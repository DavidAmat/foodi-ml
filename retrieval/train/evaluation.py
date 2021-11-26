import torch
import numpy as np
from tqdm import tqdm
from timeit import default_timer as dt
from ..model.similarity.measure import cosine_sim, l2norm, l2norm_numpy, cosine_sim_numpy
from ..utils import layers
import h5py

@torch.no_grad()
def predict_loader(model, data_loader, device):
    img_embs, cap_embs, cap_lens = None, None, None
    max_n_word = 200
    model.eval()

    pbar_fn = lambda x: x
    if model.master:
        pbar_fn = lambda x: tqdm(
            x, total=len(x),
            desc='Pred  ',
            leave=False,
        )
    print("Evaluation begins")
    max_n_samples=len(data_loader.dataset)
    max_n_samples=100
    count=0
    for batch in pbar_fn(data_loader):
        ids = batch['index']
        if len(batch['caption'][0]) == 2:
            (_, _), (_, lengths) = batch['caption']
        else:
            cap, lengths = batch['caption']
        img_emb, cap_emb = model.forward_batch(batch)
        
        if img_embs is None:
            if len(img_emb.shape) == 3:
                is_tensor = True
                
                img_embs = np.zeros((max_n_samples, img_emb.size(1), img_emb.size(2)))
                cap_embs = np.zeros((max_n_samples, max_n_word, cap_emb.size(2)))
            else:
                is_tensor = False
                img_embs = np.zeros((max_n_samples, img_emb.size(1)))
                cap_embs = np.zeros((max_n_samples, cap_emb.size(1)))
            cap_lens = [0] * max_n_samples
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy()
        
        if is_tensor:
            max_lengths = max(lengths)
            cap_embs[ids,:max_lengths,:] = cap_emb.data.cpu().numpy()
        else:
            cap_embs[ids,] = cap_emb.data.cpu().numpy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = lengths[j]
        count=count+len(ids)
        if count==max_n_samples:
            break

    # No redundancy in number of captions per image
    if img_embs.shape[0] == cap_embs.shape[0]:
        img_embs = remove_img_feat_redundancy(img_embs, data_loader)
    
    return img_embs, cap_embs, cap_lens



@torch.no_grad()
def predict_loader_bigdata(model, data_loader, device):
    img_embs, cap_embs, cap_lens = None, None, None
    max_n_word = 100
    model.eval()
    pbar_fn = lambda x: x
    if model.master:
        pbar_fn = lambda x: tqdm(
            x, total=len(x),
            desc='Pred  ',
            leave=False,
        )
    
    max_samples_eval = len(data_loader.dataset)
    #max_samples_eval = 70
    count=0
    img_embs = np.zeros((max_samples_eval, 2048), dtype=np.float)
    PARTIAL_BATCH_SIZE = 35
    print("Beginning the image part")
    b = None
    for batch in pbar_fn(data_loader):
        ids = batch['index']
        remainder = ids[0]%PARTIAL_BATCH_SIZE
        remainder = count%PARTIAL_BATCH_SIZE
        
        if b is None:
            b = torch.empty(PARTIAL_BATCH_SIZE, *batch["image"].size()[1:])
            total_ids = [None]*PARTIAL_BATCH_SIZE

        if remainder == (PARTIAL_BATCH_SIZE - 1):
            b[remainder] = batch["image"].squeeze(0)
            total_ids[remainder] = ids[0]
            img_emb = model.forward_batch_img_DBG(b)
            # cache embeddings
            img_embs[total_ids] = img_emb.mean(-1).data.cpu().numpy()
            
            b = torch.empty(PARTIAL_BATCH_SIZE, *batch["image"].size()[1:])
            total_ids = [None] * PARTIAL_BATCH_SIZE
        else: 
            b[remainder] = batch["image"].squeeze(0)
            total_ids[remainder] = ids[0]
        
        count=count+len(ids)
        if count==max_samples_eval:
            break
    
    img_embs = torch.from_numpy(img_embs)
    print("Beginning the caption part")
    
    t2i_r_at = {
        1:np.zeros(max_samples_eval),
        5: np.zeros(max_samples_eval),
        10: np.zeros(max_samples_eval)
    }
    
    i2t_r_at = {
        1:np.zeros(max_samples_eval),
        5: np.zeros(max_samples_eval),
        10: np.zeros(max_samples_eval)
    }
    
    
    
    
    count = 0    
    for batch in pbar_fn(data_loader):
        sims = np.zeros((max_samples_eval, 1))
        ids = batch["index"]

        cap_emb = model.forward_batch_cap(batch)
        cap_batch_size, cap_num_words, cap_emb_dim = cap_emb.size()
        cap_emb = cap_emb[:, :min(max_n_word, cap_num_words), :]
        cap_emb = cap_emb.to(device)
        cap_emb = cap_emb.permute(0, 2, 1)
        cap_emb = model.similarity.similarity.norm(cap_emb)
        for i in range(len(img_embs)):
            img_vector = img_embs[i].unsqueeze(0)
            img_vector = img_vector.float()
            img_vector = img_vector.to(device)
            
            txt_output = model.similarity.similarity.adapt_txt(value=cap_emb, query=img_vector)
            txt_output = model.similarity.similarity.fovea(txt_output)
            txt_vector = txt_output.max(dim=-1)[0]
            txt_vector = l2norm(txt_vector, dim=-1)
            
            img_vector = l2norm(img_vector, dim=-1)

            sim = cosine_sim(img_vector, txt_vector)
            sim = sim.squeeze(-1)
            sims[i, 0] = sim.cpu().numpy()
        
        # compute t2i for this given column of the similarity matrix
        index = ids[0]
        image_id = data_loader.dataset.data_wrapper.image_ids[index]
        inds = np.argsort(sims[:,0])[::-1]
        for k in t2i_r_at.keys(): # 1,5,10
            intersection = np.intersect1d(inds[:k], data_loader.dataset.data_wrapper.valid_answers[image_id])
            t2i_r_at[k][index] = (1 if len(intersection) > 0 else 0)
            
        count=count+len(ids)

        if count==max_samples_eval:
            #i2t_metrics = i2t_duplicated_idxs(sims, data_loader.dataset.data_wrapper.valid_answers, data_loader.dataset.data_wrapper)
            r1 = 100.0 * (np.sum(t2i_r_at[1])/len(t2i_r_at[1]))
            r5 = 100.0 * (np.sum(t2i_r_at[5])/len(t2i_r_at[5]))
            r10 = 100.0 * (np.sum(t2i_r_at[10])/len(t2i_r_at[10]))
            print(f"t2i_metrics: ", r1, r5, r10)
            
            #t2i_metrics = t2i_duplicated_idxs(sims, data_loader.dataset.data_wrapper.valid_answers, data_loader.dataset.data_wrapper)
            #print("i2t_metrics: ", t2i_metrics)    
            break
            
    count = 0  
        
    print("FINISHED T2I")
    print("--------------")
    print("Computing I2T")
    """img_Embeds = np.zeros((284000, 2048))
    max_samples_eval = 284000
    cap_embeds = np.zeros((max_samples_eval, 2048, max_n_word))
    
    for i in pbar_fn(range(len(img_embs))):
        img_vector = img_embs[i].unsqueeze(0)
        img_vector = img_vector.float()
        img_vector = img_vector.to(device)
        img_vector = l2norm(img_vector, dim=-1)
        
        sims = np.zeros((1, max_samples_eval))
        
        for batch in data_loader:
            ids = batch["index"]
            cap_emb = model.forward_batch_cap(batch)
            cap_batch_size, cap_num_words, cap_emb_dim = cap_emb.size()
            cap_emb = cap_emb[:, :min(max_n_word, cap_num_words), :]
            cap_emb = cap_emb.to(device)
            cap_emb = cap_emb.permute(0, 2, 1)
            cap_emb = model.similarity.similarity.norm(cap_emb)
            
            txt_output = model.similarity.similarity.adapt_txt(value=cap_emb, query=img_vector)
            txt_output = model.similarity.similarity.fovea(txt_output)
            txt_vector = txt_output.max(dim=-1)[0]
            txt_vector = l2norm(txt_vector, dim=-1)

            sim = cosine_sim(img_vector, txt_vector)
            sim = sim.squeeze(-1)
            sims[0, i] = sim.cpu().numpy()
        
        # compute t2i for this given column of the similarity matrix
        image_id = data_loader.dataset.data_wrapper.image_ids[i]
        inds = np.argsort(sims[0, :])[::-1]
        for k in i2t_r_at.keys(): # 1,5,10
            intersection = np.intersect1d(inds[:k], data_loader.dataset.data_wrapper.valid_answers[image_id])
            i2t_r_at[k][i] = (1 if len(intersection) > 0 else 0)
            
        count=count+len(ids)

        if count==max_samples_eval:
            
            r1 = 100.0 * (np.sum(i2t_r_at[1])/len(i2t_r_at[1]))
            r5 = 100.0 * (np.sum(i2t_r_at[5])/len(i2t_r_at[5]))
            r10 = 100.0 * (np.sum(i2t_r_at[10])/len(i2t_r_at[10]))
            print(f"i2t_metrics: ", r1, r5, r10)
            
            #t2i_metrics = t2i_duplicated_idxs(sims, data_loader.dataset.data_wrapper.valid_answers, data_loader.dataset.data_wrapper)
            #print("i2t_metrics: ", t2i_metrics)    
            break"""
        
        
    return sims


def remove_img_feat_redundancy(img_embs, data_loader):
        return img_embs[np.arange(
                            start=0,
                            stop=img_embs.shape[0],
                            step=data_loader.dataset.captions_per_image,
                        ).astype(np.int)]

@torch.no_grad()
def evaluate(
    model, img_emb, txt_emb, lengths,
    device, shared_size=128, return_sims=False
):
    model.eval()
    _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')

    begin_pred = dt()
    #commenting if it suffices to CPU this
    img_emb = torch.FloatTensor(img_emb).to(device)
    txt_emb = torch.FloatTensor(txt_emb).to(device)

    end_pred = dt()
   
    
    sims = model.get_sim_matrix_shared(
        embed_a=img_emb, 
        embed_b=txt_emb,
        lens=lengths,
        shared_size=1
    )
    sims = layers.tensor_to_numpy(sims)
    end_sim = dt()

    i2t_metrics = i2t(sims)
    print('i2t_metrics:',i2t_metrics)
    t2i_metrics = t2i(sims)
    print('t2i_metrics:',t2i_metrics)
    rsum = np.sum(i2t_metrics[:3]) + np.sum(t2i_metrics[:3])

    i2t_metrics = {f'i2t_{k}': v for k, v in zip(_metrics_, i2t_metrics)}
    t2i_metrics = {f't2i_{k}': v for k, v in zip(_metrics_, t2i_metrics)}

    metrics = {
        'pred_time': end_pred-begin_pred,
        'sim_time': end_sim-end_pred,
    }
    metrics.update(i2t_metrics)
    metrics.update(t2i_metrics)
    metrics['rsum'] = rsum

    if return_sims:
        return metrics, sims

    return metrics


@torch.no_grad()
def evaluate_bigdata(
    model, sims,device, shared_size=128, return_sims=False):
    
    model.eval()
    _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')

    begin_pred = dt()
    end_pred = dt()
    
    sims = sims.cpu().numpy()
    end_sim = dt()
    _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')
    i2t_metrics = i2t(sims)
    print('i2t_metrics:',i2t_metrics)
    t2i_metrics = t2i(sims)
    print('t2i_metrics:',t2i_metrics)
    rsum = np.sum(i2t_metrics[:3]) + np.sum(t2i_metrics[:3])

    i2t_metrics = {f'i2t_{k}': v for k, v in zip(_metrics_, i2t_metrics)}
    t2i_metrics = {f't2i_{k}': v for k, v in zip(_metrics_, t2i_metrics)}

    metrics = {
        'pred_time': end_pred-begin_pred,
        'sim_time': end_sim-end_pred,
    }
    metrics.update(i2t_metrics)
    metrics.update(t2i_metrics)
    metrics['rsum'] = rsum

    if return_sims:
        return metrics, sims

    return metrics

def t2i(sims):
    npts, ncaps = sims.shape
    captions_per_image = ncaps // npts

    ranks = np.zeros(captions_per_image * npts)
    top1 = np.zeros(captions_per_image * npts)

    # --> (5N(caption), N(image))
    sims = sims.T
    print(sims)
    for index in range(npts):
        for i in range(captions_per_image):
            inds = np.argsort(sims[captions_per_image * index + i])[::-1]
            ranks[captions_per_image * index + i] = np.where(inds == index)[0][0]
            top1[captions_per_image * index + i] = inds[0]
            #print('top1t2i:',inds[0])

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.round(np.floor(np.median(ranks)) + 1,5)
    meanr = np.round(ranks.mean() + 1,5)

    return (r1, r5, r10, medr, meanr)

def i2t(sims):
    # trying newer implementation
    npts, ncaps = sims.shape
    captions_per_image = ncaps // npts

    ranks = np.zeros(captions_per_image * npts)
    top1 = np.zeros(captions_per_image * npts)

    # --> (5N(caption), N(image))
    #sims = sims.T
    for index in range(npts):
        for i in range(captions_per_image):
            inds = np.argsort(sims[captions_per_image * index + i])[::-1]
            ranks[captions_per_image * index + i] = np.where(inds == index)[0][0]
            top1[captions_per_image * index + i] = inds[0]
            #print('top1i2t:',inds[0])

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.round(np.floor(np.median(ranks)) + 1,5)
    meanr = np.round(ranks.mean() + 1,5)

    return (r1, r5, r10, medr, meanr)


@torch.no_grad()
def evaluate_bigdata_new_metrics(
    model, sims, device, valid_answers, shared_size=128, return_sims=False, adapter=None):
    
    model.eval()
    _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')

    begin_pred = dt()
    end_pred = dt()
    
    #sims = sims.cpu().numpy()
    end_sim = dt()
    _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')
    i2t_metrics = i2t_duplicated_idxs(sims, valid_answers, adapter)
    print('i2t_metrics:',i2t_metrics)
    t2i_metrics = t2i_duplicated_idxs(sims, valid_answers, adapter)
    print('t2i_metrics:',t2i_metrics)
    rsum = np.sum(i2t_metrics[:3]) + np.sum(t2i_metrics[:3])

    i2t_metrics = {f'i2t_{k}': v for k, v in zip(_metrics_, i2t_metrics)}
    t2i_metrics = {f't2i_{k}': v for k, v in zip(_metrics_, t2i_metrics)}

    metrics = {
        'pred_time': end_pred-begin_pred,
        'sim_time': end_sim-end_pred,
    }
    metrics.update(i2t_metrics)
    metrics.update(t2i_metrics)
    metrics['rsum'] = rsum

    if return_sims:
        return metrics, sims

    return metrics


def t2i_duplicated_idxs(sims, valid_answers_imgs : dict, adapter):
    npts, ncaps = sims.shape
    captions_per_image = ncaps//npts

    r_at = {
        1:np.zeros(captions_per_image * npts),
        5: np.zeros(captions_per_image * npts),
        10: np.zeros(captions_per_image * npts)
        }
    
    #sims = sims.T
    for index in range(npts):
        image_id = adapter.image_ids[index]
        inds = np.argsort(sims[:, captions_per_image * index])[::-1]
        for k in r_at.keys(): # 1,5,10
            intersection = np.intersect1d(inds[:k], valid_answers_imgs[image_id])
            r_at[k][captions_per_image * index] = (1 if len(intersection) > 0 else 0)
    r1 = 100.0 * (np.sum(r_at[1])/len(r_at[1]))
    r5 = 100.0 * (np.sum(r_at[5])/len(r_at[5]))
    r10 = 100.0 * (np.sum(r_at[10])/len(r_at[10]))
    return r1, r5, r10

def i2t_duplicated_idxs(sims, valid_answers_caps : dict, adapter):
    npts, ncaps = sims.shape
    captions_per_image = ncaps//npts
    
    r_at = {
        1:np.zeros(captions_per_image * npts),
        5: np.zeros(captions_per_image * npts),
        10: np.zeros(captions_per_image * npts)
    }
    
    for index in range(npts):
        image_id = adapter.image_ids[index]
        inds = np.argsort(sims[captions_per_image * index])[::-1]
        for k in r_at.keys(): # 1,5,10
            intersection = np.intersect1d(inds[:k], valid_answers_caps[image_id])
            r_at[k][captions_per_image * index] = (1 if len(intersection) > 0 else 0)
            
    r1 = 100.0 * (np.sum(r_at[1])/len(r_at[1]))
    r5 = 100.0 * (np.sum(r_at[5])/len(r_at[5]))
    r10 = 100.0 * (np.sum(r_at[10])/len(r_at[10]))
    return r1, r5, r10