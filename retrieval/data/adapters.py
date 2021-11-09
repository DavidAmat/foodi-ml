from ..utils.file_utils import load_samples
from collections import defaultdict
from ..utils.logger import get_logger
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import hashlib
logger = get_logger()
import os
import pickle
class FoodiML:

    def __init__(self, data_path, data_split):

        data_split = data_split.replace('dev', 'val')
        self.data_split = data_split
        self.data_path = Path(data_path)
        self.samples_path = (
            self.data_path / 'samples'  # TODO: change to 'samples'
        )
        self.data = load_samples(self.samples_path, self.data_split)
        print(self.data.columns)
        self.image_ids, self.img_dict, self.img_captions = self._get_img_ids()

        logger.info(f'[FoodiML] Loaded {len(self.img_captions)} images annotated ')
        
        #UNCOMMENT THIS FOR VALIDATION
        if data_split == 'val' or data_split == "test":
            if not os.path.isfile("/home/ec2-user/SageMaker/foodi-ml/valid_answers.pickle"): 
                self.valid_answers = self._compute_valid_answers(self.data)
            else:
                with open("/home/ec2-user/SageMaker/foodi-ml/valid_answers.pickle", 'rb') as handle:
                    #self.valid_answers = pickle.load(handle)
                    self.valid_answers = self._compute_valid_answers(self.data)
            
            """
            with open('/home/ec2-user/SageMaker/foodi-ml/valid_answers.pickle', 'wb') as handle:
                print("Saving valid_answers")
                pickle.dump(self.valid_answers, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Correctly saved valid_answers")"""
        
    def _get_img_ids(self):
        image_ids = list(self.data["img_id"].values)
        img_dict = {}
        annotations = defaultdict(list)
        for _, row in self.data.iterrows():
            img_dict[row["img_id"]] = row["s3_path"]
            annotations[row["img_id"]].extend([row["caption"]])
        return image_ids, img_dict, annotations

    def _compute_valid_answers(self, data: pd.DataFrame):
        """Generates the valid answers taking into account mutiple images and captions. 
        For the following dataset we will create the dictionary with valid_answers:
        id    caption    hash    |    valid_answers
        0     ABC        X       |    0,1,2,4
        1     EFG        X       |    0,1,4
        2     ABC        Y       |    0,2
        3     HIJ        Z       |    3,
        4     KLM        X       |    0,1,4
        """
        print(f"Computing hashes of the captions for better performance")
        data["cap_hash"] = data["caption"].apply(lambda x : hashlib.md5(str.encode(x)).hexdigest())
        valid_answers = {}
        print(f"Computing valid answers for data_split {self.data_split} with shape {data.shape}")

        for i, row in tqdm(data.iterrows()):
            idxs_where_duplication = (data["cap_hash"] == row["cap_hash"]) | (data["hash"] == row["hash"])
            list_indexes_duplication = list(np.where(np.array(idxs_where_duplication.to_list()) == True)[0])
            valid_answers[row["img_id"]] = list_indexes_duplication
        return valid_answers
    
    def get_image_id_by_filename(self, filename):
        return self.img_dict[filename]['imgid']

    def get_captions_by_image_id(self, img_id):
        return self.img_captions[img_id]

    def get_filename_by_image_id(self, image_id):
        return self.img_dict[image_id]

    def __call__(self, filename):
        return self.img_dict[filename]

    def __len__(self, ):
        return len(self.img_captions)