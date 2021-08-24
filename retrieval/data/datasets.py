import os
import torch
import pickle
import boto3
import numpy as np
from PIL import Image
from addict import Dict
from pathlib import Path
from torch.utils.data import Dataset
import io

from ..utils.logger import get_logger
from ..utils.file_utils import read_txt, load_pickle
from .preprocessing import get_transform

logger = get_logger()


class ImageDataset(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name,
        data_split, tokenizer, lang='en',
        resize_to=256, crop_size=224,
    ):
        from .adapters import FoodiML

        logger.debug(f'ImageDataset\n {[data_path, data_split, tokenizer, lang]}')
        self.tokenizer = tokenizer
        self.lang = lang
        self.data_split = data_split
        self.split = '.'.join([data_split, lang])
        self.data_path = Path(data_path)
        self.data_name = Path(data_name)

        self.data_wrapper = (
            FoodiML(
                self.data_path,
                data_split=data_split,
            )
        )

        self._fetch_captions()
        self.length = len(self.ids)

        self.transform = get_transform(
            data_split, resize_to=resize_to, crop_size=crop_size
        )

        self.captions_per_image = 1

        logger.debug(f'Split size: {len(self.ids)}')

    def _fetch_captions(self,):
        self.captions = []
        for image_id in self.data_wrapper.image_ids:
            self.captions.extend(
                self.data_wrapper.get_captions_by_image_id(image_id)
            )

        self.ids = range(len(self.captions))
        logger.debug(f'Loaded {len(self.captions)} captions')

    def load_img(self, image_id):

        s3_key = self.data_wrapper.get_s3_key_by_image_id(image_id)

        #TODO: change to 'glovo-products-dataset-d1c9720d'
        bucket_name = "test-bucket-glovocds"

        # Boto 3
        session = boto3.Session()
        s3_resource = session.resource('s3')
        bucket = s3_resource.Bucket(bucket_name)

        # Get image as bytes an open image as image PIL
        try:
            obj = bucket.Object(s3_key)
            file_stream = io.BytesIO()
            obj.download_fileobj(file_stream)
            pil_im = Image.open(file_stream)
            image = self.transform(pil_im)
        except OSError:
            print('Error to load image: ', s3_key)
            image = torch.zeros(3, 224, 224,)

        return image

    def __getitem__(self, index):
        image_id = self.data_wrapper.image_ids[index]
        image = self.load_img(image_id)
        caption = self.captions[index]
        cap_tokens = self.tokenizer(caption)

        batch = Dict(
            image=image,
            caption=cap_tokens,
            index=index,
            img_id=image_id,
        )
        return batch

    def __len__(self):
        return self.length

    def __repr__(self):
        return f'ImageDataset.{self.data_name}.{self.split}'

    def __str__(self):
        return f'{self.data_name}.{self.split}'
