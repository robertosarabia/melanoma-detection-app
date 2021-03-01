"""
Fast image resize script: DRD@Kaggle

__author__ : Abhishek Thakur
"""

import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize(
        (resize[1], resize[0]), resample=Image.BILINEAR
    )
    img.save(outpath)


# resize train images
input_folder = "E:/documents/dev/skin cancer detection data/siim-isic-melanoma-classification/jpeg/train"
output_folder = "E:/documents/dev/skin cancer detection data/siim-isic-melanoma-classification/jpeg/train224"
images = glob.glob(os.path.join(input_folder, "*.jpg"))
Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,
        output_folder,
        (224, 224)
    ) for i in tqdm(images)
)

# resize test images
input_folder = "E:/documents/dev/skin cancer detection data/siim-isic-melanoma-classification/jpeg/test"
output_folder = "E:/documents/dev/skin cancer detection data/siim-isic-melanoma-classification/jpeg/test224"
images = glob.glob(os.path.join(input_folder, "*.jpg"))
Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,
        output_folder,
        (224, 224)
    ) for i in tqdm(images)
)
