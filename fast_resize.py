"""
Fast image resize script: DRD@Kaggle

__author__ : Abhishek Thakur
"""

import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from PIL import Image, ImageChops

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
input_folder =
output_folder =
images = glob.glob(os.path.join(input_folder, "*.jpg"))
Parallel(n_jobs=12)(
	delayed(resize_image)(
		i,
		output_folder,
		(512, 512)
	) for i in tqdm(images)
)