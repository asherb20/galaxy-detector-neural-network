import os
import argparse
from multiprocessing import Pool
import pandas as pd
from PIL import Image
from cnn.transforms import GZ2_CROP_FRACTION, CenterCropFraction

"""One-off: write center-cropped, downsized copies of the GZ2 images.

Decoding the 424px JPEGs is the main training bottleneck. The copies keep the same
subfolder/filename layout under <root>/images_160 and are already cropped, so training
skips the GZ2 crop when it uses them.

usage: python -m cnn.preresize
"""

PRERESIZED_DIR = 'images_160'
PRERESIZED_SIZE = 160
COMPLETE_MARKER = '_complete'


def preresized_dir(root):
   """Path of a finished pre-resized image folder, or None"""
   path = os.path.join(root, PRERESIZED_DIR)
   return path if os.path.exists(os.path.join(path, COMPLETE_MARKER)) else None


def use_preresized(gz2_dataset, image_dir):
   """Point a GZ2 dataset's file locations at the pre-resized copies"""
   catalog = gz2_dataset.catalog
   catalog['file_loc'] = [
      os.path.join(image_dir, subfolder, filename)
      for subfolder, filename in zip(catalog['subfolder'], catalog['filename'])
   ]


def _resize_one(job):
   source, target = job
   if os.path.exists(target):
      return
   os.makedirs(os.path.dirname(target), exist_ok=True)
   with Image.open(source) as image:
      image = CenterCropFraction(GZ2_CROP_FRACTION)(image.convert('RGB'))
      image = image.resize((PRERESIZED_SIZE, PRERESIZED_SIZE), Image.Resampling.LANCZOS)
      image.save(target, quality=95)


def preresize(root, workers):
   source_dir = os.path.join(root, 'images')
   target_dir = os.path.join(root, PRERESIZED_DIR)
   jobs = []
   for split in ('train', 'test'):
      catalog = pd.read_parquet(os.path.join(root, f'gz2_{split}_catalog.parquet'), columns=['subfolder', 'filename'])
      for subfolder, filename in zip(catalog['subfolder'], catalog['filename']):
         jobs.append((os.path.join(source_dir, subfolder, filename), os.path.join(target_dir, subfolder, filename)))

   print(f'resizing {len(jobs)} images into {target_dir}')
   with Pool(workers) as pool:
      for done, _ in enumerate(pool.imap_unordered(_resize_one, jobs, chunksize=256), start=1):
         if done % 20000 == 0:
            print(f'{done}/{len(jobs)}')

   open(os.path.join(target_dir, COMPLETE_MARKER), 'w').close()
   print('done')


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--root', default='data/gz2')
   parser.add_argument('--workers', type=int, default=os.cpu_count())
   args = parser.parse_args()
   preresize(args.root, args.workers)
