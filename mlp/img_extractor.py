import requests
import uuid
import random

class ImageExtractor:
   def __init__(self):
      pass
    
   def download_random_images(self, count=100, width=64, height=64, path='../data/images/non_galaxies/'):
      for _ in range(count):
         res = requests.get(f'https://picsum.photos/{width}/{height}')
         filename = f'{str(uuid.uuid4())}.jpg'
         if res.status_code == 200:
            with open(f'{path}{filename}', 'wb') as f:
               f.write(res.content)
               f.close()
               print(f'downloaded image: {filename}')
         else:
            print(f'failed to download image: status code {res.status_code}')

   def download_galaxy_images(self, count=100, page=1, path='../data/images/galaxies/'):
      res = requests.get(f'https://images-api.nasa.gov/search?q=galaxy&media_type=image&page={page}')
      if res.status_code == 200:
         data = res.json()
         collection = data.get('collection', {})
         items = collection.get('items', [])
         random.shuffle(items)
         if items:
            for i in range(count):
               links = items[i].get('links', [])
               if links:
                  sorted_links = sorted(links, key=lambda item: (item.get('size') is None, item.get('size') or 0))
                  link = sorted_links[0]
                  href = link.get('href')
                  if href:
                     img_res = requests.get(href)
                     if img_res.status_code == 200:
                        filename = f'galaxy_{str(uuid.uuid4())}.jpg'
                        with open(f'{path}{filename}', 'wb') as f:
                           f.write(img_res.content)
                           f.close()
                        print(f'downloaded galaxy image: {filename}')
                     else:
                        print(f'failed to download galaxy image from {href}: status code {img_res.status_code}')
      else:
         print(f'failed to fetch galaxy images: status code {res.status_code}: {res.text}')

extractor = ImageExtractor()
extractor.download_random_images(count=200)
# extractor.download_galaxy_images(count=50, page=3, path='./images/galaxies/staging/')