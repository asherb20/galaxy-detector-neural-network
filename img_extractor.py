import requests
import uuid

class ImageExtractor:
   def __init__(self):
      pass
    
   def download_random_image(self, width=128, height=128, path='./images/non_galaxies/'):
      res = requests.get(f'https://picsum.photos/{width}/{height}')
      filename = f'{str(uuid.uuid4())}.jpg'
      if res.status_code == 200:
         with open(f'{path}{filename}', 'wb') as f:
            f.write(res.content)
            f.close()
      else:
         print(f'failed to download image: status code {res.status_code}')

   def download_galaxy_images(self, count=100):
      res = requests.get(f'https://images-api.nasa.gov/search?q=galaxy&media_type=image')
      if res.status_code == 200:
         data = res.json()
         collection = data.get('collection', {})
         items = collection.get('items', [])
         print(f'found {len(items)} galaxy images from NASA API')
         if items:
            for i in range(count):
               links = items[i].get('links', [])
               if links:
                  sorted_links = sorted(links, key=lambda item: item.get('size'))
                  for link in sorted_links:
                     href = link.get('href')
                     if href:
                        img_res = requests.get(href)
                        if img_res.status_code == 200:
                           filename = f'galaxy_{str(uuid.uuid4())}.jpg'
                           with open(f'./images/galaxies/{filename}', 'wb') as f:
                              f.write(img_res.content)
                              f.close()
                           print(f'downloaded galaxy image: {filename}')
      else:
         print(f'failed to fetch galaxy images: status code {res.status_code}: {res.text}')

   def download_multiple_images(self, count=100, width=128, height=128, path='./images/non_galaxies/'):
      for _ in range(count):
         self.download_random_image(width, height, path)

extractor = ImageExtractor()
# extractor.download_multiple_images(count=3)
extractor.download_galaxy_images(3)