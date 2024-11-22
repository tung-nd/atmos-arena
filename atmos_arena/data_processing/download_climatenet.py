import requests, os
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse

### List all links to NetCDF files at a given url

def list_nc_datasets(index_url):

    # Parse target url
    reqs = requests.get(index_url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    # Find all link tags in the page and list their target href
    urls = [] 

    for link in soup.find_all('a'):
        urls.append(link.get('href'))

    # Keep only links to NetCDF file
    nc_data_urls = [x for x in urls if x.endswith('.nc')]

    return [os.path.join(index_url, x) for x in nc_data_urls]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download ClimateNet dataset.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the generated masks')
    args = parser.parse_args()
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    for split in ['train', 'test']:
        root_url = f'https://portal.nersc.gov/project/ClimateNet/climatenet_new/{split}/'
        nc_file_urls = list_nc_datasets(root_url)
        os.makedirs(os.path.join(save_dir, split), exist_ok=True)
        for url in tqdm(nc_file_urls):
            filename = url.split('/')[-1]
            req = requests.get(url)
            with open(os.path.join(save_dir, split, filename), 'wb') as f:
                f.write(req.content)