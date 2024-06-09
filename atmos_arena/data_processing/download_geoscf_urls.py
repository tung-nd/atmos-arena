import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_file(url, base_dest_folder):
    """Download a file from a URL and save it to the destination folder based on the year"""
    year = url.split('/')[8]  # Extract year from the URL
    dest_folder = os.path.join(base_dest_folder, year)
    os.makedirs(dest_folder, exist_ok=True)
    filename = os.path.join(dest_folder, url.split('/')[-1])
    # check if the file already exists
    if os.path.exists(filename):
        return url
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)
    return url

def download_from_urls(file_path, base_dest_folder, categories, num_threads=10):
    """Read URLs from file and download them concurrently into year-specific folders"""
    with open(file_path, 'r') as f:
        urls = f.read().splitlines()
        
    filtered_urls = [url for url in urls if any(cat in url for cat in categories)]

    # Create a tqdm progress bar
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(lambda url: download_file(url, base_dest_folder), filtered_urls), total=len(filtered_urls)))

# Set file path, base destination folder, and number of threads
urls_file_path = '/home/tungnd/atmos-arena/geoscf_urls.txt'
base_destination_folder = '/eagle/MDClimSim/tungnd/data/geoscf'
number_of_threads = 60
categories = ['chm_tavg']

os.makedirs(base_destination_folder, exist_ok=True)

# Start downloading files
download_from_urls(urls_file_path, base_destination_folder, categories, number_of_threads)