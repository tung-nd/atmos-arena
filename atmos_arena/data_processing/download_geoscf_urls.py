import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_file(url, base_dest_folder):
    """Download a file from a URL and save it to the destination folder based on the year"""
    year = url.split('/')[8]  # Extract year from the URL
    dest_folder = os.path.join(base_dest_folder, year)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    filename = os.path.join(dest_folder, url.split('/')[-1])
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)
    return url

def download_from_urls(file_path, base_dest_folder, num_threads=10):
    """Read URLs from file and download them concurrently into year-specific folders"""
    with open(file_path, 'r') as f:
        urls = f.read().splitlines()

    # Create a tqdm progress bar
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(lambda url: download_file(url, base_dest_folder), urls), total=len(urls)))

# Set file path, base destination folder, and number of threads
urls_file_path = '/home/tungnd/atmos-arena/geoscf_urls.txt'
base_destination_folder = '/eagle/MDClimSim/tungnd/data/geoscf'
number_of_threads = 60

if not os.path.exists(base_destination_folder):
    os.makedirs(base_destination_folder)

# Start downloading files
download_from_urls(urls_file_path, base_destination_folder, number_of_threads)