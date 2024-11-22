import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse


def get_links(url):
    """Get all links from the given URL"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

def traverse_and_collect_urls(base_url):
    """Traverse directories and collect .nc file URLs"""
    file_urls = []
    years = get_links(base_url)
    years = [year for year in years if year.startswith('Y')]
    for year in tqdm(years, desc='Years', leave=False, position=0):
        year_url = base_url + year
        months = get_links(year_url)
        months = [month for month in months if month.startswith('M')]
        for month in tqdm(months, desc='Months', leave=False, position=1):
            month_url = year_url + month
            days = get_links(month_url)
            days = [day for day in days if day.startswith('D')]
            for day in tqdm(days, desc='Days', leave=False, position=2):
                day_url = month_url + day
                files = get_links(day_url)
                files = [file for file in files if file.endswith('.nc4')]
                for file in tqdm(files, desc='Files', leave=False, position=3):
                    file_url = day_url + file
                    file_urls.append(file_url)
    return file_urls

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

def download_from_urls(urls, base_dest_folder, categories, num_threads=10):
    """Read URLs from file and download them concurrently into year-specific folders"""
        
    filtered_urls = [url for url in urls if any(cat in url for cat in categories)]

    # Create a tqdm progress bar
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(lambda url: download_file(url, base_dest_folder), filtered_urls), total=len(filtered_urls)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files from NASA portal.')
    parser.add_argument('--base_url', type=str, default='https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v1/das/',
                      help='Base URL to download from')
    parser.add_argument('--dest_folder', type=str, required=True,
                      help='Destination folder for downloaded files')
    parser.add_argument('--threads', type=int, default=60,
                      help='Number of download threads')
    parser.add_argument('--categories', nargs='+', default=['chm_tavg_1hr_g1440x721_v1'],
                      help='Categories of files to download')
    args = parser.parse_args()

    # Collect URLs
    file_urls = traverse_and_collect_urls(args.base_url)
    os.makedirs(args.dest_folder, exist_ok=True)

    # Start downloading files
    download_from_urls(file_urls, args.dest_folder, args.categories, args.threads)