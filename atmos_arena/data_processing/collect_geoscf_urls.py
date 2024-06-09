import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

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


base_url = 'https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v1/das/'

# Collect URLs
file_urls = traverse_and_collect_urls(base_url)
# Write URLs to a text file
output_file = 'geoscf_urls.txt'
with open(output_file, 'w') as f:
    for url in file_urls:
        f.write(url.rstrip() + '\n')

# # Print URLs
# for url in file_urls:
#     print(url)
