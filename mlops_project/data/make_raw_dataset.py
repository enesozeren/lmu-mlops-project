import os
import requests
from bs4 import BeautifulSoup

# Function to download a file from a given URL
def download_file(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

def main():
    '''
    Download the raw dataset from the GitHub repository
    - Repo: https://github.com/cardiffnlp/tweeteval
    - Paper: https://arxiv.org/pdf/2010.12421
    '''
    # Base URL of the GitHub repository
    base_url = "https://github.com/cardiffnlp/tweeteval/tree/main/datasets/hate"
    raw_base_url = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/hate"
    save_dir = "data/raw"
    
    # Get the webpage content
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all .txt files in the repository
    files = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.txt')]

    # Directory to save the downloaded files
    
    os.makedirs(save_dir, exist_ok=True)

    # Download each file
    for file in files:
        file_url = raw_base_url + '/' + os.path.basename(file)
        save_path = os.path.join(save_dir, os.path.basename(file))
        download_file(file_url, save_path)
        print(f"Downloaded {file_url} to {save_path}")

if __name__ == '__main__':
    main()