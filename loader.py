import requests
import json
"""
scraper.py
"""

def load_stock_names(url):
    """_summary_

    Args:
        url (_type_): the url for a given site.
    
    Returns:
        list: a list of stock abbreviations with it's associated word code
        
    """
    headers = {
        "user-agent": "To get better at coding for kjonny3@gmail.com"
    }
    if url.startswith("https:"):
        response = requests.get(url, headers=headers)
    else:
        response = requests.get(url, headers=headers)
    return response

def main():
    r = load_stock_names("https://stockanalysis.com/stocks/")
    print(r.headers['content-type'])
    print(r.text)

if __name__ == "__main__":
    main()
