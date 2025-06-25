import argparse
import requests
"""
scraper.py
"""

def build_parser():
    """
    Generates an Argument Parser
    
    Returns:
    Argparse
    """
    parser = argparse.ArgumentParser(description="Pulls all values")
    parser.add_argument()

def load_stock_names(url):
    """_summary_

    Args:
        url (_type_): the url for a given site.
    
    Returns:
        list: a list of stock abbreviations with it's associated word code
        
    """
    if url.startswith("https:"):
        response = requests.get(url)
    else:
        response = requests.get(url)
    return response
