import requests
from bs4 import BeautifulSoup as bs
import random 
import numpy as np

def get_text(url): 
    headers = {} 
    user_agent_list = [
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) Gecko/20100101 Firefox/61.0",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
        "Mozilla/5.0 (Macintosh; U; PPC Mac OS X 10.5; en-US; rv:1.9.2.15) Gecko/20110303 Firefox/3.6.15",
    ]
    headers['User-Agent'] = random.choice(user_agent_list)

    content = requests.get(url, headers = headers).text
    # time.sleep(10)
    soup = bs(content, 'lxml')
    # text = ''
    try:
        title = soup.select('#full-view-heading > h1')[0].get_text().replace('\n', '').replace('\t', '').strip()
    except IndexError:
        # print(url)
        text = ''
    else:
        try:
            abstracts = soup.select('#eng-abstract > p')
            abstract = ''
            for item in abstracts:
                abstract = abstract + item.get_text().replace('\n', '').replace('\t', '').strip()
            # abstract = soup.select('#eng-abstract > p')[0].get_text().replace('\n', '').strip()
            text = title+'.'+abstract
        except IndexError:
            text=''
     
    return text

def save_results(file_name, dict):
    np.save(file_name, dict)
    print('save: ', file_name)