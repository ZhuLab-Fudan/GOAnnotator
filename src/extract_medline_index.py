import gzip

import bs4
import torch
import xml.etree.ElementTree as ET
import os

import xmltodict
from tqdm import trange
import json
import requests  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def pubmed_from_xml_to_json_uniprot(start, end, daily=False):
    save_dir = os.path.join(BASE_DIR, "dependencies", '/pubmed/xml')
    daily_update_url = 'https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/{}'
    baseline_url = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{}'
    sub_num = 0
    article_num = 0
    docs = []
    for i in trange(start, end+1):
        file_name = 'pubmed25n{}.xml'.format(str(i).zfill(4))
        file_gz = file_name + '.gz'
        if not os.path.isfile(save_dir + file_gz):
            if daily:
                download_url = daily_update_url.format(file_gz)
            else:
                download_url = baseline_url.format(file_gz)
            
            response = requests.get(download_url)
            with open(save_dir + file_gz, "wb") as code:
                code.write(response.content)
        
        if not os.path.isfile(save_dir + file_name):
            with gzip.open(save_dir + file_gz, "rb") as f_in:
                with open(save_dir + file_name, "wb")as f_out:
                    try:
                        f_out.write(f_in.read())
                    except EOFError:
                        with open(os.path.join(BASE_DIR, "dependencies", '/pubmed/json/pubmed{}-{}.json'.format(start, i-1)), 'w') as f:
                            f.write(json.dumps(docs, indent=4, sort_keys=False))
                            print(os.path.join(BASE_DIR, "dependencies", '/pubmed/json/pubmed{}-{}.json'.format(start, i-1)))


        with open(save_dir + file_name, 'rb') as xmlfile:
            tree = ET.parse(xmlfile)
        root = tree.getroot()
        for PubmedArticle in root.findall('PubmedArticle'):
            MedlineCitation = PubmedArticle.find('MedlineCitation')

            PubmedData = PubmedArticle.find('PubmedData')
            Article = MedlineCitation.find('Article')
            ArticleIdList = PubmedData.find('ArticleIdList')
            History = PubmedData.find('History')

            
            Pub_init = {}
            Pub_json = json.loads(json.dumps(Pub_init))

            try:
                ISSN = MedlineCitation.find('MedlineJournalInfo').find('ISSNLinking').text
                Pub_json['issn'] = ISSN
            except AttributeError:
                continue
                
            Year = 'None'
            for PubMedPubDate in History.findall('PubMedPubDate'):
                if PubMedPubDate.get('PubStatus') == 'pubmed':
                    Year = PubMedPubDate.find('Year').text
                    if int(Year) >= 2024:
                        continue
                    
            Pub_json['year'] = Year
            
            if MedlineCitation.find('PMID') is not None:
                PMID = MedlineCitation.find('PMID').text
                Pub_json['id'] = PMID
            ArticleTitle = 'None'
           
            if Article.find('ArticleTitle') is not None:
                ArticleTitle = "".join(Article.find('ArticleTitle').itertext())
            Pub_json['title'] = ArticleTitle

            Abstract = 'None'
            
            if Article.find('Abstract') is None:
                continue
            if Article.find('Abstract') is not None:
                article_num = article_num + 1
                temp_text = ""
                abstract_subnum = 0
                AbstractText = Article.find('Abstract')
                for abstract_text in AbstractText.findall('AbstractText'):
                    abstract_subnum = abstract_subnum + 1
                    if abstract_text.get("Label"):
                        temp_text = temp_text + abstract_text.get("Label") + ": " + "".join(abstract_text.itertext())
                    else:
                        temp_text = temp_text + "".join(abstract_text.itertext())
                Abstract = temp_text

                if abstract_subnum > 1:
                    sub_num = sub_num + 1

            Pub_json['abstract'] = Abstract
            Pub_json['contents'] = ''
            
            if Pub_json['title'] is None:
                print("title is None")
                print(Pub_json['id'])
                Pub_json['title'] = 'None'
            else:
                Pub_json['contents'] = Pub_json['title']
            if Pub_json['abstract'] is None:
                print("abstract is None")
                print(Pub_json['id'])
                Pub_json['abstract'] = 'None'
            else:
                Pub_json['contents'] = Pub_json['contents'] + " " + Pub_json['abstract']
            


            
            docs.append(Pub_json)

        if os.path.isfile(save_dir + file_name):
            os.remove(save_dir + file_name)
        if os.path.isfile(save_dir + file_gz):
            os.remove(save_dir + file_gz)
        
    with open(os.path.join(BASE_DIR, "dependencies", '/pubmed/json/pubmed{}-{}.json'.format(start, end)), 'w') as f:
        f.write(json.dumps(docs, indent=4, sort_keys=False))
        
if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--daily", type=bool, default=False)
    args = parser.parse_args()

    pubmed_from_xml_to_json_uniprot(start=args.start, end=args.end, daily=args.daily)