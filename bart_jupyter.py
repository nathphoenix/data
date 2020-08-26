import os
import torch
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import requests
import bs4
import os
import shutil
from PIL import Image
import regex as re
from google.cloud import language
import numpy
import six
from wordcloud import WordCloud
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize,word_tokenize
from string import punctuation
started=datetime.now()
def read_article(article_URL):    
#     FOLDER_NAME = 'blog_details' #@param {type:"string"}
#     try:
#         os.mkdir(FOLDER_NAME)
#     except:
#         print("file already exist")
#     os.chdir(FOLDER_NAME)
    #@title Enter Medium Story URL
    article_URL = article_URL #@param {type:"string"}
    # article_URL = 'https://www.tmz.com/2020/07/29/dr-dre-answers-wife-divorce-petition-prenup/'
    response = requests.get(article_URL)
    soup = bs4.BeautifulSoup(response.text,'html.parser')
#     for i, li in enumerate(soup.select('ol')):
#         print(i, li.text)
    # get title
    title = soup.find(['h1','e1dc']).get_text()
    print(title)
    # get text
    paragraphs = soup.find_all(['li', 'p', 'strong', 'em', "ol", "c909", "h2"])
    txt_list = []
    tag_list = []
    with open('content2.txt', 'w') as f:
      f.write(title + '\n\n')
      for p in paragraphs:
            if p.href:
                pass
            else:
                if len(p.get_text()) > 100: # this filters out things that are most likely not part of the core article
    #                 print(p.href)
                    tag_list.append(p.name)
                    txt_list.append(p.get_text())
    # This snippet of code deals with duplicate outputs from the html, helps us clean up the data further
    txt_list2 = []
    tag_list2 = []
    for i in range(len(txt_list)):
    # #     if '\n' not in txt_list[i]:
    #     print(txt_list[i])
    #         print(len(txt_list[i]))
    #     print(tag_list[i])
    #     print()
        comp1 = txt_list[i].split()[0:5]
        comp2 = txt_list[i-1].split()[0:5]
        if comp1 == comp2:
            pass
        else:
            pass
    #Remove duplicates line
    lines_seen = set()  # holds lines already seen
    outfile = open('foot.txt', "w", errors="ignore")
    for line in txt_list:
    #     print (lines)
        if line not in lines_seen:  # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()
 
    file = open('foot.txt', "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []
    for line in article:
        sentence = word_tokenize(line.lower())
        _stopwords = set(stopwords.words('english') + list(punctuation))
        sentence=[word for word in sentence if word not in _stopwords]
        sentence = re.sub(r"([a-z\.!?])([A-Z])", r"\1 \2", line)
        sentence = sentence.replace("[^a-zA-Z]", "").split(" ")
        sentences.append(sentence) 
#         print(sentences)
    
    return sentences[:5]
def classify():
    for line in open('foot.txt', "r"):
        line = line
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "nath.json"
    language_client = language.LanguageServiceClient()
    document = language.types.Document(
        content=line,
        type=language.enums.Document.Type.PLAIN_TEXT)
    response = language_client.classify_text(document)
    categories = response.categories
    result = {}
    for category in categories:
            # Turn the categories into a dictionary of the form:
            # {category.name: category.confidence}, so that they can
            # be treated as a sparse vector.
        result[category.name] = category.confidence
    # print(line)
    for category in categories:
        print(u'=' * 20)
        print(u'{:<16}: {}'.format('category', category.name))
        print(u'{:<16}: {}'.format('confidence', category.confidence))
    # if verbose:
    #     print(text)
    #     for category in categories:
    #         print(u'=' * 20)
    #         print(u'{:<16}: {}'.format('category', category.name))
    #         print(u'{:<16}: {}'.format('confidence', category.confidence))
    data = result
    df = pd.DataFrame(result.items(), columns=['category','confidence'])
    categorized = df["category"][0]
    new = pd.read_csv("google_category.csv")
    final = new[new["Google_name"] == categorized]
    final_data = final["Bloverse_name"].to_string(index=False)
    categorized_data = final_data.replace(" ", "")
    return categorized_data
def summarization(text):
    # use bart in pytorch
    summary = summarizer(text, min_length = round(0.1 * len(text)), max_length = round(0.2 * len(text)), return_tensors='pt')
    return summary[0]['summary_text']
def split_to_keypoints(text, min_char=100, max_char=160):
    sentences = text.split('. ')
    keypoint_list = []
    for sent in sentences:
        if len(sent) > min_char and len(sent) < max_char:
            keypoint_list.append(sent)
    return keypoint_list
def split_text(text, length=4096):
    text_len = len(text)
    if text_len > length:
        return [text[i:i+length] for i in range(0, text_len, length)]
    else:
        return [text]
def clean_text():
    for line in open('foot.txt', "r"):
        text = line
    text = text.replace('\n', ' ')
    text = text.replace(r'\xa0', ' ')
    return text
def run_summarization(article_URL):
    text = read_article(article_URL)
    text = clean_text()
    category = classify()
    splitted_text = split_text(text)
    summaries = []
    keypoint_list = None
    split_len = len(splitted_text)
    if split_len>1:
        print("Long Text")
        for split_t in splitted_text:
            print(f'Processing {splitted_text.index(split_t)} of {split_len}')
            summaries.append(summarization(split_t))
        print(f'Processing Executive Summary')
        try:
            print('No recursion')
            keypoint_list = split_to_keypoints(summarization( ' '.join(summaries) ))
        except IndexError:
            print('Recursion')
            keypoint_list, _ = run_summarization( ' '.join(summaries) )
    else:
        print('Short Text')
        keypoint_list = split_to_keypoints(summarization(splitted_text[0]))
    print('Done')
    print(keypoint_list)
    return keypoint_list, summaries
    timed = datetime.now() - started
    print(timed)
path_to_model_pipeline = 'summarize_model_checkpoint.pt'
if os.path.isfile(path_to_model_pipeline):
    print('THERE and LOADING')
    summarizer = torch.load(path_to_model_pipeline)
else:
    print('SAVING')
    torch.save(pipeline("summarization"), path_to_model_pipeline)
    print('LOADING')
    summarizer = torch.load(path_to_model_pipeline)
run_summarization("https://medium.com/dose/does-the-queen-of-england-have-any-real-power-57e5750d68b1")