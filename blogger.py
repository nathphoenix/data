from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import requests
import bs4
from flask_restful import Resource
from flask import request, jsonify
import os
import shutil
import regex as re
from google.cloud import language
import numpy
import six
import lxml
import nltk

import pandas as pd
from bs4 import BeautifulSoup


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="nath.json"

class Article_title(Resource):
     @classmethod
     def get_title(cls, article_URL):

        article_URL = article_URL
        URL = "https://"
        article_URL = URL + article_URL
        response = requests.get(article_URL,  headers={"user-Agent":"chrome/79.0.3945.88"})
        soup = bs4.BeautifulSoup(response.text,'lxml')

        # get title
        title = soup.find(['h1','title']).get_text()
        return title

class Article_reader(Resource):#
    @classmethod
    def read_article(cls, article_URL):

        article_URL = article_URL
        URL = "https://"
        article_URL = URL + article_URL

        response = requests.get(article_URL,  headers={"user-Agent":"chrome/79.0.3945.88"})
        soup = bs4.BeautifulSoup(response.text,'lxml')

        title = soup.find(['h1', 'title', 'entry-title', 'strong', 'header']).get_text()
        print(title)

        # get text
        paragraphs = soup.find_all(['p'])
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
            comp1 = txt_list[i].split()[0:5]
            comp2 = txt_list[i-1].split()[0:5]
            if comp1 == comp2:
                pass
            else:
                pass

        #Remove duplicates line
        lines_seen = set()  # holds lines already seen
        outfile = open('blog.txt', "w", errors="ignore")
        for line in txt_list:
            if line not in lines_seen:  # not a duplicate
                outfile.write(line)
                lines_seen.add(line)
        outfile.close()

        with open('blog.txt', 'r') as in_file:
            text = in_file.read().replace('\xa0', ' ').replace('\t', ' ').replace('\n', '').replace("U.S.", 'US')
        text = re.sub(r"([a-z\.!?])([A-Z])", r"\1 \2", text)

        #saving the above output to a file to handle spacing issue
        f = open('blog.txt', "w")
        f.write(text)
        f.close()

        file = open('blog.txt', "r")
        filedata = file.readlines()
        print('filedata', filedata)
        article = filedata[0].split(". ")
        sentences = []
        for line in article:
            sentence = line.replace("\n"," ")
            sentence = sentence.replace("[^a-zA-Z]", "").split(" ")
            sentences.append(sentence)

class Text_classification(Resource):
    @classmethod
    def classify(cls):
        with open("blog.txt","r") as f:
            line = f.read()

        language_client = language.LanguageServiceClient()

        document = language.types.Document(
            content=line,
            type=language.enums.Document.Type.PLAIN_TEXT)
        response = language_client.classify_text(document)
        categories = response.categories

        result = {}

        for category in categories:

            result[category.name] = category.confidence

        # print(line)
        for category in categories:
            print(u'=' * 20)
            print(u'{:<16}: {}'.format('category', category.name))
            print(u'{:<16}: {}'.format('confidence', category.confidence))

        data = result
        df = pd.DataFrame(result.items(), columns=['category','confidence'])
        categorized = df["category"][0]
        new = pd.read_csv("google_category.csv")
        final = new[new["Google_name"] == categorized]
        final_data = final["Bloverse_name"].to_string(index=False)
        categorized_data = final_data.replace(" ", "")
        return categorized_data

class Similarity(Resource):
    @classmethod
    def sentence_similarity(cls, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

class Build_similarity(Resource):
    @classmethod
    def build_similarity_matrix(cls, sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #ignore if both are same sentences
                    continue
                similarity_matrix[idx1][idx2] = Similarity.sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
        return similarity_matrix

class Blogger(Resource):
    @classmethod
    def get(cls, article_URL, top_n=4):
        stop_words = stopwords.words('english')
        summarize_text = []

        # Step 1 - Read text and split it
        sentences =  Article_reader.read_article(article_URL)
        #print(sentences)
        categorized = Text_classification.classify()
        
        title = Article_title.get_title(article_URL)

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = Build_similarity.build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank_numpy(sentence_similarity_graph, alpha=0.9)


        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        #print("Indexes of top ranked_sentence order are ", ranked_sentence)

        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))
        summarized = ". ".join(summarize_text)
        summarized = nltk.word_tokenize(summarized)
        summarized = [items_len[:200] for items_len in summarized]


        title = {'summarized_text':title}
        summary = {'summarized_text':summarized}
        category = { 'article category': categorized}
        result = []
        result.append(title)
        result.append(summary)
        result.append(category)
        return  {
           'status': 'success',
           'data': result,
           'message': 'articles retrieved',
            }, 200
