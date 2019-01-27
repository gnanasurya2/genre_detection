#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 19:41:21 2019

@author: gnanasurya
"""

import pandas as pd
import glob
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option("display.max_rows", 70000)
pd.set_option("display.max_columns", 50000)
pd.set_option("display.max_colwidth", 50000)


def importing_data(news_folder, files_regex, topics):
    print("IMPORTING DATA:")
    data_table = []
    for idx, topic in enumerate(topics):
        print("->IMPORTING {} DOCUMENTS...".format(topic.upper()))
        files = glob.glob(
            "/home/gnanasurya/Desktop/bbc/{}/{}".format(topic, files_regex))
        for doc in files:
            with open(doc, 'r') as file:
                data_table.append({
                    'category': topic,
                    'category_id': idx,
                    'title': file.readline(),
                    'document': file.read()
                })
    data_frame = pd.DataFrame(data_table)
    data_frame = shuffle(data_frame)
    return data_frame


def vectorizing_words(data_frame):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data_frame["document"])
    vector = vectorizer.transform(data_frame["document"])
    print(vector.shape)


def main():
    news_folder = "/home/gnanasurya/Desktop/bbc/"
    files_regex = '*.txt'
    topics = ['tech', 'sport', 'politics', 'entertainment', 'business']
    data_frame = importing_data(news_folder, files_regex, topics)
    print(data_frame.head(1))
    vectorizing_words(data_frame)


if __name__ == "__main__":
    main()
