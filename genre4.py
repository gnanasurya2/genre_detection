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
pd.set_option("display.max_rows",70000)
pd.set_option("display.max_columns",50000)
pd.set_option("display.max_colwidth",50000)
def importing_data():
    print("IMPORTING DATA:")
    print("->IMPORTING TECH DOCUMENTS...")
    book=glob.glob("/home/gnanasurya/Desktop/bbc/tech/*.txt")
    category,document,title,category_id=[],[],[],[]
    for doc_name in book:
        with open(doc_name,'r') as file:
            category.append("tech")
            category_id.append("0")
            title.append(file.readline())
            document.append(file.read())
    print("->IMPORTING SPORT DOCUMENTS...")
    book=glob.glob("/home/gnanasurya/Desktop/bbc/sport/*.txt")
    for doc_name in book:
        with open(doc_name,"r") as file:
            category.append("sport")
            category_id.append("1")
            title.append(file.readline())
            document.append(file.read())
    print("->IMPORTING POLITICS DOCUMENT...")
    book=glob.glob("/home/gnanasurya/Desktop/bbc/politics/*.txt")
    for doc_name in book:
        with open(doc_name,"r") as file:
            category.append("politics")
            category_id.append("2")
            title.append(file.readline())
            document.append(file.read())
    print("->IMPORTING ENTERTAINMENT DOCUMENTS..")
    book=glob.glob("/home/gnanasurya/Desktop/bbc/entertainment/*.txt")
    for doc_name in book:
        with open(doc_name,"r") as file:
            category.append("entertainment")
            category_id.append("3")
            title.append(file.readline())
            document.append(file.read())
    print("->IMPORTING BUSINESS DOCUMENTS...")
    book=glob.glob("/home/gnanasurya/Desktop/bbc/business/*.txt")
    for doc_name in book:
        with open(doc_name,"r") as file:
            category.append("business")
            category_id.append("4")
            title.append(file.readline())
            document.append(file.read())
    data_frame=pd.DataFrame({"category":category,"category_id":category_id,"title":title,"document":document})
    data_frame=shuffle(data_frame)
    return data_frame
def vectorizing_words(data_frame):
    vectorizer=TfidfVectorizer()
    vectorizer.fit(data_frame["document"])
    vector=vectorizer.transform(data_frame["document"])
    print(vector.shape)
    
def main():
    data_frame=pd.DataFrame()
    data_frame=importing_data()
    print(data_frame.head(1))
    vectorizing_words(data_frame)
if __name__ == "__main__":
    main()