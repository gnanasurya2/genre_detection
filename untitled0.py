import pandas as pd
import nltk
import glob
# to randomize the dataframe
from sklearn.utils import shuffle
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
pd.set_option("display.max_rows",5000)
pd.set_option("display.max_columns",20)
pd.set_option("display.max_colwidth",7000)
def importing_data():
    book=glob.glob("/home/gnanasurya/Desktop/bbc_test/tech/*.txt")

    category,document=[],[]
    for doc_name in book:
        with open(doc_name,"r") as file:
            category.append('tech')
            document.append(file.read())
    print(" ->importing entertainment documents...")

    book=glob.glob("/home/gnanasurya/Desktop/bbc_test/entertainment/*.txt")

    for doc_name in book:
        with open(doc_name,"r") as file:
            category.append('entertainment')
            document.append(file.read())

    book=glob.glob("/home/gnanasurya/Desktop/bbc_test/business/*.txt")
    print(" ->importing business documents...")

    for doc_name in book:
        with open(doc_name,"r") as file:
            category.append('business')
            document.append(file.read())

    book=glob.glob("/home/gnanasurya/Desktop/bbc_test/politics/*.txt")
    print(" ->importing politics documents...")

    for doc_name in book:
        with open(doc_name,"r") as file:
            category.append('politics')
            document.append(file.read())

    book=glob.glob("/home/gnanasurya/Desktop/bbc_test/sport/*.txt")
    for doc_name in book:
        with open(doc_name,"r") as file:
            category.append('sport')
            document.append(file.read())
            data_frame=pd.DataFrame({'category':category,'document':document})
            data_frame=shuffle(data_frame)
    print(data_frame.head(1))
    return data_frame
def tokenizing_document(data_frame):
    print("TOKENIZING DOCUMENTS...")

    data_frame['document']=data_frame['document'].fillna("").map(nltk.word_tokenize)
    print(data_frame.head(1))
    return data_frame
def stemming_documents(data_frame):
   
    print("STEMMING DOCUMENTS...")
    data_frame['document']=[[SnowballStemmer('english').stem(word) for word in sentence] for sentence in data_frame['document']]
    print(data_frame.head(1))
    return data_frame
    
def removing_stopwords(data_frame):
    print('REMOVING STOPWORDS...')
    stop=stopwords.words('english')
    data_frame['document']=[[word for word in sentence if word not in stop] for sentence in data_frame['document']]
    print(data_frame.head(1))
    return data_frame
def removing_punctutions(data_frame):
    print("REMOVING PUNCTUTIONS...")
    data_frame['document']=[[word.lower() for word in sentence if word.isalpha()]for sentence in data_frame['document']]
    print(data_frame.head(1))
   

def main():
    data_frame=pd.DataFrame()
    data_frame=importing_data()
    data_frame=tokenizing_document(data_frame)
    data_frame=stemming_documents(data_frame)
    data_frame=removing_stopwords(data_frame)
    data_frame=removing_punctutions(data_frame)
if __name__ ==  "__main__":
    main()
    