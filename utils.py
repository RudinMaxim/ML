import json
from datetime import date, datetime
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xml.etree.ElementTree as ET
import re


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('russian'))

def datetime_handler(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4, default=datetime_handler)

def to_csv(data, filename):
    preprocessed_data = [{
        'title': d['title'],
        'description': preprocess_text(d['description']),
        'rating': d['rating'],
        'industry': d['industry'],
        'pub_date': d['pub_date'],
        'article_text': preprocess_text(d['article_text'])
    } for d in data]
    df = pd.DataFrame(preprocessed_data)
    df.to_csv(filename, index=False, encoding='utf-8')

def to_xml(data, filename):
    root = ET.Element('data')
    for item in data:
        article = ET.SubElement(root, 'article')
        title = ET.SubElement(article, 'title')
        title.text = item['title']
        description = ET.SubElement(article, 'description')
        description.text = item['description']
        rating = ET.SubElement(article, 'rating')
        rating.text = item['rating']
        industry = ET.SubElement(article, 'industry')
        industry.text = item['industry']
        pub_date = ET.SubElement(article, 'pub_date')
        pub_date.text = str(item['pub_date']) if item['pub_date'] else ''
        article_text = ET.SubElement(article, 'article_text')
        article_text.text = item['article_text']
    tree = ET.ElementTree(root)
    tree.write(filename, encoding='utf-8', xml_declaration=True)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if not word in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)
