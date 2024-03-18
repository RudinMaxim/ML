import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xml.etree.ElementTree as ET
# ========
import constants


class DataParser:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('russian'))
    def parse_habrahabr(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = soup.find_all('article')
        data = []
        for article in articles:
            title = article.find('h2').text if article.find('h2') else 'No title'
            content_div = article.find('div', class_='article-formatted-body')
            description = content_div.text if content_div else 'No description'
            data.append({'title': title, 'description': description})
        return data

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if not word in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    def to_csv(self, data, filename):
        preprocessed_data = [{'title': d['title'], 'description': self.preprocess_text(d['description'])} for d in data]
        df = pd.DataFrame(preprocessed_data)
        df.to_csv(filename, index=False)

    def to_xml(self, data, filename):
        root = ET.Element('data')
        for item in data:
            article = ET.SubElement(root, 'article')
            title = ET.SubElement(article, 'title')
            title.text = item['title']
            description = ET.SubElement(article, 'description')
            description.text = item['description']
        tree = ET.ElementTree(root)
        tree.write(filename, encoding='utf-8', xml_declaration=True)


parser = DataParser()
data = parser.parse_habrahabr(constants.url)
parser.to_xml(data, 'data/selectel.xml')
