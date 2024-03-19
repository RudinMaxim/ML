import os
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import constants
from utils import to_json, to_csv, to_xml
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


class DataParser:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('russian'))

    def web_parser(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            articles = soup.find_all(['article', 'main'])
            data = []

            for element in articles:
                title = element.find(['h1', 'h2', 'h3'])
                title_text = title.text.strip() if title else None

                content_elements = element.find_all(['p', 'div'])
                description = ' '.join(elem.text.strip() for elem in content_elements) or None

                # Количество просмотров
                rating_element = element.find('span', class_='tm-icon-counter')
                rating = rating_element.text.strip() if rating_element else None

                # Извлечение сферы деятельности
                industry_element = element.find('span', class_='industry')
                industry = industry_element.text.strip() if industry_element else None

                # Извлечение даты публикации
                date_element = element.find('time')
                date_str = date_element.get('datetime') if date_element else ''
                pub_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ') if date_str else None

                # Извлечение полного текста статьи
                article_text_elements = element.find_all('p', class_='article-text')
                article_text = ' '.join(elem.text.strip() for elem in article_text_elements)

                data.append({
                    'title': title_text,
                    'description': description,
                    'rating': rating,
                    'industry': industry,
                    'pub_date': pub_date,
                    'article_text': article_text
                })

            return data
        except requests.exceptions.RequestException as e:
            print(f'Ошибка при запросе к {url}:', e)
            return []




if __name__ == '__main__':
    # Парсинг
    parser = DataParser()
    design = parser.web_parser('https://habr.com/ru/flows/design/articles/')
    develop = parser.web_parser('https://habr.com/ru/flows/develop/articles/')
    to_json(design, 'data/design.json')
    to_json(develop, 'data/develop.json')



