import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from utils import to_json, to_csv, to_xml


# ====================


class DataParser:
    """
        Класс DataParser инициализирует необходимые ресурсы NLTK (punkt, stopwords и wordnet) и
        создает объекты WordNetLemmatizer и множество стоп-слов для русского языка.
    """

    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('russian'))

    """
        Метод web_parser принимает URL-адрес, парсит ее содержимое с помощью BeautifulSoup 
        и извлекает релевантную информацию (заголовок, описание, рейтинг, сферу деятельности и дату публикации) из HTML. 
        Извлеченные данные возвращаются в виде списка словарей.
    """

    def web_parser(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            articles = soup.find_all(['article', 'main', 'section'])
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

                data.append({
                    'title': title_text,
                    'description': description,
                    'rating': rating,
                    'industry': industry,
                    'pub_date': pub_date,
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

    # Аналитика
