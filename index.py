import os
from datetime import datetime
from itertools import chain

import requests
from bs4 import BeautifulSoup
import nltk
from nltk import ngrams, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils import to_json, to_csv, to_xml
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

class DataAnalysis:
    def __init__(self, paths_data):
        self.paths_data = paths_data
        self.data = {}
        self.load_data()

    def load_data(self):
        for path in self.paths_data:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл {path} не найден!")

            file_extension = path.split('.')[-1]

            if file_extension == 'json':
                self.data[path] = pd.read_json(path)
            elif file_extension == 'csv':
                self.data[path] = pd.read_csv(path)
            elif file_extension == 'xml':
                self.data[path] = pd.read_xml(path)
            else:
                raise ValueError("Формат не поддерживается!")

    def extract_ngrams(self, data, n=1, top=10):
        vectorizer = CountVectorizer(ngram_range=(n, n))
        X = vectorizer.fit_transform(data['article_text'])
        feature_names = vectorizer.get_feature_names_out()
        ngram_counts = X.sum(axis=0).A1
        ngram_counts_sorted = sorted([(count, feature) for feature, count in zip(feature_names, ngram_counts)],
                                     reverse=True)
        top_ngrams = [feature for count, feature in ngram_counts_sorted[:top]]
        data[f'top_{n}_grams'] = top_ngrams
        return data

    def vectorize_text(self, data, method='tfidf'):
        if method == 'tfidf':
            vectorizer = TfidfVectorizer()
        else:
            vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data['article_text'])
        return X

    def topic_modeling(self, data, method='lda', n_topics=10):
        X = self.vectorize_text(data)
        if method == 'lda':
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        else:
            model = NMF(n_components=n_topics, random_state=42)
        topic_distr = model.fit_transform(X)
        return topic_distr

    def clustering(self, data, method='kmeans', n_clusters=5):
        X = self.vectorize_text(data)
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            model = DBSCAN()
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        data['cluster'] = labels
        return data

    def evaluate_clustering(self, data, method='silhouette'):
        X = self.vectorize_text(data)
        labels = data['cluster']
        if method == 'silhouette':
            score = silhouette_score(X, labels)
        elif method == 'calinski_harabasz':
            score = calinski_harabasz_score(X, labels)
        else:
            score = davies_bouldin_score(X, labels)
        return score

    def visualize_topics(self, data, topic_distr, n_topics=10):
        df_topic_distr = pd.DataFrame(topic_distr, columns=[f'Topic {i}' for i in range(n_topics)])
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_topic_distr.T, cmap='YlGnBu')
        plt.title('Topic Distribution')
        plt.show()

    def visualize_ngrams(self, data, n=1):
        text = ' '.join(data[f'top_{n}_grams'])
        wordcloud = WordCloud(background_color='white').generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Top {n}-grams')
        plt.show()

    def visualize_attributes(self, data):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        sns.distplot(data['rating'])
        plt.title('Rating Distribution')

        plt.subplot(2, 2, 2)
        data['pub_date'].hist()
        plt.title('Publication Date Distribution')

        plt.subplot(2, 2, 3)
        sns.countplot(data['industry'])
        plt.title('Industry Distribution')

        plt.tight_layout()
        plt.show()

    def exploratory_analysis(self, data):
        self.extract_ngrams(data, n=1)
        self.extract_ngrams(data, n=2)
        self.extract_ngrams(data, n=3)
        self.visualize_ngrams(data, n=1)
        self.visualize_ngrams(data, n=2)
        self.visualize_ngrams(data, n=3)
        topic_distr = self.topic_modeling(data, method='lda')
        self.visualize_topics(data, topic_distr)
        topic_distr = self.topic_modeling(data, method='nmf')
        self.visualize_topics(data, topic_distr)
        self.visualize_attributes(data)
        self.clustering(data, method='kmeans')
        self.clustering(data, method='dbscan')
        self.clustering(data, method='agglomerative')
        print('Silhouette Score:', self.evaluate_clustering(data, method='silhouette'))
        print('Calinski-Harabasz Score:', self.evaluate_clustering(data, method='calinski_harabasz'))
        print('Davies-Bouldin Score:', self.evaluate_clustering(data, method='davies_bouldin'))




if __name__ == '__main__':
    # Парсинг
    parser = DataParser()
    design = parser.web_parser('https://habr.com/ru/flows/design/articles/')
    develop = parser.web_parser('https://habr.com/ru/flows/develop/articles/')
    to_json(design, 'data/design.json')
    to_json(develop, 'data/develop.json')

    # Аналитика
    paths_data = ['data/design.json', 'data/develop.json']
    analysis = DataAnalysis(paths_data)
    
    analysis.exploratory_analysis(analysis.data['data/design.json'])
    analysis.exploratory_analysis(analysis.data['data/develop.json'])




