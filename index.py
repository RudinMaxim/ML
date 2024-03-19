import os
from collections import Counter
from datetime import datetime
import re

import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk import ngrams, word_tokenize
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


class DataAnalysis:
    def __init__(self, paths_data):
        self.paths_data = paths_data
        self.data = {}
        self.load_data()

        self.lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('russian'))

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

        # 2.1 Поиск ключевых слов/n-грамм. Векторизация текстов

    def preprocess_text(self, text):
        """
        Функция для предварительной обработки текста
        """
        text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)  # Удаление знаков препинания и ненужных символов

        tokens = word_tokenize(text)  # Токенизация

        tokens = [word.lower() for word in tokens if
                  word.isalpha()]  # Приведение к нижнему регистру и удаление неалфавитных токенов

        tokens = [word for word in tokens if word not in self.stop_words]  # Удаление стоп-слов
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]  # Лемматизация
        return ' '.join(tokens)

    # 2.1 Поиск ключевых слов/n-грамм. Векторизация текстов
    def find_keywords(self, data, n=1):
        """
        Находит ключевые слова или n-граммы в тексте.
        """
        texts = data['description'].values
        tokens = [self.preprocess_text(text) for text in texts]
        ngram_tokens = [list(ngrams(token.split(), n)) for token in tokens]
        ngram_counts = Counter([gram for token in ngram_tokens for gram in token])
        keywords = [' '.join(gram) for gram, count in ngram_counts.most_common(10)]
        return keywords

    def vectorize_texts(self, data):
        """
        Векторизует тексты с помощью TF-IDF.
        """
        texts = data['description'].values
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        X = X.toarray()  # Преобразование разреженной матрицы в плотную
        return X, vectorizer.get_feature_names_out()

    # 2.2 Тематическое моделирование
    def topic_modeling(self, data, method='lda', n_topics=10, n_top_words=10):
        """
        Выполняет тематическое моделирование различными методами.
        """
        X, features = self.vectorize_texts(data)
        if method == 'lda':
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        elif method == 'nmf':
            model = NMF(n_components=n_topics, random_state=42)
        else:
            raise ValueError('Неизвестный метод тематического моделирования.')

        model.fit(X)
        topic_word = model.components_  # Получить топ слова для каждой темы
        vocab = np.array(features)
        topics = []
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
            topics.append(' '.join(topic_words))

        return topics

    def visualize_topics(self, data, topics, n_top_words=10):
        """
        Визуализирует результаты тематического моделирования.
        """
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < len(topics):
                topic_words = topics[i].split()[:n_top_words]
                wordcloud = WordCloud(background_color='white').generate(' '.join(topic_words))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'Topic {i + 1}', fontsize=16)
                ax.axis('off')
            else:
                ax.axis('off')
        plt.tight_layout()
        plt.show()

        # 2.3 Кластеризация

    def cluster_data(self, data, method='kmeans', n_clusters=5):
        """
        Выполняет кластеризацию данных различными методами.
        """
        X, features = self.vectorize_texts(data)
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError('Неизвестный метод кластеризации.')

        labels = model.fit_predict(X)
        return labels

    def evaluate_clustering(self, data, labels, metric='silhouette'):
        """
        Оценивает качество кластеризации с помощью выбранной метрики.
        """
        X, _ = self.vectorize_texts(data)
        if metric == 'silhouette':
            score = silhouette_score(X, labels)
        elif metric == 'calinski_harabasz':
            score = calinski_harabasz_score(X, labels)
        elif metric == 'davies_bouldin':
            score = davies_bouldin_score(X, labels)
        else:
            raise ValueError('Неизвестная метрика оценки качества кластеризации.')

        return score

        # 2.4 Разведочный анализ

    def visualize_distributions(self, data):
        """
        Визуализирует распределения признаков и целевой переменной.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        sns.distplot(data['rating'], ax=axes[0])
        axes[0].set_title('Распределение рейтингов')
        sns.countplot(data['industry'], ax=axes[1])
        axes[1].set_title('Распределение по отраслям')
        sns.distplot(data['pub_date'].dt.year, ax=axes[2])
        axes[2].set_title('Распределение по годам публикации')
        axes[3].remove()
        plt.tight_layout()
        plt.show()

    def visualize_text_features(self, data, topics):
        """
        Визуализирует зависимости темы от признаков текста.
        """
        X, features = self.vectorize_texts(data)
        topic_matrix = self.topic_modeling(data, method='lda', n_topics=len(topics))
        topic_matrix = topic_matrix.transform(X)
        for i, topic in enumerate(topics):
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 2, 1)
            sns.scatterplot(x=data['rating'], y=topic_matrix[:, i])
            plt.title(f'Topic {i + 1}: {topic[:30]}...')
            plt.xlabel('Rating')
            plt.ylabel('Topic Probability')

            plt.subplot(2, 2, 2)
            sns.boxplot(x=data['industry'], y=topic_matrix[:, i])
            plt.xticks(rotation=45)
            plt.xlabel('Industry')
            plt.ylabel('Topic Probability')

            plt.subplot(2, 2, 3)
            sns.scatterplot(x=data['pub_date'].dt.year, y=topic_matrix[:, i])
            plt.xlabel('Publication Year')
            plt.ylabel('Topic Probability')

            plt.subplot(2, 2, 4)
            wordcloud = WordCloud(background_color='white').generate(topic)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    def visualize_publication_stats(self, data):
        """
        Визуализирует статистику публикаций по времени.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=data['pub_date'].dt.year, data=data)
        plt.title('Количество публикаций по годам')
        plt.xlabel('Год')
        plt.ylabel('Количество публикаций')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.countplot(x=data['pub_date'].dt.month, data=data)
        plt.title('Количество публикаций по месяцам')
        plt.xlabel('Месяц')
        plt.ylabel('Количество публикаций')
        plt.show()


if __name__ == '__main__':
    # Парсинг
    parser = DataParser()
    design = parser.web_parser('https://habr.com/ru/flows/design/articles/')
    develop = parser.web_parser('https://habr.com/ru/flows/develop/articles/')
    to_json(design, 'data/design.json')
    to_json(develop, 'data/develop.json')

    # Аналитика
    data_paths = ['data/design.json', 'data/develop.json']
    analysis = DataAnalysis(data_paths)

    # 2.1 Поиск ключевых слов/n-грамм. Векторизация текстов
    for path in data_paths:
        data = analysis.data[path]
        keywords = analysis.find_keywords(data, n=1)  # Ключевые слова
        bigrams = analysis.find_keywords(data, n=2)  # Биграммы
        trigrams = analysis.find_keywords(data, n=3)  # Триграммы
        print(f"Ключевые слова для {path}:", keywords)
        print(f"Биграммы для {path}:", bigrams)
        print(f"Триграммы для {path}:", trigrams)

        X, features = analysis.vectorize_texts(data)
        # Дальнейшая обработка векторизованных текстов

    # 2.2 Тематическое моделирование
    for path in data_paths:
        data = analysis.data[path]
        topics_lda = analysis.topic_modeling(data, method='lda')
        topics_nmf = analysis.topic_modeling(data, method='nmf')
        print(f"Темы (LDA) для {path}:", topics_lda)
        print(f"Темы (NMF) для {path}:", topics_nmf)
        analysis.visualize_topics(data, topics_lda)
        analysis.visualize_topics(data, topics_nmf)

    # 2.3 Кластеризация
    for path in data_paths:
        data = analysis.data[path]
        labels_kmeans = analysis.cluster_data(data, method='kmeans')
        labels_dbscan = analysis.cluster_data(data, method='dbscan')
        labels_agglomerative = analysis.cluster_data(data, method='agglomerative')
        score_kmeans = analysis.evaluate_clustering(data, labels_kmeans)
        score_dbscan = analysis.evaluate_clustering(data, labels_dbscan)
        score_agglomerative = analysis.evaluate_clustering(data, labels_agglomerative)
        print(f"Качество кластеризации (k-means) для {path}: {score_kmeans}")
        print(f"Качество кластеризации (DBSCAN) для {path}: {score_dbscan}")
        print(f"Качество кластеризации (агломеративная) для {path}: {score_agglomerative}")

    # 2.4 Разведочный анализ
    for path in data_paths:
        data = analysis.data[path]
        analysis.visualize_distributions(data)
        topics = analysis.topic_modeling(data, method='lda')
        analysis.visualize_text_features(data, topics)
        analysis.visualize_publication_stats(data)
