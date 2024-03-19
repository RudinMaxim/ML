import json
from datetime import date, datetime
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xml.etree.ElementTree as ET
import re

# Инициализация лемматизатора
lemmatizer = WordNetLemmatizer()

# Инициализация, русских, стоп-слов
stop_words = set(stopwords.words('russian'))


# Обработки даты / времени
def datetime_handler(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()


# Функция для сохранения данных в формате CSV
def to_csv(data, filename):
    preprocessed_data = [{
        'title': d['title'],
        'description': preprocess_text(d['description']),
        'rating': d['rating'],
        'industry': d['industry'],
        'pub_date': d['pub_date'],
    } for d in data]
    df = pd.DataFrame(preprocessed_data)
    df.to_csv(filename, index=False, encoding='utf-8')


# Функция для сохранения данных в формате XML
def to_xml(data, filename):
    root = ET.Element('data')
    for item in data:
        article = ET.SubElement(root, 'article')
        title = ET.SubElement(article, 'title')
        title.text = item['title']
        description = ET.SubElement(article, 'description')
        description.text = preprocess_text(item['description'])
        rating = ET.SubElement(article, 'rating')
        rating.text = item['rating']
        industry = ET.SubElement(article, 'industry')
        industry.text = item['industry']
        pub_date = ET.SubElement(article, 'pub_date')
        pub_date.text = str(item['pub_date']) if item['pub_date'] else ''
    tree = ET.ElementTree(root)
    tree.write(filename, encoding='utf-8', xml_declaration=True)


# Функция для сохранения данных в формате JSON
def to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4, default=datetime_handler)


# Функция для предварительной обработки текста
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)  # Удаление знаков препинания и ненужных символов

    tokens = word_tokenize(text)  # Токенизация

    tokens = [word.lower() for word in tokens if
              word.isalpha()]  # Приведение к нижнему регистру и удаление неалфавитных токенов

    tokens = [word for word in tokens if not word in stop_words]  # Удаление стоп-слов
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Лемматизация
    return ' '.join(tokens)
