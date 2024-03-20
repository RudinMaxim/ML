from os import name

from flask import Flask, request, jsonify
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Предположим, что у вас есть DataFrame 'costs_visitors' с нужными данными
# и 'revenue' DataFrame с целевой переменной.

# Объединение данных в один DataFrame
data = pd.merge(costs_visitors, revenue, on='dt')

# Выбор признаков для модели
features = data[['costs', 'visitors']]  # Здесь вы должны указать фактические признаки
target = data['revenue']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение модели в файл
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Загрузка предварительно обученной модели
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(name)


@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных от пользователя
    data = request.get_json(force=True)

    # Преобразование данных в DataFrame
    df = pd.DataFrame(data, index=[0])

    # Предсказание с помощью модели
    predictions = model.predict(df)

    # Возвращение предсказаний
    return jsonify(predictions.tolist())


if name == 'main':
    app.run(debug=True)
