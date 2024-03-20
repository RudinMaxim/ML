import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Считывание данных
expenses = pd.read_csv('expenses.csv')
visitation = pd.read_csv('visitation.csv')
orders = pd.read_csv('orders.csv')

# Разведочный анализ данных
expenses['dt'] = pd.to_datetime(expenses['dt'])
print(expenses['costs'].describe())

plt.figure(figsize=(12, 6))
sns.lineplot(x='dt', y='costs', data=expenses)
plt.title('Advertising Costs on FaceBoom Channel')
plt.xlabel('Date')
plt.ylabel('Cost')
plt.show()

# Анализ связи затрат с количеством посетителей
visitation_fb = visitation[visitation['Channel'] == 'FaceBoom']
visitors = visitation_fb.groupby(pd.to_datetime(visitation_fb['Session Start']).dt.date).size().reset_index()
visitors.columns = ['dt', 'visitors']

# Убедимся, что столбец 'dt' в DataFrame 'visitors' имеет тип datetime64[ns]
visitors['dt'] = pd.to_datetime(visitors['dt'])

# Теперь можно безопасно объединять, так как оба столбца 'dt' будут иметь тип datetime64[ns]
costs_visitors = pd.merge(expenses, visitors, on='dt', how='left')
costs_visitors = costs_visitors.fillna(0)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
sns.lineplot(x='dt', y='costs', data=costs_visitors)
plt.title('Advertising Costs')

plt.subplot(2, 1, 2)
sns.lineplot(x='dt', y='visitors', data=costs_visitors)
plt.title('Daily Visitors from FaceBoom')
plt.tight_layout()
plt.show()

# Анализ данных о заказах и выручке
orders['Event Dt'] = pd.to_datetime(orders['Event Dt'])
revenue = orders.groupby(pd.to_datetime(orders['Event Dt']).dt.date)['Revenue'].sum().reset_index()
revenue.columns = ['dt', 'revenue']

plt.figure(figsize=(12, 6))
sns.lineplot(x='dt', y='revenue', data=revenue)
plt.title('Daily Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.show()

# Визуализация закономерностей
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
sns.lineplot(x='dt', y='costs', data=costs_visitors)
plt.title('Advertising Costs')

plt.subplot(3, 1, 2)
sns.lineplot(x='dt', y='visitors', data=costs_visitors)
plt.title('Daily Visitors from FaceBoom')

plt.subplot(3, 1, 3)
sns.lineplot(x='dt', y='revenue', data=revenue)
plt.title('Daily Revenue')
plt.tight_layout()
plt.show()