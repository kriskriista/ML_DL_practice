"""
Берутся реальные данные по домам в Калифорнии 1990 год.

Целевая переменная: MedHouseVal — медианная стоимость дома в квартале (в сотнях тысяч долларов).

8 признаков (они все в разных единицах измерения):
    MedInc — медианный доход в квартале
    HouseAge — медианный возраст дома
    AveRooms — среднее количество комнат
    AveBedrms — среднее количество спален
    Population — население квартала
    AveOccup — средняя заполненность дома
    Latitude — широта
    Longitude — долгота

Задача: прдсказать стоимость дома по признакам.

Признак с большим разбросом значений будет доминировать. 
Для этого нужно масштабирование признаков (Standard Scaler) - делается после разделения 
на тестовую и тренировочную выборки.

# # test_size - 20%  , random_state=42 - фиксирует случайность, чтобы результат можно было воспроизвести
# # X_train, X_test y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
"""
#-------------------------------------------------------------------------
# Первая часть - загружаем данные из датасета

import numpy as np
import pandas as pd 
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing.data 
y = housing.target
feature_names = housing.feature_names

print("Размер X: ", X.shape)            # (20640 - домов, 8 - признаков)
print("Признаки: ", feature_names)
print("Первые 5 строк X: ")
print(X[:5])
print("Первые 5 значений y: ", y[:5])

# Если данные из csv, то:
# import pandas as pd
# X = pd.read_csv('data.csv')  # ваши данные

#-------------------------------------------------------------------------
# Вторая часть - делим выборку

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

# train_test_split случайным образом перемешивает данные перед разделением.
# random_state — это число-зерно, которое фиксирует случайность.
# Без random_state - каждый запуск даёт разное разделение, получим разные результаты → не сможем повторить эксперимент.

print("Train размер: ", X_train.shape)      # (16512, 8)
print("Test размер: ", X_test.shape)        # (4128, 8)

#-------------------------------------------------------------------------
# Третья часть - масштабирование признаков

from sklearn.preprocessing  import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # учим параметры на train
X_test_scaled = scaler.transform(X_test)        # применяем к test

# fit_transform — учит параметры (среднее, std) и сразу применяет их к тем же данным.
# transform — только применяет (использует уже выученные параметры)

# 1. scaler.fit(X_train) — вычисляем среднее и std только на обучающих данных
# 2. X_train_scaled = scaler.transform(X_train) — масштабируем train
# 3. X_test_scaled = scaler.transform(X_test) — масштабируем test теми же параметрами

print("Среднее до масштабирования:", X_train[:, 0].mean())
print("Среднее после масштабирования:", X_train_scaled[:, 0].mean())
print("Станд. отклонение после масштабирования:", X_train_scaled[:, 0].std())

#-------------------------------------------------------------------------
# 4 часть - обучаем первую модель - лин регрессию из sklearn

from sklearn.linear_model import LinearRegression

# Создаём модель
sk_model = LinearRegression()

# Обучаем на масштабированных данных
sk_model.fit(X_train_scaled, y_train)

print("Обучение завершено")
print("Коэффициенты (веса) для каждого признака:")
for name, coef in zip(feature_names, sk_model.coef_):  
    # sk_model.coef_ - массив весов (w) для каждого признака
    print(f"  {name}: {coef:.4f}")
print(f"Свободный член (bias): {sk_model.intercept_:.4f}")
# model.intercept_ — свободный член (b)

#-------------------------------------------------------------------------
# 5 часть - предсказание и рассчет метрик

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Предсказываем на train и test
y_train_pred = sk_model.predict(X_train_scaled)
y_test_pred = sk_model.predict(X_test_scaled)

# Считаем метрики
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ЛИНЕЙНОЙ РЕГРЕССИИ")
print("="*50)
print(f"MSE на train: {train_mse:.4f}")
print(f"MSE на test:  {test_mse:.4f}")
print(f"MAE на train: {train_mae:.4f} (ошибка ~${train_mae*100:.0f}K)")
print(f"MAE на test:  {test_mae:.4f} (ошибка ~${test_mae*100:.0f}K)")
print(f"R² на train:  {train_r2:.4f}")
print(f"R² на test:   {test_r2:.4f}")

# MSE — среднеквадратичная ошибка (чем меньше, тем лучше)
# MAE — средняя абсолютная ошибка (понятнее: мы ошибаемся в среднем на ±$MAE*100 тысяч долларов)
# R² — доля объяснённой дисперсии (1.0 — идеал, 0 — хуже, чем просто среднее)

#-------------------------------------------------------------------------
# 6 часть - добавляем Ridge-регрессию (L2-регуляризация)

from sklearn.linear_model import Ridge

# Создаём Ridge модель (alpha=1.0 — сила регуляризации)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Предсказания
y_train_pred_ridge = ridge_model.predict(X_train_scaled)
y_test_pred_ridge = ridge_model.predict(X_test_scaled)

# Метрики
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ RIDGE (L2-РЕГУЛЯРИЗАЦИЯ)")
print("="*50)
print(f"MSE на train: {mean_squared_error(y_train, y_train_pred_ridge):.4f}")
print(f"MSE на test:  {mean_squared_error(y_test, y_test_pred_ridge):.4f}")
print(f"R² на train:  {r2_score(y_train, y_train_pred_ridge):.4f}")
print(f"R² на test:   {r2_score(y_test, y_test_pred_ridge):.4f}")

# Результаты Ridge должны быть очень близки к обычной линейной регрессии 
# (потому что данные не сильно переобучаются). Но в задачах с тысячами признаков 
# Ridge помогает сильно.

#-------------------------------------------------------------------------
# 7 часть - визуализация — график "предсказание vs реальность"

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Для обычной линейной регрессии
ax1.scatter(y_test, y_test_pred, alpha=0.3, s=1)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax1.set_xlabel("Реальная стоимость")
ax1.set_ylabel("Предсказанная стоимость")
ax1.set_title("Linear Regression")

# Для Ridge
ax2.scatter(y_test, y_test_pred_ridge, alpha=0.3, s=1, color='green')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax2.set_xlabel("Реальная стоимость")
ax2.set_ylabel("Предсказанная стоимость")
ax2.set_title("Ridge (L2)")

plt.tight_layout()
plt.savefig('comparison_plot.png')
plt.show()

# Каждая точка — один дом. Красная линия — идеальное предсказание 
# (предсказание = реальность). Чем ближе точки к линии, тем лучше модель.
