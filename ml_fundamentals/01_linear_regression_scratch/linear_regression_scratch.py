"""
Примеры формулировок задач, к которым применяется линейная регрессия - ПРЕДСКАЗАНИЕ:
MSE - среднеквадратическая ошибка - 1/n * sum((y_pred-y)**2) 
    1. Спрогнозируй цену квартиры в Москве на основе площади, этажа, района и года постройки.
    2. Ожидаемое время прибытия такси (ETA) в зависимости от расстояния, пробок и времени суток.
MAE - средняя абсолютная ошибка - 1/n *sum(|y_pred-y|) - выбросы в данных не должны сильно влиять на модель
    1. Прогноз количества посетителей магазина завтра. Если один день было аномально много людей (фестиваль), 
        не хотим, чтобы модель сильно дёргалась.
    2. Оценка возраста человека по фотографии — ошибка в 10 лет не в 100 раз хуже ошибки в 1 год.
    3. В данных о доходах населения есть выбросы (миллиардеры). Какую функцию потерь выбрать для предсказания зарплаты?"
        Ответ: MAE, так как MSE сильно зависит от выбросов.
"""


import numpy as np                                  # работа с числами и матрицами
import matplotlib.pyplot as plt                     # визуализация графиков
from sklearn.linear_model import LinearRegression    # готовая модель для сравнения

# Генерирую данные
np.random.seed(42)
X = 2 * np.random.rand(100, 1)              # 100 случайных чисел от 0 до 2
y = 3*X + 5 + np.random.randn(100, 1)*0.5   # у = 3*х + 5 + шум

# Линейная регрессия (градиентный спуск)
class LinearRegressionGD:
    def __init__(self, learning_rate=0.5, n_iterrations=500):   #  learning_rate - параметр, который определяет размер шага при обновлении параметров модели
        self.lr = learning_rate
        self.n_iter = n_iterrations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, x, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features, 1)
        self.bias = 0
        for i in range(self.n_iter):
            y_pred = X @ self.weights + self.bias
            loss = np.mean ((y_pred-y)**2)              # np.mean - сумма всех и деление на их количество
            self.losses.append(loss)

            dw = (2/n_samples)*(X.T @ (y_pred-y))
            db = (2/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr *dw                 # шаг градиентного спуска
            self.bias -= self.lr*db                     # шаг градиентного спуска

    def predict(self, X):
        return X @ self.weights + self.bias


# Обучаем нашу модель
model = LinearRegressionGD()
model.fit(X, y)
y_pred = model.predict(X)

# Обучаем sklearn модель для сравнения
sk_model = LinearRegression()
sk_model.fit(X, y)

# Результаты
print("=" * 40)
print("Сравнение результатов:")
print(f"Наша модель:      w = {model.weights[0][0]:.3f}, b = {model.bias:.3f}")
print(f"Sklearn модель:   w = {sk_model.coef_[0][0]:.3f}, b = {sk_model.intercept_[0]:.3f}")
print(f"Истинная функция: w = 3.000, b = 5.000")
print("=" * 40)

# Графики
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # два графика на одном табло

ax1.scatter(X, y, alpha=0.7, label='Данные')
ax1.plot(X, y_pred, 'r-', label='Наша модель')
ax1.plot(X, sk_model.predict(X), 'g--', label='Sklearn модель')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.legend()
ax1.set_title('Линейная регрессия')

ax2.plot(model.losses)
ax2.set_xlabel('Итерация')
ax2.set_ylabel('MSE Loss')
ax2.set_title('Сходимость градиентного спуска')
ax2.grid(True, alpha=0.3)

plt.tight_layout()  # Чтобы графики не налезали друг на друга.
plt.savefig('ml_fundamentals/01_linear_regression_scratch/result.png') # сохранение графиков
plt.show()