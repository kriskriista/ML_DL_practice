import numpy as np

X = np.array([1, 2, 3])
y = np.array([3, 5, 7])  # идеальная линия: w = 2, b = 0

w, b = 0.0, 0.0
lr = 0.01

for step in range(30):
    # предсказание
    y_pred = w * X + b

    # ошибка
    loss = np.mean((y_pred - y)**2)

    # градиенты
    dw = np.mean(2 * (y_pred-y) * X)
    db = np.mean(2 * (y_pred-y))

    # обновление
    w -= lr*dw
    b -= lr*db

    print(f"step {step}: w={w:.3f}, b={b:.3f}, loss={loss:.3f}")