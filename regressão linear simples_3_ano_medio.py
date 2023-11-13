import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dados de entrada
X = np.array([[5.8], [0.5], [1.9], [5.3]])
y = np.array([[3.6], [0.3], [1.5], [5.6]])

# Criando o modelo de regressão linear
modelo = LinearRegression()

# Treinando o modelo
modelo.fit(X, y)

# Coeficientes da reta de regressão
coef_angular = modelo.coef_[0][0]
coef_linear = modelo.intercept_[0]

# Fazendo previsões
y_pred = modelo.predict(X)

# Imprimindo os resultados
print('Coeficiente angular:', coef_angular)
print('Coeficiente linear:', coef_linear)

# Visualizando os dados e a reta de regressão
plt.scatter(X, y, color='blue', label='Pontos Originais')
plt.plot(X, y_pred, color='red', linewidth=2, label='Reta de Regressão')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regressão Linear Simples')
plt.legend()
plt.show()