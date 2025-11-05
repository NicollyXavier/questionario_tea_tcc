import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

# 1. Carregar dataset
df = pd.read_csv("dataset_simulado_MCHATR_deterministico.csv")

# 2. Separar entradas (X) e saída (y)
X = df.drop(columns=["Risco"])
y = df["Risco"]

# 3. Converter rótulos em números
y = y.map({"baixo": 0, "moderado": 1, "alto": 2})

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Criar e treinar o modelo
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# 6. Fazer previsões e avaliar
y_pred = modelo.predict(X_test)
print("\n🔹 Relatório de desempenho do modelo:")
print("Classes presentes no conjunto de teste:", sorted(y_test.unique()))
print(classification_report(y_test, y_pred))

# 7. Teste com exemplo novo
exemplo = [[1,0,1,0,1,0,1,1,0,1,0,1]]  # respostas simuladas
resultado = modelo.predict(exemplo)[0]
risco = {0: "baixo", 1: "moderado", 2: "alto"}[resultado]
print(f"\n🧩 Resultado previsto para o exemplo: {risco.upper()}")

# 8. Matriz de confusão
ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test)
plt.show()

# 9. Salvar o modelo treinado
joblib.dump(modelo, "modelo_mchat.pkl")

print("\n💾 Modelo salvo como modelo_mchat.pkl")
