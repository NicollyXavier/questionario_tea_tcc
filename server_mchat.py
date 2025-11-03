from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

# Carregar modelo
modelo = joblib.load("modelo_mchat.pkl")

# Inicializar o app Flask
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Servidor M-CHAT-R/F com IA está rodando ✅"

@app.route("/avaliar", methods=["POST"])
def avaliar():
    try:
        dados = request.get_json()
        respostas = dados["respostas"]

        # Criar DataFrame com as respostas
        df = pd.DataFrame([respostas], columns=[
            'P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12'
        ])

        # Fazer previsão
        pred = modelo.predict(df)[0]
        risco = {0: "baixo", 1: "moderado", 2: "alto"}[pred]

        return jsonify({"risco": risco})
    except Exception as e:
        return jsonify({"erro": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
