from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS  # Import flask_cors
import json
import pandas as pd
from ai import  train_knn_model
from ai import find_least_side_effects_drug
from ai import check_stock


app = Flask(__name__)

CORS(app)

with open("output.json", mode='r', encoding='utf-8') as file:
    data = json.load(file)


app = Flask(__name__, static_folder='vite-project/dist/assets', static_url_path='/assets')

@app.route('/')
def index():
    return send_from_directory('vite-project/dist', 'vite-project/index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('vite-project/dist', filename)


@app.route('/api/medication', methods=['GET'])
def get_medication_info():
    medication_name = request.args.get('name')
    for entry in data:
        medication = entry.get("medication")
        if medication and medication['name'].lower() == medication_name.lower():
            return jsonify(medication)
    return jsonify({"error": "Medication not found"}), 404
        




@app.route('/api/prediction', methods=['GET'])
def get_prediction():
    medication_name = request.args.get('name')
    if medication_name in data:
        return jsonify(data[medication_name])
    else:
        return jsonify([]), 200



@app.route('/api/train_model', methods=['POST'])
def train_model():
    accuracy, report = train_knn_model()
    return jsonify({"accuracy": accuracy, "report": report})

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/api/generic', methods=['POST'])
def generic():
    data = request.get_json()
    print(data)
    return jsonify(find_least_side_effects_drug(data))







