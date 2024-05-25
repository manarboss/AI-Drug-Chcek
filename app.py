from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS  # Import flask_cors
import json
import pandas as pd

app = Flask(__name__)

CORS(app)

with open("output.json", mode='r', encoding='utf-8') as file:
    data = json.load(file)


app = Flask(__name__, static_folder='vite-project/dist/assets', static_url_path='/assets')

@app.route('/')
def index():
    return send_from_directory('vite-project/dist', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('vite-project/dist', filename)


@app.route('/api/medication', methods=['GET'])
def get_medication_info():
    medication_name = request.args.get('name')
    print(medication_name)

    if medication_name in data:
        return data[medication_name]
    else:
        return [], 200





if __name__ == '__main__':
    CORS(app)
    app.run(debug=True)






