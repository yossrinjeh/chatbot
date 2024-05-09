from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import io
from chat import get_response

app = Flask(__name__)
CORS(app)

# Global variable to store result_group
result_group = None

@app.post("/predict")
def predict():
  text = request.get_json().get("message")
  response = get_response(text)
  message = {"answer": response}
  return jsonify(message)


if __name__ == "__main__":
  app.run(debug=True)
