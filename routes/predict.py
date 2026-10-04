from flask import Blueprint, request, jsonify
from services.prediction_service import predict_data

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    result = predict_data(data)
    return jsonify(result)