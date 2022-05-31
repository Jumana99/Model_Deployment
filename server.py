from flask import Flask, request, jsonify
from deepLearningMdels import execute

app = Flask(__name__)


@app.route("/")
def showHomePage():
    return "This is home page"


@app.route("/debug", methods=["POST"])
def debug():
    text = request.form["sample"]
    print(text)
    return "received"


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["sample"]
    result = execute(text)
    print(result)
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0")
