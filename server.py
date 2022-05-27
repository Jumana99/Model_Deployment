from flask import Flask, request, jsonify
from deepLearningMdels import execute

# model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    # input_query = np.array([[text]])
    result = execute(text)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
