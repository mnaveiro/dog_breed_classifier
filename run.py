from flask import Flask
from flask import render_template, request, jsonify
from keras.models import load_model

app = Flask(__name__)


# load model
ResNet_model = load_model('./saved_model/weights.best.ResNet50.hdf5')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # render web page with plotly graphs
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go', methods=['POST'])
def go():
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html'
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()