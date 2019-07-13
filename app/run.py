from flask import Flask
from flask import render_template, request, jsonify
from keras.models import load_model
import cgi, os
import cgitb; cgitb.enable()
from detector import detector
import traceback


UPLOAD_FOLDER = 'static'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load model
ResNet_model = load_model('../saved_model/weights.best.ResNet50.2.hdf5')
model = detector(ResNet_model)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # render web page with plotly graphs
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go', methods=['POST'])
def go():
    classification_results = []
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':
            img_path = os.path.join(UPLOAD_FOLDER, photo.filename)         
            photo.save(img_path)
            title = ''
            try:
                prediction = model.breed_detector(img_path)
                if prediction is not None:
                    title = 'A {} was detected. The predicted/resembling breed is {}'.format(prediction[0], prediction[1])
                else:
                    title = 'Neither a Dog or a Human was detected'

            except Exception as e:
                title = 'Unable to process file: {}'.format(img_path)
                traceback.print_exc()

            classification_results.append((title, photo.filename))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3003, debug=False, threaded = False)


if __name__ == '__main__':
    main()