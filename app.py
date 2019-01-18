from flask import Flask, make_response, request, render_template, jsonify, send_file

from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from sys import stderr
import base64
import os

from products.products import CooksterNN, EvalResult, Weights, ROOT_DIR

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(ROOT_DIR, 'images')
app.config['OUTPUT_FOLDER'] = os.path.join(ROOT_DIR, 'output')

model = CooksterNN(training=False, weights=Weights.LAST)

def validate(input_file):
    if not input_file:
        raise ValueError('File is not present in request')
    elif input_file.filename == '':
        raise ValueError('File name is not present in request')
    elif not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise ValueError('Invalid filename: ' + input_file.filename)

def upload_file(request):
    if 'file' not in request.files:
        raise ValueError('Expected file in request.files')

    file = request.files.get('file')
    validate(file)
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(path)
    return path

def eval_and_save(request):
    input_file = upload_file(request)
    output_file = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(input_file))
    result = model.eval(input_file)
    result.save(output_file)
    return (result, output_file)

@app.route('/products', methods=['POST'])
def products_from_picture():
    try:
        (result, _) = eval_and_save(request)
        return result.to_json()
    except Exception as e:
        stderr.write(str(e) + '\n')
        return BadRequest('Cannot recognize products')

@app.route('/image', methods=['POST'])
def image_post():
    try:
        (_, output) = eval_and_save(request)
        return send_file(output, mimetype='image/jpeg')
    except Exception as e:
        stderr.write(str(e) + '\n')
        return BadRequest('Cannot recognize products')

@app.route('/image/<string:filename>', methods=['GET'])
def image_get(filename):
    try:
        filename = secure_filename(filename)
        validate(filename)
        file = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.isfile(file):
            raise ValueError('Cannot find ' + file)
        return send_file(file, mimetype='image/jpeg')
    except Exception as e:
        stderr.write(str(e) + '\n')
        return BadRequest('Cannot recognize products')

if __name__ == '__main__':
    app.run(threaded=False)
