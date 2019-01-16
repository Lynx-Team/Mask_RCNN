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
    err_msg = ''
    if not input_file:
        err_msg = 'File is not present in request'
    elif input_file.filename == '':
        err_msg = 'File name is not present in request'
    elif not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        err_msg = 'Invalid filename: ' + input_file.filename
    else:
        return None

    stderr.write(err_msg + '\n')
    return BadRequest(err_msg)

def upload_file(request):
    if 'file' not in request.files:
        stderr.write('Expected file in request.files\n')
        return (None, BadRequest('Expected file in request'))

    file = request.files.get('file')
    bad_request = validate(file)
    if bad_request is not None:
        return (None, bad_request)

    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(path)
    return (path, None)

@app.route('/products', methods=['POST'])
def products_from_picture():
    try:
        (input_file, bad_request) = upload_file(request)
        if bad_request is not None:
            return bad_request
        else:
            result = model.eval(input_file)
            result.save(os.path.join(app.config['OUTPUT_FOLDER'], 'last_json_inference.jpg'))
            return result.to_json()
    except Exception as e:
        stderr.write(str(e) + '\n')
        return BadRequest('Cannot recognize products')

@app.route('/image', methods=['POST'])
def image():
    try:
        (input_file, bad_request) = upload_file(request)
        if bad_request is not None:
            return bad_request
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(input_file))
        model.eval(input_file).save(output_file)
        return send_file(output_file, mimetype='image/jpeg')
    except Exception as e:
        stderr.write(str(e) + '\n')
        return BadRequest('Cannot recognize products')

if __name__ == '__main__':
    app.run(threaded=False)
