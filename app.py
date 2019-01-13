from flask import Flask, make_response, request, render_template, jsonify, send_file

from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
import base64
import os

from products.products import CooksterNN, EvalResult

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images/'
app.config['OUTPUT_FOLDER'] = 'output/'
model = CooksterNN()

def validate(input_file):
    if not input_file:
        return BadRequest("File not present in request")
    if input_file.filename == '':
        return BadRequest("File name is not present in request")
    if not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return BadRequest("Invalid file type")
    return None

def upload_file(request):
    if 'file' not in request.files:
        return (None, BadRequest("Expected file"))

    file = request.files.get('file')
    bad_request = validate(file)
    if bad_request is not None:
        return (None, bad_request)

    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(path)
    return (path, None)

@app.route('/products', methods=['POST'])
def products_from_picture():
    (input_file, bad_request) = upload_file(request)
    if bad_request is not None:
        return bad_request
    else:
        return model.eval(input_file).to_json()

@app.route('/image', methods=['POST'])
def image():
    (input_file, bad_request) = upload_file(request)
    if bad_request is not None:
        return bad_request

    output_file = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(input_file))
    model.eval(input_file).save(output_file)
    return send_file(output_file, mimetype='image/jpeg')


if __name__ == "__main__":
    app.run(threaded=False)