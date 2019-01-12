from flask import Flask, make_response, request, render_template, jsonify, send_file

from werkzeug.exceptions import BadRequest
import base64

from products import products

app = Flask(__name__)

# model = products.ldmodel(is_training=False, load_last=False)

def validate(input_file):
    if not input_file:
        return BadRequest("File not present in request")
    if input_file.filename == '':
        return BadRequest("File name is not present in request")
    if not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return BadRequest("Invalid file type")
    return None

@app.route('/products', methods=['POST'])
def products_from_picture():
    input_file = request.files.get('file')
    validation_result = validate(input_file)
    if validation_result is not None:
        return validation_result
    # products = model.eval()
    return jsonify({'products': [{'apple': 1}, {'orange': 2}]})

@app.route('/image', methods=['POST', 'GET'])
def image():
    input_file = request.files.get('file')
    validation_result = validate(input_file)
    if validation_result is not None:
        return validation_result
    # img = model.eval()
    img = './images/products.jpg'
    return send_file(img, mimetype='image/jpeg')

if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0')
