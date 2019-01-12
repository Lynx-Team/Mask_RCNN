from flask import Flask, make_response, request, render_template, jsonify, send_file

from werkzeug.exceptions import BadRequest
import base64

from products.products import CooksterNN, EvalResult

app = Flask(__name__)
model = CooksterNN()

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
    try:
        return model.eval('./images/products.jpg').to_json()
    except:
        return BadRequest("Unexpected exception")

@app.route('/image', methods=['POST', 'GET'])
def image():
    input_file = request.files.get('file')
    validation_result = validate(input_file)
    if validation_result is not None:
        return validation_result
    try:
        img = './images/products.jpg'
        output = './output/products.jpg'
        model.eval(img).save(output)
        return send_file(output, mimetype='image/jpeg')
    except:
        return BadRequest("Unexpected exception")

if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0')
