from flask import Flask, request, render_template, flash, redirect, session, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import image_processing_2
from yolov3_flask_image_detection import yolo_detection_img

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'jamal123'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS

def save_processed_image(processed_img, prefix, filename):
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{prefix}_{filename}')
    cv2.imwrite(processed_image_path, processed_img)
    processed_image_url = url_for('uploaded_image', filename=f'{prefix}_{filename}')
    return processed_image_url

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    uploaded_file_url = None
    grayscale_image_url = None
    inverted_image_url = None
    binarized_image_url = None
    img_with_contours_url = None
    detected_number_plate_image_url = None
    dilated_image_url = None
    blurred_image_url = None

    if request.method == 'POST':
        if 'car_image' not in request.files:
            flash('No file chosen')
            return redirect(request.url)
        file = request.files['car_image']

        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Save the filename in the session variable
            session['uploaded_file'] = filename
            print(f'filepath:{file_path}')

            # YOLOV3 OBJECT DETECTION
            cropped_img, detected_number_plate = yolo_detection_img(file_path)

            grayscale_img = image_processing_2.image_to_grayscale(cropped_img)
            blurred_img = image_processing_2.blur_img(grayscale_img)
            inverted_img = image_processing_2.invert_color(blurred_img)
            binary_img = image_processing_2.binarize_img(inverted_img)
            dilated_image = image_processing_2.dilate_image(binary_img)
            img_with_contours = image_processing_2.find_contours(dilated_image, cropped_img)

            uploaded_file_url = url_for('uploaded_image', filename=filename)
            detected_number_plate_image_url = save_processed_image(detected_number_plate, 'detected_number_plate', filename)
            grayscale_image_url = save_processed_image(grayscale_img, 'grayscale', filename)
            blurred_image_url = save_processed_image(blurred_img, 'blurred', filename)
            inverted_image_url = save_processed_image(inverted_img, 'inverted', filename)
            binarized_image_url = save_processed_image(binary_img, 'binarized', filename)
            dilated_image_url = save_processed_image(dilated_image, 'dilated', filename)
            img_with_contours_url = save_processed_image(img_with_contours, 'processed', filename)

    return render_template('index.html', uploaded_file_url=uploaded_file_url,
                           grayscale_image_url=grayscale_image_url,
                           blurred_image_url=blurred_image_url,
                           inverted_image_url=inverted_image_url,
                           binarized_image_url=binarized_image_url,
                           dilated_image_url=dilated_image_url,
                           detected_number_plate_image_url=detected_number_plate_image_url,
                           img_with_contours_url=img_with_contours_url)

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='localhost', port=5891)
