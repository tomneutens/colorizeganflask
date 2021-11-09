import os
import time
from flask import (Flask, flash, render_template, redirect, request, session,
                   send_file, url_for)

from bokeh.resources import CDN, INLINE
from bokeh.embed import file_html, autoload_static
from werkzeug.utils import secure_filename

from flask_utils import (is_allowed_file, generate_random_name, is_allowed_image)

from process_image import colorize

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Somereallyrandomthings573892%&*0' #os.environ['SECRET_KEY']
app.config['UPLOAD_FOLDER'] = './uploads' #os.environ['UPLOAD_FOLDER']


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']
        # if filename is empty, then assume no upload
        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        # check if the file is "legit"
        if image_file and is_allowed_file(image_file.filename):
            filename = secure_filename(generate_random_name(image_file.filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)
            time.sleep(1) # TODO: improve to file check
            # HACK: Defer this to celery, might take time
            passed = is_allowed_image(filepath) #make_thumbnail(filepath)
            if passed:
                return redirect(url_for('predict', filename=filename))
            else:
                return redirect(request.url)


@app.errorhandler(500)
def server_error(error):
    """ Server error page handler """
    return render_template('error.html'), 500

@app.route('/images/<filename>')
def images(filename):
    """ Route for serving uploaded images """
    if is_allowed_file(filename):
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    else:
        flash("File extension not allowed.")
        return redirect(url_for('home'))

@app.route('/predict/<filename>')
def predict(filename):
    """ After uploading the image, show the prediction of the uploaded image
    in barchart form
    """
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    script, div = colorize(image_path)
    fn = secure_filename(generate_random_name(filename))
    fn_path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    os.rename(image_path,fn_path)

    ln = INLINE

    return render_template(
        'predict.html' ,
        plot_script=script,
        plot_div=div
    )

