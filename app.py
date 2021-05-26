from flask import Flask, render_template, request, url_for, redirect, flash, session
from flask_sqlalchemy import SQLAlchemy
from tensorflow import keras
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from gevent.pywsgi import WSGIServer
import joblib
import urllib.request
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

app = Flask(__name__)
app.secret_key = '459d263db7b6ef8e86afcbc68689b3e5e1b32b5f526f0475'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///app.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

bcrypt = Bcrypt(app)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

login_manager = LoginManager(app)
login_manager.login_view = "sign_in"
login_manager.login_message_category = "info"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer(), primary_key=True)
    pname = db.Column(db.String(50), nullable = False)
    email = db.Column(db.String(50), nullable = False, unique=True)
    pswd_hash = db.Column(db.String(60), nullable=False)
    phone = db.Column(db.Integer, nullable=False)
    sec_ques = db.Column(db.String(200), nullable=False)
    sec_ans = db.Column(db.String(200), nullable=False)

    @property
    def password(self):
        return self.password

    @password.setter
    def password(self, plain_text_password):
        self.pswd_hash = bcrypt.generate_password_hash(plain_text_password).decode('utf-8')

    def check_password_correction(self, attempted_password):
        return bcrypt.check_password_hash(self.pswd_hash, attempted_password)

    def __repr__(self) -> str:
        return f'{self.pname}-{self.email}-{self.pswd}-{self.phone}-{self.sec_ques}-{self.sec_ans}'

class Patient(db.Model):
    pid = db.Column(db.Integer(), primary_key= True)
    pname = db.Column(db.String(200), nullable =False)
    page = db.Column(db.Integer(), nullable=False)
    gender = db.Column(db.String(10), nullable =False)
    pphno = db.Column(db.Integer(), nullable=False)
    pemail = db.Column(db.String(50), nullable=False)
    refer = db.Column(db.String(50), nullable=False)

    def __repr__(self) -> str:
        return f'{self.pid}-{self.pname}-{self.page}-{self.gender}-{self.pphno}-{self.pemail}-{self.refer}'

def insert_patient_details() :
    pname = request.form["pname"]
    page = request.form["page"]
    gender = request.form["pgender"]
    pphno = request.form["pphno"]
    pemail = request.form["pemail"]
    refer = request.form["refer"]

    isFound = Patient.query.filter_by(pemail=pemail).first()
    if isFound :
        isFound.pname = pname
        isFound.page = page
        isFound.gender = gender
        isFound.pphno = pphno
        isFound.pemail = pemail
        isFound.refer = refer

        db.session.add(isFound)
        db.session.commit()

        session["pid"] = isFound.pid
    
    else :
        patient = Patient(pname = pname, page=page,
        gender=gender, pphno=pphno, pemail=pemail, refer=refer)
        db.session.add(patient)
        db.session.commit()

def allowed_file(filename) :
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/home', methods=["GET", "POST"])
def home() :
    return render_template("home.html")

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/patient_details')
@login_required
def patient_details() :
    patients = Patient.query.all()
    return render_template("patient_details.html", patients=patients)

@app.route('/update_patient_details/<int:pid>', methods=["GET", "POST"])
@login_required
def update_patient_details(pid):
    if request.method == "POST" :
        pname = request.form["pname"]
        page = request.form["page"]
        gender = request.form["pgender"]
        pphno = request.form["pphno"]
        pemail = request.form["pemail"]
        refer = request.form["refer"]

        patient = Patient.query.filter_by(pid=pid).first()
        patient.pname = pname
        patient.page = page
        patient.gender = gender
        patient.pphno = pphno
        patient.pemail = pemail
        patient.refer = refer

        db.session.add(patient)
        db.session.commit()

        return redirect(url_for("patient_details"))

    patient = Patient.query.filter_by(pid=pid).first()
    return render_template("update_patient_details.html", patient=patient)

@app.route('/delete_patient_details/<int:pid>')
@login_required
def delete_patient_details(pid):
    patient = Patient.query.filter_by(pid=pid).first()

    db.session.delete(patient)
    db.session.commit()

    return redirect(url_for("patient_details"))



@app.route('/brain_tumor_detection', methods=["GET", "POST"])
@login_required
def brain_tumor_detection():
    if request.method == "POST" :
        insert_patient_details()

        classes = {0: 'Glioma Tumor', 1: 'Meningioma Tumor', 2: 'No Tumor', 3: 'Pituitary Tumor'}

        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256))

            image = np.reshape(image, (1, 256, 256, 1))

            model = keras.models.load_model('./static/models/brain_tumor.h5')
            pred = model.predict(image)
            pred = classes[np.argmax(pred)]
            
            session["title"] = "BRAIN TUMOR"
            session["pred"] = pred
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="BRAIN TUMOR")


@app.route('/breast_cancer_detection', methods=["GET", "POST"])
@login_required
def breast_cancer_detection():
    if request.method == "POST" :
        insert_patient_details()

        classes = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256))

            image = np.reshape(image, (1, 256, 256, 1))

            model = keras.models.load_model('./static/models/breast_cancer.h5')
            pred = model.predict(image)
            pred = classes[np.argmax(pred)]
            
            session["title"] = "BREAST CANCER"
            session["pred"] = pred
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="BREAST CANCER")

@app.route('/lung_cancer_detection', methods=["GET", "POST"])
@login_required
def lung_cancer_detection():
    if request.method == "POST" :
        insert_patient_details()

        classes = {0: 'Adenocarcinoma (Lung) Cancer', 1: 'No Lung Cancer', 2: 'Squamous Cell Carcinoma(Lung) Cancer'}
        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (134, 134))

            image = np.reshape(image, (1, 134, 134, 3))

            model = keras.models.load_model('./static/models/lung_cancer.h5')
            pred = model.predict(image)
            pred = classes[np.argmax(pred)]
            
            session["title"] = "LUNG CANCER"
            session["pred"] = pred
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="LUNG CANCER")



@app.route('/colon_cancer_detection', methods=["GET", "POST"])
def colon_cancer_detection():
    if request.method == "POST" :
        insert_patient_details()

        classes = {0: 'Dyed Lifted Polyps', 1: 'Dyed Resection Margins', 2: 'Esophagitis', 3: 'Normal Cecum', 4: 'Normal Pylorus', 5: 'Normal Z Line', 6: 'Polyps', 7: 'Ulcerative Colitis'}
        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (100 , 100))

            image = np.reshape(image, (1, 100, 100, 3))

            model = keras.models.load_model('./static/models/colon_cancer.h5')
            pred = model.predict(image)
            pred = classes[np.argmax(pred)]
            
            session["title"] = "COLON CANCER"
            session["pred"] = pred
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="COLON CANCER")

@app.route('/covid_detection', methods=["GET", "POST"])
@login_required
def covid_detection():
    classes = {0: 'COVID-19', 1: 'Lung Opacity', 2: 'Normal'}
    if request.method == "POST" :
        insert_patient_details()

        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256))

            image = np.reshape(image, (1, 256, 256, 1))

            model = keras.models.load_model('./static/models/covid.h5')
            pred = model.predict(image)
            pred = classes[np.argmax(pred)]
            
            session["title"] = "COVID-19"
            session["pred"] = pred
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="COVID-19")


@app.route('/giloma_tumor_detection', methods=["GET", "POST"])
@login_required
def giloma_tumor_detection():
    classes = {0: 'Glioma  Tumor', 1: 'No Tumor'}
    if request.method == "POST" :
        insert_patient_details()
        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256))

            image = np.reshape(image, (1, 256, 256, 1))

            model = keras.models.load_model('./static/models/giloma_tumor.h5')
            pred = model.predict(image)
            
            predictions = pred.copy()
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1
            predictions = classes[int(predictions[0][0])]
        
            session["title"] = "GILOMA TUMOR"
            session["pred"] = predictions
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="GILOMA TUMOR")

@app.route('/meningioma_tumor_detection', methods=["GET", "POST"])
@login_required
def meningioma_tumor_detection():
    classes = {0: 'Meningioma  Tumor', 1: 'No Tumor'}
    if request.method == "POST" :
        insert_patient_details()

        
        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256))

            image = np.reshape(image, (1, 256, 256, 1))

            model = keras.models.load_model('./static/models/meningioma_tumor.h5')
            pred = model.predict(image)
            
            predictions = pred.copy()
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1
            predictions = classes[int(predictions[0][0])]
            
            session["title"] = "MENINGIOMA TUMOR"
            session["pred"] = predictions
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="MENINGIOMA TUMOR")

@app.route('/pituitary_tumor_detection', methods=["GET", "POST"])
@login_required
def pituitary_tumor_detection():

    classes = {0: 'No Tumor', 1: 'Pituitary Tumor'}

    if request.method == "POST" :
        insert_patient_details()

        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256))

            image = np.reshape(image, (1, 256, 256, 1))

            model = keras.models.load_model('./static/models/pituitary_tumor.h5')
            pred = model.predict(image)

            predictions = pred.copy()
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1
            predictions = classes[int(predictions[0][0])]

            session["title"] = "PITUITARY TUMOR"
            session["pred"] = predictions
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="PITUITARY TUMOR")

@app.route('/leukemia_detection', methods=["GET", "POST"])
@login_required
def leukemia_detection():

    classes = {0: 'Acute Lymphobiastic Leukemia (ALL)', 1: 'No Acute Lymphobiastic Leukemia (ALL)'}

    if request.method == "POST" :
        insert_patient_details()

        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256))

            image = np.reshape(image, (1, 256, 256, 1))

            model = keras.models.load_model('./static/models/leukemia_grayscale.h5')
            pred = model.predict(image)
            
            predictions = pred.copy()
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1
            predictions = classes[int(predictions[0][0])]

            session["title"] = "ACUTE LYMPHOBIASTIC LEUKEMIA (ALL)"
            session["pred"] = predictions
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="ACUTE LYMPHOBIASTIC LEUKEMIA (ALL)")

@app.route('/idc_detection', methods=["GET", "POST"])
@login_required
def idc_detection():

    classes = {0: 'No Invasive Ductal Carcinoma (IDC)', 1: 'Invasive Ductal Carcinoma (IDC)'}

    if request.method == "POST" :
        insert_patient_details()

        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (50, 50))

            image = np.reshape(image, (1, 50, 50, 1))

            model = keras.models.load_model('./static/models/idc_grayscale.h5')
            pred = model.predict(image)
            
            predictions = pred.copy()
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1
            predictions = classes[int(predictions[0][0])]

            session["title"] = "INVASIVE DUCTAL CARCINOMA (IDC)"
            session["pred"] = predictions
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="INVASIVE DUCTAL CARCINOMA (IDC)")

@app.route('/malaria_detection', methods=["GET", "POST"])
@login_required
def malaria_detection():
    classes = {0: 'Malaria (Parasitized)', 1: 'No Malaria (Uninfected)'}
    if request.method == "POST" :
        insert_patient_details()
        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (134, 134))

            image = np.reshape(image, (1, 134, 134, 3))

            model = keras.models.load_model('./static/models/malaria.h5')
            pred = model.predict(image)

            predictions = pred.copy()
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1
            predictions = classes[int(predictions[0][0])]
            
            session["title"] = "MALARIA"
            session["pred"] = predictions
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="MALARIA")



@app.route('/pneumonia_detection', methods=["GET", "POST"])
@login_required
def pneumonia_detection():
    classes = {0: 'Normal', 1: 'Opacity'}
    if request.method == "POST" :
        insert_patient_details()

        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256))

            image = np.reshape(image, (1, 256, 256, 1))

            model = keras.models.load_model('./static/models/pneumonia.h5')
            pred = model.predict(image)

            predictions = pred.copy()
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1
            predictions = classes[int(predictions[0][0])]

            session["title"] = "PNEUMONIA"
            session["pred"] = predictions
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="PNEUMONIA")

@app.route('/tuberculosis_detection', methods=["GET", "POST"])
@login_required
def tuberculosis_detection():
    classes = {0: 'Normal', 1: 'Tuberculosis'}
    if request.method == "POST" :
        insert_patient_details()

        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256))

            image = np.reshape(image, (1, 256, 256, 1))

            model = keras.models.load_model('./static/models/tuberculosis.h5')
            pred = model.predict(image)

            predictions = pred.copy()
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1
            predictions = classes[int(predictions[0][0])]

            session["title"] = "TUBERCULOSIS"
            session["pred"] = predictions
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="TUBERCULOSIS")


@app.route('/melanoma_cancer_detection', methods=["GET", "POST"])
@login_required
def melanoma_cancer_detection():
    classes = {0: 'Normal', 1: 'Opacity'}

    if request.method == "POST" :
        insert_patient_details()

        if 'file' not in request.files :
            flash(f'No file uploaded', category='danger')
            return redirect(request.url)

        img = request.files['file']
        if img.filename == "" :
            flash(f'No image slected for uploading', category='danger')
            return redirect(request.url)

        if img and allowed_file(img.filename) :
            fname = secure_filename(img.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(fname))
            
            img.save(filepath)

            session["filename"] = fname

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (100, 100))

            image = np.reshape(image, (1, 100, 100, 3))

            model = keras.models.load_model('./static/models/melanoma_cancer.h5')
            pred = model.predict(image)
            
            predictions = pred.copy()
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1
            predictions = classes[int(predictions[0][0])]

            session["title"] = "MELANOMA CANCER"
            session["pred"] = predictions
            return redirect(url_for("image_result"))
        
        else :
            flash(f'Allowed image types are - png, jpg, jpeg, bmp', category='danger')
            return redirect(request.url)

    return render_template("image_dataset.html", title="MELANOMA CANCER")

@app.route('/display_image/<filename>')
@login_required
def display_image(filename) :
    return redirect(url_for('static', filename = 'uploads/' + filename), code=301)

@app.route('/image_result')
@login_required
def image_result():
    pid = session["pid"]
    details = Patient.query.filter_by(pid=pid).first()
    return render_template("display_image_result.html", details=details)

@app.route('/heart_disease_detection', methods=["GET", "POST"])
@login_required
def heart_disease_detection():
    if request.method == "POST" :
        insert_patient_details()
        cp = request.form["cp"]
        trestbps = request.form["trestbps"]
        sch = request.form["sch"]
        fbs = request.form["fbs"]
        restecg = request.form["restecg"]
        mh = request.form["mh"]
        exang = request.form["exang"]

        size = 7
        to_predict_list = list(cp)
        to_predict_list.append(trestbps)
        to_predict_list.append(sch)
        to_predict_list.append(fbs) 
        to_predict_list.append(restecg)
        to_predict_list.append(mh)
        to_predict_list.append(exang)

        to_pred = list(map(float, to_predict_list))
        to_pred = np.array(to_pred).reshape(1, size)

        model = joblib.load("./static/models/heart_model.pkl")

        pred = model.predict(to_pred)
        pred = pred[0]

        if int(pred) == 1 :
            pred = "Heart Disease "
        else :
            pred = "No Heart Disease "

        session["para"] = to_predict_list
        session["pred"] = pred

        return redirect(url_for('heart_disease_result'))


    return render_template("heart_disease_detection.html")

@app.route('/heart_disease_result')
@login_required
def heart_disease_result():
    pid = session["pid"]
    details = Patient.query.filter_by(pid=pid).first()
    return render_template("heart_disease_result.html", details = details)

@app.route('/kidney_disease_detection', methods=["GET", "POST"])
@login_required
def kidney_disease_detection():
    if request.method == "POST" :
        bp = request.form["bp"]
        sg = request.form["sg"]
        al = request.form["al"]
        su = request.form["su"]
        rbc = request.form["rbc"]
        pc = request.form["pc"]
        pcc = request.form["pcc"]

        size = 7
        to_predict_list = list()
        to_predict_list.append(bp)
        to_predict_list.append(sg)
        to_predict_list.append(al)
        to_predict_list.append(su) 
        to_predict_list.append(rbc)
        to_predict_list.append(pc)
        to_predict_list.append(pcc)
        print(to_predict_list)
        to_pred = list(map(float, to_predict_list))
        to_pred = np.array(to_pred).reshape(1, size)

        model = joblib.load("./static/models/kidney_model.pkl")

        pred = model.predict(to_pred)
        pred = pred[0]

        if int(pred) == 1 :
            pred = "Kidney Disease "
        else :
            pred = "No Kidney Disease "

        session["para"] = to_predict_list
        session["pred"] = pred

        return redirect(url_for('kidney_disease_result'))

        
    return render_template("kidney_disease_detection.html")

@app.route('/kidney_disease_result')
@login_required
def kidney_disease_result():
    pid = session["pid"]
    details = Patient.query.filter_by(pid=pid).first()
    return render_template("kidney_disease_result.html", details = details)


@app.route('/diabetes_detection', methods=["GET", "POST"])
@login_required
def diabetes_detection():
    if request.method == "POST":
        insert_patient_details()
        pr = request.form["pr"]
        gl = request.form["GL"]
        bp = request.form["BP"]
        bmi = request.form["bmi"]
        dpf = request.form["DPF"]
        page = request.form["page"]

        size = 6
        to_predict_list = list(pr)
        to_predict_list.append(gl)
        to_predict_list.append(bp)
        to_predict_list.append(bmi)
        to_predict_list.append(dpf)
        to_predict_list.append(page)
        to_pred = list(map(float, to_predict_list))
        to_pred = np.array(to_pred).reshape(1, size)

        model = joblib.load("./static/models/diabetes_model.pkl")

        pred = model.predict(to_pred)
        pred = pred[0]

        if int(pred) == 1 :
            pred = "Diabetes "
        else :
            pred = "No Diabetes "

        session["para"] = to_predict_list
        session["pred"] = pred

        return redirect(url_for('diabetes_result'))
        
    return render_template("diabetes_detection.html")

@app.route("/diabetes_result")
def diabetes_result():
    pid = session["pid"]
    details = Patient.query.filter_by(pid=pid).first()
    return render_template("diabetes_result.html", details = details)


@app.route("/sign_up", methods=["GET", "POST"])
def sign_up():
    if request.method == "POST" :
        pname = request.form['text']
        email = request.form['email']
        pswd  = request.form['password']
        phone = request.form['phnno']
        sec_ques = request.form['secques']
        sec_ans = request.form['secans']

        user_to_create = User(pname=pname,
                              email=email,
                              password=pswd,
                              phone=phone,
                              sec_ques=sec_ques,
                              sec_ans=sec_ans)

        db.session.add(user_to_create)
        db.session.commit()

        login_user(user_to_create)
        flash(f'Account creadted successfully! You are now logged in as : {user_to_create.email}', category='success')

        return redirect(url_for('home'))
    return render_template("signup.html")


@app.route('/sign_in', methods=['GET', 'POST'])
def sign_in():
    if request.method == "POST":
        email =  request.form['email']
        pswd = request.form['password']
        attempted_user = User.query.filter_by(email=email).first()
        if attempted_user and attempted_user.check_password_correction(
                attempted_password=pswd
        ):
            login_user(attempted_user)
            flash(f'Success! You are logged in as: {attempted_user.email}', category='success')

            return redirect(url_for('home'))
        else :
            flash('Email and password does not match! Please try again.', category='danger')
            return redirect(request.url)

    return render_template("login.html")


@app.route('/forgot', methods=['GET', 'POST'])
def forgot() :
    if request.method == "POST":
        email = request.form['email']
        sec_ques =  request.form['secques']
        sec_ans = request.form['secans']
        pswd1 =  request.form['password1']
        pswd2 =  request.form['password2']

        isFound = User.query.filter_by(email=email).first()
        if isFound :
            if isFound.sec_ques == sec_ques and isFound.sec_ans == sec_ans :
                if pswd1 == pswd2 :
                    isFound.password = pswd1
                    db.session.add(isFound)
                    db.session.commit()

                    flash("Password Changed sucessfully!", category="success")
                    return redirect(url_for('sign_in'))
                else :
                    flash('Passwords does not match! Please try again.', category='danger')
                    return redirect(request.url)
            else :
                flash('Security Question or Security Answer incorrect! Please try again.', category='danger')
                return redirect(request.url)
        else :
            flash('Email not found! Please try again.', category='danger')
            return redirect(request.url)

    return render_template("forgot.html")

@app.route('/reset_password', methods=["GET", "POST"])
@login_required
def reset_password():
    if request.method == "POST" :
        pswd = request.form['password']
        pswd1 =  request.form['password1']
        pswd2 =  request.form['password2']

        if current_user.check_password_correction(attempted_password = pswd):
            if pswd1 == pswd2:
                current_user.password = pswd1
                db.session.commit()

                flash("Password Changed sucessfully!", category="success")
                return redirect(url_for('profile'))
            else:
                flash('New passwords does not match! Please try again.', category='danger')
                return redirect(request.url)
        else:
            flash('Password incorrect! Please try again.', category='danger')
            return redirect(request.url)
    return render_template("reset_password.html")

@app.route('/sign_out')
@login_required
def sign_out():
    logout_user()
    flash("You have been logged out!", category="info")
    
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=False)
