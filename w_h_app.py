# Importing essential libraries and modules
import sys
import datetime
#import requests
import json
import mysql.connector

from flask import Flask, render_template, request, Markup, flash
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from flask_sqlalchemy import SQLAlchemy
# ==============================================================================================
# db pass - icHuK9dx*z_%Mmao
# db name - id19252925_agr1986
# db user - id19252925_vijaya
# db url - https://agr1986.000webhostapp.com/

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

#city class --------------------------------



# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///weather.db'
app.secret_key = 'secret_key'

db = SQLAlchemy(app)

# render home page


@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    
     #dbconnection 
    mydb = mysql.connector.connect(
    #local database
    #host="192.168.0.104",
    #user="weather",
    #password="cA*zu2.WwkFW)2j)",
    #database="weather"
    
    #cloud database
    host="54.166.16.4",
    user="weather123",
    password="gZkg(d/Zgpjev2f3",
    database="weather"
    
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("SELECT max(id),`temp`,`soilhumidity` FROM sensor where did='kop1'")

    myresult = mycursor.fetchall()
    
    for x in myresult:
        print(x)
        tempkop1 = x[1]
        sh = x[2]
    
    return render_template('crop.html', temp1=tempkop1, sh=sh, title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'
    
    #dbconnection 
    mydb = mysql.connector.connect(
    #local database
    #host="192.168.0.104",
    #user="weather",
    #password="cA*zu2.WwkFW)2j)",
    #database="weather"
    #cloud database
    host="54.166.16.4",
    user="weather123",
    password="gZkg(d/Zgpjev2f3",
    database="weather"
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("SELECT max(id),`temp` FROM sensor where did='kop1'")

    myresult = mycursor.fetchall()
    
    for x in myresult:
        print(x)
        tempkop1 = x[1]
     
    return render_template('fertilizer.html', temp1=tempkop1, title=title)

# render disease prediction input page


#render weather page
    
#@ app.route('/w_new_index')
#def index():

#    title = 'Weather - Live Information'

#   return render_template('w_new_index.html', title=title)


# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)
    
# render weather app page

class City(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True, nullable=False)

    def __repr__(self):
        return self.name


@app.route('/w_new_index')
def index():
    def get_date(timezone):
        tz = datetime.timezone(datetime.timedelta(seconds=int(timezone)))
        return datetime.datetime.now(tz=tz).time().hour

    cities = City.query.all()
    weather = []
    #dbconnection unit 1
    mydb = mysql.connector.connect(
    #local database
    #host="192.168.0.104",
    #user="weather",
    #password="cA*zu2.WwkFW)2j)",
    #database="weather"
    #cloud database
    host="54.166.16.4",
    user="weather123",
    password="gZkg(d/Zgpjev2f3",
    database="weather"
    )

    mycursor = mydb.cursor()

    mycursor.execute("SELECT max(id),`temp`,`soilhumidity`,`airhumidity`,`pressure`,`date`,`time`,`did` FROM sensor where did='kop1'")

    myresult = mycursor.fetchall()

    for x in myresult:
        print(x)
        tempkop1 = x[1]
        sh = x[2]
        ah = x[3]
        pre = x[4]
        did = x[7]
    #dbend    
    #dbconnection unit 2
    mydb1 = mysql.connector.connect(
    #local database
    #host="192.168.0.104",
    #user="weather",
    #password="cA*zu2.WwkFW)2j)",
    #database="weather"
    #cloud database
    host="54.166.16.4",
    user="weather123",
    password="gZkg(d/Zgpjev2f3",
    database="weather"
    )

    mycursor1 = mydb1.cursor()

    mycursor1.execute("SELECT max(id),`temp`,`soilhumidity`,`airhumidity`,`pressure`,`date`,`time`,`did` FROM sensor where did='kop2'")

    myresult1 = mycursor1.fetchall()

    for x in myresult1:
        print(x)
        tempkop2 = x[1]
        sh1 = x[2]
        ah1 = x[3]
        pre1 = x[4]
        did1 = x[7]
    #dbend  
    for city in cities:
        response = requests.get(
            f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid=5b307cb345d3bef73b070c9b38e5ca4a&units=metric')

        content = json.loads(response.text)
        weather_info = {'degrees': f"{content['main']['temp']}",
                        'state': f"{content['weather'][0]['main']}",
                        'city': f"{content['name']}",
                        'time': get_date(content['timezone']),
                        'id': f"{city.id}"}
        weather.append(weather_info)
        
    return render_template("w_new_index.html", weather=weather,temp1=tempkop1,temp2=tempkop2, sh=sh, sh1=sh1, ah=ah, ah1=ah1, pre=pre, pre1=pre1)


@app.route('/add_city1', methods=['POST'])
def add_city1():
    if request.method == 'POST':
        city_name = request.form.get('city_name')

        response = requests.get(
            f'https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid=5b307cb345d3bef73b070c9b38e5ca4a&units=metric')

        if response.status_code == 404:
            flash("The city doesn't exist!")
            return redirect('/')

        cities = City.query.all()
        for city in cities:
            if city.name == city_name:
                flash("The city has already been added to the list!")
                return redirect('/')

        else:
            city = City(name=city_name)
            db.session.add(city)
            db.session.commit()

            return render_template("w_new_index.html")

@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    if request.method == 'POST':
        city = City.query.filter_by(id=id).first()
        db.session.delete(city)
        db.session.commit()
        return render_template("w_new_index.html")

# ===============================================================================================
if __name__ == '__main__':
    if len(sys.argv) > 1:
        db.create_all()
        arg_host, arg_port = sys.argv[1].split(':')
        app.run(host=arg_host, port=arg_port)
    else:
        db.create_all()
        app.run("192.168.0.102")
