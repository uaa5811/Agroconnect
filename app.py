# Importing essential libraries and modules
import sys
import datetime

import json
import mysql.connector
from flask import jsonify, send_from_directory
from flask import Flask, render_template, request, Markup, flash, send_file
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

from PIL import ImageOps, ImageFilter
import cv2
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap



# ==============================================================================================
# db pass - icHuK9dx*z_%Mmao
# db name - id19252925_agr1986
# db user - id19252925_
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

# ===============================================================================================
@app.route('/image-processing')
def image_processing():
    return render_template('image_processing.html')

@app.route('/image-processing/greyscale', methods=['POST'])
def greyscale():
    file = request.files['file']
    img = Image.open(file.stream).convert('L')
    return serve_pil_image(img)

@app.route('/image-processing/segmentation', methods=['POST'])
def segmentation():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.filter(ImageFilter.FIND_EDGES)
    return serve_pil_image(img)

@app.route('/image-processing/heatmap', methods=['POST'])
def heatmap():
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    heatmap_path = generate_heatmap(file_path)
    return send_file(heatmap_path, mimetype='image/png')

def generate_heatmap(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply a Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Normalize the image for heatmap
    normalized_image = cv2.normalize(blurred_image, None, 0, 255, cv2.NORM_MINMAX)
    
    plt.switch_backend('Agg')
    
    # Generate heatmap using matplotlib
    colormap = LinearSegmentedColormap.from_list('heatmap', ['blue', 'green', 'yellow', 'red'])
    plt.imshow(normalized_image, cmap=colormap)
   
    heatmap_path = os.path.join('static', 'heatmap_' + os.path.basename(file_path))
    plt.savefig(heatmap_path)
    plt.close()
    
    return heatmap_path
def serve_pil_image(img):
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
    
    
@app.route('/image-processing', methods=['GET', 'POST'])
def disease_prediction():
    #title = 'AgroConnect - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('image-processing.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction)
        except:
            pass
    return render_template('disease.html')  
    
    
    

# ===============================================================================================

@app.route('/chatbot')
def chatbot():
    title = 'AgroConnect - Chatbot'
    return render_template('chatbot.html', title=title)

@app.route('/chatbot-response', methods=['POST'])
def chatbot_response():
    data = request.get_json()
    message = data.get('message', '')
    response = get_chatbot_response(message)
    return jsonify({'response': response})
    

def get_chatbot_response(message):
    responses = {
        "hello": "Hello! How can I help you today?",
        "hi": "Hi there! How can I assist you?",
        "how are you": "I'm just a bot, but I'm doing great! How about you?",
        "what is your name": "I'm AgroBot, here to help you with your agricultural queries.",
        "what is AgroConnect": "AgroConnect is a platform dedicated to providing information and resources for farmers and agriculture enthusiasts.",
        "what services do you offer": "We offer a range of services including crop management tips, weather updates, and market trends.",
        "how can I register": "You can register by clicking on the 'Register' button on the top right corner of the homepage.",
        "how do I reset my password": "You can reset your password by clicking on 'Forgot Password' on the login page.",
        "what crops can I grow in summer": "In summer, you can grow crops like corn, tomatoes, cucumbers, and beans.",
        "what crops can I grow in winter": "In winter, you can grow crops like spinach, carrots, garlic, and onions.",
        "how to prevent pests": "To prevent pests, use integrated pest management techniques including natural predators, organic pesticides, and crop rotation.",
        "what is crop rotation": "Crop rotation is the practice of growing different types of crops in the same area in sequential seasons to improve soil health and reduce pests.",
        "how to improve soil fertility": "Improve soil fertility by adding organic matter, using compost, and practicing crop rotation.",
        "what are the benefits of organic farming": "Organic farming improves soil health, reduces pollution, conserves water, and promotes biodiversity.",
        "how to conserve water in farming": "Conserve water by using drip irrigation, rainwater harvesting, and choosing drought-resistant crops.",
        "what is precision farming": "Precision farming uses technology to monitor and manage field variability in crops to increase productivity and efficiency.",
        "what is hydroponics": "Hydroponics is a method of growing plants without soil, using nutrient-rich water solutions.",
        "what is aquaponics": "Aquaponics combines aquaculture (raising fish) and hydroponics (growing plants in water) in a symbiotic environment.",
        "how to start a farm": "Start a farm by researching, creating a business plan, acquiring land, and choosing the right crops or livestock.",
        "what is sustainable farming": "Sustainable farming focuses on producing food in a way that preserves the environment, supports animal welfare, and maintains farm profitability.",
        "how to control weeds": "Control weeds by mulching, hand weeding, using cover crops, and applying organic herbicides.",
        "what is no-till farming": "No-till farming is an agricultural practice where crops are grown without disturbing the soil through tillage.",
        "how to test soil pH": "Test soil pH using a soil test kit or by sending a soil sample to a laboratory for analysis.",
        "what is crop diversification": "Crop diversification involves growing a variety of crops to reduce risk and improve soil health.",
        "how to manage farm waste": "Manage farm waste by composting organic waste, recycling materials, and using waste-to-energy technologies.",
        "what are GMOs": "GMOs (genetically modified organisms) are organisms whose genetic material has been altered using genetic engineering techniques.",
        "what is organic certification": "Organic certification is a certification process for producers of organic food and other organic agricultural products.",
        "how to get organic certification": "To get organic certification, follow the organic farming standards set by your country's certification body and undergo an inspection process.",
        "what is permaculture": "Permaculture is a design philosophy that emphasizes sustainable, self-sufficient agricultural systems modeled on natural ecosystems.",
        "how to make compost": "Make compost by combining green and brown organic materials in a pile or bin and allowing them to decompose over time.",
        "what is agroforestry": "Agroforestry is the integration of trees and shrubs into agricultural landscapes to improve biodiversity and productivity.",
        "how to start beekeeping": "Start beekeeping by learning about bees, acquiring necessary equipment, and setting up hives in a suitable location.",
        "what are the benefits of beekeeping": "Beekeeping benefits include pollination of crops, production of honey and beeswax, and support for biodiversity.",
        "how to raise chickens": "Raise chickens by providing a coop, feeding them a balanced diet, and ensuring they have access to fresh water and space to roam.",
        "what is a greenhouse": "A greenhouse is a structure with walls and roof made chiefly of transparent material, such as glass, in which plants requiring regulated climatic conditions are grown.",
        "how to manage greenhouse pests": "Manage greenhouse pests by maintaining hygiene, using biological control agents, and monitoring for pest outbreaks.",
        "what is integrated pest management": "Integrated pest management is an approach to controlling pests using a combination of techniques such as biological control, habitat manipulation, and resistant varieties.",
        "how to start a farmers market": "Start a farmers market by researching local regulations, finding a suitable location, and recruiting vendors.",
        "what is urban farming": "Urban farming is the practice of cultivating, processing, and distributing food in or around urban areas.",
        "how to grow mushrooms": "Grow mushrooms by inoculating a substrate with mushroom spawn and maintaining proper temperature and humidity.",
        "what are cover crops": "Cover crops are plants grown primarily to improve soil health, reduce erosion, and manage water and nutrients.",
        "how to attract pollinators": "Attract pollinators by planting a variety of flowering plants, providing habitat, and avoiding pesticides.",
        "what is rotational grazing": "Rotational grazing is a livestock grazing strategy that involves rotating animals through different pasture areas to allow vegetation to recover.",
        "how to start a community garden": "Start a community garden by finding a suitable location, gathering community support, and organizing resources and volunteers.",
        "what is a CSA": "A CSA (Community Supported Agriculture) is a system in which consumers purchase shares of a farm's harvest in advance.",
        "how to market farm products": "Market farm products by building relationships with customers, using social media, participating in farmers markets, and creating value-added products.",
        "what is agroecology": "Agroecology is the study of ecological processes applied to agricultural production systems, aiming for sustainable and resilient farming practices.",
        "how to get funding for a farm": "Get funding for a farm through grants, loans, crowdfunding, and government programs.",
        "how to prevent blight in tomatoes": "To prevent blight in tomatoes, ensure proper spacing for air circulation, avoid overhead watering, and use fungicides if necessary.",
        "how to treat powdery mildew": "Treat powdery mildew by applying fungicides, using neem oil, or ensuring good air circulation around the plants.",
        "how to care for apple trees": "Care for apple trees by regular pruning, providing adequate water, and protecting them from pests and diseases.",
        "how to control aphids on plants": "Control aphids by using insecticidal soap, neem oil, or introducing natural predators like ladybugs.",
        "how to prevent root rot": "Prevent root rot by ensuring well-draining soil, avoiding overwatering, and using fungicide treatments if needed.",
        "what causes yellow leaves": "Yellow leaves can be caused by overwatering, nutrient deficiencies, or pest infestations.",
        "how to deal with leaf spot": "Deal with leaf spot by removing affected leaves, avoiding overhead watering, and applying fungicides.",
        "how to prevent damping off": "Prevent damping off by using sterile soil, avoiding overwatering, and providing adequate air circulation.",
        "how to treat rust on plants": "Treat rust on plants by removing affected leaves, applying fungicides, and ensuring proper spacing for air circulation.",
        "how to grow healthy cucumbers": "Grow healthy cucumbers by providing full sun, well-draining soil, regular watering, and protecting them from pests.",
        "how to prevent downy mildew": "Prevent downy mildew by ensuring good air circulation, avoiding overhead watering, and applying fungicides.",
        "how to control spider mites": "Control spider mites by using insecticidal soap, neem oil, or introducing natural predators like predatory mites.",
        "how to care for rose bushes": "Care for rose bushes by providing full sun, regular watering, and pruning to maintain shape and health.",
        "how to prevent anthracnose": "Prevent anthracnose by using resistant plant varieties, avoiding overhead watering, and applying fungicides.",
        "how to treat bacterial wilt": "Treat bacterial wilt by removing affected plants, rotating crops, and using resistant plant varieties.",
        "how to prevent scab in potatoes": "Prevent scab in potatoes by using resistant varieties, maintaining soil pH, and avoiding excessive nitrogen fertilization.",
        "how to care for citrus trees": "Care for citrus trees by providing full sun, regular watering, and protecting them from pests and diseases.",
        "how to control whiteflies": "Control whiteflies by using insecticidal soap, yellow sticky traps, or introducing natural predators like ladybugs.",
        "how to prevent verticillium wilt": "Prevent verticillium wilt by rotating crops, using resistant plant varieties, and maintaining healthy soil.",
        "how to treat fire blight": "Treat fire blight by pruning affected branches, disinfecting tools, and applying copper-based fungicides.",
        "how to care for strawberry plants": "Care for strawberry plants by providing full sun, well-draining soil, and protecting them from pests and diseases.",
        "how to control slugs and snails": "Control slugs and snails by using bait, diatomaceous earth, or introducing natural predators like birds.",
        "how to prevent fusarium wilt": "Prevent fusarium wilt by rotating crops, using resistant plant varieties, and maintaining healthy soil.",
        "how to treat sooty mold": "Treat sooty mold by controlling the insects that cause it, like aphids or whiteflies, and washing the mold off with water.",
        "how to care for blueberry bushes": "Care for blueberry bushes by providing acidic soil, regular watering, and protecting them from pests and diseases.",
        "how to control cutworms": "Control cutworms by using physical barriers, applying beneficial nematodes, or using insecticides.",
        "how to prevent blossom end rot": "Prevent blossom end rot by ensuring consistent watering and providing adequate calcium in the soil.",
        "how to treat leaf curl": "Treat leaf curl by applying fungicides, removing affected leaves, and ensuring proper air circulation.",
        "how to care for raspberry plants": "Care for raspberry plants by providing full sun, well-draining soil, and regular pruning.",
        "how to control mealybugs": "Control mealybugs by using insecticidal soap, neem oil, or introducing natural predators like ladybugs.",
        "how to prevent cedar apple rust": "Prevent cedar apple rust by removing nearby junipers, using resistant plant varieties, and applying fungicides.",
        "how to treat black spot on roses": "Treat black spot on roses by removing affected leaves, avoiding overhead watering, and applying fungicides.",
        "how to care for grapevines": "Care for grapevines by providing full sun, regular watering, and pruning to maintain shape and health.",
        "how to control thrips": "Control thrips by using insecticidal soap, neem oil, or introducing natural predators like predatory mites.",
        "how to prevent mosaic virus": "Prevent mosaic virus by using resistant plant varieties, controlling insect vectors, and practicing good garden hygiene.",
        "how to treat crown gall": "Treat crown gall by removing affected plants, disinfecting tools, and using resistant plant varieties.",
        "how to care for fig trees": "Care for fig trees by providing full sun, regular watering, and protecting them from pests and diseases.",
        "how to control earwigs": "Control earwigs by using bait, diatomaceous earth, or introducing natural predators like birds.",
        "how to prevent powdery mildew": "Prevent powdery mildew by ensuring good air circulation, avoiding overhead watering, and applying fungicides.",
        "how to treat botrytis": "Treat botrytis by removing affected plant parts, avoiding overhead watering, and applying fungicides.",
        "how to care for peach trees": "Care for peach trees by providing full sun, regular watering, and protecting them from pests and diseases.",
        "how to control Japanese beetles": "Control Japanese beetles by using traps, handpicking, or applying insecticides.",
        "how to prevent root-knot nematodes": "Prevent root-knot nematodes by rotating crops, using resistant plant varieties, and maintaining healthy soil.",
        "how to treat canker": "Treat canker by pruning affected branches, disinfecting tools, and applying fungicides.",
        "how to care for pear trees": "Care for pear trees by providing full sun, regular watering, and protecting them from pests and diseases.",
        "how to control codling moths": "Control codling moths by using pheromone traps, applying insecticides, and practicing good garden hygiene.",
        "how to prevent alternaria leaf spot": "Prevent alternaria leaf spot by rotating crops, using resistant plant varieties, and applying fungicides.",
        "how to treat downy mildew": "Treat downy mildew by removing affected leaves, avoiding overhead watering, and applying fungicides.",
         "what is farming": "Farming is the practice of cultivating the land or raising animals for food, fiber, and other products.",
        "what are the types of farming": "There are several types of farming, including subsistence farming, commercial farming, organic farming, and intensive farming.",
        "what is organic farming": "Organic farming is a method of farming that avoids the use of synthetic fertilizers and pesticides, focusing on natural processes and materials.",
        "how to start a farm": "To start a farm, you need to plan your crops or livestock, secure land, gather necessary equipment, and understand the local regulations and market.",
        "what is crop rotation": "Crop rotation is the practice of growing different types of crops in the same area across different seasons to improve soil health and reduce pests and diseases.",
        "what are cash crops": "Cash crops are crops that are grown primarily for sale and profit rather than for personal use. Examples include cotton, coffee, and sugarcane.",
        "what is irrigation": "Irrigation is the artificial application of water to the land to assist in the growth of crops.",
        "how to improve soil fertility": "Soil fertility can be improved by adding organic matter, practicing crop rotation, using cover crops, and applying appropriate fertilizers.",
        "what is sustainable farming": "Sustainable farming involves practices that maintain the productivity and usefulness of the land over the long term, often incorporating environmental stewardship and economic viability.",
        "what is monoculture": "Monoculture is the agricultural practice of growing a single crop, plant, or livestock species in a field at a time.",
        "what are cover crops": "Cover crops are plants grown to protect and enrich the soil during times when main crops are not grown.",
        "what is hydroponics": "Hydroponics is a method of growing plants without soil, using nutrient-rich water instead.",
        "what is a greenhouse": "A greenhouse is a structure with walls and a roof made chiefly of transparent material, such as glass, in which plants requiring regulated climatic conditions are grown.",
        "what are pesticides": "Pesticides are substances used to eliminate or control pests that can damage crops and livestock.",
        "what is composting": "Composting is the process of recycling organic waste, such as food scraps and yard waste, into a valuable soil amendment.",
        "what are the benefits of crop diversification": "Crop diversification can improve soil health, reduce pest and disease outbreaks, and increase economic stability for farmers.",
        "what is a farming cooperative": "A farming cooperative is an organization owned and operated by a group of farmers who work together to pool resources and share benefits.",
        "how to control weeds": "Weeds can be controlled through manual removal, mulching, using herbicides, and practicing crop rotation.",
        "what is aquaponics": "Aquaponics is a system that combines aquaculture (raising fish) and hydroponics (growing plants in water) in a symbiotic environment.",
        "what is no-till farming": "No-till farming is an agricultural technique for growing crops without disturbing the soil through tillage.",
    }
    
    
    return responses.get(message.lower(), "I'm not sure how to respond to that. Can you please rephrase your question?")


# render home page
@ app.route('/')
def home():
    title = 'AgroConnect - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'AgroConnect - Crop Recommendation'
    
     #dbconnection 
    mydb = mysql.connector.connect(
        
    #local database
    #host="192.168.0.104",
    #user="weather",
    #password="cA*zu2.WwkFW)2j)",
    #database="weather"
    
    #cloud database
    host="13.49.57.19",
    user="uaa",
    password="uaa@1234$",
    database="weather"
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("SELECT `temp`,`soilhumidity`,`airhumidity`,`pressure` FROM sensor WHERE id = ( SELECT MAX(id) FROM sensor where did='kop1')")

    myresult = mycursor.fetchall()
    
    for x in myresult:  
        print(x)
        tempkop1 = x[0]
        sh = x[1]
        ah = x[2]
        pre = x[3]
        
    
    return render_template('crop.html', temp1=tempkop1, sh=sh, ah=ah, pre=pre, title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'AgroConnect - Fertilizer Suggestion'
    
    #dbconnection 
    mydb = mysql.connector.connect(
    #local database
    #host="192.168.0.104",
    #user="weather",
    #password="cA*zu2.WwkFW)2j)",
    #database="weather"
    
    #cloud database
    host="13.49.57.19",
    user="uaa",
    password="uaa@1234$",
    database="weather"
    )
    
    mycursor = mydb.cursor()
    
    #SELECT temp,soilhumidity FROM sensor WHERE id = ( SELECT MAX(id) FROM sensor where did='kop1');
    
    mycursor.execute("SELECT `temp`,`soilhumidity`,`airhumidity`,`pressure` FROM sensor WHERE id = ( SELECT MAX(id) FROM sensor where did='kop1')")

    myresult = mycursor.fetchall()
    
    for x in myresult:  
        print(x)
        tempkop1 = x[0]
        sh = x[1]
        ah = x[2]
        pre = x[3]
     
    return render_template('fertilizer.html', temp1=tempkop1, sh=sh, ah=ah, pre=pre, title=title)

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
    title = 'AgroConnect - Crop Recommendation'

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
    title = 'AgroConnect - Fertilizer Suggestion'

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


"""@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'AgroConnect - Disease Detection'

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
    return render_template('disease.html', title=title)"""
    

    
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
    host="13.49.57.19",
    user="uaa",
    password="uaa@1234$",
    database="weather"
    )

    mycursor = mydb.cursor()

    mycursor.execute("SELECT max(id),`temp`,`soilhumidity`,`airhumidity`,`pressure`,`date`,`time`,`did` FROM sensor where id = (SELECT MAX(id) FROM sensor where did='kop1')")

    
    
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
    host="13.49.57.19",
    user="uaa",
    password="uaa@1234$",
    database="weather"
    )

    mycursor1 = mydb1.cursor()

    mycursor1.execute("SELECT max(id),`temp`,`soilhumidity`,`airhumidity`,`pressure`,`date`,`time`,`did` FROM sensor where id = (SELECT MAX(id) FROM sensor where did='kop2')")

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
            return render_template("w_new_index.html")

        cities = City.query.all()
        for city in cities:
            if city.name == city_name:
                flash("The city has already been added to the list!")
                return render_template("w_new_index.html")

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
        app.run()
