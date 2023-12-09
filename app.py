import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from PIL import Image
import os
import numpy as np
from flask import Flask, request, render_template
import cv2
from cv2 import cvtColor, Laplacian, COLOR_BGR2GRAY,Canny,COLOR_BGR2RGB
import pytesseract
from io import BytesIO
from fontTools.ttLib import TTFont
import re
import requests
from pyzbar.pyzbar import decode
import logging
import sklearn
from sklearn.cluster import KMeans


app = Flask(__name__)

# Define the folder for uploading posters
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the "uploads" directory exists, or create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Define the maximum file size (in bytes)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 2 MB

# Function to check if the uploaded file is within the size limit
def is_valid_file_size(file_path):
    return os.path.getsize(file_path) <= MAX_FILE_SIZE

# Function to check if the filename extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif', 'bmp'}



# Function to calculate average RGB values
def calculate_average_rgb(image_path):
        r, g, b = Image.open(image_path).split()
        r_avg = np.ceil(np.mean(r))
        g_avg = np.ceil(np.mean(g))
        b_avg = np.ceil(np.mean(b))
        return r_avg, g_avg, b_avg

def evaluate_indentation(image_path):
    # In this example, we'll use a simple threshold-based evaluation
    img = Image.open(image_path)
    img_data = np.array(img)
    mean_value = np.mean(img_data)
    threshold = mean_value * 0.75
    indentation_score = (np.mean(img_data) > threshold)  # Simulated result
    return indentation_score
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng+other_languages')
        return text
    except Exception as e:
        return str(e)

def analyze_image_clarity(image_path):
    img = Image.open(image_path)
    if img.mode == 'RGB':
        img = cvtColor(np.array(img), COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.uint8(img)
    clarity_score = cv2.Laplacian(img, cv2.CV_64F).var()
    return clarity_score

def analyze_clutter(image_path):
    img = Image.open(image_path)
    img = cvtColor(np.array(img), COLOR_BGR2GRAY)
    mean_value = np.mean(img)
    std_dev = np.std(img)
    threshold1 = mean_value - std_dev
    threshold2 = mean_value + std_dev
    _, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = Canny(thresholded, threshold1=threshold1, threshold2=threshold2)
    clutter_score = np.count_nonzero(edges)
    return clutter_score

def qr_code_detector(image_path):
    try:
        image = Image.open(image_path)
        decoded_objects = decode(image)
        if decoded_objects:
            for obj in decoded_objects:
                text = obj.data.decode('utf-8')
                url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                if re.match(url_pattern, text):
                    return True, obj.data.decode('utf-8')
                return False, obj.data.decode('utf-8')
        else:
            return False, "No qr code found"
    except Exception as e:
        new_var = False
        return new_var, str(e)
def extract_links_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='eng')
    url_regex = r'https?://\S+|www\.\S+'
    links = re.findall(url_regex, text)
    return links

from langdetect import detect
from langdetect.detector_factory import DetectorFactory
DetectorFactory.seed = 0

# Update extract_text_from_image function
def extract_lang_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng+other_languages')

        # Identify the language of the extracted text
        detected_language = detect(text)

        return detected_language
    except Exception as e:
        return str(e), None
import cv2
import numpy as np

def analyze_horizontal_symmetry(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    top_half = img[0:height // 2, :]
    bottom_half = img[height // 2:, :]
    if top_half.shape != bottom_half.shape:
        bottom_half = cv2.resize(bottom_half, (top_half.shape[1], top_half.shape[0]))
    bottom_half_flipped = cv2.flip(bottom_half, 0)
    diff = cv2.absdiff(top_half, bottom_half_flipped)
    hsymmetry_score = np.mean(diff)
    return hsymmetry_score


def analyze_vertical_symmetry(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    left_half = img[:, 0:width // 2]
    right_half = img[:, width // 2:]
    right_half_flipped = cv2.flip(right_half, 1)
    diff = cv2.absdiff(left_half, right_half_flipped)
    vsymmetry_score = np.mean(diff)
    return vsymmetry_score

def analyze_balance(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    left_segment = gray[:, :width // 2]
    right_segment = gray[:, width // 2:]
    left_intensity = cv2.mean(left_segment)[0]
    right_intensity = cv2.mean(right_segment)[0]
    balance_score = abs(left_intensity - right_intensity)
    return balance_score

def extract_color_palette(image_path, num_colors=5):
    img = cv2.imread(image_path)
    img = cvtColor(img, COLOR_BGR2RGB)
    pixels = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

from textblob import TextBlob

def perform_sentiment_analysis(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    return sentiment_score

def logo_detection(image_path):
    image = cv2.imread(image_path)
    im1 = int(0.1 * image.shape[0])
    region = image[:im1, -1*im1:]
    gray_logo = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, binary_logo = cv2.threshold(gray_logo, 128, 255, cv2.THRESH_BINARY)
    logo_score = cv2.countNonZero(binary_logo)

    if logo_score>10000:
        return "Logo not Detected"
    else:
        return "Logo Detected"

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng+other_languages')
        return text
    except Exception as e:
        return str(e)

@app.route("/", methods=["GET", "POST"])
def index():
    average_rgb_result = ""  # Initialize with an empty string
    indentation_result = ""
    qr_code_image = None
    extracted_text = ""
    clarity_result = ""
    clutter_result = ""
    barcode_results = []
    detected_urls = []
    url_validations = []
    show_results = False
    font_size_result = ""
    link_result=""
    lang_result=""
    hsym=""
    vsym=""
    balance=""
    color_palette=[]
    sentiment_score = 0
    sentiment_result = ""
    logo=""



    if request.method == "POST":
        # Check if the post request has a file part
        if 'poster' not in request.files:
            return "No file part"

        file = request.files['poster']

        if file.filename == '' or not allowed_file(file.filename):
            return "Invalid or no selected file"

        if file and allowed_file(file.filename):
            # Save the uploaded poster
            poster_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(poster_path)
            # Check if the file size exceeds the limit
            if not is_valid_file_size(poster_path):
                raise Exception("File size exceeds the limit (10 MB): Please upload a smaller file")

            # Calculate average RGB values
            r, g, b = calculate_average_rgb(poster_path)
            average_rgb_result = f" ({r}, {g}, {b})"

            # Evaluate indentation (replace with actual model prediction)
            indentation_score = evaluate_indentation(poster_path)
            indentation_result = f"{indentation_score}"
            extracted = extract_text_from_image(poster_path)
            extracted_text = f" {extracted}"

            extracted1=extract_links_from_image(poster_path)
            link_result=f" {extracted1}"

            lang=extract_lang_from_image(poster_path)
            lang_result=f" {lang}"

            horizontal=analyze_horizontal_symmetry(poster_path)
            hsym=f"{ horizontal}"
            vertical=analyze_vertical_symmetry(poster_path)
            vsym=f"{ vertical}"

            bal=analyze_balance(poster_path)
            balance=f"{bal}"

            clarity_score = analyze_image_clarity(poster_path)
            clarity_result = f" {clarity_score}"

            clutter_score = analyze_clutter(poster_path)
            clutter_result = f" {clutter_score}"

            sentiment_score = perform_sentiment_analysis(poster_path)
            if sentiment_score > 0:
                sentiment_result = "Positive"
            elif sentiment_score < 0:
                sentiment_result = "Negative"
            else:
                sentiment_result = "Neutral"

            logo = logo_detection(poster_path)
            logo= f"{logo}"

            color_palette = extract_color_palette(poster_path, num_colors=5)

            has_qr_code, qr_code_data = qr_code_detector(poster_path)
            qr_data = qr_code_data
            if has_qr_code:
                qr_data = f"<a href='{qr_code_data}'>{qr_code_data}</a>/safe"


            return render_template('results.html',
                                   average_rgb_result=average_rgb_result,
                                   indentation_result=indentation_result,
                                   clarity_result=clarity_result,
                                   clutter_result=clutter_result,
                                   qr_code_data=qr_code_data,
                                   link_result=link_result,
                                   lang_result=lang_result,
                                   hsym=hsym,
                                   vsym=vsym,
                                   balance=balance,
                                   color_palette=color_palette,
                                   sentiment_result=sentiment_result,
                                   logo=logo,
                                   extracted_text=extracted_text,
                                   )



            #return f"{average_rgb_result}<br>{indentation_result}<br>{size_dimension_result}<br>{clarity_result}<br>{clutter_result}<br>{extracted_text}<br>QR code data:{qr_data}"

    return render_template('index.html',
               average_rgb_result=average_rgb_result,
               indentation_result=indentation_result,
               clarity_result=clarity_result,
               clutter_result=clutter_result,
               link_result=link_result,
               lang_result=lang_result,
               hsym=hsym,
               vsym=vsym,
               balance=balance,
               color_palette=color_palette,
               sentiment_result=sentiment_result,
               extracted_text=extracted_text,
               logo=logo,
               show_results=True)
@app.route("/index")
def initial_page():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True, port=9567)

