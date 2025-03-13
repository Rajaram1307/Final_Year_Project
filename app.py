from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_mysqldb import MySQL
import bcrypt
from deepface import DeepFace
from collections import Counter
import cv2
import numpy as np
import json
import random
import re

app = Flask(__name__)


# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Rajaram001'
app.config['MYSQL_DB'] = 'user_auth'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.secret_key = 'your_secret_key'  # Required for session management

mysql = MySQL(app)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')  # Your HTML file

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and bcrypt.checkpw(password, user['password'].encode('utf-8')):
            session['user_id'] = user['id']
            session['email'] = user['email']
            return redirect(url_for('dashboard'))  # Redirect to dashboard after login
        else:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))

    return render_template('index.html')  # Render the login form

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())

        cursor = mysql.connection.cursor()
        try:
            cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed_password))
            mysql.connection.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            mysql.connection.rollback()
            flash('Email already exists or an error occurred.', 'error')
        finally:
            cursor.close()

    return render_template('index.html')  # Render the registration form

# Dashboard Route (Protected)
@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        return render_template('dashboard.html', email=session['email'])
    else:
        flash('You need to login first', 'error')
        return redirect(url_for('login'))

# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('home'))

#-------------------end of MySql module--------------------------



# Load intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

def classify_intent(user_text):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Check if the pattern is found in the user's text (case insensitive)
            if re.search(r'\b' + re.escape(pattern) + r'\b', user_text, re.IGNORECASE):
                return random.choice(intent['responses'])
    return "I'm here for you. How can I assist you?"

@app.route('/chat.html')
def chat():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    user_text = request.form['msg']
    return str(classify_intent(user_text))

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    file = request.files['image']
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    try:
        # Analyze the image using DeepFace to detect emotion
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        # Get the dominant emotion
        dominant_emotion = analysis['dominant_emotion']
        return jsonify({"mood": dominant_emotion})

    except ValueError as e:
        # Handle the case where no face is detected
        return jsonify({"mood": "no_face", "error": str(e)})

# Load face cascade classifier and face expression capture module
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_emotions():
    cap = cv2.VideoCapture(0)
    captured_expressions = []
    capture_limit = 20  # Capture 50 expressions

    while len(captured_expressions) < capture_limit:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Store dominant emotion, replacing 'neutral' with 'sad'
            emotion = result[0]['dominant_emotion']
            if emotion == "neutral":
                emotion = "sad"

            captured_expressions.append(emotion)

        cv2.imshow('Capturing Emotion...', frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    # Determine the most frequent emotion
    if captured_expressions:
        final_emotion = Counter(captured_expressions).most_common(1)[0][0]
    else:
        final_emotion = "No emotion detected"

    return final_emotion

@app.route('/start-capture')
def start_capture():
    user_mood = capture_emotions()
    return jsonify({"mood": user_mood})

if __name__ == "__main__":
    app.run(debug=True)