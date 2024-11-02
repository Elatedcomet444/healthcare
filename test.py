from flask import Flask, render_template, Response,jsonify,request,session,url_for,redirect
import cv2
import face_recognition
import os
import mediapipe as mp
import base64
import numpy as np
import re
import mysql.connector
from email.message import EmailMessage
from flask import request, redirect, url_for, flash
import smtplib
import ssl
import random
import json
import pandas as pd
# from googletrans import Translator
# from googletrans.constants import LANGUAGES
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk
from nltk.corpus import stopwords
from spacy.lang.en import STOP_WORDS
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash
import datetime
from gtts import gTTS
import pygame
from io import BytesIO
import pandas as pd
import numpy as np
from googletrans import Translator
from googletrans.constants import LANGUAGES
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key="ayyappa"
conn=mysql.connector.connect(host="localhost",user="root",password="",database="kiosk")
cursor = conn.cursor(dictionary=True) 
email_sender = 'epidermx6@gmail.com'
email_password = 'byde vwgs bpte ieks'
nltk.download('stopwords')
max_length = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def extract_face(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            face_landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks.landmark]
            x_min, y_min = min(face_landmarks, key=lambda x: x[0])[0], min(face_landmarks, key=lambda x: x[1])[1]
            x_max, y_max = max(face_landmarks, key=lambda x: x[0])[0], max(face_landmarks, key=lambda x: x[1])[1]
            face_region = image[y_min:y_max, x_min:x_max]
            return face_region

    return None
def face_encoding(image):
    face = extract_face(image)
    if face is not None:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(face_rgb)
        if encoding:
            print(encoding[0])
            return encoding[0]
        return None
def compare_faces(frame):
    # data=pd.read_csv("static/encodings.csv")
    current_encoding = face_encoding(frame)
    query="SELECT * FROM face"
    cursor.execute(query)
    result = cursor.fetchall()
    print(result)
    conn.commit()

    if current_encoding is not None:
        for i in result:
            user = i['phoneno']
            phon = i['name']
            print(user, "hoi thossfnflknkg n", phon)

            l = i.values()
            l = np.array(list(l)[2:])
            l = l.astype(float)
            distance = face_recognition.face_distance([current_encoding], l)[0]
            print(f"{user}: Distance - {distance}")

            print(distance)
            # You can set a threshold for matching here
            if distance < 0.50:
                print("user: ",user)  # Adjust the threshold as needed
                # Get user language from the register table based on name and phone number
                query = "SELECT language FROM register WHERE name=%s AND phone=%s;"
                
                cursor.execute(query, (phon, user))
                language_result = cursor.fetchone()
                print("language result:",language_result)
                if language_result:
                    user_language = language_result['language']
                    
                    print(f"User Language: {user_language}")

                    session['lang'] = user_language
                    print(session['lang'])
                    # Do something with the user language
                else:
                    print("User language not found")
                    # Handle case when user language is not found
                return user
        return False
    else:
        return "No"



# def get_user_language(user):
#     try:
#         # Retrieve user language from the register table based on name and phone number
#         query = "SELECT  language FROM register WHERE name=%s;"
#         cursor.execute(query, (user,))
        
#         result = cursor.fetchone()
#         print(result['language'])
#         print(result)
#         if result:
#             return result['language']
#         else:
#             return None
#     except Exception as e:
#         print(f"Error retrieving language: {e}")
#         return None
def email(body,mail):
    subject = 'Mail from MedNexus'
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = mail
    em['Subject'] = subject
    em.set_content(body,subtype="html")
    context = ssl.create_default_context()
    # Log in and send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender,mail, em.as_string())
def hello(phoneno):
    cursor.execute("SELECT email FROM son WHERE phone_parent=%s",(phoneno,))
    data3 = cursor.fetchall()
    print(data3)
    for row in data3:
        e=row['email']
    subject1='Registration success status'
    body1='Welcome to mednexus.Your registration is successful..Have a great day!!'
    e1=EmailMessage()
    e1['From']=email_sender
    e1['To']=e
    e1['Subject']=subject1
    e1.set_content(body1,subtype="html")
    context = ssl.create_default_context()
    # Log in and send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender,e, e1.as_string())
def translate_telugu_to_english(text):
    lang_mapping = {
    'English': 'en',
    'Hindi': 'hi',
    'Telugu': 'te'
    # Add more mappings as needed
}
    translator = Translator()
    xx=session['lang']#  xx is the language of the user language
    result = translator.translate(text, src=lang_mapping[xx], dest='en')
    return result.text
def remove_stop_words(input_text):
    tokens = input_text.split()  # Assuming input_text is a string
    stop_words = set(STOP_WORDS)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)
def get_doctor(sentence):
    # Load the fine-tuned model and tokenizer
    save_path = r"fine_tuned_bert_model"  # Update this path as per your fine-tuned model location
    loaded_model = BertForSequenceClassification.from_pretrained(save_path)
    loaded_tokenizer = BertTokenizer.from_pretrained(save_path)

    # Predefined symptoms and corresponding specializations
    symptoms_to_specialist = {
        "fever": "General Physician",
        "cough": "General Physician",
        "cold": "General Physician",
    }

    # Check for predefined symptoms
    user_input_without_stop_words = remove_stop_words(sentence)
    for symptom, specialist in symptoms_to_specialist.items():
        if symptom in user_input_without_stop_words:
            return specialist

    # Tokenize the input text for disease prediction
    input_encoding = loaded_tokenizer(user_input_without_stop_words, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    # Choose device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_encoding = {key: val.to(device) for key, val in input_encoding.items()}
    loaded_model.to(device)

    # Make prediction
    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model(**input_encoding)

    # Get predicted label
    logits = output.logits
    _, predicted_label = torch.max(logits, dim=1)

    # Convert predicted label to the original label using the label encoder
    # Assuming the label encoder has been saved with the model
    label_encoder_path = f'{save_path}/label_encoder.joblib'
    loaded_le = joblib.load(label_encoder_path)
    predicted_specialization = loaded_le.inverse_transform([predicted_label.item()])[0]

    return predicted_specialization

@app.route('/timeschedule')
def timeschedule():
    print("time schedule")
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM doctorregister ORDER BY id DESC LIMIT 1")
    current_schedule = cursor.fetchone()
    cursor.close()
    conn.close()
    return render_template('timeschedule.html', current_schedule=current_schedule)
@app.route('/update_schedule', methods=['POST'])
def update_schedule():
    # Get form data
    schedule_data = {f"schedule{i}": request.form[f'schedule{i}'] for i in range(1, 6)}

    # Create a new connection
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="kiosk")
    cursor = conn.cursor()

    try:
        # Check if a record exists
        cursor.execute("SELECT COUNT(*) FROM doctorregister")
        if cursor.fetchone()[0] > 0:
            # Update existing record
            update_query = """
            UPDATE doctorregister 
            SET schedule1=%s, schedule2=%s, schedule3=%s, schedule4=%s, schedule5=%s
            WHERE id=(SELECT id FROM (SELECT id FROM doctorregister ORDER BY id DESC LIMIT 1) as t)
            """
        else:
            # Insert new record
            update_query = """
            INSERT INTO doctorregister (schedule1, schedule2, schedule3, schedule4, schedule5)
            VALUES (%s, %s, %s, %s, %s)
            """

        cursor.execute(update_query, tuple(schedule_data.values()))
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

    return redirect(url_for('home'))

def get_time_slot(date,email1):
    time_slots_data = ['9:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00']
    sql = "SELECT * FROM appointments WHERE email = %s and date=%s"
    condition_value = (email1,date)
    cursor.execute(sql, condition_value)
    result=cursor.fetchall()
    print(result)
    if len(result)==0:
            return "9:00"
    else:
        for i in result:
            j=str(i['time'])
            # print(j[:-3])
            time_slots_data.remove(j[:-3])
            
        return time_slots_data[0]


def to_telugu(name):
    lang_mapping1 = {
    'English': 'en',
    'Hindi': 'hi',
    'Telugu': 'te'
    # Add more mappings as needed
        }
    translator = Translator()
    xx1=session['lang']
    result = translator.translate(name, src='en', dest=lang_mapping1[xx1])
    return result.text
def text_to_speech(text, language='te'):
    tts = gTTS(text=text, lang=language, slow=False)
    audio_bytes_io = BytesIO()
    tts.write_to_fp(audio_bytes_io)
    audio_bytes_io.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(audio_bytes_io)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(5)

# def get_face_encodings(image,number):
#     data=pd.read_csv("static/encodings.csv")
#     encoding = face_encoding(image)
#     if encoding is not None:
#             data[number]=encoding
#     data.to_csv("static/encodings.csv",index=False)
def insert_encodings(image, number, name):
    encoding = face_encoding(image)
    print(encoding)
    hello(number)
    if encoding is not None:
        num_encodings = 128
        encoding = encoding.astype(float)

        # Convert NumPy array to Python native type (float)
        encoding = list(map(float, encoding))
        # print(encoding)
        # print(number)
        # print(name)

        # Generate the SQL command for inserting data with an additional field for the name
        sql_insert_data = f"INSERT INTO face VALUES (%s, %s, {', '.join(['%s'] * num_encodings)});"

        # Combine all data into a single tuple for insertion
        insert_data = (name, number, *encoding)

        # Execute the SQL command to insert data
        cursor.execute(sql_insert_data, insert_data)
        conn.commit()
    else:
        return render_template('register.html', message="Face not detected")

@app.route('/')
def index():
    return render_template('home.html')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/doctorregister')
def doctorregister():
    return render_template('doctorregister.html')
@app.route('/asha')
def asha():
    return render_template('asha.html')
@app.route('/meet')
def meet():
    if 'name' in session:
        return render_template('join.html',name=session['name'])
    return render_template('join.html')
@app.route('/registerpage')
def registerpage():
    return render_template('register.html')
@app.route('/face_recognition')
def video():
    return render_template('index.html')
@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/voice')
def voice():
    sql = "SELECT * FROM appointments WHERE phoneno = %s and active = %s"

    condition_value = (session['user_phone'],1)
    cursor.execute(sql, condition_value)
    result=cursor.fetchall()
    print(session['name'])
    for i in result:
        url=i['link']
    if len(result)==0:
        telugu_text = "స్వాగతం మీరు తర్వాత పేజీ లో ఉన్న నీలం రంగు బటన్ నొక్కి సంభాషణని ప్రారంభించండి"
        text_to_speech(telugu_text)

        return render_template('voice.html')
    else:
        return redirect(url)
@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        # Get the uploaded image file
        image_file = request.files['image']

        if not image_file:
            return jsonify({'success': False, 'error': 'No file uploaded'})

        # Validate file type and size if needed
        if image_file.content_type not in ['image/jpeg', 'image/png']:
            return jsonify({'success': False, 'error': 'Invalid file type'})

        row_id = request.form['id']
        image_data = image_file.read()

        # Insert image data into the database
        sql = "INSERT INTO prescription (id, photo) VALUES (%s, %s)"
        val = (row_id, image_data)
        cursor.execute(sql, val)
        conn.commit()

        # Update appointment status
        sql = "UPDATE appointments SET active = 0 WHERE id=%s"
        val = (row_id,)
        cursor.execute(sql, val)
        conn.commit()

        # Fetch user data for the email
        sql = "SELECT phone FROM appointments WHERE id=%s"
        val = (row_id,)
        cursor.execute(sql, val)
        phone1 = cursor.fetchone()
        phone = phone1['phone']

        sql = "SELECT name, address FROM register WHERE phone=%s"
        val = (phone,)
        cursor.execute(sql, val)
        data = cursor.fetchone()
        name, address = data['name'], data['address']

        # Prepare and send the email
        subject = 'Mail from kiosk'
        body = f"Patient name is {name}. The phone number is {phone}. Patient address is {address}. Below is the prescription of the patient."
        em = EmailMessage()
        em['From'] = email_sender
        em['To'] = 'sameertalagadadeevi1778@gmail.com'
        em['Subject'] = subject
        em.set_content(body, subtype='html')

        # Attach the image
        em.add_attachment(image_data, maintype='image', subtype=image_file.filename.split('.')[-1])

        # Add SSL (layer of security)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(email_sender, email_password)
            smtp.sendmail(email_sender, 'sameertalagadadeevi1778@gmail.com', em.as_string())

        return jsonify({'success': True, 'message': 'Image uploaded successfully'})
    except Exception as e:
        print('Error:', str(e))
        return jsonify({'success': False, 'error': str(e)})

@app.route('/signin',methods=['POST'])
def signin():
    email=request.form['email']
    password=request.form['pass1']
    sql = "SELECT * FROM doctorregister WHERE email = %s and password=%s"
    # Provide a value for the condition (replace %s with the actual value)
    sql1="SELECT * FROM asha WHERE email = %s and password=%s"
    condition_value = (email,password)
    # Execute the SQL query with the provided condition
    cursor.execute(sql, condition_value)
    # Fetch the result (in this case, a single integer representing the count)
    result = cursor.fetchall()
    for i in result:
        name1=i['name']
        print(name1)
    cursor.execute(sql1, condition_value)
    result1=cursor.fetchall()
    print(result1)
    for i in result1:
        name2=i['name']
        print(name2)
    if len(result)==1:
        session['doctor']=True
        session['email']=email
        return redirect(url_for('index'))
    elif len(result1) == 1:
        session['asha']=True
        session['email']=email
        return redirect(url_for('index'))
    if email=="chintalaharishkumar8249@gmail.com" and password=="Harish@2003":
        session['admin']=True
        return render_template('home.html')
    session['admin']=True
    return render_template('login.html',message="Invalid Credentials")

@app.route('/asha_login_details', methods=['GET', 'POST'])
def asha_login_details():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Query the asha table
        cursor.execute("SELECT * FROM asha WHERE email = %s AND password = %s", (email, password))
        user = cursor.fetchone()

        if user:
            # Login successful
            session['user_id'] = user['id']  # Adjust 'id' to match your column name
            session['email'] = user['email']
            flash('Login successful!', 'success')

            session['user']=False
            session['doctor']=False
            session['asha']=True
            return render_template('home.html') # Redirect to asha dashboard
        else:
            # Login failed
            flash('Invalid email or password', 'error')

    return render_template('login.html')


@app.route('/doctor_login_details', methods=['GET', 'POST'])
def doctor_login_details():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Query the doctor table
        cursor.execute("SELECT * FROM doctorregister WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            # Check if the password matches
            if user['password'] == password:
                # Login successful
                # Adjust 'id' to match your column name
                session['email'] = user['email']
                flash('Login successful!', 'success')
                session['user']=False
                session['asha']=False
                session['doctor']=True
                return render_template('home.html')
            else:
                # Password doesn't match
                flash('Invalid password', 'error')
        else:
            # No user found with that email
            flash('Email not found', 'error')

    return render_template('login.html')

@app.route('/doctordata',methods=['POST'])
def doctordata():
    if request.method == 'POST':
        # Access form data
        name = request.form['user']
        qualification = request.form['qualification']
        experience = request.form['experience']
        phone= request.form['phone']
        email=request.form['email']
        pass1=request.form['pass1']
        sql = "INSERT INTO doctorregister VALUES (%s, %s,%s,%s,%s,%s)"
        val = (name,qualification,experience,phone,email,pass1)
        cursor.execute(sql, val)
        conn.commit()
        return render_template('home.html')
def get_db_connection():
    global conn
    try:
        if conn is None or not conn.is_connected():
            conn = mysql.connector.connect(host="localhost",user="root",password="",database="kiosk")
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None
@app.route('/ashadata',methods=['POST'])
def ashadata():
    if request.method == 'POST':
        # Access form data
        name =request.form['user']
        phone=request.form['phone']
        email1=request.form['email']
        pass1=request.form['pass1']
        # Generate the ID
        cursor.execute("SELECT MAX(id) FROM asha")
        result = cursor.fetchone()
        print(result)
        max_id = result['MAX(id)']
        print(max_id)
        if max_id is None:
            id = 1
        else:
            id = max_id + 1
        # Insert data into the table
        sql = "INSERT INTO asha (id, name, phone, email, password) VALUES (%s, %s, %s, %s, %s)"
        val = (id, name, phone, email1, pass1)
        cursor.execute(sql, val)
        conn.commit()
        return render_template('home.html')
    
@app.route('/asha_login')
def asha_login():
    return render_template('asha_login.html')

@app.route('/doctor_login')
def doctor_login():
    return render_template('doctor_login.html')

@app.route('/backend-endpoint', methods=['POST'])
def backend_endpoint():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Extract the base64-encoded frame from the data
        frame_data_url = data.get('frame', '')
        _, frame_data = frame_data_url.split(',')

        # Decode the base64 data to obtain the binary image data
        frame_binary = base64.b64decode(frame_data)

        # Convert the binary image data to a NumPy array
        frame_np = np.frombuffer(frame_binary, dtype=np.uint8)

        # Decode the NumPy array to a CV2 image
        frame_cv2 = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        matched_face_path = compare_faces(frame_cv2)
        if matched_face_path=="No":
            return jsonify({'status':'No'})
        if matched_face_path:
            print(matched_face_path)
            number=matched_face_path.split(".")[0]
            cursor.execute("SELECT name FROM register WHERE phone=%s",(number,))
            data3 = cursor.fetchall()
            print(data3)
            for row in data3:
                name=row['name']
            session['name']=name
            session['telugu']=to_telugu(name)
            session['user_phone']=number
            # hello(session['name'],session['user_phone'])
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status':'failed'})
    except Exception as e:
        # Handle any exceptions that may occur during frame processing
        print('Error processing frame:', e)
        return jsonify({'status': 'error'})


@app.route('/success')
def success():
    roomID =random.randint(1000,10000)
    url = f"http://127.0.0.1:5000/meet?roomID={roomID}"
    email(url,"narendravarma18042004@gmail.com")
    return render_template('success.html',name=session['name'])
@app.route('/register', methods=['POST'])
def register():

    if request.method == 'POST':
        # Access form data
        son_name = request.form['guardianName']
        son_phone = request.form['guardianMobile']
        son_email = request.form['guardianEmail']
        pat_name=request.form['patientName']
        pat_phone=request.form['patientMobile']
        pat_addr=request.form['address']
        pat_lan=request.form['language']
      
          # Get password from form
        captured_photo_data = request.form['capturedPhotoData']

        # Extract the base64-encoded image data
        match = re.match(r'data:image/(\w+);base64,(.+)', captured_photo_data)
        image_data = match.group(2)

        if not son_name or not son_phone or not son_email:  # Check for password as well
            flash('All fields are required.', 'error')
            return render_template('register.html')
        
        photo_path = None
        if captured_photo_data:
            match = re.match(r'data:image/(\w+);base64,(.+)', captured_photo_data)
            if match:
                image_data = match.group(2)
                decoded_image_data = base64.b64decode(image_data)
                os.makedirs('static/uploads', exist_ok=True)
                file_path = f'static/uploads/{pat_phone}.png'
                with open(file_path, 'wb') as file:
                    file.write(decoded_image_data)
                photo_path = file_path

        # Hash the password
        # hashed_password = generate_password_hash(password)


        decoded_image_data = base64.b64decode(image_data)

        # Save the image to a file
        os.makedirs('static/uploads', exist_ok=True)
        file_path = f'static/uploads/{pat_phone}.png'
        with open(file_path, 'wb') as file:
            file.write(decoded_image_data)

        # Family member data (unchanged)
        # family_member1_name = request.form.get('familyName1', '')
        # family_member1_phone = request.form.get('familyPhone1', '')
        # family_member1_address = request.form.get('familyAddress1', '')
        # family_member1_relation = request.form.get('familyRelation1', '')

        # family_member2_name = request.form.get('familyName2', '')
        # family_member2_phone = request.form.get('familyPhone2', '')
        # family_member2_address = request.form.get('familyAddress2', '')
        # family_member2_relation = request.form.get('familyRelation2', '')

        # Check if phone number already exists
        cursor.execute("SELECT * FROM register WHERE phone = %s", (pat_phone,))
        result = cursor.fetchone()
        if result:
            return render_template('register.html', message="Phone number already exists")

# Insert data into the register table
        sql_register = """INSERT INTO register (name, phone, photo_path, son_name,address,language) 
                  VALUES (%s, %s, %s, %s,%s,%s)"""
        val_register = (pat_name, pat_phone, photo_path, son_name,pat_addr,pat_lan)
        cursor.execute(sql_register, val_register)
        conn.commit()



        sql_son = """INSERT INTO son (name, phoneno, email,phone_parent) 
                     VALUES (%s, %s,%s, %s)"""
        val_son = (son_name, son_phone,son_email,pat_phone)
        cursor.execute(sql_son, val_son)

        conn.commit()
       
        session['user'] = True
       
        match = re.match(r'data:image/(\w+);base64,(.+)', captured_photo_data)
        image_data = match.group(2)
        decoded_image_data = base64.b64decode(image_data)
        os.makedirs('static/uploads', exist_ok=True)
        # Save the image to a file
        file_path = f'static/uploads/{pat_phone}.png'
        with open(file_path, 'wb') as file:
            file.write(decoded_image_data)
        
        frame_np = np.frombuffer(decoded_image_data, dtype=np.uint8)
        frame_cv2 = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        
        
        
        insert_encodings(frame_cv2,pat_phone,pat_name)


        # print(f"Name: {name}, Phone: {phone}, Address: {address}")
        return render_template('index.html')




@app.route('/translate', methods=['POST'])
def translate():
    try:
        # Parse the JSON string back into a list
        
        answers_json = request.form.get('answers')
        print(answers_json)
        translated_text = translate_telugu_to_english(answers_json)
        print(translated_text)
        answers = json.loads(translated_text) if translated_text else []
        print("Received answers:", answers)
        print("Type of answers:", type(answers))
        # Example of processing the answers (just printing for now)
        for answer in answers:
            print(answer)
        print(type(answers[0]))
        responses = {
    "What is the name of the disease you are suffering from?": answers[0],
    "How long have you been suffering from fever?": answers[1],
    "Do you have any other symptoms?": answers[2]
    }
    
        # Translate the text from Telugu to English (replace with your actual translation function)
        translated_text_lower = answers[0]
        print(translated_text_lower)
        
        # Store the lowercase translated text in the session
        session['translate_text'] = translated_text_lower
        
        # Return the lowercase translated text in the response
        return jsonify({'status': 'success', 'translatedText': translated_text_lower})
    
    except Exception as e:
        print(f"Error during translation: {e}")
        return jsonify({'status': 'error', 'message': 'Translation failed'})
    
from datetime import datetime, timedelta

def get_next_available_slot(schedule, current_time):
    for slot in schedule:
        if isinstance(slot, str):
            try:
                slot_time = datetime.strptime(slot, '%H:%M:%S').time()
            except ValueError:
                try:
                    slot_time = datetime.strptime(slot, '%H:%M').time()
                except ValueError:
                    continue  # Skip invalid time strings
        elif isinstance(slot, datetime):
            slot_time = slot.time()
        elif isinstance(slot, timedelta):
            slot_time = (datetime.min + slot).time()
        else:
            continue  # Skip invalid types

        slot_datetime = current_time.replace(hour=slot_time.hour, minute=slot_time.minute, second=slot_time.second, microsecond=0)
        
        if slot_datetime > current_time:
            return slot_time.strftime('%H:%M:%S')
    
    # If no slot is available today, return the first slot for tomorrow
    if schedule:
        if isinstance(schedule[0], str):
            return schedule[0]
        elif isinstance(schedule[0], datetime):
            return schedule[0].strftime('%H:%M:%S')
        elif isinstance(schedule[0], timedelta):
            return (datetime.min + schedule[0]).strftime('%H:%M:%S')
    
    return None  # Return None if no valid slots are found

@app.route('/predict')
def predict():
    predicted = get_doctor(session['translate_text'])
    print(session['lang'])
    print(predicted)
    
    sql = "SELECT * FROM doctorregister WHERE qualification = %s"
    condition_value = (predicted,)
    cursor.execute(sql, condition_value)
    result = cursor.fetchall()
    
    if len(result) == 0:
        return render_template('home.html', message="No doctor available related to that field")
    
    for row in result:
        email1 = row['email']
        doctor_schedule = [row['schedule1'], row['schedule2'], row['schedule3'], row['schedule4'], row['schedule5']]
    
    roomID = random.randint(1000, 10000)
    session['roomID'] = roomID
    url = f"http://127.0.0.1:5000/meet?roomID={roomID}"
    
    current_time = datetime.now()
    formatted_date = current_time.strftime('%Y-%m-%d')
    
    # Get next available time slot
    time = get_next_available_slot(doctor_schedule, current_time)
    
    if time is None:
        return render_template('home.html', message="No available time slots for this doctor")
    
    sql = "INSERT INTO appointments(phoneno,email,link,date,time,active) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (session['user_phone'], email1, url, formatted_date, time, 1)
    cursor.execute(sql, val)
    conn.commit()
    
    body = f"Hello, I am {session['name']}. This is the meeting link {url}. The time is {time}. Please check through that."
    email(body, email1)
    
    return render_template('success.html', time=time, name=to_telugu(session['name']), doctor=to_telugu(predicted))


@app.route('/asha_data_fetch')
def asha_data_fetch():
    query = """
        SELECT
            p.id AS id,
            r.name AS name,
             r.address AS address,


            p.photo AS photo
        FROM
            prescription p
        JOIN
            appointments a ON p.id = a.id
        JOIN
            register r ON a.phoneno = r.phone;
    """

    cursor.execute(query)
    results = cursor.fetchall()
    print(results)
    for row in results:
        row['photo'] = base64.b64encode(row['photo']).decode('utf-8')
    return render_template('ashadata.html', data=results)
@app.route('/ashaschedule')
def ashaschedule():
    return render_template('ashadata.html')
@app.route('/doctorschedule')
def doctorschedule():
    sql = "SELECT id,link,date,time FROM appointments WHERE email = %s and active = %s "
    condition_value = (session['email'],1)  
    cursor.execute(sql, condition_value)
    result=cursor.fetchall()
    return render_template('doctordata.html',data=result)
@app.route('/logout')
def logout():
    session.clear()
    return render_template('home.html')

from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
from googletrans import Translator
from transformers import pipeline
from gtts import gTTS
from io import BytesIO
import base64

# Set API keys
genai.configure(api_key="AIzaSyBDrzNjBunDdf68_59ocejetjIaKQLxOuM")

# Mapping of full language names to language codes for gTTS
language_code_map = {
    'en': 'english',
    'hi': 'hindi',
    'te': 'telugu',
    'ta': 'tamil',
    'bn': 'bengali',
    'mr': 'marathi',
    'gu': 'gujarati',
    'kn': 'kannada',
    'ml': 'malayalam',
    'pa': 'punjabi',
    'ur': 'urdu'
}

# Load the summarization pipeline with BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def summarize_text(text):
    try:
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        if not summary or len(summary[0]['summary_text']) == 0:
            return text
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return text

def translate_text(text, dest_language):
    translator = Translator()
    translated = translator.translate(text, dest=dest_language)
    return translated.text

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chatbot1', methods=['POST'])
def chatbot1():
    data = request.json
    text = data.get('text')
    preferred_language = data.get('language')

    if not text or preferred_language not in language_code_map:
        return jsonify({"error": "Invalid input"}), 400

    try:
        # Generate response using AI
        model = genai.GenerativeModel('gemini-pro')
        generated_response = model.generate_content(
            [
                "You are a highly skilled AI, answer the questions given within a maximum of 1000 characters.",
                text
            ]
        ).text

        # Summarize the response
        summarized_response = summarize_text(generated_response)

        # Translate the response
        translated_response = translate_text(summarized_response, dest_language=preferred_language)
        
        # Convert to speech
        tts = gTTS(text=translated_response, lang=preferred_language, slow=False)
        audio_bytes_io = BytesIO()
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)
        
        response = {
            'generated_response': generated_response,
            'summarized_response': summarized_response,
            'translated_response': translated_response,
            'audio': base64.b64encode(audio_bytes_io.getvalue()).decode('utf-8')
        }

        return jsonify(response)
    except Exception as e:
        print(f"Error during chatbot processing: {e}")
        return jsonify({"error": "An error occurred while processing the request"}), 500

@app.route('/database')
def database():
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM appointments")
    appointments = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('database.html', appointments=appointments)

@app.route('/appointment/add', methods=['POST'])
def add_appointment():
    data = request.json
    connection = get_db_connection()
    cursor = connection.cursor()
    query = """INSERT INTO appointments (email, name, phoneno, link, date, time, active) 
               VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    values = (data['email'], data['name'], data['phoneno'], data['link'], 
              data['date'], data['time'], data['active'])
    cursor.execute(query, values)
    conn.commit()
    new_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return jsonify({"id": new_id, "message": "Appointment added successfully"})

@app.route('/appointment/update', methods=['POST'])
def update_appointment():
    data = request.json
    connection = get_db_connection()
    cursor = connection.cursor()
    query = """UPDATE appointments SET email=%s, name=%s, phoneno=%s, link=%s, 
               date=%s, time=%s, active=%s WHERE id=%s"""
    values = (data['email'], data['name'], data['phoneno'], data['link'], 
              data['date'], data['time'], data['active'], data['id'])
    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Appointment updated successfully"})

@app.route('/appointment/delete', methods=['POST'])
def delete_appointment():
    data = request.json
    connection = get_db_connection()
    if connection is None:
        return jsonify({"error": "Database connection error"}), 500
    
    cursor = connection.cursor()
    query = "DELETE FROM appointments WHERE id=%s"
    cursor.execute(query, (data['id'],))
    connection.commit()
    cursor.close()
    return jsonify({"message": "Appointment deleted successfully"})

if __name__ == '__main__':
    app.run(debug=True)