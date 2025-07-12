from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from nltk.corpus import stopwords
import pytesseract
import pandas as pd
import re
from werkzeug.utils import secure_filename
import nltk
import logging
from flask_mysqldb import MySQL
import json

logging.basicConfig(filename='app_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

app = Flask(__name__, template_folder='templates')

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '8088'
app.config['MYSQL_DB'] = 'grading_system'
mysql = MySQL(app)

# Define the TF-IDF model
def calculate_similarity_tfidf(student_text, model_text, max_marks):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([student_text, model_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * max_marks, 2)

def extract_text_from_image(image_file):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

def preprocess_answers(text):
    text = text.replace('\n', ' ')
    regex = r"(A\d+)\.\s(.*?)(?=(A\d+\.\s|$))"
    matches = re.findall(regex, text, re.DOTALL)
    return {match[0].lower(): match[1].strip() for match in matches}

def remove_stopwords(text_dict):
    stop_words = set(stopwords.words('english'))
    for key, value in text_dict.items():
        if isinstance(value, str):
            words = value.split()
            filtered_words = [word for word in words if word.lower() not in stop_words]
            text_dict[key] = ' '.join(filtered_words)
    return text_dict

def evaluate_answers(student_answers, model_answers, weightage):
    scores = {}
    all_questions = set(student_answers.keys()) | set(model_answers.keys())
    for question in all_questions:
        if question in student_answers and question in model_answers:
            student_answer = student_answers[question]
            model_answer = model_answers[question]
            scores[question] = calculate_similarity_tfidf(student_answer, model_answer, weightage)
    return scores

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # Collect form data
        subject_name = request.form['subject_name']
        roll_no = int(request.form['roll_no'])
        student_name = request.form['student_name']

        # Process model file
        model_file = request.files['model_file']
        model_answers = preprocess_answers(model_file.read().decode('utf-8'))

        # Process student image files
        student_image_files = request.files.getlist('student_image_files')
        student_text = ""
        for file in student_image_files:
            student_text += extract_text_from_image(file)

        # Preprocess and evaluate answers
        student_answers = preprocess_answers(student_text)
        model_answers = remove_stopwords(model_answers)
        student_answers = remove_stopwords(student_answers)

        # Default weightage = 20 for each question
        scores = evaluate_answers(student_answers, model_answers, weightage=20)
        total_score = sum(scores.values())

        # Save results to MySQL database
        if mysql.connection:
            cur = mysql.connection.cursor()
            cur.execute(""" 
                INSERT INTO student_scores (roll_no, student_name, subject_name, total_score, scores)
                VALUES (%s, %s, %s, %s, %s)
            """, (roll_no, student_name, subject_name, total_score, json.dumps(scores)))
            mysql.connection.commit()
            cur.close()
        else:
            logging.error("MySQL connection is not established.")
            return render_template('index.html', error="Database connection failed.", results_ready=False)

        # Render results on a webpage
        return render_template(
            'index.html',
            subject_name=subject_name,
            roll_no=roll_no,
            student_name=student_name,
            scores=scores,
            total_score=total_score,
            results_ready=True
        )

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return render_template('index.html', error=str(e), results_ready=False)
    
@app.route('/update', methods=['POST'])
def update():
    try:
        # Update database logic here
        roll_no = request.form['roll_no']
        subject_name = request.form['subject_name']
        student_name = request.form['student_name']
        scores = request.form['scores']
        total_score = request.form['total_score']

        # Update the database with the form data
        cur = mysql.connection.cursor()
        cur.execute("""
            UPDATE student_scores
            SET total_score = %s, scores = %s
            WHERE roll_no = %s AND subject_name = %s AND student_name = %s
        """, (total_score, scores, roll_no, subject_name, student_name))
        mysql.connection.commit()
        cur.close()

        # After successful update, pass 'update_success' flag to the template
        return render_template('index.html', update_success=True, results_ready=True,
                               student_name=student_name, roll_no=roll_no, subject_name=subject_name,
                               total_score=total_score, scores=json.loads(scores))
    
    except Exception as e:
        logging.error(f"Error during database update: {e}")
        return render_template('index.html', error="Database update failed.", results_ready=False)

    


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
