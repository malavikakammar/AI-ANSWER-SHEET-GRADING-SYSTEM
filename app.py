from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
from nltk.corpus import stopwords
import pytesseract
import re
import nltk
import logging
import seaborn as sns
import numpy as np
import plotly.express as px
import pandas as pd

# Configure logging
logging.basicConfig(filename='app_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Define the TF-IDF model
def calculate_similarity_tfidf(student_text, model_text, max_marks):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([student_text, model_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * max_marks, 2)

# Extract text from an image using Tesseract
def extract_text_from_image(image_file):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

# Preprocess answers by extracting question-answer pairs
def preprocess_answers(text):
    text = text.replace('\n', ' ')
    regex = r"A\d\.\s(.*?)(?=A\d\.|\Z)"
    matches = re.findall(regex, text, re.DOTALL)
    return {f"a{i+1}": match.strip() for i, match in enumerate(matches)}

# Remove stopwords from the answers
def remove_stopwords(text_dict):
    for key, value in text_dict.items():
        if isinstance(value, str):
            words = value.split()
            filtered_words = [word for word in words if word.lower() not in stop_words]
            text_dict[key] = ' '.join(filtered_words)
    return text_dict

# Evaluate student answers against model answers
def evaluate_answers(student_answers, model_answers, weightage):
    scores = {}
    all_questions = set(student_answers.keys()) | set(model_answers.keys())
    for question in all_questions:
        if question in student_answers and question in model_answers:
            student_answer = student_answers[question]
            model_answer = model_answers[question]
            scores[question] = calculate_similarity_tfidf(student_answer, model_answer, weightage)
    return scores

# Hypothetical ground truth (for demonstration purposes)
def get_ground_truth(student_answers, model_answers):
    ground_truth = {}
    for question in student_answers:
        ground_truth[question] = 1 if calculate_similarity_tfidf(student_answers[question], model_answers[question], 20) > 10 else 0
    return ground_truth

# Plot confusion matrix using Seaborn
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plot accuracy graph using Plotly
def plot_accuracy_graph(accuracy_history):
    df = pd.DataFrame({'Epoch': range(len(accuracy_history)), 'Accuracy': accuracy_history})
    fig = px.line(df, x='Epoch', y='Accuracy', title='Accuracy Over Time')
    fig.show()

# Plot loss graph using Plotly
def plot_loss_graph(loss_history):
    df = pd.DataFrame({'Epoch': range(len(loss_history)), 'Loss': loss_history})
    fig = px.line(df, x='Epoch', y='Loss', title='Loss Over Time')
    fig.show()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Process route
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

        # Hypothetical ground truth and predictions
        ground_truth = get_ground_truth(student_answers, model_answers)
        predictions = {q: 1 if s > 10 else 0 for q, s in scores.items()}

        # Calculate metrics
        y_true = list(ground_truth.values())
        y_pred = list(predictions.values())
        accuracy = accuracy_score(y_true, y_pred)
        loss = np.mean([abs(s - 20) for s in scores.values()])  # Custom loss

        # Display visualizations
        plot_confusion_matrix(y_true, y_pred)
        plot_accuracy_graph([accuracy])  # Single point for simplicity
        plot_loss_graph([loss])  # Single point for simplicity

        # Render results on a webpage
        return render_template(
            'index.html',
            subject_name=subject_name,
            roll_no=roll_no,
            student_name=student_name,
            scores=scores,
            total_score=total_score,
            results_ready=True  # Flag to indicate results are ready
        )

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return render_template('index.html', error=str(e), results_ready=False)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)