# AI Answer Sheet Grading System 

This is a Flask-based AI-powered answer sheet grading system that allows instructors to upload student-written answer sheet images and automatically grade them using NLP techniques.

## Features

- Upload answer sheet images (`.jpg`, `.png`, etc.)
- Extract answers using OCR (Tesseract)
- Remove stopwords for better evaluation
- Match answers using TF-IDF and Cosine Similarity
- Automatically compute and display scores
- Store results in a MySQL database
- Update scores via form
- Error logging via `app_errors.log`

## Technologies Used

- Python (Flask)
- OCR: Tesseract
- NLP: NLTK, TF-IDF
- ML: Scikit-learn
- Database: MySQL
- Frontend: HTML (Jinja2 templates)



