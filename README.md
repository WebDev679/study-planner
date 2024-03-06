# Syllabus to Study Plan Converter

## Introduction

This Django web application allows users to upload their class syllabus in PDF format, and utilizes the GPT-4 API to generate a personalized study plan. The application processes the uploaded PDF, extracts meaningful text, and communicates with the OpenAI API to semantically analyze the content and generate a study plan tailored to the user's syllabus.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed on your system:
- Python 3.8 or higher
- Django 3.2 or higher
- OpenAI Python package
- PyMuPDF (fitz) for PDF processing

You can install the necessary Python packages using pip:

```bash
pip install Django openai PyMuPDF

git clone https://github.com/yourusername/syllabus-to-study-plan.git

cd syllabus-to-study-plan

python manage.py migrate

python manage.py runserver

```

### Access the application

Open your web browser and navigate to http://127.0.0.1:8000/upload/ to start using the application.

You can access a syllabus from the directory sample_syllabus to test the platform out. 

The demo video for this web app can be accessed through the link below 

https://drive.google.com/file/d/11GtW6ZCraK1my9ipvysU_Px4zOeC97Vq/view?usp=sharing
