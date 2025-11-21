import json
from flask import Blueprint, request, render_template, redirect, url_for, session
from models.patient import Patient
from models.questionnaire import Questionnaire
from models.marks import Marks
from extensions import db
from datetime import datetime

# Create blueprint for questionnaire routes
questionnaire_bp = Blueprint('questionnaire', __name__)

# Load questions from JSON file
with open('questions.json', 'r') as f:
    questions = json.load(f)

@questionnaire_bp.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    # Get the patient's ID from the session
    patient_id = session.get('patient_id')
    if not patient_id:
        return redirect(url_for('patient.patient_registration'))  # Redirect to registration if no patient_id

    # Retrieve the current question ID from the session
    current_question_id = session.get('current_question_id', 1)

    # Ensure the current question ID is valid
    current_question = next((q for q in questions['questions'] if q['question_id'] == current_question_id), None)
    
    if not current_question:
        # If there's no valid question, start from the first question
        session['current_question_id'] = 1
        return redirect(url_for('questionnaire.questionnaire'))  # Redirect to the first question

    if request.method == 'POST':
        # Store the patient's answer
        answer = request.form.get(f'question{current_question_id}')  # Get the answer to the current question
        question_text = current_question['question']  # Store the question text
        completion_date = datetime.now()  # Store the current date/time
        patientID = session['patient_id']  # Ensure patient_id is in session
        
        if answer:
            # Create a new Questionnaire object based on the user's answer
            new_answer = Questionnaire(
                PatientID=patientID,  # Use the patient ID from the session
                questions=question_text,  # Store the question text
                answers=answer,  # Store the user's answer
                completionDate=completion_date  # Store the current date/time
            )

            db.session.add(new_answer)  # Add the new answer to the session
            db.session.commit()  # Commit to the database

        # Determine the next question based on the answer
        next_question_id = current_question['next_question'].get(answer)

        # Check if the answer leads to "End"
        if next_question_id == "End":
            # Calculate and store marks before showing thank you page
            calculate_and_store_marks(patientID)
            return render_template('thank_you.html')  # Redirect to Thank You page

        # If there's a next question, store it in session and redirect to it
        if next_question_id:
            session['current_question_id'] = next_question_id
            return redirect(url_for('questionnaire.questionnaire'))

        # If no next question, find the next root question
        else:
            # Find the next root question
            next_root_question = next((q for q in questions['questions'] if q['question_id'] > current_question_id and 'next_question' not in q), None)
            
            if next_root_question:
                session['current_question_id'] = next_root_question['question_id']
                return redirect(url_for('questionnaire.questionnaire'))
            
        return redirect(url_for('main.index'))  # Redirect to index if no next question is found

    # If it's a GET request, render the current question
    return render_template('questionnaire.html', question=current_question)

def calculate_and_store_marks(patient_id):
    """Calculate marks based on questionnaire responses and store in Marks table"""
    try:
        # Count the number of questionnaire entries for this patient
        question_count = db.session.query(Questionnaire).filter_by(PatientID=patient_id).count()
        
        # Calculate marks (count * 5)
        calculated_marks = question_count * 5
        
        # Check if marks already exist for this patient to avoid duplicates
        existing_marks = db.session.query(Marks).filter_by(patientId=patient_id).first()
        
        if existing_marks:
            # Update existing marks
            existing_marks.marks = calculated_marks
        else:
            # Create new marks entry
            new_marks = Marks(patientId=patient_id, marks=calculated_marks)
            db.session.add(new_marks)
        
        db.session.commit()
        print(f"Marks calculated and stored for patient {patient_id}: {calculated_marks}")
        
    except Exception as e:
        db.session.rollback()
        print(f"Error calculating marks for patient {patient_id}: {str(e)}")
