from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session
from models.patient import Patient
from extensions import db

# Create blueprint for patient routes
patient_bp = Blueprint('patient', __name__)

@patient_bp.route('/patientregistration', methods=['GET', 'POST'])
def patient_registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        age = request.form['age']
        gender = request.form['gender']

        new_patient = Patient(Name=name, Email=email, Age=age, Gender=gender)
        try:
            db.session.add(new_patient)
            db.session.commit()
            session['patient_id'] = new_patient.PatientID  # Store patient ID in session
            return redirect(url_for('questionnaire.questionnaire'))  # Redirect to questionnaire page after successful registration
        except Exception as e:
            db.session.rollback() # Rollback the session in case of error
            return jsonify({"error": str(e)}), 400

    return render_template('patientregistration.html')
