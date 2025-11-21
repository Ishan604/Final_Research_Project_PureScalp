from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session , flash
from models.doctor import Doctor
from models.patient import Patient
from models.questionnaire import Questionnaire
from models.marks import Marks
from extensions import db

doctor_bp = Blueprint('doctor', __name__)


# Doctor Dashboard Route
@doctor_bp.route('/doctor_dashboard')
def doctor_dashboard():
    # Check if doctor is logged in
    if 'doctor_id' not in session or session.get('user_type') != 'doctor':
        flash('Please login as a doctor to access the dashboard', 'error')
        return redirect(url_for('auth.login'))
    
    try:
        # Get doctor information
        doctor_id = session['doctor_id']
        doctor = db.session.query(Doctor).filter_by(DoctorID=doctor_id).first()
        
        if not doctor:
            flash('Doctor information not found', 'error')
            return redirect(url_for('auth.login'))
        
        print(f"Doctor found: {doctor.Name}")
        
        # Get all patients first
        all_patients = db.session.query(Patient).all()
        print(f"Found {len(all_patients)} patients")
        
        # Process patient data with simpler queries
        patients_data = []
        for patient in all_patients:
            try:
                # Get questionnaire count for this patient
                questionnaire_count = db.session.query(Questionnaire).filter_by(PatientID=patient.PatientID).count()
                
                # Get latest questionnaire for this patient
                latest_questionnaire = db.session.query(Questionnaire).filter_by(PatientID=patient.PatientID).order_by(Questionnaire.completionDate.desc()).first()
                
                # Get latest marks for this patient
                latest_marks = db.session.query(Marks).filter_by(patientId=patient.PatientID).order_by(Marks.marks.desc()).first()
                
                patients_data.append({
                    'patient': patient,
                    'total_assessments': questionnaire_count,
                    'last_assessment': latest_questionnaire.completionDate.strftime('%Y-%m-%d %H:%M') if latest_questionnaire and latest_questionnaire.completionDate else 'Never',
                    'latest_score': latest_marks.marks if latest_marks else 'N/A',
                    'status': 'Active' if latest_marks else 'Incomplete'
                })
            except Exception as patient_error:
                print(f"Error processing patient {patient.PatientID}: {patient_error}")
                # Add patient with basic info if there's an error
                patients_data.append({
                    'patient': patient,
                    'total_assessments': 0,
                    'last_assessment': 'Never',
                    'latest_score': 'N/A',
                    'status': 'Incomplete'
                })
        
        # Get dashboard statistics with simpler queries
        total_patients = len(all_patients)
        
        # Count patients with marks
        patients_with_assessments = db.session.query(Marks.patientId).distinct().count()
        
        # Count total questionnaires
        total_assessments = db.session.query(Questionnaire).count()
        
        # Calculate average score
        avg_score_result = db.session.query(db.func.avg(Marks.marks)).scalar()
        avg_score = round(avg_score_result, 1) if avg_score_result else 0
        
        print(f"Stats - Total patients: {total_patients}, With assessments: {patients_with_assessments}, Total assessments: {total_assessments}, Avg score: {avg_score}")
        
        dashboard_stats = {
            'total_patients': total_patients,
            'patients_with_assessments': patients_with_assessments,
            'total_assessments': total_assessments,
            'avg_score': avg_score
        }
        
        return render_template('doctordashboard.html',
                             doctor=doctor,
                             patients_data=patients_data,
                             stats=dashboard_stats)
        
    except Exception as e:
        print(f"Error in doctor dashboard: {e}")
        import traceback
        traceback.print_exc()
        flash('Error loading dashboard data', 'error')
        return render_template('doctordashboard.html',
                             doctor=doctor if 'doctor' in locals() else None,
                             patients_data=[],
                             stats={'total_patients': 0, 'patients_with_assessments': 0, 
                                   'total_assessments': 0, 'avg_score': 0})

# Patient Details Route
@doctor_bp.route('/patient/<int:patient_id>')
def patient_details(patient_id):
    # Check if doctor is logged in
    if 'doctor_id' not in session or session.get('user_type') != 'doctor':
        flash('Please login as a doctor to access patient details', 'error')
        return redirect(url_for('auth.login'))
    
    try:
        # Get doctor information
        doctor_id = session['doctor_id']
        doctor = db.session.query(Doctor).filter_by(DoctorID=doctor_id).first()
        
        if not doctor:
            flash('Doctor information not found', 'error')
            return redirect(url_for('auth.login'))
        
        # Get patient information
        patient = db.session.query(Patient).filter_by(PatientID=patient_id).first()
        
        if not patient:
            flash('Patient not found', 'error')
            return redirect(url_for('doctor.doctor_dashboard'))
        
        # Get patient's questionnaires and marks
        questionnaires = db.session.query(Questionnaire).filter_by(PatientID=patient_id).order_by(Questionnaire.completionDate.desc()).all()
        marks = db.session.query(Marks).filter_by(patientId=patient_id).order_by(Marks.marks.desc()).all()
        
        
    except Exception as e:
        print(f"Error in patient details: {e}")
        flash('Error loading patient details', 'error')
        return redirect(url_for('doctor.doctor_dashboard'))