#!/usr/bin/env python3
"""
Simple database test script to check connectivity and basic queries
"""

from app import app
from extensions import db
from models.doctor import Doctor
from models.patient import Patient
from models.questionnaire import Questionnaire
from models.marks import Marks

def test_database():
    with app.app_context():
        try:
            print("Testing database connection...")
            
            # Test doctor table
            print("\n1. Testing Doctor table:")
            doctors = db.session.query(Doctor).all()
            print(f"   Found {len(doctors)} doctors")
            for doctor in doctors:
                print(f"   - Dr. {doctor.Name} ({doctor.Email})")
            
            # Test patient table
            print("\n2. Testing Patient table:")
            patients = db.session.query(Patient).all()
            print(f"   Found {len(patients)} patients")
            for patient in patients:
                print(f"   - {patient.Name} ({patient.Email})")
            
            # Test questionnaire table
            print("\n3. Testing Questionnaire table:")
            questionnaires = db.session.query(Questionnaire).all()
            print(f"   Found {len(questionnaires)} questionnaires")
            for q in questionnaires:
                print(f"   - Patient ID: {q.PatientID}, Date: {q.completionDate}")
            
            # Test marks table
            print("\n4. Testing Marks table:")
            marks = db.session.query(Marks).all()
            print(f"   Found {len(marks)} marks records")
            for mark in marks:
                print(f"   - Patient ID: {mark.patientId}, Score: {mark.marks}")
            
            # Test relationships
            print("\n5. Testing relationships:")
            if patients:
                patient = patients[0]
                print(f"   Patient {patient.Name} has:")
                print(f"   - {len(patient.questionnaires)} questionnaires")
                print(f"   - {len(patient.marks)} marks records")
            
            print("\nDatabase test completed successfully!")
            
        except Exception as e:
            print(f"Error testing database: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_database()
