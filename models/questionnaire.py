from extensions import db
from datetime import datetime

class Questionnaire(db.Model):
    __tablename__ = 'Questionnaire'
    
    QuestionnaireID = db.Column(db.Integer, primary_key=True)

    # Foreign key to Patient model
    PatientID = db.Column(db.Integer, db.ForeignKey('Patient.PatientID'), nullable=False)

    # Change to 'questions' to match your database column  
    questions = db.Column(db.String(255), nullable=False)

    # Change to 'answers' to match your database column  
    answers = db.Column(db.String(255), nullable=True)

    # Change to 'completionDate' to match your database column  
    completionDate = db.Column(db.DateTime, default=datetime.now())

    # Relationship to Patient model
    patient = db.relationship('Patient', backref=db.backref('questionnaires', lazy=True))

    def __repr__(self):
        return f"<Questionnaire {self.questions}>"
