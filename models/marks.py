from extensions import db

class Marks(db.Model):
    __tablename__ = 'Marks'
    
    marksId = db.Column(db.Integer, primary_key=True)

    # Foreign key to Patient model
    patientId = db.Column(db.Integer, db.ForeignKey('Patient.PatientID'), nullable=False) 
    marks = db.Column(db.Integer, nullable=False)
    
    # Relationship to Patient model
    patient = db.relationship('Patient', backref=db.backref('marks', lazy=True)) 

    def __repr__(self):
        return f"<Marks PatientID: {self.patientId}, Marks: {self.marks}>"
