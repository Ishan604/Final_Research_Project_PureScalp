from extensions import db

class Patient(db.Model):
    __tablename__ = 'Patient'
    
    PatientID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(100), nullable=False)
    Email = db.Column(db.String(100), unique=True, nullable=False)
    Age = db.Column(db.Integer, nullable=True)
    Gender = db.Column(db.String(10), nullable=True)

    def __repr__(self):  # String representation of the Patient object
        return f"<Patient {self.Name}>"
