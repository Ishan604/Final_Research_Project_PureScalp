from extensions import db

class Doctor(db.Model):
    __tablename__ = 'Doctor'
     
    DoctorID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(100), nullable=False)
    Email = db.Column(db.String(100), unique=True, nullable=False)
    Password = db.Column(db.String(10), nullable=False)
    Specialization = db.Column(db.String(100), nullable=True)

    def __repr__(self): # String representation of the Doctor object
        return f"<Doctor {self.Name}>"
