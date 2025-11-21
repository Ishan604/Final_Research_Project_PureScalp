from app import app
from extensions import db
from models.doctor import Doctor
from models.patient import Patient
from models.questionnaire import Questionnaire
from models.marks import Marks

# Create all tables based on the models defined in SQLAlchemy
with app.app_context():
    db.create_all()  # This will create the tables in the database

print("Database tables created successfully.")
