from flask import Flask
from extensions import db  # Import the SQLAlchemy instance

app = Flask(__name__) # Initialize the Flask application

app.config['SECRET_KEY'] = '100'  # Secret key for session management

app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc://@LAPTOP-59PVKG7G/Purescalp?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes&TrustServerCertificate=yes' # Database configuration

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable track modifications

db.init_app(app)  # Initialize the SQLAlchemy instance with the Flask app

# Register blueprints
from routes import main_bp, auth_bp, patient_bp, questionnaire_bp, recommendations_bp, doctor_bp

app.register_blueprint(main_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(patient_bp)
app.register_blueprint(questionnaire_bp)
app.register_blueprint(recommendations_bp)
app.register_blueprint(doctor_bp)

if __name__ == '__main__':
    app.run(debug=True)