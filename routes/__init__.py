# Routes package initialization
from .main_routes import main_bp
from .auth_routes import auth_bp
from .patient_routes import patient_bp
from .questionnaire_routes import questionnaire_bp
from .recommendations_routes import recommendations_bp
from .doctor_dashboard import doctor_bp

__all__ = ['main_bp', 'auth_bp', 'patient_bp', 'questionnaire_bp', 'recommendations_bp', 'doctor_bp']
