from flask import Blueprint, render_template, redirect, url_for, session
from models.patient import Patient
from models.marks import Marks
from extensions import db
from datetime import datetime

# Create blueprint for recommendations routes
recommendations_bp = Blueprint('recommendations', __name__)

# Import the recommendation lookup function
try:
    from recommendations_lookup import get_recommendation_by_score
    LOOKUP_AVAILABLE = True
except ImportError:
    print("Warning: recommendations_lookup.py not found. Using fallback recommendations.")
    LOOKUP_AVAILABLE = False

@recommendations_bp.route('/recommendations')
def recommendations():
    """Display recommendations based on patient's score from CSV file"""
    patient_id = session.get('patient_id')
    if not patient_id:
        return redirect(url_for('patient.patient_registration'))
    
    try:
        # Get patient information
        patient = db.session.query(Patient).filter_by(PatientID=patient_id).first()
        if not patient:
            return render_template('recommendations.html', error="Patient information not found")
        
        # Get patient's marks from the Marks table
        marks_record = db.session.query(Marks).filter_by(patientId=patient_id).first()
        if not marks_record:
            return render_template('recommendations.html', 
                                 error="Assessment not completed. Please complete the questionnaire first.",
                                 patient_name=patient.Name,
                                 patient_id=patient_id)
        
        score = marks_record.marks
        
        # Get recommendation from CSV file
        if LOOKUP_AVAILABLE:
            try:
                recommendation_result = get_recommendation_by_score(score)
                recommendation_text = recommendation_result.get('recommendation', 'No recommendation available')
                match_type = recommendation_result.get('match_type', 'unknown')
                source = recommendation_result.get('source', 'CSV lookup')
                image_filename = recommendation_result.get('image')
                image_url = None
                if image_filename:
                    # Build url for image in static/img/hair_recommendation_images
                    image_url = url_for('static', filename=f"img/hair_recommendation_images/{image_filename}")
                
                # Additional info for display
                additional_info = ""
                if 'matched_score' in recommendation_result:
                    additional_info = f"(Based on closest match: Score {recommendation_result['matched_score']})"
                
            except Exception as e:
                print(f"Error using CSV lookup: {e}")
                recommendation_text = get_simple_fallback_recommendation(score)
                match_type = "fallback"
                source = "Built-in logic"
                additional_info = ""
        else:
            recommendation_text = get_simple_fallback_recommendation(score)
            match_type = "fallback"
            source = "Built-in logic"
            additional_info = ""
        
        # Prepare template data
        template_data = {
            'patient_name': patient.Name,
            'patient_id': patient_id,
            'score': score,
            'assessment_date': datetime.now().strftime('%B %d, %Y'),
            'recommendation': recommendation_text,
            'match_type': match_type,
            'source': source,
            'additional_info': additional_info,
            'confidence': 1.0 if match_type == 'exact' else 0.8 if match_type == 'closest' else 0.6,
            'image_url': image_url if 'image_url' in locals() else None
        }
        
        return render_template('recommendations.html', **template_data)
        
    except Exception as e:
        print(f"Error in recommendations route: {e}")
        return render_template('recommendations.html', 
                             error=f"Unable to generate recommendations: {str(e)}",
                             patient_name=patient.Name if 'patient' in locals() else "Unknown",
                             patient_id=patient_id)

def get_simple_fallback_recommendation(score):
    """Simple fallback when CSV lookup is not available"""
    if score >= 50:
        return "Intensive hair treatment with keratin-based products and professional consultation recommended"
    elif score >= 25:
        return "Moderate hair care routine with moisturizing treatments and regular scalp massage"
    else:
        return "Basic hair maintenance with gentle products and regular care routine"
