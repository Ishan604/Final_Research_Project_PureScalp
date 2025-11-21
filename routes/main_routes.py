from flask import Blueprint, render_template, request, jsonify, current_app
import os
import torch
from config import Config
from utils import preprocess_single_image, load_class_names, create_upload_folder, is_scalp_image
from model import ScalpClassifier, ScalpValidator, load_model

# Create blueprint for main routes
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/upload')
def upload_image():
    return render_template('uploadimage.html')

@main_bp.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@main_bp.route('/quiz')
def quiz():
    return render_template('quiz.html')

@main_bp.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

@main_bp.route('/haircare')
def haircare():
    return render_template('haircare.html')

# -------- ML Inference (Analyze) --------

_cached_model = None
_cached_class_names = None
_cached_validator = None


def _load_cached_classifier():
    global _cached_model, _cached_class_names
    if _cached_model is not None and _cached_class_names is not None:
        return _cached_model, _cached_class_names

    class_names = load_class_names()
    num_classes = len(class_names) if class_names else Config.NUM_CLASSES

    model = load_model(Config.SCALP_CLASSIFIER_PATH, ScalpClassifier, num_classes=num_classes)
    _cached_model = model
    _cached_class_names = class_names if class_names else ['dandruff', 'dandruff_sensitive', 'oily', 'sensitive']
    return _cached_model, _cached_class_names


def _load_cached_validator():
    global _cached_validator
    if _cached_validator is not None:
        return _cached_validator
    validator = load_model(Config.SCALP_VALIDATOR_PATH, ScalpValidator)
    _cached_validator = validator
    return _cached_validator


@main_bp.route('/analyze', methods=['POST'])
def analyze_images():
    # Ensure upload folder exists
    create_upload_folder()

    if 'files' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No files part in the request.'
        }), 400

    files = request.files.getlist('files')
    # Enforce single file
    if len(files) > 1:
        files = files[:1]
    if not files:
        return jsonify({
            'success': False,
            'error': 'No files uploaded.'
        }), 400

    model, class_names = _load_cached_classifier()
    validator = _load_cached_validator()
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Ensure ml_models/scalp_classifier.pth exists.'
        }), 500
    # If validation is required but validator missing, block classification
    # If validation is required but validator missing, we'll fallback to classifier confidence instead of 500

    results = []
    for file in files:
        try:
            # Save file to upload folder
            filename = file.filename
            save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(save_path)

            # Validate scalp image (if validator available)
            if validator is not None:
                is_scalp, scalp_conf = is_scalp_image(save_path, validator)
                if not is_scalp:
                    results.append({
                        'filename': filename,
                        'error': 'Not a scalp image',
                        'scalp_confidence': float(scalp_conf)
                    })
                    continue
            # else: fallback will be handled after classification by confidence threshold

            # Preprocess
            image_tensor = preprocess_single_image(save_path)
            image_tensor = image_tensor.to(Config.DEVICE)

            # Predict
            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_idx = int(probs.argmax())
                pred_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                confidence = float(probs[pred_idx])

            # Low-confidence rejection when validator missing or disabled OR when validation is required but missing
            if (validator is None) and confidence < Config.CLASSIFICATION_CONFIDENCE_THRESHOLD:
                results.append({
                    'filename': filename,
                    'error': 'Low confidence prediction, likely non-scalp image',
                    'confidence': confidence
                })
            else:
                results.append({
                    'filename': filename,
                    'predicted_class': pred_class,
                    'confidence': confidence
                })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

    return jsonify({
        'success': True,
        'results': results
    })
