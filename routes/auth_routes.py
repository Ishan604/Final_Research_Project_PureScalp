from flask import Blueprint, request, jsonify, render_template, redirect, url_for , session , flash
from models.doctor import Doctor
from extensions import db

# Create blueprint for authentication routes
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        specialization = request.form['specialization']

        new_doctor = Doctor(Name=name, Email=email, Password=password, Specialization=specialization)
        
        try:
            db.session.add(new_doctor)
            db.session.commit()
            return redirect(url_for('auth.login'))  # Redirect to login page after successful signup
        
        except Exception as e:
            db.session.rollback() # Rollback the session in case of error
            return jsonify({"error": str(e)}), 400

    return render_template('signup.html')

# Doctor Login Route
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Please enter both email and password', 'error')
            return render_template('login.html')
        
        # Find doctor by email
        doctor = db.session.query(Doctor).filter_by(Email=email).first()
        
        if doctor and doctor.Password == password:  # Note: In production, use hashed passwords
            # Store doctor info in session
            session['doctor_id'] = doctor.DoctorID
            session['doctor_name'] = doctor.Name
            session['doctor_email'] = doctor.Email
            session['user_type'] = 'doctor'
            
            flash(f'Welcome Dr. {doctor.Name}!', 'success')
            return redirect(url_for('doctor.doctor_dashboard'))
        else:
            flash('Invalid email or password', 'error')
            return render_template('login.html')
    
    return render_template('login.html')

# Doctor Logout Route
@auth_bp.route('/logout')
def logout():
    # Clear session data
    session.clear()
    flash('You have been successfully logged out', 'success')
    return redirect(url_for('auth.login'))
