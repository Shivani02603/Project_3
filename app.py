#!/usr/bin/env python3
"""
app.py
Flask API Backend for AI Study Coach Frontend
Connects the beautiful UI to Tushar's ML backend
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from backend_main import StudyCoachBackend
from recommendation_engine import RecommendationEngine
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Initialize backend systems
backend = StudyCoachBackend()
rec_engine = RecommendationEngine()

print("="*60)
print("AI STUDY COACH - FLASK API SERVER")
print("="*60)
print("✓ Backend initialized")
print("✓ ML Model loaded")
print("✓ API ready to serve")
print("="*60)

# Serve the frontend
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for making predictions
    
    Request body should contain:
    {
        "study_hours_per_day": 4.5,
        "social_media_hours": 2.0,
        "netflix_hours": 1.5,
        "sleep_hours": 7.0,
        "mental_health_rating": 7,
        "attendance_percentage": 85.0,
        "diet_quality": 2,
        "exercise_frequency": 3,
        "part_time_job": 0,
        "age": 20
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'study_hours_per_day', 'social_media_hours', 'netflix_hours',
            'sleep_hours', 'mental_health_rating', 'attendance_percentage',
            'diet_quality', 'exercise_frequency', 'part_time_job', 'age'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Make prediction using backend
        prediction = backend.prediction_api.predict(data)
        
        # Get recommendations
        recommendations = rec_engine.generate_personalized_recommendations(
            prediction.get('risk_level', 'MEDIUM'),
            data
        )
        
        # Combine prediction and recommendations
        response = {
            'prediction': prediction,
            'recommendations': {
                'immediate_action': recommendations['immediate_action'],
                'motivation': recommendations['motivation'],
                'actionable_tips': recommendations['actionable_tips'][:3],
                'pattern_analysis': recommendations['pattern_analysis']
            },
            'calculated_features': {
                'total_distractions': data['social_media_hours'] + data['netflix_hours'],
                'focus_index': data['study_hours_per_day'] / (data['social_media_hours'] + data['netflix_hours'] + 0.000001)
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 500

@app.route('/api/log_session', methods=['POST'])
def log_session():
    """
    API endpoint for logging a study session
    
    Request body:
    {
        "student_id": "STU001",
        "session_data": { ... }
    }
    """
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        session_data = data.get('session_data')
        
        if not student_id or not session_data:
            return jsonify({
                'error': 'Missing student_id or session_data'
            }), 400
        
        # Log session
        result = backend.process_study_session(student_id, session_data)
        
        return jsonify({
            'success': True,
            'message': 'Session logged successfully',
            'result': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to log session'
        }), 500

@app.route('/api/weekly_stats/<student_id>', methods=['GET'])
def get_weekly_stats(student_id):
    """Get weekly statistics for a student"""
    try:
        stats = backend.session_tracker.get_weekly_stats(student_id)
        
        if 'error' in stats:
            return jsonify(stats), 404
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to get statistics'
        }), 500

@app.route('/api/user/<student_id>', methods=['GET', 'POST', 'DELETE'])
def manage_user(student_id):
    """Manage user data (GET, POST, DELETE)"""
    try:
        if request.method == 'GET':
            # Get user data
            user = backend.user_manager.get_user(student_id)
            if not user:
                return jsonify({'error': 'User not found'}), 404
            return jsonify(user), 200
            
        elif request.method == 'POST':
            # Add or update user
            user_data = request.get_json()
            backend.user_manager.add_user(student_id, user_data)
            return jsonify({
                'success': True,
                'message': 'User added/updated successfully'
            }), 200
            
        elif request.method == 'DELETE':
            # Delete user
            backend.user_manager.remove_user(student_id)
            return jsonify({
                'success': True,
                'message': 'User deleted successfully'
            }), 200
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'User operation failed'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if backend.model_loader.is_loaded() else "not loaded"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'backend': 'operational',
        'version': '1.0.0'
    }), 200

@app.route('/api/recommendations/<risk_level>', methods=['GET'])
def get_recommendations(risk_level):
    """Get recommendations for a specific risk level"""
    try:
        if risk_level not in ['LOW', 'MEDIUM', 'HIGH']:
            return jsonify({'error': 'Invalid risk level'}), 400
        
        immediate = rec_engine.get_nudge(risk_level, 'immediate')
        motivation = rec_engine.get_nudge(risk_level, 'motivation')
        actionable = [rec_engine.get_nudge(risk_level, 'actionable') for _ in range(3)]
        
        return jsonify({
            'immediate_action': immediate,
            'motivation': motivation,
            'actionable_tips': actionable
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to get recommendations'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
    print("\n🚀 Starting Flask server...")
    print("📱 Frontend: http://localhost:5000")
    print("🔌 API Base: http://localhost:5000/api")
    print("\n📚 Available Endpoints:")
    print("  POST /api/predict - Make a prediction")
    print("  POST /api/log_session - Log a study session")
    print("  GET  /api/weekly_stats/<id> - Get statistics")
    print("  GET  /api/user/<id> - Get user data")
    print("  GET  /api/health - Health check")