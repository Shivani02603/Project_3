# study_coach_backend.py
# Tushar's Backend System for AI Study Coach
# This module handles ML model integration, session tracking, and user management

import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple

# ============================================
# MODEL LOADER
# ============================================

class ModelLoader:
    """Handles loading and managing the trained ML model"""
    
    def __init__(self, model_path='trained_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained Random Forest model"""
        try:
            import joblib
            # Use joblib (same as how Shivani saves it)
            self.model = joblib.load(self.model_path)
            
            # Verify it's actually a model with predict method
            if not hasattr(self.model, 'predict'):
                print(f"⚠️  Loaded object is not a valid model (type: {type(self.model)})")
                self.model = None
                return
            
            print(f"✓ Model loaded successfully from {self.model_path}")
            
        except FileNotFoundError:
            print(f"✗ Model file not found at {self.model_path}")
            print("Please ensure Shivani's trained_model.pkl is in the same directory")
            self.model = None
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
    
    def is_loaded(self):
        """Check if model is loaded and valid"""
        return self.model is not None and hasattr(self.model, 'predict')


# ============================================
# PREDICTION API
# ============================================

class PredictionAPI:
    """API for making procrastination risk predictions"""
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.feature_names = [
            'study_hours_per_day',
            'social_media_hours',
            'netflix_hours',
            'total_distractions',
            'focus_index',
            'sleep_hours',
            'mental_health_rating',
            'attendance_percentage',
            'diet_quality',
            'exercise_frequency',
            'part_time_job',
            'age'
        ]
        self.risk_labels = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
    
    def calculate_features(self, data: Dict) -> Dict:
        """Calculate derived features from input data"""
        # Calculate total distractions
        total_distractions = data.get('social_media_hours', 0) + data.get('netflix_hours', 0)
        
        # Calculate focus index
        study_hours = data.get('study_hours_per_day', 0)
        focus_index = study_hours / (total_distractions + 1e-6)
        
        # Add calculated features
        data['total_distractions'] = total_distractions
        data['focus_index'] = focus_index
        
        return data
    
    def predict(self, student_data: Dict) -> Dict:
        """
        Make procrastination risk prediction for a student
        
        Args:
            student_data: Dictionary with student's study habits
            
        Returns:
            Dictionary with prediction results
        """
        # Calculate derived features
        student_data = self.calculate_features(student_data)
        
        # If model is loaded, use it
        if self.model_loader.is_loaded():
            try:
                # Prepare input dataframe
                input_df = pd.DataFrame([{
                    feature: student_data.get(feature, 0) 
                    for feature in self.feature_names
                }])
                
                # Make prediction
                prediction = self.model_loader.model.predict(input_df)[0]
                probabilities = self.model_loader.model.predict_proba(input_df)[0]
                
                # Get risk level
                risk_level = self.risk_labels[prediction]
                confidence = probabilities[prediction] * 100
                
                return {
                    'risk_level': risk_level,
                    'confidence': round(confidence, 2),
                    'probabilities': {
                        'LOW': round(probabilities[0] * 100, 2),
                        'MEDIUM': round(probabilities[1] * 100, 2),
                        'HIGH': round(probabilities[2] * 100, 2)
                    },
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                print(f"⚠️  Model prediction failed: {e}")
                return self._fallback_prediction(student_data)
        else:
            # Use rule-based fallback system
            print("⚠️  Using fallback prediction (model not loaded)")
            return self._fallback_prediction(student_data)
    
    def _fallback_prediction(self, student_data: Dict) -> Dict:
        """
        Fallback rule-based prediction when ML model is not available
        Uses heuristics based on study habits
        """
        focus_index = student_data.get('focus_index', 0)
        study_hours = student_data.get('study_hours_per_day', 0)
        attendance = student_data.get('attendance_percentage', 0)
        total_distractions = student_data.get('total_distractions', 0)
        
        # Rule-based risk calculation
        score = 0
        
        # Positive factors
        if focus_index >= 1.0:
            score += 3
        elif focus_index >= 0.5:
            score += 1
        else:
            score -= 2
            
        if study_hours >= 5:
            score += 3
        elif study_hours >= 3:
            score += 1
        else:
            score -= 2
            
        if attendance >= 85:
            score += 2
        elif attendance >= 70:
            score += 1
        else:
            score -= 1
        
        # Negative factors
        if total_distractions > 5:
            score -= 2
        elif total_distractions > 3:
            score -= 1
        
        # Determine risk level
        if score >= 4:
            risk_level = 'LOW'
            probs = [65.0, 25.0, 10.0]
        elif score >= 0:
            risk_level = 'MEDIUM'
            probs = [25.0, 55.0, 20.0]
        else:
            risk_level = 'HIGH'
            probs = [10.0, 25.0, 65.0]
        
        confidence = max(probs)
        
        return {
            'risk_level': risk_level,
            'confidence': round(confidence, 2),
            'probabilities': {
                'LOW': round(probs[0], 2),
                'MEDIUM': round(probs[1], 2),
                'HIGH': round(probs[2], 2)
            },
            'timestamp': datetime.now().isoformat(),
            'note': 'Prediction made using rule-based fallback (ML model not loaded)'
        }


# ============================================
# SESSION TRACKER
# ============================================

class SessionTracker:
    """Track and manage student study sessions"""
    
    def __init__(self, storage_file='sessions.json'):
        self.storage_file = storage_file
        self.sessions = self.load_sessions()
    
    def load_sessions(self) -> Dict:
        """Load existing sessions from file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_sessions(self):
        """Save sessions to file"""
        with open(self.storage_file, 'w') as f:
            json.dump(self.sessions, f, indent=2)
    
    def log_session(self, student_id: str, session_data: Dict):
        """
        Log a study session for a student
        
        Args:
            student_id: Unique identifier for student
            session_data: Dictionary with session details
        """
        if student_id not in self.sessions:
            self.sessions[student_id] = []
        
        # Add timestamp if not present
        if 'timestamp' not in session_data:
            session_data['timestamp'] = datetime.now().isoformat()
        
        self.sessions[student_id].append(session_data)
        self.save_sessions()
        
        print(f"✓ Session logged for student {student_id}")
    
    def get_student_sessions(self, student_id: str, days: int = 30) -> List[Dict]:
        """Get recent sessions for a student"""
        if student_id not in self.sessions:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_sessions = []
        
        for session in self.sessions[student_id]:
            session_date = datetime.fromisoformat(session['timestamp'])
            if session_date >= cutoff_date:
                recent_sessions.append(session)
        
        return recent_sessions
    
    def get_weekly_stats(self, student_id: str) -> Dict:
        """Calculate weekly statistics for a student"""
        sessions = self.get_student_sessions(student_id, days=7)
        
        if not sessions:
            return {'error': 'No sessions found in last 7 days'}
        
        # Calculate statistics
        total_study_hours = sum(s.get('duration', 0) for s in sessions)
        avg_focus = np.mean([s.get('focus_index', 0) for s in sessions])
        total_sessions = len(sessions)
        avg_distractions = np.mean([s.get('distractions', 0) for s in sessions])
        
        # Count risk levels
        risk_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        for session in sessions:
            risk = session.get('risk_level', 'MEDIUM')
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        return {
            'period': 'Last 7 days',
            'total_study_hours': round(total_study_hours, 2),
            'total_sessions': total_sessions,
            'average_focus_index': round(avg_focus, 2),
            'average_distractions': round(avg_distractions, 2),
            'risk_distribution': risk_counts,
            'most_common_risk': max(risk_counts, key=risk_counts.get)
        }


# ============================================
# USER MANAGER
# ============================================

class UserManager:
    """Manage student user data"""
    
    def __init__(self, storage_file='users.json'):
        self.storage_file = storage_file
        self.users = self.load_users()
    
    def load_users(self) -> Dict:
        """Load existing users from file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_users(self):
        """Save users to file"""
        with open(self.storage_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def add_user(self, student_id: str, user_data: Dict):
        """Add or update a student user"""
        user_data['created_at'] = datetime.now().isoformat()
        user_data['last_updated'] = datetime.now().isoformat()
        self.users[student_id] = user_data
        self.save_users()
        print(f"✓ User {student_id} added/updated")
    
    def get_user(self, student_id: str) -> Dict:
        """Get user data"""
        return self.users.get(student_id, {})
    
    def update_user(self, student_id: str, updates: Dict):
        """Update user data"""
        if student_id in self.users:
            self.users[student_id].update(updates)
            self.users[student_id]['last_updated'] = datetime.now().isoformat()
            self.save_users()
            print(f"✓ User {student_id} updated")
        else:
            print(f"✗ User {student_id} not found")
    
    def remove_user(self, student_id: str):
        """Remove a user"""
        if student_id in self.users:
            del self.users[student_id]
            self.save_users()
            print(f"✓ User {student_id} removed")


# ============================================
# MAIN BACKEND SYSTEM
# ============================================

class StudyCoachBackend:
    """Main backend system integrating all components"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.prediction_api = PredictionAPI(self.model_loader)
        self.session_tracker = SessionTracker()
        self.user_manager = UserManager()
        
        print("=" * 60)
        print("AI STUDY COACH - BACKEND SYSTEM INITIALIZED")
        print("=" * 60)
    
    def process_study_session(self, student_id: str, session_data: Dict) -> Dict:
        """
        Complete workflow: log session and get prediction
        
        Args:
            student_id: Student identifier
            session_data: Session details with study habits
            
        Returns:
            Complete analysis with prediction and recommendations
        """
        # Make prediction
        prediction = self.prediction_api.predict(session_data)
        
        # Check if prediction was successful
        if 'error' in prediction:
            print(f"⚠️  Prediction error: {prediction['error']}")
            # Use fallback
            prediction = self.prediction_api._fallback_prediction(session_data)
        
        # Add prediction to session data
        session_data['risk_level'] = prediction.get('risk_level', 'UNKNOWN')
        session_data['confidence'] = prediction.get('confidence', 0)
        
        # Log session
        self.session_tracker.log_session(student_id, session_data)
        
        # Get weekly stats
        weekly_stats = self.session_tracker.get_weekly_stats(student_id)
        
        return {
            'student_id': student_id,
            'current_prediction': prediction,
            'weekly_stats': weekly_stats,
            'session_logged': True
        }
    
    def get_student_report(self, student_id: str) -> str:
        """Generate text report for a student"""
        user = self.user_manager.get_user(student_id)
        stats = self.session_tracker.get_weekly_stats(student_id)
        
        report = f"""
{'='*60}
WEEKLY STUDY REPORT - {student_id}
{'='*60}

Student Info:
- Name: {user.get('name', 'Unknown')}
- Age: {user.get('age', 'N/A')}

Weekly Statistics:
- Total Study Hours: {stats.get('total_study_hours', 0)} hours
- Total Sessions: {stats.get('total_sessions', 0)}
- Average Focus Index: {stats.get('average_focus_index', 0)}
- Average Distractions: {stats.get('average_distractions', 0)} hours

Risk Analysis:
- Most Common Risk Level: {stats.get('most_common_risk', 'N/A')}
- Risk Distribution:
  * LOW: {stats.get('risk_distribution', {}).get('LOW', 0)} sessions
  * MEDIUM: {stats.get('risk_distribution', {}).get('MEDIUM', 0)} sessions
  * HIGH: {stats.get('risk_distribution', {}).get('HIGH', 0)} sessions

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
"""
        return report


# ============================================
# DEMO USAGE
# ============================================

if __name__ == "__main__":
    # Initialize backend system
    backend = StudyCoachBackend()
    
    print("\n" + "="*60)
    print("DEMO: Processing Sample Study Session")
    print("="*60)
    
    # Sample student data
    sample_session = {
        'study_hours_per_day': 4.5,
        'social_media_hours': 2.0,
        'netflix_hours': 1.5,
        'sleep_hours': 7.0,
        'mental_health_rating': 7,
        'attendance_percentage': 85.0,
        'diet_quality': 2,  # Fair
        'exercise_frequency': 3,
        'part_time_job': 0,
        'age': 20,
        'duration': 4.5  # hours studied in this session
    }
    
    # Process session
    result = backend.process_study_session('STUDENT_001', sample_session)
    
    print(f"\nPrediction Results:")
    print(f"Risk Level: {result['current_prediction']['risk_level']}")
    print(f"Confidence: {result['current_prediction']['confidence']}%")
    print(f"\nProbabilities:")
    for risk, prob in result['current_prediction']['probabilities'].items():
        print(f"  {risk}: {prob}%")
    
    print(f"\nWeekly Statistics:")
    stats = result['weekly_stats']
    print(f"Total Study Hours: {stats.get('total_study_hours', 0)}")
    print(f"Total Sessions: {stats.get('total_sessions', 0)}")
    print(f"Average Focus: {stats.get('average_focus_index', 0)}")
    
    # Generate report
    backend.user_manager.add_user('STUDENT_001', {
        'name': 'Tushar',
        'age': 20,
        'course': 'Engineering'
    })
    
    print("\n" + backend.get_student_report('STUDENT_001'))
    
    print("\n✓ Backend system demo completed!")
    print("Files created: sessions.json, users.json")