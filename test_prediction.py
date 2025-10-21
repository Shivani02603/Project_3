#!/usr/bin/env python3
"""
test_prediction.py
Quick test to verify predictions are working correctly
"""

from backend_main import StudyCoachBackend
import json

print("="*60)
print("TESTING PREDICTION SYSTEM")
print("="*60)

# Initialize backend
backend = StudyCoachBackend()

# Test data - three different scenarios
test_cases = [
    {
        'name': 'Good Student',
        'data': {
            'study_hours_per_day': 6.0,
            'social_media_hours': 1.5,
            'netflix_hours': 1.0,
            'sleep_hours': 8.0,
            'mental_health_rating': 8,
            'attendance_percentage': 95.0,
            'diet_quality': 3,
            'exercise_frequency': 4,
            'part_time_job': 0,
            'age': 20
        }
    },
    {
        'name': 'Average Student',
        'data': {
            'study_hours_per_day': 4.0,
            'social_media_hours': 2.5,
            'netflix_hours': 1.5,
            'sleep_hours': 7.0,
            'mental_health_rating': 6,
            'attendance_percentage': 80.0,
            'diet_quality': 2,
            'exercise_frequency': 2,
            'part_time_job': 0,
            'age': 20
        }
    },
    {
        'name': 'Struggling Student',
        'data': {
            'study_hours_per_day': 2.0,
            'social_media_hours': 4.0,
            'netflix_hours': 2.5,
            'sleep_hours': 5.5,
            'mental_health_rating': 4,
            'attendance_percentage': 65.0,
            'diet_quality': 1,
            'exercise_frequency': 1,
            'part_time_job': 1,
            'age': 21
        }
    }
]

# Test each case
for i, test in enumerate(test_cases, 1):
    print(f"\n{'-'*60}")
    print(f"Test Case {i}: {test['name']}")
    print('-'*60)
    
    # Print input
    print(f"Study Hours: {test['data']['study_hours_per_day']:.1f}h")
    print(f"Social Media: {test['data']['social_media_hours']:.1f}h")
    print(f"Netflix: {test['data']['netflix_hours']:.1f}h")
    print(f"Sleep: {test['data']['sleep_hours']:.1f}h")
    print(f"Attendance: {test['data']['attendance_percentage']:.1f}%")
    
    # Make prediction
    prediction = backend.prediction_api.predict(test['data'])
    
    # Print prediction
    print(f"\n📊 PREDICTION RESULT:")
    print(f"Risk Level: {prediction.get('risk_level', 'ERROR')}")
    print(f"Confidence: {prediction.get('confidence', 0):.1f}%")
    
    if 'probabilities' in prediction:
        print(f"\nProbabilities:")
        for risk, prob in prediction['probabilities'].items():
            print(f"  {risk:7s}: {prob:5.1f}%")
    
    if 'note' in prediction:
        print(f"\nNote: {prediction['note']}")
    
    if 'error' in prediction:
        print(f"\n❌ ERROR: {prediction['error']}")

# Test the complete workflow
print(f"\n{'='*60}")
print("TESTING COMPLETE WORKFLOW")
print('='*60)

result = backend.process_study_session('TEST_STUDENT', test_cases[1]['data'])

print(f"\nStudent ID: {result['student_id']}")
print(f"Session Logged: {result['session_logged']}")
print(f"\nPrediction:")
pred = result['current_prediction']
print(f"  Risk: {pred.get('risk_level', 'UNKNOWN')}")
print(f"  Confidence: {pred.get('confidence', 0):.1f}%")

if 'weekly_stats' in result and 'error' not in result['weekly_stats']:
    stats = result['weekly_stats']
    print(f"\nWeekly Stats:")
    print(f"  Total Hours: {stats.get('total_study_hours', 0):.1f}")
    print(f"  Sessions: {stats.get('total_sessions', 0)}")
    print(f"  Avg Focus: {stats.get('average_focus_index', 0):.2f}")

print(f"\n{'='*60}")
print("TEST COMPLETED")
print('='*60)

# Show what's in the prediction object for debugging
print("\n[DEBUG] Full prediction object:")
print(json.dumps(pred, indent=2))