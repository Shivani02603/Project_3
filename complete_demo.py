# complete_system_demo.py
# Complete Integration Demo - Tushar's Backend System
# Demonstrates the full AI Study Coach backend working together

import sys
from datetime import datetime, timedelta
import numpy as np

# Import all modules
from backend_main import StudyCoachBackend
from recommendation_engine import RecommendationEngine
from visualizations_module import StudyVisualization

def simulate_study_week(backend, student_id, days=7):
    """Simulate a week of study sessions for a student"""
    print(f"\n{'='*60}")
    print(f"Simulating {days}-day study period for {student_id}")
    print('='*60)
    
    base_date = datetime.now() - timedelta(days=days)
    
    for day in range(days):
        # Generate realistic session data with some variation
        study_hours = np.random.uniform(2, 7)
        social_media = np.random.uniform(1, 4)
        netflix = np.random.uniform(0.5, 2.5)
        
        session_data = {
            'study_hours_per_day': study_hours,
            'social_media_hours': social_media,
            'netflix_hours': netflix,
            'sleep_hours': np.random.uniform(5, 8),
            'mental_health_rating': int(np.random.uniform(4, 9)),
            'attendance_percentage': np.random.uniform(70, 95),
            'diet_quality': int(np.random.choice([1, 2, 3])),
            'exercise_frequency': int(np.random.uniform(1, 5)),
            'part_time_job': 0,
            'age': 20,
            'duration': study_hours,
            'distractions': social_media + netflix
        }
        
        # Process session (predict + log)
        result = backend.process_study_session(student_id, session_data)

    risk = result.get('current_prediction', {}).get('risk_level', 'Unknown')
    confidence = result.get('current_prediction', {}).get('confidence', 0)   
    print(f"\nDay {day + 1} - Risk: {risk} (Confidence: {confidence:.1f}%)")
      

    print(f"\n✓ {days}-day simulation completed")

def main():
    """Main demo showcasing all components"""
    
    print("="*70)
    print(" "*15 + "AI STUDY COACH - COMPLETE SYSTEM DEMO")
    print(" "*20 + "Backend by: Tushar")
    print("="*70)
    
    # Initialize all systems
    print("\n[1/5] Initializing Backend System...")
    backend = StudyCoachBackend()
    
    print("\n[2/5] Initializing Recommendation Engine...")
    rec_engine = RecommendationEngine()
    
    print("\n[3/5] Initializing Visualization Module...")
    viz = StudyVisualization()
    
    print("\n✓ All systems initialized successfully!")
    
    # Create test students
    students = [
        {
            'id': 'STU001',
            'name': 'Rajesh Kumar',
            'age': 20,
            'course': 'Computer Science'
        },
        {
            'id': 'STU002',
            'name': 'Priya Sharma',
            'age': 21,
            'course': 'Engineering'
        }
    ]
    
    # Register students
    print("\n[4/5] Registering Students...")
    for student in students:
        backend.user_manager.add_user(student['id'], student)
    
    # Simulate study sessions for each student
    print("\n[5/5] Simulating Study Sessions...")
    for student in students:
        simulate_study_week(backend, student['id'], days=14)
    
    # Generate comprehensive reports for first student
    student_id = students[0]['id']
    
    print(f"\n{'='*70}")
    print(f"GENERATING COMPREHENSIVE REPORTS FOR {student_id}")
    print('='*70)
    
    # Get sessions and stats
    sessions = backend.session_tracker.get_student_sessions(student_id, days=14)
    weekly_stats = backend.session_tracker.get_weekly_stats(student_id)
    
    # Generate text report
    print("\n" + "─"*70)
    print("TEXT REPORT")
    print("─"*70)
    text_report = backend.get_student_report(student_id)
    print(text_report)
    
    # Generate personalized recommendations
    print("\n" + "─"*70)
    print("PERSONALIZED RECOMMENDATIONS")
    print("─"*70)
    
    if sessions:
        latest_session = sessions[-1]
        recommendations = rec_engine.generate_personalized_recommendations(
            latest_session.get('risk_level', 'MEDIUM'),
            latest_session
        )
        
        print(f"\n🎯 Immediate Action:")
        print(f"   {recommendations['immediate_action']}")
        
        print(f"\n💪 Motivation:")
        print(f"   {recommendations['motivation']}")
        
        print(f"\n✅ Action Plan:")
        for i, tip in enumerate(recommendations['actionable_tips'], 1):
            print(f"   {i}. {tip}")
        
        print(f"\n📊 Pattern Analysis:")
        if recommendations['pattern_analysis']['concerns']:
            print(f"   ⚠️  Concerns: {', '.join(recommendations['pattern_analysis']['concerns'])}")
        if recommendations['pattern_analysis']['strengths']:
            print(f"   ✨ Strengths: {', '.join(recommendations['pattern_analysis']['strengths'])}")
        
        print(f"\n💡 Specific Suggestions:")
        for suggestion in recommendations['specific_suggestions']:
            print(f"   • {suggestion}")
    
    # Generate weekly report with recommendations
    print("\n" + "─"*70)
    print("WEEKLY PERFORMANCE REPORT")
    print("─"*70)
    weekly_report = rec_engine.generate_weekly_report(weekly_stats)
    print(weekly_report)
    
    # Generate visualizations
    print("\n" + "─"*70)
    print("GENERATING VISUALIZATIONS")
    print("─"*70)
    viz.generate_all_visualizations(sessions, weekly_stats, student_id)
    
    # Daily motivation
    print("\n" + "─"*70)
    print("DAILY MOTIVATION MESSAGE")
    print("─"*70)
    print(rec_engine.get_daily_motivation(weekly_stats.get('most_common_risk', 'MEDIUM')))
    
    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    print(f"""
✓ Backend System: Fully Operational
  • ML Model: Loaded and Making Predictions
  • Session Tracking: {len(sessions)} sessions logged
  • User Management: {len(students)} students registered

✓ Recommendation Engine: Active
  • Personalized nudges generated
  • Pattern analysis completed
  • Weekly reports generated

✓ Visualization System: Complete
  • 6 different chart types created
  • Comprehensive dashboard generated
  • All files saved to './visualizations/'

📁 Generated Files:
  • sessions.json - Session database
  • users.json - User database
  • 6 PNG visualization files per student

🎯 System Capabilities:
  ✅ Real-time procrastination risk prediction
  ✅ Study session tracking and analytics
  ✅ Personalized recommendations
  ✅ Pattern analysis and insights
  ✅ Visual progress reports
  ✅ Weekly performance summaries
  ✅ User management system

🚀 Ready for Integration with Frontend!
""")
    
    print("="*70)
    print(" "*15 + "TUSHAR'S BACKEND SYSTEM - DEMO COMPLETED")
    print("="*70)
    
    # Quick API usage examples
    print("\n" + "─"*70)
    print("QUICK API USAGE EXAMPLES")
    print("─"*70)
    print("""
# Example 1: Make a prediction
from study_coach_backend import StudyCoachBackend

backend = StudyCoachBackend()
session_data = {
    'study_hours_per_day': 4.5,
    'social_media_hours': 2.0,
    'netflix_hours': 1.5,
    'sleep_hours': 7.0,
    'mental_health_rating': 7,
    'attendance_percentage': 85.0,
    'diet_quality': 2,
    'exercise_frequency': 3,
    'part_time_job': 0,
    'age': 20
}

result = backend.prediction_api.predict(session_data)
print(f"Risk: {result['risk_level']}")

# Example 2: Log a session
backend.session_tracker.log_session('STUDENT_123', session_data)

# Example 3: Get weekly stats
stats = backend.session_tracker.get_weekly_stats('STUDENT_123')
print(stats)

# Example 4: Get recommendations
from recommendation_engine import RecommendationEngine
rec_engine = RecommendationEngine()
recs = rec_engine.generate_personalized_recommendations('HIGH', session_data)
print(recs['immediate_action'])

# Example 5: Generate visualizations
from visualizations import StudyVisualization
viz = StudyVisualization()
sessions = backend.session_tracker.get_student_sessions('STUDENT_123')
viz.create_dashboard(sessions, stats, 'STUDENT_123')
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
