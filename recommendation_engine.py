# recommendation_engine.py
# Smart Nudge Generator for AI Study Coach
# Generates personalized recommendations based on risk level and patterns

import random
from typing import List, Dict
from datetime import datetime

class RecommendationEngine:
    """Generates personalized study recommendations and nudges"""
    
    def __init__(self):
        # Define nudges for each risk level
        self.nudges = {
            'HIGH': {
                'immediate': [
                    "🚨 High procrastination risk detected! Start with just 10 minutes of study.",
                    "⚠️ Take a 5-minute walk before starting - it helps focus!",
                    "📱 Put your phone in another room before studying.",
                    "⏰ Set a timer for 15 minutes and commit to focused work.",
                    "🎯 Break your task into tiny 5-minute chunks.",
                    "💪 You've got this! Just open your books and start."
                ],
                'motivation': [
                    "Remember: Starting is the hardest part. You can do this!",
                    "Every expert was once a beginner. Take the first step!",
                    "Your future self will thank you for starting now.",
                    "Small progress is still progress. Begin!"
                ],
                'actionable': [
                    "Try the Pomodoro Technique: 25 min study + 5 min break",
                    "Study in a library or quiet space away from distractions",
                    "Use website blockers to limit social media during study time",
                    "Find a study buddy for accountability",
                    "Reward yourself after completing 30 minutes of focused work"
                ]
            },
            'MEDIUM': {
                'immediate': [
                    "📊 You're doing okay, but can improve! Try 30 focused minutes.",
                    "⭐ Good start! Maintain consistency for better results.",
                    "🔔 Take a short break after 25 minutes to stay fresh.",
                    "📝 Review your notes before starting new material.",
                    "🎵 Try studying with instrumental music if it helps."
                ],
                'motivation': [
                    "You're on the right track! Keep pushing forward.",
                    "Consistency beats intensity. Keep your routine!",
                    "You're improving! Don't stop now.",
                    "Great effort! Your hard work is showing results."
                ],
                'actionable': [
                    "Schedule specific study times in your calendar",
                    "Use active recall techniques instead of passive reading",
                    "Create a dedicated study space at home",
                    "Track your study hours to maintain accountability",
                    "Join online study groups for your subjects"
                ]
            },
            'LOW': {
                'immediate': [
                    "🎉 Excellent focus! You're on fire! Keep going!",
                    "💯 Outstanding work! Your discipline is paying off!",
                    "🌟 Keep up this amazing momentum!",
                    "🏆 You're crushing it! Maintain this consistency.",
                    "✨ Great job! You're setting a strong example!"
                ],
                'motivation': [
                    "Your dedication is inspiring! Keep it up!",
                    "You're building strong study habits. Well done!",
                    "This is what success looks like. Excellent!",
                    "You're proof that consistency works. Amazing!"
                ],
                'actionable': [
                    "Consider teaching others - it reinforces your learning",
                    "Challenge yourself with harder problems",
                    "Set new goals to push your boundaries",
                    "Help peers who might be struggling",
                    "Document your study methods to help others"
                ]
            }
        }
        
        # Study tips by category
        self.tips = {
            'focus': [
                "Use the 'Do Not Disturb' mode on all devices",
                "Study in 90-minute blocks with 15-minute breaks",
                "Keep water and healthy snacks nearby to avoid interruptions",
                "Use noise-canceling headphones or white noise",
                "Clear your desk of non-study items"
            ],
            'health': [
                "Get 7-8 hours of sleep for better memory retention",
                "Exercise for 20 minutes daily to boost brain function",
                "Eat protein-rich foods for sustained energy",
                "Stay hydrated - drink water regularly",
                "Take short walks between study sessions"
            ],
            'technique': [
                "Use the Feynman Technique: Teach concepts in simple terms",
                "Practice spaced repetition for long-term retention",
                "Create mind maps for complex topics",
                "Do practice problems immediately after learning",
                "Review material within 24 hours of learning"
            ],
            'time_management': [
                "Use time-blocking to schedule study sessions",
                "Prioritize difficult subjects when you're most alert",
                "Set daily, weekly, and monthly study goals",
                "Use apps like Forest or Focus@Will for time tracking",
                "Plan your week every Sunday evening"
            ]
        }
    
    def get_nudge(self, risk_level: str, category: str = 'immediate') -> str:
        """
        Get a random nudge for the given risk level
        
        Args:
            risk_level: 'LOW', 'MEDIUM', or 'HIGH'
            category: Type of nudge ('immediate', 'motivation', 'actionable')
        """
        if risk_level not in self.nudges:
            risk_level = 'MEDIUM'
        
        if category not in self.nudges[risk_level]:
            category = 'immediate'
        
        return random.choice(self.nudges[risk_level][category])
    
    def get_study_tip(self, category: str = None) -> str:
        """Get a random study tip from specified category"""
        if category and category in self.tips:
            return random.choice(self.tips[category])
        
        # Return random tip from any category
        all_tips = []
        for tips_list in self.tips.values():
            all_tips.extend(tips_list)
        return random.choice(all_tips)
    
    def generate_personalized_recommendations(self, 
                                              risk_level: str, 
                                              session_data: Dict) -> Dict:
        """
        Generate comprehensive personalized recommendations
        
        Args:
            risk_level: Current procrastination risk level
            session_data: Student's session data with study habits
        """
        recommendations = {
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'immediate_action': self.get_nudge(risk_level, 'immediate'),
            'motivation': self.get_nudge(risk_level, 'motivation'),
            'actionable_tips': []
        }
        
        # Add 3 actionable tips based on risk level
        for _ in range(3):
            tip = self.get_nudge(risk_level, 'actionable')
            if tip not in recommendations['actionable_tips']:
                recommendations['actionable_tips'].append(tip)
        
        # Analyze patterns and add specific recommendations
        analysis = self._analyze_patterns(session_data)
        recommendations['pattern_analysis'] = analysis
        recommendations['specific_suggestions'] = self._get_specific_suggestions(analysis)
        
        return recommendations
    
    def _analyze_patterns(self, session_data: Dict) -> Dict:
        """Analyze study patterns from session data"""
        analysis = {
            'concerns': [],
            'strengths': []
        }
        
        # Check various metrics
        study_hours = session_data.get('study_hours_per_day', 0)
        social_media = session_data.get('social_media_hours', 0)
        sleep_hours = session_data.get('sleep_hours', 0)
        focus_index = session_data.get('focus_index', 0)
        attendance = session_data.get('attendance_percentage', 0)
        
        # Identify concerns
        if study_hours < 3:
            analysis['concerns'].append('Low daily study hours')
        if social_media > 3:
            analysis['concerns'].append('High social media usage')
        if sleep_hours < 6:
            analysis['concerns'].append('Insufficient sleep')
        if focus_index < 0.5:
            analysis['concerns'].append('Low focus index')
        if attendance < 75:
            analysis['concerns'].append('Poor attendance')
        
        # Identify strengths
        if study_hours >= 5:
            analysis['strengths'].append('Excellent study hours')
        if social_media <= 2:
            analysis['strengths'].append('Good control over distractions')
        if sleep_hours >= 7:
            analysis['strengths'].append('Healthy sleep pattern')
        if focus_index > 1.0:
            analysis['strengths'].append('Strong focus')
        if attendance >= 90:
            analysis['strengths'].append('Excellent attendance')
        
        return analysis
    
    def _get_specific_suggestions(self, analysis: Dict) -> List[str]:
        """Generate specific suggestions based on pattern analysis"""
        suggestions = []
        
        concern_solutions = {
            'Low daily study hours': "Gradually increase study time by 30 minutes each week",
            'High social media usage': "Use app blockers during study hours (StayFocusd, Freedom)",
            'Insufficient sleep': "Set a consistent bedtime routine and avoid screens 1 hour before bed",
            'Low focus index': "Try the Pomodoro Technique with 25-minute focused sessions",
            'Poor attendance': "Set multiple alarms and prepare materials the night before"
        }
        
        for concern in analysis['concerns']:
            if concern in concern_solutions:
                suggestions.append(concern_solutions[concern])
        
        # Add positive reinforcement for strengths
        if analysis['strengths']:
            suggestions.append(f"✨ Keep maintaining: {', '.join(analysis['strengths'][:2])}")
        
        return suggestions
    
    def generate_weekly_report(self, weekly_stats: Dict) -> str:
        """Generate a comprehensive weekly report with recommendations"""
        risk = weekly_stats.get('most_common_risk', 'MEDIUM')
        
        report = f"""
{'='*60}
📊 WEEKLY PERFORMANCE REPORT
{'='*60}

📈 Your Statistics:
• Total Study Hours: {weekly_stats.get('total_study_hours', 0)} hours
• Study Sessions: {weekly_stats.get('total_sessions', 0)}
• Average Focus Index: {weekly_stats.get('average_focus_index', 0):.2f}
• Most Common Risk: {risk}

🎯 This Week's Focus:
{self.get_nudge(risk, 'motivation')}

💡 Immediate Action:
{self.get_nudge(risk, 'immediate')}

✅ Action Plan:
"""
        
        # Add 3 actionable tips
        for i in range(3):
            tip = self.get_nudge(risk, 'actionable')
            report += f"\n{i+1}. {tip}"
        
        report += f"\n\n🌟 Study Tips for Next Week:\n"
        for category in ['focus', 'health', 'technique']:
            tip = self.get_study_tip(category)
            report += f"\n• {tip}"
        
        report += f"\n\n{'='*60}\n"
        
        return report
    
    def get_daily_motivation(self, risk_level: str = 'MEDIUM') -> str:
        """Get daily motivational message"""
        quotes = [
            "Success is the sum of small efforts repeated day in and day out. - Robert Collier",
            "The expert in anything was once a beginner. - Helen Hayes",
            "Don't watch the clock; do what it does. Keep going. - Sam Levenson",
            "The secret of getting ahead is getting started. - Mark Twain",
            "You don't have to be great to start, but you have to start to be great. - Zig Ziglar"
        ]
        
        return f"""
{'='*60}
💪 DAILY MOTIVATION
{'='*60}

{random.choice(quotes)}

{self.get_nudge(risk_level, 'motivation')}

Today's Action: {self.get_nudge(risk_level, 'immediate')}

{'='*60}
"""


# ============================================
# DEMO USAGE
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("AI STUDY COACH - RECOMMENDATION ENGINE DEMO")
    print("="*60)
    
    engine = RecommendationEngine()
    
    # Test recommendations for different risk levels
    for risk in ['HIGH', 'MEDIUM', 'LOW']:
        print(f"\n{'='*60}")
        print(f"RECOMMENDATIONS FOR {risk} RISK LEVEL")
        print('='*60)
        print(f"\nImmediate: {engine.get_nudge(risk, 'immediate')}")
        print(f"Motivation: {engine.get_nudge(risk, 'motivation')}")
        print(f"Action: {engine.get_nudge(risk, 'actionable')}")
    
    # Generate personalized recommendations
    print("\n" + "="*60)
    print("PERSONALIZED RECOMMENDATION EXAMPLE")
    print("="*60)
    
    sample_data = {
        'study_hours_per_day': 2.5,
        'social_media_hours': 4.0,
        'sleep_hours': 5.5,
        'focus_index': 0.3,
        'attendance_percentage': 70
    }
    
    recommendations = engine.generate_personalized_recommendations('HIGH', sample_data)
    
    print(f"\nImmediate Action: {recommendations['immediate_action']}")
    print(f"\nMotivation: {recommendations['motivation']}")
    print(f"\nActionable Tips:")
    for i, tip in enumerate(recommendations['actionable_tips'], 1):
        print(f"{i}. {tip}")
    
    print(f"\nPattern Analysis:")
    print(f"Concerns: {', '.join(recommendations['pattern_analysis']['concerns'])}")
    print(f"Strengths: {', '.join(recommendations['pattern_analysis']['strengths'])}")
    
    print(f"\nSpecific Suggestions:")
    for suggestion in recommendations['specific_suggestions']:
        print(f"• {suggestion}")
    
    # Daily motivation
    print("\n" + engine.get_daily_motivation('MEDIUM'))
    
    print("✓ Recommendation Engine demo completed!")
