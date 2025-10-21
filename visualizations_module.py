# visualizations.py
# Chart Generation for AI Study Coach
# Creates visual reports and dashboards

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# Set style for professional-looking charts
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

class StudyVisualization:
    """Generate visual analytics for study patterns"""
    
    def __init__(self, output_dir='./visualizations'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_study_hours_trend(self, sessions: List[Dict], student_id: str):
        """Plot study hours over time"""
        if not sessions:
            print("No sessions to plot")
            return
        
        # Extract data
        dates = [datetime.fromisoformat(s['timestamp']).date() for s in sessions]
        hours = [s.get('duration', 0) for s in sessions]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(dates, hours, marker='o', linewidth=2, markersize=8, color='#2ecc71')
        plt.fill_between(dates, hours, alpha=0.3, color='#2ecc71')
        
        plt.title(f'Study Hours Trend - {student_id}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Study Hours', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f'{self.output_dir}/study_hours_trend_{student_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Study hours trend saved: {filename}")
    
    def plot_focus_pattern(self, sessions: List[Dict], student_id: str):
        """Plot focus index pattern over time"""
        if not sessions:
            return
        
        dates = [datetime.fromisoformat(s['timestamp']).date() for s in sessions]
        focus = [s.get('focus_index', 0) for s in sessions]
        
        plt.figure(figsize=(12, 6))
        
        # Plot focus line
        plt.plot(dates, focus, marker='s', linewidth=2, markersize=8, 
                color='#3498db', label='Focus Index')
        
        # Add threshold lines
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Good Focus')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Focus')
        plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Low Focus')
        
        plt.title(f'Focus Index Pattern - {student_id}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Focus Index', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f'{self.output_dir}/focus_pattern_{student_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Focus pattern saved: {filename}")
    
    def plot_risk_distribution(self, sessions: List[Dict], student_id: str):
        """Plot procrastination risk distribution"""
        if not sessions:
            return
        
        # Count risk levels
        risk_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        for session in sessions:
            risk = session.get('risk_level', 'MEDIUM')
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        explode =  [0.05] * len(risk_counts)
        
        plt.pie(risk_counts.values(), labels=risk_counts.keys(), autopct='%1.1f%%',
               colors=colors, explode=explode, shadow=True, startangle=90)
        plt.title(f'Procrastination Risk Distribution - {student_id}', 
                 fontsize=16, fontweight='bold')
        
        filename = f'{self.output_dir}/risk_distribution_{student_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Risk distribution saved: {filename}")
    
    def plot_weekly_comparison(self, sessions: List[Dict], student_id: str):
        """Compare weekly performance metrics"""
        if not sessions:
            return
        
        # Group by week
        weekly_data = {}
        for session in sessions:
            date = datetime.fromisoformat(session['timestamp']).date()
            week = date.isocalendar()[1]  # Get week number
            
            if week not in weekly_data:
                weekly_data[week] = {
                    'study_hours': 0,
                    'focus_index': [],
                    'sessions': 0
                }
            
            weekly_data[week]['study_hours'] += session.get('duration', 0)
            weekly_data[week]['focus_index'].append(session.get('focus_index', 0))
            weekly_data[week]['sessions'] += 1
        
        # Calculate averages
        weeks = sorted(weekly_data.keys())
        study_hours = [weekly_data[w]['study_hours'] for w in weeks]
        avg_focus = [np.mean(weekly_data[w]['focus_index']) for w in weeks]
        session_counts = [weekly_data[w]['sessions'] for w in weeks]
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Study hours
        axes[0].bar(range(len(weeks)), study_hours, color='#3498db', alpha=0.7)
        axes[0].set_title('Weekly Study Hours', fontweight='bold')
        axes[0].set_ylabel('Hours')
        axes[0].grid(True, alpha=0.3)
        
        # Average focus
        axes[1].bar(range(len(weeks)), avg_focus, color='#2ecc71', alpha=0.7)
        axes[1].set_title('Weekly Average Focus Index', fontweight='bold')
        axes[1].set_ylabel('Focus Index')
        axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        
        # Session count
        axes[2].bar(range(len(weeks)), session_counts, color='#9b59b6', alpha=0.7)
        axes[2].set_title('Weekly Session Count', fontweight='bold')
        axes[2].set_ylabel('Sessions')
        axes[2].set_xlabel('Week Number')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Weekly Performance Comparison - {student_id}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = f'{self.output_dir}/weekly_comparison_{student_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Weekly comparison saved: {filename}")
    
    def plot_distraction_analysis(self, sessions: List[Dict], student_id: str):
        """Analyze distraction patterns"""
        if not sessions:
            return
        
        social_media = [s.get('social_media_hours', 0) for s in sessions]
        netflix = [s.get('netflix_hours', 0) for s in sessions]
        study = [s.get('duration', 0) for s in sessions]
        dates = [datetime.fromisoformat(s['timestamp']).date() for s in sessions]
        
        plt.figure(figsize=(14, 8))
        
        # Stacked area chart
        plt.stackplot(range(len(dates)), study, social_media, netflix,
                     labels=['Study Hours', 'Social Media', 'Netflix'],
                     colors=['#2ecc71', '#e74c3c', '#f39c12'],
                     alpha=0.7)
        
        plt.title(f'Time Allocation Analysis - {student_id}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Session Number', fontsize=12)
        plt.ylabel('Hours', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'{self.output_dir}/distraction_analysis_{student_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Distraction analysis saved: {filename}")
    
    def create_dashboard(self, sessions: List[Dict], weekly_stats: Dict, student_id: str):
        """Create comprehensive dashboard with all metrics"""
        if not sessions:
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Study Hours Trend
        ax1 = fig.add_subplot(gs[0, :2])
        dates = [datetime.fromisoformat(s['timestamp']).date() for s in sessions]
        hours = [s.get('duration', 0) for s in sessions]
        ax1.plot(dates, hours, marker='o', linewidth=2, color='#2ecc71')
        ax1.set_title('Study Hours Trend', fontweight='bold')
        ax1.set_ylabel('Hours')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Risk Distribution Pie
        ax2 = fig.add_subplot(gs[0, 2])
        risk_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        for session in sessions:
            risk = session.get('risk_level', 'MEDIUM')
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax2.pie(risk_counts.values(), labels=risk_counts.keys(), autopct='%1.0f%%',
               colors=colors, startangle=90)
        ax2.set_title('Risk Distribution', fontweight='bold')
        
        # 3. Focus Index Pattern
        ax3 = fig.add_subplot(gs[1, :2])
        focus = [s.get('focus_index', 0) for s in sessions]
        ax3.plot(dates, focus, marker='s', linewidth=2, color='#3498db')
        ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
        ax3.set_title('Focus Index Pattern', fontweight='bold')
        ax3.set_ylabel('Focus Index')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Weekly Stats Summary
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        stats_text = f"""
        Weekly Summary
        ─────────────
        Total Hours: {weekly_stats.get('total_study_hours', 0):.1f}
        
        Sessions: {weekly_stats.get('total_sessions', 0)}
        
        Avg Focus: {weekly_stats.get('average_focus_index', 0):.2f}
        
        Main Risk: {weekly_stats.get('most_common_risk', 'N/A')}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 5. Time Allocation
        ax5 = fig.add_subplot(gs[2, :])
        avg_study = np.mean([s.get('duration', 0) for s in sessions])
        avg_social = np.mean([s.get('social_media_hours', 0) for s in sessions])
        avg_netflix = np.mean([s.get('netflix_hours', 0) for s in sessions])
        
        categories = ['Study', 'Social Media', 'Netflix']
        values = [avg_study, avg_social, avg_netflix]
        colors_bar = ['#2ecc71', '#e74c3c', '#f39c12']
        
        ax5.barh(categories, values, color=colors_bar, alpha=0.7)
        ax5.set_title('Average Daily Time Allocation', fontweight='bold')
        ax5.set_xlabel('Hours')
        ax5.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f'AI Study Coach Dashboard - {student_id}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        filename = f'{self.output_dir}/dashboard_{student_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Dashboard saved: {filename}")
    
    def generate_all_visualizations(self, sessions: List[Dict], 
                                   weekly_stats: Dict, student_id: str):
        """Generate all visualization types"""
        print(f"\n{'='*60}")
        print(f"Generating Visualizations for {student_id}")
        print('='*60)
        
        self.plot_study_hours_trend(sessions, student_id)
        self.plot_focus_pattern(sessions, student_id)
        self.plot_risk_distribution(sessions, student_id)
        self.plot_weekly_comparison(sessions, student_id)
        self.plot_distraction_analysis(sessions, student_id)
        self.create_dashboard(sessions, weekly_stats, student_id)
        
        print(f"\n✓ All visualizations generated in '{self.output_dir}/' folder")


# ============================================
# DEMO USAGE
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("AI STUDY COACH - VISUALIZATION MODULE DEMO")
    print("="*60)
    
    # Create sample data
    viz = StudyVisualization()
    
    # Generate sample sessions over 2 weeks
    sample_sessions = []
    base_date = datetime.now() - timedelta(days=14)
    
    for i in range(20):
        date = base_date + timedelta(days=i)
        session = {
            'timestamp': date.isoformat(),
            'duration': np.random.uniform(2, 6),
            'social_media_hours': np.random.uniform(1, 4),
            'netflix_hours': np.random.uniform(0.5, 2.5),
            'focus_index': np.random.uniform(0.3, 1.5),
            'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], 
                                          p=[0.3, 0.5, 0.2])
        }
        sample_sessions.append(session)
    
    # Sample weekly stats
    weekly_stats = {
        'total_study_hours': 65.5,
        'total_sessions': 20,
        'average_focus_index': 0.85,
        'most_common_risk': 'MEDIUM'
    }
    
    # Generate all visualizations
    viz.generate_all_visualizations(sample_sessions, weekly_stats, 'DEMO_STUDENT')
    
    print("\n" + "="*60)
    print("Demo completed! Check the 'visualizations' folder for outputs.")
    print("="*60)