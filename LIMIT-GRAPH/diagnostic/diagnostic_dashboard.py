# -*- coding: utf-8 -*-
"""
Diagnostic Dashboard Module
Comprehensive visualization and monitoring for long-context robustness evaluation
"""

import json
import time
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from dataclasses import asdict

class DiagnosticDashboard:
    """
    Dashboard for visualizing diagnostic evaluation results
    """
    
    def __init__(self):
        """Initialize diagnostic dashboard"""
        self.evaluation_data = {
            'similarity_evaluations': [],
            'distractor_evaluations': [],
            'alignment_evaluations': [],
            'structure_evaluations': [],
            'repetition_evaluations': []
        }
        
        self.performance_metrics = {}
        
        # Language-specific configurations
        self.language_configs = {
            'en': {'name': 'English', 'rtl': False, 'color': '#1f77b4'},
            'id': {'name': 'Indonesian', 'rtl': False, 'color': '#ff7f0e'},
            'ar': {'name': 'Arabic', 'rtl': True, 'color': '#2ca02c'},
            'es': {'name': 'Spanish', 'rtl': False, 'color': '#d62728'},
            'fr': {'name': 'French', 'rtl': False, 'color': '#9467bd'}
        }
        
    def update_similarity_data(self, similarity_results: List[Any]):
        """Update dashboard with similarity evaluation results"""
        self.evaluation_data['similarity_evaluations'].extend(similarity_results)
        self._update_performance_metrics()
    
    def update_distractor_data(self, distractor_results: List[Any]):
        """Update dashboard with distractor injection results"""
        self.evaluation_data['distractor_evaluations'].extend(distractor_results)
        self._update_performance_metrics()
    
    def update_alignment_data(self, alignment_results: List[Any]):
        """Update dashboard with semantic alignment results"""
        self.evaluation_data['alignment_evaluations'].extend(alignment_results)
        self._update_performance_metrics()
    
    def update_structure_data(self, structure_results: List[Any]):
        """Update dashboard with structure sensitivity results"""
        self.evaluation_data['structure_evaluations'].extend(structure_results)
        self._update_performance_metrics()
    
    def update_repetition_data(self, repetition_results: List[Any]):
        """Update dashboard with repetition task results"""
        self.evaluation_data['repetition_evaluations'].extend(repetition_results)
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update aggregate performance metrics"""
        self.performance_metrics = {
            'total_evaluations': sum(len(evals) for evals in self.evaluation_data.values()),
            'evaluation_types': len([k for k, v in self.evaluation_data.items() if v]),
            'last_updated': datetime.now().isoformat()
        }
    
    def create_similarity_analysis_dashboard(self) -> go.Figure:
        """Create similarity analysis dashboard"""
        similarity_data = self.evaluation_data['similarity_evaluations']
        
        if not similarity_data:
            return go.Figure().add_annotation(text="No similarity data available")
        
        # Convert to DataFrame
        df_data = []
        for result in similarity_data:
            if hasattr(result, '__dict__'):
                result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
            else:
                result_dict = result
            
            df_data.append({
                'language': result_dict.get('language', 'unknown'),
                'input_length': result_dict.get('input_length', 0),
                'similarity_band': result_dict.get('similarity_band', 'unknown'),
                'accuracy': result_dict.get('accuracy', 0),
                'response_time': result_dict.get('response_time', 0),
                'confidence_score': result_dict.get('confidence_score', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Accuracy vs Input Length", "Similarity Band Performance", 
                          "Response Time Analysis", "Confidence Calibration"),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Accuracy vs Input Length by Language
        for language in df['language'].unique():
            lang_df = df[df['language'] == language]
            fig.add_trace(
                go.Scatter(
                    x=lang_df['input_length'],
                    y=lang_df['accuracy'],
                    mode='markers+lines',
                    name=f'{language.upper()} Accuracy',
                    legendgroup='accuracy'
                ),
                row=1, col=1
            )
        
        # 2. Performance by Similarity Band
        band_performance = df.groupby('similarity_band')['accuracy'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=band_performance['similarity_band'],
                y=band_performance['accuracy'],
                name='Avg Accuracy by Band',
                marker_color='lightblue',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Response Time vs Input Length
        fig.add_trace(
            go.Scatter(
                x=df['input_length'],
                y=df['response_time'],
                mode='markers',
                name='Response Time',
                marker=dict(color='red', size=6),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Confidence vs Accuracy
        fig.add_trace(
            go.Scatter(
                x=df['confidence_score'],
                y=df['accuracy'],
                mode='markers',
                name='Confidence vs Accuracy',
                marker=dict(color='green', size=6),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Needle-Question Similarity Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_distractor_effectiveness_dashboard(self) -> go.Figure:
        """Create distractor effectiveness dashboard"""
        distractor_data = self.evaluation_data['distractor_evaluations']
        
        if not distractor_data:
            return go.Figure().add_annotation(text="No distractor data available")
        
        # Convert to DataFrame
        df_data = []
        for result in distractor_data:
            if hasattr(result, '__dict__'):
                result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
            else:
                result_dict = result
            
            df_data.append({
                'distractor_type': result_dict.get('injection_config', {}).get('distractor_type', {}).get('value', 'unknown'),
                'language': result_dict.get('injection_config', {}).get('language', 'unknown'),
                'injection_ratio': result_dict.get('injection_config', {}).get('injection_ratio', 0),
                'distractors_count': len(result_dict.get('distractors_injected', [])),
                'haystack_length_change': result_dict.get('haystack_length_change', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Distractor Types Distribution", "Injection Ratio Impact", 
                          "Language Performance", "Haystack Length Changes"),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # 1. Distractor Types Distribution
        type_counts = df['distractor_type'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                name="Distractor Types",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Injection Ratio Impact
        fig.add_trace(
            go.Scatter(
                x=df['injection_ratio'],
                y=df['distractors_count'],
                mode='markers',
                name='Injection Impact',
                marker=dict(color='blue', size=8),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Language Performance
        lang_performance = df.groupby('language')['distractors_count'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=lang_performance['language'],
                y=lang_performance['distractors_count'],
                name='Avg Distractors by Language',
                marker_color='orange',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Haystack Length Changes
        fig.add_trace(
            go.Histogram(
                x=df['haystack_length_change'],
                name='Length Changes',
                marker_color='purple',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Distractor Injection Effectiveness Dashboard",
            height=800
        )
        
        return fig
    
    def create_semantic_alignment_dashboard(self) -> go.Figure:
        """Create semantic alignment dashboard"""
        alignment_data = self.evaluation_data['alignment_evaluations']
        
        if not alignment_data:
            return go.Figure().add_annotation(text="No alignment data available")
        
        # Convert to DataFrame
        df_data = []
        for result in alignment_data:
            if hasattr(result, '__dict__'):
                result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
            else:
                result_dict = result
            
            df_data.append({
                'language': result_dict.get('language', 'unknown'),
                'max_similarity': result_dict.get('max_similarity', 0),
                'mean_similarity': result_dict.get('mean_similarity', 0),
                'std_similarity': result_dict.get('std_similarity', 0),
                'alignment_score': result_dict.get('alignment_score', 0),
                'interference_level': result_dict.get('interference_level', 'unknown'),
                'num_segments': len(result_dict.get('haystack_segments', []))
            })
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Alignment Score Distribution", "Interference Levels", 
                          "Similarity Statistics", "Language Comparison"),
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Alignment Score Distribution
        fig.add_trace(
            go.Histogram(
                x=df['alignment_score'],
                name='Alignment Scores',
                marker_color='lightgreen',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Interference Levels
        interference_counts = df['interference_level'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=interference_counts.index,
                values=interference_counts.values,
                name="Interference Levels",
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Max vs Mean Similarity
        fig.add_trace(
            go.Scatter(
                x=df['max_similarity'],
                y=df['mean_similarity'],
                mode='markers',
                name='Similarity Comparison',
                marker=dict(color='red', size=8),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Language Comparison
        lang_alignment = df.groupby('language')['alignment_score'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=lang_alignment['language'],
                y=lang_alignment['alignment_score'],
                name='Avg Alignment by Language',
                marker_color='skyblue',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Semantic Alignment Analysis Dashboard",
            height=800
        )
        
        return fig
    
    def create_structure_sensitivity_dashboard(self) -> go.Figure:
        """Create structure sensitivity dashboard"""
        structure_data = self.evaluation_data['structure_evaluations']
        
        if not structure_data:
            return go.Figure().add_annotation(text="No structure data available")
        
        # Convert to DataFrame
        df_data = []
        for result in structure_data:
            if hasattr(result, '__dict__'):
                result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
            else:
                result_dict = result
            
            df_data.append({
                'strategy': result_dict.get('shuffle_config', {}).get('strategy', {}).get('value', 'unknown'),
                'language': result_dict.get('shuffle_config', {}).get('language', 'unknown'),
                'shuffle_ratio': result_dict.get('shuffle_config', {}).get('shuffle_ratio', 0),
                'structure_change_score': result_dict.get('structure_change_score', 0),
                'coherence_score': result_dict.get('coherence_score', 0),
                'readability_score': result_dict.get('readability_score', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Structure Change by Strategy", "Coherence vs Readability", 
                          "Shuffle Ratio Impact", "Strategy Effectiveness"),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Structure Change by Strategy
        strategy_change = df.groupby('strategy')['structure_change_score'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=strategy_change['strategy'],
                y=strategy_change['structure_change_score'],
                name='Avg Structure Change',
                marker_color='coral',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Coherence vs Readability
        fig.add_trace(
            go.Scatter(
                x=df['coherence_score'],
                y=df['readability_score'],
                mode='markers',
                name='Coherence vs Readability',
                marker=dict(color='purple', size=8),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Shuffle Ratio Impact
        fig.add_trace(
            go.Scatter(
                x=df['shuffle_ratio'],
                y=df['structure_change_score'],
                mode='markers',
                name='Ratio Impact',
                marker=dict(color='green', size=8),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Strategy Effectiveness (lower change = more effective)
        strategy_effectiveness = df.groupby('strategy')['coherence_score'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=strategy_effectiveness['strategy'],
                y=strategy_effectiveness['coherence_score'],
                name='Avg Coherence Preservation',
                marker_color='gold',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Structure Sensitivity Analysis Dashboard",
            height=800
        )
        
        return fig
    
    def create_repetition_robustness_dashboard(self) -> go.Figure:
        """Create repetition robustness dashboard"""
        repetition_data = self.evaluation_data['repetition_evaluations']
        
        if not repetition_data:
            return go.Figure().add_annotation(text="No repetition data available")
        
        # Convert to DataFrame
        df_data = []
        for result in repetition_data:
            if hasattr(result, '__dict__'):
                result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
            else:
                result_dict = result
            
            df_data.append({
                'strategy': result_dict.get('repetition_config', {}).get('strategy', {}).get('value', 'unknown'),
                'language': result_dict.get('repetition_config', {}).get('language', 'unknown'),
                'repetition_count': result_dict.get('repetition_config', {}).get('repetition_count', 0),
                'repetition_density': result_dict.get('repetition_density', 0),
                'content_inflation_ratio': result_dict.get('content_inflation_ratio', 1),
                'uniqueness_preservation_score': result_dict.get('uniqueness_preservation_score', 0),
                'needle_occurrences': len(result_dict.get('needle_positions_in_repeated', []))
            })
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Repetition Strategies", "Content Inflation", 
                          "Uniqueness Preservation", "Density vs Inflation"),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Repetition Strategies Performance
        strategy_density = df.groupby('strategy')['repetition_density'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=strategy_density['strategy'],
                y=strategy_density['repetition_density'],
                name='Avg Repetition Density',
                marker_color='lightcoral',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Content Inflation Distribution
        fig.add_trace(
            go.Histogram(
                x=df['content_inflation_ratio'],
                name='Inflation Ratios',
                marker_color='lightblue',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Uniqueness Preservation by Strategy
        strategy_uniqueness = df.groupby('strategy')['uniqueness_preservation_score'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=strategy_uniqueness['strategy'],
                y=strategy_uniqueness['uniqueness_preservation_score'],
                name='Avg Uniqueness Preservation',
                marker_color='lightgreen',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Density vs Inflation Relationship
        fig.add_trace(
            go.Scatter(
                x=df['repetition_density'],
                y=df['content_inflation_ratio'],
                mode='markers',
                name='Density vs Inflation',
                marker=dict(color='orange', size=8),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Repetition Robustness Analysis Dashboard",
            height=800
        )
        
        return fig
    
    def create_comprehensive_overview(self) -> go.Figure:
        """Create comprehensive overview dashboard"""
        # Calculate summary statistics
        total_evals = sum(len(evals) for evals in self.evaluation_data.values())
        
        if total_evals == 0:
            return go.Figure().add_annotation(text="No evaluation data available")
        
        # Create overview metrics
        eval_counts = {k: len(v) for k, v in self.evaluation_data.items()}
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Evaluation Types Distribution", "Data Completeness", 
                          "Evaluation Timeline", "Performance Summary"),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Evaluation Types Distribution
        fig.add_trace(
            go.Pie(
                labels=list(eval_counts.keys()),
                values=list(eval_counts.values()),
                name="Evaluation Types",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Data Completeness
        fig.add_trace(
            go.Bar(
                x=list(eval_counts.keys()),
                y=list(eval_counts.values()),
                name='Evaluation Counts',
                marker_color='skyblue',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Mock Timeline (since we don't have timestamps)
        timeline_x = list(range(len(eval_counts)))
        timeline_y = list(eval_counts.values())
        fig.add_trace(
            go.Scatter(
                x=timeline_x,
                y=timeline_y,
                mode='lines+markers',
                name='Evaluation Progress',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Performance Summary Table
        summary_data = [
            ["Total Evaluations", str(total_evals)],
            ["Evaluation Types", str(len([k for k, v in eval_counts.items() if v > 0]))],
            ["Most Common Type", max(eval_counts.items(), key=lambda x: x[1])[0] if eval_counts else "None"],
            ["Last Updated", self.performance_metrics.get('last_updated', 'Never')]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"]),
                cells=dict(values=list(zip(*summary_data))),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Long-Context Robustness Diagnostic Overview",
            height=800
        )
        
        return fig
    
    def export_dashboard_data(self, filepath: str) -> bool:
        """Export all dashboard data to file"""
        try:
            export_data = {
                "evaluation_data": self.evaluation_data,
                "performance_metrics": self.performance_metrics,
                "export_timestamp": datetime.now().isoformat(),
                "total_evaluations": sum(len(evals) for evals in self.evaluation_data.values())
            }
            
            # Convert any non-serializable objects to dictionaries
            serializable_data = self._make_serializable(export_data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error exporting dashboard data: {e}")
            return False
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'value'):  # For Enum objects
            return obj.value
        else:
            return obj

def create_streamlit_diagnostic_dashboard():
    """Create Streamlit diagnostic dashboard interface"""
    st.set_page_config(
        page_title="LIMIT-GRAPH Diagnostic Dashboard",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 LIMIT-GRAPH Long-Context Robustness Diagnostic Dashboard")
    st.markdown("Comprehensive analysis of multilingual agent performance under various long-context challenges")
    
    # Initialize dashboard
    if 'diagnostic_dashboard' not in st.session_state:
        st.session_state.diagnostic_dashboard = DiagnosticDashboard()
    
    dashboard = st.session_state.diagnostic_dashboard
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # File upload for evaluation data
    uploaded_files = st.sidebar.file_uploader(
        "Upload Evaluation Results", 
        type=['json'],
        accept_multiple_files=True,
        help="Upload JSON files with evaluation results"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                evaluation_data = json.load(uploaded_file)
                
                # Determine data type and update dashboard
                if 'similarity_evaluations' in uploaded_file.name:
                    dashboard.update_similarity_data(evaluation_data)
                elif 'distractor' in uploaded_file.name:
                    dashboard.update_distractor_data(evaluation_data)
                elif 'alignment' in uploaded_file.name:
                    dashboard.update_alignment_data(evaluation_data)
                elif 'structure' in uploaded_file.name:
                    dashboard.update_structure_data(evaluation_data)
                elif 'repetition' in uploaded_file.name:
                    dashboard.update_repetition_data(evaluation_data)
                else:
                    # Try to auto-detect data type
                    if isinstance(evaluation_data, list) and evaluation_data:
                        sample = evaluation_data[0]
                        if 'similarity_band' in str(sample):
                            dashboard.update_similarity_data(evaluation_data)
                        elif 'distractor' in str(sample):
                            dashboard.update_distractor_data(evaluation_data)
                        # Add more auto-detection logic as needed
                
                st.sidebar.success(f"Loaded {uploaded_file.name}")
                
            except Exception as e:
                st.sidebar.error(f"Error loading {uploaded_file.name}: {e}")
    
    # Dashboard selection
    dashboard_type = st.sidebar.selectbox(
        "Select Dashboard View",
        [
            "Comprehensive Overview",
            "Similarity Analysis",
            "Distractor Effectiveness", 
            "Semantic Alignment",
            "Structure Sensitivity",
            "Repetition Robustness"
        ]
    )
    
    # Main dashboard content
    if dashboard_type == "Comprehensive Overview":
        st.header("📊 Comprehensive Overview")
        overview_fig = dashboard.create_comprehensive_overview()
        st.plotly_chart(overview_fig, use_container_width=True)
        
    elif dashboard_type == "Similarity Analysis":
        st.header("🎯 Needle-Question Similarity Analysis")
        similarity_fig = dashboard.create_similarity_analysis_dashboard()
        st.plotly_chart(similarity_fig, use_container_width=True)
        
    elif dashboard_type == "Distractor Effectiveness":
        st.header("🎭 Distractor Injection Effectiveness")
        distractor_fig = dashboard.create_distractor_effectiveness_dashboard()
        st.plotly_chart(distractor_fig, use_container_width=True)
        
    elif dashboard_type == "Semantic Alignment":
        st.header("🔗 Semantic Alignment Analysis")
        alignment_fig = dashboard.create_semantic_alignment_dashboard()
        st.plotly_chart(alignment_fig, use_container_width=True)
        
    elif dashboard_type == "Structure Sensitivity":
        st.header("🏗️ Structure Sensitivity Analysis")
        structure_fig = dashboard.create_structure_sensitivity_dashboard()
        st.plotly_chart(structure_fig, use_container_width=True)
        
    elif dashboard_type == "Repetition Robustness":
        st.header("🔄 Repetition Robustness Analysis")
        repetition_fig = dashboard.create_repetition_robustness_dashboard()
        st.plotly_chart(repetition_fig, use_container_width=True)
    
    # Performance metrics sidebar
    st.sidebar.header("Performance Metrics")
    if dashboard.performance_metrics:
        st.sidebar.metric("Total Evaluations", dashboard.performance_metrics.get('total_evaluations', 0))
        st.sidebar.metric("Evaluation Types", dashboard.performance_metrics.get('evaluation_types', 0))
        st.sidebar.text(f"Last Updated: {dashboard.performance_metrics.get('last_updated', 'Never')}")
    
    # Export functionality
    if st.sidebar.button("Export Dashboard Data"):
        export_path = f"diagnostic_dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if dashboard.export_dashboard_data(export_path):
            st.sidebar.success(f"Data exported to {export_path}")
        else:
            st.sidebar.error("Export failed")

if __name__ == "__main__":
    create_streamlit_diagnostic_dashboard()    def cre
ate_multilingual_performance_chart(self) -> go.Figure:
        """Create multilingual performance comparison chart"""
        # Aggregate data by language
        language_performance = {}
        
        for eval_type, evaluations in self.evaluation_data.items():
            for evaluation in evaluations:
                if hasattr(evaluation, 'language'):
                    lang = evaluation.language
                    if lang not in language_performance:
                        language_performance[lang] = {
                            'accuracy_scores': [],
                            'response_times': [],
                            'confidence_scores': [],
                            'evaluation_count': 0
                        }
                    
                    # Extract metrics based on evaluation type
                    if hasattr(evaluation, 'accuracy'):
                        language_performance[lang]['accuracy_scores'].append(evaluation.accuracy)
                    if hasattr(evaluation, 'response_time'):
                        language_performance[lang]['response_times'].append(evaluation.response_time)
                    if hasattr(evaluation, 'confidence_score'):
                        language_performance[lang]['confidence_scores'].append(evaluation.confidence_score)
                    
                    language_performance[lang]['evaluation_count'] += 1
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy by Language', 'Response Time by Language',
                          'Confidence by Language', 'Evaluation Count by Language'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        languages = list(language_performance.keys())
        
        # Accuracy comparison
        accuracy_means = []
        response_time_means = []
        confidence_means = []
        eval_counts = []
        
        for lang in languages:
            perf = language_performance[lang]
            accuracy_means.append(np.mean(perf['accuracy_scores']) if perf['accuracy_scores'] else 0)
            response_time_means.append(np.mean(perf['response_times']) if perf['response_times'] else 0)
            confidence_means.append(np.mean(perf['confidence_scores']) if perf['confidence_scores'] else 0)
            eval_counts.append(perf['evaluation_count'])
        
        # Add traces
        fig.add_trace(
            go.Bar(x=languages, y=accuracy_means, name='Accuracy',
                   marker_color=[self.language_configs.get(lang, {}).get('color', '#1f77b4') for lang in languages]),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=languages, y=response_time_means, name='Response Time',
                   marker_color=[self.language_configs.get(lang, {}).get('color', '#1f77b4') for lang in languages]),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=languages, y=confidence_means, name='Confidence',
                   marker_color=[self.language_configs.get(lang, {}).get('color', '#1f77b4') for lang in languages]),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=languages, y=eval_counts, name='Evaluation Count',
                   marker_color=[self.language_configs.get(lang, {}).get('color', '#1f77b4') for lang in languages]),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Multilingual Diagnostic Performance Analysis",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_arabic_specific_analysis(self) -> go.Figure:
        """Create Arabic-specific performance analysis"""
        # Filter Arabic evaluations
        arabic_evaluations = []
        
        for eval_type, evaluations in self.evaluation_data.items():
            for evaluation in evaluations:
                if hasattr(evaluation, 'language') and evaluation.language == 'ar':
                    arabic_evaluations.append({
                        'type': eval_type,
                        'evaluation': evaluation
                    })
        
        if not arabic_evaluations:
            return go.Figure().add_annotation(text="No Arabic evaluation data available")
        
        # Create Arabic-specific analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RTL Text Performance', 'Script Variation Robustness',
                          'Diacritic Handling', 'Cultural Context Adaptation'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "box"}]]
        )
        
        # RTL performance analysis
        rtl_scores = []
        text_lengths = []
        
        for item in arabic_evaluations:
            evaluation = item['evaluation']
            if hasattr(evaluation, 'accuracy'):
                rtl_scores.append(evaluation.accuracy)
                # Estimate text length (simplified)
                if hasattr(evaluation, 'needle_text'):
                    text_lengths.append(len(evaluation.needle_text))
                else:
                    text_lengths.append(50)  # Default
        
        if rtl_scores and text_lengths:
            fig.add_trace(
                go.Scatter(x=text_lengths, y=rtl_scores, mode='markers',
                          name='RTL Performance', marker=dict(color='green')),
                row=1, col=1
            )
        
        # Script variation analysis (mock data for demonstration)
        script_variations = ['Original', 'No Diacritics', 'Extra Diacritics', 'Mixed Script']
        variation_scores = [0.85, 0.78, 0.72, 0.65]  # Mock scores
        
        fig.add_trace(
            go.Bar(x=script_variations, y=variation_scores, name='Script Variations',
                   marker_color='orange'),
            row=1, col=2
        )
        
        # Diacritic handling histogram
        if rtl_scores:
            fig.add_trace(
                go.Histogram(x=rtl_scores, name='Accuracy Distribution',
                            marker_color='blue', nbinsx=10),
                row=2, col=1
            )
        
        # Cultural context box plot (mock data)
        context_types = ['Religious', 'Historical', 'Modern', 'Technical']
        context_scores = [
            [0.8, 0.85, 0.82, 0.88, 0.79],  # Religious
            [0.75, 0.78, 0.76, 0.81, 0.74],  # Historical
            [0.82, 0.86, 0.84, 0.89, 0.81],  # Modern
            [0.70, 0.73, 0.71, 0.76, 0.69]   # Technical
        ]
        
        for i, (context_type, scores) in enumerate(zip(context_types, context_scores)):
            fig.add_trace(
                go.Box(y=scores, name=context_type, x=[context_type]*len(scores)),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Arabic Language Diagnostic Analysis",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_rtl_text_analysis(self) -> Dict[str, Any]:
        """Create RTL text handling analysis"""
        rtl_analysis = {
            'total_rtl_evaluations': 0,
            'rtl_performance_metrics': {},
            'bidi_compliance_score': 0.0,
            'rendering_issues_detected': [],
            'rtl_vs_ltr_comparison': {}
        }
        
        # Analyze RTL evaluations
        rtl_evaluations = []
        ltr_evaluations = []
        
        for eval_type, evaluations in self.evaluation_data.items():
            for evaluation in evaluations:
                if hasattr(evaluation, 'language'):
                    if evaluation.language == 'ar':
                        rtl_evaluations.append(evaluation)
                    elif evaluation.language in ['en', 'id', 'es', 'fr']:
                        ltr_evaluations.append(evaluation)
        
        rtl_analysis['total_rtl_evaluations'] = len(rtl_evaluations)
        
        # Calculate RTL performance metrics
        if rtl_evaluations:
            rtl_accuracies = [e.accuracy for e in rtl_evaluations if hasattr(e, 'accuracy')]
            if rtl_accuracies:
                rtl_analysis['rtl_performance_metrics'] = {
                    'mean_accuracy': np.mean(rtl_accuracies),
                    'std_accuracy': np.std(rtl_accuracies),
                    'min_accuracy': np.min(rtl_accuracies),
                    'max_accuracy': np.max(rtl_accuracies)
                }
        
        # Compare RTL vs LTR performance
        if rtl_evaluations and ltr_evaluations:
            rtl_acc = [e.accuracy for e in rtl_evaluations if hasattr(e, 'accuracy')]
            ltr_acc = [e.accuracy for e in ltr_evaluations if hasattr(e, 'accuracy')]
            
            if rtl_acc and ltr_acc:
                rtl_analysis['rtl_vs_ltr_comparison'] = {
                    'rtl_mean': np.mean(rtl_acc),
                    'ltr_mean': np.mean(ltr_acc),
                    'performance_gap': np.mean(ltr_acc) - np.mean(rtl_acc),
                    'statistical_significance': 'mock_p_value_0.05'  # Would use proper statistical test
                }
        
        return rtl_analysis