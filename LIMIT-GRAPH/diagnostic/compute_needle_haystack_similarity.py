# -*- coding: utf-8 -*-
"""
Needle-Haystack Semantic Alignment Module
Computes semantic alignment between needles and haystack content
to evaluate context interference and attention mechanisms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
import json
from pathlib import Path

@dataclass
class SemanticAlignmentResult:
    """Results from semantic alignment analysis"""
    needle_id: str
    haystack_id: str
    language: str
    needle_text: str
    haystack_segments: List[str]
    segment_similarities: List[float]
    max_similarity: float
    min_similarity: float
    mean_similarity: float
    std_similarity: float
    alignment_score: float
    interference_level: str
    embedding_model: str

class NeedleHaystackSemanticAlignment:
    """
    Analyzes semantic alignment between needles and haystack content
    to understand context interference patterns
    """
    
    def __init__(self, embedding_models: Optional[List[str]] = None, 
                 segment_size: int = 100):
        """
        Initialize semantic alignment analyzer
        
        Args:
            embedding_models: List of embedding model names
            segment_size: Size of haystack segments for analysis
        """
        self.logger = logging.getLogger(__name__)
        self.segment_size = segment_size
        
        # Load embedding models
        if embedding_models is None:
            embedding_models = [
                "sentence-transformers/LaBSE",
                "sentence-transformers/distiluse-base-multilingual-cased"
            ]
        
        self.embedding_models = {}
        for model_name in embedding_models:
            try:
                self.embedding_models[model_name] = self._load_embedding_model(model_name)
            except Exception as e:
                self.logger.warning(f"Failed to load {model_name}: {e}")
        
        # TF-IDF vectorizer for lexical similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.alignment_results = []
    
    def _load_embedding_model(self, model_name: str):
        """Load embedding model"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        return {
            'tokenizer': tokenizer,
            'model': model,
            'device': device
        }
    
    def compute_semantic_alignment(self, needle: str, haystack: str, 
                                 language: str = 'en',
                                 embedding_model: str = "sentence-transformers/LaBSE") -> SemanticAlignmentResult:
        """
        Compute semantic alignment between needle and haystack
        
        Args:
            needle: Needle text to analyze
            haystack: Haystack text containing the needle
            language: Language code
            embedding_model: Embedding model to use
            
        Returns:
            SemanticAlignmentResult with detailed analysis
        """
        # Segment haystack into chunks with language awareness
        haystack_segments = self._segment_haystack(haystack, language)
        
        # Compute similarities between needle and each segment
        segment_similarities = []
        
        if embedding_model in self.embedding_models:
            # Use neural embeddings
            segment_similarities = self._compute_neural_similarities(
                needle, haystack_segments, embedding_model
            )
        else:
            # Fallback to TF-IDF
            segment_similarities = self._compute_tfidf_similarities(
                needle, haystack_segments
            )
        
        # Calculate alignment metrics
        max_similarity = max(segment_similarities) if segment_similarities else 0.0
        min_similarity = min(segment_similarities) if segment_similarities else 0.0
        mean_similarity = np.mean(segment_similarities) if segment_similarities else 0.0
        std_similarity = np.std(segment_similarities) if segment_similarities else 0.0
        
        # Calculate alignment score (higher = better alignment)
        alignment_score = self._calculate_alignment_score(segment_similarities)
        
        # Determine interference level
        interference_level = self._classify_interference_level(
            max_similarity, mean_similarity, std_similarity
        )
        
        result = SemanticAlignmentResult(
            needle_id=f"needle_{hash(needle) % 10000}",
            haystack_id=f"haystack_{hash(haystack) % 10000}",
            language=language,
            needle_text=needle,
            haystack_segments=haystack_segments,
            segment_similarities=segment_similarities,
            max_similarity=max_similarity,
            min_similarity=min_similarity,
            mean_similarity=mean_similarity,
            std_similarity=std_similarity,
            alignment_score=alignment_score,
            interference_level=interference_level,
            embedding_model=embedding_model
        )
        
        self.alignment_results.append(result)
        return result
    
    def _segment_haystack(self, haystack: str, language: str = 'en') -> List[str]:
        """
        Segment haystack into chunks for analysis with language-aware processing
        
        Args:
            haystack: Text to segment
            language: Language code for proper segmentation
            
        Returns:
            List of text segments
        """
        # Language-specific segmentation
        if language == 'ar':
            # Arabic text segmentation - handle RTL and word boundaries
            words = self._arabic_word_segmentation(haystack)
        else:
            # Standard word segmentation for Latin scripts
            words = haystack.split()
        
        segments = []
        
        for i in range(0, len(words), self.segment_size):
            segment = " ".join(words[i:i + self.segment_size])
            segments.append(segment)
        
        return segments
    
    def _arabic_word_segmentation(self, text: str) -> List[str]:
        """
        Arabic-specific word segmentation
        
        Args:
            text: Arabic text to segment
            
        Returns:
            List of Arabic words
        """
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Split on whitespace (Arabic words are space-separated)
        words = text.split()
        
        # Additional Arabic-specific processing
        processed_words = []
        for word in words:
            # Remove punctuation but preserve Arabic text
            cleaned_word = ''.join(c for c in word if c.isalnum() or '\u0600' <= c <= '\u06FF')
            if cleaned_word:
                processed_words.append(cleaned_word)
        
        return processed_words
    
    def _compute_neural_similarities(self, needle: str, segments: List[str], 
                                   embedding_model: str) -> List[float]:
        """
        Compute neural embedding similarities
        
        Args:
            needle: Needle text
            segments: Haystack segments
            embedding_model: Model to use
            
        Returns:
            List of similarity scores
        """
        model_info = self.embedding_models[embedding_model]
        tokenizer = model_info['tokenizer']
        model = model_info['model']
        device = model_info['device']
        
        similarities = []
        
        try:
            # Encode needle
            needle_inputs = tokenizer(
                needle, return_tensors='pt', truncation=True, 
                padding=True, max_length=512
            ).to(device)
            
            with torch.no_grad():
                needle_outputs = model(**needle_inputs)
                needle_embedding = needle_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            # Encode each segment and compute similarity
            for segment in segments:
                segment_inputs = tokenizer(
                    segment, return_tensors='pt', truncation=True,
                    padding=True, max_length=512
                ).to(device)
                
                with torch.no_grad():
                    segment_outputs = model(**segment_inputs)
                    segment_embedding = segment_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                # Compute cosine similarity
                similarity = cosine_similarity(needle_embedding, segment_embedding)[0][0]
                similarities.append(float(similarity))
        
        except Exception as e:
            self.logger.error(f"Error computing neural similarities: {e}")
            # Fallback to TF-IDF
            similarities = self._compute_tfidf_similarities(needle, segments)
        
        return similarities
    
    def _compute_tfidf_similarities(self, needle: str, segments: List[str]) -> List[float]:
        """
        Compute TF-IDF based similarities
        
        Args:
            needle: Needle text
            segments: Haystack segments
            
        Returns:
            List of similarity scores
        """
        try:
            # Combine needle and segments for vectorization
            all_texts = [needle] + segments
            
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Compute similarities between needle (index 0) and segments
            needle_vector = tfidf_matrix[0:1]
            segment_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(needle_vector, segment_vectors)[0]
            return similarities.tolist()
        
        except Exception as e:
            self.logger.error(f"Error computing TF-IDF similarities: {e}")
            # Return default similarities
            return [0.5] * len(segments)
    
    def _calculate_alignment_score(self, similarities: List[float]) -> float:
        """
        Calculate overall alignment score
        
        Args:
            similarities: List of segment similarities
            
        Returns:
            Alignment score (0-1, higher is better)
        """
        if not similarities:
            return 0.0
        
        # Alignment score considers both maximum similarity and distribution
        max_sim = max(similarities)
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Good alignment: high max similarity, low interference (low std)
        alignment_score = max_sim * (1 - std_sim) * 0.8 + mean_sim * 0.2
        
        return min(1.0, max(0.0, alignment_score))
    
    def _classify_interference_level(self, max_similarity: float, 
                                   mean_similarity: float, 
                                   std_similarity: float) -> str:
        """
        Classify interference level based on similarity metrics
        
        Args:
            max_similarity: Maximum segment similarity
            mean_similarity: Mean segment similarity
            std_similarity: Standard deviation of similarities
            
        Returns:
            Interference level classification
        """
        # High interference: high mean similarity (many confusing segments)
        if mean_similarity > 0.7:
            return "high"
        elif mean_similarity > 0.5:
            return "medium"
        elif mean_similarity > 0.3:
            return "low"
        else:
            return "minimal"
    
    def analyze_interference_patterns(self, results: Optional[List[SemanticAlignmentResult]] = None) -> Dict[str, Any]:
        """
        Analyze interference patterns across multiple results
        
        Args:
            results: Results to analyze (uses stored results if None)
            
        Returns:
            Analysis report
        """
        if results is None:
            results = self.alignment_results
        
        if not results:
            return {"error": "No results available for analysis"}
        
        # Convert to DataFrame for analysis
        df_data = []
        for result in results:
            df_data.append({
                'language': result.language,
                'max_similarity': result.max_similarity,
                'mean_similarity': result.mean_similarity,
                'std_similarity': result.std_similarity,
                'alignment_score': result.alignment_score,
                'interference_level': result.interference_level,
                'num_segments': len(result.haystack_segments)
            })
        
        df = pd.DataFrame(df_data)
        
        analysis = {
            'total_analyses': len(results),
            'languages_analyzed': list(df['language'].unique()),
            'interference_distribution': df['interference_level'].value_counts().to_dict(),
            'similarity_statistics': {
                'max_similarity': {
                    'mean': float(df['max_similarity'].mean()),
                    'std': float(df['max_similarity'].std()),
                    'min': float(df['max_similarity'].min()),
                    'max': float(df['max_similarity'].max())
                },
                'mean_similarity': {
                    'mean': float(df['mean_similarity'].mean()),
                    'std': float(df['mean_similarity'].std())
                },
                'alignment_score': {
                    'mean': float(df['alignment_score'].mean()),
                    'std': float(df['alignment_score'].std())
                }
            },
            'language_performance': {},
            'interference_patterns': {}
        }
        
        # Language-specific analysis
        for lang in df['language'].unique():
            lang_df = df[df['language'] == lang]
            analysis['language_performance'][lang] = {
                'avg_alignment_score': float(lang_df['alignment_score'].mean()),
                'avg_max_similarity': float(lang_df['max_similarity'].mean()),
                'interference_levels': lang_df['interference_level'].value_counts().to_dict()
            }
        
        # Interference pattern analysis
        for level in df['interference_level'].unique():
            level_df = df[df['interference_level'] == level]
            analysis['interference_patterns'][level] = {
                'count': len(level_df),
                'avg_alignment_score': float(level_df['alignment_score'].mean()),
                'avg_similarity_std': float(level_df['std_similarity'].mean())
            }
        
        return analysis
    
    def visualize_alignment_patterns(self, results: Optional[List[SemanticAlignmentResult]] = None) -> Dict[str, plt.Figure]:
        """
        Create visualizations of alignment patterns
        
        Args:
            results: Results to visualize (uses stored results if None)
            
        Returns:
            Dictionary of figures
        """
        if results is None:
            results = self.alignment_results
        
        if not results:
            self.logger.warning("No results available for visualization")
            return {}
        
        figures = {}
        
        # Convert to DataFrame
        df_data = []
        for result in results:
            for i, (segment, similarity) in enumerate(zip(result.haystack_segments, result.segment_similarities)):
                df_data.append({
                    'needle_id': result.needle_id,
                    'language': result.language,
                    'segment_index': i,
                    'similarity': similarity,
                    'interference_level': result.interference_level,
                    'alignment_score': result.alignment_score
                })
        
        df = pd.DataFrame(df_data)
        
        # 1. Similarity distribution by interference level
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        interference_levels = df['interference_level'].unique()
        colors = ['#2E8B57', '#4682B4', '#DAA520', '#DC143C']
        
        for i, level in enumerate(interference_levels):
            level_data = df[df['interference_level'] == level]['similarity']
            ax1.hist(level_data, alpha=0.7, label=level, 
                    color=colors[i % len(colors)], bins=20)
        
        ax1.set_xlabel('Segment Similarity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Similarity Distribution by Interference Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        figures['similarity_distribution'] = fig1
        
        # 2. Alignment score vs interference patterns
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
        fig2.suptitle('Semantic Alignment Analysis', fontsize=16)
        
        # Alignment score by language
        ax2_1 = axes2[0, 0]
        result_df = pd.DataFrame([
            {
                'language': r.language,
                'alignment_score': r.alignment_score,
                'interference_level': r.interference_level
            }
            for r in results
        ])
        
        for lang in result_df['language'].unique():
            lang_data = result_df[result_df['language'] == lang]
            ax2_1.scatter(range(len(lang_data)), lang_data['alignment_score'], 
                         label=lang, alpha=0.7, s=50)
        
        ax2_1.set_xlabel('Sample Index')
        ax2_1.set_ylabel('Alignment Score')
        ax2_1.set_title('Alignment Score by Language')
        ax2_1.legend()
        ax2_1.grid(True, alpha=0.3)
        
        # Interference level distribution
        ax2_2 = axes2[0, 1]
        interference_counts = result_df['interference_level'].value_counts()
        ax2_2.pie(interference_counts.values, labels=interference_counts.index, 
                 autopct='%1.1f%%', startangle=90)
        ax2_2.set_title('Interference Level Distribution')
        
        # Similarity heatmap for sample results
        ax2_3 = axes2[1, 0]
        if len(results) > 0:
            # Create heatmap data for first few results
            heatmap_data = []
            labels = []
            for i, result in enumerate(results[:10]):  # Limit to first 10
                heatmap_data.append(result.segment_similarities[:20])  # First 20 segments
                labels.append(f"{result.language}_{i}")
            
            # Pad shorter sequences
            max_len = max(len(row) for row in heatmap_data) if heatmap_data else 0
            for row in heatmap_data:
                while len(row) < max_len:
                    row.append(0.0)
            
            if heatmap_data:
                sns.heatmap(heatmap_data, ax=ax2_3, cmap='viridis', 
                           yticklabels=labels, cbar_kws={'label': 'Similarity'})
                ax2_3.set_title('Segment Similarity Heatmap')
                ax2_3.set_xlabel('Segment Index')
        
        # Alignment score vs max similarity
        ax2_4 = axes2[1, 1]
        ax2_4.scatter([r.max_similarity for r in results], 
                     [r.alignment_score for r in results],
                     c=[hash(r.interference_level) for r in results], 
                     cmap='viridis', alpha=0.7)
        ax2_4.set_xlabel('Max Similarity')
        ax2_4.set_ylabel('Alignment Score')
        ax2_4.set_title('Alignment Score vs Max Similarity')
        ax2_4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['alignment_analysis'] = fig2
        
        # 3. Language-specific patterns
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        
        lang_alignment = result_df.groupby('language')['alignment_score'].agg(['mean', 'std']).reset_index()
        
        x_pos = np.arange(len(lang_alignment))
        ax3.bar(x_pos, lang_alignment['mean'], yerr=lang_alignment['std'], 
               capsize=5, alpha=0.7, color='skyblue')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(lang_alignment['language'])
        ax3.set_ylabel('Average Alignment Score')
        ax3.set_title('Average Alignment Score by Language')
        ax3.grid(True, alpha=0.3)
        
        figures['language_alignment'] = fig3
        
        return figures
    
    def export_alignment_results(self, filepath: str, 
                               results: Optional[List[SemanticAlignmentResult]] = None):
        """
        Export alignment results to file
        
        Args:
            filepath: Output file path
            results: Results to export (uses stored results if None)
        """
        if results is None:
            results = self.alignment_results
        
        export_data = []
        for result in results:
            export_data.append({
                'needle_id': result.needle_id,
                'haystack_id': result.haystack_id,
                'language': result.language,
                'needle_text': result.needle_text,
                'num_segments': len(result.haystack_segments),
                'segment_similarities': result.segment_similarities,
                'max_similarity': result.max_similarity,
                'min_similarity': result.min_similarity,
                'mean_similarity': result.mean_similarity,
                'std_similarity': result.std_similarity,
                'alignment_score': result.alignment_score,
                'interference_level': result.interference_level,
                'embedding_model': result.embedding_model
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Alignment results exported to {filepath}")
    
    def generate_alignment_report(self, results: Optional[List[SemanticAlignmentResult]] = None) -> str:
        """
        Generate comprehensive alignment report
        
        Args:
            results: Results to report on (uses stored results if None)
            
        Returns:
            Formatted report string
        """
        if results is None:
            results = self.alignment_results
        
        if not results:
            return "No alignment results available for reporting."
        
        analysis = self.analyze_interference_patterns(results)
        
        report_lines = [
            "# Needle-Haystack Semantic Alignment Report",
            f"**Total Analyses:** {analysis['total_analyses']}",
            f"**Languages:** {', '.join(analysis['languages_analyzed'])}",
            "",
            "## Interference Level Distribution",
        ]
        
        for level, count in analysis['interference_distribution'].items():
            percentage = (count / analysis['total_analyses']) * 100
            report_lines.append(f"- **{level.title()}:** {count} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "## Similarity Statistics",
            f"- **Average Max Similarity:** {analysis['similarity_statistics']['max_similarity']['mean']:.3f} ± {analysis['similarity_statistics']['max_similarity']['std']:.3f}",
            f"- **Average Mean Similarity:** {analysis['similarity_statistics']['mean_similarity']['mean']:.3f} ± {analysis['similarity_statistics']['mean_similarity']['std']:.3f}",
            f"- **Average Alignment Score:** {analysis['similarity_statistics']['alignment_score']['mean']:.3f} ± {analysis['similarity_statistics']['alignment_score']['std']:.3f}",
            "",
            "## Language Performance"
        ])
        
        for lang, perf in analysis['language_performance'].items():
            report_lines.extend([
                f"### {lang.upper()}",
                f"- **Average Alignment Score:** {perf['avg_alignment_score']:.3f}",
                f"- **Average Max Similarity:** {perf['avg_max_similarity']:.3f}",
                f"- **Interference Levels:** {perf['interference_levels']}",
                ""
            ])
        
        report_lines.extend([
            "## Interference Pattern Analysis"
        ])
        
        for level, pattern in analysis['interference_patterns'].items():
            report_lines.extend([
                f"### {level.title()} Interference",
                f"- **Count:** {pattern['count']}",
                f"- **Average Alignment Score:** {pattern['avg_alignment_score']:.3f}",
                f"- **Average Similarity Std:** {pattern['avg_similarity_std']:.3f}",
                ""
            ])
        
        return "\n".join(report_lines)