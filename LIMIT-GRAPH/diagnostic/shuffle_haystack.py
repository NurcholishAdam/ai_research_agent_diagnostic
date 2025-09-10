# -*- coding: utf-8 -*-
"""
Haystack Structure Sensitivity Module
Tests agent sensitivity to haystack structure and organization
through various shuffling and reordering strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import random
import json
from dataclasses import dataclass
from enum import Enum
import re

class ShuffleStrategy(Enum):
    """Types of haystack shuffling strategies"""
    SENTENCE_SHUFFLE = "sentence_shuffle"
    PARAGRAPH_SHUFFLE = "paragraph_shuffle"
    WORD_SHUFFLE = "word_shuffle"
    RANDOM_INSERTION = "random_insertion"
    REVERSE_ORDER = "reverse_order"
    TOPIC_SCRAMBLE = "topic_scramble"
    TEMPORAL_DISORDER = "temporal_disorder"
    SEMANTIC_CLUSTERING = "semantic_clustering"

@dataclass
class ShuffleConfig:
    """Configuration for haystack shuffling"""
    strategy: ShuffleStrategy
    shuffle_ratio: float  # Proportion of content to shuffle
    preserve_needle: bool  # Whether to keep needle position fixed
    language: str
    seed: int = 42

@dataclass
class StructureSensitivityResult:
    """Result of structure sensitivity test"""
    original_haystack: str
    shuffled_haystack: str
    needle_text: str
    original_needle_position: int
    shuffled_needle_position: int
    shuffle_config: ShuffleConfig
    structure_change_score: float
    coherence_score: float
    readability_score: float
    shuffle_metadata: Dict[str, Any]

class HaystackStructureSensitivity:
    """
    Tests agent sensitivity to haystack structure through various
    shuffling and reordering strategies
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize structure sensitivity tester
        
        Args:
            seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        random.seed(seed)
        np.random.seed(seed)
        
        self.sensitivity_results = []
        
        # Language-specific sentence delimiters
        self.sentence_delimiters = {
            'en': ['.', '!', '?'],
            'id': ['.', '!', '?'],
            'es': ['.', '!', '?', '¡', '¿'],
            'fr': ['.', '!', '?'],
            'de': ['.', '!', '?'],
            'ar': ['.', '!', '?', '؟', '؍']  # Arabic question mark and other punctuation
        }
        
        # Common topic keywords for semantic clustering
        self.topic_keywords = {
            'en': {
                'science': ['research', 'study', 'experiment', 'theory', 'discovery', 'scientist'],
                'history': ['year', 'century', 'war', 'empire', 'ancient', 'historical'],
                'geography': ['country', 'city', 'mountain', 'river', 'continent', 'capital'],
                'culture': ['art', 'music', 'literature', 'tradition', 'festival', 'custom'],
                'technology': ['computer', 'internet', 'software', 'digital', 'innovation', 'device']
            },
            'id': {
                'sains': ['penelitian', 'studi', 'eksperimen', 'teori', 'penemuan', 'ilmuwan'],
                'sejarah': ['tahun', 'abad', 'perang', 'kerajaan', 'kuno', 'bersejarah'],
                'geografi': ['negara', 'kota', 'gunung', 'sungai', 'benua', 'ibu kota'],
                'budaya': ['seni', 'musik', 'sastra', 'tradisi', 'festival', 'adat'],
                'teknologi': ['komputer', 'internet', 'perangkat lunak', 'digital', 'inovasi', 'perangkat']
            },
            'ar': {
                'علوم': ['بحث', 'دراسة', 'تجربة', 'نظرية', 'اكتشاف', 'عالم'],
                'تاريخ': ['سنة', 'قرن', 'حرب', 'إمبراطورية', 'قديم', 'تاريخي'],
                'جغرافيا': ['دولة', 'مدينة', 'جبل', 'نهر', 'قارة', 'عاصمة'],
                'ثقافة': ['فن', 'موسيقى', 'أدب', 'تقليد', 'مهرجان', 'عادة'],
                'تكنولوجيا': ['حاسوب', 'إنترنت', 'برمجيات', 'رقمي', 'ابتكار', 'جهاز']
            }
        }
    
    def apply_shuffle_strategy(self, haystack: str, needle: str, 
                             config: ShuffleConfig) -> StructureSensitivityResult:
        """
        Apply shuffling strategy to haystack
        
        Args:
            haystack: Original haystack text
            needle: Needle text to track
            config: Shuffle configuration
            
        Returns:
            StructureSensitivityResult with shuffled haystack and analysis
        """
        # Find original needle position
        original_needle_position = haystack.find(needle)
        if original_needle_position == -1:
            self.logger.warning("Needle not found in haystack")
            original_needle_position = len(haystack) // 2
        
        # Apply shuffling strategy
        if config.strategy == ShuffleStrategy.SENTENCE_SHUFFLE:
            shuffled_haystack, metadata = self._shuffle_sentences(haystack, needle, config)
        elif config.strategy == ShuffleStrategy.PARAGRAPH_SHUFFLE:
            shuffled_haystack, metadata = self._shuffle_paragraphs(haystack, needle, config)
        elif config.strategy == ShuffleStrategy.WORD_SHUFFLE:
            shuffled_haystack, metadata = self._shuffle_words(haystack, needle, config)
        elif config.strategy == ShuffleStrategy.RANDOM_INSERTION:
            shuffled_haystack, metadata = self._random_insertion(haystack, needle, config)
        elif config.strategy == ShuffleStrategy.REVERSE_ORDER:
            shuffled_haystack, metadata = self._reverse_order(haystack, needle, config)
        elif config.strategy == ShuffleStrategy.TOPIC_SCRAMBLE:
            shuffled_haystack, metadata = self._topic_scramble(haystack, needle, config)
        elif config.strategy == ShuffleStrategy.TEMPORAL_DISORDER:
            shuffled_haystack, metadata = self._temporal_disorder(haystack, needle, config)
        elif config.strategy == ShuffleStrategy.SEMANTIC_CLUSTERING:
            shuffled_haystack, metadata = self._semantic_clustering(haystack, needle, config)
        else:
            # Default: no shuffling
            shuffled_haystack = haystack
            metadata = {'strategy': 'none', 'changes_made': 0}
        
        # Find new needle position
        shuffled_needle_position = shuffled_haystack.find(needle)
        if shuffled_needle_position == -1:
            shuffled_needle_position = original_needle_position
        
        # Calculate metrics
        structure_change_score = self._calculate_structure_change(haystack, shuffled_haystack)
        coherence_score = self._calculate_coherence_score(shuffled_haystack, config.language)
        readability_score = self._calculate_readability_score(shuffled_haystack, config.language)
        
        result = StructureSensitivityResult(
            original_haystack=haystack,
            shuffled_haystack=shuffled_haystack,
            needle_text=needle,
            original_needle_position=original_needle_position,
            shuffled_needle_position=shuffled_needle_position,
            shuffle_config=config,
            structure_change_score=structure_change_score,
            coherence_score=coherence_score,
            readability_score=readability_score,
            shuffle_metadata=metadata
        )
        
        self.sensitivity_results.append(result)
        return result
    
    def _shuffle_sentences(self, haystack: str, needle: str, 
                          config: ShuffleConfig) -> Tuple[str, Dict[str, Any]]:
        """Shuffle sentences within the haystack"""
        delimiters = self.sentence_delimiters.get(config.language, ['.', '!', '?'])
        
        # Split into sentences
        sentences = []
        current_sentence = ""
        
        for char in haystack:
            current_sentence += char
            if char in delimiters:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Find sentence containing needle
        needle_sentence_idx = -1
        for i, sentence in enumerate(sentences):
            if needle in sentence:
                needle_sentence_idx = i
                break
        
        # Determine sentences to shuffle
        num_to_shuffle = int(len(sentences) * config.shuffle_ratio)
        
        if config.preserve_needle and needle_sentence_idx != -1:
            # Exclude needle sentence from shuffling
            shuffle_indices = [i for i in range(len(sentences)) if i != needle_sentence_idx]
            shuffle_indices = random.sample(shuffle_indices, min(num_to_shuffle, len(shuffle_indices)))
        else:
            shuffle_indices = random.sample(range(len(sentences)), min(num_to_shuffle, len(sentences)))
        
        # Shuffle selected sentences
        shuffled_sentences = sentences.copy()
        sentences_to_shuffle = [shuffled_sentences[i] for i in shuffle_indices]
        random.shuffle(sentences_to_shuffle)
        
        for i, idx in enumerate(shuffle_indices):
            shuffled_sentences[idx] = sentences_to_shuffle[i]
        
        shuffled_haystack = " ".join(shuffled_sentences)
        
        metadata = {
            'strategy': 'sentence_shuffle',
            'total_sentences': len(sentences),
            'sentences_shuffled': len(shuffle_indices),
            'needle_sentence_preserved': config.preserve_needle and needle_sentence_idx != -1
        }
        
        return shuffled_haystack, metadata
    
    def _shuffle_paragraphs(self, haystack: str, needle: str, 
                           config: ShuffleConfig) -> Tuple[str, Dict[str, Any]]:
        """Shuffle paragraphs within the haystack"""
        # Split into paragraphs
        paragraphs = [p.strip() for p in haystack.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            # Fallback: split by double spaces or long sentences
            paragraphs = [p.strip() for p in re.split(r'\.\s+', haystack) if p.strip()]
        
        # Find paragraph containing needle
        needle_paragraph_idx = -1
        for i, paragraph in enumerate(paragraphs):
            if needle in paragraph:
                needle_paragraph_idx = i
                break
        
        # Determine paragraphs to shuffle
        num_to_shuffle = max(1, int(len(paragraphs) * config.shuffle_ratio))
        
        if config.preserve_needle and needle_paragraph_idx != -1:
            shuffle_indices = [i for i in range(len(paragraphs)) if i != needle_paragraph_idx]
            shuffle_indices = random.sample(shuffle_indices, min(num_to_shuffle, len(shuffle_indices)))
        else:
            shuffle_indices = random.sample(range(len(paragraphs)), min(num_to_shuffle, len(paragraphs)))
        
        # Shuffle selected paragraphs
        shuffled_paragraphs = paragraphs.copy()
        paragraphs_to_shuffle = [shuffled_paragraphs[i] for i in shuffle_indices]
        random.shuffle(paragraphs_to_shuffle)
        
        for i, idx in enumerate(shuffle_indices):
            shuffled_paragraphs[idx] = paragraphs_to_shuffle[i]
        
        shuffled_haystack = "\n\n".join(shuffled_paragraphs)
        
        metadata = {
            'strategy': 'paragraph_shuffle',
            'total_paragraphs': len(paragraphs),
            'paragraphs_shuffled': len(shuffle_indices),
            'needle_paragraph_preserved': config.preserve_needle and needle_paragraph_idx != -1
        }
        
        return shuffled_haystack, metadata
    
    def _shuffle_words(self, haystack: str, needle: str, 
                      config: ShuffleConfig) -> Tuple[str, Dict[str, Any]]:
        """Shuffle words within sentences (preserving sentence boundaries)"""
        delimiters = self.sentence_delimiters.get(config.language, ['.', '!', '?'])
        
        # Split into sentences
        sentences = []
        current_sentence = ""
        
        for char in haystack:
            current_sentence += char
            if char in delimiters:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Shuffle words within each sentence
        shuffled_sentences = []
        total_words_shuffled = 0
        
        for sentence in sentences:
            words = sentence.split()
            
            # Preserve needle words if required
            if config.preserve_needle and needle in sentence:
                needle_words = needle.split()
                # Simple preservation: don't shuffle if sentence contains needle
                shuffled_sentences.append(sentence)
            else:
                # Shuffle words in this sentence
                num_to_shuffle = max(1, int(len(words) * config.shuffle_ratio))
                if len(words) > 1 and num_to_shuffle > 0:
                    # Keep punctuation with last word
                    last_word = words[-1]
                    words_to_shuffle = words[:-1]
                    
                    if len(words_to_shuffle) >= num_to_shuffle:
                        shuffle_indices = random.sample(range(len(words_to_shuffle)), num_to_shuffle)
                        words_subset = [words_to_shuffle[i] for i in shuffle_indices]
                        random.shuffle(words_subset)
                        
                        for i, idx in enumerate(shuffle_indices):
                            words_to_shuffle[idx] = words_subset[i]
                        
                        total_words_shuffled += num_to_shuffle
                    
                    shuffled_sentence = " ".join(words_to_shuffle + [last_word])
                    shuffled_sentences.append(shuffled_sentence)
                else:
                    shuffled_sentences.append(sentence)
        
        shuffled_haystack = " ".join(shuffled_sentences)
        
        metadata = {
            'strategy': 'word_shuffle',
            'total_sentences': len(sentences),
            'total_words_shuffled': total_words_shuffled
        }
        
        return shuffled_haystack, metadata
    
    def _random_insertion(self, haystack: str, needle: str, 
                         config: ShuffleConfig) -> Tuple[str, Dict[str, Any]]:
        """Randomly insert filler content throughout haystack"""
        # Generate filler content based on language
        fillers = self._generate_filler_content(config.language)
        
        # Split haystack into segments
        words = haystack.split()
        num_insertions = max(1, int(len(words) * config.shuffle_ratio))
        
        # Determine insertion positions
        insertion_positions = sorted(random.sample(range(len(words)), 
                                                  min(num_insertions, len(words))))
        
        # Insert fillers
        modified_words = words.copy()
        insertions_made = 0
        
        for pos in reversed(insertion_positions):  # Reverse to maintain positions
            filler = random.choice(fillers)
            modified_words.insert(pos, filler)
            insertions_made += 1
        
        shuffled_haystack = " ".join(modified_words)
        
        metadata = {
            'strategy': 'random_insertion',
            'insertions_made': insertions_made,
            'filler_content_added': True
        }
        
        return shuffled_haystack, metadata
    
    def _reverse_order(self, haystack: str, needle: str, 
                      config: ShuffleConfig) -> Tuple[str, Dict[str, Any]]:
        """Reverse order of sentences or paragraphs"""
        # Split into sentences
        delimiters = self.sentence_delimiters.get(config.language, ['.', '!', '?'])
        sentences = []
        current_sentence = ""
        
        for char in haystack:
            current_sentence += char
            if char in delimiters:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Determine how many sentences to reverse
        num_to_reverse = int(len(sentences) * config.shuffle_ratio)
        
        if config.preserve_needle:
            # Find needle sentence and preserve its relative position
            needle_sentence_idx = -1
            for i, sentence in enumerate(sentences):
                if needle in sentence:
                    needle_sentence_idx = i
                    break
            
            if needle_sentence_idx != -1:
                # Reverse sections before and after needle separately
                before_needle = sentences[:needle_sentence_idx]
                needle_sentence = [sentences[needle_sentence_idx]]
                after_needle = sentences[needle_sentence_idx + 1:]
                
                if len(before_needle) > 1:
                    before_needle.reverse()
                if len(after_needle) > 1:
                    after_needle.reverse()
                
                shuffled_sentences = before_needle + needle_sentence + after_needle
            else:
                shuffled_sentences = sentences[::-1]
        else:
            # Simple reversal
            shuffled_sentences = sentences[::-1]
        
        shuffled_haystack = " ".join(shuffled_sentences)
        
        metadata = {
            'strategy': 'reverse_order',
            'total_sentences': len(sentences),
            'order_reversed': True
        }
        
        return shuffled_haystack, metadata
    
    def _topic_scramble(self, haystack: str, needle: str, 
                       config: ShuffleConfig) -> Tuple[str, Dict[str, Any]]:
        """Scramble content by topic/theme"""
        # Get topic keywords for language
        topics = self.topic_keywords.get(config.language, self.topic_keywords['en'])
        
        # Split into sentences
        sentences = self._split_into_sentences(haystack, config.language)
        
        # Classify sentences by topic
        sentence_topics = {}
        unclassified_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            best_topic = None
            max_matches = 0
            
            for topic, keywords in topics.items():
                matches = sum(1 for keyword in keywords if keyword in sentence_lower)
                if matches > max_matches:
                    max_matches = matches
                    best_topic = topic
            
            if best_topic and max_matches > 0:
                if best_topic not in sentence_topics:
                    sentence_topics[best_topic] = []
                sentence_topics[best_topic].append((i, sentence))
            else:
                unclassified_sentences.append((i, sentence))
        
        # Scramble within each topic group
        scrambled_sentences = [None] * len(sentences)
        
        for topic, topic_sentences in sentence_topics.items():
            indices, texts = zip(*topic_sentences)
            scrambled_texts = list(texts)
            random.shuffle(scrambled_texts)
            
            for idx, scrambled_text in zip(indices, scrambled_texts):
                scrambled_sentences[idx] = scrambled_text
        
        # Place unclassified sentences
        for idx, sentence in unclassified_sentences:
            scrambled_sentences[idx] = sentence
        
        shuffled_haystack = " ".join(filter(None, scrambled_sentences))
        
        metadata = {
            'strategy': 'topic_scramble',
            'topics_found': list(sentence_topics.keys()),
            'sentences_by_topic': {topic: len(sents) for topic, sents in sentence_topics.items()},
            'unclassified_sentences': len(unclassified_sentences)
        }
        
        return shuffled_haystack, metadata
    
    def _temporal_disorder(self, haystack: str, needle: str, 
                          config: ShuffleConfig) -> Tuple[str, Dict[str, Any]]:
        """Disorder temporal sequence of events"""
        # Find temporal markers (years, dates, temporal words)
        temporal_patterns = [
            r'\b(19|20)\d{2}\b',  # Years
            r'\b(first|second|third|then|next|finally|before|after|during)\b',  # Sequence words
            r'\b(yesterday|today|tomorrow|now|later|earlier)\b'  # Time references
        ]
        
        sentences = self._split_into_sentences(haystack, config.language)
        
        # Identify sentences with temporal markers
        temporal_sentences = []
        non_temporal_sentences = []
        
        for i, sentence in enumerate(sentences):
            has_temporal = False
            for pattern in temporal_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    has_temporal = True
                    break
            
            if has_temporal:
                temporal_sentences.append((i, sentence))
            else:
                non_temporal_sentences.append((i, sentence))
        
        # Shuffle temporal sentences to create disorder
        if temporal_sentences:
            indices, texts = zip(*temporal_sentences)
            shuffled_texts = list(texts)
            random.shuffle(shuffled_texts)
            
            # Reconstruct haystack
            result_sentences = [None] * len(sentences)
            
            # Place shuffled temporal sentences
            for idx, shuffled_text in zip(indices, shuffled_texts):
                result_sentences[idx] = shuffled_text
            
            # Place non-temporal sentences
            for idx, sentence in non_temporal_sentences:
                result_sentences[idx] = sentence
            
            shuffled_haystack = " ".join(filter(None, result_sentences))
        else:
            # No temporal markers found, fallback to sentence shuffle
            shuffled_haystack, _ = self._shuffle_sentences(haystack, needle, config)
        
        metadata = {
            'strategy': 'temporal_disorder',
            'temporal_sentences_found': len(temporal_sentences),
            'non_temporal_sentences': len(non_temporal_sentences),
            'temporal_disorder_applied': len(temporal_sentences) > 0
        }
        
        return shuffled_haystack, metadata
    
    def _semantic_clustering(self, haystack: str, needle: str, 
                           config: ShuffleConfig) -> Tuple[str, Dict[str, Any]]:
        """Cluster semantically similar content together"""
        sentences = self._split_into_sentences(haystack, config.language)
        
        # Simple semantic clustering based on word overlap
        clusters = []
        unclustered = list(enumerate(sentences))
        
        while unclustered:
            # Start new cluster with first unclustered sentence
            seed_idx, seed_sentence = unclustered.pop(0)
            current_cluster = [(seed_idx, seed_sentence)]
            seed_words = set(seed_sentence.lower().split())
            
            # Find similar sentences
            remaining = []
            for idx, sentence in unclustered:
                sentence_words = set(sentence.lower().split())
                overlap = len(seed_words & sentence_words)
                similarity = overlap / len(seed_words | sentence_words) if seed_words | sentence_words else 0
                
                if similarity > 0.2:  # Threshold for clustering
                    current_cluster.append((idx, sentence))
                else:
                    remaining.append((idx, sentence))
            
            unclustered = remaining
            clusters.append(current_cluster)
        
        # Reconstruct haystack with clustered sentences
        clustered_sentences = []
        for cluster in clusters:
            # Sort cluster by original position to maintain some order
            cluster.sort(key=lambda x: x[0])
            clustered_sentences.extend([sentence for _, sentence in cluster])
        
        shuffled_haystack = " ".join(clustered_sentences)
        
        metadata = {
            'strategy': 'semantic_clustering',
            'clusters_formed': len(clusters),
            'cluster_sizes': [len(cluster) for cluster in clusters],
            'average_cluster_size': np.mean([len(cluster) for cluster in clusters]) if clusters else 0
        }
        
        return shuffled_haystack, metadata
    
    def _split_into_sentences(self, text: str, language: str) -> List[str]:
        """Split text into sentences based on language"""
        delimiters = self.sentence_delimiters.get(language, ['.', '!', '?'])
        
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in delimiters:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if s]  # Filter empty sentences
    
    def _generate_filler_content(self, language: str) -> List[str]:
        """Generate filler content for random insertion"""
        fillers = {
            'en': [
                "Additionally,", "Furthermore,", "Moreover,", "However,", "Nevertheless,",
                "In contrast,", "Similarly,", "For example,", "In particular,", "Specifically,"
            ],
            'id': [
                "Selain itu,", "Lebih lanjut,", "Namun,", "Akan tetapi,", "Sebaliknya,",
                "Misalnya,", "Khususnya,", "Secara khusus,", "Di sisi lain,", "Dengan demikian,"
            ]
        }
        
        return fillers.get(language, fillers['en'])
    
    def _calculate_structure_change(self, original: str, shuffled: str) -> float:
        """Calculate how much the structure has changed"""
        # Simple metric based on word position changes
        original_words = original.split()
        shuffled_words = shuffled.split()
        
        if len(original_words) != len(shuffled_words):
            # Length changed significantly
            return 1.0
        
        # Calculate position changes
        position_changes = 0
        for i, word in enumerate(original_words):
            if i < len(shuffled_words) and shuffled_words[i] != word:
                position_changes += 1
        
        return position_changes / len(original_words) if original_words else 0.0
    
    def _calculate_coherence_score(self, text: str, language: str) -> float:
        """Calculate text coherence score (simple heuristic)"""
        sentences = self._split_into_sentences(text, language)
        
        if len(sentences) <= 1:
            return 1.0
        
        # Simple coherence based on word overlap between adjacent sentences
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            
            if words1 and words2:
                overlap = len(words1 & words2)
                union = len(words1 | words2)
                coherence = overlap / union if union > 0 else 0
                coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_readability_score(self, text: str, language: str) -> float:
        """Calculate readability score (simple heuristic)"""
        sentences = self._split_into_sentences(text, language)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Simple readability based on sentence length and word length
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(word) for word in words])
        
        # Normalize scores (lower is more readable)
        sentence_score = min(1.0, avg_sentence_length / 20.0)  # 20 words per sentence is baseline
        word_score = min(1.0, avg_word_length / 6.0)  # 6 characters per word is baseline
        
        readability = 1.0 - (sentence_score + word_score) / 2.0
        return max(0.0, readability)
    
    def evaluate_structure_sensitivity(self, agent, original_response: str, 
                                     shuffled_response: str, 
                                     result: StructureSensitivityResult) -> Dict[str, float]:
        """
        Evaluate agent sensitivity to structure changes
        
        Args:
            agent: The agent being tested
            original_response: Response to original haystack
            shuffled_response: Response to shuffled haystack
            result: Structure sensitivity result
            
        Returns:
            Dictionary with sensitivity metrics
        """
        # Response similarity
        original_words = set(original_response.lower().split())
        shuffled_words = set(shuffled_response.lower().split())
        
        if original_words:
            response_similarity = len(original_words & shuffled_words) / len(original_words)
        else:
            response_similarity = 0.0
        
        # Needle preservation in response
        needle_words = set(result.needle_text.lower().split())
        original_needle_overlap = len(original_words & needle_words) / len(needle_words) if needle_words else 0
        shuffled_needle_overlap = len(shuffled_words & needle_words) / len(needle_words) if needle_words else 0
        
        needle_preservation = shuffled_needle_overlap / original_needle_overlap if original_needle_overlap > 0 else 0
        
        # Structure sensitivity (lower = more sensitive to structure changes)
        structure_sensitivity = 1.0 - response_similarity
        
        # Robustness score (higher = more robust to structure changes)
        robustness_score = response_similarity * needle_preservation
        
        return {
            'response_similarity': response_similarity,
            'needle_preservation': needle_preservation,
            'structure_sensitivity': structure_sensitivity,
            'robustness_score': robustness_score,
            'structure_change_score': result.structure_change_score,
            'coherence_impact': result.coherence_score,
            'readability_impact': result.readability_score
        }
    
    def analyze_sensitivity_patterns(self, results: Optional[List[StructureSensitivityResult]] = None) -> Dict[str, Any]:
        """
        Analyze structure sensitivity patterns across results
        
        Args:
            results: Results to analyze (uses stored results if None)
            
        Returns:
            Analysis report
        """
        if results is None:
            results = self.sensitivity_results
        
        if not results:
            return {"error": "No results available for analysis"}
        
        # Convert to DataFrame for analysis
        df_data = []
        for result in results:
            df_data.append({
                'strategy': result.shuffle_config.strategy.value,
                'language': result.shuffle_config.language,
                'shuffle_ratio': result.shuffle_config.shuffle_ratio,
                'preserve_needle': result.shuffle_config.preserve_needle,
                'structure_change_score': result.structure_change_score,
                'coherence_score': result.coherence_score,
                'readability_score': result.readability_score,
                'needle_position_change': abs(result.shuffled_needle_position - result.original_needle_position)
            })
        
        df = pd.DataFrame(df_data)
        
        analysis = {
            'total_tests': len(results),
            'strategies_tested': list(df['strategy'].unique()),
            'languages_tested': list(df['language'].unique()),
            'average_metrics': {
                'structure_change_score': float(df['structure_change_score'].mean()),
                'coherence_score': float(df['coherence_score'].mean()),
                'readability_score': float(df['readability_score'].mean())
            },
            'strategy_performance': {},
            'language_performance': {},
            'shuffle_ratio_impact': {}
        }
        
        # Strategy-specific analysis
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy]
            analysis['strategy_performance'][strategy] = {
                'count': len(strategy_df),
                'avg_structure_change': float(strategy_df['structure_change_score'].mean()),
                'avg_coherence': float(strategy_df['coherence_score'].mean()),
                'avg_readability': float(strategy_df['readability_score'].mean())
            }
        
        # Language-specific analysis
        for language in df['language'].unique():
            lang_df = df[df['language'] == language]
            analysis['language_performance'][language] = {
                'count': len(lang_df),
                'avg_structure_change': float(lang_df['structure_change_score'].mean()),
                'avg_coherence': float(lang_df['coherence_score'].mean())
            }
        
        # Shuffle ratio impact
        for ratio in df['shuffle_ratio'].unique():
            ratio_df = df[df['shuffle_ratio'] == ratio]
            analysis['shuffle_ratio_impact'][str(ratio)] = {
                'count': len(ratio_df),
                'avg_structure_change': float(ratio_df['structure_change_score'].mean())
            }
        
        return analysis
    
    def export_sensitivity_results(self, filepath: str, 
                                 results: Optional[List[StructureSensitivityResult]] = None):
        """
        Export sensitivity results to file
        
        Args:
            filepath: Output file path
            results: Results to export (uses stored results if None)
        """
        if results is None:
            results = self.sensitivity_results
        
        export_data = []
        for result in results:
            export_data.append({
                'original_haystack_length': len(result.original_haystack),
                'shuffled_haystack_length': len(result.shuffled_haystack),
                'needle_text': result.needle_text,
                'original_needle_position': result.original_needle_position,
                'shuffled_needle_position': result.shuffled_needle_position,
                'shuffle_strategy': result.shuffle_config.strategy.value,
                'shuffle_ratio': result.shuffle_config.shuffle_ratio,
                'preserve_needle': result.shuffle_config.preserve_needle,
                'language': result.shuffle_config.language,
                'structure_change_score': result.structure_change_score,
                'coherence_score': result.coherence_score,
                'readability_score': result.readability_score,
                'shuffle_metadata': result.shuffle_metadata
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Sensitivity results exported to {filepath}")