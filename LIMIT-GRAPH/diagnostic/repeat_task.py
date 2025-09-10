# -*- coding: utf-8 -*-
"""
Repeated Words Replication Task Module
Tests agent ability to handle repeated content and maintain focus
on the actual needle through various repetition strategies
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

class RepetitionStrategy(Enum):
    """Types of repetition strategies"""
    EXACT_REPETITION = "exact_repetition"
    PARAPHRASED_REPETITION = "paraphrased_repetition"
    KEYWORD_REPETITION = "keyword_repetition"
    PATTERN_REPETITION = "pattern_repetition"
    NOISE_REPETITION = "noise_repetition"
    INCREMENTAL_REPETITION = "incremental_repetition"
    CONTEXTUAL_REPETITION = "contextual_repetition"
    MISLEADING_REPETITION = "misleading_repetition"

@dataclass
class RepetitionConfig:
    """Configuration for repetition task"""
    strategy: RepetitionStrategy
    repetition_count: int  # Number of repetitions to add
    repetition_ratio: float  # Ratio of repetitions to original content
    distribution: str  # "uniform", "clustered", "random"
    language: str
    preserve_needle_uniqueness: bool = True
    seed: int = 42

@dataclass
class RepetitionTaskResult:
    """Result of repetition task"""
    original_haystack: str
    repeated_haystack: str
    needle_text: str
    original_needle_position: int
    needle_positions_in_repeated: List[int]
    repetition_config: RepetitionConfig
    repetition_density: float
    content_inflation_ratio: float
    uniqueness_preservation_score: float
    repetition_metadata: Dict[str, Any]

class RepeatedWordsReplicationTask:
    """
    Tests agent ability to handle repeated content and maintain
    focus on the actual needle information
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize repeated words replication task
        
        Args:
            seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        random.seed(seed)
        np.random.seed(seed)
        
        self.repetition_results = []
        
        # Paraphrasing templates by language
        self.paraphrase_templates = {
            'en': {
                'synonyms': {
                    'said': ['stated', 'mentioned', 'declared', 'expressed', 'noted'],
                    'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
                    'small': ['tiny', 'little', 'minute', 'compact', 'petite'],
                    'good': ['excellent', 'great', 'wonderful', 'fantastic', 'superb'],
                    'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'poor'],
                    'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
                    'slow': ['sluggish', 'gradual', 'leisurely', 'unhurried', 'delayed']
                },
                'sentence_patterns': [
                    "It is worth noting that {content}",
                    "One should remember that {content}",
                    "It is important to understand that {content}",
                    "We must consider that {content}",
                    "It should be emphasized that {content}"
                ]
            },
            'id': {
                'synonyms': {
                    'besar': ['raksasa', 'besar sekali', 'sangat besar', 'luas', 'agung'],
                    'kecil': ['kecil sekali', 'mungil', 'mini', 'imut', 'cilik'],
                    'baik': ['bagus', 'hebat', 'luar biasa', 'istimewa', 'cemerlang'],
                    'buruk': ['jelek', 'tidak bagus', 'mengecewakan', 'payah', 'kurang'],
                    'cepat': ['kilat', 'gesit', 'lincah', 'sigap', 'tangkas'],
                    'lambat': ['pelan', 'santai', 'perlahan', 'tidak tergesa', 'tenang']
                },
                'sentence_patterns': [
                    "Perlu dicatat bahwa {content}",
                    "Kita harus ingat bahwa {content}",
                    "Penting untuk memahami bahwa {content}",
                    "Kita harus mempertimbangkan bahwa {content}",
                    "Harus ditekankan bahwa {content}"
                ]
            },
            'ar': {
                'synonyms': {
                    'كبير': ['ضخم', 'عظيم', 'هائل', 'جسيم', 'عملاق'],
                    'صغير': ['ضئيل', 'قليل', 'محدود', 'مصغر', 'طفيف'],
                    'جيد': ['ممتاز', 'رائع', 'عظيم', 'مذهل', 'فائق'],
                    'سيء': ['فظيع', 'مروع', 'بشع', 'مخيف', 'ضعيف'],
                    'سريع': ['عاجل', 'مستعجل', 'خاطف', 'مسرع', 'فوري'],
                    'بطيء': ['متأن', 'هادئ', 'متمهل', 'غير مستعجل', 'مؤجل'],
                    'قال': ['ذكر', 'صرح', 'أعلن', 'عبر', 'لاحظ']
                },
                'sentence_patterns': [
                    "من الجدير بالذكر أن {content}",
                    "يجب أن نتذكر أن {content}",
                    "من المهم أن نفهم أن {content}",
                    "يجب أن نعتبر أن {content}",
                    "ينبغي التأكيد على أن {content}"
                ]
                },
                'sentence_patterns': [
                    "Perlu diingat bahwa {content}",
                    "Penting untuk memahami bahwa {content}",
                    "Harus dipertimbangkan bahwa {content}",
                    "Patut dicatat bahwa {content}",
                    "Perlu ditekankan bahwa {content}"
                ]
            }
        }
        
        # Noise content templates
        self.noise_templates = {
            'en': [
                "This is additional information that may not be relevant.",
                "Some people might find this interesting to know.",
                "It has been observed in various studies and research.",
                "According to multiple sources and references available.",
                "This information comes from reliable and trusted sources.",
                "Many experts in the field have confirmed this fact.",
                "Research has shown consistent results in this area.",
                "Data analysis reveals interesting patterns and trends."
            ],
            'id': [
                "Ini adalah informasi tambahan yang mungkin tidak relevan.",
                "Beberapa orang mungkin menganggap ini menarik untuk diketahui.",
                "Telah diamati dalam berbagai studi dan penelitian.",
                "Menurut berbagai sumber dan referensi yang tersedia.",
                "Informasi ini berasal dari sumber yang dapat dipercaya.",
                "Banyak ahli di bidang ini telah mengkonfirmasi fakta ini.",
                "Penelitian telah menunjukkan hasil yang konsisten.",
                "Analisis data mengungkapkan pola dan tren yang menarik."
            ]
        }
    
    def apply_repetition_strategy(self, haystack: str, needle: str, 
                                config: RepetitionConfig) -> RepetitionTaskResult:
        """
        Apply repetition strategy to haystack
        
        Args:
            haystack: Original haystack text
            needle: Needle text to track
            config: Repetition configuration
            
        Returns:
            RepetitionTaskResult with repeated content and analysis
        """
        # Find original needle position
        original_needle_position = haystack.find(needle)
        if original_needle_position == -1:
            self.logger.warning("Needle not found in haystack")
            original_needle_position = len(haystack) // 2
        
        # Apply repetition strategy
        if config.strategy == RepetitionStrategy.EXACT_REPETITION:
            repeated_haystack, metadata = self._exact_repetition(haystack, needle, config)
        elif config.strategy == RepetitionStrategy.PARAPHRASED_REPETITION:
            repeated_haystack, metadata = self._paraphrased_repetition(haystack, needle, config)
        elif config.strategy == RepetitionStrategy.KEYWORD_REPETITION:
            repeated_haystack, metadata = self._keyword_repetition(haystack, needle, config)
        elif config.strategy == RepetitionStrategy.PATTERN_REPETITION:
            repeated_haystack, metadata = self._pattern_repetition(haystack, needle, config)
        elif config.strategy == RepetitionStrategy.NOISE_REPETITION:
            repeated_haystack, metadata = self._noise_repetition(haystack, needle, config)
        elif config.strategy == RepetitionStrategy.INCREMENTAL_REPETITION:
            repeated_haystack, metadata = self._incremental_repetition(haystack, needle, config)
        elif config.strategy == RepetitionStrategy.CONTEXTUAL_REPETITION:
            repeated_haystack, metadata = self._contextual_repetition(haystack, needle, config)
        elif config.strategy == RepetitionStrategy.MISLEADING_REPETITION:
            repeated_haystack, metadata = self._misleading_repetition(haystack, needle, config)
        else:
            # Default: no repetition
            repeated_haystack = haystack
            metadata = {'strategy': 'none', 'repetitions_added': 0}
        
        # Find all needle positions in repeated haystack
        needle_positions = []
        start = 0
        while True:
            pos = repeated_haystack.find(needle, start)
            if pos == -1:
                break
            needle_positions.append(pos)
            start = pos + 1
        
        # Calculate metrics
        repetition_density = self._calculate_repetition_density(haystack, repeated_haystack)
        content_inflation_ratio = len(repeated_haystack) / len(haystack) if haystack else 1.0
        uniqueness_score = self._calculate_uniqueness_preservation(haystack, repeated_haystack, needle)
        
        result = RepetitionTaskResult(
            original_haystack=haystack,
            repeated_haystack=repeated_haystack,
            needle_text=needle,
            original_needle_position=original_needle_position,
            needle_positions_in_repeated=needle_positions,
            repetition_config=config,
            repetition_density=repetition_density,
            content_inflation_ratio=content_inflation_ratio,
            uniqueness_preservation_score=uniqueness_score,
            repetition_metadata=metadata
        )
        
        self.repetition_results.append(result)
        return result
    
    def _exact_repetition(self, haystack: str, needle: str, 
                         config: RepetitionConfig) -> Tuple[str, Dict[str, Any]]:
        """Add exact repetitions of content segments"""
        # Split haystack into segments
        segments = self._split_into_segments(haystack, config.language)
        
        # Select segments to repeat (excluding needle if preserve_needle_uniqueness)
        segments_to_repeat = []
        needle_segment_idx = -1
        
        for i, segment in enumerate(segments):
            if needle in segment:
                needle_segment_idx = i
                if not config.preserve_needle_uniqueness:
                    segments_to_repeat.append((i, segment))
            else:
                segments_to_repeat.append((i, segment))
        
        # Limit number of segments to repeat
        num_to_repeat = min(config.repetition_count, len(segments_to_repeat))
        selected_segments = random.sample(segments_to_repeat, num_to_repeat)
        
        # Generate repetitions
        repetitions = []
        for _, segment in selected_segments:
            for _ in range(random.randint(1, 3)):  # 1-3 repetitions per segment
                repetitions.append(segment)
        
        # Insert repetitions based on distribution strategy
        repeated_haystack = self._insert_repetitions(haystack, repetitions, config.distribution)
        
        metadata = {
            'strategy': 'exact_repetition',
            'segments_repeated': len(selected_segments),
            'total_repetitions_added': len(repetitions),
            'needle_segment_preserved': config.preserve_needle_uniqueness and needle_segment_idx != -1
        }
        
        return repeated_haystack, metadata
    
    def _paraphrased_repetition(self, haystack: str, needle: str, 
                              config: RepetitionConfig) -> Tuple[str, Dict[str, Any]]:
        """Add paraphrased versions of content"""
        segments = self._split_into_segments(haystack, config.language)
        
        # Generate paraphrases
        paraphrases = []
        paraphrased_count = 0
        
        for segment in segments:
            if needle in segment and config.preserve_needle_uniqueness:
                continue
            
            if paraphrased_count >= config.repetition_count:
                break
            
            paraphrase = self._generate_paraphrase(segment, config.language)
            if paraphrase != segment:  # Only add if actually different
                paraphrases.append(paraphrase)
                paraphrased_count += 1
        
        # Insert paraphrases
        repeated_haystack = self._insert_repetitions(haystack, paraphrases, config.distribution)
        
        metadata = {
            'strategy': 'paraphrased_repetition',
            'paraphrases_generated': len(paraphrases),
            'successful_paraphrases': paraphrased_count
        }
        
        return repeated_haystack, metadata
    
    def _keyword_repetition(self, haystack: str, needle: str, 
                          config: RepetitionConfig) -> Tuple[str, Dict[str, Any]]:
        """Repeat key words and phrases throughout the text"""
        # Extract keywords from haystack
        keywords = self._extract_keywords(haystack, config.language)
        
        # Generate keyword-based repetitions
        keyword_repetitions = []
        for _ in range(config.repetition_count):
            if keywords:
                keyword = random.choice(keywords)
                # Create sentences with repeated keywords
                repetition = self._create_keyword_sentence(keyword, config.language)
                keyword_repetitions.append(repetition)
        
        # Insert keyword repetitions
        repeated_haystack = self._insert_repetitions(haystack, keyword_repetitions, config.distribution)
        
        metadata = {
            'strategy': 'keyword_repetition',
            'keywords_extracted': len(keywords),
            'keyword_repetitions_added': len(keyword_repetitions),
            'top_keywords': keywords[:5]
        }
        
        return repeated_haystack, metadata
    
    def _pattern_repetition(self, haystack: str, needle: str, 
                          config: RepetitionConfig) -> Tuple[str, Dict[str, Any]]:
        """Repeat syntactic patterns with different content"""
        # Extract sentence patterns
        patterns = self._extract_sentence_patterns(haystack, config.language)
        
        # Generate pattern-based repetitions
        pattern_repetitions = []
        for _ in range(config.repetition_count):
            if patterns:
                pattern = random.choice(patterns)
                repetition = self._generate_pattern_instance(pattern, config.language)
                pattern_repetitions.append(repetition)
        
        # Insert pattern repetitions
        repeated_haystack = self._insert_repetitions(haystack, pattern_repetitions, config.distribution)
        
        metadata = {
            'strategy': 'pattern_repetition',
            'patterns_extracted': len(patterns),
            'pattern_repetitions_added': len(pattern_repetitions)
        }
        
        return repeated_haystack, metadata
    
    def _noise_repetition(self, haystack: str, needle: str, 
                        config: RepetitionConfig) -> Tuple[str, Dict[str, Any]]:
        """Add repetitive noise content"""
        noise_templates = self.noise_templates.get(config.language, self.noise_templates['en'])
        
        # Generate noise repetitions
        noise_repetitions = []
        for _ in range(config.repetition_count):
            noise = random.choice(noise_templates)
            # Add slight variations
            variations = [
                noise,
                noise.replace("This", "That"),
                noise.replace("information", "data"),
                noise.replace("various", "multiple"),
                noise.replace("studies", "research")
            ]
            noise_repetitions.append(random.choice(variations))
        
        # Insert noise repetitions
        repeated_haystack = self._insert_repetitions(haystack, noise_repetitions, config.distribution)
        
        metadata = {
            'strategy': 'noise_repetition',
            'noise_repetitions_added': len(noise_repetitions),
            'noise_templates_used': len(set(noise_repetitions))
        }
        
        return repeated_haystack, metadata
    
    def _incremental_repetition(self, haystack: str, needle: str, 
                              config: RepetitionConfig) -> Tuple[str, Dict[str, Any]]:
        """Add incremental repetitions that build up information"""
        segments = self._split_into_segments(haystack, config.language)
        
        # Create incremental versions
        incremental_repetitions = []
        for segment in segments[:config.repetition_count]:
            if needle in segment and config.preserve_needle_uniqueness:
                continue
            
            # Create partial versions of the segment
            words = segment.split()
            for i in range(2, len(words)):
                partial = " ".join(words[:i]) + "..."
                incremental_repetitions.append(partial)
        
        # Insert incremental repetitions
        repeated_haystack = self._insert_repetitions(haystack, incremental_repetitions, config.distribution)
        
        metadata = {
            'strategy': 'incremental_repetition',
            'incremental_repetitions_added': len(incremental_repetitions),
            'segments_processed': min(config.repetition_count, len(segments))
        }
        
        return repeated_haystack, metadata
    
    def _contextual_repetition(self, haystack: str, needle: str, 
                             config: RepetitionConfig) -> Tuple[str, Dict[str, Any]]:
        """Add contextually similar but different information"""
        # Extract context from needle
        needle_context = self._extract_context(needle, config.language)
        
        # Generate contextually similar content
        contextual_repetitions = []
        for _ in range(config.repetition_count):
            similar_content = self._generate_similar_context(needle_context, config.language)
            contextual_repetitions.append(similar_content)
        
        # Insert contextual repetitions
        repeated_haystack = self._insert_repetitions(haystack, contextual_repetitions, config.distribution)
        
        metadata = {
            'strategy': 'contextual_repetition',
            'contextual_repetitions_added': len(contextual_repetitions),
            'needle_context': needle_context
        }
        
        return repeated_haystack, metadata
    
    def _misleading_repetition(self, haystack: str, needle: str, 
                             config: RepetitionConfig) -> Tuple[str, Dict[str, Any]]:
        """Add misleading information that contradicts the needle"""
        # Generate misleading versions of the needle
        misleading_repetitions = []
        
        for _ in range(config.repetition_count):
            misleading = self._generate_misleading_version(needle, config.language)
            misleading_repetitions.append(misleading)
        
        # Insert misleading repetitions
        repeated_haystack = self._insert_repetitions(haystack, misleading_repetitions, config.distribution)
        
        metadata = {
            'strategy': 'misleading_repetition',
            'misleading_repetitions_added': len(misleading_repetitions),
            'original_needle_preserved': config.preserve_needle_uniqueness
        }
        
        return repeated_haystack, metadata
    
    def _split_into_segments(self, text: str, language: str) -> List[str]:
        """Split text into meaningful segments"""
        # Simple sentence-based segmentation
        delimiters = ['.', '!', '?']
        segments = []
        current_segment = ""
        
        for char in text:
            current_segment += char
            if char in delimiters:
                segments.append(current_segment.strip())
                current_segment = ""
        
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        return [s for s in segments if s]
    
    def _insert_repetitions(self, haystack: str, repetitions: List[str], 
                          distribution: str) -> str:
        """Insert repetitions based on distribution strategy"""
        if not repetitions:
            return haystack
        
        words = haystack.split()
        
        if distribution == "uniform":
            # Distribute evenly throughout the text
            positions = np.linspace(0, len(words), len(repetitions) + 1, dtype=int)[1:-1]
        elif distribution == "clustered":
            # Cluster repetitions in one area
            cluster_center = random.randint(len(words) // 4, 3 * len(words) // 4)
            cluster_size = min(len(words) // 4, len(repetitions) * 2)
            positions = [cluster_center + random.randint(-cluster_size//2, cluster_size//2) 
                        for _ in repetitions]
            positions = [max(0, min(pos, len(words))) for pos in positions]
        else:  # random
            positions = sorted(random.sample(range(len(words)), min(len(repetitions), len(words))))
        
        # Insert repetitions at positions (in reverse order to maintain positions)
        modified_words = words.copy()
        for pos, repetition in zip(reversed(positions), reversed(repetitions)):
            modified_words.insert(pos, repetition)
        
        return " ".join(modified_words)
    
    def _generate_paraphrase(self, text: str, language: str) -> str:
        """Generate paraphrase of text"""
        templates = self.paraphrase_templates.get(language, self.paraphrase_templates['en'])
        
        # Simple paraphrasing using synonyms and sentence patterns
        words = text.split()
        paraphrased_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?')
            if word_lower in templates['synonyms']:
                synonym = random.choice(templates['synonyms'][word_lower])
                paraphrased_words.append(synonym)
            else:
                paraphrased_words.append(word)
        
        paraphrased_text = " ".join(paraphrased_words)
        
        # Apply sentence pattern if available
        if 'sentence_patterns' in templates and random.random() < 0.3:
            pattern = random.choice(templates['sentence_patterns'])
            paraphrased_text = pattern.format(content=paraphrased_text.lower())
        
        return paraphrased_text
    
    def _extract_keywords(self, text: str, language: str) -> List[str]:
        """Extract important keywords from text"""
        words = text.lower().split()
        
        # Simple keyword extraction: words longer than 4 characters, not common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                     'yang', 'dan', 'atau', 'tetapi', 'di', 'ke', 'untuk', 'dari', 'dengan', 'oleh'}
        
        keywords = []
        for word in words:
            clean_word = word.strip('.,!?')
            if len(clean_word) > 4 and clean_word not in stop_words:
                keywords.append(clean_word)
        
        # Return most frequent keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(10)]
    
    def _create_keyword_sentence(self, keyword: str, language: str) -> str:
        """Create a sentence featuring the keyword"""
        templates = {
            'en': [
                f"The {keyword} is particularly important in this context.",
                f"Many researchers have studied {keyword} extensively.",
                f"It is worth noting that {keyword} plays a crucial role.",
                f"The significance of {keyword} cannot be overstated.",
                f"Recent developments in {keyword} have been remarkable."
            ],
            'id': [
                f"{keyword} sangat penting dalam konteks ini.",
                f"Banyak peneliti telah mempelajari {keyword} secara ekstensif.",
                f"Perlu dicatat bahwa {keyword} memainkan peran penting.",
                f"Signifikansi {keyword} tidak dapat diremehkan.",
                f"Perkembangan terbaru dalam {keyword} sangat luar biasa."
            ]
        }
        
        lang_templates = templates.get(language, templates['en'])
        return random.choice(lang_templates)
    
    def _extract_sentence_patterns(self, text: str, language: str) -> List[str]:
        """Extract syntactic patterns from sentences"""
        sentences = self._split_into_segments(text, language)
        patterns = []
        
        for sentence in sentences:
            # Simple pattern extraction based on structure
            if " is " in sentence.lower():
                patterns.append("X is Y")
            elif " was " in sentence.lower():
                patterns.append("X was Y")
            elif " has " in sentence.lower():
                patterns.append("X has Y")
            elif " can " in sentence.lower():
                patterns.append("X can Y")
        
        return list(set(patterns))  # Remove duplicates
    
    def _generate_pattern_instance(self, pattern: str, language: str) -> str:
        """Generate new instance of syntactic pattern"""
        # Simple pattern instantiation
        if pattern == "X is Y":
            subjects = ["The system", "This method", "The approach", "The technique"]
            predicates = ["effective", "reliable", "important", "useful"]
            return f"{random.choice(subjects)} is {random.choice(predicates)}."
        elif pattern == "X was Y":
            subjects = ["The experiment", "The study", "The research", "The analysis"]
            predicates = ["conducted", "performed", "completed", "executed"]
            return f"{random.choice(subjects)} was {random.choice(predicates)}."
        elif pattern == "X has Y":
            subjects = ["The model", "The framework", "The algorithm", "The process"]
            objects = ["advantages", "benefits", "features", "capabilities"]
            return f"{random.choice(subjects)} has many {random.choice(objects)}."
        else:
            return "This is a generated pattern instance."
    
    def _extract_context(self, needle: str, language: str) -> Dict[str, Any]:
        """Extract contextual information from needle"""
        words = needle.lower().split()
        
        # Simple context extraction
        context = {
            'entities': [word for word in words if word[0].isupper()],
            'numbers': re.findall(r'\d+', needle),
            'keywords': [word for word in words if len(word) > 4],
            'length': len(words)
        }
        
        return context
    
    def _generate_similar_context(self, context: Dict[str, Any], language: str) -> str:
        """Generate content similar to the context"""
        templates = {
            'en': [
                "Similar information can be found in related studies.",
                "Comparable results have been observed in other research.",
                "This aligns with findings from previous investigations.",
                "Analogous patterns appear in different contexts.",
                "Related work has shown similar trends and outcomes."
            ],
            'id': [
                "Informasi serupa dapat ditemukan dalam studi terkait.",
                "Hasil yang sebanding telah diamati dalam penelitian lain.",
                "Ini sejalan dengan temuan dari investigasi sebelumnya.",
                "Pola yang analog muncul dalam konteks yang berbeda.",
                "Karya terkait telah menunjukkan tren dan hasil yang serupa."
            ]
        }
        
        lang_templates = templates.get(language, templates['en'])
        return random.choice(lang_templates)
    
    def _generate_misleading_version(self, needle: str, language: str) -> str:
        """Generate misleading version of needle"""
        # Simple misleading generation by negation or contradiction
        misleading_prefixes = {
            'en': ["Contrary to popular belief,", "It is incorrect to assume that", 
                   "Despite common misconceptions,", "It is false that"],
            'id': ["Bertentangan dengan kepercayaan umum,", "Tidak benar untuk mengasumsikan bahwa",
                   "Meskipun ada kesalahpahaman umum,", "Adalah salah bahwa"]
        }
        
        prefixes = misleading_prefixes.get(language, misleading_prefixes['en'])
        prefix = random.choice(prefixes)
        
        # Create misleading version
        return f"{prefix} {needle.lower()}"
    
    def _calculate_repetition_density(self, original: str, repeated: str) -> float:
        """Calculate density of repetitions in text"""
        original_words = set(original.lower().split())
        repeated_words = repeated.lower().split()
        
        # Count repeated words
        repeated_count = 0
        for word in repeated_words:
            if word in original_words:
                repeated_count += 1
        
        return repeated_count / len(repeated_words) if repeated_words else 0.0
    
    def _calculate_uniqueness_preservation(self, original: str, repeated: str, needle: str) -> float:
        """Calculate how well needle uniqueness is preserved"""
        needle_count_original = original.count(needle)
        needle_count_repeated = repeated.count(needle)
        
        if needle_count_original == 0:
            return 1.0 if needle_count_repeated == 0 else 0.0
        
        # Uniqueness is preserved if needle appears the same number of times
        if needle_count_repeated == needle_count_original:
            return 1.0
        elif needle_count_repeated > needle_count_original:
            # Penalize additional occurrences
            return max(0.0, 1.0 - (needle_count_repeated - needle_count_original) * 0.2)
        else:
            # Penalize missing occurrences
            return needle_count_repeated / needle_count_original
    
    def evaluate_repetition_robustness(self, agent, original_response: str, 
                                     repeated_response: str, 
                                     result: RepetitionTaskResult) -> Dict[str, float]:
        """
        Evaluate agent robustness to repetitions
        
        Args:
            agent: The agent being tested
            original_response: Response to original haystack
            repeated_response: Response to repeated haystack
            result: Repetition task result
            
        Returns:
            Dictionary with robustness metrics
        """
        # Response consistency
        original_words = set(original_response.lower().split())
        repeated_words = set(repeated_response.lower().split())
        
        if original_words:
            response_consistency = len(original_words & repeated_words) / len(original_words)
        else:
            response_consistency = 0.0
        
        # Needle focus preservation
        needle_words = set(result.needle_text.lower().split())
        original_needle_overlap = len(original_words & needle_words) / len(needle_words) if needle_words else 0
        repeated_needle_overlap = len(repeated_words & needle_words) / len(needle_words) if needle_words else 0
        
        needle_focus_preservation = repeated_needle_overlap / original_needle_overlap if original_needle_overlap > 0 else 0
        
        # Distraction resistance (higher = more resistant to repetition distractions)
        distraction_resistance = response_consistency * needle_focus_preservation
        
        # Content inflation impact
        response_length_change = len(repeated_response) / len(original_response) if original_response else 1.0
        inflation_impact = abs(response_length_change - 1.0)  # Deviation from original length
        
        return {
            'response_consistency': response_consistency,
            'needle_focus_preservation': needle_focus_preservation,
            'distraction_resistance': distraction_resistance,
            'inflation_impact': inflation_impact,
            'repetition_density': result.repetition_density,
            'content_inflation_ratio': result.content_inflation_ratio,
            'uniqueness_preservation': result.uniqueness_preservation_score
        }
    
    def analyze_repetition_patterns(self, results: Optional[List[RepetitionTaskResult]] = None) -> Dict[str, Any]:
        """
        Analyze repetition patterns across results
        
        Args:
            results: Results to analyze (uses stored results if None)
            
        Returns:
            Analysis report
        """
        if results is None:
            results = self.repetition_results
        
        if not results:
            return {"error": "No results available for analysis"}
        
        # Convert to DataFrame for analysis
        df_data = []
        for result in results:
            df_data.append({
                'strategy': result.repetition_config.strategy.value,
                'language': result.repetition_config.language,
                'repetition_count': result.repetition_config.repetition_count,
                'repetition_ratio': result.repetition_config.repetition_ratio,
                'distribution': result.repetition_config.distribution,
                'repetition_density': result.repetition_density,
                'content_inflation_ratio': result.content_inflation_ratio,
                'uniqueness_preservation_score': result.uniqueness_preservation_score,
                'needle_occurrences': len(result.needle_positions_in_repeated)
            })
        
        df = pd.DataFrame(df_data)
        
        analysis = {
            'total_tests': len(results),
            'strategies_tested': list(df['strategy'].unique()),
            'languages_tested': list(df['language'].unique()),
            'average_metrics': {
                'repetition_density': float(df['repetition_density'].mean()),
                'content_inflation_ratio': float(df['content_inflation_ratio'].mean()),
                'uniqueness_preservation': float(df['uniqueness_preservation_score'].mean())
            },
            'strategy_performance': {},
            'language_performance': {},
            'distribution_impact': {}
        }
        
        # Strategy-specific analysis
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy]
            analysis['strategy_performance'][strategy] = {
                'count': len(strategy_df),
                'avg_repetition_density': float(strategy_df['repetition_density'].mean()),
                'avg_inflation_ratio': float(strategy_df['content_inflation_ratio'].mean()),
                'avg_uniqueness_preservation': float(strategy_df['uniqueness_preservation_score'].mean())
            }
        
        # Language-specific analysis
        for language in df['language'].unique():
            lang_df = df[df['language'] == language]
            analysis['language_performance'][language] = {
                'count': len(lang_df),
                'avg_repetition_density': float(lang_df['repetition_density'].mean()),
                'avg_inflation_ratio': float(lang_df['content_inflation_ratio'].mean())
            }
        
        # Distribution impact analysis
        for distribution in df['distribution'].unique():
            dist_df = df[df['distribution'] == distribution]
            analysis['distribution_impact'][distribution] = {
                'count': len(dist_df),
                'avg_repetition_density': float(dist_df['repetition_density'].mean())
            }
        
        return analysis
    
    def export_repetition_results(self, filepath: str, 
                                results: Optional[List[RepetitionTaskResult]] = None):
        """
        Export repetition results to file
        
        Args:
            filepath: Output file path
            results: Results to export (uses stored results if None)
        """
        if results is None:
            results = self.repetition_results
        
        export_data = []
        for result in results:
            export_data.append({
                'original_haystack_length': len(result.original_haystack),
                'repeated_haystack_length': len(result.repeated_haystack),
                'needle_text': result.needle_text,
                'original_needle_position': result.original_needle_position,
                'needle_positions_in_repeated': result.needle_positions_in_repeated,
                'repetition_strategy': result.repetition_config.strategy.value,
                'repetition_count': result.repetition_config.repetition_count,
                'repetition_ratio': result.repetition_config.repetition_ratio,
                'distribution': result.repetition_config.distribution,
                'language': result.repetition_config.language,
                'preserve_needle_uniqueness': result.repetition_config.preserve_needle_uniqueness,
                'repetition_density': result.repetition_density,
                'content_inflation_ratio': result.content_inflation_ratio,
                'uniqueness_preservation_score': result.uniqueness_preservation_score,
                'repetition_metadata': result.repetition_metadata
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Repetition results exported to {filepath}")