# -*- coding: utf-8 -*-
"""
Distractor Injection Framework
Injects various types of distractors to test agent robustness in long contexts
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

class DistractorType(Enum):
    """Types of distractors that can be injected"""
    SEMANTIC_SIMILAR = "semantic_similar"
    SYNTACTIC_SIMILAR = "syntactic_similar"
    KEYWORD_OVERLAP = "keyword_overlap"
    NUMERICAL_CONFUSION = "numerical_confusion"
    TEMPORAL_CONFUSION = "temporal_confusion"
    ENTITY_SUBSTITUTION = "entity_substitution"
    NEGATION_FLIP = "negation_flip"
    CONTEXTUAL_MISLEADING = "contextual_misleading"

@dataclass
class DistractorConfig:
    """Configuration for distractor injection"""
    distractor_type: DistractorType
    injection_ratio: float  # Ratio of distractors to inject
    similarity_threshold: float  # Minimum similarity for semantic distractors
    position_strategy: str  # "random", "before_needle", "after_needle", "surrounding"
    language: str
    
@dataclass
class DistractorInjectionResult:
    """Result of distractor injection"""
    original_haystack: str
    modified_haystack: str
    needle_position: int
    distractors_injected: List[Dict[str, Any]]
    distractor_positions: List[int]
    injection_config: DistractorConfig
    haystack_length_change: int

class DistractorInjectionFramework:
    """
    Framework for injecting various types of distractors into haystacks
    to test agent robustness and attention mechanisms
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize distractor injection framework
        
        Args:
            seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        random.seed(seed)
        np.random.seed(seed)
        
        # Distractor templates by language and type
        self.distractor_templates = self._load_distractor_templates()
        
    def _load_distractor_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load distractor templates for different languages and types"""
        return {
            'en': {
                'semantic_similar': [
                    "The capital of Germany is Berlin, which is known for its rich history.",
                    "London serves as the capital city of the United Kingdom.",
                    "Rome, the eternal city, is the capital of Italy.",
                    "Madrid is the political and cultural center of Spain.",
                    "Vienna is the beautiful capital of Austria."
                ],
                'syntactic_similar': [
                    "The president of the company announced new policies yesterday.",
                    "The director of the department submitted the final report.",
                    "The manager of the team organized a meeting for next week.",
                    "The leader of the group presented the quarterly results.",
                    "The head of the division approved the budget proposal."
                ],
                'keyword_overlap': [
                    "Einstein's theory of general relativity revolutionized physics.",
                    "Newton's theory of gravity explained planetary motion.",
                    "Darwin's theory of evolution changed biological understanding.",
                    "Freud's theory of psychoanalysis influenced psychology.",
                    "Marx's theory of capitalism shaped economic thought."
                ],
                'numerical_confusion': [
                    "The building has 42 floors and was completed in 1985.",
                    "The population reached 2.5 million people in 2020.",
                    "The temperature dropped to -15 degrees Celsius yesterday.",
                    "The company reported profits of $3.7 billion last quarter.",
                    "The distance between the cities is approximately 150 kilometers."
                ],
                'temporal_confusion': [
                    "In 1969, humans first landed on the moon during the Apollo mission.",
                    "The Berlin Wall fell in 1989, ending decades of division.",
                    "World War II ended in 1945 with the surrender of Germany.",
                    "The internet became publicly available in the early 1990s.",
                    "The first iPhone was released by Apple in 2007."
                ],
                'entity_substitution': [
                    "Shakespeare wrote many famous plays including Hamlet and Macbeth.",
                    "Mozart composed beautiful symphonies and operas in the 18th century.",
                    "Picasso created revolutionary artworks in the cubist style.",
                    "Gandhi led India's independence movement through non-violence.",
                    "Churchill served as Prime Minister during World War II."
                ],
                'negation_flip': [
                    "The experiment did not succeed due to equipment failure.",
                    "The proposal was not approved by the committee.",
                    "The weather is not suitable for outdoor activities today.",
                    "The medication should not be taken with alcohol.",
                    "The building is not accessible to wheelchair users."
                ],
                'contextual_misleading': [
                    "According to recent studies, this information may be outdated.",
                    "Some experts disagree with the commonly accepted view.",
                    "Alternative theories suggest a different interpretation.",
                    "Critics argue that the evidence is insufficient.",
                    "Further research is needed to confirm these findings."
                ]
            },
            'id': {
                'semantic_similar': [
                    "Ibu kota Jerman adalah Berlin yang terkenal dengan sejarahnya.",
                    "London menjadi ibu kota Kerajaan Inggris yang bersejarah.",
                    "Roma, kota abadi, adalah ibu kota Italia.",
                    "Madrid adalah pusat politik dan budaya Spanyol.",
                    "Wina adalah ibu kota Austria yang indah."
                ],
                'syntactic_similar': [
                    "Presiden perusahaan mengumumkan kebijakan baru kemarin.",
                    "Direktur departemen menyerahkan laporan akhir.",
                    "Manajer tim mengorganisir rapat untuk minggu depan.",
                    "Pemimpin kelompok mempresentasikan hasil kuartalan.",
                    "Kepala divisi menyetujui proposal anggaran."
                ],
                'keyword_overlap': [
                    "Teori relativitas umum Einstein merevolusi fisika.",
                    "Teori gravitasi Newton menjelaskan gerakan planet.",
                    "Teori evolusi Darwin mengubah pemahaman biologi.",
                    "Teori psikoanalisis Freud mempengaruhi psikologi.",
                    "Teori kapitalisme Marx membentuk pemikiran ekonomi."
                ],
                'numerical_confusion': [
                    "Gedung tersebut memiliki 42 lantai dan selesai pada 1985.",
                    "Populasi mencapai 2,5 juta orang pada tahun 2020.",
                    "Suhu turun hingga -15 derajat Celsius kemarin.",
                    "Perusahaan melaporkan keuntungan $3,7 miliar kuartal lalu.",
                    "Jarak antara kota-kota tersebut sekitar 150 kilometer."
                ],
                'temporal_confusion': [
                    "Pada 1969, manusia pertama kali mendarat di bulan.",
                    "Tembok Berlin runtuh pada 1989, mengakhiri perpecahan.",
                    "Perang Dunia II berakhir pada 1945 dengan menyerahnya Jerman.",
                    "Internet menjadi tersedia untuk publik pada awal 1990an.",
                    "iPhone pertama dirilis oleh Apple pada 2007."
                ],
                'entity_substitution': [
                    "Shakespeare menulis banyak drama terkenal termasuk Hamlet.",
                    "Mozart menggubah simfoni dan opera indah di abad ke-18.",
                    "Picasso menciptakan karya seni revolusioner gaya kubis.",
                    "Gandhi memimpin gerakan kemerdekaan India tanpa kekerasan.",
                    "Churchill menjabat Perdana Menteri selama Perang Dunia II."
                ],
                'negation_flip': [
                    "Eksperimen tidak berhasil karena kegagalan peralatan.",
                    "Proposal tidak disetujui oleh komite.",
                    "Cuaca tidak cocok untuk aktivitas luar ruangan hari ini.",
                    "Obat tidak boleh diminum bersamaan dengan alkohol.",
                    "Gedung tidak dapat diakses oleh pengguna kursi roda."
                ],
                'contextual_misleading': [
                    "Menurut studi terbaru, informasi ini mungkin sudah usang.",
                    "Beberapa ahli tidak setuju dengan pandangan yang diterima umum.",
                    "Teori alternatif menyarankan interpretasi yang berbeda.",
                    "Kritikus berargumen bahwa buktinya tidak cukup.",
                    "Penelitian lebih lanjut diperlukan untuk mengkonfirmasi temuan ini."
                ]
            },
            'ar': {
                'semantic_similar': [
                    "عاصمة ألمانيا هي برلين المشهورة بتاريخها العريق.",
                    "لندن هي عاصمة المملكة المتحدة التاريخية.",
                    "روما، المدينة الخالدة، هي عاصمة إيطاليا.",
                    "مدريد هي المركز السياسي والثقافي لإسبانيا.",
                    "فيينا هي عاصمة النمسا الجميلة."
                ],
                'syntactic_similar': [
                    "رئيس الشركة أعلن سياسات جديدة أمس.",
                    "مدير القسم قدم التقرير النهائي.",
                    "مدير الفريق نظم اجتماعاً للأسبوع القادم.",
                    "قائد المجموعة عرض النتائج الفصلية.",
                    "رئيس الشعبة وافق على اقتراح الميزانية."
                ],
                'keyword_overlap': [
                    "نظرية النسبية العامة لأينشتاين ثورت الفيزياء.",
                    "نظرية الجاذبية لنيوتن فسرت حركة الكواكب.",
                    "نظرية التطور لداروين غيرت الفهم البيولوجي.",
                    "نظرية التحليل النفسي لفرويد أثرت على علم النفس.",
                    "نظرية الرأسمالية لماركس شكلت الفكر الاقتصادي."
                ],
                'numerical_confusion': [
                    "المبنى يحتوي على 42 طابقاً وتم إنجازه في 1985.",
                    "وصل عدد السكان إلى 2.5 مليون شخص في 2020.",
                    "انخفضت الحرارة إلى -15 درجة مئوية أمس.",
                    "الشركة أعلنت أرباحاً بقيمة 3.7 مليار دولار الربع الماضي.",
                    "المسافة بين المدينتين حوالي 150 كيلومتراً."
                ],
                'temporal_confusion': [
                    "في عام 1969، هبط البشر لأول مرة على القمر خلال مهمة أبولو.",
                    "سقط جدار برلين في 1989، منهياً عقوداً من الانقسام.",
                    "انتهت الحرب العالمية الثانية في 1945 باستسلام ألمانيا.",
                    "أصبح الإنترنت متاحاً للجمهور في أوائل التسعينات.",
                    "أطلقت آبل أول آيفون في 2007."
                ],
                'entity_substitution': [
                    "شكسبير كتب مسرحيات مشهورة منها هاملت وماكبث.",
                    "موتسارت ألف سيمفونيات وأوبرات جميلة في القرن الثامن عشر.",
                    "بيكاسو أبدع أعمالاً فنية ثورية بالأسلوب التكعيبي.",
                    "غاندي قاد حركة استقلال الهند من خلال اللاعنف.",
                    "تشرشل خدم كرئيس وزراء خلال الحرب العالمية الثانية."
                ],
                'negation_flip': [
                    "التجربة لم تنجح بسبب عطل في المعدات.",
                    "الاقتراح لم توافق عليه اللجنة.",
                    "الطقس غير مناسب للأنشطة الخارجية اليوم.",
                    "الدواء يجب ألا يؤخذ مع الكحول.",
                    "المبنى غير متاح لمستخدمي الكراسي المتحركة."
                ],
                'contextual_misleading': [
                    "وفقاً للدراسات الحديثة، قد تكون هذه المعلومات قديمة.",
                    "بعض الخبراء يختلفون مع الرأي المقبول عموماً.",
                    "النظريات البديلة تقترح تفسيراً مختلفاً.",
                    "النقاد يجادلون بأن الأدلة غير كافية.",
                    "هناك حاجة لمزيد من البحث لتأكيد هذه النتائج."
                ]
            }
        }
    
    def inject_distractors(self, haystack: str, needle: str, 
                          config: DistractorConfig) -> DistractorInjectionResult:
        """
        Inject distractors into haystack based on configuration
        
        Args:
            haystack: Original haystack text
            needle: Needle text to preserve
            config: Distractor injection configuration
            
        Returns:
            DistractorInjectionResult with modified haystack and metadata
        """
        # Find needle position in haystack
        needle_position = haystack.find(needle)
        if needle_position == -1:
            self.logger.warning("Needle not found in haystack")
            needle_position = len(haystack) // 2  # Default to middle
        
        # Generate distractors based on type
        distractors = self._generate_distractors(needle, config)
        
        # Determine injection positions
        injection_positions = self._determine_injection_positions(
            haystack, needle_position, len(needle), config
        )
        
        # Inject distractors
        modified_haystack, actual_positions = self._inject_at_positions(
            haystack, distractors, injection_positions
        )
        
        # Calculate length change
        length_change = len(modified_haystack) - len(haystack)
        
        return DistractorInjectionResult(
            original_haystack=haystack,
            modified_haystack=modified_haystack,
            needle_position=needle_position,
            distractors_injected=distractors,
            distractor_positions=actual_positions,
            injection_config=config,
            haystack_length_change=length_change
        )
    
    def _generate_distractors(self, needle: str, config: DistractorConfig) -> List[Dict[str, Any]]:
        """
        Generate distractors based on needle and configuration
        
        Args:
            needle: Original needle text
            config: Distractor configuration
            
        Returns:
            List of distractor dictionaries
        """
        distractors = []
        
        # Get templates for language and distractor type
        lang_templates = self.distractor_templates.get(config.language, 
                                                      self.distractor_templates['en'])
        type_templates = lang_templates.get(config.distractor_type.value, [])
        
        if not type_templates:
            self.logger.warning(f"No templates for {config.distractor_type.value} in {config.language}")
            return distractors
        
        # Calculate number of distractors to generate
        num_distractors = max(1, int(len(type_templates) * config.injection_ratio))
        
        if config.distractor_type == DistractorType.SEMANTIC_SIMILAR:
            distractors = self._generate_semantic_distractors(needle, type_templates, num_distractors, config)
        elif config.distractor_type == DistractorType.SYNTACTIC_SIMILAR:
            distractors = self._generate_syntactic_distractors(needle, type_templates, num_distractors, config)
        elif config.distractor_type == DistractorType.KEYWORD_OVERLAP:
            distractors = self._generate_keyword_distractors(needle, type_templates, num_distractors, config)
        elif config.distractor_type == DistractorType.NUMERICAL_CONFUSION:
            distractors = self._generate_numerical_distractors(needle, type_templates, num_distractors, config)
        elif config.distractor_type == DistractorType.TEMPORAL_CONFUSION:
            distractors = self._generate_temporal_distractors(needle, type_templates, num_distractors, config)
        elif config.distractor_type == DistractorType.ENTITY_SUBSTITUTION:
            distractors = self._generate_entity_distractors(needle, type_templates, num_distractors, config)
        elif config.distractor_type == DistractorType.NEGATION_FLIP:
            distractors = self._generate_negation_distractors(needle, type_templates, num_distractors, config)
        elif config.distractor_type == DistractorType.CONTEXTUAL_MISLEADING:
            distractors = self._generate_contextual_distractors(needle, type_templates, num_distractors, config)
        else:
            # Default: random selection from templates
            selected_templates = random.sample(type_templates, min(num_distractors, len(type_templates)))
            for i, template in enumerate(selected_templates):
                distractors.append({
                    'text': template,
                    'type': config.distractor_type.value,
                    'similarity_score': 0.5,  # Default similarity
                    'distractor_id': f"{config.distractor_type.value}_{i}"
                })
        
        return distractors
    
    def _generate_semantic_distractors(self, needle: str, templates: List[str], 
                                     num_distractors: int, config: DistractorConfig) -> List[Dict[str, Any]]:
        """Generate semantically similar distractors"""
        distractors = []
        
        # Extract key concepts from needle
        needle_concepts = self._extract_concepts(needle)
        
        # Select templates and modify them to be semantically similar
        selected_templates = random.sample(templates, min(num_distractors, len(templates)))
        
        for i, template in enumerate(selected_templates):
            # Modify template to include similar concepts
            modified_text = self._inject_similar_concepts(template, needle_concepts)
            
            distractors.append({
                'text': modified_text,
                'type': 'semantic_similar',
                'similarity_score': random.uniform(config.similarity_threshold, 0.9),
                'distractor_id': f"semantic_{i}",
                'original_template': template
            })
        
        return distractors
    
    def _generate_syntactic_distractors(self, needle: str, templates: List[str], 
                                      num_distractors: int, config: DistractorConfig) -> List[Dict[str, Any]]:
        """Generate syntactically similar distractors"""
        distractors = []
        
        # Extract syntactic pattern from needle
        needle_pattern = self._extract_syntactic_pattern(needle)
        
        selected_templates = random.sample(templates, min(num_distractors, len(templates)))
        
        for i, template in enumerate(selected_templates):
            # Modify template to match syntactic pattern
            modified_text = self._apply_syntactic_pattern(template, needle_pattern)
            
            distractors.append({
                'text': modified_text,
                'type': 'syntactic_similar',
                'similarity_score': random.uniform(0.6, 0.8),
                'distractor_id': f"syntactic_{i}",
                'pattern_applied': needle_pattern
            })
        
        return distractors
    
    def _generate_keyword_distractors(self, needle: str, templates: List[str], 
                                    num_distractors: int, config: DistractorConfig) -> List[Dict[str, Any]]:
        """Generate distractors with keyword overlap"""
        distractors = []
        
        # Extract keywords from needle
        needle_keywords = self._extract_keywords(needle)
        
        selected_templates = random.sample(templates, min(num_distractors, len(templates)))
        
        for i, template in enumerate(selected_templates):
            # Inject some needle keywords into template
            modified_text = self._inject_keywords(template, needle_keywords)
            
            overlap_ratio = len(set(modified_text.lower().split()) & 
                              set(needle.lower().split())) / len(needle.split())
            
            distractors.append({
                'text': modified_text,
                'type': 'keyword_overlap',
                'similarity_score': overlap_ratio,
                'distractor_id': f"keyword_{i}",
                'keywords_injected': needle_keywords[:2]  # Limit to 2 keywords
            })
        
        return distractors
    
    def _generate_numerical_distractors(self, needle: str, templates: List[str], 
                                      num_distractors: int, config: DistractorConfig) -> List[Dict[str, Any]]:
        """Generate distractors with numerical confusion"""
        distractors = []
        
        # Extract numbers from needle
        needle_numbers = re.findall(r'\d+(?:\.\d+)?', needle)
        
        selected_templates = random.sample(templates, min(num_distractors, len(templates)))
        
        for i, template in enumerate(selected_templates):
            modified_text = template
            
            # If needle has numbers, create similar but different numbers
            if needle_numbers:
                for num_str in needle_numbers:
                    try:
                        original_num = float(num_str)
                        # Create confusing number (±10-50% variation)
                        variation = random.uniform(0.1, 0.5) * random.choice([-1, 1])
                        confusing_num = original_num * (1 + variation)
                        
                        if '.' in num_str:
                            confusing_str = f"{confusing_num:.1f}"
                        else:
                            confusing_str = str(int(confusing_num))
                        
                        # Replace a number in template with confusing number
                        template_numbers = re.findall(r'\d+(?:\.\d+)?', modified_text)
                        if template_numbers:
                            old_num = template_numbers[0]
                            modified_text = modified_text.replace(old_num, confusing_str, 1)
                    except ValueError:
                        continue
            
            distractors.append({
                'text': modified_text,
                'type': 'numerical_confusion',
                'similarity_score': 0.7,
                'distractor_id': f"numerical_{i}",
                'original_numbers': needle_numbers
            })
        
        return distractors
    
    def _generate_temporal_distractors(self, needle: str, templates: List[str], 
                                     num_distractors: int, config: DistractorConfig) -> List[Dict[str, Any]]:
        """Generate distractors with temporal confusion"""
        distractors = []
        
        # Extract years/dates from needle
        needle_years = re.findall(r'\b(19|20)\d{2}\b', needle)
        
        selected_templates = random.sample(templates, min(num_distractors, len(templates)))
        
        for i, template in enumerate(selected_templates):
            modified_text = template
            
            # If needle has years, create confusing years
            if needle_years:
                for year_str in needle_years:
                    year = int(year_str)
                    # Create confusing year (±5-20 years)
                    year_diff = random.randint(5, 20) * random.choice([-1, 1])
                    confusing_year = year + year_diff
                    
                    # Replace year in template
                    template_years = re.findall(r'\b(19|20)\d{2}\b', modified_text)
                    if template_years:
                        old_year = template_years[0]
                        modified_text = modified_text.replace(old_year, str(confusing_year), 1)
            
            distractors.append({
                'text': modified_text,
                'type': 'temporal_confusion',
                'similarity_score': 0.6,
                'distractor_id': f"temporal_{i}",
                'original_years': needle_years
            })
        
        return distractors
    
    def _generate_entity_distractors(self, needle: str, templates: List[str], 
                                   num_distractors: int, config: DistractorConfig) -> List[Dict[str, Any]]:
        """Generate distractors with entity substitution"""
        distractors = []
        
        # Extract named entities from needle (simple approach)
        needle_entities = self._extract_entities(needle)
        
        selected_templates = random.sample(templates, min(num_distractors, len(templates)))
        
        for i, template in enumerate(selected_templates):
            # Templates already contain different entities, use as-is
            distractors.append({
                'text': template,
                'type': 'entity_substitution',
                'similarity_score': 0.5,
                'distractor_id': f"entity_{i}",
                'needle_entities': needle_entities
            })
        
        return distractors
    
    def _generate_negation_distractors(self, needle: str, templates: List[str], 
                                     num_distractors: int, config: DistractorConfig) -> List[Dict[str, Any]]:
        """Generate distractors with negation flips"""
        distractors = []
        
        selected_templates = random.sample(templates, min(num_distractors, len(templates)))
        
        for i, template in enumerate(selected_templates):
            # Create negated version of needle concepts
            modified_text = self._create_negated_version(needle, template)
            
            distractors.append({
                'text': modified_text,
                'type': 'negation_flip',
                'similarity_score': 0.8,  # High similarity but opposite meaning
                'distractor_id': f"negation_{i}",
                'negation_applied': True
            })
        
        return distractors
    
    def _generate_contextual_distractors(self, needle: str, templates: List[str], 
                                       num_distractors: int, config: DistractorConfig) -> List[Dict[str, Any]]:
        """Generate contextually misleading distractors"""
        distractors = []
        
        selected_templates = random.sample(templates, min(num_distractors, len(templates)))
        
        for i, template in enumerate(selected_templates):
            # Combine template with needle context to create misleading information
            modified_text = self._create_misleading_context(needle, template)
            
            distractors.append({
                'text': modified_text,
                'type': 'contextual_misleading',
                'similarity_score': 0.7,
                'distractor_id': f"contextual_{i}",
                'misleading_strategy': 'uncertainty_injection'
            })
        
        return distractors
    
    def _determine_injection_positions(self, haystack: str, needle_position: int, 
                                     needle_length: int, config: DistractorConfig) -> List[int]:
        """
        Determine where to inject distractors based on position strategy
        
        Args:
            haystack: Original haystack text
            needle_position: Position of needle in haystack
            needle_length: Length of needle
            config: Distractor configuration
            
        Returns:
            List of positions where distractors should be injected
        """
        positions = []
        haystack_length = len(haystack)
        
        if config.position_strategy == "random":
            # Random positions throughout haystack
            num_positions = max(1, int(haystack_length * config.injection_ratio / 100))
            positions = sorted(random.sample(range(0, haystack_length), 
                                           min(num_positions, haystack_length)))
        
        elif config.position_strategy == "before_needle":
            # Positions before the needle
            if needle_position > 0:
                num_positions = max(1, int(needle_position * config.injection_ratio / 100))
                positions = sorted(random.sample(range(0, needle_position), 
                                                min(num_positions, needle_position)))
        
        elif config.position_strategy == "after_needle":
            # Positions after the needle
            after_needle_start = needle_position + needle_length
            if after_needle_start < haystack_length:
                remaining_length = haystack_length - after_needle_start
                num_positions = max(1, int(remaining_length * config.injection_ratio / 100))
                positions = sorted(random.sample(range(after_needle_start, haystack_length), 
                                                min(num_positions, remaining_length)))
        
        elif config.position_strategy == "surrounding":
            # Positions both before and after needle
            before_positions = []
            after_positions = []
            
            if needle_position > 0:
                num_before = max(1, int(needle_position * config.injection_ratio / 200))
                before_positions = random.sample(range(0, needle_position), 
                                                min(num_before, needle_position))
            
            after_needle_start = needle_position + needle_length
            if after_needle_start < haystack_length:
                remaining_length = haystack_length - after_needle_start
                num_after = max(1, int(remaining_length * config.injection_ratio / 200))
                after_positions = random.sample(range(after_needle_start, haystack_length), 
                                               min(num_after, remaining_length))
            
            positions = sorted(before_positions + after_positions)
        
        return positions
    
    def _inject_at_positions(self, haystack: str, distractors: List[Dict[str, Any]], 
                           positions: List[int]) -> Tuple[str, List[int]]:
        """
        Inject distractors at specified positions
        
        Args:
            haystack: Original haystack text
            distractors: List of distractors to inject
            positions: Positions where to inject
            
        Returns:
            Tuple of (modified_haystack, actual_positions)
        """
        if not distractors or not positions:
            return haystack, []
        
        # Sort positions in reverse order to maintain position accuracy during injection
        sorted_positions = sorted(positions, reverse=True)
        
        modified_haystack = haystack
        actual_positions = []
        
        for i, position in enumerate(sorted_positions):
            if i >= len(distractors):
                break
            
            distractor_text = distractors[i]['text']
            
            # Insert distractor at position
            modified_haystack = (modified_haystack[:position] + 
                               " " + distractor_text + " " + 
                               modified_haystack[position:])
            
            actual_positions.append(position)
        
        # Reverse actual_positions to match original order
        actual_positions.reverse()
        
        return modified_haystack, actual_positions
    
    # Helper methods for concept and pattern extraction
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple keyword extraction (can be enhanced with NLP)
        words = text.lower().split()
        # Filter out common words
        stop_words = {'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        concepts = [word for word in words if word not in stop_words and len(word) > 3]
        return concepts[:5]  # Return top 5 concepts
    
    def _extract_syntactic_pattern(self, text: str) -> str:
        """Extract syntactic pattern from text"""
        # Simple pattern extraction based on sentence structure
        if "is" in text.lower():
            return "subject_is_object"
        elif "was" in text.lower():
            return "subject_was_object"
        elif "has" in text.lower():
            return "subject_has_object"
        else:
            return "general_statement"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        words = text.split()
        # Simple heuristic: words longer than 4 characters
        keywords = [word.strip('.,!?') for word in words if len(word) > 4]
        return keywords[:3]  # Return top 3 keywords
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text (simple approach)"""
        # Simple heuristic: capitalized words that aren't at sentence start
        words = text.split()
        entities = []
        for i, word in enumerate(words):
            if word[0].isupper() and i > 0 and words[i-1][-1] not in '.!?':
                entities.append(word.strip('.,!?'))
        return entities
    
    def _inject_similar_concepts(self, template: str, concepts: List[str]) -> str:
        """Inject similar concepts into template"""
        # Simple replacement of some words with concepts
        words = template.split()
        if len(words) > 3 and concepts:
            # Replace a random word with a concept
            replace_idx = random.randint(1, len(words) - 2)
            words[replace_idx] = random.choice(concepts)
        return " ".join(words)
    
    def _apply_syntactic_pattern(self, template: str, pattern: str) -> str:
        """Apply syntactic pattern to template"""
        # Simple pattern application
        if pattern == "subject_is_object" and "is" not in template.lower():
            words = template.split()
            if len(words) > 2:
                words.insert(2, "is")
        return " ".join(words) if 'words' in locals() else template
    
    def _inject_keywords(self, template: str, keywords: List[str]) -> str:
        """Inject keywords into template"""
        if not keywords:
            return template
        
        # Replace some words with keywords
        words = template.split()
        num_replacements = min(2, len(keywords), len(words) // 3)
        
        for _ in range(num_replacements):
            if words and keywords:
                replace_idx = random.randint(0, len(words) - 1)
                words[replace_idx] = random.choice(keywords)
        
        return " ".join(words)
    
    def _create_negated_version(self, needle: str, template: str) -> str:
        """Create negated version combining needle and template"""
        # Simple negation by adding "not" or "never"
        negation_words = ["not", "never", "no longer", "cannot"]
        negation = random.choice(negation_words)
        
        # Extract main concept from needle
        needle_concepts = self._extract_concepts(needle)
        if needle_concepts:
            concept = needle_concepts[0]
            return f"The {concept} {negation} {template.split()[-3:]}".strip()
        
        return f"It is {negation} true that {template.lower()}"
    
    def _create_misleading_context(self, needle: str, template: str) -> str:
        """Create misleading context combining needle and template"""
        # Combine needle concepts with uncertainty from template
        needle_concepts = self._extract_concepts(needle)
        if needle_concepts:
            concept = needle_concepts[0]
            return f"While some sources claim {concept}, {template.lower()}"
        
        return f"Contrary to popular belief, {template.lower()}"
    
    def evaluate_distractor_effectiveness(self, original_response: str, 
                                        distracted_response: str,
                                        needle: str) -> Dict[str, float]:
        """
        Evaluate the effectiveness of injected distractors
        
        Args:
            original_response: Agent response without distractors
            distracted_response: Agent response with distractors
            needle: Original needle text
            
        Returns:
            Dictionary with effectiveness metrics
        """
        # Calculate response similarity
        original_words = set(original_response.lower().split())
        distracted_words = set(distracted_response.lower().split())
        
        if original_words:
            response_similarity = len(original_words & distracted_words) / len(original_words)
        else:
            response_similarity = 0.0
        
        # Calculate needle preservation
        needle_words = set(needle.lower().split())
        original_needle_overlap = len(original_words & needle_words) / len(needle_words) if needle_words else 0
        distracted_needle_overlap = len(distracted_words & needle_words) / len(needle_words) if needle_words else 0
        
        needle_preservation = distracted_needle_overlap / original_needle_overlap if original_needle_overlap > 0 else 0
        
        # Calculate distraction effectiveness (lower is more effective)
        distraction_effectiveness = 1.0 - response_similarity
        
        return {
            'response_similarity': response_similarity,
            'needle_preservation': needle_preservation,
            'distraction_effectiveness': distraction_effectiveness,
            'response_length_change': len(distracted_response) - len(original_response)
        }
