# -*- coding: utf-8 -*-
"""
Needle-Question Similarity Evaluator
Evaluates multilingual agents' ability to retrieve semantically relevant needles
across increasing input lengths with varying cosine similarity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
import json
from pathlib import Path

@dataclass
class SimilarityBand:
    """Represents a similarity band for evaluation"""
    name: str
    min_similarity: float
    max_similarity: float
    color: str

@dataclass
class SimilarityEvaluationResult:
    """Results from similarity evaluation"""
    language: str
    input_length: int
    similarity_band: str
    accuracy: float
    response_time: float
    needle_found: bool
    confidence_score: float
    embedding_model: str

class MultilingualEmbeddingModel:
    """Wrapper for multilingual embedding models"""
    
    def __init__(self, model_name: str = "sentence-transformers/LaBSE"):
        """
        Initialize multilingual embedding model
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            # Fallback to a simpler model
            self.model_name = "distilbert-base-multilingual-cased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        
        for text in texts:
            try:
                # Tokenize and encode
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling of last hidden states
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.append(embedding[0])
                    
            except Exception as e:
                self.logger.warning(f"Failed to encode text: {e}")
                # Return zero embedding as fallback
                embeddings.append(np.zeros(768))  # Default BERT embedding size
        
        return np.array(embeddings)

class NeedleQuestionSimilarityEvaluator:
    """
    Evaluates agent performance across different needle-question similarity bands
    """
    
    def __init__(self, embedding_models: Optional[List[str]] = None):
        """
        Initialize similarity evaluator
        
        Args:
            embedding_models: List of embedding model names to use
        """
        self.logger = logging.getLogger(__name__)
        
        # Default multilingual embedding models
        if embedding_models is None:
            embedding_models = [
                "sentence-transformers/LaBSE",
                "sentence-transformers/distiluse-base-multilingual-cased",
                "xlm-roberta-base"
            ]
        
        self.embedding_models = {}
        for model_name in embedding_models:
            try:
                self.embedding_models[model_name] = MultilingualEmbeddingModel(model_name)
            except Exception as e:
                self.logger.warning(f"Failed to load {model_name}: {e}")
        
        # Define similarity bands
        self.similarity_bands = [
            SimilarityBand("Very High", 0.8, 1.0, "#2E8B57"),
            SimilarityBand("High", 0.6, 0.8, "#4682B4"),
            SimilarityBand("Medium", 0.4, 0.6, "#DAA520"),
            SimilarityBand("Low", 0.2, 0.4, "#CD853F"),
            SimilarityBand("Very Low", 0.0, 0.2, "#DC143C")
        ]
        
        self.evaluation_results = []
    
    def generate_needle_question_pairs(self, languages: List[str], 
                                     num_pairs_per_lang: int = 50) -> List[Dict[str, Any]]:
        """
        Generate needle-question pairs with varying similarity
        
        Args:
            languages: List of language codes
            num_pairs_per_lang: Number of pairs to generate per language
            
        Returns:
            List of needle-question pair dictionaries
        """
        pairs = []
        
        # Predefined templates for different languages
        templates = {
            'en': {
                'needles': [
                    "The capital of France is Paris.",
                    "Albert Einstein developed the theory of relativity.",
                    "The Great Wall of China was built over many centuries.",
                    "Shakespeare wrote Romeo and Juliet.",
                    "The Amazon rainforest is located in South America.",
                    "Mount Everest is the highest mountain in the world.",
                    "The Pacific Ocean is the largest ocean on Earth.",
                    "Leonardo da Vinci painted the Mona Lisa.",
                    "The human heart has four chambers.",
                    "Water boils at 100 degrees Celsius."
                ],
                'questions': {
                    'high_sim': [
                        "What is the capital city of France?",
                        "Who created the theory of relativity?",
                        "Where was the Great Wall constructed?",
                        "Who authored Romeo and Juliet?",
                        "Which continent contains the Amazon rainforest?"
                    ],
                    'medium_sim': [
                        "Which European city is known for the Eiffel Tower?",
                        "Which scientist is famous for E=mc²?",
                        "What famous wall exists in Asia?",
                        "Who wrote famous English plays?",
                        "What large forest is in Brazil?"
                    ],
                    'low_sim': [
                        "What do you know about European geography?",
                        "Tell me about famous scientists.",
                        "What are some historical constructions?",
                        "Who are notable English authors?",
                        "What ecosystems exist in South America?"
                    ]
                }
            },
            'id': {
                'needles': [
                    "Ibu kota Indonesia adalah Jakarta.",
                    "Soekarno adalah presiden pertama Indonesia.",
                    "Borobudur adalah candi Buddha terbesar di dunia.",
                    "Bahasa Indonesia menggunakan alfabet Latin.",
                    "Nasi gudeg adalah makanan khas Yogyakarta.",
                    "Pulau Komodo adalah habitat asli komodo.",
                    "Batik adalah warisan budaya Indonesia.",
                    "Gunung Merapi terletak di Jawa Tengah.",
                    "Tari Kecak berasal dari Bali.",
                    "Rendang adalah masakan Padang yang terkenal."
                ],
                'questions': {
                    'high_sim': [
                        "Apa ibu kota negara Indonesia?",
                        "Siapa presiden pertama Republik Indonesia?",
                        "Di mana letak candi Buddha terbesar?",
                        "Alfabet apa yang digunakan bahasa Indonesia?",
                        "Makanan khas apa yang berasal dari Yogyakarta?"
                    ],
                    'medium_sim': [
                        "Kota mana yang menjadi pusat pemerintahan Indonesia?",
                        "Siapa tokoh proklamator kemerdekaan Indonesia?",
                        "Candi apa yang terkenal di Indonesia?",
                        "Bagaimana sistem penulisan bahasa Indonesia?",
                        "Kuliner apa yang terkenal dari Jawa?"
                    ],
                    'low_sim': [
                        "Ceritakan tentang geografi Indonesia.",
                        "Siapa tokoh penting dalam sejarah Indonesia?",
                        "Apa saja warisan budaya Indonesia?",
                        "Bagaimana perkembangan bahasa di Indonesia?",
                        "Apa keunikan kuliner Nusantara?"
                    ]
                }
            },
            'es': {
                'needles': [
                    "La capital de España es Madrid.",
                    "Pablo Picasso fue un famoso pintor español.",
                    "El flamenco es un baile tradicional andaluz.",
                    "La paella es un plato típico de Valencia.",
                    "El Camino de Santiago es una ruta de peregrinación.",
                    "Antoni Gaudí diseñó la Sagrada Familia.",
                    "Don Quijote fue escrito por Cervantes.",
                    "La siesta es una tradición española.",
                    "El Museo del Prado está en Madrid.",
                    "La Alhambra se encuentra en Granada."
                ],
                'questions': {
                    'high_sim': [
                        "¿Cuál es la capital de España?",
                        "¿Quién fue Pablo Picasso?",
                        "¿Qué es el flamenco?",
                        "¿De dónde es originaria la paella?",
                        "¿Qué es el Camino de Santiago?"
                    ],
                    'medium_sim': [
                        "¿Qué ciudad es el centro político de España?",
                        "¿Quién fue un artista español famoso?",
                        "¿Cuál es un baile típico español?",
                        "¿Qué comida es característica de Valencia?",
                        "¿Cuál es una ruta histórica en España?"
                    ],
                    'low_sim': [
                        "Háblame de las ciudades españolas.",
                        "¿Qué sabes del arte español?",
                        "Describe la cultura española.",
                        "¿Cómo es la gastronomía de España?",
                        "¿Qué tradiciones hay en España?"
                    ]
                }
            },
            'ar': {
                'needles': [
                    "عاصمة المملكة العربية السعودية هي الرياض.",
                    "كتب نجيب محفوظ العديد من الروايات المشهورة.",
                    "يقع المسجد الحرام في مكة المكرمة.",
                    "اللغة العربية هي لغة القرآن الكريم.",
                    "الكبسة هي طبق شعبي في الخليج العربي.",
                    "تقع جامعة الأزهر في القاهرة.",
                    "البترا هي مدينة أثرية في الأردن.",
                    "يكتب العربي من اليمين إلى اليسار.",
                    "الخط العربي فن جميل ومتنوع.",
                    "شهر رمضان هو شهر الصيام عند المسلمين."
                ],
                'questions': {
                    'high_sim': [
                        "ما هي عاصمة المملكة العربية السعودية؟",
                        "من هو نجيب محفوظ؟",
                        "أين يقع المسجد الحرام؟",
                        "ما هي لغة القرآن الكريم؟",
                        "ما هو الطبق الشعبي في الخليج؟"
                    ],
                    'medium_sim': [
                        "ما هي المدينة الرئيسية في السعودية؟",
                        "من هو الكاتب العربي المشهور؟",
                        "أين توجد الكعبة المشرفة؟",
                        "بأي لغة نزل القرآن؟",
                        "ما هي الأكلة المشهورة في المنطقة؟"
                    ],
                    'low_sim': [
                        "حدثني عن المدن العربية.",
                        "ماذا تعرف عن الأدب العربي؟",
                        "صف الثقافة الإسلامية.",
                        "كيف هي اللغات في المنطقة؟",
                        "ما هي التقاليد في البلدان العربية؟"
                    ]
                }
            }
        }
        
        for lang in languages:
            if lang not in templates:
                self.logger.warning(f"No templates available for language: {lang}")
                continue
            
            lang_templates = templates[lang]
            needles = lang_templates['needles']
            
            pair_count = 0
            for needle in needles:
                if pair_count >= num_pairs_per_lang:
                    break
                
                # Generate questions with different similarity levels
                for sim_level, questions in lang_templates['questions'].items():
                    if pair_count >= num_pairs_per_lang:
                        break
                    
                    for question in questions:
                        if pair_count >= num_pairs_per_lang:
                            break
                        
                        pairs.append({
                            'language': lang,
                            'needle': needle,
                            'question': question,
                            'similarity_level': sim_level,
                            'pair_id': f"{lang}_{pair_count:03d}"
                        })
                        
                        pair_count += 1
        
        return pairs
    
    def compute_similarity(self, needle: str, question: str, 
                          embedding_model: str = "sentence-transformers/LaBSE") -> float:
        """
        Compute cosine similarity between needle and question
        
        Args:
            needle: The needle text
            question: The question text
            embedding_model: Name of embedding model to use
            
        Returns:
            Cosine similarity score
        """
        if embedding_model not in self.embedding_models:
            self.logger.warning(f"Model {embedding_model} not available, using first available")
            embedding_model = list(self.embedding_models.keys())[0]
        
        model = self.embedding_models[embedding_model]
        
        try:
            embeddings = model.encode([needle, question])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def classify_similarity_band(self, similarity: float) -> str:
        """
        Classify similarity score into bands
        
        Args:
            similarity: Cosine similarity score
            
        Returns:
            Similarity band name
        """
        for band in self.similarity_bands:
            if band.min_similarity <= similarity <= band.max_similarity:
                return band.name
        return "Unknown"
    
    def evaluate_agent_performance(self, agent, needle_question_pairs: List[Dict[str, Any]],
                                 input_lengths: List[int], 
                                 embedding_model: str = "sentence-transformers/LaBSE") -> List[SimilarityEvaluationResult]:
        """
        Evaluate agent performance across similarity bands and input lengths
        
        Args:
            agent: The agent to evaluate
            needle_question_pairs: List of needle-question pairs
            input_lengths: List of input lengths to test
            embedding_model: Embedding model to use for similarity computation
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for pair in needle_question_pairs:
            needle = pair['needle']
            question = pair['question']
            language = pair['language']
            
            # Compute similarity
            similarity = self.compute_similarity(needle, question, embedding_model)
            similarity_band = self.classify_similarity_band(similarity)
            
            for input_length in input_lengths:
                try:
                    # Create haystack of specified length
                    haystack = self._create_haystack(needle, input_length, language)
                    
                    # Evaluate agent
                    start_time = pd.Timestamp.now()
                    response = agent.query(question, haystack)
                    end_time = pd.Timestamp.now()
                    
                    response_time = (end_time - start_time).total_seconds()
                    
                    # Evaluate response
                    needle_found = self._evaluate_needle_retrieval(needle, response)
                    accuracy = 1.0 if needle_found else 0.0
                    confidence_score = self._extract_confidence(response)
                    
                    result = SimilarityEvaluationResult(
                        language=language,
                        input_length=input_length,
                        similarity_band=similarity_band,
                        accuracy=accuracy,
                        response_time=response_time,
                        needle_found=needle_found,
                        confidence_score=confidence_score,
                        embedding_model=embedding_model
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating pair {pair['pair_id']}: {e}")
                    continue
        
        self.evaluation_results.extend(results)
        return results
    
    def _create_haystack(self, needle: str, target_length: int, language: str) -> str:
        """
        Create haystack of specified length containing the needle
        
        Args:
            needle: The needle to embed
            target_length: Target length in tokens
            language: Language code
            
        Returns:
            Haystack text
        """
        # Filler text templates by language
        fillers = {
            'en': [
                "The weather today is quite pleasant with clear skies.",
                "Many people enjoy reading books in their spare time.",
                "Technology continues to advance at a rapid pace.",
                "Education plays a crucial role in personal development.",
                "Environmental conservation is important for future generations.",
                "Sports activities promote physical and mental health.",
                "Cultural diversity enriches our global community.",
                "Scientific research leads to important discoveries.",
                "Art and music inspire creativity and expression.",
                "Travel broadens our understanding of different cultures."
            ],
            'id': [
                "Cuaca hari ini sangat cerah dan menyenangkan.",
                "Banyak orang suka membaca buku di waktu luang.",
                "Teknologi terus berkembang dengan pesat.",
                "Pendidikan berperan penting dalam pengembangan diri.",
                "Pelestarian lingkungan penting untuk generasi mendatang.",
                "Aktivitas olahraga meningkatkan kesehatan fisik dan mental.",
                "Keberagaman budaya memperkaya komunitas global.",
                "Penelitian ilmiah menghasilkan penemuan penting.",
                "Seni dan musik menginspirasi kreativitas dan ekspresi.",
                "Perjalanan memperluas pemahaman tentang budaya berbeda."
            ],
            'es': [
                "El clima de hoy es muy agradable y soleado.",
                "Muchas personas disfrutan leyendo libros en su tiempo libre.",
                "La tecnología continúa avanzando rápidamente.",
                "La educación juega un papel crucial en el desarrollo personal.",
                "La conservación ambiental es importante para las futuras generaciones.",
                "Las actividades deportivas promueven la salud física y mental.",
                "La diversidad cultural enriquece nuestra comunidad global.",
                "La investigación científica conduce a descubrimientos importantes.",
                "El arte y la música inspiran creatividad y expresión.",
                "Viajar amplía nuestra comprensión de diferentes culturas."
            ],
            'ar': [
                "الطقس اليوم جميل ومشمس جداً.",
                "كثير من الناس يستمتعون بقراءة الكتب في وقت فراغهم.",
                "التكنولوجيا تتطور بسرعة كبيرة.",
                "التعليم يلعب دوراً مهماً في التطوير الشخصي.",
                "المحافظة على البيئة مهمة للأجيال القادمة.",
                "الأنشطة الرياضية تعزز الصحة الجسدية والنفسية.",
                "التنوع الثقافي يثري مجتمعنا العالمي.",
                "البحث العلمي يؤدي إلى اكتشافات مهمة.",
                "الفن والموسيقى يلهمان الإبداع والتعبير.",
                "السفر يوسع فهمنا للثقافات المختلفة."
            ]
        }
        
        if language not in fillers:
            language = 'en'  # Default to English
        
        filler_sentences = fillers[language]
        
        # Estimate tokens (rough approximation)
        needle_tokens = len(needle.split())
        current_tokens = needle_tokens
        
        haystack_parts = [needle]
        
        # Add filler text to reach target length
        while current_tokens < target_length:
            filler = np.random.choice(filler_sentences)
            haystack_parts.append(filler)
            current_tokens += len(filler.split())
        
        # Shuffle to randomize needle position
        np.random.shuffle(haystack_parts)
        
        return " ".join(haystack_parts)
    
    def _evaluate_needle_retrieval(self, needle: str, response: str) -> bool:
        """
        Evaluate if the needle information was successfully retrieved
        
        Args:
            needle: Original needle text
            response: Agent response
            
        Returns:
            True if needle was successfully retrieved
        """
        # Simple keyword-based evaluation
        needle_keywords = set(needle.lower().split())
        response_keywords = set(response.lower().split())
        
        # Check for significant overlap
        overlap = len(needle_keywords.intersection(response_keywords))
        overlap_ratio = overlap / len(needle_keywords) if needle_keywords else 0
        
        return overlap_ratio >= 0.5  # At least 50% keyword overlap
    
    def _extract_confidence(self, response: str) -> float:
        """
        Extract confidence score from agent response
        
        Args:
            response: Agent response
            
        Returns:
            Confidence score (0-1)
        """
        # Simple heuristic based on response characteristics
        if not response or len(response.strip()) == 0:
            return 0.0
        
        # Longer, more detailed responses typically indicate higher confidence
        response_length = len(response.split())
        
        if response_length < 5:
            return 0.3
        elif response_length < 15:
            return 0.6
        elif response_length < 30:
            return 0.8
        else:
            return 0.9
    
    def generate_accuracy_curves(self, results: Optional[List[SimilarityEvaluationResult]] = None) -> Dict[str, plt.Figure]:
        """
        Generate accuracy vs input length curves per similarity band
        
        Args:
            results: Evaluation results (uses stored results if None)
            
        Returns:
            Dictionary of figures by language
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            self.logger.warning("No evaluation results available for plotting")
            return {}
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([
            {
                'language': r.language,
                'input_length': r.input_length,
                'similarity_band': r.similarity_band,
                'accuracy': r.accuracy,
                'response_time': r.response_time,
                'confidence_score': r.confidence_score
            }
            for r in results
        ])
        
        figures = {}
        
        # Create plots for each language
        for language in df['language'].unique():
            lang_df = df[df['language'] == language]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Needle-Question Similarity Analysis - {language.upper()}', fontsize=16)
            
            # 1. Accuracy vs Input Length by Similarity Band
            ax1 = axes[0, 0]
            for band in lang_df['similarity_band'].unique():
                band_df = lang_df[lang_df['similarity_band'] == band]
                if not band_df.empty:
                    grouped = band_df.groupby('input_length')['accuracy'].mean().reset_index()
                    ax1.plot(grouped['input_length'], grouped['accuracy'], 
                            marker='o', label=band, linewidth=2)
            
            ax1.set_xlabel('Input Length (tokens)')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy vs Input Length by Similarity Band')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Response Time vs Input Length
            ax2 = axes[0, 1]
            grouped_time = lang_df.groupby('input_length')['response_time'].mean().reset_index()
            ax2.plot(grouped_time['input_length'], grouped_time['response_time'], 
                    marker='s', color='red', linewidth=2)
            ax2.set_xlabel('Input Length (tokens)')
            ax2.set_ylabel('Response Time (seconds)')
            ax2.set_title('Response Time vs Input Length')
            ax2.grid(True, alpha=0.3)
            
            # 3. Accuracy Distribution by Similarity Band
            ax3 = axes[1, 0]
            similarity_accuracy = lang_df.groupby('similarity_band')['accuracy'].mean().sort_values(ascending=False)
            bars = ax3.bar(range(len(similarity_accuracy)), similarity_accuracy.values)
            ax3.set_xticks(range(len(similarity_accuracy)))
            ax3.set_xticklabels(similarity_accuracy.index, rotation=45)
            ax3.set_ylabel('Average Accuracy')
            ax3.set_title('Average Accuracy by Similarity Band')
            
            # Color bars by similarity level
            colors = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#DC143C']
            for i, bar in enumerate(bars):
                if i < len(colors):
                    bar.set_color(colors[i])
            
            # 4. Confidence vs Accuracy Scatter
            ax4 = axes[1, 1]
            scatter = ax4.scatter(lang_df['confidence_score'], lang_df['accuracy'], 
                                alpha=0.6, c=lang_df['input_length'], cmap='viridis')
            ax4.set_xlabel('Confidence Score')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Confidence vs Accuracy')
            plt.colorbar(scatter, ax=ax4, label='Input Length')
            
            plt.tight_layout()
            figures[language] = fig
        
        return figures
    
    def export_results(self, filepath: str, results: Optional[List[SimilarityEvaluationResult]] = None):
        """
        Export evaluation results to file
        
        Args:
            filepath: Output file path
            results: Results to export (uses stored results if None)
        """
        if results is None:
            results = self.evaluation_results
        
        # Convert to serializable format
        export_data = []
        for result in results:
            export_data.append({
                'language': result.language,
                'input_length': result.input_length,
                'similarity_band': result.similarity_band,
                'accuracy': result.accuracy,
                'response_time': result.response_time,
                'needle_found': result.needle_found,
                'confidence_score': result.confidence_score,
                'embedding_model': result.embedding_model
            })
        
        # Save as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results exported to {filepath}")
    
    def generate_summary_report(self, results: Optional[List[SimilarityEvaluationResult]] = None) -> Dict[str, Any]:
        """
        Generate summary report of evaluation results
        
        Args:
            results: Results to summarize (uses stored results if None)
            
        Returns:
            Summary report dictionary
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            return {"error": "No results available"}
        
        df = pd.DataFrame([
            {
                'language': r.language,
                'input_length': r.input_length,
                'similarity_band': r.similarity_band,
                'accuracy': r.accuracy,
                'response_time': r.response_time,
                'confidence_score': r.confidence_score
            }
            for r in results
        ])
        
        report = {
            'total_evaluations': len(results),
            'languages_tested': list(df['language'].unique()),
            'similarity_bands_tested': list(df['similarity_band'].unique()),
            'input_length_range': {
                'min': int(df['input_length'].min()),
                'max': int(df['input_length'].max())
            },
            'overall_performance': {
                'average_accuracy': float(df['accuracy'].mean()),
                'accuracy_std': float(df['accuracy'].std()),
                'average_response_time': float(df['response_time'].mean()),
                'average_confidence': float(df['confidence_score'].mean())
            },
            'performance_by_language': {},
            'performance_by_similarity_band': {},
            'degradation_analysis': {}
        }
        
        # Performance by language
        for lang in df['language'].unique():
            lang_df = df[df['language'] == lang]
            report['performance_by_language'][lang] = {
                'accuracy': float(lang_df['accuracy'].mean()),
                'response_time': float(lang_df['response_time'].mean()),
                'confidence': float(lang_df['confidence_score'].mean()),
                'evaluations': len(lang_df)
            }
        
        # Performance by similarity band
        for band in df['similarity_band'].unique():
            band_df = df[df['similarity_band'] == band]
            report['performance_by_similarity_band'][band] = {
                'accuracy': float(band_df['accuracy'].mean()),
                'response_time': float(band_df['response_time'].mean()),
                'confidence': float(band_df['confidence_score'].mean()),
                'evaluations': len(band_df)
            }
        
        # Degradation analysis
        for lang in df['language'].unique():
            lang_df = df[df['language'] == lang]
            if len(lang_df['input_length'].unique()) > 1:
                # Calculate correlation between input length and accuracy
                correlation = lang_df['input_length'].corr(lang_df['accuracy'])
                report['degradation_analysis'][lang] = {
                    'length_accuracy_correlation': float(correlation),
                    'degradation_severity': 'high' if correlation < -0.5 else 'medium' if correlation < -0.2 else 'low'
                }
        
        return report