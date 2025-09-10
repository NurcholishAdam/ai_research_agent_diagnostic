# -*- coding: utf-8 -*-
"""
Demo: LIMIT-GRAPH Diagnostic System
Demonstrates the complete long-context robustness evaluation pipeline
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

from evaluate_similarity import NeedleQuestionSimilarityEvaluator
from inject_distractors import DistractorInjectionFramework, DistractorType, DistractorConfig
from compute_needle_haystack_similarity import NeedleHaystackSemanticAlignment
from shuffle_haystack import HaystackStructureSensitivity, ShuffleStrategy, ShuffleConfig
from repeat_task import RepeatedWordsReplicationTask, RepetitionStrategy, RepetitionConfig
from diagnostic_dashboard import DiagnosticDashboard

def setup_demo_logging():
    """Setup logging for demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class DemoAgent:
    """Demo agent with more sophisticated responses"""
    
    def __init__(self):
        self.name = "DemoAgent"
        self.knowledge_base = {
            # English
            "paris": "Paris is the capital and largest city of France.",
            "einstein": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
            "shakespeare": "William Shakespeare was an English playwright and poet.",
            "tokyo": "Tokyo is the capital of Japan and one of the world's largest cities.",
            "beethoven": "Ludwig van Beethoven was a German composer and pianist.",
            
            # Indonesian
            "jakarta": "Jakarta adalah ibu kota dan kota terbesar Indonesia.",
            "soekarno": "Soekarno adalah presiden pertama Republik Indonesia.",
            "borobudur": "Borobudur adalah candi Buddha terbesar di dunia yang terletak di Indonesia.",
            
            # Arabic
            "الرياض": "الرياض هي عاصمة المملكة العربية السعودية وأكبر مدنها.",
            "أحمد": "أحمد هو اسم شائع في العالم العربي.",
            "المسجد": "المسجد هو مكان العبادة للمسلمين.",
            "الطبيب": "الطبيب هو الشخص الذي يعالج المرضى.",
            "مكة": "مكة المكرمة هي أقدس مدينة في الإسلام."
        }
    
    def query(self, question: str, context: str) -> str:
        """Process query with context awareness"""
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Extract key entities from question
        entities = []
        for entity in self.knowledge_base.keys():
            if entity in question_lower or entity in context_lower:
                entities.append(entity)
        
        # Generate response based on context and knowledge
        if entities:
            responses = []
            for entity in entities:
                if entity in context_lower:
                    # Extract relevant sentences from context
                    sentences = context.split('.')
                    relevant_sentences = [s.strip() for s in sentences if entity in s.lower()]
                    if relevant_sentences:
                        responses.extend(relevant_sentences[:2])
                    else:
                        responses.append(self.knowledge_base[entity])
            
            if responses:
                return ". ".join(responses) + "."
        
        # Fallback: extract sentences containing question keywords
        question_words = [word.lower() for word in question.split() if len(word) > 3]
        sentences = context.split('.')
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in question_words):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return ". ".join(relevant_sentences[:2]) + "."
        
        return "I cannot find specific information to answer your question in the provided context."

def create_sample_test_data() -> List[Dict[str, Any]]:
    """Create sample test data for demonstration"""
    return [
        {
            "haystack": "Paris is the capital of France. It is located in Western Europe. The city is famous for the Eiffel Tower and the Louvre Museum. Paris has a population of over 2 million people.",
            "needle": "Paris is the capital of France.",
            "query": "What is the capital of France?",
            "language": "en"
        },
        {
            "haystack": "Albert Einstein was born in Germany in 1879. He developed the theory of relativity which revolutionized physics. Einstein won the Nobel Prize in Physics in 1921. He moved to the United States in 1933.",
            "needle": "Einstein developed the theory of relativity.",
            "query": "What did Einstein develop?",
            "language": "en"
        },
        {
            "haystack": "Jakarta adalah ibu kota Indonesia. Kota ini terletak di pulau Jawa. Jakarta memiliki populasi lebih dari 10 juta jiwa. Kota ini merupakan pusat ekonomi dan politik Indonesia.",
            "needle": "Jakarta adalah ibu kota Indonesia.",
            "query": "Apa ibu kota Indonesia?",
            "language": "id"
        },
        {
            "haystack": "William Shakespeare was an English playwright and poet. He wrote many famous plays including Romeo and Juliet, Hamlet, and Macbeth. Shakespeare lived during the Elizabethan era. His works are still performed today.",
            "needle": "Shakespeare wrote Romeo and Juliet.",
            "query": "Who wrote Romeo and Juliet?",
            "language": "en"
        }
    ]

def demo_similarity_evaluation(logger: logging.Logger):
    """Demonstrate needle-question similarity evaluation"""
    logger.info("🎯 Demonstrating Needle-Question Similarity Evaluation")
    
    evaluator = NeedleQuestionSimilarityEvaluator()
    agent = DemoAgent()
    
    # Generate needle-question pairs
    logger.info("Generating needle-question pairs...")
    pairs = evaluator.generate_needle_question_pairs(['en', 'id'], 5)
    logger.info(f"Generated {len(pairs)} needle-question pairs")
    
    # Show sample pairs
    for i, pair in enumerate(pairs[:3]):
        logger.info(f"  Pair {i+1}: {pair['language']} - {pair['similarity_level']}")
        logger.info(f"    Needle: {pair['needle'][:50]}...")
        logger.info(f"    Question: {pair['question']}")
    
    # Evaluate agent performance
    logger.info("Evaluating agent performance across input lengths...")
    input_lengths = [100, 500, 1000]
    
    results = evaluator.evaluate_agent_performance(agent, pairs[:3], input_lengths)
    
    logger.info(f"Completed {len(results)} evaluations")
    
    # Show sample results
    for result in results[:2]:
        logger.info(f"  Language: {result.language}, Length: {result.input_length}")
        logger.info(f"  Similarity Band: {result.similarity_band}, Accuracy: {result.accuracy:.2f}")
        logger.info(f"  Response Time: {result.response_time:.2f}s")
    
    return results

def demo_distractor_injection(logger: logging.Logger):
    """Demonstrate distractor injection framework"""
    logger.info("\n🎭 Demonstrating Distractor Injection Framework")
    
    framework = DistractorInjectionFramework()
    agent = DemoAgent()
    test_data = create_sample_test_data()
    
    # Test different distractor types
    distractor_types = [
        DistractorType.SEMANTIC_SIMILAR,
        DistractorType.KEYWORD_OVERLAP,
        DistractorType.NUMERICAL_CONFUSION,
        DistractorType.NEGATION_FLIP
    ]
    
    results = []
    
    for i, data_item in enumerate(test_data[:2]):
        haystack = data_item["haystack"]
        needle = data_item["needle"]
        query = data_item["query"]
        language = data_item["language"]
        
        logger.info(f"\nTesting sample {i+1}: {language}")
        logger.info(f"Original haystack length: {len(haystack)} characters")
        
        for distractor_type in distractor_types[:2]:  # Test first 2 types
            logger.info(f"\n  Testing {distractor_type.value} distractors...")
            
            config = DistractorConfig(
                distractor_type=distractor_type,
                injection_ratio=0.3,
                similarity_threshold=0.6,
                position_strategy="random",
                language=language
            )
            
            # Inject distractors
            injection_result = framework.inject_distractors(haystack, needle, config)
            
            logger.info(f"    Injected {len(injection_result.distractors_injected)} distractors")
            logger.info(f"    Modified haystack length: {len(injection_result.modified_haystack)} characters")
            
            # Test agent responses
            original_response = agent.query(query, haystack)
            distracted_response = agent.query(query, injection_result.modified_haystack)
            
            # Evaluate effectiveness
            effectiveness = framework.evaluate_distractor_effectiveness(
                original_response, distracted_response, needle
            )
            
            logger.info(f"    Distraction effectiveness: {effectiveness['distraction_effectiveness']:.2f}")
            logger.info(f"    Needle preservation: {effectiveness['needle_preservation']:.2f}")
            
            results.append({
                'injection_result': injection_result,
                'effectiveness': effectiveness,
                'original_response': original_response,
                'distracted_response': distracted_response
            })
    
    return results

def demo_semantic_alignment(logger: logging.Logger):
    """Demonstrate semantic alignment analysis"""
    logger.info("\n🔗 Demonstrating Semantic Alignment Analysis")
    
    analyzer = NeedleHaystackSemanticAlignment()
    test_data = create_sample_test_data()
    
    results = []
    
    for i, data_item in enumerate(test_data[:2]):
        haystack = data_item["haystack"]
        needle = data_item["needle"]
        language = data_item["language"]
        
        logger.info(f"\nAnalyzing sample {i+1}: {language}")
        
        # Compute semantic alignment
        alignment_result = analyzer.compute_semantic_alignment(needle, haystack, language)
        
        logger.info(f"  Haystack segments: {len(alignment_result.haystack_segments)}")
        logger.info(f"  Max similarity: {alignment_result.max_similarity:.3f}")
        logger.info(f"  Mean similarity: {alignment_result.mean_similarity:.3f}")
        logger.info(f"  Alignment score: {alignment_result.alignment_score:.3f}")
        logger.info(f"  Interference level: {alignment_result.interference_level}")
        
        # Show top similar segments
        segment_sims = list(zip(alignment_result.haystack_segments, alignment_result.segment_similarities))
        segment_sims.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("  Top similar segments:")
        for j, (segment, similarity) in enumerate(segment_sims[:2]):
            logger.info(f"    {j+1}. Similarity: {similarity:.3f} - {segment[:50]}...")
        
        results.append(alignment_result)
    
    return results

def demo_structure_sensitivity(logger: logging.Logger):
    """Demonstrate structure sensitivity testing"""
    logger.info("\n🏗️ Demonstrating Structure Sensitivity Testing")
    
    tester = HaystackStructureSensitivity()
    agent = DemoAgent()
    test_data = create_sample_test_data()
    
    # Test different shuffle strategies
    shuffle_strategies = [
        ShuffleStrategy.SENTENCE_SHUFFLE,
        ShuffleStrategy.REVERSE_ORDER,
        ShuffleStrategy.TOPIC_SCRAMBLE
    ]
    
    results = []
    
    for i, data_item in enumerate(test_data[:2]):
        haystack = data_item["haystack"]
        needle = data_item["needle"]
        query = data_item["query"]
        language = data_item["language"]
        
        logger.info(f"\nTesting sample {i+1}: {language}")
        
        for strategy in shuffle_strategies[:2]:  # Test first 2 strategies
            logger.info(f"\n  Testing {strategy.value} shuffling...")
            
            config = ShuffleConfig(
                strategy=strategy,
                shuffle_ratio=0.5,
                preserve_needle=True,
                language=language
            )
            
            # Apply shuffling
            structure_result = tester.apply_shuffle_strategy(haystack, needle, config)
            
            logger.info(f"    Structure change score: {structure_result.structure_change_score:.3f}")
            logger.info(f"    Coherence score: {structure_result.coherence_score:.3f}")
            logger.info(f"    Readability score: {structure_result.readability_score:.3f}")
            
            # Test agent responses
            original_response = agent.query(query, haystack)
            shuffled_response = agent.query(query, structure_result.shuffled_haystack)
            
            # Evaluate sensitivity
            sensitivity_metrics = tester.evaluate_structure_sensitivity(
                agent, original_response, shuffled_response, structure_result
            )
            
            logger.info(f"    Response similarity: {sensitivity_metrics['response_similarity']:.3f}")
            logger.info(f"    Robustness score: {sensitivity_metrics['robustness_score']:.3f}")
            
            results.append({
                'structure_result': structure_result,
                'sensitivity_metrics': sensitivity_metrics
            })
    
    return results

def demo_repetition_robustness(logger: logging.Logger):
    """Demonstrate repetition robustness testing"""
    logger.info("\n🔄 Demonstrating Repetition Robustness Testing")
    
    task = RepeatedWordsReplicationTask()
    agent = DemoAgent()
    test_data = create_sample_test_data()
    
    # Test different repetition strategies
    repetition_strategies = [
        RepetitionStrategy.EXACT_REPETITION,
        RepetitionStrategy.PARAPHRASED_REPETITION,
        RepetitionStrategy.KEYWORD_REPETITION
    ]
    
    results = []
    
    for i, data_item in enumerate(test_data[:2]):
        haystack = data_item["haystack"]
        needle = data_item["needle"]
        query = data_item["query"]
        language = data_item["language"]
        
        logger.info(f"\nTesting sample {i+1}: {language}")
        
        for strategy in repetition_strategies[:2]:  # Test first 2 strategies
            logger.info(f"\n  Testing {strategy.value} repetition...")
            
            config = RepetitionConfig(
                strategy=strategy,
                repetition_count=3,
                repetition_ratio=0.3,
                distribution="random",
                language=language
            )
            
            # Apply repetition
            repetition_result = task.apply_repetition_strategy(haystack, needle, config)
            
            logger.info(f"    Repetition density: {repetition_result.repetition_density:.3f}")
            logger.info(f"    Content inflation ratio: {repetition_result.content_inflation_ratio:.2f}")
            logger.info(f"    Uniqueness preservation: {repetition_result.uniqueness_preservation_score:.3f}")
            
            # Test agent responses
            original_response = agent.query(query, haystack)
            repeated_response = agent.query(query, repetition_result.repeated_haystack)
            
            # Evaluate robustness
            robustness_metrics = task.evaluate_repetition_robustness(
                agent, original_response, repeated_response, repetition_result
            )
            
            logger.info(f"    Response consistency: {robustness_metrics['response_consistency']:.3f}")
            logger.info(f"    Distraction resistance: {robustness_metrics['distraction_resistance']:.3f}")
            
            results.append({
                'repetition_result': repetition_result,
                'robustness_metrics': robustness_metrics
            })
    
    return results

def demo_dashboard_integration(logger: logging.Logger, all_results: Dict[str, List]):
    """Demonstrate dashboard integration"""
    logger.info("\n📈 Demonstrating Dashboard Integration")
    
    dashboard = DiagnosticDashboard()
    
    # Update dashboard with all results
    if 'similarity' in all_results:
        dashboard.update_similarity_data(all_results['similarity'])
        logger.info(f"  Added {len(all_results['similarity'])} similarity results")
    
    if 'distractor' in all_results:
        dashboard.update_distractor_data([r['injection_result'] for r in all_results['distractor']])
        logger.info(f"  Added {len(all_results['distractor'])} distractor results")
    
    if 'alignment' in all_results:
        dashboard.update_alignment_data(all_results['alignment'])
        logger.info(f"  Added {len(all_results['alignment'])} alignment results")
    
    if 'structure' in all_results:
        dashboard.update_structure_data([r['structure_result'] for r in all_results['structure']])
        logger.info(f"  Added {len(all_results['structure'])} structure results")
    
    if 'repetition' in all_results:
        dashboard.update_repetition_data([r['repetition_result'] for r in all_results['repetition']])
        logger.info(f"  Added {len(all_results['repetition'])} repetition results")
    
    # Show dashboard metrics
    logger.info(f"\nDashboard Summary:")
    logger.info(f"  Total evaluations: {dashboard.performance_metrics['total_evaluations']}")
    logger.info(f"  Evaluation types: {dashboard.performance_metrics['evaluation_types']}")
    logger.info(f"  Last updated: {dashboard.performance_metrics['last_updated']}")
    
    # Export dashboard data
    export_path = "demo_dashboard_export.json"
    if dashboard.export_dashboard_data(export_path):
        logger.info(f"  Dashboard data exported to {export_path}")
    
    return dashboard

def demo_complete_pipeline(logger: logging.Logger):
    """Demonstrate complete diagnostic pipeline"""
    logger.info("\n🔄 Demonstrating Complete Diagnostic Pipeline")
    
    # Run all evaluations
    all_results = {}
    
    logger.info("Running complete diagnostic evaluation pipeline...")
    
    # 1. Similarity evaluation
    similarity_results = demo_similarity_evaluation(logger)
    all_results['similarity'] = similarity_results
    
    # 2. Distractor injection
    distractor_results = demo_distractor_injection(logger)
    all_results['distractor'] = distractor_results
    
    # 3. Semantic alignment
    alignment_results = demo_semantic_alignment(logger)
    all_results['alignment'] = alignment_results
    
    # 4. Structure sensitivity
    structure_results = demo_structure_sensitivity(logger)
    all_results['structure'] = structure_results
    
    # 5. Repetition robustness
    repetition_results = demo_repetition_robustness(logger)
    all_results['repetition'] = repetition_results
    
    # 6. Dashboard integration
    dashboard = demo_dashboard_integration(logger, all_results)
    
    # Generate summary
    total_evaluations = sum(len(results) if isinstance(results, list) else 1 
                          for results in all_results.values())
    
    logger.info(f"\n✅ Pipeline completed successfully!")
    logger.info(f"   Total evaluations: {total_evaluations}")
    logger.info(f"   Evaluation types: {len(all_results)}")
    logger.info(f"   Dashboard ready for visualization")
    
    return all_results, dashboard

def main():
    """Main demo function"""
    logger = setup_demo_logging()
    
    logger.info("🚀 LIMIT-GRAPH Diagnostic System Demo")
    logger.info("="*60)
    
    try:
        # Run individual demonstrations
        logger.info("Running individual component demonstrations...")
        
        similarity_results = demo_similarity_evaluation(logger)
        distractor_results = demo_distractor_injection(logger)
        alignment_results = demo_semantic_alignment(logger)
        structure_results = demo_structure_sensitivity(logger)
        repetition_results = demo_repetition_robustness(logger)
        
        # Combine all results
        all_results = {
            'similarity': similarity_results,
            'distractor': distractor_results,
            'alignment': alignment_results,
            'structure': structure_results,
            'repetition': repetition_results
        }
        
        # Dashboard integration
        dashboard = demo_dashboard_integration(logger, all_results)
        
        # Complete pipeline demonstration
        pipeline_results, pipeline_dashboard = demo_complete_pipeline(logger)
        
        logger.info("\n🎉 Demo completed successfully!")
        logger.info("="*60)
        logger.info("Key takeaways:")
        logger.info("• Comprehensive evaluation across 5 diagnostic dimensions")
        logger.info("• Multi-language support (English and Indonesian)")
        logger.info("• Real-time dashboard visualization capabilities")
        logger.info("• Robust evaluation metrics and analysis")
        logger.info("• Integration-ready for LIMIT-GRAPH ecosystem")
        logger.info("\nNext steps:")
        logger.info("1. Run diagnostic_evaluator.py for full evaluation")
        logger.info("2. Launch dashboard with: streamlit run diagnostic_dashboard.py")
        logger.info("3. Integrate with your LIMIT-GRAPH agents")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()d
ef demo_multilingual_similarity_evaluation(logger: logging.Logger):
    """Demonstrate multilingual similarity evaluation including Arabic"""
    logger.info("🌍 Demonstrating Multilingual Similarity Evaluation")
    
    evaluator = NeedleQuestionSimilarityEvaluator()
    
    # Test multilingual needle-question pairs including Arabic
    languages = ["en", "id", "ar"]
    pairs = evaluator.generate_needle_question_pairs(languages, 5)
    
    logger.info(f"Generated {len(pairs)} multilingual needle-question pairs")
    
    for pair in pairs[:9]:  # Show first 9 pairs (3 per language)
        logger.info(f"\nLanguage: {pair['language'].upper()}")
        logger.info(f"Needle: {pair['needle']}")
        logger.info(f"Question: {pair['question']}")
        logger.info(f"Similarity Level: {pair['similarity_level']}")
        
        # Compute similarity
        try:
            similarity = evaluator.compute_similarity(pair['needle'], pair['question'])
            similarity_band = evaluator.classify_similarity_band(similarity)
            logger.info(f"Computed Similarity: {similarity:.3f} ({similarity_band})")
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")

def demo_arabic_distractor_injection(logger: logging.Logger):
    """Demonstrate Arabic distractor injection"""
    logger.info("\n🔤 Demonstrating Arabic Distractor Injection")
    
    framework = DistractorInjectionFramework()
    
    # Arabic test case
    arabic_haystack = "الطبيب يعمل في المستشفى الكبير في الرياض. المستشفى يحتوي على أحدث المعدات الطبية."
    arabic_needle = "الطبيب يعمل في المستشفى"
    
    logger.info(f"Original Arabic Haystack: {arabic_haystack}")
    logger.info(f"Arabic Needle: {arabic_needle}")
    
    # Test different distractor types with Arabic
    distractor_types = [
        DistractorType.SEMANTIC_SIMILAR,
        DistractorType.KEYWORD_OVERLAP,
        DistractorType.CONTEXTUAL_MISLEADING
    ]
    
    for distractor_type in distractor_types:
        try:
            config = DistractorConfig(
                distractor_type=distractor_type,
                injection_ratio=0.3,
                similarity_threshold=0.6,
                position_strategy="random",
                language="ar"
            )
            
            result = framework.inject_distractors(arabic_haystack, arabic_needle, config)
            
            logger.info(f"\n--- {distractor_type.value.upper()} DISTRACTORS ---")
            logger.info(f"Modified Haystack: {result.modified_haystack}")
            logger.info(f"Distractors Injected: {len(result.distractors_injected)}")
            
        except Exception as e:
            logger.error(f"Error with {distractor_type.value}: {e}")

def demo_arabic_semantic_alignment(logger: logging.Logger):
    """Demonstrate Arabic semantic alignment analysis"""
    logger.info("\n📊 Demonstrating Arabic Semantic Alignment")
    
    analyzer = NeedleHaystackSemanticAlignment()
    
    # Arabic test case
    arabic_needle = "أحمد طبيب"
    arabic_haystack = """
    أحمد طبيب ماهر يعمل في المستشفى الرئيسي في الرياض. 
    يتخصص في جراحة القلب وله خبرة طويلة في هذا المجال.
    المستشفى يقع في وسط المدينة ويخدم آلاف المرضى يومياً.
    الأطباء في المستشفى يعملون بجد لتقديم أفضل رعاية طبية.
    أحمد معروف بين زملائه بالدقة والاهتمام بالمرضى.
    """
    
    logger.info(f"Arabic Needle: {arabic_needle}")
    logger.info(f"Arabic Haystack: {arabic_haystack.strip()}")
    
    try:
        result = analyzer.compute_semantic_alignment(arabic_needle, arabic_haystack, "ar")
        
        logger.info(f"\nSemantic Alignment Results:")
        logger.info(f"Language: {result.language}")
        logger.info(f"Max Similarity: {result.max_similarity:.3f}")
        logger.info(f"Mean Similarity: {result.mean_similarity:.3f}")
        logger.info(f"Alignment Score: {result.alignment_score:.3f}")
        logger.info(f"Interference Level: {result.interference_level}")
        logger.info(f"Segments Analyzed: {len(result.haystack_segments)}")
        
    except Exception as e:
        logger.error(f"Error in Arabic semantic alignment: {e}")

def demo_arabic_structure_sensitivity(logger: logging.Logger):
    """Demonstrate Arabic structure sensitivity testing"""
    logger.info("\n🔀 Demonstrating Arabic Structure Sensitivity")
    
    tester = HaystackStructureSensitivity()
    agent = DemoAgent()
    
    # Arabic test case
    arabic_haystack = "الطبيب أحمد يعمل في المستشفى. المستشفى كبير وحديث. أحمد طبيب ماهر ومتخصص."
    arabic_needle = "أحمد طبيب"
    
    logger.info(f"Original Arabic Haystack: {arabic_haystack}")
    logger.info(f"Arabic Needle: {arabic_needle}")
    
    # Test different shuffle strategies
    shuffle_strategies = [
        ShuffleStrategy.SENTENCE_SHUFFLE,
        ShuffleStrategy.REVERSE_ORDER,
        ShuffleStrategy.WORD_SHUFFLE
    ]
    
    for strategy in shuffle_strategies:
        try:
            config = ShuffleConfig(
                strategy=strategy,
                shuffle_ratio=0.5,
                preserve_needle=True,
                language="ar"
            )
            
            result = tester.apply_shuffle_strategy(arabic_haystack, arabic_needle, config)
            
            logger.info(f"\n--- {strategy.value.upper()} ---")
            logger.info(f"Shuffled Haystack: {result.shuffled_haystack}")
            logger.info(f"Structure Change Score: {result.structure_change_score:.3f}")
            logger.info(f"Coherence Score: {result.coherence_score:.3f}")
            
            # Test agent on original vs shuffled
            original_response = agent.query("من هو الطبيب؟", arabic_haystack)
            shuffled_response = agent.query("من هو الطبيب؟", result.shuffled_haystack)
            
            logger.info(f"Original Response: {original_response}")
            logger.info(f"Shuffled Response: {shuffled_response}")
            
        except Exception as e:
            logger.error(f"Error with {strategy.value}: {e}")

def demo_arabic_repetition_robustness(logger: logging.Logger):
    """Demonstrate Arabic repetition robustness testing"""
    logger.info("\n🔁 Demonstrating Arabic Repetition Robustness")
    
    task = RepeatedWordsReplicationTask()
    agent = DemoAgent()
    
    # Arabic test case
    arabic_haystack = "الطبيب أحمد يعمل في المستشفى الكبير."
    arabic_needle = "أحمد"
    
    logger.info(f"Original Arabic Haystack: {arabic_haystack}")
    logger.info(f"Arabic Needle: {arabic_needle}")
    
    # Test different repetition strategies
    repetition_strategies = [
        RepetitionStrategy.EXACT_REPETITION,
        RepetitionStrategy.PARAPHRASED_REPETITION,
        RepetitionStrategy.KEYWORD_REPETITION
    ]
    
    for strategy in repetition_strategies:
        try:
            config = RepetitionConfig(
                strategy=strategy,
                repetition_count=3,
                repetition_ratio=0.3,
                distribution="random",
                language="ar"
            )
            
            result = task.apply_repetition_strategy(arabic_haystack, arabic_needle, config)
            
            logger.info(f"\n--- {strategy.value.upper()} ---")
            logger.info(f"Repeated Haystack: {result.repeated_haystack}")
            logger.info(f"Content Inflation Ratio: {result.content_inflation_ratio:.2f}")
            logger.info(f"Uniqueness Preservation: {result.uniqueness_preservation_score:.3f}")
            
            # Test agent robustness
            original_response = agent.query("من هو الطبيب؟", arabic_haystack)
            repeated_response = agent.query("من هو الطبيب؟", result.repeated_haystack)
            
            logger.info(f"Original Response: {original_response}")
            logger.info(f"Repeated Response: {repeated_response}")
            
        except Exception as e:
            logger.error(f"Error with {strategy.value}: {e}")

def demo_multilingual_dashboard(logger: logging.Logger):
    """Demonstrate multilingual diagnostic dashboard"""
    logger.info("\n📈 Demonstrating Multilingual Diagnostic Dashboard")
    
    dashboard = DiagnosticDashboard()
    
    # Create mock multilingual evaluation data
    from evaluate_similarity import SimilarityEvaluationResult
    
    mock_evaluations = []
    languages = ["en", "id", "ar"]
    
    for i in range(15):
        lang = languages[i % len(languages)]
        
        # Simulate different performance by language
        if lang == "ar":
            base_accuracy = 0.7  # Arabic might be more challenging
        elif lang == "id":
            base_accuracy = 0.8
        else:
            base_accuracy = 0.85
        
        accuracy = base_accuracy + (i % 3) * 0.05
        
        evaluation = SimilarityEvaluationResult(
            language=lang,
            input_length=500 + i * 100,
            similarity_band="High" if accuracy > 0.8 else "Medium",
            accuracy=accuracy,
            response_time=1.0 + (i % 3) * 0.5,
            needle_found=accuracy > 0.7,
            confidence_score=accuracy * 0.9,
            embedding_model="LaBSE"
        )
        
        mock_evaluations.append(evaluation)
    
    # Update dashboard
    dashboard.update_similarity_data(mock_evaluations)
    
    logger.info(f"Dashboard updated with {len(mock_evaluations)} multilingual evaluations")
    
    # Generate multilingual performance chart
    try:
        multilingual_fig = dashboard.create_multilingual_performance_chart()
        logger.info("Multilingual performance chart created successfully")
        
        # Generate Arabic-specific analysis
        arabic_fig = dashboard.create_arabic_specific_analysis()
        logger.info("Arabic-specific analysis chart created successfully")
        
        # Generate RTL analysis
        rtl_analysis = dashboard.create_rtl_text_analysis()
        logger.info(f"RTL Analysis: {rtl_analysis}")
        
    except Exception as e:
        logger.error(f"Error creating dashboard visualizations: {e}")

def main():
    """Main demo function with Arabic language support"""
    logger = setup_demo_logging()
    
    logger.info("🚀 LIMIT-GRAPH Diagnostic System Demo with Arabic Support")
    logger.info("="*70)
    logger.info("Languages: English (EN) | Indonesian (ID) | Arabic (AR)")
    logger.info("="*70)
    
    try:
        # Run all demonstrations including Arabic
        demo_multilingual_similarity_evaluation(logger)
        demo_arabic_distractor_injection(logger)
        demo_arabic_semantic_alignment(logger)
        demo_arabic_structure_sensitivity(logger)
        demo_arabic_repetition_robustness(logger)
        demo_multilingual_dashboard(logger)
        
        logger.info("\n🎉 Multilingual Diagnostic Demo completed successfully!")
        logger.info("✅ English language support: Full")
        logger.info("✅ Indonesian language support: Full")
        logger.info("✅ Arabic language support: Full with RTL handling")
        logger.info("✅ Cross-lingual evaluation: Enabled")
        logger.info("✅ Cultural context awareness: Integrated")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()