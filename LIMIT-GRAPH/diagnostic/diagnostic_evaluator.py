# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH Diagnostic Evaluator
Main evaluation script for long-context robustness testing
"""

import argparse
import json
import logging
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluate_similarity import NeedleQuestionSimilarityEvaluator
from inject_distractors import DistractorInjectionFramework, DistractorType, DistractorConfig
from compute_needle_haystack_similarity import NeedleHaystackSemanticAlignment
from shuffle_haystack import HaystackStructureSensitivity, ShuffleStrategy, ShuffleConfig
from repeat_task import RepeatedWordsReplicationTask, RepetitionStrategy, RepetitionConfig
from diagnostic_dashboard import DiagnosticDashboard

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('diagnostic_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_test_data(filepath: str) -> List[Dict[str, Any]]:
    """Load test data from file"""
    test_data = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.jsonl'):
                for line in f:
                    line = line.strip()
                    if line:
                        test_data.append(json.loads(line))
            else:
                test_data = json.load(f)
        
        return test_data
        
    except FileNotFoundError:
        print(f"Error: Test data file not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {filepath}: {e}")
        return []

def create_mock_agent():
    """Create a mock agent for testing purposes"""
    class MockAgent:
        def __init__(self):
            self.name = "MockAgent"
        
        def query(self, question: str, context: str) -> str:
            """Mock query method that returns simple responses"""
            # Simple keyword-based response
            question_lower = question.lower()
            context_lower = context.lower()
            
            # Look for key information in context
            sentences = context.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in question_lower.split()):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                return ". ".join(relevant_sentences[:2])  # Return first 2 relevant sentences
            else:
                return "I cannot find relevant information in the provided context."
    
    return MockAgent()

def run_similarity_evaluation(evaluator: NeedleQuestionSimilarityEvaluator,
                            agent, test_data: List[Dict], 
                            config: Dict[str, Any],
                            logger: logging.Logger) -> List[Any]:
    """Run needle-question similarity evaluation"""
    logger.info("Starting similarity evaluation")
    
    # Generate needle-question pairs
    languages = config.get('languages', ['en', 'id'])
    pairs_per_lang = config.get('pairs_per_language', 20)
    
    needle_question_pairs = evaluator.generate_needle_question_pairs(languages, pairs_per_lang)
    
    # Test different input lengths
    input_lengths = config.get('input_lengths', [100, 500, 1000, 2000])
    
    # Run evaluation
    results = evaluator.evaluate_agent_performance(
        agent, needle_question_pairs, input_lengths
    )
    
    logger.info(f"Similarity evaluation completed: {len(results)} results")
    return results

def run_distractor_evaluation(framework: DistractorInjectionFramework,
                            agent, test_data: List[Dict],
                            config: Dict[str, Any],
                            logger: logging.Logger) -> List[Any]:
    """Run distractor injection evaluation"""
    logger.info("Starting distractor evaluation")
    
    results = []
    
    # Test different distractor types
    distractor_types = [
        DistractorType.SEMANTIC_SIMILAR,
        DistractorType.KEYWORD_OVERLAP,
        DistractorType.NUMERICAL_CONFUSION,
        DistractorType.NEGATION_FLIP
    ]
    
    languages = config.get('languages', ['en', 'id'])
    
    for data_item in test_data[:config.get('max_samples', 10)]:
        haystack = data_item.get('haystack', data_item.get('context', ''))
        needle = data_item.get('needle', data_item.get('answer', ''))
        
        if not haystack or not needle:
            continue
        
        for distractor_type in distractor_types:
            for language in languages:
                try:
                    # Configure distractor injection
                    distractor_config = DistractorConfig(
                        distractor_type=distractor_type,
                        injection_ratio=0.3,
                        similarity_threshold=0.6,
                        position_strategy="random",
                        language=language
                    )
                    
                    # Inject distractors
                    injection_result = framework.inject_distractors(haystack, needle, distractor_config)
                    
                    # Test agent on original and modified haystack
                    query = data_item.get('query', f"Find information about {needle}")
                    
                    original_response = agent.query(query, haystack)
                    distracted_response = agent.query(query, injection_result.modified_haystack)
                    
                    # Evaluate effectiveness
                    effectiveness = framework.evaluate_distractor_effectiveness(
                        original_response, distracted_response, needle
                    )
                    
                    result = {
                        'injection_result': injection_result,
                        'original_response': original_response,
                        'distracted_response': distracted_response,
                        'effectiveness_metrics': effectiveness,
                        'query': query
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error in distractor evaluation: {e}")
                    continue
    
    logger.info(f"Distractor evaluation completed: {len(results)} results")
    return results

def run_alignment_evaluation(analyzer: NeedleHaystackSemanticAlignment,
                           test_data: List[Dict],
                           config: Dict[str, Any],
                           logger: logging.Logger) -> List[Any]:
    """Run semantic alignment evaluation"""
    logger.info("Starting semantic alignment evaluation")
    
    results = []
    languages = config.get('languages', ['en', 'id'])
    
    for data_item in test_data[:config.get('max_samples', 10)]:
        haystack = data_item.get('haystack', data_item.get('context', ''))
        needle = data_item.get('needle', data_item.get('answer', ''))
        
        if not haystack or not needle:
            continue
        
        for language in languages:
            try:
                # Compute semantic alignment
                alignment_result = analyzer.compute_semantic_alignment(
                    needle, haystack, language
                )
                
                results.append(alignment_result)
                
            except Exception as e:
                logger.error(f"Error in alignment evaluation: {e}")
                continue
    
    logger.info(f"Alignment evaluation completed: {len(results)} results")
    return results

def run_structure_evaluation(tester: HaystackStructureSensitivity,
                           agent, test_data: List[Dict],
                           config: Dict[str, Any],
                           logger: logging.Logger) -> List[Any]:
    """Run structure sensitivity evaluation"""
    logger.info("Starting structure sensitivity evaluation")
    
    results = []
    
    # Test different shuffle strategies
    shuffle_strategies = [
        ShuffleStrategy.SENTENCE_SHUFFLE,
        ShuffleStrategy.PARAGRAPH_SHUFFLE,
        ShuffleStrategy.REVERSE_ORDER,
        ShuffleStrategy.TOPIC_SCRAMBLE
    ]
    
    languages = config.get('languages', ['en', 'id'])
    
    for data_item in test_data[:config.get('max_samples', 10)]:
        haystack = data_item.get('haystack', data_item.get('context', ''))
        needle = data_item.get('needle', data_item.get('answer', ''))
        
        if not haystack or not needle:
            continue
        
        for strategy in shuffle_strategies:
            for language in languages:
                try:
                    # Configure shuffling
                    shuffle_config = ShuffleConfig(
                        strategy=strategy,
                        shuffle_ratio=0.5,
                        preserve_needle=True,
                        language=language
                    )
                    
                    # Apply shuffling
                    structure_result = tester.apply_shuffle_strategy(haystack, needle, shuffle_config)
                    
                    # Test agent on original and shuffled haystack
                    query = data_item.get('query', f"Find information about {needle}")
                    
                    original_response = agent.query(query, haystack)
                    shuffled_response = agent.query(query, structure_result.shuffled_haystack)
                    
                    # Evaluate sensitivity
                    sensitivity_metrics = tester.evaluate_structure_sensitivity(
                        agent, original_response, shuffled_response, structure_result
                    )
                    
                    result = {
                        'structure_result': structure_result,
                        'original_response': original_response,
                        'shuffled_response': shuffled_response,
                        'sensitivity_metrics': sensitivity_metrics,
                        'query': query
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error in structure evaluation: {e}")
                    continue
    
    logger.info(f"Structure evaluation completed: {len(results)} results")
    return results

def run_repetition_evaluation(task: RepeatedWordsReplicationTask,
                            agent, test_data: List[Dict],
                            config: Dict[str, Any],
                            logger: logging.Logger) -> List[Any]:
    """Run repetition robustness evaluation"""
    logger.info("Starting repetition evaluation")
    
    results = []
    
    # Test different repetition strategies
    repetition_strategies = [
        RepetitionStrategy.EXACT_REPETITION,
        RepetitionStrategy.PARAPHRASED_REPETITION,
        RepetitionStrategy.KEYWORD_REPETITION,
        RepetitionStrategy.NOISE_REPETITION
    ]
    
    languages = config.get('languages', ['en', 'id'])
    
    for data_item in test_data[:config.get('max_samples', 10)]:
        haystack = data_item.get('haystack', data_item.get('context', ''))
        needle = data_item.get('needle', data_item.get('answer', ''))
        
        if not haystack or not needle:
            continue
        
        for strategy in repetition_strategies:
            for language in languages:
                try:
                    # Configure repetition
                    repetition_config = RepetitionConfig(
                        strategy=strategy,
                        repetition_count=5,
                        repetition_ratio=0.3,
                        distribution="random",
                        language=language
                    )
                    
                    # Apply repetition
                    repetition_result = task.apply_repetition_strategy(haystack, needle, repetition_config)
                    
                    # Test agent on original and repeated haystack
                    query = data_item.get('query', f"Find information about {needle}")
                    
                    original_response = agent.query(query, haystack)
                    repeated_response = agent.query(query, repetition_result.repeated_haystack)
                    
                    # Evaluate robustness
                    robustness_metrics = task.evaluate_repetition_robustness(
                        agent, original_response, repeated_response, repetition_result
                    )
                    
                    result = {
                        'repetition_result': repetition_result,
                        'original_response': original_response,
                        'repeated_response': repeated_response,
                        'robustness_metrics': robustness_metrics,
                        'query': query
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error in repetition evaluation: {e}")
                    continue
    
    logger.info(f"Repetition evaluation completed: {len(results)} results")
    return results

def save_results(results: Dict[str, List], output_dir: str, logger: logging.Logger):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for eval_type, eval_results in results.items():
        if eval_results:
            output_path = os.path.join(output_dir, f"{eval_type}_results.json")
            
            try:
                # Convert results to serializable format
                serializable_results = []
                for result in eval_results:
                    if hasattr(result, '__dict__'):
                        serializable_results.append(result.__dict__)
                    else:
                        serializable_results.append(result)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, indent=2, default=str)
                
                logger.info(f"Saved {len(eval_results)} {eval_type} results to {output_path}")
                
            except Exception as e:
                logger.error(f"Error saving {eval_type} results: {e}")

def generate_summary_report(results: Dict[str, List], logger: logging.Logger) -> str:
    """Generate summary report of all evaluations"""
    report_lines = [
        "# LIMIT-GRAPH Diagnostic Evaluation Report",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Evaluation Summary"
    ]
    
    total_evaluations = sum(len(eval_results) for eval_results in results.values())
    report_lines.append(f"**Total Evaluations:** {total_evaluations}")
    report_lines.append("")
    
    for eval_type, eval_results in results.items():
        if eval_results:
            report_lines.extend([
                f"### {eval_type.replace('_', ' ').title()}",
                f"- **Count:** {len(eval_results)}",
                f"- **Status:** ✅ Completed",
                ""
            ])
        else:
            report_lines.extend([
                f"### {eval_type.replace('_', ' ').title()}",
                f"- **Count:** 0",
                f"- **Status:** ❌ No results",
                ""
            ])
    
    # Add performance insights
    report_lines.extend([
        "## Key Insights",
        "- Evaluation completed successfully across multiple diagnostic dimensions",
        "- Results provide comprehensive view of long-context robustness",
        "- Data ready for dashboard visualization and analysis",
        "",
        "## Next Steps",
        "1. Load results into diagnostic dashboard for visualization",
        "2. Analyze performance patterns across different challenge types",
        "3. Identify areas for agent improvement",
        "4. Compare results across different languages and configurations"
    ])
    
    return "\n".join(report_lines)

def main():
    """Main diagnostic evaluation function"""
    parser = argparse.ArgumentParser(description="LIMIT-GRAPH Diagnostic Evaluation")
    
    parser.add_argument("--test_data", type=str,
                       default="../data/corpus.jsonl",
                       help="Path to test data file")
    
    parser.add_argument("--evaluations", nargs="+",
                       choices=["similarity", "distractor", "alignment", "structure", "repetition"],
                       default=["similarity", "distractor"],
                       help="Types of evaluations to run")
    
    parser.add_argument("--languages", nargs="+",
                       default=["en", "id"],
                       help="Languages to test")
    
    parser.add_argument("--max_samples", type=int,
                       default=10,
                       help="Maximum number of samples to test per evaluation")
    
    parser.add_argument("--output_dir", type=str,
                       default="diagnostic_results",
                       help="Output directory for results")
    
    parser.add_argument("--log_level", type=str,
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting LIMIT-GRAPH Diagnostic Evaluation")
    
    # Load test data
    test_data = load_test_data(args.test_data)
    if not test_data:
        logger.error("No test data loaded, exiting")
        return
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Create agent (mock for demonstration)
    agent = create_mock_agent()
    logger.info(f"Created agent: {agent.name}")
    
    # Configuration
    config = {
        'languages': args.languages,
        'max_samples': args.max_samples,
        'pairs_per_language': 10,
        'input_lengths': [100, 500, 1000]
    }
    
    # Initialize evaluators
    evaluators = {}
    if "similarity" in args.evaluations:
        evaluators['similarity'] = NeedleQuestionSimilarityEvaluator()
    if "distractor" in args.evaluations:
        evaluators['distractor'] = DistractorInjectionFramework()
    if "alignment" in args.evaluations:
        evaluators['alignment'] = NeedleHaystackSemanticAlignment()
    if "structure" in args.evaluations:
        evaluators['structure'] = HaystackStructureSensitivity()
    if "repetition" in args.evaluations:
        evaluators['repetition'] = RepeatedWordsReplicationTask()
    
    # Run evaluations
    results = {}
    
    if "similarity" in evaluators:
        results['similarity_evaluation'] = run_similarity_evaluation(
            evaluators['similarity'], agent, test_data, config, logger
        )
    
    if "distractor" in evaluators:
        results['distractor_evaluation'] = run_distractor_evaluation(
            evaluators['distractor'], agent, test_data, config, logger
        )
    
    if "alignment" in evaluators:
        results['alignment_evaluation'] = run_alignment_evaluation(
            evaluators['alignment'], test_data, config, logger
        )
    
    if "structure" in evaluators:
        results['structure_evaluation'] = run_structure_evaluation(
            evaluators['structure'], agent, test_data, config, logger
        )
    
    if "repetition" in evaluators:
        results['repetition_evaluation'] = run_repetition_evaluation(
            evaluators['repetition'], agent, test_data, config, logger
        )
    
    # Save results
    save_results(results, args.output_dir, logger)
    
    # Generate summary report
    summary_report = generate_summary_report(results, logger)
    
    # Save summary report
    report_path = os.path.join(args.output_dir, "diagnostic_summary_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    logger.info(f"Summary report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print(summary_report)
    print("="*60)
    
    logger.info("Diagnostic evaluation completed successfully")

if __name__ == "__main__":
    main()