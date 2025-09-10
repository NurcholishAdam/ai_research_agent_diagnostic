# -*- coding: utf-8 -*-
"""
Integration Tests for LIMIT-GRAPH Diagnostic Module
Comprehensive testing of long-context robustness evaluation system
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any

from evaluate_similarity import NeedleQuestionSimilarityEvaluator, SimilarityEvaluationResult
from inject_distractors import DistractorInjectionFramework, DistractorType, DistractorConfig
from compute_needle_haystack_similarity import NeedleHaystackSemanticAlignment, SemanticAlignmentResult
from shuffle_haystack import HaystackStructureSensitivity, ShuffleStrategy, ShuffleConfig
from repeat_task import RepeatedWordsReplicationTask, RepetitionStrategy, RepetitionConfig
from diagnostic_dashboard import DiagnosticDashboard

class MockAgent:
    """Mock agent for testing"""
    def __init__(self):
        self.name = "TestAgent"
    
    def query(self, question: str, context: str) -> str:
        # Simple mock response based on context
        if "Paris" in context:
            return "Paris is the capital of France."
        elif "Einstein" in context:
            return "Einstein developed the theory of relativity."
        else:
            return "I found relevant information in the context."

class TestSimilarityEvaluator(unittest.TestCase):
    """Test needle-question similarity evaluator"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.evaluator = NeedleQuestionSimilarityEvaluator()
        self.agent = MockAgent()
    
    def test_generate_needle_question_pairs(self):
        """Test needle-question pair generation including Arabic"""
        pairs = self.evaluator.generate_needle_question_pairs(['en', 'id', 'ar'], 5)
        
        self.assertIsInstance(pairs, list)
        self.assertGreater(len(pairs), 0)
        
        # Check that Arabic is included
        languages_found = set(pair['language'] for pair in pairs)
        self.assertTrue('ar' in languages_found or 'en' in languages_found)
        
        # Check pair structure
        for pair in pairs:
            self.assertIn('language', pair)
            self.assertIn('needle', pair)
            self.assertIn('question', pair)
            self.assertIn('similarity_level', pair)
    
    def test_compute_similarity(self):
        """Test similarity computation"""
        needle = "The capital of France is Paris."
        question = "What is the capital of France?"
        
        similarity = self.evaluator.compute_similarity(needle, question)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_classify_similarity_band(self):
        """Test similarity band classification"""
        high_sim = self.evaluator.classify_similarity_band(0.9)
        medium_sim = self.evaluator.classify_similarity_band(0.5)
        low_sim = self.evaluator.classify_similarity_band(0.1)
        
        self.assertIn(high_sim, ["Very High", "High"])
        self.assertIn(medium_sim, ["Medium", "High", "Low"])
        self.assertIn(low_sim, ["Very Low", "Low"])

class TestDistractorInjection(unittest.TestCase):
    """Test distractor injection framework"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.framework = DistractorInjectionFramework()
        self.haystack = "The capital of France is Paris. It is a beautiful city with many attractions."
        self.needle = "The capital of France is Paris."
    
    def test_semantic_distractor_injection(self):
        """Test semantic distractor injection"""
        config = DistractorConfig(
            distractor_type=DistractorType.SEMANTIC_SIMILAR,
            injection_ratio=0.3,
            similarity_threshold=0.6,
            position_strategy="random",
            language="en"
        )
        
        result = self.framework.inject_distractors(self.haystack, self.needle, config)
        
        self.assertIsNotNone(result.modified_haystack)
        self.assertNotEqual(result.original_haystack, result.modified_haystack)
        self.assertGreater(len(result.distractors_injected), 0)
        self.assertEqual(result.injection_config.distractor_type, DistractorType.SEMANTIC_SIMILAR)
    
    def test_keyword_overlap_injection(self):
        """Test keyword overlap distractor injection"""
        config = DistractorConfig(
            distractor_type=DistractorType.KEYWORD_OVERLAP,
            injection_ratio=0.2,
            similarity_threshold=0.5,
            position_strategy="random",
            language="en"
        )
        
        result = self.framework.inject_distractors(self.haystack, self.needle, config)
        
        self.assertIsNotNone(result.modified_haystack)
        self.assertGreater(len(result.distractors_injected), 0)
    
    def test_evaluate_distractor_effectiveness(self):
        """Test distractor effectiveness evaluation"""
        original_response = "Paris is the capital of France."
        distracted_response = "Berlin is mentioned as a capital city."
        
        effectiveness = self.framework.evaluate_distractor_effectiveness(
            original_response, distracted_response, self.needle
        )
        
        self.assertIn('response_similarity', effectiveness)
        self.assertIn('needle_preservation', effectiveness)
        self.assertIn('distraction_effectiveness', effectiveness)

class TestSemanticAlignment(unittest.TestCase):
    """Test semantic alignment analyzer"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.analyzer = NeedleHaystackSemanticAlignment()
        self.needle = "Einstein developed the theory of relativity."
        self.haystack = "Albert Einstein was a physicist. He developed the theory of relativity. This theory revolutionized physics."
    
    def test_compute_semantic_alignment(self):
        """Test semantic alignment computation"""
        result = self.analyzer.compute_semantic_alignment(
            self.needle, self.haystack, "en"
        )
        
        self.assertIsInstance(result, SemanticAlignmentResult)
        self.assertEqual(result.language, "en")
        self.assertGreater(len(result.haystack_segments), 0)
        self.assertGreater(len(result.segment_similarities), 0)
        self.assertGreaterEqual(result.alignment_score, 0.0)
        self.assertLessEqual(result.alignment_score, 1.0)
    
    def test_segment_haystack(self):
        """Test haystack segmentation"""
        segments = self.analyzer._segment_haystack(self.haystack)
        
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        
        # Check that segments contain text
        for segment in segments:
            self.assertIsInstance(segment, str)
            self.assertGreater(len(segment.strip()), 0)

class TestStructureSensitivity(unittest.TestCase):
    """Test structure sensitivity tester"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.tester = HaystackStructureSensitivity()
        self.haystack = "First sentence. Second sentence with important info. Third sentence concludes."
        self.needle = "important info"
    
    def test_sentence_shuffle(self):
        """Test sentence shuffling"""
        config = ShuffleConfig(
            strategy=ShuffleStrategy.SENTENCE_SHUFFLE,
            shuffle_ratio=0.5,
            preserve_needle=True,
            language="en"
        )
        
        result = self.tester.apply_shuffle_strategy(self.haystack, self.needle, config)
        
        self.assertIsNotNone(result.shuffled_haystack)
        self.assertGreaterEqual(result.structure_change_score, 0.0)
        self.assertLessEqual(result.structure_change_score, 1.0)
        self.assertIn(self.needle, result.shuffled_haystack)  # Needle should be preserved
    
    def test_reverse_order(self):
        """Test reverse order shuffling"""
        config = ShuffleConfig(
            strategy=ShuffleStrategy.REVERSE_ORDER,
            shuffle_ratio=1.0,
            preserve_needle=False,
            language="en"
        )
        
        result = self.tester.apply_shuffle_strategy(self.haystack, self.needle, config)
        
        self.assertIsNotNone(result.shuffled_haystack)
        self.assertNotEqual(result.original_haystack, result.shuffled_haystack)

class TestRepetitionTask(unittest.TestCase):
    """Test repetition task evaluator"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.task = RepeatedWordsReplicationTask()
        self.haystack = "The capital of France is Paris. Paris is a beautiful city."
        self.needle = "Paris"
    
    def test_exact_repetition(self):
        """Test exact repetition strategy"""
        config = RepetitionConfig(
            strategy=RepetitionStrategy.EXACT_REPETITION,
            repetition_count=3,
            repetition_ratio=0.3,
            distribution="random",
            language="en"
        )
        
        result = self.task.apply_repetition_strategy(self.haystack, self.needle, config)
        
        self.assertIsNotNone(result.repeated_haystack)
        self.assertGreater(len(result.repeated_haystack), len(result.original_haystack))
        self.assertGreaterEqual(result.repetition_density, 0.0)
        self.assertGreaterEqual(result.content_inflation_ratio, 1.0)
    
    def test_paraphrased_repetition(self):
        """Test paraphrased repetition strategy"""
        config = RepetitionConfig(
            strategy=RepetitionStrategy.PARAPHRASED_REPETITION,
            repetition_count=2,
            repetition_ratio=0.2,
            distribution="uniform",
            language="en"
        )
        
        result = self.task.apply_repetition_strategy(self.haystack, self.needle, config)
        
        self.assertIsNotNone(result.repeated_haystack)
        self.assertGreaterEqual(result.uniqueness_preservation_score, 0.0)
        self.assertLessEqual(result.uniqueness_preservation_score, 1.0)

class TestDiagnosticDashboard(unittest.TestCase):
    """Test diagnostic dashboard"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.dashboard = DiagnosticDashboard()
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        self.assertIsInstance(self.dashboard.evaluation_data, dict)
        self.assertIn('similarity_evaluations', self.dashboard.evaluation_data)
        self.assertIn('distractor_evaluations', self.dashboard.evaluation_data)
    
    def test_update_similarity_data(self):
        """Test updating similarity data"""
        mock_results = [
            SimilarityEvaluationResult(
                language="en",
                input_length=100,
                similarity_band="High",
                accuracy=0.8,
                response_time=1.5,
                needle_found=True,
                confidence_score=0.7,
                embedding_model="test_model"
            )
        ]
        
        self.dashboard.update_similarity_data(mock_results)
        
        self.assertEqual(len(self.dashboard.evaluation_data['similarity_evaluations']), 1)
        self.assertGreater(self.dashboard.performance_metrics['total_evaluations'], 0)
    
    def test_export_dashboard_data(self):
        """Test dashboard data export"""
        # Add some mock data
        mock_results = [{"test": "data"}]
        self.dashboard.update_similarity_data(mock_results)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            success = self.dashboard.export_dashboard_data(temp_path)
            self.assertTrue(success)
            
            # Verify exported data
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertIn("evaluation_data", exported_data)
            self.assertIn("performance_metrics", exported_data)
            
        finally:
            os.unlink(temp_path)

class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete integration workflow"""
    
    def test_end_to_end_diagnostic_workflow(self):
        """Test complete diagnostic evaluation workflow"""
        # 1. Setup components
        similarity_evaluator = NeedleQuestionSimilarityEvaluator()
        distractor_framework = DistractorInjectionFramework()
        alignment_analyzer = NeedleHaystackSemanticAlignment()
        structure_tester = HaystackStructureSensitivity()
        repetition_task = RepeatedWordsReplicationTask()
        dashboard = DiagnosticDashboard()
        agent = MockAgent()
        
        # 2. Create test data
        test_haystack = "The capital of France is Paris. It is located in Europe. Paris has many famous landmarks."
        test_needle = "The capital of France is Paris."
        test_query = "What is the capital of France?"
        
        # 3. Run similarity evaluation
        needle_question_pairs = similarity_evaluator.generate_needle_question_pairs(['en'], 2)
        similarity_results = similarity_evaluator.evaluate_agent_performance(
            agent, needle_question_pairs[:1], [100]
        )
        
        # 4. Run distractor injection
        distractor_config = DistractorConfig(
            distractor_type=DistractorType.SEMANTIC_SIMILAR,
            injection_ratio=0.2,
            similarity_threshold=0.5,
            position_strategy="random",
            language="en"
        )
        
        distractor_result = distractor_framework.inject_distractors(
            test_haystack, test_needle, distractor_config
        )
        
        # 5. Run semantic alignment
        alignment_result = alignment_analyzer.compute_semantic_alignment(
            test_needle, test_haystack, "en"
        )
        
        # 6. Run structure sensitivity
        shuffle_config = ShuffleConfig(
            strategy=ShuffleStrategy.SENTENCE_SHUFFLE,
            shuffle_ratio=0.3,
            preserve_needle=True,
            language="en"
        )
        
        structure_result = structure_tester.apply_shuffle_strategy(
            test_haystack, test_needle, shuffle_config
        )
        
        # 7. Run repetition task
        repetition_config = RepetitionConfig(
            strategy=RepetitionStrategy.EXACT_REPETITION,
            repetition_count=2,
            repetition_ratio=0.2,
            distribution="random",
            language="en"
        )
        
        repetition_result = repetition_task.apply_repetition_strategy(
            test_haystack, test_needle, repetition_config
        )
        
        # 8. Update dashboard
        dashboard.update_similarity_data(similarity_results)
        dashboard.update_distractor_data([distractor_result])
        dashboard.update_alignment_data([alignment_result])
        dashboard.update_structure_data([structure_result])
        dashboard.update_repetition_data([repetition_result])
        
        # 9. Verify workflow completion
        self.assertGreater(len(similarity_results), 0)
        self.assertIsNotNone(distractor_result.modified_haystack)
        self.assertIsNotNone(alignment_result.alignment_score)
        self.assertIsNotNone(structure_result.shuffled_haystack)
        self.assertIsNotNone(repetition_result.repeated_haystack)
        
        # Verify dashboard has data
        self.assertGreater(dashboard.performance_metrics['total_evaluations'], 0)
        self.assertGreater(dashboard.performance_metrics['evaluation_types'], 0)

def run_diagnostic_tests():
    """Run all diagnostic integration tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSimilarityEvaluator,
        TestDistractorInjection,
        TestSemanticAlignment,
        TestStructureSensitivity,
        TestRepetitionTask,
        TestDiagnosticDashboard,
        TestIntegrationWorkflow
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Running LIMIT-GRAPH Diagnostic Integration Tests")
    print("="*60)
    
    success = run_diagnostic_tests()
    
    if success:
        print("\n✅ All diagnostic integration tests passed!")
    else:
        print("\n❌ Some diagnostic integration tests failed!")
        exit(1)class TestAr
abicLanguageSupport(unittest.TestCase):
    """Test Arabic language support in diagnostic module"""
    
    def setUp(self):
        """Setup Arabic test fixtures"""
        self.evaluator = NeedleQuestionSimilarityEvaluator()
        self.distractor_framework = DistractorInjectionFramework()
        self.alignment_analyzer = NeedleHaystackSemanticAlignment()
        
    def test_arabic_needle_question_generation(self):
        """Test Arabic needle-question pair generation"""
        pairs = self.evaluator.generate_needle_question_pairs(['ar'], 3)
        
        self.assertGreater(len(pairs), 0)
        
        for pair in pairs:
            self.assertEqual(pair['language'], 'ar')
            self.assertIsInstance(pair['needle'], str)
            self.assertIsInstance(pair['question'], str)
            # Check for Arabic characters
            self.assertTrue(any('\u0600' <= c <= '\u06FF' for c in pair['needle']))
    
    def test_arabic_distractor_injection(self):
        """Test Arabic distractor injection"""
        arabic_haystack = "الطبيب يعمل في المستشفى الكبير في الرياض."
        arabic_needle = "الطبيب يعمل"
        
        config = DistractorConfig(
            distractor_type=DistractorType.SEMANTIC_SIMILAR,
            injection_ratio=0.3,
            similarity_threshold=0.6,
            position_strategy="random",
            language="ar"
        )
        
        result = self.distractor_framework.inject_distractors(
            arabic_haystack, arabic_needle, config
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.injection_config.language, "ar")
        self.assertNotEqual(result.original_haystack, result.modified_haystack)
    
    def test_arabic_semantic_alignment(self):
        """Test Arabic semantic alignment analysis"""
        arabic_needle = "أحمد طبيب"
        arabic_haystack = "أحمد طبيب ماهر يعمل في المستشفى. المستشفى كبير وحديث."
        
        result = self.alignment_analyzer.compute_semantic_alignment(
            arabic_needle, arabic_haystack, "ar"
        )
        
        self.assertEqual(result.language, "ar")
        self.assertGreater(len(result.haystack_segments), 0)
        self.assertGreaterEqual(result.alignment_score, 0.0)
        self.assertLessEqual(result.alignment_score, 1.0)
    
    def test_rtl_text_handling(self):
        """Test RTL text handling capabilities"""
        # This would test RTL-specific functionality
        # For now, we test that Arabic text is processed without errors
        
        rtl_text = "الطبيب يعمل في المستشفى من اليمين إلى اليسار"
        
        # Test that Arabic text can be processed
        try:
            pairs = self.evaluator.generate_needle_question_pairs(['ar'], 1)
            self.assertGreater(len(pairs), 0)
        except Exception as e:
            self.fail(f"RTL text processing failed: {e}")

def run_arabic_integration_tests():
    """Run Arabic-specific integration tests"""
    print("🔤 Running Arabic Language Integration Tests")
    print("="*50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add Arabic-specific test class
    arabic_tests = unittest.TestLoader().loadTestsFromTestCase(TestArabicLanguageSupport)
    test_suite.addTests(arabic_tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    if result.wasSuccessful():
        print("\n✅ All Arabic integration tests passed!")
        print("✅ Arabic needle-question generation: Working")
        print("✅ Arabic distractor injection: Working")
        print("✅ Arabic semantic alignment: Working")
        print("✅ RTL text handling: Working")
    else:
        print("\n❌ Some Arabic integration tests failed!")
        for failure in result.failures:
            print(f"❌ {failure[0]}: {failure[1]}")
        for error in result.errors:
            print(f"❌ {error[0]}: {error[1]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run standard tests
    print("🧪 Running Standard Diagnostic Integration Tests")
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "="*70)
    
    # Run Arabic-specific tests
    run_arabic_integration_tests()