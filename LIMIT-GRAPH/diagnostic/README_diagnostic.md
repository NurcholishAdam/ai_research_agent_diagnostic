# LIMIT-GRAPH Diagnostic Module: Long-Context Robustness

## Overview

The LIMIT-GRAPH Diagnostic Module provides comprehensive evaluation of multilingual agents' ability to handle long contexts with various challenges. This module tests robustness across five key dimensions: similarity-based retrieval, distractor resistance, semantic alignment, structure sensitivity, and repetition handling.

### 🌍 Multilingual Support
- **English (EN)**: Full support with comprehensive test scenarios
- **Indonesian (ID)**: Complete language integration with cultural context
- **Arabic (AR)**: Advanced RTL support with script-aware processing
- **Spanish (ES)**: Extended Latin script support
- **French (FR)**: Romance language integration

### 🔬 Core Diagnostic Capabilities
- **Needle-Question Similarity**: Evaluates retrieval across similarity bands
- **Distractor Injection**: Tests resistance to misleading information
- **Semantic Alignment**: Analyzes needle-haystack coherence
- **Structure Sensitivity**: Measures response to content reorganization
- **Repetition Robustness**: Assesses handling of repeated content

## Architecture### Co
re Components

1. **Similarity Evaluator** (`evaluate_similarity.py`)
   - Multilingual needle-question pair generation
   - Cosine similarity computation across languages
   - Similarity band classification with cultural awareness

2. **Distractor Framework** (`inject_distractors.py`)
   - Language-specific distractor templates
   - Semantic, syntactic, and contextual misleading content
   - Cultural context-aware distractor generation

3. **Semantic Alignment** (`compute_needle_haystack_similarity.py`)
   - Cross-lingual semantic coherence analysis
   - Language-aware text segmentation
   - RTL text processing capabilities

4. **Structure Sensitivity** (`shuffle_haystack.py`)
   - Content reorganization strategies
   - Language-specific sentence boundary detection
   - Topic-based semantic clustering

5. **Repetition Robustness** (`repeat_task.py`)
   - Multilingual paraphrasing templates
   - Language-specific synonym generation
   - Cultural context preservation

6. **Diagnostic Dashboard** (`diagnostic_dashboard.py`)
   - Multilingual performance visualization
   - Cross-lingual comparison analytics
   - RTL text handling metrics

## Arabic Language Support

### 🔤 RTL (Right-to-Left) Processing
The diagnostic module provides comprehensive Arabic language support with advanced RTL text handling:

#### **Text Direction Handling**
- **Bidirectional Text**: Full BiDi algorithm compliance
- **Mixed Content**: Arabic-Latin mixed text processing
- **Direction Markers**: Unicode RTL/LTR override support
- **Rendering Compliance**: Proper Arabic text rendering validation

#### **Arabic Script Features**
- **Diacritic Processing**: Vowel marks and pronunciation guides
- **Letter Normalization**: Alef, Yeh, and Teh Marbuta variations
- **Contextual Forms**: Initial, medial, final, and isolated letters
- **Punctuation Support**: Arabic question mark (؟) and other marks

### 📝 Arabic Content Generation

#### **Needle-Question Templates**
```python
# Arabic needle-question pairs with cultural context
arabic_needles = [
    "عاصمة المملكة العربية السعودية هي الرياض.",
    "كتب نجيب محفوظ العديد من الروايات المشهورة.",
    "يقع المسجد الحرام في مكة المكرمة.",
    "اللغة العربية هي لغة القرآن الكريم."
]

arabic_questions = {
    'high_sim': [
        "ما هي عاصمة المملكة العربية السعودية؟",
        "من هو نجيب محفوظ؟",
        "أين يقع المسجد الحرام؟"
    ],
    'medium_sim': [
        "ما هي المدينة الرئيسية في السعودية؟",
        "من هو الكاتب العربي المشهور؟",
        "أين توجد الكعبة المشرفة؟"
    ]
}
```

#### **Arabic Distractor Types**
1. **Semantic Distractors**: Similar Arabic concepts
   - `"عاصمة ألمانيا هي برلين المشهورة بتاريخها العريق."`
   - `"لندن هي عاصمة المملكة المتحدة التاريخية."`

2. **Syntactic Distractors**: Similar sentence structures
   - `"رئيس الشركة أعلن سياسات جديدة أمس."`
   - `"مدير القسم قدم التقرير النهائي."`

3. **Keyword Overlap**: Shared Arabic terms
   - `"نظرية النسبية العامة لأينشتاين ثورت الفيزياء."`
   - `"نظرية الجاذبية لنيوتن فسرت حركة الكواكب."`

4. **Cultural Context**: Islamic and Middle Eastern references
   - `"شكسبير كتب مسرحيات مشهورة منها هاملت وماكبث."`
   - `"موتسارت ألف سيمفونيات وأوبرات جميلة في القرن الثامن عشر."`

### 🔄 Arabic Text Processing

#### **Morphological Analysis**
```python
# Arabic text normalization
def normalize_arabic_text(text):
    # Remove diacritics
    text = remove_diacritics(text)
    
    # Normalize letter variations
    text = re.sub(r'[آأإ]', 'ا', text)  # Alef variations
    text = re.sub(r'ة', 'ه', text)      # Teh Marbuta
    text = re.sub(r'[ىي]', 'ي', text)   # Yeh variations
    
    return text
```

#### **RTL Text Segmentation**
```python
# Arabic-aware text segmentation
def arabic_word_segmentation(text):
    # Handle RTL text boundaries
    words = text.split()
    
    # Process Arabic-specific features
    processed_words = []
    for word in words:
        # Remove punctuation but preserve Arabic text
        cleaned = ''.join(c for c in word 
                         if c.isalnum() or '\u0600' <= c <= '\u06FF')
        if cleaned:
            processed_words.append(cleaned)
    
    return processed_words
```

### 🎯 Arabic Diagnostic Tests

#### **1. Similarity Evaluation with Arabic**
```bash
# Generate Arabic needle-question pairs
python -c "
from evaluate_similarity import NeedleQuestionSimilarityEvaluator
evaluator = NeedleQuestionSimilarityEvaluator()
pairs = evaluator.generate_needle_question_pairs(['ar'], 10)
for pair in pairs[:3]:
    print(f'Needle: {pair[\"needle\"]}')
    print(f'Question: {pair[\"question\"]}')
    print(f'Similarity: {evaluator.compute_similarity(pair[\"needle\"], pair[\"question\"]):.3f}')
"
```

#### **2. Arabic Distractor Injection**
```bash
# Test Arabic distractor resistance
python -c "
from inject_distractors import DistractorInjectionFramework, DistractorType, DistractorConfig

framework = DistractorInjectionFramework()
arabic_haystack = 'الطبيب يعمل في المستشفى الكبير في الرياض.'
arabic_needle = 'الطبيب يعمل في المستشفى'

config = DistractorConfig(
    distractor_type=DistractorType.SEMANTIC_SIMILAR,
    injection_ratio=0.3,
    language='ar'
)

result = framework.inject_distractors(arabic_haystack, arabic_needle, config)
print(f'Original: {result.original_haystack}')
print(f'Modified: {result.modified_haystack}')
"
```

#### **3. Arabic Semantic Alignment**
```bash
# Analyze Arabic semantic coherence
python -c "
from compute_needle_haystack_similarity import NeedleHaystackSemanticAlignment

analyzer = NeedleHaystackSemanticAlignment()
arabic_needle = 'أحمد طبيب'
arabic_haystack = 'أحمد طبيب ماهر يعمل في المستشفى الرئيسي في الرياض.'

result = analyzer.compute_semantic_alignment(arabic_needle, arabic_haystack, 'ar')
print(f'Alignment Score: {result.alignment_score:.3f}')
print(f'Max Similarity: {result.max_similarity:.3f}')
print(f'Interference Level: {result.interference_level}')
"
```

#### **4. Arabic Structure Sensitivity**
```bash
# Test Arabic structure manipulation
python -c "
from shuffle_haystack import HaystackStructureSensitivity, ShuffleStrategy, ShuffleConfig

tester = HaystackStructureSensitivity()
arabic_haystack = 'الطبيب أحمد يعمل في المستشفى. المستشفى كبير وحديث.'
arabic_needle = 'أحمد طبيب'

config = ShuffleConfig(
    strategy=ShuffleStrategy.SENTENCE_SHUFFLE,
    shuffle_ratio=0.5,
    language='ar'
)

result = tester.apply_shuffle_strategy(arabic_haystack, arabic_needle, config)
print(f'Original: {result.original_haystack}')
print(f'Shuffled: {result.shuffled_haystack}')
print(f'Structure Change: {result.structure_change_score:.3f}')
"
```

#### **5. Arabic Repetition Robustness**
```bash
# Test Arabic repetition handling
python -c "
from repeat_task import RepeatedWordsReplicationTask, RepetitionStrategy, RepetitionConfig

task = RepeatedWordsReplicationTask()
arabic_haystack = 'الطبيب أحمد يعمل في المستشفى الكبير.'
arabic_needle = 'أحمد'

config = RepetitionConfig(
    strategy=RepetitionStrategy.PARAPHRASED_REPETITION,
    repetition_count=3,
    language='ar'
)

result = task.apply_repetition_strategy(arabic_haystack, arabic_needle, config)
print(f'Original: {result.original_haystack}')
print(f'Repeated: {result.repeated_haystack}')
print(f'Inflation Ratio: {result.content_inflation_ratio:.2f}')
"
```

## Usage Examples

### Basic Multilingual Evaluation
```bash
# Run diagnostic evaluation with Arabic support
python diagnostic_evaluator.py --languages en id ar --evaluations similarity distractor alignment

# Generate multilingual test scenarios
python -c "
from evaluate_similarity import NeedleQuestionSimilarityEvaluator
evaluator = NeedleQuestionSimilarityEvaluator()
pairs = evaluator.generate_needle_question_pairs(['en', 'id', 'ar'], 20)
print(f'Generated {len(pairs)} multilingual pairs')
"
```

### Arabic-Specific Demo
```bash
# Run comprehensive Arabic diagnostic demo
python demo_diagnostic_system.py

# This will demonstrate:
# - Arabic needle-question similarity evaluation
# - Arabic distractor injection and resistance
# - Arabic semantic alignment analysis
# - Arabic structure sensitivity testing
# - Arabic repetition robustness evaluation
# - Multilingual dashboard with Arabic metrics
```

### Integration Testing
```bash
# Run Arabic integration tests
python test_diagnostic_integration.py

# Test specific Arabic features
python -c "
import unittest
from test_diagnostic_integration import TestArabicLanguageSupport

suite = unittest.TestLoader().loadTestsFromTestCase(TestArabicLanguageSupport)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
"
```

## Arabic Performance Metrics

### RTL Text Handling Metrics
- **BiDi Compliance Score**: Bidirectional text algorithm adherence
- **Rendering Accuracy**: Proper Arabic script rendering
- **Mixed Content Handling**: Arabic-Latin text processing
- **Direction Marker Support**: Unicode RTL/LTR override handling

### Cultural Context Adaptation
- **Religious Context**: Islamic terminology and concepts
- **Historical Context**: Middle Eastern historical references
- **Modern Context**: Contemporary Arabic usage
- **Technical Context**: Arabic technical terminology

### Script Variation Robustness
- **Diacritic Handling**: Performance with/without vowel marks
- **Letter Normalization**: Handling of letter variations
- **Contextual Forms**: Connected vs isolated letter processing
- **Punctuation Processing**: Arabic-specific punctuation marks

## Dashboard Visualizations

### Multilingual Performance Charts
```python
# Create multilingual performance comparison
from diagnostic_dashboard import DiagnosticDashboard

dashboard = DiagnosticDashboard()
multilingual_chart = dashboard.create_multilingual_performance_chart()
arabic_analysis = dashboard.create_arabic_specific_analysis()
rtl_metrics = dashboard.create_rtl_text_analysis()
```

### Arabic-Specific Analytics
- **RTL Performance Trends**: Performance across text lengths
- **Script Variation Impact**: Accuracy with different Arabic forms
- **Cultural Context Sensitivity**: Performance by content type
- **Cross-lingual Comparison**: Arabic vs other languages

## Advanced Features

### Custom Arabic Content Generation
```python
# Generate custom Arabic test content
arabic_topics = {
    'علوم': ['بحث', 'دراسة', 'تجربة', 'نظرية'],
    'تاريخ': ['سنة', 'قرن', 'حرب', 'إمبراطورية'],
    'جغرافيا': ['دولة', 'مدينة', 'جبل', 'نهر'],
    'ثقافة': ['فن', 'موسيقى', 'أدب', 'تقليد']
}

# Use in semantic clustering and topic-based tests
```

### RTL Text Validation
```python
# Validate RTL text processing
def validate_rtl_processing(text):
    return {
        'is_rtl': True,
        'has_mixed_direction': check_mixed_content(text),
        'bidi_compliance': validate_bidi_algorithm(text),
        'rendering_issues': detect_rendering_problems(text)
    }
```

## Integration with Red Team Module

The diagnostic module seamlessly integrates with the LIMIT-GRAPH red team evaluation:

```python
# Combined red team and diagnostic evaluation
from redteam.enhanced_evaluator import EnhancedRedTeamEvaluator

evaluator = EnhancedRedTeamEvaluator(languages=['en', 'id', 'ar'])
results = evaluator.evaluate_enhanced(agent, scenarios, {
    'enable_arabic_rtl': True,
    'enable_cross_lingual': True,
    'context_lengths': [100, 500, 1000, 2000]
})
```

## Troubleshooting

### Common Arabic Processing Issues
1. **Encoding Problems**: Ensure UTF-8 encoding for Arabic text
2. **RTL Display**: Use proper RTL-aware text rendering
3. **Diacritic Handling**: Normalize diacritics for consistent processing
4. **Mixed Content**: Handle Arabic-Latin mixed text properly

### Performance Optimization
- **Caching**: Cache Arabic text normalization results
- **Batch Processing**: Process multiple Arabic texts together
- **Memory Management**: Optimize for Arabic text processing

## Contributing

When adding new Arabic language features:
1. Follow RTL text processing best practices
2. Include cultural context awareness
3. Add comprehensive test cases
4. Update documentation with Arabic examples
5. Ensure BiDi compliance for mixed content

## License

This module is part of the LIMIT-GRAPH project on AI Research Agent Team and follows the same licensing terms.