# Beyond the Final Layer: Intermediate Representations for Better Multilingual Calibration in Large Language Models

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Confidence calibration, the alignment of a model's predicted confidence with its actual accuracy, is crucial for the reliable deployment of Large Language Models (LLMs). However, this critical property remains largely under-explored in multilingual contexts. In this work, we conduct the first large-scale, systematic studies of multilingual calibration across six model families and over 100 languages, revealing that non-English languages suffer from systematically worse calibration. To diagnose this, we investigate the model's internal representations and find that the final layer, biased by English-centric training, provides a poor signal for multilingual confidence. In contrast, our layer-wise analysis uncovers a key insight that late-intermediate layers consistently offer a more reliable and better-calibrated signal. Building on this, we introduce a suite of training-free methods, including Language-Aware Confidence Ensemble (LACE), which adaptively selects an optimal ensemble of layers for each specific language. Our study highlights the hidden costs of English-centric alignment and offer a new path toward building more globally equitable and trustworthy LLMs by looking beyond the final layer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a comprehensive study on the calibration of LLMs, revealing that non-English languages suffer from poor calibration. A layer-by-layer analysis indicates that this issue stems from an English bias in the final layer, while intermediate layers closer to the output provide better-calibrated signals for other languages. The authors introduce LACE, a training-free approach that combines these optimal intermediate layers for each language. Experiments on MMMLU and Belebele demonstrate that LACE significantly improves calibration accuracy and enhances existing post-hoc methods.

### Strengths
1. This paper tackles the critical issue of multilingual LLM reliability at a landmark scale (6 models, 100+ languages) using rigorous, human-curated data.
2. The main finding is both clear and actionable: intermediate layers are better calibrated for non-English languages, which provides a concrete explanation for the observed language bias.
3. The experimental design is solid. 
4. The proposed LACE method is practical, effective, and works with existing post-hoc calibrators, making it highly valuable.

### Weaknesses
1. The scope is limited to multiple-choice questions. We don't know if this intermediate-layer advantage holds for open-ended generation, which is a significant gap given how LLMs are actually used.
2. LACE requires labeled validation sets for each language, which could be a significant barrier for low-resource languages that need this method the most. The data requirement may limit the widespread applicability of the solution in real-world settings.
3. The "English-bias" hypothesis doesn't fully explain why "multilingual-first" models like Aya also show better intermediate-layer calibration.

### Questions
1. Could you speculate on a path forward for testing this hypothesis on generative tasks (e.g., correlating layer-wise probabilities with human judgments)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper "Beyond the Final Layer: Intermediate Representations for Better Multilingual Calibration in Large Language Models" investigates the calibration of multilingual large language models (LLMs) and highlights the disparities in performance between English and non-English languages. Overall, the study emphasizes the importance of considering intermediate representations for enhancing the reliability of multilingual LLMs.

### Strengths
1. The paper introduces a fresh perspective on the calibration of multilingual LLMs by emphasizing the importance of intermediate layers rather than solely relying on the final layer for confidence estimation. This approach challenges established norms in the field and provides a new framework for understanding how different layers contribute to model performance.

2. The research is grounded in a comprehensive empirical analysis that utilizes high-quality datasets spanning over 100 languages. The methodology is rigorously designed, employing well-defined metrics to evaluate model performance.

### Weaknesses
1. LACE is a very interesting approach. I’m curious whether optimizing for a specific language benefits similar languages. For instance, if we optimize for Japanese, would it improve calibration for Korean or negatively impact English understanding? Exploring this could shed light on cross-lingual transfer effects.

2. The LACE experiment is conducted within a single dataset. I wonder if optimizing LACE on one dataset could enhance model performance on another for a specific language. This would help determine whether the method generalizes across datasets rather than being dataset-specific.

3. As mentioned in the limitations, all evaluated models are relatively small. It remains unclear whether the conclusions hold for larger models (e.g., >30B parameters). Investigating this would strengthen the findings.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper conducts a large-scale benchmarking study of multilingual calibration in LLMs (six model families; 100+ languages) on MMMLU and Belebele, documenting a substantial English vs. non-English calibration gap and a mid-layer “sweet spot” where ECE improves for many non-English languages. Building on this descriptive finding, the authors propose training-free confidence extractors (Best Layer, Good-Layers Ensemble) and LACE (a language-aware ensemble), sometimes combined with post-hoc calibration (temperature / isotonic), which reduce ECE on these benchmarks.

### Strengths
Timely and important problem: Multilingual calibration is underexplored and high-impact for safety/trust. The English vs. non-English gap is clearly quantified across models/datasets.

Simple, training-free methods: Best-Layer / Good-Layers / LACE are conceptually simple, require no retraining, and still deliver large ECE reductions, while remaining complementary to standard post-hoc calibration.

### Weaknesses
Limited ablation/depth of analysis. I could not find ablations on: (i) ECE binning sensitivity, (ii) computing calibration using each layer’s own prediction vs. probability on the final prediction, (iii) robustness to language-ID errors / code-switching (needed for LACE), and (iv) runtime/compute overhead of multi-layer scoring. The paper’s “Benchmarking” focus and method section do not report these studies. 

Narrow scope and stated limitations. Experiments are restricted to 7B–8B models and MCQA; the authors explicitly leave open-ended generation to future work. Methods are post-hoc rather than integrated into training. These limitations make it hard to generalize the descriptive findings beyond the evaluated setting. 


Theoretical grounding is thin. The paper offers a plausible narrative (final layer English-biased; intermediate layers more language-agnostic) but no formal analysis or causal probes explaining why the mid-layer sweet spot emerges. The results remain mostly empirical/observational.

### Questions
Ablations: How sensitive are ECE gains to the number/strategy of bins?

Layer-own predictions: If ECE/Brier are computed using each layer’s own argmax, do the mid-layer gains persist? 


Language-ID robustness: How does LACE behave under language detection errors or code-switching? Any language-agnostic variant? 


Cost/latency: What is the runtime overhead of Good-Layers / LACE vs. final-layer baselines?

Beyond MCQA: Any evidence on open-ended tasks (long-form QA, summarization) to show the phenomenon carries over?

### Soundness
2

### Presentation
3

### Contribution
2
