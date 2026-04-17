# 11Plus-Bench: Demystifying Multimodal LLM Spatial Reasoning with Cognitive-Inspired Analysis

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
For human cognitive process, spatial reasoning and perception are closely entangled, yet the nature of this interplay remains underexplored in the evaluation of multimodal large language models (MLLMs). While recent MLLM advancements show impressive performance on reasoning, their capacity for human-like spatial cognition remains an open question. In this work, we introduce a systematic evaluation framework to assess the spatial reasoning abilities of state-of-the-art MLLMs relative to human performance.  Central to our work is 11Plus-Bench, a high-quality benchmark derived from realistic standardized spatial aptitude tests. 11Plus-Bench also features fine-grained human expert annotations of both perceptual complexity and reasoning process. These annotations allow us to move beyond aggregated accuracy, enabling an instance-level, parallel analysis of human and machine cognitive profiles with predictive power. Through extensive experiments across 14 MLLMs and human evaluation, we find that current MLLMs exhibit early signs of spatial cognition. We find both convergence and divergence: while both human and MLLM cognitive effort, measured by response time and tokens generated respectively, correlates with reasoning complexity, their underlying mechanisms differ. Human correctness is highly predictable and shaped by abstract pattern complexity, whereas instance-level MLLM performance remains weakly predictable and sensitive to low-level perceptual features. Our work provides a precise characterization of the emerging yet brittle spatial reasoning in MLLMs, offering actionable insights for developing more human-like spatial intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces 11PLUS-BENCH, a new benchmark for evaluating spatial reasoning in Multimodal Large Language Models (MLLMs), grounded in standardized human spatial aptitude tests (e.g., 11+ exams). Unlike prior benchmarks that report only aggregate accuracy, this work provides fine-grained cognitive annotations—including perceptual complexity (e.g., pattern component counts), reasoning steps (e.g., spatial manipulation, logical deduction), and image-level bounding boxes. The authors conduct human evaluations (N=3) with response time as a proxy for cognitive load and compare performance across 14 MLLMs (open- and closed-source).

Key contributions include:

1. A cognitively inspired evaluation framework enabling instance-level, parallel analysis of human and MLLM "cognitive profiles."
2. Empirical findings showing that while MLLMs exhibit non-random performance correlated with human difficulty, their instance-level correctness is unpredictable and driven more by low-level visual features (e.g., image resolution) than abstract reasoning.
3. Evidence of divergent cognitive mechanisms: human accuracy is shaped by pattern complexity, whereas MLLM behavior is noisy and sensitive to superficial cues.
4. A critique of the common “single composite image” evaluation format, showing that separate image inputs yield more reliable assessment of spatial reasoning.

The work concludes that current MLLMs show early but brittle signs of spatial cognition, lacking the structured, robust reasoning seen in humans.

### Strengths
The paper is highly original in its integration of cognitive science methodology into MLLM evaluation. Rather than treating spatial reasoning as a monolithic skill, it decomposes it using psychometrically validated constructs (e.g., Spatial Visualization, Flexibility of Closure) and introduces expert-annotated cognitive features that disentangle perception from reasoning. The idea of building predictive models of correctness using these features—and comparing their explanatory power across humans and models—is novel and insightful.

The benchmark construction is rigorous: dual sourcing (public + proprietary), expert annotation by STEM-trained individuals, high inter-annotator agreement, and contamination controls (e.g., >50% of data lacks public answers). Human evaluation follows cognitive science norms (response time as a cognitive load proxy). The experimental design compares two input formats and includes both open- and closed-source models, enhancing validity. Statistical analyses (SHAP, regression, significance testing) are appropriate and well-executed.

The paper is exceptionally well-structured and clearly written. Figures 1, 4, and 6–8 effectively communicate the framework, results, and qualitative insights. Technical details (e.g., atomic reasoning operations, pattern complexity metrics) are precisely defined in the appendix. The narrative consistently links empirical findings back to the core question: Do MLLMs reason spatially like humans?

This work addresses a critical gap in MLLM evaluation: the lack of human-aligned, process-oriented spatial reasoning benchmarks. By revealing that MLLM performance is unpredictable at the instance level and driven by low-level features, it challenges assumptions of “emergent spatial intelligence” in current models. The benchmark and methodology offer a foundation for future work aiming to build more robust, human-like spatial reasoning systems—relevant to robotics, education, and scientific discovery.

### Weaknesses
1. Ambiguity in “Reasoning Steps” Annotation: The paper defines atomic operations (e.g., Spatial Manipulation, Logical Deduction), but it’s unclear how annotators decomposed multi-step problems into sequences. Were reasoning chains unique per instance? How was subjectivity controlled beyond category selection?

2. Underexplored Model “Thinking” Data: The paper notes that Gemini 2.5 Pro provides token-level “thinking” traces, but only uses total token count as a cognitive load proxy. A deeper analysis of content in these traces (e.g., do models verbalize spatial manipulations correctly?) could strengthen claims about reasoning mechanisms.

3. Task Coverage Bias: The benchmark heavily emphasizes 2D paper-folding (229 public examples) and cube nets (201), while other tasks (e.g., 3D rotation) have very few samples (<10). This may skew conclusions about MLLM weaknesses (e.g., in Flexibility of Closure).

### Questions
1. Annotation Protocol for Reasoning Chains: For tasks requiring multiple reasoning steps (e.g., paper folding), how did annotators determine the sequence and number of atomic operations? Was there a maximum step limit? Could you share an example of a full annotated reasoning chain vs. what a model generated?

2. Model Architecture Confounds: The paper groups models by open/closed source, but differences in vision encoders (e.g., CLIP vs. proprietary) or training data may explain performance gaps more than “spatial reasoning ability.” Did the authors control for these factors, or consider ablations (e.g., same LLM + different vision backbones)?

3. Contamination Risk in Public Data: While >50% of data lacks golden answers, the public set was crawled from the web using spatial reasoning keywords. Could high-performing closed-source models have seen similar problems during training? How was this risk quantified beyond “no golden answers”?

4. Actionable Pathways: The conclusion states MLLMs lack “compositional understanding.” What specific architectural or training changes does the authors’ analysis suggest? For instance, would explicit spatial operation modules (e.g., differentiable renderers) help, or is the issue more fundamental (e.g., lack of mental simulation)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
11Plus-Bench introduces a cognitively-inspired benchmark for evaluating spatial reasoning in Multimodal Large Language Models (MLLMs), grounded in standardized human spatial aptitude tests. The benchmark features fine-grained expert annotations of perceptual complexity and reasoning steps, enabling parallel, instance-level analysis of human and model cognitive profiles. Experiments across 14 MLLMs and human participants reveal that while models show early signs of spatial cognition, their performance is largely random and sensitive to low-level perceptual features, in contrast to humans whose accuracy is shaped by abstract pattern complexity and structured reasoning.

### Strengths
- Fine-Grained Annotation: Provides detailed, expert-annotated cognitive features (perceptual complexity, reasoning steps), enabling nuanced, instance-level analysis.
- Human-Machine Comparison: Enables direct, parallel analysis of human and model cognitive profiles, moving beyond aggregate accuracy to predictive modeling of correctness and cognitive effort - presenting a sound and elaborate testing framework

### Weaknesses
- Limited Scale and Diversity: The public set is relatively small (824 examples), and the private set (91) is not open, potentially limiting generalizability and reproducibility.
- Lack of data contamination study : 
   - Given the static nature of the dataset and no guardrails/process in place for mitigating/studying data contamination, the benchmark runs the risk of dilution on all metrics provided. There is a need for an elaborate data contamination study and a plan for its mitigation.
    - While the benchmark claims to minimize contamination (using private, non-public test items), there is no empirical audit or adversarial check to quantify or guarantee the absence of contamination in model pretraining data.
- Human Baseline Scope: Human evaluation is limited to three participants, raising questions on reliability
- Qualitative Analysis: The paper could benefit from more qualitative error analysis or case studies to illustrate specific model failure modes.
- Lack of an exhaustive list of models for evaluation - key opensource models like DeepSeek,Kimi and Llama families are missing from evaluation

### Questions
Major flags covered in the weakness section

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces 11PLUS-BENCH, a new benchmark inspired by human cognitive tests to evaluate the spatial reasoning of Multimodal Large Language Models (MLLMs). Featuring fine-grained annotations of perceptual and reasoning complexity, the framework enables a detailed, instance-level comparison between machine and human cognitive profiles. Experiments across 14 MLLMs reveal that while human performance is predictable and shaped by abstract complexity, MLLM performance is brittle and sensitive to low-level visual cues. The work concludes that current MLLMs show early signs of spatial cognition but lack the robust, compositional understanding seen in humans.

### Strengths
1.  **Creation of a High-Quality Benchmark :** The paper introduces a new benchmark derived from standardized psychometric tests (11+ exams), which is a strong foundation. This ensures the tasks are well-vetted for isolating specific spatial reasoning skills and are not confounded by commonsense knowledge or linguistic ambiguity often present in other VQA datasets. The inclusion of a private test split sourced from commercial providers is a significant strength, directly addressing the critical issue of data contamination in an era where web-crawled data is likely part of MLLM training sets.

2.  **Introduction of a Fine-Grained, Multi-Feature Annotation Framework:** A strength is moving beyond simple aggregate accuracy. The paper introduces a set of cognitive annotations, including `visual perception complexity` and `general reasoning process`. This framework enables an instance-level analysis, allowing researchers to investigate *why* a model might fail on a specific problem (e.g., high perceptual load vs. complex reasoning chain). This is a conceptual leap from traditional benchmarks that only report overall scores.

3.  **Novel Cognitive-Inspired Evaluation Framework:** The paper's core proposal—to conduct a *parallel analysis of human and machine cognitive profiles*—is a powerful and novel approach. It frames the evaluation not just as measuring performance, but as comparing the underlying mechanisms and failure modes. By attempting to model what features predict success and cognitive effort for both humans and MLLMs, the paper provides a blueprint for a more explanatory and insightful form of AI evaluation.

4.  **Systematic Critique of Existing Evaluation Paradigms:** The paper provides a valuable empirical critique of the common "single composite image" evaluation method used in other benchmarks. By showing that advanced models perform significantly better when presented with separate, cropped images, it demonstrates that previous results might conflate reasoning failures with more basic visual parsing challenges. This is a crucial methodological insight for the field.

### Weaknesses
*   **Subjectivity in Reasoning Process Annotation:** The annotation of "General Reasoning" requires experts to decompose a thought process into a sequence of predefined atomic operations (e.g., *Pattern Matching, Spatial Manipulation*). This methodology has two key problems:
    1.  **It assumes a discrete, serial reasoning process:** Human spatial reasoning can be holistic and parallel, not necessarily a step-by-step symbolic procedure. This annotation forces a potentially artificial structure onto a fluid cognitive process.
    2.  **It is inherently subjective:** While inter-annotator agreement was high, the decomposition itself is an interpretation. Different experts might conceptualize the steps differently, and the predefined categories may not perfectly capture the nuances of human thought.

*   **Oversimplification of Visual Perception Complexity:** "Pattern complexity" is quantified by counting atomic components like lines or surfaces (lines 202-205). This objective metric ignores the Gestalt principles of perception. For example, a complex pattern with high symmetry might be perceptually easier to process than a simpler but asymmetrical one. The metric is a simplification that may not fully capture the true perceptual load on a human or a model.

*   **Limited Scope of Spatial Capabilities:** The authors select three core capabilities (Spatial Relation & Orientation, Spatial Visualization, Flexibility of Closure) from a much broader spectrum of human spatial intelligence. They justify this by excluding skills less relevant to current MLLM architectures (e.g., kinesthetic reasoning). However, this selection may still be too narrow to make sweeping claims about MLLM spatial cognition as a whole. The findings might not generalize to other important spatial tasks like navigation, mental mapping, or complex 3D assembly.

*   **Potential Overstatement of MLLM "Randomness":** The paper claims that "instance-level MLLM performance remains largely random." However, their own results show some structure. Figure 3(b) demonstrates a positive correlation between human and MLLM accuracy, suggesting performance is not entirely random across difficulty levels. Furthermore, in the predictability analysis (Figure 7), some classifiers achieve F1 scores with statistically significant p-values (e.g., Gemini 2.5 Pro's p=0.0002), indicating a performance better than chance, even if the predictive power is weak. The narrative of "randomness" might be overstated for rhetorical effect.

*   **Lack of Systematic Error Typology:** The analysis successfully identifies which *features* predict failure (e.g., pattern complexity for humans, image resolution for MLLMs). However, it does not provide a systematic analysis of the *types* of errors models make. For instance, do MLLMs consistently fail at mental rotation beyond 90 degrees? Do they confuse reflection with rotation? The qualitative examples in Figure 8 are illustrative but not comprehensive. A categorized breakdown of error types would provide more genuinely "actionable insights" for improving models.

*   **Weak Analogy Between Human Response Time and MLLM Token Count:** The paper uses human response time as a proxy for cognitive effort and compares it to the number of tokens generated by MLLMs. This analogy is tenuous. Human response time reflects complex neural processing, memory access, and serial deliberation. MLLM token count, especially in a chain-of-thought process, is more a measure of verbosity, which can be heavily influenced by prompting, fine-tuning, and architectural biases, not necessarily correlating directly with computational "effort" or reasoning depth. The finding that token count and accuracy are uncorrelated in MLLMs is interesting but may not be directly comparable to the speed-accuracy trade-off in humans.

*   **Limited Exploration of Prompting:** The paper uses simple, fixed prompts for the MLLMs (Tables 2 & 3). MLLM performance is known to be highly sensitive to prompt engineering. The methodology does not explore whether different prompting strategies (e.g., Chain-of-Thought, asking the model to verbalize its spatial transformations) could have elicited better performance. Therefore, the results may reflect the limitations of the chosen prompts as much as the models' intrinsic capabilities.

### Questions
Please answer the points mentioned in the weakness section

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces 11PLUS-Bench, a benchmark designed to evaluate spatial reasoning in multimodal large language models (MLLMs). Unlike existing benchmarks that rely on aggregate accuracy, 11PLUS-Bench provides instance-level cognitive feature annotations to enable predictive modeling of both human and machine responses. The benchmark includes 824 public and 91 private examples derived from realistic 11+ standardized spatial aptitude tests, annotated by domain experts for three key cognitive dimensions, Spatial Relation & Orientation (SRO), Spatial Visualization (SV), and Flexibility of Closure (FoC). Evaluation across 14 MLLMs (including GPT-4o, Gemini-2.5-Pro, and Qwen-VL-2.5) and human baselines reveals a significant gap which reveals models lag far behind human accuracy (humans ≈ 80–85%, best MLLMs ≈ 30-40%) and exhibit random, poorly predictable instance-level behavior. Analysis using SHAP values and regression models finds that human correctness is primarily driven by pattern complexity and reasoning steps, while MLLM behavior is overly influenced by low-level perceptual features such as image resolution or bounding-box layout. The paper concludes that although MLLMs show early signs of spatial cognition, their reasoning remains unstable, unstructured, and far from human-like generalization.

### Strengths
* Benchmark design explicitly ties to psychometric theory (Carroll’s Three-Stratum model, spatial-visualization literature), ensuring conceptual validity.
* Introduces interpretable cognitive features (pattern complexity, reasoning type, perceptual load) enabling predictive modeling rather than aggregate scoring.
* Uses response time vs. token length to measure cognitive effort and compares predictive profiles using Random Forest and SHAP analyses.
* Reports strong annotation agreement (Pearson ≈ 0.8) and statistically significant human–model correlations in difficulty perception.
* Demonstrates that MLLMs show partial emergence of spatial cognition but are overly dependent on low-level cues.
* Private test data from commercial providers ensures low contamination to facilitate ethical and reproducibility.

### Weaknesses
* There are only three participants and a broader demographic diversity would strengthen validity.
* ~900 items may still be modest for modern MLLM evaluation and a larger set could increase statistical reliability.
* All tasks are multiple-choice which lacks open-ended reasoning or diagram generation tasks that test richer cognitive processes.
* Some key comparisons (e.g., per-model significance tests) are summarized without full confidence intervals.

### Questions
1. Was there a reason as to the limit to three participants?
2. For MLLMs, do prompt formulations (e.g., “select the best option” vs. “which image matches?”) affect accuracy?
3. Can the authors classify common model failure modes (e.g., mis-segmentation vs. transformation confusion)?
4. Could fine-tuning on annotated reasoning steps yield measurable improvements in predictive alignment with human profiles?

### Soundness
3

### Presentation
3

### Contribution
3
