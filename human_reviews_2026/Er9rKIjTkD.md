# SANEval: Open-Vocabulary Compositional Benchmarks with Failure-mode Diagnosis

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
The rapid progress of text-to-image (T2I) models has unlocked unprecedented creative potential, yet their ability to faithfully render complex prompts involving multiple objects, attributes, and spatial relationships remains a significant bottleneck. Progress is hampered by a lack of adequate evaluation methods; current benchmarks are often restricted to closed-set vocabularies, lack fine-grained diagnostic capabilities, and fail to provide the interpretable feedback necessary to diagnose and remedy specific compositional failures. We solve these challenges by introducing **SANEval** (Spatial, Attribute, and Numeracy Evaluation), a comprehensive benchmark that establishes a scalable new pipeline for open-vocabulary compositional evaluation. SANEval combines a large language model (LLM) for deep prompt understanding with an LLM-enhanced, open-vocabulary object detector to robustly evaluate compositional adherence unconstrained by a fixed vocabulary. Through extensive experiments on six state-of-the-art T2I models, we demonstrate that SANEval's automated evaluations provide a more faithful proxy for human assessment; our metric achieves a Spearman's rank correlation with statistically different results than that of existing benchmarks across tasks of attribute binding, spatial relations, and numeracy. To facilitate future research in compositional T2I generation and evaluation, we will release the SANEval dataset and our open-source evaluation pipeline.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SANEval, a new benchmark designed to evaluate compositional text-to-image (T2I) generation along three axes: spatial relations, attribute binding, and numeracy. The core innovation is an open-vocabulary evaluation pipeline that combines a language model-based prompt understanding module with an LLM-enhanced object detection module (using YOLO-E) to handle objects beyond fixed vocabularies. SANEval provides not just quantitative scores but also interpretable, structured feedback diagnosing missing, spurious, or misbound elements. The authors evaluate six state-of-the-art T2I models and demonstrate that SANEval aligns better with human judgment than existing closed-vocabulary benchmarks such as CompBench++.

### Strengths
1. Timely and relevant topic – Evaluation of compositional T2I generation is an important and underexplored problem, especially as generative models become more capable yet still fail on fine-grained prompt adherence.

2. Open-vocabulary focus – The attempt to overcome vocabulary limits of prior OD-based benchmarks (e.g., those tied to COCO classes) is a real step forward. Integrating LLM synonym reasoning with detection to match rare objects is a practical and impactful idea.

3. Structured feedback – Providing interpretable diagnostic feedback (e.g., missing or swapped objects) is valuable, addressing a longstanding complaint that most benchmarks yield only opaque scalar scores.

4. Thorough evaluation – The experiments cover multiple models, diverse prompt categories, and statistical comparisons with existing benchmarks. The degradation analysis with increasing object count is especially insightful.

5. Strong engineering – The system is well implemented, modular, and reproducible, with clear illustrations (e.g., Figures 1–4) explaining each evaluation component.

### Weaknesses
1. Incremental conceptual novelty – While well-engineered, the main idea, combining LLM reasoning with open-vocabulary detection for compositional evaluation, is a relatively straightforward extension of prior hybrid pipelines like Geneval or CompBench++. The paper oversells its conceptual novelty relative to what is essentially a systematic engineering improvement.

2. Dependence on proprietary APIs – The reliance on commercial APIs (Gemini 2.5 Flash, GPT, etc.) undermines claims of “open” benchmarking and reproducibility. The evaluation cannot be replicated without access to these closed systems.

3. Data quality concerns – Much of the dataset is LLM-generated and then “validated by humans,” but the validation process is not clearly described. It’s unclear how reliable or diverse the resulting prompts and labels are, especially for rare or ambiguous compositions.

4. Potential circularity – Because the same families of LLMs are used both to generate prompts and evaluate outputs, the benchmark risks circular evaluation biases. This is particularly problematic if the evaluated models share architectures or training corpora with the evaluators.

5. Limited human validation – The claim that SANEval correlates better with human judgments is asserted but not deeply substantiated. There’s no large-scale human evaluation or statistical analysis showing correlation coefficients with human preferences.

6. Interpretability claim is overstated – The feedback is structured and informative, but not necessarily “interpretable” in a cognitive sense. The system still relies on opaque LLM reasoning, so interpretability here is more syntactic than semantic.

7. Paper tone – The writing is overly confident and somewhat verbose, with repeated claims of “first,” “scalable,” and “comprehensive.” While the contributions are solid, they don’t quite justify that level of rhetoric.

### Questions
1. How robust is SANEval to the choice of LLMs or detectors? Have you tested with open-source alternatives to Gemini or GPT to ensure consistency?

2. Could you provide quantitative human evaluation results to validate your claim of stronger human correlation?

3. How do you handle prompts with ambiguous or relational adjectives (e.g., “a tall man next to a shorter man”)?

4. What mechanisms prevent SANEval from rewarding models that overfit to common visual priors rather than compositional fidelity?

5. How expensive (in terms of GPU or API calls) is the full benchmark run for one model?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The work proposes SANEval, an automated benchmark and scoring framework for evaluating compositional faithfulness in text-to-image generation. The core claim is that current evaluators either (i) rely on fixed vocabularies, which miss rare or fine-grained objects, or (ii) provide only coarse scalar scores, which are not actionable. SANEval addresses this by (1) decomposing prompts into explicit requirements (objects, attributes, spatial relations, and numeracy), (2) using synonym expansion plus an open-vocabulary detector to localize objects in generated images, (3) using a VLM to judge fine-grained attributes on detected crops, (4) checking spatial layout and object counts, and (5) generating structured, diagnostic feedback for each failure mode. The benchmark is applied to a suite of state-of-the-art text-to-image models, and per-dimension scores (attribute binding, spatial relationships, numeracy) are reported along with claims of improved alignment with human judgments and complementary behavior to prior benchmarks.

### Strengths
1. Problem importance. The paper focuses on a real bottleneck in current T2I systems: controllability. Capturing whether a model got “two red cars to the left of a blue bus” right is directly relevant to downstream productization and safety of generative vision systems. Framing spatial relations, numeracy, and attribute binding as three core controllability axes is well-motivated. 

2. Pipeline design / interpretability. The evaluation stack is modular and conceptually clean: prompt parsing → synonym expansion → open-vocabulary detection → attribute judgment on crops → spatial / count checks → textual feedback. This goes beyond a single scalar metric and produces human-readable explanations (“missing the second penguin,” “shirt color does not match prompt”), which is valuable both for benchmarking and for training-time improvement (e.g. RL with AI feedback). 

3. Attempt at open-vocabulary evaluation. Traditional benchmarks are constrained to a small, fixed label set (e.g. COCO categories). SANEval explicitly attempts to escape that bottleneck via synonym expansion and an open-world detector, claiming to evaluate long-tail objects and fine-grained attributes. This, if shown reliable, is a meaningful step forward for evaluation coverage. 

4. Multi-model comparison and failure characterization. The paper evaluates several strong text-to-image systems and surfaces distinct weaknesses (e.g., some models better at spatial layout, others better at numeracy, etc.). This makes SANEval look like a diagnostic tool rather than just a leaderboard generator. Tables in the main text highlight degradation under increasing prompt difficulty and differences across models.

### Weaknesses
1. Reproducibility and stability are underdeveloped.
The benchmark depends on proprietary or partially described components (e.g. Gemini-2.5-Flash for prompt parsing and attribute judgment, YOLO-E for open-vocabulary detection), some of which are not publicly reproducible. The paper promises release of data and code but does not convincingly demonstrate that the community will be able to run the full pipeline without access to closed-source commercial systems.

2. Limited validation of metric correctness.
The paper claims “strong alignment with humans,” but key details are missing: exact study size, annotator protocol, inter-annotator agreement, and per-dimension correlation (spatial / numeracy / attribute binding) between SANEval scores and human judgments at the image level. Current quantitative results emphasize differences across models and p-values vs. CompBench++, but do not provide clear, per-sample reliability numbers for SANEval itself. Without those, it remains unclear whether SANEval is actually accurate in judging success/failure, or just “plausible and convenient.”

3. Open-vocabulary claim is not yet airtight.
The core technical sell is that synonym expansion plus open-world detection solves “vocabulary mismatch.” However, no error analysis is reported for that step: How often does the system over-credit partial matches (“generic bird” instead of “albatross”)? How often does it under-credit genuinely correct rare objects because the synonym set was incomplete? This is an obvious potential criticism because it goes directly at the headline claim (“open-vocabulary evaluation”). 

4. No ablation / robustness analysis across pipeline stages.
The pipeline has multiple learned/modules stages, and failure in any stage could cascade. The paper does not report sensitivity to swapping the LLM in the Prompt Understanding Module, ablating synonym expansion, or replacing the attribute-judging VLM with an alternative. There is also no quantitative discussion of false positives / false negatives in spatial and numeracy checks caused by upstream detection errors. This makes it easy to argue the metric may be brittle, and therefore risky to trust for fine-grained leaderboard decisions.

5. Statistical framing against prior benchmarks is weak.
The comparison to CompBench++ uses significance tests on rank correlations to argue SANEval measures something different. Only reporting p-values, without effect sizes or the actual Spearman coefficients (ρ), invites criticism. A low correlation may mean “captures complementary aspects,” but it could also mean “noisy / inconsistent.” The paper currently does not disambiguate those possibilities.

6. Human prompts vs synthetic prompts.
Prompt sets are at least partially LLM-generated and curated. The paper does not quantify how similar these prompts are to natural user requests “in the wild.” A skeptical reader may ask whether the benchmark is indirectly overfit to LLM-style phrasing, and whether that inflates apparent evaluator reliability.

### Questions
1. Benchmark stability and openness
    - Will an end-to-end reference implementation using only openly available models (for prompt parsing, open-vocabulary detection, and attribute judging) be released?
    - If such a “SANEval-lite” pipeline is substituted for the proprietary backbone, how similar are model rankings and per-dimension scores?

2. Human agreement and study design

    - How large is the human evaluation set used for validating SANEval?

    - What instructions were annotators given for spatial, numeracy, and attribute correctness?

    - What inter-annotator agreement was observed?

    - What are the per-dimension image-level correlations (e.g., Spearman’s ρ / accuracy / F1) between SANEval and human judgments?

3. Error analysis for open-vocabulary claims

    - On a held-out subset with human-labeled boxes and attributes, what is precision/recall of SANEval’s object existence and attribute binding scores for long-tail categories?

    - How often does synonym expansion lead to false credit for “nearby but wrong” categories?

4. Robustness / ablations

    - How sensitive is SANEval to swapping the LLM used in the Prompt Understanding Module or the attribute-binding checker?

    - How sensitive are final scores to removing synonym expansion or constraining detection to a fixed-label detector (COCO-style)?

    - Can a single failing sub-module (e.g., missed detection) flip an otherwise-correct image from “pass” to “fail,” and how often does that happen?

6. Statistical interpretation
    - Table 3 reports significance values when comparing to CompBench++, but does not report actual effect sizes. Could the paper include rank correlation coefficients and a short interpretation (e.g., “ρ = 0.2 indicates weak monotonic agreement, suggesting that SANEval captures aspects of controllability that CompBench++ does not emphasize”)?

### Soundness
2

### Presentation
2

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
This paper presents SANEval, a new benchmark for evaluating compositional faithfulness in text-to-image (T2I) generation models. Unlike existing benchmarks that depend on closed-set object vocabularies or yield opaque single-number scores, SANEval introduces an open-vocabulary, diagnostic, and interpretable evaluation pipeline. It consists of two core modules: a Prompt Understanding Module that uses an LLM to extract objects, attributes, spatial relationships, and quantities from prompts, and an Enhanced Object Detection Module that combines an open-vocabulary detector (YOLO-E) with LLM-based synonym reasoning to reduce vocabulary mismatch. SANEval evaluates images along three axes—attribute binding, spatial relationships, and numeracy—producing not only quantitative scores but also fine-grained, structured feedback that identifies missing, hallucinated, or incorrectly bound elements. Experiments conducted on six state-of-the-art T2I models show that SANEval aligns more closely with human judgment than prior metrics and captures compositional failure modes that existing benchmarks often miss. The authors further commit to releasing the dataset, evaluation pipeline, and annotations to support future research.

### Strengths
Strong diagnostic capability:
SANEval goes beyond providing a single score—it outputs structured, interpretable feedback, explicitly identifying missing objects, incorrect attribute bindings, and count mismatches. This makes it highly useful for debugging and improving T2I systems.

Open-source commitment:
The authors plan to release the dataset, prompts, annotations, and full evaluation pipeline, which will greatly facilitate reproducibility and help standardize compositional evaluation in the community.

### Weaknesses
Limited robustness analysis:
The paper does not thoroughly examine how LLM parsing errors or object detection failures (e.g., hallucinations or missed detections) propagate through the pipeline and affect final scoring reliability.

High computational cost:
The evaluation pipeline requires multiple rounds of LLM calls and YOLO-E inference per image, which may make it expensive and impractical for large-scale evaluation on millions of samples.

Insufficient prompt diversity:
The dataset’s ~5000 prompts are largely synthetically or structurally composed, lacking coverage of real-world human-written instructions with figurative language, emotional tone, ambiguous phrasing, or complex logical relationships.

### Questions
LLM reliability
How do you ensure that the LLM does not introduce hallucinations or misinterpretations during prompt parsing or attribute evaluation? Have you considered combining rule-based logic with LLMs (a hybrid approach) to reduce such errors?

Object detection uncertainty
If the detector fails to identify a small, occluded, or stylized object, could the system incorrectly penalize a correct image? Do you plan to incorporate uncertainty estimation or confidence propagation to handle these cases?

Generalization to real-world prompts
Can you demonstrate how SANEval performs on natural, human-written prompts involving negation, multiple clauses, metaphors, or conditional instructions—rather than primarily template-based prompts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SANEval, an open-vocabulary benchmark for evaluating compositional reasoning in text-to-image (T2I) models. The authors argue that current benchmarks rely on closed-set vocabularies and provide limited diagnostic feedback. To address this, they design a modular evaluation framework combining a large language model for prompt understanding with an open-world object detector, enabling fine-grained assessment of spatial relations, attribute binding, and numeracy. SANEval outputs both quantitative scores and interpretable feedback to pinpoint specific compositional errors. The benchmark is validated against human annotations and shows complementary insights compared to prior methods like CompBench++, establishing it as a scalable and diagnostic tool for T2I evaluation.

### Strengths
* The paper introduces a well-structured benchmark that separately evaluates spatial reasoning, attribute binding, and numeracy, offering a more interpretable breakdown of compositional performance than prior holistic metrics.

* By integrating LLM-based synonym expansion with an open-world detector (YOLO-E), SANEval effectively overcomes the fixed-class limitations of existing object-detection-based benchmarks.

* The framework provides structured, human-readable feedback that identifies missing, incorrect, or extra objects and attributes, making it highly useful for model debugging. The benchmark demonstrates statistically distinct and complementary insights compared to established baselines, showing that SANEval captures novel aspects of compositional reasoning.

### Weaknesses
- The benchmark heavily relies on proprietary LLMs (e.g., Gemini-2.5-Flash) for both prompt parsing and evaluation, which may limit reproducibility.

- The qualitative feedback examples remain limited, and it is unclear how consistently the diagnostic outputs generalize across diverse prompt domains.

- The benchmark’s prompts are synthetically constructed and might not reflect the real-world user prompts especially the diversity aspect.

### Questions
How robust is SANEval to variations in the underlying LLM or detector, would the evaluation results remain consistent if different models (e.g., GPT-4o or open-source detectors) were used in place of Gemini?

### Soundness
2

### Presentation
2

### Contribution
2
