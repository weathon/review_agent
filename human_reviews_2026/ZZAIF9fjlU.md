# ProofRM: A Scalable Pipeline to Train a Generalized Math Proof Reward Model

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
Large Language Models (LLMs) have stimulated strong math reasoning abilities through Reinforcement Learning with Verifiable Rewards (RLVR). Thanks to the simplicity of directly comparing answers, the reward is both accurate and scalable. However, more challenging mathematical problems like math Olympiad or genuine mathematical research often take the form of proof-based problems where there is no guaranteed way to determine the authenticity of a proof by simply matching the answers. A reward model that can accurately evaluate diverse full proof process is necessary to operate scalable and efficient reinforcement learning to solve these problems. In this paper, we design an *scalable* data construction process that, with minimal human involvement, leverages LLMs to generate a large quantity of high-quality and diverse ``*problem-proof-check*'' triplet data, which can be used to train the proof reward model through RLVR by rewarding the correct proof checking. By utilizing different proof-generating LLMs, proof generation methods, prompts, and problem sources, we ensure the diversity of the generated problem-proof pairs in terms of difficulty, length, language style. Our human check also support the high accuracy of the checking labels. With this data generation process, we train a proof evaluator that can accurately judge across diverse datasets. Our experiments, comparing to past baselines, validate the model's effectiveness from multiple perspectives, including reward accuracy, reinforcement learning effectiveness, and test-time guidance, providing important process references and tools for enhancing LLMs' mathematical capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of verifying mathematical proofs in reinforcement learning settings. The authors argue that current RLVR (Reinforcement Learning with Verifiable Rewards) methods, which work well for problems with easily verifiable answers, struggle with proof-based problems where verification asymmetry is weak. They propose a data collection pipeline that generates diverse Question-Proof-Check (QPC) triplets through multi-dimensional LLM-aided generation, followed by LLM labeling with hierarchical human validation. The collected data is used to train a generative reward model with stabilization techniques including "LLM-as-RM-for-RM" and balanced token weighting. Experiments show the model achieves comparable performance to strong baselines on proof verification tasks.

### Strengths
1. **Well-motivated problem**: The paper clearly articulates the limitation of current RLVR approaches for proof-based mathematical problems, providing concrete evidence that verification asymmetry breaks down for complex proofs.

2. **Comprehensive data collection effort**: The multi-dimensional diversity approach (question sources, generation methods, LLM diversity) is sensible and generates a substantial dataset. The composition-level human validation strategy offers a practical trade-off between scalability and quality.

3. **Practical engineering contributions**: The LLM-as-RM-for-RM mechanism and balanced token weighting are pragmatic solutions to real training instabilities. The stable training over 312 steps with 18k samples demonstrates the effectiveness of these techniques.

### Weaknesses
1. **Limited novelty**
   The work mainly integrates existing components — multi-LLM data generation, consistency filtering, RLVR-based reward modeling — into a single pipeline.
   Conceptually it remains an **engineering consolidation** rather than a new methodological advance; the two stability heuristics are heuristic extensions without theoretical grounding or extensive empirical validation.

2. **Insufficient ablations and analysis**

   * No ablation for different data diversity dimensions, RM-for-RM or balanced token weighting.
   * No comparison between **base vs. trained ProofRM** on test-time scaling datasets.

3. **Weak test-time scaling / relative ranking ability**
   Although ProofRM performs well on single-sample accuracy, its *best-of-k* curve in Fig. 4 quickly flattens and stays below GPT-5-mini and Gemini.
   This indicates **poor calibration and limited ability to rank multiple diverse proofs**, suggesting that the model learns binary classification but not fine-grained relative judgment.
   The small and filtered evaluation set (18 problems, 48 proofs) further limits statistical reliability.

### Questions
While ProofRM shows strong single-sample accuracy in §4.1, its performance scales poorly in §4.2 (best-of-k).
Could the authors provide a deeper analysis of why good point-wise classification does not translate into effective relative ranking across multiple candidates?
Understanding whether this gap arises from binary supervision, calibration issues, or biases introduced by the stabilization methods would greatly clarify the behavior of the reward model.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents ProofRM, a reward model designed to verify the correctness of mathematical proofs expressed in natural language. The authors propose a three-stage pipeline: (1) automated data collection using LLMs with multi-dimensional variation (problem sources, proof generation methods, and generating LLMs), (2) LLM-aided labeling with composition-level human check, and (3) RLVR training using a generative reward model "LLM-as-an-RM-for-RM". The trained ProofRM achieves high accuracy on diverse test problems and demonstrates effective test-time scaling capabilities with comparable performance on best-of-k metrics.

### Strengths
The paper identifies an important and timely problem: the asymmetry of verification for mathematical proofs. The authors convincingly argue that while answer-checking has enabled breakthroughs in mathematical reasoning (DeepSeek-R1, OpenAI o1), proof verification remains a bottleneck for Olympiad-level and research-level mathematics. A comprehensive and scalable data collection pipeline is a major strength. The multi-dimensional diversity strategy including question source diversity, proof generation method diversity and LLM diversity seem sound. The composition-level human validation is quite elegant, requiring only ~100 hours of human effort versus >1,200 hours for exhaustive annotation. The paper also introduces “LLM-as-an-RM-for-RM”: an auxiliary LLM detects surface-level fluency issues (repetition, context-irrelevance, hasty conclusions) to prevent positive rewards for accidentally correct but flawed reasoning.

### Weaknesses
__1) Evaluation Limitations__

__a) Benchmark construction concerns__

The experimental section includes two types of problems: generalized check accuracy and test-time scaling. 
In the first setup, the "Mix-up" test set comes from the hierarchical sampling used during data construction. This creates a substantial bias as the baselines considered are not familiar with the dataset. While the paper states these are "different questions," they come from the same source distributions and LLM generations as training data. The second set (“student-sourced problems”) is not discussed at all, and none examples are presented. These two observations about test data sources raise concerns about the fairness of the entire evaluation in this subsection. 

The second setup considers test-time scaling evaluation on 18 problems. This is an extremely small set as the total number of the generated proofs is 48. The confidence interval for such an evaluation is likely wide. The paper should at least report statistical significance testing and, potentially, increase the number of problems considered. 

__b) Limited comparison to existing approaches__

The paper provides no baseline against existing proof RMs. The authors mention "Open Proof Corpus" with ~5k annotated samples but provide no head-to-head comparison.

The paper briefly mentions MathConstruct, IneqMath, and DeepTheorem but dismisses them based on a single author's informal experiment (Appendix A.3). This analysis lacks rigor: success rates of 84%, 70%, and 40% by a non-expert within minutes suggest these datasets may not be "fundamentally altered" but rather require targeted defense mechanisms.

__c) No computational overhead analysis__ 

The paper does not report the inference cost. Generating reasoning traces for each proof during training/evaluation incurs substantial computational overhead. How does ProofRM's throughput compare to ORM/PRM baselines? For test-time scaling with k=50, does the 8× repeated generation (64 total generations per proof) remain practical? The absence of comparison to discriminative RMs, especially time-comparison, does not benefit the work.

__2) "LLM-as-an-RM-for-RM" Approach Justification__ 

While the "LLM-as-an-RM-for-RM" approach addresses real problems, its design feels reactive rather than principled. Why should surface-level fluency checks prevent collapse, and could this inadvertently penalize valid but unconventional reasoning styles? The authors provide post-hoc manual inspection (50 samples), however, some statistically significant validation is required. Additionally, Figure 3 shows reward increasing for 18k samples, the paper acknowledges "model collapse still occasionally occurs". However, the paper provides minimal qualitative analysis of failure modes. The discussion in Appendix A.6 provides negative examples but no systematic characterization or ablation study.

__3) Overstated Scalability Claims__

The paper claims the approach is "scalable" in multiple places, but there are major bottlenecks in the pipeline. First, human validation is still required. The hierarchical sampling saves effort but doesn't eliminate human involvement. For new problem domains, the 100-hour investment must be repeated. This doesn't scale to diverse mathematical fields (e.g., topology or differential geometry). Second, the pipeline relies on the LLM-as-a-judge approach in multiple places (proof generation, labeling). This limits the approach to the tasks and domains that are LLM-friendly. The approach cannot handle diagrams, symbolic computation, or formal language. 

__4) Missing Related Work__

The paper does not discuss the outcome reward models vs. process reward models in the context of proof verification. PRMs provide step-level supervision—could this approach be combined with ProofRM?​​Limited engagement with test-time compute scaling literature beyond best-of-N. Techniques like beam search with PRMs or lookahead search could enhance ProofRM's utility.

### Questions
1) Can you provide a direct comparison to OPC and other proof verification baselines on a standardized (preferably different than the one presented in the paper) test set?
2) Have you experimented with discriminative (scalar) reward models? What is the accuracy-efficiency (especially time efficiency) trade-off compared to the generative approach?
3) How does the optimal token weight η vary across different model sizes (1.5B, 3B, 8B)? Is 0.6 universally optimal?
4) Could the pipeline be adapted to formal language (Lean/Isabelle) by replacing LLM proof generation with auto-formalization? What would be the cost-benefit trade-off?
5) How does ProofRM performance scale with training data size? Is there a clear saturation point, or do you expect continued improvement with 100k+ samples?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenging problem of verifying mathematical proofs in reinforcement learning context for LLMs. The authors argue that although RLVR performs well for problems with easily verifiable answers, the proof-based problems are lacking this verification asymmetry. They propose a scalable pipeline for creating question-proof-check (QPC) triplets by multi-dimensional diversity expansion and hierarchical human review, then train a generative reward model (ProofRM) which achieves 76.8% accuracy on diverse proof evaluation tasks.

### Strengths
Originality: The paper identifies important gap in current RLVR approaches - that verification asymmetry breaks down for proof-based problems. The multi-dimensional diversity expansion approach (varying sources, generation methods, and different LLMs) combined with composition-level human checking is creative solution for scalable data collection.

Quality: Technical contributions are quite solid, including "LLM-as-RM-for-RM" mechanism for preventing training collapse and the balanced token weighting for stabilizing output length. The detailed analysis of negative sample types in Section A.4.4 gives valuable insights about common proof failure modes.

Clarity: The motivation is articulated well with concrete examples in Table 1 and 2. Pipeline is illustrated clearly in Figure 1, and implementation details are documented thoroughly in appendix.

Significance: This work addresses practical bottleneck in scaling of mathematical reasoning - the lacking of scalable proof verification. The reported 12-fold efficiency gain in annotation process and successful RL training over 300+ steps show practical value.

### Weaknesses
The experimental scale appears limited with training set of only 24k samples and test set containing just 18 problems for best-of-k evaluation, which are too small to make robust conclusions. The 8B parameter base model is also limiting the generalizability claims. There is insufficient comparison with baseline methods - no comparison with process reward models (PRMs) or MCTS-based approaches except brief mentions, and missing the ablations on individual contributions of diversity dimensions.

The evaluation has several gaps including no evaluation on formal theorem proving, research-level mathematics, or out-of-distribution proof styles. The "Student" test set is showing significant performance drop (47.18% vs 76.81% on Mix-up), which suggests limited robustness. Methodologically, the heavy reliance on LLM-generated labels even after filtering could introduce systematic biases, and the balanced token weight formula is lacking theoretical justification. Also composition-level checking maybe misses instance-level errors.

### Questions
How does the performance change with dataset size? Could you provide learning curves that show relationship between training data volume and accuracy? What is inter-annotator agreement for human checks and how many disagreements were occurred in the 5% sampled questions?

What patterns ProofRM learns for identifying incorrect proofs? Can you show attention visualizations or feature analysis about what model focuses on? You mention briefly MCTS-based PRMs - could you provide quantitative comparison or explain why your approach is fundamentally different or better?

What is computational overhead of LLM-as-RM-for-RM during training and how this affects the training efficiency? Have you tested ProofRM on formal proofs (like Lean/Isabelle) or research-level mathematics beyond the competition problems? Could you provide statistics about which error types (from Section A.4.4) ProofRM catches most effectively and least effectively?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses a critical bottleneck in advancing Large Language Models (LLMs) for high-level mathematical reasoning: the lack of a scalable and accurate Reward Model (RM) capable of evaluating full, proof-based solutions. While current LLM successes rely on Reinforcement Learning with Verifiable Rewards (RLVR) through simple answer matching, this approach fails for proofs, where the "asymmetry of verification" diminishes.

### Strengths
1. The multi-dimensional QPC data pipeline is highly innovative, leveraging the generation power of LLMs to create diverse (in style, length, and error type) data, while utilizing a clever hierarchical human check to ensure annotation accuracy and save vast amounts of expert time.
2. The paper insightfully diagnoses why current RLVR methods are insufficient for proofs and correctly pivots the task to training a verifier/RM first.
3. The proposed fixes for RL instability, the "LLM-as-RM for RM" for thought quality and the balanced token weight strategy for sequence length variation in order to provide highly practical and significant guidance for future work on generative reward models.
4.  Explicit efforts to ensure the Proof RM learns from diverse error types (Table 3 details how methods like "Mask Completion" induce more subtle, human-like errors) suggests a strong focus on generalizable verification beyond superficial checks.

### Weaknesses
1. While the composition-level human check is crucial for scalability, its efficacy depends entirely on the homogeneity of the data within a "composition" slice. The paper could be strengthened by providing more transparency on the statistical risks: if an entire slice is dropped due to a few non-conforming samples, high-quality data might be lost, or conversely, if the sample is not representative, a bad slice might be retained.
2. The fluency check performed by the "LLM-as-RM for RM" appears to only detect syntactic or surface-level issues (e.g., repeated sentences, nonsense) to prevent model collapse. It does not seem to penalize a coherent, yet logically flawed thought process (CoT) that still reaches the correct T/F conclusion. This could allow for subtle reward hacking in the reasoning process itself.

### Questions
1. Could you provide more details on the practical application of the composition-level check? Specifically, how many samples (and what proportion) were checked per composition, and what was the final rejection rate for entire compositions versus individual samples?
2. The RM is generative. Did you explore using an internal reward signal extracted from the RM's step-by-step reasoning (e.g., using a Process Reward Model (PRM) concept) in addition to the binary T/F (Outcome Reward Model) to better supervise the fluency and correctness of the intermediate steps, rather than just the final verdict?
3. Given the multi-dimensional diversity generation (Table 3), what is the breakdown of error types (e.g., case-analysis, hallucination, subtle step-level flaws) in the final training dataset? How does the ProofRM's accuracy vary across these different categories of errors compared to baselines?

### Soundness
4

### Presentation
3

### Contribution
3
