# CCPO: Execution Consistent Preference Optimization through Computational Pacts

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 0, 6, 6

## Abstract
Execution-based verification has been shown to be effective in enhancing the mathematical reasoning abilities of large language models due to its computational soundness guarantees and dependency-aware filtering. Previous works involving preference optimization often include reward models that utilize Bradley-Terry assumptions, which fail to capture the logical dependencies and execution consistency requirements essential for scientific and computational reasoning tasks. In this paper, we introduce a novel method for generating computationally sound solutions accompanied with corresponding dependency graphs for execution-consistent preference optimization. Our approach begins with the construction of a high-quality scientific reasoning dataset by incorporating UltraFeedback prompts, base model generations, computational verification, and execution consistency results. Next, we construct dependency graphs by extracting reasoning step expressions, the computational prerequisites needed for the expressions, and the derivability relationships of the expressions from the previously collected dataset. Based on this extracted information, we generate corresponding execution consistency scores to accurately capture the mathematical verification process. Appending the generated execution consistency scores to each reasoning step results in data consisting of paired filtered reasoning steps and their corresponding execution consistency scores. Training Llama-3-8B and DeepSeekMath-7B with this corpus achieves substantial improvements across scientific reasoning domains: +17.0\% on MATH, +15.1\% on GSM8K, while extending our Scientific Feasibility Control framework to achieve 50.1\% accuracy on PhyX multimodal physics reasoning—outperforming DeepSeek-R1 (49.8\%) and OpenAI o3-mini (48.2\%)—with 91.7\% scientific validity coverage at $\alpha = 0.10$ confidence level and 73\% reduction in scientific law violations across architectures, leading to the creation of the CCPO family of models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CCPO (Execution-Consistent Preference Optimization through Computational Pacts), a framework that aims to align large language models’ reasoning with computational correctness rather than subjective human preference.

### Strengths
Addresses a relevant problem: current preference optimization methods ignore logical or computational consistency.

### Weaknesses
- Lack of credible baselines. The paper mentions DPO, IPO, but does not actually reimplement or evaluate them under the same data and hyperparameter settings. Reported numbers only compare CCPO-enhanced checkpoints to their base models, which is not sufficient for fair evaluation.

- No ablation or isolation study. There is no evidence showing which part of CCPO (dependency graph extraction, execution filtering, or conformal calibration) drives the reported gains.

- Weak theoretical rigor. The conformal guarantee depends on assumptions (deterministic execution, exchangeability) that are unrealistic for stochastic LLM decoding.

- Reproducibility concerns.   Neither code, data, nor verification scripts are released. Critical implementation details—such as dependency-graph extraction, verification engine configuration, and filtering thresholds—are unavailable. Without open resources, the reported results cannot be independently validated, which substantially limits the credibility and scientific value of the work.

### Questions
It is well known that reinforcement learning or preference optimization tends to be most effective when applied to base (unaligned) models.   If the starting checkpoint is already an instruction-tuned model such as Llama-3-8B-Instruct, how does CCPO behave in that case?  Does it still provide meaningful improvemen?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes a method for generating computationally sound solutions accompanied with corresponding dependency graphs for execution-consistent preference optimization. The paper conducts experiments on several mathematical reasoning benchmarks to evaluate the method.

### Strengths
The extensive similarities in core ideas and presentation to the prior work "Conformal Language Model Reasoning with Coherent Factuality" (ICLR 2025) undermine the meaningful assessment of this paper's unique strengths or originality.

### Weaknesses
This paper fundamentally replicates the key idea introduced in the paper "Conformal Language Model Reasoning with Coherent Factuality, ICLR 2025”. The similarities extend beyond the conceptual core to the very structure and phrasing of the argument, despite the use of different surface-level concepts. However, the authors also fail to cite the original paper.

### Questions
see the weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Code Consistency Preference Optimization (CCPO), a preference-training framework for mathematical/scientific reasoning that enforces execution consistency at the level of individual reasoning steps. CCPO constructs dependency graphs over extracted steps, verifies steps via code execution, and filters outputs using conformal prediction to guarantee calibrated coverage. The method uses multiplicative-weights self-play with graph-aware scoring, and integrates execution verification during training rather than only at inference. On MATH, GSM8K, and PhyX, the authors report gains (e.g., +17.0% MATH, +15.1% GSM8K; 50.1% PhyX with 91.7% validity coverage) for Llama‑3‑8B and DeepSeekMath‑7B variants.

### Strengths
- The paper has a clear formal goal and calibration guarantee. The authors define execution-consistent preference and provide a conformal‑prediction guarantee that the filtered output is computationally sound with probability ≥ 1 − \alpha. This offers a principled control knob absent from prior PO works. 

- Algorithm 1 builds induced subgraphs by thresholding node scores and removing nodes with missing prerequisites, aligning filtering with derivational structure. Definition 4  formalizes an execution consistency score over induced subgraphs, connecting scores to which steps survive (p. 7).

- Empirical gains show consistency across tasks. Results report +17.0% (MATH) and +15.1% (GSM8K) along with 91.7% validity coverage and a 73% reduction in scientific‑law violations. Progressive learning table shows monotonic improvements over iterations.

- The paper has a technical breadth cross math, physics, formal, and coding.

### Weaknesses
- Some mathematical specification lacks some clarity or consistency. Approximate dependency graphs are invoked in Theorem 1’s upper bound and in App. F, but their construction/quality metrics and impact on guarantees are only sketched without a concrete end‑to‑end bound on task‑level error. 
  
- The role and calibration of \beta in dependency‑aware scoring (eq. (4)) are not fully motivated; the text says \beta is calibrated via conformal prediction but does not show how this interacts with coverage guarantees.

- Execution‑consistency score r() depends on induced subgraph selection. Aadmissible UT sets and their completeness are not rigorously characterized. 

-  Results emphasize within‑model gains and a few external models on PhyX, but do not include strong PO baselines augmented with step‑wise execution filters (e.g., DPO+exec or SPPO+exec) under identical compute and prompts.

- The claim of “superior performance without external supervision” is strong, yet CCPO still relies on verified oracles and dependency construction; comparisons against alternative self‑supervised filtering schemes are limited.

- MiniF2F/HumanEval summaries are brief; it’s unclear whether improvements persist under stricter decoding/time budgets and verifier ablations.

- No statistical testing (e.g., CIs) is reported for key benchmarks; effect sizes might overlap with variance. **No direct evidence found in the manuscript.

### Questions
See Weakness.

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
2

### Summary
This paper introduces Code Consistency Preference Optimization (CCPO), a novel framework for improving mathematical reasoning in large language models through execution-based verification and dependency-aware preference optimization. The key innovation is formulating preference learning as a game-theoretic optimization problem while incorporating computational verification constraints through dependency graph construction and conformal prediction guarantees. The authors train Llama-3-8B and DeepSeekMath-7B models, achieving substantial improvements on mathematical reasoning benchmarks: +17.0% on MATH, +15.1% on GSM8K, and 50.1% on PhyX physics reasoning. The method constructs dependency graphs by extracting reasoning steps, identifying computational prerequisites, and generating execution consistency scores, then filters reasoning steps based on these scores to maintain both logical coherence and computational soundness.

### Strengths
1. **Novel Problem Formulation**: Combining execution verification with preference optimization through dependency graphs is creative and well-motivated for mathematical reasoning.

2. **Strong Empirical Results**: Consistent improvements across diverse benchmarks (MATH, GSM8K, OCW, PhyX) and multiple base models demonstrate practical effectiveness.

3. **Theoretical Framework**: Applying conformal prediction to provide coverage guarantees for reasoning step filtering is innovative.

4. **Comprehensive Evaluation**: The paper includes extensive ablations, error analysis, and comparisons with relevant baselines.

### Weaknesses
Complexity: The method is theoretically and computationally intensive. Real-time execution and graph construction may limit scalability in resource-constrained settings.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
3
