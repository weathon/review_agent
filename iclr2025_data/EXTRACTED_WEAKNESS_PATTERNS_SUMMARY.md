# Extracted Weakness Patterns for "The Art of Scaling RL Compute for LLMs"

This document summarizes concrete weakness patterns extracted from human reviews of ICLR 2025 papers that are relevant to a paper on scaling reinforcement learning compute for LLMs.

## Key Weakness Patterns Found

### 1. Limited Evaluation Scope and Generalization Claims

**Papers with this pattern:** MR.Q, GVL, Divisive Normalization

**Weakness:** Evaluations are limited to a narrow set of benchmarks, environments, or domains, yet authors make broad generality claims.

**Specific examples:**
- Evaluating on only Gym, DMC, and Atari (homogeneous environments) while claiming the method is a "generalist" algorithm
- Testing only on mathematical reasoning with single model family (Gemma) while implying broader applicability
- Using only classical networks (AlexNet, VGG-16) on CIFAR and ImageNet without testing on transfer learning or OOD tasks

**Relevance to scaling RL paper:** Papers scaling to 400K+ GPU hours should demonstrate that design choices and predictive scaling patterns hold across multiple model sizes, datasets, domains, and tasks. Claims about generality must be supported by evaluation breadth.

---

### 2. Unfair Baseline Comparisons and Hyperparameter Tuning Issues

**Papers with this pattern:** MR.Q, Process Advantage Verifiers

**Weakness:** Baseline algorithms not properly tuned or have inconsistent tuning, undercutting the validity of performance comparisons.

**Specific examples:**
- No disclosure of baseline hyperparameter settings, making it unclear if comparisons are fair
- Using different model sizes for different baselines without normalization
- Comparing against outcome reward models instead of the closest baseline (process reward models)
- No evidence that proposed method is less hyperparameter sensitive than baselines under equal conditions

**Relevance to scaling RL paper:** Large-scale studies claiming design choice comparisons must provide detailed hyperparameter descriptions, ensure baselines are fairly tuned, and normalize for compute budgets. This is critical when claiming certain design choices affect compute efficiency.

---

### 3. Missing Statistical Significance and Ablation Rigor

**Papers with this pattern:** MR.Q, PGR

**Weakness:** Ablation studies lack statistical significance testing, confidence intervals, and provide limited insight into individual component contributions.

**Specific examples:**
- Ablation results presented without significance tests or confidence intervals
- Ablations showing minimal performance gains with no clear explanation for why components are included
- Ablations designed to "revert to theory" that provide little novel insight

**Relevance to scaling RL paper:** An empirical study of design choice ablations at scale should include rigorous statistical analysis. When studying how different loss aggregation, normalization, curriculum, and off-policy choices affect performance, confidence intervals and significance tests are essential.

---

### 4. Inconsistent or Unexplained Scaling Behavior

**Papers with this pattern:** Prioritized Generative Replay, Decision Transformer

**Weakness:** Methods show inconsistent scaling trends that lack clear explanation or intuition.

**Specific examples:**
- Synthetic data ratio benefits PGR at r=0.75 but not r=0.875, with no explanation of why trend is inconsistent
- Scaling law analysis for model sizes doesn't account for inference costs, making practical scaling curves unclear
- Metrics don't account for computational trade-offs during scaling

**Relevance to scaling RL paper:** A paper on scaling should provide clear, interpretable, and reproducible scaling curves. If performance plateaus or shows non-monotonic behavior at certain scales, the paper should explain why. Include inference/training time costs in scaling analysis.

---

### 5. Sim-to-Real Gap and Real-World Validation Missing

**Papers with this pattern:** Decision Transformer, various large-scale studies

**Weakness:** Evaluation entirely in simulation/controlled settings without real-world validation or discussion of transfer challenges.

**Specific examples:**
- All evaluation in simulated environments with no case studies or real-world deployments mentioned
- Sim2Real gap acknowledged but not addressed
- Methods show improvements in offline/controlled settings but unclear how they transfer

**Relevance to scaling RL paper:** For an LLM RL paper, discuss how small-scale design choice insights (from 8B model experiments) transfer to larger models and different domains. If using synthetic or simulated settings, clearly state limitations of insights.

---

### 6. Computational Cost Not Properly Accounted For

**Papers with this pattern:** Divisive Normalization, Decision Transformer

**Weakness:** Method shows performance improvements but doesn't justify whether gains offset computational overhead. No compute-matched baselines provided.

**Specific examples:**
- 50% slower per epoch with 2% performance gain, but no comparison with training baseline longer for same total compute
- Inference time costs not included in scaling law evaluation, questioning whether "sweetspot" is a specific model size
- Training cost impacts of proposed technique not discussed despite being critical for large-scale studies

**Relevance to scaling RL paper:** For papers claiming to study "compute efficiency," all comparisons must be compute-matched or explicitly account for different computational costs. A 400K GPU-hour study should clearly document training time, inference time, and memory requirements for design choices.

---

### 7. Single Model Size, Family, or Architecture Evaluation

**Papers with this pattern:** Process Advantage Verifiers, GVL, various others

**Weakness:** Experiments conducted with only one model size/family/architecture, limiting claims about how insights scale.

**Specific examples:**
- All experiments on Gemma only, no other model families tested
- Testing on AlexNet/VGG but not scaling to modern architectures
- Robustness results on only one architecture (AlexNet) without confirming on others (VGG-16)

**Relevance to scaling RL paper:** A paper specifically about scaling must test design choices across multiple model sizes. Insights from 8B models might not hold at 13B, 34B, or 70B scales. This is critical for making generalizable claims.

---

### 8. Presentation and Clarity Issues Impacting Reproducibility

**Papers with this pattern:** R-Learning, Decision Transformer, various others

**Weakness:** Papers with complex notation, vague experimental details, or unclear design choices that hinder reproducibility.

**Specific examples:**
- Overly complex notation and theorem statements that are "very difficult to parse/understand"
- Design choices described too generally (e.g., "state, action, reward") without specific implementation details
- Missing hyperparameter selection justification
- No code release mentioned

**Relevance to scaling RL paper:** A large-scale empirical study should be highly reproducible. Include detailed hyperparameter tables, clear description of reward functions/preprocessing, and provide code for reproducibility. Avoid overly complex presentation of relatively simple ideas.

---

### 9. Missing or Unfair Baseline Comparisons

**Papers with this pattern:** Multiple papers

**Weakness:** Key baselines are missing or replaced with weaker comparisons.

**Specific examples:**
- Missing PPO comparison despite being the most general-purpose RL algorithm
- Comparing against older methods instead of recent strong baselines
- No comparison with recent related work (e.g., other PRMs when proposed method is about PRMs)
- Missing ablations comparing to simple alternatives (e.g., random masking vs. other masking strategies)

**Relevance to scaling RL paper:** When claiming that certain design choices are superior, compare against published strong baselines. For a paper on RL scaling, include comparisons with current SOTA RL approaches at the studied scales.

---

### 10. Unclear or Unstated Generalization of Key Design Choices

**Papers with this pattern:** POTEC, various others

**Weakness:** Methods make strong assumptions or design choices that don't generalize, but this limitation is not discussed upfront.

**Specific examples:**
- Algorithm assumes specific hierarchical reward structure, unclear how to generalize to different reward functions
- Number of clusters (key hyperparameter) selection guidance missing for real-world application
- Impact of clustering algorithm on performance not studied
- Method relies on accurate learned dynamics models that may fail in complex environments

**Relevance to scaling RL paper:** If the paper proposes specific design choices for RL (e.g., loss aggregation strategy, normalization scheme), clearly state assumptions about when they apply. Test robustness to variations. For "recipe" papers, provide clear guidance on hyperparameter selection.

---

## Synthesis: Key Recommendations for Scaling RL Paper

Based on these patterns, a scaling RL paper should:

1. **Breadth of Evaluation:** Test design choices across multiple model sizes (not just 8B), domains, and benchmark suites
2. **Fair Comparisons:** Clearly disclose hyperparameters for all baselines, ensure compute-matched comparisons, compare against strong recent baselines
3. **Rigorous Ablations:** Include statistical significance tests, confidence intervals, and clear intuition for each component
4. **Clear Scaling Analysis:** Show scaling trends across multiple dimensions (model size, data, compute), account for all costs (training, inference), explain any non-monotonic or unexpected behavior
5. **Reproducibility:** Provide detailed hyperparameter tables, implementation details, pseudo-code for design choices, and ideally code release
6. **Honest Limitations:** Discuss when insights might not generalize, what assumptions are made, simulation-to-reality gaps
7. **Proper Baselines:** Compare against strongest published methods, especially widely-used algorithms like PPO if claiming generality
8. **Scalable Design Guidance:** Provide clear recipes for hyperparameter selection and design choice configuration at different scales
9. **Compute Accounting:** Document training time, inference time, and memory for all design choices and model sizes
10. **Presentation:** Use clear notation, avoid unnecessary complexity, make the paper accessible to broad audience

