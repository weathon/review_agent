# SCALERL Paper: Relevant Papers and Applicable Weakness Patterns from ICLR 2025

## Executive Summary
Analysis of ICLR 2025 human reviews identified key weakness patterns from papers in related domains (RL scaling, empirical studies, training recipes, scaling laws) that are directly applicable to the SCALERL paper. These weaknesses cluster around 5 main themes: baseline fairness, evaluation scope, ablation rigor, scaling law validity, and generalization claims.

---

## Most Relevant Papers Found (5 papers with human reviews available)

### 1. **MR.Q: Towards General-Purpose Model-Free Reinforcement Learning** (R1hIXdST22)
- **Score**: 7.5 | **Decision**: Accept
- **Relevance**: Core RL algorithm work with multi-domain empirical evaluation claims
- **Key parallels to SCALERL**:
  - Claims generalist algorithm across multiple benchmark domains
  - Combines multiple algorithmic components into a single recipe
  - Empirical evaluation with fixed hyperparameters
  - Compares against multiple baseline algorithms

---

### 2. **Process Advantage Verifiers (PAVs) for RL with LLMs** (A6Y7AqlzLW)  
- **Score**: ~7.1 | **Decision**: Accept
- **Relevance**: Process-level RL improvements for LLMs with limited evaluation scope
- **Key parallels to SCALERL**:
  - RL method for improving LLM performance
  - Evaluation limited to mathematical reasoning tasks only
  - Claims about improvements in efficiency and sample complexity
  - Single model family tested (Gemma)

---

### 3. **Prioritized Generative Replay (PGR)** (5IkDAfabuo)
- **Score**: 7.5 | **Decision**: Accept  
- **Relevance**: Scaling experiments with guidance functions and inconsistent scaling patterns
- **Key parallels to SCALERL**:
  - Studies how compute/data scaling affects performance
  - Explores scaling with different conditions
  - Shows inconsistent scaling behavior across parameters

---

### 4. **Deconstructing What Makes a Good Optimizer for Autoregressive Language Models** (zfeso8ceqr)
- **Score**: 6.0 | **Decision**: Accept (after revision)
- **Relevance**: Large empirical study on hyperparameter effects on LLM training
- **Key parallels to SCALERL**:
  - Extensive empirical study of training dynamics
  - Claims about hyperparameter robustness
  - 1D hyperparameter sweeps (similar to SCALERL methodology)
  - Questions about range of tested hyperparameters

---

### 5. **Simplifying, Stabilizing and Scaling Continuous-time Consistency Models** (LyJi5ugyJx)
- **Score**: 10.0 | **Decision**: Accept (Oral)
- **Relevance**: Major paper on scaling laws and training stability at scale
- **Key parallels to SCALERL**:
  - Studies scaling behavior across compute budgets (64x64 to 512x512)
  - Scaling beyond tested ranges (inference costs, latent space effects)
  - Design choices (architecture modifications) lack supporting evidence in some cases

---

## Critical Weakness Patterns from Human Reviews

### CRITICAL SEVERITY (1 pattern)

#### **Baseline Hyperparameter Fairness**
**Source**: MR.Q (R1hIXdST22) - Reviewer 2, lines 50-58

**Reviewer Quote**:
> "Unfortunately, the paper does the opposite and provides no detail (that I could find) regarding the hyperparameter settings of the baseline algorithms, nor how or if they were tuned. This is a critical weakness of the paper, as the possibility of untuned baselines undermines the claimed performance improvements."

**Additional Evidence**:
> "One example of this is that DreamerV3 outperforms MR.Q on only 1/4 benchmarks (Atari). Notably, Atari is also the only benchmark for which the results come from the reference work. For the remaining 3 benchmarks, the authors run DreamerV3 themselves. This does not imply that the authors failed to tune DreamerV3 or that the comparison was unfair, however, given the lack of detail regarding their tuning procedure or the hyperparameter sensitivity of the methods, the result is seriously undermined."

**Applicability to SCALERL**:
- SCALERL combines multiple design choices (batch size 768, KL penalty weight, reward scaling, etc.)
- Claims state-of-the-art results compared to GRPO, DAPO, and other methods
- **Critical question**: Were GRPO and DAPO hyperparameters tuned with equal compute budget and methodology?
- **Risk**: If baselines used default hyperparameters while SCALERL was extensively tuned with 400,000+ GPU hours, improvements may reflect tuning effort rather than algorithmic superiority

---

### HIGH SEVERITY (4 patterns)

#### **1. Limited Evaluation Scope / Single Task Domain**
**Source**: PAVs (A6Y7AqlzLW) - Reviewer 1, lines 12-14

**Reviewer Quote**:
> "Evaluation only on mathematical reasoning tasks... All experiments on a single model family (Gemma)"

**Additional Context** (Reviewer 5):
> "The paper focuses its empirical evaluation on ORMs and states that there are major advantages w.r.t. them but I believe that a fair comparison would be to use PRMs since they are the closest possible baseline."

**Applicability to SCALERL**:
- SCALERL's framework claims to apply generally to 'scaling RL for LLMs,' but evaluation is limited to math tasks with specific configuration (batch size 768, max sequence length 14,336)
- **Critical questions**:
  1. Does the sigmoid scaling law hold for: Code generation? Summarization? Multi-turn dialogue? Tasks with sparse vs. dense rewards?
  2. How do task properties affect scaling law shape?
  3. Are scaling parameters universal or task-specific?

**Reviewer Expectation**:
- Evaluation on at least 3-5 diverse RL task domains beyond mathematical reasoning
- Analysis of how sigmoid parameters change across task domains
- Discussion of task properties that affect scaling law shape

---

#### **2. Missing Comparison with Closest Baselines**
**Source**: PAVs (A6Y7AqlzLW) - Reviewer 5, lines 188-195

**Reviewer Quote**:
> "The paper focuses its empirical evaluation on ORMs and states that there are major advantages w.r.t. them but I believe that a fair comparison would be to use PRMs since they are the closest possible baseline... I believe that to truly understand the utility of PAVs as neural verifiers/reward models, one would need to compare them with the same search strategy but just a different ranking scheme (PRMs vs PAVs). Could the authors please provide additional details here?"

**Applicability to SCALERL**:
- SCALERL compares against GRPO and DAPO, but unclear if these are the closest comparisons
- **Critical questions**:
  1. Are there other systematic empirical studies on RL scaling that should be compared?
  2. Are there other papers on how compute budget affects reward scaling?
  3. Are there other works on optimal batch size, KL penalty, or reward scaling for LLM RL?

**Reviewer Expectation**:
- Explicit justification of why GRPO/DAPO are the closest comparisons
- Discussion of other scaling-focused RL methods and why they weren't compared
- Analysis of SCALERL vs. other empirical scaling studies

---

#### **3. Inconsistent Scaling Behavior Not Explained**
**Source**: PGR (5IkDAfabuo) - Reviewer 4, lines 146

**Reviewer Quote**:
> "Increasing Synthetic Data ratio does not benefit PGR and the unconditional baseline equally. PGR scales better at r=0.75 than SYNTHER but neither benefits from 0.875. We would think the trend would be consistent? whats the intution behind this?"

**Applicability to SCALERL**:
- SCALERL uses sigmoid curves to model scaling: reward ∝ sigmoid(compute_budget)
- **Critical questions**:
  1. Is scaling monotonic across all compute budgets tested (up to 100,000 GPU-hours)?
  2. Are there any non-monotonic behaviors or plateaus?
  3. If scaling law parameters change with compute budget or model size, what explains these variations?
  4. Why does the sigmoid model fit better than power laws or other scaling functions?

**Reviewer Expectation**:
- Analysis of scaling curve consistency across different compute budget ranges
- Investigation of when/why sigmoid assumptions hold vs. break down
- Explanation of non-monotonic behaviors if any exist

---

#### **4. Ablation Studies Lack Statistical Rigor**
**Source**: MR.Q (R1hIXdST22) - Reviewer 2, lines 64-65

**Reviewer Quote**:
> "The ablation study fails to provide the statistical significance of the results and lacks analysis. Furthermore, the 'reverting to theory' ablations are highly unsurprising so provide little contribution, and many of the remaining ablations show minimal performance gains."

**Applicability to SCALERL**:
- SCALERL recipe combines multiple design choices: batch size 768, max sequence length 14,336, KL penalty, reward scaling, learning rate, entropy coefficient
- **Critical questions**:
  1. Which components actually drive scaling improvements with statistical significance?
  2. For each component: Is the improvement > statistical noise?
  3. How does importance vary across compute budgets?
  4. Are there redundant/unnecessary components?

**Reviewer Expectation**:
- Ablation study with error bars and significance tests
- Analysis of contribution of each design choice to final performance
- Discussion of which choices are fundamental vs. hyperparameter tuning

---

### MEDIUM SEVERITY (6 patterns)

#### **1. Single Model Architecture Evaluation**
**Source**: Implicit pattern from multiple reviews

**Applicability to SCALERL**:
- SCALERL tests only on 8B dense and 17Bx16 MoE models
- **Critical questions**:
  1. Do sigmoid scaling curves hold for other architectures?
  2. Do MoE models follow the same scaling law as dense models?
  3. How do architectural differences affect scaling parameters?
  4. Would the recipe parameters change with different architectures?

---

#### **2. Optimal Hyperparameter Ranges Unclear**
**Source**: Optimizer Study (zfeso8ceqr) - Reviewer 1, lines 25

**Reviewer Quote**:
> "For pretraining percentage and beta_2, I encourage the authors to include more extreme values to support the claim that these hyperparameters do not matter. Instead showing them they do not matter in the current small range, it is more informative to show to the readers at what extreme values the loss starts to increase significantly."

**Applicability to SCALERL**:
- SCALERL tests batch size 768 but unclear if this is truly optimal
- **Critical questions**:
  1. What happens with extreme batch sizes (1, 4096, 16384)?
  2. Where do scaling curves begin to degrade?
  3. Are there regime shifts in the scaling law at different compute budgets?

---

#### **3. Convergence Assumptions May Be Violated**
**Source**: Pattern across multiple papers

**Applicability to SCALERL**:
- SCALERL assumes sigmoid scaling with well-defined parameters
- **Critical questions**:
  1. Are the assumptions about monotonic reward improvement with compute valid?
  2. Are there cases where increasing compute doesn't improve reward?
  3. Does the sigmoid model capture all scaling behavior or miss important phenomena?

---

#### **4. Generalization Across Data Heterogeneity**
**Source**: DEPT (vf5aUZT0Fz) - implicit from dataset discussion

**Applicability to SCALERL**:
- SCALERL uses math tasks with relatively uniform structure (GSM8K, MATH)
- **Critical questions**:
  1. How does scaling law change with diverse reward structures?
  2. Do heterogeneous RL tasks (mixing multiple task types) follow the same sigmoid curve?
  3. How does data distribution shift affect scaling law fit?

---

#### **5. Design Choice Justification Missing**
**Source**: Multiple papers

**Applicability to SCALERL**:
- Critical question for each recipe component:
  1. Why is batch size 768 specifically? Would 512 or 1024 work as well?
  2. Is this universally optimal or task-specific?
  3. Why these reward scaling coefficients? Do they generalize?

---

#### **6. Inference Costs Not Accounted For**
**Source**: Scaling in Consistency Models context (implicit)

**Applicability to SCALERL**:
- During RL training with 8B or 17Bx16 models:
  1. Inference cost (forward passes to get rewards) is a significant component
  2. Larger models have higher inference costs
  3. The sigmoid scaling law may change when accounting for inference costs
  4. The optimal compute allocation might differ in practice

---

## Synthesized Recommendations by Priority

### CRITICAL
1. **Disclose all baseline hyperparameters** and provide evidence of equal tuning effort for GRPO, DAPO, and other baselines
2. **Justify baseline selection** - explicitly explain why GRPO/DAPO are the closest comparisons and not other scaling-focused RL methods

### HIGH
1. **Expand evaluation scope** - test on 3-5 diverse RL task domains (code generation, summarization, dialogue, etc.)
2. **Add statistical rigor to ablations** - include error bars, significance tests, and analysis of each component's contribution
3. **Analyze scaling curve consistency** - demonstrate monotonicity and explain any non-monotonic behaviors
4. **Validate sigmoid model choice** - compare against power laws and other functional forms

### MEDIUM
1. **Test additional architectures** - demonstrate scaling laws hold for different model families beyond 8B dense and 17Bx16 MoE
2. **Extend hyperparameter ranges** - test extreme values to identify regime boundaries
3. **Account for inference costs** - analyze total compute including forward passes for reward computation
4. **Evaluate heterogeneous task mixtures** - show robustness to task diversity
5. **Validate sigmoid assumptions** - discuss failure modes and when assumptions break down

### LOW
1. **Improve presentation clarity** - better intuitive explanations of sigmoid scaling relationships
2. **Visualize scaling relationships** more clearly in figures

---

## Key Questions from Reviewers That Apply to SCALERL

1. **Why should we believe improvements are from SCALERL recipe and not better hyperparameter tuning of baselines?**
2. **Does the scaling law generalize beyond math tasks?**
3. **Which recipe components actually matter and have statistical significance?**
4. **Are you comparing against the most relevant baselines for scaling studies?**
5. **Does sigmoid model hold in practice or are there regime shifts?**
6. **How do inference costs affect the practical scaling law?**
7. **How sensitive are results to batch size, KL penalty, and other hyperparameters?**
8. **Do scaling curves vary significantly across different model architectures?**
9. **Is the sigmoid model the best functional form or would power laws fit better?**
10. **How robust is the scaling law to reward model errors?**

---

## Summary Statistics

- **Papers analyzed with human reviews**: 5 papers
- **Total human reviews analyzed**: 26 reviews
- **Common weakness themes identified**: 7 major patterns
- **Severity distribution**:
  - CRITICAL: 1 weakness
  - HIGH: 4 weaknesses  
  - MEDIUM: 6 weaknesses
  - LOW: 1 weakness

---

**Report Generated**: 2026-04-08
**Data Source**: ICLR 2025 Human Reviews
**Analysis Scope**: Papers related to RL scaling, scaling laws, empirical studies, and training recipes
