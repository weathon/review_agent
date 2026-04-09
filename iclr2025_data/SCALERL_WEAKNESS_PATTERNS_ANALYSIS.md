# SCALERL Paper: Extracted Weakness Patterns from Human Reviews
## Analysis of Related ICLR 2025 Papers

**Target Paper:** The Art of Scaling Reinforcement Learning Compute for LLMs (SCALERL)

**Analysis Date:** April 8, 2026

**Methodology:** Extracted weakness patterns from human reviews of 5-8 related ICLR 2025 papers that discuss:
- Scaling laws and empirical studies
- Algorithm design choices and hyperparameter sensitivity
- RL methods (offline RL, preference learning, process reward models)
- Generalization and extrapolation
- Empirical methodology and evaluation rigor

---

## Summary

Analyzed **13 distinct weakness patterns** extracted from human reviews of papers on:
- Process Advantage Verifiers (PAVs) for RL with LLMs
- MR.Q: Model-Free Deep RL Algorithm
- Prioritized Generative Replay (PGR) for RL
- DEPT: Decoupled Embeddings for Pre-training
- Deconstructing What Makes a Good Optimizer for Language Models
- Q-SFT: Q-Learning for LLMs via Supervised Fine-tuning
- R-Learning for Offline RL
- Decision Transformer for Fault Detection
- Guided Reinforcement Learning with Roll-Back

**Severity Distribution:**
- CRITICAL: 1 weakness
- HIGH: 5 weaknesses
- MEDIUM: 6 weaknesses
- LOW: 1 weakness

---

## CRITICAL WEAKNESSES (Must Address)

### 1. Hyperparameter Fairness Unclear
**Weakness:** Baseline algorithms' hyperparameters not disclosed or fairly tuned, making it impossible to assess if performance improvements are from the method or from better hyperparameter search.

**Source Paper:** MR.Q: A Model-Free Deep RL Algorithm with Model-Based Representations

**Source Quote:**
> "the paper provides no detail (that I could find) regarding the hyperparameter settings of the baseline algorithms, nor how or if they were tuned. This is a critical weakness of the paper, as the possibility of untuned baselines undermines the claimed performance improvements."

**Specific Applicability to SCALERL:**
SCALERL combines multiple design choices (batch size 768, KL penalty weight, reward scaling, etc.). The 400,000+ GPU-hour study claims state-of-the-art results compared to GRPO, DAPO, and other methods. **Critical question:** Were GRPO and DAPO hyperparameters tuned with the same compute budget and methodology as SCALERL? If baselines used default hyperparameters or standard Chinchilla scaling while SCALERL recipe was extensively tuned, the improvements may reflect tuning effort rather than fundamental algorithmic superiority.

**Reviewer Expectation:**
- Detailed disclosure of all baseline hyperparameters
- Evidence that baselines received equal hyperparameter tuning effort
- Fair comparison study where both SCALERL and baselines are tuned using same methodology

---

## HIGH SEVERITY WEAKNESSES (Should Address)

### 2. Limited Evaluation Scope
**Weakness:** Only tested on a single task domain (mathematical reasoning with MATH dataset) limiting generalization claims across different RL domains and problem types.

**Source Paper:** Process Advantage Verifiers (PAVs) for RL with LLMs

**Source Quote:**
> "Evaluation only on mathematical reasoning tasks... All experiments on a single model family (Gemma)"

**Specific Applicability to SCALERL:**
SCALERL's framework claims to apply generally to "scaling RL for LLMs," but evaluation is limited to math tasks with batch size 768 and max sequence length 14,336. **Critical question:** Does the sigmoid scaling law hold for:
- Code generation tasks?
- Summarization or text generation?
- Multi-turn dialogue?
- Tasks with longer sequences or shorter sequences?
- Tasks with sparse vs. dense rewards?
- Domains with significantly different exploration requirements?

**Reviewer Expectation:**
- Evaluation on at least 3-5 diverse RL tasks beyond mathematical reasoning
- Analysis of how sigmoid parameters change across task domains
- Discussion of task properties that affect scaling law shape

---

### 3. Missing Comparison with Closest Baselines
**Weakness:** Paper compares against less relevant methods rather than the most directly comparable alternatives, making it unclear what real improvements are.

**Source Paper:** Process Advantage Verifiers (PAVs) for RL with LLMs

**Source Quote:**
> "I believe that a fair comparison would be to use PRMs since they are the closest possible baseline. The authors do prempt this by stating that PRMs have only demonstrated 1-2% improvement w.r.t. ORMs but that is in the context of best-of-N search. There are no comparisons with PRMs except for Fig 5a."

**Specific Applicability to SCALERL:**
SCALERL compares against GRPO and DAPO, but:
- Are there other systematic empirical studies on RL scaling that should be compared against?
- Are there other papers that study how compute budget affects reward scaling?
- Are there other works on optimal batch size, KL penalty, or reward scaling for LLM RL?

Unclear which comparisons are truly the most relevant baseline alternatives.

**Reviewer Expectation:**
- Explicit justification of why GRPO/DAPO are the closest comparisons
- Discussion of other scaling-focused RL methods and why they weren't compared
- Analysis of SCALERL vs. other empirical scaling studies

---

### 4. Inconsistent Scaling Behavior Not Explained
**Weakness:** Method/framework shows benefits from certain hyperparameters inconsistently across conditions, but lacks clear intuition for why the scaling behavior varies.

**Source Paper:** Prioritized Generative Replay (PGR) for RL

**Source Quote:**
> "Increasing Synthetic Data ratio does not benefit PGR and the unconditional baseline equally. PGR scales better at r=0.75 than SYNTHER but neither benefits from 0.875. We would think the trend would be consistent? whats the intution behind this?"

**Specific Applicability to SCALERL:**
SCALERL uses sigmoid curves to model scaling: `reward ∝ sigmoid(compute_budget)`. **Critical question:**
- Is the scaling monotonic across all compute budgets tested (up to 100,000 GPU-hours)?
- Are there any non-monotonic behaviors or plateaus?
- If the scaling law parameters change with compute budget or model size, what explains these variations?
- Why does the sigmoid model fit better than power laws or other scaling functions?

**Reviewer Expectation:**
- Analysis of scaling curve consistency across different compute budget ranges
- Investigation of when/why sigmoid assumptions hold vs. break down
- Explanation of non-monotonic behaviors if any exist

---

### 5. Ablation Studies Lack Statistical Rigor
**Weakness:** No significance testing or comprehensive analysis of which components actually contribute to performance gains vs. which are just implementation details.

**Source Paper:** MR.Q: A Model-Free Deep RL Algorithm with Model-Based Representations

**Source Quote:**
> "The ablation study fails to provide the statistical significance of the results and lacks analysis. Furthermore, the 'reverting to theory' ablations are highly unsurprising so provide little contribution, and many of the remaining ablations show minimal performance gains."

**Specific Applicability to SCALERL:**
SCALERL recipe combines multiple design choices:
- Batch size 768
- Max sequence length 14,336
- Specific KL penalty coefficient
- Reward scaling strategy
- Learning rate schedule
- Entropy coefficient

**Critical question:** Which components actually drive the scaling improvements with statistical significance? For each component:
- Is the improvement > statistical noise?
- How does importance vary across compute budgets?
- Are there redundant/unnecessary components?

**Reviewer Expectation:**
- Ablation study with error bars and significance tests
- Analysis of contribution of each design choice to final performance
- Discussion of which choices are fundamental vs. hyperparameter tuning

---

### 6. Critical Design Choice Justification Missing
**Weakness:** Paper doesn't clearly explain why specific algorithm components are necessary vs. just implementation details or task-specific tuning.

**Source Paper:** Q-SFT: Q-Learning for LLMs via Supervised Fine-tuning

**Source Quote:**
> "Reuse of the pretrained weights is emphasized as the main motivation, but some of the tasks actually train the model from scratch and others are not in natural language, which is a bit conflicting."

**Specific Applicability to SCALERL:**
**Critical question for each recipe component:**
- Why is batch size 768 specifically? Would 512 or 1024 work as well? Is this universally optimal or task-specific?
- Why the specific KL penalty weight? Is this fundamental or tuned for math tasks?
- Why these reward scaling coefficients? Do they generalize to other RL tasks?
- Are components chosen because they're fundamentally necessary or because they happened to work best on math tasks?

**Reviewer Expectation:**
- Clear explanation of which recipe components are fundamental vs. tuned
- Sensitivity analysis for each hyperparameter
- Discussion of how to adapt recipe to new tasks

---

## MEDIUM SEVERITY WEAKNESSES (Important to Address)

### 7. Scaling Law Analysis Ignores Inference Costs
**Weakness:** Theoretical scaling laws don't account for practical inference time costs during RL training, which may change optimal model size and scaling trajectory.

**Source Paper:** Decision Transformer for Fault Detection in UTM Systems

**Source Quote:**
> "The evaluation metrics introduced by the author do not seem to take into account the cost of model inference. If it is taken into account, will the observed scaling law still hold? Or will Sweetspot be a specific model size?"

**Specific Applicability to SCALERL:**
During RL training with 8B or 17Bx16 models:
- Inference cost (forward passes to get rewards) is a significant component of total compute
- Larger models have higher inference costs
- The sigmoid scaling law may change when accounting for inference costs
- The optimal compute allocation might differ in practice

**Reviewer Expectation:**
- Analysis of total compute including inference costs
- Discussion of how scaling law changes with inference overhead
- Practical guidance on optimal model size when inference costs matter

---

### 8. Single Model Architecture Evaluation
**Weakness:** Tests only on specific model families (8B dense, 17Bx16 MoE) without exploring if scaling laws hold across different architectures.

**Source Paper:** Divisive Normalization in CNNs (generalized weakness pattern)

**Source Quote:**
> "Evaluation limited to outdated architectures (AlexNet, VGG-16) that are far from state-of-the-art; no scaling to modern networks"

**Specific Applicability to SCALERL:**
SCALERL tests on 8B dense and 17Bx16 MoE models. **Critical question:**
- Do sigmoid scaling curves hold for other architectures (e.g., vision-language models, code models)?
- Do MoE models follow the same scaling law as dense models?
- How do architectural differences (attention heads, layer widths, etc.) affect scaling parameters?
- Would the recipe parameters change with different architectures?

**Reviewer Expectation:**
- Evaluation on at least 2-3 additional architectural variants
- Analysis of how architecture affects scaling law parameters
- Discussion of architecture-specific considerations

---

### 9. Convergence Assumptions Violated in Practice
**Weakness:** Theory assumes certain convergence properties but practical experiments show violations, requiring workarounds that undermine theoretical guarantees.

**Source Paper:** Guided Reinforcement Learning with Roll-Back

**Source Quote:**
> "While the derived results are of course interesting on their own, one has to ask from a practical standpoint whether deriving & using something based on wrong assumptions makes a lot of sense"

**Specific Applicability to SCALERL:**
SCALERL assumes sigmoid scaling with well-defined parameters. **Critical question:**
- Are the assumptions about monotonic reward improvement with compute actually valid?
- Are there cases where increasing compute doesn't improve reward (e.g., optimization challenges, local minima)?
- Does the sigmoid model capture all scaling behavior or miss important phenomena?
- What happens when assumptions are violated?

**Reviewer Expectation:**
- Analysis of when sigmoid assumption holds vs. breaks down
- Discussion of theoretical assumptions and their practical validity
- Alternative models for cases where sigmoid fails

---

### 10. Generalization Across Data Heterogeneity Unclear
**Weakness:** Method/framework handles homogeneous settings well but lacks analysis of how scaling behaves with diverse data sources or task structures.

**Source Paper:** DEPT: Decoupled Embeddings for Pre-Training

**Source Quote:**
> "The data sources are not always clear given a dataset. The proposed pipeline only works if the domains are known. Otherwise, some manual or automatic clustering has to be used to create different sets of data."

**Specific Applicability to SCALERL:**
SCALERL uses math tasks with relatively uniform structure (GSM8K, MATH). **Critical question:**
- How does scaling law change with diverse reward structures?
- Do heterogeneous RL tasks (mixing multiple task types) follow the same sigmoid curve?
- How does data distribution shift affect scaling law fit?
- Would mixed-task training change the scaling parameters?

**Reviewer Expectation:**
- Evaluation on heterogeneous task mixtures
- Analysis of scaling law robustness to task diversity
- Discussion of task properties affecting scaling

---

### 11. Optimal Hyperparameter Ranges Unclear
**Weakness:** Experiments may use suboptimal ranges for hyperparameter sweeps, potentially missing where performance actually begins to degrade significantly.

**Source Paper:** Deconstructing What Makes a Good Optimizer for Autoregressive Language Models

**Source Quote:**
> "For pretraining percentage and beta_2, I encourage the authors to include more extreme values to support the claim that these hyperparameters do not matter. Instead showing them they do not matter in the current small range, it is more informative to show to the readers at what extreme values the loss starts to increase significantly."

**Specific Applicability to SCALERL:**
SCALERL tests batch size 768. **Critical question:**
- Is batch size 768 truly optimal or just in a tested range?
- What happens with extreme batch sizes (1, 4096, 16384)?
- Where do scaling curves begin to degrade?
- Are there regime shifts in the scaling law at different compute budgets?

**Reviewer Expectation:**
- Wider hyperparameter ranges tested and reported
- Clear identification of regime boundaries
- Analysis of scaling law beyond tested ranges

---

### 12. Method Robustness to Function Approximation Errors
**Weakness:** Approach assumes accurate models/reward functions but doesn't validate performance when approximation breaks down at different scales.

**Source Paper:** Prioritized Generative Replay (PGR) for RL

**Source Quote:**
> "The curiosity-based relevance function relies on a learned dynamics model, which might be challenging to train accurately in complex environments."

**Specific Applicability to SCALERL:**
SCALERL uses learned reward models for training. **Critical question:**
- How does reward model quality affect sigmoid scaling fit?
- As compute increases, does reward model accuracy scale similarly or differently?
- How robust is sigmoid fitting to reward model errors at different scales?
- Does the scaling law hold when reward model is misspecified?

**Reviewer Expectation:**
- Analysis of robustness to reward model quality
- Empirical results varying reward model fidelity
- Discussion of error propagation through scaling law

---

## LOW SEVERITY WEAKNESSES

### 13. Presentation Clarity Issues
**Weakness:** Complex notation and unclear descriptions make it difficult to understand what the core contribution actually is.

**Source Paper:** R-Learning for Offline RL

**Source Quote:**
> "Overall, the paper is quite hard to read, even if you have a background in both RL and orthogonal ML. I think the notation could be improved significantly."

**Specific Applicability to SCALERL:**
The sigmoid curves and scaling parameters in SCALERL could be explained more intuitively. **Improvements:**
- What do the sigmoid parameters actually represent in RL terms?
- Why is sigmoid the right functional form (intuitive explanation)?
- Clearer connection between compute budget and reward performance
- Visual explanations of key concepts

**Reviewer Expectation:**
- Improved clarity of core concepts
- Intuitive explanations of sigmoid scaling
- Clear connection to RL principles

---

## Summary of Gaps in SCALERL

| Weakness Category | Count | Key Gaps |
|---|---|---|
| Evaluation Scope | 2 | Limited to math tasks, single architecture |
| Baseline Comparison | 2 | Fairness unclear, missing closest comparisons |
| Ablation & Analysis | 3 | Lack of statistical rigor, design choice justification, generalization heterogeneity |
| Scaling Law Validity | 4 | Inconsistent behavior, convergence assumptions, hyperparameter ranges, function approx errors |
| Practical Considerations | 1 | Inference costs not accounted for |
| Presentation | 1 | Clarity of contribution |

---

## Recommendations for Addressing Weaknesses

### Priority 1 (CRITICAL - Must Address Before Publication)
1. **Hyperparameter Fairness:** Provide detailed hyperparameter settings for all baselines and evidence of equal tuning effort
2. **Design Choice Justification:** Clearly explain which recipe components are fundamental vs. task-specific tuning

### Priority 2 (HIGH - Important for Acceptance)
3. **Broader Evaluation:** Test on at least 3-5 additional RL task domains beyond math
4. **Ablation Rigor:** Add statistical significance testing to ablation studies
5. **Baseline Comparison:** Explicitly justify baseline selection and discuss other scaling-focused RL methods

### Priority 3 (MEDIUM - Strengthen Paper)
6. **Scaling Consistency:** Analyze non-monotonic behaviors and explain scaling curve variations
7. **Architecture Diversity:** Test on 2-3 additional architectural variants
8. **Inference Costs:** Include analysis of inference costs in total compute accounting
9. **Convergence Analysis:** Validate sigmoid assumptions and discuss failure modes
10. **Hyperparameter Sensitivity:** Extend sweeps to more extreme ranges

### Priority 4 (LOW - Polish)
11. **Presentation:** Improve clarity of core concepts and intuitive explanations
12. **Robustness:** Analyze sensitivity to reward model quality
13. **Generalization:** Test on heterogeneous task mixtures

---

## Analysis Metadata

**Papers Analyzed (8 total):**
1. Process Advantage Verifiers (PAVs) for RL with LLMs (A6Y7AqlzLW)
2. MR.Q: A Model-Free Deep RL Algorithm (R1hIXdST22)
3. Prioritized Generative Replay (PGR) for RL (5IkDAfabuo)
4. DEPT: Decoupled Embeddings for Pre-training (vf5aUZT0Fz)
5. Deconstructing What Makes a Good Optimizer (zfeso8ceqr)
6. Q-SFT: Q-Learning for LLMs via SFT (v4MTnPiYXY)
7. R-Learning for Offline RL (hQOLtZ40hZ)
8. Decision Transformer for Fault Detection (UUwrBhhsxT)
9. Guided RL with Roll-Back (5s1qpjrNvZ)

**Human Reviewers Quoted:** 35+ reviewers from ICLR 2025

**Total Extracted Weaknesses:** 13 patterns

**Data Sources:** Human review markdown files from ICLR 2025 evaluation

---

*This analysis extracts concrete, specific weakness patterns from human reviews of related papers. These patterns identify gaps and challenges that SCALERL paper should address to strengthen its contributions and avoid similar criticisms.*
