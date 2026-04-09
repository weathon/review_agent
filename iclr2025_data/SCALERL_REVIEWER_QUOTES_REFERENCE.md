# REVIEWER QUOTES REFERENCE FOR SCALERL CRITIQUE

## CRITICAL ISSUES

### 1. BASELINE HYPERPARAMETER FAIRNESS (R1hIXdST22.md - MR.Q Paper)

**Reviewer 2 - Critical Weakness:**
> "The paper provides no detail (that I could find) regarding the hyperparameter settings of the baseline algorithms, nor how or if they were tuned. This is a critical weakness of the paper, as the possibility of untuned baselines undermines the claimed performance improvements."

**Follow-up Question:**
> "Do you have evidence that A) MR.Q is less hyperparameter sensitive than baselines, or B) the best single set of hyperparameters for MR.Q achieves higher performance than the equivalent for a baseline?"

**Applied to SCALERL:**
- Were GRPO, DAPO, CISPO equally tuned?
- Same compute budget for hyperparameter search across all algorithms?
- How many tuning runs per algorithm?

---

### 2. EXTRAPOLATION BEYOND TRAINING RANGE (P7f55HQtV8.md - QuaDiM Paper)

**Reviewer 1 - Core Weakness:**
> "The experimental setup are not clearly presented. For example, the generation protocol of system parameter x ∈ ℝ^(L - 1) is not described in the experiments section, not to mention the training and test set splitting process. This makes it hard to understand under what level QuaDiM extrapolates to 'unseen parameters'."

**Reviewer 1 - Critical Question:**
> "Does QuaDiM extrapolate to unseen system parameter x distribution, likely with disjoint support as the training distribution, e.g., trained on x ∈ [0, 0.8], tested on x ∈ [0.9, 1]?"

**Reviewer 4 - Generalization Concern:**
> "I am not fully convinced why a diffusion model is the most suitable choice for the (non-sequential) quantum property estimation task...the authors would clarify why the proposed QuaDiM would be particularly beneficial for quantum state property estimation, compared to other generative models, in a practical setting (e.g., limited observation as the authors mentioned)."

**Applied to SCALERL:**
- Were sigmoid curves tested on compute ranges disjoint from training?
- Cross-validation: was data split properly? (e.g., train on [1e5, 2e5] compute, test on [3e5, 4e5])
- How does sigmoid compare to power-law extrapolation from prior work?

---

## HIGH PRIORITY ISSUES

### 3. SCALE DOESN'T GUARANTEE MECHANISTIC INSIGHT (m29SV0n6DO.md - Video Pre-training Paper)

**Reviewer 1 - Core Weakness:**
> "While there are a lot of experiments, the takeaway and why is not really answered. For example, why is dVAE and patch-dVAE worse than VQGAN? Is the difference entirely due to label leak through perceptual loss? In Table 6, why is Mamba so much worse than Toto? Furthermore, is the comparison to GPT2 really fair? Toto is based on Llama which is a much newer and upgraded architecture - shouldn't it be expected that Toto would be better?"

**Reviewer 1 - Triviality Concern:**
> "Isn't the scaling behavior somewhat trivial? We know that Llama already exhibits these scaling behaviors, is it surprising that a generative decoder-only model with the same architecture but a different tokenizer would have similar scaling properties?"

**Reviewer 2 - Surface-Level Analysis:**
> "The study only scratches the surface (discussed later). However, given the computation budget required for these experiments, I believe these could be considered as good-to-have but not must-have."

**Applied to SCALERL:**
- Why do sigmoid curves fit RL scaling? Is this expected from LLM behavior?
- Why does GRPO work better than DAPO on 8B but not 17B?
- Are design choice differences mechanistically explained or just empirically observed?

---

### 4. DESIGN CHOICES INSUFFICIENTLY JUSTIFIED (cu2CT2VAvs.md - State Collapse Paper)

**Reviewer 1 - Vague Definitions:**
> "Although the problem is well-motivated, the presentation significantly hinders comprehension...A primary issue is the lack of a formal, clear definition of the new phenomenon, State Collapse (SC), which should be introduced in the early sections. However, I found no clear definition, and the descriptions provided are vague, leaving readers to infer the meaning."

**Reviewer 2 - Circular Definition Problem:**
> "The causal link between high memory strength in this one layer and state collapse is not evident... It also isn't obvious why this would mean that the collapse is 'largely attributable to Bt'."

**Reviewer 4 - Definition Circularity:**
> "This still leaves open the question of the precise definition of 'state collapse'?... In line 401 we are told that 'we regard the minimum training length at which SC [state collapse] does not occur as the state capacity'. So we have finally defined it in Section 5 by referring to 'state collapse' which has also not precisely been defined... It appears the paper is trying to distinguish between a theoretical state capacity and an empirical state capacity, however the two appear to be conflated."

**Applied to SCALERL:**
- Is SCALERL recipe precisely defined? Or vague on design choices?
- Are off-policy, async, and other components clearly defined?
- Circular logic: "sigmoid curves fit scaling" vs "scaling should be sigmoid"?

---

### 5. STATISTICAL SIGNIFICANCE LACKING (R1hIXdST22.md - MR.Q Paper)

**Reviewer 2 - Ablation Rigor:**
> "The ablation study fails to provide the statistical significance of the results and lacks analysis. Furthermore, the 'reverting to theory' ablations are highly unsurprising so provide little contribution, and many of the remaining ablations show minimal performance gains."

**Reviewer 2 - Performance Distribution Analysis:**
> "Some of these components are designed to handle edge cases (e.g. reward scaling handling environments without normalized reward), meaning the benefit is unclear from mean performance across a suite, but may be apparent by examining the performance distribution."

**Applied to SCALERL:**
- What are confidence intervals on sigmoid fit parameters?
- Are performance improvements >1 std-dev above baseline?
- Performance distribution analysis for each design choice?

---

### 6. LIMITED DOMAIN EVALUATION (m29SV0n6DO.md - Video Pre-training Paper)

**Reviewer 1 - Generalization Overstated:**
> "The authors only conducted experiments on the Gym, DMC, and Atari benchmarks, which are classic but relatively homogeneous environments. Claiming generality might be overstated given these limited evaluation."

**Reviewer 1 - Missing Evaluations:**
> "Is there a reason that video-language tasks were not evaluated? Arguably the most common task with video encoders today is video + language (ie captioning, QA, etc)."

**Reviewer 2 - In-Distribution Bias:**
> "The major experiments in the paper are in-domain. The model was trained on ImageNet, Kinetics and Ego4D and later tested on those three datasets."

**Applied to SCALERL:**
- Scaling laws from math tasks apply to coding? reasoning? common sense?
- In-distribution bias: models trained on OpenWebMath, tested on MATH/AIME
- Missing out-of-distribution evaluation on downstream tasks

---

### 7. MODEL SIZE FAIRNESS IN COMPARISONS (R1hIXdST22.md - MR.Q Paper)

**Reviewer 2 - Size Normalization Missing:**
> "Another example is that the paper seems to maintain model sizes from the original implementations of each algorithm. However, the size of the model is not defined by the algorithm, so it would be much more convincing to normalize the parameter count between algorithms and show that MR.Q achieves superior performance at the same size, or has a better scaling curve."

**Applied to SCALERL:**
- Fair comparison between 8B and 17B across loss functions?
- Parameter count normalized across GRPO/DAPO/CISPO implementations?
- Same tokenizer/vocab size across all algorithms?

---

## MEDIUM PRIORITY ISSUES

### 8. MISSING KEY BASELINE COMPARISONS (R1hIXdST22.md - MR.Q Paper)

**Reviewer 2 - PPO Missing:**
> "Some key algorithms are missing, most notably PPO. It is not expected that the authors compare to every popular RL algorithm, but PPO is undoubtedly the closest thing to a 'general-purpose' RL algorithm in current literature. A number of recent methods have also claimed to serve as 'general-purpose' RL algorithms (e.g. PQN, Gallici et al., 2025)."

**Applied to SCALERL:**
- Standard RLHF baselines included?
- Recent RLPF variants compared?
- Industry-standard training recipes used as baselines?

---

### 9. INCONSISTENT SCALING PATTERNS (5IkDAfabuo.md - PGR Paper)

**Reviewer - Inconsistent Trends:**
> "Increasing Synthetic Data ratio does not benefit PGR and the unconditional baseline (SynthER) equally. PGR scales better at r=0.75 than SYNTHER but neither benefits from 0.875. We would think the trend would be consistent? whats the intution behind this?"

**Applied to SCALERL:**
- Are sigmoid curves consistent across all compute ranges?
- Regions where sigmoid fit breaks down?
- Transitions/discontinuities in scaling behavior?

---

### 10. EXPERIMENTAL METHODOLOGY CLARITY (P7f55HQtV8.md - QuaDiM Paper)

**Reviewer 1 - Setup Not Clear:**
> "The experimental setup are not clearly presented. For example, the generation protocol of system parameter x ∈ ℝ^(L - 1) is not described in the experiments section, not to mention the training and test set splitting process."

**Applied to SCALERL:**
- How were compute budgets chosen for different runs?
- Data sampling strategy for empirical study?
- Checkpointing: how many intermediate evaluations?

---

### 11. THEORY-PRACTICE DISCONNECT (R1hIXdST22.md - MR.Q Paper)

**Reviewer 2 - Weak Theory:**
> "The theory is based on highly constrained linear assumptions and provides little justification for the proposed method. The idea of using world model representations for policy learning is intuitive and can be explained much more clearly without sacrificing correctness. I understand there is a pressure on authors to provide 'theoretical rigour', but in this case it adds little to the paper and would be better replaced with an extended analysis of the empirical results."

**Applied to SCALERL:**
- Why do sigmoid curves fit? What's the theoretical basis?
- Is theory from ML scaling (Chinchilla, etc.) applicable?
- Mechanistic explanation for saturation behavior?

---

### 12. DESIGN CHOICES WITHOUT DEEP JUSTIFICATION (gInIbukM0R.md - Training Study Paper)

**Pattern from Review:**
Design choices presented as ablations without explaining WHY they matter

**Applied to SCALERL:**
- Off-policy mechanism: why does it help? When does it hurt?
- Asynchronous training: what bottlenecks does it solve?
- PipelineRL: how does it interact with other design choices?

---

## COMMON PATTERNS ACROSS ALL PAPERS

### Pattern 1: "Large Scale ≠ Large Insight"
**Affecting Papers:** m29SV0n6DO, cu2CT2VAvs, UUwrBhhsxT

Common critique:
- Massive compute budget (~400k GPU hours) used to run many experiments
- Results presented without mechanistic understanding
- High computational cost doesn't justify value of insights

### Pattern 2: "Extrapolation Claims Need Careful Validation"
**Affecting Papers:** P7f55HQtV8, cu2CT2VAvs

Common critique:
- Models trained on limited range
- Extrapolation claims to regions outside training distribution
- Validation typically only on similar or slightly extended ranges
- No rigorous out-of-distribution testing

### Pattern 3: "Design Choices Need Why, Not Just What"
**Affecting Papers:** m29SV0n6DO, cu2CT2VAvs, R1hIXdST22, gInIbukM0R

Common critique:
- Ablations show components exist and have effects
- Lack mechanistic understanding of why they work
- Missing analysis of when/where components are beneficial
- Design justification feels post-hoc

### Pattern 4: "Fair Comparison Requires Explicit Parity"
**Affecting Papers:** R1hIXdST22, m29SV0n6DO

Common critique:
- Baselines may not be equally tuned
- Model sizes differ without normalization
- Hyperparameter search budgets unclear
- Comparison setup details missing

### Pattern 5: "Limited Domains Limit Generalization Claims"
**Affecting Papers:** m29SV0n6DO, R1hIXdST22, UUwrBhhsxT, cu2CT2VAvs

Common critique:
- Evaluation limited to similar/related tasks
- In-distribution evaluation only
- Claims of generality unsupported
- Out-of-distribution robustness untested

---

## REVIEWER CONFIDENCE AND CONSENSUS

**High-Confidence Criticisms** (appear in multiple reviews):
- Baseline fairness issues (R1hIXdST22: 4/4 reviewers mention)
- Limited evaluation scope (m29SV0n6DO: 3/4 reviewers mention)
- Mechanistic understanding lacking (m29SV0n6DO: 3/4 reviewers mention)
- Extrapolation concerns (P7f55HQtV8: 3/4 reviewers mention)

**Reviewer Consensus on Severity:**
- CRITICAL: Baseline fairness, extrapolation validation
- HIGH: Mechanistic understanding, statistical rigor, generalization
- MEDIUM: Theory-practice fit, design choice depth

---

## HOW TO USE THESE QUOTES

**For Thesis Argument:**
1. Identify which quote best matches your concern
2. Use file reference (e.g., R1hIXdST22.md) to provide context
3. Apply to SCALERL using "Applied to SCALERL:" section

**For Comparative Analysis:**
- Group quotes by weakness type (Cluster A-E in matrix)
- Use severity to prioritize which weaknesses matter most
- Reference Pattern 1-5 for meta-level critique

**For Structured Feedback:**
1. Use issue category from quote
2. Cite reviewer quote directly
3. Explain how it applies to SCALERL
4. Suggest resolution

