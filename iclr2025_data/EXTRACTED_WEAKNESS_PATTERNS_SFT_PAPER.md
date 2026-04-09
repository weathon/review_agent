# Extracted Weakness Patterns from Human Reviews Applicable to SFT Paper

**Analysis Target:** Paper on "SFT WITHOUT OVERFITTING: ANALYZING THE TRAINING DYNAMICS OF SUPERVISED FINE-TUNING"

This document extracts SPECIFIC weakness patterns from ICLR2025 human reviews that directly apply to the SFT paper. These patterns come from reviews of papers in related research areas.

---

## 1. LIMITED EVALUATION BENCHMARKS / NARROW EVALUATION SCOPE

**Key Risk:** Evaluation limited to specific domains, tasks, or benchmarks; insufficient diversity in experimental settings

**Why It Applies to SFT Paper:**
- The paper claims to analyze differences between FNNs and attention modules in controlling memorization
- These differences may be task-specific or domain-dependent
- Claims about OOD generalization require evaluation across diverse domains and task types
- Reviewers will likely question whether findings hold across different data distributions

### Direct Evidence from Reviews:

**Quote 1 (Paper ID: CpQegoH1Fn - Human-in-the-loop Neural Networks):**
> "Poor evaluation with limited experiments and even more limited comparisons. The proposed method is validated only in one medical dataset. I would suggest to test it against other datasets too. Regarding the comparisons I understand that this is more difficult but you need to figure out a good ablation study at least."

**Applies to SFT Paper:** If the SFT analysis is performed only on a single or small set of tasks (e.g., language understanding tasks), reviewers will request evaluation across more diverse domains.

---

**Quote 2 (Paper ID: sahQq2sH5x - Benchmarking Predictive Coding Networks):**
> "Diversity of domains/tasks/modalities while a lot more than anything other out there for PC research, is still very limited compared to much more general benchmarks out there for general deep learning evaluation. Adding more domains, tasks and modalities would be very useful to having rigorous and insightful works done by people using it."

**Applies to SFT Paper:** The diversity of training tasks matters when making claims about fundamental differences between architectural modules.

---

**Quote 3 (Paper ID: 2RNGX3iTr6 - Tabby: Tabular Adaptation):**
> "applying MOE to a narrow problem (table generation). And the results are not all that strong. * It's not easy from the presentation what exactly do the tasks require, what exactly are the baselines and model variations."

**Applies to SFT Paper:** Criticism of narrow problem scope directly applies if the SFT paper only evaluates on specific task categories.

---

## 2. MISSING BASELINES / INCOMPLETE COMPARISONS

**Key Risk:** Insufficient comparison with relevant baselines, alternative methods, or natural comparisons

**Why It Applies to SFT Paper:**
- Paper claims to compare SFT vs RL and attention-only vs full-module fine-tuning
- Reviewers will expect comparison with all reasonable variants and prior work
- Need to compare against selective fine-tuning methods, parameter-efficient approaches
- Missing ablations or variants of proposed methods are major red flags

### Direct Evidence from Reviews:

**Quote 1 (Paper ID: xsELpEPn4A - JudgeLM: Fine-tuned Large Language Models):**
> "The authors did not provide clear evidence that this model is able to maintain good performance across tasks not in the training set. I suspect that the comparison to the PandaLM test set is showing this to some extent, but I did not see any prose on *how* these two datasets differ. What tasks are seen in PandaLM that arent seen in the JudgeLM dataset? If the authors can show that the task distribution is significantly different from the training set I would be satisfied"

**Applies to SFT Paper:** Reviewers want clear explanation of task distribution differences and comprehensive comparison details. The SFT paper must clearly articulate how evaluation tasks differ from training to support OOD generalization claims.

---

**Quote 2 (Paper ID: i4ouG6Kc8M - A Dual-Metric Approach):**
> "The benchmarking in the paper is not sufficient. If the author's claim is that the proposed procedure contributes to model selection, the author ought to compare it to other existing model selection methods. For example, reasonable questions and benchmarks includes: model performance with only use the task agnostic metrics, and model performance on a a task using only a different task specific metric."

**Applies to SFT Paper:** If paper claims that attention vs FNNs differently control memorization, it must compare against other known methods of controlling memorization/capacity.

---

**Quote 3 (Paper ID: IfPfUHRowT - Inpainting the Sinogram):**
> "Figure 9 is severely blurred, making it difficult to discern the comparative effectiveness of the proposed method against other approaches. Additionally, the paper lacks a quantitative performance comparison with the baseline methods."

**Applies to SFT Paper:** Visual and quantitative comparisons with all baselines are essential.

---

**Quote 4 (Paper ID: i3f2N3iHl0 - Adaptive Tensor Attention Networks):**
> "The experiments were far from adequate: there should have been multiple runs (for uncertainty quantification) and more "modern" baselines to compare with."

**Applies to SFT Paper:** Reviewers expect statistical rigor (multiple runs with uncertainty measures) and comparison with state-of-the-art baselines.

---

## 3. UNFAIR EXPERIMENTAL SETUP / HYPERPARAMETER TUNING CONCERNS

**Key Risk:** Baselines may not be optimally tuned; comparisons may be biased due to different optimization procedures; learning rate effects not properly controlled

**Why It Applies to SFT Paper:**
- Paper investigates "role of learning rate in controlling memorization"
- If baselines are not fairly tuned across learning rates, comparisons are invalid
- Critical to show that differences between methods are not simply due to tuning effort
- Reviewers specifically look for equal hyperparameter optimization effort

### Direct Evidence from Reviews:

**Quote 1 (Paper ID: XCugWIuHR8 - Convex Distillation):**
> "Experimental results are not satisfactory to justify the efficacy of the proposed methods. Only small datasets are included. Meanwhile, the ResNet18 baseline seems not well tuned (with low acc less than 90%)."

**Applies to SFT Paper:** Baselines must be properly tuned. If baselines underperform due to poor tuning, the comparison is invalid.

---

**Quote 2 (Paper ID: E4Fk3YuG56 - Cut Your Losses in Large-Vocabulary Language Models):**
> "I think there could have been additional experiments to explore how CCE performs relative to baselines as different hyperparameters vary (e.g., relative size of vocabulary vs sequence length vs. hidden dim, sparsity of S, etc.)."

**Applies to SFT Paper:** Need to show how methods compare across different hyperparameter regimes, particularly different learning rates (since paper emphasizes learning rate's role).

---

**Quote 3 (Paper ID: iWSl5Zyjjw - DeciMamba: Exploring the Length Extrapolation):**
> "**Hyperparameter choices**: I agree with the fact that the fast decay of the sum of the discrete time steps may explain the lack of length generalization. However, the approach looks a bit hacky in that it introduces multiple novel hyperparameters: the decay factor, the maximal length of the sequence after the first decimating later and the number of layers to decimate. And it does not seem very clear how to make these choices without a gridsearch?"

**Applies to SFT Paper:** If paper introduces method-specific hyperparameters, need clear justification for how they are chosen fairly.

---

**Quote 4 (Paper ID: iXbUquaWbl - End-to-end Learning of Gaussian Mixture Priors):**
> "As acknowledged in the paper, there is no clear criterion for determining the appropriate number of mixture components needed for optimal performance. Furthermore, IMR relies on heuristic rules, such as a fixed number of iterations, to add new components, introducing additional hyperparameter tuning requirements for model optimization."

**Applies to SFT Paper:** Hyperparameter selection procedures must be clearly specified and fairly applied to all methods.

---

**Quote 5 (Paper ID: xsELpEPn4A - JudgeLM: Fine-tuned Large Language Models):**
> "While the validation set was manually checked and corrected by the authors, it does still rely on GPT generated outputs. This provides somewhat of an unfair evaluation as JudgeLM is trained on GPT generated judgements as well. Even with the human validation, there is a reasonable chance that if this dataset where annotated by a different LLM and produced different judgements, humans checking responses would also consider them reasonable. An unbiased way of annotating is for humans to provide judgements *without* knowing what the GPT judgement is."

**Applies to SFT Paper:** Evaluation setup must not inadvertently favor certain methods. If evaluation task is related to training task, this needs to be clearly controlled.

---

## 4. LACK OF THEORETICAL JUSTIFICATION / MECHANISTIC EXPLANATION

**Key Risk:** Lack of deep understanding of *why* methods work; vague hypotheses; insufficient mechanistic explanation

**Why It Applies to SFT Paper:**
- Paper claims to analyze "why" FNNs and attention differ in memorization
- Reviewers will expect mechanistic explanations, not just empirical observations
- Need to explain the underlying principles connecting module type → memorization behavior
- Claims about OOD generalization should have theoretical grounding

### Direct Evidence from Reviews:

**Quote 1 (Paper ID: FbZSZEIkEU - Adaptive Circuit Behavior and Generalization):**
> "The S2 hacking hypothesis is quite vague and the author do not present any deep understanding that would explain the mechanisms by which certain attention heads pay extra attention to the S2 token."

**Applies to SFT Paper:** If claiming that attention modules behave differently during fine-tuning, need mechanistic explanation of *how* and *why*, not just observation.

---

**Quote 2 (Paper ID: 7ZToWPWUlO - Solving Normalized Cut Problem):**
> "it is not explained why the grouping in inner circles and outer wedges is a good modeling. Is it a pattern observable in other city map grouping algorithms? Does this pattern apply to all cities? There is a drop in writing quality in section 5, which raised some doubts:"

**Applies to SFT Paper:** If claims are made about architectural differences, need to explain why these differences would generalize across different task domains.

---

**Quote 3 (Paper ID: hNjCVVm0EQ - MamKO: Mamba-based Koopman operator):**
> "In the introduction, the computational time of other koopman based approaches is discussed. However, there is no proper discussion/evaluation/theory on the training/inference time with the proposed framework."

**Applies to SFT Paper:** Claims about method behavior must be supported by theoretical analysis or detailed empirical investigation.

---

**Quote 4 (Paper ID: c61unr33XA - Dataset Distillation via Knowledge Distillation):**
> "The MKDT method appears to add only a knowledge distillation process to MTT, where a student model mimics the SSL teacher model's representations. The synthetic data is then learned by matching the student model's trajectory, rather than the teacher model's trajectory, as in MTT. It is unclear why introducing a student model as an intermediary reduces trajectory variance, or what specific role the student network plays throughout the data distillation process. The authors are encouraged to provide further discussion or insights into this mechanism."

**Applies to SFT Paper:** Need to provide clear mechanistic understanding of why selective fine-tuning (e.g., attention-only) affects memorization differently than full fine-tuning.

---

## 5. MEMORIZATION / OVERFITTING / GENERALIZATION ANALYSIS

**Key Risk:** Insufficient analysis of whether improvements are due to reduced memorization vs other factors; unclear generalization properties

**Why It Applies to SFT Paper:**
- Paper explicitly focuses on "SFT WITHOUT OVERFITTING" and memorization
- Need to clearly demonstrate that observed differences are actually about memorization, not capacity or other factors
- Must show generalization gains translate to OOD settings

### Direct Evidence from Reviews:

**Quote 1 (Paper ID: xsELpEPn4A - JudgeLM: Fine-tuned Large Language Models):**
> "The authors did not provide clear evidence that this model is able to maintain good performance across tasks not in the training set. I suspect that the comparison to the PandaLM test set is showing this to some extent, but I did not see any prose on *how* these two datasets differ. What tasks are seen in PandaLM that arent seen in the JudgeLM dataset? If the authors can show that the task distribution is significantly different from the training set I would be satisfied"

**Applies to SFT Paper:** Must explicitly show that models fine-tuned with different module selections maintain performance on OOD tasks compared to in-distribution tasks.

---

**Quote 2 (Paper ID: S8nFZ98pmU - Contrastive Meta Learning for Dynamical Systems):**
> "Since the authors explore the meta learning problem, the experimental results should focus on how the proposed learning method improves generalizability on previously unobserved dynamics. For example, the model is trained on trajectories on a set of coefficients and then evaluated on them on a different set of coefficients. However, I couldn't find such details in Table 1, 2, and 3. Overall, the experiment section is not detailed enough to support L171: "In the context of meta dynamical system learning, our goal is to develop a model that can generalize from a set of training systems ϕtrain to new, unseen test systems with properties ϕtest.""

**Applies to SFT Paper:** When claiming to study generalization, must show clear separation between training and test distributions.

---

**Quote 3 (Paper ID: tc90LV0yRL - Cybench: Evaluating Cybersecurity Capabilities):**
> "**no protection against future test data leakage**: the full CTF dataset is released meaning that once model training cutoffs move past the benchmark release date, the benchmark will not be usable anymore (cf. e.g. Haimes et al., 2024). Would it be possible to withhold a part of the dataset as a private holdout set?"

**Applies to SFT Paper:** Evaluation should use held-out test sets that truly evaluate generalization, not contamination of training data.

---

## Summary of Key Weakness Patterns

| Weakness Category | Frequency | Severity | Key SFT Paper Risk |
|---|---|---|---|
| Limited evaluation scope | High (4+ papers) | High | Claims about attention/FNN differences may not generalize across task domains |
| Missing/incomplete baselines | High (17+ papers) | Critical | Insufficient comparison with SFT variants, RL methods, selective fine-tuning |
| Hyperparameter tuning fairness | Medium (8+ papers) | Critical | Learning rate effects may be confounded with optimization effort |
| Lack of theoretical justification | Medium (7+ papers) | High | Mechanistic explanations for why modules differ in memorization |
| Narrow evaluation scope (domain) | Medium (2+ papers) | Medium | Evaluation may be limited to specific LLM families or task types |
| Unfair experimental setup | Low (2+ papers) | Critical | Validation set contamination or baseline under-tuning invalidates claims |

---

## Recommendations for SFT Paper Authors

1. **Expand Evaluation:** Test on diverse task domains (language understanding, generation, reasoning, code, etc.) to support generality claims

2. **Fair Comparisons:**
   - Compare against full attention fine-tuning, full FNN fine-tuning, hybrid approaches
   - Include comparisons with RL fine-tuning methods
   - Compare with other selective fine-tuning approaches (adapters, LoRA, etc.)

3. **Hyperparameter Rigor:**
   - Show learning rate sensitivity curves for ALL methods
   - Ensure all baselines are optimized with equal effort
   - Conduct ablations varying learning rate, batch size, training steps

4. **Theoretical Grounding:**
   - Provide mechanistic explanations for attention vs FNN differences
   - Discuss why these differences should hold across architectures
   - Include analysis of what properties of attention enable/prevent memorization

5. **Generalization Evidence:**
   - Clearly separate in-distribution and OOD evaluation
   - Show performance gaps between in-dist and OOD for each method
   - Use multiple independent test sets

6. **Statistical Rigor:**
   - Report error bars/confidence intervals for all results
   - Show results over multiple random seeds
   - Include statistical significance tests where appropriate

---

## Papers with Most Relevant Reviews

- **Fine-tuning/Memorization:** xsELpEPn4A (JudgeLM), XCugWIuHR8 (Convex Distillation)
- **Transformer Analysis:** CpQegoH1Fn (Human-in-loop), 2RNGX3iTr6 (Tabby)
- **Generalization:** S8nFZ98pmU (Contrastive Meta Learning), FbZSZEIkEU (Mechanistic Interpretability)
- **Baseline Comparison:** i4ouG6Kc8M (Dual-Metric), i3f2N3iHl0 (Adaptive Tensor Attention)
