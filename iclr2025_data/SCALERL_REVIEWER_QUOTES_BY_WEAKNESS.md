# SCALERL: Specific Reviewer Quotes Organized by Weakness Category

## CRITICAL: Baseline Hyperparameter Fairness

### Quote 1: Undisclosed Baseline Hyperparameters
**Source**: MR.Q (R1hIXdST22) - Reviewer 2  
**Context**: Reviewer examining multi-benchmark RL comparison

> "Unfortunately, the paper does the opposite and provides no detail (that I could find) regarding the hyperparameter settings of the baseline algorithms, nor how or if they were tuned. This is a critical weakness of the paper, as the possibility of untuned baselines undermines the claimed performance improvements."

**Application to SCALERL**:
- Are GRPO and DAPO hyperparameters disclosed?
- How were they tuned relative to SCALERL's 400,000+ GPU-hour study?

### Quote 2: Evidence of Unequal Comparison Setup
**Source**: MR.Q (R1hIXdST22) - Reviewer 2

> "One example of this is that DreamerV3 outperforms MR.Q on only 1/4 benchmarks (Atari). Notably, Atari is also the only benchmark for which the results come from the reference work. For the remaining 3 benchmarks, the authors run DreamerV3 themselves. This does not imply that the authors failed to tune DreamerV3 or that the comparison was unfair, however, given the lack of detail regarding their tuning procedure or the hyperparameter sensitivity of the methods, the result is seriously undermined."

**Application to SCALERL**:
- Did SCALERL run baselines themselves or report reference numbers?
- What was the tuning procedure for baselines vs. SCALERL?

### Quote 3: Model Size Normalization Missing
**Source**: MR.Q (R1hIXdST22) - Reviewer 2

> "Another example is that the paper seems to maintain model sizes from the original implementations of each algorithm. However, the size of the model is not defined by the algorithm, so it would be much more convincing to normalize the parameter count between algorithms and show that MR.Q achieves superior performance at the *same* size, or has a better scaling curve."

**Application to SCALERL**:
- Are GRPO and DAPO runs on equivalent model architectures?
- Could fairness be improved by normalizing model sizes?

---

## HIGH: Limited Evaluation Scope

### Quote 1: Single Task Domain Evaluation
**Source**: PAVs (A6Y7AqlzLW) - Reviewer 1

> "Evaluation only on mathematical reasoning tasks... All experiments on a single model family (Gemma)"

**Application to SCALERL**:
- SCALERL limited to math tasks (AIME, etc.) - does this generalize?
- Only tested on specific model families (8B Llama, 17B MoE)

### Quote 2: Lack of Diverse Reasoning Domains
**Source**: PAVs (A6Y7AqlzLW) - Reviewer 4

> "The (final) experimental section seems a bit too narrow. Although authors reference the 'conventional belief' of using mathematical correctness or relevance of steps in the introduction, they only compare to the baseline of ORM reward. It is difficult to judge how much of an improvement we should expect in other domains, as other SOTA MATH models are only briefly referenced (and not compared to) in the appendix."

**Application to SCALERL**:
- Does SCALERL recipe work equally well for code, summarization, dialogue?
- How would results change in different domains?

### Quote 3: Specific Baseline Comparison Issues
**Source**: PAVs (A6Y7AqlzLW) - Reviewer 5

> "The paper focuses its empirical evaluation on ORMs and states that there are major advantages w.r.t. them but I believe that a fair comparison would be to use PRMs since they are the closest possible baseline."

**Application to SCALERL**:
- Are GRPO/DAPO truly the closest scaling-focused baselines?
- Should other scaling studies be compared directly?

---

## HIGH: Inconsistent Scaling Behavior

### Quote 1: Non-Monotonic Scaling Not Explained
**Source**: PGR (5IkDAfabuo) - Reviewer 4

> "Increasing Synthetic Data ratio does not benefit PGR and the unconditional baseline equally. PGR scales better at r=0.75 than SYNTHER but neither benefits from 0.875. We would think the trend would be consistent? whats the intution behind this?"

**Application to SCALERL**:
- Is SCALERL's sigmoid scaling monotonic across full range?
- Are there plateaus or non-monotonic regions?
- Why sigmoid over power law - empirical justification needed?

### Quote 2: Lack of Consistency Analysis
**Source**: Optimizer Study (zfeso8ceqr) - Reviewer 1

> "I am convinced that for sufficiently small WD and epsilon, the final validation losses are nearly the same. But for warmup, batch size, and beta_2, the range are smaller and thus I am not convinced. The ratio between maximal and minimal batch sizes tried in the experiments are just 8, while the ratio is more than a thousand for learning rate."

**Application to SCALERL**:
- Is batch size 768 tested across sufficient range?
- Should extreme values be tested to identify regime shifts?

---

## HIGH: Ablation Studies Lack Statistical Rigor

### Quote 1: No Significance Testing
**Source**: MR.Q (R1hIXdST22) - Reviewer 2

> "The ablation study fails to provide the statistical significance of the results and lacks analysis. Furthermore, the 'reverting to theory' ablations are highly unsurprising so provide little contribution, and many of the remaining ablations show minimal performance gains."

**Application to SCALERL**:
- Does SCALERL include error bars on ablation results?
- Are improvements statistically significant vs. noise?
- Which design choices are fundamental vs. implementation details?

### Quote 2: Missing Component Analysis
**Source**: MR.Q (R1hIXdST22) - Reviewer 2

> "Some of these components are designed to handle edge cases (e.g. reward scaling handling environments without normalized reward), meaning the benefit is unclear from mean performance across a suite, but may be apparent by examining the performance distribution. For these in particular, further analysis of the results would strengthen the contribution of each component."

**Application to SCALERL**:
- Is each recipe component necessary?
- Are some designed for edge cases with unclear general benefit?

### Quote 3: Minimal Gains in Ablations
**Source**: MR.Q (R1hIXdST22) - Reviewer 4

> "It would be great to see a comparison to your method, but with additional synthetic trajectories simulated by your learned model. Without this experiment, it is hard to know if the compromise between model-based RL and model-free RL is the right trade-off."

**Application to SCALERL**:
- Have ablations been run with sufficient replicates?
- Are effect sizes practically meaningful beyond noise?

---

## HIGH: Missing Baseline Justification

### Quote 1: Closest Baseline Unclear
**Source**: PAVs (A6Y7AqlzLW) - Reviewer 5

> "The paper focuses its empirical evaluation on ORMs and states that there are major advantages w.r.t. them but I believe that a fair comparison would be to use PRMs since they are the closest possible baseline... I believe that to truly understand the utility of PAVs as neural verifiers/reward models, one would need to compare them with the same search strategy but just a different ranking scheme (PRMs vs PAVs)."

**Application to SCALERL**:
- Why compare to GRPO/DAPO over other scaling studies?
- What other scaling-focused RL works exist?

### Quote 2: Multiple Baseline Gaps
**Source**: MR.Q (R1hIXdST22) - Reviewer 2

> "Some key algorithms are missing, most notably PPO. It is not expected that the authors compare to every popular RL algorithm, but PPO is undoubtedly the closest thing to a 'general-purpose' RL algorithm in current literature."

**Application to SCALERL**:
- Are there other foundational scaling studies on RL?
- Should other scaling law papers be included?

---

## MEDIUM: Single Architecture Evaluation

### Quote 1: Architecture Specificity
**Source**: General pattern across scaling papers

**Implicit issue**: Architecture affects scaling laws significantly

**Application to SCALERL**:
- Do 8B dense and 17Bx16 MoE scale identically?
- How would results differ for other architectures?
- Should other model families be tested?

---

## MEDIUM: Hyperparameter Range Questions

### Quote 1: Insufficient Range Testing
**Source**: Optimizer Study (zfeso8ceqr) - Reviewer 1

> "For pretraining percentage and beta_2, I encourage the authors to include more extreme values to support the claim that these hyperparameters do not matter. Instead showing them they do not matter in the current small range, it is more informative to show to the readers at what extreme values the loss starts to increase significantly."

**Application to SCALERL**:
- Should batch size be tested beyond 768?
- Should KL penalty be tested over wider range?
- What are regime boundaries?

### Quote 2: Unclear Optimal Range
**Source**: Optimizer Study (zfeso8ceqr) - Reviewer 4

> "The plots comparing final validation loss (e.g., Figure 1) are presented so that each optimizer's optimal learning rate aligns, with the x-axis showing multiples of this optimal learning rate. However, why should different optimizers be compared over the same scale of learning rate values?"

**Application to SCALERL**:
- Are SCALERL parameters tested against baselines' optimal ranges?
- Could fairness be improved with broader testing?

---

## MEDIUM: Generalization Across Task Heterogeneity

### Quote 1: Uniform Data Assumptions
**Source**: DEPT (vf5aUZT0Fz) - Reviewer pattern

**Implicit issue**: Scaling laws may not hold under data heterogeneity

**Application to SCALERL**:
- SCALERL uses uniform math tasks (GSM8K, MATH)
- How do scaling curves change with mixed task types?
- Does reward structure heterogeneity affect sigmoid fit?

---

## MEDIUM: Inference Cost Considerations

### Quote 1: Computational Cost Not Addressed
**Source**: Consistency Models (LyJi5ugyJx) - Reviewer 3

> "In Sections 4.1 and 5.2, the paper discusses the training compute of sCM. However, including a comparison of compute efficiency with other models (e.g., ECT) would be more insightful, maybe a table or figure comparing the compute efficiency (e.g., FLOPs or training time) of sCM against ECT and other relevant baselines for a given performance level."

**Application to SCALERL**:
- Does SCALERL account for inference compute in scaling law?
- How much do forward passes for rewards contribute?
- Should inference costs change optimal compute allocation?

---

## MEDIUM: Design Choice Justification

### Quote 1: Lack of Design Motivation
**Source**: Consistency Models (LyJi5ugyJx) - Reviewer 3

> "Several design choices appear arbitrary and lack supporting evidence. For example, in Section 4.1, the authors discuss the preference for Adaptive Double Normalization over AdaGN, but there is no experimental evidence supporting this choice. It would be more insightful to add a Figure similar to Figure 5 show experimental comparison between Adaptive Double Norm and AdaGN."

**Application to SCALERL**:
- Why batch size 768 specifically?
- Why these KL penalty values?
- Where are ablations comparing design choices?

---

## MEDIUM: Convergence Assumptions

### Quote 1: Assumption Validity Questioned
**Source**: General pattern across multiple papers

**Implicit issue**: Theoretical assumptions often violated in practice

**Application to SCALERL**:
- Does sigmoid monotonically improve with all compute budgets?
- Are there optimization challenges causing saturation?
- What explains any non-monotonic behaviors?

---

## LOW: Presentation Clarity

### Quote 1: Notation and Clarity Issues
**Source**: MR.Q (R1hIXdST22) - Reviewer 4

> "Overall, the paper is quite hard to read, even if you have a background in both RL and orthogonal ML. I think the notation could be improved significantly."

**Application to SCALERL**:
- Are sigmoid curves explained intuitively?
- Is the connection to RL principles clear?
- Could visualizations be improved?

### Quote 2: Complex Methods Not Well Explained
**Source**: Optimizer Study (zfeso8ceqr) - Reviewer 2

> "The section on positional embeddings (line 269 and on) lacks details to be fully understandable without having to read another paper. Maybe beefing up that section would make the paper more self-contained."

**Application to SCALERL**:
- Is the sigmoid model fully explained?
- Are parameter meanings clear without external references?

---

## Summary: High-Impact Concerns

### Most Critical for SCALERL:
1. **Baseline hyperparameter disclosure and fairness** - determines if improvements are algorithmic or from tuning
2. **Evaluation scope beyond math tasks** - determines generalizability of scaling law
3. **Statistical rigor in ablations** - determines which components actually matter
4. **Sigmoid model validation** - determines theoretical soundness of core contribution

### Questions Reviewers Would Ask SCALERL:
1. "How can you be sure GRPO and DAPO were tuned fairly?"
2. "Does this work beyond math? What about code, summarization, dialogue?"
3. "Which recipe components are statistically significant?"
4. "Why sigmoid? Did you test power laws?"
5. "What happens with extreme batch sizes or KL penalties?"
6. "Do other model families show the same scaling curves?"

---

**Report Generated**: 2026-04-08
**Source**: Analysis of 26 human reviews from 5 relevant ICLR 2025 papers
