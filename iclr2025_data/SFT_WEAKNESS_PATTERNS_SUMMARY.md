# Weakness Patterns Relevant to SFT Paper
## "SFT WITHOUT OVERFITTING: ANALYZING THE TRAINING DYNAMICS OF SELECTIVE FINE-TUNING"

### Overview
Extracted from human review dataset with 200+ ICLR 2025 reviews. Patterns identified using targeted grep searches across multiple weakness categories.

---

## 1. LIMITED TASK DIVERSITY & NARROW EVALUATION SCOPE

### Pattern A: Insufficient Breadth of Evaluation Tasks
**Source Files:** `WCRQFlji2q.md`, `8Q0beBHq41.md`, `9NfHbWKqMF.md`

**Weakness Quote:**
> "Having the initial question formulated as 'Can VLMs play ARPGs?' and show insights from only one title, on a series of limited tasks, makes it harder to claim that the question is being comprehensively addressed."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/8Q0beBHq41.md` (lines 149)

**Relevance to SFT Paper:**
The paper evaluates selective fine-tuning (attention-only vs FNN-only vs full-model) on only two reasoning tasks (GeneralPoints and V-IRL). This pattern suggests reviewers expect evaluation on multiple diverse tasks to substantiate generalization claims. The current scope may be seen as narrow for claiming OOD generalization insights.

---

### Pattern B: Missing Scope Clarification
**Source File:** `9NfHbWKqMF.md`

**Weakness Quote:**
> "The study is primarily focused on object-centric cases, despite the availability of scene-level 3D datasets. Expanding the scope to scene-wise data could provide a broader basis for extrapolation and robustness in more complex environments."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/9NfHbWKqMF.md` (lines 12)

**Relevance to SFT Paper:**
SFT paper focuses on two reasoning tasks. Reviewers typically expect evidence that patterns discovered are not limited to the specific chosen domains but generalize more broadly across task types (e.g., different reasoning paradigms, different model sizes, different task difficulties).

---

## 2. FAIR COMPARISON CONCERNS & EXPERIMENTAL SETUP

### Pattern A: Hyperparameter Tuning Fairness
**Source File:** `dGMJ93qpfq.md`

**Weakness Quote:**
> "The proposed method brings less than 0.5%p compared to baseline for most of evaluating settings. I suggest the authors to conduct repeated experiments and present the standard deviation, and show that the proposed method is statistically significant"

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/dGMJ93qpfq.md` (lines 55)

**Relevance to SFT Paper:**
When comparing attention-only, FNN-only, and full-model fine-tuning strategies, it's critical that each approach receives equal hyperparameter tuning effort. This pattern suggests the paper should:
- Document hyperparameter search space for each variant
- Show confidence intervals/error bars
- Provide statistical significance tests
- Ensure learning rates, batch sizes, etc. are fairly tuned for all variants

---

### Pattern B: Incomplete Baseline Consistency
**Source File:** `dGMJ93qpfq.md`

**Weakness Quote:**
> "It is recommended to consider the baselines consistent in every task, as some baselines are missing in table 1 and 2. The results of baselines used in table 3 are needed in table 1 & 2."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/dGMJ93qpfq.md` (lines 138)

**Relevance to SFT Paper:**
All fine-tuning variants (attention-only, FNN-only, full) should be evaluated on all tasks consistently. Missing comparisons make it hard to assess which approach is universally better vs. task-specific.

---

### Pattern C: Sensitivity Analysis & Parameter Effects
**Source File:** `FyMjfDQ9RO.md`

**Weakness Quote:**
> "The unstable training process and sensitivity to hyperparameters are known problems of EMA-based self-distillation. This fact might lead to challenges in scaling the model."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/FyMjfDQ9RO.md` (lines 140)

**Relevance to SFT Paper:**
Different fine-tuning strategies may have different sensitivities to hyperparameters. The paper should document:
- Learning rate sensitivity for each variant
- Training stability across different seeds
- Impact of initialization strategies
- Convergence behavior differences

---

## 3. LACK OF THEORETICAL JUSTIFICATION & MECHANISTIC UNDERSTANDING

### Pattern A: Insufficient Explanation of Why Methods Work
**Source File:** `TH4gKbZS1E.md`

**Weakness Quote:**
> "The comparison primarily centers on the resulting test accuracy curves; however, it lacks the necessary theoretical justification and fundamental analysis to substantiate the findings."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/TH4gKbZS1E.md` (lines 127)

**Relevance to SFT Paper:**
Simply showing that attention-only fine-tuning outperforms full-model fine-tuning is insufficient. Reviewers expect mechanistic analysis answering:
- WHY does selective fine-tuning reduce overfitting?
- What mechanisms differ between attention and FNN layers during fine-tuning?
- How do gradient flows differ across the variants?

---

### Pattern B: Missing Qualitative Reasoning
**Source File:** `dGMJ93qpfq.md`

**Weakness Quote:**
> "Why and how is the Voronoi clustering enhancing the alignment of text and vision? Why not direct addition of class features into the text embeddings just like CoCoOp? Why the patch level features are working better than class features here? I don't want any experimental data, but concrete qualitative reasons are needed."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/dGMJ93qpfq.md` (lines 134)

**Relevance to SFT Paper:**
Provide intuitive explanations for the observed differences:
- Why should attention mechanisms be more robust to fine-tuning than FNNs?
- What properties of each layer type make them suitable/unsuitable for selective tuning?
- What theoretical principles support the findings?

---

### Pattern C: Lack of Theoretical Contribution
**Source File:** `ZSdubdbOoi.md`

**Weakness Quote:**
> "There is no theoretical guarantee of the learning outcome. This makes the whole theoretical part weak. Is there a chance to provide any theoretical guarantee on the performance of the policy learned by SRPO under some assumptions?"

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/ZSdubdbOoi.md` (lines 18)

**Relevance to SFT Paper:**
Consider providing theoretical analysis such as:
- Convergence guarantees for different selective fine-tuning strategies
- Generalization bounds comparing variants
- Analysis of when and why overfitting occurs in full-model vs. selective fine-tuning

---

## 4. GENERALIZATION BEYOND TESTED DOMAINS (OOD GENERALIZATION)

### Pattern A: Limited Evidence of Cross-Domain Transfer
**Source File:** `WCRQFlji2q.md`

**Weakness Quote:**
> "Lack of model diversity: Only the Gemma models are tested. It would be better if we can see the same findings on LLaMA or other models to show how universal this idea holds."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/WCRQFlji2q.md` (lines 142)

**Relevance to SFT Paper:**
The paper tests on two reasoning tasks (GeneralPoints, V-IRL). Reviewers would expect:
- Evidence that findings hold across different model architectures
- Different base model sizes
- Different types of reasoning tasks beyond the two presented
- Evidence of robustness to task distribution shifts

---

### Pattern B: Scope Limitations Affecting Generalization Claims
**Source File:** `hQOLtZ40hZ.md`

**Weakness Quote:**
> "There's a mismatch between the goal and the experimental results. Moreover, this limits the scope of baseline methods as most methods aim to obtain RL policies other than merely estimating difference of Q."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/hQOLtZ40hZ.md` (lines 16)

**Relevance to SFT Paper:**
The paper focuses on OOD generalization on reasoning tasks. Make sure:
- Evaluation metrics directly measure OOD generalization (not just in-distribution accuracy)
- The choice of evaluation tasks genuinely tests the hypotheses
- Results generalize beyond the specific reasoning paradigms chosen

---

## 5. INCOMPLETE BASELINES & MISSING COMPARISONS

### Pattern A: Absence of Concrete Baselines
**Source File:** `l4fMj4Vnly.md`

**Weakness Quote:**
> "Absence of a concrete baseline - the baseline the authors compare ADIFF against is a clearly inferior version of the same model, and thus guaranteeing that ADIFF would perform better than this baseline. This baseline is more of an ablation of the components of the model."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/l4fMj4Vnly.md` (lines 24)

**Relevance to SFT Paper:**
Ensure comparisons include:
- Standard full-model fine-tuning (clear baseline, not ablation)
- Other parameter-efficient fine-tuning methods (LoRA, adapters)
- Layer-specific fine-tuning strategies from prior work
- Don't only compare variants of the proposed approach

---

### Pattern B: Missing Method Diversity in Comparisons
**Source File:** `dGMJ93qpfq.md`

**Weakness Quote:**
> "Why only PromptSRC+PAP and DePT+PAP? PAP can also be attached with CoPrompt, MaPLe, CoOp, KgCoOp etc."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/dGMJ93qpfq.md` (lines 136)

**Relevance to SFT Paper:**
Test selective fine-tuning strategies on multiple base models or architectures, not just the one chosen. Show that the findings aren't specific to one model family.

---

## 6. PARAMETER EFFICIENCY & SELECTIVE FINE-TUNING

### Pattern A: Complexity vs. Efficiency Tradeoff Not Justified
**Source File:** `dGMJ93qpfq.md`

**Weakness Quote:**
> "Overall, the final solution seems to be too complicated. There are quite a few components, losses, and tricks. There was not always a good explanation of why a certain component/trick would help. Some ablation experiments are still missing. The number of trained parameters is much higher than PromptSRC and DePT. The training cost is also much higher..."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/dGMJ93qpfq.md` (lines 15)

**Relevance to SFT Paper:**
When arguing for selective fine-tuning, must justify complexity:
- Document parameter count for each variant
- Compare computational costs (FLOPs, memory, training time)
- Show cost-benefit tradeoffs clearly
- Provide ablation studies for each variant

---

### Pattern B: Missing Ablation Evidence
**Source File:** `l4fMj4Vnly.md`

**Weakness Quote:**
> "A major contribution of this work is 'cross projection' layer that distinguishes ADIFF from baseline and existing literature. However there is Insufficient evidence of importance of the cross projection layer due to two factors: [ablations are incomplete]"

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/l4fMj4Vnly.md` (lines 27)

**Relevance to SFT Paper:**
For selective fine-tuning (attention-only vs FNN-only vs full):
- Ablate each component independently
- Show what happens when you remove attention tuning, remove FNN tuning, etc.
- Quantify the contribution of each variant

---

## 7. STATISTICAL RIGOR & EMPIRICAL VALIDATION

### Pattern A: Lack of Statistical Significance Testing
**Source File:** `dGMJ93qpfq.md`

**Weakness Quote:**
> "The performance improvement over baseline seems marginal. The proposed method brings less than 0.5%p compared to baseline for most of evaluating settings. I suggest the authors to conduct repeated experiments and present the standard deviation, and show that the proposed method is statistically significant"

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/dGMJ93qpfq.md` (lines 55)

**Relevance to SFT Paper:**
- Run multiple random seeds
- Report mean ± std or confidence intervals
- Perform significance tests (t-tests) for key comparisons
- Show p-values when claiming improvements

---

### Pattern B: Insufficient Analysis Depth
**Source File:** `TH4gKbZS1E.md`

**Weakness Quote:**
> "The author merely compared KAN and MLP experimentally but did not analyze why KAN or MLP performs poorly in certain situations."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/TH4gKbZS1E.md` (lines 51)

**Relevance to SFT Paper:**
Beyond showing empirical results:
- Analyze failure modes for each fine-tuning strategy
- Show which task categories benefit from which approach
- Provide per-sample or per-layer analysis

---

## 8. REPRODUCIBILITY & EXPERIMENTAL DESIGN

### Pattern A: Insufficient Experimental Details
**Source File:** `dGMJ93qpfq.md`

**Weakness Quote:**
> "Reproducibility: This paper would have good reproducibility as most parts are simple to implement. It would be nice to supply where the Voronoi-based clustering code can be found, further improving the reproducibility."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/dGMJ93qpfq.md` (lines 190)

**Relevance to SFT Paper:**
Provide complete implementation details:
- Exact hyperparameters for each variant (learning rates, warmup schedules, etc.)
- Dataset splits and preprocessing details
- Code availability (GitHub repository)
- Reproducibility checklist items

---

## 9. OVERFITTING & GENERALIZATION ANALYSIS

### Pattern A: Overfitting Problem not Sufficiently Addressed
**Source File:** `dGMJ93qpfq.md`

**Weakness Quote:**
> "This paper addresses the overfitting problem in prompt tuning for CLIP on downstream vision tasks. The authors propose several regularization methods based on patch-level consistency losses..."

**Source:** `/home/wg25r/review_agent/iclr2025_data/human_reviews/dGMJ93qpfq.md` (lines 84)

**Relevance to SFT Paper:**
Title explicitly mentions "WITHOUT OVERFITTING." Must demonstrate:
- Clear evidence that selective fine-tuning reduces overfitting compared to full-model
- How overfitting is measured (validation loss behavior, train-test gap, etc.)
- Analysis of why selective fine-tuning prevents overfitting
- Validation set behavior across variants

---

## Summary of Actionable Recommendations

Based on these patterns, the SFT paper should:

1. **Expand evaluation scope:** Add more reasoning tasks or diverse task types beyond GeneralPoints and V-IRL
2. **Test across models:** Validate findings on different base model architectures/sizes
3. **Fair comparison:** Document hyperparameter tuning effort for all variants equally
4. **Statistical rigor:** Multiple seeds, confidence intervals, significance tests
5. **Mechanistic analysis:** Explain WHY selective fine-tuning works (gradient flow, parameter importance, etc.)
6. **Complete baselines:** Include standard methods (LoRA, adapters, other selective strategies)
7. **Ablation studies:** Show contribution of each component
8. **Reproducibility:** Code, hyperparameters, implementation details
9. **Cost-benefit analysis:** Document parameter counts and computational costs
10. **Overfitting evidence:** Clear metrics showing OOD generalization improvement

---

## File References Extracted

- `/home/wg25r/review_agent/iclr2025_data/human_reviews/TH4gKbZS1E.md` - Lack of theoretical justification pattern
- `/home/wg25r/review_agent/iclr2025_data/human_reviews/WCRQFlji2q.md` - Lack of model diversity pattern
- `/home/wg25r/review_agent/iclr2025_data/human_reviews/dGMJ93qpfq.md` - Marginal improvements, complexity, fairness patterns
- `/home/wg25r/review_agent/iclr2025_data/human_reviews/FyMjfDQ9RO.md` - Sensitivity to hyperparameters pattern
- `/home/wg25r/review_agent/iclr2025_data/human_reviews/l4fMj4Vnly.md` - Missing baselines, insufficient ablations
- `/home/wg25r/review_agent/iclr2025_data/human_reviews/ZSdubdbOoi.md` - Lack of theoretical guarantees
- `/home/wg25r/review_agent/iclr2025_data/human_reviews/9NfHbWKqMF.md` - Limited scope, OOD challenges
- `/home/wg25r/review_agent/iclr2025_data/human_reviews/hQOLtZ40hZ.md` - Mismatch between goals and evaluation
- `/home/wg25r/review_agent/iclr2025_data/human_reviews/8Q0beBHq41.md` - Narrow task diversity pattern
