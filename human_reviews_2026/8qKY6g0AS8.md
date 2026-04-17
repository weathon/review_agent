# Exploring Imbalanced Annotations for Effective In-Context Learning

- Decision: Reject
- Scores: 2, 2, 4, 2, 6

## Abstract
Large language models (LLMs) have shown impressive performance on downstream tasks through in-context learning (ICL), which heavily relies on the quality of demonstrations selected from a large annotated dataset. However, real-world datasets often exhibit long-tailed class distributions, where a few classes occupy most of the data while most classes are under-represented. In this work, we show that imbalanced annotations hurt the ICL performance by degrading the Task Learning ability and cannot be mitigated by varying the demonstration sets, selection methods, calibration methods and rebalancing methods.  To circumvent the issue, we propose a simple and effective approach termed Reweighting with Importance Factors (dubbed RIF) to enhance ICL performance under class imbalance. In particular, RIF constructs a balanced subset to estimate importance factors for each class: the ratio between the joint distribution of demonstration sets selected from balanced and imbalanced datasets. Then, we leverage the factors to re-weight the scoring function (e.g., the cosine similarity score used in TopK) during demonstration selection. In effect, RIF prevents over-selection from dominant classes while preserving the efficacy of current selection methods. Extensive experiments on common benchmarks demonstrate the effectiveness of our method, improving the average accuracy of current selection methods by up to 5.60\%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the performance degradation of In-Context Learning when demonstration examples are selected from class-imbalanced annotated datasets. The authors show this issue persists despite varying selection methods, or calibration techniques. They propose Reweighting with Importance Factors (RIF), a method that estimates class weights using a balanced subset and Bayesian optimization. During inference, RIF re-weights the scores used by standard selection methods (e.g., TopK) to counteract the imbalance.

### Strengths
1. Addresses the important and practical problem of performance degradation in In-Context Learning due to class imbalance in the annotated dataset, a real-world scenario. 
2. Proposes RIF, a method involving 'pre-hoc' score re-weighting based on importance factors estimated via Bayesian optimization on a balanced subset, conceptually contrasting with 'post-hoc' calibration.
3. RIF is technically sound and demonstrates consistent empirical improvements over vanilla selection and calibration baselines across extensive experiments (multiple datasets, imbalance ratios, LLMs).
4. Shows better computational efficiency compared to calibration methods.

### Weaknesses
1. Further Validation Suggested for Conceptual Advantages: The paper compellingly argues that RIF's advantage over undersampling lies in preserving valuable information, potentially leading to better generalization and robustness. To further solidify this key argument, the paper would benefit significantly from direct experimental validation comparing RIF against undersampling on metrics related to generalization (e.g., out-of-distribution performance) or robustness.
2. Opportunity for Broader Baseline Comparison: To provide a more comprehensive picture of RIF's performance relative to alternative strategies, the authors might consider including comparisons against potentially strong baselines that combine existing techniques. For example, evaluating RIF against hybrid approaches, such as undersampling and calibration, could further contextualize its advantages and provide a more rigorous assessment.
3. Clarifying Positioning Relative to Prior Work: While the RIF mechanism itself is novel, the paper's positioning as the "first systematic study" on "Imbalanced Annotation" affecting selection could be enhanced for clarity. Discussing how RIF specifically differs from recent related works [1,2] that also consider class imbalance in the annotated dataset for ICL demonstration selection would help readers better appreciate its unique contributions against the current research landscape.
4. Enhancing Clarity on Hyperparameter K': The results indicate performance sensitivity to the candidate size K' (Figure 2b) To aid reproducibility and practical application, the paper could be enhanced by explicitly reporting the K' value used for the main results (Table 1). Additionally, offering practical guidance or analysis on selecting K' considering the trade-off between performance and computational cost, would be valuable for readers.
5. Minor Presentation Flaws: There are inconsistencies (e.g., ByDC vs. ByCS typo), missing visualizations (e.g., RIF's direct impact on Head/Tail ratio in selection), and a lack of precision in describing the imbalanced dataset generation process.

[1] Strategic Demonstration Selection for Improved Fairness in LLM In-Context Learning (EMNLP’24)
[2] IM-Context: In-Context Learning for Imbalanced Regression Tasks (TMLR’24)

### Questions
1. Clarification on Table 1 and the Implication of phi=1 Performance: Does it evaluate performance on the same (balanced) test set while varying the imbalance ratio (phi) of only the annotation dataset used for demonstration selection? If this understanding is correct, the results consistently show that using an annotation dataset balanced via undersampling (phi=1) achieves the highest ICL performance across all scenarios. Doesn't this finding suggest that simply undersampling a real-world imbalanced dataset to achieve balance might be the most effective strategy for maximizing ICL performance on these benchmarks, potentially more so than applying RIF to the imbalanced dataset?
2. Comparison to Hybrid/Stronger Baselines: Could you comment on the potential performance of hybrid baselines like undersampling with calibration or augmentation with oversampling?
3. Understanding RIF's Distinctions from Related Work: How does RIF's contribution specifically differ from recent ICL selection methods [1,2] that address data imbalance implicitly while focusing on fairness (e.g., prioritizing underrepresented groups), or imbalanced regression?
4. Clarification on K' Value and Guidance: What specific value of K' was used to generate the main results presented in Table 1? Could you provide practical guidance or heuristics for setting K' based on dataset characteristics (e.g., imbalance ratio, tail class size) and computational constraints? Furthermore, what is the empirical computational overhead (e.g., latency increase) observed when significantly increasing K' (e.g., from 100 to 1600 in your Fig 2b experiments)?
5. Imbalanced Dataset Generation Details: Could you please provide the precise mathematical function or procedure used to determine the number of samples for intermediate classes when creating the synthetic imbalanced datasets with phi=10, 50, 100?

### Soundness
3

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
This paper mainly focuses on the class imbalance in the demonstration selection of the conventional in-context learning paradigm. In this paper, the authors demonstrate that the class distribution degrades the performance of ICL regardless of the selection methods and the classical rebalancing methods, which focus solely on class weights. Meanwhile, the conventional rebalancing weights tend to yield poor performance. Thus, a method, Reweighting with Importance Factors (RIF) is proposed to solve this problem. According to the empirical results, the RIF method achieves good performance on common benchmarks.

### Strengths
- This paper explores the class imbalance problem in the demonstration selection of in-context learning, which is useful for both academic and industrial communities.
- This paper proposes a simple method, RIF, to solve the problem.
- The results reported in the paper look good.

### Weaknesses
- The motivation of this paper is a little bit trivial. I think it would be common sense that the imbalanced data can result in degradation of performance. To enhance the paper, it would be better to provide more insights regarding this problem.

- The formulations in this paper have many problems.

### Questions
- The equation in Remark 1 seems to assume that the generation of $\boldsymbol{{\rm y}}$ does not rely on the in-context demonstrations? How do you obtain such a conclusion?

- From both Remark 1 and Eq. (3), it seems that only the distribution of demonstrations is considered. However, the prior knowledge embedded in the LLM is also essential. In this paper, the prior knowledge is not taken into consideration.

- Could you please provide more explanation regarding your estimate in Section 3.2? Specifically, how do you measure the probability? How do you measure the average probability?

- In the equation in Line 238-239, the notions $\boldsymbol{{\rm x}}$ and $\boldsymbol{{\rm y}}$ do not appear in $M(\cdot, \cdot)$.

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
4

### Summary
This paper investigates how class imbalance in annotated datasets affects the performance of in-context learning (ICL) in large language models. The authors demonstrate empirically that imbalanced annotations degrade ICL performance and propose a reweighting approach, Reweighting with Importance Factors (RIF), that adjusts the demonstration selection process to mitigate the imbalance. RIF is evaluated across multiple benchmarks and models, showing modest but consistent performance improvements.

### Strengths
* Clarity and Presentation: The paper is well-written and clearly structured. The motivation, methodology, and results are presented in an organized and easy-to-follow manner.
* Comprehensive Evaluation: The experiments are extensive, covering multiple datasets, imbalance ratios, and both open-weight and API-based models.
* Simplicity of the Proposed Method: RIF is straightforward to implement and integrates easily with existing demonstration selection methods.

### Weaknesses
* Lack of Novelty: The core idea—reweighting samples based on estimated importance factors—is not fundamentally new and aligns closely with well-known class balancing and importance sampling techniques. The contribution seems more incremental than conceptual.
* Limited Theoretical Depth: The theoretical justification is fairly standard, and the paper mostly reiterates known results in imbalance learning, adapted to the ICL setting.
* Modest Empirical Gains: While improvements are consistent, they are relatively small (typically 3–5%), and it is not entirely clear whether such gains justify a new method.
* Framing: The paper positions itself as the “first” to study imbalance in ICL, but this framing feels overstated given prior works that already examine selection biases and label imbalance in few-shot or ICL setups.

### Questions
Novelty of RIF:

* The proposed Reweighting with Importance Factors (RIF) seems closely related to standard class reweighting and importance sampling methods widely used in imbalance learning. Could the authors clarify what is fundamentally novel in RIF beyond adapting these existing ideas to the ICL setting?

Effectiveness vs. Simple Baselines:

* Have the authors compared RIF to simpler baselines, such as class-balanced sampling or uniform per-class Top-K selection? Without such baselines, it’s unclear whether the observed gains come from the reweighting mechanism itself or simply from balancing the demonstration set.

Estimation of Importance Factors:

* The paper mentions estimating importance factors via Bayesian optimization on a balanced subset. Why was this approach chosen over more straightforward estimations (e.g., using normalized class frequencies)? How sensitive is the method to the subset size or its sampling variability?

Practical Applicability:

* RIF assumes access to a balanced subset for estimating class weights. In real-world long-tailed or low-resource scenarios, tail classes may have very few examples. How practical is this assumption, and can RIF still be applied effectively when such a balanced subset is infeasible to obtain?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates how class imbalance in annotation datasets affects in-context learning (ICL) performance. The authors show that imbalanced annotations significantly degrade ICL across various tasks, and that existing calibration methods fail to address this. They propose RIF (Reweighting with Importance Factors), which estimates importance weights using a balanced subset and reweights scoring functions during demonstration selection. Experiments across different datasets show incremental improvement with accuracy gains of up to 5.6% on highly imbalanced datasets.

### Strengths
S1: The paper is well-executed experimentally, with comprehensive evaluations across multiple models (OPT, LLaMA, ChatGPT, Gemini), datasets, and imbalance ratios. The experimental protocol is solid and well explained. The explanation and figures are straightforward to understand.

S2: The finding that a larger dataset doesn't automatically address the imbalance issue is useful.

### Weaknesses
W1: This is my main concern. The problem and solution seems expected and unsurprising. It isn't surprising that class imbalance harms ICL performance. The solution to use a balanced subset to calculate a weighting factor is also hardly novel or particularly interesting. It is unclear why not just used the balanced set for ICL?

W2: The novelty seem limited. Importance sampling and reweighting for class imbalance are well known techniques addressing imbalance.

W3: The improvements are incremental and in many cases barely above noise. Looking at Table 1, many improvements are within 2% on top of a ~50% baseline. It is hard to conclude how impactful these improvements will be.

W4: It is unclear how these LLMs are selected. They seem to span old models to recent models without any clear reason or selection criterion.

### Questions
Q: It is unclear if this ICL is in the learning regime or retrieval regime please see: Pan et al: https://arxiv.org/abs/2305.09731

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies how class-imbalanced annotation pools degrade in-context learning (ICL) because selection methods over-pick head classes, biasing priors in prompts, and proposes RIF, a simple reweighting scheme that estimates per-class importance factors from a small balanced subset via Bayesian optimization and then reweights the scoring function (e.g., cosine similarity in TopK) to counteract head dominance during demo selection.

### Strengths
- Clear empirical evidence that pool-level class imbalance hurts ICL across selection methods and that common calibration schemes (CC, DC, Var-IC) fail or even neutralize advanced selectors; figures and tables isolate the effect across models and imbalance ratios.​
- Simple, method-agnostic reweighting that plugs into existing selectors without model fine-tuning and only requires model outputs; computational overhead is modest and often lower than calibration baselines in reported settings.​
- Sensible theoretical framing via importance sampling linking expected risk to class-prior mismatch in selected demonstrations, motivating per-class importance factors and reweighted selection distributions.

### Weaknesses
- Dependence on building a balanced subset $\mathcal{D}_b$ from tail classes can be impractical when tails are extremely scarce, which the paper acknowledges but does not resolve with alternatives (e.g., generative augmentation, semi-supervised density estimation).​
- Importance-factor estimation uses Bayesian optimization on $\mathcal{D}_b$ with heuristic initialization; sensitivity to surrogate choices, search budget, noise in evaluation, and cross-dataset transferability of learned weights is under-explored beyond limited robustness plots.​
- Theoretical analysis assumes shared conditional $P_c(x|y)=P_t(x|y)$ and reduces imbalance to class priors; distribution shifts in x within a class (domain shift) or long-tail feature skew are not addressed, potentially limiting guarantees in real applications ​.
- Fairness and compute parity: baselines sometimes incur extra calibration passes while RIF reports lower hours; more controlled compute-matched comparisons and wall-time/memory profiles for large-K candidate selection would strengthen efficiency claims.​
- Generation tasks aggregation into coarse “categories” (e.g., NQ) simplifies imbalance as class priors; how RIF extends to open-ended outputs, span-level evaluation, or attribute-level long tails is only preliminarily explored with small gains (e.g., EM +1.7 absolute at 100:1).​
- Reweighting operates at class level; many real-world imbalances are multi-attribute or hierarchical (topic × sentiment × length); the method does not consider multi-label, group, or continuous-attribute imbalance beyond a brief note, risking overcorrection or under-coverage.​
- The approach assumes access to class labels during selection for weight application; for tasks without explicit classes or where labels are latent (retrieval-based reasoning), mapping to “classes” is ad hoc and may inject template bias (Appendix templates).​
- Statistical rigor could improve: many tables show absolute gains but limited hypothesis testing, confidence intervals, or multiple-seed variance in the main text; robustness to prompt formats and verbalizer choices, known to sway ICL, is mostly in appendices or not systematically isolated.​
- Large-K candidate requirement to include tails increases retrieval cost; guidance to set K vs imbalance ratio is qualitative, with no adaptive candidate control or approximate nearest neighbor strategies discussed for scaling.

### Questions
- How does RIF perform when tails are extremely small (e.g., ≤5 examples/class)? Can synthetic balancing, weak labels, or external encoders help estimate reliable importance factors without $\mathcal{D}_b$?​
- What is the sensitivity of the learned weights to prompt format, verbalizers, and decoding changes, and can a single set of weights transfer across related tasks or models without re-optimization?​
- Can RIF be extended beyond discrete classes to group, attribute, or continuous-conditioned imbalance (e.g., length, domain, language) by learning importance over clusters or learned partitions?​
- For generation, can importance be defined over answer types or latent intents rather than coarse categories, and does that yield larger gains on EM/ROUGE while avoiding template-induced artifacts?​
- Under domain shift where $P_c(x|y) \neq P_t(x|y)$ can a covariate shift correction be layered with class-prior reweighting, and how would the objective and estimation change practically ​?
- What is the runtime/memory profile as K and class count grow (e.g., K = 2k, |Y| = 100), and can approximate retrieval plus per-class quotas or two-stage filtering retain gains with lower cost

### Soundness
3

### Presentation
3

### Contribution
2
