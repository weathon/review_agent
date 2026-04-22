# BM-CL: Bias Mitigation through the lens of Continual Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Biases in machine learning pose significant challenges, particularly when models amplify disparities that affect disadvantaged groups. Traditional bias mitigation techniques often lead to a {\itshape leveling-down effect}, whereby improving outcomes of disadvantaged groups comes at the expense of reduced performance for advantaged groups. This study introduces Bias Mitigation through Continual Learning (BM-CL), a novel framework that leverages the principles of continual learning to address this trade-off. We postulate that mitigating bias is conceptually similar to domain-incremental continual learning, where the model must adjust to changing fairness conditions, improving outcomes for disadvantaged groups without forgetting the knowledge that benefits advantaged groups. Drawing inspiration from techniques such as Learning without Forgetting and Elastic Weight Consolidation, we reinterpret bias mitigation as a continual learning problem. This perspective allows models to incrementally balance fairness objectives, enhancing outcomes for disadvantaged groups while preserving performance for advantaged groups. Experiments on synthetic and real-world image and tabular datasets, characterized by diverse sources of bias, demonstrate that the proposed framework mitigates biases while minimizing the loss of original knowledge. Our approach bridges the fields of fairness and continual learning, offering a promising pathway for developing machine learning systems that are both equitable and effective.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper advocates a novel BM-CL framework that reinterprets bias mitigation as a continual learning (CL) problem. The authors draw parallels between the leveling-down effect in fairness interventions (where improving disadvantaged groups harms advantaged ones) and catastrophic forgetting in CL. The proposed BM-CL combines standard bias mitigation methods (GroupDRO, ReSample) with CL techniques (LwF and EWC). Experiments on Waterbirds, CelebA, and CheXpert datasets show improvements in worst-group accuracy with minimal degradation in advantaged groups.

### Strengths
+ The core idea of framing bias mitigation as continual learning is conceptually elegant. Drawing a formal analogy between the leveling-down effect and catastrophic forgetting provides a new view that could inspire cross-fertilization between fairness and lifelong learning research.

+ BM-CL integrates easily with existing bias-mitigation pipelines by adding a CL-style regularization term. Hence the approach is clear and reproducible, with minimal additional complexity for understanding or implementation.

+ Evaluations on multiple benchmark datasets (spurious correlation, demographic bias, medical imaging) are comprehensive and show consistent trends. The code availability and ablation studies increase reproducibility.

### Weaknesses
- While providing a new lens for understanding fairness-aware ML, the proposed method is largely a straightforward combination of existing techniques (LwF, EWC, and bias-mitigation baselines) with limited algorithmic innovation. The design omits potential drawbacks (e.g., stability-plasticity conflicts under multi-attribute settings, or extension to non-image modalities).

- No new theoretical framework or analytical insight beyond the analogy to forgetting is developed. There also lacks justification for why continual-learning regularizers specifically address the fairness-accuracy trade-off beyond empirical correlation.

- While the results show improvements, the absolute differences over strong baselines (GroupDRO, ReSample) are often small (~1-2% balanced accuracy). Quantitative gains are marginal. 

- It is unclear whether these improvements are statistically significant or robust across seeds and datasets.

### Questions
1. How does BM-CL perform when the number of demographic groups increases or when group definitions overlap (intersectional fairness)?

2. Can compare BM-CL against more recent fairness methods (e.g., invariant risk minimization, domain generalization-based fairness) to contextualize competitiveness?

3. Could the authors provide a theoretical or empirical justification for why the continual-learning regularization term directly mitigates "leveling-down" beyond preventing weight drift?

4. How sensitive are results to the choice of hyperparameters such as λ (regularization strength) and pretraining ratio ρ across datasets? Do these require dataset-specific tuning?

5. Have examined the effect of CL regularization on model calibration or other fairness metrics (equalized odds, demographic parity, delta EO or DP)?

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
3

### Summary
The paper proposes BM-CL, framing bias mitigation's "leveling-down" effect as a Continual Learning "forgetting" problem. It uses a two-stage approach: ERM training followed by fine-tuning with a bias mitigation loss and a CL regularizer (LwF/EWC) to preserve performance on initially advantaged groups.

### Strengths
The paper addresses a critical and practical problem in fairness. The "leveling-down" effect is a major barrier to deploying fair models in high-stakes settings where harming any group is unacceptable.

### Weaknesses
1.Limited Novelty / Incremental Contribution: The core idea of linking "leveling-down" to "forgetting" is an interesting perspective, but the proposed solution (BM-CL) primarily combines existing, off-the-shelf techniques: standard bias mitigation losses (GroupDRO, ReSample) with standard CL regularizers (LwF, EWC) in a sequential manner. There is little fundamental algorithmic innovation presented. The contribution feels incremental rather than introducing a truly new mechanism for bias mitigation.

2.Marginal Empirical Improvement: Even when compared against the selected baselines (which, as noted previously, might not be the most appropriate ones for evaluating FWH), the performance gains offered by BM-CL are not substantial. While it often reduces the leveling-down effect (LDE), this sometimes comes at the cost of lower worst-group improvement compared to simpler baselines (e.g., compare GroupDRO-LwF vs GroupDRO on CheXpert). The improvements in balancing the trade-off are marginal and may not be significant enough to justify the method's adoption.

3. Unjustified Complexity vs. Benefit: The BM-CL framework introduces significant complexity compared to standard ERM or single-stage bias mitigation methods. It requires a two-stage pipeline, identification of best/worst groups, and tuning of additional CL-specific hyperparameters (p, λ). Given the limited novelty and marginal performance gains, the cost-benefit trade-off appears unfavorable. The paper does not sufficiently demonstrate that this added complexity yields a practically meaningful advantage.

### Questions
1.Can the authors articulate the core technical novelty of BM-CL beyond the combination of existing CL regularization techniques with existing bias mitigation losses? What is fundamentally new in the mechanism proposed?

2.Given that the empirical improvements over baselines appear marginal (Table 1), how do the authors justify the significant added complexity (two stages, extra hyperparameters) of the BM-CL framework from a practical cost-benefit perspective?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces BM-CL, a novel framework that addresses the "leveling-down effect" commonly found in bias mitigation techniques. The core contribution is the reinterpretation of bias mitigation as a task-incremental continual learning problem. Experiments demonstrate that, compared to baseline methods, this approach effectively improves worst-group accuracy while significantly minimizing the drop in best-group performance , promoting positive-sum fairness without designing new complex architectures.

### Strengths
1. It cleverly utilizes existing CL techniques in a novel way without requiring additional datasets or complex model architectures, which helps promote positive-sum fairness.
2. The paper reinterprets the bias mitigation problem as a task-incremental continual learning problem. This is a highly novel perspective that offers a new approach to addressing the issue.
3. While traditional bias mitigation methods often sacrifice the performance of advantaged groups, this framework leverages CL principles to improve outcomes for disadvantaged groups while effectively preserving knowledge for advantaged groups, thereby avoiding performance degradation.

### Weaknesses
1. Although the conceptual framework is innovative, the novelty of the contribution is limited. It primarily involves adapting existing methods to the problem posed by the paper, rather than proposing an entirely new algorithm to solve it.
2. The method assumes that explicit group labels are available during training, which may limit its broader applicability.
3. The paper does not provide theoretical guarantees for fairness convergence. The proposed loss function is effectively a combination of two conflicting objectives. While the paper empirically demonstrates that a good balance point can be found, it does not prove that such a balance point must exist or can always be achieved. Furthermore, regarding the leveling-down boundary, although the experimental results show low leveling-down values, there is a lack of theoretical analysis to indicate how severe the worst-case leveling-down might be on more complex datasets.

### Questions
1. The ablation study indicates that the regularization strength $\lambda$ is a critical hyperparameter. Beyond conducting a grid search on the validation set, is there a more principled or intuitive method for setting this parameter?
2. In Table 1, you compare BM-CL against ERM, GroupDRO, JTT, and other methods. Could you include comparisons against more recent, SOTA bias mitigation algorithms as baselines? Including these comparisons would allow for a more comprehensive evaluation of the BM-CL framework's effectiveness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Bias Mitigation through Continual Learning (BM-CL), a novel framework that reinterprets the challenge of algorithmic fairness as a domain-incremental continual learning problem. The key contribution is a methodology that leverages techniques like Learning without Forgetting and Elastic Weight Consolidation to incrementally adapt models to new fairness objectives. This approach specifically mitigates the common leveling-down effect by improving outcomes for disadvantaged groups without degrading performance for advantaged groups, thereby balancing fairness with the preservation of the model’s original knowledge, as validated on diverse synthetic and real-world datasets.

### Strengths
S1: The primary strength of this work is its novel conceptual contribution of bridging the fields of fairness and continual learning.

S2: By reinterpreting bias mitigation as a domain-incremental or task-incremental continual learning problem, this work leverages techniques like Learning without Forgetting and Elastic Weight Consolidation to incrementally adapt models to new fairness objectives.

S3: The effectiveness of BM-CL is validated across multiple, diverse real-world image datasets such as Waterbirds, CelebA, and CheXpert.

### Weaknesses
W1: Methods such as EWC and LwF can struggle with scalability to a very large number of fairness tasks or domains, and their effectiveness can be sensitive to hyperparameter tuning.

W2: The experiments are confined to image classification tasks. It remains unclear how well the BM-CL framework would generalize to other data modalities, such as tabular data or natural language, where biases are equally prevalent and challenging.

W3: The work focuses on group fairness metrics and accuracy but may not have extensively explored other important fairness notions like individual fairness or the long-term societal impact of deploying such models.

### Questions
- How exactly is the bias mitigation task structured as a continual learning problem? Is the model first trained on the original biased dataset and then fine-tuned on a de-biased or re-weighted version? Or is it exposed to different demographic groups sequentially?
- How sensitive are the results to the order in which the fairness conditions or groups are introduced? Continual learning performance is often highly dependent on task sequence.
- For the Elastic Weight Consolidation (EWC) method, how were the Fisher Information Matrix, the pretraining ratio, and the importance weight (lambda) for each parameter determined? Was lambda tuned specifically for fairness, and if so, what was the objective?
- For Learning without Forgetting (LwF), how were the soft targets from the previous model obtained and used? Was the original training data required for replay, or was it done solely with new data? If the latter, how does this affect the stability-plasticity balance?
- Were the hyperparameters (especially the critical ones for LwF and EWC like distillation temperature, regularization strength) optimized separately for each method, or was a consistent framework used? A difference in tuning effort could explain performance gaps.
- Beyond standard bias mitigation techniques, were other continual learning methods (e.g., experience replay with a small memory buffer) tested for comparison to better isolate the contribution of LwF/EWC?

### Soundness
3

### Presentation
4

### Contribution
3
