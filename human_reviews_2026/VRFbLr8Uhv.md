# Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping

- Avg Score: 4.40
- Decision: Reject
- Scores: 8, 4, 4, 4, 2

## Abstract
Differential privacy (DP) has become an essential framework for privacy-preserving machine learning. Existing DP learning methods, however, often have disparate impacts on model predictions, e.g., for minority groups. Gradient clipping, which is often used in DP learning, can suppress larger gradients from challenging samples. We show that this problem is amplified by adaptive clipping, which will often shrink the clipping bound to tiny values to match a well-fitting majority, while significantly reducing the accuracy for others. We propose bounded adaptive clipping, which introduces a tunable lower bound to prevent excessive gradient suppression. Our method improves worst-class accuracy by over 10 percentage points on Skewed and Fashion MNIST compared to unbounded adaptive clipping, 7 points compared to Automatic clipping, and 5 points compared to constant clipping.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work proposes lower-bounded adaptive clipping for differential privacy learning to address disparate impact of DP learning on minority and confusable groups. The method leads to improvement in worst-class accuracy for skewed and Fashion MNIST.

### Strengths
- The work identifies common issue of disappearing clipping bounds of current works and tackles important problem of ensuring ML fairness for minority groups and shows improvement over SOTA.
- Applied to 4 datasets (Skewed, Fashion MNIST, Adult, Dutch) and 3 architectures ResNet-18, CNN, Logistic Regression
- Testing under Realistic Constraints (DP-HPO)

### Weaknesses
- the method adds additional hyperparameters to tune increasing the complexity of the training this is a weakness common to the family of methods
- the paper could be strengthened by providing more explicit guidance or a low-cost heuristic for setting C_LB

### Questions
Hyperparameters (Target quantile γ=0.5, Multiplier τ=2.5, and learning rate ηC =0.2) across experiments are fixed, but tuning results in Table A1 show large STD (e.g., τ on Fashion MNIST is 2.7438±3.1248). Could the authors clarify the definition of "stability" used that justified fixing these parameters, despite the high variance observed in preliminary tuning results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses a critical issue at the intersection of Differential Privacy (DP) and fairness: the disparate impact of DP-trained models on different demographic groups, particularly minority or challenging subgroups. The core mechanism investigated is adaptive gradient clipping, which is standard practice in training with DP-SGD (Differentially Private Stochastic Gradient Descent).

### Strengths
The technical quality appears sound. The proposed BAC method is a direct, mathematically clear modification of existing techniques, making it easy to integrate. The experimental design is robust, contrasting BAC not only with standard DP-SGD but also with existing adaptive clipping schemes (like AUTO), providing a necessary control group to validate the contribution. The reported results clearly show performance gains on fairness metrics (e.g., disparate accuracy), suggesting the method effectively achieves its stated goal.

### Weaknesses
- The primary weakness lies in the selection and motivation of the lower bound, $C_{\text{min}}$. While the method's effectiveness hinges on this parameter, the paper does not provide sufficient theoretical guidance for its choice. Currently, $C_{\text{min}}$ appears to be a manually tuned hyperparameter. This reduces the actionability of the insight. If $C_{\text{min}}$ is set too high, it negates the benefits of adaptive clipping; if set too low, it fails to help the minority group. The paper needs a more rigorous study or a heuristic/theoretical justification for how to choose $C_{\text{min}}$ relative to, for instance, the empirical gradient norm distribution of the minority group.
- The paper positions its work as mitigating disparate impact. However, the experiments mainly compare BAC to DP-SGD variations (which are privacy-focused) rather than methods explicitly designed for fairness under DP, such as Group DP-SGD (which uses group-specific clipping thresholds or noise scales) or DP versions of re-weighting or adversarial debiasing. A weakness is the absence of a direct comparison showing how BAC's implicit fairness improvement compares to the explicit fairness control achieved by these alternative methods. Without this, the reader cannot fully assess BAC's place in the fairness-under-DP literature.
- The assessment of disparate impact appears to focus predominantly on Disparate Accuracy (difference in accuracy between groups). In many real-world applications (like loan approval or recidivism prediction), metrics like Equal Opportunity Difference (difference in False Negative Rates, $FNR$) or Predictive Parity (difference in Positive Predictive Values, $PPV$) are often more critical. The experiments are insufficient without evaluating the impact of BAC on these other crucial fairness metrics, which could potentially reveal trade-offs not visible through accuracy alone.

### Questions
- Can the authors elaborate on whether $C_{\text{min}}$ can be justified or estimated without extensive hyperparameter search? For instance, could $C_{\text{min}}$ be set to a small percentile (e.g., the $5^{\text{th}}$ percentile) of the historical L2 gradient norms observed on the entire training set, or perhaps only on the minority/underperforming subgroup?
- Does the enforcement of $C_{\text{min}}$ affect the required noise level for a fixed privacy budget $\epsilon$, compared to a standard (unbounded) adaptive clipping method? Intuitively, a bounded clip norm could stabilize the bound variance, but a formal discussion on the impact of BAC on the final noise scale and the $\epsilon$ calculation is needed.

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
3

### Summary
The paper studies DP-SGD with adaptive clipping (similar to Andrew et al., 2021), which privately tracks a quantile of per-sample gradient norms and updates a global clipping bound so that roughly a target fraction of gradients are clipped. It identifies a failure mode for this method: as training progresses and most gradients shrink, the estimated proportion above the bound drops, the bound keeps shrinking, and can collapse toward zero. This disproportionately hurting minority or hard groups whose gradients remain larger.

To prevent this, the authors add a tunable lower bound $C_{LB}$ on the clipping bound (bounded adaptive clipping): when the adapted bound would fall below $C_{LB}$, they clip at $C_{LB}$ instead. Experiments on image (Fashion-MNIST, Skewed-MNIST) and tabular (Adult, Dutch) datasets show improved worst-class accuracy and competitive macro accuracy versus constant clipping, unbounded adaptive clipping, and AUTO (Bu et al., 2023). Because the proposed method introduces an extra hyperparameter, they also evaluate with DP-HPO and report similar or better performance under the accounted privacy budget.

### Strengths
The paper identifies a failure mode of earlier adaptive clipping methods and proposes a simple fix, with experiments demonstrating that it alleviates the issue. It's clearly written.

### Weaknesses
- Theorem 3.2 does not provide a precise privacy guarantee. The privacy–accuracy trade-off would be much clearer if the authors specified the resulting $\epsilon$ as an explicit function of $T$, the subsampling rate and the noise multipliers $\sigma_{grad}, \sigma_{count}$. In its current form, the guarantee is hard to interpret.

- While the mean-estimation example is interesting, it seems specific. Is the failure primarily driven by the setup in which the minority group is strictly smaller than the majority? How general is the phenomenon beyond that specific data structure?

- For image data, the “group” is defined by the class label, which isn’t a protected attribute, so the fairness interpretation is unclear.

### Questions
- Theorem 3.2 seems to rely on Lemma 3.1, which assumes both Gaussian mechanisms have sensitivity 1. While counting has sensitivity 1, the private gradient mean (after averaging) does not. I assume the privacy amplification by subsampling can also complicates the results. How do you handle sensitivity for this? 

- How does proposed method perform when the size of the minority group is similar to that of the majority group? 

- How does the method perform in terms of other fairness metrics such as per-group FPR/TPR, gap between group accuracies, etc.?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper identifies a critical issue in differentially private (DP) learning: adaptive clipping methods can lead to vanishing clipping bounds, which disproportionately harm minority or challenging classes. The authors propose a simple yet effective solution—introducing a lower bound on the clipping threshold—and demonstrate its efficacy across multiple datasets and models. The work is well-motivated, methodologically sound, and thoroughly evaluated. It addresses an important problem at the intersection of privacy and fairness, with practical implications for real-world DP training. Experiments across MNIST, Adult and Dutch, show improved worst-class and subgroup accuracy, with competitive macro accuracy, under both optimal hyperparameters and DP-HPO.

### Strengths
1.	This manuscript first identifies a well-identified problem, via a failure mode of existing adaptive clipping methods, where clipping bounds collapse during training, leading to unfair outcomes. The toy example in Figure 1 is particularly effective in illustrating this issue.
2.	Authors propose a simple and effective Solution, i.e. DP-HPO. Its bounded adaptive clipping is easy to implement, requires minimal modification to existing DP-SGD pipelines, and comes with a clear privacy guarantee.
3.	The privacy analysis is rigorous, leveraging Gaussian DP composition to account for both gradient and clipping-bound updates.
4.	The paper provides extensive details on hyperparameters, datasets, and experimental setups.

### Weaknesses
1.	The DP-HPO introduce a new hyperparameter i.e. the lower-bound of adaptive clipping bound C_LB. The paper shows robustness, but provides limited guidance on principled selection., this could be a practical barrier.
2.	The paper has a limited theoretical analysis about fairness. While motivated by fairness, the paper does not provide a theoretical analysis of how bounded clipping improves fairness guarantees (e.g., in terms of fairness definitions like equalized odds or demographic parity).
3.	The paper compares to AUTO and constant/unbounded clipping. It should provide comparisons with other fairness-oriented DP methods (e.g., DP-SGD-Fair by Xu et al., 2021, FairDP by Liu et al. 2022).

### Questions
1.	Can the authors provide a simple theoretical intuition or bound on how the lower bound mitigates disparate impact?
2.	Could the authors provide a sensitivity analysis or a heuristic for setting C_LB?
3.	Have the authors considered evaluating other fairness metrics (e.g., demographic parity, equal opportunity) beyond accuracy parity?
4.	DP-HPO is proposed based on normalized DP-SGD (De et al., 2022). How does the proposed adaptive clipping bound C_LB working with SGD, affect the fairness? The related work shows DP-SGD has the fairness problem. But, Lemma 3.1 and Theorem3.2 are both about privacy.
5.	I admit that The proposed DP-HPO is a simple and effective method, but it seem a little incremental novelty, via introduce the C_LB.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper diagnoses a failure mode in DP training. It claims that existing adaptive clipping methods cause disparate impact by shrinking the clipping bound to "tiny values" to accommodate the majority group. This, in turn, suppresses the larger gradients from minority or "challenging" samples, harming their performance.
The authors propose "bounded adaptive clipping" as a solution. This method is a minor modification that introduces a tunable hyperparameter, which acts as a floor, preventing the clipping bound from collapsing to zero. The paper shows this simple fix improves worst-class accuracy on skewed image datasets.

### Strengths
1. This paper focuses on an important part of DP training.
2. The paper clearly identifies and illustrates a failure mode for unbounded adaptive clipping, where the bound collapses and ignores minorities.

### Weaknesses
1. The novel part of this paper is the max() function. This is a minor heuristic, not a new framework.
2. Baseline is not well selected. Why pick the auto clipping? My understanding is that auto clipping is good for hyperparameter tuning since it does not require for clip bound. Why do you want to compare your proposed method with them? I think De et al.(https://arxiv.org/pdf/2204.13650) may be a good choice. They achieve good performance on many datasets. If your method plus theirs can achieve new SoTA results on CIFAR-10 or CIFAR-100 dataset will make your method more stronger.
3. The datasets are toy datasets. I know for a DP paper, it may not be easy for training with ImageNet but at least use CIFAR-10/100.
4. The improvements are not consistent.  Sometimes the proposed method is better than baseline for eps=1 and 4, sometimes it is better for eps=2. Could authors provide more explanation for this? Some improvements are limited.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
