# INO-SGD: Addressing Utility Imbalance under Individualized Differential Privacy

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
Differential privacy (DP) is widely employed in machine learning to protect confidential or sensitive training data from being revealed. As data owners gain greater control over their data due to personal data ownership, they are more likely to set their own privacy requirements, necessitating individualized DP (IDP) to fulfil such requests. In particular, owners of data from more sensitive subsets, such as positive cases of stigmatized diseases, likely set stronger privacy requirements, as leakage of such data could incur more serious societal impact. However, existing IDP algorithms induce a critical utility imbalance problem: Data from owners with stronger privacy requirements may be severely underrepresented in the trained model, resulting in poorer performance on similar data from subsequent users during deployment. In this paper, we analyze this problem and propose the INO-SGD algorithm, which strategically down-weights data within each batch to improve performance on the more private data across all iterations. Notably, our algorithm is specially designed to satisfy IDP, while existing techniques addressing utility imbalance neither satisfy IDP nor can be easily adapted to do so. Lastly, we demonstrate the empirical feasibility of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces the novel problem of **utility imbalance** caused by individualized differential privacy (IDP). In IDP, each data owner specifies their own privacy level, which offers flexibility but leads to uneven model performance across owners with different privacy budgets. The authors propose **Individualized Noisy Ordered SGD (INO-SGD)**, an algorithm that **adjusts the weighting of data within each batch** to improve outcomes for owners with stricter privacy requirements while maintaining the same privacy guarantee as standard IDP-SGD. Through extensive experiments, the authors demonstrate that INO-SGD enhances the utility of more private groups, effectively mitigating utility imbalance, while maintaining or even improving overall model performance.

### Strengths
**1. Novel and Important Problem Identification:**
The paper introduces the new problem of utility imbalance under individualized differential privacy. It shows that while IDP gives users control over their privacy, existing algorithms unintentionally penalize those with stronger privacy. Its focus on how varying privacy budgets create uneven model performance highlights an important direction for future research in differential privacy.

**2. Strong Theoretical Justification and Clear Presentation:**
The paper provides a well-structured and convincing theoretical motivation for the identified problem. By connecting the imbalance to sampling rates and clipping thresholds through concepts like Minority Initial Drop and Biased Optimization Objective, it offers clear mathematical insight into the cause of the issue. Theorem 3.1 and its supporting proofs further reinforce the soundness of the argument.

**3. Comprehensive Empirical Validation:**
The experiments are extensive and well-executed, covering a diverse range of datasets, models, and privacy settings. The results consistently show that INO-SGD improves the performance of more private groups, effectively mitigating utility imbalance, while maintaining or even enhancing overall model utility.

**4. Novel and Well-Designed Mechanism:**
The idea behind INO-SGD is to down-weight less important data within each batch to improve performance on more private data throughout training. This approach is practical and broadly applicable, with potential use in learning problems that involve fairness or group balance.

### Weaknesses
**1.** In some parts, the paper’s wording seems to overstate the performance gains of INO-SGD. For example, lines 444–446 claim that *Across training stages, INO-SGD significantly increases the recall on validation data groups associated with more private owners (left) at the expense of a **small decrease** in the recall on less private ones (right).* However, from Figure 6, it appears that for larger $\varepsilon$ values (less private owners), IDP-SGD clearly outperforms INO-SGD, and the performance drop for these groups is roughly equal to the gain for more private ones. Additionally, Figures 5 and 7 use validation loss as the utility metric for the MNIST dataset, while recall/accuracy are used for CIFAR-10 and CIFAR-100-FV. Is there a specific reason for this inconsistency? If not, I would suggest using a consistent evaluation metric across datasets for clarity.

**2.** The paper states that INO-SGD guarantees the same privacy level as IDP-SGD (Theorems 2.2 and 3.3). While the paper mentions that INO-SGD consumes privacy budgets at the same rate as IDP-SGD, and therefore maintains the same number of training iterations, it remains unclear why the two methods have exactly the same privacy guarantee at **each iteration**. For instance, INO-SGD involves computing the Batch Importance Function (BIF) from $f_{\text{tail}}$ and $B_t$, as well as sorting losses—operations specific to INO-SGD. Could the authors provide more intuition or justification for why these additional computations do not affect the per-iteration privacy guarantee?

### Questions
See the Weaknesses section, please.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies utility imbalance under individualized differential privacy (IDP). It shows that even on balanced datasets, IDP-SGD (SAMPLE/SCALE) can disadvantage owners (or groups) with stricter privacy (smaller ε) by creating MID (minority initial drop) during training and BOO (biased optimization objective) at convergence. To mitigate this, the authors propose INO-SGD, which (i) sorts per-batch examples by loss, (ii) uses a Tail Importance Function (TIF) to define a Batch Importance Function (BIF), and (iii) rescales clipped gradients by per-rank importance scores before adding Gaussian noise—while keeping each datum’s modular sensitivity bounded by its individualized clipping threshold, thereby preserving IDP.

### Strengths
The problem framing is clear. IDP can itself create imbalanced learning dynamics (MID/BOO) even when class counts are balanced. This problem actually step away practical DP deployment.

The intuition makes sense. To my understanding, it aims to increase the sensitive users' utility by trading off some non-sensitive users' utility. Meanwhile, the loss-ordered importance scheme is designed to not increase per-datum sensitivity (keeps ΔA^d ≤ Co(d)) while re-weighting gradient contributions.

### Weaknesses
The narrative argues standard class-imbalance fixes don’t satisfy IDP or get neutralized by clipping (oversampling/weighting/rescaling), but empirical comparisons with IDP-compliant alternatives are missing.

Claims rest on formal sensitivity bounds and RDP accounting; there’s no auditing checking if per-owner attack success (e.g., per-group LiRA, TPR/FPR rates) still meet TPR/FPR<=exp(ε). Though formal analysis makes sense, using auditing to check if there is any bug in implementation is also important.

The main figures emphasize per-group recall and validation loss/accuracy. To assess “imbalance”, measures like worst-group accuracy should be reported directly.

### Questions
Please check the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper identifies a utility imbalance that arises when training with individualized DP (IDP): data owned by users with stricter privacy budgets (e.g., sensitive subpopulations) receives less effective learning, degrading their group utility even when dataset sizes are balanced. The authors analyze why IDP-SGD induces MID/BOO-style dynamics and propose INO-SGD, which (i) sorts per-batch examples by loss, (ii) constructs a tail importance function (TIF) and corresponding batch importance function (BIF) to assign importance weights, and (iii) scales clipped per-example gradients by these data-dependent weights without exceeding each owner’s clipping threshold, thereby preserving IDP guarantees. They prove IRDP privacy with the same accounting as IDP-SGD and show that INO-SGD can obtain gains on MNIST, CIFAR-10/100, improving the more private groups’ recall while preserving or improving overall accuracy.

### Strengths
+ The paper clearly identifies and analyzes a practical failure mode of IDP training, showing how stricter privacy budgets systematically depress group utility even without data imbalance.

+ The proposed method preserves individualized DP guarantees without additional privacy cost, because it reweights only after clipping and uses the same accounting as IDP-SGD.

+ The experimental results consistently improve performance for highly private users while keeping or improving global accuracy, demonstrating that the approach meaningfully reduces utility imbalance in practice.

### Weaknesses
- The empirical evaluation lacks breadth against strong IDP-compliant alternatives, as it focuses mainly on vision benchmarks and does not compare to a wider set of within-IDP reweighting or sampling schemes.

- The method provides limited guidance for selecting the importance-function hyperparameters, leaving open how to tune them or manage trade-offs across privacy groups in real deployments.

### Questions
Refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
