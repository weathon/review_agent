# The Tail Tells All: Estimating Model-Level Membership Inference Vulnerability Without Reference Models

- Decision: Reject
- Scores: 4, 2, 8, 4

## Abstract
Membership inference attacks (MIAs) have emerged as the standard tool for evaluating the privacy risks of AI models. However, state-of-the-art attacks require training numerous, often computationally expensive, reference models, limiting their practicality. We present a novel approach for estimating model-level vulnerability, TPR at low FPR, to membership inference attacks without requiring reference models. Empirical analysis shows loss distributions to asymmetric and heavy-tailed and suggest that most points at risk from MIAs to have moved from the tail (high-loss region) to the head (low-loss region) of the distribution. We leverage this insight to propose a method to estimate model-level vulnerability from the training and testing distribution alone: using the absence of outliers from the high-loss region as a predictor of the risk. We evaluate our method, the TNR of a simple loss attack, across a wide range of architectures and datasets and show it to accurately estimate model-level vulnerability to the SOTA MIA attack (LiRA). We also show our method to outperform both low-cost (few reference models) attacks such as RMIA and other measures of distribution difference. We finally evaluate the use of non-linear function to evaluate risk and show the approach to be promising to evaluate the risk in large-language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a new metric to approximate the vulnerability of a model to SOTA MIAs without the need to train additional reference models. As empirical analysis shows, loss distribution on training data points tends to be long-tailed. Their proposed metric is based on the idea that during training, atypical training points tend to move from high-loss (tail) to low-loss (head) of the loss distribution as a consequence of memorisation. Identifying such samples can then provide an estimate of MIA vulnerability. The proposed metric estimates the true negative rate (TNR) of a model at a fixed FPR for an attack to account for such tail-to-head records.

### Strengths
- The proposed approach does not rely on reference models and is computationally feasible.
- The new metric is tested against a diverse set of models (including LLMs) and image classification datasets.
- The estimated MIA vulnerability using the new metric is compared against MIA vulnerability estimates using other SOTA metrics in the literature, such as LT-IQR.
- Authors(s) use the proposed metric to estimate MIA vulnerability for SOTA MIAs such as LiRA and RMIA (with 64 shadow models), thereby providing empirical evidence supporting the validity of their method.

### Weaknesses
- The metric appears to be sensitive to the choice of pre-set threshold used to compute LOSS TNR at fixed FPR as demonstrated by Eq (4). It is possible that the choice of threshold changes depending on the choice of datasets/ models. Furthermore, it is unclear whether the author(s) vary the threshold for different experimental settings. Nor is it clear if their proposed metric is robust to the choice of threshold.
- In Line 344, the authors conjecture "An exponential fit would imply that as LOSS TNR increases, member identification becomes easier as the model memorises more difficult samples...". But they provide no empirical evidence to support this. As a reviewer, I would appreciate a plot (if feasible) equivalent of Table 3. Furthermore, it is not specified in the paper if the results shown in Table 3 pertain to a specific training setting or it is generalizable to other datasets/models.
- The figures/ tables in the paper lack the necessary information to clearly convey the author(s)' intended message. They are supposed to be self-contained with as little need to refer to text in the paper as possible:
    - Table 1: Does "Ours" in the table refer to the Loss TNR at fixed FPR? Can you make it clear in the caption or the table?
    - Table 2: Same as Table 1. It lacks the context to interpret the contents of the table.
    - Table 3: Same as Table 1. It lacks the context to interpret the contents of the table.
- In all figures, you use TNR@FNR instead of TNR@FPR.
- Line 335: I suppose you mean “that a linear model may not be the most appropriate given the task at hand.”
- Figure 5's caption says, "The LIRA TPR@FPR LOSS as a linear function of the LOSS AUC evaluated on LLMs." Assuming this refers to the subfigure on the right, the metric in the subfigure on the x-axis is LOSS TNR@FNR and not the LOSS AUC as the caption suggests. It will cause confusion about the effect/ claim the author(s) intend to convey using the figure.

### Questions
**Questions**: If the author(s) can address the weaknesses detailed above, I would be amenable to revising my initial assessment.

**Suggestions**: 
The author(s) provide ample evidence (in Section 4, Hypothesis) to support their proposed metric, but there is a lack of empirical evidence that would be necessary to solidify their argument, as detailed among the weaknesses. A bigger issue with the paper is its poor presentation. The author(s) need to improve their presentation, which, in its current version, makes the paper a rather difficult read. I have detailed some of the presentation issues in the weaknesses.

### Soundness
3

### Presentation
1

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
This paper proposes a method for estimating model-level vulnerability to membership inference attacks without training reference models. The key idea is that the absence of high-loss (tail) samples in the training loss distribution correlates with the model’s susceptibility to MIAs. The authors empirically show that the TNR of a simple loss-based attack can predict the TPR of LiRA at low false positive rates. The approach aims to offer a low-cost privacy risk estimation method suitable for large or resource-limited settings.

### Strengths
- The empirical results cover multiple datasets and architectures, demonstrating correlation between the proposed metric (LOSS TNR) and LiRA’s TPR at FPR.

- The paper is clearly written and easy to follow.

### Weaknesses
- The paper focuses on model-level vulnerability estimation, but it is unclear what the real-world application scenario of such a metric is. In privacy evaluation, MIAs are primarily defined as worst-case, sample-level privacy breaches (determining whether a particular record was in training), as evidenced by Carnili et al.. A model-level average metric offers little actionable guidance: practitioners either need to test specific data records (for auditing) or evaluate defense mechanisms under realistic attack settings. The proposed metric seems to produce a correlation measure with LiRA or RMIA, but it is unclear how this would be used in practice. The paper does not provide any concrete use cases or deployment scenarios.

- The technical novelty of the work is also limited. The core contribution amounts to computing the True Negative Rate (TNR) of a standard loss-based attack. This idea lacks theoretical grounding or statistical justification. The absence of analytical insights weakens the contribution, making it less suitable for a top-tier venue such as ICLR.

- The evaluation is restricted to LiRA and RMIA, both of which rely on output-distribution differences. However, many MIAs operate under different assumptions: Reference-calibrated or label-only attacks (e.g., He et al., 2024; Ye et al., 2022) rely on label confidence or query perturbations, not continuous loss values. It remains unclear whether the proposed estimator generalizes to those attack families.

- Finally, although the authors compare their approach with metrics like LT-IQR AUC and train-test gap, these baselines are not designed for model-level vulnerability estimation. As a result, the comparison does not convincingly demonstrate the proposed method’s superiority or distinct advantages.

### Questions
- What are the real-world application scenarios where a model-level vulnerability metric is practically useful for privacy evaluation?

- What theoretical or statistical justification supports using the TNR of a simple loss attack as a valid estimator of MIA vulnerability?

- Does the proposed method generalize beyond LiRA and RMIA to other attack types, such as label-only or reference-calibrated MIAs?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a new method to estimate a model’s vulnerability to membership inference attacks without training any reference models. It is shown that samples most vulnerable to LiRA are the ones that were moved from the high-loss tail of the distribution to the low-loss region during training. By analyzing the loss distributions, the MIA vulnerability can be predicted. The approach outperforms SOTA methods such as RMIA in predicting LiRA’s overall success rate, while requiring no reference models.

### Strengths
- The proposed method is way more efficient than previous approaches.
- The method is a very good and efficient indicator to approximate vulnerability to MIAs after training a model.
- The paper was very easy to read and to follow.
- With an adaptation of the LOSS TNR to the LOSS AUC, the method can even be applied to LLMs.

### Weaknesses
- While LiRA and RMIA are computationally more demanding, these attacks can be used to predict membership for individual samples. The proposed method cannot predict membership for individual samples, but only estimates the vulnerability to membership inference attacks on a model level.

Misc:
- In line 82, the sentence seems to be incomplete and has "achieve" two times within the sentence.
- In line 260, "Appendix" and the closing brackets are missing.

### Questions
Q1: Is it somehow possible to extend this approach to allow for sample-level membership predictions?  
Q2: Why use the LOSS AUC only for LLMs? Did you also try it for other "traditional" models?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper propses a low cost method to identify points that will be vulnerable to privacy leakage. The method works by tracking the loss of the points in the training set and comparing to the test set. The author further conduct experiments on LLMs.

### Strengths
The paper makes solid contributions. It substantially reduces the computational cost of estimating membership inference vulnerability by removing the need for reference models. It also establishes a strong empirical relationship between true MIA performance and its proposed proxy (the LOSS TNR metric), demonstrating that simple loss-based statistics can reliably estimate privacy risk. Finally, it conducts extensive experiments across diverse architectures and datasets, reinforcing the robustness and generalizability of its findings.

### Weaknesses
My biggest concern is with the limited scale of the image experiments. The tails tend to disappear when the generalization gap is low. For example, finetuning a large transformer models (like ViT) on CIFAR datasets. I think this represents an important case for the authors to consider.

### Questions
above

### Soundness
3

### Presentation
3

### Contribution
3
