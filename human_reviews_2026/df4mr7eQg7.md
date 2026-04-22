# Class-Wise Disparity in Adversarial Training: Implicit Bias Perspective

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 8, 2

## Abstract
Disparities in class-wise robust accuracies frequently arise in adversarial training, where certain classes suffer significantly lower robustness than others, even when trained on balanced data. This phenomenon has been identified and termed robust fairness in prior work, highlighting the challenge of ensuring equitable robustness across classes.
In this work, we investigate the root causes of such disparities and identify a strong correlation between the norms of head parameters (i.e., the last layer’s weights) and class-wise robust accuracies. Our theoretical and empirical analyses show that adversarial training tends to amplify these disparities by disproportionately affecting head norms, which in turn influence class-wise performance.
To address this, we propose a simple yet effective solution that mitigates these imbalances by directly fine-tuning the head parameters while keeping the feature extractor fixed. Unlike existing methods that rely on class reweighting or remargining strategies, our approach requires no validation set and introduces minimal computational overhead.
Experiments across various datasets and architectures demonstrate that our method significantly reduces disparities in class-wise robust accuracies without degrading overall performance, providing a practical and principled step toward improving robust fairness in adversarial learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper systematically investigates the problem of class-wise robustness disparity that is prevalent in adversarial training, and reveal that the root cause of this disparity stems from the strong correlation between the classifier head parameter norm and class-wise robustness. To this end, the authors propose two low-cost mitigation methods, HWNwB and DecoSAM, to alleviate the head norm imbalance and enhance the worst-class robustness, respectively. The effectiveness of the proposed methods is validated on multiple datasets.

### Strengths
1. Theoretical support. The paper explains at the theoretical level that a larger classifier head norm can lead to an increase in robustness disparity.
2. Empirical validation. The positive correlation between the classifier head norm and class-wise robust accuracy is empirically demonstrated through statistical correlation analysis.

### Weaknesses
1. Lack of performance stability. HWNwB and DecoSAM vary significantly in effectiveness under different attacks and experimental settings. The former is more effective for within-training PGD metrics and score equalization, while the latter performs better on AA and Worst-Class (WC). This difference raises doubts about which method to choose for practical applications.
2. Insufficient data size. Although the authors show that the near-zero WC on ImageNet is not favorable for fairness assessment, the experimental results can be verified on a medium-sized dataset such as Tiny-ImageNet-200 / ImageNet-100.
3. Lack of discussion on different perturbation budgets. Both training and evaluation are basically fixed at ℓ∞, ε = 8/255 (PGD and AA). The lack of experiments under different budgets (e.g., ε ∈ {4, 8, 16}/255) and different norms (ℓ₂) makes it difficult to judge the robustness of the methods under varying attack strengths and norms.
4. Insufficient discussion of model architectures. The current experiments focus on the WRN family and lack experiments on mainstream architectures such as ViT/DeiT/ConvNeXt, which makes it difficult to assess the utility of the defense methods on the latest architectures.
5. Some theoretical assumptions are strong. The derivation relies on assumptions such as “ψ(x_adv) ≈ ψ(x)”, which may not hold under strong attacks or distributional bias scenarios. The authors need to explicitly cite the assumptions or have an explicit analysis of the assumptions to prove that the assumptions hold.
6. Lack of comparison with other methods for addressing AT fairness. Although the paper discusses class-wise disparity and mentions its potential correlation with robust fairness, there is a lack of comparison and discussion of existing fair adversarial training methods at the experimental and methodological level.

### Questions
Please examine the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates class-wise robustness disparities in adversarial training, where some classes become significantly less robust than others despite balanced data. The authors identify a strong correlation between the norms of classifier head weights and class-wise robust accuracies, showing that adversarial training implicitly amplifies these norm imbalances, leading to uneven robustness. To address this, they propose Head Weights Normalization with Bias (HWNwB) and Decoupled Sharpness-Aware Minimization (DecoSAM), which adjust only the classifier head while keeping the feature extractor fixed. Extensive experiments across multiple datasets and adversarial training algorithms demonstrate that these methods substantially reduce class-wise robustness gaps with minimal computational cost and without degrading overall robustness.

### Strengths
1. The paper offers a novel and well-motivated perspective on class-wise disparity in adversarial training by interpreting it as an implicit bias problem related to head-weight norm imbalance. This viewpoint goes beyond existing fairness or reweighting approaches.
2. The proposed methods, HWNwB and DecoSAM, are simple, lightweight, and practical. HWNwB requires no additional training, while DecoSAM involves only one epoch of head-only fine-tuning, making them computationally efficient.
3. Extensive experiments across multiple datasets demonstrate consistent improvements in worst-class robustness and reduced class-wise disparity, with minimal impact on average accuracy or overall robustness.

### Weaknesses
1. While the paper provides strong correlation evidence, it does not offer a full causal analysis showing whether head norm imbalance is the root cause or merely a symptom of deeper optimization dynamics.

2. HWNwB and DecoSAM are both post-hoc or head-only fine-tuning methods, meaning they depend on an already adversarially trained model. The paper does not explore whether integrating these ideas directly into the training process could yield better or more stable results.

3. The empirical improvements, while consistent, are moderate in some cases; the gains in worst-class robustness often come with small drops in clean or average accuracy, which could be discussed more thoroughly.

4. The paper aims to study and improve robust fairness methods; however, the main experimental tables (e.g., Tables 2 and 3) mainly compare the proposed approaches with standard adversarial training baselines rather than with existing robust fairness methods. Moreover, in Table 4, it is not clearly specified which adversarial training algorithms were used as the underlying models for comparison.

5. The claim that the method is compatible with a wide range of adversarial training algorithms makes the contribution appear less distinctive.

6. The contribution stating that the paper theoretically and empirically demonstrates that adversarial training induces norm imbalances leading to class-wise performance disparities seems less novel, as a similar phenomenon has already been analyzed in prior work such as FRL.

### Questions
1. It would be interesting to investigate whether integrating HWNwB and DecoSAM directly into the training process, rather than applying them post hoc, could lead to better or more stable results.
2. It is not clearly explained what motivated the authors to focus specifically on the classifier head; the paper does not sufficiently justify why the head layer, rather than other components of the model, was chosen as the central point of analysis for class-wise disparity.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the problem of **class-wise disparity in adversarial training**, where different classes show varying robustness even in balanced datasets. The authors reveal a strong correlation (ρ ≈ 0.95) between **the L₂-norms of class-specific head weights** and **their robust accuracies**, suggesting that adversarial optimization introduces an implicit bias into the classifier head.

They formalize this phenomenon theoretically by linking class hardness to gradient gaps and head-norm growth, then propose two lightweight solutions to mitigate it:

- **HWNwB (Head Weights Normalization with Bias)**: post-training normalization of classifier heads while preserving bias terms.
- **Deco-SAM (Decoupled Sharpness-Aware Minimization)**: adaptive class-wise fine-tuning that balances robustness through SAM-based optimization.

Extensive experiments across multiple datasets (CIFAR-10/100, STL-10, OfficeHome) and adversarial training methods (PGD-AT, TRADES, MART, ARoW) show significant reductions in fairness disparity with minimal cost and no validation set required.

### Strengths
- Establishes a clear theoretical link between class hardness, gradient gaps, and head-norm disparity.
- Combines theoretical rigor with thorough empirical validation.
- Introduces two lightweight, algorithm-agnostic mitigation strategies requiring no validation data.
- Demonstrates consistent fairness improvements across datasets and training methods.
- Writing, figures, and appendix materials are clear and reproducible.

### Weaknesses
- Experiments cover small–to–mid-scale benchmarks (CIFAR, STL-10, OfficeHome) but lack validation on **large-scale settings** (e.g., ImageNet-like regimes), so scalability and generality remain untested.
- Robustness evaluation is primarily based on PGD and AutoAttack, which already provide strong coverage. However, including at least one additional optimization-based (e.g., CW) or adaptive attack (e.g., BPDA/EOT) would make the evaluation more comprehensive and confirm that the improvements are not attack-specific.
- Robustness evaluation fixes the perturbation budget at ε=8/255 and focuses on PGD / AutoAttack. Including **ε-sweep experiments** (e.g., evaluating robustness across different attack magnitudes) would clarify how stable the proposed fairness improvements remain as adversarial strength increases.

### Questions
- **Deco-SAM hyperparameter clarity.** The paper lacks detail and sensitivity analysis for **τ**. Could you report the default τ, the rationale for its choice, and a brief ablation (e.g., τ ∈ {…}) showing how **worst-class accuracy**, **class-wise variance**, and **PGD/AA robustness** change across τ and learning-rate schedules?
- **ε-sweep robustness.** Experiments fix the perturbation budget at **ε = 8/255** under PGD-20 and AutoAttack. Have you evaluated whether the **fairness improvements persist across different ε values** (i.e., varying attack strengths)? A compact ε-sweep would clarify the stability of the effect.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper investigates the class-wise performance disparities in adversarial training: despite balanced class frequencies, some classes end up with much lower robust accuracy than others. The authors identify a empirical correlation between the ℓ₂-norms of the classifier head weights and class-wise robust accuracy, where classes with larger head norms tend to have higher robustness. They show that adversarial training exacerbates this norm imbalance, and that this drives the disparity across classes. To mitigate it, the authors propose two lightweight methods that adjust or fine-tune the head parameters with the feature extractor frozen. Experimental results show reductions in class-wise disparity.

### Strengths
The authors reframe class-wise robustness gaps through an implicit bias in head-weight norms, revealing a tight link between last-layer weight norms and per-class robust accuracy—a fresh angle beyond data imbalance or attack heuristics.

The authors provide theoretical and empirical evidence tying adversarial training to growing head-norm disparities and, in turn, to uneven robust accuracy across classes.

### Weaknesses
Disparities in class-wise robustness have already been extensively studied both theoretically and empirically [1–3]. While the observed correlation between the ℓ₂-norms of classifier head weights and class-wise robust accuracy is interesting, it is unclear how is the identified correlation can be related with existing findings, and the paper’s contribution appears incremental relative to prior analyses that connect class margins, logit norms, and adversarial robustness (it is more like a different perspective of the same issue, rather than discovering a under-discovered issue in the disparities of class-wise robustness). Moreover, based on the presented experiments, the proposed method neither clearly outperforms existing approaches nor demonstrates strong complementarity with them.

Several claims seem over-stated or implicit. For example, why is "no validation set required" an important issue? Especially in terms of adversarial training, the usage or acquisition of a validation set seems trivial.

Several important assumptions are deferred to the Appendix. I would strongly suggest the authors to include the assumptions in the main paper with proper justifications. Otherwise, the theortical claims risk being misinterpreted.

The discussions seem limited to $l_{∞}$ attacks. Robustness under stronger or diverse attacks (AutoAttack, multi-target) is underexplored.

[1] Xu, Han, et al. "To be robust or to be fair: Towards fairness in adversarial training." International conference on machine learning. PMLR, 2021.

[2] Ma, Xinsong, Zekai Wang, and Weiwei Liu. "On the tradeoff between robustness and fairness." Advances in Neural Information Processing Systems 35 (2022): 26230-26241.

[2] Wei, Zeming, et al. "Cfa: Class-wise calibrated fair adversarial training." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
