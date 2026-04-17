# Tackling Fake Forgetting through Uncertainty Quantification

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Machine unlearning seeks to remove the influence of specified data from a trained model. While metrics such as unlearning accuracy (UA) and membership inference attack (MIA) provide baselines for assessing unlearning performance, they fall short of evaluating the reliability of forgetting. In this paper, we find that the data points misclassified by UA and MIA still have their ground truth labels included in the prediction set from the uncertainty quantification perspective, which raises the issue of fake forgetting. To address this issue, we propose two novel metrics inspired by conformal prediction that provide a more reliable evaluation of forgetting quality. Building on these insights, we further propose an unlearning framework that integrates conformal prediction into the Carlini & Wagner adversarial attack loss, which can effectively push the ground truth label out of the conformal prediction set. Through extensive experiments on image classification tasks, we demonstrate both the effectiveness of our proposed metrics and the superiority of our framework. Code is available at https://anonymous.4open.science/r/MUCP-60E4.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper identifies limitations in existing machine unlearning metrics like unlearning accuracy (UA) and membership inference attacks (MIA), arguing that they fail to account for uncertainty and confidence, leading to "fake forgetting" where misclassified forget data points still retain ground-truth labels in the model's uncertainty-quantified prediction sets. 
To address this, the authors apply conformal prediction to recover such data and propose two new metrics: Conformal Ratio (CR) and MIA Conformal Ratio (MIACR). They also introduce a general unlearning framework (CPU) that integrates conformal prediction with Carlini & Wagner (C&W) adversarial attack loss to push ground-truth labels out of the prediction set during training-based unlearning. 
Experiments are conducted on image classification tasks (CIFAR-10, Tiny ImageNet) using ResNet-18 and ViT models, showing how CR and MIACR captures the fake forgetting, and how the proposed CPU performs under lens of conformal prediction.

### Strengths
- The manuscript is well-written and easy to follow. 
- The core idea of incorporating uncertainty quantification (via conformal prediction) into machine unlearning evaluation is novel and timely, addressing a real gap in current metrics that overlook confidence levels. This perspective could inspire more robust assessments in unlearning research.
- The proposed metrics (CR and MIACR) are straightforward extensions of conformal prediction concepts, providing a more comprehensive view by balancing coverage and set size.
- The unlearning framework (CPU) is a practical enhancement to existing training-based methods, demonstrating improvements in forgetting quality across baselines like Finetune and Random Label.

### Weaknesses
Overall, although I like the motivation and idea of this manuscript, I think the experiments are need to be improved. The metrics seem more sensitive to whether forgetting occurred at all, rather than the quality of forgetting.

- The definition of "fake forgetting" may be overly broad. Conformal prediction naturally expands prediction sets to achieve high coverage (e.g., 95% confidence), so recovery of ground-truth labels is expected even in well-unlearned models. 
  - For instance, lines 154-157 note recovery in the Retrain (RT) baseline (e.g., 30.6% in Table 2 for 10% random forgetting), which undermines the claim that this indicates incomplete forgetting. In random forgetting scenarios, which inherently oppose generalization, high probability on ground-truth labels despite misclassification is not anomalous.

- Existing metrics (UA, RA, TA) in Table 1 raise concerns about baseline performance: Finetune shows low UA (e.g., 3.84% for 10% forgetting), suggesting under-forgetting, while Random Label has degraded RA/TA, indicating over-forgetting. This makes it unclear if the observed fake forgetting is due to metric limitations or poor unlearning quality. 
  - Random forgetting benchmarks generally underperform (low UA across methods), so focusing more on class-wise forgetting might provide clearer insights.

- Coverage values are high even for RT (e.g., ~0.94), potentially inflated by samples that were never forgotten (due to low UA). 
  - In CIFAR-10 results, other methods show higher coverage than RT, leading to worse (higher) CR scores, but this may simply reflect their poorer UA, not superior fake forgetting detection.
  - Restricting analysis to mislabeled data (as in Table 2) might yield a fairer metric for fake forgetting quality, rather than mixing in UA effects.

- Tiny ImageNet results may have sanity issues: e.g., NegGrad (coverage 0.999)/(set size 0.949) yielding CR 2.184 doesn't align with expected calculations. Also, RT often shows more fake forgetting per CR than baselines, possibly contradicting the motivation. 

- In class unlearning (Table 10), set sizes often equal the total number of classes with coverage=1, implying all classes (including ground-truth) are in the conformal prediction set. This suggests nearly all forgetting is "fake," which seems counterintuitive and may indicate over-relaxation in conformal sets.

- For CPU, updating \hat{q} every epoch during training could introduce instability in the objective. Is there theoretical justification for convergence, or empirical evidence of stability?

- Notation for conformal sets \mathbb{C}(x) is inconsistent between CR and MIACR, as they apply to different contexts (multi-class vs. binary membership). Separating notations would improve clarity.

### Questions
- Could you clarify why recovery in RT is treated as fake forgetting, given its role as the gold-standard baseline?

- In class unlearning, why does coverage=1 across methods imply widespread fake forgetting? Does this suggest conformal prediction is too permissive for multi-class tasks?

- Have you tested CPU on stronger baselines where UA is near-perfect, to isolate fake forgetting from under-forgetting effects?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper claims that unlearning judged “successful” by UA/MIA still does not correctly evaluate the unlearning methods because these methods leave the true label in a conformal prediction (CP) set. They show that a large portion of misclassified “forgotten” points remain recoverable under CP.  The paper proposes two CP-inspired metrics: CR (Coverage divided by Set Size) and MIACR (probability that the CP set equals {non-member} in the MIA setup). They also propose a new unlearning paradigm, CPU training loss, that pushes the true label’s non-conformity above a CP threshold to exclude it from the set, which is inspired by CW attack.

### Strengths
1. The paper is well-written and the new metrics and methods are presented clearly.
2. In the setting of bias removal, where the goal is to make the model’s performance poor, the presented metrics and the method would be good additions to the literature.
3. There are several ablation studies for further illustration of the hyper-parameters of the presented metrics and method.

### Weaknesses
1. I believe the authors could better motivate the problem. I think this presented metrics and method would have made much more sense if they had specifically mentioned that their goal is the setting of “bias removal” in [1]. However, the authors generalize those metrics to two of the settings in [1]: 1.Bias removal and 2.User privacy. However, most of their claims in the setting of user privacy are incorrect. I try to briefly explain some of these fallacies: 
   - As mentioned in [1] (which is the work authors also refer to), in the setting of user privacy, the goal is to train a model that behaves similar to the retrained model. So the retrain model is the gold standard, and this definition have been followed in dozens of seminal prior work.Therefore a claim such as the one in line 154 of the submitted manuscript “Even for the RT baseline, UA does not reliably assess whether a data point has truly been forgotten, since 30.6% of UA misclassified data points can still be recovered by conformal prediction” shows a problem with the use-case of your metric for this setting. You cannot claim that the retrain models for privacy settings are not unlearned enough!
    - Authors claim that MIAs don’t measure the user privacy correctly because some of the samples they classify as non-training samples still have high probabilities for the correct label. That is again the expected behavior from the retrain models on the forget samples in the user privacy setting. As mentioned in [2], the retrain models that have not seen the forget samples during training, perform similar to the test samples when making predictions on the forget samples. What MIA does is to reveal if there is privacy leakage through the over-confidence of the model on the forget samples due the the fact that they were used during the training. The goal of the MIA method is to detect whether forget samples are identifiable from the test samples. That exactly aligns with the definition of user privacy, and there is nothing “fake” in what they measure. Some of the recent works in unlearning even utilize MIAs that have a correct baseline of 50% AUC for the retrain models in detecting forget samples vs. test samples [2,3], which basically agrees to the fact that to a retrain model forget samples are the same as unseen (test) samples. That is also why in other prior works the gap with the retrain model is computed because although a larger value represents under-unlearning and forget samples that are more similar to the training samples, the smaller values also represent an over-unlearning behavior [1,4]. Over-unlearning (in the user privacy setting) is often due to the enforcing poor performance on the forget samples (which is exactly what your proposed method tries to achieve). According to [1] that you have cited as a reference as well: “Success from a privacy perspective is associated with a forget error only as high as that of retraining-from-scratch”.

2. Although authors have used [1] as the justification for their metrics, it seems they have not used the method they introduced specifically for the setting of bias removal as a baseline!

3. It would be great to also compare the presented metrics with more recent metrics for unlearning evaluation (e.g., U-MIA and U-LIRA) [3].



### Minor weaknesses:

1. It would have been better to rely on more recent unlearning methods to showcase your metrics [1,2,5,6,7,8].

------------
[1] Kurmanji, M., Triantafillou, P., Hayes, J., & Triantafillou, E. (2023). Towards unbounded machine unlearning. Advances in neural information processing systems, 36, 1957-1987.

[2] Ebrahimpour-Boroojeny, A., Sundaram, H., & Chandrasekaran, V. Not All Wrong is Bad: Using Adversarial Examples for Unlearning. In Forty-second International Conference on Machine Learning. 

[3] Cadet, X. F., Borovykh, A., Malekzadeh, M., Ahmadi-Abhari, S., & Haddadi, H. (2025, June). Deep Unlearn: Benchmarking Machine Unlearning for Image Classification. In 2025 IEEE 10th European Symposium on Security and Privacy (EuroS&P) (pp. 939-962). IEEE.

[4] Shi, W., Lee, J., Huang, Y., Malladi, S., Zhao, J., Holtzman, A., ... & Zhang, C. (2024). Muse: Machine unlearning six-way evaluation for language models. arXiv preprint arXiv:2407.06460.

[5] Zhang, B., Dong, Y., Wang, T., & Li, J. Towards Certified Unlearning for Deep Neural Networks. In Forty-first International Conference on Machine Learning.

[6] Cha, S., Cho, S., Hwang, D., Lee, H., Moon, T., & Lee, M. (2024, March). Learning to unlearn: Instance-wise unlearning for pre-trained classifiers. In Proceedings of the AAAI conference on artificial intelligence (Vol. 38, No. 10, pp. 11186-11194).

[7] Bonato, J., Cotogni, M., & Sabetta, L. (2024, September). Is retain set all you need in machine unlearning? restoring performance of unlearned models with out-of-distribution images. In European Conference on Computer Vision (pp. 1-19). Cham: Springer Nature Switzerland.

--------------

Finally, I would be willing to increase the rating if my concerns are addressed.

### Questions
1. How authors would justify their metrics in the setting of user privacy, where over-unlearning can be as bad as under-unlearning?

2. In section 2.3.2 for unlearning in CIFAR-10 authors mention they need more than 1000 samples to avoid abnormal $\hat q$ values. How does this number of samples changes for larger datasets such as TinyImagenet? Does that scale with the size of dataset? What about the difficulty of the dataset (leading to lower confidence values) or model complexity? How would they affect the number of samples needed for calibration?

3. Do you have the results on MIA scores used for your unlearning framework?

### Soundness
2

### Presentation
3

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
The paper investigates the phenomenon of Fake Forgetting in machine unlearning, where models appear to forget through misclassification, yet the true labels remain within the prediction set from an uncertainty quantification perspective. To address this, the authors propose two new conformal prediction–based metrics, CR (Conformity Rate) and MIACR (Model Information Alignment Conformity Rate), designed to more reliably evaluate unlearning quality. Furthermore, they integrate conformal prediction into a Carlini & Wagner–based forgetting loss to exclude true labels from the conformal prediction set, achieving more controlled and interpretable forgetting. The method is empirically validated on CIFAR-10 and Tiny-ImageNet using ResNet-18 and ViT backbones.

### Strengths
1.The paper identifies an underexplored issue in current unlearning evaluation—fake forgetting—and provides a clear motivation for moving beyond error-based metrics.
2.Introducing conformal prediction into the unlearning framework for both evaluation (CR/MIACR) and optimization is conceptually sound and adds interpretability to the unlearning process.
3.Experimental results on CIFAR-10 and Tiny-ImageNet demonstrate promising improvements under small-model settings, with clearly presented figures and well-written explanations.
4.The paper is well-organized, logically coherent, and easy to follow, making the technical contributions accessible to a broad machine learning audience.

### Weaknesses
1.The fake forgetting phenomenon is not entirely new—similar discussions on misinterpretation of forgetting through uncertainty or distributional bias have appeared in small-model unlearning literature (e.g., Becker & Liebig 2022, Fast Yet Effective Machine Unlearning).
2.The proposed metrics (CR and MIACR) are derived from conformal prediction, whose core idea of distribution-free coverage control is well-established. The novelty primarily lies in the application domain, not in methodological advancement.
3.There is a potential overfitting between the evaluation metric and the training objective—since conformal-based losses are optimized directly, the improvement in CR/MIACR may not generalize to other unlearning quality measures.

### Questions
See the comments above

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
4

### Summary
This paper reveals "fake forgetting" in machine unlearning, where metrics like UA and MIA miss misclassified forget data recoverable (>50%) via conformal prediction sets, as these traditional metrics focus solely on point predictions without accounting for model uncertainty, allowing residual knowledge of forget data to persist in plausible label sets. This paper introduces two new metrics: CR (balancing coverage and set size) and MIACR (excluding membership hints) and a CPU (Conformal Prediction Unlearning) framework augmenting training-based methods with a new loss to exclude true labels from sets. Experiments on CIFAR-10 (ResNet-18) and Tiny ImageNet (ViT) show UA gains (up to 9.23%), MIA improvements, and preserved accuracies, with Grad-CAM confirming deeper forgetting; yet, reliance on exchangeability assumptions needs broader testing.

### Strengths
- **Clarity and Motivation**: The paper is clearly written, the problem is well described, and the motivation for introducing conformal-based unlearning is convincing, making the contribution accessible and easy to follow.

- **Novel Identification of "Fake Forgetting"**: The paper insightfully uncovers a critical flaw in existing unlearning metrics (UA and MIA) by demonstrating through conformal prediction that over 50% of misclassified forget data remains recoverable in prediction sets.

- **Innovative Metrics and Framework**: It proposes two well-motivated metrics: CR (balancing coverage and set size) and MIACR (enforcing strict membership exclusion).

### Weaknesses
**Major Weaknesses**

1. **Limited Baseline Comparisons**: Only three classic unlearning methods (RT, FT, RL) are evaluated, with no inclusion of more recent or diverse baselines (e.g., LoTUS from CVPR 2025), potentially overlooking how CPU performs against state-of-the-art alternatives.

2. **Limited Experimental Scope**: Evaluations are restricted to image classification tasks on small datasets (CIFAR-10 and Tiny ImageNet) with specific models (ResNet-18 and ViT), lacking  larger-scale benchmarks, which questions generalizability to real-world unlearning scenarios. It would be interesting to see the extension of this work on conditional image generation as well. 


3. **Potential Metric Overfitting and Privacy Evaluation Gaps**: The CPU framework directly optimizes a loss function tailored to exclude true labels from conformal prediction sets, which underpins the proposed CR and MIACR metrics, raising concerns of circularity where improvements on these metrics may not reflect broader unlearning quality or privacy guarantees, despite cross-evaluations on UA and MIA; furthermore, while basic MIA is used, there's no testing against advanced attacks (e.g., model inversion or backdoor recovery) or real-world metrics like differential privacy, limiting claims of practical forgetting efficacy.

**Minor Weaknesses**

- **Qualitative Visuals Could Be Expanded - Grad-CAM Size**: The Grad-CAM heatmaps (Figure 1, Table 1) provide intuitive evidence of forgetting shifts but are small and could be larger to better convey the message.

- **Qualitative Visuals Could Be Expanded - Teaser Figure Absence**: A teaser figure could significantly help to convey the message to the reader.

### Questions
1. Given the work's focus on image classification, I appreciate if the authors could provide insight into how the CPU framework might be adapted to multimodal, LLMs, or image generation tasks. It would be valuable to suggest experiments evaluating its performance on datasets like LAION or unlearning in vision-language models/diffusion models to assess its broader applicability.

### Soundness
3

### Presentation
3

### Contribution
3
