# How Does Fine-Tuned Foundation Models Help for Long-Tailed Data

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Deep long-tail learning is a challenging visual recognition problem that trains models on long-tailed distributed datasets.
In the last decade, a large number of methods have been proposed to solve the problems caused by imbalanced data.
Many methods have been proven useful in learning a deep model from scratch, such as ResNet or ResNeXt, but they have not been validated as effective in fine-tuning the pre-trained foundation models, such as CLIP or ViT. If users inappropriately apply these long-tail learning methods, it may result in worse accuracy than expected.
However, there is no scientific guideline for these methods in the existing literature.
In this paper, we first collect the widely used methods of existing long-tail learning and then conduct extensive and systematic experiments to provide a guideline for the accurate use of these methods in fine-tuning foundation models.
Furthermore, we observe that the current comparison protocol ignores the influence of training cost and hyperparameter selection, which may potentially lead to unfair comparisons and biased results.
Motivated by our empirical studies, we propose a unified fine-tuning framework for long-tailed recognition.
Experimental results demonstrate that the proposed framework outperforms existing methods on multiple long-tailed datasets, including ImageNet-LT, Places-LT, CIFAR100-LT, and iNaturalist 2018.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper systematically studies how classic long-tailed learning methods perform when applied to fine-tuning foundation vision models. It evaluates seven categories of techniques (re-sampling, augmentation, class-balanced losses, classifier design, etc.) under both full and parameter-efficient fine-tuning. The authors find that many traditional methods do not transfer well, while a combination of cosine classifier, square-root sampling, Balanced Softmax/logit adjustment, and label smoothing works reliably. They conclude with a unified fine-tuning framework that outperforms prior long-tail methods across multiple benchmarks.

### Strengths
1 The paper provides a comprehensive and systematic empirical study of seven major categories of long-tailed learning methods on foundation models

2 Based on extensive experiments, the authors deliver a well-validated unified fine-tuning framework that consistently improves long-tailed performance across multiple datasets and backbones

### Weaknesses
(1) I do not fully agree with the authors’ claim in the introduction that “to the best of our knowledge, there has not been a systematic study on how to fine-tune foundation models under a long-tailed distribution.” In fact, LIFT [a] has already provided a systematic analysis of imbalance issues under long-tailed settings and explored various strategies. Works such as [b] and [c] have also examined biases in foundation models under long-tailed distributions, and LPT [d] offers a deeper investigation as well. I recommend that the authors reconsider the positioning of their contribution and more precisely articulate the gap their work aims to fill, rather than relying on an overly broad claim.

(2) The results on several benchmarks do not appear to surpass LIFT, which achieves competitive performance with only 10 training epochs and minimal additional techniques. From this standpoint, it is difficult to assess the practical significance and novelty of the proposed method.

(3) Many of the examined techniques and strategies depend heavily on training hyperparameters such as the number of epochs, learning rate. More experiments are required to understand how sensitive the proposed tricks are to these hyperparameters and to more fully validate their robustness.

[a]  Shi, Jiang-Xin, et al. "Long-tail learning with foundation model: Heavy fine-tuning hurts." arXiv preprint arXiv:2309.10019 (2023). ICML2024

[b] Chen, Jiahao, et al. "Rethinking the Bias of Foundation Model under Long-tailed Distribution." arXiv preprint arXiv:2501.15955 (2025). ICML2025

[c] Wen, Xin, et al. "What makes clip more robust to long-tailed pre-training data? a controlled study for transferable insights." Advances in Neural Information Processing Systems 37 (2024): 36567-36601.

[d] Dong, Bowen, et al. "Lpt: Long-tailed prompt tuning for image classification." arXiv preprint arXiv:2210.01033 (2022). ICLR

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper systematically evaluates long-tail learning methods, including re-sampling and loss functions for fine-tuning foundation models, i.e., CLIP and ViT. A unified framework that outperforms existing approaches on imbalanced datasets has been proposed, and empirical guidelines have been provided for the long-tailed learning community. However, evaluations have been carried out on limited model architectures, and the proposition is mainly validated through accuracy and efficiency. The impact on the learned representations remains undiscovered.

### Strengths
- The paper provides a systematic empirical study of long-tail learning methods and offers actionable insights, e.g., recommending Balanced Softmax and Square-root sampling for fine-tuning foundation models on imbalanced data, which is valuable for the whole long-tailed learning community.

-  Proposes a novel framework combining optimal methods to achieve trade-offs between performance and computational cost.

- Tests on 4 datasets with detailed observations, e.g., hyperparameter sensitivity analysis, examine robustness of methods like LADE, noting instability with improper hyperparameters.

### Weaknesses
- Only CLIP/ViT are considered. Extending to other architectures (e.g., DINO) could strengthen the generalizability.

- Mentions potential leakage between ImageNet and IN21K-ViT, but doesn’t quantify how its impact was mitigated. Additionally, the BALLAD baseline is omitted in Table 10, which makes superiority claims less convincing.

- The motivation is intuitive, and the work's unified framework combines existing methods, but groundbreaking algorithmic novelty lacks emphasis.

- Some tables (e.g., resampling results across datasets) could be consolidated for brevity. Moreover, the style of the tables is not unified. Some are full-bordered, and some are three-line tables.

### Questions
1. What is the impact brought by the new design to the learned representations? Qualitative results and more detailed ablations are preferred to indicate the effectiveness of the proposed method on learned representations.

2. Why were only CLIP and ViT tested? How about DINO?

3. LADE collapses in Fig. 2. Does this reveal fundamental limitations of logit adjustment for foundation models, or is it fixable via hyperparameter tuning?

4. Beyond acknowledging potential leakage between ImageNet and IN21K-ViT, what specific steps were taken to ensure contamination didn't inflate results?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a systematic study of how classical long-tail learning methods perform when applied to foundation models (CLIP and ViT) instead of training from scratch. The research categorizes existing methods into 7 groups: (1) Re-sampling, (2) Data Augmentation, (3) Class-sensitive Loss, (4) Balanced Classifier, (5) Knowledge Distillation, (6) Ensemble Learning, and (7) Other tricks. Through extensive experiments across two fine-tuning paradigms (Full Fine-Tuning and Parameter-Efficient Fine-Tuning) and four standard datasets (CIFAR100-LT, Places-LT, ImageNet-LT, iNaturalist 2018), they provide empirical guidelines for practitioners. The authors then propose a unified framework combining the most effective methods and demonstrate competitive performance compared to state-of-the-art approaches.

### Strengths
- The paper provides a timely revisit of how existing methods perform when pre-trained models are adopted, which is practically beneficial.
- The experimental setup is well-structured and thorough seven method categories covering major long-tail learning approaches over four datasets. The paper also provides detailed hyperparameter specifications and comprehensive ablation studies, facilitating reproducibility and follow-up research.
- The work provides actionable insights, clearly showing which methods work best in different settings.
- This work also considers training costs, computational efficiency, and hyperparameter sensitivity. This practical consideration is crucial for real-world deployment.

### Weaknesses
- The work is purely empirical. While systematic evaluation has value, the contribution is relatively modest. The proposed ultimate framework is also simply a combination of best-performing existing methods without deeper insight into why these combinations work synergistically.
- No statistical significance testing is reported. It would be beneficial to rule out experimental noise with multiple independent runs and increase the reliability of results.

### Questions
- Do the authors expect these findings to generalize to other foundation models beyond CLIP and ViT (e.g., DINOv2, MAE, SigLIP2)?
- What properties of pre-trained representations make them more suitable to certain long-tail learning techniques or hyper-parameters?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper investigates how classic long-tailed (LT) learning techniques fare when fine-tuning foundation models (CLIP, ViT) under both full fine-tuning (FFT) and parameter-efficient fine-tuning (PEFT). It benchmarks seven families of methods (re-sampling, data augmentation, class-sensitive losses, balanced classifiers, and “other tricks”), and synthesizes an “ultimate framework” combining Cosine classifier + Square-root sampling + Balanced Softmax (BS) + Label Smoothing (LS). This framework yields consistent gains across ImageNet-LT, Places-LT, CIFAR100-LT, and iNaturalist-2018, sometimes surpassing recent LT methods, while noting that naive data augmentation have inconsistent effects on
performance across different datasets and models.

### Strengths
1: The paper systematically reviews and tests representative techniques (e.g., ROS/RUS/Square-root, Rand/AutoAugment, Focal/LDAM/CB/BS/LA/LADE, Cosine/τ-norm, mixup/LS) across FFT/PEFT and backbones

### Weaknesses
1: While this study presents ample empirical results, the findings remain largely superficial and fail to uncover the intrinsic mechanisms by which long-tailed data distributions influence fine-tuning. Prior works, such as [1] and [2], provide theoretical insights into the geometric properties of contrastive representations learned from balanced versus imbalanced datasets. Other studies [3, 4] investigate the underlying mechanisms of long-tailed learning from an empirical perspective. In the context of CLIP, [5] offers a promising direction for exploring how fine-tuning with long-tailed data impacts downstream performance. Incorporating such theoretical or representational analyses would substantially deepen the paper’s contribution and explanatory power.

[1] Dissecting supervised contrastive learning, Graf et al., ICML 2021.

[2] Geometry of Long-Tailed Representation Learning: Rebalancing Features for Skewed Distributions, Yi et al., ICLR 2025.

[3] Imbalance trouble: Revisiting neural-collapse geometry, Thrampoulidis et al., NeurIPS 2022. 

[4] What Makes CLIP More Robust to Long-Tailed Pre-Training Data? A Controlled Study for Transferable Insights, Wen et al., NeurIPS 2024. 

[5] Decipher the Modality Gap in Multimodal Contrastive Learning: From Convergent Representations to Pairwise Alignment, Yi et al., arxiv.

### Questions
1: Why does RUS×N negatively affect tail performance? Beyond reporting the empirical trend, the paper should try to provide a deeper analysis of the underlying mechanism. For instance, can the authors examine representation drift or head-biased margin dynamics as N increases? Such analyses would clarify whether the degradation arises from overfitting to majority classes, loss of feature diversity in tails, or instability in the learned decision boundaries.

2: Please provide a deeper mechanistic analysis of why the combination of Cosine normalization, BS, and LS exhibits robustness under long-tailed (LT) fine-tuning for foundation models (FMs). For instance, an examination of weight norms, feature angular distributions, and class-wise margins before and after fine-tuning would help explain the underlying dynamics contributing to this robustness.

### Soundness
2

### Presentation
3

### Contribution
2
