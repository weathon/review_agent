# WRF4CIR: Weight-Regularized Fine-Tuning Network for Composed Image Retrieval

- Decision: Reject
- Scores: 4, 2, 6, 8

## Abstract
Composed Image Retrieval (CIR) task aims to retrieve target images based on reference images and modification texts. Current CIR methods primarily rely on fine-tuning vision-language pre-trained models. However, due to the large-scale nature of pre-trained models and the limited training data for CIR task, significant overfitting commonly occurs during fine-tuning, resulting in poor generalization. To address this issue, in this paper, we introduce Weight-Regularized Fine-tuning network for CIR, termed WRF4CIR. Specifically, during the fine-tuning process, we apply adversarial perturbations to the model weights for regularization, where these perturbations are generated in the opposite direction of gradient descent. Intuitively, WRF4CIR increases the model’s learning difficulty on training data, effectively mitigating overfitting. Technically, WRF4CIR explicitly regularizes the flatness of the weight loss landscape, enhancing the model’s robustness to weight perturbations and improving generalization. Extensive experiments on benchmark datasets demonstrate that WRF4CIR significantly narrows the generalization gap and achieves substantial improvements over existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a novel Weight-Regularized Fine-tuning network for CIR, termed WRF4CIR.  By introducing adversarial perturbations to the model weights, which act inversely to gradient descent, WRF4CIR increases the difficulty of learning the model on the training data, effectively mitigating overfitting. Extensive experiments on benchmark datasets demonstrate that WRF4CIR significantly narrows the generalization gap and achieves significant improvement over existing methods.

### Strengths
1. The idea of ​​providing perturbations to the parameters in the direction opposite to the gradient is interesting.
2. There are sufficient comparative tests and experimental analysis

### Weaknesses
1. Regarding the overfitting issue, as the training data decreases, it may not be overfitting, but insufficient training. I think this part of the analysis is not sufficient.
2. I think the starting point of this article is good, but the analysis and method design need to be improved.
3. How much improvement does Q-Former bring to the method? It seems that Q-Former plays an important role in the method, but the ablation experiment does not provide corresponding results.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper aims to address the significant overfitting problem that occurs when fine-tuning VLPs on CIR tasks, which often have limited training data. The authors propose a weight-regularized fine-tuning method called WRF4CIR, which applies adversarial perturbations to the model weights to flatten the loss landscape and, consequently, improve generalization .

### Strengths
The motivation is clear and the experimental results show some improvement on the FashionIQ and CIRR datasets.

### Weaknesses
1. The primary weakness of this paper is the positioning of its core contribution. The authors claim to propose a "novel" Weight-Regularized Fine-tuning network, with the central idea of finding "flatter minima" in the loss landscape to improve generalization.This concept is not new; it is extensively studied in the deep learning community. More importantly, the proposed optimization objective (Eq 1 7, Eq 3), $\min_{\theta} \max_{\delta} \mathcal{L}(\theta+\delta)$, is functionally identical to the core mechanism of Sharpness-Aware Minimization (SAM) (Foret et al., 2021) and its variants (like ASAM). SAM is a highly prominent and widely used regularization technique that explicitly seeks flat loss regions by finding adversarial perturbations (via gradient ascent) in the weight space. Critically, the paper fails to cite or even mention SAM anywhere. WRF4CIR appears to be a direct application of SAM (or a very similar adversarial weight perturbation technique) to the CIR task.
2. The paper's optimization objective (Eq 3) is a $\min \max$ problem. This requires an inner loop of gradient ascent to find the perturbation $\delta$ that maximizes the loss. The authors correctly state this, noting the perturbation is "generated in the opposite direction of gradient descent". The direction of gradient descent is $-\nabla \mathcal{L}$, so its opposite (gradient ascent) should be $+\nabla \mathcal{L}$. However, in Algorithm 1, Line 5, the formula given for the perturbed model is:$f_{\theta+\delta}(t)\leftarrow f_{\theta}(t)-\gamma\frac{L(f)}{||L(f)||}||f_{\theta}(t)||$ Here, $L(f)$ is defined in Line 4 as the gradient $\nabla_{f_{\theta}}l_{g2t}(...)$. This means the applied perturbation $\delta$ is in the direction of $-\nabla \mathcal{L}$, which is gradient descent, the exact opposite of the required gradient ascent and contradicts the paper's own text and objective.
3. The authors candidly admit in Section 5.3 and Appendix A.8 that the method requires a second backpropagation, significantly increasing training time (e.g., from 2.85h to 4.78h on an A100, a ~68% increase) . This is a substantial computational overhead. As a mitigation, the authors propose a "random weight perturbation" variant ($WRF4CIR_{RWP}$). However, according to Table 3, this faster variant's performance (e.g., 66.18 Rmean on FashionIQ) is significantly worse than the full WRF4CIR (67.34) and only achieves about half the performance gain over the baseline (64.80) . This makes the method's practical utility questionable. The authors need to provide a stronger justification for why the 2.54% (67.34 vs 64.80) Rmean gain is worth a nearly 70% increase in training cost.
4. The authors claim in Appendix A.4 that the method is "not particularly sensitive" to the choice of the perturbation strength $\gamma$ and that $\gamma=0.001$ is a "highly consistent" optimal value. However, Figure 4(a) and Figure 7  clearly show the opposite. In Figure 4(a), the performance for both FashionIQ (R@10) and CIRR (R@1) hits a sharp peak around $\gamma=0.001$ and then immediately drops. Figure 7 (Left) also shows the BLIP2 ViT-G performance on FashionIQ peaking at 0.001 and then dropping sharply .

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces WRF4CIR, a weight regularization framework designed to improve the generalization of composed image retrieval (CIR) models fine-tuned from large vision–language pretraining (VLP) models (e.g., CLIP, BLIP-2). The authors observe that CIR datasets are relatively small, leading to strong overfitting when fine-tuning powerful VLPs, and propose a remedy for this issue.

### Strengths
- The paper highlights an important and often overlooked phenomenon in current CIR literature and provides a detailed analysis to demonstrate it.
- The authors conduct extensive experiments across multiple backbones and CIR datasets, supporting the robustness of their findings.
- The paper clearly explains the proposed method and its underlying motivation, making it easy to follow and reproduce.

### Weaknesses
- Core idea: To the best of my understanding, the authors implement an existing method within the CIR framework. In this case, the paper lacks clear methodological novelty. Nevertheless, the work remains valuable to the community due to its detailed analysis and comprehensive experiments.

- (Continuing previous point previous point) Transparency and attribution: The paper does not provide sufficient transparency regarding the origin of the proposed method. This aspect should be explicitly discussed, beyond a brief mention in the related work section, and include proper statement and and citation. If the authors claim full novelty, I can point to several earlier papers employing very similar techniques. My overall rating is heavily influenced by this issue and will be changed according to the authors’ clarification in the rebuttal.

- Backbone accuracy: In Tables 1 and 2, the backbone sizes of the baselines are inaccurate. For example, CASE uses ViT-B rather than ViT-L. The authors should carefully verify the backbone details for all compared methods, as this is a crucial element for fair and accurate comparison.

- Figure 2 interpretation: The intention behind Figure 2 is to illustrate performance gaps across three datasets. However, averaging recall values (e.g., Recall@1 or Recall@5) across different datasets is a bad idea, as each dataset has a different retrieval candidate pool that directly affects recall magnitudes. For instance, a 10% Recall@1 on a pool of one million images may reflect strong performance, while 80% Recall@1 could be weak on a pool of ten images. It would be clearer to show separate bars per dataset, or focus on one dataset in the main figure and move similar trends for the others to the appendix.

### Questions
- Figure 5c - what is ‘best’ and ‘last’ results? the authors should provide more details regarding this figure.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors focused on the composed image retrieval. Considering the significant overfitting issues in the existing methods, the authors proposed a weight-regularized fine-tuning network for CIR. Specifically, the designed method includes adversarial perturbations to the model weights for regularization. Experimental results prove the effectiveness of the proposed method.

### Strengths
1. The authors investigated the overfitting problem in the VLP-based CIR task, which has not been systematically analyzed in previous works.
2. The designed weight-regularized fine-tuning network can mitigate the overfitting issue, thus improving the final retrieval performance.
3. Experimental results prove the effectiveness of the proposed method.

### Weaknesses
1. Overfitting is a common issue in many VLP-based methods for the CIR task. While the authors did not apply WRF4CIR to previous methods to verify the effectiveness of their designed method across baseline approaches.
2. The reported results on FashionIQ are ambiguous: some methods are evaluated on the VAL-split, whereas others are evaluated on the original-split, as noted in [1].

[1] A Comprehensive Survey on Composed Image Retrieval. ACM TOIS 2025.

### Questions
As listed above.

### Soundness
3

### Presentation
3

### Contribution
3
