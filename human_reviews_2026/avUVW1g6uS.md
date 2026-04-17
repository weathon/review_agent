# SAMerging: Sharpness-aware Model Merging via Multi-Teacher Knowledge Distillation

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Model merging offers a lightweight alternative to joint multi-task learning (MTL), which is often costly or data-prohibitive. While the task arithmetic seems promising, it is brittle to coefficient scaling, and we observe that recent approaches, such as AdaMerging, that learn these coefficients, remain sensitive to initialization. This raises a key question: can merging coefficients be learned in a principled, label-free way? We introduce SAMerging, a method that learns coefficients by seeking flat minima. Our approach is grounded in two theoretical contributions. First, we derive a flatness-aware PAC-Bayes generalization bound for the merged model, featuring a novel cross-task heterogeneity term that quantifies expert-task mismatch. Second, this analysis guides us to frame merging as multi-teacher knowledge distillation on a small, unlabeled dataset. We formally show that minimizing the student-teacher KL divergence tightens an upper bound on the merged model's excess risk. We then employ Sharpness-Aware Minimization (SAM) to find robust solutions that generalize better. Empirically, SAMerging establishes a new state of the art on vision and NLP benchmarks. Notably, it surpasses AdaMerging with accuracy gains of $+4.5\%$ on TA-8 and $+11.7\%$ on TALL-20. This is achieved with remarkable data efficiency, using $10\times$ fewer calibration data and proving effective even in data-scarce settings with as few as $16$ examples per task. Furthermore, it requires no original training data and incurs no additional inference-time or memory overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
First, the authors derive a PAC-Bayes generalization bound for the merged model, which highlights the importance of finding "flat" minima in the loss landscape and introduces a "cross-task heterogeneity" term to quantify the mismatch between different models. Second, they frame the model merging problem as a multi-teacher knowledge distillation task. This involves minimizing the KL divergence between the merged model's predictions and the expert models' predictions on a small, unlabeled dataset. To find solutions that generalize well, they employ Sharpness-Aware Minimization, which explicitly seeks out flat regions in the loss landscape.

### Strengths
* The derivation of the PAC-Bayes bound (Theorem 2) and the excess risk bound (Theorem 3) logically connects the goals of finding flat minima and minimizing KL divergence to the ultimate objective of better generalization. The results convincingly demonstrate the superiority of SAMerging over existing data-dependent methods, particularly in its data efficiency.
* The connection between the theoretical insights and the final algorithm is explained logically, making the design choices easy to follow.
* By providing a robust, data-efficient method, SAMerging addresses key limitations of prior work, such as sensitivity to initialization and high data requirements. It achieves state-of-the-art performance with minimal calibration data and no inference overhead.

### Weaknesses
* The theoretical proofs and methods used in this paper, such as SAM and multi-teacher knowledge distillation, have been widely explored in previous work. The contribution here appears to be more of a combination of existing techniques applied to model merging, rather than a novel approach.
* The paper acknowledges this limitation, noting that the NTK regime is most accurate near the pretrained initialization. It remains unclear how well this assumption holds in practice, especially when the fine-tuned models have diverged significantly from the pretrained model.
* The experiments are primarily focused on classification tasks. While these are standard benchmarks, the paper's claims would be strengthened by evaluating SAMerging on a more diverse set of tasks, particularly on modern LLMs.

### Questions
It adds computational cost during the merging process due to the use of SAM, which involves an additional forward/backward pass to find the "worst-case" perturbation. The authors acknowledge this as a "calibration-time cost" but do not quantify it relative to other methods. A brief analysis of this trade-off would provide a more complete picture of the method's practicality.

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
3

### Summary
This paper proposes SAMerging, a method for model merging in the context of multi-task learning. SAMerging selects layer-wise merging coefficients by optimizing for flat minima and aligning with expert teachers via multi-teacher knowledge distillation on small amounts of unlabeled data. The theoretical backbone includes a PAC-Bayes generalization bound for the merged model, introducing a new cross-task heterogeneity term and connecting optimization of the merged model's sharpness and KL fit to rigorous generalization control. Empirically, SAMerging is tested on a range of computer vision and NLP benchmarks, consistently outperforming both data-free and data-dependent baselines, including AdaMerging, using significantly fewer calibration data, with no added inference overhead.

### Strengths
- This paper establishes a detailed PAC-Bayes generalization bound for MTL model merging and introduces an explicit cross-task heterogeneity term. This analysis provides motivation for practical design choices and clarifies failure modes.
- The method proposed in this paper addresses how to train a model with excellent performance even under zero initialization, which differs from previous works that require task arithmetic information.
- This paper also presents extremely detailed ablation experiments, which fully demonstrate the superiority of the SAMerging method.

### Weaknesses
- The paper does not include a comparison with the work ProDistill [1]. This is because the paper mentions that it requires 1,600 samples to achieve optimal performance, while the number of samples used in ProDistill is far fewer than that required by SAMerging. Is it necessary to further validate the conclusion that SAMerging requires fewer samples by comparing it with ProDistill?
- The description of experimental details is insufficient. The paper does not provide specific values for parameters such as ρ and η, nor does it conduct basic ablation experiments on these hyperparameters—these details need to be supplemented.
- The paper does not mention the memory overhead or time overhead during training. As the number of tasks increases, is the memory overhead completely proportional to the number of tasks? If so, how to address such significant memory overhead? If possible, please provide relevant calculation formulas or training methods.

[1] Jing Xu, Jiazheng Li, and Jingzhao Zhang. Scalable model merging with progressive layer-wise
distillation. (arXiv:2502.12706), May 2025. doi: 10.48550/arXiv.2502.12706. URL http:
//arxiv.org/abs/2502.12706. arXiv:2502.12706.

### Questions
My questions are listed with my weaknesses above.

I'm excited to engage with the authors to clear up the aspects I don't fully understand and I'm optimistic that with some iteration this paper can be made stronger.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SAMerging, a framework for data-efficient and label-free model merging. The authors derive a flatness-aware PAC-Bayes generalization bound that connects model sharpness with cross-task heterogeneity, providing theoretical insight into when merging succeeds. They further reinterpret coefficient learning as multi-teacher knowledge distillation, minimizing the KL divergence between the merged model and its experts while incorporating Sharpness-Aware Minimization (SAM) for better generalization. Extensive experiments on TA-8, TALL-14/20, and GLUE benchmarks show consistent competitive results.

### Strengths
1 SAMerging achieves consistent performance gains over both data-free and data-dependent baselines across multiple benchmarks.
2 The paper has a good organizational structure.

### Weaknesses
1 While the PAC-Bayes and SAM integration are new, the overall combination of KD + SAM resembles existing fine-tuning or merging extensions (e.g., AdaMerging + SAM). The contribution feels evolutionary rather than fundamentally new, since the method is essentially “AdaMerging + SAM + KD reformulation” without deeper empirical diversity.

2 The paper devotes extensive space to mathematical derivations (PAC-Bayes, NTK linearization, and multiple lemmas), but the empirical link between theory and practice is unclear. There is no ablation or visualization showing how the “flatness” or “heterogeneity” terms actually correlate with the final performance, making the theoretical results appear decorative rather than explanatory.

3 The experimental validation is confined to relatively standard benchmarks (TA-8, TALL-14/20, GLUE) using image classification and text classification tasks. This setting does not reflect the diversity or difficulty of modern model-merging scenarios. In particular,
all tasks share similar architectures and backbone initializations (CLIP or GPT-2), there are no tests on heterogeneous architectures, domain shifts, or large-scale multimodal models.

4 Several recent and competitive merging methods (e.g., Twin-Merging, PCB-Merging, 2024–2025) are missing, weakening the empirical thoroughness of the study.

5 Figure 2(a) is redundant, as its information is already presented in Table 1.

### Questions
Q1: Which previous does the experimental setup (e.g., Table 1) in this paper follow?

### Soundness
2

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
2

### Summary
This paper derives a flatness-aware PAC-Bayes generalization bound that provides theoretical guidance for the design of model merging methods. The authors further propose SAMerging, which improves model merging through multi-teacher knowledge distillation on a small, unlabeled dataset. Theoretically, the paper proves that SAMerging tightens an upper bound on the merged model’s excess risk.

### Strengths
1. This paper theoretically identifies the key factors that influence model merging performance, providing guidance for the design of new merging methods.

2. The proposed method is also supported by solid theoretical analysis, which enhances the rigor and credibility of the approach.

3. The experimental results demonstrate that the proposed method achieves promising performance across various benchmarks.

### Weaknesses
1. **Unclear connection between the proposed method and the core Theorem 2.** The method introduced in Section 3.1 appears to have a weak connection with the main theoretical contribution presented in Theorem 2. In other words, it is unclear how the core theoretical result in Theorem 2 technically guides the design of the proposed SAMerging method in concrete technical details. Besides, Section 3.1 and Section 3.2 appear to follow two different theoretical frameworks, with limited connection between them.

2. **Reliance on training data.** The proposed approach depends on access to training data, which may limit its practical applicability, especially given the existence of several data-free model merging methods.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
3
