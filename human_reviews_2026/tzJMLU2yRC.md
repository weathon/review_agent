# Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Positive-Unlabeled (PU) learning aims to train a binary classifier (positive vs. negative) where only limited positive data and abundant unlabeled data are available. While widely applicable, state-of-the-art PU learning methods substantially underperform their supervised counterparts on complex datasets, especially without auxiliary negatives or pre-estimated parameters (e.g., a 14.26% gap on CIFAR-100 dataset). We identify the primary bottleneck as the challenge of learning discriminative representations under unreliable supervision. To tackle this challenge, we propose NcPU, a non-contrastive PU learning framework that requires no auxiliary information. NcPU combines a noisy-pair robust supervised non-contrastive loss (NoiSNCL), which aligns intra-class representations despite unreliable supervision, with a phantom label disambiguation (PLD) scheme that supplies conservative negative supervision via regret-based label updates. Theoretically, NoiSNCL and PLD can iteratively benefit each other from the perspective of the Expectation-Maximization framework. Empirically, extensive experiments demonstrate that: (1) NoiSNCL enables simple PU methods to achieve competitive performance; and (2) NcPU achieves substantial improvements over state-of-the-art PU methods across diverse datasets, including challenging datasets on post-disaster building damage mapping, highlighting its promise for real-world applications. Code: https://github.com/Hengwei-Zhao96/NcPU.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses Positive-Unlabeled (PU) learning on complex datasets, where existing methods struggle to learn discriminative representations under unreliable supervision. The authors propose NcPU, which combines a noisy-pair robust non-contrastive loss (NoiSNCL) with a phantom label disambiguation (PLD) scheme for conservative negative supervision. The method is theoretically analyzed under the EM framework and empirically validated on complex datasets, showing promising results.

### Strengths
1. The t-SNE visualization clearly demonstrates that the proposed method enables the model to learn more discriminative and meaningful features than previous approaches.
2. The authors provide a theoretical analysis to show the effectiveness of their proposed method based on EM framework. 
3. Experimental results show that the proposed method surpasses previous methods by a considerable margin.

### Weaknesses
1. The novelty of the proposed method is relatively limited. Specifically, it is constructed by incorporating existing techniques such as BYOL, class prototypes, and SAT. The main contribution of the method lies in the introduction of the modified loss term, $\\tilde{\\mathcal{L}}$. 
2. The proposed method requires substantially longer training time compared to previous approaches. Specifically, the authors train for 1500 epochs to achieve the best performance, whereas competing methods, such as WSC, are trained for only 250 epochs in their original paper, requiring roughly one-sixth of the training time.

### Questions
1. As shown in Equation (7), the authors adopt a square-root loss in the proposed NoiSNCL objective. Consequently, the gradient does not vanish even when the two features of $x_i$ and $x_j$ are identical. Specifically, when $\\tilde{q}_i^\\top \\tilde{q}_j = 1$, the gradient magnitude becomes $\\frac{2}{\\| q_i \\|_2^2}$. Would this non-vanishing gradient potentially lead to overfitting or training instability?

### Soundness
2

### Presentation
2

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
This paper identifies an insight that noisy pairs tend to dominate the representation learning process, as their gradient magnitudes overwhelm those from the clean pairs. To address this, it proposes a PU learning framework called NcPU. The method introduces two key components: 1) a noisy-pair robust supervised non-contrastive loss (NoiSNCL) that aligns intra-class representations while tolerating noisy pairs through gradient analysis, and 2) a phantom label disambiguation (PLD) module that refines supervision through regret-based label updating. Experimental results demonstrate the framework's effectiveness, with additional results showing its applicability in post-disaster building damage mapping tasks.

### Strengths
- This paper provides an insight by identifying and formally analyzing the "noisy-pair gradient dominance" problem in traditional contrastive learning. 
- Building on this insight, the proposed method is well-designed. The experimental evaluation is comprehensive, demonstrating state-of-the-art performance across multiple benchmark datasets.
- The paper is generally well-written, explaining the proposed method, with detailed descriptions.

### Weaknesses
- The dual-network architecture inevitably increases training cost compared to simpler PU methods. A quantitative comparison of computational overhead would help practitioners evaluate the trade-offs.
- While performance on standard benchmarks is strong, validation under more challenging conditions (e.g., extreme class imbalance, very limited positive samples) would better demonstrate the method's robustness.

### Questions
Please refer to the weaknesses.

### Soundness
3

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
4

### Summary
This paper introduces NcPU, a non-contrastive PU learning framework that addresses the representation separability in PU learning. The proposed method integrates a noisy-pair robust non-contrastive loss (NoiSNCL) and a phantom label disambiguation (PLD) module. NoiSNCL enhances intra-class representation alignment under noisy supervision, while PLD refines pseudo labels through prototype-based, regret-aware updates. Theoretically, the framework is grounded in an EM-based interpretation, and empirically, NcPU achieves substantial gains across multiple datasets, even surpassing supervised baselines.

### Strengths
1. The paper clearly identifies the representation separability issue as the bottleneck in PU learning, and the motivation is well justified and convincing.
2. The proposed noisy-pair robust loss effectively mitigates the influence of false labels, a common problem in PU learning under weak supervision.
3. The proposed method achieves significant and consistent improvements, even outperforming supervised models on several benchmarks.
4. The paper is well-organized and clearly written, making the technical ideas easy to follow.

### Weaknesses
1. The loss function  $\tilde{\mathcal{L}}_r = 2\sqrt{1 - \langle q, k \rangle}$may cause numerical instability when $\langle q, k \rangle \approx 0$, since the gradient involves a term  $\frac{1}{\sqrt{1 - \langle q, k \rangle}} \to \infty$. Although the composed gradient may remain finite, the authors should discuss the numerical stability and possible mitigation strategies.  

2. The idea of introducing robust non-contrastive alignment has conceptual overlap with prior work such as the CoTAP loss proposed in “Semantic Concentration for Self-Supervised Dense Representations Learning” (TPAMI 2025).  
   While the problem settings differ, it would strengthen the contribution to discuss distinctions or novel theoretical insights explicitly.  

3. A naïve self-supervised baseline (e.g., BYOL or DINO pretrained representations combined with standard PU learning) is missing, which would help isolate the contribution of NoiSNCL and PLD from pretrained feature quality.  

4. The paper still uses BYOL as its base self-supervised method, which has been surpassed by recent approaches like iBOT (ICLR 2022) and DINO v1/v2.  Future work could benefit from integrating NcPU with stronger backbone frameworks to further validate its robustness.  

5. The proposed framework involves multi-view alignment and prototype-based refinement, which may increase training time and computational cost.  A runtime comparison with existing PU learning or non-contrastive baselines would clarify efficiency and practical feasibility.

Overall, this paper is worthy of acceptance. Once the above issues are adequately addressed, I would be inclined to raise my score.

### Questions
Please see the weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3
