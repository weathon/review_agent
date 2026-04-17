# Leveraging Label Dependencies for Calibration in Multi-Label Classification through Proper Scoring Rule

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Modern Deep Neural Networks (DNNs) trained by using cross entropy for binary or multi-class classification are known to produce poorly calibrated probability estimates. While various calibration methods have been proposed, only a few addresses the challenge of calibrating Multi-Label Classification (MLC) tasks. Multi-label classification is essential in real-world applications, as most objects or instances naturally belong to multiple categories, and the associated labels often exhibit strong interdependencies. A key difficulty in calibrating MLC models lies in effectively considering the information of label interdependencies. Existing methods that attempt to model the  label interdependencies often lack rigorous statistical justification or they consider the labels are independent or lacks being strictly proper - a property which induces calibrated predicted probabilities upon minimization. In this work, we introduce a novel loss function, \emph{Correlated Multi-Label Loss (CMLL)}, that explicitly captures label interdependencies while satisfying the properties of a strictly proper loss. Our method leverages pairwise label correlations to incorporate dependency information into the training process and is proven to be Fisher consistent. Extensive experiments on three publicly available benchmark multi-label datasets demonstrate the effectiveness of our approach. Our proposed method significantly reduces calibration error while maintaining state-of-the-art classification accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Correlated Multi-Label Loss (CMLL), a novel loss function designed to improve the calibration of deep neural networks in multi-label classification (MLC) tasks. Unlike conventional losses such as Binary Cross-Entropy, which assume label independence, CMLL explicitly models pairwise label dependencies while maintaining the property of being strictly proper, ensuring reliable posterior probability estimates. The authors provide theoretical guarantees, proving that CMLL is both Fisher consistent and $\ell_2$-Lipschitz continuous, and they derive a generalization bound that scales linearly with the number of labels. Extensive experiments on benchmark datasets (PASCAL VOC 2012, MS-COCO, and WIDER-A) demonstrate that CMLL significantly reduces calibration errors without compromising classification accuracy, establishing it as an effective and theoretically grounded approach for trustworthy multi-label learning.

### Strengths
1. The proposed Correlated Multi-Label Loss (CMLL) innovatively combines pairwise label dependency modeling with the property of strict properness, bridging a clear gap between calibration theory and multi-label learning.

2. The paper provides formal proofs showing that CMLL is strictly proper, Fisher consistent, and $\ell_2$-Lipschitz continuous, and it derives a generalization bound with interpretable dependence on the number of labels.

### Weaknesses
1. The notation in this paper could be made clearer. For example, when describing $\boldsymbol{h}(\mathcal{X})$ and $Y$, it would be helpful to explicitly clarify what their rows and columns represent;

2. Although Lemma 1 seems intended to express the difference between two labels, based on my understanding of the notation, the computation of $\tau$ appears to measure the discrepancy between two instances rather than between a pair of labels.

3. In Assumption 1, it seems that the loss function $L$ corresponds to the proposed CMLL loss. If this is the case, it would be helpful to specify the valid ranges of $M$ and $B$. In particular, under certain extreme cases, the $\log$ term in Equation (3) might lead to an unbounded $M$, which could invalidate Assumption 1 and consequently affect the soundness of Theorem 2.

### Questions
Please carefully check the Weaknesses.

Minor comment:

1. In Lemma 1, could the authors clarify whether the dataset $\mathcal{D}$ is defined as D = \{(\varepsilon_i, Y_i)\}_{i=1}^n?

### Soundness
2

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
5

### Summary
This paper introduces a new loss function, Correlated Multi-Label Loss (CMLL), for improving confidence calibration in multi-label classification. The authors argue that existing methods either assume label independence or lack theoretical guarantees such as strict propriety. CMLL is designed to explicitly model pairwise label correlations while maintaining the property of being a Strictly Proper Scoring Rule (PSR). The paper also provides theoretical justification (Fisher consistency and generalization analysis) and experimental validation on three standard datasets (PASCAL VOC, MS-COCO, and WIDER-A), showing improvements in calibration metrics while maintaining comparable accuracy.

### Strengths
1. The work tackles the important and underexplored problem of multi-label confidence calibration, which is highly relevant for real-world applications where label dependencies are common (e.g., medical imaging, scene recognition).

2. Experiments across multiple datasets and architectures (ResNet-50 and ViT-B/32) demonstrate consistent improvement in calibration metrics such as ACE and MCE.

### Weaknesses
1. The proposed method’s originality is somewhat incremental compared to recent works such as [Chen et al., TIP 2024][Peng et al., CVPR 2024; TPAMI 2025]. These papers also explore correlation-based or dependency-aware regularization for calibration. The current submission does not clearly articulate how CMLL is fundamentally different or superior in modeling dependencies beyond reformulating correlation alignment as a proper scoring rule.

2. The experiments do not include a comparison with [Chen et al., TIP 2024], which introduced both a multi-label calibration method and comprehensive evaluation metrics for multi-label confidence calibration.

3. Only simple baselines (BCE, Focal Loss, TWL, LDACE-CCL) are used. Missing comparisons with state-of-the-art multi-label calibration methods significantly weakens the empirical validation.

4. The paper evaluates only on relatively standard architectures (ResNet-50 and ViT-B/32) and basic multi-label baselines. Recent multi-label recognition backbones (e.g., ASL [Ridnik et al., ICCV 2021], ML-Decoder, or transformer-based decoders) are not included, making it difficult to assess general applicability.

[Chen et al., TIP 2024] Dynamic Correlation Learning and Regularization for Multi-Label Confidence Calibration.

[Peng et al., CVPR 2024; TPAMI 2025] Perception/Semantic Aware Regularization for Sequential Confidence Calibration.

### Questions
See Weakness.

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
4

### Summary
This paper tackles the problem of poor confidence calibration in modern deep neural networks for multi-label classification tasks. This is a crucial issue, as miscalibrated models are unreliable in safety-critical applications and often involve multiple labels per instance. The authors identify a key gap in current methods: existing "proper scoring rules" (PSR) losses like BCE ignore label dependencies, and other losses that model the dependencies like focal loss are not PSR. To fix this, the paper introduces a new loss function called Correlated Multi-Label Loss (CMLL), including a regularization term penalizes the difference between the model's predicted label correlations and the ground-truth label correlations. The authors provide a key theoretical proof that their combined CMLL loss is still a PSR loss.

### Strengths
1. The paper is very well-motivated. Calibration is a known, hard problem in MLC. This work is well-positioned. It directly addresses the limitations of recent key papers.

2. The main claim isn't just based on intuition. 

3. The experimental results are good.

### Weaknesses
The paper's entire theoretical foundation rests on the claim that CMLL is a PSR. A PSR must be uniquely minimized when the prediction $\hat{\rho}$ equals the true probability $\rho$. However, CMLL is a weighted trade-off between the BCE loss (a PSR) and a new correlation term. It is possible that a model will sacrifice perfect calibration (increasing the BCE loss) to better match the in-batch label correlations (decreasing the new term). This means the minimum of the CMLL loss is no longer guaranteed to be at the point of perfect calibration in particle. Since the proposed correlation term is calculated in-batch, the loss for any single sample dependent on the other samples present in its batch. This formulation contradicts the standard definition of a PSR, which is based on the expectation $E_{x,y}[L(h(x), y)]$. Also it makes the training gradient highly sensitive to batch composition and sampling noise. 

The $\lambda$ is the most important part of the proposed method, as it controls the trade-off between calibration and the regularization term. The paper simply states $\lambda=1$ is used for all experiments with no justification, ablation study, or sensitivity analysis.

### Questions
Please refer to above

### Soundness
3

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
The calibration of multi-label deep neural networks is considered. The paper introduces the Correlated Multi-Label Loss (CMLL), a novel loss function designed to improve calibration in Multi-Label Classification (MLC) tasks by explicitly capturing label interdependencies. CMLL is proven to be a strictly proper loss and to be Fisher consistent. The loss incorporates dependency information by minimizing the absolute difference between the empirical correlation of the predicted scores for label pairs and the correlation of their ground truths. Extensive experiments on three benchmark datasets, PASCAL VOC, MS-COCO, and WIDER-A, demonstrate that CMLL reduces calibration error while maintaining classification accuracy compared to some other popular loss functions.

### Strengths
1. The work proposed the Correlated Multi-Label Loss (CMLL) and established a generalization bound for it.  
2. Empirical evaluation of CMLL in terms of the accuracy and calibration on multiple real-world multi-label datasets.

### Weaknesses
More experiments are necessary.  
   -- a. Only one metric (hamming loss) is used for evaluating the accuracy of multi-label classification. In the modern multi-label learning literature, more metrics such as mAP, OF1, CF1 are widely employed.  
   -- b. Lack of comparison against state-of-the-art baselines. Comparison to SOTA multi-label losses, such as Ridnik et al., 2021; and Cheng & Vasconcelos (2024), is necessary for validating the superiority of the proposed CMLL loss against losses that do not take into consideration label dependency.

### Questions
There should be a space between text and Parenthesis (.

### Soundness
3

### Presentation
3

### Contribution
2
