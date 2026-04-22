# QUEST: A robust attention formulation using query-modulated spherical attention

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6, 6

## Abstract
The Transformer model architecture has become one of the most widely used in deep learning and the attention mechanism is at its core. The standard attention formulation uses a softmax operation applied to a scaled dot product between query and key vectors. We explore the role played by norms of the queries and keys, which can cause training instabilities when they arbitrarily increase. We demonstrate how this can happen even in simple Transformer models, in the presence of easy-to-learn spurious patterns in the data. We propose a new attention formulation, QUEry-modulated Spherical aTtention (QUEST), that constrains the keys to a hyperspherical latent space, while still allowing individual tokens to flexibly control the sharpness of the attention distribution. QUEST can be easily used as a drop-in replacement for standard attention. We focus on vision applications while also exploring other domains to highlight the method's generality. We show that (1) QUEST trains without instabilities and (2) produces models with improved performance (3) that are robust to data corruptions and adversarial attacks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper analyzes the role of the norm of queries and keys in the attention module, proposes an alternative variant to the standard attention mechanism, and demonstrates the effectiveness of the proposed attention variant across multiple domains.

### Strengths
1. The designed attention variant serves as a drop-in replacement for the standard approach, requiring only simple modifications.

2. Experiments were conducted across multiple domains, including image classification, language modeling, Graph Transformers, and time series tasks.

### Weaknesses
see questions.

### Questions
1.**Question regarding Fig. 1**. In the third row of Fig. 1, the standard ViT attends to the bird, yet it still makes an incorrect prediction. I hope the authors can provide a clear explanation to help readers understand this phenomenon.

2.**The baselines in Table 2 appear relatively low**. First, when using DeiT and MAE recipes for supervised training standard ViT-Base/Large on ImageNet, I did not encounter training instability issues. Second, using the DeiT-3 training method (fourth row), the authors report a baseline accuracy of 82.7%, which is lower than the official DeiT-3 result of 83.8% (which I have successfully reproduced). Third, I request the authors to provide the training curves.

3.**Lack of large-scale experiments**. Although the effectiveness of the proposed method has been validated across multiple domains (cf. Tables 2, 5, 10), the model parameters used are relatively small. As a simple replacement for standard attention, I believe more large-scale experiments should be conducted.

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
3

### Summary
This paper presents QUEST, a technique designed to enhance training stability and feature learning in Attention mechanisms. The authors investigate how attention logit magnitudes and the norms of Query (Q) and Key (K) matrices influence training dynamics. They find that applying global normalization to the K matrix can cause the model to prematurely focus on less significant features during early training phases, leading to suboptimal convergence. Additionally, since K's norm influences gradient magnitudes, this approach creates training instability. These findings are supported by prior research and demonstrated through a simplified example.
The proposed solution involves applying L2-normalization exclusively to the K matrices while leaving Q matrices unnormalized. This design prevents K from dominating the training process while preserving Q's ability to modulate attention sharpness on a per-token basis through softmax temperature control. To validate their approach, the authors perform experiments training vision transformer models both with and without QUEST, demonstrating its effectiveness.

### Strengths
1) The paper addresses a significant issue affecting transformers across diverse applications and offers a straightforward solution to enhance their stability and performance. 

2) The paper provides a thorough analysis of the attention mechanism and the function of its components. Claims regarding current limitations of Scaled Dot-Product Attention and proposed alternatives are validated through prior research and a toy example, which reinforce the paper's arguments and empirically demonstrate QUEST's effectiveness in addressing these issues.

3) QUEST's impact is illustrated through visualizations, which support the assertion that it prevents converging to less meaningfull attention weights.

4) The authors evaluate QUEST across multiple transformer architectures and tasks, demonstrating superior performance in nearly all experimental settings.

### Weaknesses
1) The main weakness of this work is that the reported improvements are relatively small, often around 1% or less. This raises some concern, especially since it appears that each experiment was run only once. Given the number of experiments conducted, it would be helpful to include a statistical summary of the results. Could the authors provide some form of statistical analysis of the improvements across tasks? Even a simple aggregate analysis could offer a clearer picture of the overall gains.

2) The paper assumes that the primary issue arises from the normalization of the K vectors, suggesting that Q should remain unnormalized during optimization. However, experiments involving only Q normalization also seem to yield positive effects. In the toy experiment, Q appears less stable than K but still more stable than the baseline, and in some cases even achieves better accuracy, particularly for certain learning rate values, where Q-Norm seems easier to optimize in the QUEST setup. This trend is also reflected in the ablation study, where Q-Norm improves over the baseline and performs comparably to K-Norm. Did the authors further investigate Q-Norm beyond the results presented in the paper?

### Questions
See Weaknesses

### Soundness
3

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
3

### Summary
This paper revisits the instability issues in Transformer training, which arise from uncontrolled growth of query/key norms in standard softmax attention. The authors propose a simple yet effective variant, QUEST, that normalizes only the keys while keeping queries unnormalized. This breaks the mutual amplification between query and key norms, stabilizing training and allowing each query to control its own attention sharpness. The method is a drop-in replacement for standard attention and is evaluated across diverse domains. Experiments show that QUEST improves training stability, robustness to corruptions and adversarial attacks, and sometimes accuracy.

### Strengths
**Well-motivated**: Identifies a concrete cause of Transformer instability (growing norms) and proposes an intuitive fix requiring minimal code change.

**Strong Empirical Results**: Demonstrated consistent gains on multiple benchmarks (ImageNet, ADE20K, WikiText-103, GraphGPS, UEA datasets).

**Robustness Improvements**: Shows better performance under adversarial attacks (FGSM, PGD, AutoAttack) and data corruptions.

**General Applicability**: Works as a drop-in replacement across architectures (ViT, CrossViT, Transformer-XL, Graph Transformers).

**Theoretical & Empirical Insight**: Offers a helpful interpretation of attention norm dynamics and a controlled toy experiment demonstrating the effect of spurious correlations.

### Weaknesses
**Experiments:** Most experiments emphasize vision tasks; evaluations in NLP and other domains are relatively light.

**Novelty:** Normalizing keys is a simple modification; some may view it as an extension of prior QKNorm works rather than a fundamentally new paradigm.

**Ablation:** The comparison between normalizing only queries vs. only keys could be expanded.

**Analysis**:Theoretical analysis is mostly heuristic and lacks rigorous mathematical derivation or verified data support.

### Questions
1. Can the authors provide a more formal theoretical justification or analytical evidence beyond heuristic explanations?

2. In NLP tasks, does QUEST consistently outperform QKNorm or LayerNorm-based stabilizations?

3. More modality/field need to be test. Can QUEST be extended to multi-modal or diffusion Transformer architectures?

4. Previous [1] attributes training instability mainly to attention logit explosion and proposes QKNorm as a normalization-based remedy, how does QUEST theoretically or empirically differ from QKNorm in addressing this issue? Specifically, can the authors clarify whether QUEST’s key-only normalization provides comparable stability at large model scales, and if so, why this asymmetric normalization would outperform the symmetric approach proposed in [1]?

5. In NLP tasks, does QUEST consistently outperform QKNorm or LayerNorm-based stabilizations?


This is an interesting work, and I will raise my score if authors address all my concerns.

[1] Scaling Vision Transformers to 22 Billion Parameters

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
For transformer architecture, in order to solve training instabilities when the norm of queries and keys arbitrarily increase, this paper propose  a new attention  (QUEST). This mechanism map the keys to a hyperspherical latent space and single token are still able to sharpen the attention distribution. Empirical experiments show the effectiveness of the proposed methods.

### Strengths
The paper is well-written and easy to follow. It has clear motivation and it identifies an attention instability that arises from growing norms of queries and keys. Based on that, this paper proposes an elegant solution driven by mathematical intuition and compare with relevant methods.

The proposed method is easy to be implemented.

This paper validate proposed method on broad experiments(vision, language, graph transformers and time-series.) and achieve superior performance. Moreover, in vision area, the explainability and robustness analysis are provided.

### Weaknesses
Training instability is not very proper since this paper does not analyse the optimization process but only after optimization. 

Technical depth is limted since it is conceptual but not theoretical. The analysis is mostly based on observation and training instability is affected by the network Jacobian which is lack in the paper.

The scope is moderate

### Questions
See above

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a novel attention mechanism, QUEST (QUEry-modulated Spherical aTtention) to mitigate instability (attention collapse) during the training. The authors argue that the instabilities may come from spurious correlation, or stealing attention due to norm of a certain key. QUEST attempts to solve this by normalizing the key vectors. This coulb be interpreted between standard attention and QK-Norm attention, which focuses on the cosine similarity rather than the dot product. This paper demonstrates that QUEST stabilizes training and improves the performance through various experiment.

### Strengths
1. This paper clearly reformulates the problem. To calculate attention logit, which shows the relevance between query and key, the two obvious choices are  dot-product vs cosine similarity. The authors suggest a solution in between.

2. The authors present a broad experiments results.

3. Quest shows performance gain in IN-C experiment, which implies that QUEST can aggregate information from broad region.

### Weaknesses
1. toy example is too limite. Though the authors devote large portion of the paper into the toy example, the model does not have query, key projection head. The role of projection head is to extract information from embedding vectors. Analyzing attention without the learnable projection head is not compelling.

2. The theoretical analysis is not clear. The auithors argue that QUEST can  mitigate the trainig instability but does not show this metric with other method.

### Questions
1. Why does query normalization perform worse?
2.  Please clarify the elliptical-quest about scaling.
3. Please provide deeper comparison with QK-norm method.

### Soundness
3

### Presentation
3

### Contribution
3
