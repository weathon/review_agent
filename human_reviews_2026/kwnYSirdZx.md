# Adam Can Mitigate Class Imbalance Without Element-Wise Gradient Normalization

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
Adam has remained a dominant optimization algorithm in deep learning for a decade. Recent studies reveal that Adam mitigates the class imbalance by normalizing element-level gradients to balance gradients across classes. However, this interpretation relies on an assumption that gradients between different classes are fully orthogonal. In this paper, we further investigate the assumption. We observe that inter-class gradient orthogonality can be low, particularly during the initial training stages, yet Adam still mitigates class imbalance. This suggests that Adam may not reduce class imbalance by normalizing element-level gradients. Through the ablation of Adam, we further support that class imbalance can be alleviated without element-wise gradient normalization. This work reveals that, even with inter-class gradient coupling, Adam mitigates class imbalance by normalizing gradients across iterations. During early training, the model primarily fits high-frequency class data; as the loss for these diminishes, it adapts to low-frequency classes. Due to the inter-iteration normalization, the gradient magnitudes for low-frequency classes then approximate the initial high-frequency gradients. This mechanism helps Adam mitigate class imbalance. Consequently, we demonstrate that this mechanism necessitates at least layer-wise gradient normalization across iterations, since most neural networks exhibit layer-level inconsistencies between forward and backward propagation. Finally, we further explore potential limitations in Adam’s ability to address the inconsistencies.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper, demonstrates that Adam can mitigate class imbalance by balancing the magnitudes of gradients across iterations. It argues that the layer-wise dynamics normalization can address a layer-level in-consistency between forward propagation and back-propagation while Adam may not fully address this issue. So to tackle that the authors introduce a scaling factor proportional to the initialized weight magnitudes. In a more general sense the paper does an analysis across Adam with various theoretical and experimental results.

### Strengths
- Very extensive and well designed experiments in different modalities and models (both image and NLP data) with very good comparisons and variations of Adam.
- Very good presentation of the problem, the idea, and the proposal. The paper is very easy to read and digest from the audience.
- Very nice theoretical touches in the methodology of the paper that make the paper look more complete.

### Weaknesses
- I don't really see anything bad with the paper. I like the motivation and the analysis. The only theoretical guarantee stems in Eq. 15 if I am not mistaken right? Is there anything else that can be proven for this work? Like proof of convergence? I looked at the Appendix and couldn't find anything.

### Questions
What about any comparisons with RMSProp? For example in Figure 1 it's unfair to compete with SGD since we know that Adam is already an improvement. But I do see the SGD as a baseline.

In general I really enjoyed reading the paper and everything was very well organized from the abstract to the Appendix. I think this is a solid contribution to ICLR this year.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors investigate why Adam reduces class imbalance problems, challenging the prior hypothesis that relies on orthogonality between gradients of different classes. They perform multiple ablations and demonstrate that Adam's success comes from normalizing gradients across iterations.

### Strengths
- The paper is well-written, and each section of the paper is presented with a clear focus. Their claims are well-supported, and the authors provide detailed intuitions and experiments. 
- The ablations are clearly presented and well-designed. Each experiment shows evidence for / against specific hypotheses. For example, to test the importance of element-wise gradient normalization, the authors compared standard Adam with a modified version (Adam-LDN) which replaces element-wise normalization with layer-wise normalization. Because these two have very similar behaviors, this disproves the hypothesis that the benefits of Adam rely on element-wise normalization. The authors also do a good job of presenting the differences between the different variants and are explicit about what behaviors are shared.   
- Adam-S-LDN has competitive performance to Adam, indicating that the authors were able to replicate the benefits of Adam into the layer-wise dynamics normalization with rescaling to account for imbalanced initialization.

### Weaknesses
- A significant portion of the paper is dedicated to demonstrating that gradients across classes are not orthogonal, and there are convincing experiments which illustrate this point. The author claims that these experiment demonstrate novel results which other papers disagree with. However, this seems like a strawman argument to me; my understanding of prior works is that they believe that Adam succeeds because the gradient norm and Hessian trace have strong correlation, which does not appear to be contradictory. Therefore, this decreases the novelty of this finding.
- The paper focuses on understanding Adam for class imbalance, which is a somewhat narrow field. However, their conclusions of Adam stabilizing optimization dynamics across different iterations should extend outside of this setting.

### Questions
- In Line 57, you claim "Kunstner et al., 2024... relies upon the assumption that the inter-class gradients are fully orthogonal". My understanding of this paper's claims was that Adam outperforms SGD in heavy-tailed class imbalance scenarios because the gradient norm and Hessian trace have high correlations, arguing that this enables Adam's normalization to behave similarly to diagonal preconditioning. Could you elaborate on why this conclusion assumes that the inter-class gradients are orthogonal? 
- Previous works have found that heavy-tailed class imbalance leads to a more significant performance gap between Adam and SGD. Did you do any ablations to understand the extent of the class-imbalance on the performance of Adam vs Adam-S (and other variations)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper examines the factors contributing to the effectiveness of the Adam Optimizer in learning on class-imbalanced datasets. The authors demonstrate that the earlier interpretation of normalization across classes is dependent on the assumption of orthogonal gradients; therefore, the interpretation is not accurate. They show that the reason for the effectiveness is more related to the inter-iteration normalization (in the layer).

### Strengths
The paper is clearly readable and understandable.

### Weaknesses
Soundness: The authors provide plots for the training loss, but don’t give us insights into the generalization. This is problematic because the final model's usage would depend on its convergence, which is not evaluated. Hence, the effect of the proposed scaling is not observable in practice. 

Formalism: The main claim of normalization across classes is only analyzed in the extreme cases of the binary classification setting, as presented in Eqs. 1 and 2, which hinders the validity of the claim. Furthermore, the claims made are not supported by sound mathematical theorems and lemmas; therefore, the validity of the claims cannot be established.

Analysis of Only the Early Phase of Training: It's unclear to me how the early phase of analysis (Fig. 3) is applicable to the final results of the model.

### Questions
The plot for Eq. 11 is based on Cosine Similarity for orthogonality, hence in Fig. 4a, it should be low value for orthogonality. However, the results seem to be counterintuitive. Could the authors please elaborate on that in more detail?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates why the Adam optimizer is able to better deal with class-imbalanced data (compared to SGD).
Previous works suggested that Adam mitigates class imbalance because its element-wise gradient normalization approximates per-class normalization, assuming gradients from different classes are orthogonal.
This paper challenges that assumption and shows empirically that inter-class gradient orthogonality is often low (especially early in training).
To study Adam, the authors introduce a variant of Adam, Adam-LDN, which removes element-wise normalization and instead performs layer-wise dynamics normalization. 

Overall, I find the main argument of the paper to be convoluted. The Adam-LDN optimizer is somewhat related to ADAM but there is no proof of strong argument to explain how similar this algorithm is to ADAM so it's unclear if it is indeed a good surrogate. One part of the paper (section 4.3) also talks about layerwize normalization, but that section seems somewhat disconnected from the class imbalance problem (see questions below).

### Strengths
1. Challenges the gradient orthogonality assumption
2. Experiments: demonstrates results across both language (GPT-2 on WikiText-103) and vision tasks showing consistency of findings.

### Weaknesses
1. Limited novelty: the work primarily offers an interpretation and minor ablations of Adam rather than a substantially new optimizer
2. Lack of theoretical rigor: the analysis is largely heuristic, there is no formal convergence analysis.
3. Positioning vs related work. The proposed layer-wise scaling overlaps conceptually with prior layer-wise trust/ratio ideas and other layer-wise Adam variants (see comments/questions below).

### Questions
- The main argument is section 4.2 that boils down to saying that once high accuracy is achieved on 1 class, say c_1, then the relative contribution of the low-frequency class c_2 progressively dominates the weight updates. However, it seems to me that the same argument applies to normalized gradient descent if I'm not mistaken, so it's not clear to me why this explains the particular property of ADAM to mitigate class imbalance. Can you please comment on this?

- The paper claims: "To harmonize optimization dynamics, we introduce layer-specific scaling factor". However, prior works have introduced similar scaling factors I think, e.g. LAMB introduces a layer-wise trust ratio to scale updates.

- Missing prior work: the paper "Deconstructing What Makes a Good Optimizer for Language Models" by Zhao et al. (2024) proposes a variant called Adalayer which is a layer-wise variant of Adam.

- One part of the paper (section 4.3) talks about layerwize normalization, but that section seems somewhat disconnected from the class imbalance problem. Specifically, the argument there is that scaling a layer's weights by a constant factor leaves the forward output unchanged, but rescales the gradients (inversely), which creates layer-specific imbalances that can slow optimization. This is an interesting observation about optimization dynamics in general, but it seems conceptually distinct from the paper's main question: how Adam mitigates class imbalance across classes. The link to class imbalance is only loosely implied, can you elaborate on what the connection is?

Small typos:
- Equation (13) and (14): lr should be squared

### Soundness
3

### Presentation
3

### Contribution
2
