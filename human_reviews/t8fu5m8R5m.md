# Adversarially Robust Anomaly Detection through Spurious Negative Pair Mitigation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Despite significant progress in Anomaly Detection (AD), the robustness of existing detection methods against adversarial attacks remains a challenge, compromising their reliability in critical real-world applications such as autonomous driving. This issue primarily arises from the AD setup, which assumes that training data is limited to a group of unlabeled normal samples, making the detectors vulnerable to adversarial anomaly samples during testing. Additionally, implementing adversarial training as a safeguard encounters difficulties, such as formulating an effective objective function without access to labels. An ideal objective function for adversarial training in AD should promote strong perturbations both within and between the normal and anomaly groups to maximize margin between normal and anomaly distribution. To address these issues, we first propose crafting a pseudo-anomaly group derived from normal group samples. Then, we demonstrate that adversarial training with contrastive loss could serve as an ideal objective function, as it creates both inter- and intra-group perturbations. However, we notice that spurious negative pairs compromise the conventional contrastive loss for achieving robust AD. Spurious negative pairs are those that should be mapped closely but are erroneously separated. These pairs introduce noise and misguide the direction of inter-group adversarial perturbations. To overcome the effect of spurious negative pairs, we define opposite pairs and adversarially pull them apart to strengthen inter-group perturbations. Experimental results demonstrate our superior performance in both clean and adversarial scenarios, with a 26.1% improvement in robust detection across various challenging benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel approach for robust Anomaly Detection (AD) under adversarial attacks by generating pseudo-anomaly groups from normal group samples. It demonstrates that adversarial training with contrastive loss could serve as an ideal objective function, as it creates both inter- and intra-group perturbations. It defines opposite pairs and adversarially pulls them apart to strengthen inter-group perturbations to overcome the effect of spurious negative pairs. Experimental results demonstrate superior performance of the proposed method in both clean and adversarial scenarios, with a 26.1% improvement in robust detection across various challenging benchmark datasets.

### Strengths
1. It proposes COBRA (anomaly-aware COntrastive-Based approach for Robust AD), a novel method that mitigates the effect of spurious negative pairs to learn effective perturbations.

2. It employs numerous strong adversarial attacks for robustness evaluation, including PGD-1000, AutoAttack, and Adaptive AutoAttack.

3. The experiments span various datasets, including large and real-world datasets such as Autonomous Driving, ImageNet, MVTecAD, and ISIC, demonstrating COBRA’s practical applicability. Additionally, It conducts ablation studies to examine the impact of various COBRA components.

### Weaknesses
1. It might be too expensive to compute the anomaly score since it needs to compute the similarity score over the entire training dataset $D_{train}$. Also, it is unclear how $D_{train}$ will affect the performance. It can provide an empirical analysis of how the size of $D_{train}$ affects both performance and inference time.

2. Although it evaluates several strong attacks including AutoAttack, Adaptive AutoAttack and black-box attacks, it doesn’t evaluate adaptive attacks that are specially designed for the proposed anomaly score. It should follow the guidelines in [1] to design such adaptive attacks. 

3. From Table 4, it seems $A^3$ is weaker than PGD-1000, which is not expected. Some explanations are needed here. For example, it can provide a detailed analysis of why this result occurs, including potential hypotheses about the interaction between $A^3$ and the proposed method.

[1] Tramer, Florian, et al. "On adaptive attacks to adversarial example defenses." Advances in neural information processing systems 33 (2020): 1633-1645.

### Questions
1. Could you analyze the computational complexity of the proposed anomaly score? Will different $D_{train}$ affect the performance? 

2. Could you evaluate adaptive attacks that are specially designed for the proposed anomaly score?

3. Could you explain why $A^3$ is weaker than PGD-1000 in Table 4?

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
The goal of this paper is to enhance the robustness of anomaly detection systems against adversarial attacks. It introduces a new method called COBRA, which addresses the issue of spurious negative pairs and strengthens the distinction between normal and anomalous samples through contrastive learning.

### Strengths
1. The paper presents an approach to enhance the robustness of anomaly detection methods, specifically addressing spurious negative pairs—an issue that traditional contrastive learning methods have not adequately tackled.
2. Figure 1 is well-crafted, providing a clear visualization of the COBRA method.
3. The paper demonstrates extensive experimentation across a variety of datasets, covering both high-resolution and low-resolution data.

### Weaknesses
1. The claim that spurious negative pairs weaken inter-group perturbation in adversarial contrastive learning (Introduction, lines 66-69) requires supporting experimental evidence. The paper lacks clear results or a detailed explanation to substantiate this point.

2. The experiments rely solely on ResNet-18. Evaluating the method with different architectures or feature extractors would provide better insight into the generalizability of the approach.

3. The rationale behind using a k-class classifier for anomaly crafting is unclear. The paper does not sufficiently explain how training with k-class classifiers improves anomaly crafting or what the specific benefits are.

4. There is a lack of experimentation with different epsilon values during adversarial training (in the paper, a fixed epsilon of 2/255 is used for high-resolution data, and 4/255 for low-resolution data). Testing across a range of epsilon values would offer a more comprehensive evaluation of robustness.

5. The balance between clean accuracy and robust accuracy is not well-addressed. For example, in Table 2, the ZARND method shows both high clean accuracy and robust accuracy in the Low Res experiment, but the balance between these is not analyzed in depth.

6. The criteria for hard/mild augmentations are unclear. The paper does not provide a clear definition or justification for the augmentation types used.

7. The distinction between adversarial effects and augmentation effects is insufficiently explained. It’s unclear whether performance improvements are due to adversarial training or augmentations alone. This could be clarified through additional analysis.

8. Table 6 lacks sufficient explanation. Terms such as "L opposite" are not clearly defined, and the application of these terms within the experiments is not fully explained.

### Questions
1.  Could you please provide experimental evidence to support the claim that spurious negative pairs weaken inter-group perturbations in adversarial contrastive learning? (Introduction, lines 66-69)

2. Could you explain why ResNet-18 was chosen as the sole architecture for your experiments? It would be helpful to understand whether your proposed method generalizes well to other architectures or feature extractors, and why this aspect was not explored.

3. What is the rationale behind using a k-class classifier for anomaly crafting? It would be beneficial to know how this approach specifically improves the anomaly crafting process compared to alternative methods.

4. Could you elaborate on why different epsilon values were not explored during adversarial training? Wouldn’t varying the epsilon values offer further insights into the robustness of your method?

5. What criteria did you use to define hard and mild augmentations? It would be helpful if you could provide more details on how these augmentations impact the model’s performance.

6. How do you distinguish the effects of adversarial training from those of augmentations (Hard/Mild) ? Additional experiments clarifying the contributions of each would be greatly appreciated.

7. Could you clarify the terms used in Table 6, such as "L_{Opposite}" ? It would be valuable to understand how these terms were applied in the experiments and their significance.

Additionally, I would appreciate responses to the above weaknesses points.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel method for adversarially robust anomaly detection. The proposed method enhances robustness in anomaly detection by generating pseudo-anomalies through hard transformations and mitigating the effect of spurious negative samples with a tailored contrastive loss. Experiments on various datasets demonstrate that the proposed method achieves superior performance in both clean and adversarial scenarios.

### Strengths
1. This paper identifies limitations in existing anomaly detection methods and challenges in using  contrastive learning for adversarial training, offering solutions to improve robustness in anomaly detection.
2. Through extensive experiments, this paper provides a comprehensive understanding of the strengths and limitations of the proposed method.

### Weaknesses
1. This paper does not follow the ICLR citation format and contains several typos affecting clarity. 
  - Page 4, line 207: In the phrase "where $\Upsilon(x^{b+i})$ is considered as $x^{b}$", it seems that $x^{b}$ should be replaced with $x^{i}$.
  - Page 6, line 291: $x+\delta$ should be used as an input to $\mathcal{L}(\cdot, \cdot)$ instead of $x+\delta^{\ast}$.
  - Page 6, lines 317 and 319: There are typos involving "setting 4".
2. The paper includes unclear or inconsistent content that hinders understanding of the proposed method. 
  - Page 2, line 107: The formulation $x^{\ast}=x_{k}^{\ast}$ includes the undefined symbol $k$.
  - In Figure 1, an augmented sample and its corresponding opposite sample (e.g., $\text{A}\_1$ and $\text{A}\_1^{\Upsilon}$) are used in $\mathcal{L}\_{\text{COBRA}}$. However, in equation (2), the opposite sample of the original sample (e.g., $\text{A}^{\Upsilon}$) is used instead.
  - In Figure 1 and the “Adversarial Training Step” paragraph on page 6, a single adversarial example $x_{adv}$ is generated for the augmented samples $x_i$ and $x_j$. However, the generation process is unclear and needs a step-by-step explanation, especially on how the perturbation is optimized in $\mathcal{L}\_{\text{COBRA}}$.
3. In the EXPERIMENTS section, the analysis of experimental results appears before the experimental setup, evaluation details, and implementation details, which limits understanding. Additionally, the analysis is too brief to fully convey the implications of the experimental results, and no analysis is provided for Figure 3.

### Questions
1. Could the authors address the first and second weaknesses to improve the clarity of the paper?
  - Update the citation format to follow the ICLR style and fix the typos.
  - Define $k$ and resolve the inconsistency regarding the opposite sample in Figure 1 and equation (2).
  - If possible, include pseudocode for generating the adversarial example.
2. In the EXPERIMENTS section, could the authors first present the experimental setup and evaluation details, and implementation details, followed by an analysis of the results? Additionally, could the authors provide a more detailed analysis for each experimental result and include an analysis of Figure 3?
3. On page 7, lines 375-377, it states that replacing adversarial training with clean training improved clean detection performance to 90.7%. Could the authors also report how robust accuracy was affected?
4. For the PGD attack, the number of iterations was set to 1000. Does this actually increase the effectiveness of the adversarial attack? Could the authors compare the robust accuracy with fewer iterations, such as 20 or 100?

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
The paper addresses adversarial robustness in anomaly detection settings, where conventional models are vulnerable due to training limitations that exclude labeled data and anomaly samples. The authors propose a framework that generates pseudo-anomalies (synthetic outliers) through data augmentations and employs a contrastive loss function. They claim a modifications to the contrastive objective mitigates the effect of spurious negative pairs by maximizing the margin between normal and anomaly samples, although, it seems class collision still occurs despite their modification. The authors experiment with well-known benchmark datasets, demonstrating significant improvement in robust detection performance across real-world scenarios under adversarial conditions.

### Strengths
-	The paper addresses the topic of adversarial robustness in anomaly detection, which is a known limitation in semantic and defect-based anomaly detection, and unlabeled OOD detection.
-	The authors provide an extensive set of experiments, comparing their method with well-known approaches in a comprehensive manner.
-	The literature review includes relevant works.
-	The modifications proposed are straightforward and appear to yield strong results, though their simplicity raises some questions about the underlying mechanisms driving the reported improvements.

### Weaknesses
-	The approach to mitigating spurious negative pairs through opposite pairs is not clearly effective in addressing class collision. The use of a single-pairing strategy for forming the positives, inherited from InfoNCE/NT-Xent suggests unresolved issues. 
-	The source of performance gains is unclear. It is not evident whether improvements stem from the redefinition of the loss function, thresholding, opposite pairs, the anomaly crafting strategy, etc. A clearer breakdown of these contributions would clarify the method’s impact.
-	The term "spurious pairs" seems misapplied, as it should be referred to as "class collision" in the context of contrastive learning. Using standard terminology would align the paper with accepted practices in the field. Refer to  Arora et al. (2019, "A Theoretical Analysis of Contrastive Unsupervised Representation Learning").
-	It is difficult for the reader to rapidly identify what is being considered as positive and negative samples within the formulation. A clear understanding of this within the framework is critical for the reader's comprehension.

### Questions
-	Does the source of performance improvements primarily come from the redefinition of the loss function (e.g., the inclusion of opposite pairs) or from the way anomalies are crafted? Ablation studies with a detailed breakdown of each part of the framework, including replacements with other contrastive objectives, could clarify this.
-	Have the authors considered techniques from recent or ongoing research that expand the set of positives for normal samples to include all other normals in the batch? This approach could potentially enhance robustness by promoting compact in-distribution representations at the same time increasing the margin between normal and anomalous samples.
-	On page 1 (line 051), the authors refer to a “novel thresholding approach (19).” It’s unclear why this is considered novel, especially since the cited work is from 2013. The authors should clarify already in the introduction what aspect of their thresholding method is novel or provide more context to justify the claim.
-	The introduction could be improved by breaking down and summarizing the key components of the methodology. Currently, it requires readers to navigate back and forth to piece together the details. The authors should use the introduction to outline their approach comprehensively, giving readers a clear understanding of what will be discussed in detail later in the methodology section.
-	On page 1 (line 57), the authors state that the "optimal objective function should maximize the margin between the distributions of normal and anomaly samples." The rationale behind how maximizing this margin, while also promoting compactness of in-distribution samples, contributes to adversarial robustness is not clearly explained. The authors should provide additional insights on why this separation is relevant and how it enhances the model's robustness against adversarial attacks. 
-	In Section 3.1, the explanation regarding the generation of anomalies and the definition of opposite pairs needs to be clearer. Authors should easily introduce the definitions, for any given normal sample, what is being considered positives, which are negatives, and where/how the opposite pairs come within this settings. A more direct and clearer explanation would greatly enhance the reader's comprehension of the framework.
-	The paper refers to InfoNCE but cite SimCLR paper. Moreover, the implementation details and mathematical definitions follows more NT-Xent. Could the authors clarify whether the loss formulation adheres strictly to InfoNCE or NT-Xent? Additionally, in the context of defining negative pairs, the explanation on Page 4, Line 214, seems to align with NT-Xent, particularly since you describe the negative samples as "the other samples’ augmented view." This typically implies $2(N-1)$ negative samples per anchor, following NT-Xent. Is this interpretation correct, or does your formulation differ in the way negatives are constructed within the batch?
-	Would the authors consider reformulating Equations (1) and (2) to follow the standard mathematical notation used in InfoNCE or NT-Xent? The current two-summation format may be confusing, especially since $P(x_i)$ is a singleton. Using a more familiar notation could improve clarity and make it easier for readers to understand the loss function.

### Soundness
3

### Presentation
3

### Contribution
3
