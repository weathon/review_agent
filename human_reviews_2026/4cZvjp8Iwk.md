# SNAPHARD CONTRAST LEARNING

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6, 6

## Abstract
In recent years, Contrastive Learning (CL) has garnered significant attention due to its efficacy across various domains, spanning from visual and textual modalities. A fundamental aspect of CL is aligning the representations of anchor instances with relevant positive samples while simultaneously separating them from negative ones. Prior studies have extensively explored diverse strategies for generating and sampling contrastive (i.e., positive/negative) pairs. Despite the empirical success, the theoretical understanding of the CL approach remains under-explored, leaving questions such as the rationale behind contrastive-pair sampling and its contributions to the model performance unclear.
This paper addresses this gap by providing a comprehensive theoretical analysis from the angle of optimality conditions and introducing the SnaPhArd Contrast Learning (SPACL). Specifically, SPACL prioritizes hard positive and hard negative samples during constructing contrastive pairs and computing the contrastive loss, rather than treating all samples equally. Experimental results across two downstream tasks demonstrate that SPACL consistently outperforms or competes favorably with state-of-the-art methods, showcasing its robustness and efficacy. A comprehensive ablation study further examines the effectiveness of SPACL's individual components to verify the theoretic findings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a screening-based contrastive learning method to jointly sample hard positive and hard negative samples in contrastive learning for better representation diversity (via positives) and discriminability (via negatives). It also provides a theoretical analysis of sampling contrastive pairs and conditions for representation collapse.

### Strengths
1. Introduces a novel contrastive learning approach that jointly samples hard positives and negatives using a screening mechanism, with theoretical analysis supporting its design.
2. Shows strong empirical performance in 4 image classification benchmarks across and compares its method to supervised, self-supervised, and weakly-supervised baseline methods. Also compares its method to link prediction baselines in 3 benchmarks.
3. Provides a comprehensive ablation to show the effectiveness of the proposed method

### Weaknesses
1. In the supervised contrastive learning setting, the risk of false negatives is minimal since positives and negatives are explicitly defined by labels. However, in the self-supervised setting, false negatives can naturally occur when semantically similar samples are mistakenly treated as negatives. As SPACL’s focus on hard negatives can amplify the detrimental effect of false negatives, a discussion of false negatives would be beneficial.
2. I understand that “relying solely on a single hard sample poses the risk of solution collapse, as indicated by Theorem 2.1”. However, Figure 4 shows that increasing the number of hard positives beyond 4 leads to lower performance. Could the authors explain why increasing the number of hard positives (which are intended to prevent collapse) can lead to worse performance compared to using only one hard sample?

### Questions
1. What does the error bar in Table1, Table 2 indicate? 
2. Please address questions in the weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SnaPhArd Contrastive Learning (SPACL), which focuses on hard positive and negative examples within the contrastive learning (CL) framework. The authors analyzed the optimality and crash risks of CL and propose a method designed to overcome these challenges by using carefully selected contrasting pairs. SPACL demonstrates empirical performance by outperforming existing SOTA methods on multiple downstream tasks, including image classification and knowledge graph link prediction.

### Strengths
- The theoretical analysis provides a unique perspective for contrastive learning.
- Extensive experiments demonstrate the effectiveness of the method in this paper.
- The writing is clear and well-structured.

### Weaknesses
- This paper presents a detailed theoretical analysis; however, its heavy reliance on numerous equations makes the ideas somewhat obscure. The core concepts may be difficult for non-specialist readers in the field to grasp. Adding a few intuitive illustrations might enhance the overall completeness of the paper.

- While the paper demonstrates the efficacy of SPACL, the computational cost compared to other methods is not discussed in depth. Understanding the trade-offs in terms of computational requirements could be valuable, especially for real-world applications.

- I noticed that the experiments in this paper were conducted on classic datasets such as CIFAR and ImageNet. I understand that these are the standard datasets used in previous contrastive learning works. However, I have a more exploratory question: given the rapid and relatively mature development of VLMs (including large models across various domains), can the proposed method in this paper truly provide a valuable tool for the academic or industrial community?

### Questions
This paper verifies the effectiveness of SPACL on image classification and knowledge graph completion tasks. What challenges might SPACL encounter when dealing with other modalities or tasks? What are the advantages and disadvantages of SPACL when applied to different types of data or tasks?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the role of hard sample mining in Contrastive Learning (CL). The authors provide a theoretical analysis of the InfoNCE loss, deriving optimality conditions to motivate the importance of hard samples. Based on this analysis, they propose SnaPhArd Contrast Learning (SPACL), a method that systematically screens for and utilizes hard positive and hard negative samples. The paper reports strong empirical results on image classification and knowledge graph link prediction tasks, showing improvements over a selected set of baseline methods.

### Strengths
- **Clear Theoretical Grounding**: A major strength is the clean and elegant theoretical motivation derived from the InfoNCE loss function. This provides a principled "why" for the algorithm's design, which is commendable.
- **Thorough Ablation Studies**: The paper includes an extensive set of ablation experiments that methodically dissect the SPACL algorithm. This rigorous analysis convincingly demonstrates the utility of each proposed component within its framework.

### Weaknesses
- **Limited Scope of Experimental Comparison**: The evaluation could be strengthened by including a broader set of baselines. Comparisons are notably missing against: (a) the current state-of-the-art paradigm of Masked Image Modeling (e.g., MAE), and (b) influential non-InfoNCE contrastive methods like DINO. This makes it difficult to assess the method's standing in the wider field of self-supervised learning.
- **Lack of Discussion on Complexity**: The proposed method adds several components (e.g., an adversarial generator and a discriminator) that increase complexity. The paper would benefit from a discussion on the trade-off between this added complexity and the resulting performance gains.

### Questions
- For clarity and reproducibility, we suggest moving key experimental details, such as the backbone architecture, from the appendix to the main paper.
- Could you please clarify the computational overhead (e.g., in terms of training time or GPU memory) introduced by the SPACL components, particularly the adversarial generator and discriminator, compared to a simpler baseline like SimCLR?
- It is recommended to provide more details on the usage of LLMs in research ideation/writing.

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
2

### Summary
Contrastive learning has attracted considerable attention in recent years and has demonstrated strong empirical performance across various tasks. To address the issue of potential false negative samples in the construction of feature pairs, this paper conducts a comprehensive theoretical analysis from the perspective of optimality conditions and proposes the SPACL framework. Specifically, the method improves the process of constructing positive and negative pairs by selecting “hard” positive and “hard” negative samples, rather than treating all samples equally. The experimental results show that the proposed approach achieves excellent performance.

### Strengths
S1: This paper provides a relatively thorough theoretical analysis, outlining the optimality conditions and highlighting the beneficial role of hard samples in constructing feature pairs. The proposed geometric analysis offers a novel perspective and is compelling.

S2: At the methodological level, SPACL integrates the selection of distant positive samples with adversarially generated hard negative samples, which effectively enhances the discriminative capability of the model and serves as a valuable complement to existing methods.

S3: The extensive experimental results clearly validate the effectiveness of the proposed method, showing consistent and significant improvements over other methods across various benchmark datasets.

S4: The paper provides a number of illustrative figures, which are helpful for understanding the proposed method.

### Weaknesses
W1: Intuitively, the proposed method could incur some additional computational cost. Thus, providing an analysis of memory usage and throughput would help to more comprehensively evaluate the efficiency of the method.

W2: In the ablation study, the authors state that the performance drop in the “w/o AN” setting indicates that adversarially generated negative samples help sharpen the boundary of the negative region; otherwise, relying solely on batch-based or queue-based negatives would lead to a more blurred boundary. It would be helpful to provide a more fine-grained explanation or additional visualizations to further support this claim.

### Questions
Please refer to the weaknesses section. The main issues are concentrated on the performance analysis and experiments, as well as the need for further explanation or visualization in the ablation studies.

### Soundness
3

### Presentation
3

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
The paper presents a theoretical and empirical investigation into contrastive learning (CL) with a focus on hard positive and negative sample selection. The authors first analyze the optimality and collapse conditions of CL through geometric and gradient-based formulations of the InfoNCE loss. Building upon this theoretical framework, they propose SPACL, a new algorithm that strategically emphasizes “hard” samples - instances that are challenging for the model to distinguish, while filtering out trivial ones.

SPACL introduces two core innovations: 1) Hard Positive Selection: Identifies the most diverse and dissimilar augmented views via a farthest-point iterative process, thereby mitigating representation collapse. 2) Hard Negative Generation and Screening: Incorporates adversarially generated negatives and a relative screening mechanism to retain only the most informative (hard) negative samples.

Empirical results across multiple benchmarks, including CIFAR-10/100, ImageNet(-100), and three knowledge graph datasets (WN9, FB15K-237, FB15K), show consistent improvements over state-of-the-art methods such as SimCLR, MoCo, SupCon, and VarCon. Theoretical derivations are complemented by extensive ablation studies demonstrating the contribution of each module.

### Strengths
1. This paper provides complete analyses of why and how sample hardness influences contrastive optimization, connecting gradient geometry and convex hull dynamics.

2. The dual hard-sample selection (positive and negative) is elegantly motivated and empirically validated.


3. Strong performance across multiple benchmarks and supervision levels, with consistent margins of improvement.


4. Clearly isolate the role of each component (anchor selection, hard positives, adversarial negatives, and screening).


5. Demonstrates applicability in both visual and relational (knowledge graph) contexts.

### Weaknesses
1. The derivations assume idealized settings (e.g., constant auxiliary encoder ggg, unit-sphere normalization) that may not hold in practical large-scale CL systems. The paper could better discuss these approximations.


2. While effective, SPACL introduces additional overhead via adversarial negative generation and iterative positive selection. The paper lacks a quantitative runtime or resource comparison to baselines.


3. Although two distinct domains are tested, the method’s performance in text or multimodal CL remains unexamined.


4. Although λ parameters are ablated, the dependence of SPACL’s stability on these values may warrant further analysis, especially for large-scale datasets.

### Questions
1. Could the authors provide a more explicit analysis of SPACL’s computational and memory complexity relative to MoCo or SupCon? How feasible is adversarial negative generation in large-scale pretraining?


2. Have the authors tested SPACL with transformer-based encoders (e.g., ViT, CLIP) to verify whether the observed benefits generalize beyond convolutional backbones?


3. The hardness definition for positives depends on dissimilarity averaging. Have alternative formulations (e.g., mutual information or gradient-based difficulty) been explored?


4. Since the adversarial component introduces an inner min–max loop, did the authors encounter stability issues during training? If so, how were they mitigated?


5. The conclusion mentions potential applicability to negative-free CL frameworks (e.g., BYOL, SimSiam). Could the authors clarify how the proposed geometric analysis might translate to such setups?

### Soundness
4

### Presentation
3

### Contribution
4
