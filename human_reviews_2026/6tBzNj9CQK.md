# Autonomous Source Knowledge Selection in Multi-Domain Adaptation

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4, 2

## Abstract
Unsupervised multi-domain adaptation plays a key role in transfer learning by leveraging acquired rich source information from multiple source domains to solve target task from an unlabeled target domain. However, multiple source domains often contain much redundant or unrelated information which can harm transfer performance, especially when in massive-source domain settings. It is urgent to develop effective strategies for identifying and selecting the most transferable knowledge from massive source domains to address the target task. In this paper, we propose a multi-domain adaptation method named Autonomous Source Knowledge Selection (AutoS) to autonomosly select source training samples and models, enabling the prediction of target task using more relevant and transferable source information. The proposed method employs a density-driven selection strategy to choose source samples during training and to determine which source models should contribute to target prediction. Simulteneously, a pseudo-label enhancement module built on a pre-trained multimodal modal is employed to mitigate target label noise and improve self-supervision. Experiments on real-world datasets indicate the superiority of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an unsupervised multi-domain adaptation method named Autonomous Source Knowledge Selection (AutoS), which aims to address the challenge of redundant or irrelevant information from massive source domains. The method autonomously selects relevant source training samples and models through a density-driven selection strategy, while employing a pseudo-label enhancement module based on a pre-trained multimodal model to reduce target label noise and enhance self-supervision. Experiments are conducted on the Office31, OfficeHome, DomainNet126 and DomainNet datasets.

### Strengths
1. The research motivation of this paper is clearly articulated, and the framework diagram is well-designed.

2. The paper includes comprehensive ablation studies and visualizations, which strengthen the presented work.

### Weaknesses
1. The performance advantage of the proposed method over the state-of-the-art ones is marginal. On the four benchmark datasets, the highest improvement over the state-of-the-art is only 0.4%, and the method even underperforms some baselines in certain cases. This raises serious doubts about the method's practical effectiveness and its overall contribution.

2. The ablation study presented in Table 3 fails to provide compelling evidence for the effectiveness of the individual components. The performance gain of the full model over the base model is merely 0.3%. Such a minuscule improvement makes it difficult to convincingly argue that the added modules contribute significantly to the overall framework.

3. The methodology section suffers from a critical lack of clarity. Numerous symbols used in the mathematical formulations are not defined or explained, which severely hinders the reader's ability to understand the technical details and reproduce the work. A thorough review and supplementation of all mathematical notation throughout the paper is strongly recommended.

### Questions
Please refer to the weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an autonomous source knowledge selection framework for multi-source domain adaptation. The method progressively filters out unhelpful source domains and samples via a density-driven selection strategy, and adapts a target model by fusing selected source models and pseudo-labels from a external CLIP. The overall design reduces noise from redundant sources and enhances cross-domain generalization through cross-modal supervision and prompt tuning.

### Strengths
1. The method is modular and integrates several known techniques (e.g., density estimation, prompt tuning, CLIP supervision).
2. This paper shows some promising empirical results on standard benchmarks.

### Weaknesses
1. The method assumes the availability of domain labels for all source domains. In practice, especially with large-scale web or industrial data, domain boundaries are often unknown or ambiguous. The entire framework depends on identifying and discarding full source domains, which is risky.
2. The problem of transferring from multiple labeled domains to an unlabeled target is a classic transfer learning setting. However, the method relies on CLIP, itself already mitigates many of the traditional domain shift issues. This creates a mismatch between the paper’s motivation and its solution. (and CLIP prompt tuning has been studied to overcome dataset shift already, It feels somewhat excessive to use prompt tuning here, given that it’s only used for generating better guidance.)
3. Recent works are not compared [1]. PACS is a well-established testbed for domain selection and is notably missing here. ImageNet family can also be considered (ImageNet, ImageNet-C, ImageNet v2, etc)
4. The method reads as a combination of existing ideas (density filtering, prompt tuning, model averaging) without a unifying intuition. There is no concrete observation or empirical motivation driving the design of the multi-step pipeline. Why density? Why federated aggregation vs. others? Why this specific prompt tuning strategy? 
5. Source selection can be risky e.g., wrongly discarding helpful domains or reinforcing spurious alignment. 
6. With the rise of strong foundation models that already exhibit strong zero-shot and transfer capabilities, the need for complex multi-source adaptation pipelines has reduced significantly. In this context, it would be helpful for the authors to clarify the motivation and relevance of their setting in modern real-world scenarios, especially where domain boundaries are often unknown and data is unstructured or unlabeled.

[1] Training multi-source domain adaptation network by mutual information estimation and minimization

### Questions
See weaknesses.

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
3

### Summary
- Task: Unsupervised multi-domain adaptation
- Goal: To solve a target task in an unlabeled target domain by leveraging source information obtained from multiple source domain.
- The paper proposes a selection strategy to determine which source domains are most useful for learning in unsupervised multi-domain adaptation.
- To mitigate label noise, the authors introduce a multi-modal model along with a novel prompt tuning loss designed to train it.
- The effectiveness of the two proposed contributions is demonstrated through experiments, showing consistent performance improvements and validating the proposed framework.

### Strengths
- Recognizing that not all source information is equally useful in unsupervised multi-domain adaptation, the authors propose a selection strategy.
   - Each source domain is assigned a weight according to its relevance, enabling the model to selectively utilize information that is more beneficial for target domain adaptation.
- In parallel with employing a multi-modal model, the authors propose a loss function designed to enhance the alignment of the target adaptation model.

### Weaknesses
- Lack of analysis
   - While  the assumption that not all source information is useful is understandable, the paper would be more convincing if experimental evidence supporting this claim were provided.
   - The proposed loss function $\mathcal{L}_{\mathrm{ex}}$
 appears to designed for the prompt tuning objective of a foundation model, but its precise mathematical definition and formulation should be explicitly stated.
   - Additional experiments are needed to justify the validity of the hyperparameter settings used in the proposed method.
   - The interpretation and significance of the proposed components are generally underexplained and lack in-depth analysis.
- Representation quality
   - In Eq. (1), the position of $(x_k^{s}, \hat{y}_k^{s}) \in \mathcal{D}_k^{S}$ within the argmin expression appears to be incorrect.
   - Although the meaning of 𝜇 in Eq. (1) can be inferred, it should be explicitly described in the text to improve clarity.
   - The dataset used in Fig. 3 should be clearly specified.
   - The overall quality of the figures in experiments section is relatively low and should be improved.
      - The font size in Fig. 4 is too small for readability.

### Questions
- Assumption and experimental evidence
   - The assumption that not all source information is useful in unsupervised multi-domain adaptation is understandable; however, the paper would be stronger if this assumption were supported by empirical evidence through experiments.
- Clarification and validation of the proposed loss
   - Among the proposed loss functions, $\mathcal{L}_{\mathrm{ex}}$  appears to represent a learning objective for prompt tuning within a foundation model. Its precise definition and mathematical formulation should be clearly stated.
- Hyperparameter justification
   - Additional experiments are required to justify the chosen hyperparameter settings and demonstrate their robustness across different configurations.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a two‑stage source knowledge selection method for multi‑source unsupervised domain adaptation (MS‑UDA). The proposal can: 1. autonomously select and collect transferable samples via a density‑controlled, target‑driven criterion, and 2. adapt to the target with prompt‑tuning. Experiments on several cross-domain vision datasets show competitive accuracy among several recent baselines, with lower running time and GPU memory. Ablations suggest that different components of the proposal contribute.

### Strengths
The paper targets a practical scalability issue in MS‑UDA, i.e., addresses a valid and important research gap that is significantly important.

 Design of selection signal and two‑stage training with prompt‑only fine-tuning can conceptually reduce the overall complexity.

Ablations indicate that each component of the proposal contributes to the performance.

### Weaknesses
The paper’s novelty might be the specific density-aware keep/drop rule coupled with federated aggregation. Other core components, such as source/domain selection or weighting, are better explored.

The absolute performance gains compared with other baselines appear to be marginal.

Potential dependence on a foundation model for pseudo-label production. Gains may partially stem from CLIP priors rather than the selection scheme, especially when referring to Tables 1 and 2; performance may degrade in domains where CLIP is weak. 

Table 5 lacks units for running time and GPU memory, while only empirical results are given, lacking theoretical evidence. This makes the assessment of the actual trade-off difficult.

The quality of visualization (aspect ratio, font size, etc.) needs improvement starting from Fig. 3; similarly, there are some writing errors, e.g., multi-modal “modal” in the abstract.

### Questions
Elaborate further to potentially address the CLIP’s related worries, as in the Weakness section.

To help Table 5 make more sense, please specify units and provide more complete information about the computational platform. A few more benchmarks are desired for this part. Please also elaborate on theoretical complexity where applicable.

 Regarding the experimental configuration, please at least add the related information of the federated learning setup. Moreover, please specify if an ideal federated learning scenario is assumed.

On Office-Home, the source-free variant can slightly beat the “default” setting. With source information, such results appear to be counterintuitive. Please elaborate on this.

Check the manuscript carefully to avoid small writing errors and update figures for easier reading. Other minor suggestions include enlarging Fig.1’s font size and updating Fig.2’s upper part to directly contain the information of federated learning.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new method for selecting source domains which are relevant to a specific target domain under a multi-source domain adaptation setting. Specifically, they introduce a closeness measure based on how source domains match a defined density distribution to target samples and re-weight the source domain accordingly. They also introduce a self-supervised learning strategy that uses a pre-trained foundation model to bootstrap the learning using the selected source domains. Experiments show positive improvements of this method over prior methods.

### Strengths
- The paper tackles an important problem of selecting relevant source domains for a target domain. 

- The idea of using pre-trained models to sub-select the source domains instead of re-training the source domains separately is very interesting.

### Weaknesses
- The paper lacks any sound intuition of technical depth as to why the proposed framework should work. The sequence of steps are just presented without adequate explanation as to what each of those are supposed to achieve, and why no other good alternatives exist. For example, L216-240 in Sec 3.3 has many successive equations but none of the notations are explained. $\Gamma, \pi, \sigma, \lambda$ are all used but none of them are grounded in previous notation or explained what they mean, without which they just seem like runaway arguments. 

- In addition to above, the paper does not adequately support the hypothesis on why the gating function in Eq 8 should work. At the very least, it should be explained what does $\frac{1}{K}-\sigma$ means, and why is it a good threshold. while theoretical justification is out of the scope, intuition behind this should atleast be explained. 

- Use of foundational model in Sec 3.4 is not clear, and Eq 13 is not supported well in the rest of the arguments.

### Questions
- The paper uses a VLM foundational model, so it should be specified what other compared models also use the foundation model and what do not. In addition, the zero-shot accuracy using the foundation model on the target domain also has to be presented as a comparison. 

- Is the source domain selection done once before training or is it done continuously during training? As the features are learnt, it might happen that few source domains which were removed might again become relevant, how do the authors address this case?

- are there any examples of what source domains are considered "related" in datasets like say, DomainNet? Also, if source Domain A is considered to be "relevant" for target Domain B, then will this also be the case when source is B and target is A? are the operations commutative?

### Soundness
1

### Presentation
2

### Contribution
3
