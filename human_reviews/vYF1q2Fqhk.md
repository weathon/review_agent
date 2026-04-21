# Hierarchical-Latent Generative Models are Robust View Generators for Contrastive Representation Learning

- Avg Score: 4.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 3

## Abstract
A growing area of research is exploiting pre-trained generative models as a data source for contrastive representation learning, generating anchors and the associated positive views through perturbations of the latent codes. In this study, we make significant advances in this field by formalizing the properties of a specific category of generative models, which we term Hierarchical-Latent. We show how the intrinsic properties of these models can successfully be used to create robust views for contrastive learning, outperforming not only previous methods' performance but also surpassing classic approaches trained with genuine real data. The proposed framework is evaluated on different generators and contrastive learning techniques, also investigating the effects of employing a discriminator to filter out low-quality images. Eventually, we test continuous sampling in our experiments, where the generator dynamically samples new synthetic data during contrastive training of the encoder, showing competitive or faster training time with respect to a real-data approach, while allowing a virtually unlimited training set.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors introduce and explore a category of generative models termed Hierarchical-Latent (HL) models. This work demonstrates how the distinctive characteristics of these models, which operate using multiple latent spaces in a hierarchical fashion, can be harnessed to generate robust positive views for contrastive representation learning. By employing HL models as data sources for self-supervised learning and devising specific perturbation strategies for different latent vectors, the authors achieve significant improvements in representation learning compared to previous methods and even surpass the performance of training on real data. Additionally, this study proposes a continuous sampling approach for generating additional training data in real time, revealing its competitive or faster performance compared to traditional data loading techniques. Overall, the paper formalizes the HL model category, highlights its effectiveness in self-supervised learning, and introduces a practical method for augmenting training data.

### Strengths
(1) This paper introduces the new concept of Hierarchical-Latent (HL) models and their application in self-supervised learning. The formalization of HL models as a distinct category within generative models, along with their utilization for data generation, presents a fresh perspective.

(2) The paper meticulously conducts experiments, compares the proposed approach with existing methods, and includes comprehensive ablation studies to validate its efficacy. Furthermore, the introduction of the continuous sampling technique to augment training data is practical.

(3) The paper is clear in presentation. This work effectively describes the concepts, with a logical flow from defining HL models to their application in self-supervised learning. The use of illustrative figures enhances comprehension.

(4) This paper holds significance in the fields of self-supervised learning and generative models. The proposal of continuous sampling as a means to augment training data online addresses a pertinent issue, potentially reducing the gap between synthetic and real data in classifier training.

### Weaknesses
(1) The paper lacks a deeper theoretical analysis of why and how the Hierarchical-Latent models would significantly enhance the paper. Providing theoretical insights into the relationships between hierarchical latent, perturbations, and representation learning could strengthen the paper's contributions.

(2) This work lacks a comprehensive comparison with strong baseline methods. A more extensive set of baseline models and techniques should be considered to provide a more thorough evaluation of the proposed approach.

(3) Visualization of the hierarchy of latent spaces in HL models or diagrams illustrating this hierarchical structure would enhance understanding. 

(4) A more in-depth discussion of the trade-offs and limitations of using generative models for self-supervised learning compared to real data would be valuable.

### Questions
(1) Could the authors provide more theoretical insights into the hierarchical nature of HL models' latent spaces and their implications for representation learning? 

(2) Could the authors expand the baseline comparisons by including more state-of-the-art self-supervised learning methods and possibly discussing how the proposed approach compares to conventional supervised learning using real data?

(3) Would it be possible to include visualization to illustrate the hierarchical structure of latent spaces in HL models? 

(4) Have the authors examined or experimented with the adaptability of HL models and the suggested approach in fields beyond images, such as text or audio data?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper discusses the use of hierarchical-latent generative models as robust view generators for contrastive representation learning. The authors propose a framework that utilizes the properties of hierarchical-latent models to create robust views for contrastive learning, outperforming previous methods and even surpassing approaches trained with real data. The model is novel and the performance is shown to be better than the current baselines.

### Strengths
(1) The problem that the paper aims to solve is significant and the method propose is feasible.
(2) The paper forms the problem in a theoretical way, which is technically sound.
(3) The performance of the proposed method is superior based on the experimental section.

### Weaknesses
(1) In proposition 2.2, the paper says g(.) is a mapping to x but in the equation it says g(.) maps to y instead.
(2) The paragraph above assumption 3.1, the paper says the last layer model fine-grained details of the data, but I think this can also be done but the last layer of any neural networks, such as MLP.
(3) I would also suggest the paper to add complexity analysis to their method as this might be a concern if too many latent variables have to be generated.

### Questions
Please refer to my comments under "Weakness".

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on using image generation models for contrastive self-supervised learning. Specifically, it employs a hierarchical structure of multiple latent vectors in the generative models to change the global to local information of generated images. New images are generated based on progressive perturbations of the latent vectors through hierarchies. The key message is that as the level in the hierarchy increases (from global latent variables to local ones), the perturbation distances in the latent space to change the semantics of generated images increase (i.e., at more local levels of latent vectors, changing the semantics of generated images requires more aggressive perturbation.) The authors then use the hierarchical regime to generate images for self-supervised learning and propose to continuously sample generated images to increase the total training size. The authors show SOTA results on training SimCLR and SimSiam using BigBiGan and StyleGAN2 to generate images.

Despite good efforts, the current shape of the paper lacks important technical details to make it sound and clear enough to be accepted at ICLR.

### Strengths
Originality: this work starts from an interesting perspective: using hierarchical levels of latent codes to generate images for representation learning and study the effect of perturbation in terms of the maximum distance allowed in the latent space to avoid semantic change. This is new to the reviewer’s knowledge.

Clarity: overall, the paper is clearly written, with an easy-to-follow presentation. The theoretical results are clearly stated, generated samples are nicely illustrated, and empirical results are concisely presented.
   
Significance: using generative models as data sources is an interesting and important problem, and this paper provides potentially useful techniques and insights for future researchers.

### Weaknesses
Quality: there are severe gaps in the theoretical part despite the adequate empirical results and analyses. The details are in the questions section.

### Questions
1. **Proposition 2.2**: it is very unclear how the authors deduced such an important statement, which itself also has ambiguity. The authors claim Proposition 2.2 is a reformulation from Li et al. (2022b), but there are no proofs or theoretical discussions that support this (upon the reviewer’s inspection, there is no directly equivalent statement in Li et al. (2022b)). In Li et al. (2022b), semantic consistency is defined as a bounded difference between mutual information terms, where the mutual information is between the generated images and the label. However, none of these is clearly stated in this submission, nor are there any discussions or linkages. In terms of the ambiguity, what does the right arrow mean? Is it convergence in probability, in distribution, or something else? Why does such convergence (or the mathematical property the authors intended to show) suffice the expressing the semantic consistency?

2. **Table 1**: it is quite debatable if InfoNCE loss is a good metric to measure semantic shift/consistency of images, which is also a central component this work’s conclusions rely on (Table 1 directly supports Assumption 3.1 empirically, which is the key message of the paper and does not have other rigorous theoretical justification). The authors did not justify at all why InfoNCE loss can align well with the intention of Proposition 2.2 regarding Semantic Consistency. In Proposition 2.2, semantic consistency is defined in terms of the consistency of true labels. However, the authors did not show how InfoNCE can *preserve* or approximate such consistency property when labels are not present. The absence of task information/labels does not automatically justify InfoNCE as the valid metric.

Other questions or comments:
1. It is not very clear to the reviewer why, in Table 1 left, different chunks use a different number of training samples for the Monte Carlo simulations.
2. Whether “robust” is the best word for this submission is debatable when it essentially means task-agnostic.

### Soundness
1 poor

### Presentation
3 good

### Contribution
3 good
