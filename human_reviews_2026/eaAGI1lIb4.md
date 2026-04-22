# XIL: Cross-Expanding Incremental Learning

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6, 6

## Abstract
Class-Incremental Learning (CIL) traditionally assumes that all tasks share a similar domain distribution, limiting its applicability in real-world scenarios where data arrive from evolving environments. 
We introduce a new problem setting, Cross-Expanding Incremental Learning (XIL), which extends CIL by requiring models to handle class-incremental data across distinct domains and to expand class-domain associations bidirectionally.
In this setting, new classes should be integrated into previously seen domains, while earlier classes are extended to newly encountered ones, a capability we refer to as bidirectional domain transferability (BiDoT).
To address XIL, we present a new framework, Semantic Expansion through Evolving Domains (XEED), which leverages domain-specialized prompts, residual-guided representation modulation, and evolving prototype embeddings to expand class semantics across previously encountered domains.
We further introduce the BiDoT Score, a novel metric for quantifying the degree of BiDoT.
Extensive experiments on benchmark datasets with significant domain shifts demonstrate that XEED outperforms existing CIL baselines by a large margin in both standard accuracy and BiDoT scores, establishing a strong foundation for realistic continual learning under domain-evolving conditions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces Cross-Expanding Incremental Learning (XIL)—a setting where both classes and domains evolve—and proposes XEED, a rehearsal-free framework that (a) learns domain-specialized prompts, (b) modulates class features with domain residuals using a diffusion model (IP-Adapter + SDXL), and (c) performs prototype-based classification with evolving, domain-aware prototypes. Also, define the BiDoT Score to measure bidirectional domain transferability. XEED substantially improves BiDoT and accuracy on PACS, Office-31, and DomainNet versus strong CIL/prompt baselines.

### Strengths
The paper introduces a new continual learning setting, specifically,

1) Clean formulation of XIL + dedicated BiDoT metric.

2) Method components are modular and leverage frozen encoder + small learned parts.

3) Privacy-friendlier than exemplar replay; aligns with synthetic replay trends.

### Weaknesses
1) The paper admits that no directly comparable XIL baselines exist and hence reuses CIL prompt-based methods (S-Prompts, CODA-P, CPrompt, etc.) as proxies. Since XIL is a new setting, the paper adapts existing CIL methods as baselines rather than comparing with purpose-built domain-evolving or generative continual learning frameworks. This makes it difficult to fully assess XEED’s relative performance or to verify whether its improvements come from the new setting or the framework design itself.

2) XEED’s strongest BiDoT improvements depend critically on synthetic image generation using IP-Adapter + SDXL. The ablation study shows that removing this component causes F-BiDoT to plummet (e.g., from 65.19 → 20.91 on PACS and 33.63 → 4.47 on DomainNet). This dependence implies that the system’s success hinges on access to a large diffusion model and high-quality generation. The method may fail or become impractical if the computing is limited or the diffusion model struggles with domain realism.

3) Limited or no experiment in common continual learning benchmarks such as CIFAR100, TinyImagenet200, ImageNetsubset100, Food, Cars, ImageNet-R/A.

### Questions
Please refer to the weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents cross-expanding incremental learning where not only class changes occur but also domain shift happens at the same time. To address this problem, authors proposes Semantic Expansion through Evolving Domains (XEED).

### Strengths
1) although it intersects with existing setting, the XIL problem is new. 
2) the proposed solution, namely XEED, performs well.

### Weaknesses
1) the following references also deal with domain adaptation and continual learning. I suggest to review them in the paper.

[1] Towards Cross-Domain Continual Learning
[2] Cross-Domain Continual Learning via CLAMP

2) please kindly explain the difference of your works with cross-domain continual learning as proposed in [1] and [2]

3) I suggest to elaborate more on the real-world context of your setting. This allows readers to appreciate more on your contributions. Also, it is suggested to link directly your problem with a concrete dataset that you use. Is there any dataset that can represent your problem?

4) the prompt selection mechanism is non-parametric and perhaps over-simplified.

5) prompt selection accuracy should be reported.

6) the domain order should affect the result. it should be detailed in the paper.

### Questions
1) the following references also deal with domain adaptation and continual learning. I suggest to review them in the paper.

[1] Towards Cross-Domain Continual Learning
[2] Cross-Domain Continual Learning via CLAMP

2) please kindly explain the difference of your works with cross-domain continual learning as proposed in [1] and [2]

3) I suggest to elaborate more on the real-world context of your setting. This allows readers to appreciate more on your contributions. Also, it is suggested to link directly your problem with a concrete dataset that you use. Is there any dataset that can represent your problem?

4) the prompt selection mechanism is non-parametric and perhaps over-simplified.

5) prompt selection accuracy should be reported.

6) the domain order should affect the result. it should be detailed in the paper.

### Soundness
2

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
This paper proposes the Cross-Expanding Incremental Learning (XIL), a new setting of continual learning. The XIL presents the problem of existing setting of class-incremental learning (CIL), where the model learns from the data of same domains. XIL emphasizes the  capability of transferring knowledge across different domains and proposes a new settiing. A corresponding method XEED is proposed to address the XIL. XEED is a generative replay-based method which leverages a pre-trained diffusion model and CLIP to generate samples that are transferable across different domains. Domain-specific prompts and prototypes are proposed to preserve knowledge from different tasks. Several experiments are conducted to validate the effectness of XEED.

### Strengths
1. This paper consider a new paradigm of continual learning, where the capability of transferring across different domains are considered.
2. This paper is well writen with good presentation and clear motivation.
3. The technique of generating exemplars with representation modulation is noval and sound.

### Weaknesses
1. Using the pre-trained diffusion model and CLIP. Although this paper is overall good, I have several concerns about using the pre-trained diffusion model. Since a good exemplar or sample for transferring can be generated, the knowledge related to the domains and classes is already encoded in the pre-trained model. As such, do we still need continual learning of another model? Most existing generative replay-based methods train a generative model during the continual learning, which is incremental learning process parallel to the training model. However, if the model needs another model which already has the knowledge of future data, since we already have a good one, what is the meaning of incremental learning of that model? 
2. The proposed method XEED is simple and preserving prompts and prototypes specific to distinct domains or classes has been widey studied in the community of continual learning. 
3. The capability of transferring across domains, the major challenge defined in this paper, seems mainly benefit from the generative replay. It seems that the major problem defined in this paper is solved by introducing knowledge of other models while the other parts of this method have mere innovation. One can argue that with such a replay mechamism, existing CIL methods can be easily transferred to the setting of XIL.
4. Results on dataset specially designed for the domain and class incremental learning, e.g., CoRE50 [1]. should be included. The datasets used in the experiments are not designed for incremental learning. The CoRE50, which is designed for both the CIL and Domain-incremental learning tasks, can be a good benchmark of XIL.

[1] Vincenzo Lomonaco and Davide Maltoni. Core50: a new dataset and benchmark for continuous object recognition. In Conference on Robot Learning, pages 17–26, 2017.

### Questions
Please refer to the weaknesses.

### Soundness
2

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
3

### Summary
This paper introduces Cross-Expanding Incremental Learning (XIL), a new, more realistic problem setting that extends traditional Class-Incremental Learning (CIL) by requiring models to learn new classes that arrive from distinct, evolving domains. The core challenge is achieving bidirectional domain transferability (BiDoT), where the model must generalize previously learned classes to new domains and new classes to old domains, even for combinations never seen during training. To address this, the authors propose a novel framework called Semantic Expansion through Evolving Domains (XEED), which utilizes domain-specialized prompts to adapt to different domains, a generative model to synthesize images for unseen class-domain pairs, and evolving prototypes to continuously expand the classifier's semantic space. Accompanied by a new evaluation metric, the BiDoT Score, experiments show that XEED significantly outperforms existing CIL methods, which struggle in this dynamic setting, thereby establishing a strong foundation for continual learning in real-world, non-static environments.

### Strengths
The paper's foremost strength is its introduction of the Cross-Expanding Incremental Learning (XIL) problem, which establishes a more realistic and challenging benchmark for continual learning by forcing models to simultaneously handle new classes and evolving data domains.

The proposed XEED presents a highly innovative solution, ingeniously using a generative model to synthesize data for unseen class-domain pairs, thereby directly addressing the core challenge of bidirectional knowledge transfer.

### Weaknesses
The paper does not sufficiently explain how the baseline methods, originally designed for the standard CIL setting, were adapted to the new multi-domain XIL setting. This is particularly concerning for methods like S-Prompts, which show drastically lower performance, raising questions about whether the comparison is entirely fair

As a paper proposing a new problem setting and a novel evaluation metric, the absence of publicly available code is a major drawback. This lack of resources makes it extremely difficult for the research community to verify the reported results, adopt the XIL benchmark, and fairly compare future methods, thus hindering follow-up research.

While the XEED is effective, its core components: domain-specific prompts, generative replay, and prototype-based classifiers, are existing techniques.

### Questions
NO

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
3

### Summary
This paper introduces Cross-Expanding Incremental Learning (XIL), a new setting that extends class-incremental learning to handle data from distinct and evolving domains. XIL requires models to perform bidirectional domain transferability, integrating new classes into old domains and adapting old classes to new ones. To address this challenge, the authors propose XEED, a framework that combines domain-specialized prompts, residual-guided modulation, and evolving prototypes to expand class semantics across domains. They also introduce the BiDoT Score, a metric that measures how well models generalize across unseen class–domain combinations. Experiments on benchmark datasets with significant domain shifts show that XEED outperforms existing methods in both accuracy and BiDoT scores.

### Strengths
- The paper is clearly written and well-structured, allowing readers to easily follow the logical flow and understand the motivation and proposed method at both conceptual and technical levels.

- The empirical results convincingly demonstrate the effectiveness of the proposed approach across multiple standard benchmark datasets.

### Weaknesses
- The **quality and fidelity of synthetic exemplars** present a potential limitation. The proposed method assumes that diffusion-based generation can effectively represent unseen domain–class combinations. However, such generative processes may produce low-fidelity or stylistically inconsistent samples, which could introduce noise into the evolving prototype updates and distort the learned feature space. A more detailed analysis of the generation quality—either through visual inspection, quantitative metrics such as FID or LPIPS, or ablation studies—would help validate whether the synthetic exemplars truly enhance domain transfer rather than acting as noisy augmentations.

- Another concern lies in the **high computational cost and reliance on diffusion models**. The method depends on a pre-trained diffusion generator to synthesize cross-domain exemplars, which is computationally expensive due to multiple denoising steps, high GPU memory requirements, and significant I/O overhead. While diffusion models contribute to the quality of generated samples, their use raises questions about scalability and practicality in large-scale or real-time incremental learning scenarios. A discussion of the computational budget, inference latency, and trade-offs between efficiency and performance would strengthen the paper’s experimental transparency.

- Finally, the framework’s **dependence on frozen backbones and domain prompts** may restrict adaptability under severe domain shifts. The feature extractor remains fixed during prompt tuning, assuming pre-trained features are sufficiently general for new domains. In cases where the domain gap is large (e.g., transferring from natural images to sketches or depth maps), this assumption may not hold, leading to suboptimal adaptation even with domain-specific prompts. Allowing partial backbone fine-tuning or incorporating adaptive normalization could improve flexibility and robustness in such scenarios.

### Questions
- How do the authors ensure that the diffusion-generated exemplars are of sufficient visual and semantic quality to represent unseen domain–class combinations?

- Could noisy or low-fidelity exemplars negatively affect the evolving prototype updates, and if so, how is this mitigated?

- How scalable is XEED when applied to larger datasets or a greater number of incremental tasks and domains?

- Why was the backbone 
𝑓
𝜙
f
ϕ
	​

 kept frozen during prompt tuning, and have the authors experimented with partial fine-tuning or adapter layers?

### Soundness
3

### Presentation
3

### Contribution
3
