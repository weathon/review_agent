# MaGA: Machine-Guided Amnesiac Unlearning through Target Feature Disentanglement

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
The security of training data has raised the ``Right to be Forgotten'' policy to protect the privacy of data providers, leading to an urgent need for effective Machine Unlearning. However, existing unlearning methods often face a trade-off dilemma between fully erasing the influence of target data and preserving the overall model capability. 
To address this, we first investigate the intrinsic characteristics of class concepts learned during model pretraining, revealing that these concepts are often entangled at the feature pattern level. Based upon this insight, we introduce Machine-Guided Amnesiac (MaGA), a novel unlearning framework to manipulate the unlearning process via leveraging Multi-modal Large Language Models to estimate conceptual similarities between features. These similarities are encoded in a transition matrix to assign suitable perturbing labels for re-alignment of target data to achieve unlearning. This facilitates effective unlearning, as it perturbs the concepts related to target instances, thus reducing undesired model disruption. 
Furthermore, we propose a Fragment-Absorb strategy to disentangle the influence of target concepts through a positive-negative feature noise pair. During unlearning, both feature noises are leveraged to impede target feature patterns while enhancing the remaining desired features. This promotes selective forgetting of target data influence, smoothing complete unlearning while mitigating the risks of under-unlearning or over-unlearning.
Extensive experiments conducted across typical unlearning tasks and diverse datasets demonstrate that our approach outperforms existing baselines, effectively removing target data while preserving the model generalization on retained data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes MaGA, a new machine unlearning framework that integrates Multi-modal Large Language Models (MLLMs) to guide the unlearning process. The method introduces two key ideas: 1. A transition matrix constructed from MLLM-estimated conceptual similarities to generate perturbing labels for unlearning; 2. A Fragment-Align strategy that injects positive and negative feature noises to disentangle target features and preserve generalization. Experiments are conducted on CIFAR-10, CIFAR-20, and CIFAR-100 across class-wise, sub-class, and random unlearning tasks. Results suggest that MaGA achieves competitive performance compared with existing methods, offering improved unlearning effectiveness (lower MIA) and retained accuracy close to retrained “gold” models.

### Strengths
1. Novel conceptual direction: Leveraging MLLMs for semantic similarity estimation in unlearning is an interesting and creative approach that bridges recent advances in multimodal understanding and privacy-oriented model editing.
2. Relatively well-designed experiments: Includes comparisons to multiple baselines, ablation studies, sensitivity analysis, and visualization (t-SNE), showing careful experimental design.

### Weaknesses
1. High computational overhead and scalability concerns. MaGA heavily depends on MLLM inference and feature noise generation. The method is only demonstrated on small datasets (CIFARs), and it remains unclear how it scales to larger datasets (e.g., ImageNet) or higher-dimensional data. The claimed precomputation trick for the transition matrix is not empirically verified.
2. Lack of theoretical justification. The notion of “feature disentanglement” and the claimed semantic alignment between perturbing and target labels are conceptually appealing but lack theoretical analysis or measurable metrics. The method’s convergence or stability is not discussed.
3. Limited experimental diversity. All experiments are on small-scale vision datasets. The method’s generality to real-world or cross-domain settings (e.g., text, multimodal, federated unlearning) is untested.
4. Relation to prior art unclear. The proposed Fragment-Align approach is conceptually close to existing feature-noise-based unlearning methods. The paper does not sufficiently differentiate MaGA in terms of objective function or theoretical insight.
5. Efficiency not convincingly addressed. Although the paper claims 59% time savings compared to full retraining, the absolute time cost of MLLM queries is omitted, and the comparison seems unfair because the baseline retraining cost depends on dataset scale.
6. Poor writing. The author should improve the illustration through adjusting figure, table and more fluent expression.

### Questions
See the weaknesses.

### Soundness
1

### Presentation
1

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
The key idea of this work is to maintain a balance between effective forgetting of targeted concepts and reducing undesired model disruption. To achieve this, this work first constructs a Transition matrix that aligns perturbed labels by considering inter-conceptional similarities. Additionally, to balance over-unlearning and under-unlearning, this work introduces positive and negative perturbation noises, thereby maintaining retained knowledge while selectively forgetting the target concepts.

### Strengths
1) This paper explores the over-unlearning and under-unlearning, which a key challenge of machine unlearning. To address it, this work proposes a balanced mechanism that preserves retained knowledge while effectively forgetting targeted concepts.

2) The use of semantic concept modeling (inter-concept similarity via a transition matrix) is insightful and could inspire follow-up research on concept-aware unlearning.

### Weaknesses
1) The theoretical foundation behind the proposed method is unclear. In particular, the derivation of positive and negative noise terms in Equations (8) and (9) does not guarantee the generation of semantically meaningful concepts. Previous studies on model inversion and adversarial attacks have shown that generating interpretable concepts requires explicit constraints, which are not discussed or justified in this paper.

2) The experiments focus mainly on classification tasks, with no evaluation on multi-modal or generative tasks. Since the method operates at the concept level, it would be particularly relevant and insightful to test it on conceptual generative models.

3) The paper does not clearly justify why a Multimodal Large Language Model is required to construct the Transition Matrix. It remains unclear whether this component is central to the proposed method. Moreover, the lack of ablation studies makes it difficult to assess its contribution and necessity.

4) The experimental results are not fully convincing. For example, Table 1 (class-wise unlearning on CIFAR-100 using ResNet-18) shows inconsistent improvements, and the use of bolded values is confusing. Furthermore, the evaluation omits comparison with important recent baselines such as SalUn [1].

[1] SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation

### Questions
1) How to ensure that the positive and negative noise vectors correspond to semantically meaningful concepts rather than arbitrary perturbations?

2) Is MLLM specifically used to build the Transition Matrix? Would a simpler embedding-based similarity approach (e.g., CLIP or Word2Vec) yield comparable results?

3) Could the approach generalize to generative models or LLMs?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce a novel unlearning method which focuses on addressing the forget-utility tradeoff for unlearning.

For a specific target image, their method first identifies other classes of images (called perturbing class) with similar feature representations as the target class, with the goal of mapping the representations of the target image to such similar classes.\
Then, the authors learn two noise matrices to add to the target image - a positive noise matrix which enhances the features common with the perturbing class and a negative noise matrix which enhances the features that are not representative of the target class.

This helps them to design a novel forget loss which maps the features of the target image to representations of a different class without damaging important features.

The authors demonstrate successful unlearning on a variety of unlearning settings with significant improvement in MIA accuracy in many cases.

### Strengths
- The paper is well written and easy to follow.
- The authors introduce a novel algorithm with a focus on preserving model utility.
- The authors provide comprehensive experiments on CIFAR-10, CIFAR-100 and CIFAR-20 using Resnet18 and ViT.

### Weaknesses
The main weakness is some missing baselines. 
Kindly compare with other SOTA methods like [1] and [2].
Even though [2] is designed for class-wise unlearning, it is designed using a similar rationale of class-discriminatory feature space manipulation. 



[1] Kurmanji, Meghdad, et al. "Towards unbounded machine unlearning." Advances in neural information processing systems 36 (2023): 1957-1987.\
[2] Kodge, Sangamesh, Gobinda Saha, and Kaushik Roy. "Deep unlearning: Fast and efficient gradient-free class forgetting." Transactions on Machine Learning Research (2024).

### Questions
- What happens if the model logits are used to form the transition matrix instead of the MLLM ?
- Is there any rationale behind the design choice of a True / False question for the prompt to the MLLM ?
- In Figure 1,  please indicate which one is the image from the target class and perturbed class. Are these masks found using optimization ?  If not, kindly include some examples of positive and negative noise matrices. 
- Please specify the function $\phi$ in Eq. 6
- Is there any constraint on the feature noise matrices ?
- Additionally, it is unclear what the optimizations of Eq 7 and 8 are doing.  The subscript i of $N_i$ indicates that it is specific to $x_i$, however that does not seem to be the case from the optimization itself. Please give an intuition as to why only $N$ is used rather than $N+x_i$ inside the loss functions.

### Soundness
3

### Presentation
4

### Contribution
3
