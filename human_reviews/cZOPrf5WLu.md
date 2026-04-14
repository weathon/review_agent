# Learning on LoRAs: GL-Equivariant Processing of Low-Rank Weight Spaces for Large Finetuned Models

- Decision: Reject
- Scores: 3, 5, 8

## Abstract
Low-rank adaptations (LoRAs) have revolutionized the finetuning of large foundation models, enabling efficient adaptation even with limited computational resources. The resulting proliferation of LoRAs presents exciting opportunities for applying machine learning techniques that take these low-rank weights themselves as inputs.  In this paper, we investigate the potential of Learning on LoRAs (LoL), a paradigm where LoRA weights serve as input to machine learning models. For instance, an LoL model that takes in LoRA weights as inputs could predict the performance of the finetuned model on downstream tasks, detect potentially harmful finetunes, or even generate novel model edits without traditional training methods.  We first identify the inherent parameter symmetries of low rank decompositions of weights, which differ significantly from the parameter symmetries of standard neural networks. To efficiently process LoRA weights, we develop several symmetry-aware invariant or equivariant LoL models, using tools such as canonicalization, invariant featurization, and equivariant layers. We finetune thousands of text-to-image diffusion models and language models to collect datasets of LoRAs. In numerical experiments on these datasets, we show that our LoL architectures are capable of processing low rank weight decompositions to predict CLIP score, finetuning data attributes, finetuning data membership, and accuracy on downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The work introduces the Learning on LoRAs (LoL) paradigm; LoL is an ML model which has LoRA adapters as input and it outputs predictions about the weights.

### Strengths
Shows that LoRA adapters have information in them that can be learned.
I guess the idea of learning a model for LoRA adapters is technically novel, then again the fact that weights encode information and that information can be extracted isn't a wild thought.

### Weaknesses
I don't really understand why this is at all useful...

### Questions
- The model allows you to predict the accuracy of the model if a set of LoRA weights is applied; how is this useful? Why not just apply the LoRA weights and actually measure the accuracy?

- Why is it useful to predict how many data points the model was fine tuned on... If I have a set of LoRA adapters, I'll try every single one and the set of LoRA weights that does best is the one I'll use (regardless of how many data points it was trained on...).

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
In this paper, the authors explore the potential of Learning on LoRAs (LoL), a paradigm where LoRA weights are used as inputs to machine learning models. They fine-tune thousands of text-to-image diffusion models and language models to collect datasets of LoRAs and develop several symmetry-aware, invariant or equivariant LoL models using techniques such as canonicalization, invariant featurization, and equivariant layers.

### Strengths
1. Several GL(r) equivariant and invariant neural architectures are proposed to effectively process weights of LoRAs.
2. Creating novel datasets for Learning on LoRAs
3. Authors conduct experiments across various finetuned models.

### Weaknesses
1. The detailed information regarding the models, such as the number of parameters and the training costs associated with different model structures and methods, is not sufficiently clear.
2. The motivation behind the task selection for LoL models is not clearly articulated. For instance, the LoL models aim to predict hyperparameters, CLIP scores, and training images of fine-tuned diffusion models using their LoRA weights. However, it is not immediately clear why these tasks are important or how they contribute to advancing the field. 
3. Other methods used for comparison in this work are not clearly defined and lacks sufficient discussion.

### Questions
1. The LoRA model datasets used in this work are derived from Stable Diffusion V1.4 and Qwen2. A key question is whether the LoL model will remain effective if LoRA models are obtained from different versions, such as Stable Diffusion V2.1 or Qwen V2.5. 

2. Is there a clear rationale behind the choice of model architecture for the LoL models? Specifically, is using an MLP sufficient for the tasks at hand, or could other architectures be more effective?

3. The paper mentions that a large number of LoRA models were collected. How many GPU hours during the collection of these LoRA models?

4. The practical usage of the LoL models is not explicitly addressed. Are there any potential real-world applications except the dataset size prediction?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Learning on LoRAs (LoL), an approach where low-rank adaptation weights, commonly used for efficient fine-tuning of large foundation models, serve as inputs for predictive models. The authors develop several GL(r)-equivariant and invariant neural architectures that process LoRA weights for various downstream tasks, such as predicting model accuracy, analyzing finetuning attributes, and performing data membership inference. The models leverage geometric deep learning techniques like canonicalization and invariant featurization, allowing efficient performance prediction without loss of structural information. The authors validate their approach through experimentation across diverse tasks on newly generated datasets of LoRA weights, including diffusion finetunes and language models finetunes.

### Strengths
1 - The paper contributes to the field by capitalizing on the structural properties of LoRA weights, enabling efficient analysis and predictions without needing full model access or extensive retraining. By focusing on parameter symmetries in LoRA weights, the method facilitates tasks like predicting model performance on downstream tasks and analyzing fine-tuning attributes directly from the LoRA weights themselves without requiring compute- and parameter-heavy inferences. 

2 - The mathematical rigor in defining GL-invariant and GL-equivariant properties strengthens the reliability of the model design. In particular, Theorem 1 and Theorem 2 (Section 4)​ define the necessary properties of the proposed models, providing a clear foundation for future work in this domain. By establishing these invariance properties, the authors ensure that their architectures are theoretically grounded in symmetry-aware learning. The use of canonicalization and alignment methods for symmetry processing is well-supported by previous studies in geometric learning (see Maron et al., 2019) and further extends the concepts to the LoRA domain. 

3 - The paper demonstrates extensive experiments across multiple datasets (Section 5, Tables 2 and 3)​. Notably, the authors generate three novel datasets of LoRAs (Section 5.1)​, including CelebA-LoRA, Imagenette-LoRA, and Qwen2-ARC-LoRA, providing a robust basis for evaluating their models.

### Weaknesses
1 - The paper could benefit from a more thorough introduction to concepts like GL-invariance, O-invariance, and O-Align before discussing them within the context of Learning on LoRAs (LoL). While these concepts are central to the proposed architecture, they may be unfamiliar to some readers, especially those not specialized in geometric deep learning. Including a dedicated background section or a more extensive explanation would make the work more accessible and improve readability.

2 - GL-net’s superior performance may be partly due to a higher number of parameters or computational requirements compared to other LoL architectures. This could imply a cost-performance trade-off, where GL-net’s success on specific tasks may not be as scalable or efficient in scenarios requiring high throughput or limited compute resources. An analysis of this cost-performance trade-off would provide helpful insight into its practical applicability.

3 - The visuals, such as Figure 3 showcasing CLIP predictions, could benefit from clearer labeling and expanded captions to help interpret performance differences among models​. While the data is comprehensive, more explanatory notes within the figures would enhance accessibility for readers.

4 - The GL(r)-equivariant models, while theoretically sound, are complex and may pose practical challenges for adoption. The multi-step processing involving canonicalization, invariant featurization, and the invariant head layer adds architectural overhead (Section 3.3.3). While this complexity is justified for theoretical exploration, a discussion on trade-offs and simplifications would make it easier for practitioners to implement the approach. Previous work in weight-space learning, such as Navon et al. (2023), has shown simpler approaches that could be integrated as baselines to contrast model complexity and efficiency​.

### Questions
I have no major concerns and consider the paper as a strong submission, but it would be good if the authors could elaborate on the following issues:

1 - Can the proposed methodologies (such as GL-equivariant architectures) be directly applied to other LoRA-based fine-tuning variants, like DoRA (Liu et al., 2024)? If modifications are necessary for such variants, what aspects of the approach would need adaptation, particularly in handling their unique low-rank structures?


2 - Is the performance boost primarily due to the model’s increased number of parameters or computational requirements compared to other LoL techniques (Section 5.2, Table 2)? If so, how would you describe the cost/performance trade-off for GL-net in practical applications, especially in contexts where computational efficiency is prioritized?

3 - In Table 4, GL-net shows relatively lower performance on the dataset size prediction task. Could this lower performance be attributed to overfitting? If so, would any regularization techniques be effective here, and are there any architectural adjustments that could improve generalization on this specific task?


4 - The authors briefly mention future applications of GL-equivariant networks, such as model merging and LoRA editing (Conclusion)​. Could authors please elaborate on the specific challenges or potential approaches you foresee for these applications? In particular, how might the GL-net architecture be adjusted or extended for these tasks, which may have different requirements compared to the predictive tasks presented?


References

 1- Maron, Haggai, et al. "Invariant and equivariant graph networks." arXiv preprint arXiv:1812.09902 (2018).

 2 - Navon, Aviv, et al. "Equivariant architectures for learning in deep weight spaces." International Conference on Machine Learning. PMLR, 2023.

3- Liu, Shih-Yang, et al. "Dora: Weight-decomposed low-rank adaptation." arXiv preprint arXiv:2402.09353 (2024).

### Soundness
3

### Presentation
3

### Contribution
4
