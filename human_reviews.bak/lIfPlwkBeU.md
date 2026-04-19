# Multi-conditioned Graph Diffusion for Neural Architecture Search

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3

## Abstract
Neural architecture search automates the design of neural network architectures usually by exploring a large and thus complex architecture search space. To advance the architecture search, we present a graph diffusion-based NAS approach that uses discrete conditional graph diffusion processes to generate high-performing neural network architectures. Our method is based on the idea of classifier-free guidance, which we introduce for graph diffusion models. We then propose a multi-conditioned classifier-free guidance approach applied to graph diffusion networks to jointly impose constraints such as high accuracy and low hardware latency. Unlike the related work, our method is completely differentiable and requires only a single model training. In our evaluations, we show promising results on six standard benchmarks, yielding novel and unique architectures at a fast speed, i.e. less than 0.2 seconds per architecture. Furthermore, we demonstrate the generalisability and efficiency of our method through experiments on ImageNet dataset.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper introduces a differentiable generative technique based on diffusion, enabling training with a single model and reducing search time in NAS. They utilize discrete conditional diffusion procedures and opt for a classifier-free approach for the graph diffusion model, proposing a multi-conditioned diffusion guidance approach.

### Strengths
- They present a multi-conditioned graph diffusion model.
- They conducted experiments not only in the vision domain but also in the NLP domain.

### Weaknesses
- The authors insist that the potential of classifier-free guidance in graph diffusion networks remains unexplored (and also said that the current classifier-free guidance approaches operate only on the image synthesis and are single-conditioned), but there are some works exploring classifier-free guidance in graph diffusion networks e.g., [1]. This is not an exact statement. 
-  Many core ideas overlap with the related work [2] that the authors did not mention, such as the first adoption of diffusion models in the NAS field, the utilization of positional embedding, and so on.
- For a new condition, a significant cost would be involved in training a diffusion model again. When the search space becomes vast, it could pose a problem.
- The proposed method seems to be confined only to the cell-based search space.
- The used baselines seem outdated. It would be beneficial to include more recent baselines such as [3].
- Discretizing the real-valued condition into multiple classes doesn't seem like a natural approach. In a real-world scenario, when finding architecture and label pairs for training the diffusion model, it seems difficult to avoid class imbalance issues concerning the given threshold, $f_{\text{th}}$, which can lead to a decrease in the model performance.

***
### References 
[1] Hoogeboom, Emiel, et al. "Equivariant diffusion for molecule generation in 3d." International conference on machine learning. PMLR, 2022.

[2] An, Sohyun, et al. "DiffusionNAG: Task-guided Neural Architecture Generation with Diffusion Models." arXiv preprint arXiv:2305.16943 (2023).

[3] Shala, Gresa, et al. "Transfer NAS with meta-learned bayesian surrogates." The Eleventh International Conference on Learning Representations. 2022.

### Questions
- How did you select $\lambda$, which is a parameter to weight the importance of nodes and edges, in Eq (2)?
- How did you handle the upper triangular matrix when generating the edge matrix, e.g., through masking the lower part of the matrix or by directly utilizing the upper triangular matrix during the training of the diffusion model?
- How can we apply this method to not the cell-based search spaces?
- How do you adjust the guidance scale $\gamma$ in Eq (5)?
- Is positional embedding not involved in the sampling process?
- How were the (architecture, label) pairs sampled when training the conditional diffusion model?
- Why wasn't the variation for each experiment provided? Is there a specific reason for this?
- Why does the feasibility decrease as the latency constraint value increases in Table 4?
- Could you provide a more detailed explanation of the process used to derive the search time in Table 5? Is that the query time?
- In Eq (6), is the proposed method simply concatenating multiple constraints?
- How many training samples are used in the ablation study on novelty and uniqueness?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The proposed method is based on graph diffusion and introduces the concept of classifier-free guidance for NAS. It uses discrete conditional graph diffusion processes to generate high-performing neural network architectures. The approach can impose constraints such as high accuracy and low hardware latency by using a multi-conditioned classifier-free guidance technique. Unlike existing methods, this approach is fully differentiable and requires only a single model training. The authors demonstrate the effectiveness of their method by achieving promising results on six standard benchmarks and showcasing fast architecture generation, taking less than 0.2 seconds per architecture. They also demonstrate the generalizability and efficiency of their method on the ImageNet dataset.

### Strengths
- This paper is well written and easy to follow.
- This paper is the first work to introduce classifier-free diffusion model for neural architecture generation.
- This paper conduct extensive experiments on various benchmarks.

### Weaknesses
The reviewer's main concerns are as follows:

- Lack of experimental support for the contribution
    - The proposed method introduces a multi-conditioned diffusion guidance technique as one of its primary contributions. However, most experiments have been conducted with a single objective, primarily focusing on accuracy. This limitation makes it challenging to have confidence in the claim about the capabilities of the proposed method. To address this issue, it is recommended to conduct additional experiments under various scenarios with multi-objectives or multi-conditioned criteria, encompassing factors like accuracy, latency constraints, and memory footprint constraints simultaneously.

- Insufficient Analysis:
   - Are the generated graphs always valid neural architectures following the rules of the given search space? For example, let's say that nodes represent operations, and edges represent connections between operations. It is possible that nodes are created, but edges connecting nodes are not generated (i.e., isolated nodes). In this case, we can consider the generated neural architecture as invalid. Consequently, experiments related to the validity of the proposed method should be performed on diverse benchmarks like NB101, NB301, NB-NLP, etc. 
   - There is no analysis concerning the guide scale (r), which is expected to have a substantial impact on the types of neural architectures generated. Thus, a thorough exploration of this aspect is necessary.

- Unnatural problem formulation:
  - The paper's approach to discretizing the target variable, particularly in the context of accuracy, may appear unnatural. Instead of searching for neural architectures that maximize the final accuracy, the paper defines the goal as finding architectures included in the top 5%. This approach needs clarification or justification to enhance the clarity of the problem formulation.
- Retraining the diffusion model:
  - The paper utilizes a classifier-free diffusion model, which necessitates retraining every time the objective or conditions change. This retraining process is not trivial and can be costly. 
- Unfair comparison with existing baselines:
  - Many experiments in the paper utilize queries as metrics, but this approach might not be entirely fair to existing baselines. This is because the proposed method incurs an additional cost in the form of training the diffusion model, which is separate from the queries themselves. For example, the 8.3 hours of training required for NB301 is a significant time that cannot be disregarded. In this regards, Table 5 should be labeled more accurately to reflect the true cost as "0.001 + diffusion model training time" instead of simply "0.001."
- Baseline performance:
  - It is observed that the baseline performance for each benchmark is not at a state-of-the-art level, and the performance of the proposed method does not appear to be particularly impressive. This raises concerns about the competitiveness and efficacy of the proposed method in comparison to existing baselines.

### Questions
Please address the concerns in Weaknesses.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new diffusion-based NAS algorithm called DiNAS, This method use graph diffusion models to generate architectures. Specifically, the method uses conditional diffusion model to control the properties of the generated architectures, e.g., performance and latency. The entire framework can be optimized by gradient-based approaches. The model is evaluated on several NAS benchmarks to demonstrate the effectiveness.

### Strengths
1.	This paper is well-written and easy to follow.
2.	Using graph diffusion to generate architectures is a reasonable idea.

### Weaknesses
1.	The method is very straightforward. The method designs lack the relation with NAS problems, making the technical contribution limited.
2.	There has been diffusion-based NAS method [1], but the paper does not mention the difference between the two methods.
3.	All experiment are conducted on NAS benchmarks. In my opinion, NAS benchmarks provide quick feedback for developing new NAS methods, but NAS methods still need evaluation out of NAS benchmark to avoid overfitting on these benchmarks.
4.	In table 5, the search time of DiNAS excludes the performance estimation time of architecture training. It is unfair to compare with other methods with performance estimation time, e.g. NASNET, DARTS.
[1] DiffusionNAG: Task-guided Neural Architecture Generation with Diffusion Models

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
