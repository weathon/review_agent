# Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 2

## Abstract
Large neural networks are typically trained for a fixed computational budget, creating a rigid trade-off between performance and efficiency that is ill-suited for deployment in resource-constrained or dynamic environments. Existing approaches to this problem present a difficult choice: training a discrete collection of specialist models is computationally prohibitive, while dynamic methods like slimmable networks often lack the flexibility to be applied to large, pre-trained foundation models. In this work, we propose *Nested Subspace Networks (NSNs)*, a novel architectural paradigm that enables a single model to be dynamically and granularly adjusted across a continuous spectrum of compute budgets at inference time. The core of our approach is to re-parameterize linear layers to satisfy a nested subspace property, such that the function computed at a given rank is a strict subspace of the function at any higher rank. We show that this entire hierarchy of models can be optimized jointly via an uncertainty-aware objective that learns to balance the contributions of different ranks based on their intrinsic difficulty. We demonstrate empirically that NSNs can be surgically applied to pre-trained LLMs and unlock a smooth and predictable compute-performance frontier. For example, a single NSN-adapted model can achieve a 50\% reduction in inference FLOPs with only a 5 percentage point loss in accuracy. Our findings establish NSNs as a powerful framework for creating the next generation of adaptive foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes a way to re-parametrize linear layers into low-rank factorization $W=A^{R \times d_{in}} \cdot B^{d_{out} \times R}$ such that the compute budget can be dynamically adjusted without retraining by constructing the  effective weight matrix from the first r rows of A and first r columns of B. In other words, the effective layers forms a filter chain of images for the ranks r < R (a.k.a Nested Subspace Property / Layer). To train the layer effectively, the work leverages an uncertainty-weighted training objective by Kendall+2018 that balances the learning across ranks. The experiments show that the nested subspace layers can be introduced into pretrained LLMs to create a smooth rank-based accuracy-vs-compute trade-off that remains adjustable at test-time.

### Strengths
- The approach is highly flexible and practical as it can be broadly applied to any pretrained models with linear layers through a straight forward Singular Value Decomposition. Rather than requiring the specification of a suitable rank up front that may not be known for a given problem and architecture the method allows to "shut up and train" to find an appropriate effective rank.
- The resulting architecture is intepretable in the sense that it is straight-forward to inspect how much each ranks learns to represent both empirically and theorectically (the offered theoretical discussion provides a good intution of the approximation trade-offs involved).

### Weaknesses
- Section 4.1 investigates whether a single NSN, trained with the multi-rank objective, can match the performance of multiple, specialized model, but only does so for an somewhat artificial image classification setup where regular MLP baselines are trained from scratch. What is currently missing is a similiar quantitive comparison with LoRA baselines to establish whether an NSN evaluated for rank r matches a LoRA model that exclusively trained for rank r. This would make this work comparable with prior baselines and delinate the benifits of NSN learning vs predefined-rank models.
- An explicit desideratum is granularity of adaption, but the trade-off here only works in discrete 'matrix rank' steps for each layer. Since different layers at different depths learn to represent different things it is not clear that a uniform rank pick across the entire model is ideal. Arguably, the most granular adaption would be picking the ideal rank size for each layer but it is not obvious how to do that efficiently. 
- The work mentions the connection to alternative weight-learning approaches such as GradNorm but there is no comparison or analysis on why the Kendall-loss formulation is the best choice.

**Additional comments**

- The core idea of the work could be introduced and visualized more clearly, e.g. by illustrating an example A and B and the resulting effective weight matrix. The right column of Figure 1 does not make it very clear that there is only 1 instance of A and B and not a duplication of many A and Bs.
- L182 "Following Kendall et al. (2018), we use the standard uncertainty-weighted surrogate objective"; I recommend expanding on this to make the work more self-contained and accessible to readers who may not be familiar with  
- "a strict and dynamic computational budget" - I found this confusing to read; it might be better to concretely lay out the use cases that this work has in mind.
- L115 typo: that *form* a nested hierachy

Aside: The manuscript uses visually highlighted "Takeaway" messages that spell out what the reader may conclude from each section. I am not sure if this is helpful because it can feel repetitive, or, if it does not align with the reader's conclusion, somewhat presumptuous. For example, L276 "smooth and predictable trade-off" placed before any experimental results reads more like an unverifiable claim than a takeaway.

### Questions
- It is not clear how the anchor and variant rank are chosen. Section 2.2 says that the ranks are "sampled" but I am not sure why and how specifically. Is this applied consistently across training runs or is there a random element to it?
- It is claimed that the log-variances "serve as an emergent proxy for the effective expressiveness of each rank-specific model" but it is not clear why that needs to be the case since they are as far as I can tell freely trainable parameters. The same goes for the "uncertainty-weighted surrogate" - how do we know that these parameters act as a meaningful surrogate for uncertainty?
- Are you planning to release your source code and a drop-in layer implementation?

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
This paper proposes Nested Subspace Networks (NSNs), an architectural  framework for neural networks that enables continuous and dynamic  adaptation to a range of computational budgets through a nested subspace reparameterization of linear layers. NSNs introduce a means to  dynamically select model "rank" at inference, with theoretical backing  for smooth interpolation across ranks and an uncertainty-aware  multi-rank training objective to promote optimality across the entire  rank spectrum. Empirical evaluations include MLP experiments on CIFAR-10, ablation studies, and applications to several large pre-trained language models (LLMs), demonstrating that NSNs achieve competitive accuracy with significantly reduced computational cost.

### Strengths
- The paper proposes a conceptually novel framework, Nested Subspace Networks, that unifies multiple model capacities within a single parameterization. 
- The approach is validated through controlled experiments, ablation analyses, and applications to pre-trained LLMs, supporting the claims of dynamic adaptability and compute–performance efficiency.
- A key practical merit is the post-hoc applicability of NSNs: the framework can be retrofitted onto existing pre-trained models without retraining from scratch, addressing a critical deployment challenge for large-scale foundation models.

### Weaknesses
- Equation 2 needs clarification. Does the variant rank $r$ change in different epoch or remain unchanged during training. Is $\mathcal{L}_{CE}(k)$ the cross entropy loss calculated when the linear weight using its first $k$ rows and columns?
- Implementation details missing. The paper does not explain how Logits Regularization, Residual Orthogonality, and Hidden Regularization (as shown in Table 1) are implemented, nor whether these are used concurrently with the main “Two CEs” objective or as independent ablations.
- Ambiguity in Section 4.1. The experiment uses CIFAR-10 but states that “inputs are ImageNet last-layer embeddings.”
Please clarify this setup: are ImageNet features used merely as a fixed embedding extractor, and are all MLP layers subsequently replaced with NSN layers?

### Questions
1. Why choosing exponential in Equation 3 as the coefficient.

2. Is there any empirical evidence about assumption 1.

3. In Figure 7, it is shown that the model's score on the evaluation  benchmarks decreases continuously and monotonically as the rank is  reduced. How can we prove that this decline represents the desired  hierarchical degradation in reasoning ability, rather than simply a  score reduction caused by a decrease in the model's overall expressive  power? 

For instance, is the score drop a result of the model becoming  inconsistently capable of solving tasks of the same difficulty level  (i.e., succeeding on some while failing on others)? Or is it due to the  model becoming uniformly incapable of solving a whole category of  problems at a certain difficulty level? Only the latter would  demonstrate an effective adjustment of reasoning and computational  effort according to task difficulty.

4. A recent paper [1] on implicit regularization in matrix factorization (which is not cited in Related Works) also analyzes the hierarchical training dynamics that emerge when progressively increasing the matrix rank—an idea conceptually related to the nested structure proposed in this paper (see, for example, their Proposition 1). One of their key findings is that even without explicit rank constraints, the model naturally evolves from low-rank to high-rank representations during optimization. It would strengthen the theoretical positioning of this work if the authors could discuss or compare the necessity of explicitly enforcing rank constraints in NSNs.

[1] Connectivity Shapes Implicit Regularization in Matrix Factorization Models for Matrix Completion, NeurIPS 2024.

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
This paper presents a model architecture called Nested Subspace Networks (NSNs), which can be applied on top of modern neural network architectures both during pretraining and after pretraining. NSNs consist of a series of low rank subnetworks that approximate the original neural network, thereby reducing computational FLOPs during training, adaptation, and inference. A novel optimization procedure based on uncertainty estimation is also proposed to jointly optimize all subnetworks within the NSN architecture. Both theoretical analysis and experimental results are provided to demonstrate the effectiveness of the proposed NSNs and the optimization method.

### Strengths
- The paper is overall well written and well motivated.
- Improving the computational efficiency of large deep models and large language models is of practical importance.
- The proposed NSN model architecture is clear and easy to follow.

### Weaknesses
- The NSN architecture does not appear to be highly novel. It essentially applies the slimmable neural network [1] on top of a low rank model architecture (across the rank dimensions).
- The proposed "training with multi rank uncertainty" procedure is not clearly explained. I suggest adding an algorithm box to illustrate the detailed training steps.
- Only FLOPs are reported; wall clock time would be a more informative metric to demonstrate both training and inference speedups.
- It is not clear, during model training, what the overall cost of training an NSN is compared to the standard training of a dense model with a similar number of parameters.

[1] https://arxiv.org/abs/1812.08928

### Questions
- The methods proposed in [2] seem to be very promising on top of slimmable networks. I wonder if they are also applicable to NSNs.


[2] https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Universally_Slimmable_Networks_and_Improved_Training_Techniques_ICCV_2019_paper.pdf

### Soundness
2

### Presentation
3

### Contribution
1
