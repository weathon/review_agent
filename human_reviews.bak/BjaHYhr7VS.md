# Not all parameters are equal: a Hessian informed differential learning rate for deep learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 6

## Abstract
Differential learning rate (DLR), a technique that applies different learning rates (instead of a single one) to different model parameters, has been widely used in deep learning and achieved empirical success via its various forms. For example, parameter-efficient training (PET) applies zero learning rates to most parameters so as to significantly saves the computational cost; adaptive optimizers such as Adam apply the coordinate-wise learning rate to accelerate the convergence.

At the core, DLR leverages the observation that different parameters can have different loss curvature, which is hard to characterize in general. We propose the Hessian-informed differential learning rate (Hi-DLR), an efficient approach that captures the loss curvature of parameters for any model and optimizer adaptively. Given a proper grouping of parameters, we empirically demonstrate that Hi-DLR can improve the convergence by dynamically determining the learning rates during the training. Furthermore, we can quantify the influence of different parameters and freeze the less-contributing parameters, which leads to a new PET that automatically adapts to various tasks and models.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a differential learning rate strategy called Hi-DLR. The algorithmic novelty of Hi-DLR lies in diagonalizing the Hessian matrix $H$ and extending the first-order approximation of $\mathbf{G}_t^{\top} \mathbf{g}_t^{\text{optim}} / \left(\mathbf{g}_t^{\text{optim}}\right)^{\top} \mathbf{H}_t \mathbf{g}_t^{\text{optim}}$ from [1] from ULR to Differential Learning Rate (DLR). The authors also adopt a per-parameter influence derived from Hi-ULR to select influential parameters for parameter-efficient training. From the reviewer's perspective, this definition of parameter influence is novel, although it is the optimal solution of equation 3.1 with a diagonalized version of the Hessian matrix. The reviewers welcome discussions with other reviewers and the Area Chair to determine if this definition of influence demonstrates conceptual novelty.

[1] Automatic gradient descent with generalized Newton’s method. arXiv, 2024.

### Strengths
- The paper is well-motivated and designed to create suitable learning rates for different parameters.
- The authors also provide applications of the proposed Hi-DLR to NAM and LoRA.
- The per-parameter influence metric is interesting.

### Weaknesses
**Evaluation of Hi-DLR**: All training loss figures in this paper are plotted with respect to iterations (e.g., Figures 4 and 5). Could the authors provide wall-clock time comparisons between Hi-DLR and baseline methods, in addition to the iteration-based plots. This would help demonstrate whether Hi-DLR provides real-world speedups.

### Questions
The authors state that they consistently observe existing PET methods selecting highly influential parameters, which have approximately ($10^4 \times$ ) higher PPI than the majority of model parameters. Why is this the case? The PPI metric is an estimation of parameter influence, so what is the corresponding ground truth for parameter influence? Without ground truth, how can we be certain that PET methods have indeed selected the highly influential parameters?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes Hessian informed differential learning rates for deep learning, which is derived based on the second-order Taylor approximation of the loss function. Experiments on synthetic data and real data are conducted to demonstrate the effectiveness of the proposed learning rates.

### Strengths
1. The authors conducted extensive experiments covering multiple different settings, from synthetic data and toy objective functions to various advanced practical learning tasks.

### Weaknesses
1. The novelty of the proposed method is questionable. The idea is very similar to the derivation of Newton’s method, and the approximation of the Hessian with the diagonal matrix is not novel either, for example

Andrei, N. A diagonal quasi-Newton updating method for unconstrained optimization. Numer Algor 81, 575–590 (2019). https://doi.org/10.1007/s11075-018-0562-7

2. The logic of the proposed method is not convincing. The authors mentioned in the caption of Table 1 that they apply the proposed learning rates to AdamW. However, the proposed learning rates are derived based on equations (2.1) and (2.3), where the vector $\mathbf{d}$ in (2.1) is replaced by  $ \boldsymbol{\eta}\_{[K]} \mathbf{g}\_{[K]}^{\mathrm{optim}} $. If the learning rates are applied to AdamW, why shouldn’t we set the vector $\mathbf{d}$ in (2.1) to be the actuarial difference between iterates of AdamW, taking the entry-wise adaptive learning rates of AdamW into consideration? I think the notation $ \mathbf{g}\_{[K]}^{\mathrm{optim}} $ is not explained well either. I suggest that the authors should clarify how exactly their method integrates with AdamW.

3. The paper does not provide sufficient experimental details. Although some are provided in the appendix, the information provided are insufficient to reproduce the results. For example, I do not find any experimental details about the training of the ViT. For example, did the authors consider data augmentation? What is the batch size for training ViT, and what exact version of ViT is used on the various data sets? (e.g., what is the classifier, what are the dimensions of heads, is dropout used, how many patches are each image split into, etc). The authors provide no codes either. 

4. For relatively large models, as far as I can tell, the paper only considers applying the proposed learning rates for fine-tuning. It is important to also demonstrate the performance of the proposed method in pre-training. Therefore, I suggest that the authors should include experiments on pre-training certain models such as ViTs.

5. The presentation of the paper is not clear enough, and the paper needs some significant revision. As mentioned in the points above, there are notations and experimental setups that are not explained clearly. Moreover, I feel that this paper lacks a proper introduction section.

### Questions
I suggest that the authors should respond to the weaknesses pointed out above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces a novel method called Hessian-informed Differential Learning Rate (Hi-DLR) to optimize the training of neural networks by adapting learning rates based on the curvature of the loss function, which enhances convergence across various tasks. It highlights the limitations of existing Parameter-Efficient Tuning (PET) methods, demonstrating that no single PET method is universally effective, as performance varies significantly with different datasets and model architectures. The authors propose a flexible, model-agnostic meta-framework that adaptively selects the most effective PET methods and parameter groups based on their Per-Parameter Influence (PPI) during training.

### Strengths
The paper presents the Hessian-informed Differential Learning Rate (Hi-DLR) method, which enriches the approximation of Hessian information to leverage the varying loss curvature of different parameters through adaptive learning rates, enhancing the training efficiency of deep learning models .

The authors propose an efficient algorithm for computing Hi-DLR, incorporating a novel diagonalization technique that significantly reduces computational costs while effectively separating the contributions of different parameter groups, thus facilitating faster training without sacrificing performance .

The paper introduces a flexible, model-agnostic meta-framework for Parameter-Efficient Tuning (PET) that utilizes per-parameter influence to dynamically select trainable parameters. This adaptive approach allows for improved performance across various tasks and models, addressing the limitations of existing PET methods

### Weaknesses
1. This method includes other time-consuming steps and is incomplete without measurement. Under the same conditions, would a standard AdamW approach achieve a similar result?


2.The effectiveness of Hi-DLR depends on appropriate parameter grouping; suboptimal groupings can lead to less effective learning rate adjustments, potentially hindering performance. In the paper, parameter groups are adjusted manually, which requires explanation. Additionally, for large models, the hyperparameter K warrants further discussion.

3.When group number K is a large number, the update period will increase as its learning rate update every \phi iterations, which might influence the convergence.

### Questions
Could the authors provide the source code for reproduction?

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
3

### Summary
The main idea of this paper is to use different learning rates for different parameter groups. This is a common knowledge among optimizer designs that for different part of the model, e.g., the weight, bias, head, norm, etc., need different learning rates, however, how to search for the best learning rates remains open. This paper proposes to use the second-order approximate of the loss function to solve for the adaptive learning rates. In addition, the second-order approximation are done by regression on random sampled directions. The proposed algorithm significantly enhances training efficiency. Moreover, the paper introduces a per-parameter influence index, which identifies the most influential parameters, facilitating more efficient parameter-specific training.

### Strengths
Well-written, the intuition of this paper is clear, if the shape of a convex optimization surface is given, we can obtain the optimal learning rates for each directions directly. This can be done on top of any gradient momentum, and preconditioning regularizations. 

Although the dimension of the parameters are large, the authors pick a few directions in the space (K directions) to get a low dimensional second-order approximation for learning rate searches.

The idea is simple but effective.

The authors also designed a per-parameter influence, that can differentiate the most influencial parameters for training. By freezing all other parameters under a PPI threshold, the authors can largely reduce the training cost of models, while preserving most of the testing accuracies.

### Weaknesses
Although everything diagonalized, the computational cost of the search is relatively high when fitting the second-order model, especially when $K$ is large.

It seems to me that the algorithm in principle is partly based on search, that you first move towards K directions defined by the split of K parameter groups, and than decides the curvature of each direction (fit a second order function) to get a learning rate on the parameter groups. I wonder if the algorithm can actually be designed more directly, for example instead of randomly sampled $\eta$s, fix several points on K lines, and than estimate curvature, make the learning rate on a line the negative curvature, etc. 

The authors discussed the empirical comparison with other adaptive learning rate algorithms like prodigy and adaptation, etc. however, they did not discuss the relationship of their adaptation with previous adaptive algorithms in principle.

### Questions
$d_k$ is used both as an update direction, and as a notation of dimensionality in your paper.

In Equation (3.1), and when fitting the quadratic function, are you using the gradient calculations, or the corrected gradients like the momentum gradients in Adam? Does that mean second order Taylor expansion on any direction? It would be better if the authors could discuss how does their adaptive gradient interact with Adam's momentum, and adaptive schedule. Are they completely orthogonal, or how they influence each other?

For LoRA, the optimal learning rates ratios for matrix A and B can be fixed (Hayou et al., 2024) . How does your schedule interact with this ratio? If $\lambda$ is fixed, then your algorithm cannot work further on Lora+?

### Soundness
3

### Presentation
4

### Contribution
3
