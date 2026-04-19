# Learning Structured Sparse Neural Networks Using Group Envelope Regularization

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3

## Abstract
We propose an efficient method to learn both unstructured and structured sparse neural networks during training, utilizing a novel generalization of the sparse envelope function (SEF) used as a regularizer, termed {\itshape{weighted group sparse envelope function}} (WGSEF). The WGSEF acts as a neuron group selector, which is leveraged to induce structured sparsity. The method ensures a hardware-friendly structured sparsity of a deep neural network (DNN) to efficiently accelerate the DNN's evaluation. Notably, the method is adaptable, letting any hardware specify group definitions, such as filters, channels, filter shapes, layer depths, a single parameter (unstructured), etc. Owing to the WGSEF's properties, the proposed method allows to a pre-define sparsity level that would be achieved at the training convergence, while maintaining negligible network accuracy degradation or even improvement in the case of redundant parameters. We introduce an efficient technique to calculate the exact value of the WGSEF along with its proximal operator in a worst-case complexity of $O(n)$, where $n$ is the total number of group variables. In addition, we propose a proximal-gradient-based optimization method to train the model, that is, the non-convex minimization of the sum of the neural network loss and the WGSEF. Finally, we conduct an experiment and illustrate the efficiency of our proposed technique in terms of the completion ratio, accuracy, and inference latency.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper introduces a regularizer called Weighted Group Sparse Envelope Function that is used with a proximal gradient type of algorithm to sparsify neural networks.

### Strengths
Addresses the important problem of sparsifying neural networks.

### Weaknesses
The literature review only briefly touches previous works that directly optimize the L0 penalty and only cites old papers from before 2017 and omits more recent works in this direction. See below for one example.

The description of the method has problems, raising reproducibility issues:
- It is unnecessarily complicated and could confuse the reader, For example, in eq (13) theta^T A_s_j theta is just the L2 norm of the theta on index set s_j.
- It has undefined terms. For example SEF is referred in the introduction as the sparse envelope function, but its definition is never given, but is referred to as s_k^**  in Remark 1 and later, and s_k^** is never defined.

The paper is missing a Conclusion section and a discussion of the weaknesses of the proposed method.

Experimental evaluation is lacking in many respects:
- It is not clear what the AdaHSPG+ algorithm is because it is not defined.
- Experiments with comparisons are done only on two easy datasets. More challenging datasets such as CIFAR-100 or Imagenet are not included.
- The comparison is only with group Lasso based methods, relying on a statement from a Bui et al 2021 paper claiming that they work best. No works after 2021 that are not based on group Lasso are included in the evaluation. For example, Guo et al "Network pruning via annealing and direct sparsity control", IJCNN 2021 obtains better results on CIFAR-10 with VGG16.

### Questions
- How does the proposed method perform on CIFAR-100?
- Lasso based methods are known to bias the weights while imposing sparsity, obtaining suboptimally trained NNs. How is that avoided in the proposed approach?
- How are the regularization parameters lambda_l chosen to agree with the desired sparsity k?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a regularization approach, the Weighted Group Sparse Envelope Function (WGSEF), for inducing structured sparsity in neural networks, facilitating hardware-accelerated deep learning. It introduces an efficient calculation method for WGSEF and a proximal-gradient optimization technique to train sparse models. The paper's experimental results demonstrate the method's ability to maintain or improve accuracy while achieving computational efficiency.

### Strengths
- The paper introduces WGSEF, a novel generalization of the sparse envelope function for structured sparsity.

- It provides an efficient technique for the calculation of WGSEF and its proximal operator.

### Weaknesses
- The paper's presentation and writing quality require improvement; the experimental section ends abruptly, seems the paper may be incomplete.

- The paper uses benchmark datasets like CIFAR10 and Fashion-MNIST, and benchmark architectures like VGG16, ResNet18, and MobileNetV1, which are standard choices for evaluating the performance of deep learning methods. However, these benchmarks might not fully reveal the practical efficiency and applicability of the proposed WGSEF regularization method in real-world, large-scale problems. For example, [Chen et al. 2021] tests its HSPG in structural prune ResNet on ImageNet, which could provide a stronger testament to the method's practicality.

- The process for grouping parameters before regularization is not specified. An exploration of automated grouping algorithms, such as in

Chen, Tianyi, et al. "OTOv2: Automatic, Generic, User-Friendly." The Eleventh International Conference on Learning Representations. 2022.

could be beneficial when applied with WGSEF.

### Questions
see the above weaknesses part

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a weight regularization in deep learning with the goal of improving the hardware efficiency of neural networks. Specifically, they propose a method to learn structured and unstructured sparsity pattern in weights of neural network. They base their work on the sparse envelope function and extend it to the group case.

### Strengths
The paper provide details formulation and derivation of their method. Their experimental results show improvement against baselines based on Group Lasso to increasing group sparsity ratio.

### Weaknesses
- This work extend the sparse envelope function (Back & Refael (2022)) into group sparsity. So, it appears as an extension which is applied to deep learning training. To evaluate their theoretical contribution could the authors comment on how their analysis differs from the SEF? Perhaps, comment on the challenges in the extension from SEF to the group case. For example, some corollary are direct extension of derivations already done by SEF paper.

- What is the performance of WGSEF (algorithm 2) alone. This comparison is needed with WGSEF + HSPG to determine the advantages of proposed WGSEF.

- To evaluate the results, experiment are needed on popular deep learning architectures (Unet, Transformers). Please include standard deviation over the reported numbers. LeNet is a very basic network. Please update results with a neural network commonly used in the literature. Moreover, characterization of the method is missing. A figure on how the performance varies as a function of the hyper-parameter on the sparsity. This would be helpful for sensitivity analysis of the method. 

- Almost all deep learning training are done with variants of Adam optimizer. I wonder if the authors could provide analysis on Adam as opposed to SGD.

- Literature review is not comprehensive. For example, literature on l1 convex relaxation to l0 is missing (one example [1]). It is not clear how the contribution of this work differs from prior work on inducing group sparsity for efficient neural network architecture.

- The organization of the paper needs improvement. Coherency can be improved. Here are some points. For example, in Section 5, the authors start talking about SGD which was not the main focus on the paper till this Section. Or again, the end of Section 5, the paper discusses some related works that are not relevant to the proposed method. I found many statements of the paper not related to the main theme of the paper, and also re-statement of results from prior works (e.g., Corollary 5.0.1). Moreover, I recommend to shorten the findings of Beck & Refael (2022) in the intro. Not clear why all those details are included in the intro before main contribution. Around (1) and (2), please mention that l0-norm is a pseudo norm. In find algorithm 1 redundant (Comparing to 2). Labels of figures are very small. Please increase the fontsize, and also use the table formatting suggested by the conference. I do find (1) and (2) unnecessary in the intro as the paper is based on SEF. Please justify why these equations must be there? I recommend to remove.

Other comments.

- Could the authors explain how the group SEF (their method) differs from group lasso? I find them related; however, the formulation no where refers to lasso or l1 type regularization.

- I thank the authors for their thorough related works; however, this introduction is missing a statement on "how their method differs from prior works". Please comment.

- Second paragraph of intro on generalization. Please also include advantages of over-parameterization. Otherwise, the sentence read a biased view. overparameterization does not result in an overfit (there is a double descent) if the amount of training data is very large. 

- Please elaborate what the  "complete inverse problem" is (in the intro).

- Section 5 has related works on SGD which breaks down the flow of the paper. Could the authors explain why such related works on SGD is provided there? Recommend to move the paragraph after Algorithm 2 into appendix.

- Which experiment (dataset) results of Table 1 refer to.


[1] Donoho, D. L., & Elad, M. (2003). Optimally sparse representation in general (nonorthogonal) dictionaries via l1 minimization. Proceedings of the National Academy of Sciences, 100(5), 2197-2202.

### Questions
see above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
