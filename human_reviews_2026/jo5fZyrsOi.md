# Gradient-based Dynamic Sparse Training with Adaptive Rewinding

- Avg Score: 3.00
- Decision: Reject
- Scores: 6, 4, 0, 2

## Abstract
Deep neural networks (DNNs) deliver state-of-the-art performance across domains but impose prohibitive computational and memory costs. Pruning mitigates this challenge by removing unimportant parameters, yet conventional post-training pruning and reset-to-initial sparse training approaches incur high retraining costs or degrade performance on large models. To improve stability, prior post-training work suggests rewinding weights to intermediate checkpoints, though at the expense of costly offline analysis. We propose GDSTAR, a Gradient-based Dynamic Sparse Training framework with Adaptive Rewinding that supports models of different sizes and complexities without offline retraining. During training, GDSTAR (1) dynamically identifies stable rewind points using the Frobenius norm of gradients, (2) selects weights for pruning using accumulated gradient magnitudes, and (3) ensures stable optimization using a controlled pruning rate with exponential decay. Experiments across diverse DNNs and datasets show the efficiency and scalability of GDSTAR, which achieves up to 96% sparsity while maintaining accuracy, with only a 0.94% average drop compared to dense models. Compared to the state-of-the-art sparse training approach, GDSTAR improves accuracy by an average of 0.72% (up to 2.13%), under the same sparsity ratios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces GDSTAR (Gradient-based Dynamic Sparse Training with Adaptive Rewinding), a sparse training framework that dynamically identifies optimal rewind points and prunes unimportant weights during training. The method leverages the Frobenius norm of gradients to determine stable rewind points without costly offline analysis and uses accumulated squared gradients to select weights for pruning. GDSTAR applies a decay-based pruning schedule to progressively reduce weights, maintaining training stability and model performance.

### Strengths
The method is well-motivated, providing a unified approach to integrate pruning and adaptive rewinding directly into training, reducing offline retraining costs.
Using the Frobenius norm of gradients as an online stability measure is novel and effectively avoids expensive multiple training runs required by prior rewinding methods.
The algorithm is theoretically coherent and can be integrated easily into standard training pipelines without architectural changes.

### Weaknesses
Experiments are limited to vision classification tasks on small-scale datasets (MNIST, CIFAR-10), leaving uncertainty about the method’s scalability to large-scale datasets or transformer-based architectures.
Some hyperparameter choices (e.g., decay rate, pruning rate) appear heuristic and lack theoretical justification. The paper does not discuss whether the proposed hyperparameters remain effective across different networks and tasks.
The comparison to state-of-the-art sparse training or pruning methods is narrow. The recent related works [1][2][3][4][5][6][7][8] should be compared and discussed.

[1] Rigging the Lottery: Making All Tickets Winners, ICML 2020
[2] Sparse Training via Boosting Pruning Plasticity with Neuroregeneration, NeurIPS 2021
[3] Top-kast: Top-k always sparse training NeurIPS 2020
[4] AC/DC: Alternating Compressed/DeCompressed Training of Deep Neural Networks, NeurIPS 2021
[5] Dynamic Sparsity Is Channel-Level Sparsity Learner, NeurIPS 2023
[6] Advancing Dynamic Sparse Training by Exploring Optimization Opportunities, ICML 2024
[7] NeurRev: Train Better Sparse Neural Network Practically via Neuron Revitalization, ICLR 2024
[8] A Single-Step, Sharpness-Aware Minimization is All You Need to Achieve Efficient and Accurate Sparse Training, NeurIPS 2024

### Questions
How does GDSTAR perform on larger-scale datasets (e.g., ImageNet) or non-convolutional architectures like Transformers or LLMs?
Can the authors provide concrete wall-clock or FLOP-based speedup results to quantify the real efficiency gain?
How sensitive is the method to hyperparameters such as pruning rate and decay rate, and can these be adapted automatically?
Could the gradient-based rewind and pruning mechanism generalize to structured or hardware-friendly sparsity patterns?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a method to dynamically train sparse networks. The key problem being addressed here is that of the high cost associated with post-training pruning methods. To eliminate post-training pruning, the authors propose a Frobenius norm-based method to determine optimal rewind checkpoints, identify which set of weights to prune, and at what rate the pruning should occur. The authors conduct experiments on multiple neural networks and on two datasets to show the affect of various hyperparameters on the performance of GDSTAR.

### Strengths
1) The idea of using gradients Frobenius norm to determine the checkpoints is a very interesting one and intuitively makes sense.
2) The authors have done a good job of providing enough background and motivation to appreciate their work. 
3) The method does seem to outperform the baseline on a variety of networks and at very high levels of sparsity.
4) The analyses in Section 5 and the Appendix (e.g., impact of prune rate, decay rate, and chunking) are well-executed and provide valuable insights into the method's hyperparameters.

### Weaknesses
1) From the description of the methodology, it seems that a lot of work has to be done to obtain pruned networks, which can potentially increase the training time.
2) This is similar to the above point, but I am not convinced if the method is scalable to larger networks or datasets.
3) I think the figures in the paper are really interesting and can help drive the point home, but currently, they are a bit too small for a reader to engage with them.

### Questions
Overall, I appreciate the paper and the ideas it presents. I think answering the questions below can make the paper a lot more solid and well-rounded:
1) What is the overhead associated with GDSTAR?

a) An analysis or at least a few paragraphs on the training time and the memory cost of the method could help the readers make a better judgment if they would like to apply this approach to their use case or not. For example, at a sparsity level, a plot with accuracy on the x-axis and training time/memory cost on the y-axis would provide a clear picture of all the methods being compared to GDSTAR.

b) Again, an analysis or a few paragraphs on how the GDSTAR approach, in its current form, will scale compared with larger models and/or datasets?

2) For the accuracy reported in Table 2, is it for a single run or multiple runs? If multiple runs, then having standard deviations would be useful.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes a method of "dynamic sparse training" using an online method to determine the rewound point, i.e. as used for the Lottery Ticket Hypothesis, but without the cumbersome and extremely compute expensive offline methodology of the LTH weight rewinding. The method appears to be an iterative pruning method aside from that, using a pruning schedule over training. The authors evaluate the method on MNIST with extremely tiny "LeNet" models from the LTH paper, and ResNet-50/ResNet34/VGG-19 on CIFAR10. The results show marginally better text accuracy for the proposed method than simply training from initialization.

### Strengths
* Weight rewinding in the LTH literature is a fascinating research topic, with much to contribute in understanding and removing the exhaustive search usually done to find the rewind point. Any progress on that front, as this paper claims to make in terms of finding an-online method of identifying a rewind point in dense training, is interesting.
* The authors seem to have some familiarly with NN pruning and sparse training work from the systems community that I'm not familiar with as a sparse training researcher within the ML community, and it's always interesting to learn of related work across fields.

### Weaknesses
* The motivation of this paper and the results presented do not make much sense in the context of the state-of-the-art in dynamic sparse training (DST) literature they claim their method belongs to, or indeed even just generally the state-of-the-art (SOTA) in sparse training. Existing DST methods do not require or use weight rewinding, while demonstrating better sparsity and generalization results than the authors present on larger models and datasets.
* The background cites none of the predominant dynamic sparse training work over the last 5 years, never mind the SOTA work from within the last year. For example, Sparse Evolutionary Training (SET), Rigging the Lottery Ticket (RigL), MEST, Gradual Magnitude Pruning . The mainly cited paper (McDanel et al) is an arxiv preprint from 2022 that appears to have not been published anywhere, and outlines a pruning (not sparse training) method.
* The paper relies on evaluation on only tiny toy datasets (MNIST, CIFAR10), and in the case of CIFAR-10, completely inappropriate and over-parameterized models for CIFAR-10 (VGG-19, ResNet-34 and ResNet50. Even in this setting the results do not achieve as high a sparsity or generalization on CIFAR-10 as demonstrated by existing DST methods (e.g. SET/RigL).
* On the only part of the paper I believe is anywhere near a novel and interesting contribution, online weight rewinding, the authors compare their online weight rewinding method - i.e. choosing weights from an iteration k - and compare those to the baselines of random initialization (k=0) and dense training (full training). A proper evaluation would compare with the exhaustive weight rewinding of LTH the authors motivate their work with. Using any weight rewinding (i.e. k>0) is going to surpass the generalization of k=0 trivially.

I can understand missing one or two citations, or coming from a different research community and being unaware of SOTA papers. This however can only be described as the willful ignorance of an entire field of research clearly demonstrated by both citing an excellent and comprehensive survey paper of the field of sparse training (Torsten Hoefler et al.) while co-opting the terminology within that survey paper, citing none of the work from it, and writing a motivation that makes no sense in context of what is written in the survey paper alone.

### Questions
* How does your method compare to SOTA dynamic sparse training methods such as RigL, SET and MEST? 
* How does your online rewinding method compare to more appropriate baselines, including the exhaustive LTH weight rewinding, and perhaps even a random k weight rewinding point.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes GDSTAR, a Gradient-based Dynamic Sparse Training framework with Adaptive Rewinding, aiming to achieve efficient sparse training without offline retraining.
Experiments on 11 DNNs (LeNet, ResNet, VGG, EfficientNet, etc.) over MNIST and CIFAR-10 show that GDSTAR achieves up to 96% sparsity with an average 0.94% accuracy drop, outperforming prior dynamic sparse training methods like Procrustes by 0.72% on average.

### Strengths
1. Proposes an online adaptive rewinding mechanism using the Frobenius norm of gradients.
2. Well-motivated and mathematically clear.
3. Demonstrates performance gains.

### Weaknesses
1. Experiments are focused on MNIST and CIFAR-10, which are very small datasets. It would be good to have the experimental results on ImageNet.
2. The paper does not provide quantitative runtime or FLOPs comparisons.
3. The experiments only compare the proposed method with Procrustes, but no other sparse training baselines.

### Questions
1. Could GDSTAR be extended to ImageNet-scale models or non-CNN architectures (e.g., transformers or ViTs)?
2. How often is the optimal rewind point updated? Is frequent updating beneficial or redundant after early stabilization?
3. How much additional time or memory does online Frobenius-norm computation add compared to Procrustes or standard training?

### Soundness
3

### Presentation
2

### Contribution
2
