# Bigger Isn’t Always Memorizing: Early Stopping Overparameterized Diffusion Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Diffusion probabilistic models have become a cornerstone of modern generative AI, yet the mechanisms underlying their generalization remain poorly understood. In fact, if these models were perfectly minimizing their training loss, they would just generate data belonging to their training set, i.e., memorize, as empirically found in the overparameterized regime. We revisit this view by showing that, in highly overparameterized diffusion models, generalization in natural data domains is progressively achieved during training before the onset of memorization. Our results, ranging from image to language diffusion models, systematically support the empirical law that memorization time is proportional to the dataset size. Generalization vs. memorization is then best understood as a competition between time scales. We show that this phenomenology is recovered in diffusion models learning a simple probabilistic context-free grammar with random rules, where generalization corresponds to the hierarchical acquisition of deeper grammar rules as training time grows, and the generalization cost of early stopping can be characterized. We summarize these results in a phase diagram. Overall, our results support that a principled early-stopping criterion - scaling with dataset size - can effectively optimize generalization while avoiding memorization, with direct implications for hyperparameter transfer and privacy-sensitive applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper analyze the learning dynamics of highly overparameterized diffusion models, which are supposed to memorize training data when sufficiently optimized. However, during training they first **generalize before memorizing/overfitting**. Empirically, the authors investigate image and language diffusion models on small datasets and consistently observe this early-stopping generalization. In this regime, the model achieves imperfect generalization with relatively lower validation loss, novel but lossy generations, and partial reproducibility. Motivated by these observations, the authors propose an early-stopping metric $\tau_{\text{mem}} \propto P$.

Finally, the authors provide a Random Hierarchy Model (RHM) perspective on this early-stopping generalization: according to previous work, to learn the $\ell$-th layer in the RHM model, one needs exponential data size $m^{\ell+1}$. With limited samples, the model only learns lower-level structures, which prevents full generalization.

### Strengths
1. The authors investigate the learning dynamics [1] of diffusion models on both images and language, providing insights into generalization vs. memorization and the training process of diffusion models. They also run a broad set of experiments supporting their arguments.
2. With the RHM model, the authors aim to characterize learning dynamics as learning different levels of dataset structure, i.e., **how much data is required to learn a given structural level**, aligning with coarse-to-fine learning behavior [1].

[1] Wang, Binxu. *An analytical theory of power law spectral bias in the learning dynamics of diffusion models.* NeurIPS 2025.

### Weaknesses
1. **A comprehensive and practical ablation on $\tau_{\text{mem}}$ is missing.** For instance, I would expect a clear scaling/regression plot for $\tau_{\text{mem}}$–$P$ validating the linear relationship, and analysis of how the coefficients depend on the data distribution and model size.

2. **The RHM theory does not fully justify memorization.** It explains partial generalization before memorization, but several definitions are missing: What is an empirical version of RHM data? How are errors at different levels $\ell$ in Figure 5 computed, and are they train or test errors? There is no rigorous distinction between empirical and population losses within RHM. 

   As a result, The kernel-regression setup (Sec. 3.3) introduced to explain memorization and the linear dependency $\tau_{\text{mem}} \propto P$ feels disconnected from the RHM story.

3. **Some claims are not fully supported by experiments.** “At some time $\tau_{\text{mem}}$, the models begin to diverge. This divergence coincides with the onset of memorization” (L200–202). In Figure 2, inter-model similarity decreases monotonically, while similarity to training data increases monotonically; $\tau_{\text{mem}}$ is not clearly special. “With direct implications for hyperparameter transfer and privacy-sensitive applications” (L26–27). I do not see a straightforward justification—please elaborate (e.g., via Stable Diffusion experiments or deeper analysis as in point 1).

### Questions
1. In the caption of Figure 1, you state “$\tau_{\text{mem}}$ scales approximately linearly with $P$.” Is this driven by using equal intervals for $P=\{2048,4096,8192,16384\}$?
2. How does $\tau_{\text{mem}}$ change with different optimizers and schedules (e.g., AdamW vs. Adam, warmup/cosine)? This seems crucial for practicality.
3. For language diffusion models, can you provide some generated text samples (pre- and post-$\tau_{\text{mem}}$) for illustration?

*My current rating is a provisional assessment and may be updated after author responses and discussion with other reviewers.*

### Soundness
2

### Presentation
2

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
The paper investigates training dynamics of diffusion models with respect to generalization and memorization. Empirically authors show that diffusion models first learn to generate samples from *entire* data distribution (generalization), and after certain point it learns to generate samples from *training* data distribution (memorization).

### Strengths
- The presentation is clear.
- The fact that diffusion models first try to generalize before memorizing is a new observation. 
- The authors conducted extensive experiments on various modalities.

### Weaknesses
- As in [Deep Double Descent](https://arxiv.org/abs/1912.02292), the number of training epochs is also included in training capacity. Hence the fact that memorization time and dataset size having linear dependency is not very surprising. 
- The observation might not be very practically applicable, because most practical vision diffusion models are trained on very large dataset. Also since it's been reported that memorization happens at the *concept* level, it would be very hard to quantify *validation error* and hence the correct *early stopping point*.
- The paper is primarily scientific report, where most contents align with the existing perspective.

### Questions
- Is the FID score at $\tau_\text{mem}$ comparable to FID score of same model trained on whole dataset?
- If so then would this imply that even 2048 images are enough to represent the whole cifar10 dataset?

### Soundness
4

### Presentation
4

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
This paper investigates the dynamic transition from generalization to memorization of diffusion models during the training process. The authors demonstrate across image and language models that generalization is, in fact, achieved progressively during training before the onset of memorization, finding an empirical law that the memorization time is proportional to the dataset size. Ultimately, the results suggest that generalization and memorization are distinct temporal phases, implying that a principled, dataset-size-aware early-stopping criterion can be an optimal strategy for preserving generalization and avoiding memorization in large diffusion models

### Strengths
Strengths:
* Besides image generation, they also study the memorization of the masked diffusion model in the text modality, which is novel in the generalization-memorization field to me.
* They use the reproducibility of two different models trained over two disjoint datasets to show that the score function attempted to learn the real underlying distribution at the early stage.
* The random hierarchy model further provides some interesting insights to the learning process, such as the partial generation.

### Weaknesses
Weaknesses:
* In Section 3.1, you showed that the transition point $\tau_{mem}$ scales approximately linearly with the training set size. Do you think the distribution complexity, such as the intrinsic dimension of the data and the entropy of the data, also influences the transition point? Besides, for the latent diffusion model and pixel diffusion model, is there any differences on the transition point? Including more factors into your study would make this work more thorough and robust.
* The key claim of this paper is that the model first generalizes at an early stage but then memorizes after $\tau_{mem}$. But I am concerned about calling the first stage generalization. Although the validation loss indeed decreased in the first stage, it may still be too high, and the score function hasn’t learned a good distribution. As you also visualized in Figure 2 (right), the generated images before $\tau_{mem}$ have bad quality and thus cannot be treated as good generalizations. Then, the early stopping strategy fails in this case. I feel that the early stop is effective only when both the sample size and network size are large, which is also supported by Figure 6.
* The dynamic transition and the linear relation between $\tau_{mem}$ and dataset size have also been revealed in prior work [1]. Could you compare the novelty of your work?

[1]: Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training. https://arxiv.org/abs/2505.17638

### Questions
Do you also visualize the partial generalization in real images? Is there any prior work proposing hierarchy data frameworks for images?

### Soundness
3

### Presentation
3

### Contribution
2
