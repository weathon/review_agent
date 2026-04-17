# Sample-Efficient Differentially Private Fine-Tuning via Gradient Matrix Denoising

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
We address the challenge of sample efficiency in differentially private fine-tuning of large language models (LLMs) using DP-SGD. While DP-SGD provides strong privacy guarantees, the added noise significantly increases the entropy of gradient matrices, disrupting their low-rank structure and slowing optimization. We propose a post-processing algorithm that leverages random matrix theory to denoise gradients, restore low-rank structure, and improve alignment with the original signal. Applied to DP-SGD fine-tuning of RoBERTa model family on GLUE tasks and Qwen and Llama family on DART and E2E datasets, our method improves sample efficiency compared to state-of-the-art approaches, substantially reducing training time when optimal performance is not required. This work demonstrates that matrix recovery techniques can enhance the utility of private language model training without compromising privacy guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies that DP noise disrupts the inherent low-rank structure of gradient matrices. They propose a post-processing method that uses Random Matrix Theory to denoise the gradients on a per-layer basis. The core idea is to identify and reconstruct the signal singular values that emerge from the noise. The authors' experiments on RoBERTa models show that this method can achieve target validation accuracies in significantly fewer training steps, although it does not always achieve the best final accuracy.

### Strengths
The paper is well-written, and the proposed method is described with sufficient detail to be easily followed and understood. Besides, the authors provide a practical and honest analysis of their method's performance.

### Weaknesses
W1: The method's effectiveness is doubtful due to theoretical contradictions, such as the "Norm Correction" (Step C) deviating from the RMT optimal estimate (Step B). This correction provides an inflated gradient magnitude that could harm optimizers, and the method's reliance on an empirically-tuned $\kappa$ (Step A) further questions its robustness.

W2: The paper claims improved "sample efficiency" but ignores the massive computational overhead of performing SVD on every linear layer at every step.

### Questions
Q1: Could the authors provide a concrete comparison of the total training time or the average time per training step against the baseline? 

Q2: How sensitive is the algorithm's final performance to the choice of the hyperparameter $\kappa$? 

Q3: Could the authors provide an ablation on the performance without the $\kappa$-thresholding  and norm correction (Step C) to isolate its effects?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies private fine-tuning of LLMs using DP-SGD. Compared to the standard SGD, DP-SGD adds clipping and Gaussian noise in order to protect privacy. Both would affect the model performance and decrease training efficiency. This paper mainly addresses the part with added Gaussian noise. Motivated from random matrix theory, the paper proposes to add a post processing step after clipping and noise addition to recover original gradients from noisy observations. The new methods can reach 90% or 95% of the state-of-the-art performance with reduced training steps when fine-tuning RoBERTa on GLUE tasks.

### Strengths
I think adding a post processing step to denoise the gradients is a good idea. The motivation and the design of the algorithm are presented clearly. The studied topic is interesting to the community, as fine-tuning LLMs indeed poses privacy concerns. Introducing efficient algorithms for private fine-tuning of LLMs is of great importance.

### Weaknesses
1. SVD is always required for the denoising step to compute all singular vectors and singular values of gradient matrices. This adds significant amout of computational overhead. Although the paper claims on sample efficiency, computational efficiency is also or even more important in many cases. What is the total running time to reach the claimed performance? If per-step cost is significantly higher, it is unfair and potentially misleading to only compare number of steps.

2. It turns out random matrix theory provides good solutions but mainly works for the asymptotic case. There are lots of approximations and additional fixes to make it work in pratical non-asymptotic regime. I doubt whether the denoise function used in the paper is still optimal. The fixes by $\kappa$ and norm rescaling also look artificial and fragile to me. These fixes make the method seem less like a clean solution but more like a heuristic that requires careful tuning and may not be robust if a different model and a different tasks are used. Moreover, should the noise in $\kappa \sigma (\sqrt{n}+\sqrt{m})$ be $\sigma C/|B|$? Is this a typo or a wrong algorithm is used?

3. I do not think the goal in the experiment section to match 95% of SOTA with less steps makes sense. First, as said in 1, running time should also be considered. Second, in my opinion, the more important problem to be addressed in DP fine-tuning is the reduced performance due to privacy noise. Designing algorithms that achieves better trade-off between utility and privacy, i.e., better performance under the same privacy budget, is more interesting. Can the proposed algorithm reach or surpass the performance of SOTA with the same number of iterations or with the same training time?

4. The experiments are very limited. Only 4 tasks are considered on a single RoBERTa family. It is unclear whether the algorithms also work for other model family, other tasks, and scale to larger models.

### Questions
1. Are Figures 1 and 2 required to be this large? More interesting results can be presented if this space is saved.

2. The notation on Gaussian noise is not consistent and gives lots of confusion. In line 145 and eq. (2), it is $N(0, \sigma C)$ but $N(0, \sigma^2 C^2)$ in many other places.

3. What is the purpose of Section 3.1? Is Improvement(t) ever used? This metric is not private and could leak sensitive information.

4. Improvement in cosine similarity does not directly imply the algorithm should perform better. One reason is that there is still per-sample gradient clipping.

### Soundness
2

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
2

### Summary
This paper proposes a post-processing denoising method for DP fine-tuning of LLMs using DP-SGD. By applying random matrix theory-based denoising (singular value shrinkage and norm correction) to the noisy gradients of linear layers, the method aims to restore low-rank structure and improve sample efficiency. The approach is evaluated on GLUE benchmarks with RoBERTa models, showing consistent reductions in the number of training steps needed to reach target accuracy under DP constraints. The method leverages the post-processing invariance of DP to preserve privacy guarantees.

### Strengths
-	This paper addresses an important and practical challenge: improving sample efficiency in DP fine-tuning of LLMs.
-	The authors applies theoretically motivated random matrix denoising to the DP-SGD context, with clear exposition and solid background. The method is modular, simple to implement, and preserves DP guarantees via post-processing.
-	Empirical results on GLUE tasks and RoBERTa models show consistent speedup in convergence (steps to reach target accuracy).

### Weaknesses
-	The method does not always improve final accuracy; in some cases, the baseline outperforms the proposed approach, and this is not analyzed in depth.
-	The threshold hyperparameter (κ) is tuned on a single dataset; robustness and generalization are not validated.
-	The method only addresses noise-induced degradation, not the often more severe impact of gradient clipping in DP-SGD.

### Questions
- How does the method compare to alternative DP optimization or post-processing strategies?
- How robust is the method to the choice of threshold κ and norm correction? Did you tune these on each task/model or fix them universally?
- How does the method compare to alternative DP optimization or post-processing strategies?

### Soundness
3

### Presentation
2

### Contribution
3
