# A Guide to Training Consistency Models

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
While the theoretical foundations of consistency models are well-understood, their practical implementation is often hindered by complex and entangled training pipelines. The interplay between critical components is not always transparent, making systematic improvement difficult. To address this, we propose a practical training playbook for consistency models. Our approach begins with a na\"ive baseline and proceeds to deconstruct the training process, isolating and examining the impact of key modules: time step discretization, time conditioning, loss weighting, time sampling strategies, the auxiliary task with variable upper limit, and distribution-level losses. This modular analysis provides a clear view of how each element contributes to the overall performance. Following this guide, we demonstrate the ability to build models that achieve both state-of-the-art results and substantially faster convergence. Notably, for the first time, consistency models trained from scratch now surpass the leading EDM diffusion model on CIFAR-10 under the same network architecture. They achieve a remarkable $1$-step FID of $2.53$ and a $2$-step FID of $1.92$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes several training techniques to improve the sample quality of consistency models, including time step discretization, time preconditioning, time weighting and distribution, etc. The authors evaluate their method on CIFAR-10.

### Strengths
Improving the discrete consistency model (CM)’s sample quality is an important topic, as it does not require Jacobian-vector products (JVPs). Continuous-time CMs rely on JVPs, which are not well supported by ML libraries such as PyTorch and thus add additional implementation burden.

### Weaknesses
My biggest concern is that the paper starts from a naive baseline, without considering already well-established training techniques. For example, the baseline (A) starts from a 1-step FID of 6.40, which is significantly higher than existing baselines such as iCT and ECT. Therefore, although Table 2 shows that their techniques improve performance from this naive baseline, it fails to show that it improves over iCT or ECT. The only comparison against these baselines is in the final result table, which does not show the effect of each design choice proposed in this paper when applied to those baselines.

Next, I list the points regarding each proposed component:

* **Time step discretization:** The motivation behind the proposed discretization is that it minimizes the overall discretization error coming from finite difference approximation. However, in Table 6(j), the continuous-time model (Analytic) underperforms the discrete CM. Doesn't this show that the discretization error does not matter when the discrete CM employs a sufficiently small step size?
* **Weighting function:** The reasoning behind w(t) is not convincing. Why do we want to match the gradient magnitude of CM and diffusion models? They are optimizing different losses with different purposes. Also, the diffusion model’s gradient magnitude depends on specific instances. For example, once it achieves a minimum, the expected gradient will be zero.
* Did you mean once you plug (12) into (11), the norm of (11) is the same as the norm of (10)? Why?
* **Auxiliary variable-upper-limit integration task:** ECT’s time-step upper bound is zero. See [https://github.com/locuslab/ect/blob/4311059770f54821d151a9b0e1f76770a5f3930e/training/loss.py#L45-L50](https://github.com/locuslab/ect/blob/4311059770f54821d151a9b0e1f76770a5f3930e/training/loss.py#L45-L50). It’s not clear if we need this technique given that the baseline already does not use the time upper bound.
* **MMD loss:** The performance improvement is marginal (in Table 2, it does not improve at all). I’d suggest removing it and instead using that space to explain some contents in Appendix C.1.

Finally, as noted by the authors, one important limitation is that the paper only evaluates on CIFAR-10. Although I understand that the ImageNet-64 experiments may be beyond the authors’ available resources, it is also the case that empirical findings in CIFAR-10 often do not transfer to ImageNet or larger datasets. One way to address this is to start from a pre-trained diffusion model and show ablations with relatively small batch sizes, as done in ECT.

### Questions
* Can you set A in Table 2 to ECT/iCT and show the effects of your techniques from there?
* In Table 6(b), why is linear decrease performing similarly to the proposed exponential decrease?
* ECT uses a sigmoid multiplier on CIFAR-10. See [https://github.com/locuslab/ect/blob/4311059770f54821d151a9b0e1f76770a5f3930e/training/loss.py#L45-L50](https://github.com/locuslab/ect/blob/4311059770f54821d151a9b0e1f76770a5f3930e/training/loss.py#L45-L50). Can you compare against this in Table 6(b)?

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
The paper proposes several techniques for improving the traning of consistency models and flow-map consistency models. By progressively adding modular improvements over a simple baseline, the paper clearly identifies the benefits and advantages of each added component. The result is a clear guideline on how to imprve consistency training, and the model achieve competitive results on CIFAR-10.

### Strengths
The resulting method is relatively simple, especially compared to other consistency model training procedures. Most proposed components are well motivated and justified, and verified with extensive empirical evaluation. The final results on CIFAR-10 are competitive.

### Weaknesses
As also mentioned by the authors, the method is only evaluated on CIFAR-10. This makes it hard to understand if the proposed improvements translate to other datasets. In addition, it would be very interesting to see how directly the various design choices work on different datasets and settings, i.e. how much additional tuning is required to get this method to work on other datasets. Ideally, it would be a great selling point if the design choices translate well with minimal tuning, but from the manuscript it is hard to tell. While I understand that not every lab has access to much computational resources, the authors could have used latent imagenet 256 x 256 which has a latent resolution of 32x32x4, with the smallest DiT settings, which I believe shouldn't surpass by much the GPU requirements of CIFAR-10. Another weakness is that the authors did not provide the code, which makes it hard to verify some of the claims made in the paper (see questions).

### Questions
- The baseline seems too good to be true to me. From my understanding, that's a finite difference approximation of a continuous consistency model with a constant small $dt$, trained from scratch, without weighting functions and with uniform time distribution, with the l2 loss and without discretization curricula. That alone gets 6.4 1-step FID, which is already much better than the original CT. Could the authors please provide insights on this?
- The loss in equation 5 reduces to simple l2 loss when s=0. I wonder how did the authors manage to get such good results with a simple L2 loss, while most papers need to use pseudo-huber loss or variants. 
- Do you use float32 precision?

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
4

### Summary
Recent methods for training consistency models rely on numerous design choices, often set empirically, that significantly impact performance. This paper empirically investigates the design choices for the  critical components of training consistency models and suggests best practices for training consistency models. These components include discretization schedule, time preconditioning, time sampling, loss weighting,  targets in loss function, as well as the expression for the loss function itself which in this case, is inductive moment matching style distributional loss. The analysis is done on CIFAR-10, and the resulting model gets competitive FID on CIFAR-10 for 1-step and 2-step sampling.

### Strengths
1. Consistency models have been widely adapted to distill diffusion models or train 1-step or 2-step from scratch. The detailed empirical analysis in this paper is useful for the research community as it provides guidelines in training such models in practice.
2. The main paper has been written in an easy to understand manner and various design decisions have been explained clearly (though the proofs in the appendix can be simplified).

### Weaknesses
1. Results and analysis is done only on CIFAR-10 which is quite small. It would have been useful to show that the configuration from CIFAR-10 can be transferred to IMageNet-64. 
2. Errors in Time Preconditioning and typos in some other expressions:  
- The general form for time preconditioning  $c(t) = K \int \dfrac{dt}{h(t)} + C$. Upon substituting $h(t) = \epsilon e^{ - \mu t}$, we get $c(t) = \dfrac{K}{\epsilon \mu} e^{\mu t} + C$. This function is exponentially decreasing when it should be exponentially increasing to get $c’(t)h(t)$ as constant. This expression is also different from the one in the paper $c(t) = \dfrac{e^{-\mu t} - 1}{\mu}$.
-  In line 1142 and line 1147: After taking the limit, $r$ should be replaced with $t$ in the expression.
3. From Table 2, the role of introducing some of the components is unclear. For instance, time-aware MMD loss has similar 1-step performance and only marginal gain (+0.01 FID) on 2 step over the other design choices. The recommended time preconditioning seems to result in worse 1-step performance. In Table 3 for variable-upper-limit, the granular analysis done for constant-upper limit is missing. Therefore, it is difficult to understand the contributions of different components like discretization function, time preconditioning, loss weighting etc. for variable upper limit case.

### Questions
1. What time preconditioning was used in practice? The recommended expression from time conditioning differs from the one that can be derived as per the reasoning in the paper.

### Soundness
2

### Presentation
3

### Contribution
2
