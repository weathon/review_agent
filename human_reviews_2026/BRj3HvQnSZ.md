# Unbiased Gradient Estimation for Event Binning via Functional Backpropagation

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
Event-based vision encodes dynamic scenes as asynchronous spatio-temporal spikes called events. To leverage conventional image processing pipelines, events are typically binned into frames. However, binning functions are discontinuous, which truncates gradients at the frame level and forces most event-based algorithms to rely solely on frame-based features. Attempts to directly learn from raw events avoid this restriction but instead suffer from biased gradient estimation due to the discontinuities of the binning operation, ultimately limiting their learning efficiency. To address this challenge, we propose a novel framework for unbiased gradient estimation of arbitrary binning functions by synthesizing weak derivatives during backpropagation while keeping the forward output unchanged. The key idea is to exploit integration by parts: lifting the target functions to functionals yields an integral form of the derivative of the binning function during backpropagation, where the cotangent function naturally arises. By reconstructing this cotangent function from the sampled cotangent vector, we compute weak derivatives that provably match long-range finite differences of both smooth and non-smooth targets. Experimentally, our method improves simple optimization-based egomotion estimation with 3.2\% lower RMS error and 1.57$\times$ faster convergence. On complex downstream tasks, we achieve 9.4\% lower EPE in self-supervised optical flow, and 5.1\% lower RMS error in SLAM, demonstrating broad benefits for event-based visual perception.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed a framework for unbiased gradient estimation of arbitrary binning functions by synthesizing weak derivatives during backpropagation while keeping the forward output unchanged. Experiments show that the proposed method is benificial for downstream tasks like optical flow estimation and SLAM.

### Strengths
- The proposed method addresses event-based visual perception from a new perspective via biased gradient estimation.
- The proposed method improves simple optimization-based egomotion estimation and facilitates fast convergence.
- The proposed method is beneficial for downstream tasks like optical flow and SLAM.

### Weaknesses
- If it is possible, please evaluate your proposed method on more flow estimation methods, like event-based meshflow estimation.
- The proposed method has many design choices and parameter setups. If it is possible, please enrich the parameter analyses and conduct more detailed ablation studies to help better understand the design choices. Besides, it would be nice to evaluate the proposed method under different event data representations.

### Questions
- Would you consider making the source code and the implementation publicly available to foster future research in this line?
- The event-based optical flow estimation has the advantage of temporal-continuous prediction. How about your solution for temporal-continuous prediction? This could be discussed in detail.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to address a long-standing core problem in event-based vision: the biased gradient estimation caused by the discontinuity of the binning operation. Traditional methods, such as using smooth kernels or surrogate gradients, introduce bias during backpropagation, leading to inefficient learning. To solve this, the authors propose a novel framework named "Functional Backpropagation" (FBP). The core idea is to lift the discrete binning function into a continuous functional space. By leveraging weak derivatives and integration by parts from functional analysis, the FBP framework derives a synthesized, unbiased gradient estimate during backpropagation without altering the forward pass output. The authors theoretically prove that this synthesized gradient approximates long-range finite differences. Experimental results show that the proposed method achieves improved performance on downstream tasks such as motion estimation, optical flow, and SLAM.

### Strengths
The paper is built on a very solid mathematical foundation. The authors skillfully employ tools from functional analysis, such as weak derivatives and integration by parts, to provide a novel and theoretically unbiased solution to the prevalent problem of gradient estimation for discontinuous binning operations in the event-based vision domain. This approach of lifting a discrete problem into a functional space for a solution is highly inspiring and represents a rigorous attempt to fundamentally address the gradient issue.

### Weaknesses
1. The experimental figures and setup descriptions are insufficient. The readability of some figures is poor. The legend in Figure 3 (e.g., G_fd > 0) is not explained in the caption. The plots on the left of Figure 4 are missing a y-axis title. The experimental setup in Section 4.1 (Bias Analysis) is not clearly described. The authors mention estimating the "finite-difference bias" by subtracting "numerical gradients" but do not detail the specific method for calculating these numerical gradients (e.g., central difference, forward difference) or the choice of step size.

2. The paper's readability could be improved. The paper introduces numerous mathematical symbols and theories from functional analysis (e.g., Fréchet derivative, Riesz representation theorem), which may be unfamiliar to many researchers in the machine learning and computer vision communities, significantly raising the barrier to entry. Despite the novelty of the method, the paper lacks a standalone algorithm section or pseudocode to guide implementation. It is difficult for readers to discern from the theoretical derivations in Section 3 how "Functional Backpropagation" is actually implemented in modern deep learning frameworks like PyTorch or JAX. This significantly hinders the quick understanding and reproduction of the work.

3. The depth of the experimental validation are insufficient. The choice of the reconstruction kernel $l(\cdot)$ is crucial to the method. The authors select a linear spline kernel in Section 3.3 but do not provide an ablation study to validate the effect of other choices, leaving the superiority of the current selection unsupported by experimental evidence. While the paper compares FBP with gradients from the raw kernel, a more crucial comparison would be against other established gradient estimation techniques in the field, such as various Surrogate Gradients (SG) methods. 

4. The current experimental setup fails to sufficiently demonstrate whether FBP offers a significant advantage over these existing techniques.The reported performance improvement is limited. In several applications, the improvement brought by the method is relatively modest (e.g., a 9.4% EPE reduction in optical flow and a 5.1% Abs RMS error reduction in SLAM). The authors primarily apply the method to frameworks based on Contrast Maximization (CM). However, the CM method itself suffers from several other well-known bottlenecks, such as high sensitivity to initialization, slow convergence, and a tendency to converge to degenerate solutions. While the proposed method theoretically improves the gradient, its limited practical impact weakens the motivation for targeting "gradient bias" as the most critical problem to solve within the CM framework.

### Questions
1. Formally, Eq. (14) seems to just replace the original binning kernel $k(\cdot)$ with a new kernel $\kappa(\cdot)$ formed by the convolution of $l(\cdot)$ and $k(\cdot)$. Could the authors provide a more intuitive explanation as to why the gradient derived from the FBP framework is theoretically "unbiased", whereas directly using a smooth kernel (like a Gaussian) is "biased"? Where does its fundamental advantage lie?

2. In the experiments for Eq. (20), the authors use truncated kernels. Is the fundamental source of gradient bias the artificial truncation of the kernel, or is it the discrete sampling of the IWE space? If it is the former, would using an untruncated kernel (or increasing its bandwidth) significantly reduce the bias? If it is the latter, does this imply that increasing the IWE resolution (i.e., decreasing $\Delta$) is a more direct solution than optimizing the gradient itself?

3. The derivation of Eq. (12) relies on approximating the Dirac delta function with a reconstruction kernel $l(\cdot)$. The authors chose a linear spline kernel in their experiments. What is the specific impact of the choice of $l(\cdot)$ on the unbiasedness and variance of the final gradient? Does an "optimal" choice for $l(\cdot)$ exist in theory?

4. In Section 4.2, the authors use optimizers like L-BFGS-B and trust-ncg, which rely on first- or even second-order information. In real-time applications like robotics and SLAM, efficient (quasi-)Newton methods are crucial. Could the authors comment on whether the FBP framework naturally supports efficient computation of second-order derivatives (i.e., Hessian-vector products, HVP)? Does it have advantages or face new challenges in this regard compared to traditional surrogate gradient methods?

### Soundness
3

### Presentation
1

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
The paper contributes a theoretical framework for unbiased gradient estimation in event-based vision, addressing the problem of discontinuities in event binning functions. The authors derive weak derivatives via integration by parts and demonstrate improvements in several tasks such as motion estimation, optical flow, and SLAM.

### Strengths
1. The paper is clearly written and well-organized. 
2. The mathematical exposition is detailed and precise, with good use of notation and references.
3. Figures and tables are informative and visually consistent, effectively supporting the text.

### Weaknesses
1. No ablation on reconstruction kernel choices.
2. Limited comparison to related works, like comparison to prior gradient approximation methods.
3. Readability could be improved with intuitive explanations or schematic examples.

### Questions
1. Although convergence speed improves, the computational overhead of functional derivatives is briefly mentioned but not rigorously profiled. A breakdown of runtime cost per gradient step would be helpful.
2. How sensitive is the method’s performance to the choice of reconstruction kernel? Would different kernels change the unbiasedness or convergence behavior?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the biased gradient estimation in event-based vision due to binning function discontinuities. The proposed functional backpropagation framework lifts binning functions into functional space to enable unbiased gradient computation via weak derivatives without Dirac deltas. The paper claims that the weak derivative is equivalent to long-range finite differences and generalizes to both smooth and discontinuous kernels. Experiments are conducted on velocity estimation, optical flow estimation and SLAM.

### Strengths
1. The topic is good. The paper analyzes a fundamental issue in event cameras, namely the gradient problem in the binning function, which is a relatively small but important issue.
2. The paper features rigorous mathematical derivations and proofs, demonstrating a solid theoretical foundation.
3. The method proposed in this paper has the potential to contribute to some fundamental applications in the field, such as velocity estimation as mentioned in the paper.

### Weaknesses
1. Some captions of the figures are insufficient, which affects the readability of the paper. For Figures 1 and 2, it is recommended to add more detailed explanations of the symbols and the content depicted in the figures. This will help readers better understand the key components and the underlying concepts illustrated in these figures.
2. The current structure of the paper writting is not well balanced. I believe the method and analysis sections in the early part are overly lengthy, while the experimental section in the main text is clearly too brief.
3. In the part of velocity estimation, the proposed methods show no advantages in some cases in Linear Velocity Estimation.
4. There are some issues with the optical flow estimation section. First, are the comparison methods used in the paper up-to-date? This needs to be carefully verified. The related work section cites mostly older references. Second, what are the advantages of the proposed method compared to other existing deep learning algorithms? Can deep learning models implicitly address the effects caused by binning within the network computation? Third, I think the visualization of the method's result is not satisfying, and using the ERAFT inference results as ground truth is also not reasonable enough.

### Questions
As listed in the "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
3
