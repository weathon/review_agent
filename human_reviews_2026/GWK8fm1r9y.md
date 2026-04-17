# Flow Straight and Fast in Hilbert Space: Functional Rectified Flow

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Many generative models originally developed in finite-dimensional Euclidean space have functional generalizations in infinite-dimensional settings. However, the extension of rectified flow to infinite-dimensional spaces remains unexplored. In this work, we establish a rigorous functional formulation of rectified flow in an infinite-dimensional Hilbert space. Our approach builds upon the superposition principle for continuity equations in an infinite-dimensional space. We further show that this framework extends naturally to functional flow matching and functional probability flow ODEs, interpreting them as nonlinear generalizations of rectified flow. Notably, our extension to functional flow matching removes the restrictive measure-theoretic assumptions in the existing theory of \citet{kerrigan2024functional}. Furthermore, we demonstrate experimentally that our method achieves superior performance compared to existing functional generative models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper establishes the rectified flow in Hilbert space, by providing a theoretical result proving the superposition principle in separable Hilbert Space. This justifies the extension of the usual flow matching loss in the Hilbert space,  providing an alternative approach to Kerrigan's Functional Flow matching that not only requires the existsnce of the Radon Nykodim derivative but also the target data to live in the Cameron Martin space of the initial Gaussian Measure.  
The paper implements the Flow matching in the Hilbert space by embedding the hilbert space element through functional mapping via implicit neural representation / transformers.

### Strengths
The paper has a solid contribution regarding the superposition principle in separable Hilbert Space, whose significance may even extend beyond the flow matching.  It provides an alternative to the flow matching techniques like Functional Flow matching in the infinite dimensional space.   The paper also provides valuable insights that the proposed theoretical framework can realize the paths/dynamics proposed on other works (Functional Probability Flow ODE, Functional Flow Matching)

### Weaknesses
While the theoretical contributions deserves much credit, but the paper suffers from several weaknesses in its presentations. 

### 1.  
The theoretical novelty seems orthogonal to the methodological advantage in the current presentation of the paper.  This paper guarantees that "the flow matching of usual type (Lipman et al, Liu et al)" is theoretically justified on the Hilbert space.  However, as a method, (4) on its own is not novel, and it is impossible in the current set of results to conclude whether the performance advantage is coming from theoretical soundness of the proposed approach or from the power of the Hilbert-space embedding machinations like Transformer / INR.
In order to channel the theoretical novelty to practical significance, I believe an extensive, corner case experiment is required to show that FRM overcomes the problem of FFM; for instance, one may come up with the case in which there is no well defined RN derivative, and show that FRM outperforms FFM that is applied as an an approximate procedure ( with the same representation power allowed to both methods).  Some effort is needed to fundamentally isolate the power of the transformer from the threotical novelty. 


### 2. 
The presentation of the novelty of this work was very hard to follow-- the EFM's theoretical requirements are noted, but to what end? I do agree that these requirements are more impractical than the ones required by the proposed approach, but they have not been shown to emerge as the real problem in practice.  So many methods in machine learning are applied to the real world problem with hand-wavy assumptions --- these ambiguities would be of interest in machine learning only if they present a real threat.     Without the presenting that a "real threat" is triggered by impractical functional assumptions, the motivation of overcoming EFM's assumptions is weak. 

At the same time, if the goal of this work is purely theoretical,  I feel that EFM deserves different treatment in this paper, as a more restrictive framework purported to the same theoretical goal, for example.

### Questions
Please see the weaknesses section. Does the theoretical soundness of this framework, as a method, provide an advantage over other comparative methods in axis independent from the architectural representation power or way of embedding an element of hilbert space element? 

In a similar note, is there any "practical" cases in which X1 lies outside of the Cameron Martin Space. Can you come up with a synthetic case for example to show that the proposed approach is **empirically** superior to functional flow matching? (I guess adding jumps / disconitniuty might do a trick when we set H = L2. )

### Soundness
3

### Presentation
1

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
The authors demonstrate theoretically the soundness of most general Rectified Flows on infinite-dimensional spaces, greatly relaxing the assumptions imposed by previous works. They propose three ways to train the proposed method, FRF, which they evaluate on two different modalities, on which they significantly improve on the SOTA.

### Strengths
- The work is well-anchored in the literature regarding the problem it tackles (flows on infinite-dimensional spaces). Good comparisons are offered at the end of section 4.
- The framework proposed generalises the previous results of the field, and essentially match what we would expect Rectified Flows to be like in this setting; that is to say, they recover essentially the same method, finally, in theory.
- There are good theoretical discussions in the appendix – namely, about how reasonable the assumptions are in this work. In general, the work in the appendix seems very thorough and introduces the topic well.
- The proof of the main theorem seems sound, and is well-organised, in general.

### Weaknesses
- On the theoretical part of the paper, there is, I believe, a great problem in the general presentation: as I first read the paper (without the appendix first), I completely missed out on the part about the superposition, and, when I came back to read the abstract, I was surprised to find it there. In general, I would argue that too much of the technical content is hidden in the appendix. Understandably, the content is highly technical and arguably too dense for the main part of a Machine Learning paper. However, essentially omitting main results (included in the abstract itself) seems wrong, and, if space is an issue, other parts (for instance, background) can be trimmed on. I think this is important. Probably some informal/intuitive explanations can be given.
- While the generality of this method on the interpolation choice and the weaknesses of the other methods are pointed out, it required going through the appendix quite thoroughly to get an idea of what the assumptions made in this work were.
- While the work is well-grounded with respect to the generative functional literature, it is less so with respect to the generative modelling/flow-based modelling literature, arguably.
- The empirical validation, I would argue, is the main weakness of this work, and perhaps the method approach in general. It seems that the training objective is pretty standard, so then section 5 must be the novelty, empirically, where different architectures are proposed. Then, in section 6, only one architecture is tried out on each of the datasets. I can imagine that the neural operators can only work in the Navier-Stokes case, but that which worked on MNIST surely can be run on CelebA, unless I am mistaken?
- Such low FID difference on MNIST, especially at such low FIDs too, may not be used as evidence for improvement. I am not even certain it is a good metric for MNIST. (I agree, though, that it shows that the method works.)
- I would refrain from using $\mathcal{O}$ to express “order of” with constants following.

### Questions
1. To be certain: what are the main assumptions made in this work?
2. Although they are certainly all essentially equivalent, could you justify formulating this as Rectified Flows (which are not really considered state-of-the-art), instead of, say, Flow Matching [1] or Stochastic Interpolants [2]?
3. How does this work relate to stochastic processes? Is it the case, similar to finite-dimensional (typically Euclidean) data, that we can recover the same marginals without the Wiener process (probability flow ODE [3])?
4. Not being an expert on functional ML, how do you train your models on *images*? And why is it a relevant experiment whatsoever? Is there an advantage to this representation over the typical tensorial CxHxW one?
5. Why did you not try out all applicable methods on the same datasets?

Overall, I think this is a good work, but some clarifications must be made, and mostly the work would benefit a lot from a slightly different presentation, in my opinion.


### References
[1] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. “Flow Matching for Generative Modeling”

[2] Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden. “Stochastic Interpolants: A Unifying Framework for Flows and Diffusions”

[3] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole. “Score-Based Generative Modeling through Stochastic Differential Equations”

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper extends the **Rectified Flow (RF)** framework to **infinite-dimensional Hilbert spaces**, providing a theoretical foundation for functional generative modeling.  
The proposed **Functional Rectified Flow (FRF)** generalizes the original rectified flow (Liu et al., 2022) beyond Euclidean domains, offering a deterministic and mathematically rigorous formulation for *function-valued random variables*.  
Overall, the work aims to lift rectified flow theory into infinite-dimensional settings, positioning it as a unifying framework for functional generative modeling.

### Strengths
1. The formulation is mathematically solid, carefully extending continuity equations and rectified flows to Hilbert spaces via the *superposition principle*.  
2. The work unifies **rectified flow**, **functional flow matching**, and **functional probability flow ODEs** under a single theoretical umbrella, clarifying their relationships.  
3. Results show modest but consistent improvement in *FID* and *density MSE* metrics compared to functional diffusion processes (FDP) and functional flow matching (FFM).

### Weaknesses
1. The main theoretical advancement—lifting rectified flow to separable Hilbert spaces—is **mathematically correct but conceptually incremental** relative to prior work on functional flow matching (Kerrigan et al., 2024) and functional diffusion models (Franzese et al., 2023). The novelty largely lies in formalizing the *marginal-preserving property* in infinite dimensions rather than introducing fundamentally new generative insights.
2. Experimental improvements (e.g., FID 0.43 → 0.41 on MNIST) are marginal and within statistical noise.  Experiments are confined to relatively small datasets and fixed architectures; no ablation or analysis of functional-space regularization is included.

### Questions
1. Can the authors provide more intuition on the *practical implications* of Hilbert-space rectification — e.g., does it enable higher-resolution function generation or better memory scaling?  
2. Could the marginal-preserving theorem be empirically verified (e.g., via sample-wise trajectory interpolation analysis)?  
3. Is there a computational or convergence benefit of FRF over functional flow matching in terms of training stability or sampling complexity?  
4. Would FRF remain valid if the underlying Hilbert space is replaced by a Banach space, or if data lies on mixed discrete–continuous domains?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a rigorous theoretical extension of rectified flows to infinite-dimensional Hilbert spaces, establishing a unified framework for functional generative modeling. The authors prove marginal-preserving property for rectified flows in this settings. It is demonstrated that functional mapping of flows and ODE of probabilistic flows arise as special cases within their framework. Empirically, the method achieves state-of-the-art results on images and scientific datasets using three different neural architectures.

### Strengths
- The Hilbert space extension of rectified flows is novel and rigorous, addressing key limitations of prior functional models by relaxing restrictive measure-theoretic assumptions.
- The work successfully unifies several functional generative models, providing a cohesive theoretical perspective.
- Extensive experiments across multiple architectures (INR, Transformer, Neural Operator) and data types demonstrate consistent superiority over competitive baselines.

### Weaknesses
- The theory relies on the assumption that the stochastic process $\mathbb X$ is "pathwise continuously differentiable." Although this is more plausible than absolute continuity of measures, it is still a non-trivial assumption about the smoothness of data trajectories. It would be useful to briefly discuss the practical validity of this assumption for real functional data (e.g., images as functions, solutions to partial differential equations). Such a statement is not obvious if, for example, the image contains contrasting edges.
- While the experiments are convincing, they are limited to three datasets. Including a more diverse set of functional data, such as irregularly sampled time series or 3D shapes, would further strengthen the claim of general applicability. Furthermore, a direct comparison of training/sampling speed versus other functional models (FDP, FFM) would be highly informative, given that one of the key motivations for rectified flows is efficiency. Of the three datasets listed, two are datasets with images for which the need to apply this functional technique is not at all obvious -- models built on conventional finite-dimensional principles work well with them.
- The excellent performance is demonstrated by plugging into established, powerful architectures (INR, Transformer, FNO). An open question is how much of the performance gain is due to the proposed theoretical framework versus the inherent capacity of these architectures. An ablation studying a simpler backbone could help isolate the contribution of the rectified flow objective itself.

### Questions
- The assumption of pathwise differentiability is central to your theory. How is it implemented or tested during training, especially when using discrete data to approximate functions?
- The paper highlights computational efficiency as a motivation. Could you provide quantitative results (e.g., wall-clock time, number of function evaluations) comparing the sampling speed of your FRF against functional diffusion or flow matching models?
- In sentence 18, you note that Lipschitz's global conditions can be applied to neural networks. Did you use such methods (e.g., spectral normalization) in your experiments to ensure the correctness of the ODE formulation? If not, did you observe any instabilities during numerical integration?

### Soundness
3

### Presentation
3

### Contribution
3
