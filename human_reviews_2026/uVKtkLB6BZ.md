# Adapting Noise to Data: Generative Flows from learned 1D Processes

- Decision: Reject
- Scores: 4, 2, 6, 10

## Abstract
We introduce a general framework for learning data-adaptive latent distributions (noise)
in generative models based on 1D quantile functions through minimizing a statistical
discrepancy between noise and data samples. Our quantile-based parameterization naturally
adapts to heavy-tailed or compactly supported target distributions while shortening transport
paths by capturing marginal structure. This construction, originally motivated by the study
of 1D processes beyond the usual diffusion, integrates seamlessly with standard training
objectives, including flow matching and consistency models. Numerical experiments
highlight both the flexibility and the effectiveness of our approach, achieved with minimal
computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a method to learn the initial distribution of a flow matching model. Usually, this distribution is chosen to be Gaussian noise. Here, they parameterize the initial distribution with a learnable quantile function. Beyond the initial distribution, this distribution also determines all intermediate distributions (commonly called probability path), while having a fixed linear interpolant. They jointly learn this initial distribution with the velocity field. Experiments on synthetic data sets and small-scale image datasets show the validity of the method.

### Strengths
- This is a natural idea that has not been explored in the literature as much and that could be a powerful way of improving flow-based generative models.
- It is a simple training objective that has minimal computational overhead.

### Weaknesses
- Overall, the writing of the paper could be improved significantly. The motivation is not well-explained in the text (both in the introduction and later in the text). Further, illustrations and examples are lacking.
- The experiments are limited and the presented results are not very strong. For example, for CIFAR10, the flow baseline is worse than standard baselines (https://github.com/facebookresearch/flow_matching). FM achieves an FID on CIFAR10 <=3.0.
- The training objective requires more elaboration: The parameters phi that parameterize the initial distribution underlie a trade-off: They can be either used to  minimize the first or second term in the training objective L(theta, phi). Therefore, even for lambda=0, minimizing this objective might be valid (i.e. one minimizes then effectively the residual variance of the CFM loss). As such discussions are at the core of the idea, it would be good to elaborate on this more.

### Questions
- L26: "Consistency models like the recently introduced inductive moment matching (IMM) Zhou et al. (2025)" → Consistency Models are generally speaking different from IMM. I would rather present them as different methods.
- Proposition 2 is known prior to this work, e.g. it is a special case of Proposition 4 in [1] and should be referenced.
- Why are Consistency Models discussed? They are not really used anywhere?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a novel generative modeling framework that constructs flow-based models using learned one-dimensional noising processes. Instead of relying on a fixed Gaussian latent distribution, the method learns the noise distribution directly through quantile functions that adapt to the data. This formulation integrates naturally with the flow matching framework, enabling more flexible and data-dependent noise modeling. The authors further illustrate the approach through several examples of one-dimensional processes, including the Wiener process, the Kac process, and an MMD gradient flow, and show that learning quantile-based noise can substantially enhance the flexibility and transport efficiency of generative models.

### Strengths
1. The idea of constructing generative flows through learnable 1D quantile processes is original.

2. By using quantile parameterizations, the approach can handle distributions with compact support or heavy tails, going beyond the Gaussian assumptions typical in flow and diffusion models.

3. The framework is compatible with standard objectives such as Flow Matching and Inductive Moment Matching, showing practical extensibility.

### Weaknesses
1. Lack of sufficient baselines.

The paper lacks adequate baseline comparisons to clearly demonstrate the advantages of the proposed method. In Section 5.1, no baseline is provided for reference, and Figure 5 includes only a single baseline whose selection and description are not well explained. The experimental evaluation should include more detailed quantitative comparisons against standard diffusion or flow-based models to better substantiate the claimed improvements.

2. Clarity and presentation issues.

The overall clarity of the paper can be improved. The abstract does not effectively summarize the key contributions and contains some redundancy. For example, the first and third sentences are quite similar. Several methodological details are also unclear. For instance, the statement “we pre-train our quantile” (Line 356) does not explain why pre-training is necessary or which experiments rely on it. Similarly, the introduction of the regularization term that penalizes the expected negative log-determinant of the Jacobian (Line 374) is mentioned without justification or analysis of its impact. These elements should be clarified to improve the transparency and reproducibility of the work.

3. Expressive power of one-dimensional processes.

The paper does not provide sufficient theoretical or empirical evidence regarding the expressive power of using one-dimensional denoising processes. While the decomposition into independent one-dimensional components makes the approach more tractable, it may limit the model’s ability to capture complex dependencies across dimensions. A deeper discussion or ablation study evaluating this trade-off would strengthen the paper’s technical soundness.

### Questions
1. Generality of one-dimensional flows

Is there a universal or systematic way to construct one-dimensional flows, beyond the three specific examples discussed in Section 4.1?

2. Sampling efficiency

What is the sampling time or computational cost of the proposed method compared to standard flow matching or diffusion-based models?

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
The authors propose a framework for 1D per-dimension noising processes for generative 
models. They propose to learn the latent distribution to reduce the transport paths of 
generative models. The latent distribution is modeled via learned quantile functions, which 
are modeled via rational quadratic splines. The quantile functions are learned from data by 
minimizing the Wasserstein-2 distance between the data distribution and the modeled latent 
distribution.

Main contribution:
- Decomposition of multidimensional flows into 1D noising processes
- Quantile-based formulation of latent distribution: learn the quantile function of latent
noise instead of fixing it to a Gaussian distribution
- Experimental validation on synthetic data (checkerboard, funnel, Gaussian mixture)
and image data (MNIST, CIFAR-10)

### Strengths
- Minimal computational overhead via rational quadratic splines
- improved noise distribution adapted to target
- Good explanation of the math fundamentals
- provides a general framework for independent 1D noising processes and an
expressive way to parameterize them in practice via quantile functions and rational quadratic splines

### Weaknesses
- Missing ablation study on the velocity not exploding outside of the support of the
distribution. A simple 2D example showing the vector field would be nice.
- Missing benchmarks on larger problems -> how scalable and stable is this approach
with an increasing problem dimension? -> potentially unstable quantile training
- lack of quantitative metrics
- unclear generalization capability for shifting data distributions

### Questions
The weight on the quantile loss and the regularization weight are both new
hyperparameters that need to be tuned. How sensitive are they to different problems? On the funnel target, the quantiles are pre-trained -> another hyperparameter.
- How stable is the joint optimization of quantile and flow networks in practice?
- Is learning quantiles equivalent to learning transport maps under certain
assumptions?
- How does the learned quantile noise compare to learned latent priors in VAEs or
normalizing flows?
Can this method scale to modern high-resolution diffusion tasks?
- What happens for out-of-distribution conditional data?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper introduces an approach that uses one dimensional processes and quantile functions to learn generative models in a component-wise manner. The author shows how this approach is compatible with the flow matching and consistency model frameworks, and can better handle difficult settings such as heavy tails and compact supports.

### Strengths
Originality: The core ideas (e.g., using 1D processes and quantile functions to learn generative models in a way that is compatible with consistency and flow matching frameworks, etc) are, to the best of my knowledge, original and innovative. 

Clarity: the paper is written in a clear and self-contained manner. Even in the more technical portions, everything is defined and explained clearly. This is a major strength of the paper. 

Quality: I find the quality of the theoretical and empirical sections to be sufficient. While one can always perform more experiments on more datasets/simulations, the current experiments sufficiently demonstrate/support the main points of the paper. While I did not check proofs/appendix in detail, the technical portions of the main paper are, to the best of my knowledge, sound and correct. 

Significance: The topic of learning generative models is timely and significant. The proposed method is also a significant contribution in my opinion, more than enough to meet the bar for ICLR.

### Weaknesses
There are minor points and questions which I bring up below: 

- When the authors mention the difficulty of learning multimodal and heavy-tailed targets on page 1, Hagemann and Neumayer (2021) and Salmona et al (2022) are cited. However, there are other highly relevant literature that should have been cited. These include:

- Concentration of Measure for Distributions Generated via Diffusion Models. R Ghane, A Bao, D Akhtiamov, B Hassibi
- On the Statistical Capacity of Deep Generative Models. E Tam, D Dunson
- Copula & Marginal Flows: Disentangling the Marginal from its Joint. M Wiese, R Knobloch, R Korn

- Runtime/computational costs: does the one dimensional approach that the authors propose lead to higher runtime or computational complexity in practice compared to other FM/consistency based approaches? (I am NOT looking for any computational complexity bounds/results, I am mainly interested in just a couple of sentences that comment on the runtime/computational aspects of things so readers can get a rough idea).

### Questions
See above section

### Soundness
3

### Presentation
4

### Contribution
4
