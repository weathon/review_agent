# On the Mode-Seeking Properties of Langevin Dynamics

- Decision: Reject
- Scores: 3, 5, 5, 5

## Abstract
The Langevin Dynamics framework, which aims to generate samples from the score function of a probability distribution, is widely used for analyzing and interpreting score-based generative modeling. While the convergence behavior of Langevin Dynamics under unimodal distributions has been extensively studied in the literature, in practice the data distribution could consist of multiple distinct modes. In this work, we investigate Langevin Dynamics in producing samples from multimodal distributions and theoretically study its mode-seeking properties. We prove that under a variety of sub-Gaussian mixtures, Langevin Dynamics is unlikely to find all mixture components within a sub-exponential number of steps in the data dimension. To reduce the mode-seeking tendencies of Langevin Dynamics, we propose Chained Langevin Dynamics, which divides the data vector into patches of constant size and generates every patch sequentially conditioned on the previous patches. We perform a theoretical analysis of Chained Langevin Dynamics by reducing it to sampling from a constant-dimensional distribution. We present the results of several numerical experiments on synthetic and real image datasets, supporting our theoretical results on the iteration complexities of sample generation from mixture distributions using the chained and vanilla Langevin Dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper considers the task of sampling from a multimodal density using Langevin-like SDEs. The authors considers the standard Euler discretisation of the over damped Langevin diffusion and show that the chain does not discover the modes with high probability after a number of iterations that is exponential in the dimension when the target is a mixture of (sub)-gaussian distributions. To improve over this slow mixing, the paper proposes chained Langevin dynamics, essentially a Gibbs-type approach where Langevin dynamics are applied to batches of the chain instead of the full vector.

### Strengths
The paper considers an interesting problem, since slow mixing in multimodal settings is an issue. The authors also propose an algorithm to obviate such slow mixing.

### Weaknesses
- The paper is hard to read and contains several paragraphs that are of dubious relevance (e.g. lines 141-144 mentioning GANs, several paragraphs on score-based diffusion models, lines 150-161, 179-200, where it is not clear these are relevant for this article). The introduction is long and overall unclear.

- The authors do not seem comfortable with the MCMC literature, e.g. referring to the Euler discretisation of the overdamped Langevin diffusion as "stochastic gradient Langevin Dynamics", citing a paper that introduces a completely different approach based on subsampling. 

- It is not clear if the authors consider the usual sampling context vs. the generative modelling context: is the score of the data distribution known? Moreover the paper contains plenty of references to the generative modelling literature, but it is not clear what is the purpose, and I can only imagine it is to increase their chances of publication. 

- It is unclear if the chained algorithm that is proposed has even the right stationary distribution, nor if it has any useful property for sampling purposes. Moreover, one needs to have the conditional densities to run the algorithm in the first place.

-  The experiments, especially those on the MNIST data, miss explanations on the obtained results, so it is unclear how to interpret them. 

- Overall, I find the paper should go through a substantial rewriting and thus should be rejected.

### Questions
- Is the idea behind the chained algorithm (i.e. decomposing the process in blocks) based/influenced on previous works? If so, the authors should mention the relevant papers.

-Could the authors give any justification for their algorithm?

- Lines 170-171: why is there a minus sign in front of the drift term?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper coniders Langevin Dynamics for multimodal distributions (actually mixtures of gaussians, with a small extension to the case of of a large-variance gaussian plus some localized subgaussians). They prove in some toy model cases that Langevin Dynamics does not reach good approximation to these distributions in short time. As a "solution" to that problem, they lower the dimensionality by partitioning the data vector in equal pieces and sequentially generating the pieces rather than generating them all together. This improves on the "curse of dimension" part of the difficulty.

### Strengths
The topic of the paper (rate of approximation of multimodal distributions) is of interest and the problem of finding a good scalable strategy in these problems is largely open.

### Weaknesses
The setting of study is mainly a toy model, whose assumptions may or may not hold in realistic scenarios. 
A discussion of limitations of the core/crucial Assumptions 1 and 2 of the paper is dearly missed.

I think that this is a good step in that direction, but I'm not sure if the paper is already valuable as is, mainly due to the doubts I have as to how general Assumptions 1 and 2 actually are, compared to typical applications.

### Questions
1) Why is the setup of Assumption 1 general enough to deserve a paper?

More precisely, can one say that there are other kinds of multi-modality that appear in applications or not? 
If yes, then what are the cases not covered by Assumption 1?
If not, in what sense does Assumption 1 subsume the main difficulties related to multi-modality?

2) Is it correct to say that the underlying principle of Assumption 2 is not that distinct from the case of Assumption 1, other than allowing the "concentrated/independent modes" part of the distribution to be given by more general kinds of measures?

### Soundness
3

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
4

### Summary
This article examines the behavior of Langevin dynamics (LD), a sampling method commonly used in generative modeling, especially in contexts where data distributions are multimodal (contain multiple clusters or “modes”). The authors highlight a key limitation of traditional Langevin dynamics: its difficulty in sampling from all modes in a multimodal distribution, as it often struggles to transition between modes in high-dimensional settings.

To address this, the authors propose Chained Langevin Dynamics (Chained-LD), a novel approach that divides the data into smaller patches and samples these sequentially, conditioned on previously generated patches. This method reduces the likelihood of mode-seeking (focusing on only a few modes) by lowering the dimensionality for each sampling step, which in turn accelerates convergence to the correct distribution.

Key contributions and findings of the paper include:
1. **Theoretical Analysis:** The study provides a rigorous analysis of examples where the traditional Langevin dynamics can fail to capture all modes, especially under Gaussian and sub-Gaussian mixtures.
2. **Chained Langevin Dynamics:** The paper introduces Chained-LD as a solution and establishes its convergence properties, showing that it can sample all modes in a distribution.
3. **Empirical Validation:** Experiments on synthetic datasets and real image datasets (such as MNIST) successfully conducted showing how Chained-LD can capture all modes compared to standard LD.

### Strengths
The paper is well-written, easy to read, and highly structured, with thorough documentation that enhances its clarity. The computational approach is straightforward to follow, and the main message of the work is communicated effectively.

One of the most compelling aspects of the paper is the recursion on the vector $n_t$, which presents an innovative approach to the field; to the best of my knowledge, this recursion is novel and has significant implications on the examples presented. The test setting introduced is particularly interesting, as it highlights how SGMs tend to emphasize majority classes, potentially overlooking minority representations in the data.

Additionally, the authors provide a detailed description of the behavior within a specific test setting, which is later extended to the framework of annealed Langevin dynamics. This progression in the analysis is logical and well-integrated, offering valuable insights into the dynamics of SGMs in multimodal distributions.

### Weaknesses
1. **Missing Citations:** The paper does not reference the recent works, like *Conforti, Dumus, and Gentiloni-Silveri (2023)*, which provides critical insights into the KL divergence bounds. Given the significance of these results for KL bounds, integrating this reference would strengthen the theoretical grounding of the paper.
2. **Real-World Applicability and Initial Distribution:** The results are derived based on an initial distribution that is not known in advance, raising questions about their direct applicability to real-world scenarios. In practical applications, data is typically standardized and rescaled, and initializations do not start from a well-defined mode $P^{(0)}$. It would be helpful to understand if and how these findings could be adapted to scenarios where the data distribution and modes are not known beforehand. Moreover, I suggest to include that a direct application of these findings would be finding fair algorithms on less represented group. Could the authors discuss potential approaches for adapting their method to scenarios where the initial distribution is unknown or to explain any limitations in applying their results to such cases?
3. **Handling of Balanced Modes:** The study assumes a setting where one mode is notably dominant. In reality, distributions often lack such stark imbalances. How would the results change when no mode significantly outweighs others? An exploration of how the method performs when modes have similar probabilities would clarify its robustness and generalizability. Could the authors include an additional experiments or theoretical analysis for the case where modes have similar probabilities? This would help demonstrate the method's robustness across different types of multimodal distributions.
4. **Assumption 2 and Gaussian Mixtures:** On page 5, it is claimed that Assumption 2 is automatically satisfied for Gaussian mixtures, yet this is not elaborated upon. This assumption, particularly in points (iv) and (v), introduces technical constraints that lack intuitive explanations for real-world distributions. A more detailed justification for these points would enhance clarity. Could the authors provide a brief proof or explanation for why Gaussian mixtures satisfy Assumption 2, particularly focusing on points (iv) and (v)?
5. **Selection of Patch Size $Q$:** The algorithm’s effectiveness relies on the choice of patch size $Q$. However, guidance on how $Q$ should be selected w.r.t. the dimension $d$ for different applications is not provided. Insights into an optimal or adaptive selection process of this parameter could improve the method’s usability in diverse contexts. Could the authors provide a heuristic or guideline for selecting $Q$ based on d and the characteristics of the data, or to discuss the sensitivity of their method to different choices of $Q$?
6. **Score Function Estimation:** It is unclear how $s_\theta$, the conditional score function estimator, is obtained and whether it is computed offline or dynamically updated. An explanation of the estimation method and its computational cost would help assess the algorithm’s practicality. Could the authors give a brief description of the estimation procedure for $s_\theta$ in the main text or appendix, including whether it's pre-computed or updated during sampling?
7. **Impact on Input Geometry:** The segmentation of the sample into patches for processing may alter the overall geometry of the input vector $x_0$. How does this segmentation affect the coherence and structure of the original data, and does it risk distorting key relationships within the vector? Could the authors discuss or analyze how their patch-based approach preserves or affects important structural relationships in the data?
8. **Proposition 1 and Convergence Bounds:** On page 7, Proposition 1 references a TV distance condition bounded by $\varepsilon Q/d$ but does not justify why this specific bound is chosen. Furthermore, the criteria for calling this result a “convergence bound” are unclear and would benefit from further clarification on its theoretical significance. Could the authors share intuition or justification for the specific bound $\varepsilon Q/d$ and clarify what it is meant by "convergence bound" in this context?
9. **Computational Efficiency:** The proposed approach, particularly with Chained Langevin Dynamics, appears more computationally intensive than traditional methods. Can the authors quantify the additional computational load required for convergence in comparison to previous approaches? Understanding the trade-offs between computational cost and performance is essential for evaluating the practical feasibility of this algorithm. Could the authors include a computational complexity analysis or empirical runtime comparison between their method and traditional Langevin dynamics?

I’m open to improving the evaluation score, as this paper shows real potential in highlighting biases in SGMs, especially around minority representation. Addressing the weaknesses—such as clarifying assumptions, real-world applicability, and computational efficiency—would greatly enhance its impact and make it a valuable reference for understanding and mitigating biases in generative models.

### Questions
See the **Weaknesses** section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper investigates the mode-seeking tendency of stochastic gradient Langevin dynamics (SGLD) and proposes Chained Langevin dynamics with the objective of sampling from multi-modal distributions more accurately. The paper first studies the convergence of SGLD on a multi-modal target with a high-variance, low density, background mode, and shows hardness results of sampling from the sub Gaussian mixture in sub-exponential time in the data dimension. Chained Langevin dynamics seek to alleviate this issue by dividing the input vector into patches and applying SGLD to the conditional distributions $p(x^q \vert x^1, \dots, x^{q-1})$, only, thus reducing the effective dimension of the sampled variable. The proposed chained Langevin dynamics are then combined with annealing schemes and used to train a score-based generative model on toy-datasets and MNIST image datasets.

### Strengths
- The paper makes theoretical contributions on the mode seeking properties of Langevin dynamics with and without annealing and makes methodological contributions to improve SGLD.
- The paper investigates an interesting, and to me novel test-case of having a high-variance low-density background mode. Such settings could be useful benchmarks for sampling and generative modelling, particularly for approaches that have scientific applications in mind.
- The chained Langevin diffusion is a simple, easy to implement concept with potentially high impact in sampling and score-based generative modelling.
- To me it is quite impressive that the LSTM architecture (somewhat untypical for diffusion models) achieves the translation from initial samples to target samples on the image dataset example. However, I don't fully understand how this works, since annealed Langevin is normally initialised from a Gaussian distribution. Some explanations on how the generative model is applied, here, would be helpful.

### Weaknesses
- To me, the paper appears to be two different papers in one. It is hard to make out a story and it is unclear whether the paper attempts to make contributions in Langevin sampling or score-based generative modelling. I think the paper should make a clearer distinction between sampling from a known target distribution (for which we know the Stein score but don't have samples) and generative modelling (for which we have access to samples but not the Stein score). These two settings are quite different, even though they are mathematically related. This also influences how the theoretical and empirical contributions are perceived as a reader.
- The theory is supposed to motivate the chained Langevin algorithm. However, I see a gap between the motivating failure case of SGLD and the proposed chained LD, at least in the main text of the paper. Firstly, chained LD is introduced in Algorithm 1 as a training algorithm of a generative model trained on data. The theoretical results, on the other hand, regard a sampler from a fixed target distribution. However, the paper does not explain why vanilla score-based generative models would fail on mixtures of sub-Gaussian distributions in the same way as SGLD, and thus why chained LD is needed for the training of the generative model. Secondly, the theory of chained LD is not fully developed, and I guess the mixture of sub-Gaussians would not satisfy the implied assumptions of m-strongly log concavity in Proposition 1.
- If the papers main focus is a new sampling methodology, I think it would be good to justify chained LD more thoroughly from a theory point of view, or to make more thorough evaluations of the sampler, comparing it to other adaptions of Langevin sampling and annealing methods. For example, can something be said about the optimal number of patches between $Q=1$ and $Q=d$, or is this a tuning parameter? I am also curious if related strategies to chained Langevin have been developed before. It seems a reasonable approach, as it looks like a mixture of an autoregressive sampler and a Langevin sampler. Putting chained Langevin into context with related works on other variations of vanilla Langevin would be helpful. 
- If the paper proposes a new method for score-based generative modelling, it would be interesting to see how it compares to vanilla diffusion models on standard benchmarks. For example, reporting FID scores of annealed chained LD on standard image datasets like CIFAR10 would typically be expected for a paper at ICLR, even if the scores are not state of the art. Of course it is hard to make improvements, here, since score-based generative models are so optimised, but I could see how slicing the sample into chunks could speed up convergence in some way or another.

### Questions
- I am not familiar with the recent literature on convergence results for Langevin dynamics. I would expect that the difficulty of sampling from multi-modal distributions with unadjusted Langevin algorithms is well-known. While I consider the background noise an insightful novel test-case, how are the theoretical contributions in this paper on the convergence of Langevin algorithms different from existing results on unadjusted Langevin algorithms? 
- What is the trade-off that one needs to make when choosing the number of patches for chained Langevin? Is there an optimal patch number between $Q=1$ and $Q=d$?
- For the score-based generative model, could you clarify what forward process you used? In diffusion models, one typically picks a forward process that converges to a Gaussian distribution and learns the reverse process from Gaussian samples to data. How did you initialise the reverse diffusion in the image translation experiments show-cased in figure 3, where the diffusion is seemingly not initialised from a Gaussian?

### Soundness
3

### Presentation
2

### Contribution
3
