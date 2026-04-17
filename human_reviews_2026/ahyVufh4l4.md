# Define latent spaces by example: optimising over the outputs of generative models

- Decision: Reject
- Scores: 8, 4, 4, 6

## Abstract
Modern generative AI models like diffusion and flow matching can sample from rich data distributions, but many downstream tasks — such as experimental design or creative content generation — require a higher level of control than unconstrained sampling. Here, the challenge is to efficiently identify outputs that are both probable under the model and satisfy task-specific constraints. Often, the evaluation of samples is expensive and lack gradients — a setting known as black-box optimisation. In this work, we allow black-box optimisation on top of diffusion and flow matching models for the first time by introducing surrogate latent spaces: non-parametric, low-dimensional Euclidean embeddings that can be extracted from any generative model without additional training. The axes can be defined via examples, providing a simple and interpretable approach to define custom latent spaces that express intended features and is convenient to use in downstream tasks. Our proposed representation is Euclidean and has controllable dimensionality, permitting direct application of standard optimisation algorithms. We demonstrate that our approach is architecture-agnostic, incurs almost no additional computational cost over standard generation, and generalises across modalities, including images, audio, videos, and structured objects like proteins.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper deals with sampling of generative models, aiming at identifying outputs that are probable given the estimated density but also that satisfy task-specific constraints. The authors propose to build a simple "surrogate latent space" that allows to conduct directly an optimization with regards to various tasks. The approach is evaluated in the context of text-to-image generation and protein design, exhibiting interesting results when one considers the low computation cost of the proposed approach.

### Strengths
* the authors adopt a constructivist approach to define their surrogate latent space, stating by establishing key principles it should support then explaining, step by step, how the space is defined and which properties they assume on the bijective functions that allow the mapping to the latent space of the model. The make a real effort of clarity, referring to the appendices for mathematical details and variations in their approach. Each time, the assumptions and their implementation are well motivated.
  - they also clearly credit previous works, in particular (Bodin et al., 2024) 

* the related work is short but efficiently highlights the novelty of the proposed approach with regards to three families of approaches.

* the proposed approach is agnostic to the generative model considered (as long as it has a latent space with -- quite common -- properties) and thus the type of data handled.
  - The approach is evaluated in two very different tasks, namely two experiment in text-to-image generation and one for protein design. Each time, the approach is compared to a recent and relevant baseline.
  - The quantitative results (Table 1) are conducted over 30 repetitions and the results report the median and 90% confidence interval.

### Weaknesses
* the experiment on images (section 5.2) is conducted for three prompts only. However, a previous work such as (Denker et al, 2025) report their results on one prompt only. They nevertheless conduct a more ambitious experimetn on the 10 classes of MNIST (although this dataset may not be the most relevant). One must also note that the experimemnt in section 5.3 also deals with text-to-image generation and is conducted at a much larger scale. Hence, the experiment of section 5.2 can be considered as an illustrative example and is thus, in that sense, well conducted. 
  - for future works (*not* rebuttal) one can suggest to conduct a similar experiment on a larger number of prompts and reporting aggregated performance. Reporting the 90% confidence interval for individual prompts is nevertheless more intersting than reporting an "average median performance and its standard deviation".

* In Table 1, the quantitative results are better for the proposed approach for two over three prompt, as highlighted, but one must consider various variant of the approach to get this result. However, the proposed approach is much less costly to compute and, in that sense, the results can be considered as intersting in comparison to the threee baselines.

* The results for protein design (Table 2) are reported in the appendix only. One can understand that there is a limited place but the main values should at least be reported in the main text, on lines 467-472.

* minor:
  - equations line 240 and 323 are not numbered
  - Snyder (1987) --> (Snyder, 1987) -- or ... problem *of* Snyder (1987)
  - although Bayesian Optimisation (BO) is defined on line 064, the full term could be used in the related works, line 341 (there is enough place)
  - references on lines 431 shoul use `\cite` instead of `\citet`
  - reference (Denker et al, 2025) line 510 reports the arxiv preprint while it has been published at Frontiers of Probabilistic Inference workshop at ICLR 2025.
  - the ICLR 2026 guideline asked for a dedicated section in the appendix explaining possible LLM usage. See "The Use of Large Language Models (LLMs)" on [this page](https://iclr.cc/Conferences/2026/AuthorGuide)

### Questions
Overall, the weaknesses identified are fairly minor and could be corrected by slightly reorganizing/rewriting certain sentences.

### Soundness
4

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
3

### Summary
This paper develops a method for optimization of deterministic generative models over arbitrary value functions. It does so by constructing a surrogate latent spacing using seed samples, which in combination with blackbox optimization procedures (like BO), recovers higher scoring samples than the seeds themselves, and in comparison to random search in latent space. The method is demonstrated on a number of modalities, including text-conditioned image generation, audio, video, and protein sequences.

### Strengths
- The paper tackles an important problem in optimizing arbitrary functions for deterministic-trajectory generative models with high dimensional latent space (e.g., DDIM, flow matching, etc.). The problem is clearly stated in the paper.
- The proposed solution is elegant and intuitive, appears to perform well on different modalities and without training, and the qualitative results are convincing.
- The paper is overall clearly written, though some critical sections on methodological details were somewhat dense.

### Weaknesses
- In general the paper may be suffering from having too many results with too little space to describe all of them in full, but critical parts are then not sufficiently described.
- This may be a lack of familiarity on my part, but section 3.1 and 3.2 could perhaps use some detail, as I was confused about the dimensionalities of the intermediate spaces and how a bijective mapping back onto the higher-dimensional latent space (z) is achieved with uniqueness guarantees. Perhaps some graphic intuition would help.
- The quantitative evaluations could be more systematic, or possibly just more clearly presented. The main result on the quality of the method is Figure 4, which, if I understand correctly, are evaluated over 1 million prompts of the same variety with the 3 different features involving vehicles in different terrains. The method clearly does better than random search, though the scoring seems arbitrary / badly scaled since a black image scores 20. More generally, it does not compare against other, more competitive methods as presented in Table 1, which seems to be only shown for the 3 prompt value functions in the different columns. Similarly, for the other modalities, while examples of the surrogate latent space and samples are given (e.g., for audio and images), I missed their quantitative evaluations, and in relation to other methods. Overall, it’s clear from the results that the proposed method works, but I’m unsure to what extent (how well relative to others) and in what settings. To be clear, I’m not saying it has to beat SOTA on every evaluation, just that it’s somewhat unclear what and how systematic the experimental setups are.
- Main result from Table 1 is essentially interpreted in one line (line 433), but there seems to be more insights to be gained by comparing across the different setups, and the table lacks highlighting or just some kind of visual guidance to help the readers make their conclusions.

### Questions
- What is the dimensionality of epsilon? Is the dimensionality of w = K? If so, how is the transformation to/from the lower-dimensional unit sphere orthant (w) mapped to the original higher-dimensional z- and x-space with bijective mappings (line 210-222)?
- intuitively, do the K seeds need to satisfy some kind of diversity criterion to construct a  latent space with a meaningful span? If so, how to evaluate this? Table 1 reports results on the quality and diversity of downstream generation with oversampling (e.g., best 1-of-6), but not relative to the quality / diversity of these seeds.
- are there systematic quantitative evaluations beyond the 3 example prompts in Table 1?
- Just to clarify, Fig 4 median and CI are computed over 1 million targets from the combination of the 3 attributes?
- typos on line 395 and 397? 100/500 should be 200 and 600?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a method to map high-dimensional representation spaces of deep generative models to low-dimensional latent spaces using example-based transformations. The approach is training-free and can be applied across various generative models for different modalities. It is qualitatively evaluated on image and protein datasets and quantitatively compared with baseline methods that require additional training, achieving comparable performance.

### Strengths
1. The paper addresses an interesting problem of enhancing controllability in modern generative AI models when generating new samples, which could broaden their range of applications. Considering that real-world images are often complex, defining latent spaces based on given examples is a reasonable and intuitive idea. 
2. The latent spaces constructed by the proposed method satisfy desirable properties such as validity, uniqueness, and approximate stationarity. 
3. The proposed method is evaluated on two distinct data modalities, and the results are promising.

### Weaknesses
1. Since the latent space is determined by the provided examples, it would be helpful to clarify the assumptions or requirements for selecting these examples. For instance, in Figure 4, should the examples collectively cover all key elements (e.g., matte black, dull hovercraft vehicle, vineyards) in the target image? 
2. Table 1 compares several sampling strategies for the proposed method, but it remains unclear which strategy performs best or how to choose among them. Providing selection criteria or guidance would strengthen the paper. 
3. According to Table 1, the proposed method does not significantly outperform baseline methods on quantitative metrics such as Reward and Diversity. While its training-free nature is a clear advantage, the reported training cost of some baselines (e.g., 4 GPU hours for Adjoint Matching) is not excessively high. Furthermore, including the inference or generation time (e.g., the average time to generate a finalized target example) would make the performance comparison more comprehensive.

### Questions
See "Weaknesses"

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the task of Latent Space Optimization (LSO) in diffusion/flow-based generative models. Authors argue that naive LSO in such models suffers from (i) Inefficient optimization due to the high dimensionality of the latent space, and (ii) difficulty to  stay on the valid support of the generative model. To address these challenges, authors propose a method for constructing lower-dimensional "surrogate latent spaces" for optimization with 3 main properties:
- **Validity** of all the latent within the space
- **Uniqueness** of the mapping between each latent in the surrogate space to the generated outputs.
- **Approximate** stationarity• of the surrogate space, where similarity between the generated objects approximately depends on the euclidean distance between their latents in the proposed space.

The proposed method constructs a small surrogate search space $U$ around a set of seed examples. To ensure uniqueness, it maps each $u\in U$ through a Weight Chart (Angular or Knothe–Rosenblatt) to a unit-norm, non-negative weight vector and uses a deterministic inversion/decoding path. To ensure validity, seed latents are combined in an inner latent via Latent Optimal Linear transport and then mapped back to the model latent. The authors argue that $U$ is approximately stationary because (i) the Weight Chart ties similarity largely to the dot product of the unit weights—hence to Euclidean distance in $U$—and (ii) in the isotropic inner latent, cosine similarity concentrates and is well-approximated by that weight dot product.

The authors evaluate their approach across multiple modalities (images, audio, video, proteins) against baselines operating in the original latent space. The authors also provide ablations and visualizations in support of the proposed method.

### Strengths
- The proposed method is training-free and is demonstrated across multiple modalities (images, audio, video, proteins).

- The proposed method keeps optimization on the model’s support (on-manifold), addressing a core challenge in latent-space optimization.

- Defining the search space from seed examples yields a more interpretable and steerable optimization process.

- Results
	- The method matches or exceeds fine-tuning–based baselines on the evaluated image benchmark without additional training.
	- The experiments show standard black-box optimizers (CMA-ES, BO) succeed inside the surrogate space, whereas the same optimizers often fail in the original latent (e.g., CMA-ES collapses to black images).
	- In the protein design experiments, it improves recovery metrics and the number/diversity of successful designs compared to standard sampling.

### Weaknesses
- Seed dependency
    - The attainable solution set is limited by the span, diversity, and quality of the chosen seed examples.
    - Each new optimization target (e.g., prompt or objective) typically requires its own seed set to construct the surrogate space.

- Approximate stationarity lacks a formal guarantee and may degrade as the surrogate dimension grows.
 
- Minor: The method assumes (quasi-)deterministic inversion/decoding; using stochastic samplers or strong guidance can weaken the uniqueness and approximate stationarity properties.

### Questions
See the weaknesses

### Soundness
3

### Presentation
4

### Contribution
3
