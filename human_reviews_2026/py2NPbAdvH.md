# Discrete Bayesian Sample Inference for Graph Generation

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Generating graph-structured data is crucial in applications such as molecular generation, knowledge graphs, and network analysis. However, their discrete, unordered nature makes them difficult for traditional generative models, leading to the rise of discrete diffusion and flow matching models. In this work, we introduce GraphBSI, a novel one-shot graph generative model based on Bayesian Sample Inference (BSI). Instead of evolving samples directly, GraphBSI iteratively refines a belief over graphs in the continuous space of distribution parameters, naturally handling discrete structures. Further, we state BSI as a stochastic differential equation (SDE) and derive a noise-controlled family of SDEs that preserves the marginal distributions via an approximation of the score function. Our theoretical analysis further reveals the connection to Bayesian Flow Networks and Diffusion models. Finally, in our empirical evaluation, we demonstrate state-of-the-art performance on molecular and synthetic graph generation, outperforming existing one-shot graph generative models on the standard benchmarks Moses and GuacaMol.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Discrete Bayesian Sampling Inference (BSI), a novel framework for graph generation. The core idea is to model the generative process not through discrete state transitions, but by evolving a continuous belief state, which is argued to better capture the dynamics of discrete variable evolution. The authors provide a theoretically grounded framework to derive the training and sampling methodologies. GraphBSI is evaluated on molecular graph generation benchmarks, where it demonstrates significant performance improvements over current state-of-the-art models.

### Strengths
- The proposed Discrete BSI presents a compelling and theoretically interesting mechanism for discrete data generation. The idea of operating on a continuous belief state to model the evolution of discrete variables addresses a key challenge in generative modeling.

- The application of BSI to graph generation yields significant performance improvements over state-of-the-art methods on established benchmarks like Moses and GuacaMol. This highlights the practical efficacy of the approach.

- The theoretical foundations are presented rigorously, with (mostly) well-stated theorems and proofs that are easy to follow.

### Weaknesses
- My primary concern is the precise positioning of the paper's contribution. It is somewhat ambiguous whether the main novelty lies in the general framework for discrete generation (placing it alongside with discrete diffusion, flow models, etc.) or specifically in its application to graph generation. Clarifying this would significantly strengthen the paper's impact by helping the reader understand its broader context and significance. Honestly I would be happy to improve the score if this question is properly addressed. 

- Insufficient Ablation of Performance Gains: While the empirical gains are impressive, the source of these improvements is not fully elucidated. The provided ablation studies on noise levels and time distortion are helpful, but a deeper analysis is needed to isolate which specific components of the GraphBSI model are most critical to its success.

- The theoretical framework is solid but I would appreciate if the authors can provide some easy-to-follow explanations alongside. Some of the results are slightly against intuition (e.g. Theorem 2 where the lower bound is a squared error for categorical variables) and can be difficult to understand at the first glimpse. Also, I would appreciate if the authors can compare their methods with a few existing continuous state discrete diffusion/flow models, both theoretical and empirical. Such as [1, 2, 3]. 

[1] H. Stark. et al. "Dirichlet Flow Matching with Applications to DNA Sequence Design". ICML 2024

[2] O. Davis. et al. "Fisher flow matching for generative modeling over discrete data". NeurIPS 2024

[3] H. Zheng. et al. "Continuously augmented discrete diffusion model for categorical generative modeling". Arxiv.

### Questions
- Q1: Though the title states that the model is designated for graph generation, my main question is whether the model is specifically working for graph generation, or it can bring similar improvements to other discrete generation tasks. If this is designed for graph generation, I would like to see more on how can BSI improve the graph generation? If not and BSI can generally improve the discrete generation, it might be ideal to conduct generative tasks to modalities other than graphs, such as text modeling, protein co-design, etc[1][2]. 

- Q2: I look into the appendix and understand that the loss in eq. 5 is derived through the lower bound of log-likelihood. Honestly this is a bit surprised as from the assumption, the x is sampled from a softmax version of z, it seems cross-entropy loss to be more reasonable. Could the authors elaborate more on advantage of choosing the loss to be squared distance ? I am convinced with the derivation but It just seems a bit unnatural. 

- Q3: In the abstract, the authors claim that the advantage of BSI comes from introducing a continuous belief z that can better capture the evolution of dynamics. Would the authors provide more evidence/justification/intuition on why this works? 


[1] A. Campbell et al. A Continuous Time Framework for Discrete Denoising Models. NeurIPS 2023

[2] A. Campbell et al. Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design. ICML 2024

### Soundness
2

### Presentation
3

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
The paper proposed a method based on BSI for graph generation, achieving outstanding performance.

### Strengths
The paper clearly presents the method being used. It is easy to follow.

It has a thorough structure, including theoretical guarantees and experimental analysis.

The experimental results are impressive, especially with only 50 steps.

### Weaknesses
Regarding the soundness of the paper, it is not clear how this work relates to graphs, and it seems mainly adapted for discrete data, and the transformers and graph features mainly follow the previous work. I would like to ask the authors to clarify whether they are the first to adapt BSI to discrete data modalities. If yes, is the main approach is adding softmax operations on the original feature z used in common BSI?

Beyond that, I kindly ask the authors to clarify better the relationship between BSI and BFN. The paper states that BFN is a more generalizable version of BSI, but a more thorough discussion would help the reader understand the connection better. In terms of application, how does this help in your implementation or analysis, and what can BFN not achieve?

Additionally, mathematically, what is the difference between this method and a continuous diffusion on the simplex space, or a continuous diffusion on the logits space with softmax applied? If different, what makes BFN more advantageous than these normal diffusion formulations?

Clarifying these points would help me better understand the contribution of the paper.

### Questions
1. Which preprocessing procedure (which version of the implementation) the authors apply for the MOSES dataset?
2. Do you have results with 5 or 10 steps, since the method already reaches very high performance with only 50 steps?
3. How do the authors evaluate the robustness of the framework, separately in terms of hyperparameters during training, and sampling?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents GraphBSI, a new one-shot generative model for discrete graphs based on Bayesian Sample Inference (BSI).
Unlike conventional diffusion or Bayesian Flow Network (BFN) models, GraphBSI performs generation by refining a belief distribution in parameter space rather than evolving discrete samples directly. The authors derive a categorical BSI formulation and show that in the continuous-time limit, it becomes a stochastic differential equation (SDE). They further generalize this to a family of SDEs with a noise control parameter γ, allowing smooth interpolation between deterministic probability-flow ODEs and stochastic samplers. On standard molecule generation benchmarks (GuacaMol and MOSES), GraphBSI achieves  superior results across most metrics.

### Strengths
Extends the Bayesian Sample Inference framework to discrete graphs.

Introduces both Euler–Maruyama and Ornstein–Uhlenbeck discretization schemes.

Outperforms baselines on key metrics with fewer sample steps.

### Weaknesses
How sensitive is GraphBSI to the choice of the precision schedule β(t)? The paper only mentioned a monotonically increasing schedule.

Since GraphBSI is similar to BFN, a direct comparison with GraphBFN regarding training time would better highlight its practical advantages.

In addition, GraphBFN is not included in Table 2. Including GraphBFN results would allow for a clearer assessment of GraphBSI’s improvements over prior BFN-based approaches. The sampling_step=50 for GraphBSI is also missed in Table 2.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper extends Bayesian Sample Inference (BSI) to categorical/graph generation, introduces an SDE view with a controllable noise family ($\gamma$), and studies two discretizations (EM/OU). On molecular benchmarks, the method achieves competitive performance with relatively few function evaluations (e.g., 50/500 NFE). The contribution is integration-oriented: it adapts BFN/BSI ideas to discrete graphs with a clearer Bayesian update and an SDE formulation, offering a tunable trade-off between speed and quality. The writing is generally clear, but several theoretical statements and implementation details would benefit from refinement.

### Strengths
- **Originality**: Systematically ports BSI to **discrete/categorical graphs** and unifies deterministic ODE and stochastic sampling paths via **SDE + controllable noise**. This “bridging” removes some practical limitations of prior BFN-style derivations and constitutes originality via domain adaptation and simplification.  
- **Quality**: A clear training objective (ELBO) and a family of samplers (controlled by $\gamma$) are provided, with strong **inference efficiency** (competitive at low NFE). Ablations on $\gamma$ and time grids are helpful; engineering appears solid.  
- **Clarity**: Key intuitions (“belief contraction,” “same marginal family”) are conveyed through diagrams and concise algorithms; the separation of main text and appendix reads well.  
- **Significance**: For **discrete structure generation** (notably molecular graphs), the **low-step, efficient** generation is practically relevant. Methodologically, it offers a clean, implementable link between BFN/BSI and diffusion-SDE paradigms, with potential impact on broader discrete domains.

### Weaknesses
1) **ELBO Reconstruction Term** In the main theorem, the reconstruction term is written as $\mathbb{E}[p(x|z)]$ instead of the standard $\mathbb{E}[\log p(x|z)]$. While the conclusion that this term is independent of $\theta$ still holds, the ELBO should use the expected log-likelihood. Please unify the notation in the main text and appendix.

2) **Weighting in Theorem 5** To be strictly consistent with the loss in the main text, a factor $\tfrac{1}{2}$ is needed, i.e.,  
$$
\lambda(t) = \frac{\beta'(t)}{2} \cdot \frac{(\beta(t) + \beta_0)^2}{\beta(t)^2}.
$$  
The current statement may double the scaling. A short clarification aligning Theorem 5 with Eq. (5) would help.

3) **Over-Strong “Affine Equivalence” Wording** The appendix claims an affine equivalence between the receiver distribution and BSI observations, but the covariance scaling does not match exactly. A safer phrasing is “same-order approximation in the small- $\alpha$ limit,” avoiding over-commitment.

4) **Final Decoding Inconsistency** The text mentions sampling from $\mathrm{Cat}(\mathrm{softmax}(z))$ while algorithms return `Quantize(f_\theta(\cdot,1))`. Since sampling vs. quantization affects diversity and faithfulness differently, please unify the terminology and state the default choice with a brief rationale.

5) **Empirical Support for “Same Marginal” is Indirect** The claim that different $\gamma$ share the same marginal is mainly supported through downstream metrics. A lightweight check in $z$ -space (e.g., comparing first and second moments at a few $t$, $\gamma$) would make the claim more tangible—no large-scale new experiments needed.

6) **Comparative/Robustness Details Could Be Lightly Strengthened** If space permits, adding small variance bands over a few seeds on key plots or clarifying a “same wall-clock budget” comparison protocol would improve perceived fairness and reproducibility.

### Questions
1) Could you add a minimal appendix probe comparing means/covariances of $z_t$ across a few $t$ and $\gamma$ values as an intuitive check for the “same marginal” claim? This can be very lightweight.  
2) For Theorem 5, do you plan to include the missing $\tfrac{1}{2}$ factor in a revision/erratum and explicitly state the condition under which it is strictly aligned with Eq.(5)?  
3) For the final decoding, which default path do you recommend—**categorical sampling** or **argmax quantization**—and why (stability, diversity, alignment with prior work)?  
4) If adding figures is difficult, would you consider providing a reference configuration in the repo for a “**same wall-clock budget**” comparison (fixed batch/GPU/time) so others can verify fairness?  
5) For the “affine equivalence” phrasing, would you consider noting in the appendix that “same-order approximation” is more accurate when covariance scalings differ, to avoid readers inferring strict equivalence?

### Soundness
3

### Presentation
4

### Contribution
4
