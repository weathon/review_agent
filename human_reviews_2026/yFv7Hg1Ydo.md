# Spatiotemporal Imputation with Graph-Informed Flow Matching

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Missing data is a common challenge in spatiotemporal systems, arising in applications such as air quality monitoring and urban traffic management. Traditional machine learning approaches, like recurrent and graph neural networks, rely on iterative propagation, which tends to accumulate errors over time and space. Recent diffusion-based methods mitigate error propagation but require iterative sampling and often depend on problem-agnostic Gaussian priors, limiting both efficiency and effectiveness. To address these limitations, we propose GiFlow, a Graph-Informed Flow Matching framework for spatiotemporal imputation. GiFlow replaces the typical Gaussian prior with a graph-informed prior constructed via spatiotemporal filtering of observable signals, which better aligns the source distribution to the target and thereby simplifies the generation trajectory. The flow field is parameterized by a hybrid vector field model that integrates spatial attention, temporal attention, and spatiotemporal propagation, enabling joint modeling of spatial and temporal dependencies. Unlike diffusion models, GiFlow is trained via direct regression and supports deterministic, few-step generation at inference. Extensive experiments on both synthetic and real-world datasets with different missing patterns and missing rates demonstrate that the proposed GiFlow outperforms the state-of-the-art approaches in spatiotemporal imputation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a Graph-Informed Flow Matching framework for spatiotemporal imputation and conducts extensive experiments to validate the effectiveness of the proposed method.

### Strengths
- S1： This paper conducts experiments on four datasets from the air quality and traffic domains, considers two types of missing patterns, and reports three categories of evaluation metrics.

### Weaknesses
- W1：As shown in Tables 1, 3, and 4, the performance improvement of GiFlow over the baselines is relatively small.
- W2：The paper lacks comparisons with the most recent spatiotemporal imputation methods from 2024 and 2025, such as ImputeFormer (KDD 2024) and CoFill (IJCAI 2025).
ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation, KDD 2024.
Filling the Missings: Spatiotemporal Data Imputation by Conditional Diffusion, IJCAI, 2025.
- W3：Although the paper claims that the proposed method is more efficient and requires fewer generation steps than diffusion-based models, it lacks theoretical analysis of computational complexity, as well as empirical evaluations of model parameters and runtime efficiency. Therefore, the experimental evidence supporting these claims is incomplete.
- W4：This paper does not provide code for reproducibility.

### Questions
See the weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces GiFlow, a flow matching-based framework for spatiotemporal imputation that replaces the standard Gaussian prior with a graph-informed prior constructed through adaptive spatiotemporal filtering of observations. This design aligns the initial and target distributions, thereby reducing the transport cost and improving generation efficiency. The velocity field is parameterized by a hybrid architecture that integrates spatial attention, temporal attention, and spatiotemporal propagation to jointly capture dependencies across space and time. Unlike diffusion-based approaches, GiFlow is trained via direct regression and allows deterministic few-step inference, achieving competitive or superior results on synthetic, air quality, and traffic datasets under diverse missing data patterns.

### Strengths
- **Timely and relevant topic**: The paper tackles spatiotemporal imputation using flow matching, a rapidly growing and high-interest area that has recently gained attention as an efficient alternative to diffusion models.

- **Novel prior design**: Introducing a structured and more informative prior instead of sampling from a simple noise distribution is a meaningful and elegant idea that improves the efficiency of the generative process.

- **Solid theoretical foundation**: The mathematical formulation of the method is sound and clearly presented, providing good intuition about how the proposed flow evolves between the initial and target distributions.

- **Comprehensive experimental study**: The paper includes a diverse and well-chosen set of experiments, covering synthetic and real-world datasets, which effectively demonstrate the method’s strengths and practical relevance.

### Weaknesses
- **Clarity and intuition**: Although the mathematical foundation of the paper is solid, the method could be explained in a more intuitive manner. Some parts of the derivation and motivation could benefit from additional explanations or diagrams to help readers grasp the underlying idea more easily.
- **Minor writing issues**: There are a few minor typographical or grammatical errors, e.g., the word "geenrative" in line 59, which should be corrected.
- **Figure clarity**: While Figure 1 has an appealing design, it does not clearly illustrate the proposed pipeline or how the components interact. Maybe it could be improved.
- **Computational efficiency not demonstrated**: The paper claims that the method requires fewer inference steps, but this is not empirically analyzed in the main text. Section C.3 mentions using 20 steps, but there is no discussion of how performance scales with the number of steps or how this compares to diffusion-based baselines in terms of speed or computational cost.
- **Missing discussion on consistency models**: Since the paper establishes a connection between flow matching and diffusion models, it would be interesting to include a brief discussion on consistency models. These models can be interpreted as a discrete, distilled, and consistency-enforced formulation of flow matching, aiming for higher inference efficiency. In the context of time-series imputation, the recently proposed CoSTI [2] model follows this line of thought. CoSTI can perform probabilistic imputations in a single step, but requires multiple runs to obtain deterministic estimates such as the median. Including a short discussion or even a small comparison in terms of inference efficiency would help clarify how GiFlow relates to this family of models and where it stands in terms of trade-offs.

[1] Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency models.https://arxiv.org/abs/2303.01469

[2] Javier Solís-García, Belén Vega-Márquez, Juan A. Nepomuceno, and Isabel A. Nepomuceno-Chamorro. Costi: Consistency models for (a faster) spatio-temporal imputation. Knowledge-Based Systems, 327:114117, 2025. https://arxiv.org/abs/2501.19364

### Questions
- Figure 1 clarification: What do the colored dashed lines in Figure 1 represent? It is not entirely clear how they relate to the components of the proposed flow or to the data propagation process.

- Model size and complexity: Could the authors report the number of parameters in GiFlow (and optionally, compare it to baselines)? This would help assess the model’s scalability and computational footprint.

- Addressing reviewer concerns: I would be glad to revise my evaluation if the authors can improve upon some of the points mentioned above.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In this paper, authors provide a flow-based method for spatiotemporal data imputation. It contains comprehensive theoretical discussion and experiments to demonstrate its effectiveness in four dataset.

### Strengths
1. The theoretical discussion of this method is solid.
2. It is necessary to discuss some parameters, such as filtering factors.

### Weaknesses
1. Some ablation study needs to be concluded, such as the spatial temporal components in this method.
2. Are there existing flow-based methods for spatiotemporal imputation? If so, it is necessary to include them in baseline comparisons.
3. Are there some results related to block missing in PEMS datasets?
4. In the introduction, it says that a key limitation of diffusion model is problem-agnostic Gaussian priors. It might be useful to show some visualizations in the dataset to show that it is not a good choice for Gaussian priors.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a generative framework for imputing missing values in spatiotemporal data by integrating flow matching with graph-informed priors. Unlike conventional RNN- or GNN-based models that rely on iterative propagation and suffer from error accumulation, and unlike diffusion-based imputers that depend on Gaussian priors and iterative sampling, GiFlow introduces a deterministic, few-step generation process that directly models the conditional data distribution. The key innovation lies in constructing a graph-informed prior through adaptive spatiotemporal filtering of observed signals, which aligns the source and target distributions and provably reduces transport cost. The model’s vector field combines spatial and temporal attention with spatiotemporal propagation to jointly capture dependencies. Theoretical analysis establishes bounds on the filtering’s receptive field and its influence on transport efficiency. Experiments on synthetic, air quality, and traffic datasets demonstrate that GiFlow consistently outperforms state-of-the-art baselines—both diffusion-based and neural methods—across diverse missing patterns and rates.

### Strengths
1.	Replacing a problem-agnostic Gaussian with a graph-informed prior constructed by adaptive spatiotemporal filtering is well-motivated and technically concrete.
2.	The paper goes beyond intuition: it proves a transport-cost advantage (Theorem 1) of the graph-informed prior over a Gaussian start under FM, and analyzes how filtering factors control the receptive field with an explicit truncation-error bound.
3.	On air-quality (Air-36, AQI) and traffic (PeMS04/08) benchmarks, GiFlow is competitive or superior to RNN/GNN/diffusion baselines under both point- and block-missing regimes evaluated by MAE/RMSE/MAPE.

### Weaknesses
1.	Flow Matching (FM) explicitly allows non-Gaussian, problem-tailored priors; GiFlow’s choice to plug in a graph-filtered prior is a natural (arguably straightforward) instantiation of that flexibility rather than a fundamentally new paradigm. The paper motivates FM vs. diffusion and claims “first” to integrate a graph-informed prior for spatiotemporal imputation, but related ideas—FM with problem-tailored priors—are already known separately. The combination may read as engineering integration rather than a conceptual innovation.
2.	The paper emphasizes “deterministic, few-step” generation as a benefit over diffusion. That’s good for speed, but it gives up calibrated uncertainty, which is the key for imputation in scientific/operational settings. The paper neither quantifies uncertainty nor contrasts GiFlow’s point estimates with probabilistic metrics, so the claimed modeling advantage is just one-sided in speed/efficiency. 
3.	The prior is produced by adaptive spatiotemporal filtering with filtering factors chosen by minimizing equation 6. But X1 is the ground-truth complete signal, unavailable at test time. How are τ selected without leakage? Is there a learned predictor of τ from observables only, or are global τ tuned offline and fixed at inference? This is a key practicality gap that weakens the “adaptive” claim unless clarified and demonstrated.
4.	The paper states code will be open-sourced upon acceptance. Reproduction is impossible in the review process.

### Questions
1. The motivations of using diffusion models-based methods for imputation task should be further clarified. Why can they help to avoid the accumulation of errors compared to RNNs/GNNs-based approaches?

### Soundness
3

### Presentation
3

### Contribution
2
