# Tracking Temporal Dynamics of Vector Sets with Gaussian Process

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6

## Abstract
Understanding the temporal evolution of sets of vectors is a fundamental challenge across various domains, including ecology, crime analysis, and linguistics. 
For instance, ecosystem structures evolve due to interactions among plants, herbivores, and carnivores; the spatial distribution of crimes shifts in response to societal changes; 
and word embedding vectors reflect cultural and semantic trends over time. 
However, analyzing such time-varying sets of vectors is challenging due to their complicated structures, which also evolve over time. 
In this work, we propose a novel method for modeling the distribution underlying each set of vectors using infinite-dimensional Gaussian processes. 
By approximating the latent function in the Gaussian process with Random Fourier Features, we obtain compact and comparable vector representations over time. 
This enables us to track and visualize temporal transitions of vector sets in a low-dimensional space.
We apply our method to both sociological data (crime distributions) and linguistic data (word embeddings), demonstrating its effectiveness in capturing temporal dynamics. 
Our results show that the proposed approach provides interpretable and robust representations, offering a powerful framework for analyzing structural changes in temporally indexed vector sets across diverse domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a novel approach to modeling and visualizing the temporal dynamics of sets of vectors. The central idea is to represent each time-dependent vector set as probability distributions using Gaussian Processes approximated via Random Fourier Features (RFF). This produces compact K-dimensional representations that can be tracked and visualized over time using PCA.
Experiments have been conducted on synthetic and real data.

### Strengths
Coupling the Gaussian Process modeling with a finite dimensional representation using Random Fourier Features is promissing.
The initial results suggest potential benefits.

### Weaknesses
The overall method sounds interesting but lacks clarity.

•	Some estimated densities show high values in empty regions. The heat maps do not reflect completely the behaviours.
•	PCA interpretation is not always straightforward (PC1/PC2 excluded for crimes). How much variance is explained? And why choosing PCA3 and 5 sometimes and how much variance do they explain?
•	Part of the method is unclear: it uses Metropolis-Hastings for optimization although these methods are samplers. How do you use this for optimization?
•	The crime data lacks connection to external factors (e.g., policy changes etc.. ) for interpretation

In summary, it is an interesting idea to represent evolving vector sets as trajectories in RFF weight space, but execution and validation need strengthening.

### Questions
See weakness and related questions.

### Soundness
2

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
This paper proposes a framework for modeling how sets of vectors evolve over time using Gaussian Processes (GPs) approximated via Random Fourier Features (RFF). By representing each vector set as a K-dimensional embedding derived from a GP-based density model, the method captures global distributional structures at each time step. Temporal changes are then visualized through PCA projections of these embeddings, enabling interpretable analysis of dynamics in a low-dimensional space. Experiments on synthetic data, Chicago crime distributions, and semantic shifts in English words demonstrate that the approach effectively captures both spatial and semantic transitions.

### Strengths
- The problem of tracking the temporal dynamics of vector sets is important.
- The authors conducted experiments using multiple synthetic and real datasets with the proposed method and examined the analysis results.

### Weaknesses
- The proposed method using RFF-based GP and PCA is merely a combination of existing methods and lacks novelty.
- Approaches considering the temporal evolution of weights for basis functions, such as spectral methods, have been widely used for a long time.
- The advantage of using Gaussian processes is not clear.

### Questions
- What does the label z in equation (2) refer to?
- It might be interesting to consider the time evolution of weight coefficients in continuous time, as in spectral methods for numerical simulation.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposed to model the temporal evolution of a set of vectors via Gaussian processes (GPs). GPs are hard to compute due to its use of kernel method in the covariant matrix, so random Fourier features (RFFs) are used to approximate the inference of GPs.

### Strengths
- The proposed method is a simple yet elegant way of dealing with vector sets.

### Weaknesses
- The novelty of this paper is somewhat limited. The main technical contribution here is applying GPs to vector sets, and the idea of using RFFs to work with the inference of GPs are not novel.
- The authors used a 2-dim PCA to visualize the multi-dimensional data. Why don't use a better method such as t-SNE?
- There is no comparison with other related methods that models evolution of vector sets.

### Questions
- L230: I think it should be $b_k \sim \mathrm{Unif}[0, 2\pi] $, not $[0, 1]$.

### Soundness
2

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
1

### Summary
This paper proposes a novel framework for modeling and visualizing how sets of vectors evolve over time, a problem relevant to diverse domains such as ecology, crime analysis, and linguistics. The core idea is to represent each time-indexed vector set as a distribution modeled using Gaussian Processes, and then to obtain a compact, comparable vector representation through Random Fourier Features.

By sampling cosine basis functions and estimating their weights, the authors express each vector set as a K-dimensional weight vector. Applying Principal Component Analysis to these weight vectors across time enables interpretable low-dimensional trajectories that capture the temporal transitions of distributions.

### Strengths
1. The paper introduces an innovative approach to representing and tracking the evolution of vector sets over time by combining Gaussian Processes with Random Fourier Features. 
2. The paper leverage PCA method to provide intuitive understanding of temporal dynamics.
3. The paper is well-structured, with clear methodological exposition, illustrative figures, and detailed experimental setups.

### Weaknesses
1. It seems like the experiments, while diverse (synthetic, crime, and linguistic datasets), remain primarily qualitative.
2. The paper applies a single set of hyperparameters (e.g., RFF dimension K=30, Gaussian kernel bandwidth) across all experiments, without justification or sensitivity analysis.

### Questions
1. The model estimates RFF weights via a Metropolis–Hastings sampler, which can be computationally intensive. What is the runtime and scalability behavior with respect to the number of samples, time steps, and feature dimensions?
2. Have you considered how these temporal trajectories could be used in downstream tasks?

### Soundness
2

### Presentation
2

### Contribution
2
