# Fast Generation, Forecasting, and Imputation of Multivariate Irregular Time Series with OUFlow

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
We propose OUFlow, a general-purpose time-series generative model that robustly handles irregular sampling and generates sequences at arbitrary time points. OUFlow integrates latent dynamics governed by a mixture of Ornstein-Uhlenbeck processes with expressive target distributions via normalizing flows. Leveraging our analytically derived, efficiently computable likelihoods and posteriors for high-dimensional time series, OUFlow supports unconditional time-series generation, probabilistic forecasting, and imputation from partial observations within a unified model after a single training phase. It also enables explicit likelihood evaluation (e.g., for anomaly detection), clustering via modes of the latent OU process, and, in some cases, denoising under noisy supervision. By exploiting parallelization through the scan algorithm, OUFlow attains logarithmic runtime scaling in the number of generated points, while maintaining high accuracy in all three tasks. Comprehensive experiments on both synthetic and real-world datasets demonstrate that OUFlow consistently outperforms other models capable of all three tasks, in both generation quality and computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel generative model OUFlow, focusing on three core tasks (generation, forecasting, and imputation) for multivariate, irregularly sampled time series data.

### Strengths
1. The paper has solid academic theory, with rigorous logic for the recommendation of the OU method, and the appendices supplement detailed formula derivations.
2. The experimental part is comprehensive, covering the three tasks mentioned in the paper as well as hyperparameter design.
3. The writing is well-organized, and all the figures follow a consistent, standard format.

### Weaknesses
1. Writing: What is "unconditional generation"? There are many formula derivations; although the proofs are sufficient, they seriously affect reading.
2. There is a slight disconnection between the Introduction section and the experimental section. The Introduction mainly emphasizes advantages in time complexity, but the experimental section contains only limited discussions on time complexity, and the results are not best.
3. Regarding the authors' statement that "making it suitable for broader applications such as anomaly detection", I did not find related discussions in the main text.
4. The paper has solid theoretical calculations, but its main goal is "Fast GENERATION". However, when look at the time complexity analysis and experimental results (Table 2, Table 3), the supposed "fast generation" advantage isn’t actually shown. What’s more, for time series tasks, the paper only uses a few datasets for comparison, which makes the experimental results less convincing.

### Questions
1. Table 1 shows that OUFlow does not show significant advantages in terms of time complexity. This can be observed in Table 1, Figure 3, Figure 4, and Figure 5, which is still a certain gap compared with the ACSSM method. The authors should provide a more sufficient explanation for this.
2. For the imputation task design, why is the random mask method not adopted, but instead a long time segment is selected for imputation? Meanwhile, the explanation given by the authors for the dataset shuffling method is somewhat far-fetched. These settings are different from those of previous imputation tasks.

### Soundness
3

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
3

### Summary
This paper presents OUFlow, a novel generative model designed for generation, forecasting, and imputation of multivariate irregular time series. The core innovation lies in integrating Ornstein–Uhlenbeck (OU) processes, linear observation, and normalizing flows to avoid the numerical integration steps required by traditional differential equation-based models. Experimental results on multiple datasets demonstrate that the proposed model outperforms existing methods in terms of both generation quality and computational efficiency.

### Strengths
1. The study addresses the challenging and highly relevant problem of generating multivariate regular time series—a topic of significant interest in both academic and applied contexts.
2. The model design exhibits a degree of novelty by combining the analytical properties of OU processes with the expressive power of normalizing flows, representing an interesting and promising direction.
3. The experimental section of the paper is quite good, covering the three core tasks of generation, forecasting, and imputation. The validation across multiple datasets demonstrates a substantial amount of work, which is commendable.

### Weaknesses
1. Although the paper claims OUFlow is a novel generative model, its core components (mixtures of OU processes, linear observation models, and normalizing flows_ are all established techniques. The primary contribution appears to be the combination and application of these methods to time series tasks, rather than the introduction of a fundamentally new theoretical or architectural breakthrough.
2. The paper states in Table 1 that OUFlow’s generation complexity is O (N+log(N+K)), repeatedly emphasizing its efficiency. However, the model’s theoretical time complexity is not optimal. Moreover, while Figure 5 compares generation time against the number of generated time points, it neglects the equally important impact of the number of observations on performance. Additional experiments are necessary to fully substantiate the claimed superiority in generation speed.
3. For experiments​:
a) While the selected baseline models (e.g., LatentSDE, DSPD) are reasonable, the paper overlooks several state-of-the-art generative models based on structured state space models (SSMs). Given the strong performance of recent SSM-based architectures (e.g., Mamba) in time-series modeling, comparisons on forecasting and imputation tasks should be included to ensure a comprehensive evaluation.
b) The original design intent and primary application scenarios of the ACSSM model are focused on forecasting and imputation tasks. The experimental results in this paper show that the performance of the OUFlow model is quite similar to ACSSM on forecasting and imputation tasks, yet demonstrates a significant performance gap on the generation task. It is recommended to supplement the comparisons with models specifically designed for generation tasks to enable a more equitable assessment of the model's comprehensive capabilities.
c) Evaluation criteria for generation tasks are not yet standardized across existing research, with different models employing varied methodologies, thereby reducing the comparability of results. For instance, the DSPD model utilizes a discriminative evaluation approach, where a classifier is trained and an accuracy rate of 50% is deemed the optimal indicator of generation quality. In contrast, this paper relies solely on the Mean Time-Averaged Energy Score (Mean TAES) as a single metric, failing to delve into the critical temporal properties of the generated sequences (such as spectral characteristics, long-term dependencies, or inter-variable correlations). 
d) More mainstream datasets in the community should be used.
4. Section 3.5 notes that "in many cases, only a subset of modes is effectively learned, with the weights for all other modes approaching zero," yet the appendix D.3 suggests that increasing the number of modes improves performance. These statements appear contradictory. More detailed ablation studies are needed to clarify the influence of Mand the necessity of auxiliary loss functions.
5. About writing quality: 
a) The abstract is overly brief and fails to adequately summarize the methodology and contributions.
b) The introduction begins with references to large language models and image generation, which are only loosely connected to the main topic of time-series analysis, weakening the focus.
c) The logic in the third paragraph is unclear and lacks emphasis, making it difficult to discern the key points.

### Questions
As in weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a time series modeling framework for forecasting, generation and imputation built on the idea of switching OU latent process which maps to observations through a normalizing flow, thereby defining an invertible transformation between the latents and the observations. This modeling choice allows the system to model irregularly spaced points and thus generalize to more real-world scenarios. The experiments indicate that it leads to improved performance as well as diverse generations, and in particular the choice of linear parameterizations leads to closed form evaluation of a number of relevant conditional distributions.

### Strengths
- The authors combine strong generative models (normalizing flows) with a latent OU process to model irregularly sampled time-series, pushing the frontier of real-world time-series modeling.
- The design choices allow computing of important conditional distributions tractable in closed form, thereby allowing ease of both sampling as well as likelihood computation.
- Empirically, they show that the proposed approach outperforms some of the baselines that they consider, built on top of state-space models as well as diffusion-style generative models.

### Weaknesses
- A potential weakness of the proposed method is that the choice of linear evolution of the latents is governed at the trajectory level as opposed to dynamically updated throughout the time-series, which feels quite limiting and non compositional. 
- While the authors compare to some of the baselines, it would be nice to have a comparison against Time-Grad (Rasul et. al) which is one of the predominant methods for doing time series forecasting.
- Most of the quantitative results highlight performance as opposed to training and inference time, for both generation and forecasting/imputation tasks. It would be good to get a comparative estimate of the time required to train the proposed model (in contrast to baselines) as well as the computational cost at inference.

The authors should compare and contrast with the following existing works that look at time series forecasting using continuous time samplers as well as switching linear systems:

*Chen, Yu, et al. "Recurrent interpolants for probabilistic time series prediction." arXiv preprint arXiv:2409.11684 (2024).*

*Linderman, Scott W., et al. "Recurrent switching linear dynamical systems." arXiv preprint arXiv:1610.08466 (2016).*

*Halmos, Peter, Jonathan Pillow, and David A. Knowles. "System Identification for Continuous-time Linear Dynamical Systems." arXiv preprint arXiv:2308.11933 (2023).*

*Rasul, Kashif, et al. "Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting." International conference on machine learning. PMLR, 2021.*

### Questions
- Maybe I did not understand this part correctly, but how do the authors parallelize training given the latents follow a linear dynamical system? What are the assumptions and simplifications made to allow for this parallelized training as opposed to ODE-RNN?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces OUFlow, a unified framework that innovatively combines normalizing flows with Ornstein-Uhlenbeck SDEs for time series generation, forecasting, and imputation. Its key strengths include enabling exact likelihood computation, continuous-time modeling, and scalable inference by leveraging analytic solutions to SDEs, thus avoiding numerical solvers.

However, the work has significant limitations. The motivation is unclear, lacking a definitive problem statement or justification for why existing models are insufficient. The core assumption of a linear SDE process is potentially restrictive and not well-justified for complex, real-world dynamics. Empirically, the study is limited to a few clean, periodic datasets, raising concerns about generalizability to noisy or high-dimensional real-world data. Furthermore, it lacks a theoretical analysis of the model's expressiveness or convergence guarantees.

### Strengths
S1. The paper addresses time series generation, forecasting, and imputation under a unified framework, using appropriate metrics like energy distance and TAES, demonstrating a comprehensive and rigorous evaluation across multiple tasks.

S2. OUFlow creatively combines normalizing flows with Ornstein-Uhlenbeck-based SDEs, enabling exact likelihood computation and continuous-time modeling, offering a theoretically grounded alternative to discrete diffusion models.


S3. By leveraging analytic solutions to linear SDEs, the model avoids numerical solvers, ensuring stable training and scalable inference—particularly beneficial for long or irregularly sampled time series.

### Weaknesses
W1. The paper introduces OUFlow as a normalizing flow model for time series, but it fails to clearly define the core problem it aims to solve. The motivation is scattered. There is no discussion of why existing SDE-based or flow models are insufficient for the tasks, nor a compelling real-world use case that necessitates the proposed approach. This weakens the paper's narrative and perceived impact.

W2. The model assumes the latent process follows a linear SDE with block-diagonal drift matrix (Eq. 53). While this enables analytic solutions (Eq. 55), it is a strong and potentially restrictive assumption. Real-world time series often exhibit nonlinear, non-stationary, or high-dimensional dynamics. The paper does not justify why a linear OU process is sufficient or how the model would generalize to more complex dynamics. This raises concerns about expressiveness and modeling capacity.

W3. The experiments use only four datasets. While Lorenz63 is a classic chaotic system, the others are standard but relatively clean and periodic. There is no evaluation on high-dimensional, irregularly sampled, or real-world noisy data (e.g., medical, financial, or sensor data with missing values). This limits the generalizability and practical relevance of the results. Moreover, the absence of comparison to strong baselines weakens the empirical contribution.

W4. Despite being a flow-based model, the paper does not provide any theoretical analysis of the model’s expressiveness (e.g., universal approximation, density coverage), convergence, or consistency. For instance, under what conditions does the learned flow converge to the true data distribution? How does the OU prior affect the posterior? The derivation focuses on computational tricks (e.g., Woodbury) but lacks deeper theoretical insights that would elevate the work beyond an engineering contribution.

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
