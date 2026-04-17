# Elucidating the design space of deep stochastic processes for image enhancement

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
In this work, we investigate deep stochastic processes for image enhancement. We show that existing approaches can be interpreted as instances of Ornstein–Uhlenbeck processes, diffusion bridges, or diffusion processes, each represented by a stochastic differential equation. As a result, we consolidate 11 methods into a unified mathematical framework and present them in a systematically structured table. This perspective separates the definition of processes from the schedulers and samplers that were originally used. Furthermore, we provide a modular library that implements the proposed methods and facilitates the integration of additional approaches with minimal coding effort. 
In order to perform comprehensive empirical evaluation among considered approaches, we evaluate them on four image enhancement tasks: super-resolution, colorization, low-light enhancement, and deraining with identical backbones and training protocol ensuring fair and meaningful comparison. The experiments highlight that, while most methods achieve similar results, there are exceptions that make some refinement strategies more effective than others, which we further analyze and explain.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on deep stochastic processes for image enhancement, including Ornstein-Uhlenbeck processes, diffusion bridges, and diffusion processes. This paper unifies them into a single framework and develops a PyTorch library for easy implementation. The authors claim that most methods perform similarly and plain diffusion often outperforms other complex variants.

### Strengths
1. The paper coherently categorizes 11 existing deep stochastic process methods for image enhancement into three classes and clarifies core design differences between approaches.
2. The authors developed a PyTorch library that supports easy implementation, lowering barriers for future research.
3. This paper identifies specific performance-limiting factors (e.g., low temperature harming OU process performance).

### Weaknesses
1. The paper reads more like a verification report of existing methods rather than an original research contribution. While it reproduces and compares several known techniques, it fails to clarify which specific theoretical differences among these methods lead to their varying performance. A deeper analysis of the underlying theoretical factors that account for the observed performance gaps would make the contribution more insightful and meaningful.
2. The work lacks guiding insights or implications for future research directions. There is little discussion on how the presented results could inspire further development or improvement of current methodologies.
3. The paper does not introduce any novel theoretical framework or superior implementation strategy that advances the SOTA.
4. The experimental findings, for example, the observation that a plain diffusion model outperforms more complex variants, are interesting, but they are not supported by sufficient theoretical analysis or explanation.

### Questions
1. Could the authors clarify the key theoretical differences among the compared methods that lead to their performance variations?
2. How might the current findings provide insights or guidance for future research directions?
3. Can the authors offer a theoretical explanation for why the plain diffusion model outperforms more complex variants? Does the amount of training data or training scale have an impact?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper unifies recent “stochastic process” approaches in image enhancement (diffusion, OU, bridge) under a single SDE framework, provides a unified table and three representative propositions, and includes FM, ResShift, and InDI as special cases within specific SDE formulations. It reports systematic comparisons across four tasks—super-resolution, upsampling, low-light enhancement, colorization, and deraining—concluding that conditional stochastic processes do not significantly outperform standard diffusion. The authors also release a modular codebase to facilitate implementation with different process paths. The goals and motivations are clear, and the work offers strong value in framework unification.

### Strengths
1. **Clear unified perspective and standardized presentation:** The paper systematically formulates these image enhancement methods in a general form, decoupled from the scheduler/sampler, facilitating fair comparison and reproducibility.
2. **Comprehensive empirical baselines:** It covers DM-VP, FM, IR-SDE, ResShift, InDI, (BBDM/DDBM/I2SB/GOUB/UniDB), unifies the number of steps, and reports PSNR/SSIM/LPIPS across multiple tasks. The finding that most methods perform similarly provides valuable insight for the community.
3. **Engineering contribution:** A unified, extensible PyTorch library is provided, supporting replication and modular experimentation.

### Weaknesses
**Experimental Setting:**

1. To ensure fairness, the authors unified network architectures, loss functions, parameterizations, and sampling formulas across methods—leading to the conclusion that performance differences are minor. However, this may actually introduce a different kind of unfairness. Each method may require distinct hyperparameters or architectures to reach its theoretical potential. Using one universal configuration or directly adopting settings from prior work could prevent methods from achieving their best possible performance. For example, in Table 2 (deraining) and Tables 7–8 (NFE=100), UniDB performs notably worse. Comparing methods’ theoretical merits should involve finding their optimal performance points rather than enforcing universal parameters, as even DDPM might fail to outperform an autoencoder under such constraints.
2. The paper lacks details on the parameterization strategies used in training. These methods typically fall into three categories: (1) parameterizing the inverse probability $p_\theta(x_0|x_t)$ (essentially $\epsilon_\theta$), (2) parameterizing the velocity field $v_\theta(x_t)$, and (3) SDE-based parameterization. The parameterization type strongly affects the sampling process. For instance, if SDE-based parameterization is used, Mean-ODE tends to perform well, while Euler-ODE does not.

### Questions
1. Regarding Weakness 1, it is recommended that the authors tune hyperparameters more thoroughly to find each method’s optimal configuration and update the main experimental tables accordingly. Although this might not be feasible during the rebuttal stage, such revisions should be considered for the camera-ready version or future submissions, along with a discussion of this issue in the experimental setup.
2. Regarding Weakness 2, could the inconsistent parameterization schemes have caused the anomalies observed in Table 9, where several results are abnormally low? If not, the authors should provide an explanation.

Finally, I believe that differences among these methods do exist. Even from Table 2, one can see that bridge-based models perform mediocrely in super-resolution—worse than DM-VP—but significantly outperform both diffusion and OU models in colorization. With more datasets and fairer comparisons, more universal findings are likely to emerge.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper integrates 11 methods of SDEs  in a systematic table and proposes an unified mathematical framework for definition to present the stochastic processes. They also provide a modular library (so far information about this library is only limited to description provided in this paper) that implements the methods and evaluate them on four image enhancement tasks: super-resolution, colorization, low-light enhancement to improve comparison.

### Strengths
This paper serves more like a review paper to integrate SDEs diffusion methods in a systematic table. They also provide a modular library. This seems like will bring out convivence for implementing related methods.
modular library

### Weaknesses
This paper serves more like a review paper. 
1., the introduction section ignored some important events when reviewing literatures. 
In P1. Line 50, 4th paragraph, they should have referred to papers on diffusion models.
2. I would suggest to add reference links to the table to make it easier for readers to access the references.

### Questions
1. I wonder what is the benefit of this unified framework. 
2. Did you choose specific parameters to represent in the table? It would be beneficial to explain how you get to a formula that is different from the original paper where there is a different. For example, did you set $\theta_t=0$ in UniDB?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the fragmentation in deep stochastic process-based image enhancement (IE) methods by proposing a unified mathematical framework to consolidate 11 state-of-the-art approaches. These methods are categorized into three classes of stochastic processes—standard diffusion models, Ornstein-Uhlenbeck processes, and diffusion bridges—each formalized via SDEs, transition kernels, and base distributions.

The authors separate method definitions from their original schedulers/samplers, enabling fair comparison. They implement a modular library (Ito Vision) to standardize implementations and conduct comprehensive experiments across four IE tasks: super-resolution (FFHQ), low-light enhancement (LOL), colorization (ImageNet), and deraining (Rain1400).

Key findings include: (1) Most methods achieve comparable performance, with plain diffusion models often outperforming specialized OU/bridge methods; (2) Low-temperature settings in OU processes (e.g., InDI) lead to collinearity bias, limiting detail recovery; (3) Ancestral sampling is the most consistent sampler, while deterministic samplers cause oversmoothing for diffusion bridges.

### Strengths
- Foundational Unification of Fragmented Methods: The paper fills a critical gap in IE research by formalizing 11 diverse methods into a single mathematical framework (Table 1).
- The experiments set a new standard for fairness in IE comparisons with identical backbones and various tasks. Ablations on samplers (Tables 9-10) and discretization strategies (Table 11) provide actionable insights.
- The paper goes beyond surface-level comparisons to explain why some methods underperform with analysis of temperature and collinearity.

### Weaknesses
- The framework is validated primarily on moderate-resolution inputs (e.g., 256×256 for super-resolution, 320×320 for low-light enhancement) but lacks testing on:
  - High-resolution IE: Professional scenarios (e.g., 1K/2K image restoration) require handling larger feature maps—It is unclear if the unified framework scales to these sizes (e.g., U-Net backbones may suffer from memory constraints).
  - Real-world noise artifacts: The paper assumes synthetic degradation (e.g., bicubic downsampling for super-resolution, synthetic rain for deraining) but ignores real sensor noise (Poisson shot noise, ISP artifacts) common in low-light photography.

- All experiments use U-Net backbones, while modern IE methods increasingly adopt Transformer/DiT architectures for better long-range feature modeling. It is unclear if the unified framework’s SDE/transition kernel definitions are compatible with Transformer-based score networks. The paper does not compare U-Net with Transformer backbones under the same stochastic process, missing an opportunity to validate backbone impact.

### Questions
- Have you tested the unified framework on ultra-high-resolution inputs (e.g., 1024×1024) or real-world noisy images? If so, how do process performance (e.g., LPIPS) and computational cost (e.g., memory) scale? If not, what modifications (e.g., hierarchical SDEs) would be needed to support these scenarios?
- Have you explored using DiT backbones with the unified framework? If so, did you need to adjust the transition kernel definitions to accommodate token-wise attention? How do Transformer-based methods compare to U-Net in terms of performance (e.g., FID) and efficiency?

### Soundness
3

### Presentation
2

### Contribution
3
