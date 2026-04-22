# GAGA: Gaussianity-Aware Gaussian Approximation for Efficient 3D Molecular Generation

- Avg Score: 2.67
- Decision: Accept (Poster)
- Scores: 2, 2, 4

## Abstract
Gaussian Probability Path based Generative Models (GPPGMs) generate data by reversing a stochastic process that progressively corrupts samples with Gaussian noise. Despite state-of-the-art results in 3D molecular generation, their deployment is hindered by the high cost of long generative trajectories, often requiring hundreds to thousands of steps during training and sampling. In this work, we propose a principled method, named GAGA, to improve generation efficiency without sacrificing training granularity or inference fidelity of GPPGMs. Our key insight is that different data modalities obtain sufficient Gaussianity at markedly different steps during the forward process. Based on this observation, we analytically identify a characteristic step at which molecular data attains sufficient Gaussianity, after which the trajectory can be replaced by a closed-form Gaussian approximation. Unlike existing accelerators that coarsen or reformulate trajectories, our approach preserves full-resolution learning dynamics while avoiding redundant transport through truncated distributional states. Experiments on 3D molecular generation benchmarks demonstrate that our GAGA achieves substantial improvement on both generation quality and computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to improve the training and sampling efficiency of diffusion models by truncating the diffusion noise-adding process. For zero-mean data, the authors design an empirical method to select diffusion timesteps, which is validated on molecular generation tasks. Experimental results show performance improvements over methods like EDM and GeoLDM.

### Strengths
1. The proposed method is simple and serves as a "free lunch": it eliminates redundant training and sampling overhead in existing models, boosting both efficiency and generation quality.

2. The writing is clear and easy to follow

### Weaknesses
1. The core idea of truncating the noise-adding process is overly straightforward. Similar concepts have already been explored in iDDPM, where the redundancy of original diffusion schedules was identified and new schedules were designed. This method essentially truncates the existing schedule, which introduces two critical issues:
    - Theoretical inconsistency: Truncation increases the approximation error of the Gaussian prior. The distance between $x_{T^*}$ and the standard Gaussian is larger than that between original $x_T$ and the standard Gaussian.
    - Practical inconvenience: A new hyperparameter $T^*$ (or two hyperparameters $\epsilon_{dep}$, $\epsilon_{DS}$) is introduced , which needs to tune through experiments. 

Instead of truncation, a more viable approach could be adjusting the SNR decay rate of the schedule to slow down decay, thereby compressing the time interval during which the distribution approximates a Gaussian.

2. Limited significance of the method: Performance improvements are more pronounced for models with DDPM schedule (e.g., EDM, GeoLDM). Intuitively, the DDPM schedule has a fast SNR decay rate, leading to many high-noise timesteps—hence the truncation method yields more obvious gains. However, for methods that already address schedule redundancy (e.g., GeoBFN, SLDM), the efficiency improvements from this approach are expected to be far more limited.

3. Gaps in research logic and writing: 
    - The paper introduces three metrics to measure the discrepancy between the current distribution and a Gaussian distribution: cumulant tensors (Eq. 8), mutual information (Eq. 12), and cumulative distribution functions (Eq. 14). The theoretical analysis relies on the first metric, while the second and third are used in experiments—creating a logical gap. Is there a need to include so many similar metrics? Would retaining the least computationally expensive one suffice?
    - Eq. 26 lacks implementation details for \text{MI}_{\text{rows}}: Does it compute the mutual information between all pairs of rows and then average the results? Eq. 12: Does it use row-wise or column-wise mutual information? What is the physical meaning of rows and columns? For the matrix representation of $x_t$: If $x_t$ represents coordinates, is the matrix of size $N \times 3$ (where N is the number of atoms) or $(3N) \times d_{\text{feat}}$ (where $d_{\text{feat}}$ is the feature dimension)? If the latter, does the independence measured by mutual information depend on the network used to extract features?
    - Unclear visualization in Figure 3: The "Gaussianity" in Figure 3 is represented by colors; numerical values would be more concrete. 

Ref:
Improved Denoising Diffusion Probabilistic Models.

Straight-Line Diffusion Model for Efficient 3D Molecular Generation.

Unified Generative Modeling of 3D Molecules via Bayesian Flow Networks.

### Questions
1. Is the zero-mean assumption a major limitation of your method? Could normalization be applied to handle data with non-zero means, regardless of the original data distribution?
2. What would be the performance if timestep truncation is applied only during sampling (not training)? Would this harm generation quality?
3. A curious question: for element types alone, is their distribution closer to a Gaussian than that of images?

### Soundness
3

### Presentation
2

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
This paper proposes an algorithm to calibrate the noise schedule for diffusion model training and sampling. The central idea is to identify a minimal log signal-to-noise ratio (log-SNR), $\lambda_\min$ (in the sense of [1]), at which the noised latent variable attains sufficient Gaussianity. The schedule is then truncated at this point to avoid unnecessary and potentially harmful levels of noising. The effectiveness of this calibration method is demonstrated on several diffusion models for molecule generation.

[1] Kingma, Diederik, and Ruiqi Gao. "Understanding diffusion objectives as the elbo with simple data augmentation." Advances in Neural Information Processing Systems 36 (2023): 65484-65516.

### Strengths
- Clear Motivation: The paper is well-motivated, addressing the common and practical problem of inefficiently wide noise schedules in diffusion models.
- Strong Empirical Support: A key strength is the empirical demonstration that baseline models often operate on an unnecessarily broad SNR range. The results convincingly support the claim that truncating this range to an "effective" one can be done without degrading model performance.
- Practical Significance: The proposed method offers a practical tool for scheduler calibration. A significant benefit is the potential to accelerate inference by reducing the number of sampling steps, even when applied post-hoc to models trained with a standard, non-calibrated scheduler.

### Weaknesses
- Insufficient Positioning: A significant weakness is the lack of thorough positioning against the extensive existing literature on noise schedulers, SNR analysis, and related diffusion model theory (e.g., [1,2,3]). This omission causes the work to feel disconnected from established formalism in the area. Consequently, the development of the method appears somewhat ad-hoc rather than being rigorously derived from first principles.
- Clarity and Readability: The paper's clarity could be improved. A considerable portion of the text is dedicated to standard, boilerplate descriptions of diffusion models, which detracts from the space available to elaborate on the novel contributions.
- Undefined Key Terminology: A central term, "data identity," is used repeatedly throughout the manuscript but is never formally defined. This ambiguity leaves a core concept of the paper open to interpretation and hinders the reader's ability to fully grasp the method.
- Unconventional Presentation: In a minor editorial point, the results tables use boldface to highlight the proposed method's results rather than the conventional practice of bolding the best-performing method for a given metric. This unconventional choice makes direct performance comparisons less intuitive.

[1] Kingma, Diederik, and Ruiqi Gao. "Understanding diffusion objectives as the elbo with simple data augmentation." Advances in Neural Information Processing Systems 36 (2023): 65484-65516.

[2] Falck, Fabian, et al. "A Fourier Space Perspective on Diffusion Models." arXiv preprint arXiv:2505.11278 (2025).

[3] Gao, Ruiqi, et al. "Diffusion models and gaussian flow matching: Two sides of the same coin." The Fourth Blogpost Track at ICLR 2025. 2025.

### Questions
1. Could the authors please provide a formal, mathematical definition for the term "data identity" as it is used in the context of this paper?
2. The paper mentions benefits for training. Could the authors clarify precisely how the proposed calibration method reduces training time?

### Soundness
3

### Presentation
1

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
This paper proposes an interesting approach for accelerating the training and sampling procedures of 3D molecular diffusion and flow-based models using Gaussian Approximation. The key idea is that molecular data becomes "sufficiently Gaussian" early on in the forward noising process; therefore, the trajectory can be truncated and replaced by closed-form Gaussian distributions. Experiments yield promising results in terms of improved sampling and training efficiency, as well as enhanced generation quality, on two 3D molecular datasets.

### Strengths
* The idea of assessing the Gaussianity of the data during the noising process to accelerate the training and sampling is novel and sound. Additionally, leveraging this specifically for molecular data is interesting (although some claims in this regard require further clarification, as noted in the weaknesses).
* The paper is well-written and clearly explained.
* The experiments covering two common datasets and a few baselines are rather extensive and show promising results.

### Weaknesses
* The main assumption in this paper is that molecular data converges to Gaussianity faster than other modalities, e.g., images. However, this assumption is neither theoretically justified nor empirically verified. Proposition 3.1 requires that the initial distribution is more Gaussian; however, can this be demonstrated? Additionally, based on Figure 2, the authors claim that image data retains recognizable features for more steps than molecular data; however, this can be misleading, as images are much more familiar to the human eye than molecules. Quantitative or qualitative results are needed to verify this claim. For instance, it would be interesting to plot the proposed dependency evaluator $Dep(x_t)$ and the distributinal similarity $D_t$ over different timesteps during the noising process for an image dataset and a molecular dataset (of course, they should be adequately scaled to accommodate the different data scales and ranges).
* While the paper does a good job at explaining how $T^* $ is obtained, it is missing some details on how the training and sampling procedure is adapted. It seems that the segment $[T^*, T]$ is simply truncated; however, the paper should formally clarify this. For example, what are the exact modifications to the training and sampling algorithms from EDM [1]?
* All reported experimental results lack standard deviations across different runs, which makes it hard to assess the statistical significance of the claimed results. Also, the code is not provided.
* It would be a great addition to the paper if the proposed Gaussianity test is used in other settings beyond accelerating training and sampling procedures. In some cases, efficiency is not the primary concern, and the user may be willing to sacrifice efficiency for improved generation quality. I believe the proposed approach can be leveraged in this setting. For example, given a pre-specified noise schedule, find the total time steps $T$ such that the optimal $T^* =1000$ or use the proposed Gaussianity test to guide the design of noising schedules such that $T^*=T$. This setting would enable comparison with baselines for the same computational budget, potentially yielding improved generation results because of better allocation of time steps.

I would be willing to increase my score if the authors address these points convincingly.

**References**

[1] Hoogeboom, Emiel, et al. "Equivariant diffusion for molecule generation in 3d." International conference on machine learning. PMLR, 2022.

### Questions
1. Is the term "Gaussian Probability Path based Generative Models (GPPGMs)" known in the literature? If yes, please cite the corresponding papers. If not, please somehow clarify this or use more well-known terms.
2. Figure 1 is not clear. What is "complex", and what is the difference between the two backward passes? Please clarify the figure and the caption.
3. In Equation 2, $\alpha_{t-\Delta t|t}$ is not defined.
4. In Equation 7, why is $\tilde v_t$ chosen in this way? Please clearly explain this. 
5. In Equation 10, shouldn't $m>=3$?
6. In line 225, it is stated that "sparse molecular coordinates around equilibrium are closer to Gaussian". Please elaborate and give concrete qualitative or quantitative arguments.
7. When estimating all the quantities required for computing $T^*$, e.g., the variance of the Gaussian, the dependency evaluator, and the distributional similarity, do you sample multiple noise vectors for each data sample and each noise level?
8. What are the values of $\epsilon_{dep}$ and $\epsilon_{DS}$ and how are they chosen?

### Soundness
2

### Presentation
3

### Contribution
3
