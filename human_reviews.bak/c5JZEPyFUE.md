# Dynamical Diffusion: Learning Temporal Dynamics with Diffusion Models

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Diffusion models have emerged as powerful generative frameworks by progressively adding noise to data through a forward process and then reversing this process to generate realistic samples. While these models have achieved strong performance across various tasks and modalities, their application to temporal predictive learning remains underexplored. Existing approaches treat predictive learning as a conditional generation problem, but often fail to fully exploit the temporal dynamics inherent in the data, leading to challenges in generating temporally coherent sequences. To address this, we introduce Dynamical Diffusion (DyDiff), a theoretically sound framework that incorporates temporally aware forward and reverse processes. Dynamical Diffusion explicitly models temporal transitions at each diffusion step, establishing dependencies on preceding states to better capture temporal dynamics. Through the reparameterization trick, Dynamical Diffusion achieves efficient training and inference similar to any standard diffusion model. Extensive experiments across scientific spatiotemporal forecasting, video prediction, and time series forecasting demonstrate that Dynamical Diffusion consistently improves performance in temporal predictive tasks, filling a crucial gap in existing methodologies. Code is available at this repository: https://github.com/thuml/dynamical-diffusion.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a new diffusion model for time series data incoporating the temporal dependencies between states in the forward and backward processes. Extensive empirical evidence shows that the model indeed outperforms regular denoising diffusion most notably on the SEVIR and Turbulence Flow datasets.

### Strengths
- **A clear problem and a sound solution**: The paper identifies a significant gap in how current diffusion models handle temporal data - they treat predictive learning primarily as conditional generation without fully leveraging temporal dependencies in the data. The authors address this limitation by modelling explicitly temporal transitions at each diffusion step (in both the forward and backward processes) through a mixture process.
- **Thorough evaluation**: The authors conduct a strong empirical evaluation using datasets from various domains: scientific spatiotemporal forecasting, video prediction, and time series forecasting. The experiments also include ablations and analyses of key components like dependent noise and the $η$ parameter. In addition to the reported metrics, qualitative visualizations illustrate the strengths of the model espcially compared to regular diffusion.
- **Good presentation**: The paper is well-written, with a logical flow from motivation to results.

### Weaknesses
- **Why diffusion models?**: I understand that the main contribution of the paper is extending the capability of diffusion models, and as such it makes sense to focus the evaluation of DyDiff vs DPM. However, I think a general motivation for diffusion/generative models for dynamical data is needed (see questions). Additionally, the authors could discuss how their approach in general relates to previous methods (see questions).

- **Derivations for the reverse process and conditional reverse process are incomplete**: I have a hard time following the derivations in appendix A2 for $q(x_{t-1}^1|x^1_t)$ and $q(x^1_{t-1}|x^1_t, x_0^{-P:1})$, which I find to be too brief. Can the authors explain how $q(x_{t-1}|x_t, x_0^{-P:1})$ is obtained from the forward process?
 
- **Dataset descriptions lack context**: The Turbulence Flow dataset is introduced without context about what it represents (including what each frame represents physically) or why it's relevant for evaluation. For SEVIR, while mentioned as a "spatiotemporal Earth observation dataset", the significance of Vertically Integrated Liquid (VIL) prediction is not explained. The same goes for the RoboNET, BAIR, and time-series datasets. A few sentences introducing the datasets used are needed in each experimental subsection. 

- **The metrics used are not defined**: Critical evaluation metrics (CRPS, CSI, FVD, PSNR, SSIM, LPIPS) are used without proper definition or motivation. This makes interpreting the results difficult. A formal (brief, if needed) definition of these metrics is crucial for the assessment of the empirical evaluation of the paper. 

- **Important baselines are missing**: The evaluation is missing important baselines like DYffusion [1], which is mentioned when motivating the present work, but not compared to it empirically. Other more recent relevant work is Rolling Diffusion [2], which the authors may choose to not discuss given how recently it was published.

- **The code is not provided as part of the submission**: while this is optional, I believe it is a useful practice to ensure reproducibility. 

**nitpicks**
- could you bold the best numbers in table 8 appendix D2?
- 'Dyffusion' is misspelled in page 2, last paragraph

[1] "DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting", Salva Rühling Cachay, Bo Zhao, Hailey Joren, Rose Yu, NeurIPS 2023

[2] "Rolling Diffusion Models", David Ruhe, Jonathan Heek, Tim Salimans, Emiel Hoogeboom, ICML 2024

### Questions
- What are the advantages of using diffusion-like models for dynamical data compared to previous predictive approaches?
- Are there any features of diffusion models that previous methods cannot achieve? can you demonstrare these empirically?
- Can you extend your derivations to show how $q(x_{t-1}|x_0^{-P:s})$ is obtained from the forward process for both the ddpm and ddim sampling methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors address temporal learning dynamics - an underexplored area in diffusion models. They introduce Dynamical Diffusion, a simple (essentially one term) modification of existing diffusion dynamics to directly incorporate this temporal dependency in both the forward and backward processes. They show that tracking the temporal similarities results in higher quality samples across multiple modalities.

### Strengths
I think this is a good paper. In my view, the motivation is more akin to how momentum methods developed in optimization - in both the cases, tracking the temporal dynamics was important. The idea, in that regard, isn't too novel, but I am glad to see the large number of experiments the authors have performed. I think this will be a paper whose method will be used as benchmark by future papers.

### Weaknesses
N/A

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Dynamical Diffusion (DyDiff) is a novel framework that enhances diffusion models for temporal predictive learning tasks. It introduces temporally aware forward and reverse processes that explicitly model temporal transitions at each diffusion step, addressing the challenge of integrating temporal dynamics into diffusion processes. DyDiff achieves this by incorporating a mixture of historical states controlled by a new schedule parameter γ, allowing it to capture temporal dependencies more effectively. The framework is theoretically sound and can be efficiently trained and sampled similar to standard diffusion models. Experiments across scientific spatiotemporal forecasting, video prediction, and time series forecasting demonstrate that DyDiff consistently outperforms standard diffusion models, particularly in generating temporally coherent sequences and improving performance over longer time horizons.

### Strengths
The paper introduces a novel framework called Dynamical Diffusion (DyDiff) that incorporates temporal dynamics into diffusion models for predictive learning tasks. This represents an original approach to addressing limitations of existing diffusion-based methods for temporal data. The authors creatively combine ideas from standard diffusion models with explicit modeling of temporal transitions.

The paper is generally well-written and structured logically. Key concepts and the proposed method are explained clearly with helpful illustrations. The authors provide pseudocode for the algorithms, which aids in understanding the implementation. Some technical details are relegated to appendices to maintain flow in the main text.

### Weaknesses
1. The paper does not adequately address why temporal dynamics need to be explicitly modeled, given that neural network architectures like transformers can inherently learn such relationships. Specifically:
	-	Lack of justification: The authors do not provide a clear explanation or empirical evidence for why explicitly modeling temporal dynamics is necessary, given that modern architectures like transformers are theoretically capable of learning temporal relationships on their own.
	-	Limited comparison: There is no direct comparison between the proposed Dynamical Diffusion method and transformer-based approaches that implicitly learn temporal dynamics. This makes it difficult to assess the true value added by the explicit modeling of temporal relationships.
	-	Insufficient analysis: The paper does not thoroughly analyze the limitations of existing architectures in capturing temporal dynamics, which would help justify the need for the proposed approach.
	-	Overlooked alternatives: The authors do not discuss potential alternatives for improving temporal modeling within existing frameworks, such as modifications to transformer architectures or attention mechanisms.
	-	Unclear efficiency trade-offs: The paper does not address whether the explicit modeling of temporal dynamics introduces additional computational overhead compared to letting a neural network learn these relationships implicitly.

2. The paper primarily compares Dynamical Diffusion with standard diffusion models. To better establish its significance, the authors should:
- Include comparisons with other state-of-the-art methods in temporal predictive learning, not just diffusion-based approaches.
- Provide a more comprehensive literature review to contextualize their work within the broader field of temporal predictive learning.

3. The paper could benefit from more extensive ablation studies to understand the contribution of different components of the proposed method. Additionally, a thorough analysis of hyperparameter sensitivity would strengthen the work. The authors could:
- Conduct ablation studies on the key components of Dynamical Diffusion.
- Analyze the sensitivity of the method to different hyperparameters, especially the newly introduced γ̄_t schedule.
- Provide guidelines for selecting optimal hyperparameters for different types of temporal data.

### Questions
See weakness

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
3

### Summary
The authors proposed a new method to approach the problem of predictive learning in the context of diffusion models. Instead of utilizing conditional generation, in which the generation is conditioned on the history, their method devises a forward process that gradually incorporates the history while simultaneously adds noise to the data. In this way, the process mixes the "diffusion time" and the "data time". Training can be done using the usual denoising score matching method.

### Strengths
Mixing temporal transitions with diffusion process is a novel idea. The experiments seem extensively done and the proposed method does deliver better performance when compared to DPM in many cases, though not as powerful as heavy models such as iVideoGPT

### Weaknesses
1. The writing can at various places uses some improvements. For instance, section 1, from the third paragraph onwards, appear to have large overlap with the abstract. In section 5, why the papers mentioned as "related work" described in the 1st paragraph are related seem to be obscure.  
2. The paper does not contain much insight about why iterative scheme like the one described in (4)-(5) is a helpful one, or more precisely, why it is a good structure to impose. Even adding some simple explanations on the basic features of the scheme, such as " larger values of $t$ correspond to a stronger emphasis on historical states", will help bringing some insights.

### Questions
1. The author(s) stressed that the method is "theoretically sound" (repeated twice) and "theoretically guaranteed". It'd be good if the author(s) can clarify a. in which sense is it theoretically sound? b. What precisely is "guaranteed" theoretically? 
2. I'm confused by the sentence "Notably, the noise factor $\sqrt{1-\alpha_t}$ remains consistent with theoriginal diffusion process along the denoising axis. Hence, the new forward process preserves the same signal-to-noise ratio for any diffusion step $t$". Surely the authors do not mean $\sqrt{1-\alpha_t}$ is independent of $t$, as opposed to what the notation suggests?

### Soundness
3

### Presentation
2

### Contribution
3
