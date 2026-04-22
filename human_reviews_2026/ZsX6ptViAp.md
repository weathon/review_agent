# Control-Augmented Auto-Regressive Diffusion for Data Assimilation

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 0, 6

## Abstract
Despite recent advances in test-time scaling and finetuning of diffusion models, guidance in Auto-Regressive Diffusion Models (ARDMs) remains underexplored. We introduce an amortized framework that augments pretrained ARDMs with a lightweight \emph{controller} network, trained offline by previewing future ARDM rollouts and learning stepwise controls that anticipate upcoming observations under a terminal cost objective. We evaluate this framework in the context of data assimilation (DA) for chaotic spatiotemporal partial differential equations (PDEs), a setting where existing methods are often computationally prohibitive and prone to forecast drift under sparse observations. Our approach reduces DA inference to a single forward rollout with on-the-fly corrections, avoiding expensive adjoint computations and/or optimizations during inference. We demonstrate that our method consistently outperforms four state-of-the-art baselines in stability, accuracy, and physical fidelity across two canonical PDEs and six observation regimes. We will release code and checkpoints publicly.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes an application for data assimilation using a controlled diffusion model, but it is currently unclear what the motivation for doing so is. I am currently unsure of the differences between it and other guidance diffusion models. The author needs to have sufficient discussion on this. If my concerns are resolved, I will increase the rating.

### Strengths
1. Integrating random control mechanisms into the generation of ARDM to solve the prediction drift problem caused by observation sparsity in chaotic system data assimilation.
2. Balancing computational efficiency and physical fidelity, it not only avoids the high overhead of accompanying computation or integrated optimization, but also verifies the rationality of the results through physical diagnostic indicators (total variation, dissipation rate), which is in line with practical scientific application scenarios.

### Weaknesses
1. The author needs to explain the motivation behind using diffusion models for DA, which was not mentioned at all in the introduction.
2. How is the diffusion model mentioned by the author trained?
3. The current method section does not have an intuitive overview as it is difficult to understand. It seems that the proposed method is similar to classifier/classifier-free guidance, except that a model is added for control. Please discuss the difference between the two.
4. What is control in the paper, is it a neural network or is it equation 8?
5. Data assimilation was first proposed in weather forecasting, and the proposed method should be validated on this type of data, rather than just considering two unrealistic chaotic systems. There are many datasets such as ERA5 that can support model training.
6. Why are there no traditional DA methods, such as those based on Kalman filtering. These baselines are necessary to demonstrate the advantages of the proposed method.
7. What other advantages does the proposed method have besides being more accurate than baselines?
8. The proposed method needs a more detailed overview to reflect its process and advantages, for example in the caption of Figure 1.
9. Missing introduction to the limitation.
10. Can you add classifier-free guidance directly for generation? What are the issues with doing so? Why does the proposed method have advantages?
11. Are observations all regular? However, in reality, observations are often irregular. Can this be applied?

### Questions
See the weaknesses.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes CADA, a finetuning framework that augments a pretrained autoregressive diffusion model with a lightweight controller. The controller is trained offline, stepwise controls during each denoising sub-step, thereby amortizing the otherwise expensive inner optimization. As a result, data assimilation reduces to a single feed-forward, avoiding adjoint computations or per-step test-time optimization.
On two canonical chaotic PDEs and across six observation regimes, CADA outperforms four state-of-the-art baselines in stability and accuracy, and better adheres to standard physical diagnostics. Ablation studies further show that amortization is crucial for robust long-horizon performance.

### Strengths
- The paper proposes a lightweight controller–based finetuning scheme for ARDMs. To the best of my knowledge, it is novel and interesting.

- The framework is applied to data assimilation and demonstrates strong empirical performance.

- The paper provides a complete exposition of the model design rationale, training strategy, and operational workflow.

### Weaknesses
- The controller is trained on short preview windows and synthetic regimes; under longer horizons, different noise/observation operators, or higher-dimensional settings, the amortized policy might be off-distribution and drift. (No formal stability guarantees.)

- Baselines are mostly diffusion-based DA, missing classical 3DVAR/EnKF/4DVAR for broader context.

- Setups skew “clean” (canonical PDEs); more realistic NWP-style experiments would help.

- The method section is somewhat hard to follow, but the details become much clearer after reading Algorithms 1–3 in the appendix. I recommend moving these algorithms to the main text for better readability and flow.

### Questions
I think this paper is above the acceptance threshold. If the authors address the points below, I’d be inclined to raise my score to 8.

- The paper’s baselines are limited to diffusion-based DA methods; comparisons to classical DA (e.g., 3DVAR/4DVAR/EnKF) are missing. Please add at least a small-scale EnKF or 3DVAR baseline, or provide a principled justification for infeasibility under your setup (e.g., cost, operator mismatch, tuning burden). This would materially strengthen the empirical claims and better situate the work relative to established DA practice.

- Please elaborate on how the method scales to high-dimensional, real-world systems. What are the main computational or modeling challenges when moving from synthetic PDEs to large-scale NWP-like setups?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper proposes a control-augmented framework for Auto-Regressive Diffusion Models (ARDMs) applied to data assimilation in chaotic spatiotemporal PDEs. The approach trains a lightweight controller network offline to provide stepwise corrections during ARDM rollouts, anticipating future observations under a terminal cost objective. The method aims to avoid expensive adjoint computations during inference while improving stability and accuracy under sparse observations.

### Strengths
- Addresses an important problem: data assimilation for chaotic PDEs with sparse observations

- Reports improvements over four baselines across multiple PDEs and observation regimes

### Weaknesses
- Poor presentation and organization: The paper fails to clearly state the problem before diving into methodology, forcing readers to reconstruct the narrative themselves

- Unclear notation and equations: Key equations (7, 9) lack proper explanation. Variables like $y$ and $Φ$ are introduced without clear context or connection to the overall framework
Disconnected sections: Section 2.2 appears unmotivated and its relevance to the rest of the paper is unclear

- Insufficient training details: The training procedure is poorly explained, making reproducibility difficult even with promised code release

- Missing computational analysis: No comparison of computational costs versus baselines, despite the method appearing computationally intensive

### Questions
- What exactly is the problem formulation? Can you state it clearly upfront?

- Can you provide a clearer explanation of equation (9) and its role in the method?

- What are the computational costs (wall-clock time, memory) compared to the four baselines?

- How is the controller network trained in practice? What is the training pipeline? Please provide the code.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new framework called Control-Augmented Data Assimilation (CADA) that improves how diffusion models incorporate observational information when making predictions for chaotic systems like weather or fluid dynamics. It introduces a lightweight controller network trained offline to anticipate future observations and inject small “control” corrections into each diffusion step. This allows data assimilation to occur in a single forward rollout, avoiding expensive optimisation or adjoint computations used by existing methods. Experiments on the Kuramoto–Sivashinsky and Kolmogorov flow PDEs show that CADA yields more accurate, stable, and physically consistent long-horizon forecasts than state-of-the-art diffusion-based baselines. The approach provides a general recipe for embedding control into generative dynamics for broader sequential inverse problems such as weather and climate modelling

### Strengths
The general problem the paper addresses -- data assimilation in PDEs and weather models -- is an important one. The paper builds on a strong line of work using neural surrogates and diffusion for data assimilation in PDEs is very well presented. The writing is clear. Figures are excellent. The messages are clean. The contribution is novel as far as I'm aware and neat. ICLR is an appropriate venue for the work.

### Weaknesses
I felt the paper did a very good job at explaining the high-level narrative. I felt that it was less strong in explaining the more fine-grained technical details. This is obviously challenging in a short paper, but I was left with some quite significant questions that 

-- Clarity of experimental setup: A schematic is needed to clarify what observations are available, when, and how evaluations are performed; the practical relevance to real weather forecasting (with continuous, dense data) is uncertain.

-- Applicability and motivation: The chosen case should be linked to a concrete application domain, possibly exploring varied or mixed spatio-temporal observation densities.

-- Computational cost: No discussion or comparison of training and inference costs is provided; scaling, efficiency, and fairness of baseline comparisons should be addressed.

-- Controller network limitations: Training a separate controller for each observation regime limits flexibility; a discussion of trade-offs, data requirements, and generalization to new regimes is needed.

-- Methodological clarity (Eq. 8–10): The theoretical connection between the tilted objective and the amortised control formulation is unclear; there’s concern about loss of KL regularization and guarantees that the new policy stays close to the prior.

-- Baseline tuning and fairness: Details on hyperparameter tuning, especially guidance strength in Shysheya et al., are missing; the surprising baseline performance raises concerns about whether comparisons are fair

### Questions
A schematic explaining the experimental setup would have helped me understand the protocol — what observations are accessed when by what methods and at what lead times are evaluations performed against GT. Linked to this, it’s not clear to me that the studied case is of direct relevance to weather forecasting where measurements are usually continuous in time (and often fairly dense) at least for medium range forecasting in the Global North.  It might be interesting to consider different spatio-temporal densities of observational data if you want to make the case for different application domains. For example, a mix of high and low-density observation regions, or random space-time masking.  Do the authors have a specific application in mind which motivates the current setup?   

A comparison / discussion of computational cost is missing. This seems important e.g. is the new method very expensive compared to the others at training / test time? Could the baseline methods be improved if the computational cost of training / inference was matched? I presume that this isn’t the case but some discussion of scaling and numbers would be useful to make the argument tight. 

Instead of a pure inference time technique (like reconstruction guidance), the authors propose using the outputs of a controller network to guide the sampling. This requires training a separate controller network for each new observation regime. This is quite a big limitation versus reconstruction guidance. There should probably be a discussion about the trade-off between extra training/inference cost, as well as how this cannot be applied out-of-the-box for new observation regimes. This could potentially be complemented by a discussion about the data requirements needed for training the controller network—if I start observing a new regime, how much data will I have to collect until I can train a controller network and then use that to perform inference?

I have a question around equation 8. My understanding here is that the paper starts by motivating the tilted approach (Eq. 7), which leads to the cost in Eq. 8 which contains beta. However, as the authors note, direct tilting is intractable, so a different direction is taken where they inject the amortised controls into the pretrained policy (i.e. modify the sampling process by injecting the u information). To train the controls, they use the cost in Eq. 10 where they set beta to 0 (but I am not even sure that comes from the same considerations as Eq. 8). Regarding this, I would be curious if there are any guarantees that the resulting policy is still “close enough” to the prior one? In Eq. 8 this is controlled by the KL divergence, but in their case can the additional amortised information actually modify the policy quite a lot since there is no constraint on the KL?

How were the hyperparameters for the baselines fine-tuned? In particular, the extent to which the method in Shysheya et al is able to leverage the observations is known to depend highly on the guidance strength which needs to be tuned. Have the authors experimented with that to see if the baseline comparison is fair? Moreover, what is the conditioning scenario they used? Based on my experience, I’m a bit surprised about the low quality of the results of Shysheya et al. Moreover, in the related work line 307-309 it says “Autoregressive methods (Shysheya et al., 2024; Gao et al.) and DiffDA (Huang et al., 2024) improve stability but remain computationally expensive due to inference-time optimization.” Is this really the case for Shysheya et al?

Small detail: I felt that the presentation of the anchored windows, prospective guidance section and the additional explanations in Appendix C is quite convoluted and could probably be presented in a cleaner manner.

Small detail: I am a bit confused about why in Alg. 1 line 3 the controller network does not ingest z_{t+1}^{(s)} too, but it might be a typo because I think it should.

Small detail: The authors mention that “to strengthen the assimilation signal, we evaluate the arrival cost not only at observation indices but also at their intermediate denoising sub-steps, using Tweedie estimates of the forecast state at each sub-step.” I am confused about where that is used because Alg 1. shows that the arrival cost is only output at the end of the S DDIM steps. Does this mean that when training the controller, the authors sometimes output x_{t+1} based on Tweedies’ estimates rather than going through all the S DDIM steps and use that as the output of CONTROLLEDSTEP in Alg. 2?

### Soundness
3

### Presentation
4

### Contribution
3
