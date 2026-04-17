# SCOPED: Score–Curvature Out-of-distribution Proximity Evaluator for Diffusion

- Decision: Accept (Poster)
- Scores: 2, 8, 6, 6

## Abstract
Out-of-distribution (OOD) detection is essential for reliable deployment of machine learning systems in vision, robotics, and reinforcement learning. We introduce Score–Curvature Out-of-distribution Proximity Evaluator for Diffusion (SCOPED), a fast and general-purpose OOD detection method for diffusion models that reduces the number of forward passes on the trained model by an order of magnitude compared to prior methods, outperforming most diffusion-based baselines and approaching the accuracy of the strongest ones. SCOPED is computed from a single diffusion model trained once on a diverse dataset and combines the Jacobian trace and squared norm of the model’s score function into a single test statistic. Rather than thresholding on a fixed value, we estimate the in-distribution density of SCOPED scores using kernel density estimation, enabling a flexible, unsupervised test that, in the simplest case, only requires a single forward pass and one Jacobian–vector product (JVP), made efficient by Hutchinson’s trace estimator. On four vision benchmarks, SCOPED achieves competitive or state-of-the-art precision-recall scores despite its low computational cost. The same method generalizes to robotic control tasks with shared state and action spaces, identifying distribution shifts across reward functions and training regimes. These results position SCOPED as a practical foundation for fast and reliable OOD detection in real-world domains, including perceptual artifacts in vision, outlier detection in autoregressive models, exploration in reinforcement learning, and dataset curation for unsupervised training.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an efficient out-of-distribution detection method utilizing the ratio between the score and the curvature. The key idea of the model is based on the intuition of isotropic Gaussian approximation, where the output is close to 1 in the typical set. The proposed method, SCOPED, is evaluated in reinforcement learning data and a standard OOD detection framework.

### Strengths
1. It is not hard to understand the key idea of the method. The manuscript is well-written.

2. I believe the method further extends by using the curvature information of the diffusion model. 

3. Reinforcement learning benchmark is interesting, although I do not think this is a standard evaluation framework like CIFAR-10 (ID) against CIFAR-100 (OOD).

### Weaknesses
1. My biggest concern about the paper is the structural similarity of the proposed method against DiffPath [1]. DiffPath shows significance against existing methods since 1) it does not require a diffusion model suited for ID data, 2) it has a small computation cost. This paper follows a similar framework, with only marginal improvements. This questions the significance of the approach.

2. Furthermore, the empirical results of the proposed method on single and double step lags behind DiffPath. If the variant of the method can utilize multiple time steps (e.g., <5 since computing J is slower than computing F) can outperform DiffPath, it can have merits. Unless I can find a good reason to apply the proposed method.

3. The evaluation benchmark seems largely borrowed from [1], but evaluation on the texture dataset is missing. It would be interesting to evaluate the OOD texture data.

4. While it is relatively minor, it would be better to note the actual runtime of the SCOPED compared against DiffPath.

5. Furthermore, I believe Table 1 can be improved. "Diffusion-based" and "Curvature and Diffusion-based" should be revised to represent the "ID-dependent method" and "ID-agnostic method", respectively, for readability.


[1] Out-of-Distribution Detection with a Single Unconditional Diffusion Model, NeurIPS 2024.

### Questions
See Weakness

### Soundness
2

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
4

### Summary
This paper introduces SCOPED (Score-Curvature Out-of-distribution Proximity Evaluator for Diffusion), a novel and highly efficient method for Out-of-Distribution (OOD) detection using pre-trained diffusion models.

The primary problem this work addresses is the high computational cost of existing diffusion-based OOD methods, which often require numerous sequential model evaluations to integrate a full denoising trajectory.

The core contribution is a new, computationally cheap OOD test statistic: the "score-curvature ratio" ($T(x)$). This statistic is motivated by information geometry and compares the squared norm of the model's score function ($||s(x)||^2$) to the local curvature of the log-probability density (the trace of the score's Jacobian, $Tr(\nabla_x s(x))$). The intuition is that for in-distribution (ID) samples, these quantities are closely related, while OOD samples will show a significant discrepancy.

Key features of the method are:

1. **High Efficiency:** Instead of a full path, SCOPED only requires evaluation at one or two _independent_ noise levels (e.g., an "early" and "mid-level" step).
    
2. **Fast Curvature:** The Jacobian trace is estimated efficiently using Hutchinson's trace estimator, which only requires a single Jacobian-vector product (JVP) per evaluation, making its cost (approx. $1F+1J$) comparable to a forward pass.
    
3. **Unsupervised Calibration:** The method is calibrated by fitting a Kernel Density Estimator (KDE) to the $T(x)$ scores from the ID dataset. The final anomaly score for a new sample is its negative log-likelihood (NLL) under this KDE.
    

The authors evaluate SCOPED on standard vision (CIFAR, SVHN, CelebA) and reinforcement learning (DMC, D4RL) benchmarks. The results demonstrate that SCOPED achieves competitive or state-of-the-art (SOTA) AUROC scores while reducing the number of model evaluations by an order of magnitude compared to prior diffusion-based OOD detectors.

### Strengths
-  **High Computational Efficiency:** This is its primary advantage. It reduces the number of model evaluations by an order of magnitude (e.g., from ~10-1000 for path-based methods to just $1-2$) by avoiding sequential denoising paths.

-  **Strong Theoretical Grounding:** The method is not an arbitrary heuristic. It's based on a well-motivated principle from information geometry—the "score-curvature ratio"—which links the model's score to the local curvature of the data distribution.

-  **Excellent Performance:** Despite its low computational cost, SCOPED achieves competitive or state-of-the-art OOD detection accuracy (AUROC) across multiple benchmarks, demonstrating that it does not sacrifice performance for speed.

-  **Generality Across Domains:** The paper successfully validates the *same* method on two very different domains: standard computer vision (CIFAR, SVHN) and high-dimensional robotic control/reinforcement learning (DMC, D4RL).

-  **Practical Unsupervised Design:** The entire system is calibrated *only* using in-distribution (ID) data. The use of Kernel Density Estimation (KDE) provides a flexible, non-parametric way to set the OOD threshold without needing any OOD examples.

- **Parallelizable:** Unlike path-based methods where each step depends on the last, SCOPED's evaluations (e.g., at $t=1$ and $t=300$) are independent and can be run in parallel, further increasing its real-world speed.

### Weaknesses
- Agnostic to Downstream Model Uncertainty: The most significant conceptual limitation is that the OOD score is completely decoupled from the downstream task model (e.g., the classifier). SCOPED is a distributional OOD detector, not a model-centric one.
It only answers: "Does this input look like the in-distribution dataset?" It cannot answer: "Is this input hard for the main classifier?"
Consequently, this method cannot detect in-distribution samples with high epistemic uncertainty i.e., "hard" or atypical ID samples that the main classifier is likely to misclassify. This is a key capability of other OOD methods, often measured by AUPR on correct vs. incorrect ID predictions, which is not addressed here.

- Dependence on Foundation Model & Calibration Data: The method's effectiveness relies on two separate components: a pre-trained "foundation" diffusion model and the ID data used for KDE calibration.

- The quality and domain of the diffusion model cap the performance. It is unclear how SCOPED would perform if the diffusion model's training data (e.g., CelebA) is too distant from the main task's ID data (e.g., medical X-rays), as the "measuring tool" itself may lack sensitivity in the correct feature space.

- The paper does not discuss the sensitivity to the KDE hyperparameters which is a critical free parameter that could significantly impact the robustness of the resulting OOD score.

- The choice of timesteps ($t=1$ and $t=300$) is presented as a robust combination of "fine detail" and "coarse structure" based on an SNR curve, but it remains an empirically-chosen heuristic.

### Questions
- Why you didn't evaluate scoped using openood benchmark so we can see its effect on Near and Far-OOD per each ID dataset?

- How did you choose the timesteps t=1 and t=300 and do we need to tune them as hyperparameters ?

- The KDE calibration step is critical, but KDEs can be sensitive to hyperparameter choices like bandwidth. Could you provide a sensitivity analysis for the KDE bandwidth? How was this parameter selected for the paper's experiments, and how much does performance degrade if it's not optimal?

- The vision experiments all rely on a single DDPM trained on CelebA. How robust is SCOPED's performance to this choice? For example, how would it perform if the foundation model's domain were very distant from the ID task's domain 

- The choice of $t=1$ and $t=300$ seems to work well as a heuristic. How would you recommend a practitioner select these timesteps for a completely new dataset and a different diffusion model architecture?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces **SCOPED**, an OOD scoring method for diffusion models that leverages a **score–curvature ratio** \(T(x)=\|s_\theta(x,t)\|^2 / \mathrm{Tr}(\nabla_x s_\theta(x,t))\) at one or two noise levels. The score comes from the trained diffusion model at a chosen timestep, while the curvature (trace of the score Jacobian) is efficiently estimated via a single **Hutchinson JVP**. To normalize across datasets/models/timesteps, the method fits a **KDE** on in-distribution \(T(x)\) values and uses \(-\log\) density as the anomaly score. SCOPED aims to capture **typicality** without running full diffusion trajectories, yielding strong OOD separation on both **vision** and **RL** benchmarks with **very low NFE** and full **batch parallelism**.

### Strengths
- **Simple, model-reuse design:** one forward + one JVP per timestep; no extra networks or training.
- **Theory-motivated:** connects OOD detection to **typicality** via score norm vs. curvature balance.
- **Highly efficient:** drastic reduction in **NFE** vs. path-based baselines; trivially parallelizable over samples/timesteps.
- **Competitive accuracy:** reports **SOTA/near-SOTA AUROC** while being significantly faster.

### Weaknesses
While I’m not a domain expert, I do have the following concerns:

**1. Diffusion model choice vs. dataset/task fit**  
I’m not fully certain the paper establishes that the proposed statistic is portable across diffusion families and problem settings. If the backbone is latent diffusion, the score/curvature live in latent space while many intuitions and downstream decisions operate in pixel space; the semantics of “typicality” may differ between these spaces and make cross-setup comparisons hard to trust. Conversely, pixel-space score models (EDM/VE/VP) are heavier and their noise schedules differ, so the same two-timestep heuristic and thresholding may not transfer. The reliance on ε/v parameterizations also introduces schedule-specific conversions to score; small mismatches in schedule, EMA usage, or conditioning (class-conditional vs. unconditional, text guidance) can change the magnitude and distribution of the statistic. Architectural shifts (UNet ↔ DiT) and training recipes (noise schedule, optimizer, data filtering) could further alter curvature behavior, raising concerns about whether results reflect the method or the particular diffusion recipe chosen for each dataset/task.

**2. Real-world complexity and high-resolution data**  
On complex, high-resolution images, a single global score/curvature ratio may blur heterogeneous content: small out-of-domain regions can be averaged away by large in-domain areas, while rare but in-domain structures might be spuriously flagged. As resolution grows, Hutchinson-based curvature estimates tend to be noisier, sign instabilities near zero can occur, and numerical sensitivity increases, which can propagate into unstable OOD scores. Calibration learned on a narrow ID slice may drift across cameras, ISPs, compression levels, lighting, or seasonal/style shifts; near-OOD shifts are more common in practice than far-OOD dataset swaps and may be harder to separate. The choice of resolution and noise levels interacts with SNR and texture statistics, so operating points that work on small benchmarks may not hold at 512²–1024² or beyond. Finally, long-tailed categories and mixed scenes introduce class/region dependence in the statistic, complicating threshold comparability and raising the risk of biased false positives/negatives in deployment.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

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
This paper presents a novel approach to detecting out-of-distribution inputs by combining a diffusion model’s score function norm with the Jacobian trace curvature into a single typicality statistic. This statistic is then calibrated using kernel density estimation to accurately represent the in-distribution density. The key advantage of this approach lies in its efficiency, as it computes each score using only a single forward pass and a single Jacobian vector product, utilizing Hutchinson’s trace estimator. This avoids the need for serial trajectory integration and significantly reduces the number of evaluations. Moreover, this approach enables full parallelism, further enhancing its computational efficiency.

One notable aspect of the paper is the offline selection of noise levels. This is achieved through an SNR study, which helps determine the optimal noise level for the given task. The paper shows that evaluating early and mid-schedule steps, such as $t=1$ and $t=300$, remains robust without tuning on out-of-distribution data.

### Strengths
- **Computational efficiency and performance across domains:** This paper attains competitive AUROC score on CIFAR‑10/100, SVHN, and CelebA while reducing computational burdens, and it perfectly separates several shifts on DeepMind Control Suite.

- **Data-driven calibration and noise selection:** kernel density estimation over in‑distribution statistics yields a calibrated anomaly score, and noise levels are picked from an offline SNR curve such as $t=1$ and $t=300$, avoiding out-of-distribution set leakage while maintaining robustness.

### Weaknesses
- The technical contribution of this paper appears similar to the previous study [1] that employs JVP and trace operators. While I acknowledge that this paper proposes a practical approach using KDE for calibration, the main idea is quite similar to that of the previous paper. They should clarify the technical differences and the reasons why this approach is more meaningful.     

- This paper primarily focused on relatively low-resolution image diffusion models. However, recent diffusion-based generative models have  been capable of generating high-resolution images by adapting VAE-style dimension-reduction approaches. If this method is empirically constrained to pixel-space diffusion models, the authors should provide a clear explanation for this limitation and identify the underlying numerical errors.   

**Reference**

[1] Jeon, Dongjae, Dueun Kim, and Albert No. "Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes." ICML2025

### Questions
- I was curious about comparison between this method with unsupervised approaches that employ VAE-style encoders and decoders. The paper mentions that it offers unsupervised approaches, but I’m not sure about the differences and contributions compared to classical VAE-style out-of-distribution detection. Table 1 provides a partial performance comparison in C10, but I couldn’t delve into the fundamental strength of its high-level understanding.

### Soundness
3

### Presentation
2

### Contribution
2
