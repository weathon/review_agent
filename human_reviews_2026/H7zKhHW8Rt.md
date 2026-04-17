# Constrained Probabilistic Diffusion Model for Seismic Data Reconstruction Using a Restoration Operator Based on a Deep Image Prior

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Seismic data acquisition is of the utmost importance in the oil industry, as it allows the representation of subsurface geological features. Several factors affect the number of sources or receivers during the acquisition, impacting seismic data quality. Different methodologies have been developed for reconstruction based on generative models for seismic signals. Methods based on diffusion models (DM) have emerged in this context for seismic data reconstruction, guiding the image generation or the reverse process to solve the reconstruction problem using a closed-form solution and deep learning-based solvers. However, a disadvantage of these methodologies is that they cannot extract all the features necessary to represent the data domain, which is crucial for accurate reconstruction. As a result, the entire DM must be re-trained for each experiment, leading to high computational costs due to its complexity. We propose DM-RODIP, an alternative DM approach for seismic data reconstruction, where the reverse process of a pre-trained DM is guided toward a reconstruction problem solution using a restoration operator based on a Deep Image Prior. The proposed method was evaluated on synthetic and field data, demonstrating superior reconstruction performance with improvements of up to 10.2 dB in PSNR and 0.09 in SSIM for synthetic data, and 1.0 dB in PSNR and 0.04 in SSIM for field data, outperforming state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces DM-RODIP, a method for seismic data reconstruction that integrates a diffusion model (DM) with a Restoration Operator based on a Deep Image Prior (DIP). The key idea is to guide the reverse diffusion process using a pretrained restoration network (an Attention U-Net trained in DIP fashion), thereby avoiding full retraining of the diffusion model for each reconstruction task. The method aims to reduce computational cost of retraining while improving reconstruction quality on both synthetic and field seismic datasets. Experiments demonstrate performance  gains over  baselines such as CCSeis-DDPM and standalone DIP reconstructions.

### Strengths
* Reduced computational overhead through reuse of pretrained diffusion models
The main practical contribution lies in reusing a pretrained diffusion model and optimizing only a lightweight Deep Image Prior (DIP) restoration network for each reconstruction task. This design potentially lowers GPU cost, training time, and data requirements compared to full retraining of diffusion models.

* The proposed method seems to outperform the baselines and improve the quality of reconstruction.

### Weaknesses
* The conceptual novelty is limited. The proposed framework primarily combines existing components — a pretrained diffusion model, a Deep Image Prior, and a restoration prior— without introducing a new theoretical idea or methodological principle. There is no additional insight as to why this combination achieves better performance. 

*  Contribution is ambiguous. The paper frames the method as a new diffusion variant (“Constrained Probabilistic Diffusion Model”), but in practice it is just  reusing a pretrained DM with a DIP-based refinement. The key contribution therefore lies in connecting previously established ideas rather than advancing diffusion modeling itself.

* The paper  lack of theoretical or empirical analysis of the DRP.  The paper does not analyze the goal and role of the  restoration operator in  the diffusion process. The guidance from the restoration operator is not clear. 

* The scale of the experiments is limited. This conceptually should work on other inverse problems, however, no other image restoration or inverse problem is discussed.

### Questions
How does including the DRP affect the computation cost and performance?
How does the restoration operator mathematically constrain or modify the reverse diffusion dynamics? Is there an objective function or probabilistic interpretation linking the DIP output to the diffusion posterior?

### Soundness
2

### Presentation
3

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
This paper proposes DM-RODIP, a constrained probabilistic diffusion framework for seismic data reconstruction under sparse and irregular sampling conditions. The method integrates three components: a pre-trained diffusion model (DM) serving as a probabilistic prior of seismic data distribution, a Deep Image Prior (DIP) module acting as an implicit structural regularizer, and a Restoration Operator formulated as a proximal optimization step enforcing measurement consistency. Unlike conventional diffusion-based reconstruction methods that require retraining the entire model for each dataset or sampling configuration, DM-RODIP achieves efficient reconstruction without retraining, by coupling DIP and Restoration into the reverse diffusion loop. Experiments on synthetic and field datasets (Land 3D, Stratton) show consistent improvements over DIP-only and DDPM-based baselines.

### Strengths
1. The paper elegantly combines diffusion models (probabilistic priors) with proximal optimization and implicit deep priors. The Restoration Operator enforces physical measurement constraints, bridging the gap between generative modeling and data-consistent inversion. The overall pipeline is theoretically sound and computationally efficient.

2. Unlike previous DM-based methods that are highly data-specific, this approach decouples distribution learning (offline) from reconstruction (online), allowing direct adaptation to new observation patterns.

3. Visual examples convincingly show better layer continuity and fewer artifacts in reconstructed seismic sections. The method is applicable not only to seismic reconstruction but also to other ill-posed imaging problems (denoising, super-resolution, inpainting).

### Weaknesses
1. The paper lacks quantitative evidence showing the distinct contributions of DIP and the Restoration Operator. For instance, results with “DM only”, “DM + Restoration”, and full “DM + DIP + Restoration” would help clarify each module’s necessity.
2. The paper describes the DIP–Restoration coupling intuitively but lacks a formal Bayesian or probabilistic justification.
3. The workflow diagram (Fig. 1) is not intuitive. It illustrates the iterative reverse diffusion loop but does not clearly differentiate between the training phase (DM pre-training) and the inference/reconstruction phase. I recommend redesigning the figure to explicitly show two parts: (1) Offline training of the diffusion model using full seismic data; (2) Online reconstruction using DIP and Restoration Operator on observed sparse data.
4. The method is compared mainly with DIP and DDPM; missing stronger baselines such as conditional or plug-and-play diffusion models.
5. Several notations are undefined or underexplained.​

### Questions
1. Can you provide quantitative evidence (PSNR/SSIM) showing the effect of removing the DIP module?
2. How sensitive is reconstruction performance to hyper parameters in the Restoration Operator?
3. Is the Restoration Operator applied at every diffusion step or intermittently?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DM-RODIP, which uses the reverse process of a pre-trained diffusion model to generate or reconstruct seismic data. At each step, it uses a recovery operator based on DIP to guide inverse sampling. This approach aims to preserve the statistical prior captured by the diffusion model during the generation process, while also reconstructing the seismic data. Finally, the experimental results using synthetic and measured (Stratton 3D) data demonstrated that DM-RODIP achieves superior reconstruction quality compared to several baseline methods, such as DIP and CCSeis-DDPM.

### Strengths
1. Combining pre-trained DM (e.g. capturing global statistics) with DIP (e.g. leveraging structural priors while only requiring observations) is an interesting idea.

2. The proposed solution avoids the need to retrain the entire diffusion model for each specific reconstruction scenario, thereby reducing computational costs.

3. The experimental results on seismic imaging are good, so it might be useful for scenarios involving sparse or mismatched training data, compensating for the limitations of pre-trained diffusion models in domain-specific features.

### Weaknesses
Although the solution is interesting, the main idea is not entirely novel, but rather a combination of existing ideas and their engineering implementation, i.e., the method is an extension of CDDIP (Goyes-Penafiel et al., 2025) which adopts an additional existing calibration method (restoration prior).

Although the paper aims to reduce computational costs by eliminating the need to retrain the deep model, the algorithm itself often carries out DIP (with optimisation) and proxy steps during inference, which can be computationally intensive. The paper lacks detailed comparisons of runtime, memory consumption, per-DIP optimisation costs and overall inference time. This makes it impossible to verify the claim that computational costs are lower.  For example, if DIP uses the current $X_{t−1}$ as input for fitting observations at each step, this amounts to iterative optimisation (and calling the forward model multiple times) during the inference phase. Furthermore, to my knowledge, DIP itself suffers from overfitting due to its reliance on early stopping strategies. Solving a DIP subproblem at each step during backpropagation (as per line 4 of Algorithm 1) could be computationally intensive, potentially introducing noise or instability during optimisation. A further clarification is expected.

The comparison methods primarily involve DIP (isolated) and CCSeis-DDPM. Although the paper cites several works on diffusion-guided or projection-related approaches (such as DPS and DiffPIR), it neither quantifies these as strong baselines for comparison nor integrates them.

### Questions
Line 288. In Algorithm1, what is the $\sqrt{\Sigma}$?

It's better to present some failed cases and analyse limitations (such as degradation under high noise or extremely sparse sampling conditions).

How to prevent overfitting? As the DIP is involved, this approach risks overfitting observational noise or implicitly reusing observations multiple times.

How do you crop/normalise Straton 3D data?

### Soundness
3

### Presentation
3

### Contribution
3
