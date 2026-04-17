# LaMbDA: Local Latent Embedding Alignment for Cross-modal Time-Series Diffusion

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
We present a mutually aligned diffusion framework for cross‑modal time‑series generation that treats paired modalities X and Y as complementary observations of a shared latent dynamical process and couples their denoising trajectories through stepwise alignment of local latent embeddings. We instantiate this as LaMbDA (Local latent eMbedDing Alignment), a lightweight objective that enforces phase consistency by encouraging local latent neighborhoods of X and Y to inhabit a shared local manifold. LaMbDA augments the diffusion loss by incorporating first-order sequence-contrastive and second-order covariance alignment terms across modalities at matched timesteps. Aligning their local embeddings allows each modality to help denoise the other and resolve ambiguities throughout the reverse process. Human biomechanics provides a strong testbed for this approach: paired, synchronized measurements (e.g., joint kinematics and ground‑reaction forces) capture the same movement state while reflecting practical constraints such as sensor dropout and cost. We evaluate LaMbDA extensively on biomechanical data and complement this with controlled studies on canonical synthetic dynamical systems (Lorenz attractor; double pendulum in non‑chaotic and chaotic regimes) to probe generality under varying dynamical complexity. Across all these settings, aligning local latent statistics consistently improves generation fidelity, phase coherence, and representation quality for downstream probes, without architectural changes or inference overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a diffusion-based latent manifold alignment method to achieve two-modality learning for dynamical systems.

### Strengths
The paper is well-structured, and the model is clearly presented. The theoretical foundations are included. Experiments demonstrate the mechanism and the superior performance.

### Weaknesses
The baseline comparisons are missing. Some analysis needs to be added.

### Questions
1.	In table 4, can the author explain why the result of Y|X is similar for different ablation scenarios? 
2.	The author is suggested to give some raw trajectory visualization for X and Y to show their synchronized behaviors, which motivates the alignment loss. 
3.	The baseline of diffusion model, dynamic prediction models, and other time-series methods is lacking. For example, other methods may perform well in the reconstruction of Fig. 2B. 
4.	The sensitivity analysis with respect to the hyperparameters is lacking. 
5.	The limitations are absent.

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
3

### Summary
This paper proposed a difussion framework for cross-modal latent alighment called LaMbDA. The framework used one order and two order loss for reguliraztion of the model training. The model was trained and tested on synsetic Lorenz system data, and locomotor data. The model experiment showed the alignment with MSE and FID, compared with self-supervised learning, showed the learned representation using UMAP, did ablation study on terms in the loss function. The figures and tables support their claim that the alignment increases the generation fidelity.

### Strengths
The paper gave a novel frame work with two identical diffusion for the latent alignment. I have checked the formulas in Section 3, they all make sense and does not have ambiguity or miss leading notation. The paper did plenty experiments to support its claim, from the values of the figures and tables. All the details of the implementation I could found described in the Appendix.

### Weaknesses
The experiment with the alignment of angles-moments, angles-GRF, moments-GRF, seems all 3-D vectors if I understand correctly. If like that, I would like to see the author could align diffrent high dimensional observations.

### Questions
1. Have you tried to see the results of different latent dimension?

2. For some of the pairs, angles-moments, angles-GRF, moments-GRF,  the difference of the results of X on Y and Y on X is large, for example in Table 1 (0.14 vs 0.07, 0.19 vs 0.06, 0.07 vs 0.03 for MSE, same big difference for FID) , have you considered the reason why this happens? 

3. Do you have an order of sampling X and Y within time step and the first step, does this involve bias and relate to the values of X conditioned on Y and Y conditioned on X?

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
4

### Summary
In this work, the authors proposed a multi-modal alignment framework named LAMbDA for cross-modal time-series data. Given the two paired observed modalities X and Y from a shared latent dynamical process, the main idea of the paper is to add the alignment regularizors on the latent hidden states of these two modalities data to add the inductive biases constraints to the model. LaMbDA adds this alignment using a combination of a first-order sequence-contrastive loss and a second-order covariance alignment term. On the experimental side, the authors evaluate the effectiveness of the proposed method on synthesized Lorenz dynamics dataset and the human biomechanical locomotor datasets.

### Strengths
1. The paper is well-written and easy for the audiance to read and follow.
2. The modeling of observed multi-modal data and its related alignment tasks are critical problems in dynamical systems.
3. The proposed LAMbDA framework significantly improves cross-modal generation performance compared to the non-aligned baseline across multiple modality pairs and metrics (e.g., MSE, FID, Predictive score).

### Weaknesses
1. I think that the main algorithm novelty of this work LaMbDA mainly comes from the combination of a not new contrastive loss term [1] and a covariance loss (second-order), it's actually a bit empirical. The difference is only that this paper puts them onto this new dynamical system alignment application context.
2. While focused in dynamical systems scenarios, the proposed alignment method does not explicitly enforce alignment based on the dynamics or flow of the system (e.g., considering vector fields or preserving local structures beyond the simple Taylor approximation). 
3. In the experiments, the paper only compares the generative performance against non-aligned models. And also some baselines, like MSE and SimCLR, are actually weak, To be considered as a performant method, it must benchmark LAMbDA's synthesis quality against other powerful generative architectures with alignment.

[1] Representation Learning with Contrastive Predictive Coding. 2018. Oord, et al.

### Questions
I have no more questions, other concerns please relate to my weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
