# Sparse-Compression Diffusion Models

- Decision: Reject
- Scores: 0, 2, 4, 6

## Abstract
Diffusion models have demonstrated capabilities in generating synthetic data, especially the pictural ones. However, the conventional reverse diffusion process merely overfits in data denoising. Empirical studies indicate that the denoising mechanism obstructs the diffusion models to generate robust physical synthetic data. In this work, we propose the Sparse-Compression Diffusion Models (SCDM), with a novel reverse diffusion process that enforces compression. The compression attempts to recover the low-dimensional and sparse representations, potentially enabling the ability to capture the latent factors of some rules. SCDM demonstrates great generalization on datasets of images and physical scenarios. Our experiments also prove that the sparse latent representations learned by SCDM align well with scientific theories in physical scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes SCDM, a diffusion framework that inserts a compression-sparsity module in each reverse step: encode $x_t\rightarrow z_t$, project onto a mixture of orthogonal subspaces, apply an ISTA-style soft-threshold, and then predict the noise from the resulting sparse code to form the usual DDPM update rule. The claimed contributions are (1) integrating "sparse rate reduction" into diffusion, and (2) demonstrating that sparse latents "align with scientific theories," using toy/small physics datasets (circle, oscillators, pendulums).

### Strengths
Pursuing a sparse latent structure during reverse diffusion is a reasonable direction. The paper attempts to formalize this via rate-reduction terms (eq. 5–8).

### Weaknesses
1. The problem statement is very unclear and overstated. The paper asserts that standard reverse diffusion "merely overfits in data denoising" and "obstructs… robust physical synthetic data," but provides no evidence or references establishing this problem as real or general. This is presented in the abstract/introduction and is used to justify the method. However, the experiments do not demonstrate real-world failures of DDPM beyond small toy settings, and does not reveal why is it that DDPM fails. To me, this appears to be a flaw in the experiments.
2. The core claim that the learned latents "align well with scientific theories" is asserted (what does scientific theories even mean?), but the evidence is qualitative plots of sinusoidal-like activation norms (Fig. 4) and small metric gains on tiny datasets. This does not validate alignment with the governing physical laws in a robust way.
3. The projection operator in algorithm 1 is not defined clearly anywhere in the paper.
4. Eq. 14 mixes the usual DDPM mean with $z_{t-1}^{K}$ in place of $x_{t}$. There is no derivation showing that this substitution yields a valid reverse process.
5. All datasets are tiny and insufficient to evaluate a general generative modeling claim about "robust physical data synthesis."
6. The motivation and what exactly is new are very hard to track.
7. Numerous grammatical errors and awkward phrasing (e.g., "pictural," "merely overfits in data denoising" in the abstract), plus inconsistent use of technical terms, make the paper feel unpolished.

### Questions
See weaknesses.

### Soundness
1

### Presentation
1

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
The paper proposes Sparse-Compression Diffusion Models (SCDM), an extension of DDPMs that introduces a compression–sparsity mechanism into the reverse diffusion step. The method draws heavily on the CRATE (Yu et al., 2023) framework for sparse rate reduction, combining subspace projection and ISTA-style soft-thresholding to encourage low-dimensional latent representations. The authors claim this yields more interpretable and physically robust generative models, supported by experiments on simple physical datasets (oscillators and pendula).

### Strengths
1. The idea of embedding sparse representation principles into diffusion modeling is conceptually sound and moderately novel in motivation.
2. The paper is structurally clear and provides a full algorithmic description (Algorithm 1), which makes replication possible in principle.
3. The attempt to link sparse representations with physical interpretability is an appealing research direction.

### Weaknesses
1. Limited Theoretical Novelty
- Nearly all equations—including the compression objective $$R(z)-R_c(z;U)-\lambda\|z\|_0$$, the log-det rate formulation, the softmax-weighted subspace projection, and the ISTA sparsification step—are taken directly from the CRATE paper with minimal adaptation.
- The only  change is placing these CRATE blocks inside the DDPM reverse step and feeding the sparse code into the noise predictor. This is an implementation variant, not a new theory.
- There is no probabilistic or variational derivation showing that the added term improves the diffusion likelihood or score estimation.
2. Weak Mathematical Rigor
- Notation and definitions are inconsistent (e.g., non-differentiable $$ \| \cdot \|_0 $$ with no surrogate use).
- The information-theoretic "coding rate" is adopted from CRATE without re-deriving its relevance to diffusion modeling.
- No analysis is given of how the added compression term affects convergence or stability of the reverse process.
3. Insufficient Empirical Rigor
- Metrics: RMAE, MDN %CE, and %VFE measure geometric fidelity of trajectories, not generative quality. They do not test likelihood, diversity, or coverage. 
- Baselines: The comparison excludes the 'Physics-Informed Diffusion Model' (Shu et al., JCP 2023) despite citing it. SCDM also uses an extra ISTA bottleneck, so model capacity differs from DDPM/DDIM baselines.
- Datasets: All datasets are low-dimensional (2–4 vars) and small. These tasks are trivial for diffusion models and cannot demonstrate generalization or robustness.
- Reporting: 
    -- No multiple seeds, confidence intervals, or significance tests. Figures lack scales and quantitative analysis.
    -- Not clear how individual CRATE components influence sample quality
4. Citation and Scholarly Issues
- Roughly half the references are arXiv-only. Foundational works (VAE, DDPM, SSC) have wrong years or missing venues(Arxiv references even  for papers which have been published in ML venues)
5. Writing and Presentation
- English usage is often incorrect ("successful generating," "merely overfits in data denoising") and many claims are vague ("intrinsic factors of some rules").
- Assertions of "alignment with scientific theories" are rhetorical; no quantitative mapping to actual equations is shown.

### Questions
1. How does the compression–sparsity term modify or complement the diffusion ELBO?  
2. Why are standard generative metrics (NLL, MMD, FID) absent?  
3. Why was the physics-informed diffusion baseline omitted despite citation?  
4. Could simpler non-diffusion models (e.g., sparse VAE) achieve similar performance on these datasets?

SCDM is incremental and under-evidenced. The theoretical framework is a direct transplant of CRATE into diffusion modeling, and the experiments are confined to trivial physical systems with inadequate baselines and metrics. The work lacks the mathematical and empirical depth required for a top-tier venue.

-----
Recommended Improvements
------
1. Provide a formal derivation connecting the compression–sparsity term to diffusion likelihood or score matching.  
2. Expand experiments to higher-dimensional datasets and include standard generative metrics.  
3. Add physics-informed diffusion and sparse generative models as baselines.  
4. Report mean +- std across multiple runs and perform significance tests.  
5. Correct reference list and cite archival versions where available.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel reverse diffusion process to force the model to better learn low-dimensional and sparse latent representations. The formula description in the paper is detailed and concise, and experiments have been conducted to verify the effectiveness of this method.

### Strengths
- This paper proposes a novel integration of sparse rate reduction with diffusion models which has not been previously explored in the diffusion model literature. 
- Tests on 7 physical datasets covering diverse low-dimensional physical systems, with rigorous baselines (DDPM, DDIM, SDD) and three physics-relevant metric.
- The thesis is structured coherently, with clear writing and concise descriptions of the proposed methods. The logical flow of methodology, experiments, and results facilitates easy comprehension of the technical contributions.

### Weaknesses
- For quantitative accuracy assessment, the manuscript states that SCDM is compared with 3 baselines (as shown in Table 1), yet Line 346 claims a comparison with 5 baselines—this quantitative discrepancy requires clarification. Furthermore, 3 baselines may be insufficient to fully substantiate the superiority of SCDM, especially given that SCDM does not achieve the best performance across all metrics in Table 1. This limits the persuasiveness of its claimed effectiveness.
- The work proposes a novel reverse process for diffusion models, which in principle could be extended to downstream tasks such as image generation. However, the current study only focuses on sparse compression, with no supplementary experiments to validate SCDM’s performance in other tasks (e.g., image quality evaluation via FID, or model efficiency metrics like inference time/parameter overhead). This narrow scope fails to demonstrate its versatility as a general-purpose diffusion model.
- SCDM’s compression process relies on two core steps: projection and transformation. Ablation studies to isolate and quantify the individual contributions of these two steps are critical to dissecting the model’s working mechanism—without such analyses, the relative importance of each component remains unclear, and the rationality of the proposed pipeline cannot be fully verified.
- The manuscript would benefit from replacing raster graphics with vector graphics (e.g., figure 2). Vector graphics offer higher resolution and scalability, which enhances the clarity and reproducibility of key results—an important aspect of academic manuscript presentation.

### Questions
- See Weaknesses.

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
2

### Summary
The authors propose a novel design/regularization procedure incorporated into the noise predictor of a diffusion denoiser. They perform the denoising using a compressed, sparse representation of x_t, referred to as z_t. The compressed, sparse representation is trained to have these properties by following CRATE from Yu et al. 2023. They show that this intervention improves sampling performance on datasets which rightfully lie on a low-dimensional manifold. They also show that these learned representations match the known latent variables of their datasets.

### Strengths
Novel idea, good sampling results on their chosen datasets. I cannot speak to the validity of the sparse latent representations aligning with the known latent variables of the physical systems that these diffusion models are modelling.

### Weaknesses
I am a bit confused about the construction of their denoising net. Their noise prediction network takes in z_t, the compressed representation of x_t. Shouldn't the noise component of x_t, which the network is trying to predict, be incompressible? So shouldn't z_t be throwing away information that the noise prediction network g needs to do its job? My best guess is that their implementation doesn't actually work this way and they are really predicting x_0 hat instead of epsilon hat, or something like that. Otherwise I have no idea how this achieves reasonable sampling quality. Perhaps this is all fine and I am just terminally confused. 

"non-markova"-> non markovian

### Questions
Please address my confusion above in the weaknesses section

### Soundness
2

### Presentation
2

### Contribution
2
