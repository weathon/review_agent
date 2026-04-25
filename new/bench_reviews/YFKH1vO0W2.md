Now let me search for calibration papers.Now I have enough information to write the final review. Let me compile it.---

## Summary

DiffSDA extends diffusion autoencoders (DiffAE) to the sequential setting, introducing a probabilistic model with joint (non-factorized) static/dynamic priors, an efficient LDM-EDM decoder (63 NFEs), and an LSTM-based sequential semantic encoder that extracts static code $s_0$ and dynamic codes $\mathbf{d}_0^{1:V}$ through architectural inductive biases rather than explicit regularizers. The paper additionally proposes a new evaluation protocol for real-world sequential disentanglement, using three high-resolution in-the-wild visual datasets (VoxCeleb, CelebV-HQ, TaiChi-HD) not previously used for this task, and demonstrates strong quantitative improvements over SPYL and DBSE on TIMIT audio as well.

---

## Strengths

- **Novel extension of DiffAE to sequential disentanglement (Sec. 4.1, Eq. 4–8):** To the authors' knowledge (and consistent with the related work landscape), this is the first model to factorize static and dynamic factors within a diffusion autoencoder framework. The probabilistic formulation allowing dependent static/dynamic priors is a meaningful departure from VAE-based methods that require explicit independence constraints, mutual information losses, and numerous hyper-parameters.

- **Quantitatively compelling audio results (Table 3):** On TIMIT, DiffSDA achieves a 42.29% dynamic EER versus 31.11% for DBSE (the prior state of the art), a ~36% relative improvement in the disentanglement gap — and this is measured with EER, a metric that *directly* tests whether dynamic factors are separable from static ones. Seven baselines are included, making the comparison credible and the gain substantial.

- **Real-world evaluation protocol with three new high-resolution datasets (Sec. 5):** Prior sequential disentanglement work (MMNIST, MUG 64×64, synthetic sequences) evaluates on low-resolution, controlled data. The introduction of VoxCeleb and CelebV-HQ at 256×256 is a genuine community contribution: the baseline results in Fig. 3 confirm that existing VAE-based methods fail at this scale, while DiffSDA succeeds qualitatively.

- **Practical scalability (Sec. 4.2):** Integrating LDM's VQ-VAE compression with EDM's 63-NFE sampler is a concrete engineering contribution that makes diffusion-based video processing tractable without sacrificing generation quality.

---

## Weaknesses

### Fatal
*None.* The paper does not present fundamentally flawed proofs, fabricated data, or an unsupported core claim at the level that would invalidate all results.

### Major

- **Visual evaluation metrics (AED, AKD) primarily measure reconstruction/generation quality, not disentanglement quality.** Table 2 reveals that DiffSDA's reconstruction quality is 3–4 orders of magnitude better than VAE-based baselines on MSE (e.g., 3.0e−7 vs. 0.001 on MUG; 5.9e−4 vs. 0.012 on CelebV-HQ). Since AED and AKD are perceptual distance metrics computed on reconstructed outputs, a model with vastly superior reconstruction will automatically score better on these metrics regardless of whether its latent space is actually more disentangled. The swap metric measures the distance between a swap output and the source video — a sharper, more faithful diffusion decoder will trivially produce lower AED/AKD on both the static-frozen and dynamics-frozen conditions. Table 1's headline improvements (e.g., AKD on CelebV-HQ: 6.932 vs. 28.69/39.16) thus cannot be disentangled from the backbone quality gap. The audio EER in Table 3 sidesteps this problem because EER measures classification accuracy on latent representations, not perceptual output quality. An analogous disentanglement-specific visual metric (e.g., accuracy of a linear probe predicting identity from $d_0^\tau$, or predicting motion/pose from $s_0$) is absent from the visual evaluation, leaving the claim of superior visual disentanglement relying solely on metrics that are blind to this distinction.

- **No ablation studies.** DiffSDA introduces three distinct components over baselines: (i) the LSTM-based factorized semantic encoder, (ii) the EDM decoder replacing DDIM, and (iii) the LDM backbone. The reconstruction quality gap (Table 2) strongly suggests the LDM/EDM backbone upgrade alone—applied to the baselines' encoders—would dominate measured improvements. Without at minimum comparing: (a) DiffSDA encoder + baseline DDIM decoder, (b) baseline encoder + DiffSDA decoder, or (c) a non-disentangled LDM autoencoder on the same swap metrics, it is impossible to attribute observed improvements to the disentanglement mechanism specifically versus the backbone quality improvement. This gap makes it impossible to evaluate the paper's core architectural claims.

### Minor

- **Zero-shot generalization is assessed qualitatively only (Sec. 5.2).** Zero-shot disentanglement is presented as a distinct contribution of the evaluation protocol ("for the first time"), yet the evaluation consists of cherry-picked qualitative examples (Figs. 1, 4). No AED/AKD scores are reported for the zero-shot setting, and no competing method is evaluated zero-shot for comparison. This section constitutes a demonstration, not a scientific evaluation.

- **PCA-based multifactor exploration is qualitative and post-hoc (Sec. 5.3).** Traversing PCA directions of the latent space is standard practice (inherited directly from DiffAE, Preechakul et al. 2022) and does not constitute evidence of structured multifactor disentanglement without quantitative validation. The paper itself notes that some PCA directions control "image blurriness," which suggests that generation artifacts, rather than semantically meaningful factors, can enter the latent space.

### Trivial

- **Unconditional generation (Sec. 5.4) lacks standard generation quality metrics** such as FVD or FID, which would be expected for a visual generative model claiming state-of-the-art quality.

---

## Nice-to-Haves

- Adding a non-disentangled diffusion baseline (e.g., a frame-independent DiffAE without the LSTM factorization) evaluated on the same swap metrics would directly test whether the disentanglement mechanism adds value over simply having a better decoder — this is the single most informative missing experiment.
- Training a linear probe to predict identity from $d_0^\tau$ and motion/pose from $s_0$ would provide a disentanglement-specific metric analogous to EER for the visual domain, closing the gap between the audio and video evaluations.
- A quantitative zero-shot evaluation (AED/AKD on held-out datasets) would substantiate the zero-shot claim.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

1. **"The generative model in Eq. 4 uses a joint prior that does not encourage independence"** — The paper explicitly motivates the joint prior as a design choice (expressiveness, non-autoregressive sampling, causality). This is a stated architectural decision, not an oversight. Criticizing the absence of a factorized prior without the paper making that claim is scope creep.

2. **"Argument (i) is incorrect: the LSTM can encode average motion into h^V"** — The critic is correct that h^V encodes the full sequence temporally. However, since $s_0$ is a single vector shared identically across all $\tau$ in the decoder, it literally cannot carry frame-specific dynamic information (which by definition varies per $\tau$). The inductive bias is weaker than a formal regularizer but is not "incorrect as stated." The concern reduces to whether average/global dynamic information leaks into $s_0$, which is a nuance worth noting but is softer than the harsh framing.

3. **Criticism of AED/AKD being "borrowed from animation" and not "introduced by the paper"** — The paper explicitly acknowledges these metrics are adapted from Siarohin et al. (2019). There is no false novelty claim here.

4. **"Results are sensitive to cherry-picking in Section 5.3"** — Qualitative exploration is standard in latent space analysis papers. The "blurriness" PCA component concern is kept (minor), but the generic charge of cherry-picking is removed.

---

## Novel Insights

The most incisive insight across reviewers is the structural mismatch between the paper's primary evaluation setting (visual, AED/AKD) and its secondary evaluation setting (audio, EER): the audio metrics directly measure whether static representations are invariant to dynamics and vice versa, whereas the visual metrics measure perceptual fidelity of outputs from a system with a fundamentally better decoder than all baselines. This asymmetry means the paper has strong evidence for disentanglement in the domain where it is hardest to claim novelty (audio, where VAE-based methods also achieve reasonable results) and weaker evidence for disentanglement in the domain where it claims the most novelty (high-resolution video). A paper that directly bridged this gap with a visual disentanglement metric would be substantially more convincing.

---

## Suggestions

1. **Run the most important missing control:** Take the DiffSDA decoder (LDM + EDM) but use a *non-disentangled* single-code encoder (no static/dynamic split). Evaluate on the same swap metrics. If this baseline scores comparably to full DiffSDA on Table 1, the disentanglement architecture adds little over the backbone; if it scores much worse, the disentanglement mechanism is validated.
2. **Add a linear probe disentanglement metric for visual data:** Train classifiers on $s_0$ to predict dynamic attributes, and on $\mathbf{d}_0^\tau$ to predict static attributes. Near-chance accuracy constitutes direct evidence of factorization.
3. **Report AED/AKD for zero-shot conditions** alongside qualitative figures.

---

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Comparison |
|---|---|---|---|
| `hBGavkf61a.md` – Diffusion Bridge AutoEncoders | 7.25 | Accept (Spotlight) | Diffusion AE novel contribution with strong ablations and theoretical grounding. More technically rigorous than DiffSDA. |
| `Lut5t3qElA.md` – V3 Unsupervised Disentanglement | 6.40 | Accept (Poster) | Novel disentanglement across modalities; experiments limited to toy data. Accepted despite simpler experimental scope. |
| `iKd99CYwPX.md` – Deterministic Diffusion Sequential | 4.50 | Reject | Diffusion for sequential tasks, incremental contribution, weaker novelty. |
| `AOSsLRKQrX.md` – DisFormer Disentangled Visual Dynamics | 3.50 | Reject | Disentangled object representations for video dynamics; rejected for evaluation gaps and limited scope. |

DiffSDA sits above DisFormer (3.5) and the sequential diffusion reject (4.5): it is genuinely novel, covers multiple modalities, introduces a new benchmark, and has strong audio quantitative results. However, the central visual disentanglement claim is not independently validated (metric confounding) and there are no ablations — concerns more fundamental than V3's "only simple datasets" limitation (6.4) or DBAE's presentation issues (7.25). Positioned below the accepted papers that had cleaner experimental validation, the score lands at **5.0** — borderline reject reflecting a paper with real contributions that fall short of convincingly supporting its primary claim in the visual domain due to the metric confounding and ablation gaps.

**Originality:** Moderate-high (first diffusion-based sequential disentanglement).
**Research question importance:** High (unsupervised disentanglement on real-world video is an open problem).
**Claims well-supported:** Partially — audio claims are well-supported; visual claims have a methodological gap.
**Soundness of experiments:** Moderate — AED/AKD design is appropriate for visual animation transfer but problematic for disentanglement claims when decoder quality differs dramatically across compared methods.
**Clarity of writing:** Good.
**Value to community:** Moderate-to-high if ablation and controlled experiments were added.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>