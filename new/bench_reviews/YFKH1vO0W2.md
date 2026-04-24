## Summary

This paper introduces DiffSDA, a diffusion-based sequential disentanglement autoencoder that extends diffusion autoencoders to video sequences by factorizing latent representations into static and dynamic components. The model employs a latent diffusion model (LDM) with a pretrained VQ-VAE and an EDM sampler to achieve efficient high-resolution video processing (256×256), and evaluates on real-world datasets (VoxCeleb, CelebV-HQ, TaiChi-HD) as well as TIMIT audio.

## Strengths

- **First diffusion sequential disentanglement on real-world high-resolution video.** The paper extends diffusion autoencoders to the sequential setting and demonstrates qualitatively strong swap and reconstruction results on challenging, high-resolution face and human-motion datasets (Figs. 1, 3, 4, 6; Tables 1–2), moving beyond the toy datasets common in prior sequential disentanglement work.
- **Scalable architecture.** The combination of LDM for latent space efficiency and EDM for fast sampling (63 NFEs) enables practical processing of 256×256 videos (Sec. 4.2). This engineering contribution addresses a genuine computational bottleneck in prior diffusion autoencoders.
- **Modality flexibility.** Replacing the U-Net with an MLP yields competitive audio disentanglement results on TIMIT (Table 3), suggesting the framework generalizes beyond vision with minimal architectural changes.
- **Real-world evaluation protocol.** The paper adopts high-resolution video datasets not previously used for sequential disentanglement and proposes animation-derived unsupervised swap metrics (AED/AKD) to reduce reliance on label-dependent classifiers (Sec. 5.1).

## Weaknesses

### Fatal

None.

### Major

- **Decoder-capacity mismatch invalidates quantitative comparisons as disentanglement evidence.** DiffSDA uses a latent diffusion decoder with a pretrained VQ-VAE and EDM sampler, while the baselines (SPYL, DBSE) use conventional VAE/GAN decoders. Tables 1 and 2 report orders-of-magnitude better reconstruction MSE and substantially lower swap metrics, but these gains primarily reflect the superior generative capacity of the diffusion pipeline rather than superior disentanglement of the latent space. The paper never isolates the disentanglement contribution from the decoder contribution—for example, by comparing against a non-factorized diffusion autoencoder using the exact same decoder, or by using the same decoder architecture for all methods. Conflating reconstruction quality with disentanglement quality is a fundamental evaluation flaw.
- **Disentanglement claim rests on weak, unverified inductive biases.** The only mechanisms enforcing the static/dynamic split are architectural heuristics: the static vector $s_0$ is shared across frames and the dynamic vectors $d_0^\tau$ are low-dimensional (Sec. 4.2). The loss (Eq. 8) contains no disentanglement regularization, mutual information penalties, or identifiability constraints. The paper provides no single-factor reconstruction ablations (e.g., reconstruct using only $s_0$ with $d$ set to mean, or vice versa) and no information-leakage quantification (e.g., classifiers predicting static attributes from $d$). Without such evidence, the central claim that the model achieves unsupervised sequential disentanglement is not empirically substantiated.

### Minor

- **Equation 4 omits the base measure.** The joint distribution in Eq. 4 is written as a product of transition/conditional densities without the base measure $p(\mathbf{x}_T^{1:V}, s_T, \mathbf{d}_T^{1:V})$. This is a notation issue that should be corrected for mathematical precision.
- **The “unconstrained prior” claim is overstated.** The introduction states the model “has no constraints on the prior distribution of the static and dynamic latents” (Sec. 1). While the training objective (Eq. 6) indeed does not use $p_{T0}$ for the semantic factors, the generation procedure explicitly employs a standard Gaussian diffusion prior ($\mathbf{z}_T \sim \mathcal{N}(0, I)$) and a DDIM reverse process (Sec. 4.2). The distinction between training flexibility and inference prior should be stated more accurately.
- **Zero-shot evaluation lacks quantitative metrics.** Section 5.2 presents only qualitative zero-shot swap results (Figs. 1, 4). Reporting AED/AKD on cross-dataset pairs (e.g., train VoxCeleb → test MUG) would strengthen the generalization claim beyond hand-selected visualizations.
- **TIMIT audio results are ambiguous.** The static EER of 4.43% is worse than all recent baselines except FHVAE (Table 3). The large disentanglement gap is driven by an unusually high dynamic EER (46.72%). Without verifying that the dynamic factor encodes useful phonetic content (e.g., via phoneme classification), it is unclear whether the high dynamic EER reflects successful purification of speaker identity or simply an uninformative/noisy dynamic representation.

### Trivial

None.

## Nice-to-Haves

- Controlled baseline with identical decoder capacity (non-factorized diffusion autoencoder) to isolate the contribution of the static/dynamic split.
- Single-factor reconstruction ablations and information-leakage tests to empirically verify disentanglement.
- Quantitative zero-swap metrics on cross-dataset pairs.
- Theoretical or empirical discussion of why architectural inductive biases suffice in this setting despite established impossibility results for unsupervised disentanglement.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Misleading multifactor claim:** The harsh reviewer argued the multifactor contribution is misleading because the model outputs a single static and dynamic vector. However, the paper explicitly titles the section “Toward Multifactor Disentanglement,” states the model has the “potential to further disentangle,” and lists “multifactor exploration” (not capability) in the contributions. The hedging is appropriate.
- **Overstated “no real-world protocol” claim:** The reviewer argued related fields already use VoxCeleb and TaiChi-HD. The paper’s claim is specifically about *sequential disentanglement* evaluation protocols, and it acknowledges animation work in Section 2. The claim is narrower than the reviewer implied.
- **Fatal mathematical inconsistency in Eq. 4:** The reviewer framed the omitted base measure as invalidating the probabilistic model. This is a notation omission, not a fundamental methodological flaw; the training objective relies only on standard score matching anyway.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

1. Add a non-factorized diffusion autoencoder baseline using the identical LDM-EDM decoder to determine how much of the quantitative improvement comes from the decoder versus the static/dynamic factorization.
2. Perform single-factor reconstruction ablations (reconstruct with only $s$ or only $d$) and train classifiers to predict static attributes from dynamic factors (and vice versa) to verify that information leakage is low.
3. Clarify Eq. 4 by including the base measure, and rephrase the “unconstrained prior” claim in the introduction to distinguish between the lack of regularization during training and the Gaussian prior used at generation time.

## Score and Decision

**Calibration anchors used:**
- `/home/wg25r/review_agent/human_reviews/hBGavkf61a.md` (DBAE, avg 7.25, Accept Spotlight): Principled diffusion autoencoder framework with theoretical guarantees and fair, thorough comparisons. DiffSDA is below this due to weaker theoretical grounding and unfair decoder comparisons.
- `/home/wg25r/review_agent/human_reviews/Lut5t3qElA.md` (V3 disentanglement, avg 6.40, Accept Poster): Clean statistical methodology with clear motivation, but limited to simpler datasets. DiffSDA has more impressive real-world scale but weaker methodological justification.
- `/home/wg25r/review_agent/human_reviews/ehr4oTe6XI.md` (Gromov-Monge disentanglement, avg 5.50, Accept Poster): Good empirical performance but unclear logical connection between method and disentanglement. DiffSDA is comparable—better real-world results, but similarly unclear mechanism and additionally confounded by decoder asymmetry.
- `/home/wg25r/review_agent/human_reviews/AOSsLRKQrX.md` (DisFormer, avg 3.50, Reject): Toy datasets, no ablations, weak disentanglement evidence. DiffSDA is clearly above this because it tackles real-world high-resolution data.
- `/home/wg25r/review_agent/human_reviews/etnG659OB9.md` (CauF-VAE, avg 3.00, Reject): Supervised setting, missing baselines, weak experiments. DiffSDA is clearly above this.

DiffSDA makes a genuine contribution by scaling sequential disentanglement to real-world high-resolution video via diffusion, and the qualitative results are compelling. However, the two major weaknesses—decoder-capacity mismatch and lack of empirical verification for disentanglement—are serious enough that the paper should not be accepted without revision. The numerical comparisons are uninterpretable as disentanglement evidence, and the central claim is supported only by heuristic architectural biases. These issues place the paper around the borderline/medium range.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>