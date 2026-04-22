# Degradation-Aware All-in-One Image Restoration via Latent Prior Encoding

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Real-world images often suffer from spatially diverse degradations such as haze, rain, snow, and low-light, significantly impacting visual quality and downstream vision tasks. Existing all-in-one restoration (AIR) approaches either depend on external text prompts or embed hand-crafted architectural priors (e.g., frequency heuristics); both impose discrete, brittle assumptions that weaken generalization to unseen or mixed degradations. To address this limitation, we propose to reframe AIR as learned latent prior inference, where degradation-aware representations are automatically inferred from the input without explicit task cues. Based on latent priors, we formulate AIR as a structured reasoning paradigm: (1) which features to route (adaptive feature selection), (2) where to restore (spatial localization), and (3) what to restore (degradation semantics). We design a lightweight decoding module that efficiently leverages these latent encoded cues for spatially-adaptive restoration. Extensive experiments across six common degradation tasks, five compound settings, and previously unseen degradations demonstrate that our method outperforms state-of-the-art (SOTA) approaches, achieving an average PSNR improvement of 1.68 dB while being three times more efficient. Code will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DAIR, a degradation aware all-in-one restoration approach which aims to tackle unknown degradations. Existing methods require human manual inputs, prompts or hand-crafted priors to provide degradation information to the model, which presents challenges when dealing with unseen degradations. DAIR is built to automatically tackle unseen and compound degradations with a 'which, where, what' approach. The authors propose latent prior encoding to identify 'which' features from the encoder are required for restoration. A degradation map is then constructed which dictates 'where' the model should restore followed by cross-modal fusion to determine the content to reconstruct. A decoder (3WD) uses this information to produce the restored output. DAIR achieves SOTA performance.

### Strengths
1. The paper is well written and easy to follow.
2. The paper tackles an important problem of existing all-in-one restoration approaches which is addressing unknown degradations.
3. The idea of pre-training a VAE to learn degradation representations is interesting.
4. DAIR achieves SOTA performance with significant boosts in performance.
5. The overall approach has many components whose importance is validated by an ablation study.

### Weaknesses
See questions for ways to address major weaknesses.

**Major**:

 1. Lack of experiments to demonstrate the core motivation of the architecture: The authors show the importance of their proposed modules in Table 6. However, this does not support the ‘which, where, what’ approach, i.e., the core motivation of the work.

 2. Lack of benchmarking on unseen task restoration: There are only experiments on unseen mixed degradations in the main paper, which does not support the claim in the abstract and introduction. 

 3. The YOLOv12 experiment is claimed as a contribution but has no other mention in the main paper. If it is to be claimed as a contribution, the experiment must be moved from the supplementary to the main paper.

**Minor**:
1. Citation style: Brackets are missing around the citations in many instances.
2. To the best of my knowledge, PromptIR does not use manual prompt and is blind. So, that needs to be changed throughout the paper.

### Questions
**More ablations (major weakness 1)**: 
1. Is there a benefit in using the luminance and chrominance encoders? How is performance affected if the VAE feature is directly used?
2. Can the authors provide experiments showcasing the ‘which’ and ‘what’ features? For instance, can the authors provide experiments to show that the output of LPE is ‘which’ features to use for decoding (and similarly for ‘what’)?
3. Can the authors provide experiments to confirm that the VAE captures degradation priors?

**Lack of benchmarking (major weakness 2)**: The real-world experiments on non-compound degradations need to be moved to the main paper. Additionally, can the authors provide quantitative comparisons with other methods for these tasks? Also, can the authors provide results on real mixed degradations (such as LOLBlur [1])?

**Fig. 5**: In Fig. 5, what features were used to plot the t-SNE?

[1] Zhou, Shangchen, Chongyi Li, and Chen Change Loy. "Lednet: Joint low-light enhancement and deblurring in the dark." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
DAIR reframes all‑in‑one image restoration (AIR) as learned latent prior inference. Guided by these learned priors, the method reasons about which features to route, where to restore, and what content to reconstruct (the “which–where–what” paradigm). Results illustrated in the manuscript seem good but limited.

### Strengths
1. Moving from prompts/architectural biases to latent prior inference gives a coherent, task‑agnostic design that maps naturally to which–where–what reasoning.
2. The 3WD module relies on element‑wise gating for linear (in spatial size) complexity, avoiding quadratic attention overhead. This yields favorable compute/param budgets in multi‑task settings.
3. Notable improvement in downstream detection (YOLOv12‑L), with 34.9 mAP, supporting practical impact beyond pixel metrics.

### Weaknesses
1. The experimental setting is not standard compared with other all-in-one methods. The authors shall add additional two settings in Promptir or Perceive-ir.
2. The main fig for the whole framework is bad and confusing. For example, I can't figure out where is the input for degradation map in (c) referred in every stage of (a).
3. The questions of 'which, where and what' is good but lacks some convincing proofs or experimental results. It makes the manuscript more like a story made on the framework rather than solving key problems with insights.
4. As is stated by the authors that the proposed method do not rely on textual or degradation prompts, the loss of SupCon for VAE pretraining actually involves degradation label, which means the method still need supervision from degradation information.

### Questions
1. The same as weakness1, the compared method like AdaIR do the different settings, and why the authors choose a unique setting. Have doubts on the performance of normal experimental settings.
2. Typos in line 277-278 shall be fixed.
3. The training is conducted on a single Nvidia 3060 with a large scale of data. I have doubt on the training cost like day length for that.
4. Further studies would help: frequency cues inside degradation map, LC luminance‑only vs. luminance+chrominance, and latent fusion variants.
5. Why do the authors choose desnowing and low-light enhancement for single tasks? What about other single tasks?

### Soundness
3

### Presentation
2

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
The paper proposes DAIR, a degradation-aware, all-in-one image restoration framework that eliminates manual prompts and hand-crafted priors. It reframes AIR as latent prior inference: a VAE-style encoder infers multi-scale latent codes and a global descriptor from the degraded image; a learnable degradation map (DM) tells the network where to restore; a µ-guided latent fusion decides what to restore; and a lightweight 3WD (which–where–what decoding) module performs spatially adaptive reconstruction with claimed linear complexity.

### Strengths
1. Clear problem reframing: Casting AIR as latent prior inference is a clean, unified alternative to prompt-driven or frequency-heuristic approaches.
2. Well-structured reasoning design: The which–where–what decomposition (latent-prior selection, degradation maps, and µ-conditioned fusion) is conceptually coherent and easy to follow.
3. Efficiency claims: 3WD replaces quadratic attention with element-wise modulation, yielding linear-in-pixels complexity and large reported speed/memory savings.

### Weaknesses
1. The conceptual gap to earlier non-prompt AIR (e.g., contrastive degradation embeddings, frequency mining) could be articulated and empirically isolated more sharply; the VAE-based latent prior may read as an incremental integration of known ideas.
2. The encoder is pre-trained on mixtures (including some compound degradations) and then frozen. How this compares, apples-to-apples, to baselines’ pre-training or data exposure is unclear; fairness details (data volume, augmentations, early stopping) should be expanded.
3. The complexity discussion focuses on the inner attention form. End-to-end wall-clock and memory benchmarks (same HW, resolution, batch size) vs. SA/CA across multiple image sizes would better substantiate the claimed 85–257× speedups.
4.  “Unseen” tests are limited to two combos (haze+snow, low-light+rain). It would help to include more OOD types (e.g., underwater, compression artifacts, lens flare) and real-capture datasets to support robustness claims.

### Questions
See Weaknesses.

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
4

### Summary
The paper proposes DAIR, a unified framework that reframes all-in-one restoration as latent prior inference rather than prompt-following or hand-crafted inductive biases. A VAE-style latent prior encoder infers multi-scale degradation codes directly from a degraded image; these codes drive a three-part “which–where–what” reasoning pipeline: (i) which features to route (latent-prior-modulated encoders), (ii) where to restore (a learnable degradation map combining spatial features with FFT cues), and (iii) what to reconstruct (a µ-guided latent fusion that adaptively scales/shifts structural and chromatic branches). A lightweight 3WD decoder performs element-wise, degradation-guided attention with linear complexity.

### Strengths
1. Clear which–where–what decomposition and a linear-time 3WD decoding mechanism; thoughtful integration of luminance/chrominance branches plus FFT-based cues for degradation localization. 
2. Strong empirical coverage: common, compound, and unseen degradations; consistent average PSNR/SSIM gains; tangible downstream benefits on object detection.
3. Solid ablations (latent priors, DM, µ-fusion, 3WD) and efficiency analysis (FLOPs, linear attention vs. SA/CA) that support the design choices.

### Weaknesses
1. Many benchmarks are synthetic (e.g., SOTS, Rain100L/CCD, compound datasets). Real-capture datasets (wild rain/snow/low-light, RAW/ISP pipelines) are underrepresented; robustness to sensor noise/ISP and compression artifacts remains unclear.
2. Heavy reliance on PSNR/SSIM; limited perceptual/user-study evidence. Including LPIPS, MUSIQ/NIQE, or human preference tests would better validate perceptual gains and avoid PSNR-overfitting.
3. While prior AIR (PromptIR/ADAIR/DFPIR/DiffUIR) are covered, results against task-specialized SOTA per degradation (e.g., latest LLIE, dehaze, derain, deblur transformers/diffusions) are not uniformly comprehensive on each dataset split. 
4. The encoder pretraining mixes single/compound degradations, then is frozen; risk of domain overfitting or latent collapse is only qualitatively addressed. Quantitative analyses of latent separability/stability across seeds and cameras would help.
5. Degradation maps are compelling but lack quantitative localization validation (e.g., overlap with synthetic corruption masks, perturbation or pointing-game metrics). Similarly, how µ-fusion decisions correlate with true degradation semantics is not rigorously measured.
6. Linear 3WD is attractive, but end-to-end latency on 2–8K images, memory usage under tiling, and comparison to windowed attention are not reported.

### Questions
1. Do latent priors transfer to real-capture datasets (e.g., LLIE in the wild, real rain/snow) and RAW pipelines? Any results on smartphone benchmarks or cross-sensor generalization without finetuning?
2. How stable are the learned {xℓ, µ} across runs/seeds? Provide cluster separability (t-SNE/UMAP) with degradation labels, NMI/ARI, and center/variance over seeds; any evidence of latent collapse under heavy compound degradations?
3. On synthetic corruptions with known masks, what is the IoU/ROC of DM against ground truth per degradation (rain streaks, haze veil, snow flakes, low-light masks)? Does DM remain calibrated under unseen mixes?
4. Sensitivity to FFT features (remove magnitude/phase), LC branching (luma-only vs chroma-only), FiLM parameterization (γ/β forms), and temperature/β in VAE loss? Please include variance across 3–5 runs.
5. Could you report LPIPS/MUSIQ or a user study on a subset? Any failure cases (e.g., color shifts, ringing, detail hallucination) compared to PromptIR/ADAIR?

### Soundness
3

### Presentation
3

### Contribution
2
