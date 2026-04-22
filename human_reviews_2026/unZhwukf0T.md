# CAT-VIDEO: CORRUPTION-AWARE TRAINING FOR ROBUST VIDEO DIFFUSION MODELS

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Latent Video Diffusion Models (LVDMs) have achieved state-of-the-art generative quality for image and video generation; however, they remain brittle under noisy conditioning, where small perturbations in text or multimodal embeddings can cascade over timesteps and cause semantic drift. Existing corruption strategies from image diffusion (Gaussian, Uniform) fail in video settings because static noise disrupts temporal fidelity. In this paper, we propose **CAT-Video**, a corruption-aware training framework with structured, data-aligned noise injection tailored for video diffusion. Our two operators—*Batch-Centered Noise Injection (BCNI)* and *Spectrum-Aware Contextual Noise (SACN)*—align perturbations with batch semantics or spectral dynamics to preserve coherence. CAT-Video yields substantial gains: BCNI reduces FVD by **31.9%** on WebVid-2M, MSR-VTT, and MSVD, while SACN improves UCF-101 by **12.3%**, outperforming Gaussian, Uniform, and even large diffusion baselines like DEMO (2.3B) and Lavie (3B) despite training on $\mathbf{5}\times$ less data. Ablations confirm the unique value of low-rank, data-aligned noise, and theory establishes why these operators tighten robustness and generalization bounds. CAT-Video thus sets a new framework for robust video diffusion, and our experiments show that it can also be extended to autoregressive generation and multimodal video understanding LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CAT-LVDM, a corruption-aware training framework for Latent Video Diffusion Models designed to improve robustness against imperfect or noisy text prompts. The core of the framework consists of two structured noise injection strategies:

1. BCNI: Perturbs text embeddings along intra-batch semantic directions to increase entropy while maintaining semantic alignment.

2. SACN: Injects noise along dominant spectral modes (via SVD) to enhance low-frequency smoothness and temporal coherence.

The authors provide theoretical justification for both methods, analyzing conditional entropy and Wasserstein distance bounds. Experiments are conducted on several datasets (WebVid-2M, MSR-VTT, MSVD, UCF-101), where the proposed methods demonstrate quantitative improvements (e.g., reduced FVD) compared to uncorrupted baselines and simpler noise injection techniques.

### Strengths
1. The paper provides a theoretical analysis for its proposed methods (BCNI and SACN)

2. The paper addresses the practical problem of training video models on large-scale, noisy web data. The idea of corruption-aware training for video diffusion is considered interesting and worthwhile.

3. The method demonstrates clear quantitative performance gains (e.g., FVD, SSIM, PSNR) over uncorrupted baselines and naive (Gaussian/Uniform) noise baselines on the tested datasets.

4. The paper is generally well-written and easy to follow.

### Weaknesses
1. A major concern is that the method was only validated on an older LVDM architecture (DEMO). Its effectiveness and scalability on modern, state-of-the-art models (e.g., DiT-based) are unproven

2. Despite quantitative gains, the practical impact is undermined by the generated videos. the visual results not unsatisfactory. 

3. he paper lacks a clear justification for applying BCNI primarily to caption-rich datasets (WebVid-2M, MSR-VTT) and SACN only to a class-labeled dataset (UCF-101). 

4. The evaluation is missing key comparisons. It fails to compare against stronger, modern LVDM baselines and does not empirically validate the necessity of "video-specific" corruption. 

5. The core idea of perturbing conditions to improve robustness has been explored in the image domain, limiting the paper's technical novelty.

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces CAT-Video, a training framework that improves video generation models' resilience to imperfect inputs. The authors address a key limitation of standard noise-injection techniques, which often disrupt video coherence. Their solution involves two novel methods: Batch-Centered Noise Injection (BCNI) to maintain semantic consistency within a batch, and Spectrum-Aware Contextual Noise (SACN) to preserve smooth temporal dynamics.

### Strengths
​​+  a New Problem​​: It is one of the first to systematically address how to make video AI models robust to imperfect instructions, focusing on a key weakness in existing methods that break video coherence.

​​+  Its two new techniques, BCNI and SACN, are simple and add almost no extra cost, yet outperform much larger and more expensive models.

​​+ Extensive testing across different datasets and metrics proves the methods reliably enhance video quality and motion.

### Weaknesses
- Is this problem meaningful in the research field?  In what kind of scenerios, noise will introduced into the input? Does the model robust for  adversarial attack? 
- Does the proposed model work for large video models?

### Questions
Please refer to the strengths and weaknesses

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
3

### Summary
This paper introduces CAT-Video, a corruption-aware training framework that improves robustness in latent video diffusion models (LVDMs) under noisy or imperfect conditioning.
The core contribution lies in two structured corruption strategies: Batch-Centered Noise Injection (BCNI), which injects noise along the deviation from the batch mean, and Spectrum-Aware Contextual Noise (SACN), which perturbs only the low-frequency spectral components of conditioning embeddings. Theoretical analysis supports tighter generalization bounds and improved temporal fidelity, and extensive experiments across multiple benchmarks demonstrate empirical gains over conventional Gaussian and Uniform corruption techniques as well as large-scale diffusion models.

### Strengths
1.	The paper presents a clear exploration of corruption-aware training for video diffusion, supported by solid theoretical derivations that extend existing analyses to the temporal setting.
2.	Extensive experiments across four standard benchmarks and additional evaluations on autoregressive and multimodal models confirm broad empirical robustness.
3.	Ablation studies, sensitivity analyses, and detailed appendices ensure reproducibility; released code and metrics further strengthen the work’s transparency.

### Weaknesses
1.	Although the authors claim model-agnostic applicability, most analyses are conducted on the DEMO backbone with OpenCLIP-based encoders. This limits the universality claim, while evaluating BCNI/SACN on one or two additional diffusion backbones would strengthen the evidence.
2.	While CAT-Video demonstrates strong robustness on standard datasets, further evaluation under more realistic, noisy, or weakly aligned conditions would better reflect its practical robustness.
3.	Related work on corruption-aware methods in image diffusion and other noisy-input domains is not sufficiently discussed. This section primarily centers on LVDMs, which are not the main focus of this work.
4.	Additional qualitative visualizations would help clarify how CAT-Video improves visual coherence compared with other corruption strategies.

### Questions
1.	Would combining multiple corruption techniques during training (e.g., applying BCNI followed by SACN) be beneficial, or would such composite perturbations destabilize the training process? Some insight into this interaction could be valuable for readers.
2.	How are subsets of training data (e.g., 2 M vs. 10 M in the original DEMO paper) selected? Are they random samples or curated based on specific criteria?
3.	For very long or high-resolution videos, does computing BCNI or SACN introduce significant additional computational overhead during training?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes CAT-Video, a corruption-aware training framework for latent video diffusion models (LVDMs). It introduces two structured, low-rank embedding perturbations: Batch-Centered Noise Injection (BCNI) and Spectrum-Aware Contextual Noise (SACN). The idea is to inject data-aligned noise during training to improve robustness to noisy text/multimodal conditioning and reduce semantic drift across timesteps. The method is supported by a theoretical sketch (entropy/Wasserstein/score-drift bounds) and experiments on WebVid-2M, MSR-VTT, MSVD, and UCF-101, with reported FVD gains over Gaussian/Uniform corruption and some large diffusion baselines; there are also brief extensions to autoregressive models and multimodal video understanding

### Strengths
Robustness of T2V models to noisy/ambiguous conditioning is important and under-tested. The paper articulates temporal error accumulation in diffusion and motivates structured corruption specifically for video.

BCNI perturbs along deviations from batch mean; SACN perturbs along principal spectral modes with exponentially decayed variances. Both add minimal overhead and have a single scale hyper-parameter.

### Weaknesses
1. Novelty is limited relative to prior “noisy/structured conditioning” literature; positioning is dated.
Injecting noise into conditioning embeddings (Gaussian/Uniform, token-level swaps/replacements) and exploiting structure/low-rank directions have been explored extensively in image diffusion and multimodal finetuning (e.g., corruption-aware pretraining, token-level perturbations, NEFTune-style noisy embeddings). The paper cites several such works but doesn’t clearly differentiate BCNI/SACN as more than “reasonable engineering variants” adapted to video. A crisper technical delta vs. structured conditioning noise in recent video works is missing. Suggest: provide head-to-head against stronger structured baselines, not only isotropic Gaussian/Uniform or simple temporal gradients (TANI/HSCAN as defined). For instance, compare to learned/adaptive corruption schedules, prompt-space adversarial perturbations with temporal regularizers, or curriculum-style noise aligned to caption syntax/scene cuts

2. Experimental scope feels behind current 2025-26 standards; datasets/architectures and metrics are narrow.
Core results rely on WebVid-2M/MSR-VTT/MSVD/UCF-101 with FVD-centric reporting. Modern T2V evaluation has moved toward longer videos, higher resolutions, compositional control, and stronger perception-aligned metrics (e.g., VBench subsets with motion/physics consistency, human studies with calibrated protocols). The paper mentions VBench/EvalCrafter in passing, but full tables are pushed to the appendix; the main text should surface these with stronger analysis and significance tests. Also, many baselines listed are from earlier generations of models; direct comparisons to current-gen LVDMs/DiT-style video transformers trained at scale are missing. Actionable:
Add long-horizon (≥10–16 s) and high-res evaluations;
Include modern SOTA baselines trained on 2024–2026 corpora (and/or reproduce baseline training with matched compute);
Report human preference and motion/consistency metrics prominently in main text.

3. Claims vs. baselines are hard to validate due to scale mismatch and unclear fairness.
Several “beats larger models with 5× less data” statements are made, but the compared methods differ in architecture, parameterization, pretraining recipes, and dataset curation. Without matched compute / strong-to-strong comparisons (e.g., same backbone with/without CAT; or retrained SOTA with the authors’ code), it’s difficult to attribute gains to BCNI/SACN rather than other confounders. Please (i) include paired-control runs on the same modern backbone at matched data/compute; (ii) show scaling curves (data, steps, guidance, steps vs. FVD) for clean vs. CAT; (iii) report statistical significance (mean±std is in the appendix, but main-text needs tests across seeds).

4. Theory is mostly appendix-level and not tightly coupled to practice.
The paper sketches entropy/Wasserstein/score-drift bounds and a D/d “complexity gap,” but practical instantiations (how d is chosen/estimated online, how SVD in SACN scales with sequence length, and why the assumed low-frequency dominance universally holds) are not convincingly validated. The main text should tie specific theorems to measurable proxies (e.g., Lipschitz estimates, score-norm smoothness, sensitivity slopes) and show correlations across datasets/noise levels. Provide ablations on rank d, spectral weighting schedules, and wall-clock overhead broken down by operator.

5. The multimodal video understanding experiment (AVSD) is conducted at 0.5B scale only (LLaVA-OV-0.5B-FT and PAVE-0.5B). By ICLR-26 standards, this falls short of contemporary MLLM practice (7B/13B and stronger video-language backbones). For a credible “model-agnostic” claim, please include ≥7B variants (e.g., PAVE-7B/13B or comparable video-LLaVA baselines

6. The paper claims “we also validate scalability by extending CAT to autoregressive video generation (NOVA)…,” but NOVA is not tabulated in the main paper. Table 3(a) discusses scalability to AR using MAGVIT/CogVideo as references, while NOVA-specific results are not shown; the authors point to Appendix Tables 14–16, yet Table 14 aggregates AR results by corruption type (BCNI/SACN/Gaussian/Uniform) without a clear NOVA row, making the claim hard to verify from the main text. Please provide an explicit NOVA baseline line with matched settings (params, data, steps) and report AR scaling curves (data/compute vs. FVD) to substantiate “model-agnostic” robustness.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
