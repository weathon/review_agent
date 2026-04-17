# StyleAT: Defending Face Recognition Against Semantic Attacks

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
With face-recognition models now embedded in everyday authentication and surveillance, recent works have pinpointed a critical weakness: these models remain acutely vulnerable to adversarial semantic edits. I.e., adversarially produced semantic alterations to the input, such as slight aging or pose changes, can induce misclassifications. Certain existing attacks are powerful, but they can be computationally costly, rendering them inadequate for developing defenses (e.g., through adversarial training). To fill the gap, we introduce BoundStyle, a potent semantic attack operating in StyleGAN’s rich latent space to maximize misclassification rates. Notably, BoundStyle is significantly more efficient than equally powerful attacks, making it suitable for adversarial training. Building on BoundStyle, we develop StyleAT, an efficient adversarial training scheme that incorporates low-budget attack variants yet defends against stronger and unseen semantic attacks. We evaluate on two datasets unseen during training (LFW and VGG-Face) and five models, and find that StyleAT boosts robust accuracy against state-of-the-art attacks (DiffPrivate and BoundStyle) and outperforms common defenses (DOA and classical filters) in various settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces BOUNDSTYLE, a semantic adversarial attack that manipulates StyleGAN3 latent codes to generate perturbed face images capable of evading face recognition (FR) systems while preserving perceptual identity for human observers. Additionally, the authors propose STYLEAT, an adversarial training framework that incorporates three distinct triplet losses computed on clean images, BOUNDSTYLE-edited samples, and PGD pixel-space perturbations to enhance model robustness against semantic attacks. Experimental evaluation across five FR architectures and two benchmark datasets demonstrates that BOUNDSTYLE exhibits strong transferability across different recognizers while achieving approximately 9.5× faster generation speed compared to DiffPrivate.

### Strengths
The paper presents BOUNDSTYLE, an efficient StyleGAN3 latent-space attack that is approximately 9.5× faster than DiffPrivate, enabling practical adversarial training at scale. 

Building on this efficiency, the authors propose STYLEAT, which combines clean, semantic-attack, and PGD pixel-level triplet losses to improve robustness across five face recognition models and two unseen datasets.

### Weaknesses
1. The paper uses “semantics” as edits to expression/age/accessories with identity preserved, but provides no quantitative disentanglement or taxonomy of which semantic factors are actually being manipulated and protected.
2. At line 62, the authors claim that BOUNDSTYLE produces only semantic edits; however, the paper lacks both qualitative visualizations and quantitative validation to substantiate this claim.
3. This work represents an incremental contribution with limited novelty. The methodology is quite unclear, and without proper architecture, it is very difficult to follow. A complete end-to-end pipeline might help.
4. The attack objective (Section 4.1) uses the recognizer on preprocessed crops of generated images, i.e., F(C(G(l+δ)). Applying crop/alignment after editing may itself alter pixel evidence and confound what counts as a “semantic” change.
5. All target FR backbones are classic CNNs with one Swin-T. No recent SOTA recognizers (e.g., MagFace, AdaFace) are included, weakening robustness and generalizability claims.
6. The comparison presented is methodologically flawed, as it evaluates results using β = 3 (BOUNDSTYLE or STYLEAT) against z=6 for DiffPrivate. This creates an unfair benchmark since DiffPrivate demonstrates superior robustness at z=3, which would be a more appropriate comparison point for evaluating the proposed approaches.
7. The paper does not showcase statistical testing.
8. The paper claims high visual fidelity through inversion and pivotal tuning, asserting that edits appear unchanged to human observers. However, no quantitative perceptual metrics (SSIM, PSNR, LPIPS) were provided, nor were any human evaluation studies conducted to support these claims.

### Questions
(1) How does the paper quantitatively measure and verify that BOUNDSTYLE manipulates only semantic attributes (expression, age, accessories) while preserving identity? Can authors provide a systematic taxonomy or disentanglement analysis of which semantic factors are being modified and protected?

(2) Regarding the claim at line 62 that BOUNDSTYLE produces only semantic edits, can authors provide any qualitative visualizations or quantitative metrics to validate this assertion?

(3) Could authors provide a comprehensive end-to-end architectural diagram or pipeline illustration to clarify the methodology? The current presentation makes it difficult to understand the complete workflow and implementation details.

(4) In Section 4.1, the attack objective applies face recognition to preprocessed crops of generated images (F(C(G(l+δ)))). How does this account for the fact that post-generation cropping and alignment may introduce pixel-level artifacts that confound the distinction between semantic and identity-related changes?

(5) The proposed method should be validated against recent state-of-the-art face recognizers such as MagFace and AdaFace. How would their inclusion affect the robustness and generalizability claims, given that the current evaluation relies primarily on classic CNN architectures with only one Transformer model (Swin-T)?

(6)  Can the author justify the choice of comparing β=3 for your methods against z=6 for DiffPrivate? Since DiffPrivate demonstrates superior robustness at z=3, wouldn't this be a fairer and appropriate baseline for comparison?

(7) The paper should include statistical significance tests to validate that the reported performance improvements are statistically meaningful rather than within noise margins. 

(8) The claim of high visual fidelity asserts that edits appear unchanged to human observers. Can the author provide quantitative perceptual quality metrics (SSIM, PSNR, LPIPS) and conduct human evaluation studies to substantiate these claims?

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
3

### Summary
This paper studies semantic adversarial attacks on face recognition. It introduces a new attack called **BOUNDSTYLE**, which operates in the latent space of StyleGAN3 to produce realistic facial edits that fool recognition systems.  BOUNDSTYLE is efficient, tunable, and achieves high attack success with realistic outputs.  Building on it, the authors propose **STYLEAT**, an adversarial training method that uses fast BOUNDSTYLE variants to improve robustness.  STYLEAT combines latent-space semantic attacks and small pixel-space perturbations during training.  Experiments on **LFW** and **VGG-Face** show that BOUNDSTYLE is about 9.5× faster than the diffusion-based DiffPrivate.  The paper claims that STYLEAT is the first defense specifically designed for general semantic attacks on face recognition.  The experiments are carried out on two datasets (LFW and VGG-Face) using pre-trained face recognition backbones.

### Strengths
-   **Clear methodological contribution:** The paper introduces both a new semantic attack (BOUNDSTYLE) and a corresponding defense (STYLEAT), forming a coherent attack–defense framework.
    
-   **Practical efficiency:** BOUNDSTYLE is nearly an order of magnitude faster than prior semantic attacks, making it feasible for adversarial training and large-scale robustness evaluation

### Weaknesses
-   **No high-level block diagram of the pipeline.**  
    The paper provides a clear pseudocode (Alg. 1 in the appendix) but lacks a single illustration or block diagram showing the end-to-end attack → defense training pipeline, which makes the method harder to digest quickly.
    
-   **No analytic mapping between attack budgets in different latent spaces.**  
    BOUNDSTYLE’s budget (β in StyleGAN3 latent space) and DiffPrivate’s budget (‖Δz‖ in diffusion latent space) are treated as separate knobs without a principled conversion. The paper therefore compares attacks empirically but provides no theory or metric to equate perceptual magnitudes across the two spaces.
    
-   **Transferability claim is overstated relative to absolute potency.**  
    BOUNDSTYLE shows _good transferability_ (off-diagonals close to diagonals), but DiffPrivate drives _lower absolute_ robust accuracies in some settings (notably VGG-Face), meaning DiffPrivate can be a more destructive white-box attack even if it transfers less well. The paper reports both effects but leans on transferability as a strength without emphasizing the potency tradeoff enough.
    
-   **Baselines, datasets, and backbone coverage are limited.**  
    DOA and simple filters are not fully representative baselines for latent-space semantic attacks (DOA targets ad-hoc physical accessories), and STYLEAT is adversarially trained with BOUNDSTYLE, which naturally advantages it against that family of edits. The experimental scope is also narrow: evaluation is on two datasets (LFW, VGG-Face) using pre-trained backbones, without experiments on larger or cleaner FR datasets (e.g., VGGFace2, Glint360K) or broader transformer backbones. These choices weaken claims of broad generalization.

### Questions
-   **Computation claim:** Could the authors provide a FLOP or compute-based analysis to support the reported 9.5× speedup?
    
-   **Other concerns:** Please refer to the weaknesses section for additional points regarding clarity, evaluation setup, and baseline choices.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the issue of face recognition (FR) systems being vulnerable to semantic attacks (e.g., changes in age or pose). Existing attacks are either computationally costly or have limited effectiveness, making it difficult to develop effective defenses. Furthermore, there is a lack of established defenses against general semantic attacks.
To address these issues, this paper presents two main contributions:
BOUNDSTYLE: A novel and efficient semantic attack method that operates in StyleGAN's latent space. Experiments show that while maintaining a high attack success rate, it is approximately 9.5 times faster than state-of-the-art (SOTA) attacks like DiffPrivate , making it well-suited for adversarial training.
STYLEAT: An adversarial training defense scheme based on BOUNDSTYLE. This scheme enhances model robustness using a low-cost variant of BOUNDSTYLE , and it also integrates a component to defend against pixel-level perturbations.
The experimental evaluation was conducted on two datasets (LFW and VGG-Face) and five FR models. The results show that STYLEAT significantly improves the model's robust accuracy against SOTA attacks (including DiffPrivate, which was unseen during training), with up to a 46.3% increase in the gray-box setting, for example , and it outperforms existing defense methods (such as DOA and standard filters)。

### Strengths
This paper's strengths lie in its dual contribution: it introduces both an efficient, powerful semantic attack (BOUNDSTYLE) and an effective, generalizable defense (STYLEAT). A significant advantage is its solution to the feasibility problem of adversarial training; BOUNDSTYLE's high efficiency (about 9.5 times faster than DiffPrivate ) directly overcomes the high computational cost that previously made such defenses impractical. The STYLEAT defense demonstrates strong efficacy and generalization, proving effective not only against the BOUNDSTYLE attack used in its training but also generalizing to unseen SOTA attacks like DiffPrivate, with robust accuracy improving up to 28.6% 6and 46.3%, respectively. Moreover, BOUNDSTYLE itself is a potent attack, matching DiffPrivate in white-box scenarios while exhibiting clearly superior black-box transferability. Finally, the paper provides a key design insight by incorporating LAdvPix, based on the observation that some attacks like DiffPrivate may include imperceptible pixel perturbations , a hypothesis confirmed by ablation studies.

### Weaknesses
1.Limitation of Defense Evaluation: Although the attack (BOUNDSTYLE) was tested on 5 different FR backbones 1, the defense's (STYLEAT) effectiveness appears to be evaluated only on the ResNet model. 
2.Empirical Defense: As the authors acknowledge in the conclusion, STYLEAT is an empirical defense and does not provide provable security guarantees8。

### Questions
1. Why was the STYLEAT defense only evaluated on the ResNet backbone ? Given that the attack tests covered multiple architectures (such as SwinT), showing STYLEAT's defensive effectiveness on these different architectures (particularly its cross-architecture generalization) would make the paper's conclusions more convincing？

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a  StyleGAN3-latent, tunable semantic attack on face recognition (BOUNDSTYLE) and an adversarial-training scheme that mixes fast BOUNDSTYLE variants with light PGD in pixel space to improve robustness against general semantic attacks (STYLEAT).  Experiments span five FR backbone networks and are evaluated on two well-known face recognition datasets. The results demonstrate BOUNDSTYLE exceeds/matches a norm-bounded baseline by about 10x faster. STYLEAT achieves better white-box robustness.

### Strengths
1. This paper focuses on semantic attacks, which are more practical than norm-bounded noise. It launches tunable and fast attacks and achieves about 10x faster.

2. Proposed defense improves robustness to unseen attack in white- and gray-box settings. Black-box heatmaps show proposed attacking methods transfer across architectures more than the baseline.

### Weaknesses
1. Robustness is reported only on LFW and VGG-Face with limited positive pairs at a fixed FPR target; these are aging, small-scale verification sets. No tests on harder, modern FR benchmarks (e.g., IJB-C/IJB-S, MegaFace reruns) or large-scale ID/verification datasets. 
2. The proposed attack - STYLEAT is applied to ResNet only; robustness gains may not generalize to other strong backbones (e.g., ArcFace-like ResNet100, ViT-based SOTA). 
3. The paper qualitatively limits β/‖Δz‖ to “preserve identity,” but provides no human study or perceptual metric (e.g., face-ID match rate against a separate high-accuracy verifier, or MOS/LPIPS) to confirm “same person” under edits. 
4. Baseline DiffPrivate is evaluated as a norm-bounded variant with capped iterations (∥Δz∥≤6, 70 iters), which may understate its best-known strength; there’s no “best-effort” setting matching wall-clock or quality across methods.
5. Results emphasize robust accuracy at a fixed FPR on small test sets; no ROC tradeoffs, no calibration robustness, and no analysis under different thresholds or across demographics/poses.  The 9.5× speedup is measured with batch=1 on a single A6000, without reporting attack convergence/tuning across GPUs or batched/mixed-precision settings; unclear generality. STYLEAT includes pixel-space PGD partly to counter “imperceptible perturbations” in DiffPrivate, inferred via filter sensitivity. This is indirect evidence; more direct spectral/energy analyses would strengthen the claim.

### Questions
1. (Important) How do you quantify that BOUNDSTYLE edits preserve identity? Please report verification rates against a strong external FR (not used in training or thresholding), plus perceptual metrics (LPIPS/FID) or a small human study stratified by β. 

2. Could the authors add a matched wall-clock comparison (equal time budgets) and a best-effort comparison (no norm cap; tuned steps) for DiffPrivate vs. BOUNDSTYLE, to remove confounds from ∥Δz∥ bounds and iteration caps? Do the robustness gains transfer when defending SwinT/MobileFace/RepVGG (and newer ArcFace-style models)? Any signs of robust overfitting across backbones or datasets?

3. How do results change if you (a) vary the FPR target (e.g., 1e-3, 1e-5), (b) recalibrate post-training, and (c) plot full ROC/DET curves under attack?   Can the authors report results on harder, modern FR benchmarks (IJB-C/S, CFP-FP, AgeDB, CALFW, CPLFW) and larger positive/negative sets to reduce variance?  

4. Beyond filter sensitivity, do the authors have frequency-domain or gradient-alignment analyses showing DiffPrivate introduces non-semantic high-frequency components, and that STYLEAT’s pixel PGD directly targets them? Or if this is inapplicable to the current method?

5. Please ablate T, β, α for BOUNDSTYLE during training vs testing, report compute/accuracy trade-offs, and show whether per-sample random starts are essential. Inversion and pivotal tuning are often brittle. Which encoder, learning rates, and stopping criteria were used? What cache reuse ratio and wall-clock overhead do they add to training?

### Soundness
2

### Presentation
3

### Contribution
2
