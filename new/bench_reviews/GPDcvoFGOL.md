Now let me check the calibration papers referenced by the Human Finder reviewer.Now I have enough context to write the consolidated review. Let me verify the specific claim about Eq. (5) and multi-layer propagation before finalizing.Now I have all the information I need. Let me write the consolidated review.

---

## Summary
The paper introduces a "second-order lens" for interpreting individual MLP neurons in CLIP-ViT by analyzing how each neuron's residual-stream contribution propagates through subsequent attention layers (value pathways) to the output representation. The authors characterize these second-order effects as: (1) concentrated in late layers, (2) sparse—significant for fewer than 2% of images per neuron—and (3) approximately rank-1, enabling a single direction per neuron to be decomposed via sparse coding into text descriptions that reveal polysemantic behavior. Two downstream applications are demonstrated: automated generation of semantic adversarial examples exploiting spurious neuron polysemy, and zero-shot segmentation surpassing prior work.

---

## Strengths

- **Novel and well-motivated interpretability lens.** The paper makes a convincing case that neither direct (first-order) effects (near-constant in CLIP) nor indirect (ablation) effects (obscured by self-repair) are suitable for interpreting MLP neurons. The second-order lens addresses both limitations simultaneously, and Table 1 provides clear empirical support: mean-ablating second-order effects drops ImageNet accuracy to 29.6% vs. 52.3% for indirect effects, and PC#1 explains 48.2% of variance vs. 11.0% for indirect effects.

- **Strong empirical characterization.** The sparsity finding (significant for < 2% of images per neuron, confirmed in Figure 3 "w/o large/small norm" conditions) and the approximate rank-1 structure (Figure 3 "rec. from PC #1" matching baseline accuracy nearly perfectly) are systematically established and together motivate the sparse text decomposition approach.

- **Creative adversarial application.** Exploiting polysemantic neuron structure to generate semantic adversarial images is genuinely novel. Table 3 shows clear and consistent outperformance over all baselines, including the especially difficult "ship→truck" task where the only non-zero success rate is the proposed method (5.7 vs. 0.0 for all baselines).

- **Strong zero-shot segmentation results.** Table 4 shows consistent improvement over all prior methods including the authors' own TextSpan across all three metrics (Pix. Acc., mIoU, mAP). Qualitative results (Figure 7) visually confirm more complete object coverage.

- **Clear derivation and presentation.** The mathematical derivation of Eq. (5) from the residual stream perspective (following Elhage et al. 2021) is clean, and the paper honestly enumerates its limitations (query/key effects, neuron-neuron interactions, attack pipeline failure modes) in Section 6.

---

## Weaknesses

### Fatal
*None.*

### Major

- **The quantitative validation does not directly confirm semantic faithfulness of neuron descriptions.** The paper's central interpretability claim is that the sparse text decompositions reveal *what a neuron is doing*. The introduction states "we show that these concepts correctly track which inputs activate a given neuron (Section 4)," but Section 4's actual validation metric is ImageNet accuracy after replacing all neurons' second-order effects with their sparse text reconstructions. This measures the *approximation quality* of the representation, not whether the recovered text labels are faithful descriptions of individual neuron function. The qualitative evidence (Figure 5, Table 2) is compelling and self-consistent, but an activation-prediction experiment—e.g., testing whether the top decomposed concepts predict which images yield large-norm second-order effects—would provide direct evidence for the semantic interpretation claim.

- **The rank-1 claim is partially undermined by the 48.2% variance figure.** The paper observes empirically that PC#1 explains 48.2% of variance—meaning over half the variation is not captured. The term "approximately rank-1" requires the reader to accept this on faith without seeing how PC#2 and PC#3 contribute, and without a threshold analysis for when the approximation breaks. The downstream accuracy test recovers well, but accuracy is a coarse metric that could mask per-neuron failures in the approximation. Reporting the cumulative variance explained by top-k components (not just PC#1) and whether rank-1 quality differs between early vs. late neurons would make this claim substantially more credible.

### Minor

- **Adversarial success rates are modest in absolute terms and depend on manual curation.** Absolute rates range from 5.3% to 22.7%, and the pipeline includes manual removal of images that fail to depict the required objects. While the relative improvement over baselines is large and the limitation is honestly disclosed in Section 5.1, the abstract's framing of "mass production" of adversarial examples is somewhat stronger than the evidence supports given the filtering step and narrow scope (5 binary CIFAR-10 class pairs). Pre- and post-filtering success rates should be reported, and the attack's scalability to multi-class settings remains unshown.

- **Segmentation evaluation does not isolate the second-order contribution.** The segmentation method selects neurons by `|⟨r_n^l, M_text(c_i)⟩|`, uses the second-order direction `r_n^l`, and averages spatial activation maps. The paper compares against TextSpan (first-order directions) but does not explicitly ablate the neuron-selection step using first-order directions while keeping the rest of the pipeline constant. As a result, it is unclear whether the improvement comes from the second-order directions specifically, from using raw activation maps (vs. TextSpan's head-level decomposition), or from both.

- **No analysis of what fraction of neurons are interpretable.** The paper shows 4 hand-picked neurons with clean decompositions (Table 2, Figure 5). What fraction of the ~3,072 neurons per layer yield semantically meaningful, non-noisy decompositions? Without this, the claimed scalability of the approach is asserted but not demonstrated. Showing failure cases alongside successes is standard practice in interpretability work.

- **Generalizability is limited to CLIP-ViT.** The derivation in Eq. (5) and the decomposition via CLIP's shared text-image space are both CLIP-specific. The paper presents results on ViT-B-32 and ViT-L-14 (appendix) and acknowledges OpenAI variants only. Whether the second-order lens extends to other vision-language models or standard ViTs is unexplored.

### Trivial

- The selectivity "<2% of images" is based on a norm threshold whose choice is not fully specified in the main text. The Harsh Critic notes that "a random class-conditional neuron would also appear selective on ImageNet" but this doesn't undermine the authors' findings since they are measuring actual norm distributions. Still, reporting the threshold explicitly would improve reproducibility.

- The segmentation threshold of 0.5 is applied without tuning or justification, and sensitivity to hyperparameters (number of top neurons, layer range) is not characterized in the main paper.

---

## Nice-to-Haves

- **Comparison to Sparse Autoencoder (SAE) baselines.** SAEs (cited: Bricken et al. 2023; Rajamanoharan et al. 2024) are the dominant approach for decomposing polysemantic neurons. A comparison of OMP-based decomposition vs. SAE-based decomposition on the same CLIP model—measuring both reconstruction quality and downstream task performance—would clarify whether the second-order direction + OMP pipeline offers advantages over the standard approach.

- **Causal intervention experiments.** A direct test—e.g., activating/suppressing specific decomposed text directions in neuron activation space and measuring the predicted change in output logits—would validate that decompositions are causally faithful rather than purely correlational descriptions.

- **Adversarial attack in multi-class or broader settings.** Extending to full CIFAR-10 or ImageNet-scale multi-class classification, or reporting targeted attack success rates compared to gradient-based semantic adversarial methods, would substantially strengthen this application.

- **Reporting variance explained by top-k PCs.** Showing PC#1 through PC#5 cumulative variance, and segmenting by early vs. late neurons, would clarify the scope of the rank-1 approximation.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Harsh Critic Issue 1 — "Eq. (5) does not capture multi-layer propagation through later attention blocks."**
**Removed.** This is a misreading. Eq. (5) explicitly sums over all subsequent layers l' from l+1 to L: `Σ_{l'=l+1}^L Σ_h Σ_i ...`. The formula captures the neuron's additive contribution being read by every subsequent attention head's value pathway across all later layers—which is precisely what "through all the consecutive attention heads" means in the residual stream framework (Elhage et al. 2021). The critic is correct that higher-order cascading paths (neuron → attention modifies it → another attention reads modified version) are not captured, but that is a standard approximation in mechanistic interpretability and is the reason the authors call it "second-order"—i.e., paths of length 2, not length 3+. The empirical validation in Figure 3 confirms the approximation is accurate. The phrase "total contribution" in the introduction is imprecise but does not constitute a structural mismatch between claim and computation.

**Spark — "Segmentation baselines are outdated/inappropriate; MaskCLIP, CLIP-Surgery, SAM should be included."**
**Removed (scope creep).** The paper's primary contribution is a mechanistic interpretability method for CLIP neurons; zero-shot segmentation is an *application* used to validate utility. The relevant comparison class is prior work that uses CLIP internals for segmentation (Chefer et al. 2021, TextSpan), which the paper covers. Demanding comparison against full segmentation pipelines (SAM, MaskCLIP) that use different paradigms and supervision would be scope creep for an interpretability paper.

**Harsh Critic — Criticism of lack of comparison to PGD/gradient-based attacks as evidence the adversarial pipeline is weak.**
**Removed.** Gradient-based adversarial attacks (PGD) operate in a fundamentally different threat model (perturbation of pixel values) vs. the paper's semantic generation pipeline (images on the natural image manifold). Comparing them is category error. The paper's contribution is the *interpretability-guided* generation of natural semantic adversarial images, not a claim to outperform L∞-bounded attacks.

---

## Novel Insights

The most genuinely novel observation is the conjunction of three empirical findings that together license a tractable interpretability pipeline: (1) the second-order effect is sparse (< 2% of images per neuron), (2) it is approximately rank-1 (supported by near-perfect accuracy recovery in Figure 3), and (3) this rank-1 direction lives in CLIP's shared text-image space, making sparse text decomposition via OMP directly applicable. Each finding alone would be interesting; together they justify interpreting individual neurons by a small list of words—bridging the mechanistic interpretability and automated concept discovery literatures in a way neither had previously achieved for CLIP neurons. The adversarial application further demonstrates that polysemanticity is not merely a theoretical curiosity but an operational vulnerability exploitable without gradient access.

---

## Suggestions

1. **Add a direct activation-prediction experiment.** For a held-out set of images and neurons, predict which images will have large-norm second-order effects using the decomposed text concepts, and report precision/recall. This would directly validate "concepts correctly track which inputs activate a given neuron."
2. **Report variance explained by PC#2 through PC#5** and analyze whether rank-1 quality varies systematically between early and late layers or interpretable vs. uninterpretable neurons.
3. **Quantify interpretability coverage.** Report the fraction of neurons whose top-4 decomposed text descriptions are judged semantically coherent (either via human study or via an automated activation-tracking metric), to substantiate the scalability claim.
4. **Report pre- and post-filtering numbers for the adversarial pipeline** (how many images were generated total vs. how many passed the content filter), so success rates reflect the full pipeline cost.
5. **Ablate second-order vs. first-order directions for neuron selection in segmentation.** Replace `r_n^l` with the first-order direction in the neuron selection step while keeping everything else constant, to isolate the second-order lens's contribution to the segmentation gain.

---

## Score and Decision

**Calibration:**

- **TextSpan (5Ca9sSzuDp.md, 8,8,8,8, oral):** Same first author, same venue, predecessor work interpreting CLIP attention heads. That paper introduced the text-based decomposition framework for attention heads and zero-shot segmentation from image patches. The current paper extends this to MLP neurons, which is harder and less explored, adds the second-order lens (novel), and introduces a new adversarial application. However, the segmentation improvement over TextSpan is modest (~1.5% pixel acc), and the adversarial application has low absolute rates. The contribution is meaningful but narrower in scope than the full TextSpan framework.

- **PatchSAE (imT03YXlG2.md, 6,6,6,8):** CLIP interpretability with sparse methods; similar territory (understanding CLIP internals with sparse representations), comparable technical depth. That paper had clarity/presentation issues that brought scores to 6-6-6-8. This paper is better organized and has cleaner applications.

- **Describe-and-Dissect (Rnxam2SRgB.md, 3,6,5,5, rejected):** Neuron interpretation via LLMs; weaker evaluation (entirely qualitative human studies), more opaque pipeline. This paper is considerably stronger methodologically.

**Assessment:** The paper is a solid, well-executed contribution extending an established framework to a more challenging domain (neurons). The methodology is technically sound and validated. The main limitations—indirect semantic validation, modest adversarial success rates, modest segmentation gains—do not undermine the core novelty but temper the strength of the contribution. This sits comfortably above PatchSAE's ~6.5 average and below TextSpan's 8.0, placing it around **6.5**.

**Originality:** High — the second-order lens for neuron interpretation in CLIP is genuinely new.  
**Importance of research question:** High — neuron-level interpretability in VLMs is underexplored and practically relevant.  
**Claims vs. evidence:** Moderate — the core rank-1 approximation is empirically well-supported; the semantic faithfulness claim relies too heavily on qualitative illustration.  
**Soundness of experiments:** Moderate-high — rigorous in the ablation structure; limited by narrow adversarial settings and indirect validation.  
**Clarity:** High — the paper is well-written and clearly structured.  
**Value to community:** High — both applications are useful and the interpretability analysis opens concrete directions.

**Decision: Accept (poster)**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>