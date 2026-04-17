---
job_id: c9cc5043-9fb0-456e-92c8-a4731191fd72
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: TADsPhlp32.pdf
paper: Structural Semantic Features for Improved AI-Generated Fake Image Detection
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a representation-learning based method for AI-generated image detection, integrating structural features with an existing hybrid detector; this falls squarely within representation learning for computer vision and general ML.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Works, Methodology, Experiments, Results, Conclusion) are present, in English, and the work includes concrete methods, equations, and extensive quantitative evaluation. While there are weaknesses (e.g., limited ablations, some experimental ambiguities), there is no fatal methodological flaw or missing core section that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts, meta-instructions to reviewers, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper proposes augmenting the AIDE AI-generated image detector with a new “structural semantic” feature extractor based on cuboidal partitioning of the image. The method recursively partitions an image by greedily selecting axis-aligned cuts that maximally reduce the sum of squared pixel-feature errors, then uses the normalized cumulative gains from the top \(N=1024\) splits as a structural feature vector, which is compressed and concatenated with AIDE’s patchwise and CLIP-based features. Experiments on GenImage, AIGCDetect, and Chameleon show improved mean accuracy over AIDE on GenImage, second-best mean performance on AIGCDetect, and consistently second-best performance on the challenging Chameleon benchmark, suggesting these structural features provide complementary information.

## Strengths

1. **Clear, simple structural feature formulation with reasonable intuition.**  
   The core idea in Section 3.2, formalized in **Equations (1)–(3)**, is straightforward: treat the image as a segment, compute SSE \(e_S = \sum_{p_i\in S}\|p_i-\mu_S\|^2\), recursively split on the cut that maximally reduces SSE, and define a normalized cumulative gain curve \(\hat g_i = \frac{1}{e_I}\sum_{j=1}^i \hat g_j\). This gives a 1D structural signature that encodes how “structured” the image is at different levels. The math is simple but internally consistent, and the link to hierarchical segmentation / quadtree-style representations is clear.

2. **Architecture integration is clean and modular.**  
   **Figure 2** effectively shows how the proposed structural feature extractor is added as a third branch alongside AIDE’s existing Patchwise and Semantic modules, and that only the structural extractor and final MLP head are trained while the AIDE backbones are frozen. This design is easy to reproduce and suggests the method can be plugged into other detectors with minimal engineering.

3. **Strong empirical performance on GenImage, where structural features are most justified.**  
   In **Table 1**, the proposed method improves AIDE’s mean accuracy on GenImage from 86.88% to 89.56% (+2.68 absolute). It is best or second-best for most generators, including clear gains on difficult ones like BigGAN (66.89 → 73.64) and consistent improvements on ADM, GLIDE, VQDM, and Wukong. Given GenImage’s focus on modern diffusion models, this is a meaningful empirical result.

4. **Nontrivial robustness across benchmarks, particularly for human-deceptive content.**  
   Even though AIGCDetect is more mixed (see below), **Table 3** shows that on the particularly challenging Chameleon dataset the method is consistently second-best under both ProGAN (58.91%) and SD v1.4 (61.39%) training regimes, close to or slightly below AIDE. Given that Chameleon images are explicitly chosen to fool humans, this supports the claim that the structural features are not overfitting to simple artifacts and preserve OOD robustness.

5. **Qualitative visualizations help interpret the contribution.**  
   - **Figure 1** overlays the cuboidal partitioning on a WFIR face and highlights regions near the ear and a hair-like artifact that are isolated by the partitioning. This gives a concrete, intuitive sense of what “structural semantics” means here, and how the partition tree can focus on local anomalies that AIDE misses.  
   - **Figure 3** shows 13 failure cases of AIDE where the proposed method flips the decision, with confidence scores transitioning from <50% to >50%. While not deeply analyzed, this supports the claim that the structural branch is correcting nontrivial mistakes rather than just re-weighting obvious cases.

6. **Positioning relative to AIDE is coherent and honest about mixed outcomes.**  
   The paper does not pretend that adding the new expert improves all subsets; Section 4.8 openly notes mild performance drops on some AIGCDetect subsets and frames the method as a context-dependent expert. This mixture-of-experts framing, citing Hansen & Salamon (1990), is conceptually sound and matches the empirical tables.

7. **Computation / reproducibility details are adequate.**  
   Section 4.3 gives training datasets, learning rates, batch size, number of epochs, GPU type, and approximate training time for each benchmark. Combined with the architectural clarity of **Figure 2** and the explicit setting \(N=1024, M=256\), this is sufficient to reproduce the main results once code is available.

## Weaknesses

1. **Limited novelty: mostly a straightforward feature add-on to AIDE with modest conceptual leap.**  
   The main technical move is to take an existing cuboidal partitioning / SSE-gain hierarchy (Ahmmed et al., 2022; Haque et al., 2025) and feed its cumulative gain curve as a feature into an existing hybrid detector (AIDE). There is no new learning objective, no new partitioning criterion, and no adaptation of the partitioning to the forgery task. From a representation learning standpoint, this feels like adding a hand-crafted global statistic to an existing feature soup. The paper claims to be the “first application” of such structural analysis to AIGC detection, which is probably accurate, but the conceptual advance is incremental rather than deep.

2. **Insufficient ablations disentangling where the gains come from.**  
   There is no systematic ablation study isolating the contribution of the structural branch. Missing key experiments include:
   - A direct comparison between AIDE re-trained under *exactly the same* training protocol and “Ours” on each benchmark (the AIDE numbers seem taken from prior work, but this is not explicit). Without re-training the baseline in the same codebase, it is hard to attribute the +2.68% on GenImage solely to structural features.  
   - A model that uses *only* structural features (no patchwise or CLIP), to show how much discriminative power is in the partitioning itself.  
   - Variants that change \(N\) (number of splits) or remove the GELU/FC compression to test sensitivity.  
   Given how central the “structural semantics” claim is, the absence of these ablations weakens the causal story and makes the results look more like “another feature helps a big ensemble” than “structural semantics are specifically valuable”.

3. **Methodological details of the partitioning and feature extraction are underspecified.**  
   Section 3.2 leaves several implementation-critical aspects vague:
   - It states that \(p_i\) is a “feature vector (e.g., RGB values)” in **Equation (1)**. Is it exactly 3D RGB? Are channels normalized? Are any higher-level features (e.g., from CLIP or SRM-filtered responses) used instead? This matters because SSE on raw RGB is sensitive to illumination and color, which might not be the most robust structural cue.  
   - The search over “all cuts” for \(\hat g = \max_{\forall\text{cuts}} g\) in **Equation (2)** is not defined precisely: are candidate cuts evaluated at every pixel location, at a grid, or with some stopping criterion? How is computational cost controlled for \(N=1024\) splits on high-res images?  
   - The hierarchical strategy “always selecting the sub-segment that offers the greatest potential gain” needs more explicit pseudo-code. For example, are all current leaves considered and the one with maximal possible gain chosen, or is the tree traversed depth-first?  
   These ambiguities are nontrivial for anyone trying to reimplement the feature extractor faithfully.

4. **Some experimental inconsistencies and possible typos in AIGCDetect results.**  
   **Table 2** appears to contain obvious errors:  
   - “CNNSpot | ProGAN: 104.00” is impossible for an accuracy metric that should be in \([0,100]\).  
   - Column names like “StyleGAN | StyleGAN | CnVbGAN | StoGAN | StoGAN2 | MediPentary | Adide | SOTA” are confusing or likely mis-typed (e.g., “CnVbGAN” presumably “CycleGAN”; “StoGAN/ StoGAN2” presumably “StarGAN / StarGAN2”; “MediPentary” and “Adide” are unclear; “SOTA” is not a dataset).  
   - Several mean scores for methods like UnivFD (“Mean 50.73”) look inconsistent with the rest of the row, given many entries above 80–90.  
   These issues make it difficult to trust the exact comparative numbers on AIGCDetect, and they suggest the table may have been adapted from prior work without adequate verification.

5. **Inconsistent performance and lack of analysis for the drops.**  
   While the paper acknowledges drop-offs on some AIGCDetect subsets, the analysis is too high-level: it attributes them generically to contexts where “few structural inconsistencies” are present. A more rigorous analysis would, for example:
   - Quantify per-subset gains/drops relative to AIDE and correlate them with specific content types (faces, scenes, product images).  
   - Provide failure examples where the structural features mislead the detector and analyze what patterns they are responding to.  
   This is important because **Table 2** shows noticeable declines on some subsets that matter in practice (e.g., some StyleGAN variant and diffusion model columns), and the paper currently hand-waves these away.

6. **Missing broader comparison to other recent semantic / structural AIGC detectors.**  
   The related work section focuses mainly on “classic” AIGC detectors (frequency, noise patterns, CLIP-based UnivFD, DIRE, PatchCraft, AIDE) and on the older structural similarity / partitioning literature. It does not engage with several recent methods that also exploit *higher-level structural or semantic inconsistencies* in AIGC (see “Potentially Missing Related Work” below). As a result, the conceptual novelty is overstated, and the paper does not clearly explain how a global SSE-gain curve differs in effect from, say, semantic reconstruction error or semantic anomaly detection.

7. **Limited interpretability of the learned structural embedding.**  
   The paper argues that the cumulative gain curve “effectively encodes the image’s organizational hierarchy” and **Figure 1** shows partitions on a single face, but there is no further analysis of how these curves differ between real and fake images. For instance, plotting average \(\hat g_i\) curves for real vs. fake on a held-out set could show whether fakes tend to have higher gains at certain scales, or whether the curves are just generic complexity measures. Without such diagnostics, the titular “structural semantic features” remain somewhat opaque and the term “semantic” feels like an overclaim for what is, mathematically, purely low-level SSE.

8. **Minor but nontrivial clarity issues and typos.**  
   There are several smaller problems that collectively reduce polish and precision:  
   - In **Table 1**, “Spec | 52.0 0” is likely a typo.  
   - Some references are misformatted or partially duplicated (e.g., multiple “Tan et al. (2023)” lines without consistent numbering; Rombach et al. listed under “Robinson et al.”; inconsistent inline citation styles).  
   - Section 4.6 states “As shown in Table 3, while our model is not the outright best performer…” but then treats being second-best as “crucial validation” without nuance; a more balanced phrasing would acknowledge that all methods are very close numerically (mid- to high-50s).

Overall, while the idea is reasonable and the GenImage gains are promising, the combination of limited novelty, missing ablations, and some experimental-table issues prevents this from being a clear, unambiguous advance.

## Potentially Missing Related Work

Below are directly related works that are not cited or discussed but are important for positioning this paper’s contribution around semantic/structural cues in AIGC detection:

1. **Kang et al., “SARE: Semantic-Aware Reconstruction Error for Generalizable AI-Generated Image Detection”, 2025.**  
   - Relevance: SARE focuses on semantic-aware reconstruction error, explicitly measuring semantic inconsistencies between an image and its caption-guided reconstructions. This is conceptually close to the paper’s claim that “structural semantics” help detect inconsistencies beyond low-level artifacts.  
   - Suggestion: Discuss in Section 2.1 as another approach that targets higher-level structure/semantics, and compare qualitatively in the introduction when motivating semantic/structural cues.

2. **Fu et al., “PiD: Generalized AI-Generated Images Detection with Pixelwise Decomposition Residuals”, 2025.**  
   - Relevance: PiD leverages residual signals via pixelwise decomposition to uncover forgery clues independent of semantic content, similar in spirit to using aggregated low-level statistics (here, SSE gains).  
   - Suggestion: Add to Section 2.1 as a modern artifact-based detector and clarify how global hierarchical SSE features differ from localized residual statistics.

3. **Tan et al., “Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images”, 2026.**  
   - Relevance: This work formalizes semantic anomaly detection in AIGC and introduces structured annotations for semantic-level inconsistencies. It strongly overlaps with the paper’s narrative about “structural semantics” and violations of physics/anatomy.  
   - Suggestion: Cite in the Introduction and Section 2.2 when discussing semantic/structural inconsistencies (Kamali et al., 2024) and position your approach relative to explicit semantic anomaly detection.

4. **Zheng et al., “Breaking Semantic Artifacts for Generalized AI-generated Image Detection”, 2024.**  
   - Relevance: Analyzes how detectors overfit to semantic artifacts and proposes patch-based methods to mitigate this. This directly touches the risk that global structural statistics, like the ones here, might capture semantic bias rather than genuine generative artifacts.  
   - Suggestion: Discuss in Section 2.1 as a cautionary counterpart, and in Section 4.8 when reflecting on contexts where your structural features hurt performance.

5. **Zisman & Shaham, “RealStats: A Rigorous Real-Only Statistical Framework for Fake Image Detection”, 2026.**  
   - Relevance: Proposes a real-only statistical detection framework with interpretable scores. Given that your structural features are also statistical summaries of images, comparison to RealStats would strengthen the statistical grounding of your approach.  
   - Suggestion: Mention in Related Work and briefly contrast your discriminative, feature-augmentation approach with their generative / real-only approach.

6. **Rajan & Lee, “Stay-Positive: A Case for Ignoring Real Image Features in Fake Image Detection”, 2025.**  
   - Relevance: Argues that focusing exclusively on generative artifacts rather than both real and fake features improves generalization. Your structural features may pick up dataset-specific real-structure biases, which could explain some drops on AIGCDetect.  
   - Suggestion: Add to Section 4.8’s discussion of when extra experts hurt, and acknowledge the tension between richer features and overfitting to non-artifact structure.

7. **Xiao et al., “Unveiling Perceptual Artifacts: A Fine-Grained Benchmark for Interpretable AI-Generated Image Detection”, 2026.**  
   - Relevance: Introduces a benchmark centered on interpretable perceptual artifacts. This aligns with your qualitative examples in **Figure 3** and could be a natural additional benchmark for structural features.  
   - Suggestion: Mention in Related Work as a resource for future evaluation and interpretability analysis.

8. **Yu et al., “SemGIR: Semantic-Guided Image Regeneration based method for AI-generated Image Detection and Attribution”, 2024.**  
   - Relevance: Uses image-to-text-to-image regeneration and compares original/regenerated images to detect fakes, explicitly leveraging semantic guidance and structural differences. It is directly relevant to your focus on higher-level structural semantics.  
   - Suggestion: Discuss in Section 2.1/2.2 as a semantically guided method; contrast regeneration-based structural differences with your purely image-internal partitioning.

Including and engaging with these works would make the paper’s contribution clearer and more honestly scoped.

## Questions

1. **Implementation details of the partitioning and features.**  
   - What exactly is the pixel feature vector \(p_i\) in **Equation (1)**? Is it 3D RGB, normalized RGB, grayscale intensity, or something else (e.g., SRM-filter responses)?  
   - How are candidate cuts enumerated to compute \(\hat g = \max_{\forall\text{cuts}} g\) in **Equation (2)**? Do you consider all possible row/column splits, or a subsampled grid? What is the complexity as a function of image size and how is it handled in practice?  
   - Are there stopping criteria other than reaching \(N=1024\) splits (e.g., minimum segment size or minimum gain)?

2. **Baseline alignment and training protocols.**  
   - Are the AIDE numbers in **Tables 1–3** obtained by running your own implementation under *exactly* the same training pipeline as “Ours”, or are they directly copied from Yan et al. (2025)? If the latter, can you provide results for a re-trained AIDE baseline to ensure a fair comparison?  
   - Relatedly, were any hyperparameters for the structural branch or MLP head tuned on validation sets that overlap with the test generators?

3. **Effect of varying \(N\) and feature compression.**  
   - Why choose \(N=1024\) and \(M=256\)? Did you run experiments with smaller \(N\) (e.g., 256 or 512) and different \(M\)?  
   - Is there any performance trade-off between larger \(N\) (more structural detail) and overfitting on smaller datasets like Chameleon? Some ablation curves would be very informative.

4. **Behavior of the structural features alone.**  
   - How well does a detector that uses only the structural feature branch perform compared to AIDE, UnivFD, etc., particularly on GenImage? Even approximate single-branch results would help quantify whether the structural features are strong stand-alone discriminators or mainly useful as a weak but complementary expert.

5. **Clarification of AIGCDetect table and metrics.**  
   - Please verify and correct **Table 2**, especially the “ProGAN 104.00” entry and the dataset column names (“CnVbGAN”, “StoGAN”, “MediPentary”, “Adide”, “SOTA”). It is currently unclear what each column corresponds to and whether 104% is a typo.  
   - Confirm that “Mean” is simple arithmetic mean of per-generator accuracies and not weighted by dataset size.

Addressing these questions with concrete experiments (especially ablations and corrected tables) would significantly increase my confidence in the claims.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating
3: good.  
The core method is mathematically sound and the GenImage and Chameleon experiments are broadly convincing, but missing ablations and issues in Table 2 reduce confidence in some empirical claims.

## Presentation Rating
2: fair.  
The overall narrative and figures are understandable and the core equations are clearly stated, but important implementation details are missing, several tables (especially Table 2) have apparent errors, and the related work omits key recent semantic/structural detectors.

## Contribution Rating
2: fair.  
The work likely has practical value as an add-on to strong baselines and achieves state-of-the-art on GenImage, but the conceptual novelty is limited and the lack of deeper analysis/ablations prevents it from being a clear, substantial step forward.

## Overall Rating
4: marginally below the acceptance threshold. But would not mind if paper is accepted.  
The idea of adding a hierarchical structural feature branch to AIGC detectors is sensible and yields solid gains on GenImage with reasonable robustness on Chameleon, which is nontrivial. However, the contribution is relatively incremental, key ablations and implementation details are missing, and there are clear problems with one of the main result tables. With stronger experimental analysis, corrected/clarified comparisons, and deeper engagement with recent semantic-focused literature, this could be elevated to a clear accept; in its current form it falls slightly short of ICLR standards.

## Reviewer Confidence
4: confident.  
I am familiar with AIGC detection literature, checked the math in Section 3 carefully, and examined the tables/figures in detail; while some implementation specifics are missing, my overall assessment is unlikely to change drastically, though a strong ablation section and corrected tables could move my recommendation upward.