=== CALIBRATION EXAMPLE 17 ===

# Final Consolidated Review
## Summary
This paper identifies a largely overlooked failure mode of attribute-based Vision-Language Model (VLM) adaptation: a small fraction (<7%) of "spuriously correlated attributes" in the attribute pool exerts disproportionate influence on model predictions, causing poor OOD generalization. To address this, the authors propose **Spurious Attribute Probing (SAP)**, which uses an MLLM and Concept Bottleneck Models (CBMs) to identify and filter spurious attributes from the pool, complementing existing attribute-based methods; and **Spurious Attribute Shielding (SAS)**, a plug-and-play subsidiary-loss module that constructs pseudo-categories from spurious attributes to discourage reliance on them during PEFT training. Experiments across 11 datasets and 3 generalization tasks with 11 baselines report consistent OOD improvements of over 2% in most settings.

---

## Strengths

- **Compelling, falsifiable empirical motivation.** Table 1 cleanly demonstrates that manually removing <7% of spurious attributes yields +2.36% new-category accuracy for CPL and +1.62% for ArGue, without sacrificing base accuracy. The CBM weight analysis in Figure 1 further shows that spurious attributes occupy 2 of the top-3 influential slots, giving an intuitive, quantitative hook for the "Black Sheep" label.

- **Broad experimental scope.** The evaluation spans 11 diverse datasets, 3 generalization tasks (base-to-new, cross-dataset transfer, domain generalization), and 11 baselines covering prompt tuning, adapters, and training-free methods — an unusually thorough experimental protocol for the field.

- **Creative adversarial evaluation via counter group (Table 2).** Constructing a test subset filtered to remove images with high semantic similarity to spurious attributes, and then showing SAS closes the gap between test-set and counter-group accuracy (by ~4–6%), is a distinctive and convincing way to isolate the effect of spurious attributes on evaluation.

- **Effective plug-and-play design with efficiency analysis.** SAS integrates without architectural changes; Table 5 shows the selective 10%-category trick reduces training overhead to near-baseline levels (~4h51m vs. 4h37m for CoCoOp) while preserving ~83% of the full-SAS gains — a practical contribution for the community.

- **Adaptive threshold outperforms all fixed values.** Table 4 empirically validates the adaptive γ_c strategy across a grid of fixed values (0.0–1.0), showing it is not simply cherry-picked.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Detailed per-dataset, per-baseline results are entirely in supplementary (Supp. Mat. E), not the main paper.** Figure 3 shows aggregate scatter plots but provides no individual baseline gains, no per-dataset breakdown, and no indication of cases where performance decreases. Reviewers and readers cannot verify the "average improvement of over 2% in most baselines" claim — specifically, which baselines benefit how much, and whether any baselines regress — without accessing materials outside the paper. For a contribution whose core value is empirical, this is a significant structural deficit.

- **The adaptive threshold has an underacknowledged failure mode.** γ_c is defined as the *lowest* weight among core attributes A_c^+. If even one core attribute has a near-zero CBM weight (e.g., a visually subtle but definitionally critical feature), γ_c collapses toward 0, and essentially all non-core attributes get flagged as spurious — flooding the SAS pipeline with false positives. Table 4 ablates fixed γ values and shows that low γ (≈0.0) does degrade performance, validating this concern. The authors acknowledge the false-positive risk in general (Section 3.3: "potentially leading to false positives"), but do not address this specific degenerate case of the adaptive rule.

- **CBM trained on 16-shot data conflates coincidental and structural co-occurrences.** The CBM that assigns attribute weights is fitted on the same 16 training images per class used for PEFT. An attribute that appears in 12 of 16 training images for "mountain bike" will receive high weight regardless of whether it is a structural spurious correlation or a coincidence of the specific 16 shots. The paper does not report sensitivity to training-set seed, nor does it discuss this confound. Since the downstream identification and shielding of spurious attributes depend directly on these CBM weights, this is a meaningful gap in robustness analysis.

- **SAS pseudo-categories address spurious attributes in isolation, not in co-occurrence.** The spurious correlation problem arises because spurious attributes co-occur *with* the target class in training images (e.g., mountain bike + trees). SAS constructs pseudo-categories from images of the spurious attribute *alone* (e.g., just trees). This teaches the model to distinguish trees from mountain bikes, but does not directly disrupt the spurious correlation arising from joint presence. The paper acknowledges purity concerns about noise in synthetic images (Section 3.4) but does not address this co-occurrence mismatch. The empirical results suggest the approach still works, but the theoretical gap deserves explicit discussion.

### Minor

- **Main qualitative motivation study (Table 1) covers only 4 of 11 datasets.** FGVCAircraft, SUN397, Flowers102, and DTD are shown; ImageNet, Caltech101, EuroSAT, StanfordCars, etc., are not. While the full evaluation in Figure 3 uses all 11, the motivational study that justifies the entire framework might look different on datasets with different spurious-attribute profiles (e.g., EuroSAT satellite imagery). This should at minimum be acknowledged.

- **GPT-4V API dependency creates reproducibility and accessibility concerns.** SAP requires GPT-4V Turbo at "high" image-understanding level across 16 shots × many categories × 11 datasets. Neither the total API cost nor the number of queries is reported. The paper mentions Supp. Mat. B includes results with alternative MLLMs, which partially mitigates this, but these alternatives are not discussed in the main body. Given that reproducibility is an ICLR core value, the main text should at minimum state approximate costs and provide the open-source alternative results.

- **Counter group evaluation has a partial circularity concern.** The counter group is constructed by filtering images with high semantic similarity to the *same* spurious attributes identified by SAP. If SAP produces false positives (misidentifying a core attribute as spurious), the counter group will exclude images containing that core attribute — inflating the apparent performance gap between test and counter group and making SAS look more effective on it than it is. The paper does not discuss this dependency.

- **No ablation isolating the quality of SAP-identified spurious attributes as input to SAS.** Figure 3 shows "Baseline + SAP" and "Baseline + SAS" separately, which is useful. However, there is no experiment showing SAS with *random* or *unfiltered* attributes as pseudo-category inputs. Table 4's γ sweep is the closest proxy, but it only ablates the threshold, not the quality of the MLLM-derived non-core/core split. This would clarify whether SAS gains are specifically tied to the spurious-attribute identification quality of SAP.

### Tiny

- Section 3.2's description of the heatmap annotation process (sampling 5 images, visual inspection) is very brief. It gives no detail on how ambiguous cases were resolved and does not report inter-annotator agreement — relevant since this is the foundation of the "Black Sheep" empirical motivation.

- The loss in Eq. 3 computes the subsidiary softmax only over J_c ∪ {c} — pseudo-categories are local to their target class. If two classes share a spurious attribute (e.g., "road" for both vehicle and scooter), the subsidiary task does not see this shared structure. This design choice is not discussed.

---

## Nice-to-Haves

- Compare SAS against standard distributionally robust optimization baselines (e.g., GroupDRO) adapted for VLMs to demonstrate that gains are not merely from generic robustness techniques.
- Report wall-clock time including offline costs: Stable Diffusion image generation and LAION retrieval time, currently omitted from Table 5's efficiency analysis.
- Validate SAP's precision and recall with human annotation on a dataset with known ground-truth spurious correlations (e.g., Waterbirds or CelebA) to anchor claims about identification quality.
- Provide a theoretical explanation for why ℒ_pse promotes invariant feature learning rather than simple multi-task generalization; the current framing is heuristic.
- Investigate open-source MLLM alternatives (e.g., LLaVA) as drop-in replacements for GPT-4V to lower adoption barriers; if Supp. Mat. B already contains this, promote the key finding to the main body.
- Include a control variant where pseudo-categories are built from random (non-spurious) attributes, to firmly distinguish "any auxiliary task" from "specifically spurious-attribute shielding."

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **[REMOVED — scope creep] Criticism that SAS is not evaluated on VQA, captioning, or retrieval.** The paper explicitly scopes itself to classification generalization tasks. Not doing VQA is not a weakness; it could be noted as future work.
- **[REMOVED — field norm] Demanding confidence intervals / standard deviations for all tables.** The paper reports 3-run averages with distinct initialization. This is standard practice in the prompt-tuning and few-shot VLM literature (e.g., CoCoOp, PromptSRC, MaPLe all report single or 3-run means without CIs). This does not warrant a standalone weakness.
- **[REMOVED — scope creep] Criticism that the paper does not extend to subtle spurious correlations (lighting, texture biases, compositional patterns).** The paper explicitly targets "discrete, nameable" visual attributes and makes no claim otherwise.
- **[REMOVED — assumption without evidence] Claim that Tip-Adapter comparison is unfair because it is training-free.** A training-free baseline is a legitimate lower-bound comparator; including it demonstrates SAS adds value. The asymmetry does not favor the authors' method.
- **[REMOVED — factually disputable] Claim that the ArGue\* prompt-engineering result "weakens motivation for the full SAP pipeline."** The paper's argument is precisely that naive prompting produces only marginal reduction (6.06% → 5.76%), motivating the full SAP pipeline. The motivation is logically valid.
- **[WEAKENED → NICE-TO-HAVE] Demand for theoretical proof of why ℒ_pse induces invariant features.** This is an empirical systems paper; requiring theoretical proofs is not standard for this community or this contribution type.
- **[WEAKENED → NICE-TO-HAVE] Requesting larger-scale SAS comparison against group DRO.** While useful, this would require re-implementing DRO under the VLM/PEFT framework — a non-trivial extension beyond the paper's stated contribution.

---

## Novel Insights

The most genuinely insightful contribution is the operationalization and quantification of the "Black Sheep" phenomenon: despite comprising <7% of the attribute pool, spurious attributes occupy two of the top-three most influential slots in CBM-based weight rankings (Figure 1), and their removal produces OOD gains that exceed what proportional removal would predict. This asymmetric leverage — small in count, large in influence — is a novel empirical finding that connects attribute-based VLM adaptation failures to the debiasing literature in a concrete, actionable way. The counter-group evaluation framework (Table 2) is also a methodologically interesting contribution: rather than relying purely on aggregate OOD accuracy, it constructs a minimally spurious-attribute-contaminated test subset, providing a more targeted probe of spurious-reliance. Both insights are transferable to related work on attribute-based adaptation beyond the specific SAP/SAS implementations.

---

## Suggestions

1. **Move per-dataset, per-baseline numerical results into the main paper**, even in compact form (e.g., a single summary table with per-task aggregates per baseline). Scatter plots in Figure 3 are informative but insufficient for independent verification of the core claim.

2. **Add a robustness check on the adaptive threshold** by testing its behavior on at least one dataset where known core attributes have low CBM weights, to validate the adaptive rule doesn't degenerate.

3. **Report CBM weight stability across training seeds** (e.g., variance of identified spurious attributes across the 3 initialization runs). This would directly address the 16-shot confound concern.

4. **Include one experiment using random attributes as SAS pseudo-category inputs** as a control for the "any auxiliary task" hypothesis, even on a single dataset. This is critical to establishing that the gains are specifically due to *spurious* shielding.

5. **Add a brief discussion of the co-occurrence mismatch** in Section 3.4: acknowledge that pseudo-categories represent isolated spurious attributes rather than co-occurring contexts, and hypothesize why the subsidiary objective still reduces reliance (e.g., by suppressing feature responses to those visual patterns regardless of context).

6. **Promote the key open-source MLLM ablation from Supp. Mat. B to the main body**, even a single sentence with main numbers. This directly addresses the reproducibility concern without requiring extra experiments.

---

**Overall assessment:** The paper identifies a real and underappreciated problem, provides a crisp empirical motivation, and delivers a practical solution that works broadly across baselines and datasets. The methodological choices (adaptive threshold, 16-shot CBM, isolated pseudo-categories) each carry genuine gaps that the paper does not fully acknowledge. The structural decision to relegate all quantitative detail to the supplementary is a notable submission weakness. Nevertheless, the core idea is sound and the empirical scale is impressive. With the structural and ablation deficiencies addressed — particularly surfacing detailed results in the main paper and adding the random-attribute control — this would be a solid contribution.

**Novelty:** Moderate-to-good. The "Black Sheep" framing and SAP pipeline are novel; SAS is a creative application of known debiasing ideas.
**Technical soundness:** Moderate. Several design choices have gaps the paper does not address.
**Empirical support:** Good in scope; insufficient in main-body accessibility and ablation completeness.
**Significance:** High for the VLM/PEFT community; plug-and-play integration is a practical asset.
**Clarity:** Good prose; poor quantitative presentation due to supplementary over-reliance.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 8.0, 6.0]
Average score: 7.0
Binary outcome: Accept
