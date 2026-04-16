## Summary

The paper proposes a “second-order lens” to study CLIP-ViT neurons by tracing their influence through subsequent attention heads to the output, rather than via direct logit lenses or intervention-based indirect effects. The authors empirically characterize these second-order effects (late-layer, sparse, approximately low-rank), then approximate each neuron’s effect by a single direction in CLIP’s joint text–image space and decompose that direction into a sparse combination of text embeddings. They argue this yields polysemantic neuron descriptions, and use these analyses to (i) construct semantic adversarial examples and (ii) perform zero-shot segmentation that improves over prior CLIP-based attribution methods.

## Strengths

- **Clear formalization of a nontrivial lens on neurons.**  
  The paper gives a mathematically explicit definition of the “second-order effect” (Eq. 5) by propagating each neuron’s residual contribution through later attention-value paths into the output, separating input-dependent “attention-weighted activations” from input-independent matrices. This is a concrete and analyzable object that goes beyond first-order logit lenses and naive ablation-based indirect effects.

- **Useful empirical characterization of CLIP’s MLP neurons.**  
  For CLIP ViT-B/32 (and partially ViT-L/14 in the appendix), the paper shows:
  - Second-order effects are concentrated in moderately late layers (8–10), via mean-ablation experiments (Fig. 3, “w/o all neurons”).
  - Individual neurons’ second-order effects matter for only a small subset of images, supported by selective ablation of large-norm vs small-norm effects (Fig. 3, “w/o large norms” vs “w/o small norms”).
  - Second-order effects of a neuron admit a strong first principal component; using PC#1 plus a bias recovers near-baseline ImageNet accuracy (Fig. 3, “rec. from PC #1”) and explains substantially more variance than indirect effects (Table 1: 48.2% vs 11.0%).

- **Zero-shot segmentation performance is strong and simple.**  
  The segmentation pipeline—select neurons whose directions \(r_n^l\) align with the text embedding of a class and average their spatial activations—achieves the best reported results on ImageNet-Seg (Table 4), outperforming established attribution methods and the prior TextSpan approach across all metrics (Pixel Acc, mIoU, mAP). Qualitative masks (Fig. 7) indeed look more complete than those from first-order token-based attribution.

- **Ambitious link from mechanistic analysis to adversarial attacks.**  
  The semantic adversarial pipeline (Sec. 5.1, Fig. 6) is conceptually appealing: it connects neuron-level polysemy, text descriptions, an LLM, and a text-to-image model to generate on-manifold adversarial examples that exploit spuriously shared neuron directions between classes. Quantitatively, the “second order” pipeline outperforms random neurons, an indirect-effect analogue, and a nearest-text baseline on several CIFAR-10 binary tasks (Table 3).

- **Writing and structure are clear.**  
  The problem motivation (limitations of direct and indirect effects), the derivation of second-order effects, and the transition to applications are all presented in a logically coherent and accessible way. Limitations (e.g., ignoring query/key pathways and neuron–neuron interactions) are explicitly acknowledged in Sec. 6.

## Weaknesses

### Fatal

None. The paper is conceptually ambitious but not fundamentally unsound: the second-order effect is clearly defined, and the empirical claims broadly match the presented evidence. However, several core interpretability claims are overstated or weakly validated.

### Major

- **(1) Overclaiming “total contribution” and functional semantics for a partial pathway.**  
  By construction, Eq. (5) only traces the flow of a neuron’s output through later MSA **values** into the class token and projection; it holds queries/keys, subsequent MLPs, and layernorm behavior fixed. The paper is explicit about focusing on “flow of information from the neurons through the attention layers” (Sec. 3.2, Fig. 2) and later notes as a limitation:  
  > “We investigated how the neurons flow through individual consecutive attention *values*, and ignored the effect of neurons on consecutive queries and keys in the attention mechanism. … We leave it for future work.” (Sec. 6)  
  Despite this, the abstract and main text repeatedly describe \(\phi_n^l(I)\) as “the total contribution to the output, flowing via all the consecutive attention heads” and then treat the resulting direction \(r_n^l\) as “the direction that the neuron writes to in the joint representation space.” This framing elides that:
  - There are additional causal channels (via Q/K, later MLPs, and normalization-induced coupling) that are not modeled.
  - The “second-order” lens is therefore one particular linearized pathway, not the neuron’s full functional impact.
  The paper does not provide empirical evidence that the value-pathway dominates the neuron’s behavioral effect, nor that ignoring the other channels is harmless for their interpretive conclusions. This weakens the causal/mechanistic interpretation of \(\phi_n^l\): it is a useful diagnostic quantity but not fully justified as *the* neuron function.

- **(2) Rank-1 / “one direction per neuron” is supported only indirectly and partially.**  
  The claim that each neuron’s second-order effect can be captured by a single direction \(r_n^l\) and a scalar coefficient is central: it underpins using PCA, then sparse text decomposition of \(r_n^l\). The evidence, however, is limited:
  - Task-level: replacing each \(\phi_n^l(I)\) with \(x_n^l(I) r_n^l + b_n^l\) preserves ImageNet accuracy (Fig. 3, “rec. from PC #1”). This shows the rest of the variance is not critical for classification accuracy, but classification is known to be robust to substantial internal perturbations; this does not prove that the neuron’s effect is essentially one-dimensional.
  - Aggregate variance: Table 1 shows PC#1 explains 48.2% of the variance of second-order effects (vs 11% for indirect effects). While this is a meaningful difference and supports “more low-rank than indirect,” 48% is far from rank-1 in a strong sense. The paper does not report:
    - Distribution across neurons (some may be far from rank-1).
    - The nature or potential importance of the remaining ~52% variance.
  As a result, the strong interpretive narrative—“each neuron writes along a single semantic direction”—is not tightly supported; the evidence shows a useful low-dimensional approximation, not that the rest is negligible or non-semantic.

- **(3) Sparse text decompositions are only weakly validated as faithful neuron descriptions.**  
  Section 4 argues that sparse decompositions of \(r_n^l\) into text embeddings “describe” neuron functions and reveal polysemanticity. Current evidence is limited:
  - Quantitatively, Fig. 4 evaluates decompositions only by whether replacing \(r_n^l\) with \(\hat r_n^l\) sustains ImageNet accuracy. This assesses representational coverage of the text dictionary, not whether the *specific sparse combination* chosen for each neuron corresponds to human-meaningful concepts.
  - The method uses up to 128 text components per neuron. Such large sets are a poor fit to human-level interpretability and allow many plausible linear approximations in a high-dimensional embedding space; recovery of performance with big \(m\) does not guarantee that the identified texts correctly or uniquely capture what the neuron tracks.
  - Qualitative examples (Table 2, Fig. 5) show some attractive cases where texts match top-activating images, but are cherry-picked and anecdotal. There is no systematic evaluation (automatic or human) of:
    - How often descriptions are coherent vs nonsensical.
    - Whether descriptions predict which images a neuron has large second-order effect on.
    - How polysemy manifests across the neuron population.
  Without such evaluation, the “automated interpretation of neurons” remains plausible but not convincingly demonstrated; the decompositions could easily be post-hoc rationalizations in a dense embedding space.

- **(4) The adversarial generation evaluation is narrow and heavily confounded.**  
  While the adversarial pipeline is creative, the current evaluation does not clearly demonstrate that neuron-level interpretability is the critical factor:
  - Scope and manual filtering: only five CIFAR-10 binary tasks, 100 generated images per task per run, and manual removal of images that do not match the intended semantics (“we manually remove images that include \(c_2\) objects or do not include \(c_1\) objects”). This manual curation blurs how “mass-production” and success rates (Table 3) would look in a fully automatic setting.
  - Complex generative stack: the pipeline chains words \(W_v\) → LLM → text-to-image model → CLIP classification. Failures can come from any stage. Baselines (“random neurons”, “indirect effect”, “similar words”) differ both in how words are chosen and in how those words interact with the LLM and generator. The higher success of the “second order” pipeline could arise because its word lists happen to be more visually concrete or more easily rendered, not because they uniquely expose true spurious neuron features.
  - Lack of human/heuristic controls: there is no comparison to simple hand-designed spurious prompts (e.g., adding plausible co-occurring objects or backgrounds) passed through the same LLM+generator stack. That would help isolate whether interpretability adds signal beyond intuitive prompt engineering.
  Taken together, the current results suggest that the second-order-derived words are *a* reasonably effective way to seed adversarial prompts, but they do not yet substantiate the stronger claim that mechanistic neuron interpretations are driving the attack’s advantage.

- **(5) The segmentation method does not clearly rely on the full interpretability pipeline.**  
  For zero-shot segmentation, the method uses:
  - Neuron directions \(r_n^l\) and text embeddings \(M_{\text{text}}(c_i)\) to select top neurons by absolute dot product.
  - Raw neuron patch activations to form attribution maps.  
  This already relies on the notion of a neuron-associated direction in joint space, but **not** on the sparse text decomposition. The experiments do not test:
  - Whether using second-order-based \(r_n^l\) is substantially better than using, e.g., first-order directions, token-level features, or simple correlations with class logits.
  - Whether performance is sensitive to the number of neurons (fixed at 200), choice of layers (8–10), or selection metric (absolute dot vs signed).
  Given this, it is unclear how much of the segmentation improvement derives from the second-order lens per se versus more generic late-layer, text-aligned neuron selection. The method is strong empirically, but the link to “interpreting neurons and then using that interpretation” is looser than claimed.

### Minor

- **Indirect effects comparison is somewhat narrow.**  
  The paper fairly notes that indirect effects suffer from self-repair (citing McGrath et al., 2023), and Table 1 shows that mean-ablating indirect effects in layer 9 of ViT-B/32 harms accuracy less and has lower PC#1 variance explained than second-order effects. However, the conclusion that indirect effects “fail to capture the neurons’ function” is based on one ablation mode and one model/layer; more nuanced interventions (e.g., causal scrubbing variants) are not explored, so claims about “failure” should be slightly toned down to “less informative under this ablation protocol.”

- **Sparsity (<2% images) is not fully quantified across neurons.**  
  The paper states:  
  > “the second-order effect of each individual neuron is significant only for less than 2% of the images”  
  and supports this with a per-neuron experiment where ablating effects on high-norm vs low-norm images has different impact (Fig. 3 “w/o small norm” vs “w/o large norm”). However, the thresholding and statistics across the full neuron population are not clearly reported (e.g., distribution over neurons of the fraction of images where ablation matters). This makes the “<2%” claim somewhat heuristic.

- **Design choices lack ablation at the application level.**  
  Choices like:
  - top-100 neurons in layers 8–10 for adversarial generation,
  - top-200 neurons for segmentation,
  - pool constructions (10k vs 30k words vs ∼28k class descriptions),  
  are only indirectly justified (Fig. 4 for reconstruction accuracy) and not examined for their impact on adversarial success or segmentation performance.

### Trivial

- The use of “rank-1” language is stronger than the empirical numbers justify. Phrasing such as “approximately low-rank, with a dominant PC” would better match the reported 48.2% variance explained and minimal accuracy drop, without implying near-perfect rank-1 structure.

## Nice-to-Haves

- A quantitative evaluation of neuron descriptions: for example, given the sparse text set for a neuron, predict which images (from a held-out set) should have large second-order effect, and measure correlation or accuracy. Even a small-scale human study or automatic proxy would greatly strengthen the “automated interpretation” claim.
- More detailed analysis of how often neurons appear clearly polysemantic vs monosemantic, e.g., by measuring semantic diversity among top-k phrases per neuron or via clustering.
- Sensitivity analyses for segmentation: vary the number of neurons, the layers included, and the threshold; show if the method is robust or requires careful tuning.

## Removed Points

These points are flagged to be removed or de-emphasized; treat them with caution, as they either overreach given the paper’s content or reflect misunderstandings.

- **“Second-order effect is not a mechanistic object because it ignores layernorm and other components.”**  
  While it is true that Eq. (5) ignores layernorm in the main derivation, the authors explicitly say:  
  > “Throughout the paper, we ignore layer-normalization terms to simplify derivations. We address layers-normalization in detail in Appendix A.6.”  
  and frame their lens as focusing on the value pathway through MSAs. This makes the definition a controlled approximation rather than an unacknowledged flaw. The issue is already captured above as overclaiming “total contribution”; it does not make the construct inherently invalid.

- **“Indirect effect comparison is unfair because they did not explore alternative interventions.”**  
  The harsh version suggests this as a methodological error. The paper never claims to exhaust all possible indirect-effect designs; it compares a standard mean-ablation approach and shows a clear difference (Table 1). Demanding exploration of every causal intervention variant is outside the paper’s scope; the fair criticism is just that the conclusion should be slightly softer, already noted under Minor.

- **“The sparsity (<2% images) claim is unsupported.”**  
  The paper does run a concrete experiment (selecting images with large vs small \(\|\phi_n^l(I)\|\) norms and ablating) and demonstrates the selective impact (Fig. 3). While the global “each neuron <2%” summary is somewhat compressed, this is not a fatal or clearly incorrect statement; the concern is about precision, already mentioned under Minor.

- **“Zero-shot segmentation is unrelated to the second-order lens.”**  
  This overstates things: segmentation *does* use the learned neuron directions \(r_n^l\) (which come from second-order effects) to select neurons per class. The valid criticism is that the method does not require the *textual sparse decompositions*, not that it ignores second-order analysis altogether.

## Novel Insights

The paper’s most genuinely new insight is that, for CLIP ViT image encoders, tracing neuron influence through later attention-value paths yields a signal that is (a) concentrated in late layers, (b) significantly more low-rank than standard indirect effects, and (c) sufficiently structured to support simple, text-aligned segmentation that outperforms more elaborate attribution schemes. This suggests that a relatively narrow slice of the network—the late MLP→MSA-value→class-token path—encodes a surprisingly interpretable and actionable portion of the model’s decision circuitry, even if that slice does not capture the neuron’s full causal role.

## Suggestions

- Clarify the framing of the second-order lens: explicitly present it as a *value-path-based approximation* to neuron influence, and avoid language like “total contribution.” Briefly discuss why you expect this pathway to be informative (e.g., empirical evidence that late value paths dominate logit movement) and what is plausibly missed by ignoring Q/K and later MLPs.
- Strengthen rank-1 claims by adding neuron-level statistics:
  - Distribution over neurons of variance explained by PC#1.
  - Perhaps 2–3 example neurons where PC#2 also carries substantial, qualitatively distinct structure, to show limitations.
  Replace “rank-1” phrasing with “strong dominant principal component” where appropriate.
- Add at least one quantitative evaluation of neuron descriptions:
  - For a subset of neurons, use their sparse text sets to predict which of a held-out sample of images fall in the neuron’s “high-effect” set, and report correlation/precision.
  - Alternatively, have annotators judge whether the top descriptions plausibly describe the common content of high-\(\|\phi_n^l(I)\|\) images.
- For adversarial examples:
  - Report the proportion of generated images discarded at each manual filtering stage and, if feasible, provide “fully automatic” success rates without manual curation.
  - Add a control baseline that uses hand-picked plausible spurious cues or random visually concrete words, passed through the same LLM+generator stack, to test whether neuron-derived cues add signal beyond generic prompt engineering.
- For segmentation:
  - Provide ablations over number of neurons, layers used, and selection rule (signed vs absolute dot), to better isolate which parts of the design contribute most to performance.
  - Compare using second-order-based \(r_n^l\) vs simpler alternatives (e.g., first-order effects or class-conditional gradients) as neuron selection features.

On standard conference axes:  
- **Originality:** High in terms of defining the second-order lens for neurons and linking it to text decompositions and downstream tasks.  
- **Importance of question:** Moderate-to-high; mechanistic understanding of CLIP and feature-based vulnerabilities is an active and impactful area.  
- **Support for claims:** Mixed. Empirical characterization and segmentation are solid; the strongest interpretability and causal claims are under-supported.  
- **Soundness of experiments:** Generally sound but somewhat narrow (single main model; small adversarial evaluation; limited ablations).  
- **Clarity:** Good overall; technical derivations and experimental setups are understandable.  
- **Value to community:** Positive, especially as an exploratory mechanistic tool and as an improved segmentation method, but the interpretability narrative should be tempered.

## Score and Decision

### Calibration

I compared this paper against several human-reviewed works:

- **5Ca9sSzuDp (“Interpreting CLIP’s Image Representation via Text-Based Decomposition”; scores 8,8,8,8)**  
  A very strong, tightly validated interpretability paper about CLIP’s attention heads and tokens, with extensive experiments and clearer alignment between decompositions and behavior. The current paper is less mature empirically (narrower evaluations, weaker validation of descriptions).

- **imT03YXlG2 (PatchSAE for CLIP; scores 6,6,6,8)**  
  Uses sparse autoencoders for CLIP features with richer ablations and clearer comparative baselines; still some open questions but overall stronger in methodological robustness than the present paper.

- **Rnxam2SRgB / 01ep65umEr (neuron description methods; scores mostly 5–6, often rejected)**  
  These share challenges around validating neuron descriptions and avoiding cherry-picked examples. The present paper’s validation is at a similar or slightly better level (due to segmentation and adversarial applications), but still not at the bar of the best-accepted work.

- **JCCPtPDido (jet expansions; scores 5–6, mostly rejected)**  
  Conceptually interesting expansion of residual computation with questions about approximation faithfulness. The current paper is in a similar category: interesting construct, but key approximation/faithfulness questions not fully answered.

Relative to these anchors, this submission is clearly stronger than the weakest neuron-description papers that received 3–5, because it delivers a solid segmentation method and a well-specified new lens. However, it falls short of the 7–8 range occupied by the best CLIP interpretability and sparse-feature works, mainly because its interpretability and adversarial claims are not yet rigorously substantiated.

A fair calibrated score is **6.0**: promising and valuable, but with significant evidential and framing issues that, in my judgment, keep it below the typical acceptance bar for a top-tier venue this year.

Given that many recent mechanistic interpretability papers with comparable or slightly stronger validation have been rejected or borderline (scores ~5–6), and factoring in the relatively weak adversarial evaluation and overstatements about “total contribution,” my meta-level decision leans to rejection at this stage, with encouragement to resubmit after strengthening the empirical support and tightening claims.

MY FINAL SCORE: <pineapple>6.0</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>