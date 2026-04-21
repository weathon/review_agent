Now I have sufficient information to write the final review. Let me compile it.

## Summary

The paper introduces the "second-order lens" for interpreting individual MLP neurons in CLIP-ViT, analyzing the pathway from a neuron through subsequent attention heads to the output representation. The authors characterize these second-order effects as concentrated in late layers, sparse (significant for <2% of images), and approximately rank-1, enabling a sparse text decomposition that reveals polysemantic neuron behavior. Two downstream applications are demonstrated: semantic adversarial attacks exploiting neuron polysemy and improved zero-shot segmentation over prior work.

## Strengths

- **Well-motivated and principled derivation of second-order effects.** The paper clearly articulates why first-order effects (near-constants in CLIP, per Gandelsman et al., 2024) and indirect effects (obscured by self-repair, Section 3.2, Table 1) are inadequate for neuron interpretation, and Equation 5 provides a clean decomposition of the second-order effect into attention-weighted (input-dependent) and OV-circuit (input-independent) terms.

- **Strong empirical characterization of second-order effect properties.** Table 1 directly demonstrates advantages over indirect effects: 48.2% vs. 11% variance explained by the first PC, and a 29.6% accuracy drop (closer to zero = larger effect) vs. 52.3% upon mean-ablation at layer 9. Figure 3's "w/o large norm" vs. "w/o small norm" lines effectively demonstrate the sparsity property, and "rec. from PC #1" shows the rank-1 approximation preserves accuracy.

- **Creative adversarial attack application.** The pipeline in Section 5.1 that uses discovered neuron polysemy to find spurious concept overlaps and generate semantic adversarial images is a genuinely interesting connection between interpretability and robustness. Table 3 shows the second-order method uniquely succeeds on ship→truck (5.7% vs. 0% for all baselines) and substantially outperforms on dog→deer (22.7% vs. 6.3%).

- **Qualitative evidence supports interpretability of decompositions.** Tables 2 and 5, and Figure 5, show alignment between discovered text descriptions and top-activating images that is visually compelling and consistent with the claimed polysemantic behavior.

- **Results generalize across model sizes.** The paper validates key findings (layer concentration, sparsity, rank-1 approximation) on both ViT-B-32 and ViT-L-14, and on ImageNet-R in addition to ImageNet (referenced in Section 3.3 and Appendix A.1).

## Weaknesses

### Fatal
None.

### Major

- **No causal validation of individual neuron interpretations.** The paper defines second-order effects as attributions with frozen attention weights (Equation 5), which is a linearized decomposition, not an interventional one. Mean-ablation experiments (Figure 3) show that second-order effects are important *in aggregate*, but individual neuron interpretations (e.g., "this neuron writes toward yachts and cabriolets") are validated only qualitatively. There is no experiment that intervenes on a specific neuron and verifies the predicted downstream effect (e.g., suppressing a "yacht"-writing neuron decreases similarity to yacht images). This gap means the claim that the second-order lens reveals "neuron functionality" relies on correlation-based attributions rather than causal evidence. The paper acknowledges this implicitly by noting it ignores neuron effects on queries/keys (Section 6), but the more fundamental issue is the lack of per-neuron causal validation.

- **The sparse text decomposition is validated only by reconstruction accuracy, not semantic correctness.** Section 4 evaluates the decomposition entirely by whether replacing φ_n^l(I) with the text-reconstructed version preserves downstream ImageNet accuracy (Figure 4). This tests sufficiency for the task, not interpretability of the discovered descriptions. A decomposition that assigns effects to arbitrary orthogonal text directions with appropriate magnitudes could preserve accuracy while being semantically uninformative. The paper provides no quantitative measure of whether images containing the described concepts actually activate the corresponding neurons at higher rates than random concepts. The Section 4 text mentions "these concepts correctly track which inputs activate a given neuron" (line 39), but the evidence for this is purely the qualitative match in Figure 5 and Table 2.

- **Manual post-hoc filtering in adversarial evaluation.** The paper reports adversarial success rates after "manually remov[ing] images that include c₂ objects or do not include c₁ objects" (Section 5.1). This filtering is applied to both the method and baselines, but it makes the reported success rates unrepresentative of the method's practical effectiveness: a significant fraction of generated images fail to contain the intended content, and the success rates are computed over only the images that pass manual curation. The absolute success rates (5.3–22.7% out of 100) are modest even after filtering, which raises questions about the "mass-production" framing.

### Minor

- **The 48.2% variance explained by the first PC is only moderate.** While the rank-1 approximation preserves classification accuracy (Figure 3, "rec. from PC #1"), classification accuracy is a coarse metric that can be robust to substantial noise. The remaining 51.8% of variance is uncharacterized—it could contain interpretability-relevant structure that is being discarded. This deserves at least a brief discussion.

- **Segmentation improvement over TextSpan (the authors' own prior work) is modest.** The gains are +1.6 pixel accuracy, +0.9 mIoU, +0.8 mAP (Table 4). The paper does not compare against dedicated zero-shot segmentation methods, only CLIP attribution baselines. The improvement may simply reflect ensemble averaging over 200 neuron activation maps rather than a deeper insight from the second-order lens.

- **The abstract's claim of "new model capabilities" overstates the contributions.** The segmentation improvement is marginal over prior work, and the adversarial attack has low absolute yield. "New model capabilities" suggests capabilities the model did not previously possess, whereas both applications are better characterized as extensions of existing interpretability-driven techniques with modest quantitative improvements.

## Nice-to-Haves

- Causal validation of individual neuron interpretations via activation patching or path patching would substantially strengthen the core claim.
- Quantitative evaluation of text description accuracy (e.g., measuring whether described concepts predict neuron activation patterns on held-out data).
- Reporting adversarial results both with and without manual filtering, or providing the filtering rate so readers can assess what fraction of generated images are discarded.
- Comparison against dedicated zero-shot segmentation methods (not just attribution-based ones) in Table 4.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"The second-order effect is an attribution, not a validated causal decomposition"** — While the lack of per-neuron causal validation is a real concern (kept as Major), the harsh critic's framing that this "assumes attention weights are fixed, ignoring the neuron's effect on subsequent queries/keys" conflates the approximation with an error. The paper explicitly acknowledges this in Section 6 and the frozen-attention approach is standard in circuit analysis (OV-circuit analysis per Elhage et al., 2021). The issue is lack of validation, not that the approach is fundamentally flawed.

- **"The indirect effects comparison confounds the probing method with the quantity being probed"** — The mean-ablation vs. variance-explained comparison in Table 1 is a reasonable way to show that second-order effects have more interpretable structure than indirect effects. Asking for alternative interventions (resampling ablation) is a nice-to-have, not a weakness, since the paper's goal is to motivate the second-order lens rather than exhaustively study indirect effects.

- **"The difference between using single words vs. descriptions... raises concerns about whether the decomposition is discovering genuine structure or overfitting to the dictionary"** — Figure 4 shows that different pools converge at large m, which suggests stability rather than overfitting. The critic's concern is speculative and partially contradicted by the paper's own evidence.

- **"Requesting comparison against alternative probing methods for indirect effects"** — This is outside the paper's scope; the paper introduces the second-order lens, not a comprehensive comparison of all possible indirect-effect probes.

- **Criticisms of abstract/introduction "overclaiming"** — The actual paper language ("our results indicate that an automated interpretation of neurons can be used for model deception and for introducing new model capabilities") is moderate. The phrase "new model capabilities" in the abstract is the strongest claim; I've addressed it as a minor weakness above, but it does not invalidate the paper.

- **"Remove the manual filtering step" as a major demand** — This is methodologically relevant but the harsh critic elevates it too far. Both the method and baselines undergo the same filtering, so the comparison is fair even if absolute rates are hard to interpret. I've kept this as a minor weakness.

## Novel Insights

The paper's observation that second-order effects reveal polysemantic neuron behavior that can be *directly exploited* for semantic adversarial attacks is a genuinely useful connection: it suggests that interpretability tools don't just explain models but can uncover attack surfaces. The asymmetric success rates across tasks (22.7% for dog→deer vs. 5.3% for horse→automobile) hint that some class pairs share more neuron-level structure than others—an insight that could guide both robustness research and further interpretability work.

## Suggestions

- Run a targeted causal validation: for a small set of neurons, suppress the neuron's activation and measure whether the predicted class similarity drops for the text descriptions found in the decomposition. Even 5–10 neurons would transform the paper's claim from "the decomposition preserves accuracy" to "the decomposition predicts causal effects."
- Report the manual filtering rate (what fraction of images were removed) in the adversarial experiments, and ideally provide results both with and without filtering.
- Tone down "mass-production of semantic adversarial examples" to "generation of semantic adversarial examples" given the modest absolute success rates and manual curation involved.

## Evaluation

**Originality:** The second-order lens is a natural and technically sound extension of the logit lens to neurons, well-motivated by the failure of direct and indirect effects. The connection to adversarial attacks is creative. The sparse text decomposition methodology is relatively standard (OMP with a text dictionary), but the application to neurons via the rank-1 second-order direction is novel.

**Importance of research question:** Understanding neurons in CLIP is an important and timely question, and the paper makes a meaningful contribution by providing a tractable framework for it.

**Claims supported:** The empirical characterization of second-order effects is well-supported. The claims about individual neuron functionality and the semantic correctness of text decompositions lack causal/validation evidence, making them the paper's main vulnerability.

**Soundness of experiments:** The experimental setup is reasonable but has notable gaps: no per-neuron causal validation, no quantitative evaluation of text description accuracy, and manual filtering in adversarial experiments.

**Clarity:** The paper is well-written with clear derivations and good figures. The core ideas are presented accessibly.

**Value to community:** The second-order lens framework and the demonstrated connection between interpretability and adversarial robustness are valuable contributions that will likely spur further work.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| TextSpan (Interpreting CLIP's Image Representation) | 5Ca9sSzuDp.md | 8.00 | This paper's prior work on CLIP attention heads; the current paper extends from heads to neurons, a harder problem. Similar scope but more novel at time of publication. The current paper is a clear followup with a genuine advancement, but with weaker validation. |
| Sparse Feature Circuits | I4e82CIDxv.md | 8.00 | Directly comparable in spirit: uses SAE-based features for causal circuits in LMs. Has stronger causal validation (ablation of features → measurable behavioral change). More mature causal methodology than our paper. |
| Scaling and Evaluating SAEs | tcsZt9ZNKD.md | 8.20 | Much larger scale and more thorough evaluation; sets the bar for high-quality interpretability work. Our paper is below this in rigor. |
| Mechanistic basis of ICL | aN4Jf6Cx69.md | 9.00 | Very rigorous mechanistic analysis with phenomenological models. Much deeper causal analysis. Not directly comparable in topic. |
| TeLLMe: LLMs Explain Neurons | 01ep65umEr.md | 5.25 | Uses LLMs to explain neurons in vision models. Our paper has a more principled methodology but similarly lacks strong causal validation. Our paper is somewhat stronger. |
| Summing Up the Facts (Additive motif) | P2gnDEHGu3.md | 5.25 | Mechanistic analysis of LLMs using direct logit attribution, similar in spirit but reviewers found insufficient depth. Our paper has more applications but similar validation gaps. |
| TRACER | PoB6QGAM38.md | 3.00 | Claims causal explanations without causal experiments. Our paper is significantly better than this—it does have mean-ablation validation and qualitative support—but shares the pattern of attributions presented as functional roles. |
| Chess SAE | Wxl0JMgDoU.md | 2.50 | SAE-based interpretability limited only to threat response. Our paper has broader scope and more applications. |

The paper sits between the high-scoring interpretability papers (8+, which have causal validation or massive scale) and the medium-low papers (5, which have principled ideas but insufficient validation). The lack of per-neuron causal validation is the main separator from the 8-range papers, while the solid empirical characterization and creative applications separate it from the 5-range papers. The closest comparison is with TeLLMe (5.25) and Summing Up the Facts (5.25), which have similar validation gaps, but our paper has stronger methodological grounding and more thorough characterization. TextSpan (8.0) is the most topically similar, but it had the advantage of being first. Given the paper sits above the medium-low cluster but below the high-scoring causal-validated papers, a score in the 6-7 range is appropriate.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>