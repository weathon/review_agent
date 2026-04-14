## Summary
TopoLM augments a GPT-2-style transformer with a spatial smoothness loss (joint optimization with cross-entropy) that induces brain-like topographic organization without fitting to neural data. The model spontaneously produces language-selective clusters whose response profiles and verb/noun selectivity—including the null result for abstract stimuli—match patterns observed in human fMRI, extending the TDANN principle from vision to language. Performance on downstream tasks (GLUE) improves, while BLiMP and Brain-Score show minor degradation.

---

## Strengths

- **Emergent brain-like clustering without neural supervision.** The model is trained solely on text; the agreement with fMRI-measured verb/noun topography (Moran's I = 0.81 vs. 0.11 for non-topographic baseline, brain data I = 0.96) arises purely from the combined task + spatial loss. This is a non-trivial and well-controlled demonstration.

- **Replication of a specific null result (concrete > abstract verb/noun dissociation).** TopoLM reproduces the Moseley & Pulvermüller (2014) finding that verb/noun clustering exists for concrete words (I = 0.80) but not abstract words (I = 0.23, p < 0.001), while the non-topographic baseline shows no such difference (I = 0.11 vs. 0.12). Predicting a null result with a quantitative model is far harder to dismiss as coincidence than a positive correlation.

- **Spatial smoothness outperforms local connectivity (Topoformer-BERT) for brain-relevant clustering.** Topoformer-BERT shows high unthresholded Moran's I (0.66) but the clusters do not reach significance for the verb/noun contrast, whereas TopoLM's clusters are significant and match the anatomical arrangement. This controlled comparison establishes that the specific mechanism (smoothness loss) matters.

- **Explicit falsifiable criteria set up a priori.** The three success criteria for the core language system (distinct clusters emerge, consistent profiles across clusters, profiles match brain) make the evaluation principled rather than post hoc. The honest acknowledgment that criterion 3 is only partially met, and the demonstration that the non-topographic model fails equally, is methodologically transparent.

- **GLUE improvement (+3 pts) over the non-topographic baseline.** Unlike topographic vision models that sacrifice task performance, TopoLM improves on downstream GLUE benchmarks, consistent with the spatial loss acting as regularization.

---

## Weaknesses

- **Potential circularity between training distance metric and evaluation metric.** The spatial loss uses ℓ∞ norm to define unit neighborhoods during training (Eq. 1). The evaluation of clustering uses Moran's I with Queen contiguity, which is defined by the same ℓ∞ radius. The model is optimized for a criterion that is then used, through the same distance geometry, to measure its success. This does not invalidate the brain-alignment results (which use independent fMRI data), but it means the Moran's I score on model activations alone is not an independent validation of emergent organization. The authors should either use a different distance metric for Moran's I evaluation, or explicitly justify why this alignment is not circular.

- **fMRI sampling substantially inflates both models' clustering scores, compressing the apparent advantage of topography.** Before sampling, the gap between TopoLM and the non-topographic baseline is I = 0.48 − 0.11 = 0.37. After applying fMRI-like sampling to both, the gap shrinks to I = 0.81 − 0.60 = 0.21, and the non-topographic baseline with sampling (0.60) already exceeds unsampled TopoLM (0.48). This information appears in Figure 3B and Appendix Figure 10, but the narrative in the main text emphasizes "topo + sampling" without adequately flagging that sampling substantially closes the gap. The remaining distance to brain data (I = 0.96 vs model I = 0.81) is also unaddressed—no statistical test is provided for whether this gap is significant.

- **No sensitivity analysis for α = 2.5.** The paper states α was chosen after "extensive hyperparameter search," but the reader cannot evaluate how brittle this choice is. Given that α directly trades task performance against spatial organization, and given that BLiMP already drops 5 points, a plot of (BLiMP, Brain-Score, Moran's I) across a range of α values is essential. Without this, it is unclear whether the chosen α is a principled sweet spot or a cherry-picked equilibrium.

- **BLiMP degradation is non-trivial.** A 5-point drop (0.76 → 0.71) is presented as "slight," but on a scale from chance (~0.50) to near-ceiling (~0.85 for this model class), this represents a meaningful loss of grammatical knowledge. Some BLiMP subtasks may show larger drops masked by averaging. This is not fatal but should be reported at the subtask level to allow assessment.

- **No multi-seed reproducibility analysis.** The spatial positions of clusters are not verified across independent runs. If cluster locations vary randomly by seed, the claim that TopoLM predicts the "canonical" organization of the brain's language system is substantially weakened—what is being matched may depend on luck of initialization rather than a systematic emergent property. At a minimum, Moran's I statistics and the concrete/abstract dissociation should be reported across multiple seeds.

- **No statistical significance testing for cross-model differences in Table 1.** Single-point estimates are given for BLiMP, GLUE, and Brain-Score with no confidence intervals, standard deviations, or significance tests across seeds or GLUE tasks. The 2-point Brain-Score difference and 3-point GLUE difference could plausibly be within run-to-run variability.

- **Brain-Score null result deserves more discussion.** TopoLM achieves 0.78 vs. 0.80 for the non-topographic baseline. The paper claims "functionally and spatially aligned model of language processing" in the abstract, but the functional alignment (measured by Brain-Score ridge regression) is slightly *worse* with topography. This is an important null result: the spatial inductive bias does not improve functional alignment as measured by encoding models. The paper acknowledges this, but a discussion of why (e.g., spatial loss compresses representational geometry in ways that reduce linear decodability) would strengthen the paper.

---

## Nice-to-Haves

- **Ablation of spatial loss during GLUE fine-tuning.** Fine-tuning uses the same α = 2.5 spatial loss; a simple condition without the spatial loss during fine-tuning would directly test the regularization hypothesis for the GLUE improvement.

- **α sensitivity curve.** Even a simple line plot of BLiMP / Brain-Score / Moran's I vs. α would confirm the chosen value lies on a reasonable Pareto frontier rather than a local optimum.

- **Neighborhood size and number ablation table.** Footnote 5 claims little effect, but quantifying this claim (e.g., a 2×2 table of neighborhood size × count) would reassure the reader that the efficiency approximation does not sacrifice quality.

- **Mechanistic explanation for concrete > abstract clustering.** Why would spatial smoothness specifically produce greater clustering for concrete than abstract words? A brief analysis (e.g., whether concrete words cluster more tightly in embedding space or exhibit more consistent co-occurrence patterns) would add scientific depth.

- **Held-out stimulus generalization.** Testing cluster selectivity on semantic categories not used during localization would rule out potential overfitting to the specific Fedorenko et al. (2010) localizer stimuli.

- **Reporting computational overhead.** Training time and memory overhead relative to the non-topographic baseline are not reported; these would help practitioners.

- **Seed overlap visualization.** Overlaying language-selective unit maps across a few independent runs to assess spatial consistency is low cost and directly answers the reproducibility question.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Confounded Topoformer-BERT comparison (Review 2)]:** The paper explicitly and prominently states "Topoformer-BERT is a *baseline*, but not a control" (Section 3) and lists the architectural differences (corpus size, attention heads, bidirectionality, training objective). The comparison is correctly framed and the asymmetry is not a flaw.

- **[Clustering algorithm is "ad hoc" (Harsh Critic)]:** The evolutionary greedy clustering algorithm is a reasonable and common approach in neuroimaging (starting from most selective unit, adding neighbors, applying FDR threshold). While parameter choices are not ablated in the main text, calling this "ad hoc" mischaracterizes a standard procedure. The threshold justification is a nice-to-have at most.

- **[Moseley & Pulvermüller comparison is "purely qualitative" (Harsh Critic)]:** The paper is transparent that no fMRI data is available for this dataset and explicitly describes it as qualitative corroborative evidence. The quantitative Moran's I analysis on model activations is provided (Figure 4B), and the paper does not treat this as "independent confirmation"—this criticism misreads the paper's framing.

- **[Spatial axes not anatomically grounded (Harsh Critic)]:** The paper does not claim to predict anterior-posterior or dorsal-ventral axes; the comparison is to functional selectivity patterns (verb vs. noun), not to anatomical coordinates. This is a genuine scope limitation, but criticizing TopoLM for lacking retinotopic-style grounding imposes a requirement outside the paper's stated scope. The Discussion's limitation paragraph honestly addresses this.

- **[Training loss gap of 3.7% is "non-negligible" without more analysis (Harsh Critic)]:** The downstream task performance (GLUE, Brain-Score) provides the relevant assessment of this gap's impact. Flagging the raw loss difference without contextualizing it against task outcomes is an incomplete criticism.

- **[Demanding theoretical proofs or wiring cost calculation (Spark Finder)]:** The paper is empirical and the wiring cost motivation is cited from Margalit et al. (2024). Requiring explicit wiring length computation is not a standard expectation for this type of systems/NeuroAI paper.

---

## Novel Insights

The most genuinely novel observation across the reviews, beyond the paper's stated contributions, is the interaction between fMRI readout sampling and the evaluation metric: the non-topographic model with sampling (I = 0.60) already substantially exceeds unsampled TopoLM (I = 0.48), suggesting that the observed advantage of topography is partly a measurement artifact of the fMRI blurring kernel rather than purely an emergent property of the architecture. This raises a deeper question: are the brain's observed verb/noun clusters as spatially sharp as the raw I = 0.96 suggests, or is the brain's high I itself partly an artifact of voxel smoothing over pre-organized tissue? If so, the model might need even sharper internal organization than currently measured to account for the brain's true fine-grained structure. The paper does not engage with this question, and doing so would strengthen the theoretical framing considerably.

---

## Suggestions

1. **Address the ℓ∞ circularity:** Report Moran's I with an alternative contiguity metric (e.g., Rook contiguity or ℓ¹ distance) as a sensitivity check; if results are robust, the circularity concern is substantially mitigated.
2. **Run at least 3–5 random seeds** and report mean ± std of Moran's I and the concrete/abstract dissociation; a seed overlap heatmap would also directly demonstrate whether cluster locations are stable.
3. **Add an α sensitivity plot** spanning at least one order of magnitude; mark the chosen α and show BLiMP, GLUE, Brain-Score, and Moran's I on the same axes.
4. **Report BLiMP subtask-level results** to identify which grammatical phenomena are most affected by the spatial loss.
5. **Provide a no-spatial-loss GLUE fine-tuning ablation** to test the regularization hypothesis for the +3 pt improvement.
6. **Discuss the Brain-Score null result more explicitly** in the main text: what does it mean that spatial topography adds no functional alignment improvement by encoding-model metrics?

---

## Evaluation

- **Novelty:** High. Extending the TDANN spatial-smoothness principle from vision to language is a genuine first; replicating a specific null result (concrete > abstract) in a forward model is a strong scientific move.
- **Technical soundness:** Moderate-high. The spatial loss formulation and training setup are sound; the main concerns are α sensitivity, seed stability, and the evaluation metric circularity.
- **Empirical support:** Good. Three datasets, one with quantitative fMRI comparison and one with a null result replication; the absence of neural supervision during training strengthens the inference. The lack of error bars and multi-seed results is a gap.
- **Significance:** High for NeuroAI. Provides a computational hypothesis for language system functional organization with testable predictions.
- **Clarity:** Good. The three-criterion framing is effective, and limitations are honestly discussed; some results (fMRI sampling gap compression) would benefit from more prominent presentation.

MY FINAL SCORE: <pineapple>7.2</pineapple>