Now I have sufficient calibration data. Let me write the final review.

## Summary

TopoLM introduces a transformer language model with an explicit 2D spatial arrangement of units, trained with a combined next-token prediction and spatial smoothness loss. The spatial smoothness loss encourages nearby units to have correlated activations, mimicking cortical wiring-length minimization. Without any fitting to brain data, TopoLM develops language-selective clusters, verb/noun-selective clusters, and—most strikingly—replicates the empirical finding that concrete words produce verb/noun clustering while abstract words do not. The paper extends the TDANN spatial smoothness principle from vision to the language system.

## Strengths

- **Clean, minimal model design**: Adding spatial positions and a smoothness loss to a standard transformer, with all other components held constant and a non-topographic baseline (α=0) with identical architecture, makes attribution straightforward (Section 3). This is the right experimental design for isolating the effect of spatial smoothness.

- **Concrete>abstract verb-noun asymmetry is a genuinely non-trivial, compelling result**: TopoLM reproduces the Moseley & Pulvermüller (2014) finding—strong verb/noun clustering for concrete words (I=0.80) but not abstract words (I=0.23, p<0.001)—while the non-topographic baseline shows no such difference (I=0.11 vs. 0.12, p>0.05) (Figure 4b). This is not trivially predicted by the spatial loss alone and demonstrates that spatial smoothness interacts with the representational geometry to produce content-dependent organization matching empirical brain data.

- **Rigorous neuroimaging methodology**: The paper properly uses FDR correction, functional localizers, Moran's I with Queen contiguity, and conventional stimulus contrasts from the neuroscience literature (Sections 4-5). This is considerably more rigorous than typical ANN-brain comparison work.

- **Extension of TDANN principle from vision to language**: The paper provides evidence that the spatial smoothness principle (Margalit et al., 2024), previously established for visual cortex, also drives brain-like organization in the language system (Section 7). This conceptual contribution—suggesting a single cortical principle across domains—is meaningful.

- **Task performance is largely preserved**: The spatial loss comes at modest cost (BLiMP −5pts, Brain-Score −2pts) while GLUE improves (+3pts), suggesting the spatial organization is achievable without severely compromising model utility (Table 1).

## Weaknesses

### Fatal
None.

### Major

- **Moran's I evaluation is partially circular, and the paper's strongest quantitative claim (I=0.81 vs. 0.96) conflates generic smoothness with brain-specific topology**: The spatial smoothness loss directly encourages spatial autocorrelation of activations, and Moran's I measures spatial autocorrelation of derived contrast maps. Showing high Moran's I for a model trained for spatial smoothness is partially expected by construction. The critical question is whether the *specific* spatial organization (not just its degree of smoothness) matches the brain. The paper evaluates degree-of-clustering (Moran's I) but not topological similarity (e.g., whether cluster boundaries, shapes, and relative positions match). The non-topographic baseline's jump from I=0.11 to I=0.60 with fMRI sampling (Appendix Figure 10) further illustrates how much of the quantitative gap is driven by the smoothing process rather than by meaningful topographic structure. The headline I=0.81 (with sampling) vs. 0.96 (brain) comparison is therefore less informative than it appears; a more discriminating evaluation would test whether brain-*unlike* dimensions also cluster at comparable Moran's I values. This does not invalidate the paper, but it significantly weakens the main quantitative brain-comparison claim.

- **Overclaimed framing: "successfully predicts" and "driven by a unified spatial objective"**: The abstract states TopoLM "successfully predicts the emergence of a spatially organized cortical language system" and that results "suggest" organization is "driven by a unified spatial objective." Three of the four demonstrated properties (language-selective clusters emerging, consistent response profiles across clusters, verb-noun clusters for any representational dimension along which stimuli differ) are expected consequences of adding spatial smoothness to any model with semantic content. The one genuinely non-trivial prediction—the concrete>abstract asymmetry—is the paper's strongest contribution, but it is presented as one result among many rather than as the core finding. The claim of "successfully predicts" implies confirmed novel predictions, which overstates the evidence. Additionally, Brain-Score slightly *decreases* for TopoLM (0.78 vs. 0.80), which is somewhat paradoxical if the organization is truly "brain-like"—the paper acknowledges this but does not fully resolve the tension (Section 6, Table 1). The framing should be moderated to match what is demonstrated.

- **No negative control for non-brain-like dimensions**: The paper does not test whether dimensions that should NOT correspond to brain organization (e.g., word frequency bins, syntactic subcategory, letter count) also produce spatially smooth clusters with comparable Moran's I. This is the essential control for distinguishing "spatial smoothness produces brain-like organization" from "spatial smoothness produces smooth clusters for any dimension of variation in the representations." Without this, the matches to brain data could be generic consequences of smoothness rather than specific predictions.

### Minor

- **fMRI readout sampling parameters lack sensitivity analysis**: The FWHM=2.0mm and unit distance=1.0mm parameter choices have a substantial effect on the main results (I=0.48→0.81 for TopoLM; I=0.11→0.60 for baseline). The paper reports results both with and without sampling, which is good, but varying these parameters would determine how robust the brain-comparison gap is to this degree of freedom. This matters because the primary quantitative claim (I=0.81 approaching brain's 0.96) depends on these choices.

- **Imperfect response profile match is underexplored**: Figure 2c shows TopoLM's core language system response profile has sentences ≈ unconnected words > jabberwocky > nonwords, whereas the brain shows sentences > {unconnected words, jabberwocky} > nonwords. The paper attributes this to a "general shortcoming of the base transformer model" and notes the baseline also fails here. This is fair, but it means the spatial loss does not improve this critical functional pattern — a notable gap in the "brain-prediction" narrative that deserves more discussion.

- **Topoformer-BERT comparison confounds**: The comparison with Topoformer-BERT involves different architecture, training data, scale, and objective. The paper acknowledges this partially (calling it "a baseline, but not a control") but still uses the comparison to support claims about spatial smoothness vs. local connectivity. The confounds limit what can be concluded from this comparison.

## Nice-to-Haves

- **Representational geometry analysis for the concrete/abstract asymmetry**: Measuring the functional (non-spatial) separability of verb-noun contrasts for concrete vs. abstract words in the non-topographic baseline (e.g., via RSA or decoding) would establish whether the clustering asymmetry follows from inherent transformer representations or is genuinely novel.

- **Novel model-generated predictions**: The paper claims TopoLM is "stimulus-computable" and can "discover new candidates for spatial clustering" (Section 7). Showing specific novel predictions that go beyond post-hoc matching of existing brain datasets would substantially strengthen the contribution.

- **Dataset-by-dataset Brain-Score breakdown**: The overall Brain-Score decrease (0.78 vs. 0.80) would benefit from analysis of which datasets drive the decrease, and whether TopoLM outperforms on spatially-sensitive subsets.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Validation loss gap as a critical issue**: The harsh critic raises the 3.075 vs. 2.966 validation loss gap as a "meaningful degradation." This is modest and is acknowledged in the paper's BLiMP/Brain-Score analysis. The paper also shows GLUE improves. This is not a standalone major weakness.

- **α=2.5 without reported sensitivity**: The critic notes no data shown for the α tuning claim in Footnote 4. This is a minor point—hyperparameter selection without published sensitivity analysis is standard practice, and the paper does report both the topographic and non-topographic models for comparison.

- **No error bars on Table 1**: The critic notes this, but single-run evaluation is the norm for this type of comparison, and the differences are small enough that variance would likely not change the conclusion.

- **Random spatial permutation across layers means no coherent cortical sheet**: The paper acknowledges this in Section 7 ("Limitations"). This is an inherent design trade-off, not an oversight. The model is not meant to be a full cortical sheet simulation.

- **Grammar/typos/formatting issues**: Removed per hard rules.

## Novel Insights

The concrete>abstract verb-noun asymmetry result reveals something genuinely interesting: spatial smoothness is not a content-blind operation—it interacts with the intrinsic representational geometry of the underlying language model. Concrete words, which have richer sensory-motor grounding in the transformer's learned representations, produce separable verb/noun clusters under spatial smoothness, while abstract words do not. This suggests that cortical topographic organization may preferentially emerge for representational dimensions that are already more spatially or semantically separable in the underlying functional code—a "smoothness amplifies pre-existing structure" principle that goes beyond the simple claim that smoothness alone produces brain-like maps.

## Suggestions

- Add a negative control experiment: test whether non-brain-relevant dimensions (e.g., word frequency bin, orthographic length) also produce high Moran's I clusters in TopoLM. If they do, the brain matches are generic; if they don't, this substantially strengthens the paper's core claim.

- Moderate the abstract framing: replace "successfully predicts" with "is consistent with" or "reproduces," and replace "driven by a unified spatial objective" with "consistent with a spatial smoothness principle playing a role in" — the evidence supports that spatial smoothness is a contributing factor, not that it is the sole driving principle.

- Report the concrete>abstract asymmetry as the paper's headline finding rather than burying it among more expected results — it is the only result that cannot be trivially derived from the model design.

## Calibration Comparison

**Anchors examined:**
- THqWPzL00e (TopoNets): avg 7.50, Accept (Spotlight) — very similar paper (TopoLoss for vision+language, brain-like topography). TopoNets had cleaner baselines, broader architecture coverage, and less overclaim, but less depth on specific neuroscience predictions.
- rySLejeB1k (Emergent Orientation Maps): avg 7.33, Accept (Spotlight) — topographic V1 model with emergent pinwheels, strong mechanistic result.
- lm4PYXBId5 (Video model-brain RSA): avg 7.00 — large-scale benchmarking, solid but incremental.
- eoB6JmdmVf (Speech/text LMs brain alignment): avg 4.75, Reject — limited novelty, some analytical gaps.
- hbon6Jbp9Q (LM per brain region): avg 2.33, Withdrawn — no novel neuroscience insights, methodological weaknesses.
- InRaT76E2S (Activation Decay): avg 2.50, Withdrawn — overclaimed spatial smoothness generalization with flawed proofs.
- LD0qz8j8Zm (Brain-inspired ESN sub-circuits): avg 4.0, Withdrawn — partially circular evaluation, design consequences as insights.

TopoLM sits between the high-scoring TopoNets (7.5) and the medium-scoring brain-ANN papers (~4-5). It has a genuinely compelling non-trivial result (concrete>abstract asymmetry) and a clean model design, but its main quantitative evaluation has partial circularity, its framing overclaims, and it lacks critical negative controls. Compared to TopoNets (7.5), TopoLM is weaker because TopoNets had broader validation, less overclaim, and better baselines. Compared to medium-scoring brain-ANN papers (~4-5), TopoLM is stronger because it has a clean non-trivial result and rigorous neuroimaging methodology. The concrete>abstract finding, combined with the rigorous methodology, places it above borderline but the circularity and overclaim pull it down from TopoNets territory.

**Score: 5.5** — Accept-level quality on the strength of the concrete>abstract result and clean model design, but significant concerns about evaluation circularity, overclaiming, and missing negative controls prevent a higher score.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>