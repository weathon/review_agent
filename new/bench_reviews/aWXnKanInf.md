Now I have all the information needed. Let me write the final consolidated review.

## Summary

TopoLM extends the TDANN spatial smoothness principle from vision to language by training a GPT-2-style transformer with a spatial correlation loss that encourages nearby units on a 2D grid to have correlated activations. The model develops emergent spatio-functional organization—including language-selective clusters and verb/noun-selective clusters—that qualitatively and quantitatively matches neuroimaging data. The most striking finding is that TopoLM replicates the concrete-but-not-abstract verb–noun clustering pattern from Moseley & Pulvermüller (2014), a result the non-topographic baseline completely fails to produce.

## Strengths

- **Concrete-abstract verb–noun result is genuinely non-trivial (Section 5.2, Figure 4).** TopoLM shows strong verb/noun clustering for concrete words (I = 0.80) but not abstract words (I = 0.23, p < 0.001), while the non-topographic baseline shows no difference (I = 0.11 vs. 0.12, p > 0.05). This cannot be explained by the spatial loss alone—spatial smoothness treats all inputs identically—so the difference must emerge from the interaction between task-learned representations and the smoothness constraint. Matching Moseley & Pulvermüller (2014) provides compelling evidence for the model's neuroscientific relevance.

- **Clean, well-motivated model design (Section 3, Eq. 1–2).** The spatial correlation loss transfers naturally from the TDANN vision framework, and the formulation avoids hard architectural constraints (unlike Topoformer's local connectivity), preserving full attention connectivity. The random permutation of spatial positions across layers (footnote 2) is a reasonable design choice to prevent degenerate solutions.

- **Emergence of language-selective clusters with consistent response profiles (Section 4, Figure 2A-B).** Multiple language-selective clusters emerge across the topographic tissue, and response profiles across clusters are consistent with one another, matching the uniformity observed across the brain's core language system subregions (Fedorenko et al., 2024).

- **Important research question.** Whether the spatial smoothness principle from vision neuroscience generalizes to language is a significant and timely question. Even if the evidence is incomplete, the approach and the question itself advance the field.

- **Informative Topoformer-BERT comparison (Section 5.1–5.2).** Topoformer-BERT shows high Moran's I (0.66–0.85) but few significant verb-noun-selective units, demonstrating that spatial autocorrelation alone does not imply brain-like functional organization. This helps isolate the contribution of spatial smoothness loss versus local connectivity constraints.

## Weaknesses

### Fatal
None.

### Major

- **Partial circularity between training objective and primary evaluation metric.** The spatial loss (Eq. 1) encourages nearby units to have correlated activations. Moran's *I*, the primary quantitative metric in Sections 5.1–5.2, measures spatial autocorrelation of contrast maps—which is a direct downstream consequence of the spatial smoothness the model was trained to achieve. High Moran's *I* after training with spatial loss is therefore partially tautological: optimizing for spatial smoothness yields spatial smoothness. The meaningful claim should be about what *semantically defined* clusters emerge, not just their spatial autocorrelation. The concrete-abstract finding partially addresses this (spatial smoothness alone cannot explain the concrete/abstract difference), but the primary quantitative comparisons in Figures 3B and 4B still rely heavily on Moran's *I* as the central metric. A model trained with a weaker smoothness constraint might produce the same semantic clusters with lower Moran's *I*, which would be equally brain-like in the sense that matters. The paper would be substantially strengthened by evaluating semantic cluster content independently of spatial autocorrelation—for example, through classification accuracy of cluster labels, or by comparing the *types* of errors the model makes versus the brain.

- **Failure to match the core language system's defining functional signature, despite claiming "functionally and spatially aligned" (Abstract).** The paper itself defines successful alignment as requiring sentences > {unconnected words, jabberwocky} > nonwords (Section 4, line 166). Figure 2C shows TopoLM's response profile is: sentences ≈ unconnected words > jabberwocky > nonwords. The sentences > unconnected words distinction indexes syntactic processing beyond lexical semantics and is the defining signature of the brain's language system. TopoLM does not capture this. The paper acknowledges the mismatch ("response profiles *mostly* match") and notes the baseline also fails, but the abstract still claims "functionally and spatially aligned model of language processing in the brain"—this is overstated. A model that fails the brain's most basic functional signature should not claim full functional alignment.

- **Overclaiming that the brain's functional organization is "driven by" the spatial smoothness principle (Abstract, Discussion).** The paper states results "suggest that the functional organization of the human language system is driven by a unified spatial objective" and "provides evidence that this principle of spatial smoothness indeed generalizes across cortex." The paper demonstrates *compatibility*: applying spatial smoothness to a language model produces organization interpretable as brain-like. It does not establish that the brain *uses* this principle—the causal direction is not established, and the evidence is consistent with multiple alternative mechanisms. The discussion's final sentence ("our results suggest that the spatial smoothness principle leads to topographic organization consistent with the spatio-functional organization") is more carefully hedged; the abstract should match this tone.

### Minor

- **fMRI readout sampling amplifies clustering substantially in both models, and the paper could be clearer about relative contributions (Figure 3B).** Without sampling, TopoLM achieves I = 0.48 vs. the baseline's I = 0.11; with sampling, I = 0.81 vs. 0.60. The sampling increases the non-topographic baseline's clustering 5.5× and TopoLM's 1.7×, meaning sampling accounts for a large share of the apparent "brain-like" clustering in both models. While the paper presents both conditions and the sampling is methodologically justified (fMRI voxels do smooth signals), the sampled results are presented as primary, and the paper could be more explicit about how much of the brain-like clustering is attributable to the spatial loss versus the sampling procedure.

- **Mean absolute activation as the response metric conflates baseline magnitude with stimulus selectivity (Section 4, line 172).** The brain data uses BOLD signal *change*—a differential measure—while the model uses mean absolute activation. A unit that is always highly activated will show high "response" regardless of stimulus. This methodological mismatch could contribute to the failure to distinguish sentences from unconnected words, and is worth addressing with a differential measure (e.g., activation relative to baseline) in future work.

- **No sensitivity analysis for α = 2.5 (Section 3).** The choice of α determines the balance between task performance (validation loss 3.075 vs. 2.966) and spatial organization, and thus shapes all downstream results. While the paper states α was chosen after "extensive hyperparameter search," no sensitivity analysis is provided, leaving the reader unable to assess whether the concrete-abstract finding is robust or an artifact of this specific configuration.

- **The "virtually no cost" characterization of performance trade-offs understates the BLiMP decrease (Section 7, Table 1).** A 5-point decrease on BLiMP (0.71 vs. 0.76) on a minimal-pair benchmark probing grammatical knowledge is non-trivial, as is the 2-point decrease on Brain-Score. "Virtually no cost" would be more defensible if qualified with a discussion of what these magnitudes mean for the benchmarks in question.

### Trivial
None significant.

## Nice-to-Haves

- **Genuinely predictive experiments**: Use TopoLM to predict candidate clusters for linguistic features *not yet tested* in neuroimaging (e.g., animacy, argument structure), then test those predictions experimentally. This would transform the contribution from replication to prediction and would more strongly support the "driven by" claim.

- **Control for alternative cluster interpretations**: The paper shows verb-noun clusters emerge, but does not test whether other interpretable but non-brain-like clusters also emerge (e.g., frequency-based, length-based, positional). This would help assess whether the verb-noun correspondence is specific or coincidental.

- **Individual unit selectivity profiles**: Showing graded selectivity within clusters (rather than binary membership) would strengthen the analogy to cortical neurons.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Topoformer-BERT comparison is confounded"**: The paper explicitly acknowledges this—"Topoformer-BERT is a baseline, but not a control" (Section 3)—and the comparison is presented to illustrate a qualitative point about spatial autocorrelation vs. functional selectivity, not as a controlled experiment. Removed as the paper is transparent about the comparison's limitations.

- **"Random permutation suggests the spatial loss produces degenerate solutions"**: The permutation is a design choice to prevent a specific failure mode; the paper shows it is necessary (Figure 12). This does not imply the loss is "not doing meaningful work"—it means the optimization landscape requires a straightforward architectural intervention. Removed as the inference is not supported.

- **"No multiple random seeds"**: While a valid concern in principle, this is a standard nitpick for a single-model study with multiple qualitative and quantitative evaluations. Removed as a reproducibility nitpick.

- **"Missing evolutionary clustering sensitivity analysis"**: The clustering algorithm's thresholds (10 units, p < 0.05 FDR) are reasonable choices and the resulting clusters are visualized. Requesting sensitivity for every hyperparameter of an analysis pipeline is a generic ask. Removed as generic.

- **"Request for broader suite of architectures"**: The paper studies one architecture (GPT-2-small). Requesting more is scope creep; the paper's contribution is about the principle, not the architecture. Removed as generic weakness.

- **Formatting/style nitpicks and typo complaints**: Removed per hard rules.

## Novel Insights

The concrete-abstract asymmetry in verb-noun clustering (Section 5.2) reveals something important that goes beyond the paper's stated contribution: the spatial smoothness principle does not merely produce spatial autocorrelation—it selectively amplifies organization that is already latent in the task-learned representations. Concrete words have richer sensorimotor grounding than abstract words, leading to more spatially coherent representational structure in the model. When combined with the smoothness constraint, this latent structure is amplified into explicit spatial clustering. This suggests that spatial smoothness acts as a *magnifier* of existing representational structure rather than a *creator* of new structure—a distinction that the paper does not make but that has important implications for understanding the brain's developmental trajectory.

## Suggestions

- Replace the abstract's "functionally and spatially aligned" with "spatially organized with partial functional alignment" or similar, acknowledging the sentences ≈ unconnected words failure. Similarly, soften "driven by a unified spatial objective" to "consistent with a unified spatial objective."
- Add a semantic-content-based evaluation metric independent of Moran's *I* (e.g., classification accuracy of cluster labels, or cluster purity relative to known brain region assignments) to decouple the evaluation from the training objective.
- Present results *without* fMRI readout sampling as the primary comparison, with sampled results as a secondary analysis—this gives a cleaner picture of what the spatial loss alone achieves.
- Replace mean absolute activation with a differential response measure (activation relative to a baseline condition) to better match the brain's BOLD signal change metric.

## Score and Decision

**Calibration anchors:**
- **TopoNets** (avg 7.50, Accept Spotlight): `/home/wg25r/review_agent/human_reviews/THqWPzL00e.md` — Very similar paper applying spatial smoothness loss to both vision and language models. TopoLM goes deeper on language-specific neuroscience validation (verb-noun, concrete-abstract) but has more circularity in its primary evaluation metric. TopoNets had its own overclaiming issues ("broad suite" for limited architectures). TopoLM is somewhat weaker due to the partial circularity and the failure to match the brain's defining functional signature.
- **Emergent Orientation Maps** (avg 7.33, Accept Spotlight): `/home/wg25r/review_agent/human_reviews/rySLejeB1k.md` — Topographic cortical organization in vision models. Stronger validation in the vision domain; TopoLM extends this to language with partially weaker validation.
- **Speech LM brain alignment** (avg 4.75, Reject): `/home/wg25r/review_agent/human_reviews/eoB6JmdmVf.md` — Brain alignment analysis with real contributions but significant methodological limitations. TopoLM is clearly stronger, with a non-trivial predictive finding.
- **Learning Multiple Representations** (avg 2.33, Withdrawn): `/home/wg25r/review_agent/human_reviews/hbon6Jbp9Q.md` — Overclaimed brain analogies without rigorous validation. TopoLM is much stronger, with genuine empirical findings and quantitative comparisons.

TopoLM sits between the medium-scoring papers (4-5) and the high-scoring topographic model papers (7+). The concrete-abstract verb-noun finding is a genuinely strong and non-trivial result. However, the partial circularity between the training objective and the primary evaluation metric, the failure to match the brain's defining functional signature, and the overclaiming in the abstract are significant issues. The paper's real contribution—demonstrating that spatial smoothness interacts with task-learned representations to produce brain-like organization in language—is solid, but the framing goes beyond what the evidence supports. Relative to TopoNets (7.50), which had cleaner evaluation despite its own overclaiming, TopoLM's evaluation circularity and functional alignment gap pull it down by roughly 1-1.5 points.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>