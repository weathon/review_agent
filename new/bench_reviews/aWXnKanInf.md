The paper content was provided in the user message. Let me re-read key sections carefully to verify the harsh critic's claims.

Let me now carefully assemble the review, cross-referencing against the actual paper content.

Key facts to verify from the paper:

1. **Figure 2C response profile**: The paper says "TopoLM (blue) *mostly* matches the neural data (green), but not exactly; however, the non-topographic baseline model (orange) fails to capture neural patterns as well." And in §4: "they do not have higher activation than unconnected words as in brain data. However, looking across the entire language selective network, the response profile of the non-topographic baseline model similarly fails to capture the neural response profile (Figure 2C), suggesting a general shortcoming of the base transformer model, rather than a weakness of topography."

2. **Moran's I values**: From §5.1: Non-topo baseline I=0.11; TopoLM I=0.48 before sampling; TopoLM I=0.81 after sampling; non-topo after sampling I=0.60 (Appendix). The neural data I=0.96.

3. **Topoformer-BERT**: "we find no evidence for noun-verb selective clustering... before thresholding, the model does exhibit a high degree of clustering competitive with TopoLM (Moran's I = 0.66 before sampling, 0.85 after sampling)."

4. **Abstract claims**: "closely match the functional organization in the brain's language system" and "predicts the emergence of a spatially organized cortical language system as well as the organization of functional clusters selective for fine-grained linguistic features empirically observed in human cortex"

5. **Brain-Score**: 0.78 (TopoLM) vs 0.80 (non-topo), BLiMP: 0.71 vs 0.76, GLUE: 0.68 vs 0.65.

6. **α=2.5**: "We chose this value of α after extensive hyperparameter search. In particular, lower values of α do not adequately encourage the development of topography, while greater values impede task performance and the development of meaningful representations."

7. **Random spatial permutation**: "We randomly permute these positions for each layer such that each layer has a unique spatial encoding." - This is explicitly done to "abstract away from feed-forward propagation."

Now let me assess each criticism:

**Harsh Critic #1**: "Do the brain-like language network claims really require topography?"
- Point 1: "Key effects are not clearly topography-specific" - The paper shows that the topo and non-topo models are both misaligned with neural data in the same way for sentences vs. unconnected words. This is fair. However, the paper explicitly acknowledges this: "they do not have higher activation than unconnected words as in brain data" and notes this is a weakness of the base transformer model. The claim of the paper is about *spatial organization*, not about this specific functional signature. The key claim IS about spatial organization (language-selective clusters emerging). The non-topo model cannot develop spatial clusters by definition since it has no spatial arrangement. So Point 1 partially misunderstands the paper's core claim - the paper's main claim is about *spatial organization*, and the non-topo model literally cannot have spatial organization. But the critic is right that the claim of "closely matching functional organization" is overstated given the failure in this key ordering.

- Point 2: "The network-level uniformity claim is not tied to topography" - This is asking for a comparison of non-topo model's functional clustering with TopoLM's, which doesn't make sense because the non-topo model has no spatial layout. The non-topo model can't have "spatial" clusters. However, one could do a comparable analysis by applying the same clustering to hidden units (without spatial arrangement). The paper doesn't do this, which is a fair gap.

- Point 3: "Brain-like core language system is functionally underspecified" - The paper does acknowledge this limitation. The paper says "the response profiles are not a perfect match to the brain data." But the abstract says "closely match." This is overclaiming.

**Harsh Critic #2**: "Verb/noun clusters match human cortex rests entirely on Moran's I and coarse smoothing"
- Point 1: "Moran's I conflates true category-selective clustering with any low-frequency spatial structure" - This is a legitimate concern. Moran's I measures spatial autocorrelation, not category selectivity per se. However, the paper does use t-tests to define language-selective regions and then shows that the clusters that emerge have the right response profiles. The Moran's I is used as a measure of *degree of clustering*, not as the sole evidence. The paper also shows the visual maps in Figures 3-4. But the critic is right that no shuffle control or alternative-contrast control is performed.

- Point 2: "fMRI-like smoothing almost guarantees high Moran's I" - This is partially correct. The non-topo model reaches I=0.60 after smoothing, but TopoLM reaches 0.81 vs. the brain's 0.96. So smoothing alone doesn't guarantee the high values, but it does inflate them substantially. This is a legitimate concern. The paper does show pre-smoothing values (TopoLM: 0.48, non-topo: 0.11), which is more impressive, but the paper doesn't include controls like shuffled labels or alternative contrasts.

- Point 3: "No direct alignment between model and brain locations of clusters" - This is correct. The paper only shows that clusters *exist* somewhere on the grid, not that they are in brain-analogous locations. Given the random permutation, this is expected—but the "brain-like" language is stronger than warranted by just showing clusters exist. This is a fair criticism.

- Point 4: "Concrete vs. abstract word result can be an artefact of activation magnitude differences" - This is speculative but plausible. No magnitude control is done. This is a minor concern.

**Harsh Critic #3**: "Brain-alignment claims are not commensurate with small but systematic performance drops"
- The paper says "virtually no cost" but shows 5-point BLiMP decrease and 2-point Brain-Score decrease. The abstract says "closely match the functional organization" but Brain-Score goes down. The discussion says "virtually no cost to performance or functional brain alignment." These are quantitative drops. However, no error bars are provided, making it impossible to determine significance. This is a fair criticism of the claims being too strong, though the drops are relatively small.

**Harsh Critic #4**: "The topographic mechanism is only weakly linked to wiring cost and cortical constraints"
- This is a conceptual point about whether the random per-layer permutation undermines the wiring cost interpretation. The paper does explicitly address this: "Our goal is to abstract away from feed-forward propagation as much as possible, as the hierarchical organization of the brain is quite different from that of a language model." The critic is right that this weakens the wiring cost interpretation but the paper is trying to abstract from the specific cortical hierarchy. The paper says the smoothness loss is "an indirect but efficient approach to minimizing local wiring-length" and the smoothness is the claim. This is a valid conceptual limitation to note but not a fatal flaw—the paper explicitly aims to test the smoothness principle, not claim to model actual cortical connectivity.

Now let me consider the **Neutral Reviewer's** and **Spark's** points:

- Neutral Reviewer's points are largely reasonable and well-calibrated. The incomplete match to brain response profiles, limited α sensitivity analysis, Moran's I limitations, and the lack of mechanistic account are all valid.

- Spark's points about α sensitivity and fMRI readout confounds are fair, but some (like "comparison with alternative regularization strategies") are scope creep—the paper is about a specific topographic objective, not about whether other regularizers also produce clusters.

Now for the **Human Finder's** points:
1. fMRI readout inflating clustering metrics - Fair concern
2. Limited model scale - This is a generic complaint; the paper uses a reasonable model size for a proof-of-concept
3. Unfair comparison with Topoformer - The paper already acknowledges this and calls it "a baseline, but not a control"
4. Hyperparameter sensitivity of α - Fair
5. Functional advantage unclear beyond descriptive alignment - Somewhat fair but this is a neuroscience modeling paper, not an ML systems paper
6. Response profile mismatch - Already noted by paper

Let me now compose the final review.

Calibration analysis:
- TopoNets (similar topic, topographic models): Accept/Spotlight, scores 8,8,6,8 = avg 7.5. That paper had a more general loss function, more architectures, better performance claims, and was seen as a clear advance.
- Topoformer (the predecessor this paper compares to): Reject, scores 6,8,5,6,5 = avg 6.0. Major issues were weak brain-link, poor presentation, and limited evaluation.
- Credit-based SOM: Accept/Poster, scores 5,8,6,6 = avg 6.25. This paper was conceptually novel but had functional advantage questions.
- Brain-like Functional Organization in LLMs (withdrawn/reject): scores 3,5,6,1. Major issue: conceptual confusion and limited novelty.

TopoLM has:
- Clear novelty in extending TDANN to language
- Good comparison with actual fMRI data
- Real limitations in the Morans I metric and overclaiming
- A single model/architecture size
- Acknowledged limitations about incomplete match

Given the calibration, TopoLM is stronger than Topoformer (which was rejected at ~6.0) because it has much better evaluation and comparison with real brain data. But it has weaker claims than TopoNets (which was accepted at ~7.5) because TopoNets was more general, tested more architectures, and had cleaner performance claims. TopoLM's overclaiming is a genuine issue.

I'd place TopoLM around 5.5-6. It has a meaningful contribution but with significant overclaiming and methodological concerns that weaken the stated conclusions. The core finding—that a spatial smoothness objective produces emergent clustering that partially matches brain organization—is solid and interesting. But the "closely match the functional organization" claim and the use of Moran's I as the primary quantitative evidence leave the paper in a middle ground where the contribution is real but not as strong as presented.</think>Now I have a thorough understanding of the paper, all reviewer inputs, and the calibration papers. Let me compose the final review.

## Summary

TopoLM extends the TDANN spatial smoothness principle from vision models to transformer language models by adding a 2D spatial arrangement of units with a spatial correlation loss that encourages nearby units to have similar activation profiles. When trained jointly with next-token prediction, this model develops emergent clusters corresponding to a language-selective network and verb/noun-selective clusters that preferentially emerge for concrete over abstract words—replicating findings from human neuroimaging studies. The model retains competitive performance on BLiMP, GLUE, and Brain-Score benchmarks relative to a non-topographic baseline.

## Strengths

- **Novel and meaningful extension of topographic modeling from vision to language.** Applying the TDANN spatial smoothness loss (previously validated only in vision) to a transformer LM is a clean and important conceptual contribution. The fact that this single principle produces multiple brain-like organizational phenomena (language-selective clusters, verb/noun selective clusters, concrete-vs-abstract distinction) is noteworthy.

- **Serious engagement with real neuroimaging data and established experimental paradigms.** The paper tests TopoLM against three independent neuroimaging contrasts (Fedorenko localizer, Hauptman verb/noun contrasts, Moseley concrete/abstract contrasts), including direct comparison with available fMRI data. The use of a Fedorenko-style functional localizer applied *in silico* is methodologically sound, and the fMRI readout sampling simulation is appropriately motivated.

- **Careful non-topographic control with matched architecture and training setup.** The paper trains both TopoLM and a non-topographic baseline with identical hyperparameters (except α=0), providing a valid internal comparison for the effect of the spatial loss. This is essential and properly executed.

- **Task performance is largely preserved.** Table 1 shows TopoLM trades only modest BLiMP (−5 pts) and Brain-Score (−2 pts) decreases for substantive gains in spatial brain-likeness, while GLUE performance actually improves (+3 pts). This addresses a key concern from prior topographic models that spatial constraints severely degrade task performance.

- **Interesting negative result with Topoformer-BERT.** The finding that Topoformer-BERT shows high Moran's I clustering *before thresholding* but *no units survive significance* in verb/noun contrasts (Figure 13–14) is an important discriminative result: it distinguishes spatial smoothness from mere spatial autocorrelation induced by local connectivity constraints.

## Weaknesses

### Major

- **Moran's I alone is insufficient to establish brain-like category-selective clustering, and the paper lacks critical controls.** Moran's I measures spatial autocorrelation, not category-selective patchiness per se. A high Moran's I can arise from broad gradients, low-frequency structure, or simply from the Gaussian smoothing applied to simulate fMRI readout—which itself inflates the non-topographic baseline from I=0.11 to I=0.60 (Appendix Figure 10). The paper does not include basic sanity checks such as label shuffling (computing verb-vs-verb or random-split contrasts under the same pipeline) to demonstrate that the observed clustering is specific to the verb/noun contrast rather than to generic activation structure. The Topoformer-BERT result is itself revealing: high Moran's I before thresholding (I=0.85 with sampling) despite *zero significant units*, demonstrating that spatial autocorrelation of smoothed, unthresholded maps is not diagnostic of meaningful category selectivity. The paper should have included shuffled-label controls and reported clustering on thresholded maps as the primary metric. While the visual maps (Figures 3c, 4a) *appear* to show genuine patches, the quantitative backbone relies too heavily on a metric that conflates multiple sources of spatial structure.

- **The "closely match the functional organization" claim in the abstract and discussion is overstated.** The model's core language system does not reproduce the key brain signature where sentences > unconnected words (Figure 2C). The paper acknowledges this ("the response profiles are not a perfect match to the brain data") and attributes it to a "general shortcoming of the base transformer model," but the abstract states that representations "closely match the functional organization in the brain's language system" and "predicts...organization of functional clusters selective for fine-grained linguistic features empirically observed in human cortex." A model that fails a primary functional signature of the language localizer is at best a *partial* match, and this should be reflected proportionally in the abstract and framing, not only in the body text.

- **No sensitivity analysis for the spatial loss weight α.** The value α=2.5 was chosen after "extensive hyperparameter search," but no quantitative results are shown for how topographic metrics, task performance, or brain alignment vary with α. Since α directly controls the tradeoff between spatial organization and task performance, and the claim is that spatial smoothness is a *unifying principle* driving organization, demonstrating robustness across a range of α values is essential. This is not a missing ablation in the margin—it goes to the heart of whether the brain-like organization depends on a specific hyperparameter choice.

- **The conceptual link between the spatial smoothness loss and cortical wiring-length minimization is weakened by the random per-layer permutation.** Units are placed on a 2D grid then randomly permuted for each layer, explicitly destroying any systematic relationship between spatial position and connectivity. The paper invokes the wiring-length minimization principle from Margalit et al. (2024), but in a transformer with full attention (no locality constraint), the smoothness loss is simply an activation regularizer that encourages locally correlated units—it cannot directly minimize wiring length because there is no spatially-structured connectivity pattern to minimize. The paper acknowledges this choice (footnote 2) but still uses "wiring cost" language that oversells the mechanistic interpretation. The spatial smoothness principle is an interesting hypothesis, but the current implementation demonstrates that a *correlation-based regularizer on a 2D grid* produces clustering, not that cortical wiring constraints drive the organization.

### Minor

- **The non-topographic baseline is not directly compared on spatial clustering metrics for the language localizer (Section 4).** The paper shows TopoLM's language-selective clusters visually (Figure 2a) but does not provide an analogous analysis for the non-topographic model. Since the non-topographic model has no spatial arrangement, one cannot compute Moran's I on it directly, but could random-assign it to a grid and show that no consistent clusters form. This would more sharply demonstrate that the spatial loss, not just the underlying LM representations, is responsible for the observed organization.

- **Brain-Score comparison is fundamentally a test of representational content, not spatial organization.** The ridge regression readout used for Brain-Score does not leverage the spatial layout of TopoLM—it operates on pooled hidden states. Thus, Brain-Score measures whether the spatial loss degrades functional alignment (it does, by 2 points), but cannot confirm that spatial organization *improves* alignment. This limitation should be noted.

- **No error bars or statistical significance is reported for any benchmark (Table 1).** It is impossible to assess whether the 5-point BLiMP decrease or 2-point Brain-Score decrease are within noise or robust.

- **The concrete-vs-abstract word result (Section 5.2) claims replication of Moseley & Pulvermüller, but the analysis differs methodologically.** Moseley & Pulvermüller used anatomically defined ROIs, while TopoLM uses whole-grid clustering. The paper acknowledges this ("since the model lacks defined 'anatomical regions'"), but still frames the result as replication. Given the very different methodologies, "consistent with" or "parallels" would be more appropriate than "replicates."

### Trivial

- The discussion of limitations focuses narrowly on the lack of coherent tissue across layers, but omits discussion of the conceptual gap between smoothness-as-regularizer and smoothness-as-wiring-minimization.

## Nice-to-Haves

- A sweep over α values showing how topographic metrics (Moran's I on thresholded maps, language-selective cluster count), task metrics (BLiMP, GLUE), and Brain-Score jointly vary. This would clarify how robust the results are and help future work calibrate the topography-performance tradeoff.

- Additional linguistic contrasts beyond verb/noun to test which categories emerge and which do not. The paper mentions "syntactic and semantic categories" broadly (Introduction) but only tests two contrasts. Systematic probing would reveal the scope and limits of the smoothness principle.

- A comparison with an alternative regularization approach (e.g., L2 penalty on activations) to isolate whether the spatial structure specifically matters, or whether any regularizer that reduces representational noise would produce similar clustering patterns when projected onto a 2D grid.

- Visualization of how organization develops across layers (not just snapshots of layer blocks) to assess whether the model captures hierarchical structure analogous to the brain's anterior-posterior gradient.

## Removed Points

- *Demand for larger models or more architectures.* The paper uses a GPT-2-small scale model as a proof-of-concept for a new principle. Demanding scaling to larger models is scope creep for a first demonstration. That said, a brief discussion of expected scaling behavior would be valuable.

- *Criticisms of Topoformer-BERT comparison fairness.* The paper explicitly states Topoformer-BERT is "a baseline, but not a control" due to differences in training data, objective, and architecture. The comparison is appropriately framed. The unfairness favors Topoformer-BERT (the baseline), not TopoLM.

- *Demand for direct spatial alignment between model maps and brain locations.* The model uses random spatial layouts per layer with no anatomical correspondence by design. Expecting region-to-region mapping would require a fundamentally different model architecture. Showing that clusters *exist somewhere* on the grid is the appropriate test given this design.

- *Formatting and reproducibility nitpicks.* Missing standard deviations and hyperparameter details are standard; code is available. The evolutionary clustering algorithm and neighborhood parameters are described in sufficient detail for reproduction.

- *Request for a mechanistic account of why specific clusters emerge.* While this would be interesting, the paper's stated goal is to demonstrate that the smoothness principle can predict brain-like organization, not to explain why verb/noun categories specifically emerge. The paper's contribution is showing the *fact* of emergence under a general principle, not providing a detailed mechanistic theory of category formation.

## Novel Insights

The finding that Topoformer-BERT shows high Moran's I (pre-thresholding) but zero significant verb/noun-selective units is a genuinely novel observation that distinguishes *statistical spatial autocorrelation* from *functionally meaningful spatial selectivity*. This has implications beyond this paper: any claim about brain-like topographic organization in ANNs needs to show not just spatial smoothness but also that the smooth regions carry the right functional selectivity. The distinction between these two senses of "clustering" is underappreciated in the topographic modeling literature and deserves more prominence than it receives in the current paper.

## Suggestions

1. **Add shuffled-label and random-split controls for Moran's I.** Compute verb-vs-verb and noun-vs-noun contrasts through the same pipeline. If these also produce high I values, the metric is confounded. If they produce low I values, this strengthens the verb/noun finding substantially.

2. **Report clustering metrics on thresholded maps as the primary result.** Since unthresholded Moran's I is inflated by smoothing and can be high without any meaningful selectivity (as shown by Topoformer-BERT), the paper should use thresholded-map metrics as primary and report unthresholded as supplementary.

3. **Tone down the abstract and introduction.** Replace "closely match" with "partially match" or "share key features with," and note the sentences-vs-unconnected-words failure upfront. The current framing creates a mismatch between the confident abstract and the more nuanced body.

4. **Provide at least a minimal α sensitivity analysis.** A simple table with α ∈ {0, 1, 2.5, 5, 10} showing BLiMP, Brain-Score, and Moran's I (thresholded) would dramatically strengthen the claim that spatial smoothness robustly drives brain-like organization.

5. **Distinguish more carefully between the smoothness-as-regularizer finding and the smoothness-as-wiring-principle claim.** The former is well-supported by the data; the latter requires connectivity constraints that the current model lacks.

## Score and Decision

**Calibration comparison:** TopoNets (similar domain, accepted at Spotlight, avg 7.5) demonstrated topographic losses across multiple architectures with strong performance retention and cleaner evaluation. TopoLM is narrower (one architecture, one scale) and has more overclaiming. Topoformer (rejected, avg 6.0) had weaker evaluation and brain-link claims but weaker methodology overall. Credit-based SOM (accepted poster, avg 6.25) had a novel training mechanism but limited functional advantage analysis, similar to TopoLM's situation. TopoLM has a solid core finding but weakened by (1) overclaiming relative to the evidence, (2) a primary quantitative metric (Moran's I) that conflation issues undermine, and (3) missing α sensitivity analysis. The paper makes a real and interesting contribution—extending TDANN to language and showing emergent clustering—but the gap between what is demonstrated and what is claimed is too large in its current form.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>