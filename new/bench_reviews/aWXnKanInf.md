Now let me run calibration searches in parallel to anchor the score.Now I have enough to write the final authoritative review.

---

## Summary

TopoLM is a transformer language model augmented with a per-layer spatial smoothness loss (adapted from the TDANN framework in vision neuroscience) that causes units within each attention and MLP layer to self-organize into spatially contiguous clusters reflecting linguistic function. The paper demonstrates that these emergent clusters qualitatively and quantitatively replicate patterns observed in human fMRI data—including a spatially organized core language system, verb- and noun-selective clusters, and crucially, clustering specific to *concrete* but not *abstract* verb-noun contrasts. The work extends the spatial smoothness principle from visual cortex (Margalit et al., 2024) to the language system, providing a unified computational explanation for cortical topography across modalities.

---

## Strengths

- **Concrete vs. abstract verb/noun dissociation (Section 5.2, Figure 4):** The paper's single strongest empirical result. TopoLM shows substantially higher spatial autocorrelation for concrete verb/noun contrasts (Moran's *I* = 0.80) versus abstract words (*I* = 0.23, *p* < 0.001), directly replicating the Moseley & Pulvermüller (2014) finding. The non-topographic baseline shows no such differentiation (*I* = 0.11 vs. 0.12, *p* > 0.05). This is a nuanced, non-trivial prediction that genuinely distinguishes topographic organization from random clustering.

- **Rigorous quantitative evaluation using real fMRI data (Section 5.1, Figure 3b):** The paper obtains Hauptman et al. (2024) fMRI data and computes Moran's *I* with Queen contiguity as a standardized metric against ground-truth neural maps (*I* = 0.96). TopoLM with fMRI readout sampling reaches *I* = 0.81 versus *I* = 0.60 for the non-topographic baseline—an interpretable, direct comparison rather than qualitative visual inspection.

- **fMRI readout simulation methodology (Section 3, Figure 1c):** The authors explicitly model the spatial coarsening of fMRI by applying Gaussian smoothing (FWHM 2.0 mm, unit distance 1.0 mm) *before* computing selectivity maps. This is methodologically important: it converts a model *I* of 0.48 to 0.81 and is correctly applied to both models, keeping the comparison fair and interpretable.

- **Emergence of brain-like organization purely from text + spatial loss:** The language-selective clusters and verb/noun selectivity emerge without fitting to any brain data, solely from next-token prediction and spatial smoothness—making the result a genuine generalization rather than a post-hoc fit.

- **Honest positioning of the Topoformer-BERT comparison (Section 3, Footnote 6; Section 7):** The authors explicitly acknowledge that Topoformer-BERT is "not a control" due to differences in training data, objective, architecture, and head count. The qualitative finding that Topoformer-BERT achieves high raw Moran's *I* but no significant verb/noun-selective units (Section 5.1) is reported transparently and provides genuine contrast between spatial smoothness and local connectivity as topographic mechanisms.

- **Direct application of functional localizer paradigm (Section 4):** Adapting Fedorenko et al. (2010)'s sentence/nonword localizer to define language-selective units enables direct, condition-by-condition comparison against human data rather than purely correlational alignment.

---

## Weaknesses

### Fatal
None.

### Major

- **Absence of a size- and data-matched alternative topographic mechanism baseline.** The paper's central scientific thesis is that the TDANN *spatial smoothness loss* specifically is what drives brain-like language organization. However, the only topographic comparison is Topoformer-BERT—explicitly flagged as having a different architecture, training data, objective, and head count. No experiment isolates spatial smoothness loss from other conceivable topographic inductive biases (e.g., local connectivity) at the same scale. The evidence only shows that "TopoLM with spatial loss > TopoLM without spatial loss," which supports the utility of *a* topographic constraint but not the specific smoothness formulation. The claim in Section 7—"TopoLM extends the principle of cortical response smoothness...providing evidence that this principle of spatial smoothness indeed generalizes across cortex"—is not fully separable from the null that any topographic constraint would achieve similar results. This gap is real and would require at least a matched-architecture, matched-data alternative-mechanism baseline to close properly.

### Minor

- **Per-layer random spatial permutation limits the cortical tissue analogy.** Section 3 (footnote 2) explains that unit positions are randomly permuted independently for each attention and MLP layer to prevent the model from simply propagating one spatial pattern through the network—producing 24 independent topographic maps rather than a coherent cortical surface. The paper acknowledges this directly in Section 7: "there is as such no coherent tissue across the entire system as in the brain." While the justification is principled, it means within-layer clustering is partly expected by design (each layer independently minimizes spatial loss), and cross-layer spatial coherence cannot exist. The abstract's characterization of the model as providing "a functionally and spatially aligned model of language processing in the brain" is therefore somewhat imprecise about what "spatially aligned" means.

- **Response profile mismatch is understated in the abstract and introduction.** Section 4 reveals that TopoLM does not replicate the canonical sentence > unconnected words ordering—a foundational and well-replicated signature of the human language system. The paper correctly attributes this to the base transformer (the non-topographic baseline has the identical problem), but the abstract claims the model "closely match[es] the functional organization in the brain's language system" without flagging this. The qualifier "mostly" in Section 4 does not adequately foreshadow the importance of the mismatch. This asymmetry between abstract claims and section-level caveats should be corrected.

- **Spatial loss during GLUE fine-tuning may not be a fair comparison.** Section 6 states that both during pre-training and fine-tuning, TopoLM uses the combined task + spatial loss (α = 2.5), while the non-topographic baseline has α = 0 throughout. The paper hypothesizes that the 3-point GLUE improvement comes from spatial loss serving as regularization. However, this means the fine-tuning comparison is not purely about the model's pre-trained representations; TopoLM receives an additional regularization benefit during fine-tuning that the baseline does not. This confound is worth stating explicitly so readers do not over-interpret the GLUE result as purely a representational advantage.

- **Discussion language slightly overclaims causality.** The Discussion states the results "suggest that the functional organization of the human language system is *driven by* a unified spatial objective" (emphasis added). What the evidence shows is that this spatial objective *can produce* organization resembling brain patterns; it does not establish that the brain uses this specific objective, especially since other mechanisms (metabolic cost, developmental gradients, local wiring constraints) could produce similar organization. The claim should be softened to something like "is *consistent with*" or "*can be explained by*."

- **BLiMP 5-point decrease deserves subcategory analysis.** The paper calls the 5-point BLiMP decrease (0.76 → 0.71) "slight." While this framing is defensible, knowing which linguistic phenomena are most affected (morphological agreement, syntactic islands, etc.) would clarify whether spatial regularization trades off with specific grammatical knowledge—relevant for the paper's claim that TopoLM is a better cognitive model of language.

### Trivial

- The harsh critic notes the model has "24 Transformer blocks" with independent spatial maps, but the model actually has 12 Transformer blocks each with a separate attention and MLP layer (totaling 24 spatial maps). The distinction matters for accurate description when comparing to biology.

---

## Nice-to-Haves

- A version where spatial positions are shared or consistently mapped across layers—even with layer-specific transformations—would strengthen the cortical tissue analogy and is the most natural architectural extension.
- Reporting Moran's *I* trajectories across layers in the main text (rather than appendix only) would let readers judge which layers most closely correspond to brain measurements.
- A mechanistic analysis of *why* concreteness modulates verb/noun clustering in TopoLM (e.g., whether concrete words have more distinctive distributional contexts) would strengthen the causal story.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Sensitivity of Moran's I to neighborhood hyperparameters not shown for key metrics"** (Harsh Critic): The paper (footnote 5) states that neighborhood size and count have "little effect" on task performance and topographic metrics, and the architectural choices were determined via hyperparameter search. Requesting full sensitivity analyses for the primary metrics is reasonable as a nice-to-have but does not rise to a weakness given the footnote disclosure—removed per the nitpick-on-reproducibility rule.

- **"Cluster size threshold of ≥10 units is arbitrary with no sensitivity analysis"** (Harsh Critic): This is a standard methodological choice in spatial analysis; the paper uses it consistently and transparently. Sensitivity analyses here are not standard practice in the field and would not materially affect the main claims—removed per the soft rule on field norms.

- **"Applying spatial loss during GLUE fine-tuning could inflate GLUE scores relative to the baseline"** is listed here in its stronger form: the critic suggests this might "inflate GLUE scores" unfairly. This is kept in Weaknesses (Minor) because it's a real confound, but the more extreme framing of deliberate inflation is removed—the paper is transparent and likely applies the same procedure consistently. What remains is a labeling concern, not a methodological flaw.

- **Generic strength: "Public code availability"** (Strength Finder): Removed per rule on generic strengths not citing specific section/table/figure evidence.

---

## Novel Insights

The concrete/abstract verb-noun dissociation (Section 5.2) provides a genuinely informative test case: the replication of the *absence* of abstract clustering alongside the presence of concrete clustering is stronger evidence for principled brain-like organization than demonstrating clustering alone. Combined with the Topoformer comparison showing that local connectivity produces high raw Moran's *I* but no significant selectivity, the paper makes a subtle point that topographic *smoothness* and topographic *functional organization* are separable—clustering that survives significance thresholding with coherent selectivity profiles is categorically different from arbitrary spatial correlation. This distinction between surface-level clustering and functionally interpretable clustering is a contribution to methodology in this subfield, not just a property of TopoLM specifically.

---

## Suggestions

1. Add a controlled topographic baseline at matched scale (same architecture, same data, local connectivity instead of smoothness loss) to directly test the TDANN smoothness principle versus alternative mechanisms.
2. Revise the abstract to acknowledge the response profile mismatch honestly alongside the successes.
3. Clarify the GLUE fine-tuning comparison by either removing spatial loss from TopoLM fine-tuning or explicitly framing it as "spatial loss as regularization" rather than a representational advantage.
4. Soften "driven by" in the Discussion to "consistent with" or "can be explained by."

---

## Score and Decision

**Calibration anchors used:**
- **THqWPzL00e (TopoNets)** — Accept Spotlight, scores 8/8/6/8 (mean 7.5). Closest topical match: also introduces topographic language (and vision) models using a spatial smoothness loss, broad validation across multiple architectures. TopoNets is somewhat broader in scope (vision + language) but less rigorous in neuroscience validation than TopoLM, which uses actual fMRI data for quantitative Moran's *I* comparison.
- **emMMa4q0qw (Vision CNNs / ventral stream alignment)** — Accept Poster, mean 7.0. Brain alignment paper with strong experiments, solid empirical contribution.
- **LM4PYXBId5 (100 Neural Networks and Brains)** — Accept Poster, mean 7.0. Large-scale brain alignment benchmarking; empirically thorough but lower novelty.

**Positioning:** TopoLM is more narrowly scoped than TopoNets but has more direct neuroscience validation (actual fMRI data, Moran's *I* against ground truth maps, functionally interpretable clustering through significance thresholding). Its major weakness—the absence of a controlled topographic baseline isolating the TDANN principle—is genuine and limits the causal interpretation, placing it slightly below TopoNets. The response profile mismatch and per-layer permutation limitation are acknowledged and understood, not hidden. The core claim (extending spatial smoothness to language cortex) is well-supported by the concrete/abstract dissociation result. This is solidly an Accept, likely at the poster level, scoring somewhat below TopoNets' spotlight-level 7.5.

**Axes summary:**
- *Originality*: Good — clear extension of TDANN to language, novel Moran's *I* comparison against fMRI
- *Importance of research question*: High — understanding cortical topography in language is an open problem
- *Claims well-supported*: Mostly — core concrete/abstract finding strong; TDANN principle attribution needs a controlled baseline
- *Soundness of experiments*: Good — rigorous statistical methods, honest reporting, appropriate caveats
- *Clarity of writing*: Good, with a few overclaims in abstract/discussion
- *Value to community*: Solid contribution as a proof-of-concept extending spatial smoothness to language

**Final score: 6.5**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>