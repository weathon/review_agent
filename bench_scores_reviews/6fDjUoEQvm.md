## Summary

HyperDAS is a transformer-based hypernetwork that automates two key steps in distributed alignment search (DAS): (1) selecting which token positions in base and counterfactual residual streams encode a target concept, and (2) identifying an orthogonal linear subspace that mediates that concept via a Householder transformation. Conditioned on a natural language concept description and the frozen target model's hidden states, the hypernetwork produces soft token-pairing weights and a dynamically rotated subspace, achieving state-of-the-art disentanglement scores on the RAVEL benchmark with Llama3-8B. The paper also includes a substantive discussion of faithfulness risks — the concern that a powerful supervised interpretability method may be steering rather than discovering — and shows how architectural and training constraints (sparsity loss, one-to-one snapping) mitigate these risks.

---

## Strengths

- **End-to-end automation of token localization, a genuine pain point in the field.** Prior DAS-based methods rely on heuristics such as "always use the last entity token," which the paper shows is not universally valid (Figure 4 reveals that HyperDAS selects JSON syntax tokens in deep layers — a previously unknown storage location). HyperDAS removes this manual step via differentiable attention, a non-trivial contribution with demonstrated gains (Disentangle score: 84.7 vs. 76.0 for MDAS in Table 3a).

- **Conceptually clean faithfulness analysis with concrete evidence.** The paper goes beyond performance numbers to analyze *whether* the learned interventions are faithful. Figure 7 empirically demonstrates pathological failure modes (no sparsity loss → many-to-one alignment that hacks the soft training objective; too much sparsity loss → degenerate hidden-state blending), and quantifies that all three variants achieve ~94% disentanglement under soft weights — making the sparsity loss's design motivation concrete rather than hand-wavy.

- **Layer-resolved interpretability findings.** Figure 4's breakdown of which token types are targeted at shallow, middle, and deep layers reveals genuine mechanistic insight: counterfactual entity tokens are robustly targeted from the earliest layers (~84%), while base prompt targeting evolves from noise (shallow) to entity-focused (middle) to syntax-token-focused (deep). This multi-layer analysis is more fine-grained than what MDAS provides and constitutes a secondary scientific contribution.

- **Structured Householder vector analysis (Figures 5–6).** The PCA clustering and pairwise cosine similarities of Householder vectors across attributes (e.g., Longitude/Latitude cosine similarity = 0.97, Country/Language = 0.79) provide an interpretable geometric signature of what HyperDAS has learned, enabling falsifiable predictions about which attributes share representational geometry.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Single model evaluation severely limits generalizability claims.** All results are reported exclusively for Llama3-8B. It is unknown whether HyperDAS generalizes to other architectures (Mistral, Gemma, GPT-family), other scales, or models with different tokenizers and entity-encoding behaviors. A method claiming to automate mechanistic interpretability should demonstrate it is not overfitted to one model's quirks. This is the most significant gap in the experimental section.

- **Notational inconsistency in softmax direction (Section 3.2).** The paper states "we apply a column-wise softmax G = ColumnSoftmax(G^i)" (Eq. 8). If this normalizes each column of G ∈ R^{B×(C+1)} over the B base tokens (i.e., Sum(G_{(★,c)}) = 1 by construction), then the sparse attention loss (Eq. 13) — which fires only when Sum(G_{(★,c)}) > 1 — would *never* activate. Conversely, Eq. 9 constructs a weighted sum over counterfactual tokens c for each base token b, which semantically requires the row sum Sum(G_{(b,★)}) ≈ 1, i.e., row-wise normalization. Inspection of Figure 2's raw matrix (e.g., the column for "[BOS]" base token sums to ≈ 1.0 over counterfactual tokens) further supports the actual implementation using row-wise softmax. This appears to be either a labeling error in the paper ("column-wise" where "row-wise" is intended) or a genuine discrepancy between the text and code. The ambiguity directly affects the reader's understanding of how the sparse loss functions and should be clarified or corrected.

- **Symmetric variant failure is unexplained and concerning.** Symmetric-All-Domains achieves an average Disentangle score of only 54.8%, and the per-domain Symmetric variant collapses for verbs (42.3 Causal vs. 93.0 for Asymmetric, Table 3a). The paper attributes this to asymmetric behavior in positional assignments (Figure 8) but offers only a qualitative description. Symmetry ought to be a desirable property of a faithful interpretability method — the fact that enforcing it substantially degrades performance (and that the better-performing asymmetric variant selects different entity token positions depending on whether the prompt is base or counterfactual) raises unresolved questions about what HyperDAS is discovering. This deserves a mechanistic explanation: does the asymmetry reflect genuinely different "read" vs. "write" circuits, or is it an artifact of the training procedure?

- **Limited baseline comparison.** The only baseline is MDAS. There is no ablation against DAS with the same token position as MDAS (to isolate the gain from automated token selection), nor any comparison against LoReFT (Wu et al., 2024, which is cited). Without a "DAS + HyperDAS token selection oracle" ablation, it is unclear whether the performance gains come primarily from the hypernetwork's token localization, the subspace identification, or the joint training objective.

### Minor

- **Single Householder transformation has limited expressivity.** The paper uses a single Householder reflection (Eq. 10) to rotate the initial subspace R^l. A single Householder matrix is a rank-1 update to the identity (a reflection about one hyperplane) and has limited ability to span an arbitrary rotation in d=4096 dimensions. No justification is provided for why one transformation suffices, and no ablation studies the effect of using more Householder transformations. The approach may be sufficient empirically, but this should be validated.

- **Training-inference discretization gap is unquantified.** The method snaps the soft alignment matrix G to hard 1-1 correspondences at inference time (Eq. 14). The paper argues the sparse loss bridges this gap, but provides no direct measurement of the performance degradation from soft → hard alignment. A table or figure comparing soft-weighted vs. snapped performance would make this design choice more transparent.

- **Layer selection methodology is underspecified.** Table 3a reports results from "the best layer between 10 and 15." It is not stated whether this best layer was selected on the validation split or the test split, and how this selection is separated from the test evaluation. If selected post-hoc on the test set, this could inflate reported numbers.

- **High cosine similarities across all attributes may indicate subspace conflation.** Figure 6 shows that all city-domain attribute pairs share cosine similarity ≥ 0.69 for their Householder vectors. The paper interprets within-attribute clusters as evidence of disentanglement, but does not address whether all high-similarity vectors are simply capturing "city-ness" rather than genuinely distinct attribute subspaces. A comparison against random baselines or against the mean city representation direction would clarify this.

- **Multi-token selection (53%) is presented without analysis.** The paper notes HyperDAS selects multiple tokens 53% of the time. It is unclear whether these multi-token selections are interpretable (e.g., sub-word splits of the same entity) or represent a failure mode. No analysis of the quality or interpretability of multi-token selections is provided.

### Tiny

- **Figure 2's matrix layout is the transpose of G as defined in the text.** G ∈ R^{B×(C+1)} has rows = base tokens and columns = counterfactual tokens, but Figure 2 displays "rows = counterfactual tokens, columns = base tokens." The paper mentions this (Section 3.6), but the inconsistency between the mathematical convention and the figure makes notation hard to track throughout.

- **Sparsity loss schedule sensitivity.** The loss weight λ increases linearly from 0 to 1.5 starting at 50% of training steps. No sensitivity analysis for this schedule is provided, and Figure 7 shows the method is sensitive to the magnitude of λ. A brief ablation would strengthen confidence in reproducibility.

---

## Nice-to-Haves

- **Cross-architecture generalization experiment.** Even one additional model (e.g., Pythia-12B or Gemma-7B) would substantially strengthen the generalizability claim.

- **Cross-domain zero-shot transfer within RAVEL.** Training on Cities and evaluating on Nobel Laureates would test whether the hypernetwork learns a general concept localization algorithm or memorizes entity-specific patterns.

- **Ablation isolating token selection vs. subspace learning.** Compare (i) MDAS token position + HyperDAS subspace, (ii) HyperDAS token position + fixed subspace, and (iii) full HyperDAS. This would partition the performance gain by source.

- **Synthetic ground-truth validation.** Evaluating on a task with known mechanism (e.g., modular arithmetic) would provide a stronger faithfulness check beyond the architectural constraints already argued.

- **Distilling HyperDAS outputs into static probes.** After HyperDAS identifies the concept location and subspace, distilling this into a per-concept static probe would reduce inference cost and yield a cleaner interpretability artifact.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"Automating interpretability overstates the contribution."** The paper's title includes "Towards" and the text is explicit that supervision is still required per concept. The framing accurately describes what is automated (the search over positions and subspaces). REMOVED as the concern misreads the paper's own qualified claim.

- **"The method is evaluated only at a fixed layer l and layer search is still manual."** True, but the paper explicitly scopes the contribution to within-layer localization and shows multi-layer sweeps in Figure 3b. Criticizing the absence of automated layer search is scope creep for this paper. REMOVED.

- **"Zero-shot generalization to unseen concepts is unaddressed."** The paper does not claim zero-shot capability; this is scope creep. Moved to Nice-to-Have.

- **"No discussion of confidence intervals / statistical significance."** For RAVEL-scale benchmarks structured as large counterfactual intervention suites, single-run evaluation is standard in the field. REMOVED per the rule on non-standard rigor requirements.

- **"Missing related work X."** Per review instructions, no external citations can be confirmed; all such criticisms are removed.

- **"The contribution list in the introduction is implicit rather than bulleted."** Pure formatting nitpick. REMOVED.

- **"Unfair comparison because HyperDAS trains jointly on all attributes while MDAS trains separately per attribute."** When evaluating HyperDAS-Asymmetric (per-domain), the comparison to MDAS is fair. When evaluating All-Domains variants, the joint training actually appears to *hurt* HyperDAS performance (80.7 vs. 84.7), so any asymmetry in this comparison benefits MDAS, not HyperDAS. REMOVED per the rule on comparisons that are asymmetric in favor of the baseline.

---

## Novel Insights

The most genuinely novel empirical insight from these reviews (cross-validated against the paper) is that HyperDAS, when applied to deep layers of Llama3-8B, targets JSON syntax tokens for intervention rather than entity tokens — a finding that extends and challenges the prevailing assumption in knowledge editing that entity attributes are invariably stored at entity token positions. This is not merely a validation of prior work but a discovery of previously unknown storage locations, and it emerges directly from removing the manual token-selection heuristic. Additionally, the Householder vector cosine similarity analysis (Figure 6) offers a new geometric vocabulary for describing attribute disentanglement — the finding that Latitude/Longitude share nearly identical subspaces (0.97 similarity) while Country/Longitude are more separated (0.69) constitutes a testable prediction about how Llama3-8B jointly represents geographically correlated attributes.

---

## Suggestions

1. **Clarify the softmax direction definitively.** State explicitly whether the softmax in Eq. 8 normalizes over base tokens (column-wise, Sum(G_{(★,c)}) = 1) or counterfactual tokens (row-wise, Sum(G_{(b,★)}) = 1), and verify consistency with the sparse attention loss. If it is row-wise (as Eq. 9 and Figure 2 suggest), update the paper text accordingly.

2. **Add at least one non-Llama evaluation.** Even a smaller model on a subset of RAVEL domains would substantially de-risk the generalizability question.

3. **Mechanistically explain asymmetric token selection.** Provide an empirical analysis of whether the second-to-last (base) vs. last (counterfactual) token preference reflects a "read from early token" vs. "read from accumulated representation" circuit, referencing residual stream norm or attention score patterns.

4. **Ablate number of Householder transformations.** Report performance with 1, 2, 4, and 8 Householder reflections to justify the single-reflection design choice and quantify any expressivity ceiling.

5. **Quantify soft-to-hard alignment performance drop.** Add a row to the results table showing performance under soft (weighted) inference vs. hard (snapped) inference, to directly measure the training-evaluation gap the sparsity loss is designed to close.

6. **Clarify layer selection protocol.** Explicitly state whether the "best layer between 10 and 15" is selected on a held-out validation set, and separate this from the test evaluation to avoid any appearance of post-hoc selection.

---

**Overall evaluation axes:**
- **Novelty:** Moderate-to-high. Applying hypernetworks to automate token-position search in mechanistic interpretability is a creative and previously unexplored direction.
- **Technical soundness:** Moderate. The core architecture is reasonable, but the notational inconsistency in the softmax direction and the unexplained failure of the symmetric variant leave open questions about whether the method operates as described.
- **Empirical support:** Moderate. State-of-the-art results on RAVEL are meaningful, but confined to a single model and a single benchmark with only one baseline comparison.
- **Significance:** Moderate-to-high. Automating token position localization is a genuine and practically important bottleneck; the method's success validates that end-to-end optimization can replace heuristic token selection.
- **Clarity:** Moderate. The high-level story and diagrams are clear, but the method section contains notational ambiguities (softmax direction, matrix transpose in Figure 2) that require careful tracking to follow correctly.