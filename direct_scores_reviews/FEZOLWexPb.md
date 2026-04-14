---

## Summary
MAESTRO is a self-supervised set representation learning framework for cytometry data. It combines a Set Transformer backbone (ISAB-based encoder, SAB-based decoder) with a non-random block masking autoencoder objective, a Sinkhorn optimal transport reconstruction loss, and a DINO-style student-teacher self-distillation framework. The key engineering contribution is an asymmetric student-teacher design where the teacher processes the full cell set (up to 1.3M cells) without gradient computation, while the student processes a masked, subsampled subset, enabling scalable SSL over variable-sized biological sets. The method is evaluated on 1,514 whole-blood cytometry samples across 14 cohorts and 11 clinical phenotypes, outperforming manual gating, clustering, and other set-learning baselines on disease diagnosis, age, sex prediction, and cell-type distribution retrieval.

---

## Strengths

- **Asymmetric teacher-student scalability design.** The paper's central engineering insight — letting the teacher encode the full set (no backprop required) while the student trains on a subsampled masked subset — is a clever and practically impactful solution to the scalability wall that breaks prior set-learning methods (Set Transformer, OTKE) at cytometry scale. Prior works are explicitly capped at 10K cells; MAESTRO handles 1.3M cells per sample.

- **Non-Random Block Masking (NRBM) with cosine-similarity ordering.** This masking strategy groups biologically similar cells and masks contiguous blocks in similarity space, so the model must reconstruct coherent subpopulations rather than isolated cells. This is domain-appropriate and nontrivial: random masking of cell sets would likely be insufficient given the redundancy structure of cytometry data, and the ablation (Table 1) confirms that removing masking entirely is catastrophic (AUROC drops from 0.992 to 0.955).

- **Multi-cohort evaluation with clinical breadth.** The dataset of 1,514 samples across 14 cohorts, 3 collection sites, and 11 phenotypes is substantively larger and more heterogeneous than typical computational immunology benchmarks. Evaluating on both global (diagnosis/age/sex) and local (cell-type distribution retrieval across 16 cell types) tasks gives a thorough picture of representation quality.

- **Permutation invariance formally established.** Theorems 1–4 with proofs in the appendix verify that the ISAB, PMA, and SAB blocks satisfy the necessary equivariance/invariance properties. This goes beyond the typical "we claim permutation invariance" treatment in related work.

- **Sinkhorn OT reconstruction loss as a permutation-invariant training objective.** Using Sinkhorn distance rather than element-wise MSE for set reconstruction is principled: it avoids the need for a canonical ordering of the reconstructed set, which is a genuine requirement for set-structured outputs.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Mathematical inconsistency in the self-distillation loss (Eq. 14 and Algorithm 3, Step 8).** Equation (14) defines:
  $$\mathcal{L}_{\text{SD}} = \frac{1}{m}\sum_{i=1}^m \text{KL}(\text{softmax}(f_s(\mathbf{x}_i)/\tau) \| \text{softmax}(f_t(\mathbf{x}_i)/\tau))$$
  where $\mathbf{x}_i \in \mathcal{S}_m$ are individual cells and $f_s, f_t$ are applied *element-wise*. However, Equations (8), (11), and (12) define $f_s$ and $f_t$ as *set encoders* that produce a single global vector $\mathbf{z} \in \mathbb{R}^D$ via PMA pooling over the entire input set. Applying a set encoder to individual elements is architecturally undefined. Algorithm 3, Steps 4 and 6 further define $\mathbf{z}_s = f_s(\mathcal{S}')$ and $\mathbf{z}_t = f_t(\mathcal{S})$ as single vectors, yet Step 8 indexes them as $\mathbf{z}_s^i, \mathbf{z}_t^i$ over $|\mathcal{S}_M|$ elements — contradicting Steps 4 and 6. The likely intended formulation is a set-level KL divergence between $\text{softmax}(\mathbf{z}_s/\tau)$ and $\text{softmax}(\mathbf{z}_t/\tau)$, treating the 1024-dim embedding as logits (consistent with DINO/iBOT practice). If instead distillation is computed over pre-PMA element-level features (before pooling), this must be stated explicitly and the encoder definition revised. As written, the self-distillation objective is unimplementable as specified. This must be corrected.

- **Confounded baseline comparison: data volume and supervision regime simultaneously differ.** The baselines (Deep Sets, Set Transformer) are both (a) restricted to 10K cells due to scalability limitations and (b) trained in a *supervised* manner, while MAESTRO uses the full set through the teacher and is evaluated via *linear probing* after SSL pre-training. These are two simultaneous confounds. MAESTRO's observed advantage could stem from seeing substantially more data per sample, from the self-supervised pre-training objective, or from the architectural design — the current experimental design cannot distinguish between these. A controlled experiment with MAESTRO restricted to 10K cells would isolate the architectural advantage; a comparison with at least one SSL baseline using the same Set Transformer backbone (without distillation) would isolate the contribution of self-distillation. The ablation in Table 1 addresses the latter within MAESTRO variants, but not against external baselines.

### Minor

- **Figure 1 encoder architecture discrepancy.** The extracted image description of Figure 1 shows both the teacher and student encoders using SAB (Self-Attention Block) blocks, while Equation (8) explicitly defines the encoder as using three ISAB (Induced Set-Attention Block) blocks followed by PMA. The decoder (Eq. 9) correctly uses SABs. This discrepancy between the figure and the equations creates confusion about the actual encoder architecture; the text body (Section 3.1.2, 3.2.2) is consistent with ISABs in the encoder, suggesting the figure may be mislabeled, but this must be corrected.

- **Sinkhorn Algorithm 2 is non-standard.** The standard Sinkhorn-Knopp algorithm initializes $A_{ij} \propto e^{-D_{ij}/\varepsilon}$ and then alternates row/column normalizations. Algorithm 2 instead initializes $A$ uniformly and applies the exponential update ($A_{ij} \leftarrow A_{ij} \cdot e^{-D_{ij}}$) *inside* the normalization loop after each normalization step, which does not correspond to standard Sinkhorn-Knopp. The pseudocode should either match a cited formulation or include a convergence argument for this variant.

- **No statistical significance for main results.** Table 1 (ablation) reports confidence intervals, but Figure 4 (linear probing comparison) and Figure 5 (cell-type retrieval) present results without error bars or significance tests. With 1,514 samples split across 14 cohorts and 11 phenotypes, the effective evaluation set is modest, and performance differences between methods could overlap within sampling variance.

- **Ablation only covers disease diagnosis.** The ablation (Table 1) is performed solely on the disease classification task. Age regression, sex classification, and cell-type distribution retrieval are presented as equally important evaluations but have no corresponding ablation, leaving unclear whether the contributions of NRBM, multi-rate masking, and self-distillation are consistent across tasks.

### Tiny

- **"Non-Random Block Masking" naming is misleading.** The anchor element *is* selected randomly (Algorithm 1, Step 1). The "non-random" label refers to the ordering of subsequent masking by similarity, which is structurally appropriate but could confuse readers. A name like "Similarity-Ordered Block Masking" would more accurately describe the procedure.

- **Radar plot in Figure 5b uses an inverted convention.** The caption states "less coverage across the map means lower error," which inverts the usual radar chart convention (larger area = better). While the caption does explain this, the visualization is counterintuitive and may cause misreading; inverting the axis or re-plotting with coverage representing performance would improve clarity.

---

## Nice-to-Haves

- **MAESTRO at 10K cells control.** Train and evaluate MAESTRO with the same 10K cell cap used by the baselines. This single experiment would cleanly demonstrate whether the performance gain is primarily architectural or primarily from processing more cells per sample.

- **Cross-cohort (leave-site-out) evaluation.** Given 3 collection sites and known batch effects, evaluating representations trained on two sites and probed on the third would substantially strengthen the generalization claim and be more informative for clinical deployment.

- **Quantitative reconstruction evaluation.** Section 4.2 evaluates reconstruction solely via UMAP visualization. Reporting held-out Sinkhorn reconstruction error as a function of masking rate (e.g., 50%, 75%, 90%) would add quantitative rigor to this section.

- **Rare cell type stratified analysis.** The averaged MAE in Figure 5 masks potential failures on rare populations (<1% frequency), which are clinically important (e.g., regulatory T cells, mast cells). Stratified error reported separately for rare vs. common cell types would reveal whether NRBM inadvertently under-samples rare populations.

- **Batch effect quantification in latent space.** Figure 3 uses visual clustering to argue that batch effects are controlled. A quantitative metric (e.g., kBET or silhouette score partitioned by cohort vs. diagnosis) would provide stronger evidence that the embeddings are not confounded by technical variation.

- **Attention weight visualization.** PMA attention weights over cell types would offer biological interpretability — which populations most drive the sample-level embedding — and help validate that the model attends to biologically meaningful signals.

---

## Removed Points
*These points are flagged for removal; treat with caution.*

- **Critic's concern about the "first attention-based SSL set architecture" claim vs. point-cloud SSL literature.** The paper scopes this claim explicitly to "the context of single-cell data" and the challenges specific to cytometry (permutation invariance, variable cardinality up to 1.3M, no positional structure). Comparing against point-cloud SSL methods (Point-MAE, etc.) is out of scope; those methods rely on spatial proximity and geometric structure absent in cytometry data.

- **Critic's concern that ISABs for scalability is "asserted without quantitative ablation" in Section 3.1.2.** The claim "ISABs alone are insufficient" is supported by the ablation and by the design discussion; the paper argues the student-teacher framework and masking are necessary additions, not just ISABs. The phrasing is out of place in a definitions section, but this is a pure style issue.

- **Critic's claim that comparison to supervised baselines is "unfair" because it benefits MAESTRO.** The baselines are supervised because that is how they are designed and published; applying them in SSL mode would require substantial re-engineering. The comparison is inherently asymmetric, but the asymmetry is disclosed. The remaining valid concern (data volume confound) is retained above.

- **Critic's suggestion that the batch effect might inflate demographic predictions.** While worth considering, this is speculative and not supported by evidence in the review. The dataset uses BatchControlHD2 technical controls and the paper demonstrates batch correction in the appendix.

- **Generic strength: "the paper is well-written."** Removed as non-specific.

---

## Novel Insights

The most genuinely novel insight in this paper — beyond the standard claim of "combining SSL techniques for a new domain" — is the asymmetric teacher-student design where the teacher's lack of backpropagation is leveraged as a *feature*, not just a limitation: because the teacher does not require gradient memory, it can process arbitrarily large sets that would be infeasible for the student, turning the EMA teacher into a scalable "full-set reference encoder" rather than merely a momentum-updated regularizer. This reframing of the teacher model as a computationally cheap global-context provider for a resource-constrained local encoder is a transferable idea beyond cytometry — it could apply to any domain where full-data forward passes are feasible but full-data backpropagation is not (e.g., very long genomic sequences, large-scale graph datasets).

---

## Suggestions

1. **Resolve the self-distillation loss formulation.** Rewrite Equation (14) and Algorithm 3 Step 8 to be consistent with the encoder definitions. If the loss is set-level (KL between softmax(z_s/τ) and softmax(z_t/τ)), state this explicitly and explain why the 1024-dimensional embedding is treated as logits. If element-level distillation occurs at pre-PMA representations, define this intermediate output formally and update Equations (11)–(12) accordingly.

2. **Add a controlled 10K-cell MAESTRO variant.** This single ablation, reported in Table 1 or a new table, would answer the most pressing experimental concern and significantly strengthen the claim that the architecture — not data volume — drives the gains.

3. **Harmonize Figure 1 with the encoder definition in Eq. (8).** Ensure the figure labels (SAB vs. ISAB) match the mathematical definition. If the figure is a simplified schematic, add a note clarifying that encoder blocks are ISAB (not SAB).

4. **Extend the ablation to all evaluation tasks.** Report the Table 1 ablation results for age regression and cell-type distribution retrieval in addition to disease classification to confirm consistent contribution of each component.

5. **Correct or cite the Sinkhorn algorithm variant.** Either bring Algorithm 2 into line with standard Sinkhorn-Knopp (initializing from the cost matrix exponential) or cite the specific variant used and note its convergence properties.

---

## Evaluation

- **Novelty:** Moderate-to-good. The individual components (Set Transformer, MAE, DINO) are established; the novelty is in their integration and the asymmetric teacher-student scaling design for biological set data. The NRBM masking is a domain-specific contribution of moderate novelty.
- **Technical soundness:** Mixed. The architecture and training procedure are overall sound, but the self-distillation loss formulation has a clear inconsistency that undermines confidence in the reported implementation. The Sinkhorn pseudocode is non-standard. These are correctable but present.
- **Empirical support:** Adequate in breadth (multi-cohort, multiple tasks), but weakened by the baseline confound and absence of statistical significance tests in the main comparisons. The ablation (Table 1) is informative but limited to one task.
- **Significance:** High in the applied domain. Scalable SSL for cytometry sample representations fills a genuine and documented gap, with direct clinical implications. The method enables analyses (e.g., full-set immune profile vectorization) that were previously infeasible.
- **Clarity:** Good overall, with notable exceptions in the self-distillation formulation and the Figure 1 architecture diagram.

MY FINAL SCORE: <pineapple>5.4</pineapple>