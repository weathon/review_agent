## Summary

MAESTRO is a self-supervised set representation learning framework for high-dimensional cytometry data. It combines a Set Transformer backbone (ISAB/SAB/PMA blocks), a novel Non-Random Block Masking (NRBM) strategy, Sinkhorn Optimal Transport reconstruction, and an EMA teacher–student self-distillation objective to learn permutation-invariant, fixed-dimensional representations of immune profiles containing hundreds of thousands of cells. Evaluated on 1,514 whole-blood cytometry samples spanning 14 cohorts and 11 disease phenotypes, MAESTRO outperforms supervised set-learning baselines and the field-standard manual gating on diagnosis classification, age regression, and unsupervised cell-type distribution retrieval.

---

## Strengths

- **Genuine scalability to massive sets in a self-supervised regime.** The paper is transparent that existing set methods (Set Transformer, OTKE, Deep Sets) fail at the scale of cytometry data (up to 1.38M cells). By splitting the computation between a full-set EMA teacher (no backprop) and a subset-trained student, MAESTRO avoids the quadratic memory wall while retaining a global reference signal. This asymmetric compute split is the key architectural insight enabling SSL at this scale, and it is a concrete, non-trivial contribution.
- **NRBM as a biologically motivated masking strategy.** Masking cells that are semantically similar to a random reference cell removes redundant, easily-reconstructed clusters and forces the model to infer distinct cell populations. The ablation (Table 1) shows this contributes meaningfully beyond random masking (+1.3 pp accuracy, +1.2 pp F1), and the reconstruction visualization (Figure 2) demonstrates that MAESTRO can recover entire UMAP "islands" of cells absent from the unmasked input — a qualitatively striking result.
- **Sinkhorn OT as a permutation-invariant reconstruction objective.** The choice of Sinkhorn distance over MSE or Chamfer distance for set reconstruction is well-motivated: it provides soft, globally optimal matchings without imposing a spurious ordering on the reconstructed set. This is the appropriate loss function for the task and its adoption from optimal transport is principled.
- **Strong empirical results across multiple diverse evaluation tasks.** MAESTRO outperforms all baselines on three structurally different downstream tasks — a classification task (diagnosis), a regression task (age), and an unsupervised retrieval task (16-class cell-type distribution). The cell-type distribution retrieval (Figure 5) is especially compelling because no label information is used; it validates that the set-level embedding preserves local, element-level biological structure.
- **Technical control batch analysis.** Showing that the BatchControlHD2 technical controls cluster tightly in embedding space (Figure 3), despite exhibiting raw batch effects, provides concrete and domain-appropriate evidence that the learned representations reflect biology rather than technical variation.

---

## Weaknesses

### Fatal

*(None. The paper has substantive issues but no single flaw that entirely invalidates the contribution.)*

### Major

- **Self-distillation loss is formally ill-defined.** The encoder $f$ (Eq. 8) maps an entire input set $\mathcal{S}$ to a *single* vector $\mathbf{z} \in \mathbb{R}^D$ via PMA pooling. Yet Eq. 14 writes the distillation loss as a sum over elements $\mathbf{x}_i \in \mathcal{S}_m$, calling $f_s(\mathbf{x}_i)$ and $f_t(\mathbf{x}_i)$ for individual cells — which is undefined under the architecture in Eq. 8, because $f_s$ and $f_t$ operate on sets, not individual elements. Algorithm 3 step 8 compounds the confusion by writing $\mathbf{z}_s^i$ and $\mathbf{z}_t^i$ as if there are per-element latents, whereas each forward pass yields exactly one pooled vector. It is plausible that the actual implementation computes a single KL divergence between $\text{softmax}(\mathbf{z}_s/\tau)$ and $\text{softmax}(\mathbf{z}_t/\tau)$, but this is not what is written. The paper must provide a mathematically consistent statement of this loss. As written, it is impossible to reproduce the training procedure from the main text.

- **Decoder mechanism is underspecified.** Eq. 9 states $g(\mathbf{z}) = \text{Linear}(\text{SAB}_2(\dots(\text{PMA}(\mathbf{z}))))$, but PMA is defined as $\text{PMA}(\mathcal{S}) = \text{MAB}(\mathbf{s}, \mathcal{S})$ where $\mathcal{S}$ is a set — not a single vector. How $\mathbf{z} \in \mathbb{R}^D$ is expanded into a variable-sized set of $|\mathcal{S}_M|$ elements via PMA is never explained. If a fixed number of learnable seed tokens are used as queries and $\mathbf{z}$ as context, the output has a fixed size unrelated to the input subset size, which would be a design flaw. If the number of seed tokens is matched to $|\mathcal{S}_M|$ dynamically, that mechanism must be stated. This is not a nitpick — the decoder is half of the model.

- **Comparison conflates two confounders.** MAESTRO is compared against Deep Sets and Set Transformer (i) run in *supervised* mode and (ii) restricted to 10,000 cells, while MAESTRO is (i) self-supervised with linear probing and (ii) given the full set (up to 1.38M cells). These two differences are both independently advantageous to MAESTRO and are never disentangled. To isolate architectural merit, the paper needs at minimum one ablation: MAESTRO trained and evaluated on the same 10,000-cell random subsets as the baselines. Without this, it is impossible to determine whether MAESTRO's gains come from its architecture or simply from seeing 100× more data.

### Minor

- **"Supervised approaches" mischaracterization of Deep Sets and Set Transformer.** The paper states "Deep Sets and Set Transformer are supervised approaches, restricting their use to labeled datasets" (Section 2, Section 1). These are *architectures*, not inherently supervised methods. Both can in principle be adapted to a self-supervised objective. The intended meaning (that standard/published applications of these have been supervised, and no SSL adaptation exists for cytometry) is defensible but should be stated precisely; the current framing is misleading and could confuse ML readers about the nature of these models.

- **Theorems 1–4 are not novel contributions.** The permutation equivariance of MAB/ISAB/SAB and permutation invariance of PMA are direct corollaries of results in Lee et al. (2019). Presenting them as theorems in the main paper overstates the theoretical contribution and consumes space that could explain more novel aspects of the method.

- **Masking rates A, B, C are never defined in the main text or algorithm.** Figure 1 shows three masking rates (A, B, C) applied to the student, but neither the main text nor Algorithm 3 describes what these rates are, how they are chosen, or whether they correspond to independent augmentation views applied simultaneously or sequentially. This is a meaningful architectural choice whose description is absent.

- **Batch effect confounding is not fully ruled out.** Samples come from 14 cohorts at three institutions, and diagnosis is likely correlated with cohort membership (each cohort being a specific clinical study). The paper shows technical controls cluster well post-embedding, which is necessary but not sufficient: it does not rule out residual cohort-level signals aligning with diagnosis groups. A cross-cohort held-out evaluation would significantly strengthen the biological validity claims.

- **Ablation does not cover Sinkhorn OT vs. simpler reconstruction losses.** Sinkhorn OT is a central design choice and is computationally expensive. No ablation compares it against Chamfer distance or MSE on unordered sets, leaving its contribution empirically unvalidated relative to cheaper alternatives.

- **Figure 5b caption is self-contradictory.** The caption reads: "As we move away from the center the MAE gets higher, less coverage across the map means lower error." The second clause contradicts the first (less coverage = lower error, but moving away from center = higher error). This confusion is a concrete readability problem.

### Tiny

- **"Algorithm 0" naming error.** Section 3.2.1 refers to "Algorithm 0" but the algorithm block is labeled Algorithm 1. Minor but indicates incomplete revision.
- **Hyperparameters ($\tau$, $\alpha$, masking rates) confined to appendix** without brief summary in the main text, reducing immediate reproducibility.

---

## Nice-to-Haves

- **MAESTRO restricted to 10K cells ablation.** Beyond the fairness argument, this would quantify the actual marginal gain of processing larger sets, giving insight into when sampling suffices and how much full-set access helps.
- **Cross-institution generalization experiment.** Training on cohorts from two Penn locations and testing on the third would demonstrate that embeddings generalize beyond training site distributions.
- **Self-supervised baselines.** Implementing a basic MAE or DINO objective on the same Set Transformer backbone would directly test whether MAESTRO's gains over supervised baselines come from the SSL paradigm in general or from MAESTRO's specific design.
- **Sensitivity analysis for NRBM reference element.** How much does performance vary if the reference cell is an outlier versus a representative centroid? This would clarify the robustness of NRBM.
- **Scalability plot.** An empirical GPU memory and wall-clock time curve as a function of set size (1K, 10K, 100K, 1M) for MAESTRO and baselines would substantiate the scalability claim quantitatively rather than relying on anecdote.
- **Attention map visualization.** Showing which cell subpopulations PMA/SAB attends to for different diagnoses would provide mechanistic interpretability useful for immunologists.
- **Sinkhorn OT ablation vs. Chamfer/MSE.** See minor weaknesses above; the empirical justification for this specific choice is absent.

---

## Removed Points

*These points are flagged to be removed or heavily discounted — treat them with caution.*

- **"Unfair comparison because comparison is asymmetric and benefits MAESTRO" (removed per rules).** The Harsh Critic correctly identifies the comparison as unfair but for valid reasons — MAESTRO genuinely processes more data AND is SSL vs. supervised. This is NOT a case where the asymmetry intentionally benefits the baseline to prove a stronger point. The weakness is retained in Major above.
- **Manual gating dimensionality mismatch (removed).** The critic notes that MAESTRO's 1,024-dim embeddings are compared to manual gating proportion vectors of lower dimensionality and that this advantages MAESTRO under linear probing. However, this is scope creep — the paper's central claim is that learned set representations improve over domain-standard analysis pipelines. Manual gating proportion vectors represent the standard feature engineering approach; arguing that one should artificially pad or restrict dimensionality misses the point of the comparison.
- **No external dataset validation (removed from weaknesses, retained as nice-to-have).** Requiring validation on an external public dataset (e.g., FlowCAP) is a reasonable suggestion but is not a standard requirement for a methods paper of this type in the ML/computational biology literature. Moved to Nice-to-Haves.
- **Statistical significance testing (removed from weaknesses).** The critic calls for p-values or confidence intervals on all comparisons. Standard deviations are reported in Table 1. For the primary comparisons in Figure 4, single-run evaluation is standard practice for this scale and domain. This does not rise to a methodological flaw.
- **Teacher computational overhead deferred to appendix (removed).** The paper explicitly states runtime/memory details are in Appendix F.2 and that the teacher does not require backpropagation. Deferring detailed numbers to the appendix is acceptable; the mechanism explaining why it is feasible is present in the main text.
- **Spark Finder "Input Size Contradiction" (removed).** The abstract says "uses all of a sample's cells" and Figure 1 says "N Cells are sampled." There is no contradiction: the teacher model processes the full set; the student processes a sampled subset. The abstract's claim refers to the teacher-provided signal, which does use all cells.
- **Reconstruction loss applied to masked vs. all cells (removed).** Algorithm 3 step 7 computes SinkhornDistance($\mathcal{S}_M$, $\hat{\mathcal{S}}$), reconstructing the full sampled subset from the masked version. Figure 2 explicitly shows masked cells being reconstructed, confirming the intended training signal.
- **Rare cell population detection limitation (removed from weaknesses).** The paper itself acknowledges this limitation in Appendix F.6. Criticizing a stated limitation is redundant.

---

## Novel Insights

The most genuinely insightful architectural idea — not sufficiently emphasized in the paper itself — is the asymmetric compute split between teacher and student as a means of achieving scalable SSL on massive sets. Because the teacher's forward pass requires no backpropagation, its memory cost is constant regardless of set size, enabling it to consume the full patient-level set as a global target while the student trains on manageable subsets under masked reconstruction. This is a principled and elegant solution to the fundamental tension between set completeness and gradient computation, and it is more broadly applicable beyond cytometry (e.g., computational pathology on WSIs, ecology datasets with variable population sizes). NRBM's insight — that masking cells semantically similar to each other (rather than random cells) destroys easy-to-predict redundancies and forces reconstruction of biologically distinct minority populations — also has potential as a general principle for self-supervised learning on multimodal, clustered set data where element redundancy is high.

---

## Suggestions

1. **Fix the self-distillation loss equation.** Rewrite Eq. 14 to match what is actually computed. If the loss is a single KL divergence between $\text{softmax}(\mathbf{z}_s/\tau)$ and $\text{softmax}(\mathbf{z}_t/\tau)$, write that explicitly. If it is computed element-wise, explain how per-element latents are obtained from the set encoder.
2. **Fully specify the decoder.** Explain how PMA is repurposed to expand $\mathbf{z} \in \mathbb{R}^D$ to $|\mathcal{S}_M|$ elements, including what the queries are and how their count is determined.
3. **Add a 10K-cell MAESTRO ablation.** Train and evaluate MAESTRO on the same 10K random subset as the baselines, and include it in Figure 4/Table 1. This single experiment would substantially address the fairness concern and demonstrate the genuine value of large-set processing.
4. **Describe masking rates A, B, C in the main text.** Add a sentence or small table clarifying what the three rates are and whether they correspond to multiple augmentation views per forward pass.
5. **Rename "Online Tokenizer" to "EMA Teacher" or "Momentum Teacher."** The current terminology (borrowed from iBOT's VQ-VAE tokenizer context) misrepresents what is being done and will mislead ML readers.
6. **Fix Figure 5b caption.** Rewrite the self-contradictory sentence about coverage and MAE.
7. **Add cross-cohort held-out evaluation.** Designate one or two cohorts as test-only to empirically evaluate cross-institution generalization.
8. **Ablate Sinkhorn OT against Chamfer distance.** This would confirm that the permutation-invariant OT loss is specifically responsible for reconstruction quality, not just the masked autoencoding objective.