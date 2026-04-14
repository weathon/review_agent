=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
## Summary

DPLM-2 extends the discrete diffusion protein language model (DPLM) to jointly model amino acid sequences and 3D structures within a single Transformer-based framework. The core technical contributions are: (1) a Lookup-Free Quantization (LFQ) structure tokenizer that converts backbone coordinates to discrete tokens significantly more effectively than VQ-VAE; (2) a LoRA-based warm-up from sequence-pretrained DPLM to efficiently transfer evolutionary knowledge to multimodal learning; and (3) a self-mixup training strategy to reduce exposure bias. Trained on PDB + AFDB-SwissProt (∼220K structures) initialized from DPLM, the model achieves competitive performance across unconditional co-generation, folding, inverse folding, and motif scaffolding tasks.

---

## Strengths

- **LFQ tokenizer demonstrably outperforms VQ-VAE for protein structure**: Fig. 2 provides compelling quantitative evidence — LFQ-4k achieves TM-score 0.91 and RMSD 3.31 on CAMEO 2022 vs. VQ-VAE-1k's 0.80 / 4.14, while also converging in 2 days vs. 15 days on 8 A100s. The analysis of token-secondary structure correspondence (Fig. 2B) further demonstrates that structure tokens capture semantically meaningful local geometry.

- **Strong motif-scaffolding results with multimodal conditioning**: DPLM-2 achieves a co-generation success rate of 0.53 vs. RFDiffusion's 0.40 and solves 19/24 problems—outperforming all baselines when using joint structure+sequence motif input and co-generation output. This is a qualitatively meaningful result that specialized structure-only models cannot directly replicate and represents the paper's most compelling empirical finding.

- **Secondary structure distribution closest to natural proteins**: Fig. 4A shows that structure-based models (RFDiffusion, MultiFlow) over-represent α-helices relative to PDB, while DPLM-2 produces balanced helix/sheet/loop proportions most closely matching natural proteins. Fig. 4C confirms this with ternary plots — MultiFlow clusters in helix-rich regions while DPLM-2 spans the natural distribution.

- **Data and compute efficiency for multimodal modeling**: DPLM-2 achieves competitive multimodal generation using only ∼220K structural examples and open-source pre-trained weights, in contrast to ESM3's 1.4B–98B parameters trained on massive synthetic datasets. The open-source commitment (models + training/inference code) meaningfully lowers the barrier for community adoption.

- **Consistent scaling behavior across model sizes**: Table 3 and 4 show systematic improvement from 150M → 650M → 3B across folding and inverse folding tasks, providing evidence that DPLM-2 follows expected scaling laws.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Missing specialized inverse folding baselines (Table 4)**: The inverse folding evaluation compares DPLM-2 only against other generalist multimodal models (MultiFlow, ESM3). Purpose-built inverse folding models such as ProteinMPNN and ESM-IF1 are entirely absent. These are the canonical, widely-used benchmarks for this task. The claim of "competitive performance" in inverse folding is unverifiable without these comparisons, as the paper may be comparing only against weaker generalist baselines. This is a significant omission given that inverse folding is presented as a core conditional task.

- **Conditional independence assumption (Eq. 2) lacks justification in the main paper**: The factorization log p_θ(z_i, s_i|·) = log p_θ(z_i|·) + log p_θ(s_i|·) is a strong assumption that decouples the structure and sequence output heads at each residue. For a model whose core claim is learning the joint distribution p(s, z), this means all cross-modal residue-level coupling must be implicitly absorbed into the shared Transformer representations, with no explicit joint prediction term. The paper flags this as discussed in §G (unavailable), but this assumption deserves at minimum: (a) acknowledgment of what it trades away, and (b) an ablation comparing the factored vs. a non-factored joint output head. As written, the main paper's most consequential architectural decision is almost parenthetical.

- **Catastrophic forgetting unresolved despite LoRA design intent**: LoRA warm-up was explicitly motivated by preventing catastrophic forgetting of sequence knowledge (§3.2: "we apply LoRA to limit too much deviation to the original parameters"). Yet Table 5 shows DPLM-2 (650M) underperforms DPLM (650M) on HumanPPI (84.44% vs 86.41%) and Metal Ion Binding (74.28% vs 75.15%). The paper attributes this to smaller training data (200K vs. 45M) but does not close the loop on whether LoRA rank, applied layers, or data mixing is the limiting factor. The data scale gap between DPLM-2 (200K structural examples) and SaProt (ESMFold structures for all ~240M UniRef50 sequences) is enormous and largely explains the SaProt gap — but the degradation *relative to DPLM itself* remains unexplained given that LoRA was supposed to prevent exactly this. This limits the model's claim to be a general-purpose protein foundation model.

### Minor

- **Codebook size selection unjustified**: Fig. 2 identifies LFQ-8k (8192 tokens) as "the best compression-reconstruction trade-off" (TM-score 0.93, RMSD 2.58 on CAMEO 2022), yet DPLM-2 uses LFQ-4k (4096 tokens, TM-score 0.91, RMSD 3.31). The reason for this choice (presumably vocabulary size or memory constraints) is not stated. The gap between LFQ-4k and LFQ-8k is not negligible; quantifying the downstream impact on generation quality would clarify whether this is a meaningful performance-cost tradeoff.

- **Folding performance gap with ESMFold is understated**: DPLM-2 (650M + SFT) achieves TM-score 0.84/0.89 on CAMEO 2022 vs. ESMFold's 0.85/0.93. The top-half TM-score gap (0.89 vs. 0.93) is substantial and the paper's description of this as "close performance" is optimistic. The authors' implicit argument — that DPLM-2 is a generalist model while ESMFold is purpose-built — is valid and should be stated explicitly rather than left as a subtext.

- **Training distribution of (t_d, t_s) not specified**: DPLM-2 uses independent noise schedulers for each modality, enabling flexible conditional sampling by setting one modality's noise to 0 at inference. However, how (t_d, t_s) pairs are sampled during training is not described in the main paper (deferred to §A.2). Whether the training distribution adequately covers the folding regime (t_d ≈ T, t_s ≈ 0) and inverse folding regime (t_d ≈ 0, t_s ≈ T) directly determines whether conditional sampling is well-supported or requires out-of-distribution generalization.

- **ESMFold oracle circularity not acknowledged**: ESMFold is used both as the source of training data (AFDB-SwissProt predicted structures) and as the evaluation oracle for designability (sc-TM). This creates a well-known circularity: generated structures that are optimized to match ESMFold's predictions may appear designable in sc-TM without being genuinely so. While this is a known limitation of the field, the paper does not acknowledge it, even in the limitations section.

- **Compute efficiency claim lacks quantitative support**: The paper characterizes DPLM-2 as "data and compute efficient" compared to ESM3, but provides no training FLOPs, wall-clock training time, or inference latency comparison. "Efficient" relative to a 98B-parameter model is a low bar; without numbers, the efficiency claim is qualitative.

### Tiny

- **Metric direction inconsistency in Table 2**: The column header reads "avg. pdb-TM (↑)" but the text in §4.1 states "lower values indicating greater novelty." If novelty is the goal, the annotation should be "↓". This inconsistency makes it unclear what direction is considered "better" for this metric.

- **LFQ-4k reconstruction ceiling propagates to generation upper bound**: The structure tokenizer has a TM-score ceiling of ~0.91 on CAMEO 2022 for LFQ-4k. Even a perfect generative model with this tokenizer cannot exceed this reconstruction quality for folding. This hard ceiling on folding performance is not discussed and would explain some of the gap vs. ESMFold.

- **Motif scaffolding success criteria differ across evaluation types**: The paper uses pLDDT > 70 for sequence-based evaluation and sCTM > 0.8 for structure/co-generation evaluation (§4.4). These are different thresholds applied to different metrics, which makes cross-method numerical comparisons in Fig. 5 not fully apples-to-apples.

---

## Nice-to-Haves

- **Alternative evaluation oracle**: Using AlphaFold3 or OmegaFold as a secondary oracle alongside ESMFold would help mitigate oracle circularity concerns and provide a more conservative estimate of designability.
- **Representation learning recovery experiment**: Showing whether continued sequence-only fine-tuning (or a higher LoRA rank) can recover DPLM-2's downstream predictive performance to DPLM levels would clarify whether the forgetting is architectural or simply a data quantity issue.
- **Contact map / solvent accessibility distribution analysis**: Secondary structure statistics (Fig. 4) are well-analyzed, but verifying that the joint distribution of generated proteins matches natural proteins in long-range contact patterns would provide stronger evidence of global structural realism.
- **Tokenization precision for functional sites**: The LFQ-4k tokenizer's RMSD of 3.31Å introduces structural imprecision that may be especially problematic for enzyme active pockets and binding interfaces requiring sub-Angstrom precision. A focused analysis on how reconstruction error affects design of functional sites would clarify the practical ceiling.
- **Joint embedding visualization**: A t-SNE or UMAP projection of the learned residue representations colored by both fold type and sequence family would help verify that the model genuinely integrates both modalities rather than maintaining them as parallel channels.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"100 sampling iterations is oracle selection"** (Harsh Critic): The paper states "DPLM-2 adopts argmax decoding for 100 sampling iterations." In discrete diffusion, this refers to 100 iterative denoising steps with argmax selection at each step — it is not 100 independent samples with best-of-N selection. The harsh critic's interpretation of oracle-selection inflating performance is a misread of the inference procedure.
- **"No statistical significance testing"** (Harsh Critic): Single-run evaluation without confidence intervals is the standard norm for large-scale protein generation benchmarks in this field. This is not a meaningful weakness for the community standards this paper is evaluated against.
- **"Wet-lab validation absent"** (Spark Finder): Absence of experimental wet-lab validation is standard for computational protein design publications at ICLR and is not a weakness relative to community expectations. All baselines (RFDiffusion, MultiFlow, ESM3) similarly lack wet-lab validation.
- **"Unfair characterization of ESM3 as sequence-first"** (Harsh Critic): The characterization is directionally accurate — ESM3 does use modality dropout that prioritizes sequence, and its generative protocol in practice is cascaded. The harsh critic's objection that this "conflates intent" does not materially misrepresent ESM3's behavior in the benchmarks.
- **"Self-mixup in appendix is insufficient"** (Harsh Critic): While self-mixup is a claimed contribution, its main-paper mention is adequate for a supporting technique; deferred details are standard for ICLR format given page limits.
- **"FAPE loss fails to preserve global topology"** (Harsh Critic): FAPE loss, while local, is computed over all pairwise frame relationships and is a standard reconstruction objective in backbone structure modeling. The assertion that FAPE cannot capture global topology is an overstatement; if this were a real failure mode, it would appear in the reconstruction TM-scores.
- **Requests for multimer evaluation** (Spark Finder): Multimer modeling is explicitly scoped out in the paper's limitations section and is a distinct research problem from single-chain design. Its absence is not a weakness.

---

## Novel Insights

The combination of two observations — the LFQ-4k tokenizer's RMSD reconstruction error (~3.31Å on CAMEO 2022) and the conditional independence factorization in Eq. 2 — points to a structural tension at the heart of DPLM-2 that the paper does not fully surface: the model learns a joint distribution but predicts structure and sequence tokens independently at the output head, meaning residue-level geometry-identity couplings (e.g., a buried hydrophobic residue requiring a specific backbone dihedral) must be captured entirely through the shared transformer representations rather than through any explicit joint output distribution. This implicit coupling mechanism may explain why co-generation outperforms cascaded generation numerically (Table 2, scTM 0.925 vs 0.921/0.907), but only marginally — the joint model is not exploiting a fundamentally richer joint prediction. A future architecture that predicts a proper joint (z_i, s_i) distribution (e.g., a small per-residue joint head rather than two independent heads) could meaningfully improve coupling fidelity, especially for functional residues where backbone geometry and residue identity are tightly co-determined.

---

## Suggestions

- **Add ProteinMPNN and ESM-IF1 to Table 4**: This is the most important fix for the inverse folding section. Present DPLM-2's AAR vs. specialized models with the explicit framing that DPLM-2 is a generalist — the comparison still provides a useful reference point and the existing numbers suggest DPLM-2 (3B) may approach specialized models.
- **Move the (t_d, t_s) training distribution description to the main paper**: Even a single sentence specifying the joint sampling strategy would allow readers to evaluate the conditional generation coverage without consulting appendix §A.2.
- **Address the LFQ-4k vs. LFQ-8k choice explicitly**: Either justify the choice as a vocabulary-size/memory tradeoff with a cost estimate, or show an ablation on downstream generation quality between the two codebook sizes.
- **Resolve the pdb-TM metric direction annotation in Table 2**: Add a footnote or revise the header to clarify whether higher or lower pdb-TM is "better" in the context of novelty vs. quality tradeoff.
- **Add a brief discussion of the conditional independence assumption to the main text**: Even 2–3 sentences acknowledging what the factorization implies and why the shared transformer representations are expected to be sufficient (or what empirical evidence supports this) would substantially strengthen §3.1.
- **Expand the LoRA warm-up analysis**: Report LoRA rank, which layers are adapted, and whether varying these parameters can recover representation task performance — this directly tests the paper's efficiency claim and resolves the open question about catastrophic forgetting vs. data scale.

---

**Overall assessment across axes:**
- *Novelty*: Moderate-to-good. The combination of LFQ tokenization + discrete diffusion multimodal extension + LoRA warm-up is genuinely novel in the protein design context, though each individual component builds on prior work.
- *Technical soundness*: Mostly sound, with the conditional independence assumption being a notable open question requiring fuller treatment.
- *Empirical support*: Mixed. The motif scaffolding and unconditional generation results are convincing; the inverse folding evaluation is incomplete without ProteinMPNN/ESM-IF1 baselines; representation learning results reveal a genuine limitation.
- *Significance*: Good. The open-source, data-efficient multimodal modeling paradigm addresses a real community need, and the motif scaffolding results are practically meaningful.
- *Clarity*: Good overall, with specific issues in metric annotations, inference protocol description, and noise scheduler specification.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 3.0]
Average score: 6.3
Binary outcome: Accept
