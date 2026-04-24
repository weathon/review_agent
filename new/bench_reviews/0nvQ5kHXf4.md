Now let me run calibration searches to properly anchor my score.Now I have sufficient information to write the final consolidated review. Let me synthesize all findings.

---

## Summary

WASI (Weight-Activation Subspace Iteration) is a method for resource-constrained fine-tuning of transformer models on edge devices. It jointly compresses weight matrices and activation maps into low-rank subspaces by exploiting the empirically observed stability of the intrinsic weight subspace during fine-tuning. The method runs both forward and backward passes entirely in the compressed representation, enabling inference to benefit from the compressed architecture — unlike LoRA-style adapters. Results are reported on ViT, SwinT, and TinyLlama across five datasets, with on-device hardware validation on a Raspberry Pi 5.

---

## Strengths

- **Joint compression of both weights and activations in a unified framework (Eqs. 8–11, Sec. 3.3)**: Unlike LoRA (which preserves inference cost) and ASI (which only compresses activations), WASI compresses both components simultaneously. The forward pass ($A_{i+1} = A_i R_i^T L_i^T$) and gradient propagation both operate in the compressed representation, enabling true inference-time savings — a principled and coherent design.

- **Subspace reuse validated to preserve convergence (Fig. 3b)**: The WSI mechanism avoids re-running full SVD at every iteration by using Gram-Schmidt orthogonalization. The experiment shows WSI requires 1.36× fewer FLOPs than full SVD to achieve the same accuracy, and at matched FLOPs beats full SVD by ~35% — directly validating the core efficiency claim.

- **Real on-device hardware measurements on Raspberry Pi 5 (Sec. 4.4, Fig. 8)**: Wall-clock measurements showing a consistent 1.4× speedup over vanilla training at ε=0.9 are among the most credible results in the paper; actual hardware timing is far more informative than FLOPs estimates alone.

- **Principled information-loss control via explained-variance threshold ε (Eqs. 5–7)**: Unlike ASVD/FWSVD, which lack a theoretical basis for truncation, WASI ties truncation to a single interpretable hyperparameter that controls compression–accuracy trade-off across all layers and architectures.

- **Breadth of empirical validation**: Experiments span ViT and SwinT on five datasets (CIFAR-10/100, CUB, Flowers, Pets), with additional LLM experiments on TinyLlama/BoolQ. SwinT results (Fig. 6) show consistent Pareto improvement across all five datasets.

---

## Weaknesses

### Fatal

None.

### Major

- **Yang et al. (2023a) ["Efficient Low-Rank Backpropagation for Vision Transformer Adaptation"] cited but not benchmarked**: This NeurIPS 2023 work specifically reduces backpropagation cost for ViT adaptation via low-rank gradient approximation — arguably the most directly comparable published method for WASI's primary setting. The paper references it in the bibliography and in the related work section but does not include it in any experiment. Section 4.1 notes baselines are "as discussed in Secs. 1, 2, Appendix A.5," implying Appendix A.5 contains a justification, but this cannot be verified in the main submission. Without comparison against Yang et al. (2023a), the claim of state-of-the-art efficiency on ViT is unsubstantiated. The SVD-LLM comparison, while technically present, is partially undermined by the paper's own acknowledgment (Sec. 1, Sec. 2) that SVD-LLM "cannot be directly applied to all vision transformer-based models." This creates an experimental structure that avoids the most relevant competing method while including one the authors concede is ill-fitted to the task.

- **Weight subspace stability validated only in one narrow setting**: The foundational empirical claim — that the essential subspace remains stable during fine-tuning and can be reused across iterations — is verified in Fig. 3a for a single (architecture=ViT, dataset=Pets, ε=0.8) combination. This is the central assumption supporting the entire method. If stability is architecture- or dataset-dependent (e.g., weaker on SwinT or more aggressive ε values), the method's behavior could degrade in untested regimes. No multi-setting analysis is provided.

- **Update rule in Eq. 11 and Algorithm 1 lacks clarity**: Algorithm 1 takes the weight tensor $W_{i,(t)}$ as input and produces $L_{i,(t)}, R_{i,(t)}$ via subspace iteration. But Eq. 11 writes the update as $L_i R_i \leftarrow L_i R_i + \eta \cdot \partial\tilde{\mathcal{L}}/\partial W_i$, updating the *product* rather than the individual factors. The paper does not explain how the updated product is re-factored into individual $L_i$ and $R_i$ for the next iteration of Algorithm 1 — whether by re-materializing the full matrix and applying WSI (which may be expensive), or by some other mechanism. The published algorithm as written is operationally incomplete. Appendix A.1 likely provides detail, but this is load-bearing algorithmic description that should appear in the main text.

### Minor

- **Abstract-level compression claims lack context**: The abstract states "reducing memory usage by up to 62×" without noting this applies to linear layers within MLP blocks. While Section 4.1 explicitly discloses this scope and references extended results with attention layers in Appendix B.3, it would be clearer to briefly qualify this in the abstract (e.g., "MLP linear layers") to avoid misreading.

- **TinyLlama experiment is too limited to support generalization claims**: The LLM experiment (Sec. 4.3, Fig. 7) fine-tunes only the last 5 layers at ε=0.1 — an extremely aggressive compression that is not used anywhere in the vision experiments (ε ∈ {0.4, …, 0.9}). Accuracy values are only referenced via Fig. 7 rather than stated numerically in text. These design choices make it difficult to evaluate whether WASI genuinely generalizes to decoder-only LLMs or whether the narrow scope was chosen specifically because it worked. The claim "without accuracy loss" needs quantitative grounding.

- **Lack of WSI-only vs. ASI-only vs. WASI ablation**: WASI is presented as combining WSI (new contribution) with an improved ASI (same group's prior work). While WASI vs. ASI comparisons appear in Fig. 5, no experiment isolates WSI's contribution alone. The fraction of performance gain attributable to WSI vs. the improved dynamic-programming ASI vs. their combination is unclear.

- **Speedup comparison with ASI lacks discussion**: ASI achieves 1.56× speedup on compact CNNs (Sec. 2); WASI achieves 1.4× speedup on ViT (Sec. 4.4). The paper does not discuss this apparent regression when moving from CNNs to transformers. Some explanation (e.g., attention-layer overhead not compressed, larger model fixed costs) would be informative.

### Trivial

None worth flagging beyond what is above.

---

## Nice-to-Haves

- Convergence curves (accuracy vs. epoch) for WASI vs. vanilla at different ε values would clarify whether lower-ε runs converge more slowly or show instability — currently the Pareto-front plots hide temporal training dynamics.
- Extending the TinyLlama experiment to full fine-tuning (all layers) at ε values comparable to the vision experiments would substantially strengthen the generalization claim.
- A brief main-text clarification of how Eq. 11's product-update is operationalized (i.e., whether the full weight matrix is reconstructed for WSI at the next step) would resolve the algorithmic ambiguity without requiring the reader to consult Appendix A.1.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic — "headline efficiency numbers presented without disclosure"**: The paper explicitly states in Section 4.1 "focusing on linear layers within multi-perceptron blocks for fair comparison with previous methods (extended results with attention layers in Appendix B.3)." The disclosure exists in the experimental setup; the abstract not repeating it is acceptable practice. Removed as overstated.

- **Harsh Critic — "SVD-LLM comparison is structurally unfair"**: The paper acknowledges SVD-LLM is not designed for vision transformers. Under the hard rules, if asymmetry favors the baseline (SVD-LLM gets lower FLOPs via LoRA adapters, which WASI cannot match on compute), this is intentionally asymmetric to prove a narrower point (memory efficiency). This critique is partially valid for the FLOPs sub-claim but not for the memory comparison. Removed as framed; retained only as the Yang et al. 2023a missing-baseline concern.

- **Harsh Critic — "LBP-WHT attribution conflates Yang et al. (2023b) and (2023a)"**: The related work section says "Gradient Filter (Yang et al., 2023b)..." and later "LBP-WHT (Yang et al., 2023b)...". Yang et al. (2023b) is "Efficient On-Device Training via Gradient Filtering" which does use Hadamard-like operations. This is likely a labeling inconsistency rather than a citation error affecting scientific content. Removed as formatting/minor labeling issue.

- **Harsh Critic — "GaLore and LoRA not benchmarked"**: GaLore targets LLM pre-training memory reduction (optimizer states), not the on-device inference-efficient fine-tuning setting WASI targets. LoRA adds inference overhead and is scoped out by design. Removing per soft rules (outside stated scope).

- **Strength Finder — "generality across transformer architectures including TinyLlama"**: This strength conflicts with the verified weakness (only 5 layers, ε=0.1 only). Moved to removed per rule that strength vs. weakness disagreement means weakness wins.

---

## Novel Insights

WASI's most distinctive architectural decision is treating the compressed model (L_i R_i) as the *actual model* rather than an adapter: inference runs directly on the factored representation, eliminating the LoRA merge step and preserving compression benefits at deployment. The combination of warm-started subspace iteration with an explained-variance threshold is a principled mechanism that stabilizes rank selection (avoiding the brute-force rank search of prior ASI) while preserving information-theoretic interpretability. The empirical result that subspace stability enables SVD-free iteration across epochs — saving 1.36× FLOPs over repeated SVD at matched accuracy — is a practically useful and underappreciated finding. Its main limitation is that it has only been validated in one setting.

---

## Suggestions

1. Include Yang et al. (2023a) as a direct comparison on ViT/SwinT classification, or provide a clear main-text argument for why it is not applicable (summarizing what Appendix A.5 says).
2. Validate rank stability across at least 2–3 additional (architecture, dataset, ε) combinations to give the foundational hypothesis broader empirical support.
3. Clarify Eq. 11 in the main text: state explicitly whether $W_i$ is reconstructed from $L_i R_i$ for the WSI step at the next iteration, and what the memory cost of that reconstruction is.
4. Match ε values in the TinyLlama experiment to those used for vision transformers (0.4–0.9) and report accuracy numerically in text to enable meaningful comparison.
5. Add a 2-row ablation table (WSI-only vs. improved-ASI-only vs. WASI) on one dataset to isolate each component's contribution.

---

## Score and Decision

**Calibration anchors consulted:**

| Paper | Path | Avg Score | Comparison to WASI |
|-------|------|-----------|-------------------|
| LDAdam | `/human_reviews/Zkp1GuHerF.md` | 7.0 (Accept) | Stronger: convergence proofs, broader LLM experiments, thorough baselines |
| LoRAM | `/human_reviews/s7DkcgpRxL.md` | 6.2 (Accept) | Similar character (practical memory-efficient training), stronger LLM coverage, comparable real-hardware validation |
| DeLoRA | `/human_reviews/X1U74IwuxG.md` | 5.25 (Accept) | Similar tier: real contribution, some experimental gaps |
| MoE-SVD | `/human_reviews/ho7ZUS1z8A.md` | 5.0 (Reject) | Similar tier: SVD-based compression, moderate novelty |
| EVA | `/human_reviews/DM6Q45HWSk.md` | 4.75 (Reject) | Similar: SVD-initialized LoRA, missing ablation and baselines |
| TPI-LLM | `/human_reviews/0cadcLKbt7.md` | 4.0 (Withdrawn) | Edge deployment system; has real hardware but weaker novelty |
| ELR-Diffusion | `/human_reviews/edx7LTufJF.md` | 2.5 (Withdrawn) | Much weaker: missing critical baselines, copied text |

**Positioning:** WASI sits between LoRAM (6.2) and EVA (4.75). It has stronger on-device hardware validation than most and a genuine contribution (joint weight-activation compression in stable subspaces for transformers). However, the missing Yang et al. (2023a) comparison, one-setting stability validation, and algorithmic clarity gap in Eq. 11 keep it below the 6+ band. The TinyLlama experiment is too limited to materially strengthen the contribution. Anchoring between DeLoRA (5.25, accepted with weaknesses) and EVA (4.75, rejected due to baseline gaps), and weighting the unaddressed baseline concern as a major issue, I arrive at **5.0**.

**Axes:**
- *Originality*: Moderate — joint weight-activation low-rank training for transformers is a clear extension of prior ASI work; the subspace-iteration for weights is new and sensible.
- *Importance of research question*: High — on-device transformer training is practically important and understudied.
- *Claim support*: Partial — core claims well-supported for vision transformers, but the Yang et al. (2023a) gap and limited stability evidence weaken confidence.
- *Soundness of experiments*: Moderate — good breadth across datasets, genuine hardware evaluation, but key baseline missing and LLM experiment inadequate.
- *Clarity of writing*: Good — well-organized, though Eq. 11 update rule needs clarification.
- *Value to research community*: Moderate — practical value for edge deployment; code released.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>