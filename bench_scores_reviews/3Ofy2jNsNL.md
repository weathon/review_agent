## Summary
ACT-IN-LLM proposes compressing only Key and Value tokens *within* LLM transformer layers (rather than before the LLM) for high-resolution Multimodal LLMs. The Adaptive Compression Module (ACM) uses the last token's row of the previous layer's attention weights to guide which vision K/V tokens to retain, while preserving all Query tokens and hidden states. The authors provide a unified compression-matrix formulation, a theoretical argument that K/V-only compression yields a better low-rank approximation of full attention than query-or-all compression, and controlled experiments showing a substantial +5.5 point improvement over the best prior compression baseline (FastV) on high-resolution benchmarks while using ~83% of full-model forward-pass time.

---

## Strengths

- **Clearly motivated architectural choice with directly supporting evidence.** Fig. 2(a) concretely demonstrates that dropping tokens in early LLM layers causes up to ~15% degradation on high-resolution tasks, while general benchmarks suffer only ~3%, motivating the need for deferred, layer-wise compression. Fig. 2(b) shows that tokens ignored in early layers gain attention importance in later ones, directly motivating the retention of all query tokens to allow such "recovery."

- **Controlled comparison across compression methods.** Table 2 holds the backbone, training data, cropping strategy, and total token budget constant across all baselines, isolating the effect of the compression strategy. This is methodologically strong and notably better than many cross-paper comparison tables. The 5.5-point gain over FastV on the HR average (45.4 vs. 39.9) and strong gains on ChartQA (46.1 vs. 35.0), DocVQA (45.2 vs. 38.6), and InfoVQA (31.6 vs. 27.7) are substantial.

- **Effective "w/o train" performance.** The model achieves 43.5 HR average even without retraining (vs. FastV's 38.7 w/o train), indicating that the architectural design itself—not just the training recipe—carries the improvement.

- **Hierarchical compression ratio design is principled and validated.** The ablation in Table 4a shows that a hierarchical approach (r_i < r_j < r_p) consistently outperforms flat compression ratios, which aligns with the observed sparsification of attention in deeper layers (Fig. 2b). The separate HR/LR ratios are also empirically justified via the low-rank degree analysis (Fig. 5a).

- **Theoretical framework provides useful structure.** The compression-matrix formulation (Eq. 7–9) cleanly unifies Pre-LLM, FlexAttention, and ACM strategies in a single framework, making Table 1's complexity comparison precise and enabling formal comparisons.

---

## Weaknesses

### Fatal
None.

### Major

- **Critical ablation missing: Q-compression vs. K/V-only compression.** The central architectural claim is that retaining all Query tokens provides an "implicit error correction mechanism." This is the primary justification for the asymmetric design Com(I, C^K, C^V) over the symmetric Pre-LLM Com(C^Q, C^K, C^V). Yet no experiment directly compares within-LLM Com(C^Q, C^K, C^V) against within-LLM Com(I, C^K, C^V). Without this ablation, one cannot determine whether the gains in Table 2 come from *in-LLM* compression generally (regardless of whether Q is compressed) or specifically from *retaining Q tokens*. This single experiment would either validate or significantly reframe the paper's core contribution.

- **Theory-algorithm gap in the core theoretical claims.** Theorem 2 proves the *existence* of compression matrices C^K, C^V achieving a bounded approximation error—but the actual ACM selects tokens via a deterministic top-k rule driven by the previous layer's last-row attention weights. There is no proof or empirical validation that this specific selector achieves (or approaches) the approximation bound. Similarly, Theorem 3's dominance claim ("with probability 1−o(1)") is stated for any C^K, C^V, including the ones found by arbitrary heuristics, making the theorem true but potentially vacuous as a justification for the proposed algorithm specifically. The paper should either (a) prove that the proposed top-k selector satisfies the approximation conditions, or (b) frame these theorems explicitly as existence motivations rather than guarantees.

- **Notation inconsistency in Eq. 9 undermines the theory section.** In Eq. 1, **A**_{i,h} is defined as the *post*-softmax attention weight. Yet in Eq. 9, the compressed attention is written as softmax(**C**^Q **A** (**C**^K)^⊤)·**C**^V **V**, which places **A** inside another softmax, implying it is the *pre*-softmax logit matrix **Q**·**K**^⊤/√D. This notational inconsistency propagates through Theorems 1–3 and should be resolved, as it currently makes Section 4 difficult to verify.

### Minor

- **The text-guided (attention-weight) selection provides only marginal gains over AvgPool-1D.** Table 4b shows: Attention-weight (HR 45.35 / General 75.04) vs. AvgPool-1D (HR 45.08 / General 75.06). The difference is 0.27 HR points and effectively zero on general benchmarks. While all ACM variants substantially outperform Pre-LLM (39.15 / 72.28), the claimed advantage of *text-guided* adaptive selection—arguably the paper's most novel design element—is not strongly supported. This does not invalidate the paper, but the contribution should be recalibrated: the main gain appears to come from in-LLM K/V compression itself, not specifically from last-row attention guidance. A stronger comparison (last row vs. average text rows vs. current query row) would clarify this.

- **Positional encoding structure after K/V subsampling is unaddressed.** When a subset of K/V positions is sampled, the spatial layout of retained vision tokens becomes irregular. For OCR/document tasks where spatial position carries meaning, this could matter. The paper does not discuss whether any form of position-aware selection or re-indexing is applied.

- **Efficiency claims are incompletely reported.** The paper claims "~20% training/inference time reduction" but reports only single forward-pass times (in ms). No training throughput (iterations/second or total wall-clock time) is reported, and no breakdown of prefill vs. decode latency is provided. Since K/V compression affects the prefill stage more than decoding (where KV cache is already truncated), actual end-to-end generation speedup may differ significantly from prefill-only numbers.

- **The "+6.3% improvement" headline figure is inconsistent with reported data.** The abstract reports "6.3% improvement over existing token compression techniques" while the introduction says "6.2%" and Table 2 shows 45.4 − 39.9 = 5.5 points. The paper should standardize and clearly define which metric/aggregation produces this number.

### Tiny

- **Section 5.2 text is numerically confusing.** The sentence "ACT-IN-LLM(0.5B) achieves 54.58%, while ACT-IN-LLM(3B) reaches 67.00%, resulting in a 6.23% gain when scaling from 3B to 7B" mixes two different model transitions in the same sentence.

- **Head-averaging details for ACM are underspecified.** Eq. 3 states ACM uses "the averaged attention weight from the i−1-th layer" (**A**_{i−1}), but the paper does not specify whether averaging is over heads only, over both heads and batch dimension, or how head-heterogeneous importance signals are reconciled. This matters for reproducibility.

- **No limitations section.** The paper does not discuss where the method may fail—e.g., tasks requiring many spatially dispersed fine details, long decoding sequences where important token identity evolves, or architectures where FFN rather than attention dominates compute (limiting practical efficiency gain).

---

## Nice-to-Haves

- **Ablate last-row attention vs. other text-token guidance signals** (average over all text rows, max over text rows, current generation token). This would either strengthen or narrow the contribution of the text-guidance design choice, which is currently hard to assess from Table 4b alone.

- **Visualization of spatially retained tokens across layers given different queries.** For a method claiming "text-guided adaptive compression," showing which image regions survive at early vs. mid vs. later layers for different query types (e.g., "What is the chart title?" vs. "Count the objects in the bottom-left") would provide compelling qualitative evidence.

- **Text-only benchmark evaluation** (e.g., MMLU) to confirm that modifying self-attention K/V across all LLM layers does not degrade the backbone's language capabilities.

- **Training wall-clock and GPU memory during training**, not just inference metrics, to fully substantiate the training efficiency claim.

- **Failure case analysis**: examples where ACT-IN-LLM fails relative to the full-token model would clarify practical limitations and help practitioners know when to apply this method.

---

## Removed Points
*These points were flagged for removal—treat them with caution.*

- **"Unfair zero-shot comparison" (Spark Finder):** The reviewer argued that comparing "Ours w/o train" vs. "FastV w/o train" is unfair because FastV is designed to be training-free while ACT-IN-LLM requires training. This is **removed**: the purpose of the "w/o train" rows is precisely to show that the architectural design of ACT-IN-LLM alone (independent of training) outperforms FastV (the best training-free prior method). This comparison intentionally holds the asymmetry to make a *stronger* point in the paper's favor, which is methodologically sound.

- **Missing related works (all three reviewers):** Per instructions, references to missing related works are removed, as the reviewer cannot verify existence of external sources.

- **Figure 6 suspicious data (Critic):** The table extracted from Figure 6 appears to show identical values for Full, Cabstractor, UHD, and Ours—this is almost certainly a PDF extraction artifact from overlapping figure elements, not an actual data error in the paper.

- **"Baselines may not be fully optimized" (Critic):** The paper explicitly states all methods are trained under identical conditions (same epochs, training data, LR, slicing strategy). The concern that some methods might perform better under their own "optimal" regimes is a general caveat about any controlled study, not a specific flaw here.

- **Broader impact section requested (Critic):** Absence of a broader impact discussion is a formatting/convention request, not a substantive weakness relevant to ICLR review.

---

## Novel Insights

The most actionable insight from the review synthesis is a potential reframing of the paper's contribution: the evidence in Table 4b suggests that the primary gain of ACT-IN-LLM may come from *the location and type of compression* (within-LLM, K/V only) rather than from *the specific text-guided attention-based selection mechanism*. If an ablation showing within-LLM K/V compression with AvgPool-1D achieves nearly the same result, the paper's main contribution is the compression strategy architecture (where and what to compress), with the text-guided selector being an optional enhancement. This reframing, while potentially deflating one claimed novelty, would actually make the theoretical argument (Theorems 1–3 about K/V-only approximation quality) more central and self-consistent, and the paper would still represent a meaningful contribution.

---

## Suggestions

1. **Run the within-LLM Com(C^Q, C^K, C^V) ablation** — apply the same hierarchical in-LLM compression but also compress Q tokens (matching Pre-LLM architecturally but placed inside the LLM). This experiment is essential to isolate whether "in-LLM placement" or "Q retention" drives the gain.

2. **Clarify and tighten the theoretical claims** — either rename Theorem 2 as a "motivating existence result" and not treat it as an algorithmic guarantee, or add Proposition/Corollary showing that the top-k attention-based selector satisfies the approximation conditions under the stated assumptions.

3. **Fix the Eq. 9 notation** — introduce separate notation for pre-softmax logits (e.g., **L** = **Q**·**K**^⊤/√D) and post-softmax weights (**A** = softmax(**L**)), and be consistent throughout Section 4.

4. **Report actual training throughput** (steps/sec, total hours, GPU memory during backward pass) alongside the inference timing to fully substantiate the claimed ~20% efficiency reduction.

5. **Add a last-row vs. text-average attention guidance ablation** to Section 5.4, since this is the specific design choice within the ACM step and is currently untested against natural alternatives.

---

**Evaluation summary:**

- **Novelty:** Moderate-to-good. The specific combination of within-LLM K/V-only compression with text-guided adaptive selection for high-resolution MLLMs is genuinely new, though the closest antecedents (FastV, FlexAttention) are nearby.
- **Technical soundness:** Moderate. The empirical method is sound and well-controlled; the theoretical section has real notation and connectivity issues that need resolution before it can be taken at face value.
- **Empirical support:** Good. The controlled comparison in Table 2 is the paper's strongest asset; the +5.5 HR-average point gain over FastV under matched training is convincing. The scaling experiments in Section 5.2 are suggestive but lack ablation baselines.
- **Significance:** Good. Efficient high-resolution MLLM inference is a practically important problem, and the proposed approach addresses a real limitation of prior token-dropping strategies.
- **Clarity:** Adequate. High-level narrative is easy to follow, but several technical details critical for reproducibility and for evaluating the theory are underspecified or inconsistent.