Now let me search for calibration papers to anchor the score.Now I have all the information I need to write the consolidated review. Let me produce it.

---

## Summary

ACT-IN-LLM proposes to improve computational efficiency in high-resolution Multimodal Large Language Models (MLLMs) by adaptively compressing only the *Key* and *Value* tokens (while preserving all Query tokens) at specific decoder layers inside the LLM, guided by previous-layer attention weights. Unlike prior Pre-LLM or Early-LLM strategies that permanently discard tokens before or very early in the model, the method retains full queries throughout, preventing irreversible information loss. A theoretical analysis argues that K/V-only compression is a better low-rank approximation of full self-attention than schemes that also compress Q, and controlled experiments on high-resolution and general VQA benchmarks show clear improvements over existing compression baselines.

---

## Strengths

- **Well-motivated empirical diagnosis.** Figure 2 provides compelling direct evidence: dropping tokens earlier causes larger performance gaps (up to 15% on HR tasks), and attention heatmaps across layers show that tokens unimportant early can become important later. This cleanly motivates the in-LLM, non-destructive approach.

- **Clear and novel mechanism.** Compressing only K/V while retaining full Q is a principled design choice that is distinct from all listed baselines (Pre-LLM, Early-LLM, FlexAttention). The complexity shift from O((N+L)²D) to O((N+L)(M+L)D), correctly stated in Table 1, is the right trade-off for this use case.

- **Strong controlled comparison (Table 2).** Under identical training conditions, ACT-IN-LLM outperforms all compression baselines by 5.5% on the average HR score. Notably, even the "without training" variant outperforms all trained compression baselines — a compelling demonstration of the method's inherent advantage.

- **Comprehensive evaluation.** Experiments span controlled comparisons (Table 2), multi-scale LLM experiments (Fig. 7, 0.5B–7B), SOTA comparison (Table 3), and multiple ablation dimensions (compression ratio, method, and layer position).

- **Theoretical framework.** The unified formulation (Eq. 9) and Theorem 3 provide a useful conceptual scaffold for comparing compression strategies, even if the formal proofs have limitations (see Weaknesses).

---

## Weaknesses

### Fatal
*None.*

### Major

- **FlashAttention incompatibility is unaddressed.** The ACM selects K/V tokens based on the explicit attention matrix **A**_{i-1} extracted from the previous layer (Eq. 4). Extracting explicit full attention matrices is incompatible with FlashAttention, which is the standard hardware-efficient attention kernel used in modern LLM deployments. Without FlashAttention, memory overhead and actual latency may be substantially higher than reported (the paper's measurements are on a single V100 with likely standard attention). This directly undermines the efficiency claims and is not discussed anywhere in the paper. If the method requires materializing O((N+L)²) attention matrices at each ACM layer, the memory savings shown in Table 2 (18.8 GB vs. 19.9 GB — barely 6%) would worsen significantly.

- **The theoretical superiority claim is existence-based, not constructive.** Theorem 2 proves that *there exist* matrices C^K and C^V satisfying the low-rank approximation bound (Eq. 11), but the actual algorithm uses top-k token selection guided by the last-token row of the previous layer's attention. There is no theorem establishing that this specific heuristic achieves or approximates the bound from Theorem 2. Similarly, Theorem 3 shows that C^K/C^V-only compression beats all-component compression *in principle*, but not that the implemented procedure achieves this. The paper's claim of "theoretical superiority" therefore exceeds what the proofs actually establish.

- **Efficiency framing is partially misleading.** The abstract and introduction claim "reducing vision tokens by ~60%" and "training/inference time by ~20%." In practice (Table 2): time reduction is 17% (621→515ms) and memory reduction is only 6% (19.9→18.8 GB). Crucially, since all queries are retained, the FFN still processes the full (N+L)-length sequence at every compressed layer, and the Q·K^T product is O((N+L)·(M+L)·D) — symmetric with FlexAttention (Table 1). The "60% vision token reduction" applies only to the K/V side of attention, not to the sequence length processed by the rest of the model. While Table 1 makes this clear, the headline language creates a misleading impression that sequence length is reduced through the stack.

### Minor

- **Last-token guidance not ablated as a design choice.** The method uses the last row of the attention matrix (attention from the final token to all tokens) to score K/V importance. The paper states this encodes "complete multimodal context" but provides no direct validation. The ablation in Table 4b compares attention-weight vs. pooling-based compression, but does not compare *which token's* attention row is used (last token vs. mean over text tokens vs. previous generated token during decoding). This is a core design decision with no empirical justification beyond intuition.

- **Within-stage token indices are frozen, reducing adaptivity.** Section 3.3 states: "for efficiency, we keep the vision tokens index to be identical in each stage." This means the same K/V token set is used across all layers within each stage (early/middle/late), while the paper's framing of "layer-wise" adaptive compression implies per-layer adaptation. The practical adaptivity is therefore coarser than implied.

- **Default 70% ACM layer choice is not the best performer.** Table 5 shows that 50% ACM layers achieves higher performance on both general (75.25 vs. 75.04) and HR benchmarks (46.12 vs. 45.35) than the chosen default of 70%, at only modestly higher latency (552ms vs. 515ms). The paper describes 70% as yielding "the best performance and efficiency trade-off" without a principled criterion, and the choice appears somewhat arbitrary.

### Trivial

- Table 4a uses inconsistent notation (e.g., `{2/2, 2/2, 2/2/2}` — three entries in the last position vs. two elsewhere), making it difficult to parse the configuration format for plain compression rows.

---

## Nice-to-Haves

- **Plug-and-play validation on off-the-shelf public models.** The "without training" result in Table 2 is compelling but only tests the authors' own trained model at inference time without fine-tuning. Demonstrating ACM applied to LLaVA-1.5-HD or InternVL2 without any fine-tuning would greatly strengthen the practical plug-and-play claim.

- **Empirical validation of the error-correction claim.** The paper argues that retaining full queries provides "inherent error correction" (Sec. 3.2), but this is never directly tested. An ablation measuring output fidelity degradation with and without full queries (perhaps by comparing to a version that also compresses Q) would directly validate whether this mechanism operates as described.

- **Analysis of which tokens are selected across layers.** It would be informative to visualize whether the selected K/V token sets actually differ across layers (validating layer-wise adaptivity) or whether similar spatial regions dominate throughout. This directly bears on whether per-stage and per-layer compression decisions add value.

- **Discussion of FLOPs breakdown.** A decomposition of where the efficiency gains actually come from (attention, FFN, or other) and where the ACM overhead lands would help readers understand the true computational trade-off.

- **Video MLLM applicability.** Briefly addressing whether the method extends to video inputs, where N is dramatically larger, would expand the paper's relevance.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic] "Retaining all tokens" is internally inconsistent with "~60% token reduction."** The paper explicitly gives the complexity as O((N+L)(M+L)D) in Table 1 and Eq. 3, and never claims sequence-length reduction through FFN. The "60%" refers to K/V token count. The framing is imprecise but the technical description is correct. This is partially retained above as a "minor framing issue" rather than a structural contradiction.

- **[Human Finder] Hierarchical compression contradicts early-layer observation.** The critic claims using r_i < r_j < r_p (lighter compression early, heavier later) contradicts the observation of denser early attention. In fact, this is *consistent*: smaller r = less compression = more tokens retained, applied to early layers where attention is denser. The criticism is factually incorrect.

- **[Human Finder] "Limited evaluation on fine-grained OCR tasks."** The paper evaluates on VQA-text, ChartQA, DocVQA, and InfoVQA — all text-rich/OCR-demanding benchmarks. The criticism that fine-grained tasks are missing is not supported by the paper content.

- **[Harsh Critic] SOTA comparison (Table 3) is heterogeneous and does not support strong ranking claims.** The paper itself contextualizes this comparison by noting data size differences (e.g., "87.2% of InternVL2 performance with 24% data"), so the authors do not misrepresent it as a controlled apples-to-apples comparison. Within the compressed-token group, the comparison is fair. Keeping as a minor note would be scope creep since the authors are appropriately upfront about the differences.

- **[Harsh Critic/Spark] Requests for statistical significance, confidence intervals, or multiple seeds.** Single-run evaluation is standard practice in MLLM benchmarking at this scale; demanding significance testing is outside community norms.

- **[Harsh Critic] Section 5.2 scaling shows data/model size scaling only for ACT-IN-LLM without baselines.** True, but the section's stated goal is "does ACT-IN-LLM improve with scale," not "does it scale better than baselines." Criticizing its absence is scope creep.

---

## Novel Insights

The paper's most genuinely novel observation — demonstrated both empirically and theoretically — is that the appropriate "unit of compression" within an LLM attention mechanism is the *K/V pair alone*, not all three of Q/K/V. The argument is sound: since the output of attention is Softmax(Q·K^T)·V, compressing K and V reduces the number of "information sources" the model attends to, but compressing Q additionally reduces the number of "question positions" that can look up those sources, causing a second-order accuracy loss. The controlled experimental confirmation in Table 4b — where even non-adaptive pooling of K/V outperforms pre-LLM compression by a wide margin — lends empirical weight to this mechanistic insight beyond the paper's own proposed method.

---

## Evaluation on Key Axes

- **Originality:** High. The K/V-only within-LLM compression strategy is a genuinely new framing distinct from all baselines. The mechanism is principled and well-differentiated.
- **Importance of research question:** High. High-resolution MLLMs face a real quadratic-complexity bottleneck, and the ~9% gap between compressed and full-token models on HR tasks is a meaningful problem.
- **Claims well-supported:** Moderate. The core claim of empirical superiority over compression baselines is well-supported (Table 2). The "theoretical superiority" claim is overstated given existence-only proofs.
- **Soundness of experiments:** Good, with caveats. Controlled comparisons are fair; the FlashAttention compatibility gap means real-world deployment efficiency may differ from reported numbers.
- **Clarity of writing:** Good overall, with minor notation inconsistencies in ablation tables.
- **Value to the research community:** High. The K/V-only insight is portable and the empirical recipe (Table 2) is reproducible within the paper's backbone.

---

## Score and Decision

**Calibration anchors:**
- **LLaVA-Mini** (UQJ7CDW8nb): Scores 8,6,6,6 → Accepted. More dramatic compression (1 vision token), clean pre-fusion + compression story, strong results on 11+ benchmarks. Stronger than ACT-IN-LLM on impact and novelty.
- **Oryx MLLM** (ODiY6pbHZQ): Scores 6,6,6,6 → Accepted. Solid engineering system paper, somewhat incremental (mostly design choices). Similar breadth to ACT-IN-LLM.
- **SparseVLM** (1xG3MN1RRW): Scores 3,5,6,6,6 → Rejected. Also has the FlashAttention incompatibility problem, plus more severe performance degradation. ACT-IN-LLM's results are clearly stronger and more principled.
- **eRAM-V** (GtlV6o1yUy): Scores 3,6,5,6 → Rejected. Has internal design contradictions and weaker empirical support.
- **Visual Token Grouping** (ym1dS37mZE): Scores 6,5,3 → Rejected. Limited ablations, similarity to Q-Former not addressed.

**Positioning:** ACT-IN-LLM is clearly above SparseVLM and eRAM-V (rejected, ~3-5 range). The controlled comparison (Table 2) with same-backbone, same-data settings is more rigorous than most rejected papers in this space. The FlashAttention concern and existence-based theory are real weaknesses but do not invalidate the core empirical finding. The paper sits below LLaVA-Mini in impact (that paper has more dramatic compression and a more surprising insight). It aligns most closely with Oryx MLLM in quality — solid contribution with clear novelty, some incremental aspects, and one addressable engineering concern.

**Final score: 6.0** — Weak accept. The paper makes a genuine and well-supported contribution to efficient high-resolution MLLM design. The FlashAttention concern is the one issue that should be explicitly addressed in a revision; the theoretical claims should also be scoped more carefully to existence results rather than implemented guarantees.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>