## Summary
The paper proposes ACT-IN-LLM, a vision-token compression scheme for high-resolution multimodal LLMs that keeps all query tokens but compresses keys/values within selected layers using a text-guided, last-token attention heuristic. It provides a unified formulation of token compression, theoretical low-rank approximation results, and extensive experiments showing improvements over several existing compression baselines and competitive performance relative to larger, non-compressed MLLMs.

## Strengths
- Clear and well-scoped problem: improving efficiency for high-resolution MLLMs while mitigating the performance loss observed with pre-/early-LLM compression, supported by diagnostic experiments in Fig. 2 (layer-wise dropping) and attention visualizations.
- Methodologically simple and implementable: ACM is just a K/V subsampling module before self-attention (Eq. 3–6, Fig. 3–4), making it easy to integrate into standard decoder-only MLLMs without architectural surgery.
- Strong controlled baseline comparison: Table 2 uses a shared encoder (CLIP-ViT-L/14), LLM (Vicuna-7B), cropping scheme, and training setup for all compression methods while varying only the compression strategy, and shows a sizable gain over the previous best (FastV) on high-resolution benchmarks (45.4 vs. 39.9) with modest time/memory savings.
- Broad empirical coverage: comparisons span both general and high-resolution benchmarks, multiple compression paradigms (Pre-LLM, interaction-based, Early-LLM), and multiple backbones/scales (Sec. 5.2, Fig. 7).
- Ablations are relatively thorough: Tables 4(a–c) and 5 explore compression ratios, layer placement, and alternative ACM implementations, giving useful insight into what aspects of the design matter for the efficiency–accuracy trade-off.

## Weaknesses

### Fatal
None.

### Major
- **Theory–algorithm disconnect and overstated theoretical claims.**  
  Sec. 4.2–4.3 (Theorems 1–3) are stated for generic compression matrices \(C^K, C^V\) under Assumption 1 and show that *there exist* KV-only compression matrices that approximate full attention better than other patterns. However, the implemented ACM is a *specific* deterministic top‑k selection driven by last-layer attention to the last token (Eq. 4–6); there is no argument that the actual selection matrices produced by this heuristic satisfy the conditions or concentration bounds of Theorem 2, nor that they realize the probabilistic comparisons of Theorem 3. The text repeatedly phrases this as “we theoretically show the superiority of our proposed ACM” (Sec. 4.1, Abstract, lines 47–48), which suggests a guarantee about the *implemented* algorithm, not just about the existence of some KV-compression. This is a substantive overclaim: the core empirical method is only loosely related to the theoretical constructions. The theory is valuable as heuristic support for KV-only compression but does not rigorously back the claimed superiority of ACT-IN-LLM itself.

- **Conceptual overstatement about “retaining all tokens across layers” and “implicit error correction.”**  
  The paper claims that ACT-IN-LLM “retains all tokens across layers, ensuring an implicit error correction mechanism that mitigates the loss of critical information” (Intro, lines 45–47; Fig. 1(b) text). In reality, the ACM permanently discards subsets of vision tokens from K/V in each layer (Eq. 6), so once a token is dropped from K/V at later layers, it cannot newly influence subsequent computations. This is qualitatively similar in irreversibility to pre-/early-LLM compression—information can only propagate through whatever influence it had before being removed from K/V. The key difference is *when* and *which* tokens are dropped, not that ACT-IN-LLM fundamentally avoids irretrievable loss. The strong “all tokens retained” / “error correction” narrative is therefore misleading relative to the actual mechanism.

- **Evidence does not convincingly isolate the contribution of the “text-guided” last-token heuristic.**  
  The central story is that ACM is “text-guided” by using the last row of the previous layer’s attention \(\mathbf{A}_{i-1}[N+L,:]\) as importance scores for visual tokens (Sec. 3.2). However, (i) the paper does not rigorously justify why this particular row is a good proxy for vision–text interaction, especially under causal decoding where the last token’s attention is shaped by prior outputs and context; and (ii) Table 4(b) indicates that a simple AvgPool-1D ACM (no attention guidance) slightly *outperforms* the attention-based ACM on the “general” benchmarks and is essentially tied on high-resolution. This suggests that the main gains may come from the *location* (in-layer KV-only compression) and the ratio schedule, not from the text-guided top‑k heuristic. Yet the abstract and conclusion attribute performance to “text-guided” adaptive compression without acknowledging that simpler, text-agnostic variants perform comparably.

- **Over-interpretation of SOTA comparisons where architecture/training are not controlled.**  
  Table 3 compares ACT-IN-LLM to a heterogeneous set of MLLMs that differ in vision encoders, token budgets, pretraining and SFT data sizes. The text (Sec. 5.3–5.4, lines 326–328) presents these results as evidence that ACT-IN-LLM “obtains the SOTA performance” among ≤1k-token methods and is “competitive” with large non-compressed models, implicitly ascribing much of this to the compression strategy. However, without controlled experiments that swap compression while holding architectures and data fixed, these numbers primarily demonstrate that the *overall system* is strong, not that the proposed compression is the salient causal factor. The paper should soften these SOTA claims or explicitly state that they are indicative, not controlled.

### Minor
- **Ambiguity and potential issues in the “text-guided” mechanism under causal decoding.**  
  Sec. 3.2 uses the last row of the average attention \(\mathbf{A}_{i-1}[N+L,:]\) as the guide, but does not clarify precisely which token this corresponds to in realistic prompting/generation regimes (e.g., inclusion of system prompts, user text, and intermediate outputs). It is also not discussed how, during generation, evolving outputs might create a feedback loop where earlier mistakes in the last-token sequence influence future compression. While this may not invalidate the method empirically, it weakens the conceptual interpretation as faithfully “text-guided.”

- **Some implementation details of ACM are underspecified.**  
  For reproducing and analyzing the method, more clarity would help on: how attention weights are averaged across heads (Eq. 3 only says “averaged attention weight”); the exact construction and shape of the sampled mask \(\bar{\mathbf{M}}_i\) in Eq. 6 (dimensions in the text and the figure are slightly confusing); and whether top‑k selection is done per head or on an aggregated score. These are not fatal, but they do matter for faithful reimplementation and for understanding how sparsity patterns in Fig. 4(b) arise.

- **Assumption 1 is only weakly connected to ACM’s concrete design.**  
  Assumption 1 (“vision tokens receive much less attention than text tokens”) is empirically supported by Fig. 5(b) and underpins the existence results in Theorem 2, but it is not analyzed how sensitive ACM’s benefits are to this property. For example, if future MLLMs are trained to attend more strongly to visual tokens, would the same KV-only scheme remain optimal? Some discussion or a small-scale experiment probing this sensitivity would strengthen the theoretical story.

- **Scaling experiments lack baseline comparisons at the same scale.**  
  Sec. 5.2 and Fig. 7 show that ACT-IN-LLM improves with larger LLMs and more SFT data, but there is no corresponding full-token or simple-compression baseline under the same scaling. These curves thus mainly show that “bigger models with more data work better,” not that the proposed compression scales more favorably than alternatives.

- **Fig. 6/table description is a bit confusing.**  
  The parsed figure caption and tabular numbers in lines 220–250 look stylized and may be condensed for illustration, but the main text (lines 276–278) could be clearer about what is *measured* vs. schematic, and about the exact settings behind the reported “∼65% times compared with the full model” statement.

### Trivial
- Minor wording/claim tightening: e.g., softening phrases like “ensure no vital information is lost” and “we theoretically demonstrate superiority” to better match what is actually proved and observed.
- Minor clarifications in Sec. 3.3 on maintaining identical indices within each stage and how that interacts with hierarchical ratios.

## Nice-to-Haves
- A focused experiment that more directly isolates the “text-guided” component: e.g., comparing (i) current last-token attention-based ACM, (ii) uniform/random subsampling with same per-layer ratios and positions, and (iii) static spatial or slice-level heuristics, all under identical training, to quantify how much additional benefit the attention heuristic brings beyond KV-only, in-layer compression.
- More qualitative visualizations of which visual regions are preserved/dropped across layers for representative questions, to assess whether selection aligns with human notions of relevance.
- A short theoretical or empirical analysis of how ACM behaves during auto-regressive decoding (with growing text) vs. in the pure encoding regime used in most evaluations.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the theory is “almost entirely disconnected” from the algorithm.**  
  While it is true that the theorems use existence arguments and do not analyze the exact top‑k heuristic, the unified formulation in Sec. 4.1 explicitly represents ACM as \(\text{Com}(\mathbf{I}, C^K, C^V)\), and the empirical Fig. 5 is designed to tie Assumption 1 to observed MLLM behavior. The issue is overclaiming, not complete disconnection, so I treat this as a *major* but not “fatal” flaw.
- **Assertion that attention-based ACM is clearly not better than AvgPool-1D.**  
  Table 4(b) (as described in the text) indicates near parity, not decisive inferiority. The main valid point is that the text-guided heuristic isn’t clearly *better*; claiming it is *worse* is not supported strongly enough to emphasize.
- **Allegation that Fig. 6 uses extrapolated (“theoretical bound”) numbers beyond OOM.**  
  The parsed table looks schematic and may reflect PDF-to-text artifacts. There’s no explicit evidence in the main text that the authors extrapolate invalid data; this should not be framed as a reliability concern.
- **Speculation that the FastV baseline is not trained under comparable conditions.**  
  Sec. 5.1 explicitly states that “all other settings” including training schedule are kept constant and that FastV and “FastV w/o train” are both included. Without contradictory evidence, I will not treat fairness of their FastV reimplementation as a serious weakness.
- **Criticism that interaction-based methods “fail to fully align” is unsupported.**  
  This is a rhetorical claim in related work; given our constraints on external validation, I do not weigh it as a substantive flaw.

## Novel Insights
None beyond the paper’s own contributions; the main issues are standard but important: the theoretical section overstates what is actually proved about the concrete algorithm, and the empirical narrative somewhat over-attributes gains to the “text-guided” aspect and to the compression method when some gains may instead come from generic KV-only, in-layer compression and broader architectural/training choices.

## Suggestions
- Reposition the theory as establishing the *potential* benefits of KV-only compression under Assumption 1, and clearly state that the current top‑k, last-token ACM is a heuristic motivated by (but not guaranteed by) that theory; adjust wording in the abstract, Sec. 4, and conclusion accordingly.
- Clarify the information-loss story: emphasize that ACT-IN-LLM postpones and distributes K/V dropping in a way that empirically harms performance less than pre-/early-LLM compression, rather than claiming it “ensures no vital information is lost.”
- Add an ablation that more tightly isolates the effect of attention-based/text-guided selection vs. simple or random K/V subsampling under the same layer schedule and ratios, and report any gains or ties honestly in the narrative.
- Expand Sec. 3.2 with more detail on how the last token is defined in typical prompts and during generation, how attention is aggregated across heads, and how \(\bar{\mathbf{M}}_i\) is shaped and applied, to improve reproducibility and interpretability.
- In the SOTA comparison section, explicitly acknowledge that many differences are due to heterogeneous backbones and training recipes, and present Table 3 primarily as evidence that ACT-IN-LLM can be embedded in a strong system, rather than as isolating the causal effect of the compression scheme.

### Evaluation on key axes
- **Originality:** Moderate–good. KV-only, in-layer compression is not entirely novel conceptually, but the specific combination for high-res MLLMs with a simple ACM and unified theoretical framing is a useful contribution.
- **Importance of question:** High. Efficient high-resolution multimodal reasoning is a significant and timely problem.
- **Support for claims:** Empirical support for “KV-only, in-layer compression can be strong” is solid; support for the stronger claims about theoretical guarantees, information preservation, and uniquely beneficial “text-guidance” is weaker and overstated.
- **Soundness of experiments:** Generally sound for the controlled Vicuna-7B comparison; less so for broader SOTA claims where confounders remain.
- **Clarity:** Good overall, with localized ambiguities in ACM implementation and theoretical framing.
- **Value to community:** Reasonably high as an engineering contribution; with toned-down claims and clarified analysis, it would be a solid efficiency paper.

## Score and Decision

### Calibration anchors
- **High-score anchors (>7):**
  - `/home/wg25r/review_agent/human_reviews/SI2hI0frk6.md` (Transfusion, avg 7.6, oral): very strong, clean contributions and tight alignment between theory, method, and experiments; clearly stronger overall than this paper.
  - `/home/wg25r/review_agent/human_reviews/gJeYtRuguR.md` (multi-exit ViT + token reduction, avg 7.5, poster): strong empirical study of token reduction with clear, limited claims; comparable problem area but fewer theoretical overclaims than ACT-IN-LLM.
- **Medium-score anchors (4–6):**
  - `/home/wg25r/review_agent/human_reviews/Uhj5OxAz7I.md` (Matryoshka multimodal models, avg 6.0, poster): simple, well-executed token granularity control with clean experiments and modest claims; similar in spirit and quality, slightly better narrative tightness than ACT-IN-LLM.
  - `/home/wg25r/review_agent/human_reviews/mb2ryuZ3wz.md` (Adaptive length image tokenization, avg 5.75, poster): interesting variable-length tokens with good analysis but some baseline and complexity concerns; comparable mix of strengths/weaknesses.
  - `/home/wg25r/review_agent/human_reviews/abHtkQkumD.md` (Learnable context-aware attention mask, avg 5.0, reject): interesting idea with somewhat underspecified method and overclaims; slightly weaker empirically than ACT-IN-LLM.
- **Low-score anchors (<3):**
  - `/home/wg25r/review_agent/human_reviews/N0vzm0vwyR.md` (CNN pruning via conditional MI, avg 2.33): unclear motivation and weak experimental clarity; significantly weaker than the current paper.
  - `/home/wg25r/review_agent/human_reviews/rKMz6cDE7W.md` (streaming attention approximation, avg 2.33, reject): more serious methodological issues and weaker validation than here.

Relative to these anchors, ACT-IN-LLM has stronger and broader experiments than the low-band papers and is clearly more mature. It is roughly on par with the medium-band token compression works (Matryoshka, adaptive tokenization), but with more overclaim in theory and conceptual framing, which brings it slightly below the best of them. It is noticeably weaker than the high-band works in terms of claim–evidence alignment.

Positioning it in this landscape, a calibrated score around 5.5 (borderline but with clear value) seems appropriate.

**Final recommendation:** Reject, primarily due to overstated theoretical and conceptual claims and the lack of clear isolation of the “text-guided” mechanism’s contribution, despite otherwise solid empirical results.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>