Now let me look at the human reviews for calibration.Now let me search for additional calibration papers.Now I have all the information I need. Let me write the consolidated final review.

---

## Summary

This paper introduces **echo embeddings**, a method for extracting high-quality text embeddings from autoregressive (causal) LMs without modifying the architecture to use bidirectional attention. By repeating the input text with an autoencoder-style prompt and pooling representations from the second occurrence—which has full causal access to all first-occurrence tokens—the method achieves >5% absolute improvement over classical LM embeddings in zero-shot settings and approximately matches LLM2Vec-unsupervised (which requires additional MLM fine-tuning). In supervised fine-tuning, echo embeddings match or modestly outperform bidirectionally-converted LMs under compute-matched conditions. The paper challenges the prevailing assumption that bidirectional attention is necessary for strong text embeddings.

---

## Strengths

- **Simple and elegant core idea with genuine practical utility.** Repeating the input so that second-occurrence tokens attend causally to all first-occurrence tokens is a neat mechanism requiring no architectural change or additional training. Practitioners can immediately adopt it with any instruction-tuned causal LM.

- **Strong, consistent zero-shot improvements.** Echo embeddings achieve 48.64 average MTEB (full compute) and 49.02 (compute-matched) vs. 42.38 for classical LM embeddings and 43.69 for PromptEOL. Crucially, the compute-matched variant *nearly matches* LLM2Vec-unsupervised (49.43), which requires additional MLM fine-tuning—a significant practical advantage.

- **Well-motivated synthetic analysis.** The S1/S2 construction in Section 3.1 clearly illustrates the failure modes of mean and last-token pooling under causal masking: mean pooling dilutes late discriminatory information; last-token pooling misses early discriminatory information. Figure 2 shows echo embeddings handling both structures simultaneously at ~0.95+ accuracy. This is a didactically strong motivation.

- **Breadth of ablations.** The paper ablates pooling strategy (Table 4), backbone model and scale (Table 2), bidirectional attention casting (Table 3), prompt sensitivity (Figure 3), and compute budget curves (Figures 4–5). The finding that echo embeddings are robustly insensitive to prompt choice, while PromptEOL has high variance (Figure 3), is particularly valuable for practitioners.

- **Honest accounting of compute cost.** The paper explicitly acknowledges the 2× compute overhead, explains the matching strategy (halving input length at inference; halving training steps during fine-tuning), and provides detailed budget curves rather than hiding the trade-off.

- **Table 3 provides a useful nuance.** The result that naively casting to bidirectional attention often *hurts* (LLaMA-2-7B: 47.27 → 43.03; S-LLaMA-1.3B: 40.14 → 35.77) unless the backbone was pre-trained with non-standard methodology (Mistral-7B) is an important finding for the community.

---

## Weaknesses

### Fatal
*(None. The core zero-shot result is solid and the contribution stands.)*

### Major

1. **Critical ablation missing: repetition without the task-framing prompt.**
The paper's mechanistic claim is that input repetition enables causal tokens to attend bidirectionally. However, the zero-shot setup bundles together two distinct components: (a) the autoencoder-style instruction ("Rewrite the following paragraph: S. The rewritten paragraph: S'") and (b) the duplicated input that exposes second-occurrence tokens to the full first occurrence. The paper does not isolate these. A control—feeding the input twice with no reconstruction instruction ("S; S") versus the full echo prompt—would clarify whether the benefit comes from the structural repetition itself or from the task-conditioning. Without this, the paper establishes that the echo *recipe* works well, but not definitively that repetition (as opposed to a particular kind of autoencoder prompting) is the active mechanism. This matters because the paper's conceptual framing foregrounds repetition as the key insight.

2. **Fine-tuning gains are marginal without statistical support, yet are presented as validating strong architectural claims.**
In Table 5, the full-compute echo (64.68) beats classical causal (63.98, +0.70) and classical bidirectional (64.23, +0.45). Compute-matched echo (64.66) beats classical bidirectional by 0.43 points. In Table 7, echo compute-matched (64.66) sits between GritLM-public (64.70) and LLM2Vec-supervised (64.80)—essentially a three-way tie within ~0.2 points. None of the fine-tuned comparisons report variance across seeds or statistical significance. Given that LoRA fine-tuning on mixed datasets is sensitive to seed and data order, differences of this magnitude are well within typical noise. The abstract's phrase "matching or outperforming bidirectionally-converted LMs … even with an identical compute budget" is technically supported in the point estimates but overclaims relative to the evidential strength.

3. **The synthetic example is not verified to generalize to real benchmark performance.**
The S1/S2 construction is compelling as illustration, but the paper only gestures to "Appendix C" to establish that this failure mode appears in real MTEB data. The appendix is not provided in the submission and cannot be verified. More substantively, the toy example is hand-designed to make classical embeddings fail; whether the gains on real MTEB tasks are actually driven by this mechanism or by other factors (e.g., additional processing depth from seeing input twice, or the autoencoder framing inducing more useful hidden states) is not established. A brief analysis in the main paper—e.g., correlating per-task improvement with task characteristics that match the S1/S2 failure modes—would meaningfully strengthen the causal story.

### Minor

4. **Exclusively instruction-tuned models.** All experiments use instruction-tuned variants (Mistral-7B-Instruct, LLaMA-2-7B-Instruct, S-LLaMA-1.3B-Instruct). Whether the echo prompt is understood by base (non-instruction-tuned) models is untested. If instruction-following capability is a prerequisite for the method to work, this limits applicability to the instruction-tuned subset of causal LMs and should be explicitly scoped.

5. **Pooling strategy switch between settings lacks explanation in the main body.** Mean pooling is strongly preferred in zero-shot (echo: 48.64 vs. 31.55 last-token), but last-token pooling is used in fine-tuning. The explanation ("consistent with prior work, last-token with trainable EOS token slightly outperforms mean pooling") is deferred to Appendix F and given only briefly in Section 4.5. Because this choice affects all fine-tuned results, the main paper should discuss it with at least a brief quantitative comparison.

6. **Compute-matching via input halving is not analyzed for short-text tasks.** Halving input length disproportionately affects tasks with naturally short inputs. The paper does not characterize how many MTEB tasks fall below a "safe" truncation threshold or whether any task categories are systematically harmed by the input halving. The compute-matched echo results are actually *slightly better* than full echo in some settings (Table 1: 49.02 vs. 48.64), which deserves more explanation rather than the current brief comment that "it is unclear to what degree we would observe this improvement in other settings."

### Trivial

7. **Conclusion overstates the scope of the claim.** The conclusion says "the common assumption that bidirectional architectures are crucial for high quality embedding models is false." Given that (a) fine-tuning gains are marginal, (b) the mechanism is not fully isolated, and (c) Mistral-7B with naive bidirectional casting already approaches echo performance (Table 3), the more defensible claim is that *repetition is a viable alternative to architectural modification*, not that bidirectionality is universally unnecessary.

---

## Nice-to-Haves

- **Attention pattern analysis on real data.** Probing or attention visualization experiments showing that second-occurrence tokens actually attend meaningfully to first-occurrence tokens across real MTEB inputs (not just the synthetic S1/S2 dataset) would substantially strengthen the mechanistic narrative.

- **Base model evaluation.** Testing on at least one base (non-instruction-tuned) model would clarify the dependency on instruction-following capability and either expand or appropriately scope the claims.

- **Long-document retrieval characterization.** An explicit breakdown of performance (and compute cost) on MTEB retrieval tasks with long documents would help practitioners understand when the 2× compute overhead is most problematic.

- **Variance/significance for fine-tuned results.** Given the small margins in supervised settings, even a simple two-seed comparison would clarify which differences are meaningful.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic, Issue 1 — "Compute-matched comparison is structurally unfair"]: WEAKENED, partially removed.** The paper is fully transparent about its compute-matching strategy (input halving, step halving). The characterization of this as a "structural" failure is excessive; the paper is honest about trade-offs and never claims the comparison is architecture-equivalent. The residual concern (framing) is captured under Weakness 2 (fine-tuning margins) and Weakness 6 (input halving effects).

- **[Harsh Critic — "Figure 4 overstated: echo and Classical+Bi are equal at all budgets"]: REMOVED.** The extracted numbers in the figure table appear to reflect a PDF parsing artifact or rounding of visual data. The paper text (line 294–299) explicitly describes a crossover: "when the budget is greater than approximately 64 tokens, echo embeddings outperforms Mistral-7B classical embeddings with bidirectional attention." This is internally consistent; the exact values extracted should not be treated as ground truth.

- **[Harsh Critic — Consistency of classical baseline prompt ("Write a paragraph: S'") being unfair]: REMOVED.** The "classical" prompt is a standard inference-time task-specification prompt similar to what prior work uses. The paper's claim is not that echo has better prompt engineering but that it has a structural advantage; what matters is that the classical baseline uses a reasonable prompt, which it does.

- **[Harsh Critic — Questioning LLM2Vec's additional fine-tuning stages making comparison unfair]: REMOVED.** The asymmetry here favors the *baseline* (LLM2Vec requires more training), not the authors. This is an intentionally stronger comparison that, if anything, understates echo's advantage. By the hard rules, such asymmetries should be removed.

---

## Novel Insights

The most genuinely novel aspect is not just the method itself but the finding in Table 3: for most causal LMs, naively casting to bidirectional attention at inference time *hurts* performance—sometimes severely. This suggests that the widely-held belief that bidirectionality is simply better for embeddings is model- and pre-training-dependent, and that the actual obstacle in causal LMs is not the attention mechanism per se but the lack of context access for early tokens—an obstacle that repetition can resolve without touching the architecture. The compute-budget curves (Figures 4–5) further show that the crossover point (~64–128 tokens) is well below the standard MTEB budget of 512 tokens, meaning that for virtually all practical evaluation settings, echo embeddings can be used without net compute penalty. These together make a coherent argument that the architecture modification step in LLM2Vec/GritLM may be more disruptive than beneficial, and that a simple prompt-engineering approach recovers most of the benefit—a practically important finding.

---

## Suggestions

1. **Add the repetition-vs.-prompt ablation in the main paper.** Compare "x; x" (plain repetition) with "Rewrite: x; Rewritten: x" (full echo) and with "Rewrite: x; Rewritten:" (instruction only, no second input) as a 3-way comparison. This single ablation would significantly strengthen the mechanistic claim.

2. **Bring Appendix C's real-data failure-mode analysis into the main paper.** Even a single figure showing which MTEB task categories show the S2-type failure in classical embeddings, and how echo repairs them, would ground the motivation empirically.

3. **Report at least two random seed runs for Table 5/6/7.** Given the sub-1-point margins, this is essential for honest scientific communication.

4. **Discuss instruction-tuned vs. base model dependency explicitly.** Even if not testing base models, acknowledging this as a known limitation in the main text is important for practitioners evaluating where to apply the method.

---

## Score and Decision

**Calibration:**

- *Making Text Embedders Few-Shot Learners* (wfLuiDjQ0u): Accepted poster, scores 8/6/6/8 (~7 avg). A paper of similar scope: simple training recipe for LLM-based embeddings, strong MTEB results, thorough ablations. That paper also lacked some mechanistic depth but was accepted on the strength of its empirical contribution and simplicity.

- *Scaling Sentence Embeddings* (V0CUOBWUHa): Rejected, scores 6/5/6 (~5.7 avg). Weaker empirical results, unclear baselines, mixed task performance. Echo embeddings is clearly stronger.

- *Pooling and Attention* (CWAvMSNUqT): Rejected, scores 5/3/5/5 (~4.5 avg). Narrower contribution, inconsistent results, no clear mechanism. Echo is considerably better.

- *Bitune* (NzEIjnIIzv): Rejected, mixed scores 8/5/10/6. Similar problem space (bidirectional information in decoder LLMs) but focused on generation tasks, not embeddings. Comparable in concept; echo is more practically validated on an established benchmark.

**Positioning:** Echo embeddings sits comfortably in the tier of "Making Text Embedders Few-Shot Learners"—both papers have simple, elegant ideas, strong zero-shot results, broad ablations, and some mechanistic gaps. The zero-shot contribution of echo embeddings is arguably the stronger practical result (matching unsupervised fine-tuning methods without any fine-tuning). However, the missing mechanistic ablation (repetition vs. task prompt), the marginal fine-tuned gains without statistical support, and the conclusion-level overclaiming pull it slightly below that paper's ~7 average. I place it at **6.5**: a solid, above-threshold contribution with clear practical value, genuine originality, and one fixable but real gap in mechanistic rigor.

**Assessment along axes:**
- *Originality*: Good — the echo embedding idea is novel, well-motivated, and distinct from prior work.
- *Importance of research question*: High — unified generative/embedding models are a pressing practical need.
- *Claims well-supported*: Partially — zero-shot claims are well-supported; fine-tuning "outperform" framing is somewhat overclaimed.
- *Soundness of experiments*: Good for zero-shot; adequate but incomplete for fine-tuning (no variance).
- *Clarity*: Good — the paper is well-written and experiments are clearly described.
- *Value to community*: High — the method is immediately usable and the negative result about bidirectional casting (Table 3) is independently informative.

**Final Score: 6.5 / Accept**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>