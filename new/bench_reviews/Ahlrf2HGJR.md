Now I have all the information I need. Let me compose the final review.

## Summary

This paper introduces "echo embeddings," a simple method for extracting high-quality text embeddings from autoregressive language models without architecture changes or fine-tuning. By repeating the input and pooling representations from the second occurrence—which has access to all tokens via causal attention—echo embeddings circumvent the need for bidirectional attention. Zero-shot echo embeddings improve over classical LM embeddings by +6.26 on MTEB (56 datasets) and nearly match LLM2Vec-unsupervised, which requires additional MLM fine-tuning. The paper also evaluates fine-tuned echo embeddings, claiming they match or outperform bidirectionally-converted LMs.

## Strengths

- **Simple, novel, and well-motivated core idea.** Repeating input so the second occurrence can attend to the full context under causal attention is clever, requires no architecture changes or training, and is trivially applicable to any autoregressive LM. This is a genuine contribution to the embedding literature.

- **Strong zero-shot results.** Table 1 shows a +6.26 absolute improvement over classical embeddings on MTEB (48.64 vs. 42.38) and even the compute-matched variant (49.02) outperforms classical embeddings. The improvement holds across 10 of 12 subcategories. This is a substantial margin that robustly supports the core claim.

- **Mechanistic evidence for bidirectional information flow.** Figure 2C directly demonstrates that embeddings of only the *A* portion in Structure 2 inputs can discriminate s⁺ from s⁻ even though discriminatory information resides solely in the *B* portion. This validates the claimed mechanism that echo embeddings encode bidirectional information under causal attention.

- **Prompt robustness.** Figure 3 shows echo embeddings have low variance across different prompt wordings while all echo prompts outperform all classical prompts. This practical robustness differentiates echo from PromptEOL, which is highly prompt-sensitive.

- **Consistency across models and scales.** Tables 2 and 6 show echo outperforms classical and PromptEOL on both S-LLaMA-1.3B and LLaMA-7B, and fine-tuning gains are consistent on S-LLaMA-1.3B (62.01 vs. 61.26).

- **Honest compute-matching methodology.** The paper consistently reports compute-matched variants alongside full-compute results (Tables 1, 5, 7; Figures 4, 5), transparently assessing the practical cost-benefit tradeoff. The compute analysis (Section 4.7) shows echo outperforms classical+bidirectional for budgets above 64 tokens (zero-shot) and 128 tokens (fine-tuned).

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed fine-tuning results without significance testing.** The abstract states echo embeddings are "matching or outperforming bidirectionally-converted LMs in an apples-to-apples comparison, even with an identical compute budget." However, Table 5 shows margins of only 0.45 points (64.68 vs. 64.23 for Classical+Bi) and 0.70 points (64.68 vs. 63.98 for Classical) on MTEB—sub-1-point differences on a 56-dataset average with no standard deviations, confidence intervals, or significance tests reported anywhere in the paper. Table 7 shows compute-matched echo (64.66) is actually *behind* LLM2Vec-supervised (64.80) and GritLM (64.70), not matching or outperforming them. The zero-shot results (6+ points) robustly support the core claim; the fine-tuning results do not establish superiority, and the abstract's language overclaims this dimension. The paper should either tone down the fine-tuning claims or provide the statistical rigor needed to support them.

- **Missing ablation isolating repetition from the reconstruction prompt.** Echo embeddings combine two components: (a) repeating the input in the sequence, and (b) prompting the model to "repeat/rewrite" the input. The paper ablates prompt wording (Figure 3) but never tests repetition *without* a reconstruction-oriented prompt (e.g., simply concatenating "S. S." with no task instruction). The paper itself states the prompt is needed "to encourage the second occurrence to actually 'encode' information about the first" (Section 3.2). Without this ablation, we cannot determine whether the architectural repetition or the task instruction drives the improvement, leaving the mechanism under-specified. If mere repetition without a task prompt produces similar gains, the paper's framing of the prompt as essential is incorrect; if it produces no gains, the title's claim that "repetition improves embeddings" is misleading—it is the *prompted repetition* that matters.

### Minor

- **Compute-matching in fine-tuning confounds training duration with data exposure.** Section 4.4 states that compute-matched echo embeddings are "train[ed] for half as many steps" as classical embeddings. This means the echo model sees half the training data, introducing a confound: if the fine-tuning margin shrinks under compute-matching (it does—from 0.70 to 0.68 vs. Classical), we cannot tell whether this reflects compute-dependence of echo's advantage or simply undertraining. Matching total FLOPs by adjusting batch size rather than steps would isolate the effect more cleanly. The paper is transparent about this choice, which partially mitigates the concern.

- **Mistral-7B bidirectional attention nearly matches echo.** Table 3 shows Mistral-7B with bidirectional attention achieves 58.24 vs. echo at 59.78 on MTEB-MINI—a small gap that suggests for some models, the simpler bidirectional conversion is nearly as effective. The paper acknowledges this and attributes it to "non-standard pre-training methodology," but this limits the generality of the claim that echo is necessary for all autoregressive models.

### Trivial
None.

## Nice-to-Haves

- Analysis of embedding space geometry (isotropy, cosine similarity distributions) to connect empirical gains to the theoretical motivation about bidirectional information flow.
- Per-dataset scatter plot comparing echo vs. classical to reveal whether improvements are uniform or concentrated in specific task types (e.g., retrieval vs. STS).
- Failure mode analysis: on which MTEB subsets does echo *not* help or even hurt (e.g., Clustering: 33.80 vs. 34.14)?

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Asymmetric prompts in fine-tuning introduce a confound" (Harsh Critic).** The critic claims that echo queries getting "Query: S / Query again: S'" while classical queries get "Instruct: {instruction} / Query: S" introduces a confound. This is not a confound—it is the echo method itself. The "again" portion IS the echo. Comparing echo (with its full pipeline) vs. classical (with its full pipeline) is the intended comparison. Removed because the "asymmetry" is the method, not a design flaw.

- **"Comparison to a properly-trained bidirectional model at 7B scale" (Harsh Critic).** Demanding a direct comparison where the *only* difference is causal vs. bidirectional attention with identical training goes beyond the paper's stated scope and would require training a new bidirectional LM from scratch at 7B scale. The paper's comparison to LLM2Vec-supervised (which does use bidirectional attention at 7B scale with a different training pipeline) is adequate for the stated goals.

- **"Compute-matching halves input length, changing what text is being embedded" (Harsh Critic).** The paper is explicit about this tradeoff (Section 4.2: "we also consider compute-matched echo embeddings in which we halve the length of the input"). This is a design choice transparently reported, not a hidden flaw. The analysis in Figures 4 and 5 further explores different budget levels, giving the reader full information.

- **"Synthetic dataset is somewhat circular" (Harsh Critic).** The critic notes the toy dataset is constructed to exhibit exactly the failure mode causal attention creates. The paper uses it as illustration (Section 3.1), and the real-data MTEB results vindicate its relevance. The paper itself frames it as a motivating example, not as standalone proof.

- **"LLM2Vec is still ahead on the main metric" (Harsh Critic).** The paper is careful in its framing: it says echo "perform similarly" and "offers a simple alternative" (Section 4.3), while acknowledging LLM2Vec requires additional fine-tuning. The 0.79-point gap (48.64 vs. 49.43) for a zero-shot method vs. an unsupervised fine-tuning method is reasonably characterized as "similar."

- **Formatting/style nitpicks** (any minor presentation issues): Removed per hard rules.

## Novel Insights

The paper makes an underappreciated connection between the prompt design for echo embeddings and the concept of "inference-time compute" for representations. By showing that the task instruction at inference time can fundamentally change what information is encoded in representations—even without any weight updates—echo embeddings point to a broader research direction: understanding how inference-time interventions (not just scaling compute, but changing the task framing) can reshape the information geometry of LM representations. This is distinct from prior work on inference-time compute scaling and suggests a complementary axis of investigation.

## Suggestions

- Add a single ablation: test echo embeddings with a neutral prompt like "S. S." (no reconstruction instruction) to isolate the contribution of the task prompt vs. the repetition structure. This is the most impactful experiment the paper is missing.
- Soften the fine-tuning claims in the abstract from "matching or outperforming" to "competitive with" or add standard deviations / bootstrap confidence intervals for the fine-tuning MTEB scores to support the stronger claim.
- Report per-category breakdowns for fine-tuning results (as done for zero-shot in Table 1) to help readers understand where echo helps and where it does not.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| DoLa | `/home/wg25r/review_agent/human_reviews/Th6NyL07na.md` | 7.25 | Simple inference-time trick without fine-tuning with strong results; echo embeddings is comparable in novelty but overclaims fine-tuning |
| StreamingLLM | `/home/wg25r/review_agent/human_reviews/NG7sS51zVF.md` | 7.50 | Simple method with strong analysis; echo embeddings is slightly weaker due to missing ablation and overclaim |
| MoE as Embedding | `/home/wg25r/review_agent/human_reviews/eFGQ97z5Cd.md` | 6.67 | Similar profile (trick for embeddings from LLMs); echo embeddings has stronger zero-shot results and better mechanistic analysis |
| NV-Embed | `/home/wg25r/review_agent/human_reviews/lgsyLSsDRe.md` | 7.50 | Comprehensive system-level contribution; echo embeddings is a simpler, more targeted contribution |
| NUDGE | `/home/wg25r/review_agent/human_reviews/MYw74B77KQ.md` | 6.0 | Non-parametric embedding fine-tuning; echo embeddings has a more novel and broadly applicable core idea |
| Late Chunking | `/home/wg25r/review_agent/human_reviews/74QmBTV0Zf.md` | 4.75 | Simple embedding method with weaker empirical support; echo embeddings clearly surpasses this |
| CPLLM | `/home/wg25r/review_agent/human_reviews/fnBYPL5Ged.md` | 2.0 | Overclaimed marginal improvements without significance testing; echo embeddings is far stronger with 6+ point zero-shot gains |
| GritLM | `/home/wg25r/review_agent/human_reviews/BC4lIvfSzv.md` | 7.00 | Unified generation+embedding model; echo embeddings is a simpler alternative that doesn't require bidirectional attention |

The paper's zero-shot contribution (simple, no-training, +6.26 MTEB, mechanistic validation) places it solidly in the range of accepted papers like DoLa (7.25) and MoE-Embedding (6.67). The overclaimed fine-tuning results and missing ablation keep it below the 7.5 tier of StreamingLLM/NV-Embed. The zero-shot results are genuinely strong and the method is novel—this is not a marginal-contribution paper like CPLLM or Late Chunking. I position this at 7.0: solidly above borderline, with clear contribution but notable weaknesses that should be addressed.

**Originality:** High. The idea of repeating input for bidirectional information under causal attention is genuinely novel and simple.

**Importance:** High. The question of whether bidirectional attention is truly necessary for embeddings is of broad interest, and the affirmative answer via a simple method has practical impact.

**Claims support:** Mixed. Zero-shot claims are well-supported (+6.26 on MTEB with mechanistic evidence). Fine-tuning claims are overclaimed (sub-1-point margins without significance testing).

**Experimental soundness:** Good for zero-shot, weak for fine-tuning (no variance estimates, missing ablation).

**Clarity:** Good. The paper is well-organized and the method is clearly described.

**Community value:** High. A simple, no-training method that substantially improves autoregressive LM embeddings will be widely adopted.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>