## Summary

The paper introduces "echo embeddings," a simple method for extracting text embeddings from autoregressive language models without modifying the architecture to use bidirectional attention. By repeating the input and extracting embeddings from the second occurrence (which can attend to all tokens in the first occurrence via causal attention), the method captures bidirectional context at inference time. Echo embeddings yield strong zero-shot results (~5% improvement over classical embeddings on MTEB, nearly matching LLM2Vec-unsupervised) and modestly outperform baselines in the fine-tuned setting, even under compute-matched conditions.

## Strengths

1. **Simple, elegant, and practically useful insight.** The core idea—repeat the input and pool from the second occurrence—is easy to implement on any causal LM without architectural changes or training, which is valuable for practitioners seeking zero-shot embeddings.

2. **Strong zero-shot results with comprehensive evaluation.** Echo embeddings improve over classical mean-pooled embeddings by ~6 points on MTEB average (48.64 vs 42.38) and over PromptEOL by ~5 points (43.69), nearly matching LLM2Vec-unsupervised (49.43) which requires additional MLM fine-tuning (Table 1). Evaluation covers the full 56-dataset MTEB, multiple backbone models (Mistral-7B, LLaMA-2-7B, S-LLaMA-1.3B), both zero-shot and fine-tuned regimes, and ablations on pooling, prompts, and compute budgets.

3. **Clean synthetic motivation.** The S1/S2 construction effectively isolates the failure mode of causal attention: mean-pooling exaggerates early-token similarity while last-pooling misses early tokens. Figure 2 clearly shows echo embeddings resolve both failure modes. The follow-up in Figure 2C showing that even the A-portion of echo embeddings can discriminate s+ from s− (when discriminative info is only in B) is particularly compelling evidence for bidirectional information flow.

4. **Compute-matched analysis strengthens the practical case.** The authors consistently report compute-matched variants (halving input length, halving training steps). That compute-matched echo still outperforms classical embeddings in most settings (49.02 vs 42.38 in Table 1; 64.66 vs 64.23 for Classical+Bi in Table 5) makes the result more convincing.

5. **Prompt robustness.** Figure 3 demonstrates that echo embeddings have low variance across prompt choices and that all echo prompts outperform all classical prompts, reducing concerns about prompt engineering being the primary driver of gains.

## Weaknesses

### Fatal
None.

### Major

- **Missing ablation isolating repetition from prompt/task framing effects.** The paper's central claim is that repetition enables bidirectional information flow, but the echo condition always pairs repetition with an autoencoding-style prompt ("Rewrite the following paragraph: S. The rewritten paragraph: S'") while the classical baseline uses a different prompt ("Write a paragraph: S'"). No experiment applies the same "rewrite" prompt without repetition, or repeats the input without an autoencoding instruction. This makes it impossible to attribute gains cleanly to the repetition mechanism vs. the prompt framing. While Figure 3 shows low prompt sensitivity, all tested echo prompts still involve both repetition and autoencoding instructions simultaneously. This is critical because the paper's mechanistic argument—that causal masking is the dominant failure mode and echo embeddings functionally restore bidirectionality—rests on this attribution. (Section 3.2, 4.2)

- **Fine-tuned improvements are marginal, undermining the claim that echo embeddings "circumvent the need for bidirectional attention."** In Table 5, echo embeddings (64.68) outperform classical+bidirectional (64.23) by only 0.45 MTEB points, and even compute-matched echo (64.66) only improves by 0.43 points over classical+bi. Similar small margins appear in Table 6 (62.01 vs 61.26 for S-LLaMA). Without error bars or statistical significance tests, these differences could be within noise. The paper's framing—"paving the way towards a unified architecture for all NLP tasks" and "circumvents the need for bidirectional attention"—is overstated given that the practical advantage in the most relevant (fine-tuned) setting is this marginal. (Abstract, Conclusion, Table 5)

- **Compute-matching via input truncation confounds information loss with compute savings.** Halving the input length to match compute (the primary strategy) removes actual semantic content, making the comparison between "echo on shorter input" and "classical on full input" not truly apples-to-apples in terms of information processed. The paper acknowledges echo underperforms at very small budgets (Section 4.7: "for very small computational budgets (fewer than 64 tokens), we find that Mistral-7B classical embeddings with bidirectional outperform echo embeddings"), but Figure captions still claim echo "outperforms Classical+Bi embeddings across all budgets" which contradicts the inline table data (Figure 5: 64 tokens shows Echo 70.5 vs Classical+Bi 71.2). (Section 4.6–4.7, Figures 4–5)

### Minor

- **Figure captions overclaim relative to the data.** As noted above, Figure 4 states "Echo embeddings outperform Classical+Bi embeddings across all budgets" but at 64 tokens they are tied (56.5 both), and Figure 5 shows echo underperforming Classical+Bi at 64 tokens (70.5 vs 71.2). The body text partially corrects this but the captions remain misleading. (Figures 4–5)

- **No per-task or per-category analysis of where echo embeddings fail.** The paper only reports category averages. In Table 1, echo underperforms PromptEOL on Classification (72.01 vs 73.84) and Reranking (47.56 vs 48.44) by non-trivial margins. Understanding which task types benefit and which don't would strengthen the paper and help practitioners. (Tables 1, 5)

- **The synthetic dataset lacks characterization.** The S1/S2 construction uses GPT-4-generated strings, but no quantitative characterization (length distributions, lexical overlap statistics, dataset size) is provided in the main text, making it hard to assess how realistic or representative the failure mode is. (Section 3.1)

- **The mechanistic claim about early tokens capturing bidirectional information is plausible but under-probed.** Figure 2C shows that cosine similarities using only A-portion embeddings can discriminate s+ from s−, but this does not demonstrate that individual token states encode balanced bidirectional information the way true bidirectional encoders do. More direct probing (attention visualization, representational similarity analysis) would strengthen this claim. (Section 3.2, Figure 2C)

- **The switch from mean pooling (zero-shot) to last-token pooling (fine-tuned) disconnects from the theoretical motivation.** The paper's key argument is about mean pooling over bidirectional representations, but the fine-tuned results use last-token pooling with a trainable EOS token. The theoretical framework is not updated to explain why echo embeddings also help in this setting. (Section 4.4–4.5)

### Trivial
- Minor inconsistency: the paper uses "MTEB" and "MTBEMini" spelling variants in different places (e.g., Figure 5 caption says "MTBEMini").

## Nice-to-Haves

- Ablation experiments isolating repetition from prompt effects (same prompt with and without repetition; repetition with neutral prompt).
- Analysis of echo embeddings on longer documents where the 2x sequence length cost is most significant, and where truncation-based compute matching is least feasible.
- Evaluation with more than two repetitions to test whether additional repetitions yield further gains (which would support the bidirectional information flow hypothesis) or plateau (suggesting a different mechanism).
- Probing experiments or attention visualizations on real data confirming that early second-occurrence tokens attend to later first-occurrence tokens.
- Comparison with stronger supervised baselines (E5-mistral, NV-Embed, etc.) to establish absolute performance positioning.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Missing comparison to stronger recent baselines like NV-Embed, BGE-large, or E5-mistral"** (from Spark/human finder): The paper compares to the most directly comparable methods (LLM2Vec, GritLM) that use the same or similar backbone models with bidirectional attention, which is the relevant baseline for their claim. Comparing to entirely different architectures (encoder-only models) would not directly address the paper's question about whether architectural modification is needed within the same model class.

- **"The method lacks novelty—it's just repetition"** (from human finder): While simple, the insight that repetition under causal attention functionally provides bidirectional context for embeddings, and the systematic validation that this works nearly as well as explicit bidirectional conversion, represents a genuine contribution. Simplicity can be a strength when supported by thorough validation.

- **"Reproducibility concerns about undisclosed hyperparameters or code availability"** (implied by Spark): The paper provides training details (LoRA r=16, α=16, τ=1/50, lr=8e-4, batch size 2048, GradCache) and prompt templates. This is standard practice for the field.

- **"No theoretical grounding or formal analysis"** (from Neutral Reviewer): Theoretical analysis of why causal repetition approximates bidirectional representations is nice-to-have but not standard for empirical embedding papers. The field does not require formal proofs for method contributions.

- **"The S1/S2 dataset is overly simplistic"** (from Harsh Critic): The toy dataset is explicitly designed as a motivating example to illustrate a specific failure mode, which it does effectively. The paper does not claim this synthetic dataset alone proves the real-world mechanism—it supplements it with MTEB evaluation.

- **"Bidirectional baselines may be poorly tuned"** (from Harsh Critic): The paper uses the same fine-tuning methodology for classical+bidirectional as for classical+causal, which is a reasonable apples-to-apples comparison. The bidirectional models are already disadvantaged (they require architectural modification), so the comparison is if anything favorable to the paper's method if the baseline is suboptimal.

## Novel Insights

The paper makes a genuine and non-obvious observation: that the widely-held assumption that bidirectional attention is essential for high-quality embeddings from language models can be largely circumvented by a simple inference-time repetition trick. This is notable because it suggests that much of the representational power lost by causal masking can be recovered without any architectural changes, and the gains appear primarily in the zero-shot setting where no fine-tuning is available. However, the fine-tuned results reveal that the advantage shrinks dramatically, suggesting that while repetition effectively addresses the zero-shot bidirectional information deficit, supervised training can largely close the gap through other means. This raises an important nuance: echo embeddings' main value proposition is as a zero-shot technique, not as a replacement for architectural innovations in the fine-tuned regime.

## Suggestions

- Add a control experiment: apply the same autoencoding prompt but without input repetition, and apply input repetition with a neutral (non-autoencoding) prompt. This would cleanly isolate the contribution of repetition vs. prompt framing and substantially strengthen or revise the mechanistic claim.
- Tone down claims to match the evidence: replace "circumvents the need for bidirectional attention" and "paving the way towards a unified architecture" with more measured framing like "provides a simple and effective zero-shot alternative to bidirectional attention conversion" and "shows that bidirectional attention is not strictly necessary for competitive zero-shot embeddings."
- Report per-category and per-dataset results showing where echo embeddings lag behind baselines (e.g., Classification and Reranking in Table 1), and discuss potential reasons.
- Correct the misleading figure captions in Figures 4–5 to accurately reflect that echo embeddings do not outperform Classical+Bi at all token budgets.

## Evaluation

**Originality:** The core insight—repeating input to leverage causal attention for bidirectional context—is simple but non-obvious and has not been proposed in this form. However, the methodological contribution beyond "repeat and pool" is limited; the empirical validation does the heavy lifting.

**Importance of research question:** Important. Whether architectural changes are needed for embeddings from autoregressive LMs is a timely and practically relevant question.

**Claim support:** The zero-shot claims are well-supported. The mechanistic claim about bidirectional information flow is plausible but under-probed (missing ablation). The fine-tuned claims are overstated relative to the marginal improvements observed. The "circumvents bidirectional attention" framing is stronger than the evidence warrants.

**Experimental soundness:** Broad evaluation across models, settings, and ablations. Compute-matching is imperfect but transparently reported. Missing the key ablation separating repetition from prompt effects.

**Clarity:** Generally well-written; the toy example is effective. Some overclaiming in captions and framing.

**Value to community:** High practical value for zero-shot embedding extraction. Moderate conceptual value—the paper provides evidence that bidirectional attention is less essential than believed, but the mechanism could be partially explained by prompt effects rather than pure bidirectional information flow.

## Score and Decision

For calibration, I considered:
- **GritLM** (similar topic, unified gen/rep model): Accepted with scores 6-8. More complex method, stronger empirical results, also had compute-matching concerns.
- **Pooling and Attention** (LLM embedding methodology): Rejected with scores 3-5. Limited novelty, incremental contribution.
- **Bitune** (bidirectional attention for LLMs): Borderline/rejected with scores 5-10. Similar bidirectional attention topic, concerns about marginal improvements.
- **Contextual Document Embeddings** (embedding approach): Accepted with scores 6-8. Simple but effective method, thorough evaluation.

This paper is stronger than the rejected papers (it has a genuine insight and thorough evaluation) but weaker than the top accepted papers (marginal fine-tuned improvements, missing key ablation, overclaiming). Its main value is as a zero-shot technique, where the improvements are substantial. The missing ablation separating repetition from prompt effects is a real gap that weakens the mechanistic story. Given the calibration, the paper falls in a borderline-accept range — the zero-shot contribution is valuable, but the overclaiming and missing ablation prevent a higher score.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>