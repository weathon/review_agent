Now I have enough information to write the final review. Let me synthesize everything.

## Summary

TNT introduces a two-stage training paradigm for deep memory modules (e.g., Titans, TTT) that decouples training efficiency from inference performance. Stage 1 uses a hierarchical memory architecture—global memory for long-range context and parallel local memories with periodic state resets that break sequential dependencies to enable context parallelism for non-linear recurrences. Stage 2 fine-tunes local memories at smaller chunk sizes. The method achieves a genuine ~7.7× training speedup at matched chunk sizes over Titans while improving average perplexity from 25.07 to 23.13.

## Strengths

- **Periodic reset mechanism for context parallelism is a genuine and practical innovation** (Eq. 6, Section 4.1.1): Breaking sequential dependencies in non-linear RNNs by resetting local memory to a learned $W_{\text{init}}$ at shard boundaries is a clean, implementable solution to a long-standing problem. The 7.68× speedup at matched chunk size (Table 1, TNT CL={8} vs. Titans C=8) validates this convincingly.

- **Quality improvement over deep memory baselines is substantial and simultaneous with speedup**: Table 2 shows TNT's best Stage 1 avg perplexity (23.13) markedly outperforms the best Titans (25.07) and vanilla Transformer (23.58), breaking the typical speed-accuracy tradeoff. This is a key result.

- **Hierarchical memory design is well-motivated and empirically validated**: The ablation (Table 3) shows catastrophic degradation without global memory (PPL jumps from 21.04 to 25.60), confirming the global module is essential for compensating information lost at local reset boundaries. The Q-K projection ablation also shows a 0.97 PPL increase when removed, validating the compression-retrieval mismatch hypothesis.

- **Clear problem decomposition matching solutions**: Section 3 identifies three distinct challenges (efficiency, compression-retrieval mismatch, chunksize mismatch) each addressed by a specific component (hierarchical memory + resets, Q-K projection, Stage 2 fine-tuning), making the design rationale transparent.

- **Linear runtime scaling with sequence length validated empirically**: Figure 4 demonstrates TNT scales linearly while Titans scales worse, and at 32K context TNT (CL=128) is 1.3× faster than FlashAttention—a notable practical result for long-context scenarios.

## Weaknesses

### Fatal
None.

### Major

- **The headline "up to 17×" speedup (Abstract, Table 1) conflates architectural contribution with chunk-size effects**: The 17.37× figure compares TNT CL={64} against Titans C=8—different chunk sizes with different performance profiles. The fair same-chunksize comparison (TNT CL={8} vs. Titans C=8) yields 7.68×, less than half the headline. While the paper does state the 7.68× figure in Section 5.2 ("using an identical local memory chunksize of 8, TNT is already 7.7× faster"), the abstract, introduction, and conclusion all lead with 17×. Part of the 17× comes from simply using a larger chunk size—something any baseline could also do at the cost of accuracy. The paper's framing implies TNT's architectural innovations deliver 17×, when the true architectural contribution is ~7.7×. This matters because it misrepresents the magnitude of the method's contribution.

- **Ablations do not control for parameter count when adding local memory modules**: The paper labels all models "150M parameters" but never specifies how parameters are allocated across components. Table 3's ablation shows incremental PPL improvement from 21.04 (1 local module) to 20.15 (4 local modules), but it is unclear whether this improvement comes from the hierarchical multi-resolution design or simply from having more memory module parameters. The paper must either (a) demonstrate parameter-controlled comparisons where total parameter count is held fixed as modules are added, or (b) explicitly report parameter counts per component. Without this, the quality improvement from adding modules is uninterpretable—it could simply reflect 4× the memory module parameters.

### Minor

- **Stage 2 fine-tuning provides marginal improvement at the full model level**: The best Stage 2 result (23.09 avg PPL) improves over the best Stage 1 result (23.13) by only 0.04 PPL—less than 0.2% relative improvement requiring 5% additional compute. This is within typical run-to-run variance at this scale (no variance is reported). At the ablation level (1 local module), Stage 2 improves from 21.04 to 20.86 (0.18 PPL), which is more meaningful. The two-stage paradigm is a core claimed contribution but its empirical contribution at full model scale is minimal, suggesting it may not be worth the additional complexity. That said, Stage 2 is cheap (5% compute) and provides a consistent if small improvement.

- **Quality evaluation only at 16K context length despite long-sequence motivation**: The paper emphasizes long-sequence modeling and demonstrates speed advantages up to 32K, but quality (Table 2) is evaluated only at 16K context. The hierarchical design's information tradeoffs—local resets discarding context, global memory compensating—may worsen at longer contexts. Evaluation at 32K or longer would strengthen the claims.

- **Shard length $S_L$ is a key hyperparameter that is never ablated**: $S_L$ controls how much context local memories see before reset, directly affecting the parallelism-quality tradeoff. The paper uses $S_L$=2048 for speed benchmarks and $S_L$=4096 for quality evaluation without justifying these choices or testing sensitivity.

### Trivial
None.

## Nice-to-Haves

- Analysis of information loss at reset boundaries: A plot of perplexity as a function of position relative to shard boundaries would reveal whether the global memory adequately compensates for local reset information loss.
- Scale evaluation beyond 150M/10B tokens: Even a 500M experiment would increase confidence that TNT's advantages persist at larger scales.
- Parameter breakdown per component to settle the parameter-scaling question definitively.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Q-K Projection equation appears to omit the query vector (Harsh Critic Issue 4)**: The parsed equation (Eq. 7) renders the local memory retrieval as $f(W_t, \sum k_\tau k_\tau / \|k_\tau\|)$ without $q_t$. However, the text in Section 4.1.2 clearly states: "projecting the query $q_t$ onto the subspace spanned by previously observed keys" and describes the projection matrix $P = \sum k_\tau k_\tau^T / \|k_\tau\|^2$ maintained as a running sum. The text makes clear $q_t$ is involved in the projection. This is almost certainly a parser artifact rather than a fundamental error in the paper—removed as a weakness.

- **Challenge 1 is "more of an engineering obstacle than a research contribution" (Harsh Critic Section Notes)**: This is a subjective judgment. The paper addresses it with the hierarchical memory architecture and periodic resets, which constitute a genuine research contribution. Removed.

- **Missing related works (Harsh Critic, Strength Finder)**: Per instructions, I do not flag missing related works.

- **No variance reported (Harsh Critic Section Notes)**: At 150M/10B tokens, single-run evaluation is standard practice in this field. Moved to trivial/nice-to-have.

- **Reproducibility concerns about undisclosed hyperparameters**: Per instructions, removed as nitpick about reproducibility.

- **Request for Transformer comparison with FlashAttention for speed**: The paper already provides this comparison in Table 1 and Figure 4. The Transformer with FlashAttention is faster than TNT, which the paper acknowledges: "our implementation does not yet outperform highly optimized baselines like the Gated Transformer with FlashAttention." Removed as factually addressed.

- **Strength Finder's claim that multi-resolution improvement "captures genuinely complementary multi-scale information rather than merely adding parameters"**: This conflicts with the verified Major weakness about uncontrolled parameter scaling. Removed from strengths since the parameter scaling concern undermines the certainty of this claim.

## Novel Insights

The paper's most interesting tension is that its strongest contribution (periodic resets for context parallelism) and its weakest empirical validation (Stage 2 fine-tuning) are both framed as equally important. The reset mechanism alone, at matched chunk sizes, delivers a clear ~7.7× speedup with quality improvement—a result that stands on its own. The two-stage paradigm and multi-resolution local memories, while conceptually sound, add complexity without proportional empirical payoff. The paper would be stronger if it led with the reset-based parallelism result rather than the inflated 17× figure, and honestly characterized Stage 2 as a marginal refinement rather than a co-equal contribution.

## Suggestions

- Report the 7.68× matched-chunksize figure prominently in the abstract alongside (or instead of) the 17× figure, and clearly separate the architectural speedup from the chunk-size speedup in a dedicated analysis.
- Add a parameter budget table showing how parameters are allocated across global memory, each local module, and the rest of the model, and run at least one ablation where total parameters are held constant while varying the number of local modules.
- Evaluate quality at 32K context to validate the long-sequence claims.
- Ablate $S_L$ to characterize the parallelism-quality tradeoff curve.

## Score and Decision

**Calibration anchors:**
- ParaRNN (6.5, Oral): Parallel training of nonlinear RNNs via Newton's iterations, 7B scale, custom CUDA kernel. TNT is weaker due to smaller scale (150M vs 7B), overclaimed headline speedup, and no custom kernel.
- In-Place TTT (7.33, Oral): Test-time training framework, 4B model, 128K context, principled objective. TNT is weaker due to smaller scale, limited context evaluation, and overclaiming.
- Fast-dLLM (7.0, Poster): 27.6× speedup with metric inflation flagged by reviewers. TNT has similar inflation issue (17× vs 7.68×) but less severe.
- TTT Done Right (5.60, Poster): Large-chunk TTT, multi-modal evaluation. TNT is comparable—both address TTT efficiency with chunk-size tradeoffs, TNT has clearer mechanism but more overclaiming.
- ATLAS (5.33, Reject): Memory modules for long context, same research group. TNT is stronger due to concrete practical speedup contribution.
- Factorization Memory (2.0, Reject): RNN with sparse memory, insufficient evaluation. TNT is much stronger with genuine innovation and strong results.

TNT's genuine contribution (periodic reset for ~7.7× speedup at matched chunk sizes with quality improvement) places it above rejected papers, but the overclaimed 17× headline, uncontrolled parameter scaling in ablations, and marginal Stage 2 contribution keep it below the strong accepts. Comparable to TTT Done Right but with more overclaiming; stronger than ATLAS due to practical speedup.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>