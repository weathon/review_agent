Now I have thoroughly read the paper and checked claims. Let me compile my final review.

## Summary

This paper introduces "subgoal imbalance" as a lens for understanding why autoregressive (AR) models struggle on planning/reasoning tasks, and proposes Multi-Granularity Diffusion Modeling (MGDM)—a discrete diffusion model with token-level difficulty-based reweighting—as a solution. The key insight is that diffusion training decomposes hard subgoals into multiple easier denoising objectives with varying context availability, unlike AR models which must predict hard tokens given only left context. MGDM achieves dramatic improvements on three structured tasks: 91.5% vs 45.8% on Countdown, 100% vs ~20.7% on Sudoku, and widening gaps on SAT with increasing difficulty.

## Strengths

- **Insightful subgoal imbalance framing (§3.1):** The synthetic planning task cleanly isolates the phenomenon where certain positions are predictively much harder given only left context. Figure 2 shows AR models require exponentially more data and fail at PD ≥ 2, while diffusion achieves 100% accuracy across all planning distances. This is a clear, well-controlled demonstration of a real limitation.

- **Strong empirical gains on structured reasoning tasks (§4, Tables 1–2, Figure 4):** The results are dramatic: a 6M diffusion model achieves 100% Sudoku accuracy where LLaMA-13B achieves only 32.9%. On Countdown 5, MGDM (85M) reaches 46.6% vs GPT-2 Scratch (85M) at 5.1%. These gaps are large enough to be clearly meaningful, not marginal improvements.

- **Mechanistic loss analysis (Figure 3):** The comparison of $-\log p_{\text{AR}}(x_n | x_{1:n-1})$ vs $-\log p_{\text{DM}}(x_n | x_{\neq n})$ vs $-\log p_{\text{DM}}(x_n | x_{1:n-1})$ for a hard subgoal (PD=3) provides concrete evidence that diffusion decomposes hard objectives into progressively easier ones across timesteps—a mechanism AR models cannot exploit.

- **MGDM token-level reweighting is effective (Eq. 8, Table 3):** The adaptive difficulty-based reweighting $v(x_{t,n}) = \alpha(1 - \exp(-u(\cdot)))^\beta$ improves TopK accuracy from 87.3% to 91.5% on Countdown 4, demonstrating it is a meaningful extension beyond vanilla discrete diffusion. Figure 5 shows it converges faster and to higher accuracy.

- **"Regretful Compromise" error analysis (§4.4, Figure 6b):** This is a novel diagnostic showing AR models accumulate planning errors in early equations that cascade into calculation errors later (48.9% calculation error rate in Eq3 vs near-zero in Eq1). This goes beyond reporting accuracy to explain *why* AR fails.

- **Decoding speed–accuracy trade-off (Figure 6a):** The T=1 result (75% vs AR's 45.8% on Countdown 4 at 10× speed) shows the advantage isn't purely from iterative refinement—there is genuine improvement from a single parallel pass.

## Weaknesses

### Fatal
None.

### Major

- **Claim about "sophisticated language understanding" is unsupported by the evaluation scope.** The abstract promises relevance to "sophisticated language understanding and problem-solving tasks," and the introduction frames the contribution as addressing "complex reasoning and long-term planning" and "sophisticated language understanding." However, all three evaluation tasks—Countdown, Sudoku, and SAT with 5–9 variables—are purely combinatorial constraint-satisfaction problems with finite, well-defined solution spaces. They require no semantic or world knowledge, no ambiguous interpretation, and no natural language understanding. These tasks are precisely the genre where bidirectional context and global constraint satisfaction are most advantageous, making it unclear whether the advantages generalize to open-ended natural language reasoning. The paper acknowledges this implicitly in the conclusion ("we aim to demonstrate the potential advantages"), but the abstract and introduction language substantially overclaims the breadth of the contribution. This mismatch between claim scope and evaluation scope is the paper's most significant weakness.

- **The sources of MGDM's advantage are not fully disentangled.** The performance gap between AR and diffusion could arise from three confounded factors: (a) bidirectional context (seeing future tokens), (b) iterative refinement (multiple denoising passes), and (c) the diffusion ELBO training objective. The paper includes a "teacherless" baseline (§3.1) that has bidirectional context but no iterative refinement, but this baseline "fails to adequately fit the training data"—which only shows this particular implementation doesn't work, not that bidirectional context alone is insufficient. A more informative ablation would be an AR model with bidirectional attention trained with an MLM-style objective, and a non-iterative diffusion model (T=1 with full bidirectional context). The loss analysis in Figure 3 partially addresses this (showing $-\log p_{\text{DM}}(x_n | x_{1:n-1})$ decreasing with timestep suggests the diffusion objective provides benefit beyond just context), but a direct architectural ablation in the experiments would be much more convincing. Without it, the reader cannot determine whether the gains come from the diffusion paradigm per se or from architectural properties that are well-understood (bidirectional context is known to help constraint satisfaction tasks).

### Minor

- **Proposition 1 is more of a conceptual statement than a formal result.** It states that subgoal difficulty can differ based on model parameterization—something that is intuitively obvious and doesn't provide quantitative bounds or novel characterization. Its value is as a conceptual framing rather than a theoretical contribution, but calling it a "Proposition" oversells its formality.

- **MGDM hyperparameter sensitivity is underexplored.** Table 3 shows α and β matter (the best configuration α=0.25, β=2 achieves 91.5% vs 87.3% without reweighting), but there is no analysis of how these are selected, whether they transfer across tasks, or how sensitive performance is to these choices.

- **The GPT-4 comparison (Table 2) is not apples-to-apples.** MGDM is a task-specific 85M model trained on 500K examples; GPT-4 is a large general model prompted with 5 examples. The paper does acknowledge the token cost advantage and positions this as showing "the modeling paradigm sometimes outweighs the sheer number of parameters," which is fair as motivation, but the comparison should not be read as showing MGDM is more capable than GPT-4 in any general sense.

### Trivial
None.

## Nice-to-Haves

- Evaluation on at least one natural language reasoning benchmark (e.g., GSM-8K, MATH, or multi-step QA) to test whether the advantages generalize beyond combinatorial puzzles.
- A bidirectional-attention AR ablation (e.g., MLM-trained model) on the same tasks to isolate the contribution of the diffusion objective from bidirectional context.
- Equal-compute comparisons with AR+beam-search or AR+verification to put the "without search techniques" claim on a more level playing field.

## Removed Points

These points are flagged by reviewers but are removed or weakened because they are either factually wrong, scope creep, or misunderstandings of the paper:

- **"SAT instances with 5-9 variables are trivially solvable by dedicated SAT solvers"** — This is scope creep; the paper is comparing neural modeling paradigms on these tasks, not proposing to compete with SAT solvers. The NP-completeness of SAT is cited for its theoretical significance as representing a broad class of constraint satisfaction problems, not for practical solving.

- **"Sudoku results are uninformative compared to a constraint-propagation solver"** — Same scope creep issue. The comparison is between AR and diffusion as neural approaches, not against classical solvers.

- **"The inference-time compute comparison is misleading"** — While partially valid, the T=1 result (75% vs 45.8%) uses a single forward pass on the full sequence and is faster than AR's N sequential forward passes. This is a genuine architectural advantage of parallel decoding, not merely a compute budget trick. The paper is transparent about the comparison conditions. The concern about excluding search from AR baselines is softened by the Stream-of-Search comparison in Table 1.

- **"Teacherless baseline is broken"** — The paper explicitly explains why teacherless training fails and notes it "can be conceptualized as a special case of diffusion without an iterative denoising process," which is an informative negative result showing that bidirectional context alone is insufficient without iterative refinement. The criticism that this "only tells us the implementation is broken" is partially addressed by the paper's analysis.

- **"Proposition 1 is trivially true"** — While not formally deep, the proposition serves as a conceptual scaffold for the subsequent analysis, and Figure 2 provides the empirical substantiation. The value is in the framework, not the formal statement.

## Novel Insights

The "Regretful Compromise" diagnosis (§4.4) is a genuinely novel analytical contribution: it decomposes AR errors into planning errors (early positions) that cascade into calculation errors (later positions), showing a 48.9% calculation error rate in the final equation. This suggests a concrete mechanism for how left-to-right decoding creates compounding failures in multi-step reasoning tasks—not merely that AR models perform worse, but that they fail in a structurally predictable way that diffusion models avoid through bidirectional correction. This is more informative than raw accuracy comparisons.

## Suggestions

- Add a bidirectional-attention AR baseline (e.g., BERT-style MLM on the same data) to isolate the effect of the diffusion objective from context access.
- Moderate the framing language: replace "sophisticated language understanding" with "structured constraint-satisfaction reasoning" in the abstract and introduction, or add one natural language reasoning evaluation to substantiate the broader claim.
- Provide guidance on selecting α and β for MGDM, and report whether the best configuration transfers across tasks (Countdown vs. Sudoku vs. SAT).

## Score and Decision

**Calibration anchors:**
- **Block Diffusion** (avg 8.0, Oral): Novel theoretical interpolation between AR and diffusion, strong SOTA on language modeling benchmarks. Our paper has narrower evaluation scope (puzzles only) and less theoretical depth on the model itself.
- **SymmetricDiffusers** (avg 8.0, Oral): Discrete diffusion on permutations, evaluated on combinatorial tasks. Similar domain, but with cleaner theoretical grounding and evaluation. Our paper has a broader conceptual framing (subgoal imbalance) but weaker evaluation breadth.
- **Latent Diffusion with LLMs** (avg 3.0, Reject): Also claims diffusion improves reasoning over AR, also evaluates on toy tasks, also has confounded comparisons. Our paper is substantially stronger—deeper analysis, better baselines, dramatic empirical gaps, and a mechanistic explanation for the gains.
- **Limits of Deep Learning** (avg 7.0, Poster): Theoretical analysis of architectural limitations with empirical corroboration. Similar spirit and scope of contribution.
- **Overclaiming papers** (avg 4.67–5.75): Papers that overclaim beyond their evaluation scope receive scores in this range.

Our paper sits clearly above the rejected "Latent Diffusion with LLMs" (3.0) because it has deeper analysis, stronger baselines, a clearer mechanistic explanation, and dramatic empirical improvements. But it sits below "Block Diffusion" (8.0) and "SymmetricDiffusers" (8.0) because its evaluation scope is limited to puzzles, it overclaims language understanding capabilities, and it doesn't fully disentangle the sources of its advantage. It's comparable to "Limits of Deep Learning" (7.0) in spirit—identifying an important architectural limitation and providing analysis—but with weaker theoretical contributions and overclaimed scope.

The paper makes a genuine contribution in identifying and analyzing subgoal imbalance, proposing an effective method, and providing mechanistic insight. The major weaknesses—overclaimed scope and confounded attribution—are significant but don't invalidate the core empirical and analytical findings.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>