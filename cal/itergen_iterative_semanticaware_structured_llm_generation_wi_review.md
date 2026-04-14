=== CALIBRATION EXAMPLE 10 ===

# Final Consolidated Review
## Summary

ITERGEN introduces a framework for iterative, grammar-guided LLM generation that enables bidirectional navigation through partially-generated output at the level of grammar symbols (non-terminals and terminals) rather than raw tokens. By maintaining a symbol-to-position map via an incremental LR parser and managing the KV cache coherently across forward/backward calls, the system lets users programmatically enforce semantic constraints mid-generation through targeted backtracking and resampling. The framework is evaluated on SQL generation (9 models, Spider benchmark), privacy leakage prevention (Enron email dataset), and Vega-Lite specification generation.

---

## Strengths

- **Grammar-symbol navigation abstraction is genuinely novel.** The core `forward(stop_symbol, n)` / `backward(stop_symbol, n)` API that operates over BNF grammar symbols—rather than raw token counts—is a conceptually clean and practically useful abstraction that prior work (GUIDANCE, SYNCHROMESH, PICARD) does not provide in this general form. The symbol position map built incrementally via the LR parser's reduce operations is the technical enabler that makes this work across arbitrary user-provided grammars.

- **KV-cache-coherent backtracking is a meaningful engineering contribution.** Maintaining cache coherence during bidirectional navigation avoids the expensive KV-prefill step on every retry, making iterative resampling feasible in practice. This is non-trivial and directly enables the efficiency advantage ITERGEN claims over full-restart approaches.

- **Breadth of evaluation.** Testing across 9 LLMs spanning base, instruction-tuned, and code-specific variants (0.5B–8B parameters) on three distinct task domains provides meaningful coverage and shows consistent, non-cherry-picked improvements. The CRANE follow-on work (Banerjee et al., 2025) cited in related work further validates the framework as a reusable building block.

- **The decoding trace with recurrence penalty addresses a real practical problem.** Maintaining a tree of decoded tokens and penalizing previously explored paths during backtracking is a pragmatic solution to the looping problem; the ablation study in the appendix provides partial empirical support.

---

## Weaknesses

### Fatal
None.

### Major

- **Critical missing ablation: backtracking vs. grammar guidance alone.** The paper never compares ITERGEN against full-output rejection sampling (generate complete SQL until schema-valid, restart otherwise). Without this baseline, it is impossible to determine whether the gains come from (a) targeted partial backtracking at the grammar-symbol level, or (b) merely applying semantic constraints during any generation loop. This is the most important experiment to establish the core claim about the value of grammar-symbol-level navigation. The absence of this comparison is a significant gap.

- **Ambiguous headline accuracy figures.** The abstract states "18.5% mean improvement over state-of-the-art." From Table 1, the absolute differences in overall accuracy (ITERGEN vs. SYNCode) average to approximately 6.4 percentage points across 9 models. The 18.5% figure is recoverable as a relative improvement: ITERGEN's mean accuracy (41.63%) vs. SYNCode (35.22%) yields ~18.2% relative gain—but this is never stated as a relative improvement in the abstract or conclusion. Readers may interpret this as an absolute gain of 18.5 percentage points, which is incorrect. The reporting should be made unambiguous.

- **Privacy experiment: narrow leak definition and weak baselines.** The paper defines a "leak" as verbatim output of a known email address, which allows ITERGEN to solve the task through exact-match blacklisting. While useful, this definition is not disclosed prominently in the main text (only in the appendix). More critically, the privacy experiment compares only against unconstrained STANDARD generation—neither SYNCode nor any other constrained decoding baseline is included. This makes it impossible to attribute the improvement to backtracking specifically (vs. any grammar-enforced constraint). Additionally, perplexity is used as a "response quality gauge," but perplexity measures fluency, not informativeness; a model generating fluent but uninformative text would appear unharmed by this metric.

### Minor

- **Formal definition error in Section 3.2.** The definition of the Forward function reads: "let $O_f \in \Sigma^*$ be the output after the call to the **backward** function." This should say "forward function." Errors in formal definitions undermine the precision that a systems paper at this level requires.

- **Hyperparameter selection methodology.** `max_iter=20` and γ=0.7 are selected based on "a small subset of the training dataset" with no cross-validation or held-out split described. For ICLR, this warrants at least a brief description of how many training examples were used and whether the hyperparameter selection could have overfit. The appendix ablation partially addresses sensitivity but does not resolve the selection methodology concern.

- **KV-cache management details insufficient for reproducibility.** The mechanism for maintaining KV-cache coherence during backward operations is described in a single paragraph with no pseudocode. Since KV-cache manipulation is not natively supported by most HuggingFace inference paths and must be custom-implemented, readers wishing to reproduce or extend the work cannot do so without access to the source code alone.

- **Vega-Lite evaluation: inconsistent model coverage.** SQL uses 9 models; Vega-Lite uses 3, with no stated justification (computational cost, grammar complexity, etc.). This asymmetry weakens the generalizability claim for the Vega-Lite domain.

- **Extra-token distribution artifact unanalyzed.** The paper correctly explains that the LR parser's lookahead requirement forces the model to generate one extra token that is then discarded. This means the model operates in a context that is subsequently modified—a subtle distributional effect. The paper correctly flags it as an implementation detail users need not handle, but the potential impact on output quality is not characterized anywhere.

### Tiny

- **Behavior at max_iter exhaustion unspecified.** When the iteration limit is reached, the paper does not state whether ITERGEN returns the best partial output, an invalid output, or raises an error. This matters for understanding the floor behavior of the system.

- **Failure case analysis confined to appendix.** The main paper would benefit from at least a summary of failure modes (e.g., when does ITERGEN fail to converge? which grammar structures cause the most backtracking?).

---

## Nice-to-Haves

- **Accuracy-latency trade-off analysis.** A plot of accuracy vs. mean backtracking iterations (or wall-clock time) would help practitioners decide on max_iter settings and understand the cost-benefit curve.

- **Comparison to dedicated privacy-preserving generation approaches** (e.g., DP-based decoding, post-hoc filtering) to contextualize the 100% leak prevention result. The current setup proves ITERGEN solves this instantiation of the problem; comparison to alternatives would establish when ITERGEN is the right tool.

- **Overhead scaling with grammar complexity and generation length.** The reported time overhead is for fixed SQL and Vega-Lite grammars; characterizing how overhead scales with grammar size and output length would help users assess applicability to larger grammars (e.g., full Python, nested JSON schemas).

- **Iteration distribution across the dataset.** Figure 6 in the appendix covers Vega-Lite for one model; making this a first-class result (for SQL as well) would reveal whether convergence is typically fast or whether the max_iter bound is frequently hit.

- **Batch generation support.** Acknowledged as a limitation, but even a brief discussion of the technical challenges and a possible design sketch would strengthen the paper's standing as a general-purpose generation framework.

---

## Removed Points

*These points are flagged for removal; treat with caution.*

- **Harsh Critic: "Statistical significance is never assessed."** For greedy-decoding evaluation on standard NLP benchmarks (Spider, NLV Corpus), single-run evaluation is the accepted norm in the field. Demanding CIs or p-values for these experiments imposes a methodological standard not applicable to this community. The concern is valid only for the very small per-model differences (e.g., 1.2 pp), but those are marginal results, not the paper's headline claims.

- **Harsh Critic: "ITERGEN would be better suited to OSDI/EuroSys."** Venue opinion; not a technical criticism.

- **Harsh Critic: "No theoretical analysis of convergence or distribution preservation."** This is an empirical systems paper; demanding theoretical proofs is outside the scope of ICLR empirical contributions in this subarea. KEEP the distribution distortion concern as an empirical limitation (captured above), but remove the demand for formal proofs.

- **Spark Finder: "Direct comparison to GUIDANCE/LMQL."** The paper correctly explains that GUIDANCE's `stop_at` works with regular expressions and does not support grammar-level backtracking, while LMQL also lacks this. A direct head-to-head would measure different capabilities; the paper's related work treatment is sufficient.

- **Spark Finder: "Larger dataset / more models."** The Spider dataset (1034 examples) and 9-model zoo are above average for this type of systems evaluation; requesting more is generic scope creep.

- **Positive Reviewer: "Open source and reproducibility" as a strength.** Generic; applies to many papers.

- **Harsh Critic: "18.5% is not reproducible from Table 1."** As shown above, the figure is recoverable as a relative improvement using Table 8 averages (41.63% vs. 35.22%). The criticism is not that the number is fabricated, but that it is ambiguously presented—this is retained as a Minor weakness.

---

## Novel Insights

The most underappreciated insight in the paper is the *shift in granularity for constraint enforcement*: existing grammar-guided methods enforce constraints at the token level (masking invalid next tokens), while ITERGEN enforces them at the *non-terminal* level (backtracking entire semantic units). This separation of concerns means the constraint logic is written in terms of program semantics (e.g., "is this column name in the schema?") rather than token prediction, making the API substantially more expressive for end-users. A secondary insight—visible in the privacy experiment—is that exact-match blacklisting implemented *during generation* via grammar backtracking is both simpler and more efficient than post-hoc filtering or prompt engineering, and the perplexity data in Table 2 (only minor degradation) supports that this does not substantially harm fluency. What remains uncharacterized, and would be the most valuable contribution of a follow-up, is whether the grammar-symbol-level backtracking policy systematically finds *better* semantic fixpoints than token-level rejection sampling, or whether the efficiency gain is the primary advantage.

---

## Suggestions

1. **Add the rejection-sampling ablation.** Implement a baseline that generates complete SQL/Vega-Lite outputs and restarts from scratch whenever a semantic constraint is violated (with the same max_iter budget). This single experiment would cleanly establish whether the grammar-symbol-level *targeting* of backtracking contributes over the semantic constraint checking itself.

2. **Clarify the 18.5%/17.8% figures.** In both the abstract and conclusion, state explicitly that these are relative improvements: "(18.5% relative improvement, from 35.2% to 41.6% absolute accuracy)." This takes one sentence and removes a significant source of ambiguity.

3. **Expand the privacy experiment.** Add SYNCode as a privacy baseline (grammar-constrained generation without backtracking) to disentangle the grammar constraint from the backtracking contribution. Consider replacing perplexity with a task-appropriate utility metric such as BERTScore against a reference non-leaky response.

4. **Fix the formal definition in Section 3.2.** "Let $O_f \in \Sigma^*$ be the output after the call to the **backward** function" in the Forward function definition should read "forward function."

5. **Add a pseudocode block for KV-cache management.** Even 10–15 lines of pseudocode in an appendix showing how cache indices are tracked and truncated during backward calls would substantially improve reproducibility.

6. **State max_iter exhaustion behavior explicitly.** Add one sentence to Section 3.2 or the experimental setup clarifying what ITERGEN returns when the iteration limit is reached.

---

**Overall assessment:** ITERGEN is a solid, well-motivated systems contribution with a genuinely useful grammar-symbol navigation abstraction and consistent empirical improvements across multiple models and domains. The paper is hampered primarily by ambiguous reporting of headline claims, a critical missing ablation (targeted backtracking vs. full restart), and a thin privacy experiment that cannot be fully interpreted. The technical core is sound and the framework is already demonstrably useful (evidenced by the CRANE citation). Addressing the missing ablation and clarifying the accuracy reporting would substantially strengthen the paper's standing.

- **Novelty:** Moderate-to-high; the grammar-symbol-level bidirectional navigation API is original and practically useful.
- **Technical soundness:** Sound at the systems level, with notable gaps in formal definitions and hyperparameter methodology.
- **Empirical support:** Adequate breadth across models, but missing the most critical ablation and relying on a weak privacy experimental design.
- **Significance:** High for practitioners building structured generation systems; the CRANE follow-up suggests real downstream impact.
- **Clarity:** Good overall, marred by the formal definition error and ambiguous presentation of headline numbers.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 6.0]
Average score: 6.7
Binary outcome: Accept
