## Summary
DRAFT proposes an iterative framework for refining tool documentation for LLMs via self-driven trial-and-error interaction. The method decomposes documentation improvement into three phases—Experience Gathering (Explorer), Learning from Experience (Analyzer), and Documentation Rewriting (Rewriter)—augmented by a diversity-promoting exploration strategy (similarity constraint + self-reflection) and a tool-adaptive termination mechanism (BLEU + embedding similarity). Experiments on ToolBench and RestBench across three LLMs show consistent gains in Correct Path Rate and Win Rate, with supporting evidence from tool retrieval and human evaluations.

---

## Strengths

- **Genuine novelty in the application of execution feedback to documentation refinement.** Prior work (e.g., EasyTool) rewrites documentation one-shot using LLMs without grounding in actual tool execution traces. DRAFT uniquely grounds each revision in real tool responses (e.g., actual parameter errors, return field structures), enabling documentation that reflects actual tool behavior rather than a paraphrase of the original. This is a meaningful and underexplored direction.

- **Diversity-promoting exploration is a principled design.** The combination of embedding-based similarity constraint (Eq. 2) with self-reflection for regeneration is a thoughtful mechanism that addresses coverage failure in naive iterative exploration. The ablation in Table 2 confirms it matters: removing it drops CP% from 88 to 84 and Win% from 71 to 69 on TMDB with GPT-4o.

- **Tool-adaptive termination addresses a genuine over-iteration failure mode.** Figure 6 shows a non-monotonic performance curve across iterations, confirming that running too many refinements degrades performance. The mechanism is ablated in Table 2 (drop to 80% CP / 68% Win without it), giving direct empirical backing rather than just a theoretical claim.

- **Multi-stage evaluation is distinctive.** The paper validates improvements not only on downstream tool-use task performance (Table 1) but also on tool retrieval quality (Table 3) and human comprehension (Table 4). The retrieval analysis is particularly compelling: it shows that rewritten documentation is semantically improved even independent of any generation task, reducing concerns that gains are task-specific artifacts.

- **Cross-model transfer shows documentation improvements are model-agnostic.** DRAFT documentation refined with GPT-4o improves performance of GPT-4o-mini and Llama-3-70B (Table 1). On ToolBench, GPT-4o-mini + DRAFT (47% CP) even surpasses GPT-4o without DRAFT (37% CP), suggesting that documentation quality is a meaningful bottleneck worth targeting.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Missing single-pass-with-feedback ablation undermines the core claim about iteration.** The paper's central claim is that *iterative* trial-and-error refinement is necessary. However, the ablation in Table 2 only removes the diversity or termination mechanisms—it never compares DRAFT against a single-pass rewrite in which the Rewriter is given a batch of tool execution results at once without iteration. Without this, it is impossible to determine whether the iterative loop adds value beyond simply exposing the model to tool output once. This is the most critical missing experiment.

- **Win% evaluation is underspecified.** Win% is the primary metric for three out of six experimental columns in Table 1, yet the paper describes it only as computed by a "ChatGPT-based evaluator" (Section 3.1) with no disclosure of: which specific model was used, the prompt template, whether evaluation was blinded to method identity, whether position/order of outputs was randomized, or any measure of evaluator consistency. This is a major reproducibility and validity concern.

- **Algorithm 1 has a structural issue and confusing termination logic.** As written, when the break condition fires at Line 16 (`if Δ > τ then Break`), execution jumps out of the loop *before* Line 19 executes (`D̃ ← D̃ ∪ t_i`). This means the final converged documentation `t_i` is never added to the output set. The algorithm as presented returns all *pre-convergence* versions but not the converged one—which is the opposite of the intended behavior. Additionally, naming Δ the "degree of change" when it is computed as a similarity metric (BLEU + cosine sim, where higher = more similar = *less* change) is internally inconsistent. The paper says "we consider the iterative process to have converged when there is minimal change," which matches the logic (high Δ = converge), but the terminology inverts intuition. This should be corrected and clarified.

- **No computational cost analysis.** DRAFT makes multiple sequential LLM calls per tool (Explorer → Analyzer → Rewriter) across up to 5 iterations. No API call counts, token consumption, wall-clock times, or cost estimates are reported. Given that gains on ToolBench with GPT-4o are +6% CP and +1% Win over EasyTool, the cost-benefit tradeoff is a meaningful practical question. This is especially important because EasyTool is described as a single-pass rewrite, which would be substantially cheaper.

### Minor

- **Cross-model generalization claim is stronger than the evidence.** The only experiment with an alternative backbone is Figure 7, using Llama-3-70B on RestBench-TMDB only. The paper concludes from this that the approach generalizes across models, but one dataset and one alternative backbone are insufficient to substantiate "robust cross-model generalization capabilities." Testing on at least RestBench-Spotify and ToolBench would meaningfully strengthen this claim.

- **Ablation covers mechanisms but not modules.** The ablation (Table 2) validates the diversity and termination mechanisms, but there is no ablation isolating the Analyzer module. A Rewriter-only variant (using `(t_{i-1}, e_i, r_i)` directly without the Analyzer's intermediate suggestions `s_i`) would clarify whether the Analyzer is adding value beyond the information already available. Similarly, the exploration directions `d_i` produced by the Rewriter are not independently evaluated.

- **Retrieval results are not uniformly positive.** Contriever on Spotify @10 slightly decreases from 49.6 to 49.2 with DRAFT, and BM25 on Spotify @1 is unchanged (43.9 vs 43.9). The paper does not discuss these non-improvements, which suggests the benefits may be tool/dataset-dependent.

- **Human evaluation is small-scale and lacks agreement statistics.** Only 3 annotators evaluate 50 cases, with no inter-annotator agreement (e.g., Fleiss' κ) reported. For RestBench accuracy, 70% of cases are labeled "Equal," which suggests either the task is difficult to judge or many cases show no clear improvement—this deserves discussion rather than being passed over.

### Tiny

- **EasyTool backbone model mismatch.** The paper uses GPT-4o as DRAFT's backbone and compares against EasyTool, which uses "ChatGPT" (presumably GPT-3.5 or earlier). This comparison is not apples-to-apples in terms of the rewriting model's capability, which inflates the apparent advantage. Rerunning EasyTool with GPT-4o would give a cleaner comparison.

- **ToolBench subset claim scope.** Since only the I3-Instruction subset is used, claims about ToolBench performance should be scoped accordingly.

---

## Nice-to-Haves

- **Compare against in-context few-shot demonstrations from explored examples.** A natural alternative to rewriting documentation is to provide the Explorer's gathered (query, result) pairs directly as few-shot demonstrations during inference. This would not modify the documentation and is simpler to deploy. If DRAFT significantly outperforms this, it provides a strong practical argument for the documentation-rewriting approach.

- **Task-completion metric beyond path correctness.** CP% measures whether the model's tool-call sequence contains the ground-truth subsequence, not whether the final user query is answered correctly. A model could follow the right tool path but misuse the returned results. Adding a task-completion or answer-quality metric would strengthen the claim of practical benefit.

- **Sensitivity analysis for termination threshold τ.** The paper sets τ = 0.75 without discussion of how sensitive results are to this value. A small sweep would validate the chosen value and inform practitioners deploying DRAFT on new tool sets.

- **Trajectory visualization across iterations.** The paper shows a single before/after snapshot (Figure 2 right panel). Showing how the documentation evolves iteration-by-iteration for a concrete example would directly substantiate the claim that each iteration adds meaningful signal rather than noise.

- **Documentation length and completeness/conciseness tradeoff analysis.** The paper claims DRAFT produces documentation that is simultaneously more complete and more concise. These can be in tension. Tracking average documentation length across iterations and its relationship to performance would clarify whether gains come from enriched content, streamlined content, or both.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Figure 1(c) labeling concern (Harsh Critic).** The critic flagged that Figure 1(c) appears to show raw documentation winning 92.5% of the time, which contradicts the paper's claims. This is almost certainly a parsing artifact from the text extraction of the figure—the column headers (Raw vs. Improved) were likely swapped in parsing. The paper's narrative, its Table 1 results, and the figure caption all consistently claim that DRAFT-improved documentation is preferred. This should not be treated as a genuine paper flaw.

- **"Causal claim under-justified" / other bottlenecks not separated (Harsh Critic).** The critic argues the introduction must first prove documentation is the *dominant* bottleneck over planning errors, schema complexity, etc. This is scope creep: the paper's contribution is to show that documentation quality *is* a meaningful bottleneck worth addressing, not that it is the only one. Showing consistent downstream gains is sufficient justification. Removed.

- **Section 2.5 "marketing-oriented" (Harsh Critic).** Critique of the tone of a summary section—this is a style/formatting nitpick. Removed.

- **"No constraints on hallucinated rewrites" (Harsh Critic).** Speculative concern about the Rewriter fabricating tool behavior not observed in exploration. While theoretically valid, the paper does not claim to prevent hallucination and this falls outside stated scope. Additionally, the Rewriter is always conditioned on actual tool execution traces (r_i), which grounds its rewrites. Removed as a separate weakness; the human evaluation (Table 4, accuracy dimension) provides partial empirical evidence against systematic hallucination.

- **"Natural-language feedback is asserted better than scalar feedback without direct evidence" (Harsh Critic).** The paper discusses this design choice in the context of the broader LLM feedback literature (Section 4, Learning from Feedback) and provides sufficient conceptual justification. Not requiring an ablation of scalar vs. NL feedback is reasonable scope given the paper's focus. Removed.

- **"Fully automated / dynamically maintaining / explainability are over-claims" (Harsh Critic).** These are summary statements about the system's properties, not empirical claims being advanced as contributions. Critiquing them as insufficiently proven is a style nitpick on how system features are described. Removed.

- **Generic "missing limitations section" as a standalone weakness.** The absence of a dedicated limitations section is a presentation preference; the paper discusses several limitations inline (e.g., performance degradation from over-iteration, cost savings from termination). Removed as a standalone weakness, though the cost/safety concerns are kept in the main weaknesses.

- **"Related work is not sufficiently comparative" (Harsh Critic).** Vague and not specific enough to act on. Removed per instructions (no missing related works).

---

## Novel Insights

The spark finder's observation about providing explored examples as few-shot in-context demonstrations at test time (rather than baking them into documentation) is the most actionable novel insight from the synthesis. This baseline is not in the paper, and if it performs comparably to DRAFT, the justification for the more expensive documentation-rewriting pipeline weakens significantly. Conversely, if DRAFT outperforms it, that result would be one of the strongest possible empirical arguments for the paper's approach—since it would show that reformulating knowledge into persistent documentation is better than retaining raw interaction traces. This experiment is missing and is the single highest-leverage addition the authors could make to the paper. A secondary insight is that the retrieval gain analysis (Table 3) implicitly validates that DRAFT's documentation improvements are not specific to any generation model or prompt style, because retrieval is a model-agnostic downstream task—this is a stronger form of generalization evidence than the cross-model generation experiments, and the paper undersells it.

---

## Suggestions

1. **Add a single-pass-with-feedback baseline.** Run the Rewriter once using a batch of N Explorer traces (where N = average iterations in DRAFT) without the iterative loop. This is the minimal ablation to justify the iterative architecture and should be straightforward to implement.

2. **Add in-context few-shot demonstration baseline.** At inference time, prepend the (query, parameters, result) triples collected by DRAFT's Explorer to the prompt without modifying the documentation. Compare performance against DRAFT documentation. This tests whether documentation rewriting provides persistent, generalizable benefit over ephemeral context augmentation.

3. **Fully specify the Win% evaluator.** Publish the exact model name (version), evaluation prompt, position randomization protocol, and any cross-run consistency check. Without these details, Win% results cannot be reproduced or verified.

4. **Fix Algorithm 1 output logic.** Move Line 19 (`D̃ ← D̃ ∪ t_i`) to before the break check, or add explicit handling to output `t_{i-1}` when convergence is triggered, and rename Δ to something like "stability score" or "convergence score" to avoid the "degree of change" terminology inversion.

5. **Report cost per tool.** Add a table showing average API calls, tokens consumed, and approximate monetary cost per tool for DRAFT vs. EasyTool. Even rough estimates would allow practitioners to make an informed tradeoff decision.

6. **Rerun EasyTool with GPT-4o backbone.** Use the same backbone for EasyTool to make the comparison model-controlled. If DRAFT still outperforms, the advantage is attributable to iteration and feedback rather than model capability.

7. **Extend Figure 7 to all datasets.** Using Llama-3-70B as backbone on TMDB, Spotify, and ToolBench (rather than just TMDB) would meaningfully substantiate the cross-model generalization claim.

---

**Overall assessment:** DRAFT addresses a real and underappreciated problem—that human-written API documentation is often misaligned with LLM comprehension—and the core idea of using actual tool execution feedback to iteratively improve documentation is genuinely novel and practically significant. Empirical results are consistent across models and benchmarks, and the multi-dimensional evaluation (task performance + retrieval + human judgment) is a notable strength. The paper is held back primarily by the absence of a critical ablation (single-pass-with-feedback), underspecification of the Win% evaluator, and an algorithmic presentation issue in Algorithm 1. These are fixable gaps, and if addressed, the paper would stand on substantially stronger footing. As submitted, the work is promising and above the workshop tier, but the missing ablation is a genuine threat to the core narrative and should be resolved before the contribution can be considered fully established.