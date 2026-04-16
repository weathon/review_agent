Now I have a solid understanding of the paper and its comparison points. Let me compile the final review.

## Summary

This paper proposes DRAFT, a framework that iteratively refines tool documentation for LLMs through a self-driven trial-and-error process consisting of three phases: experience gathering (Explorer simulates tool usage), learning from experience (Analyzer identifies documentation issues), and documentation rewriting (Rewriter updates the documentation). The framework includes a diversity-promoting exploration strategy and a tool-adaptive termination mechanism. Experiments on ToolBench and RestBench across three LLMs show consistent improvements in tool-use performance, with some evidence of cross-model generalization and improvements in tool retrieval.

## Strengths

- **Well-motivated and practical problem**: The paper identifies a genuine and understudied bottleneck in tool learning—human-written tool documentation is often incomplete, redundant, or inaccurate for LLM consumption. The examples in Figure 1(a) effectively illustrate these issues, and the problem is practically important given the growing use of tool-augmented LLMs.

- **Clear and modular framework design**: The Decomposition into Explorer, Analyzer, and Rewriter with iterative feedback is intuitive and mirrors human learning-through-practice. The diversity-promoting exploration strategy and tool-adaptive termination mechanism are sensible design choices with clear motivation. The ablation study (Table 2) confirms they contribute positively.

- **Consistent empirical improvements across models and datasets**: DRAFT achieves improvements in both CP% and Win% across all three evaluated LLMs (GPT-4o, GPT-4o-mini, Llama-3-70B) on all three dataset configurations, including outperforming the EasyTool baseline. Notably, GPT-4o-mini with DRAFT documentation even surpasses GPT-4o without DRAFT on ToolBench (CP%: 47 vs. 37).

- **Cross-model generalization evidence**: Figure 7 shows that when Llama-3-70B is used as the backbone for DRAFT, the resulting documentation still improves all three evaluated models, suggesting the improvements capture genuine tool knowledge rather than model-specific quirks. GPT-4o as backbone yields better results, which is a reasonable finding that benefits from stronger foundation models.

- **Multiple auxiliary evaluations beyond core task performance**: The paper includes tool retrieval improvements (Table 3), human evaluation of documentation quality on completeness/conciseness/accuracy (Table 4), and iteration analysis (Figure 6), providing a broader assessment than just a single headline metric.

## Weaknesses

### Major:

- **Conflation between "better documentation" and "better task performance"**: The core claim is that DRAFT improves documentation *quality* and bridges a comprehension gap, but the primary evidence is downstream task performance (CP%, Win%) when rewritten text is used as in-context tool description. This shows that rewritten *prompts* work better, not necessarily that the documentation is more faithful to the API. The human evaluation provides some evidence, but for RestBench accuracy, 70% of cases were rated "Equal" and only 30% favored DRAFT—which the paper overclaims as "significant improvements." The gap between "better benchmark scores" and "more faithful documentation" is not adequately bridged.

- **No control for what information is actually being added to the documentation**: DRAFT's pipeline generates exploratory queries, captures tool outputs, and feeds these to the Analyzer and Rewriter. The resulting documentation likely embeds example call patterns, edge cases, and usage traces. Without characterizing what content ends up in the rewritten docs (e.g., are there now de facto few-shot examples in the description?) and without controlling for documentation length or information volume against baselines, it is unclear whether gains come from genuinely "better documentation" or from injecting additional supervision into the prompt. A one-shot rewrite baseline (having GPT-4o rewrite documentation once without iterative exploration) would help isolate the value of the iterative feedback loop, and this experiment is notably absent.

- **LLM-as-judge evaluation protocol lacks transparency and controls**: Win% is evaluated via "pairwise comparisons by a ChatGPT-based evaluator," but the paper provides no detail on whether evaluator sees system identities, whether comparisons are randomized, how ties are handled, or what prompts the evaluator uses. Since DRAFT uses GPT-4o to rewrite docs and the evaluator is also ChatGPT-based, there is a legitimate concern about shared stylistic preferences. This is a known limitation in the field but still undermines the strongest claims.

### Minor:

- **Termination mechanism is not grounded in downstream performance**: The tool-adaptive termination criterion (Δ > τ where Δ combines BLEU and embedding similarity between consecutive doc versions) measures *stability of the rewriting process*, not whether the documentation has reached optimality for downstream use. The ablation in Table 2 shows that without adaptive termination, performance drops, but this doesn't establish that Δ ≈ τ at the right point—just that running all iterations to completion is worse. A more direct validation would track downstream performance versus Δ.

- **Limited ablation depth**: Only two ablation variants (w/o diversity, w/o adaptive) are presented on a single dataset with a single LLM. Key design choices are unexplored: the contribution of the Analyzer phase versus direct rewriting from exploration feedback, the impact of varying the similarity threshold φ and termination threshold τ, and the effect of removing history from the Rewriter.

- **Computational cost not reported**: DRAFT requires multiple LLM calls per tool per iteration (Explorer, Analyzer, Rewriter across up to 5 iterations). The paper lacks any discussion of API costs, token consumption, or runtime, which is important for practical applicability—especially since the claimed advantage over manual documentation is scalability.

- **Cross-model generalization is tested within a narrow model family**: The evaluated models (GPT-4o, GPT-4o-mini, Llama-3-70B) are all large decoder-only transformers trained on similar corpora. Claims about "robust cross-model generalization" would be stronger with evaluation on more diverse architectures or smaller models.

### Trivial:

- The claim in Section 2.5 that DRAFT "dynamically maintains an accurate and up-to-date representation of evolving features" suggests adaptability to API changes, but no experiment tests this scenario (e.g., modifying tool behavior mid-deployment).

## Nice-to-Haves

- A one-shot rewrite baseline to isolate the contribution of iteration and tool-use feedback versus simply having a strong model rewrite documentation once.
- Sensitivity analysis on φ and τ hyperparameters.
- A catalog of what specific content changes DRAFT introduces (e.g., missing parameter constraints added, edge cases documented, examples inserted) to characterize the nature of improvements.
- Cost analysis (token consumption and API cost per tool).
- Evaluation on tools with already high-quality documentation to test whether DRAFT could harm such cases.

## Removed Points

- **Termination condition is "perverse" or logically inverted (Harsh Critic Point 3)**: Upon checking Eq. 5 and Algorithm 1, the logic is correct: Δ measures similarity (high values = similar docs = convergence), and stopping when Δ > τ is the intended behavior. The critic incorrectly interpreted "degree of change" as implying Δ should be low for convergence, but the actual implementation correctly stops when docs become similar. The concern about Δ not reflecting downstream performance is kept as a minor weakness, but the claim of a perverse reversal is removed.

- **Missing baseline such as prompt engineering / DSPy baselines (Human Finder / PLAY2PROMPT analogy)**: The paper already compares against EasyTool (a GPT-based documentation rewriting method) and DFSDT (a reasoning method). Adding one-shot rewrite is a nice-to-have but not a required missing baseline—the existing baselines are reasonable for the problem setting.

- **No discussion of multi-tool composition scenarios (Human Finder / PLAY2PROMPT analogy)**: The ToolBench I3-Instruction subset used in experiments *is* specifically designed for multi-tool scenarios (complex instructions requiring multiple tools from different categories), so this concern does not apply.

- **Risk of hallucinated documentation modifications (Human Finder)**: This is a valid but speculative concern. The paper's human evaluation on accuracy (Table 4) shows DRAFT improves accuracy on ToolBench (56% vs 0%) and RestBench (30% vs 0%), providing some evidence that hallucination is not a dominant failure mode. Downgraded to minor concern rather than a standalone weakness.

- **EasyTool not being evaluated on Spotify dataset is asymmetric (Harsh Critic)**: This is noted in the paper and EasyTool does not support this dataset. This asymmetry does not favor DRAFT—it means DRAFT has no documentation-rewriting competitor on Spotify. It deserves mention but is not a methodological flaw in the comparison as presented.

- **Hyperparameter sensitivity and cost analysis (Neutral Reviewer)**: These are practical concerns but standard for this type of work. Moved to Nice-to-Haves.

## Novel Insights

The cross-model generalization finding—that documentation refined using one model's exploration feedback also improves other models—is genuinely interesting and suggests the improvements capture tool-relevant facts rather than just prompt-tuning for a specific model's idiosyncrasies. This finding, if validated more broadly, would support the practical amortization of DRAFT's cost by running it once with a strong model and distributing the resulting documentation to weaker models. However, this finding should be interpreted cautiously given the narrow model family tested.

## Suggestions

- Add a simple one-shot rewrite baseline (single-pass GPT-4o rewrite without exploration or iteration) to isolate the contribution of the iterative feedback mechanism.
- Report token consumption and approximate cost per tool to enable practitioners to assess feasibility.
- Provide a qualitative analysis of what changes DRAFT makes to documentation (e.g., categorize as: adding missing parameter constraints, removing redundant text, correcting inaccurate descriptions, adding usage examples) to ground the claim about documentation quality improvement.
- Report standard deviations or confidence intervals for key metrics, at minimum from a few random seeds, to establish robustness.

## Score and Decision

Calibration against similar papers:
- **PLAY2PROMPT** (very similar: iterative tool documentation refinement, zero-shot): scores 3–5, rejected. DRAFT has a more thorough evaluation (3 datasets, 3 models, human eval, retrieval), but shares similar concerns about evaluation metric reliability and missing simple baselines.
- **ToolEVO** (tool adaptation with MCTS): scores 1–6, accepted as poster. Had weaker presentation and methodology concerns but tackled a novel problem (dynamic tool evolution).
- **MAC-CAFE** (iterative KB refinement): scores 1–6, rejected. Had fundamental concerns about methodology and limited novelty.
- **Tool Decoding** (constrained decoding for tool use): scores all 6, poster. Well-scoped contribution with clear experimental methodology.

DRAFT's contribution—automated iterative documentation refinement—is novel and well-motivated, with consistent empirical results across multiple models and datasets. However, the evidence for the strongest claims about "documentation quality" is weaker than presented (primarily downstream task performance, with human eval showing mostly "equal" judgments), and several methodological gaps (no one-shot baseline, no analysis of what actually changes in the docs, limited ablation) prevent a confident assessment of what DRAFT is really doing. The core idea is sound and practically useful, but the execution and evaluation don't fully back the narrative.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>