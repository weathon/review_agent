# FormalML: A Benchmark for Evaluating Formal Subgoal Completion in Machine Learning Theory

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Large language models (LLMs) have recently demonstrated remarkable progress in formal theorem proving. Yet their ability to serve as practical assistants for mathematicians, filling in missing steps within complex proofs, remains underexplored. We identify this challenge as the task of subgoal completion, where an LLM must discharge short but nontrivial proof obligations left unresolved in a human-provided sketch. To study this problem, we introduce FormalML, a Lean 4 benchmark built from foundational theories of machine learning. Using a translation tactic that converts procedural proofs into declarative form, we extract 4,937 problems spanning optimization and probability inequalities, with varying levels of difficulty. FormalML is the first subgoal completion benchmark to combine premise retrieval and complex research-level contexts. Evaluation of state-of-the-art provers highlights persistent limitations in accuracy and efficiency, underscoring the need for more capable LLM-based theorem provers for effective subgoal completion.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a Lean 4 theorem proving benchmark, FormalML, with problems extracted from two formalization projects about machine learning theory. Unlike existing benchmarks, FormalML focuses on subgoal completion, rather than proving theorems from scratch -- motivated by the fact that users of interactive theorem provers are generally interested in calling automation in the middle of a partial proof. Experiments show that the benchmark is still generally challenging for models at the 7B scale especially at Pass@1, with STP doing best at 63% Pass@32. The paper shows several other analyses of adding premise retrieval (with mixed improvements) and expert iteration (generally improving model performance).

### Strengths
The paper shows a novel angle on dataset construction for theorem proving, focusing on subgoals. The extraction pipeline should be easy to use in other repositories beyond the two used here. It's good to see this expanding the evaluation of prover models beyond mathlib and including projects from the broader Lean community, which helps it represent more use cases of Lean. I'd expect this benchmark to be used by future work in LLMs for theorem proving, which currently has significant momentum.

The paper is generally very clear and describes the pipeline and the results well.

The results and analyses are also interesting, with techniques that are generally adopted now in LLM reasoning (like long CoT in recent reasoning models) not always being beneficial. Thus, this evaluation is complementary to existing benchmarks, not just showing the same patterns (besides being well motivated).

### Weaknesses
I have some issues with the evaluation and the interpretation of the results.

For evaluation, it would be helpful to know how many of the goals are solvable with simple automation tactics in Lean, such as grind, aesop, or canonical. Users generally try these (or similar, more domain-specific, eg ring) tactics first when they want proof automation. This would help readers get a better understanding of what fraction of the goals are indeed challenging for traditional automation.

For the interpretation, I think that actually some of the models seem to do rather well. Again, depending on whether these subgoals are out of reach of traditional automation, 25% pass@1 actually does not sound bad to me. Plus, STP with Pass@32 already achieves 63%. Thus, I suggest being less bold on the claim that they "are inadequate as practical tools" if the reference for this is this pass@1 or pass@4 evaluation.

Given the rapid progress in this area, and progressive gains in efficiency, it's likely that smaller models on smaller budgets might achieve similar performance on this benchmark in not too long. While FormalML might be useful to track that progress, there's a risk of it being saturated quickly. It would help to perhaps include other projects that are at the moment significantly more challenging, to motivate the community to work on those. If the best results on level 3 goals (in Figure 4) are already over 60%, it's seems that the current methods are perhaps on track to tackle them with smaller budgets/models given sufficient efficiency improvements on all fronts.

For the evaluation of retrieval, it seems that there's no model doing retrieval itself, and the experiment is mainly testing the ability of models to use in-context statements. While this is a useful analysis, the descripiton is a bit misleading. For instance, the conclusion is that models "often struggle with premise retrieval". But retrieval here is done manually, not by the model, so the model is not struggling with retrieval, but rather in using the premises. Thus, here I suggest simply rephrasing the conclusion.

### Questions
- What is the performance of simple automation tactics on FormalML?
- Did you analyze the examples of theorems solved by DeepSeekProver without CoT that failed with CoT? What seem to be the most common failure modes with CoT when that happens (ie when it succeeds without it)?
- In the retrieval experiment, how do models perform in a "perfect retrieval" scenario, where you give them exactly (and only) the premises used in the proof? That would give a sort of ceiling for the impact of retrieval here.
- Was there any reason to not include other repositories (such as those used in the Expert Iteration experiment) in the benchmark? It seems to me that the pipeline should be very generally applicable.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes **FormalML**, a Lean 4-based benchmark for evaluating LLM performance in *subgoal completion*—a task designed to align with the vision of "mathematicians providing high-level proof frameworks while models verify intermediate conclusions" (as illustrated in Figure 1). The benchmark extracts 4,937 subgoals from optimization and probability theories using the custom `to_theorem` tactic, which decomposes procedural proofs into declarative subgoals. Evaluations of state-of-the-art theorem provers (e.g., STP, DeepSeek-Prover-V2) reveal underwhelming performance, though this is partially attributed to the subgoals’ nature: most can be solved with basic tactics like `rw` and `simp`, making them "out-of-distribution" for models not trained on such granular steps.

### Strengths
1. **Vision Alignment with Practical Research Needs**: The paper identifies subgoal completion as a critical gap in LLM-aided formal theorem proving, directly targeting the real-world workflow where mathematicians prioritize "inspiration and intuition" (high-level frameworks) over tedious technical details. This vision addresses a tangible pain point in mathematical research and formal verification.
2. **Novel Benchmark for a Underexplored Task**: FormalML fills a void in existing benchmarks by focusing on subgoal completion (rather than full-proof generation) and integrating premise retrieval and research-grade ML theory contexts. It provides a structured, quantifiable foundation for evaluating LLM capabilities in intermediate proof steps.
3. **Transparent Analysis of Model Limitations**: The paper acknowledges that poor model performance is partially due to the "out-of-distribution" nature of the subgoals (solvable via basic tactics like `rw`/`simp`). This honest attribution avoids overstating limitations and frames performance gaps as actionable, context-dependent challenges.

### Weaknesses
While the paper’s core vision of human-AI collaboration in formal theorem proving is compelling, the implementation of subgoal extraction via `to_theorem` is criticized for being overly simplistic and failing to fully realize Figure 1’s intended workflow. A key unresolved question is how training on these subgoals impacts models’ performance on standard full-proof generation tasks.

1. **Overly Simplistic Subgoal Extraction via `to_theorem`**: The `to_theorem` tactic decomposes procedural proofs into subgoals tied to "single-step tactic state transitions," which fails to capture the nuance of Figure 1’s vision. These subgoals are not the "non-trivial intermediate conclusions" mathematicians would need help with—instead, they are granular, tactic-dependent steps that lack the complexity of real-world research gaps.
2. **Limited Justification for Subgoal Utility**: Since most extracted subgoals can be solved with basic Lean tactics (e.g., `rw`, `simp`), the paper does not fully explain why LLMs are needed for this task. The subgoals do not address the "hard technical gaps" that would truly liberate mathematicians, reducing the practical impact of the benchmark.
3. **Incomplete Analysis of Training Transfer**: The paper evaluates model performance on subgoal completion but does not explore how training on these simplistic subgoals affects performance on standard full-proof generation tasks. This omission leaves a critical gap in understanding the tradeoffs of subgoal-focused training.

### Questions
1. The `to_theorem` tactic extracts subgoals from existing procedural proofs, but these subgoals are far simpler than the "non-trivial intermediate conclusions" in Figure 1. Do the authors have plans to refine subgoal extraction (e.g., integrating human annotations to mark meaningful research gaps) to better align with the paper’s core vision?
2. Most subgoals can be solved with basic tactics like `rw` and `simp`, raising questions about their utility. How do the authors plan to evolve FormalML to include more complex, tactic-agnostic subgoals that reflect the "hard technical details" mathematicians actually struggle with?
3. The paper notes that poor model performance is partially due to "out-of-distribution" subgoals. Have the authors tested whether training models explicitly on these `to_theorem`-extracted subgoals improves performance—and if so, does this training cause performance drops (i.e., "catastrophic forgetting") on standard full-proof generation tasks?
4. Given that basic tactics already solve most subgoals, what concrete advantages does an LLM-based approach offer for subgoal completion? For example, can LLMs handle edge cases or premise combinations that basic tactics cannot, and if so, how is this validated in the benchmark?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces FormalML, a Lean 4 benchmark targeting subgoal completion: filling in short but nontrivial proof steps within research-level ML theory proofs rather than producing full proofs end-to-end. The authors implement a Lean tactic, `to_theorem`, that extracts line- or segment-level subgoals from procedural proof scripts, and curate 4,937 problems spanning optimization and probability (with premise retrieval needs and complex contexts). A broad evaluation of recent LLM-based provers shows modest Pass@1, steep drops with difficulty, mixed gains from premise retrieval, and poor efficiency for long-CoT methods. They also show expert iteration improves performance, especially at low budgets.

### Strengths
1. Novelly shifts focus from full-proof generation (competition-style) to research-adjacent subgoal completion, a practical interface between human sketches and formal verification. Benchmarks combine premise retrieval with complex research contexts (from Optlib/FoML), which most prior Lean benchmarks avoid.

2. Introduces a symbolic extraction tactic (`to_theorem`) that converts procedural Lean steps into declarative subgoals at line/segment granularity, novel and broadly useful for dataset creation and tooling.

3. Clear, reproducible dataset construction pipeline. Thorough experiments across tree-search and whole-proof families, multiple budgets, domain splits, and difficulty levels; plus targeted retrieval experiments and an efficiency metric. Additionally includes error analysis (sorry vs. Lean errors) and a concrete ablation on expert iteration showing meaningful gains.

4. Provides a large and focused ML-theory benchmark likely to steer future model/training design (e.g., retrieval-aware training, efficiency-aware decoding).

### Weaknesses
1. Difficulty is proxied by proof length. That can correlate but isn’t identical to semantic difficulty. It would be good to add alternative signals and/or compare with human annotations.

2. I would go more carefully about the claim that long-CoT "shows no benefit". It seems supported on FormalML but may be dataset-specific. Also, long-CoT prompts also prime models for full-proof search, not subgoal snippets. More work would be needed before concluding broadly on this.

3. Authors note both single-line and multi-line segment extraction. It would be good to show how segment length affects success, retrieval load, and Pass@K. This would inform optimal chunking for future datasets.

4. Authors evaluate models some of which may have seen Optlib/FoML/mathlib during pretraining. How did you guard against data leakage? Any checks that extracted subgoals aren’t near-duplicates of public training proofs?

### Questions
1. Concentration on ML theory (Optlib + FoML) may limit generality to other formal domains (algebraic topology, number theory, verification). Consider adding a small cross-domain diagnostic split (e.g., mathlib analysis/algebra lemmas) to evaluate transfer and prevent domain-overfitting.

2. Consider adding refinement-at-most-N-steps interactive baselines (tool-use or repair loops) to gauge whether light interactivity beats one-shot generation.

3. Table 3 lists “Goedel-Prover-V2-8B” with size "7B". Can you double check this is consistent?

4. Slightly inconsistent terminologies: both "Mcdiarmid" and "McDiarmid" appear; both "Godel" and "Goedel" are spelt.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces FormalML, a benchmark of 4937 problems extracted from machine learning theory in a subgoal completion context (filling in sorries). The authors also use expert iteration to significantly improve the abilities of two open-source models.

### Strengths
- The paper curates a large and interesting benchmark for theorem proving in Lean sourced from existing and reputable data sources.
- Almost all leading theorem-proving models are included and at multiple sampling budgets. 
- The authors analyze current models from various perspectives: low computational budgets, premise retrieval, and long CoT.
- The authors show the effectiveness of expert iteration on their benchmark.

### Weaknesses
- The paper is limited to the domain of machine learning theory, which is somewhat narrow. Other than ease of data sourcing, there is no apparent reason why this needs to be the case. It would be great to expand the work to other domains as well. 
- Relatedly, while the findings are interesting, the authors do not comment on how the benchmark can inform and advance downstream research in formal theorem proving, especially in other domains. 
- As stated, the premise retrieval part of the paper does not consider embedding-style retrieval and only completion-based retrieval with the argument that using the same model enhances efficiency. However, a small embedding model could be cheap and produce better results.

### Questions
- The gains from expert iteration are very drastic for Goedel-Prover-8B and DeepSeek-V2, but the two models exhibit different patterns across sample budget. It is surprising that Goedel-Prover is showing uniform gains across all sample budgets, while DS-V2 shows very large gains for pass@1 and little gain for pass@32. Do you have any insights into this?
- How do the insights from this benchmark transfer to theorem proving in general? Do you envision the target to be a model specialized for machine learning theory proving?

### Soundness
3

### Presentation
3

### Contribution
3
