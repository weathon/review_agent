# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work presents a benchmark and empirical evaluation for LLM-based instructed code editing, which clearly fits ICLR’s scope on representation learning, code generation, and datasets/benchmarks.

## Minimum Quality
Pass ✅.  
The paper is in English and contains all required components (abstract, introduction, related work, benchmark construction / methodology, benchmark statistics, experiments/results, discussion/conclusion). The empirical methodology is reasonably sound for a benchmark paper; I see no fatal methodological errors or grossly misleading theoretical claims.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts to manipulate automated reviewing or hidden prompts aimed at LLM reviewers.

---

# Expected Review Outcome:

## Summary

The paper introduces **EditBench**, a benchmark for evaluating LLMs on *instructed code editing* grounded in in-the-wild IDE usage. The authors collect real VS Code interactions where users highlight code and provide natural-language edit instructions, then curate 109 core problems (expanded to 540 via multilingual comment translation) with executable test harnesses in Python and JavaScript. They evaluate 40 diverse LLMs, analyze performance across problem difficulty, edit categories, and context configurations (e.g., with/without highlighted regions and cursor position), and show that current models still struggle on these realistic edit tasks.

## Strengths

1. **Strong motivation and clear problem focus.**  
   The paper cleanly identifies a gap between existing code benchmarks (mostly from scratch generation or synthetic edits) and the rapidly growing *instructed edit* interaction in modern coding assistants. The distinction between editing an existing file under ambiguous natural instructions vs solving small stand‑alone programming puzzles is well articulated in Section 1 and Table 1, making the need for this benchmark very clear.

2. **Grounding in real-world IDE workflows.**  
   A key strength is that EditBench is built from *actual* user interactions in VS Code, via a custom extension (Figure 2). The pipeline of logging instruction, full file context, highlighted span, cursor location, and the user-accepted model output is well described in Section 3.1 and 3.2. Compared to annotator-written prompts in CanItEdit/EditEval or exercise-derived prompts in Polyglot, the examples in Table 2 convincingly show that EditBench captures much messier, underspecified instructions (e.g., “optimize the computation by better batching the latter part”) and diverse ways of expressing the same functional intent, which is much closer to deployed assistant usage.

3. **Careful benchmark curation with nontrivial test harness construction.**  
   The paper is transparent about the fact that not every logged interaction became a benchmark item, and describes a multi-step curation process: deduping, dropping trivial or stylistic edits, filtering out ambiguous cases, and then having a 5‑person expert team design test harnesses and conduct a second-pass review (Section 3.2). Using an agent only for environment setup while relying on humans for the tests and PII screening is a thoughtful design choice. This is important, because constructing robust tests for messy real-world edits is significantly harder than for LeetCode-style functions.

4. **Benchmark statistics substantiate the “more realistic and diverse” claim.**  
   Table 1 and Figure 3 are used effectively. Table 1 shows that EditBench’s instruction and code-context lengths are much larger and more variable than those of CanItEdit, EditEval, and Polyglot (e.g., code context length mean 5642 ± 7567 vs 258 ± 185 in EditEval), and that it includes 5 natural languages and 2 programming languages. Figure 3 shows a wide distribution of Python imports, with 74 unique imports versus ≤25 for baselines, which supports the claim that the benchmark covers more varied software stacks and libraries. These quantitative comparisons back up the qualitative examples in Table 2.

5. **Large, systematic multi-model evaluation with insightful analyses.**  
   Evaluating 40 models from many families (GPT, Claude, Qwen, Llama, Mistral, DeepSeek, Gemini, GLM, etc.) is nontrivial and gives the benchmark immediate practical relevance. Figure 4 clearly shows the performance range and the strong gap between closed and open models, with only claude-sonnet‑4 exceeding 60% pass@1. The easy vs hard split and the analysis that “hard” items tend to have shorter instructions but longer highlighted code are insightful for understanding failure modes. The context ablation study in Table 3 (code-only vs +highlight vs +highlight+cursor) is especially useful: it demonstrates that giving highlighted code usually helps (5/7 models) but cursor position has mixed effects, which is exactly the kind of nuance one wants from a benchmark paper.

6. **Analysis across edit categories, not just a single scalar score.**  
   The categorization into feature addition, feature modification, bug fixing, and optimization, and the subsequent per-category performance analysis, add substantial value beyond a single leaderboard. Figure 5 shows that different models have different strengths (e.g., Qwen3-coder-flash is relatively strongest on bug fixes, Claude more on feature modification), and that optimization tasks are generally under-served by current models. This makes EditBench more diagnostic and can guide future model or training-data design.

7. **Good clarity and presentation, especially figures and tables.**  
   Figures 1–5 are generally clear and well integrated with the text. For example, Figure 1’s pipeline-style visualization makes the benchmark’s setting (instruction + code + highlight + cursor → LM edit → test cases) immediately intuitive even to readers unfamiliar with modern IDE assistants. Table 2’s side-by-side comparison of EditBench vs CanItEdit/EditEval instructions within the same task categories is an effective way to communicate the shift from templated to messy real-world prompts.

## Weaknesses

1. **Limited programming-language coverage and somewhat small “core” set.**  
   While the paper emphasizes diversity, the actual *functional* coverage is restricted to two programming languages (Python and JavaScript) and a core of 109 unique problems; the 540 problem count largely comes from multilingual comment translation. Table 1 shows 2 PLs vs up to 5 in Polyglot, and Section 3.2 admits that only a subset of collected interactions were converted into harnessed problems. This weakens the generality claims to “real-world code editing” across ecosystems: for example, no statically-typed or multi-file languages (e.g., Java, C#, Go, Rust) are represented, and no build systems or frameworks outside what can be reasonably unit-tested are considered. The paper should be clearer that EditBench is currently focused on single-file Python/JS functions and smaller scripts, and that performance may not transfer to large, strongly-typed, multi-module codebases.

2. **Multilingual aspect is partially synthetic and under-analyzed.**  
   The benchmark’s advertised 5 natural languages are largely obtained by translating problem comments using GPT‑4o (Section 3.2), following HumanEval‑XL’s procedure. This is reasonable for scaling, but means that most non-English instances are not *independently collected* real usage, and the translation process itself could simplify or normalize the instructions. The paper does not analyze model performance by natural language nor check whether translations preserve the “messy” nature of original prompts. As a result, the “multilingual, in-the-wild” framing is somewhat overstated; for the non-English splits, this is closer to synthetic multilingualization of English tasks. A table similar to Table 2 with original vs translated non-English prompts, or a breakdown of pass@1 by language, would greatly clarify how much value the multilingual expansion actually adds.

3. **Potential bias from using model-generated solutions in curation.**  
   Section 3.2 notes that GPT‑4o and Claude Sonnet 3.7 were used to generate example solutions “to give insight into possible solutions” for annotators constructing tests. Since the original accepted user edits also come from some LMs, there is a real risk that test harnesses are inadvertently aligned to patterns favored by these specific models (for example, particular decomposition strategies, argument orders, or API usage), which could inflate their scores and penalize alternative-but-correct solutions. The paper states that annotators were asked to make tests generalizable, but does not provide concrete checks (e.g., percentage of problems where another human independently wrote a different correct solution that still passes all tests). A few spot statistics or failure cases where alternative valid implementations were initially rejected by tests would be helpful.

4. **Evaluation protocol for editing is not fully aligned with how IDE tools operate.**  
   The experiments use a “regenerate the entire file” strategy: the model is always asked to output the complete modified file, regardless of what was highlighted (Section 4, “Code Editing Methods”). Yet many real tools perform *span* edits (only replacing the highlighted region) or patch-style diffs. The paper partly compensates by providing highlighted spans and cursor positions, but Table 3 shows that models do not always benefit from cursor information, and no experiment tests a “patch output” protocol. This mismatch raises the question of whether the core difficulty is in edit planning vs in robust whole-file regeneration. An ablation that compares whole-file vs span-only editing, at least for a subset of models, would strengthen the claim that EditBench measures editing ability rather than robustness to file-level regeneration.

5. **Limited methodological detail on the easy/hard split and difficulty characterization.**  
   The easy/hard split is defined as problems solved by at most \(k\) models (with \(k = 20\)), but the paper only briefly mentions that this yields a roughly even split and that hard problems have shorter instructions and longer highlights. There is no figure or table quantifying these distributions, nor analysis of whether “hardness” correlates with specific libraries (which Figure 3 could be used for) or categories (beyond a short comment that categories are “roughly evenly distributed”). A more systematic characterization would help ensure that hard problems are not dominated by edge cases in test harnesses or quirks from a subset of models.

6. **Scope of comparisons to existing benchmarks is somewhat shallow.**  
   While Table 1 and the qualitative examples in Table 2 do a good job comparing prompt statistics and style, the actual correlation analysis with Polyglot, SWE-Bench, and Chatbot Arena (Section 5.2) is very light. The correlation coefficients are low-to-moderate, but the paper stops at a qualitative explanation (“code-centric I/O”, “interaction modality”, “real-world intent”) without deeper probing. For example, it would be informative to inspect concrete cases where a model scores high on Polyglot but low on EditBench, or vice versa, and tie these to specific features visible in Figure 1 (highlight, cursor, longer context). As written, the correlation discussion feels more speculative than empirical.

7. **Related work on broader code benchmarks and pretraining is slightly thin.**  
   The related work section covers many editing and evaluation datasets (HumanEval, MBPP, CanItEdit, Polyglot, SWE-Bench, etc.) but omits some influential code benchmark and pretraining efforts that are relevant for positioning this work in the larger ecosystem. For instance, CodeXGLUE, GraphCodeBERT, and unified pretraining for program understanding/generation have shaped how code benchmarks and model evaluation are structured. While this omission is not fatal, it weakens the broader contextualization of where EditBench sits among prior code-understanding benchmarks.

## Potentially Missing Related Work

1. **Lu et al., “CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation,” 2021.**  
   CodeXGLUE proposes a comprehensive benchmark suite for code understanding and generation tasks, including code completion, translation, and defect detection. It is directly relevant as a precursor to later code benchmarks and should be discussed in Section 2 as part of the “Coding Benchmarks” narrative, clarifying that EditBench adds a real-world instructed editing setting rather than replacing these tasks.

2. **Ahmad et al., “Unified Pre-training for Program Understanding and Generation,” 2021.**  
   This work presents a unified pretraining framework for both program understanding and generation, which many current code LMs implicitly build upon. It would be useful to reference this in the related work when discussing how modern LLMs acquire the abilities being evaluated by EditBench, especially in Section 2’s first paragraph.

3. **Guo et al., “GraphCodeBERT: Pre-training Code Representations with Data Flow,” 2021.**  
   GraphCodeBERT introduces code representation pretraining with explicit data-flow modeling, influencing later code models and benchmarks. Citing it in Section 2 would better situate EditBench among code-understanding benchmarks and highlight that while previous work targets structural understanding, EditBench emphasizes realistic, instruction-driven code edits.

## Questions

1. **Span vs file-level editing.**  
   Have you experimented with prompting models to *only* edit the highlighted span (or to output a diff/patch) rather than regenerating the full file? If so, how does pass@1 change, and do the context ablation trends in Table 3 still hold? If not, could you comment on why you chose full-file regeneration as the default, given that many IDE assistants work in patch mode?

2. **Bias from example solutions during test creation.**  
   Can you provide quantitative evidence that test harnesses are not overfit to the LM-generated example solutions (e.g., percentage of problems where a human wrote an alternative correct solution that passes all tests, or where different solutions from multiple models are accepted)? Any statistics here would increase confidence in benchmark soundness.

3. **Natural-language breakdown and multilingual robustness.**  
   You mention 5 natural languages and GPT‑4o translation of comments. Could you report pass@1 per language, and indicate how many items per language are *original* user-written vs translated? This would clarify whether EditBench is primarily an English benchmark with translated variants or genuinely multilingual in practice.

4. **Effect of training data contamination.**  
   Since the data is harvested from real users, some of the contexts or instructions could conceivably be present in model training logs (for models that train on public data or tool telemetry). Did you take any steps to ensure that the benchmark problems (or test harnesses) are not publicly accessible, or to detect approximate training-set leakage? Even a brief argument about why contamination is unlikely would be useful.

5. **Difficulty characterization beyond “solved by ≤k models”.**  
   Have you considered human difficulty ratings (e.g., from your annotator pool) or alternative difficulty definitions (such as requiring both top- and mid-tier models to fail) for the easy/hard split? Some discussion of why the chosen heuristic is robust, and potentially a histogram of per-problem solve rates, would help readers interpret the large easy–hard gap discussed in Section 4.1.

## Flag For Ethics Review

No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating
3: good.  
The benchmark construction and evaluation methodology are generally solid, with thoughtful curation and multiple analyses, though some aspects (e.g., bias from example solutions, multilingual expansion, and the editing protocol choice) could be probed more deeply.

## Presentation Rating
3: good.  
The paper is well organized and readable, with effective use of figures and tables, though some claims about multilinguality and difficulty would benefit from more detailed quantitative breakdowns.

## Contribution Rating
4: excellent.  
The paper provides a timely and impactful benchmark that addresses a clearly under-evaluated but practically crucial capability of LLMs, grounded in real IDE usage and accompanied by broad multi-model evaluation.

## Overall Rating
8: Accept, good paper (poster).  
The work offers a high-quality, carefully constructed benchmark that fills an important gap in evaluating LLMs for real-world instructed code editing. Despite some limitations in language coverage, multilingual construction, and depth of certain analyses, the benchmark is clearly valuable to both academia and industry, and the experimental section already yields nontrivial insights about current models’ strengths and weaknesses.

## Reviewer Confidence
4: confident.  
I am familiar with code-generation/editing benchmarks and LLM evaluation, have carefully read the methodology and results, and feel reasonably certain in this assessment, though I would welcome further details from the authors on the questions raised above.