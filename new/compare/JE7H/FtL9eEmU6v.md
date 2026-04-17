---
job_id: da845004-4f22-4197-8732-10e7575f6143
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: FtL9eEmU6v.pdf
paper: EditBench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work introduces a code-editing benchmark for LLMs, squarely within “datasets and benchmarks” and “representation learning for code / language” topics listed in the ICLR CFP.

## Minimum Quality
Pass ✅.  
The paper is in English and contains Abstract, Introduction, Related Work, Benchmark Construction / Methodology (Sections 3–4), Evaluation & Results (Section 5, plus Appendix E), and Conclusion / Limitations (Section 6 & F). Methods and experiments are technically sound at a high level, with nontrivial analysis and clear empirical evaluation.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts to manipulate LLM reviewers or hidden prompts in the text.

---

# Expected Review Outcome:

## Summary

The paper introduces **EditBench**, a benchmark for evaluating LLMs on *in-the-wild instructed code edits*. The authors collect real user instructions and code contexts via a VS Code extension, then curate 109 core Python/JS problems with human-written test harnesses and translate them into 5 natural languages, yielding 540 tasks. They evaluate 40 LLMs, analyze performance across problem difficulty, contextual information (highlighted code, cursor), and edit categories, and show that EditBench is challenging, exhibits substantial variation in model behavior, and correlates only weakly with existing edit or chat benchmarks.

## Strengths

1. **Clear gap and strong motivation, well supported by data.**  
   The paper convincingly identifies a missing piece in current coding benchmarks: realistic *instructed edit* tasks grounded in real IDE workflows rather than educational or synthetic prompts. The comparisons in **Table 1** (problem source, natural / programming languages, context lengths, highlight support) and the qualitative contrasts in **Table 2 / Table 5** between EditBench instructions and those in CanItEdit / EditEval make this gap concrete and easy to understand.

2. **Careful, multi-stage benchmark construction based on in-the-wild data.**  
   The data pipeline is thoughtfully designed: a VS Code extension for live collection (Section 3.1, **Figure 2**), privacy controls (Appendix A), filtering out trivial/ambiguous tasks with explicit examples (Section C), and a two-annotator process for test harness creation (Section 3.3). The explanation of why highlighted code and cursor are crucial to disambiguate messy user instructions is persuasive, and the “remove” example in Appendix E nicely illustrates why these context signals matter.

3. **Substantial empirical study across many models and conditions.**  
   Evaluating **40** diverse LLMs (Table 6) is already nontrivial. The authors go beyond a flat leaderboard:  
   - Overall performance distribution and open vs closed trends (**Figure 4**).  
   - Ablation over context variants (code only, +highlight, +cursor) in **Table 3**, which concretely shows that highlighted code usually helps while cursor position has inconsistent effects.  
   - Analysis by context length (**Table 7**), difficulty split (easy vs hard via number of models solving, Section 5.1, Table 8), and edit category performance (**Figure 5**).  
   This makes the benchmark immediately useful for diagnosing model behavior rather than just ranking them.

4. **Realistic, diverse code and instruction characteristics.**  
   The distribution plots in Appendix B support the claim that EditBench is closer to real practice:  
   - **Figure 6** and **Figure 7** show the variety of programming and natural languages in the raw extension data.  
   - **Figure 8–10** show long-tailed distributions for highlighted span length, instruction length, and file context length (with many long contexts and short, underspecified instructions).  
   - **Figure 3** and **Figure 12** demonstrate EditBench’s larger variety of Python libraries (74 imports vs 15–25 in other benchmarks), hinting at more realistic, application-level code.  
   Together, these visuals give a good empirical justification for the “real-world” claim.

5. **Insightful evaluation findings relevant to practitioners.**  
   Several results have practical implications for building coding assistants:  
   - Closed models still dominate, but some open models (e.g., glm-4.5, kimi-k2, deepseek-chat-v3.1) are competitive (**Figure 4**).  
   - Performance strongly depends on edit category: bug fixing is relatively easy, feature addition and optimization are tougher (**Figure 5**).  
   - Even among strong models, the easy/hard split leads to average gaps of nearly 60 percentage points, and hard problems are characterized by very short instructions but nontrivial highlighted code (Table 8).  
   These observations are likely to inform both training data design and product decisions for IDE-integrated LLMs.

6. **Ethics and privacy handled reasonably for a code benchmark.**  
   The paper describes IRB approval, user-configurable privacy modes, PII screening by annotators, and a conservative stance on data release (Appendix A, F.3). For a benchmark built from live user code, this level of attention is important and reassuring.

## Weaknesses

1. **Limited coverage of programming languages and interaction patterns.**  
   Despite a broad raw data distribution (**Figure 6**), EditBench-core ends up with 104 Python and 9 JavaScript problems (Section C). Other languages visible in the telemetry (PHP, HTML, Java, etc.) are dropped. This strongly biases the benchmark toward Python-centric back-end / ML workflows and a small subset of JS (5/9 problems are React). The conclusions about “real-world instructed edits” are therefore somewhat narrow; a model tuned specifically for Python might look more capable than one that is stronger on, say, Java / C# or modern TypeScript tooling. The paper acknowledges this in Section 6, but the abstract and introduction somewhat oversell cross-language generality.

2. **Translation-based “multi-linguality” is shallow and may not reflect real mixed-language edits.**  
   EditBench-complete is created by translating problem comments into other natural languages using GPT-4o (Section 3.2). This means that non-English variants are *synthetic*, not separate real prompts authored in those languages. The code itself remains largely unchanged, and the kinds of messy multilingual phenomena that arise in real coding (mixed-language comments, variable names, error messages, etc.) are not captured. While the authors do some spot-checking with native speakers, they do not analyze whether translation induces artifacts (e.g., unnatural phrasing or over-specification) that differ from original prompts. As a result, the claim in the abstract that the benchmark “comprises of 5 natural languages” risks being interpreted more strongly than what is actually provided.

3. **Evaluation metric and problem difficulty definition are somewhat under-specified and could use more rigor.**  
   The paper adopts pass@1 from prior coding benchmarks, but does not deeply discuss its suitability for *editing* scenarios where multiple implementations might satisfy the test harness but differ greatly in edit locality or adherence to the user’s instruction. Moreover:  
   - The easy/hard split is defined as “solved by ≤ k models” with k=20 (Section 5), yet there is no sensitivity analysis to k, nor is there a discussion of potential circularity: problem hardness is evaluated via the same models that are then analyzed by hardness.  
   - There is no explicit mathematical formulation of the scoring function or formal description of the sampling process (e.g., are results averaged over multiple seeds? Only temperature 0 with one sample). This is mostly clear from text, but a compact “Evaluation Protocol” box with a precise definition would avoid ambiguities about model stochasticity and make reproducing scores easier.

4. **Lack of more granular error analysis and qualitative inspection of failures.**  
   The results section remains fairly high level. We see category-level averages (**Figure 5**), correlation to other leaderboards, and length-based strata (Tables 7–8), but it stops short of detailed error modes. For example:  
   - How often do models misunderstand the *instruction* vs fail on code-level reasoning vs break formatting / imports?  
   - Are failures on optimization tasks usually due to incorrect asymptotic complexity, micro-optimizations that break semantics, or not touching the right region of code?  
   - For editing-specific pathology, how often do models modify untouched parts of the file despite prompts that say “only change highlighted section”?  
   A handful of representative failure cases (similar to the “remove” example, but for each category) would make the benchmark more actionable for model developers and would also sanity-check that tests actually reflect instruction adherence rather than just “one particular solution”.

5. **Test harness construction process is only partially validated, and potential label noise is not quantified.**  
   Section 3.3 explains that annotators use user instruction, full code, highlight, and cursor to design tests, and that they drop ambiguous problems. However, the paper does not offer:  
   - Any inter-annotator agreement statistics or cross-checking beyond “second annotator review”.  
   - Any empirical sanity checks comparing the original user-accepted edits to the benchmark’s test oracle (e.g., replaying the user-accepted model output to ensure it passes).  
   - Any estimate of test flakiness or false negatives (correct edits that still fail some tests because the harness encodes a narrower interpretation than users intended).  
   For a dataset that heavily leans on human interpretation of messy instructions, some measurement of reliability would strengthen confidence that pass@1 is a meaningful ground truth rather than just “agreement with one annotator’s reading”.

6. **Context ablation experiments raise hypotheses but are not fully explored.**  
   **Table 3** shows nontrivial and sometimes counter-intuitive behavior when toggling highlighted code and cursor position (e.g., glm-4.6 collapsing from 56.48% to 44.81% when both highlight and cursor are present, qwen3-coder unaffected or worsened by extra context, and o3-mini degrading with highlight). However, the paper does not probe *why*:  
   - Are prompts for cursor / highlight information poorly phrased for some models?  
   - Does the extra markup exceed models’ effective context windows causing truncation on long files?  
   - Are some models overfitting to the highlighted span and ignoring needed surrounding lines?  
   Given that “context-dependent problems” are a core selling point of EditBench (Figure 1), a deeper analysis here would significantly add to the paper’s insights.

7. **Some claims about benchmark uniqueness relative to prior edit work could be sharpened with more systematic comparisons.**  
   The related work section briefly mentions CodeEditorBench, InstructCoder, CanItEdit, EditEval, and Polyglot. **Table 1** includes basic statistics for three of them. However:  
   - There is no direct *empirical* comparison on overlapping model sets for all of those benchmarks (only Polyglot and Chatbot Arena are used in Section 5.2).  
   - The discussion that EditBench is only weakly correlated with Polyglot and Chatbot Arena is suggestive but does not fully unpack *what* capabilities are differently exercised. For instance, how many of EditBench’s hard problems are solved in Polyglot but not vice versa?  
   More systematic cross-benchmark analysis would clarify whether EditBench is complementary mainly because of “messier instructions”, longer contexts, different libraries, etc., versus differences in test construction.

8. **Minor clarity / math issues and formatting friction.**  
   A few small points that, while not fatal, could be improved:  
   - Percentages and correlation numbers are often given without clear confidence intervals or test statistics (Section 5.2 mentions p-values for Pearson r only once).  
   - The definition of functional categories (feature addition, modification, fixing, optimization) is described qualitatively (Section 4), but no clear annotation protocol or resolving of ambiguous multi-category cases is described.  
   - The example code in **Figure 11** has several typos / syntax issues (e.g., load-detenv, get relevant_documents, DIVIDER = "-"10), which may confuse readers, even though they are presumably anonymized or simplified.

Overall, these weaknesses point to missing depth in analysis and some limitations in benchmark scope, rather than fundamental flaws in the core idea or implementation.

## Potentially Missing Related Work

N/A.  
The paper cites the most directly relevant recent code-edit and real-world-code benchmarks (CanItEdit, CodeEditorBench, InstructCoder, SWE-Bench and variants, SWT-Bench, InterCode, LiveCodeBench, Aider Polyglot), as well as several program repair and code-editing LLM works. I do not identify any glaringly missing, clearly-on-point prior work that is unmentioned.

## Questions

1. **Reliability of test harnesses.**  
   - Did you verify that the original user-accepted edit (from the extension) passes your final test cases for that problem? If so, what fraction of problems have at least one such “known good” solution that passes?  
   - Have you observed any cases where multiple plausible edits (e.g., behavior-preserving refactors) are rejected by the harness?

2. **Edit locality and instruction adherence.**  
   - Currently pass@1 is purely test-based. Did you consider incorporating constraints on the fraction of lines changed or on modifying only the highlighted region (e.g., via diff analysis) to penalize models that rewrite the entire file unnecessarily? If so, what prevented you from including such metrics?  
   - Do you see many cases where a model passes tests but clearly violates the instruction (e.g., ignoring a performance constraint while still satisfying functional tests)?

3. **Context ablation behavior.**  
   - For glm-4.6 and o3-mini, why does adding highlight/cursor information seem harmful in **Table 3**? Can you provide any quantitative evidence (e.g., context length histograms conditioned on these prompts) or qualitative examples illustrating failure modes?  
   - Would a model-specific prompt tuning (e.g., different formatting for highlights for models trained with editor-style prefixes) change these conclusions?

4. **Non-English performance.**  
   - Do you have per-language breakdowns of pass@1 on translated tasks (e.g., English vs Spanish vs Chinese)? Are there significant drops, or is performance similar across languages?  
   - Given that both instructions and comments are translated, did you observe any systematic translation artifacts (e.g., length inflation, more explicit instructions) that might make non-English variants easier/harder?

5. **Benchmark evolution and contamination.**  
   - You mention in F.1/F.2 that you plan to continuously expand EditBench. Do you intend to version tasks and leaderboards to distinguish “old” problems that may become contaminated from new ones?  
   - Have you considered any automated contamination checks similar to those in LiveBench / LiveCodeBench, given that some of the raw user code is likely already public on GitHub?

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The benchmark construction, evaluation protocol, and empirical analysis are methodologically sound overall, with well-chosen models and careful context ablations. Some aspects (test harness reliability, depth of failure analysis, and metric discussion) could be more rigorously justified, but there are no obvious fatal flaws.

## Presentation Rating

3: good.  
The paper is generally well written and organized, with informative figures and tables (notably **Figure 1–5**, **Table 1–3,7–8**). A few sections could use tighter mathematical specification and more precise framing of multilingual claims, but the main ideas are clear.

## Contribution Rating

3: good.  
The work makes a meaningful and timely contribution by introducing a realistic code-editing benchmark, backed by a substantial in-the-wild dataset and extensive evaluation. The conceptual innovation is more on the data / evaluation side than on algorithms, but the benchmark fills a real gap and will likely be widely used.

## Overall Rating

8: Accept, good paper (poster).  
The benchmark is thoughtfully constructed, empirically grounded, and already provides nontrivial insights into LLM code-editing behavior. While there are limitations in language coverage and room for deeper analysis, the positives clearly outweigh the negatives, and the work is well aligned with ICLR’s interest in realistic, robust evaluation of large models.

## Reviewer Confidence

4: confident.  
I am familiar with LLM code benchmarks and editing tasks, and I carefully examined the methodology, equations/definitions (e.g., hardness split via k, pass@1 setup), and experimental design. Some implementation details of test construction and translation cannot be fully verified from the text alone, but I am reasonably confident in my assessment.