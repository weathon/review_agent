---
job_id: 78351c28-526a-4f4b-8c43-eb6ed9083732
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: blFpxJ3A08.pdf
paper: LPFQA: A Long-Tail Professional Forum-Based Benchmark for LLMs’ Evaluation
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work proposes a benchmark dataset and evaluation protocol for LLMs, directly aligned with “datasets and benchmarks” and “representation learning for language and other modalities” within ICLR’s scope.

## Minimum Quality
Pass ✅.  
The paper contains an Abstract, Introduction, Related Work, Method / Benchmark Construction (Section 3), Statistics (Section 3.3), Experiments and Analyses (Section 4), and Conclusion. While there are issues with rigor and clarity, there are no fatal structural omissions or obvious theoretical errors that alone warrant desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
The paper includes prompts used to control LLMs *within the dataset pipeline*, but there are no instructions targeted at reviewers or attempts to manipulate the review process.

---

# Expected Review Outcome:

## Summary

The paper introduces LPFQA, a benchmark of 505 questions collected and synthesized from professional technical forums spanning 20 domains (e.g., physics, biology, finance, engineering) to evaluate LLMs on long‑tail, professional knowledge and complex reasoning. The authors describe an automated three‑phase pipeline using MLLMs/LLMs to extract questions and answers from forum screenshots, convert them into multiple‑choice or short‑answer formats, and then refine them via expert verification and empirical difficulty filtering. They evaluate 12 contemporary LLMs on LPFQA (and two filtered subsets, denoted as LPFQA\(^-\) and LPFQA\(^{=}\)), analyze per‑domain performance patterns, and run ablations involving a code interpreter and web search tools.

## Strengths

1. **Timely focus on long‑tail, professional questions.**  
   Constructing a benchmark from specialized technical forums (Section 3.2, Appendix D) targets a clearly under‑served evaluation regime: rare, fragmented knowledge that is likely under‑represented in pre‑training corpora. This goes beyond standard exam or textbook‑style questions and matches the intuitive notion of “corner cases” practitioners ask online.

2. **End‑to‑end data pipeline is clearly conceptualized.**  
   Figure 1 provides a reasonably clear, high‑level view of the pipeline from crawling discussion links, capturing screenshots, generating QA pairs with an MLLM, filtering and labeling with an LLM, to expert verification and empirical difficulty adjustment. The modular separation into data collection, automated QA generation, and expert validation phases is sensible and easy to follow.

3. **Breadth of domains and qualitative diversity.**  
   As summarized in Figure 2 and Section 3.3, LPFQA spans 20 fields, with non‑trivial coverage in Physics, Mathematics, and Biology (each >60 questions) and a tail of engineering, finance, law, and medical questions. The example Q&A in Appendix B shows that many items do demand precise, domain‑specific knowledge (e.g., Wright–Fisher generational extinction, stellar rotation determinants, or engineering power conversion setups).

4. **Multi‑dimensional evaluation intent.**  
   The benchmark is explicitly designed to touch “knowledge depth, reasoning ability, terminology comprehension, and contextual analysis” (Section 3.1), and the construction prompts (Appendix C) push for multi‑step reasoning and professional terminology. This is a useful step beyond one‑dimensional accuracy on simple trivia.

5. **Inclusion of tool‑use ablations is informative.**  
   The experiments with a code interpreter and web search tools (Section 4.2.2, Tables 3 and 4) are simple but interesting: performance mostly *decreases* when tools are added. This is a useful empirical cautionary signal that tool‑augmented reasoning is not automatically beneficial on long‑tail professional tasks and that retrieval may introduce misleading context.

6. **All results are reported across many frontier models.**  
   Tables 1 and 2 show evaluation of 12 strong, contemporary LLMs, and Figure 3 visualizes per‑domain performance for each model. Even with the benchmark modest in size, this dataset could be useful as a compact, discriminative stress test for high‑end models.

## Weaknesses

1. **Benchmark size and statistical reliability are not adequately justified.**  
   With only 505 questions across 20 domains (and further reduced to 436 in LPFQA\(^-\) and 421 in LPFQA\(^{=}\)), each field often has fewer than ~30 questions (Figure 2). Given that domain‑wise results are heavily relied upon in Figure 3 and Figure 4 for cross‑disciplinary claims, the paper never provides any uncertainty estimates (e.g., standard errors or confidence intervals) or reliability analyses. For instance, if a field has only ~10 items (e.g., Chem, Med, DS), a 1‑question difference already changes accuracy by 10 percentage points, which can completely flip the ranking in Figure 4(b–c). Without per‑field sample size annotations in these figures or any statistical testing, the claimed cross‑model rankings by domain are quite fragile.

2. **Heavy reliance on LLM/MLLM generation and judging, with limited discussion of leakage and circularity.**  
   The pipeline depends on MLLMs to parse screenshots and generate QA pairs, and then LLMs for duplicate removal, ambiguity filtering, conversion to MC/short‑answer, and later for automatic grading of short answers (Appendix C). Yet the paper does not specify *which* models are used at each step, nor whether any overlap exists between those construction models and the evaluated models (which include GPT‑4.x, GPT‑5, Gemini‑2.5‑Pro, DeepSeek variants, etc.). This raises real concerns of data contamination and circularity: if GPT‑4.1 was used in the QA construction or as the grading model, its behavior may be implicitly baked into the benchmark. The scoring prompt in Appendix C (“Prompt of metric for answer evaluation”) suggests a single LLM judge performing strict equivalence checks on final answers, but there is no equation or precise description of how multiple judges, if any, are combined or how robustness to judge bias is ensured. For a benchmark paper, the lack of transparency here seriously undermines trust in the reported numbers.

3. **Long‑tail property and difficulty hierarchy are asserted, not quantified.**  
   A central selling point is “long‑tail professional forum‑based” and “hierarchical difficulty”. However, the paper never defines or quantifies “long‑tail” in the context of language. There is no measurement of rarity (e.g., token frequency in web corpora, alignment with MMLU or other benchmark distributions, or coverage of niche tags in the source forums). Similarly, difficulty levels are only mentioned qualitatively: the authors say in Section 3.2.3 that they classify items into difficulty levels using multiple LLMs and empirical accuracy, but there is no description of (i) the mapping from accuracy bands to difficulty labels, (ii) how many levels exist, or (iii) how these levels relate to the “knowledge depth / reasoning / terminology / context” dimensions. Equations defining, for example, a difficulty score \(d_i = 1 - \frac{1}{M} \sum_{m} \mathbf{1}[\text{model}_m \text{ correct on item } i]\) or similar are absent, and Figure 2 / Figure 5 visualize only counts, not difficulty strata. The claimed “tiered difficulty structure” is therefore not substantiated.

4. **Some empirical analyses appear inconsistent or simply incorrect relative to the tables/figures.**  
   In Section 4.1 (“Overall performance”), the paper claims: “Among all evaluated systems, **DeepSeek‑V3 demonstrates the most balanced and consistent performance across disciplines, with no apparent weaknesses, and can thus be regarded as the overall best‑performing model. GPT‑5 exhibits strong competitiveness...**”. This seems plainly contradicted by **Table 1** and **Table 2**, where DeepSeek‑V3 has one of the *lowest* overall scores (32.60 on LPFQA; 37.54 / 35.59 on the filtered sets), and GPT‑5 consistently obtains the top score. Moreover, in Figure 3, the radar chart for DeepSeek‑V3 visually shows large depressions in multiple fields (e.g., Misc, Eng), indicating clear weaknesses. This misreading of their own results raises concerns about the care taken in the analysis and undermines confidence in other qualitative claims derived from the figures.

5. **Key evaluation details are missing or under‑specified, especially for short‑answer grading.**  
   While the authors include the grading prompt in Appendix C, they never clarify in the main text how multiple‑choice vs short‑answer items are mixed in computing the “Score” reported in Tables 1–4. Questions include multi‑label MC items (e.g., Q&A 2 in Appendix B) and short answers with multiple “core knowledge points”. Yet there is no formal metric definition (e.g., per‑question score \(s_i \in \{0,1\}\) with exact match on option sets or judge‑based acceptance of free text) or explanation of how multi‑label correctness is checked. The grading prompt mandates *exact agreement* with the reference answer and evaluation points (Step 4), but does not cover approximate matches, partially correct lists, or answers expressed in equivalent but different wording. Without a clear mathematical description of the scoring function \(S(\text{prediction}, \text{reference}, \text{eval\_points})\), it is hard to interpret the absolute accuracy numbers and to know whether scores are more about surface form matching than substantive correctness.

6. **Limited and somewhat confusing ablation interpretation.**  
   In Table 3 (code interpreter) and Table 4 (search), many models’ scores change only a few percentage points. The paper averages “scores” but the Δ column is reported in percentages without specifying whether it is absolute or relative; given that the base scores in Table 1 are already ~30–47, the entries like “2.89%↓” are ambiguous and not derived from any presented equation. Moreover, the conclusion that “LPFQA primarily reflects domain knowledge rather than reasoning ability” solely from CI not helping is overstated. Tool integration quality (prompting, budget, environment) is a huge confounder; there is no control experiment showing that code interpreter *does* help on a reasoning‑focused benchmark within this same evaluation framework.

7. **Presentation issues and inconsistencies materially hurt clarity.**  
   - **Table 2**’s column headers are “LPFQA” and “LPFQA ”, apparently missing the superscripts “\(^-\)” and “\(^{=}\)”, which forces the reader to infer which column is which from the text.  
   - Section 4.2.1 refers to Figure 5 for both LPFQA\(^-\) and LPFQA\(^{=}\), but Figure 5 (as shown) overlays both distributions in a bar chart without a clear legend explaining which color corresponds to which subset beyond “LPFQA\(^-\)” and “LPFQA\(^{=}\)”; the caption simply repeats “Quality distribution of each field in filtered LPFQA.”  
   - The multi‑label MC example in Appendix B (Q&A 2) uses 4 correct options, while the “Prompt for multi‑choice” in Appendix C explicitly requires “exactly one correct answer”. This contradiction calls into question how strictly the construction guidelines were followed.  
   - Throughout Section 4.1, the text refers to “Figures 3 and 4” but the in‑text lettering (“Figure 4a”, “4b”, “4c”) is cramped and the mapping to subplots in images **img‑4.jpeg** and **img‑5.jpeg** is not immediately obvious. The narrative should directly point to specific subfigures (e.g., “Figure 4(a) Average scores”).

8. **No comparison or correlation with existing benchmarks, despite claiming complementary properties.**  
   The related work section extensively discusses MMLU, BIG‑bench, HELM, Arena‑Hard, HLE, etc., but the experimental section never attempts to correlate LPFQA performance with any standard benchmark for the overlapping set of models. Even a simple scatter plot of LPFQA score vs MMLU/MMLU‑Pro or Arena‑Hard rank for a few models would help substantiate the claim that LPFQA captures different capabilities (e.g., deeper professional long‑tail knowledge). Without such analysis, LPFQA might just be another small, noisy QA set whose marginal benefit over existing resources is unclear.

9. **Domain coverage is wide but shallow, and the “interdisciplinary” claim is overstated.**  
   While it is true that questions are sourced from multiple professional forums, the dataset still treats each question as belonging to *one* field (CS, Bio, Phys, etc.). There is no explicit construction of cross‑disciplinary items where, say, physics and finance must be integrated. The prompts *suggest* cross‑disciplinary reasoning, but the only concrete examples in Appendix B are mostly single‑domain. Also, some fields (e.g., Data Science and Big Data Technology with only 3 items, En with 9) have such minimal coverage that domain‑wise claims there are not very meaningful.

10. **Math / notation clarity issues around filtered sets.**  
    The notation LPFQA\(^-\) and LPFQA\(^{=}\) is introduced in Section 4.2.1, but the set relationships are not clearly formalized. The text states that from 505 items they drop those that all models miss, obtaining 436 items (LPFQA\(^-\)), and then “based on LPFQA\(^-\)” they drop those all models solve, obtaining 421 items (LPFQA\(^{=}\)). A simple set‑notation equation such as  
    \[
    \text{LPFQA}^- = \{q \in \text{LPFQA} : \exists m, \text{model}_m \text{ correct on } q\}
    \]
    \[
    \text{LPFQA}^{=} = \{q \in \text{LPFQA}^- : \exists m_1, m_2, \text{model}_{m_1} \text{ correct on } q, \text{model}_{m_2} \text{ incorrect on } q\}
    \]
    would eliminate ambiguity, but no such explicit definitions are given. This becomes more problematic when interpreting Table 2, where both columns are simply labeled “LPFQA”, and it is unclear which scores correspond to which filtered set without cross‑referencing the text.

Overall, while the idea of a professional‑forum‑based long‑tail benchmark is quite appealing, the current paper leaves too many methodological and analytical gaps to be fully convincing as a robust, reference‑quality benchmark for the community.

## Potentially Missing Related Work

1. **Kojima et al., “Large Language Models are Zero‑Shot Reasoners”, 2022.**  
   Directly relevant to the paper’s focus on evaluating reasoning abilities of LLMs across tasks. Should be discussed in Section 2 when contrasting LPFQA’s complex reasoning evaluation with instruction‑prompting approaches for zero‑shot reasoning, and potentially referenced in the ablation section when interpreting reasoning vs knowledge.

2. **Zhong et al., “Factual Probing Is [MASK]: Learning vs. Learning to Recall”, 2021.**  
   This paper explicitly probes factual knowledge recall of language models, closely aligned with LPFQA’s emphasis on specialized, long‑tail factual knowledge. It should be cited in Section 2.1 and referenced when discussing that LPFQA targets “knowledge depth” versus generic factual probing.

3. **Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text‑to‑Text Transformer” (T5), 2020.**  
   A foundational work on large text‑to‑text models evaluated over many tasks. Given that LPFQA evaluates a broad set of LLMs in a similar spirit, this work should be mentioned in Related Work as part of the landscape of multi‑task benchmarks and large‑scale model evaluations.

4. **Brown et al., “Language Models are Few‑Shot Learners” (GPT‑3), 2020.**  
   Since the benchmark evaluates models in a near zero‑shot setting (no task‑specific training) and focuses on general‑purpose LLM capabilities, this paper is foundational and should be cited in the Introduction or Related Work when discussing the evolution of LLMs and their evaluation paradigms.

5. **Kiela et al., “Dynabench: Rethinking Benchmarking in NLP”, 2021.**  
   Dynabench proposes a dynamic, user‑in‑the‑loop benchmarking platform that explicitly aims to capture difficult, adversarial, and realistic examples, conceptually similar to collecting challenging questions from professional forums. It should be discussed in Section 2.2 as an alternative “user‑centric” benchmark, with a contrast drawn between dynamic vs static benchmarks.

6. **Zellers et al., “HellaSwag: Can a Machine Really Finish Your Sentence?”, 2019.**  
   HellaSwag is an adversarial commonsense reasoning benchmark with carefully crafted distractors. Given that LPFQA also emphasizes carefully designed multiple‑choice distractors (Appendix C), this work should be referenced when discussing prior work on challenging MC benchmarks targeting reasoning.

7. **Lin et al., “NumerSense: Probing Numerical Commonsense Knowledge of Pre‑trained Language Models”, 2020.**  
   NumerSense systematically probes a specific aspect of knowledge (numerical commonsense). LPFQA likewise targets specialized knowledge in narrow domains. It would be appropriate to mention this work in Section 2.1 when framing LPFQA as another specialized probing benchmark and to possibly compare the notion of “long‑tail knowledge” with “numerical commonsense”.

## Questions

1. **LLM / MLLM usage specifics.**  
   Which exact models (and versions) were used for (a) screenshot understanding and QA generation, (b) QA quality filtering, (c) MC/short‑answer conversion, and (d) automatic grading of short answers? Are any of these models (or their close variants) included among the 12 evaluated models? If yes, please quantify potential overlap and comment on contamination risks.

2. **Scoring function formalization.**  
   Can you provide a precise mathematical definition of the scoring function used to compute each model’s “Score” in Tables 1–4, explicitly handling: single‑label MC, multi‑label MC (like Q&A 2), short‑answer questions with multiple core knowledge points, and the LLM‑judge decisions? A concise equation or pseudocode would address a major ambiguity.

3. **Difficulty levels and long‑tail quantification.**  
   How exactly are difficulty levels defined and assigned based on LLM accuracy during empirical testing? Can you provide: (i) explicit thresholds used to split items into easy/medium/hard (or finer levels), (ii) distributions of items across levels per domain, and (iii) any quantitative evidence that LPFQA items are indeed drawn from the long tail (e.g., frequency of key terms in a large web corpus or comparison to MMLU distributions)?

4. **Inconsistency around DeepSeek‑V3 vs GPT‑5.**  
   The text in Section 4.1 currently describes DeepSeek‑V3 as “overall best‑performing” despite the tables/figures showing GPT‑5 as the top performer by a clear margin. Was this a mistake, or is there another metric (e.g., variance, worst‑case performance) on which DeepSeek‑V3 is superior? Please clarify and, if it is an error, correct the analysis.

5. **Correlation with existing benchmarks.**  
   Have you computed correlations (e.g., Pearson or Spearman) between LPFQA scores and other standard benchmarks such as MMLU, MMLU‑Pro, Arena‑Hard, or HLE for the overlapping set of models? Even partial results (for a subset of models) would greatly strengthen the argument that LPFQA captures complementary aspects of model competence.

6. **Reproducibility of tool‑use ablations.**  
   For the code interpreter and search tool experiments, what prompts, temperature, tool‑call budgets, and timeout settings were used? Were these tuned for any model or kept fixed across all? Clarifying this would help interpret whether the negative Δ in Tables 3 and 4 reflects intrinsic mismatch between tools and long‑tail tasks, or simply suboptimal agent prompt design.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The benchmark idea is interesting and the pipeline is plausible, but several core experimental and methodological aspects (judge model details, scoring function, difficulty definition, misinterpretation of results) are underspecified or inconsistent, limiting confidence in the conclusions.

## Presentation Rating

2: fair.  
The high‑level narrative is understandable and figures help, particularly Figure 1 and Figure 3, but multiple inconsistencies (e.g., mislabeling in Table 2, contradictory statements about DeepSeek‑V3, conflicting constraints on MC questions) and missing formal definitions make the paper harder to follow than it should be.

## Contribution Rating

2: fair.  
A specialized, professional‑forum‑based benchmark is potentially valuable, but the dataset is small, empirical analysis is limited, and key claims about long‑tail properties and difficulty hierarchy are only qualitatively supported. The contribution is incremental compared to existing challenging QA benchmarks and dynamic evaluation platforms.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper tackles an important and timely problem with a conceptually appealing benchmark and provides some useful analyses and ablations. However, methodological opacity around LLM usage and grading, lack of quantitative evidence for the “long‑tail” and “hierarchical difficulty” claims, inconsistencies between text and figures/tables, and limited statistical rigor collectively keep it below the bar I would expect for an ICLR benchmark paper. A revision that clarifies these aspects, strengthens empirical validation, and cleans up the presentation could make this a stronger candidate in the future.

## Reviewer Confidence

4: confident.  
I am familiar with LLM evaluation and benchmarking work and carefully examined the methodology, tables, and figures, though some construction details depend on unreleased implementation specifics that I cannot fully verify.