# RefineBench: Evaluating Refinement Capability of Language Models via Checklists

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Can language models (LMs) self-refine their own responses? This question is increasingly relevant as a wide range of real-world user interactions involve refinement requests. However, prior studies have largely tested LMs' refinement abilities on verifiable tasks such as competition math or symbolic reasoning with simplified scaffolds, whereas users often pose open-ended queries and provide varying degrees of feedback on what they desire. The recent advent of reasoning models that exhibit self-reflection patterns in their chains-of-thought further motivates this question. To analyze this, we introduce RefineBench, a benchmark of 1,000 challenging problems across 11 domains paired with a checklist-based evaluation framework. We evaluate two refinement modes: (1) guided refinement, where an LM is provided natural language feedback, and (2) self-refinement, where LMs attempt to improve without guidance. In the self-refinement setting, even frontier LMs such as Gemini 2.5 Pro and GPT-5 achieve modest baseline scores of 31.3% and 29.1%, respectively, and most models fail to consistently improve across iterations (e.g., Gemini-2.5-Pro gains only +1.8%, while DeepSeek-R1 declines by –0.1%). By contrast, in guided refinement, both proprietary LMs and large open-weight LMs (>70B) can leverage targeted feedback to refine responses to near-perfect levels within five turns. These findings suggest that frontier LMs require breakthroughs to self-refine their incorrect responses, and that RefineBench provides a valuable testbed for tracking progress.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces RefineBench, a new benchmark with 1,002 problems across 11 domains designed to evaluate the refinement capabilities of Large Language Models. It uses a novel checklist-based framework to test two modes:
1. Self-Refinement (no feedback, $f_t = \emptyset$)
2. Guided Refinement (with feedback $f_t$)

The primary contribution is the finding that even frontier LMs like Gemini 2.5 Pro struggle significantly with self-refinement, showing minimal gains (e.g., +1.8%) across iterations. However, in guided refinement, these same models effectively use targeted feedback to achieve near-perfect scores. This suggests LMs possess refinement abilities but lack the direction of what to fix.

### Strengths
1. This paper introduces RefineBench, a high-quality benchmark for evaluating refinement in complex, non-verifiable domains like law and humanities, moving beyond simple math problems.
2. The quality of this benchmark is very high, using real-world problems and a novel checklist-based evaluation framework that was rigorously validated by Ph.D. domain experts (96.1% appropriateness).
3. The authors identify a key bottleneck: LMs can improve with feedback but are fundamentally poor at self-refining because they cannot identify their own errors.

### Weaknesses
1. The "self-refinement" failure mode is not precisely identified. The paper concludes models "lack direction on what to fix". However, the evidence (e.g., Figure 6) suggests the model failed to identify that a problem existed at all. This is a failure of self-verification or error-detection. Models aren't necessarily unable to fix errors, but rather they incorrectly conclude their initial answer is already "complete and correct"  and thus stop trying to refine.
2. The "Guided Refinement" setting likely overstates true refinement capability by effectively testing instruction-following. The feedback provided is not a realistic, high-level critique. Instead, it's a list of explicit, atomized commands derived directly from the failed checklist items (e.g., "The response should accurately..." in Appendix K), which is a much simpler task for LMs.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents RefineBench, a benchmark of 1002 problems across 11 domains. Each problem includes a checklist to help evaluators assess LLM responses consistently and accurately. The dataset combines verifiable STEM tasks with non-verifiable, free-form tasks. Experiments show that guided refinement enables most LLMs to reach correct answers after multiple turns, whereas self-refinement does not achieve comparable gains.

### Strengths
- The paper introduces a new benchmark with a relatively large problem set and clear per-problem checklists, enabling more reliable evaluation of LLMs’ reasoning abilities.

- The analyses are clear and highlight that self-refinement remains challenging, particularly due to LLMs’ difficulty in identifying specific errors and determining how to adjust initial answers.

### Weaknesses
- The study uses GPT-4.1 as the sole evaluator, which may introduce bias. Incorporating a second independent LLM-as-judge or human auditing would strengthen the evaluation.

- For problems that originally include images, textual descriptions may omit important details. Expanding the benchmark to a multimodal setting would address this limitation.

### Questions
1. The paper finds a key bottleneck is that LLMs struggle to pinpoint detailed issues or the direction of correction from the initial response. Do the authors have insights or proposals on how to improve this?

2. The paper notes that many existing benchmarks focus on math or symbolic reasoning rather than open-ended questions, yet RefineBench still contains a moderate share of math and math-like tasks. How are truly open-ended tasks represented, and could their proportion be increased?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RefineBench, a benchmark aimed at probing whether LLMs can perform self-refinement, either independently or with external guidance. In this context, external guidance refers to evaluation checklist items that the model previously failed on. The authors evaluate 37 LLMs and highlight several insights: (1) LLMs generally struggle to refine their own responses, though thinking models show slightly better self-refinement; (2) LLMs can refine themselves when given failed checklist items, but still fail to address issues that are not explicitly pointed out in partially guided setups.

### Strengths
1. The benchmark is well-curated, covering a wide range of topics and domains. The manual quality control process also seems solid.
2. The evaluation spans a large number of models, showing a commendable level of comprehensiveness.
3. The findings are interesting - especially the comparison between thinking models and standard ones. As the paper notes, whether refinement itself is beneficial has been extensively studied and debated in prior work, but revisiting this question in the context of reasoning models is valuable.

### Weaknesses
1. I have concerns about using the same checklist for both external guidance and evaluation. Could this create potential leakage, where models optimize for missing checklist items instead of genuinely improving quality? It's unclear whether the provided guidance leads to real improvement or just better checklist completion.
2. Discussion of related work is strangely organized. The CriticBench line of work seems most relevant and should probably be introduced earlier in Section 2. In contrast, the part on multi-turn benchmarks feels less directly connected and can be toned down.

### Questions
The analysis on whether test-time scaling helps refinement is intriguing. However, it's limited to Gemini-2.5-Pro in Figure 5. It would be great to expand the analysis and see whether similar trends hold across other reasoning models.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This dataset presents a multi-topic refinement benchmark with university-level problems and evaluation checklists. The paper presents the dataset curation process and evaluation process, as well as extensive experiments with LLMS. This is a fantastic paper, with one significant let down -- there is no human evaluation/verification of the evaluation pipeline -- which really weakens the results.

### Strengths
- Very interesting problem.
- The dataset is a significant contribution. 
- Human evaluation of checklist generation.
- Summary statistics of the dataset and comparison to other benchmarks are included and well-presented.
- Extensive experiments.

### Weaknesses
I really only found one letdown in this paper, but it is a big one --  there is no human evaluation/verification of the evaluation pipeline. To be strong, there needs to be a human verification of a sample of the end-to-end evaluation process. How do we know how good the LLMs are at comparing the answer to the checklist and providing good feedback? This is instrumental to understanding the results. 

I also think that a good baseline would have been to compare the success with human feedback (desirable).

### Questions
(line 190 -- step 3) -- If this was manually reviewed by the authors, why did you need LLMs to create the checklists from reference answers? 

-- Why did you not have human verification of the eval pipeline, and can it be reasonably added?

### Soundness
2

### Presentation
3

### Contribution
3
