# KEIC: A Framework and Dataset to Self-Correcting Large Language Models in Conversations

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Large language models (LLMs) are adept at generating coherent and fluent responses within conversational contexts. Recent studies also demonstrate that LLMs can follow the user preference in an extremely long-term setting. Nevertheless, there is still lack of comprehensive research exploring LLMs to dynamically update their knowledge in response to corrections of misinformation provided by users during dialogue sessions. In this paper, we present a unified framework termed Knowledge Editing In Conversation (KEIC), along with a 1,781 human-annotated dataset, devised to assess the efficacy of LLMs in aligning the user update in an in-context setting, wherein the previous chat containing a false statement that conflicts with the subsequent user update. Through systematic investigations on more than 25 LLMs using various prompting and retrieval-augmented generation (RAG) methods, we observe that the contemporary LLMs exhibit a modicum of proficiency in this task. To enhance their self-correction abilities, we propose a structured strategy to handle the information update in a multi-turn conversation. We demonstrate that our approach is effective and suggest insights for research communities in this emerging and essential issue.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work focuses on a question in the context of multi-turn conversations, that is *if LLMs can dynamically update their knowledge in response to corrections of misinformation provided by users.*. This task is also formulated as `Knowledge Editing In Context (KEIC)`.

Under this topics, this work has made three contributions:

1. This work proposes a unified framework `KEIC` to decomposes dialogues into four disjoint phases, which can standardize the evaluation.

2.Derived from the CoQA dataset, this work has developed a human-annotated dataset of 1781 instances.

3. This work has designed four methods for simulating the user corrections in LLMs.

4. Experiments on massive LLMs have shown the effectiveness of the proposed method.

### Strengths
1. The proposed KEIC framework can well formalize the in-context knowledge editing task for LLMs in conversations, which allows us to better process the dynamic knowledge updating. Especially the decomposition of different types of utterances.

2. Constructs a high-quality 1,781-instance human-annotated dataset  with clear definitions (such as  "effective new fact" ).

3. Four practical and model-agnostic correction methods are proposed (OTC, Verification, Reiteration, Deletion), which can be adopted in various scenarios.

4. Extensive experiments are conducted and a set of key insights are also investigated.  This research can help the future studies in the community.

### Weaknesses
1. The dataset is only limited to  YN (Yes or No) questions and is also limited to some specific domains. The generalizability to more diverse real conversational scenarios has not been well-proven.
﻿
2. The major contribution is the task definition, dataset definition (sec 2). However, the novelty of the methodology part is limited.  Such methods use some pre-defined prompts, which may show some biases. Meanwhile, it seems like the `Deletion` is very costly.
﻿
3. The baseline setting is not very strong. Thus, the experiments can show the positive effectiveness but is hard t show how well it is.

### Questions
1. It would be better to discuss the application of Sec 2.2 more.

2. Please summarize the most highlighted findings of this paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper formalizes Knowledge Editing In Conversation (KEIC)—updating misinformation within an ongoing dialogue without changing parameters—and introduces a unified framework plus a 1,781-instance human-annotated dataset (CoQA-derived, Yes/No). It benchmarks 25+ LLMs with prompting/RAG and proposes four model-agnostic strategies (including Reiteration and an iterative external correction). Results indicate modest KEIC proficiency overall and report improvements from structured strategies on subsets of models/tasks, with evidence primarily based on the released dataset and analyses centered on GPT-3.5. The work frames an emerging problem and offers initial methodology and empirical observations rather than a definitive solution.

### Strengths
1. Clear definition of KEIC as non-parametric conversational knowledge updating, and a four-phase decomposition that makes the task structured and testable.
2. A 1,781-instance human-annotated dataset with minimally edited new facts provides a practical foundation for controlled evaluation and future benchmarks.
3. The work proposes model-agnostic strategies with graded intervention strength, including OTC, Verification, Reiteration, and Deletion, combining prompting with NLI/RAG components while considering cost and practicality.
4. The study addresses a capability of practical importance for long-horizon dialogue and offers actionable guidance such as favoring explicit negation, placing updates near the test turn, and using reiteration for greater stability.

### Weaknesses
1. The paper lacks dedicated Related Work and Limitations sections in the main text and instead places them in the appendix, which is misaligned with ICLR standards and makes it difficult to assess novelty, positioning, and scope; a concise comparison to prior work and a clear limitations discussion should be moved into the main body.

2. The dataset is relatively small and skewed toward Yes/No and CoQA-derived narratives, limiting coverage of open-ended answers, multi-hop reasoning, numerical or temporal updates, non-English conversations, and realistic misinformation distributions; expanding scope and diversity and adding harder update scenarios would strengthen claims.
The evaluation relies heavily on template-based corrections and automatic Y/N scoring, with limited human assessment of coherence, persona consistency, side effects, and long-horizon stability; incorporating blinded human judgments, persistence tests across many subsequent turns, and measurements of latency and cost would improve robustness.

3. The proposed methods are not universally effective across models. In Figure 4 the OTC intervention does not consistently improve performance across models, and subsequent validation centers on GPT‑3.5, limiting representativeness; report per-model results with confidence intervals or significance tests and extend validation to multiple model families and sizes, including contemporary frontier models.

4. The oracle analysis for Reiteration in Figure 4c is used to hypothesize that auto-generated rewrites would perform similarly to human-written ones without direct evidence, so the authors should run a controlled experiment using auto-generated rewrites under the same protocol and compare against human versions with matched controls and statistical testing.

### Questions
1. Why does the main text omit Related Work and Limitations while devoting substantial space to analyzing a single model’s (GPT‑3.5) behavior?
2. Can you explain why Yes/No is used as the primary evaluation rubric and how you handle partially correct or ambiguous answers (including AE-produced N/A or ties)?
3. Beyond oracle (human) rewrites, can you run a matched-protocol comparison on a representative Dval subset between Reiteration with model-generated rewrites and the human version; given that different models vary in rewrite quality, how will you control for this, and do your conclusions still hold across models?
4. Why is the comprehensive analysis focused on GPT‑3.5, and how do you substantiate the generalizability of key conclusions (e.g., template sensitivity, Reiteration outperforming OTC)?
5. The CAM vs. CBA proximity effect is currently observed on GPT‑3.5; how will you ensure or verify that the same phenomenon holds for other models (families/sizes)?
6. Deletion effectively keeps only correct history, so improvement is intuitive but costly; how do you define and measure cost and determine when it is worth using, and what happens if Deletion is imperfect (e.g., fails to remove all incorrect content or removes useful context)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a critical and challenging task: Knowledge Editing In Conversation (KEIC), aimed at evaluating and enhancing LLMs' ability to dynamically update their knowledge and correct misinformation in multi-turn dialogues. The authors formalize the KEIC framework, segmenting conversations into false, update, test, and other phases, and precisely defining what constitutes an "effective new fact." To address this task, four model-agnostic user correction methods are proposed: One-Turn Correction (OTC), Verification, Reiteration, and Deletion. Reiteration and Deletion are identified as particularly effective. A high-quality, human-annotated KEIC dataset, comprising 1,781 instances derived from CoQA, is constructed to support this research, covering factual and non-factual narrative stories.

### Strengths
1. This paper discusses a crucial research question. The capability of in-context KE is highly demanded in realistic LLM products. Different from other knowledge conflict work, the paper focuses on user profiling.
2. The dataset  CoQA contains 1781 instances with human annotation, and has a potential impact for related research.
3. The evaluation is extensive, covering closed and open-sourced LLM families.

### Weaknesses
1. My major concern can be the lack of comparison with existing methods. The paper primarily investigates in-context correction strategies within multi-turn dialogues. Some related methods deserve credit and comparison.
a. parameter-editing KE methods (e.g., MEND, ROME) 
b. retrieval-based approaches when used specifically for knowledge correction (Memory-Based Model Editing at Scale)
c. methods that address knowledge conflict (ConflictBank)

2. Another concern involves the top-performing Deletion method. While it effectively removes "false knowledge," this information is not entirely useless, as the process of correction itself can provide valuable context.

3. A clearer illustration of the novelty of the paper will help enhance the persuasiveness.

### Questions
Please see Weakness

### Soundness
3

### Presentation
2

### Contribution
2
