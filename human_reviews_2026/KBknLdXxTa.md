# From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Large language models now solve many benchmark math problems at near‑expert levels, yet this progress has not fully translated into reliable performance in real‑world applications. We study this gap through contextual mathematical reasoning, where the mathematical core must be formulated from descriptive scenarios.We introduce CORE-MATH, a benchmark that repurposes AIME and MATH-500 problems into two contextual settings: Scenario Grounding (SG), which embeds abstract problems into realistic narratives without increasing reasoning complexity, and Complexity Scaling (CS), which transforms explicit conditions into sub‑problems to capture how constraints often appear in practice. Evaluating 61 proprietary and open‑source models, we observe sharp drops: on average, open‑source models decline by 13 and 34 points on SG and CS, while proprietary models drop by 13 and 20.  Error analysis shows that errors are dominated by incorrect problem formulation, with formulation accuracy declining as original problem difficulty increases. Correct formulation emerges as a prerequisite for success, and its sufficiency improves with model scale, indicating that larger models advance in both understanding and reasoning. Nevertheless, formulation and reasoning remain two complementary bottlenecks that limit contextual mathematical problem solving. Finally, we find that fine‑tuning with scenario data improves performance, whereas formulation‑only training is ineffective. However, performance gaps are only partially alleviated, highlighting contextual mathematical reasoning as a central unsolved challenge for LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles a very relevant gap, i.e. models that perform well on abstract math often struggle when the same problems are presented in a realistic narrative. The authors build a benchmark by transforming existing items from AIME and a filtered MATH-500 subset into two contextualized variants. The first, Scenario Grounding (SG), wraps the same core math in a short story while keeping the solution unchanged. The second, Complexity Scaling (CS), conceals some explicit conditions behind small sub-problems, requiring the model to first identify what is needed before solving. They evaluate a large set of models and observe a clear drop from the original to SG and a further drop to CS. They also analyze where errors originate and introduce three formulation metrics (formulation accuracy, necessity, and sufficiency) to distinguish between “setting up the right math” and “doing the math.”. The authors test training strategies and find that simple SFT on scenario-style data gives measurable improvements without harming abstract performance. 

I see the contribution as timely and practically important. The benchmark design, although not perfect, addresses a real need: testing whether models can extract the correct structure from a story. The formulation-vs-reasoning framing is helpful for analysis, and the training results give a simple recipe that others can try. The novelty is, however, moderate (a new benchmark plus analysis), but the empirical scope and the focus on formulation make it valuable to the community, provided the evaluation is strengthened.

### Strengths
The work isolates a phenomenon that has been widely reported, ie, strong models on abstract math often fail when conditions are embedded in text. The two-variant construction is simple and scalable, and the error analysis is quite illuminating, showing that setting up the math is frequently the key difficulty. I also appreciate that the authors do not stop at measurement, but also include initial training interventions that enhance contextual performance without compromising abstract skills. The paper is mostly clear, and the benchmark seems practical to adopt.

### Weaknesses
My main concerns are about measurement validity and fairness. The SG/CS transformations would benefit from multiple independent annotators and a quantitative audit to ensure equivalence and controlled change. The formulation metrics rely on an LLM judge. The human validation sample is eventually too small, and there is no check with a different judge family or a symbolic/exact solver, where feasible. The open-source vs proprietary comparison lacks confidence intervals to assess robustness. Contamination or near-duplicate checks are not presented, which matters for items derived from widely circulated math sets. The SFT pipeline filters scenarios using one solver, which can introduce selection bias and may not generalize.

### Questions
1) Could you report inter-annotator agreement and a small difficulty-parity audit for SG, and a clearer rubric for CS to show it does not change the task class? 
2) Would you consider re-scoring a stratified subset with a different judge family, some exact/symbolic checks where possible, and a larger blinded human pool?
3) It would also help to provide confidence intervals or bootstraps, and to balance sampling across model groups (or clearly separate the summaries). 
4) Please comment on step-level near-duplicate analysis and on the sensitivity of the SFT gains to the choice of solver used for filtering.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CORE-MATH, a benchmark for evaluating LLMs on contextual math reasoning. The benchmark converts problems from AIME and MATH-500 into scenario-based narratives and more complex narratives with embedded sub-problems, Complexity Scaling. The authors evaluate 61 and proprietary open-source LLMs and find that performance drops sharply when moving from abstract to contextual settings. Through error analysis and targeted experiments, the main limitation is identified as incorrect problem formulation rather than failures in step-by-step reasoning. The paper also examines how fine-tuning and formulation-specific training can reduce these errors and shows gains, but a large gap remains. The work demonstrates that contextual math reasoning is still an open challenge for current LLMs.

### Strengths
- The research question is well motivated and the main finding is relevant. The paper provides a useful observation: the authors identify contextual complexity as a general bottleneck that limits current LLMs' reliability on multi-step reasoning.

- The paper provide insightful error analysis. The detailed breakdown of error types (Figure 2) demonstrates that formulation errors predominate across architectures in contextual settings, which is often overlooked in prior work.

### Weaknesses
- The paper builds several automatic annotation and categorization steps and relies on an LLM judge with only light human checks. The authors state that they use o1-mini to decide whether a model output is mathematically equivalent to the reference solution, and that manual checks on Qwen3-14B and Qwen3-32B show more than 90% agreement with human judgments. However, the paper does not report the exact sample sizes, selection protocol, prompts, or agreement statistics, e.g., Cohen's kappa, Pearson/Spearman correlation of scores, or confidence intervals. The heavy use of an LLM-as-judge without full validation details becomes a concern.

- The error analysis uses GPT-5 to assign error categories to outputs from other models, but the paper does not provide the prompts or templates that define each category. Also, there is no study of judge bias or stability, e.g., prompt ablations, temperature sweeps, and no human expert evaluation. A small human audit with inter-rater agreement would be useful. 

- CORE-MATH draws from AIME-2024, AIME-2025, and a filtered subset of MATH-500, with each original item converted into two variants, SG and CS. This means the total pool starts from a relatively small number of sources. The paper probes robustness by adding two extra SG versions for AIME-2024 and reporting an averaged score, SG Avg@3, but it does not report standard deviation or standard error, and it does not repeat this check for CS or the other subsets. It is not clear whether that the results could be sensitive to specific paraphrases; Another concern is the limited scale in CORE-MATH, which makes results less convincing.

### Questions
- The proposed SFT setup helps but does not remove the drop under CS. The paper positions this as a first step, but the current training recipe and analysis do not yet show a clear path to solve the remaining gap. What suggestions do the authors have to further address this research gap, or how would the authors justify the limited improvement through their method?

- Typo: the caption reference "Figure 4" should be "Table 5."

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
The authors introduce CORE-MATH, a new benchmark that repurposes problems from established sources like AIME and MATH-500 into more realistic, narrative-driven scenarios. This is done through two variations: "Scenario Grounding," which embeds problems in a narrative, and "Complexity Scaling," which conceals conditions within sub-problems. Through an extensive evaluation of 61 models, the paper demonstrates a significant drop in accuracy on these contextual tasks. The primary cause of failure is identified as "problem formulation"—the inability to correctly extract the core mathematical structure from the narrative. The authors find that while model scale helps, it doesn't solve the issue, and that end-to-end fine-tuning on scenario-based data improves performance, whereas training a model solely for formulation is ineffective.

### Strengths
1. The work tackles the highly relevant and important gap between benchmark success and practical, real-world capability in LLMs, pushing research beyond abstract problem-solving.
2. The CORE-MATH benchmark is well-designed. By building upon trusted sources (AIME, MATH-500) and systematically creating two distinct types of contextual challenges (SG and CS), the authors provide a controlled framework for analyzing this problem.
3. The study is thorough, evaluating a wide array of 46 open-source and 15 proprietary models. The analysis goes beyond simple accuracy metrics, effectively identifying problem formulation as the key bottleneck through both qualitative and quantitative evidence.

### Weaknesses
The primary weakness of the work lies in the lack of detail regarding the benchmark construction process, which is critical for ensuring the benchmark's validity and reproducibility. The paper states that an LLM (01-mini) was guided by structured prompts to generate the contextual variants, which were then reviewed by human experts. However, several key aspects unclear:
1. Why choose o1-mini but not stronger models? Did the authors compare its performance with other frointer LLMs?
2. What was the protocol for the human expert review to guarantee mathematical equivalence? How many experts reviewed each problem, and what was the procedure for resolving disagreements to ensure the final scenarios were valid?

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates LLMs’ mathematical reasoning in contextual scenarios, where the underlying math must first be formulated from narrative descriptions before being solved. It introduces CORE-MATH, a benchmark that repurposes AIME and MATH-500 problems into two controlled variants: Scenario Grounding (SG), which embeds problems in realistic narratives without altering the core math, and Complexity Scaling (CS), which hides explicit conditions behind simple sub-problems to mimic how constraints appear in practice.

Across 61 models, the authors observe substantial accuracy drops from the original abstract problems to SG—and even larger drops on CS—implicating problem formulation as a primary failure mode. Training experiments indicate that fine-tuning on scenario data improves robustness, whereas a formulation-only pipeline is ineffective.

### Strengths
The problem is interesting, as it focuses on realistic contextualization of mathematical reasoning.

The paper introduces a new benchmark, CORE-MATH, which is built upon AIME 2024, AIME 2025, and MATH-500 datasets.

The proposed Scenario Grounding (SG) and Complexity Scaling (CS) strategies are effective for constructing contextual mathematical problems that test both problem formulation and reasoning capabilities.

### Weaknesses
Is the mapping from the original problem to the narrative automatic or human-assisted?

How do the authors ensure that the mapping is accurate? The generated narrative may not exactly match the original mathematical problem, which could alter its meaning.

The Scenario Grounding (SG) and Complexity Scaling (CS) strategies are used for data construction. How do the authors guarantee that these transformations do not change the underlying problem semantics?

Would it be possible to train models on similar data types so that their performance can be better preserved across contextualized settings?

### Questions
Since AIME and MATH datasets may have already been heavily used or augmented during LLM training, how would the results differ if the authors included other benchmarks, such as MathOdyssey?

### Soundness
4

### Presentation
3

### Contribution
3
