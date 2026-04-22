# Reasoning-Intensive Regression

- Avg Score: 3.67
- Decision: Reject
- Scores: 2, 2, 4, 4, 4, 6

## Abstract
AI researchers and practitioners increasingly apply large language models (LLMs) to what we call reasoning-intensive regression (RiR), i.e., deducing subtle numerical scores from text. Unlike standard language regression tasks, e.g., for sentiment or similarity, RiR often appears instead in ad-hoc problems such as rubric-based scoring, modeling dense rewards in complex environments, or domain-specific retrieval, where much deeper analysis of context is required while only limited task-specific training data and computation are available. We cast four realistic problems as RiR tasks to establish an initial benchmark, and use that to test our hypothesis that prompting frozen LLMs and finetuning Transformer encoders via gradient descent will both often struggle in RiR. We then propose MENTAT, a simple and lightweight method that combines batch-reflective prompt optimization with neural ensemble learning.  MENTAT achieves up to 65% improvement over both baselines, though substantial room remains for future advances in RiR.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes “Reasoning-Intensive Regression (RiR)”,  a subset of text-to-score problems that purportedly require numerical reasoning rather than shallow feature extraction. It classifies four different tasks such as math error detection, instruction following, pairwise RAG comparison, and essay grading into continuous targets to form a benchmark, arguing that standard LLM prompting and small-encoder fine-tuning both struggle. The authors also introduce MENTAT, combining batched prompt evolution with a small neural aggregator over multiple rollouts, and advocate CCC (concordance correlation coefficient) alongside NMSE to avoid variance-collapse pathologies.

### Strengths
- The paper identifies a failure mode of distance-only metrics (NMSE) in regression based reasoning tasks and motivates CCC to capture agreement

- Presents a simple, lightweight method (MENTAT) that is shown to improve over frozen prompting and a finetuned encoder on the proposed tasks in the evaluation

- Provides an initial, multi-task benchmark with a clear claim that some real-world scoring problems need deeper analysis than typical sentiment/similarity regression

### Weaknesses
- Conceptual motivation for “RiR” and the three-level taxonomy is not well motivatied. RiR is described as “fuzzy,” and the levels are explicitly “informal,” with no clear definition on what tasks could be classified this way. This needs to be strengthened further

- Task taxonomy and task definitions for levels is under-justified. Several tasks (e.g., math) are converted into regression by design (predict fraction until first error), which evaluates a surrogate metric rather than native task success. The authors don’t justify why this surrogate is decision-relevant, nor report native metrics (e.g., accuracy, exact match) alongside the other metrics they report. I would love to hear the author side argument on this.

- MENTAT novelty is unclear relative to prior prompt-optimization. The “batch-reflect” prompt evolution over worst-case subsets looks close in spirit to existing prompt-evolution methods; the draft does not provide a thorough, controlled comparison [1] .


- Aggregator/loss choice lacks justification. The order-invariant MLP over rollout stats and the training objective (emphasizing CCC/NMSE) are presented without ablations on alternative aggregators (non-parametric, rank-based) or loss functions.


- It is unclear what role does NeoBERT play and whether it is fair comparison in the light of the experiments. The paper mixes a finetuned encoder (NeoBERT) with frozen LLMs; this crosses regimes. Matched-budget baselines (e.g., LoRA-tuned small LLMs, stronger encoders) are missing, so the relative claims are hard to interpret.

[1] prompterator: Iterate efficiently towards more effective prompts. Sucik et al, 2023

### Questions
Task casting: For math and pairwise-RAG, why is the regression surrogate (e.g., fraction until first wrong step) the right target?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper studies the problem of reasoning-intensive regression, which requires the LLMs to deduce subtle numerical scores from text. The authors proposes a prompt optimization paradigm by LLM self reflection and showcase the improvement compared to baseline head fine-tuning and prompt hand-crafted fine-tuning methods.

### Strengths
The study of reasoning-intensive regression of LLMs is indeed of crucial importance and there lacks work that seriously investigate the limitation of existing LLMs in use cases like LLM as a judge. The direction that goes beyond hand crafted prompt tuning is interesting and appealing.

### Weaknesses
1. The benchmarking tasks in Section 2 is rather confusing. Why we choose these tasks as benchmarks for reasoning-intensive regression? There lacks sufficient discussion on the coverage of these tasks and what is the performance of existing either open source or closed source LLMs.

2. The proposed method MENTAT requires additional training and inference burden, there lacks a clear description of the motivation behind this method. In addition, there lacks explanation of what types of data the additional MLP is trained on. What is the input and output of the MLPs? What is the scale of this additional training part upon the original LLMs? What is overall the additional inference cost of current method compared to the standard fixed template of existing LLMs or few shot templates?

3. Disclaimer: I am not familiar with this field of reasoning-intensive regression. But I do think there is huge space to improve on the writing side to make the paper readable for broader audience. I have read many papers, including those in the area I am not familiar with, and this is really one of the few times I found a paper's wordings and technical descriptions make me feel super puzzled almost throughout the paper, making it extremely hard to follow. I would highly recommend adding more relevant equations to explain the tasks and metrics more detailedly, instead of using long sentences and only word descriptions of the methodology. Overall, I do not think this work does a good job in explaining clearly the method proposed.

### Questions
See questions in weakness.

### Soundness
4

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Reasoning-Intensive Regression (RiR), an interesting category of tasks where models performs detailed reasoning and output precise numeric values. It presents a benchmark of tasks such as math error detection, essay grading, showing that standard fine-tuning often collapses to mean predictions, while prompting large models yields reasoning but poor numeric calibration. To address this, the authors propose MENTAT, a lightweight method that first optimizes prompts, then does multiple rollouts, get regression outputs from them, and combine them using an external MLP. The paper is well written however, at times it felt overly verbose. The methodology itself is sound, but noveltywise, it felt like a mixture of prompt optimization and self consistency. More importantly, based on the framing of the task, the paper misses important baselines to compare against to showcase the efficacy of MENTAT.

### Strengths
1. This paper introduces Reasoning-Intensive Regression, which is a timely task to investigate into when LLMs are sufficiently powerful to finish important tasks. 
2. The four different benchmark datasets are quite interesting to be framed for reasoning-intensive regression, although I believe the task choices could be even more well thought out.

### Weaknesses
1. MENTAT seems like using self-consistency on top of prompt optimization. I am not sure whether about the contribution in that aspect, as it is well known that both of them should substantially boost performance.
2. "optimizing prompts for RiR tasks has the fairly unique property that the patterns across examples are at least as important as the per-example error." - I believe this is a general theme of most automated prompt optimization methods. More importantly, authors claim that "MENTAT’s prompt evolver is centered around asking the LLM to jointly reason about tens of mistakes at once". I am not sure it is accurate given methods like APO/ PE2 / TextGrad has already covered those areas. 

3. I think the authors need to cover more baeslines here. Modern Prompt optimization methods should definitely be covered as baselines to prove the efficacy of the suggested approach. Also, the original self consistency can be a good baseline to compare against.

[1] Pryzant R, Iter D, Li J, Lee YT, Zhu C, Zeng M. Automatic prompt optimization with" gradient descent" and beam search. arXiv preprint arXiv:2305.03495. 2023 May 4.
[2] Ye Q, Axmed M, Pryzant R, Khani F. Prompt engineering a prompt engineer. arXiv preprint arXiv:2311.05661. 2023 Nov 9.

### Questions
Please check the weaknesses. The claims need to be substantiated and coverage of prompt optimization methods also needs to be improved, although I understand that is not the primary focus of the problem itself.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MENTAT, a streamlined framework to improve the LLM's performance on reasoning-intensive regression (RiR) tasks. This is a two stage process, starting with auto prompt optimization, followed by ensemble learning. It only involes inference on frozen LLMs, and does not require large training datasets, making it lightweight and applicable in practical real world settings.

### Strengths
- The MENTAT framework automates the prompt engineering process, which can be very useful, especially in complex scenarios like RiR, where traditionally requires extensive amount of human involvement.
- Ensemble learning is an interesting way to take advantage of the probabilistic nature of LLMs, and increase the regression quality.

### Weaknesses
- When introducing MENTAT in Sec.3, it lacks important details, specifically authors should explain
  - Phase 1
    - How to select the bottom $\sqrt{n}$ rollouts in the training set. Are the rollouts ranked by a combination of CCC/NMSE? If so, is the ratio constant, and is the ratio the same as the one used for MLP training?
    - How are the prompts structured in `error analysis` and `prompt refinement`? Does the setup vary between models and/or tasks? Should include all the prompts used in the Appendix.
    - Consider include the conversation of a full iteration. This can be added to the Appendix.
- In Sec.4, results lack ablation study and analysis of the MENTAT framework
  - The performance gap between different veresions of MENTAT makes me wonder if the success of MENTAT is primarily driven by the aggregation of the LLM's probabilistic rollout or is it due to a better prompt generated from the automated process. Authors should include ablation study wrt basic/detailed human crafted prompts (HCP), specifically `HCP Avg` and `HCP MLP` in the results, and compare with their MENTAT counter parts to better analyse the framework. For HCP MLP, training should use the same amount of data and HPT as in MENTAT Phase 2.
  - Authors should add a figure for the distribution of `{Var(all rollouts for question i)}` for each task with HCP and MENTAT Prompt. This should help analyse if aggregation is the key in obtaining a better regression performance.
  - To help understand the effect of prompt optimization, authours can include a `NMSE/CCC vs Iteration` plot in the training process, consider increasing the total iterations for long term analysis. As stated in Appendix.C, MLP is very lightweight, and it should be feasible to include plots of both with and without training the aggregator at each iteration.
  - Add variance to Table.1,2 if space allows. If not, this can be included in the Appendix.
- In Appendix.F, include all the LLM discovered prompts for entries that appear in Table.1,2

### Questions
- In Sec.2, regarding the dataset and their evalatuion metrics,
  - Detecting Mathematical Errors, why filter out problems with correct solutions that existed in the original dataset? I think this should be included to truly reflect the real world setting.
  - Instruction Following, original paper used binary label for each requirement $r_i$, and labels are averaged to produce the final score. Why change to $s_i \in [0,1]$ for each $r_i$, and adopt harmonic mean?
  - Pairwise RAG Comparison, why not use the evaluation metric from the original paper, namely RAG-QA Arena?
- How much data were used to train MLP?
- Table.1 mentioned that NeoBERT performs better with "1000 training + 500 validation", why not add an additional column to the results?
- Why not use a universal evaluation model (potentially more powerful than the test model) in error analysis and prompt refinement?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the concept of reasoning-intensive regression, a type of language regression task that requires intensive reasoning from an LLM. To benchmark it, the author designed a benchmark with 4 tasks, spanning different levels of required reasoning capability. Lastly, the author designed a combined evolution and neural aggregation method to better solve this benchmark.

### Strengths
The paper is well-written, the benchmark and the methodology are well-presented.

### Weaknesses
* While presenting a new benchmark for RIR, the paper lacks an error analysis addressing the current models’ or methods’ pitfalls on the benchmark.
* Additionally, regarding the different levels of regression tasks and their corresponding benchmark designs, it would be better to include some quantitative or qualitative comparisons showing the differences between level-3 reasoning-intensive tasks and levels 2 and 1 (e.g., average CoT length, confidence, self-consistency, etc.).
* For the experiment setup with Mentat, I feel that a very important practical question is how much data is allocated to train or supervise the method (in phase 1 and phase 2). Data efficiency is crucial, especially since collecting regression data is costly. Therefore, a data-point efficiency experiment is needed (beyond the two options of 100 and 500 budget).
* I’m also curious — in Table 1, how would the method compare to few-shot learning with carefully selected few-shot examples? This seems to be an important baseline as well.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper defines a new task family called Reasoning-Intensive Regression (RiR)—language-based regression problems that require explicit multi-step reasoning instead of shallow feature extraction. The authors argue that standard encoder fine-tuning collapses to narrow score ranges, while directly prompting frozen LLMs yields coarse, discretized outputs.
This paper establish an initial benchmark of four diverse RiR tasks (mathematical-error detection, instruction-following quality, pairwise RAG comparison, and essay grading) and propose MENTAT, a lightweight two-phase method combining batch-reflective prompt optimization with neural ensemble aggregation.
Across all tasks, MENTAT improves Concordance Correlation Coefficient (CCC) over both encoder and prompting baselines, showing that simple prompt evolution plus multi-rollout calibration can recover both reasoning depth and numeric precision.

### Strengths
1. The proposed Reasoning-Intensive Regression (RiR) benchmark captures an emerging but under-explored problem space—tasks requiring both deep reasoning and precise numeric scoring (e.g., reward modeling, rubric-based evaluation). This formulation is well-motivated and relevant to ongoing LLM-judging and alignment research.
2. The authors convincingly argue that normalized MSE can be misleading due to output collapse and advocate using the Concordance Correlation Coefficient (CCC), which jointly considers variance, mean alignment, and correlation. This insight is both intuitive and empirically supported, representing a meaningful contribution to LLM evaluation methodology.
3. MENTAT’s two-phase structure—batch-reflective prompt optimization followed by neural aggregation—is simple, effective, and closely mirrors how humans iteratively refine prompts and aggregate judgments. It provides measurable gains while remaining lightweight and reproducible across models.

### Weaknesses
1. The mathematical RiR task relies on ProcessBench, which explicitly annotates the first erroneous reasoning step. While this allows continuous regression scoring, such step-level annotations are rare in most math datasets, limiting scalability and generalizability.
2. Instruction-Following uses gpt-oss-20b (an open-source LLM) while other tasks use GPT-5 or GPT-4.1. The authors briefly claim this is “for reproducibility and generalization validation,” but no comparison is given to show GPT-5 behaves similarly on that task. 
3. Although the paper claims MENTAT is lightweight, it involves multiple LLM inference rounds and K-fold rollouts per sample. The actual compute cost, token budget, or latency trade-off is not quantified. Moreover, comparisons to recent prompt-optimization or ensemble calibration methods are absent, leaving uncertainty about relative efficiency.

### Questions
1. Your prompt evolution stage always selects approximately √n of the worst samples for reflection. Have you experimented with smaller or larger batches, or grouping by error type rather than just score percentile?
2. Could the MLP aggregation in Phase 2 be replaced by a non-trainable closed-form rule (e.g., median plus variance correction)? If possible, please provide a comparison table between simple aggregation baselines (average, median, trimmed mean) and the learned MLP.

### Soundness
3

### Presentation
4

### Contribution
3
