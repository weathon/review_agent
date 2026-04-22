# Memorize or Generalize? Evaluating LLM Code Generation with Code Rewriting

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Large language models (LLMs) have recently demonstrated exceptional code generation capabilities. However, there is a growing and active debate whether LLMs are mostly doing memorization (i.e., replicating or reusing large parts of their training data) 
versus generalization (i.e., beyond training data). Existing evaluations largely proxy memorization with surface/structural similarity, thereby conflating benign reuse with harmful recall and neglecting task correctness under semantic variation. We define memorization behaviorally as failure at high similarity and introduce a semantic perturbation code rewriting, which rewrites a semantically different answer at a similar difficulty level for a given coding question, then reverse-engineers a novel coding question. We further propose Memorization Risk Index (MRI), a normalized score that combines two signals: (i) how similar the model’s answer for the rewritten task is to the original ground-truth solution, and (ii) how much performance drops from the original task to its rewritten counterpart. MRI is high only when both conditions hold—when the model outputs similar code but fails the perturbed task—thereby capturing harmful memorization rather than benign reuse. Empirical evaluations on code generation benchmarks MBPP+ and BigCodeBench reveal remarkable findings that (1) the memorization risk alleviates as LLMs scale up, and (2) supervised fine-tuning (SFT) improves accuracy while worsening memorization, (3) reinforcement learning with PPO achieves a more balanced trade-off between memorization and generalization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a code-rewriting pipeline to diagnose harmful memorization in code generation tasks in LLMs. To probe this, the authors evaluate models on altered versions of standard coding problems. They quantify and summarize risk with a single index, the Memorization Risk Index (MRI), which is high only when both the model’s output on the rewrite remains highly similar to the original solution and there is a relative accuracy drop (RAD) from original to rewritten tasks. Empirically, across MBPP+ and BigCodeBench, the study reports a consistent story where memorization risk generally falls with larger model scale on simpler problems but persists on harder ones. The authors also found that SFT tends to increase raw accuracy and memorization risk, while PPO better balances accuracy and risk, often keeping MRI near its base levels.

### Strengths
Clear Pipeline Setup: The papers provide a clean pipeline structure, giving an illustrative understanding of the author’s proposed setup. The authors also commit to releasing the evolved tasks, prompts, and code upon publication, which strengthens reproducibility and contribution to the research community for similar research.

Harmful Memorization Framing: The work defines harmful memorization as a joint event, failure after a semantic shift, alongside high similarity to the original solution, rather than treating any code reuse as memorization. The authors also clearly separate style robustness (e.g. paraphrase/mutation) and distinguish its lens from contamination-based definitions.

### Weaknesses
Scope Limited to Python:
The paper aims to characterize harmful memorization in code generation but evaluates exclusively on Python tasks. That makes other programming languages' settings uncertain, as different language ecosystems may differ materially from the author’s claims. would substantively strengthen the claim that the findings. A cross-language pilot would substantively strengthen the claim that the observed scaling trends and SFT vs. PPO trade-offs generalize beyond Python.

Dataset-level MRI:
Metric aggregation could possibly mislead. In this paper, MRI is computed as the product of two corpus-level aggregates, so “mean similarity” and “overall accuracy drop” may be driven by different subsets of items in the dataset. When you multiply these two dataset numbers, you can overstate or understate the true co-occurrence of high similarity and failure at the item level.

### Questions
Is there any reason why the evaluation is limited to Python? 

Is every experiment reported as Pass@1 (including the scaling plots)?

Small Comments:
- A few editorial issues “Budge:” in App. G. I think you meant Budget? 
- Small inconsistency in reported compute (Sec. F says 2 GPUs; earlier 4 A100s are mentioned).

### Soundness
2

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
5

### Summary
This paper proposes a code‐rewriting evaluation framework to diagnose harmful memorization in LLM code generation. The authors construct “semantically edited” variants of known solutions and reverse-engineer task descriptions, then measure MRI, a diagnostic score defined as Similarity × RAD (relative accuracy drop). Experiments on MBPP+ and BigCodeBench, across multiple model sizes and fine-tuning strategies, suggest that larger models reduce harmful memorization on easier tasks, while harder tasks retain non-trivial MRI; moreover, SFT tends to raise MRI whereas PPO achieves a better accuracy–memorization balance. The work is well scoped (focusing on harmful memorization rather than benign reuse), the rewriting pipeline is innovative, and MRI is simple and model-agnostic. However, the metric design has notable weaknesses: RAD can be inflated at low baseline accuracy, the similarity component (equal-weight AST + edit distance) lacks ablations and comparisons to alternative overlap measures, and the fine-tuning analysis remains largely observational without an MRI-aware training recipe. Overall, the paper is promising and practically relevant, but would benefit from robustness analyses (RAD stabilization, similarity ablations), broader metric comparisons, and closing the loop by incorporating MRI into training/selection protocols.

### Strengths
1. Well‑scoped formulation of harmful memorization. The paper explicitly targets harmful memorization—cases where high code similarity coincides with a meaningful accuracy drop—rather than benign reuse. The MRI construct (Similarity × RAD) operationalizes this notion in a way that is easy to interpret and analyze.
2. Innovative code‑rewriting evaluation pipeline. By making a single semantic change to a reference solution and then reverse‑engineering the task description, the pipeline creates tasks that are superficially similar yet semantically distinct. This design probes whether models can go beyond rote recall to genuine reasoning, improving ecological validity over paraphrase/noise baselines.
3. Simple, model‑agnostic metric with diagnostic value. The MRI score combines a similarity component with a relative performance‑drop component, yielding a unified signal that is high only when both similarity and degradation are high. Its two‑factor structure helps practitioners diagnose whether failures stem from brittle recall or broader capability gaps.

### Weaknesses
1. RAD is inflated at low baseline accuracy, which in turn overstates MRI. RAD is defined as \max\{0,\tfrac{Acc(T_{ori}) - Acc(T_{rew})}{Acc(T_{ori})}\}. When Acc(T_{ori}) is small, even a tiny absolute drop yields a large relative decrease, which elevates MRI (since MRI = Similarity × RAD). This effect is especially visible on BigCodeBench, where lower baseline accuracy coincides with persistently higher RAD and non‑zero MRI.
2. Similarity uses an equal‑weight average of AST and edit distance, but lacks ablations and alternative metrics. The paper fixes similarity as the unweighted mean of AST‑level similarity and token edit‑distance similarity. However, it does not report ablations (AST‑only, edit‑only, or weighted variants) or threshold‑sensitivity analyses. It also does not compare against common token‑overlap metrics such as n‑gram/Jaccard or MinHash. This under‑supports the design choice and limits the interpretability of what the similarity score is actually capturing (semantics vs. surface form).
3. The “Impact of Fine‑Tuning Strategies on Memorization” section is largely observational and lacks actionable training guidance using MRI. The paper observes that SFT tends to increase task accuracy but also raises MRI, while PPO appears to strike a better balance. However, MRI/Sim/RAD are not incorporated into the training objective, early stopping, or model selection (e.g., as regularizers, constraints, or explicit rewards). As a result, the findings remain descriptive rather than prescriptive, without a concrete, reproducible recipe for using MRI to guide training decisions.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates memorization versus generalization in LLM code generation by introducing an evaluation methodology called code rewriting. The authors distinguish between harmful memorization (high similarity to training solutions coupled with performance drops on semantically altered tasks) and benign code reuse. They propose the Memorization Risk Index (MRI), which multiplies solution similarity with relative accuracy drop under semantic perturbations. The paper evaluates multiple LLM families (Qwen, Llama) across MBPP+ and BigCodeBench benchmarks, examining how model scale and fine-tuning strategies (SFT vs. PPO) affect memorization. Key findings include: (1) memorization generally decreases with model scale, especially on simpler tasks; (2) SFT improves accuracy but increases memorization risk; (3) PPO (RL) offers a better memorization-accuracy trade-off.

### Strengths
- The code rewriting approach offers a novel way to evaluate memorization that goes beyond surface-level similarity metrics. The introduction of MRI to evaluate memorization is quite interesting as it helps capture harmful memorization as failure under high similarity on code-rewriting tasks.

- The experimental methodology is rigorous, with quality assurance through both automated (GPT-5 as judge) and manual validation. The evaluation spans multiple model families (Qwen, Llama), sizes (0.5B-70B), benchmarks (MBPP+, BigCodeBench), and training approaches (base, SFT, PPO), providing comprehensive coverage.

- The inclusion of semantic-preserving perturbations (mutation, paraphrase) as baselines effectively differentiates memorization from robustness issues.

### Weaknesses
- Need more justification around the design of the MRI score (why multiplication operation chosen)
- While quality assurance is performed, only 10% manual validation is reported. What was the acceptance rate from the LLM judge? How many generated tasks were rejected? 
- The paper reports aggregate MRI scores but provides limited insight into why models fail on rewritten tasks. Are failures due to specific semantic changes (e.g., sorting, filtering operations)? Qualitative analysis of high-MRI cases would provide actionable insights. The paper spends more time in formulating the problem rather than talking about ablations.
- The concepts introduced in this paper, i.e., memorization decreases with size, SFT increases memorization whereas PPO alleviates it aren't novel. The authors should have spend more analysis on why are the models struggling with code rewrites

### Questions
- Very difficult to follow the results with a combination of tables and results. Table 1 should be presented like Table 2 (or maybe merged) to show all the results in one place and the graphs maybe added to drive more discussion.
- There are some obvious questions from the table - The trend of RAD paraphrasing is not clear? (the trend is reversed in some family of models)
- You mention "modify one logic" in ground truth solutions. What types of logic changes are most common in your dataset (e.g., change operators, add constraints, reverse logic)? Is there diversity in change types, and does MRI vary by change type?
- Why does mutation (semantic-preserving) cause higher RAD than paraphrase? What specific mutation types (character noise, word scrambling, capitalization) are most problematic? Does this suggest brittleness distinct from memorization and does it indicate problems in real world scenarios ?

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
The paper studies when code LLMs “memorize” rather than generalize. It introduces code rewriting—constructing a new task by modifying the ground-truth solution’s semantics while keeping the same signature and similarly worded prompt—to create semantically different but surface-similar problems.

### Strengths
The paper’s strength is its clear behavioral framing of harmful memorization and the introduction of the MRI, which effectively separates harmful memorization from benign code reuse. It provides a well-structured empirical study across models, showing that scaling reduces risk, SFT increases it, and PPO balances both, with strong baselines confirming MRI’s robustness.

### Weaknesses
1.How sensitive is MRI to the AST/token weighting and to the choice of code parser? Any robustness checks across parsers/grammars?
2. The rewritten tasks are produced by LLMs. This risks distributional coupling between generation artifacts and the evaluated models, and raises the question of whether MRI partly reflects quirks of the rewriting pipeline rather than memorization per se.
3. The 50/50 AST vs edit weighting is plausible but unvalidated. Some memorization (e.g., templated control flow with different identifiers) may not be well captured. A learned or ablated weighting, plus code-embedding similarity, could calibrate MRI.
4. For SFT vs PPO, what are the data overlaps between training and evaluation (including rewritten variants)? Any risk that SFT learns rewrite artifacts more than PPO?

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
