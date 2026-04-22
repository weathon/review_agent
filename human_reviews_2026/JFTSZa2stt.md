# Sage: A Scalable Framework for Evaluating LLM-as-a-Judge Without Human Effort

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
LLM-as-a-Judge has been widely adopted as an evaluation method and served as supervised rewards in model training. However, existing benchmarks for LLM-as-a-Judge are mainly relying on human-annotated ground truth, which introduces human bias that undermines the assessing reliability and imposes scalability constraints. To overcome these limitations, we introduce Sage, a novel evaluation suite that assesses the quality of LLM judges without necessitating any human annotation. Inspired by axioms of rational choice theory, Sage introduces two new lenses for measuring LLM-as-a-Judge: local self-consistency (pair-wise preference stability) and global logical consistency (transitivity across a full set of preferences). We curate a dataset of 650 questions by combining structured benchmark problems with real-world user queries. Our experiments demonstrate both the intrinsic stability of our metrics and their high correlation with supervised benchmarks like LLMBar and RewardBench2, confirming SAGE's reliability as an evaluation suite for LLM-as-a-Judge. Based on Sage, we reveal that current *state-of-the-art* LLMs exhibit significant robustness deficiencies when acting as judges; even the top-performing model, Gemini-2.5-Pro and GPT-5, fails to maintain consistent preferences in nearly a quarter of difficult cases. We attribute this to a new phenomenon called **situational preference** which explain why explicit rubrics or criteria can help model judge consistently across answer pairs. Our further analysis shows that fine-tuning LLM-as-a-Judge is an unreliable method which further induce human bias, while multi-agent judges, deep reasoning can enhance performance through different means.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SAGE proposes a fully automatic framework to evaluate “LLM-as-a-Judge” without human gold labels by measuring (i) local self-consistency via a symmetrized pairwise protocol (Intra-Pair Instability, IPI) and (ii) global logical coherence via the minimum edits to reach a weak total order (TOV). Using 650 questions (RewardBench2 + WildChat-1M) and 6 candidate answers per question, the paper reports low metric variance, strong correlations with existing human-grounded judge benchmarks, and analyses showing that rubric-guided judging and multi-model panels help, while fine-tuned judges can regress.

### Strengths
- Clear, axiomatic motivation for order-consistency as a desideratum for LLM judges; precise formalization of IPI (positional self-consistency) and TOV (global transitivity to a weak total order).
- Practical, scalable evaluation protocol (round-robin, symmetrized) with easy-to-reproduce ingredients; sensible EASY vs HARD tiers via answer diversity vs homogeneity.
- Empirical metric stability (very low variance) and reported correlations with LLMBar/RewardBench2 that argue SAGE is at least a useful proxy for robustness/accuracy without human labels.
- Useful analyses: rubric prompting reduces IPI/TOV, model “reasoning depth” helps, mixed outcomes for fine-tuned judges, and modest gains from panel aggregation.
- Ethical/reproducibility commitments (release of code/dataset/prompts; curated sourcing).

### Weaknesses
- Proxy vs validity gap: IPI/TOV assess internal logical/positional consistency, not external correctness; the paper relies on correlations to argue usefulness, but this does not establish causal or sufficient validity—especially on subjective tasks. More direct evidence is needed that lowering IPI/TOV truly improves application outcomes (e.g., arena rankings stability, RLHF learning signals).
- Model- and prompt-sensitivity: The symmetrized protocol and prompts may themselves induce artifacts (e.g., sensitivity to tie handling, abstentions, verbosity formatting). The paper does not systematically vary prompt templates, tie policies, or judge “instruction strength” and report robustness envelopes.
- TOV computation & scalability: Minimizing edits to a weak total order is a nontrivial combinatorial problem; complexity/algorithmic details and performance for larger n are not fully fleshed out (the work uses n=6). Practicality for n≫6 and the effect on confidence intervals are unclear. 
- Construction of HARD set: “Hardness” is instantiated by generating all 6 answers from a single capable model. This conflates source homogeneity with true near-ties, and risks model-specific quirks. A more model-agnostic construction or human sanity checks on difficulty would strengthen claims.
- Multi-agent analysis is thin: The negative ChatEval result may be configuration-dependent; ablations (num-rounds, argument exchange format, judge selection) are missing, so the conclusion that “debate fails to help” is too strong.
- No cost/throughput accounting: A selling point is scalability “without human effort,” but the token/runtime cost of full round-robin, symmetrized judging across 650×C(6,2) pairs and multiple runs is not reported; practicality for frequent benchmarking or larger suites is unclear. 
- External impact evidence: Claims that SAGE improves arena stability and RLHF safety are suggestive; stronger end-to-end studies (e.g., swap reward model/arena judge using SAGE-selected judges and show better downstream stability/accuracy) are missing.

### Questions
- Can you run an ablation where the only change in an arena/auto-rater or RLHF loop is replacing the judge with a SAGE-selected one, and show narrower Elo CIs or higher human agreement?
- Can you report results under ≥3 substantially different judge prompts (instruction strength, rubric placement, formatting) and show that rankings and absolute scores remain stable?
- How frequently does the judge return “equal quality”? How are frequent ties treated in IPI/TOV, and how sensitive are the metrics to the proportion of ties?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SAGE, a framework for evaluating the quality of LLMs used as judges, free of human-annotated ground truth. The work is motivated by the inherent biases, scalability issues, and high costs associated with human annotations. Inspired by axioms of rational choice theory, SAGE assesses LLM judges on two intrinsic properties:

* Intra-Pair Instability (IPI): A measure of local self-consistency, which quantifies how often a judge's preference for one of two answers flips when their presentation order is swapped.
* Weak Total Order Violation (TOV): A measure of global logical consistency, which calculates the minimum number of judgment changes required to make a judge's full set of pairwise preferences transitive.

The framework includes a curated dataset with two difficulty tiers: SAGE-EASY (comparing answers from models of varying quality) and SAGE-HARD (comparing subtly different answers from a single advanced model). The authors evaluate 13 prominent LLMs and find that even top models exhibit significant deficiencies in robustness, particularly on the SAGE-HARD set. A key finding is the identification of "situational preference", a phenomenon in which a judge's criteria appear to shift across different answer pairs for the same question. The paper shows that this can be mitigated by explicitly prompting the model to generate and use a fixed rubric.

### Strengths
* **Originality**: SAGE is original in formalization and framework design, though conceptually close to earlier analyses of LLM judge bias (TrustJudge, PandaLM). The identification and naming of the "situational preference" phenomenon provides a new vocabulary for describing a key failure mode in LLM reasoning.
* **Quality**:  The framework's claims are credible and robust. The authors provide (1) a theoretical analysis grounded in Conformal Prediction to prove the stability of their metrics, (2) empirical experiments showing low variance across repeated runs, and (3) external validation showing strong correlation with established human-annotated benchmarks.
* **Clarity**: The paper is clear and understandable. It has clear definitions of IPI and TOV, combined with the excellent schematic in Figure 2, which allows the reader to grasp the entire methodology with ease.
* **Significance**:  By removing the dependency on human annotators, SAGE offers a practical path forward for benchmarking LLM judges at a scale and cost that was previously prohibitive. This can accelerate research and development by providing a more efficient tool for diagnosing and improving LLM reliability.

### Weaknesses
* LLM judge bias is not a new topic, it has been discussed in earlier papers like MT-bench. . Also when I search related literatures, I find a recent work TrustJudge seems to formulate highly similar to this work, although its experiments are not as sound as this paper. 
* Incomplete Proxy for Accuracy: While the paper shows a strong correlation with accuracy benchmarks, consistency is not a substitute for correctness. A model could be perfectly consistent in its preference for a plausible-sounding but factually incorrect answer. The authors acknowledge this by framing SAGE as a proxy, but it remains an inherent limitation of the approach. This limitation could be fundamental, because axioms of rational choice theory guarantee consistency rather than accuracy.
* Dependency on Candidate Answers: The evaluation is dependent on the quality and diversity of the candidate answer set. While the paper's method for generating SAGE-EASY and SAGE-HARD is reasonable, the results ultimately depend on this specific answer generation process.

### Questions
1. In the multi-agent experiments, panel-based approaches (POLL) improved performance, whereas the debate-based framework (ChatEval) degraded it. This is a surprising result. What is your hypothesis for why a debate would make judgments less robust? Could it be that the debate process itself can be steered by fallacious arguments, amplifying inconsistency?
2. Have you considered applying SAGE to evaluate human annotators? Since the framework is fully automated, it seems it could be used to measure the internal consistency of human evaluators and potentially identify those who are more reliable, providing a quantitative handle on the "noisy and inconsistent data" problem you describe in Figure 1.

### Soundness
4

### Presentation
4

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
This paper introduces Sage, a framework for evaluating the consistency of LLM-based judges. Specifically, the authors introduce two metrics: IPI for measuring positional consistency and TOV for measuring transitive consistency. On a curated dataset of 650 questions, current LLM-based judges are found to have serious deficiencies, but rubrics, multi-agent judges, and extending reasoning can improve robustness.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed metrics (IPI and TOV) are motivated and relevant to known issues with LLM-based judges.
3. The evaluation is thorough, and additional analysis is meaningful and offers practical recommendations for creating more consistent LLM-based judges.

### Weaknesses
1. The framing of Sage as "scalable framework" and "without human effort" is strange. I don't feel the cost of human annotation is particularly meaningful on an evaluation dataset of only ~650 questions, and further scaling would introduce additional complexity/challenges when using the dataset.
2. I don't buy the argument that Sage is a strong proxy for "accuracy" either against human preferences or some other objective ground truth. While it correlates well, it's easy to imagine a judge that is perfectly consistent (e.g., IPI=0 and TOV=0) but systematically wrong. Ultimately, I view Sage as a piece in a broader evaluation suite, for example: Sage to measure self-consistency, RewardBench [1] to measure human agreement, and JudgeBench [2] to measure objective correctness.

[1] https://arxiv.org/abs/2403.13787

[2] https://arxiv.org/abs/2410.12784

### Questions
Some random comments that didn't factor into my evaluation:
- In the related work, you might compare against TrustJudge [3] (contemporaneous work), which investigates the same two inconsistencies.
- Fig. 3b is difficult to read.
- L265. "evaluat" -> "evaluate"
- L408. "justice judgments" -> "just judgements"
- L423. "Figure 5" -> "Table 5"

[3] https://arxiv.org/abs/2509.21117

### Soundness
4

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
This paper presents an automatic method for judging LLM as a judge. It presents 2 metrics for judging llm as a judge, namely, Intra-Pair Instability (IPI), and Weak Total Order Violation (TOV). 

Intra-Pair Instability (IPI) measures how often a model changes its preference when the same two answers are swapped. Weak Total Order Violation (TOV) measures how many pairwise judgments must be fixed to make all preferences logically consistent.

The authors curated a 650 samples dataset from RewardBench2 and WildChat-1M and then tested the 2 proposed metrics on RewardBench2 and LLMBar for human-llm correlation.

### Strengths
The paper’s main strength is its human-free and scalable framework, which removes the need for expensive and subjective human labels while allowing large-scale, repeatable evaluation of LLM judges. The proposed SAGE method offers a clear way to measure how stable and reliable judge models are. 

The experiments are also comprehensive, covering 13 major LLMs (both open and closed models) and analyzing fine-tuned judges and multi-agent systems, giving useful insights into how these factors affect consistency and fairness.

### Weaknesses
1. Generalization to WildChat-1M: The author tested human-llm correlation on A.(RewardBench2) and B.(LLMBar), and the benchmark dataset SAGE they proposed is the combination of A.(RewardBench2) and C.(WildChat-1M). It is unclear whether high correlations between human-llm can be generalized to the WildChat-1M given that there might be domain shifts. 

2. Consistency does not necessarily equal correctness:  a model could be stably wrong. This limits how much the metrics reflect true judging quality beyond internal coherence.

3. Limited interpretability of TOV and IPI scores: The values of these metrics are not intuitively meaningful, and the paper does not define clear thresholds for what counts as a “good” or “bad” judge.

### Questions
How can the human-llm judge consistency generalization from (RewardBench2 and LLMBar) to Wildchat-1M be tested/justified?

### Soundness
2

### Presentation
3

### Contribution
2
