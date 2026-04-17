# AutoLibra: Agent Metric Induction from Open-Ended Human Feedback

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Agents are predominantly evaluated and optimized via task success metrics, which are coarse,
rely on manual design from experts, and fail to reward intermediate emergent behaviors.
We propose AutoLibra, a framework for agent evaluation, that transforms open-ended
human feedback e.g. “If you find that the button is disabled, don’t click it again”, or “This
agent has too much autonomy to decide what to do on its own” into metrics for evaluating
fine-grained behaviors in agent trajectories. AutoLibra accomplishes this by grounding
feedback to an agent’s behavior, clustering similar positive and negative behaviors, and
creating concrete metrics with clear definitions and concrete examples, which can be used for
prompting LLM-as-a-Judge as evaluators. We further propose two meta-metrics to evaluate
the alignment of a set of (induced) metrics with open feedback: “coverage” and “redundancy”.
Through optimizing these meta-metrics, we experimentally demonstrate AutoLibra’s ability
to induce more concrete agent evaluation metrics than the ones proposed in previous
agent evaluation benchmarks and discover new metrics to analyze agents. We also present
two applications of AutoLibra in agent improvement: First, we show that AutoLibra
serve human prompt engineers for diagonalize agent failures and improve prompts iterative.
Moreover, we find that AutoLibra can induce metrics for automatic optimization for agents,
which makes agents improve through self-regulation. Our results suggest that AutoLibra is a
powerful task-agnostic tool for evaluating and improving language agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel agent evaluation framework, AutoLibra, that leverages human feedback to automatically induce evaluation metrics which can be used in LLM-as-a-Judge. AutoLibra incorporates a meta-evaluation to evaluate the quality (coverage and redundancy) of induced metric. The paper also demonstrates using AutoLibra to iteratively improve agent systems.

### Strengths
1. The paper is well-written.

2. The paper studies an important problem in agent evaluation and proposes a interesting and promising solution.

3. The paper propose a novel perspective of using human feedback to induce metrics for agent evaluation.

3. Most design choices, such as which model to choose for each step, are well justified.

4. The paper demonstrates clear effectiveness of AutoLibra in improving agent systems.

### Weaknesses
1. AutoLibra relies on feedback from end users or experts. This creates a major concern about the practical use of AutoLibra. Real-world feedback can be noisy, ambiguous, or high-level. Please clarify the minimum quality and granularity required. Please also compare the overhead of collecting expert feedback against alternatives (e.g., expert‑designed metrics).

2. The construction and evaluation details can be elaborated to strengthen the soundness of this work (see Questions).

### Questions
1. Line 212: What does the “similar results” mean? Which metrics are used to assess similarity? Why does the model used to rate agent trajectories have to have similar results as o3-mini high (the model used in the step of behavior clustering).

2. Line 265: Why do you choose 20 different sets? What if users of AutoLibra uses more or fewer number of sets? How should users set this parameter in practice?

3. Line 319-320: Why is the complexity of the evaluation of social interaction considered high?

4. Section 3.3 and Table 1 show the step-wise agreement scores between AutoLibra and human annotation. How do these numbers translate to end-to-end agreement about the agent performance? Please report the end-to-end agreement scores in cases where such evaluation is necessary?

5. Line 334-335: How were the failure categories proposed by authors/experts? Were they developed independently of the AutoLibra‑induced metrics?

6. In general, what specifications (quality, granularity, quantity, annotator expertise) are needed to achieve the reported performance? How does AutoLibra deal with ambiguous feedback that has multiple possible ways to interpret, or high-level feedback that are hard to ground (e.g., “The agent doesn’t meet my requirements”)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces AutoLibra, a framework which converts unstructured human feedback on agentic trajectories into interpretable eval metrics. This paper would be stronger if a more comprehensive validation of LLM grader performance was provided.

### Strengths
- The loop to extract metrics and automatically measure coverage is novel (from what I have seen). 
- These metrics should help the autograders score new cases (pos and negative examples is a nice touch).
- validated approach on 20% held out set.
- metrics as a function of observed trajectories is a very cool idea.

### Weaknesses
- only 118 trajectories human labelled, with each trajectory only taking 5 mins. This is quite a small sample size IMO. It would be good to see this methods applied to and validated against a larger set.
- This method heavily relies on the LLM performance. The paper should dedicate more ablation studies and effort into current LLM proficiency at this task. 
- given the importance of LLM performance at this task I think a method such as the one detailed here https://arxiv.org/abs/2507.03772, which allows a more detailed investigation into LLM behaviours, should be explored.  
- I think you should cite scalable oversight work as quite related e.g.works such as Constitutional AI (Bai et al., 2022), RLAIF/AI feedback (e.g., Lee et al., 2023/Anthropic) and maybe even some control work. 
- Figure 4 is hard to interpret. e.g. annotation lines and plot lines looking very similar. Is this actual data?

### Questions
see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work provides an evaluation framework for LLM agents that induces fine-grained behavioral metrics from open-ended human feedback, then uses LLM-as-a-Judge to score trajectories, and meta-metrics (Coverage, Redundancy) to select/optimize the metric set. 

Pipeline is as far as I understood is: feedback grounding --> behavior clustering --> metric induction --> LLM-judging --> meta-evaluation (Coverage/Redundancy).

Experiments are conducted on CoGym, Sotopia, WebArena, WebVoyager, and Baba-is-AI and they find >0.85 human agreement for grounding, judging, and meta-evaluation. Results show Coverage peaking at N=6–10 induced metrics, with Coverage reaching 88% on WebArena/WebVoyager. They also show 20% improvement on Baba-is-AI when optimizing using induced metrics rather than success rate directly.

### Strengths
- Produces fine-grained, actionable behavioral metrics (e.g., Access Barrier Handling, Error Recovery and Adjustment, Navigation Accuracy).

- Discovers failure modes that were not captured in pre-existing benchmark taxonomies (e.g., WebVoyager: Query/Search Strategy Efficiency (approx 7%), Final Output Quality ( approx18%)).

- Demonstrates self-regulated improvement: optimizing induced metrics yields ~20% success gain on Baba-is-AI without directly optimizing success rate.

- Step-wise human validation is consistently >0.85, which is a strong reliability signal for an LLM-based pipeline.

### Weaknesses
In general there are not too many strong weaknesses:

- LLM dependence and limited visibility in clustering:

Please report clustering stability: fix the optimal number of metrics (N), run at least three different random seeds, and quantify how similar the resulting metric sets are (for example, by matching clusters and comparing overlap, or by providing a small human-judged semantic comparison across samples).

- Generalization not demonstrated across domains:

A small cross-dataset test would clarify this. For example: induce metrics on WebArena, then evaluate Coverage and Redundancy on held-out feedback from WebVoyager and CoGym without re-introducing the metrics.

If any of these experiments or measurements already exist in the appendix, please point me to the exact section or figure.

### Questions
Questions for Authors:

- Can you include (or point to) a cross-dataset transfer evaluation?

- Can you provide order-of-magnitude compute/token/$ estimates for one induction loop? It seems pretty costly to me. 

- How stable is the induced metric set across random seeds for clustering/induction?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes AutoLibra, a framework that converts open-ended human feedback on agent trajectories into a set of interpretable, fine-grained behavioral metrics, then uses these metrics with LLM-as-a-Judge for evaluation and for agent improvement. 

The pipeline includes (i) grounding free-form feedback to concrete behaviors, (ii) clustering them into metric definitions with examples, and (iii) assessing a metric set with two meta-metrics—coverage (alignment with observed feedback) and redundancy (non-overlap/spuriousness). The authors also demonstrate applications to prompt engineering and automatic self-regulation/optimization of agents.

### Strengths
1. Timely problem & clear formulation. Moving beyond task-success rates to behavioral evaluation is important for LLM agents interacting with humans. The coverage/redundancy meta-metrics give a principled way to select metric sets that actually reflect what people say they want. 

2. Interpretability & reusability. Metric definitions with examples are human-legible and reusable across tasks, which is valuable for human–AI interaction workflows (e.g., aligning internal diagnostics with user-visible rubrics).

3. Closed-loop potential. Showing that the same induced metrics can inform both evaluation and improvement (prompt iteration or self-regulation) helps bridge the usual gap between “LLM as a judge” and tangible agent gains.

4. Practicality. The approach is comparatively lightweight (no bespoke simulators or dense labels) and could plug into existing logs/feedback pipelines.

### Weaknesses
1. Judge-dependence & circularity. Every stage (grounding, clustering, judging) leans on LLMs. Without strong cross-judge and human-only checks, it risks metric drift or Goodhart effects (agents optimize to a judge’s quirks rather than human satisfaction).

2. Sensitivity of the meta-metrics. Coverage and redundancy depend on how “aspects” are extracted and granularized; small parsing or clustering changes could alter the frontier. The paper would be stronger with sensitivity analyses (judge model, prompt, seed, granularity).

3. Causal link to human value. It remains unclear when improving these induced metrics causally improves human satisfaction or task utility, especially in socially nuanced settings; correlations may not hold under domain shift.

4. Generalization & robustness. How stable are the induced metrics across tasks, annotator populations, and time? Are they robust to verbose/hedging outputs or to adversarial behaviors that superficially satisfy rubrics?

5. Cost & reproducibility details. A fuller accounting of inference cost, annotation effort, and step-by-step prompts would help others reproduce and operate this in production settings.

### Questions
1. Missing ralted work. Your pipeline appears sensitive to the evaluation language and judge behavior. Building on findings that evaluation rubrics/languages can themselves shape agent behavior, please (a) discuss how your approach relates to and differs from Wang et al., 2025 (ICML) [1], and (b) provide robustness evidence that induced metrics remain valid under judge swap (different families/sizes), prompt/rubric rephrasings, and formatting/verbosity changes. Can you show that optimizing your metrics causally improves human satisfaction across such perturbations (anti-Goodhart tests, cross-judge A/Bs, and human-only checks)?

2. Robustness to gaming and verbosity. Do agents that learn to be longer, more self-reflective, or apologetic inflate metric hits without better task outcomes? 

3. Noisy/contradictory feedback and metric stability. How stable are aspect extraction, clustering, coverage, and redundancy under label noise, annotator disagreement, and imbalanced feedback distributions? 

4. If metrics are induced on domain A and evaluated on domain B (or earlier vs. later time slices), how do coverage, redundancy, and downstream gains degrade? Do you have a principled refresh/update policy that avoids drift without overfitting to recent logs?


[1] Wang, Z., Zhang, Z., Fang, F. &amp; Du, Y.. (2025). M$^3$HF: Multi-agent Reinforcement Learning from Multi-phase Human Feedback of Mixed Quality. Proceedings of the 42nd International Conference on Machine Learning, in Proceedings of Machine Learning Research 267:65429-65448 Available from https://proceedings.mlr.press/v267/wang25el.html.

### Soundness
3

### Presentation
3

### Contribution
3
