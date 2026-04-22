# SciNav: A General Agent Framework for Scientific Coding Tasks

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6, 4

## Abstract
Autonomous science agents, built on large language models (LLMs), are increasingly being investigated to generate hypotheses, design experiments, and produce reports. Prior science agents primarily focus on open-ended scientific problems, where such outputs—hypotheses, experiments, or analyses are inherently subjective and thus difficult to evaluate rigorously. In contrast, existing scientific coding benchmarks provide tasks with clearly defined, executable outputs that enable objective assessment. However, current agent-based approaches to these benchmarks remain engineering-driven pipelines, lacking principled framework design. This mismatch exposes a gap: the absence of end-to-end, principled science agent frameworks for scientific coding tasks. We address this gap by focusing on scientific coding tasks, where evaluation can be made rigorously, and introducing an agent framework SciNav (Scientific Navigator) that enables more effective solution exploration. Our framework is designed to operate efficiently under constrained search budgets, moving beyond reliance on pre-defined success metrics and prolonged search cycles. Inspired by findings that comparative judgments often reveal finer-grained quality differences and therefore provide greater discriminative power than absolute scoring, our framework leverages pairwise relative judgments within a tree search process to select top-K promising solution branches, prune low-potential ones, and progressively narrow down the solution candidates on the selected branches guided by relative comparisons. We demonstrate our agent's effectiveness across different types of tasks on two benchmarks. Experiments show that SciNav significantly outperforms direct prompting and prior agents like OpenHands and Self-Debug across different base models, task types, and difficulty levels, and exceeds different frontier comparators such as random selection and LLM absolute scoring. These results confirm the strength of our agent design and highlight the effectiveness of relative judgment–guided top-K search for high-quality scientific coding, marking a step toward more practical science agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This study introduces SciNav, an LLM-based agent framework designed for scientific coding tasks. SciNav employs Top-K Comparative Tree Search to address these tasks under constrained computational budgets. Experimental results demonstrate that SciNav surpasses baseline methods, achieving up to a 24% increase in success rate and a 7.8-point absolute improvement in valid execution rate.

### Strengths
-	This paper addresses a clearly defined and underexplored problem: scientific coding tasks, which connect scientific reasoning with executable code generation and allow for objective evaluation.
-	The proposed Top-K Comparative Tree Search introduces a novel use of relative LLM judgments for iterative code refinement. It shows consistent empirical improvements over strong baselines.
-	The paper is well-organized with detailed component analysis, ablation studies, and transparent experimental setup across multiple benchmarks and LLMs.

### Weaknesses
-	The primary concern with this study is the gap between the authors’ claims and their actual contributions: the paper frames SciNav as a general “principled agent framework,” but the work focuses narrowly on code-generation heuristics without formal theoretical grounding or demonstration of broader scientific reasoning.
-	The use of the term “principled” in the title and narrative is inappropriate, as the method is not derived from explicit first principles or formal justification. Currently it is a structured heuristic rather than a principled framework in the scientific sense. I highly recommend the authors to use more appropriate terms.
-	The paper lacks quantitative analysis of computational cost and statistical significance of performance gains brought by the proposed method, making claims of efficiency and reliability only partially supported.

### Questions
Please see comments above. One additional question: can scientific coding serve as a valid representation of scientific reasoning tasks? How does the proposed system enhance scientific reasoning in open-ended environments?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SciNav, an agent for scientific coding problems that frames solution search as a Top-K Comparative Tree Search (TKCTS). Instead of relying on absolute LLM scores or task-specific metrics during exploration, SciNav repeatedly performs pairwise (relative) judgments among candidate programs, prunes low-potential branches, and refines the top-K trajectories via self-debug and self-improvement loops. The system is evaluated on ScienceAgentBench and DA-Code, claiming consistent gains over Direct Prompting, Self-Debug, and OpenHands, with ablations suggesting relative judgments beat random or absolute scoring for frontier selection.

### Strengths
1. Clear problem focus: Targets scientific coding where outputs are executable and evaluable, avoiding the fuzziness of “end-to-end science agents.” 

2. Methodical framing: TKCTS with self-debug/self-improve is a clean, modular agent design; Algorithm 1 is easy to follow and implement. 

3. Relative judgments: Sensible use of pairwise comparisons; the frontier-comparator ablation shows consistent benefits over random and absolute scoring. 

4. Cross-dataset evaluation: Evidence on ScienceAgentBench and DA-Code, with task-type and difficulty breakdowns and an informative error analysis.

### Weaknesses
1. Compute fairness & unclear budgets.
The paper fixes step counts but does not report token, runtime, or dollar budgets across methods. Without normalized compute, it’s unclear if performance gains stem from the algorithm or simply more compute. Reporting tokens / wall-clock / $ per task and re-running under a fixed compute budget is necessary.

2. Small absolute gains and low success-rate regime.
Improvements are modest (e.g., \~2–4% absolute SR gains) and overall SR remains low (\~15–19%), as seen in Table 2. The practical significance is unclear without confidence intervals, per-task breakdowns, or hypothesis testing.

3. LLM-as-judge bias / circularity risk.
If the same model family both generates and judges, relative scoring may reflect stylistic familiarity rather than correctness signal. This risks overestimating benefit of pairwise comparison. Cross-model judging or position-randomization baselines are missing.

4. Baselines not fully representative.
The paper omits competitive selection/reranking baselines such as tournament selection over best-of-N, MCTS with learned value, or verification-guided heuristics. These are relevant comparisons that could challenge the novelty claim.

5. Sparse statistical reporting.
Only 2–3 runs per setting, no reported CIs, no significance tests, and unclear sampling protocol for DA-Code. More rigorous variance reporting is needed, especially given the stochastic nature of LLM benchmarking.

6. Under-leveraged partial execution signals.
The paper emphasizes “no task-specific metric at run time,” but many tasks allow cheap checks (import/compile, partial test subsets, lints). A hybrid relative-judgment + lightweight execution signal could materially improve the agent — and the omission feels like an avoidable limitation rather than a principled choice.

7. Reproducibility gap.
Prompts are included, but code is “will be released upon acceptance.” Given the importance of queueing, frontier selection, and Elo parameters, anonymized code or pseudocode for comparator internals would improve credibility.

### Questions
1. Compute parity:
Can you report tokens, wall-clock time, and $ cost per task for each method? Do results hold under strict compute-budget matching?

2. Judge/model decoupling:
Did you evaluate cross-model judging (e.g., Claude evaluates GPT solutions and vice versa)? If not, please include — this is crucial to rule out style bias.

3. Pair selection policy:
How exactly are candidate pairs chosen for comparison? Uniform random? Score-based? Uncertainty-based? Please add an ablation isolating this choice.

4. Comparator stability:
How sensitive performance is to the Elo update parameters and number of comparison calls? A plot of success-rate vs. comparison budget would help.

5. DA-Code sampling clarity:
How were the 100 DA-Code tasks selected and stratified? Please release task IDs and sampling seeds to enable exact replication.

6. Hybrid signal experiment:
Have you tested combining pairwise judgments with cheap verification signals (static checks, partial tests)? This seems easy to add and directly addresses observed failure modes.

7. Baseline strengthening:
Can you add tournament-selection, majority-vote ranking over best-of-N, or MCTS-style search? If not, please justify why these are not relevant or already covered by TKCTS.

### Soundness
2

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
3

### Summary
The authors propose and evaluate SciNav, a "framework" (agent) that performs scientific coding. The agent performs search taking into account a constrained budget, and uses pairwise (comparative) jugements rather than absolute judgements to guide the search. They show SciNav outperforms prior agents like OpenHands.

### Strengths
- The empirical result of outperforming OpenHands and Self-Debug is quite compelling
 - nice to see search budgets taken into account in the framework
 - nice to see you leveraging existing benchmarks rather than creating a new one
 - ablation that suggests relative judgements are helping (a little bit)

### Weaknesses
- Gains are somewhat modest (~2%-3%), so the impact of the work seems a little limited
 - Comparison with genetic algorithm approaches to coding (e.g., in AI Scientist) would be useful 

Minor:
 - Abstract takes way to long too get to the goal and contribution - should be stated in first or second sentence. (The abstract gives the impression at first you're going to propose a benchmark)
 - Would be worth expanding on use of relative judgements in AI, e.g., it's the basis of A/B testing, preference optimization (e.g., DPO), and other methods.

### Questions
- A common approach in other agent-based coding tasks is to use genetic algorithms to merge different coding ideas (e.g., AIScientist), rather than expanding a single parent. It'd be nice to know how your "single parent" approach would compare. Do you have any intuitions about this?
 - It seems that your contribution is more proposing and evaluating TKCTS as a great framework for agent coding, rather than the (narrower) use of comparative judgements. Is that a reasonable reframing of the contribution?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors focus on improving scientific coding agents by proposing SciNav, a framework that treats problem-solving as a structured search guided by relative evaluation. They introduce Top-K Comparative Tree Search (TKCTS), which allows the agent to explore, compare, and refine code solutions through pairwise relative judgments rather than absolute scoring. SciNav integrates components for planning, self-debugging, self-improvement, and frontier selection using an Elo-based ranking mechanism. The authors evaluate SciNav on scientific coding benchmarks to demonstrate that this principled, comparison-driven approach leads to more effective and reliable solution generation than the baseline agents.

### Strengths
(1) The paper is well-motivated and clearly defines the need for principled frameworks for scientific coding tasks with verifiable outputs.

(2) It presents a structured search method combining relative judgments and iterative refinement, supported by consistent quantitative improvements over existing agent baselines.

### Weaknesses
* Evaluation is limited to two controlled benchmarks, leaving uncertainty about generalization to real-world or open-ended scientific tasks.
* Reliance on LLM-as-judge comparisons may introduce bias, as the same models both generate and evaluate solutions.
* The fixed and narrow search budget restricts exploration, and scalability to more complex tasks remains unclear.

### Questions
* Was a cost or runtime comparison performed to quantify the additional computation introduced by pairwise judgments and iterative search relative to baselines?
* The TKCTS relies on relative judgments by an LLM-as-judge. How consistent are these judgments across multiple runs or judging models? Would cross-model evaluation (e.g., using a different LLM as the judge) yield stable rankings?
* The framework uses a fixed budget (five initial solutions, three debug steps, ten total exploration steps). Why were these values chosen, and have the authors tested sensitivity to these parameters?
* Given that relative judgments guide the search process, was any validation (e.g., human evaluation or ground-truth correctness checks) performed to verify that the LLM-judge’s preferences align with actual code quality?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SciNav (Scientific Navigator), a framework for autonomous science agents designed to tackle scientific coding tasks. The core contribution is the Top-K Comparative Tree Search (TKCTS) algorithm, which replaces absolute scoring with pairwise relative judgments during solution exploration. SciNav integrates several components: initial multi-plan generation, self-debug, iterative self-improvement, and a frontier comparator based on relative LLM judgments, to progressively refine code solutions under constrained computational budgets.

Experiments on ScienceAgentBench and DA-Code show that SciNav performs best compared to baselines such as OpenHands and Self-Debug. Ablations also show that relative comparison helps compared to random selection and absolute scoring.

### Strengths
S1. The relative judgment–guided Top-K search is a well-motivated methodological idea that builds on prior insights about the reliability of pairwise evaluation and applied in an agentic setting.

S2. The experiments are reasonable, covering two benchmarks, several LLM backbones, and detailed component ablations. The experiments for the contributions of each component, including initial plan diversity, self-improvement, and the comparator strategy are appreciated.

### Weaknesses
While the results and experiments are good, my main concerns center around how much we can interpret from them which I'm happy to change with some clarification.

First, the paper does not report error bars or statistical significance. This makes it hard to assess whether observed performance differences are meaningful or consistent across runs.

Second, it is important, especially when we consider deployment to also compare the cost of each agent/ablation involved. How many extra LLM calls/tokens are used for SciNav vs. Self-debug to obtain the performance increases? What is the cost of inference time or $ cost to have extra LLM calls? 

Without this information, it is hard to assess the tradeoff/value of SciNav. For instance, how much of this performance gain is just a result of scaling test-time compute.

### Questions
What is the K in top-K and the comparison budget for the experiments?
How does changing this impact the performance? Given the main contribution is the comparator it would be helpful to see how much performance is impacted by these hyperparamter choices.

### Soundness
2

### Presentation
3

### Contribution
2
