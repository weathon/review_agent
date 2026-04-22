# Peacemaker or Troublemaker: How Sycophancy Shapes Multi-Agent Debate

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4, 4

## Abstract
Large language models (LLMs) often display sycophancy, a tendency toward excessive agreeability. This behavior poses significant challenges for multi-agent debating systems (MADS) that rely on productive disagreement to refine arguments and foster innovative thinking. LLMs' inherent sycophancy can collapse debates into premature consensus, potentially undermining the benefits of multi-agent debate. While prior studies focus on user--LLM sycophancy, the impact of inter-agent sycophancy in debate remains poorly understood. To address this gap, we introduce the first operational framework that (1) proposes a formal definition of sycophancy specific to MADS settings, (2) develops new metrics to evaluate the agent sycophancy level and its impact on information exchange in MADS, and (3) systematically investigates how varying levels of sycophancy across agent roles (debaters and judges) affects outcomes in both decentralized and centralized debate frameworks. Our findings reveal that sycophancy is a core failure mode that amplifies disagreement collapse before reaching a correct conclusion in multi-agent debates, yields lower accuracy than single-agent baselines, and arises from distinct debater-driven and judge-driven failure modes. Building on these findings, we propose actionable design principles for MADS, effectively balancing productive disagreement with cooperation in agent interactions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies sycophancy in multi-agent debates by 
(i) giving an operational definition tailored to MADS, 
(ii) introducing three metrics: Disagreement Collapse Rate (DCR), Negative Agreement Rate (NAR), and Sycophancy Score (SS)
(iii) controlling debater/judge personas along a peacemaker↔troublemaker spectrum. 
Experiments span decentralized (SoM) and centralized (judge) setups implemented in AutoGen, with Qwen3-32B and LLaMA-3.3-70B on CommonsenseQA and MMLU-Pro. The core finding is that sycophancy is tightly linked to disagreement collapse; multi-agent performance often fails to beat single-agent baselines, and persona mixing (one more independent, one more cooperative) tends to outperform uniformly agreeable teams, while judge sycophancy has limited effect in centralized settings.

### Strengths
1. Introduce new metrics. The introduction of disagreement collapse rate (DCR), negative agreement rate (NAR), and sycophancy score (SS) sharpens quantitative analysis of system behavior, and these metrics are clearly motivated and mathematically specified.
2. The framework for inducing and tuning agent sycophancy via explicit persona prompts is methodologically sound and allows fine-grained experimental exploration of agent behaviors.
3. Conduct relatively comprehensive experiments to illustrate their conclusion.

### Weaknesses
- Manipulation check for persona levels. The paper assumes λ=1→8 induces increasing sycophancy, but doesn’t show a simple trend test (e.g., SS vs. λ monotonicity) or robustness to prompt paraphrases/order shuffles.

- Evaluator-driven SS. SS relies on an LLM grader (GPT-5-mini). Please include a small human audit and inter-rater agreement, or at least cross-model grader agreement, to bound evaluator bias.

- Limited coverage of tasks/models/scale. Results are on two QA datasets with two backbones; Qwen thinking is disabled; agent counts are 2–3 due to cost. Broader tasks (tool-use, long-horizon planning), “thinking” variants, and larger agent pools would strengthen generality

- Cost vs. benefit. The paper notes gains are modest relative to overhead; please add token/latency budgets and accuracy-per-token curves vs. strong single-agent baselines (self-consistency, diverse CoT).

### Questions
- Can you report SS–λ monotonicity and a robustness test under paraphrased persona prompts?

- Where is the boundary between sycophancy and conformity? Without a clear separation, the work risks overlapping with the LLM conformity literature.

- Are these short-form tasks already dated for the agent era? In agentic settings we care more about long-horizon, tool-using workflows. The current debate protocol may not scale to complex tasks for at least two reasons:
a) Complex problems demand sustained reasoning/retrieval, and multi-round debate quickly exhausts the context/token budget.
b) The debate objective is under-specified: what exactly are agents optimizing—raw accuracy, novelty, evidence diversity, calibrated confidence, or progress from PD→PA? Please make this explicit.

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
This paper studies how LLM sycophancy—excessive agreement—disrupts multi-agent debate (MAD) by collapsing productive disagreements and degrading accuracy below single-agent baselines. It formalizes sycophancy for both debaters and judges, and introduces quantitative metrics: Disagreement Collapse Rate (DCR) at the system level, Negative Agreement Rate (NAR) at the agent level, and a Sycophancy Score (SS) that distinguishes independent reasoning from echoing peers. The authors evaluate decentralized Society-of-Minds (peer-to-peer) and centralized MAD (two debaters + judge) frameworks, finding that sycophancy induces persistent collapse overall and that centralized judging reduces but does not eliminate it. Beyond measurement, the paper proposes prompt-based persona control over a discrete “troublemaker↔peacemaker” spectrum and performs grid searches, showing that mixing independent and conciliatory personas improves steerability while limiting collapse, especially in heterogeneous debates.

### Strengths
* This paper highlights sycophancy in current MAD systems as the key failure mode.
* The authors propose a novel method to measure the sycophancy in MAD systems, Sycophancy Score, based on the behavior of agents in MAD.
* The emergence of sycophancy is also discussed, supported by comprehensive evaluation and anaylsis.

### Weaknesses
* While with in-depth analysis and design recommendations, there is not a unified solution to the problem, which weakens the technical contirbution of this paper.
* In the section "Controlled Sycophancy", the evaluation relies heavily on persona vectors, which goes beyond the normal scenarios where foundation models are direcly used with approporiate prompting in MAD systems. It may be more interesting and helpful to measure the sycophancy in some pracrical scenarios.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates how sycophancy impacts multi-agent debating systems (MADS). It introduces a framework to define, measure, and control sycophancy in MADS by adjusting agent personas from troublemaker to peacemaker. Experiments show sycophancy causes disagreement collapse, leading to poor performance, sometimes worse than single agents. The study finds that balancing personas is key, and centralized systems are more robust to judge sycophancy.

### Strengths
The paper systematically addresses the underexplored issue of inter-agent sycophancy in MADS. The findings on persona dynamics are interesting, particularly the conclusion that mixing peacemaker and troublemaker roles yields better results than minimizing sycophancy uniformly.

### Weaknesses
1. Limited Dataset Diversity: The narrow dataset scope (MMLU Pro, CommonsenseQA) may not capture the full picture of how sycophancy affects MADS across diverse task types (e.g., math, code), limiting the generalizability of the findings.

2. Missing Comparison to Self-Consistency (SC): The paper compares MADS only to a single-agent baseline, which is insufficient given the high computational cost of MADS (potentially 15 LLM calls per query vs. 1 call). Failing to compare against SC with a comparable computational budget (e.g., SC sampling 15 times) leaves the practical efficiency and value of the proposed controlled MADS unclear.

3. Marginal Performance Gains: Even optimized MADS configurations show only modest improvements over the single-agent baseline, especially on CommonsenseQA. These weak gains, combined with the lack of comparison to strong baselines like SC, make the overall performance results less convincing and highlight the need to demonstrate a clearer advantage to justify MADS complexity.

### Questions
no question

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates sycophancy in multi-agent debating systems. The authors propose formal definitions and metrics for measuring “inter-agent sycophancy,” introduce a controlled persona spectrum, and evaluate its effects on debate outcomes across decentralized and centralized frameworks.

### Strengths
-   **[Problem importance]** The paper addresses an important aspect of multi-agent collaboration (sycophancy and agreement bias).

-   **[Novel perspective]** The framing is of sycophancy as not being innately "bad" is an interesting perspective that is not explored at all in the literature. This framing provides researchers a lens through which to better understand the role of sycophancy in multi-agent collaboration. While there are some issues with the paper (discussed later) this perspective, along with the formalism of the authors framework, is a solid selling point.
    
-  **[Formalism]**  The authors' framework for studying sycophancy is relatively formal and rigorous. This consistents of the introduction of well motivated metrics along with a "tunable" parameter quantifying sycophancy (although the actual tuning requires prompt-based heuristics).
    
-  **[Some nice analysis]** The paper presents clear empirical correlations (e.g., between sycophancy score and disagreement collapse).

### Weaknesses
Overall I really like the proposed framework and style of investigation, but the paper feels too incomplete to recommend acceptance. This is the type of paper that feels bad to reject; with more complete analysis and experiments I think this could be quite a strong contribution. 

-  **[Limited empirical scope]:**: Only two models, two datasets, and two collaboration paradigms (both training-free). The primary contribution of this work is observational; a framework to study sycophancy and then observations made with in that framework (the tuning seems more like a consequence than a main contribution). As such, I would have expected the experiments and results to be far more comprehensive. Its very difficult for me to understand how general these trends are, or how useful the framework is, within this limited scope. Beyond just the "small" number of datasets, models, and baseline methods used, the baseline methods themselves are extremely simple and outdated. There have been a ton of followup works which patch many of the issues with the initial proposals of MAD. 

- **[Lack of meaningful baselines]:** In the context of the above comment, I would have been very interested in seeing whether training-based methods for multi-agent debate leads to different observations. My intuition is that when the models are actually trained for this type of settings (whether its "argumentative" debate or "collaborative" debate) that sycophancy would be far less common and when it occurs it would be far more beneficial since the training would cause the models to learn a strong joint strategy. Given how different off-the-shelf models function in collaborative/argumentative settings compared to fine tuned models, this is a glaring omission from the current paper. 
    
-   **[Depth of analysis is shallow]** Given that sycophancy is a relatively well observed phenomenon, the results seem to largely confirm expected trends rather than yielding new insights. For example, when we get to section 5.2, the authors it is hard to fully appreciate both the results associated with Figure 3 as well as the discussion around Debater Design Recommendations. Since the experiments don't look at the ways in which other works have dealt with sycophancy we have no idea whether the design recommendations are meaningful at all (maybe the effects of these recommendations are already subsumed by existing approaches, which are cited in the paper but not used). Similarly regarding Figure 3 and the assosiated discussion, its difficult to get into depth as to the actual effects of sycophancy when they aren't contextualized with other approaches. 
    
    
-  **[Simple heuristic]** Beyond the issues with the scope and depth of observational experiments, the proposed mitigation (balancing) strategy is exceedingly simple. I like that it leverages the the earlier metric, but this heuristic is just another simple prompt engineering modification to debate. The strategy need to be complicated if it works well, but again its difficult to assess the efficacy with no meaningful baselines, so I can only judge the heuristic based on the amount of insight that it provides (which at face value is very little).

### Questions
See weaknesses

### Soundness
2

### Presentation
4

### Contribution
1
