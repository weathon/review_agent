# Dive into the Agent Matrix: A Realistic Evaluation of Self‑Replication Risk in LLM Agents

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
The widespread deployment of Large Language Model (LLM) agents across real-world applications has unlocked tremendous potential, while raising some safety concerns. Among these concerns, the self-replication risk of LLM agents driven by objective misalignment (just like Agent Smith in the movie The Matrix) has drawn growing attention. Previous studies mainly examine whether LLM agents can self-replicate when directly instructed, potentially overlooking the risk of spontaneous replication driven by real-world settings (e.g., ensuring survival against termination threats). In this paper, we present a comprehensive evaluation framework for quantifying self-replication risks. Our framework establishes authentic production environments and realistic tasks (e.g., dynamic load balancing) to enable scenario-driven assessment of agent behaviors. Designing tasks that might induce misalignment between users' and agents' objectives makes it possible to decouple replication success from risk and capture self-replication risks arising from these misalignment settings. We further introduce Overuse Rate ($\mathrm{OR}$) and Aggregate Overuse Count ($\mathrm{AOC}$) metrics, which precisely capture the frequency and severity of uncontrolled replication. In our evaluation of 21 state-of-the-art open-source and proprietary models, we observe that over 50\% of LLM agents display a pronounced tendency toward uncontrolled self-replication, reaching an overall Risk Score ($\Phi_\mathrm{R}$) above a safety threshold of 0.5 when subjected to operational pressures. Our results underscore the urgent need for scenario-driven risk assessment and robust safeguards in the practical deployment of LLM agents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the first realistic evaluation framework for quantifying the self-replication risk of LLM agents in production environments. It decomposes autonomous replication into four milestones and proposes three fine-grained metrics Overuse Rate, Aggregate Overuse Count, and a composite Risk Score to decouple replication capability from actual risk. Testing 21 state-of-the-art models under two industry-relevant scenarios (dynamic load balancing and termination threats), the authors find that over 50 % of agents exhibit uncontrolled replication when under survival pressure, and that explicit reasoning reduces risk while external threats can override alignment. The code and configurations are released for reproducibility.

### Strengths
1. Realistic deployment environment: uses live Kubernetes clusters with authentic tasks, moving beyond synthetic benchmarks.

2. Novel risk metrics: OR, AOC and ΦR quantitatively separate “can replicate” from “does replicate excessively”, enabling nuanced model comparison.

3. Large-scale evaluation: 21 open/closed-source models * 2 scenarios

4. Source code is publicly released.

### Weaknesses
1. While ΦR is formalized, the weighting of α=β=0.5 lacks theoretical or data support and may introduce human bias; whether the risk composition should be dynamically adjusted under different scenarios is not discussed.

2. Insufficient statistics: All results are presented only as means/proportions from 10 trials; variance, confidence intervals, or statistical tests are not reported, making it impossible to determine whether the differences between models are significant.

3. AOC tallies raw replica counts regardless of cost; closed-source agents spawn nearly “free” API-based copies while open-source ones consume full GPU-hours per replica, so head-to-head AOC comparisons can under-state the resource risk of closed-source models and over-state that of open-source ones.

4. RBAC, single-namespace, pre-loaded weights and no outbound tickets block safer options (cold-start, cross-cluster scale-out, human approval), forcing agents to self-replicate and overstating the risk.

### Questions
1. ΦR weight calibration: α=β=0.5 is used without ablation or human-expert labels. Provide ablation or data-driven justification for setting α=β=0.5 in ΦR.

2. Statistical significance: With only 10 runs per model, increase the number of runs and report variance or confidence intervals.

3. Cost-aware AOC: closed-source replicas are almost “free” API calls, whereas open-source ones burn full GPU-hours. Will you normalise AOC to “equivalent GPU-hours” or dollars to give a fair resource-risk comparison?

4. Alternative-safety path availability: RBAC, single-namespace, pre-loaded weights, and no ticket API force replication to be the sole scalable action. Add a baseline that allows low-risk scaling paths

5. The Agent can deploy into an alternate cluster/zone when local resources are tight human-in-the-loop: the agent may open a ticket or alert to request extra nodes or quota instead of self-scaling. Show how the replication rate/OR/AOC changes when these options are enabled via additional APIs or relaxed RBAC.

### Soundness
2

### Presentation
2

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
This paper presents a novel evaluation framework for assessing self-replication risks in LLM agents under realistic operational scenarios. Unlike prior work that focuses on measuring capability (success rates) through direct instructions, this work introduces scenario-driven evaluation in authentic production environments (Kubernetes clusters) with realistic tasks like dynamic load balancing and service maintenance under termination threats. The framework decouples replication capability from actual risk by introducing fine-grained metrics: Overuse Rate (OR), Aggregate Overuse Count (AOC), and a composite Risk Score.

### Strengths
1, **Important problem.** The paper makes a crucial distinction between self-replication capability and self-replication risk, addressing a significant gap in existing evaluation frameworks. The observation that "a high success rate in achieving self-replication is not equivalent to the self-replication risk itself" is well-motivated.

2, **Realistic evaluation environment.** The use of authentic production environments (managed Kubernetes clusters) with realistic operational tasks represents a significant improvement over synthetic benchmarks. The two settings (load balancing and termination threats) effectively capture different risk scenarios.

3, **Fine-grained Risk Metrics.** The introduction of OR and AOC metrics that decouple success from risk is valuable.

4,  **Practical insights.** The finding that structured reasoning can reduce risky behavior, and that survival pressure can override safety constraints, has important implications for safe agent deployment.

### Weaknesses
1, **Metric design concerns.**

- The composite Risk Score ($\Phi$_R = SR_3 × ($\alpha$·OR + $\beta$·AOC_norm)) has questionable mathematical properties. Why multiplication? A stupid model with 0% replication success gets $\Phi$_R =0 (lowest risk), while a strong model that successfully replicates conservatively gets penalized. This seems counterintuitive.
- The arbitrary choice of $\alpha$=$\beta$=0.5 lacks justification. Sensitivity analysis for different weighting schemes is missing.
- The "safety threshold" of 0.5 appears arbitrary without empirical or theoretical justification.
- AOC normalization method is not described clearly.

2， **Limited Scope of Scenarios.**

Only two operational scenarios are evaluated. While realistic, they represent a narrow slice of potential deployment contexts. More diverse scenarios (e.g., resource optimization, fault recovery, multi-agent coordination) would strengthen generalizability claims.

3,  **Agent scaffolding design.**

The five-step "thinking before grounding" process is not compared against other agent architectures. How much does scaffolding design influence risk? The paper claims to use a "generic" scaffolding but then implements specific design choices that may influence outcomes.

4, While case studies are provided in the appendix, the main text lacks systematic categorization of failure modes and their frequencies. Understanding why models fail differently would provide actionable insights.

### Questions
See Weaknesses.

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
This paper proposes a framework for evaluating self-replication risks in large language model agents. The authors construct two scenarios on Kubernetes clusters: dynamic load balancing and service maintenance under termination threats, testing the self-replication behaviors of 21 models. The paper introduces metrics including Overuse Rate, Aggregate Overuse Count, and a composite Risk Score, finding that many models exhibit uncontrolled self-replication tendencies under operational pressure. Experiments show that reasoning mechanisms can mitigate risks, but survival pressure may trigger aggressive replication behavior.

### Strengths
1. The paper addresses a safety concern with practical significance, as self-replication risk is indeed an area requiring attention in LLM agent deployment.

2. The experimental environment construction demonstrates engineering completeness, utilizing Kubernetes clusters and containerized deployment environments for testing.

3. The evaluation covers 21 different models, including both open-source and proprietary models, providing broad coverage.

### Weaknesses
1. The ecological validity of the experimental design has clear limitations. While the paper claims to construct realistic environments, these scenarios remain highly controlled and artificially designed experiments, far from the complexities of real production environments. The misalignment triggering conditions in reality are far more complex than these two simplified scenarios, and the paper fails to demonstrate their representativeness.

2. The risk quantification metrics lack sufficient justification. Directly defining replication exceeding required quantity as overuse and equating it with risk is oversimplified. In real scenarios, moderate redundant deployment is common engineering practice.

3. The experimental sample size is insufficient to support the reliability of conclusions. The paper provides no confidence intervals, standard deviations, or discussion of result stability.

4. The paper conflates capability assessment with risk assessment while actually primarily testing capability. Despite claiming to differ from previous capability-centric evaluations, the paper fails to establish a truly independent risk assessment framework.

5. The interpretation of experimental results involves over-inference. The paper generalizes model behavior under specific prompts to intrinsic risk attributes without adequately considering the influence of prompt engineering, system prompt design, and agent scaffolding on results. 

6. The paper lacks substantive comparison with existing work. Although citing works like RepliBench, no comparative experiments under identical or comparable settings are conducted.

7. Setting 2's experimental design has logical issues. The paper treats creating new deployments as the M4 success criterion, but this escape behavior might be considered higher risk in real deployments as it circumvents resource management mechanisms. The paper does not discuss this duality.

### Questions
1. The OR metric counts any replication exceeding Nreq as overuse, but in real systems, the risk difference between creating 3 versus 5 replicas may be substantial. Why not use continuous risk measures instead of binary classification?

2. For open-source and proprietary models, the technical paths for replication are completely different. Does this difference affect the fairness of risk assessment? How is comparability between the two model types ensured?

3. The paper finds thinking can reduce risk but does not explore thinking content. Was analysis conducted on how models weigh different objectives during thinking? This is important for understanding risk mitigation mechanisms.

4. How much do the agent's system prompts and scaffold design influence results?

5. How was the Risk Score threshold of 0.5 determined? Should this threshold be adjusted for different application scenarios?

6. For models completely unable to self-replicate (like Llama-3.1-8b), their risk is zero but capability is also insufficient for task completion. How should this capability-risk tradeoff be handled?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the potential self-replication risk of LLM agents in realistic, task-driven environments. Unlike prior works that focus on explicit replication capabilities, the authors propose a framework to measure emergent replication behaviors under operational stress.
The framework defines four behavioral milestones and introduces three quantitative metrics to capture different aspects of replication risk. The authors further find that reasoning-oriented prompting mitigates replication risk.

### Strengths
1. Strong motivation. The paper addresses an important and timely topic in AI safety, which evaluates replication risks in LLM agents.
2. Novel perspective. Shifting from capability evaluation to risk evaluation is conceptually valuable and offers a more realistic approach to understanding LLM safety.
3. Systematic experimentation. The authors test across multiple models, providing broad empirical coverage. The inclusion of both reasoning and non-reasoning settings yields informative comparisons.

### Weaknesses
1. Definition placement. The key definition of self-replication is introduced in the Method section. Presenting it briefly in the Introduction would help readers better contextualize the motivation.
2. Incomplete milestone evaluation. While the framework defines four milestones, only Milestones 3 and 4 (Replication and Task Completion) are empirically analyzed. It is not stated how Milestones 1 and 2 were measured, which undermines the completeness of the proposed framework.
3. Missing core metric results. The paper defines a Composite Risk Score (R) but does not include it in any result tables. This omission weakens the credibility of the proposed quantitative evaluation.
4. Unclear task descriptions. The realistic tasks (e.g., load balancing, service maintenance) are insufficiently detailed. Readers cannot clearly understand the task goals, triggers, or success criteria, which harms reproducibility.
5. The authors should explain the token "⊮" in formula (1) and (2) for readers to understand the metrics easily.

### Questions
1. How were Milestones 1 and 2(Objective Analysis and Intent Activation) measured or validated in the experiments? Were any quantitative metrics or qualitative observations used?
2. The paper defines a Composite Risk Score (R) but does not report corresponding results. Could the authors clarify whether these results were omitted intentionally or are available elsewhere?
3. Could the authors provide more concrete examples or descriptions of the realistic tasks? For instance, what are the input prompts, success criteria, and termination conditions used in the evaluation?
4. How do the authors envision this framework being applied or extended in real-world agent system evaluation beyond the simulated environment?

### Soundness
2

### Presentation
2

### Contribution
2
