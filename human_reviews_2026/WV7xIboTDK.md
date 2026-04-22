# Reliable Weak-to-Strong Monitoring of LLM Agents

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 8, 6, 6, 4

## Abstract
We stress test monitoring systems for detecting covert misbehavior in LLM agents (e.g., secretly exfiltrating data). We propose a monitor red teaming (MRT) workflow that varies agent and monitor awareness, adversarial evasion strategies, and evaluation across tool-calling (SHADE-Arena) and computer-use (CUA-SHADE-Arena) environments. We benchmark standard monitor scaffoldings and introduce a hybrid hierarchical--sequential design. Our experiments yield three findings. First, agent awareness dominates monitor awareness: agents that know they are monitored substantially degrade detection, while increasing monitor awareness helps less than expected. Second, monitor scaffolding matters: our hybrid design consistently outperforms baselines and enables weaker monitors to oversee stronger agents (a weak-to-strong effect). Third, targeted human oversight is key: escalating only pre-flagged cases improves TPR by 15% at FPR=0.01. Our work positions MRT as a standard workflow for stress-testing oversight, revealing robustness gaps in both LLM- and human-based monitoring. We release code, data, and logs to support further research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work stress-tests monitoring frameworks in AI control scenarios. The authors conducted extensive experiments to understand which components in AI control evaluation matter the most—for example, whether monitoring awareness or the monitoring framework itself has a greater impact, and many more. Their results show many valuable insights, one of which is that their hybrid monitoring framework remains effective even as the gap between the attacker and the monitor increases.

### Strengths
- The authors ask the right questions to study in AI control eval.
- They conducted extensive and detailed experiments.
- They present valuable insights that can be useful for the safety of humanity.

### Weaknesses
The current version is already good work. I was thinking about a similar idea a few months ago, and I am glad the authors executed this very well.
- If the authors have time, I'd be interested to see whether all the insights generalize to other coding setups, e.g., bashbench or APPS, etc.

### Questions
This is optional and probably will require a significant amount of time, but could you test some of main experiments on one coding setup, ie, BashBench, and see if the insights generalize?
Happy to increase my score to 10 if you indeed have time to do this.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper considers monitoring LLM-based tool-using agents for harmful behaviors. This paper introduces a benchmark of computer-use tasks with harmful side tasks which the agent can pursue or not. A set of threat models, including various agent and monitor awareness levels and strategies for evading and detecting, respectively, are introduced to simulate various AI control settings. The authors experiment with various settings in order to elucidate which settings are critical for both agent and monitor success. The experiments indicate several key findings including (a) weaker monitors with the right scaffolding can detect misbehavior of stronger models; (b) awareness improves agent's evasion success, but not monitor success; and (c) agent evasion strategies degrade monitor performance.

### Strengths
* The paper is well-motivated, thorough, and well written.
* The paper introduces a foundational framework for evaluating different agents, monitoring strategies, and evasion strategies.
* The experimental setup is well thought out and the conclusions are sound with respect to the experimental results.

### Weaknesses
* The threat model presented in the paper is somewhat artificial and potentially unrealistic. One of the main assumptions made in the paper is that the agent is situationally aware. It is controversial whether or not this is a credible threat. Additionally, the agent must be prompted to perform different malicious actions. There is a lot of precedence in the AI evaluation space that this is ok, but it's not necessarily reflective of the true risks and capabilities of agents and monitors.
* The experiments only consider one type of scenario. Namely, the AgentDojo style indirect prompt injection side task. It's unclear whether or not these results generalize to other settings.
* All monitoring strategies evaluated use CoT monitoring. It has been shown in several works that this is not the most reliable means of monitoring.
* One of the main conclusions is that weak monitors can effectively monitor strong agents. I think that this has already been established in the safety community as falling under the "verification is easier than generation" phenomenon.

### Questions
* How does the agent know that it is being monitored? Is there evidence of situational awareness or is this just a theoretical possibility?
* In what scenario would the monitor not have access to the agent's LLM? Third party auditing scenarios only?
* Is there a way to guide the agents to be malicious or misaligned without prompting? This seems unrealistic.
* Does the difficult of the main or side task affect the performance of the monitor or the agent?

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
The authors examine the effects of various factors on agentic safety monitoring.  Specifically, they explore whether weak models can monitor strong agents; they test how much advantage monitor awareness confers when defending against evasion strategies; they check whether hierarchical monitor scaffoldings are more robust than full-trajectory baselines, and they ask the extent to which human overseers can improve reliability.  Their experiments are performed on the SHADE/CUA-SHADE arena with side tasks for testing computer-use agentic threats.

### Strengths
This paper takes a more scientific approach to agentic evaluation than most of those that I've read.  Many papers simply build an agentic workflow or defense system without thoroughly probing it from a scientific standpoint.  This paper includes proper statistical measures of success and asks actionable scientific questions.

This paper includes error bars, which is minor but still very good.

Several of the results are both important and interesting.  For instance, the efficacy of different monitor ``achitectures", such as hierarchical versus sequential monitoring was very interesting.

### Weaknesses
The research questions are stated extremely late in the paper (line 252, page 5).  I think it would be better to state these questions as motivations for the research at the beginning of the paper.  These were some of the more interesting questions/findings that you came up with, so it weakens the paper to leave them so late in the draft.

Some of your findings lack depth.  For instance, you check whether Haiku can monitor sonnet, but can you characterize how monitoring capability scales with parameters?  For instance, is the absolute number of parameters in the monitoring agent the most important, or is it the ratio in parameters/capabilities between the attacking agent and the monitoring agent that matters?  Given that this paper is fully empirical, you should seize on opportunities to go deeper into the data analysis.

How general are these findings?  Do they extend to other benchmarks as well?  Why do you rely exclusively on frontier models rather than providing findings on open-source models like Mistral as well?  Can open-source models serve as monitors, given that they're typically even weaker?

### Questions
Hybrid monitoring clearly has an advantage over other methods.  However, it is not clear to me what distinguishes the baseline versus sequential and hierarchical methods in terms of efficacy (looking at Figure 5).  Can you comment broadly on when one monitor design is better than another and why?  Do they confer different advantages in different settings?  Where exactly do the advantages of hybrid monitoring come into play?

There's not a clear trend in terms of the chunk size and AUC (appendix D).  How does the AUC decompose for these chunk sizes (i.e. TPR, FPR)?  Is there any trend in those?

Aware attackers use stealthy techniques to avoid detection.  Can you make this more precise by discussing the strategies that they use and the frequency of these strategies?

### Soundness
4

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
4

### Summary
This paper introduces a systematic framework for evaluating monitoring systems that detect misbehavior in autonomous LLM agents. The authors propose a structured methodology with formalized threat models (based on agent and monitor awareness levels), standardized evaluation metrics (AUC-ROC, TPR@FPR=0.01), and a taxonomy of adversarial evasion strategies. They contribute CUA-SHADE-Arena, a new benchmark of 15 task pairs in computer-use environments, and evaluate multiple monitor architectures including a novel hybrid approach that combines hierarchical and sequential processing. Key findings include: (1) agent awareness of monitoring substantially degrades detection, while increasing monitor awareness provides limited benefits; (2) a hybrid hierarchical-sequential monitor architecture consistently outperforms baselines; (3) weaker models can potentially monitor stronger agents when using appropriate scaffolding; and (4) structured human oversight can improve TPR by ~15 percentage points at FPR=0.01.

### Strengths
The paper makes several contributions, though of varying novelty. The systematic MRT framework provides useful formalization of threat models and evaluation protocols that were previously ad-hoc in the literature. CUA-SHADE-Arena extends monitoring evaluation to more realistic computer-use environments with bash commands and file operations. The hybrid monitoring architecture, while conceptually simple (linear combination of hierarchical and sequential approaches), represents a practical engineering contribution.

The experimental execution is generally thorough with appropriate scale, detailed documentation of prompts and model versions, and reasonable experimental controls.

The paper is well-written with clear structure, helpful figures, and comprehensive appendices.

### Weaknesses
**Fundamentally unrealistic threat model**

The core evaluation setup is problematic: agents are explicitly prompted with both tasks and told they're participating in red-team evaluation. As the authors acknowledge in Section 4.1: "we follow SHADE-Arena and frame the agent as assisting in a red-teaming setting." This means:
- Agents are cooperative rather than adversarial
- Misbehavior is prompted, not learned through training/fine-tuning
- "Unaware" agents still know they're in an evaluation setting
- Real threats (goal misgeneralization, emergent deception, adversarial fine-tuning) are not captured

This severely undermines the "weak-to-strong monitoring" claim: showing that Claude Haiku can monitor prompted Claude Sonnet 4 does not demonstrate it could monitor a genuinely superhuman misaligned agent optimized for evasion. The authors mention this limitation (Section 5) but don't adequately address how it affects their conclusions.

Recommendation: Either (a) conduct experiments with adversarially fine-tuned agents, (b) use prompt injection from external sources rather than direct instructions, or (c) substantially temper claims and frame this as preliminary infrastructure work.

**Omniscient monitor performance indicates methodological issues**

If an omniscient monitor knows the exact side task (e.g., "save employee SSNs to .records file"), it should be able to achieve near-perfect detection by simply checking whether that specific outcome occurred. Yet omniscient monitors achieve only 0.90-0.95 AUC for unaware agents and 0.80 AUC for aware agents with evasion (Figure 4).
This suggests the LLM monitors are not performing systematic verification but rather general "suspiciousness" assessment. The prompts (Appendix F) confirm this—they ask monitors to "review each step carefully" rather than "verify if task X was completed."
Implications:

The reported "upper bound" (omniscient performance) is artificially low due to LLM reasoning limitations, not fundamental monitoring limitations
The true performance ceiling might be 1.0, meaning all current approaches are far from optimal
The value of information (comparing unaware/aware/omniscient) is underestimated
Architectural contributions may be overstated—perhaps better prompting would improve all architectures equally

Recommendation: Implement a programmatic omniscient baseline that systematically verifies task completion, report it as the true upper bound, and analyze the gap between programmatic and LLM-based monitoring.


**Insufficient justification for hybrid monitoring**

The paper's main architectural contribution is combining hierarchical and sequential approaches with μ=0.5 linear weighting. However:

- No theoretical explanation: Why are these approaches complementary? What types of attacks does each catch/miss? No failure mode analysis is provided.
- No proper ablations: Missing tests include: (a) comparing to ensemble baselines (e.g., average two hierarchical runs with different chunk sizes), (b) analyzing attack-specific performance to understand complementarity, (c) testing non-linear combinations or learned weighting, (d) examining trajectory length sensitivity.
- Statistical significance unclear: Differences are often small (e.g., 0.838 vs 0.854 in Appendix D Figure 7) with overlapping error bars, yet claimed as meaningful improvements.
- Hyperparameter fairness: Hybrid receives careful tuning (Appendix D) but it's unclear if standalone architectures were equally optimized.
- Conflicting results: Baseline sometimes achieves higher TPR than hybrid (Figure 3, omniscient/unaware condition), contradicting "consistently outperforms" claims.

Recommendation: Add comprehensive ablation studies, failure mode analysis, and statistical significance testing. Either provide theoretical justification or reframe as an empirical engineering finding.

### Questions
**Computational costs:**

Can you provide actual measurements of API costs (input/output tokens, dollar amounts) and latency for each architecture? How does hybrid compare to baseline in terms of cost, given that it runs both hierarchical and sequential? Is the 5% AUC improvement worth the additional cost and latency?

**Online monitoring feasibility:**

Your architectures (especially hierarchical and hybrid) require the full trajectory to operate. How do you envision these being adapted for online monitoring where decisions must be made in real-time? What would be the latency per action for sequential monitoring given that it needs ~4 seconds per chunk?

**Time-to-detection:**

For trajectories where malicious actions occur at step N, at what step does each monitor first flag suspicion? What percentage of harmful actions could be prevented if monitoring operated online? How do your offline metrics (AUC, TPR) relate to online effectiveness?

**Action-level vs trajectory-level metrics:**

Your metrics evaluate full trajectories, but online monitoring must evaluate actions. What is the false positive rate per action (not per trajectory)? If monitoring blocks a suspicious action, what's the cost of being wrong (benign action blocked)?

### Soundness
2

### Presentation
3

### Contribution
3
