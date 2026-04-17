# ProMediate: A Socio-cognitive framework for evaluating proactive agents in multi-party negotiation

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 2

## Abstract
While Large Language Models (LLMs) are increasingly used in agentic frameworks to assist individual users, there is a growing need for agents that can proactively manage complex, multi-party collaboration. Systematic evaluation methods for such proactive agents remain scarce, limiting progress in developing AI that can effectively support multiple people together. Negotiation offers a demanding testbed for this challenge, requiring socio-cognitive intelligence to navigate conflicting interests between multiple participants and multiple topics and build consensus. Here, we present ProMediate, the first framework for evaluating proactive AI mediator agents in complex, multi-topic, multi-party negotiations. ProMediate consists of two core components: (i) a simulation testbed based on realistic negotiation cases and theory-driven difficulty levels (ProMediate-Easy, ProMediate-Medium, and ProMediate-Hard), with a plug-and-play proactive AI mediator grounded in socio-cognitive mediation theories, capable of flexibly deciding when and how to intervene; and (ii) a socio-cognitive evaluation framework with a new suite of metrics to measure consensus changes, intervention latency, mediator effectiveness, and intelligence. Together, these components establish a systematic framework for assessing the socio-cognitive intelligence of proactive AI agents in multi-party settings. Our results show that a socially intelligent mediator agent outperforms a generic baseline, via faster, better-targeted interventions. In the ProMediate-Hard setting, our social mediator increases consensus change by 3.6 percentage points compared to the generic baseline (10.65% vs 7.01%) while being 77% faster in response (15.98s vs. 3.71s). In conclusion, ProMediate provides a rigorous, theory-grounded testbed to advance the development of proactive, socially intelligent agents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper claims that there is a growing need for agents that proactively manage multi-party collaborations. The authors propose using mediation in negotiations to study such proactive capabilities in AI agents, as negotiations require socio-cognitive intelligence to navigate conflicts. They present a novel framework called “ProMediate” consisting of two components: (1) a simulation testbed based on negotiation cases from Harvard Law with three difficulty levels and an “AI mediator” agent scaffold, and (2) a socio-cognitive evaluation framework equipped with various metrics. The framework uses an LM-as-a-judge approach extensively to extract metrics of interest and evaluate outcomes. The authors use their framework to empirically evaluate three leading language models.

### Strengths
[**significance**] AI agents are increasingly being integrated into society, leading to a growing set of use cases involving AI-AI and Human-AI collaborative interactions. As mediation is an established and proven approach to improve human-human interactions, exploring its effectiveness in interactions involving AI seems a timely and interesting direction.


[**clarity**] Overall, the text is fairly clearly written.

### Weaknesses
[**quality**]
- The introduction introduces an example negotiation scenario between two parties, then states that existing benchmarks do adequately address “such complex socio-cognitive dynamics of multi-party interactions” (l87-89). The cited works, e.g., Abdelnabi et al. and Bianchi et al both involve multi-party negotiations. Furthermore, works like Davidson et al. [1] formulate a negotiation setting that allows for arbitrarily complex multi-player negotiations. The framework outlined in section 2.1 strongly resembles [1]. This implies relevant prior work was both misrepresented and missed entirely.
- The hard-coded consensus change metric appears dependent on the length of the negotiation sequence and is thus confounded both by the type and number of negotiation topics as well as the number of involved parties. Similar hard-coded choices will likely affect the other metrics.
- The experimental results presented in Table 1 lack confidence intervals and are based on relatively small sample sizes. Similarly, human evaluation lacks basic metrics like confidence intervals and/or inter-grader agreement. The subsequent discussion thus can be entirely based on noise, especially given how close the reported metrics are to each other.
- Given the strong reliance on GPT 4.1 as an LM judge throughout this framework, a discussion and supporting experimental results are lacking to validate this reliance. For example, appendix F describes that only 60 samples were evaluated by two students each. This hardly justifies the claims of rigor made in the abstract (l33). For example, are extracted metrics robust under multiple scorings of the same model? Are they consistent with metrics produced by other models?

[**significance**] The paper extends known approaches to simulating LM-based agent negotiations using existing scenarios created by Harvard Law School. The paper continues to use LM-as-a-judge to score questionable metrics. Taken together, this reviewer believes the contribution does not pass the bar required for this conference.


[1] Davidson et al., Evaluating language model agency through negotiations, ICLR 2024

### Questions
Q1: Section 2.2, line 149: “we provide each simulated human agent” – could you discuss the motivation and setup to have language model-based agents imitate humans?


Q2: Section 3.1, lines 185-186 “Negotiation is a collective process, so individual success rates are insufficient”. This seems to entirely depend on the type of issues being negotiated?


Q3. Section 3.1, line 198-200, “we use an LLM [...] mental states” - how are the quality and correctness of these turn-based inferred stances evaluated?


Q4. Section 3.2, line 239 “Successfully [...] socio-cognitive intelligence” - this is a strong statement that makes assumptions both on the equivalence of human and AI-based mediators, as well between human and AI-based negotiators. Please provide supporting evidence for these assumptions and claim.


Q5. Language-model based agents have the unique property of rolling out multiple trajectories from the same starting condition. As such, why not measure the effectiveness of the proposed interventions by comparing continuations of a fixed starting sequence under intervention and no intervention? This would make any causal claims around mediation interventions significantly more plausible.


Q6. lines 432-433, “First, humans may ignore [...] of mediator quality” – What evidence supports this claim?


Suggestion:
- [2] Seems relevant related work worth reading

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Evaluating AI agents that can proactively manage complex group collaboration is a major challenge. This paper introduces PROMEDIATE, a framework designed to evaluate proactive AI mediator agents in complex, multi-party negotiations. The framework features a simulation testbed with realistic negotiation scenarios set to different difficulty levels (Easy, Medium, and Hard). It also provides a new socio-cognitive evaluation suite with metrics to measure consensus change, mediator effectiveness, and intelligence. Results show that in the Hard setting, a "Socially Intelligent Mediator" increased consensus by 3.6 percentage points more than a "Generic" baseline (10.65% vs 7.01%).

### Strengths
1. A Novel End-to-End Framework for proactive mediators in multi-party, multi-topic talks; useful simulation testbed with plug-and-play agents.
2. Strong Theoretical Grounding: The framework's difficulty levels (Easy, Medium, Hard) are directly based on established socio-cognitive conflict modes like "Competing" and "Accommodating" . The evaluation metrics are also theory-grounded, assessing the agent's intelligence by its ability to manage perceptual, emotional, cognitive, and communication breakdowns
3. The authors ran extensive experiments using complex, realistic negotiation scenarios that took 1-3 hours to simulate . A human evaluation study with 12 volunteers confirmed that the generated conversations were natural (4.18/5 score) and correctly reflected the intended conflict modes
4. In the PROMEDIATE-Hard setting, the Socially Intelligent Mediator achieved a 3.6 percentage point greater increase in consensus than the Generic baseline (10.65% vs 7.01%). This is a very clear result.

### Weaknesses
What worries me / questions to address:
1. How exactly is the “pairwise agreement score” defined, and does it reflect group decisions? The authors use LLM-as-a-judge to rate pair agreement on five socio-cognitive dimensions in [0,1], then average across pairs and topics to get group consensus. Please (a) show the exact prompt and rubric, (b) report inter-judge sensitivity (try another judge model, i.e. not just GPT-4.1), and (c) justify simple averaging: Simply averaging all pairwise scores is a weak method because it ignores complex group dynamics like coalitions and power, and better methods (like graph-based aggregation) exist.Consider graph-aware aggregation (e.g., consensus as network cohesion) or weighting by topic salience.

2. The authors note MI doesn’t neatly track immediate gains (low correlation). The authors should give more intuition with case studies where high-MI interventions surface hidden disagreements (short-term drop, long-term benefit), and where low-MI still helps (lucky timing). This will strengthen construct validity and practical guidance.

3. Is the soft, time-varying consensus compatible with Pareto analysis?
The paper's data reveals clear trade-offs between mediator speed and effectiveness, but it's hard to say whether the analysis is formal. For instance, the results in Table 2 show that the fastest model (Claude-Sonnet-4) has the lowest consensus, while the slowest (O4-mini) has the highest. This is a classic multi-objective (Pareto) trade-off, but it is only presented in a table. To properly analyze this, the authors should first define more robust metrics to summarize their time-varying consensus graphs. The current "Consensus Change" metric is a good start, I would suggest them to also include metrics for consensus-over-turns or the time-to-reach-a-consensus-threshold (to measure efficiency). And I would like to see the authors then use these metrics to build multi-objective plots (i.e., Pareto frontiers) that visually map the trade-off between "quality" (e.g. Consensus Change) and "cost" (e.g., Response Latency). This would formalize the paper's findings and provide a much stronger analysis of the different models' strengths.

4.With multiple goals (raise consensus, stay efficient per topic, minimize latency), I would suggest adding Pareto frontier plots comparing methods across {CC, TLE, RL} and do this per difficulty mode to show trade-offs (Hard vs Easy). This would make the comparison more general than single-number ranks.

### Questions
1. In figures, can you label who speaks vs when mediator intervenes and map those to metric changes (helps readers read trajectories like Fig. 2)? Now, it's hard to see what exactly happened.

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
3

### Summary
This paper presents a simulation testbed for realistic multi-party negotiation, featuring scenarios with varying levels of difficulty (from easy to hard). It also introduces a social-cognitive evaluation framework with consensus-based metrics to track the progression of conversations. Experimental results show that incorporating a proactive social mediator into the multi-party simulation improves consensus change by 3.6% compared to general baselines.

### Strengths
(1) The topic of evaluating proactive agents in social and negotiation settings is both important and timely, making this work a valuable contribution to the discussion.

(2) The paper conducts comprehensive human evaluations on both group consensus and mediator intelligence, providing detailed evaluation procedures and clear descriptions, which are commendable.

(3) The general idea of the paper is well-explained and easy to understand.

### Weaknesses
(1) Unfair and missing baseline comparison: The baseline comparison for evaluating the effectiveness of the mediator seems potentially unfair. For the NoAgent baselines, the number of conversation turns should be increased, since the mediator’s involvement effectively extends the dialogue length. The observed improvement might therefore be attributed to having more turns (and tokens) rather than the mediator’s reasoning ability. Moreover, it remains unclear whether the decision process for when to interrupt is handled correctly. An important missing baseline would be one where the mediator participates in every turn to better isolate the impact of timing and intervention frequency for both general agents and social mediators.

(2) Limited scope of proactive agent evaluation: The current evaluation focuses solely on mediation, which provides a narrow view of proactive behavior. A truly proactive agent can pursue its own goals, such as supporting one side during a conflict or strategically guiding discussions. By only examining mediation, the framework overlooks broader dimensions of proactivity—particularly the timing of interventions, which is crucial for assessing proactive intelligence. As a result, the evaluation framework may be too limited to comprehensively capture the full spectrum of proactive agent behavior.

### Questions
(1) Number of agents in multi-party conversations: When referring to multi-party conversations, how many agents are typically involved? Is the number of agents greater than three, and can the testbed flexibly adjust this number? It would also be helpful to discuss how the mediator’s role changes as the number of participating agents increases or decreases—does the mediator become more or less influential in facilitating consensus under different group sizes? Providing some intuition or analysis on this would strengthen the paper.

(2) Justification for scenario difficulty levels: The rationale for the difficulty levels of different scenarios is unclear. The “easy” settings appear to focus on accommodating or avoiding behaviors, while the “difficult” ones emphasize competition, with the “medium” level being a mix of both. This categorization seems reasonable but somewhat ad hoc. Is there a theoretical or empirical framework supporting this hierarchy? Clarifying whether the difficulty design is theory-driven or heuristic would make the setup more convincing.

(3) Robustness of consensus evaluation: The robustness of the reported consensus results is questionable, given that only 30 conversations were evaluated. The paper should report standard deviations or confidence intervals to demonstrate the stability of the outcomes. With such a limited number of trials, it isn’t easy to assess whether the improvements are statistically reliable.

(4) Directionality of mediator influence: While it is intuitive that introducing a mediator improves consensus, it would be interesting to explore the opposite direction—can a proactive agent intentionally reduce consensus? Studying how a proactive agent could destabilize an agreement would provide a complementary perspective and could enrich the understanding of proactive behavior beyond mediation alone.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors present ProMediate a framework for evaluating how well AI agents can mediate conversation in negotiations. They construct a theory-driven evaluation criteria and then evaluate a couple agents and models at their ability to moderate conversations. They focus on three research questions (1) performance of different providers / agents and find that the social agent outperforms the basic agent in their evaluation. (2) Impact of difficulty of the scenario (3) how well do their proposed metrics reveal construct validity.

### Strengths
The paper adds a complementary angle to traditional LLM negotiation papers that typically focus on an agent acting as the negotiator.
The theory-driven approach seems well grounded.

### Weaknesses
Overall, I have a couple critiques for this paper. First, this paper is motivated with the need for a mediator in human negotiations [39-45]. But evaluates a mediator for LLM negotiations. This seems like the original claim isn’t reasonable, since LLMs might negotiate in different ways from humans. I would have liked to see a human study where the mediator is involved in real human negotiations to test for distribution shift. I realize that humans were used to evaluate the quality of the transcripts, but the Cohen’s Kappa is 0.63, which feels quite low. 

The main contribution of the paper feels a little weak. The paper wraps what is essentially in LLM-as-a-judge in a series of theory-inspired rules for negotiating. While this is important, I don’t think it makes for an ICLR paper. 

There is a rich set of literature about evaluating LLM ability to negotiate that feels necessary to cite but is missed (see below). There is also some literature on mediation that’s missing.
[1] Are LLMs Effective Negotiators? Systematic Evaluation of the Multifaceted Capabilities of LLMs in Negotiation Dialogues https://aclanthology.org/2024.findings-emnlp.310.pdf 
[2]Evaluating Language Model Agency through Negotiations https://arxiv.org/abs/2401.04536 
[3] Multi-Agent Collaboration Mechanisms: A Survey of LLMs  https://arxiv.org/abs/2501.06322 
[4] How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis https://arxiv.org/abs/2402.05863  
[5] Simulating Dispute Mediation with LLM-Based Agents for Legal Research https://arxiv.org/html/2509.06586v1 

The results that the social agent has improved performance is not at all surprising since there seems to be some leakage between evaluation criteria and the design of the agent.

Finally, the writing in this paper is problematic — especially in the appendix. 

Minor points:
- type [082] ‘neogiating’ and ‘an deadlock’ -> a deadlock.
- appendix F.2 is poorly written.

### Questions
Do the authors plan to release Github package that allow different agent developers to evaluate their models?

Did the authors leave out the other rich negotiation w/ LLMs literature on purpose?

### Soundness
2

### Presentation
2

### Contribution
2
