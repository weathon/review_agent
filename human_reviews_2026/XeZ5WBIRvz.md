# When Ethics and Payoffs Diverge: LLM Agents in Morally Charged Social Dilemmas

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Recent advances in large language models (LLMs) have enabled their use in complex agentic roles, involving decision-making with humans or other agents, making ethical alignment a key AI safety concern. While prior work has examined both LLMs' moral judgment and strategic behavior in social dilemmas, there is limited understanding of how they act when moral imperatives directly conflict with rewards or incentives. To investigate this, we introduce MORAL Behavior in Social Dilemma SIMulation (MoralSim) and evaluate how LLMs behave in the prisoner's dilemma and public goods game with morally charged contexts. In MoralSim, we test a range of frontier models across both game structures and three distinct moral framings, enabling a systematic examination of how LLMs navigate social dilemmas in which ethical norms conflict with payoff-maximizing strategies. Our results show substantial variation across models in both their general tendency to act morally and the consistency of their behavior across game types, the specific moral framing, and situational factors such as opponent behavior and survival risks. Crucially, no model exhibits consistently moral behavior in MoralSim, highlighting the need for caution when deploying LLMs in agentic roles where the agent's "self-interest" may conflict with ethical expectations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work studies how LLMs behave when moral obligations conflict with incentives in social dilemmas. The authors present a simulation framework called MoralSim that essentially is a collection of situations that reflect social dilemmas expressed in natural language. The authors evaluate several LLMs considering Prisoner's Dilemma and Public Good games. The findings show that models vary widely in their tendency to act morally and lack consistent ethical behavior across situations, suggesting that LLMs may be unreliable for decision-making roles requiring alignment with ethical norms.

**General note**: I reviewed this manuscript for a previous conference. I would like to note that I went through the paper carefully, noting the main differences. The core contributions are essentially the same apart from the causality analysis; hence, I find myself in a position to repeat my critiques. Please note that the list of weaknesses below takes into consideration the revised version of the paper with respect to my initial review. I would also like to note that my comment about the naming of the “morality score” was addressed, and in this version, the authors talk about “cooperation score”. I noted the following changes with respect to the version I reviewed below (these are the main ones I was able to clearly identify):

- The morality score has been renamed cooperation score.
- A section about causal effect estimation has been added (plus a discussion of the treatment effects later in the paepr).
- The discussion of the results regarding Q3 and Q4 has been extended.
- There is a new section about reasoning trace analysis in the appendix.

These points/observations did not affect my review. I treated this paper completely *as new*. I believe that these are independent conferences and we should consider the manuscripts in their own right each time. I added this note explicitly in order to avoid potential criticisms about the fact I raised again some of the concerns of my previous review (since those parts are unchanged).

### Strengths
- This is indeed a very interesting topic.
- The paper is well written and very easy to follow. The evaluation of the work is conducted.

### Weaknesses
- This is a well-written paper, but it is difficult to identify a substantial contribution with respect to the state of the art. In fact, the papers listed under “LLMs in Game Theory Settings” focus directly or indirectly on decision-making problems in social dilemmas. In fact, the authors do not really consider actual moral frameworks in their analysis as these works do.  The authors claim that various prior research has already explored LLMs’ moral reasoning and strategic behaviour separately. The authors of [Tennant et al., 2024] (actually published and presented at ICLR 2025) essentially not only present a variety of games (including the Iterated Prisoner's Dilemma) but also study how to run a fine-tuning procedure to examine the effects of moral decision-making.
- The authors present very limited analysis in terms of sensitivity to variations of the prompts, including the agent setup that is discussed in Section 3.3 (e.g., descriptions of the setting, personal memory, and current task). These might have substantial effects and should be discussed by the authors in my opinion.
- It is quite surprising that the authors considered moral dilemmas, but the actual dynamics of the responses are only partially analysed/considered by the authors. In fact, the authors mainly focus on the choice of the single agents. The authors also consider the opponent alignment, but this is quite confusing since it appears to the reviewer quite orthogonal to the problem of acting morally. The authors consider the relative payoff, but considering the fact that these are classic (repeated) games, the analysis of the actual cumulative payoff would have probably been more informative from a game theory point of view.
- The survival rate score is interesting, but it appears rather disjointed considering the core topic of the paper. With respect to the statistical validity of the simulations, it is unclear how different repetitions of the games have been implemented. 
- The paper essentially lacks a related work section. The authors moved it to the Appendix, but that is outside the 10 pages of the main body of the paper. It seems to me a way for going above the 10 page limit: this is not fair towards the other ICLR authors in my opinion.

### Questions
I do not have specific questions for the authors. My core concern is about the very limited contribution of this work with respect to the state of the art.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents MORALSIM, a new framework for evaluating the moral behavior of LLM agents in situations where ethical norms conflict with personal interests and benefits. The authors investigate the behavior of 9 modern language models (including Claude-3.7-Sonnet, GPT-4o, Deepseek-R1, o3-mini, and others) in two classic game scenarios: Prisoner's Dilemma and the Public Goods Game. Each game is embedded in three different moral contexts: Contractual Reporting, Privacy Protection, and Green Production. The full factorial design of the experiment is used, varying the type of game, the moral context, the opponent's behavior (always cooperating / always betraying) and the risk of survival. The key result: none of the tested models demonstrates consistent moral behavior in all scenarios. The proportion of morally oriented actions varies from 7.9% (Qwen-3) to 76.3% (GPT-4o-mini). The authors use causal analysis (Average Treatment Effects) to determine the factors influencing moral decisions, and show that the structure of the game, the specific moral context, and the opponent's behavior have the greatest impact.

### Strengths
- A full-factor design with clear manipulations (type of game, moral context, survival risk, opponent behavior) allows you to isolate the effects of each factor
- Quality of analysis. Analysis of ~3500 reflections of agents reveals decision-making mechanisms. Causal assessment through ATEs yields quantitative effects with confidence intervals. It is shown that profit maximizer models (Deepseek-R1, Qwen) rely on profit maximization, while more cooperative ones (Claude, GPT-4o) more often take into account moral considerations.
- All models show a decrease in morale precisely when the user is most vulnerable (at risk of bankruptcy), which raises important issues of AI security.
- The code is open, the prompts are documented in detail.

### Weaknesses
1. PD and PGG only. Other structures (for example, Trust Game, Stag Hunt) could reveal other patterns of moral behavior. An extension to asymmetric games would be especially valuable.
2. In the real world, agents can often negotiate, which significantly changes the dynamics of cooperation. The authors acknowledge this, but do not investigate it.
3. For some models (Claude-3.7-Sonnet, Gemini-2.5-Flash), versions without reasoning mode were used for cost reasons. Given that the analysis has shown the importance of reasoning, this limits the conclusions.
4. Multi-agent scenarios (N>2) could better reflect social dynamics and collective responsibility.

### Questions
1. You have shown invariance to paraphrases, but how sensitive are the results to more fundamental changes in the presentation of the problem? For example, what if we present the same dilemmas through different metaphors or change the order in which options are presented?
2. Does the moral behavior of agents change as they gain experience in repetitive games? Are there signs of "moral learning" or adaptation of strategies over time?
3. Moral norms vary between cultures. Do you plan to investigate how different cultural contexts affect the moral behavior of LLM agents?

### Soundness
3

### Presentation
4

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
The authors introduce MoralSim, a benchmark that focuses on studying LLM behaviors in scenarios where explicitly forced to trade-off between moral behavior and rewards. The authors encase these scenarios in realistic settings to provide greater fidelity to underlying LLM behavior and perform analysis of different behavioral questions in LLMs to find that

### Strengths
- The paper is well-written and the experiments are well-designed. 
- The settings developed by the paper more clearly expose trade-offs in moral behavior compared to prior work. 
- There are clear behavioral takeaways from the paper, e.g., opponent behavior meaningfully steers LLM actions or that moral context improves the morality of LLM behaviors.

### Weaknesses
While the games offer a step towards realism, they still do not capture the full nuances of reality. In particular:
- All the games are two-player. Many real-world settings involve multiple players with different levels of power and interlocking incentives.
- The text of the games themselves is still somewhat unrealistic. There is an explicit payoff structure described in the system prompt (e.g., Figure 2), whereas in the real world an LLM agent would need to uncover those trade-offs themselves.
However, I realize that these nuances are quite tricky to incorporate and would make some of the analysis less clean, so I don't think that should hold the paper back.

There is no discussion or analysis of whether or not the LLMs participating in the games recognize that they are in the game. Recent research into scheming (https://www.antischeming.ai/) suggests that frontier LLMs may recognize that they are being evaluated, which could question the validity of the research results. However, it is difficult to assess scheming without access to the full reasoning trace, so I also don't count this as a strong weakness.

### Questions
- Would it be possible at all to analyze whether the LLMs recognize that they are playing a game and if that would alter their behavior in any sense? Would it be possible to attempt to induce "evaluation-awareness" into the LLMs (e.g., by being more suggestive with the wording that the environment is a game) and see how that affects the results?

### Soundness
2

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
5

### Summary
The paper investigates the behavior of large language models (LLMs) as agents in social dilemmas that include moral constraints. Specifically, it explores how these models make decisions when moral principles conflict with strategies that yield higher rewards. To address this, the authors propose a new evaluation framework called MoralSim (Moral Behavior Social Dilemma Simulation), which integrates classic game-theoretic environments (such as repeated Prisoner's Dilemma and Public Goods Game) with real-world moral scenarios.

### Strengths
- The MoralSim framework combines classic game-theoretic scenarios with real-world moral dilemmas, enabling a systematic and comprehensive evaluation. This framework is more structured than previous fragmented tests. For example, compared to the MACHIAVELLI benchmark—which primarily uses text adventure games to assess agent behaviorarxiv.org—MoralSim employs formal game structures integrated with moral contexts, allowing for more controlled and easily quantifiable comparisons. This could be better than just simple QA.

### Weaknesses
- Scientificness of the evaluation of this assumption. Although multi-agent framework could provide a more vivid setting for revoke LLM's decision under certain scenarios, it could still be a question of how real and how consistent these evaluations are. According to my experience these testings are easily be changed by small parts of prompts. Yet, this paper don't provide a convincing enough evidence to illustrate the scintificness of this testing. 

- The novelty is somewhat incremental. Although the MoralSim framework’s integration of morality and game theory is commendable, its concepts overlap with some existing work [1,2,3]. Also, please discuss these papers in the main paper more to let audiance familiar with context and existing research as well as the differences. Especially [2,3] already reported the betray behavior of LLMs

[1] Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark

[2] Moral Alignment for LLM Agents 

[3] Survival Games: Human-LLM Strategic Showdowns under Severe Resource Scarcity

### Questions
- How authors evaluate the testing is stable instead of impacted by small part of prompt design? How often do models change archetype between Goal-oriented vs Neutral prompts? Provide a confusion matrix of archetypes across prompts.

- Please include a comparison table vs MACHIAVELLI, GovSim, Survival Games, and moral-reward alignment detailing the unique “human-harm resource” dimension and survival horizon.

### Soundness
3

### Presentation
2

### Contribution
2
