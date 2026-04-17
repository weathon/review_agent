# Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People

- Decision: Accept (Oral)
- Scores: 8, 4, 8

## Abstract
Many emerging applications of AI—from scientific discovery to medical diagnosis—require agents to seek information strategically: forming hypotheses, asking targeted questions, and making decisions under uncertainty. In high-stakes settings with limited resources, do language models (LMs) behave like rational agents? Drawing on insights from human cognition, we develop methods to evaluate and enhance agentic information-seeking. First, we introduce a decision-oriented dialogue task called Collaborative Battleship, in which a Captain must balance exploration (asking questions) and action (taking shots), while a Spotter must supply accurate, contextually-grounded answers. Compared to human players (N=42), we find that many LM agents struggle to ask informative questions, produce accurate answers, and identify high-utility actions. To address these gaps, we develop novel Monte Carlo inference strategies for LMs inspired by Bayesian Experimental Design (BED). For Spotter agents, our approach boosts accuracy by up to 14.7% absolute over LM-only baselines; for Captain agents, it raises expected information gain (EIG) by up to 0.227 bits (94.2% of the achievable noise ceiling). Combined, these components yield sharper targeting (+0.303–0.374 F1), and enable weaker LMs, such as Llama-4-Scout, to outperform both humans (8% → 82% win rate) and frontier models (0% → 67% win rate vs. GPT-5) at ≈1% of GPT-5's cost. We replicate these findings on Guess Who?, where our methods significantly boost accuracy (+28.3–42.4 p.p.), demonstrating their general applicability for building information-seeking agents.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The present paper evaluates and improves the ability of LLMs to ask goal-directed questions and take actions in a dynamic environment. For this, the authors design a novel task called Collaborative Battleship. They run a human study on this task and compare human performance to that of various LLMs. This revealed a performance gap that the authors then subsequently address by developing Bayesian-inspired inference-time strategies for LLMs, leading to significant improvement in LLM performance.

### Strengths
I enjoyed this paper a lot. It has all the ingredients for a great paper: a new task, human evaluation, a decent set of different models that are evaluated, a novel technique for improving models, and demonstration that the findings generalize to another domain. The paper was exceptionally well written and easy to follow. The method is clean and simple, yet effective. CaptainQA is an interesting agentic test bed for LLMs and the whole methodology fits thematically well into ICLR.

### Weaknesses
I found the usage of the term Bayes-rational weird. Best to my knowledge, this is not an accepted term in the literature. It implies that the strategies developed by the authors are Bayes-optimal, which is not the case (as also noted by the authors). To avoid this confusion, I would suggest using a different term instead.

There is not so much negative to say about the paper. Perhaps the only downside is that, while the results and methods are interesting, they are not groundbreaking. For me, that is the only reason for giving this paper a score of 8 (instead of the full 10).

Minor:
* RSA not defined (p9).

### Questions
The indicator function in Equation 1 seems to be redundant (unless I am missing something).

The authors find that GPT-5 does not significantly benefit from Bayesian question or move selection. This raises the question of whether the proposed is useful for future models. Do you think that there is a risk of this being the case?

I was a bit confused by the description of Equation 7. Why is \pi_{t+1}^a introduced but then never used? Why p_{t+1}^{hit} defined as a distribution over questions? That seems strange. Why is u_t^* mentioned twice under step 2?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Collaborative Battleship, a novel two-player task designed to evaluate the trade-offs between information-seeking (asking questions) and exploitation (taking actions) in language model (LM) agents under communication constraints (yes/no answers). The authors collected a human dataset, BATTLESHIPQA (N=42 participants, 126 games), establishing human baselines for question quality and strategic play. They compare human performance to various LMs, finding that while weaker LMs struggle, frontier reasoning models approach human levels. To improve agent rationality, the paper proposes inference-time strategies based on Bayesian Experimental Design (BED), using Sequential Monte Carlo (SMC) to approximate belief states and guide question selection (maximizing EIG), action selection (maximizing hit probability), and the explore/exploit decision (one-step lookahead). These strategies significantly boost performance, enabling weaker LMs (Llama-4-Scout) augmented with Bayesian methods to outperform average humans and stronger LMs like GPT-5 in win rate, at substantially lower computational cost. Findings were replicated on the Guess Who? game. The authors provide IRB approval details and plan to release the dataset and code.

### Strengths
**Novel Task and Dataset:** Collaborative Battleship provides a clean, interpretable environment for studying the explore/exploit dilemma and grounded communication. The BATTLESHIPQA dataset, collected from human interactions, is a valuable contribution for benchmarking and analysis.

**Principled Bayesian Framework:** The application of BED principles (EIG maximization, belief updating via SMC, MAP action selection) provides a strong theoretical grounding for the proposed inference-time strategies.

**Strong Empirical Results:** The Bayesian strategies demonstrate significant improvements in LM agent performance across multiple metrics (accuracy, EIG, F1, win rate). The finding that augmenting weaker LMs can lead to super-human and SOTA-LM-beating performance at lower cost is particularly compelling.

**Generalization:** Successful replication of performance gains on the Guess Who? task suggests the framework's potential applicability to other information-seeking domains.

**Transparency and Reproducibility:** The paper includes ethical considerations for human subjects, clear plans for releasing code and data, and explicit disclosure of AI assistance.

### Weaknesses
**Limited Scope:** While the paper introduces an interesting method, its evaluation is confined to a specific domain defined by the authors, with generalization demonstrated only on one additional ad-hoc task. This limited scope makes it difficult to assess the method's potential for broader applicability and overall impact.

**"Surpass Humans" Claim Qualification:** The claim that augmented LMs surpass human performance needs stronger qualification. Details on human participants' prior experience, the number of trials per condition, and how performance compares when normalizing for computational resources (cost, latency, tokens) are necessary for a fair comparison. The large cost disparity is noted but not integrated into the primary win-rate comparisons.

**Rationality Framing:** The paper frames the goal as building "rational" agents using Bayes-optimal principles (BED). However, human behavior often follows boundedly rational heuristics. The evaluation primarily uses game performance (win rate, F1) as a proxy for rationality, potentially conflating task success with optimal information processing under constraints.

**Potential Scaffolding Effects:** The interaction between different components (question generation, answering, strategy selection, potential code generation for grounding) needs further ablation. It's unclear if performance gains could stem from implicit prompt leakage or interactions between modules rather than purely the Bayesian logic.

### Questions
1. Could you provide more details on the human study participants? What was their prior experience with Battleship? How were the number of human trials balanced against LM evaluations in terms of total interaction opportunities or budget? 

2. Could you please provide performance results (e.g., win rate, F1 score) normalized by computational cost (tokens, latency, API cost)?  Cost-controlled comparisons (e.g., win rate vs. budget curves) would strengthen the claims about efficiency.

3. Can you provide ablations that isolate the capabilities of the Captain (questioner/actor) and Spotter (answerer) roles? For instance, how does a strong Captain perform with a weak Spotter, and vice-versa? How much does the specific code generation strategy contribute to the Spotter's grounding? 

4. How sensitive are the results of the Bayesian strategies to hyperparameters like the SMC particle count, the number of candidate questions sampled (k for $Q_{Bayes}$), the decision discount factor ($\gamma$ for $D_{Bayes}$), and decoding parameters (e.g., temperature)?

5. What was the inter-rater reliability for the manual annotation of human questions and gold answers in BATTLESHIPQA?

6. Consider citing and comparing with SPIN-Bench (Yao et al., 2025) which evaluates LLMs in multi-agent cooperative and strategic settings (like Hanabi)  and documents coordination failures, making it relevant context for evaluating interactive agent strategies.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Paper introduces the collaborative battleship game, where a “captain” and a “spotter” collaborate in the game of battleship. The captain must decide between asking the spotter a question, or taking a hit at a board position, thereby having to balance information seeking and reward seeking.

The authors collected the BATTLESHIPQA human dataset where pairs of humans collaborated to play the game (42 participants, 21 pairs). This establishes human performance, the kind of questions humans ask, and how accurately the human (spotter) responds. 

They then evaluated 15 LMs in the role of the spotter (i.e. Q&A), finding that the best frontier models can do well, but they show greater degradation in hard (i.e. context dependent) questions. Then, they evaluate 3 LMs in the role of the captain, with various “bayes rational” strategies.  The “bayes rational” strategies improve the performance of weak LMs to the level of / slightly beyond GPT-5.

Finally, they extended the bayesian inference framework to other information seeking games and see some performance gains.

### Strengths
There are numerous strengths with this paper. To start, the collaborative battleship game sets up an interesting scenario for the captain agent, where they not only have to balance information seeking vs. reward seeking behaviour, but also maintain uncertainty about the correctness of its collaborator (noisy spotter agent). Information seeking in the space of natural language is an interesting setting to study.

The human dataset provides valuable information to ground “human level” performance, the type of questions asked, and human-like behaviour. The LM evaluations are comprehensive, with interesting findings including code generation boosting spotter accuracy. The “bayes rational” captain strategies improving worse models to perform at GPT-5 level has useful application implications. 

Overall, the paper is very well written and presented.

### Weaknesses
A number of simplifying assumptions were made in the “bayes rational” modelling choices. For instance, modelling with fixed $\epsilon$ (as the authors already point out) and $\gamma$ (not sure how this is set), and only modelling single-step look-ahead are simplifying assumptions. I do not think this detracts from the main point of the paper, as to my understanding this paper is about improving empirical performance of weak LMs using some cognitively inspired “bayes rational" strategies. Nevertheless, it is worth pointing this out for scientific rigour. 

However, there are a few simplifications that I am quite confused about, and likely warrant deeper discussions. I ask them in the question section below.

### Questions
### On assumptions

**(1)**
is maximizing EIG (Eq 5, L214) in this game the “right” thing to do to maximize performance of hitting a ship? My understanding is that in some games from previous works (e.g. ActiveACRE, blickets, feature world) [1,2,3], the explicit goal is to find out how the environment works. In these cases, directly optimizing for information gain is the “right” thing to do. 

In battleships, seeking information is only ever in service of the goal of hitting ships. Therefore, rather than computing the post-question hit probability (Eq 7) with the max-EIG question (Eq 5), i.e. $p_{t+1}^{hit} (q_{t}^{*} | x, H)$, shouldn’t we directly find $\arg\max_q p_{t+1}^{hit}(q | x, H)$ instead, and use that in the decision rule in L217? Is there no scenario in which this will change which question is selected? This is discussed empirically in L405-412 but it would be good to discuss a bit theoretically as well. 

**(2)**
I do not think the information gained from the “shoot” action is appropriately considered. The act of shooting perfectly reveals information about whether or not a tile contains a ship, akin to asking a noise-less spotter “is there a ship on this tile”. Indeed, for a 8x8 board and a budget of 40 shots, 40/64 = 62.5% of tiles can be revealed this way. Thus, shouldn’t the captain’s decision really be between (i) asking a question based on post-hit probability (per point (1) above), and (ii) choosing a shot that will both hit a current target and/or increase post-hit probability of the next shot? 


### Minor clarifications

1. What does Figure 3 b and c error bars denote? Please label / state in figure caption
2. How is the discount factor $\gamma$ in L216 selected?

---

[1] Piriyakulkij, Top, et al. "Doing experiments and revising rules with natural language and probabilistic reasoning." Advances in Neural Information Processing Systems 37 (2024): 53102-53137.

[2] GX-Chen, Anthony, et al. "Language Agents Mirror Human Causal Reasoning Biases. How Can We Help Them Think Like Scientists?." arXiv preprint arXiv:2505.09614 (2025).

[3] Sawyer, Danny P., et al. "Can foundation models actively gather information in interactive environments to test hypotheses?." arXiv preprint arXiv:2412.06438 (2024).

### Soundness
4

### Presentation
3

### Contribution
3
