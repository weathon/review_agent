# General search techniques without common knowledge for imperfect-information games, and application to superhuman Fog of War chess

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 8, 4, 2

## Abstract
Since the advent of AI, games have served as progress benchmarks. Meanwhile, imperfect-information variants of chess have existed for over a century, present extreme challenges, and have been the focus of decades of AI research. Beyond calculation needed in regular chess, they require reasoning about information gathering, the opponent’s knowledge, signaling, _etc_. The most popular variant, _Fog of War (FoW) chess_ (a.k.a. _dark chess_), has been a major challenge problem in imperfect-information game solving since superhuman performance was reached in no-limit Texas hold’em poker. We present _Obscuro_, the first superhuman AI for FoW chess. It introduces advances to search in imperfect-information games, enabling strong, scalable reasoning. Experiments against the prior state-of-the-art AI and human players---including the world's best---show that _Obscuro_ is significantly stronger. FoW chess is the largest (by amount of imperfect information) turn-based zero-sum game in which superhuman performance has been achieved and the largest game in which imperfect-information search has been successfully applied.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a game-playing agent, named Obscuro. Obscuro is the first artificial agent that has achieved superhuman performance in Fog-of-war Chess, an imperfect information variant of Chess. Obscuro is a fully search-based agent that builds on the previous state-of-the-art agent in FoW Chess. The main algorithm behind Obscuro, Knowledge-limited Unfrozen Subgame Solving (KLUSS), is a generalization of KLSS. The paper reports good results against the previous state-of-the-art agent, human players of various skill levels, as well as the number one-ranked human player. Finally, the proposed improvements and their importance to the final performance of Obscuro are validated in a series of ablation experiments.

The paper is clearly written overall, with a well-motivated introduction, detailed algorithmic description, and a comprehensive experimental section that includes strong evaluations against both prior state-of-the-art methods and human players. The ablation studies effectively validate the importance of the proposed improvements. However, the clarity is weakened by missing or inconsistent details, such as incorrectly labeled figures, missing descriptions of the subgame solving algorithm in the main text, and ambiguities in the ablation setup. Some claims, like FoW Chess being the largest imperfect-information game where search has been applied, seem unsupported.

### Strengths
**Originality.**
The experiments convincingly support the claim that Obscuro is the first AI agent to achieve superhuman performance in FoW Chess.

**Clarity**
The whole paper is nicely and concisely written. The introduction motivates the need for scalable search in imperfect information games and the reasons it’s substantially harder than in perfect information games.
The agent’s description is clearly and thoroughly written, providing detailed explanations of each component.

**Soundness.**
The experimental section is comprehensive, including detailed evaluations against both previous state-of-the-art methods and human players of different skill levels, up to the top-rated player on chess.com, providing strong empirical evidence for the quality of the proposed method.
The ablation studies consistently demonstrate performance improvements of the proposed modifications, confirming that the improvements are both well-justified and impactful.

### Weaknesses
**Clarity.**
Several parts of the paper suffer from unclear or inconsistent presentation:
* Lines 216–220 introduce a connectivity graph with nodes labeled by their distance from the circled node and one node marked with an asterisk, yet no such labeling appears in the corresponding figure, making it difficult to follow the explanation.
* The description of the main algorithm omits details about the subgame solving algorithm, describing it only in the appendix.
* It is not clear which opponent was used in the ablation experiments and what the “above four improvements” in the fourth ablation (GT-CFR Only) refer to
* It is also not clear which components were and were not used in some of the ablations

**Soundness.**
The claims that FoW Chess is the largest (measured by the amount of hidden information) turn-based game and the largest imperfect information game where search has been successfully applied are not based on any concrete numbers comparing the game to other such games.
The first sentence in the conclusion is misleading, as Obscuro requires a strong value function to reach such a strong play (as shown by one of the ablation experiments).

### Questions
* Is the set $P$ from Section 3 an infoset of the acting player?
* Are the ablations run against the previous state-of-the-art algorithm for FoW Chess?
* Do you have any intuition why two-sided GT-CFR scored so low as compared to one-sided GT-CFR in the ablations?

Minor comments:
* Line 53: A missing citation to DeepStack
* Lines 147-150: The problem of deciding whether two histories belong to the same common-knowledge closure has been studied by Solinas et al. [1]

[1] Solinas, Christopher, et al. "History filtering in imperfect information games: algorithms and complexity." Advances in Neural Information Processing Systems 36 (2023): 43634-43645.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents Obscuro, the first AI to reach a superhuman level in the chess variant Fog of War chess (FoW), and reaches a new state-of-the-art for this game type.
The main innovation lies in its new search adaptations.
In fact, the agent is entirely based on real-time search, rather than training a new neural network policy or value function; it relies on classical Stockfish 14 for standard chess for their node evaluation.
Related to search, they introduce knowledge-limited unfrozen subgame solving (KLUSS) search as an improvement over the KLSS algorithm. Next, they utilize growing-tree counterfactual regret minimization (GT-CFR), modifying it to one-sided GT-CFR. Here, they use predictive CFR+ (PCFR+) for equilibrium finding. To expand the game tree, they utilize the polynomial upper confidence bounds for trees (PUCT) algorithm.
The authors also provide game footage of their agents, along with an elaborate appendix that explains the search techniques in detail.

### Strengths
+ The engine runs on regular consumer hardware (the details are provided in Appendix B.5.).
(Preferably, refer to Appendix B.5 directly in the main paper.)

+ The presented agent Obscuro wins both against the previously best FoW engine and the strongest human player on chess.com by a significant margin (+302 +/- 29 Elo over ZS21, +241 +/- 274 Elo over top human).

+ The paper presents numerous ablation studies that highlight the benefits of their individual modifications to existing algorithms.

+ The paper is well formulated and guides the reader throughout the content.

+ It includes a detailed appendix with pseudocode that provides a detailed explanation of the search techniques.

Relevancy:
The paper appears to be highly relevant for a major audience.

Overall Assessment
Overall, I believe it is a great paper that pushes the state-of-the-art in the field of imperfect information games to a new level.

### Weaknesses
- "required no large-scale computation to learn a value function or blueprint strategy"
-> This is true, but the evaluation function was taken from Stockfish 14, which underwent a long period of development. The authors also show in their ablation studies that the evaluation function plays a significant role in their playing strength.
Maybe clarify this statement a bit.

- The authors do not mention whether they will make their source code and/or engine publicly available.

- It was not directly clear to me at first sight in which perspective the published games were, i.e. if Obscuro was White or Black. -> Please add a comment in the paper that the games are always in the view of Obscuro.

- The sample size (20 games  (+16 =0 -4), +241 +/- 274 Elo) against the top human was not large enough to 100% claim superiority over the top human. You may also provide the relative Elo superiority with error bounds in the main paper. The top human could also try to find fundamental weaknesses in its play over time. This seems, however, not likely as it attempts to find a near-equilibrium strategy.
I also consider 20 games to be a good sample size, given it's hard to find top players playing an engine. 

- The paper would benefit from an additional evaluation environment despite FoW, to highlight the broader applicability of their search agent. This alternative environment could be poker or Stratego for example.

Minor things:
"Liu et al. (Liu et al., 2023) introduced a safe variant of KLSS"
-> Here, Liu et al. is mentioned twice. You can avoid this by using \citet{}

### Questions
"4. Two-sided GT-CFR only, against ZS21. In this ablation, we turned off all the above im-
provements 1, 2, 3, and 4, and matched the resulting agent against that of ZS21. This
serves to isolate the effect of using GT-CFR compared to using the LP-based equilibrium
computation and iterative deepening node expansion as in ZS21.
In a 1,000-game match, the GT-CFR version scored 72.6% (+711 =30 -259)."

-> This is a bit unexpected to score so highly here, when turning off all 3 previous improvements, right?
I likely misunderstood what was done here.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Obscuro, an AI system to achieve superhuman performance in Fog of War (FoW) Chess, a long-standing benchmark in imperfect-information game solving. The key technical contribution is a suite of scalable real-time search techniques—most notably Knowledge-Limited Unfrozen Subgame Solving (KLUSS) and one-sided GT-CFR—that circumvent the need to compute or enumerate common-knowledge sets, which are intractable in FoW Chess (up to ~10¹⁸ states). The authors demonstrate Obscuro’s superiority through extensive experiments: it wins 85.1% of 1,000 games against the prior SOTA (ZS21) and 80% of 20 games against the world’s top-ranked human player (rating 2318 on chess.com).

### Strengths
Significant Technical Advance: The paper directly addresses a fundamental scalability bottleneck in imperfect-information game solving—the reliance on common knowledge—and proposes practical, theoretically motivated approximations (KLUSS) that enable real-time search in previously intractable settings.

Strong Empirical Validation: The evaluation is comprehensive:
Large-scale AI-vs-AI matches (1,000–10,000 games) with statistical significance (z > 5).
Human evaluation across skill levels (100 games vs. players rated 1450–2006, 97% win rate).
A decisive 16–4 victory over the #1 human player, with p = 0.011 (binomial test).

Extensive ablation studies isolating the contribution of each component (e.g., KLUSS alone improves win rate from 58% to 85% vs. ZS21).
Algorithmic Innovation: The combination of one-sided GT-CFR, last-iterate strategy selection, and purification is well-motivated and empirically validated. The use of Stockfish as a node evaluator is pragmatic and effective.

Reproducibility & Transparency: Private game links are provided; hardware requirements are modest (consumer CPU); pseudocode and detailed appendices support reproducibility.

### Weaknesses
The article is not clearly articulated and lacks proper formatting. It fails to explicitly highlight the core contributions of the KLUSS algorithm and one-sided GT-CFR (which may represent the most significant departure from prior methods).

Stockfish is designed for chess with perfect information. Using it as an evaluation function for the game tree in FoW chess is not entirely reasonable, since the winning conditions of the two games differ. Please provide an explanation.

### Questions
Stockfish is designed for chess with perfect information. Using it as an evaluation function for the game tree in FoW chess is not entirely reasonable, since the winning conditions of the two games differ. Please provide an explanation.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents Obscuro, a search-based AI agent achieving superhuman performance in Fog of War chess, introducing general subgame-solving algorithms that do not rely on explicit common-knowledge reasoning. The authors propose the KLUSS framework that prunes high-order belief states considered strategically irrelevant. This is integrated into algorithms like PCFR+ for efficient tree expansion. Expirical results show that Obscuro surpasses the previous SOTA system and human experts.

### Strengths
- The paper addresses a central limitation of existing imperfect-information search methods. KLUSS provides a pragmatic and scalable alternative that extends the reach of search-based techniques.
- The system combines ideas from recent developments in counterfactual regret minimization, tree expansion policies, and real-time planning, demonstrating a coherent and well-engineered design.
- The empirical results are strong. The performance can be achieved with relatively small computation, underscoring the method's practical efficiency.

### Weaknesses
- The interaction between PCFR+, one-sided GT-CFR, and KLUSS lacks a unified theoretical analysis. Each component has individual convergence guarantees under specific settings, but their concurrent use in a dynamically expanding and pruned search tree leaves correctness unproven.
- Although framed as a general search method, Obscuro’s performance critically depends on Stockfish’s perfect-information evaluation function. This component embeds extensive domain knowledge from conventional chess, potentially limiting the generality of the claimed framework.
- The paper can also benefit from adding tests on other imperfect-information domains such as poker or Stratego, which strengthens the empirical basis for calling the approach “general-purpose.”

### Questions
See weaknesses

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces *Obscuro*, a superhuman Fog of War (FoW) chess agent.
It improves the SOTA by using better SOTA algorithms for the individual steps. The most significant improvement results from replacing LP-based equilibrium computation with GT-CFR. Further improvements are generated by careful engineering.

### Strengths
- very good motivation
- great outcomes, producing the first superhuman FoW agent
- overall well-written paper

### Weaknesses
- low originality by mainly engineering the current SOTA
- contributions only clearly stated in the ablation
- use of crafted, not learned value-function
- no learning at all, only search
- overuse of footnotes decreases readability

### Questions
- Can you motivate the choices of your adaptations?
- What are the implications of your work in the field? I.e., how can your work contribute to solving other tasks?
- The approach uses advanced search techniques and does not utilize any learning algorithm or learnable parameters. What makes this paper a reinforcement learning paper?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 6

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper contributes a Fog-of-War chess bot that is ostensibly state of the art. It performs real-time search without trying to untangle unwieldly "I know that you know..." loops that usually choke imperfect-information solvers.

### Strengths
Obscuro seems to be the first superhuman AI in the fog of war chess variant.
The paper does a good job of explaining the difficulties associated with developing AI in this kind of setting.

### Weaknesses
- My biggest concern is the significance of the contribution. Even if all of the claims in the paper are completely accurate (and see below for concerns on that front), it's not clear to me that applying mostly known tricks to develop superhuman AI in this niche chess variant constitutes enough for acceptance. Despite being a lifelong chess fan, and even a chess variants fan, I've never heard of fog of war chess. I don't think the abstract's claim that FoW chess has been "the main challenge problem in imperfect information games" is accurate. 

- Building on the previous point, the methods used are interesting but not super exciting, and don't seem to be very generalizable. The work would be stronger if the authors could demonstrate or at least argue why their innovations are useful beyond FoW chess.

- The results section is quite short, and I'm not convinced that the authors have done everything they can to put Obscuro up to the toughest tests.

- The paper is somewhat unprofessionally written (e.g. section 4.1; the 4-line title).

### Questions
- What is the broader significance?
- How could the methods generalize?

### Soundness
2

### Presentation
1

### Contribution
2
