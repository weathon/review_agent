# Breaking Algorithmic Collusion in Human-AI Ecosystems

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
The growing adoption of AI agents is giving rise to ecosystems where these agents interact not only with each other but also with humans. We study such mixed ecosystems in the context of repeated pricing games, modeling AI agents as playing equilibrium strategies. We then analyze defections, where a human manually performs the task instead of their AI agent, thereby replacing an equilibrium player with a no-regret strategy. Motivated by how populations of AI agents can sustain supracompetitive prices, we ask whether high prices persist under such defections. Our main finding is that even a single human defection can destabilize collusion and drive down prices, while multiple defections push prices close to competitive levels. We further show how the nature of collusion changes under defection-aware AI agents and under greedy (follow-the-leader) defection strategies. Taken together, our results characterize when algorithmic collusion is fragile—and when it persists—in mixed ecosystems of AI agents and humans.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies collusion dynamics in pricing between different kinds of players. In particular, AI agents are modeled as colluding and human players are modeled as defecting from that collusions. The paper derives theoretical results that characterize the robustness of collision to defection under different numbers of players and types of strategic adaptation.

### Strengths
The paper is on an interesting topic and very well written and motivated. The theoretical results do speak to the authors research questions and as a non-theorist I found them interesting.

### Weaknesses
I don’t believe ICLR is the appropriate venue for this work. There is very little content here related to AI/ML except in the cover story of the variables. The works cited support this as most of the work is from economics, algorithmic game theory and related fields.

### Questions
Can we connect these theoretical results to empirical modeling with actual LLM or DeepRL agents? 

Why do we need to assume that one player is the “human” and one is the “LLM”. Human’s might also collude. LLMs might not? It feels that this assumption is doing a lot of work and potentially confusing what the results actually mean.

### Soundness
3

### Presentation
4

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
This paper analyzes the stability of algorithmic collusion in iterated pricing games when one or more of the players "defects" against the others by following a no-regret strategy. Algorithmic collusion is an interesting setting of study, as it models the interactions between AI systems and humans in markets where collusion (e.g. high price fixing) can be detrimental for consumers. These kinds of scenarios are becoming extremely relevant as agentic AI is deployed in the wild.

The authors prove a few results in different settings:

**One defector:** If a single seller from the median-profit set defects (by switching to a follow-the-leader strategy), the market price falls to about $\frac{1+\log N}{N}$ where $N$ is the number of players.

**Many defectors:**  The authors show that adding more no-regret defectors drives prices down exponentially in M (the number of defectors).

**Defection-aware AI (extension):** If the AI agents can re-equilibrate after observing a defection, prices remain high: at least $1 - \frac{1}{K} - \frac{r(T)}{T}$ where $K$ is the number of discrete actions (product of discretizing the prices into K buckets), $T$ is the time horizon of the game and $r(T)$ is the external regret of the no-regret strategy (defection strategy).

### Strengths
This is a very well written paper that tackles the timely issue of algorithmic collusion. In particular the results, although theoretical, give practical, actionable insights that are useful to alleviate algorithmic collusion caused by non-myopic multi-agent interactions.

1. **Clarity:** The paper is extremely well written. All the necessary background is covered as well as important mathematical definitions required to analyze and interpret the results. 

2. **Importance of the problem:** The problem of study is becoming increasingly relevant as agentic systems are deployed and forced to interact with humans and amongst each other. This kind of work can be very beneficial to prevent and mitigate potential negative consequences of multi-agent interaction; algorithmic collusion for price setting is an especially likely one to occur. More specifically, "manually" introducing defections (by setting low prices) is proven to drive the time-average of the prices down in iterated Bertrand games.

3. **Mathematical rigor:** Results are proved with tight (or matching) bounds in the single- and multi-defector regimes, plus clean extensions.

### Weaknesses
Although the paper is very rigorous it has some major weaknesses that reduce my confidence in accepting it:

1. **Lack of experimental results:** No experiments (even toy) to illustrate convergence of prices under standard online learners or simple scripted LLM agents. Empirical evidence could potentially ground the theoretical results. Authors also note this is a stylized model.

2. **Modeling assumptions:** The assumptions that most weaken the paper, in my view, are: Equilibrium/coordination: core results presume AI sellers are already in a stable repeated-game equilibrium (and, in the extension, can quickly re-equilibrate after a defection), which abstracts away the messy training and deployment dynamics that often prevent such coordination; Which seller defects: the sharpest guarantees rely on the defector coming from the median-profit set, a “typical seller” condition that may not hold in many markets and can materially change the predicted price drop if high-profit leaders defect instead.

3. **Relevance to ICLR:** Given that the paper has no experimental part and that the results come from explicit modeling of the interaction dynamics between agents, there are clearly no learning representations aspect in the paper (deep learning).

I would be willing to reconsider my score and be more confident in my assessment if the authors mitigate/address the issues mentioned above.

### Questions
1. How often is the “median-profit defector” condition expected in realistic markets? Can the bound extend to arbitrary defectors (e.g. lower-profit defectors)?
2.  How sensitive are the main results of the paper to different tie-breaking rules? Could the authors give their intuition?

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
This work analyzes algorithmic collusion in repeated $N$-player pricing game with a discrete price grid in mixed human-AI ecosystems. AI agents are assumed to play (approximate) equilibrium strategies in the repeated game, whereas human players are assumed to defect by switching to no-regret strategies. The paper analyzes how such defections affect the market price with a single (median-profit) defector, multiple defectors, greedy/FTL defectors, and defection-aware AI agents. This paper provides theoretical proofs on how algorithmic collusion varies under defections.

### Strengths
1. The overall writing is easy to follow. No major errors. 
2. Precise formalization and clear proofs of theorems and lemmas. 
3. Useful scope of extensions. Covers single vs. multiple defectors, no-regret vs. FTL, and defection-aware agents, which sharpens the paper’s scope.

### Weaknesses
1. No empirical evidence. This paper is purely theoretical without any experiments or simulations of RL agents or LLM agents. Since it claims relevance to RL/LLM agents, adding empirical evidence would validate the theory in practice. 
2. This paper assumes RL/LLM agents would play equilibrium strategies. However, prior work suggests RL [1,2] and LLMs [3,4] may not be able to converge to Nash equilibria in repeated interactions and results vary across different RL algorithms and LLMs. Either justify this or soften the assumption.
3. Lack of convergence proof of AI algorithms under defections. There is no convergence/stability analysis for RL/LLM algorithms under defections (and no experiments). Provide a convergence claim for a representative algorithm class showing trained agents approximate the analyzed scenarios.

[1] Mazumdar, E., Ratliff, L. J., Jordan, M. I., & Sastry, S. S. (2019). Policy-gradient algorithms have no guarantees of convergence in linear quadratic games. _arXiv preprint arXiv:1907.03712_.

[2] Zhang, K., Yang, Z., & Başar, T. (2021). Multi-agent reinforcement learning: A selective overview of theories and algorithms. _Handbook of reinforcement learning and control_, 321-384.

[3] Akata, E., Schulz, L., Coda-Forno, J., Oh, S. J., Bethge, M., & Schulz, E. (2025). Playing repeated games with large language models. _Nature Human Behaviour_, 1-11.

[4] Lorè, N., & Heydari, B. (2024). Strategic behavior of large language models and the role of game structure versus contextual framing. _Scientific Reports_, _14_(1), 18490.

### Questions
1. What theoretical evidence supports the assumption that deployed AI (RL/LLM) agents play (approximate) equilibrium strategies in this setting?
2. Can you add empirical analysis with standard RL/LLM agents to support theoretical results? 
3. Alternatively, can you provide a convergence/stability guarantee (or a clear non-convergence statement) for a representative RL algorithm under defections?
4. Could you include a brief comparison table listing the closest papers and, for each: assumptions, agent models, main results? Please highlight what is new here and note where your assumptions match or differ from prior work.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates how human defections from algorithmic collusion influence prices in mixed human–AI markets. They model human defections as no-regret or follow-the-leader (FTL) strategies, AI collusion as prices sustained by equilibrium strategies and human-AI markets as repeated Bertrand pricing games. Their analysis shows that when a typical agent (with median profit) switches from an equilibrium strategy (AI) to a no-regret strategy (human), the market price goes down by a polynomial factor of the number of players and exponentially when multiple humans defect, regardless of their identities. They also show that equilibrium collusion can persist when AI agents are defection-aware. All claims are proven analytically, with tight matching bounds and detailed proofs in the appendix.

### Strengths
The paper presents a stylised setting to analytically characterise the price effects caused by agents deviating from the equilibrium profile and makes a solid theoretical contribution. The methodology appears to be correct and reproducible. Generally, the paper is well structured with clarity in the definitions and the whole approach. The problem the authors are investigating is timely and needs research attention.

### Weaknesses
The authors do acknowledge the limitations of the simplifications in the current work for analytical tractability. However, I find that the motivation that AI agents play Nash equilibria in the repeated game needs strengthening. The authors use the terminology of “AI agents” and “no-regret learners,” but mathematically treat them as given strategy classes, not as adaptive processes. Thus, while the topic (AI collusion) is socially and technically relevant, the learning connection is mainly in the narrative of the paper. So the core contribution is a game-theoretic characterisation of equilibrium outcomes under mixed strategy profiles, not an analysis of learning dynamics or convergence.

### Questions
The approach of the authors and their theoretical contributions is a stepping stone for more research in the field. I believe the paper would be strengthened a lot if they provided at least some simulations with simple learning agents to better persuade the validity of their assumptions for the AI agents and show how their theory indeed bridges empirical observations. Also, I found the modelling rather orthogonal to the standard AI modeling, e.g., the Nisan and Kolumbus (2022), where AI are algorithms play the game repeatedly using some algorithm and agents either report their values in a meta-game or play a Nash equilibrium (or something else less sophisticated). Is there a reason that this paper models the human as playing the algorithm and the AI as playing the Nash equilibrium? From this, I don't really understand why something is termed "AI" and something is "human" - also, I don't understand why something should be termed algorithmic collusion when this is not the result of some algorithms training together and reaching a collusive outcome, but rather an assumption that a Nash equilibrium is played. In general, I don't see why something is distinctively AI vs human here and maybe the authors could clarify that.

### Soundness
3

### Presentation
2

### Contribution
2
