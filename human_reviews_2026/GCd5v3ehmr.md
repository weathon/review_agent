# MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Developing Large Language Models (LLMs) to cooperate and compete effectively within multi-agent systems (MASs) is a critical step towards more advanced intelligence. While reinforcement learning (RL) has proven effective for enhancing reasoning in single-agent tasks, its extension to multi-turn, multi-agent scenarios remains underexplored due to the challenges of long-horizon credit assignment and agent-specific advantage estimation. To address these challenges, we introduce **MARSHAL**, an end-to-end RL framework that incentivizes **M**ulti-**A**gent **R**easoning through **S**elf-play wit**H** str**A**tegic **L**LMs in both cooperative and competitive games. MARSHAL features a turn-level advantage estimator that aligns learning signals with each interaction for credit assignment, and an agent-specific advantage normalization to stabilize multi-agent training. By learning with self-play across cooperative and competitive games, MARSHAL agents trained from Qwen3-4B develop strong strategic abilities, with up to $28.7$\% performance improvements in held-out games. More importantly, the capability acquired through self-play generalizes beyond games, yielding consistent performance gains of MASs in reasoning benchmarks. When integrated into leading MASs, our MARSHAL agent achieves significant zero-shot performance gains of up to $10.0$\% on AIME, $7.6$\% on GPQA-Diamond, and $3.5$\% on average across all benchmarks. These results establish self-play in strategic games as a powerful approach for developing generalizable multi-agent reasoning capabilities in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces MARS (Multi-agent RL method for LLMs), a new framework for training Large Language Models in multi-agent environments. The method includes: 
1) the introduction of turn-level rewards to enable more fine-grained credit assignment, 
2) an agent role-based advantage normalization technique, which aims to reduce training variance by calibrating advantage signals relative to the performance of rollouts from the same role,
3) self-play against a variety of partners opponents in competitive games.

The authors conduct experiments using Qwen3-4B as the backbone model. They train specialist agents on individual games representing diverse dynamics: Tic-Tac-Toe (competitive, deterministic), Kuhn Poker (competitive, uncertain), and Mini Hanabi (cooperative, uncertain). They also train a generalist agent (MARS-multi) on all games simultaneously. Evaluation is performed on held-out games with similar dynamics (Connect Four, Leduc Hold'em, Simple Hanabi). The results indicate that MARS specialist agents outperform baselines (SPIRAL, Qwen3-4B) on their respective held-out tasks. The paper also evaluates the agents on general STEM reasoning benchmarks, finding mixed results in a single-agent setting but small, consistent improvements when integrated into a multi-agent framework (AutoGen). The ablation suggests playing with mixed opponents is critical.

### Strengths
- The algorithmic contributions are very clear and well motivated.  Turn-level reward shaping for credit assignment and role-conditioned advantage normalization are simple, intuitive modifications to GRPO that target known pain points (variance, difficulty credit assignment).
- The experimental design is solid, as the choice of training and testing environments is commendable. The authors systematically cover competitive, cooperative, deterministic, and uncertain dynamics. Evaluating on held-out games (e.g., training on Tic-Tac-Toe and testing on Connect Four) provides a solid test of generalization to similar dynamics, which is a key claim of the paper.
- The results are positive when tested in case of generalization. We see that MARS agents outperform baselines on held-out games (Connect Four, Leduc Hold'em) is a significant result. This suggests the method is not just overfitting to the training games but is learning underlying strategies for that specific dynamics.

### Weaknesses
- All experiments use small, two-player environments with constrained action/communication spaces. This limits external validity; N-player settings introduce additional non-stationarity, credit assignment, and population-diversity challenges, so evidence scaling beyond 2-player cases is important to substantiate generality claims.

### Questions
- Ablation results in Table 5 shows that without either turn-level reward or agent-specific advantage normalization resulted in significant performance drop in Tic/Kuhn trained models, but not necessarily true for Hanabi-trained model in Kuhn Poker and Leduc Hold'em settings, do you have any intuition on why that is the case?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MARS, an RL framework for training LLMs via self-play using a mixed curriculum of cooperative (Hanabi) and competitive (Tic-Tac-Toe, Kuhn Poker) games.1 The authors claim this training, supported by two methodological tweaks to the GRPO algorithm ("turn-level advantage estimation" and "agent-specific advantage normalization") 1, leads to skills that generalize. The experimental results, particularly the transfer of these skills to downstream multi-agent systems like AutoGen and MAD 1, are quite strong and genuinely interesting.

However, I have some *very significant* concerns about the paper's framing, its engagement with closely related prior work, and the experimental design, all of which muddy the central claims of methodological novelty.

### Strengths
This is a solid piece of work, and I want to start by highlighting what I think is really excellent here.

1. **Originality & Significance (The Core Idea):** The most significant contribution here, in my view, is the demonstration of *skill transfer from a simple game curriculum to complex, downstream multi-agent systems*. This is a really big deal. Prior work like SPAG or SPIRAL showed self-play can boost *single-agent* reasoning benchmarks, but this paper is the first I've seen to convincingly argue that these skills transfer to *cooperative and competitive software systems* like AutoGen and MAD. The 10.0% gain on AIME and 12.5% on GPQA-Diamond are non-trivial and suggest this is a very promising research direction.
2. **Quality (Experimental Rigor):** The experimental setup is high-quality and, in one specific area, exceptionally well-designed.
   - The evaluation is comprehensive: held-out games (Table 1) , single-agent benchmarks (Table 2) , and downstream MAS (Table 2).
   - **Skill-System Mapping:** This is, frankly, the strongest part of the paper. The analysis showing that the agent trained on adversarial games (Tic-Tac-Toe) excels in the *competitive* MAD framework, while the agent trained on the *cooperative* game (Hanabi) excels in the *collaborative* AutoGen framework, is fantastic. This provides powerful evidence for the central thesis that MARS is teaching distinct, transferable skills, not just generically "making the model smarter."
   - **Ablations (at first glance):** The ablation studies in Table 4 (showing self-play beats a fixed opponent) and Table 5  (showing the method components are necessary) seem to validate the design choices. (I have major questions about these, see Weaknesses, but they are presented as being thorough).
3. **Clarity:** The paper is very well-written and easy to follow. The central ideas are communicated effectively.
   - **Figure 2** does an excellent job of visually explaining the flaw in the naive GRPO and how the proposed "sum-then-normalize" method is different.
   - **Table 3** is a superb piece of qualitative evidence. Showing the "Intent Recognition" thought-trace from Hanabi ("Maybe they want me to play this card?") mapping to AutoGen ("Maybe the assistant is not sure?") makes the quantitative results in Table 2 so much more believable.
4. **Significance (Broader Impact):** If these results hold, this work provides a scalable, data-efficient paradigm for "social alignment." The idea of autonomously training agents in a "social sandbox" to develop complex, generalizable skills like "theory of mind" is highly significant for both the MARL and LLM-agent communities.

### Weaknesses
Despite the strong results, I have major problems with the paper's claims to *methodological* novelty. The authors have either missed or ignored extremely relevant, contemporaneous work, and the experimental design fails to decouple the most important variables.

1. **Major Novelty Issue: "Agent-Specific Normalization" vs. SPIRAL's RAE.**
   - **The Problem:** The authors claim "agent-specific advantage normalization" is a *novel technique* to solve the problem of "heterogeneous roles" in multi-agent training.
   - **The Criticism:** The paper's *own primary baseline*, **SPIRAL (Liu et al., 2025)** , explicitly identifies and solves *the exact same problem*. SPIRAL calls its solution **"Role-conditioned Advantage Estimation (RAE)"**. Based on its description, RAE *also* computes separate advantages, $A_0(s, a)$ and $A_1(s, a)$, for each role".
   - **The Evidence:** From all available information, MARS's "agent-specific normalization" and SPIRAL's "RAE" appear to be functionally identical. Yet, the authors *never once mention RAE* in the entire paper, even when citing SPIRAL. The ablation study ("w/o agent-specific" in Table 5) only proves that *some* kind of role-based normalization is needed, it does *not* prove that the MARS version is new or superior to RAE. This is a major, major oversight and frankly misrepresents the methodological contribution.
2. **Confounding Variables: The Method vs. The Curriculum.**
   - **The Problem:** The authors present MARS as a complete package (Method A + Curriculum X) and show it beats SPIRAL (Method B + Curriculum Y).
   - **The Criticism:** This is a classic confounding variable problem. The MARS "package" differs from SPIRAL in *two* key ways: (1) the advantage estimation algorithm and (2) the training curriculum (MARS adds cooperative games like Hanabi 1, while SPIRAL is purely zero-sum).
   - **The Evidence:** The paper provides *no evidence* to decouple these factors. How do we know the impressive gains on AutoGen, AIME, and GPQA aren't *entirely* due to the superior *curriculum* (i.e., adding Hanabi was the only thing that mattered) and have *nothing* to do with the paper's proposed "sum-then-normalize" or "agent-specific" algorithms? The paper is just completely silent on this.
   - **What's Missing:** The critical missing experiment is: **SPIRAL's algorithm (with RAE) trained on MARS's mixed curriculum.** If that setup achieves the same results as MARS, then this paper's methodological claims are largely invalidated. This is a huge hole in the experimental design.
3. **Concurrent Novelty: "Turn-Level" Credit Assignment.**
   - **The Problem:** The authors correctly identify a "critical flaw" in the original GRPO "Process Supervision" setting, which uses a "normalize-then-sum" approach, and propose a "sum-then-normalize" fix.
   - **The Criticism:** They weren't the only ones to spot this. A paper on OpenReview (**`h83vIG5Hre`**, pub. **June 8, 2025**) *also* identified that trajectory-level estimation is "inadequate" and proposed a "fine-grained turn-level advantage estimation" strategy.
   - **The Evidence:** This concurrent paper proposed **"MT-GRPO"**, which uses a *different* fix (a weighted sum of independently normalized turn-level and outcome-level advantages, e.g., $\hat{A}_{i,1} = \hat{A}^{T}_{i} + \lambda\hat{A}^{O}_{i}$). This means the "best way to fix GRPO" was an open question at the time of submission. MARS presents its solution as *the* solution, without acknowledging or comparing it to other concurrent proposals. This again weakens the claim of methodological novelty.
4. **Misleading Framing: The "Novelty" of Mixed-Motive Games.**
   - **The Problem:** The paper repeatedly frames its contribution as "The inclusion of cooperative games is a key differentiator" and contrasts this with the "narrow, adversarial-only mindset of prior work".
   - **The Criticism:** This is an oversimplification and a bit of a strawman. The *concept* of using mixed-motive games is not new.
   - **The Evidence:** The authors *themselves* cite **Xu et al. (2023)** , which used the "Werewolf" game. The "Werewolf" paper *explicitly* states the game "involves both cooperation and competition".The *real* novelty of MARS isn't the *idea* of using a non-adversarial game; it's the demonstration that a *simple, modular curriculum* of prototype games (pure-coop, pure-comp) can *generalize to downstream MAS*. This is a much stronger and more accurate claim, and I'd urge the authors to reframe their contribution this way.

### Questions
My overall rating is "Borderline Accept" because the *results* (the generalization to AutoGen/MAD) are genuinely exciting and significant. However, my final decision is entirely contingent on the authors' ability to provide satisfactory answers to the following questions, which directly address the weaknesses I've outlined.

1. **"Agent-Specific Normalization" vs. SPIRAL's RAE:** I'm going to be very direct here. Your "agent-specific advantage normalization" appears to be functionally identical to the "Role-conditioned Advantage Estimation (RAE)" proposed in **SPIRAL (Liu et al., 2025)** 5, which is your main baseline.
   - Can you please provide a clear, algorithmic explanation of how your method *differs* from RAE?
   - If it is not different, this must be acknowledged. If it *is* different, why did you not compare against RAE in your ablations? This really needs to be addressed, as it undermines a core methodological claim.
2. **Decoupling Method vs. Curriculum:** Your key comparison against SPIRAL is confounded. You changed both the algorithm *and* the game curriculum. How can you be sure that your performance gains are not *entirely* due to your superior curriculum (i.e., just adding Hanabi)? To support your methodological claims, you *must* provide evidence that decouples these factors. Can you provide results for either:
   - **(a)** The SPIRAL algorithm (with RAE) trained on *your* mixed (Tic/Kuhn/Hanabi) curriculum?
   - **(b)** Your MARS algorithm trained on SPIRAL's *adversarial-only* curriculum?
3. **Regarding Concurrent Work on GRPO:** Your "sum-then-normalize" fix for GRPO's process supervision flaw is one solution. However, concurrent work from June 2025 (OpenReview `h83vIG5Hre`) 8 proposed an alternative, **"MT-GRPO"**. Could you please comment on why your solution is preferable to theirs (a weighted, multi-signal advantage)?
4. **Regarding Novelty Framing:** You claim the "inclusion of cooperative games is a key differentiator". Given that you cite **Xu et al. (2023)** , which used the "Werewolf" game (a mixed cooperation/competition game), would you be willing to reframe your novelty? It seems to me the real, and more impressive, contribution is demonstrating *generalization from a simple, modular curriculum of game prototypes to complex, downstream MAS*.
5. **Ablation Result in Table 5:** In Table 5, you ablate the "agent-specific" component. The performance drop for the Hanabi specialist is described as "mild" (55.55 -> 52.50). This seems counter-intuitive. Hanabi is an asymmetric, imperfect-information *cooperative* game where player roles and perspectives are fundamentally different and critical to success. Why is the effect so much smaller here than in the adversarial games? Does this suggest your component is less crucial for cooperative scenarios?

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
MARS is an end-to-end reinforcement learning framework that enhances multi-agent reasoning capabilities of large language models through self-play in strategic games, achieving 28.7% performance improvement in games and 10.0% and 12.5% gains on multi-agent reasoning benchmarks AIME and GPQA-Diamond, respectively.

### Strengths
1. The paper introduces a genuinely novel approach by adapting GRPO (Group-Relative Policy Optimization) to multi-agent settings, which hasn't been explored before in this context

2. The framework moves beyond traditional supervised learning approaches for multi-agent LLMs, establishing self-play as a viable training paradigm

3. The experimental design is comprehensive, covering diverse game types and reasoning benchmarks

### Weaknesses
The paper significantly weakens its contribution by failing to compare against recent and relevant multi-agent reasoning methods that have shown strong performance. Missing Comparison with like AFlow, ToT...

### Questions
Weak Transfer Learning Justification

The paper doesn't adequately explain why strategic game skills transfer to math problems (AIME, MATH-500....) or science questions (GPQA) (Reasoning problem). The connection between game-playing abilities and general reasoning is assumed but not validated
No analysis of which specific game skills contribute to reasoning improvements.

The paper doesn't discuss failure cases or when MARS performs poorly.

### Soundness
3

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
3

### Summary
The paper proposes MARS, an end-to-end reinforcement learning framework to improve multi-agent reasoning in LLMs via self-play across both cooperative and competitive strategic games. The key issue solved is that, direct applying GRPO to the multi-turn, multi-agent structure of self-play will introduce significant challenges of long-horizon credit assignment and agent- specific advantage estimation. MARS propose a turn-level, sum-then-normalize advantage estimator for long-horizon credit assignment within multi-turn trajectories, and Agent-specific advantage normalization to handle heterogeneous roles and asymmetric payoff scales in multi-agent settings. Plenty of experiments show that MARS trains Qwen3-4B can bring Improved strategic performance.

### Strengths
1. The problem motivation and the solution is very clear, i.e. long-horizon credit assignment and role heterogeneity for multi-turn multi-agent RL with LLMs.
2. The method is very simple as an improvement to GRPO, including Turn-level advantage estimator and Agent-specific advantage normalization, 
3. The experimental setting is diverse, including a portfolio of six strategic, two-player games to cultivate a mixed set of multi-agent capabilities, both debate (MAD) and cooperation (AutoGen), as well as fixed opponents (MCTS/NE/CFR) and detailed MAS benchmarks (MATH, GSM8K, AQUA-RAT, AIME24, AMC23, MMLU-STEM, GPQA-Diamond).
4. When integrated into leading multi-agent systems,  MARS agent achieves significant performance gains of 10.0% on AIME and 12.5% on GPQA-Diamond.

### Weaknesses
1. The sum-then-normalize estimator centers on batch means across turns/trajectories. It can stabilize the result, but discards the value function structure and may be sensitive to batch composition across diverse games.
2. Agent-specific normalization is defined over role-based subgroups. In more complex settings (N-player, heterogeneous roles, non-stationary curricula), subgroup granularity and sample efficiency could be problematic.
3. The ability of transferring to MAS may conflate multiple factors. The MAS gains is driven by better turn-taking discipline, concise formatting, or truly strategic multi-agent reasoning? 
4. Scaling. 200 training steps and 4B base model are modest. Can the performance gain be obtained for the MAS on 7B/14B or larger models.
5. The contribution is mainly about the advantage function computation, lacking of improvement about topology or collaboration method

### Questions
1. In eq.4, how the parameters alpha and l are choose? Is the performance is sensitive to values?
2. For agent-specific normalization, if the sub-groups are sub-groups (e.g., one role acts more frequently or has lower variance rewards)? Is the formulation adaptive to weight or normalize per subgroup?
3. In MAD/AutoGen, do competitive-trained agents consistently outperform cooperative-trained agents in debate, and vice versa? 
4. When generalized to multiagent system, do the MARS models need further optimization? Can the author give more explanation, why the MARS collaborative and competitive capabilities are useful for mathematics and general QA task? In tab.1 MARS(multi.) doesn’t have the best performance when testing.

### Soundness
3

### Presentation
3

### Contribution
2
