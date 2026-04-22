# Offline Equilibrium Finding in Extensive-form Games: Datasets, Methods, and Analysis

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Recently, offline reinforcement learning (Offline RL) has emerged as a promising paradigm for solving real-world decision-making problems using pre-collected datasets. However, its application in game theory remains largely unexplored. To bridge this gap, we introduce ***offline equilibrium finding*** (Offline EF) in extensive-form games (EFGs), which aims to compute equilibrium strategies from offline datasets. Offline EF faces three key challenges: the lack of benchmark datasets, the difficulty of deriving equilibrium strategies without access to all action profiles, and the impact of dataset quality on effectiveness. To tackle these challenges, we first construct diverse offline datasets covering a wide range of games to support algorithm evaluation. Then, we propose BOMB, a novel framework that integrates behavior cloning within a model-based method, enabling seamless adaptation of online equilibrium-finding algorithms to the offline setting. Furthermore, we provide a comprehensive theoretical analysis of BOMB, offering performance guarantees across various offline datasets. Extensive experimental results show that BOMB not only outperforms traditional offline RL methods but also achieves highly efficient equilibrium computation in offline settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces the problem of Offline Equilibrium Finding (Offline EF) in imperfect-information extensive-form games (IIEFGs), where the goal is to compute equilibrium strategies using only pre-collected datasets without environment interaction. The authors propose the BOMB framework, which combines behavior cloning (BC) and model-based (MB) methods, and provide theoretical guarantees under strong dataset coverage assumptions. They also contribute a set of offline datasets for benchmarking, collected using random, expert, and learning-based policies. Extensive experiments are conducted on small-scale games to validate BOMB against offline RL baselines and its individual components.

### Strengths
1. Offline EF is a meaningful and underexplored direction that bridges offline learning and game theory. The paper clearly motivates the problem and differentiates it from offline RL and online equilibrium finding.

2. The paper includes extensive experiments covering two-player, multi-player, and hybrid dataset settings.

### Weaknesses
1. The theoretical guarantees rely on strong assumptions such as uniform coverage or ε-equilibrium coverage, which are impractical in real-world settings. The analysis does not address more realistic, partial-coverage settings. In Theorem 4.2, dataset $\mathcal D$ requires sufficient state reached, and the BC strategy is directly weighted into dataset $\mathcal D$. This implies that the $\varepsilon$-equilibrium is bounded with $\alpha$, and if $\alpha$ is too small, $|\mathcal D|$ must be very large. Moreover, I find the proof section of the main text to be rather redundant.

2. All experiments are conducted on small-scale games that can be solved exactly with tabular methods. There is no evaluation on larger games or real human gameplay data, limiting the claim of practical impact.

3. The datasets are generated using high-variance RL algorithms (e.g., Deep CFR, PSRO), which contradicts the motivation of avoiding simulators and may not reflect real-world data distributions.

4. The NashConv values are high compared to standard online methods, and the performance of BOMB in fully offline settings without fine-tune remains unclear.

5. Overstated claim in (Line 52) " there has been no dedicated study addressing the offline setting in multi-player games". See [1-12]. I haven't read all the papers in recent years, so there should be more similar works.

6. The algorithm BOMB is not novel in IIEFGs. Combine behavior strategy and equilibrium strategy is a common refinement in solving IIEFGs [13, 14]. The algorithm appears rather inefficient, and the training overhead seems quite substantial.

7. The BC strategy and MB strategy appear to be trained separately, which leads to a problem: the MB strategy cannot handle the slight suboptimality of the EFG structure. For instance, P1 optimal strategy should be to execute action A (with an expected value of +1), but if P1 execute action B (with an expected value of +0.99), the MB strategy will not know how to respond when Player 1 executes action B. For instance, if P1 acts B, then Player 2's expected value for action C is -1 and action D is 0, Player 2's MB strategy would clearly choose action D. However, within the BOMB framework, this MB strategy would not be sampled at all.

8. Relevant works have been compiled only up to 2021. In recent years, considerable work has been undertaken towards resolving IIEFG, including [15-27].

[1] Yuheng Jing, Kai Li, Bingyun Liu, Yifan Zang, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng. Towards Offline Opponent Modeling with In-context Learning. ICLR 2024

[2] Hang Xu, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng. Dynamic Discounted Counterfactual Regret Minimization. ICLR 2024

[3] Jingxiao Chen, Weiji Xie, Weinan Zhang, Yong Yu, Ying Wen. Offline Fictitious Self-Play for Competitive Games. arxiv 2024

[4] Runyu Lu, Yuanheng Zhu, Dongbin Zhao. Constrained Exploitability Descent: An Offline Reinforcement Learning Method for Finding Mixed-Strategy Nash Equilibrium. ICML 2025

[5] Junren Luo, Wanpeng Zhang, Mingwo Zou, Jiongming Su, Jing Chen. Offline PSRO with Max-Min Entropy for Multi-Player Imperfect Information Games. 2022 China Automation Congress

[6] Hai Zhong, Xun Wang, Zhuoran Li, Longbo Huang. Offline-to-Online Multi-Agent Reinforcement Learning with Offline Value Function Memory and Sequential Exploration. AAMAS 2025

[7] Weijun Zeng, Yinghao Li, Xiaosi Chen, Zijie Chang, Fei Ge. GNN-ReBeL: Enhancing Neural Belief Representation for Imperfect-Information Games. IEEE SMC 2025

[8] Weijun Zeng, Yinghao Li, Xiaosi Chen, Zijie Chang, Fei Ge. An Investigation of Subgame Depth in ReBeL: Impact on Convergence and Performance in Imperfect-Information Games. IEEE SMC 2025

[9] David Sychrovský, Michal Šustr, Elnaz Davoodi, Michael Bowling, Marc Lanctot, Martin Schmid. Learning Not to Regret. AAAI 2024

[10] David Sychrovský, Martin Schmid, Michal Šustr, Michael Bowling. Meta-Learning in Self-Play Regret Minimization. arxiv 2025

[11] Fuxiang Zhang, Chengxing Jia, Yi-Chen Li, Lei Yuan, Yang Yu, Zongzhang Zhang. Discovering Generalizable Multi-agent Coordination Skills from Multi-task Offline Data. ICLR 2023

[12] Zhengdao Shao, Liansheng Zhuang, Houqiang Li, Shafei Wang. COPSRO: An Offline Empirical Game Theoretic Method With Conservative Critic. IEEE TNNLS 2025

[13] Gabriele Farina, Christian Kroer, Tuomas Sandholm. Regret Minimization in Behaviorally-Constrained Zero-Sum Games. ICML 2017

[14] Brian Hu Zhang, Tuomas Sandholm. On the Outcome Equivalence of Extensive-Form and Behavioral Correlated Equilibria. AAAI 2024

[15] Noam Brown, Anton Bakhtin, Adam Lerer, Qucheng Gong. Combining Deep Reinforcement Learning and Search for Imperfect-Information Games. NeurIPS 2020

[16] Gabriele Farina, Christian Kroer, Tuomas Sandholm. Faster Game Solving via Predictive Blackwell Approachability: Connecting Regret Matching and Mirror Descent. AAAI 2021

[17] Brian Hu Zhang, Tuomas Sandholm. Subgame solving without common knowledge. NeurIPS 2021

[18] Hang Xu, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing. AutoCFR: Learning to Design Counterfactual Regret Minimization Algorithms. AAAI 2022

[19] Enmin Zhao, Renye Yan, Jinqiu Li, Kai Li, Junliang Xing. AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning. AAAI 2022

[20] Martin Schmid, Kevin Waugh, Matej Moravčík, Nolan Bard, Neil Burch, Rudolf Kadlec, Finbarr Timbers, Marc Lanctot, Josh Davidson, G. Zacharias Holland, Elnaz Davoodi, Alden Christianson, Michael Bowling. Student of games: A unified learning algorithm for both perfect and imperfect information games. Science 2023

[21] Stephen Marcus McAleer, Gabriele Farina, Marc Lanctot, Tuomas Sandholm. ESCHER: Eschewing Importance Sampling in Games by Computing a History Value Function to Estimate Regret. ICLR 2023

[22] Weiming Liu, Haobo Fu, Qiang Fu, Wei Yang. Opponent-Limited Online Search for Imperfect Information Games. ICML 2023

[23] Linjian Meng, Zhenxing Ge, Pinzhuo Tian, Bo An, Yang Gao. An Efficient Deep Reinforcement Learning Algorithm for Solving Imperfect Information Extensive-Form Games. AAAI 2023

[24] Boning Li, Zhixuan Fang, Longbo Huang. RL-CFR: Improving Action Abstraction for Imperfect Information Extensive-Form Games with Reinforcement Learning. ICML 2024

[25] Yuheng Jing, Bingyun Liu, Kai Li, Yifan Zang, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng. Opponent Modeling with In-context Search. NeurIPS 2024

[26] Boning Li, Longbo Huang. Efficient Online Pruning and Abstraction for Imperfect Information Extensive-Form Games. ICLR 2025

[27] Linjian Meng, Tianpei Yang, Youzhi Zhang, Zhenxing Ge, Shangdong Yang, Tianyu Ding, Wenbin Li, Bo An, Yang Gao. Efficient Last-Iterate Convergence of Counterfactual Regret Minimization Algorithms. NeurIPS 2025

[28] Matthew Thomas Jackson, Uljad Berdica, Jarek Luca Liesen, Shimon Whiteson, Jakob Nicolaus Foerster. A Clean Slate for Offline Reinforcement Learning. NeuIPS 2025

### Questions
1. Definition 2.1 assumes a unique equilibrium strategy $\sigma^*$. How is the gap metric defined when multiple equilibria exist? I found this metric unreasonable. Why not use NashConv?

2. The state representation includes information sets, game-level info, and available actions. How does BOMB encoding different states in various games?

3. What are the computational and statistical barriers to applying BOMB to larger games, and how can they be addressed?

4. How does BOMB compare to recent online methods like Deep CFR and PSRO? Can we fine-tune BOMB model to achieve better performance?

5. Can you show results against MoBRAC [28]?

### Soundness
2

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
The paper introduces "offline equilibrium finding", which is the problem of finding a Nash equilibrium of a (in general, imperfect-information, sequential) game. Various algorithms are studied, including a model-based approach (in which a model is learned from the dataset, and then fed to known equilibrium computation algorithms); a behavioral cloning approach (in which one attempts to train a strategy that directly emulates the strateg(ies) used to generate the data); and a combination of these, which the authors call the Behavior clOning and Model-Based (BOMB) method.

### Strengths
The problem of learning a game from an offline dataset seems interesting and important to me. The paper is clear, and the algorithms presented are simple and seemingly effective in experiments, at least in the two-player zero-sum setting.

### Weaknesses
* Appendix D raises more questions than it answers for me. If I understand correctly, Cui and Du (2022) showed that, under the unilateral coverage assumption, a Nash equilibrium can be computed in a Markov game, essentially by making pessimistic assumptions about the values in the remainder of the game. But the counterexample in Theorem D.5 *is* a Markov game, if you view the decisions of the two players as simultaneous, which is allowable in this case. Another way of saying this is the following: in the counterexample in Theorem D.5, $(a_1, b_1)$ is always *an* equilibrium, no matter the value of the '?' node. Thus BC works (and I think this implies that BC always works under unilateral coverage assumption?) But it might not be the *unique* equilibrium, so MB does not necessarily work.

  As far as I can tell, this is a (perhaps unnecessary) weakness of the techiques presented in this paper: indeed, Zhang and Sandholm [1] have a similar "pessimism-optimism"-style algorithm for finding equilibria of incomplete extensive-form game models, and in their paper, it certainly suffices for the model ("pseudogame", in their paper) to have what the present paper calls unilateral coverage (and infinite data). I think if you make a similar "pessimistic" assumption, MB should also work under unilateral coverage. Thoughts?

  On a similar note, Zhang and Sandholm [1] have similar ideas to the Cui and Du (2022) paper, for extensive-form games instead of Markov games. Perhaps that is the more relevant citation, since this paper concerns extensive-form games; indeed, I'd like to see some discussion about the results in this paper as compared to [1].

* Experiments, RQ3-4: why not use MB-CFR for CCE computation? CFR is guaranteed to find CCEs, but not Nash, so this feels backwards to me. 
* Related question: What is "NashConv" for MB-CFR multi-player games? Is it the Nash gap of the "marginalized" strategy created by taking the marginals of the correlated strategy profile created by CFR? If so, this should be stated explicitly. 
* L446: worth mentioning here that computing Nash equilibria, even in normal-form games, is PPAD-hard beyond the two-player zero-sum case (see e.g., [2]), so there are some limitations regarding what one can hope to do efficiently here.

Nitpicks and minor issues:
- "X is guaranteed to be Y if and only if Z" is the same statement as "X is Y if Z". If you mean that "X is Y if Z, and moreover if any of the conditions in Z is broken then X is not necessarily Y" (which you do seem to mean), you should explicitly state the second part as well.
- "States" and "histories", as defined and used by this paper, are basically the same thing, since your definition of "state" basically uniquely identifies a history. I'd pick one of these two words/notations (probably "histories", to align with the EFG literature) and stick to it, for consistency. If for some reason it is relevant that the history is actually represented as a tuple of those items rather than just some abstract representation, you can say something like "We identify histories with tuples of the form ..."
- $\epsilon$-equilibrium coverage is a stronger condition than uniform coverage, right? Because uniform coverage is just the condition $C(s, a) > 0$ for all $(s, a)$?

[1] BH Zhang, T Sandholm (NeurIPS 2020), "Small Nash equilibrium certificates in very large games"

[2] X Chen, X Deng, SH Teng (JACM 2009), "Settling the complexity of computing two-player Nash equilibria"

### Questions
1. Perhaps a strange question, but might be good food for thought: are you assuming that the dataset $\mathcal D$ comes from some (possibly correlated) strategy profile? i.e., it is possible that, for two distinct states $s, s'$ in the same information set of a player $i$, the distribution $\mathcal D$ assigns different distributions over the actions at $s$ and $s'$, hence "breaking" the information set. Do you allow this? If so, how do you deal with this, especially in behavioral cloning? If not, have you thought about what would happen if you do allow it?
2. Any responses to anything I've said above?

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
This paper proposes a hybrid framework, BOMB, for offline equilibrium finding in extensive-form games. The core idea is to mix the behavior cloning (BC) policy from the offline dataset with the policy learned by online model-based (MB) equilibrium-learning algorithms, including CFR and PSRO. The authors provide a dataset with random or expert game trajectories and learn game models for subsequent online learning. Theoretically, they provide necessary and sufficient conditions (about data coverage) for the MB and BC policies to approach equilibrium. Through experiments, they verify that under a learning-based estimation of the policy mixing parameter, BOMB outperforms BC, MB, and two applicable offline RL algorithms.

### Strengths
1. It is important to examine the open problem of offline equilibrium finding, especially under the imperfect-information setting.

2. This paper makes significant efforts in creating offline datasets and demonstrating the advantage of using a hybrid policy in the offline setting, providing both theoretical and empirical analysis.

### Weaknesses
1. From my perspective, the advantage of BOMB mainly comes from the existence of a learnable mixing parameter. However, the paper does not explain how the learning-based estimation method (Figure 3) works. Estimating the mixing parameter simply based on a similarity vector is intuitively unsound, since it completely ignores the game structure or the distribution of offline data.

2. While Theorem 4.4 and Theorem 4.5 seem correct to me, the proof of Theorem 4.6 is quite insufficient. Line 1308 suggests that Theorem 4.4 guarantees the Nash gap of the MB policy is smaller than that of the BC policy, which is not true even in the authors' own experiment of 5-player Kuhn poker (Figure 8). Even if the statement is correct, arbitrarily mixing the two policies could result in a performance drop rather than improvement without further theoretical analysis.

3. The authors claim that the MB framework’s performance is shown to be independent of the specific algorithm used. This is quite counterintuitive and only verified in the simplest 2-player Kuhn poker for CFR and PSRO (Figure 19).

4. The typos should be checked and corrected in the paper. For example, on Page 4, it should be "Liar's Dice" in Figure 2, "state" in Line 174, and "CFR (Zinkevich et al., 2007)" in Line 204.

### Questions
See Weaknesses.

### Soundness
2

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
2

### Summary
This paper argues that offline equilibrium finding is under-explored, and proposes new datasets, a new method (BOMB), and experiments validating the new method.

### Strengths
The research direction is promising and several experiments are performed.

### Weaknesses
I wish there was more on the related works.

It is mentioned that there is existing work that does offline learning in Markov games. I would have appreciated more insight into this previous research and why or why not it doesn't work for EFGs.

I would have appreciated more insight into previous research on real-world games like football, such as papers by Karl Tuyls et al. like "Game Plan: What AI can do for Football, and What Football can do for AI": how does this line of research relate or not relate to offline equilibrium finding?

And there do seem to be some papers that come up when one searches for "offline multi-agent reinforcement learning". How do these apply or not apply to offline equilibrium finding?

### Questions
see "weaknesses"

### Soundness
3

### Presentation
3

### Contribution
2
