# An Investigation of Batch Normalization in Off-Policy Actor-Critic Algorithms

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Batch Normalization (BN) has played a pivotal role in the success of deep learning by improving training stability, mitigating overfitting, and enabling more effective optimization. However, its adoption in deep reinforcement learning (DRL) has been limited due to the inherent non-i.i.d.\ nature of data and the dynamically shifting distributions induced by the agent’s learning process. In this paper, we argue that, despite these challenges, BN retains unique advantages in DRL settings, particularly through its stochasticity and its ability to ease training. When applied appropriately, BN can adapt to evolving data distributions and enhance both convergence speed and final performance. To this end, we conduct a comprehensive empirical study on the use of BN in off-policy actor-critic algorithms, systematically analyzing how different training and evaluation modes impact performance. We further identify failure modes that lead to instability or divergence, analyze their underlying causes, and propose the Mode-Aware Batch Normalization (MA-BN) method with practical actionable recommendations for robust BN integration in DRL pipelines. We also empirically validate that, in RL settings, MA-BN accelerates and stabilizes training, broadens the effective learning rate range, enhances exploration, and reduces overall optimization difficulty. Our code is available at: \url{https://anonymous.4open.science/r/bn_rl468305485}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the effectiveness of applying Batch Normalization (BN) to Deep Reinforcement Learning (DRL), particularly within off-policy actor-critic (AC) frameworks. While BN is a well-known technique for stabilizing the training of deep learning models, its applicability to DRL is not straightforward due to the nature of the IID data from the replay buffer.

This paper empirically investigates a study focusing on a critical, yet underexplored, aspect: the distinct behaviors of BN's training mode versus its evaluation mode at different stages of the AC update.
The authors found the failure modes that collapse the stability and performance, for example, when BN is used in in training mode during the actor update (Critic-I step).

Based on this observation, the paper proposes Mode-Aware Batch Normalization (MA-BN), a set of simple, practical guidelines for setting the BN modes. They demonstrate empirically that MA-BN not only stabilizes training but also outperforms Layer Normalization (LN) and no-normalization baselines, accelerates convergence, and enhances exploration.

### Strengths
This paper provides a method for decomposing the problem of why BN fails and succeeds in DRL, especially in AC settings.

1) The notation is clear and easy to understand. They point out cases where BNs are used as Actor I and II, and Critic I, II, and III as described in Algorithm 1.

2) The analysis is comprehensive, covering multiple algorithms (DRQv2, SAC, DDPG, SD-SAC), benchmarks (DMC, Mujoco, Atari), and even related work (CrossQ in Appendix D).

3) The paper demystifies why BN has been so troublesome in RL, providing a concrete, actionable diagnosis.

4) It provides an immediate, simple, and "drop-in" solution (the MA-BN "ETT" critic config) for practitioners.

### Weaknesses
Although the paper explores the BNs in AC settings comprehensively, the justifications for the final MA-BN configuration are not uniformly rigorous. The analysis for Critic-I (Section 3.1) is conclusive, but the analyses for Critic-III and the Actor (3.2 and 3.3) are less definitive.

1) Justification for Critic-III T (Target Critic)

The analysis in Section 3.2 (Fig. 4) shows that ETT is strong, but ETE (eval mode) with soft-updated BN stats ("ETE+BN-soft") is also competitive and even superior in one environment. The paper's choice of ETT for "simplicity" feels slightly arbitrary, as "ETE+BN-soft" is arguably just as simple and perhaps more intuitive.

2) Justification for Actor TT
The analysis for the actor mode (Section 3.3, Fig. 5) is also inconclusive. "TT" is chosen as the default, but "ET" performs best for SAC on Mujoco. The authors justify "TT" as "maximizing stochasticity, but the results imply the optimal actor mode may be algorithm- or environment-dependent.

### Questions
1) Actor choice
The choice of Actor TT seems based on good performance on DMC, even though ET was best for SAC on Mujoco (Fig 5). This suggests the optimal actor mode might be algorithm-dependent. How critical is the TT choice for the DRQv2 results in Section 4? Would performance degrade significantly if you used ET instead?

2) Figure clarity
To help readers better understand the paper, I would suggest adding more explanation of the results in the figure captions. For example, the connection between BN effectiveness and the experiment and results in Section 4.4. (Figure 8) seems unclear. 

3) Method applicability
In this paper, the effectiveness of BN is investigated in online off-policy AC RL settings. Are the insights in the paper applicable to offline RL settings, where the estimation of values is more important and more complex due to the biased distribution between the dataset and the environment?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper explores how Batch Normalization (BN) behaves within deep reinforcement learning in off-policy actor-critic methods. Through empirical analysis, authors show that BN can still offer benefits when configured correctly. They uncover that improper selection between BN’s training and evaluation modes can lead to instability or divergence, and propose Mode-Aware Batch Normalization (MA-BN), as the best configuration. Authors show that MA-BN can improve exploration and hyperparameters stability.

### Strengths
Putting aside the motivation issues described later, the paper is technically well written. All experiments are clearly explained, and the figures actually demonstrate the phenomena being studied. I have no remarks on the design of the existing experiments. The only thing missing is the details necessary for reproducibility. I advise the authors to explicitly add all hyperparameters and describe the setup of each experiment in detail. Currently, this is not the case.

### Weaknesses
Overall, the paper is currently more like a collection of unrelated experiments without clear single motivation and lacks clear conclusive experiments.

If the goal is to demonstrate the advantages of BN over other normalization methods, more extensive comparisons with state-of-the-art algorithms are required (at least on the level of modern algorithms, like SimBa, BroNet, etc). Judging by the results in Figure 6, even the best found BN configuration doest not outperform LN. While authors do not claim state-of-the-art, this results undermines the inital motivation that BN is worth such attention at all. If LN works just fine, why bother? I think paper does not address this question convincingly (in contrast to CrossQ which focuses on simplification and faster training).

If the goal is to analyse unique properties that BN can give in comparison to other methods (like mentioned improved exploration), a more in-depth and comprehensive analysis is needed, including not only LN but other methods of regularization and exploration. I liked the experiments on improving exploration and stability to hyperparameters the most. These results, with more detailed development, may actually be novel and could be useful and interesting. However, very little attention is paid to them. Exploration studied in over simplified setup, which may not transfer to the more difficult and stochastic environment (good similar example is the recent study on the effect of small batch size on exploration, see https://arxiv.org/abs/2310.03882). What other properties make BN unique in comparison to the other methods? Such results would be significantly more insightful.

As for the main analysis regarding the BN modes, it is not convincing. If I understand results correctly, the authors are essentially reproduced CrossQ's conclusions for Critic 1. Moreover, authors write: “… reinforcing our claim that distribution mismatch is a key factor behind the degradation of training-mode performance”. Why this claim and it’s confirmation is novel, given the CrossQ results? Seems like CrossQ addressed this problem specifically and with similar methods before. As for the rest of the analysis of Critic 2-3, it is completely unnecessary in Cross-Q as it removes target networks (which is a good thing which we must strive for anyway). Given that CrossQ is already quite old (first appeared in 2019), and stood the test of time, I do not understand the motivation to study BN based on DRQ and not CrossQ. In conclusion, I believe that changing one hyperparameter is not sufficient for a separate method name.

Overall, reading the paper gives a mixed impression. To summarise, I feel that despite the (in my opinion) complete technical validity of the results, they are insufficiently significant and novel for publication at ICLR, and it would be more suitable for publication in for example TMRL, which explicitly values technical correctness over the possible impact.

### Questions
1. What are the advantages/motivation on using DRQ instead of Cross-Q?
2. Given the rising need for continual learning, do you expect that BN will perform on the same level in the streaming setups? (see https://arxiv.org/abs/2410.14606 or https://arxiv.org/abs/2411.15370)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper examines the use of Batch Normalization (BN) in off-policy actor-critic reinforcement learning, with a focus on how different BN modes - training vs. evaluation - affect learning dynamics. The authors propose Mode-Aware Batch Normalization (MA-BN) as a solution to observed instability issues, claiming that it enhances exploration and improves robustness to hyperparameter variation. While the premise might appear promising, the overall execution and depth of analysis fall short of making a meaningful contribution.

### Strengths
The paper is technically competent and clearly structured. Figures are informative and do illustrate the discussed effects. The idea of making BN mode selection adaptive (MA-BN) is at least conceptually reasonable, and the authors deserve credit for a clean empirical setup. However, beyond these basic merits, the work struggles to justify its necessity or originality.

### Weaknesses
The study lacks a coherent purpose and fails to articulate a strong motivation. It is unclear whether the intent is to promote BN as a superior normalization method or to simply catalogue its behavior under certain conditions. The findings largely reiterate what prior work -particularly CrossQ has already shown, yet without acknowledging the redundancy. More importantly, BN does not demonstrate any real advantage over Layer Normalization (LN), making the central claim weak and unconvincing. The exploration experiments, while potentially interesting, are shallow and performed in oversimplified environments that do not reflect the challenges of realistic RL tasks. Overall, the work feels incremental, derivative, and unlikely to influence future research directions.

### Questions
(1) Why choose DrQ over CrossQ as the foundation of the analysis? The decision to base the study on DrQ instead of the more relevant CrossQ framework seems arbitrary and poorly justified.

(2) How would BN perform in continual learning setups? BN fundamentally depends on batch-level statistics, which makes it ill-suited for non-stationary or streaming scenarios where data distributions evolve over time. Have the authors considered how MA-BN might behave under distribution drift? Would it require constant re-estimation of running statistics, or would that simply destabilize learning further?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a comprehensive empirical investigation of Batch Normalization (BN) in off-policy actor-critic algorithms. The authors analyze all possible BN placement and mode combinations (training vs evaluation) for both the actor and critic networks, identifying configurations that lead to instability or divergence. They find that using BN in evaluation mode for the critic’s actor-update step (Critic-I) avoids a distribution mismatch between the buffer and current policy, while training mode generally works best elsewhere. Based on this study, they propose Mode-Aware Batch Normalization (MA-BN), a principled configuration that mproves stability, accelerates convergence, widens the usable learning-rate range, and modestly enhances exploration.

### Strengths
- **Comprehensive and systematic study:** The paper carefully evaluates BN modes at every relevant point in the off-policy actor-critic pipeline, filling an important empirical gap in RL normalization literature.  
- **Clarity and motivation:** The motivation is clear. BN is underused in RL due to non-i.i.d. data, and the paper provides a principled exploration of why.  
- **Important Insights:** The results lead to a concrete, easy-to-implement recipe (MA-BN) that can directly guide practitioners.  
- **Relevance:** Understanding normalization behavior in off-policy RL is increasingly important for scalable visual and continuous-control settings.  
- **Presentation:** The paper is generally well-written and easy to follow, with thorough experiments on both continuous and discrete benchmarks.

### Weaknesses
1. **Terminology:** The use of “training” and “evaluation” modes for BN may confuse readers, since both modes are used during RL training. Terms like *instantaneous BN* (using batch statistics) vs *running-average BN* might reduce ambiguity.  
2. **Exploration claims:** The authors attribute enhanced exploration to BN’s stochasticity. While plausible, there is no explicit mechanism linking BN to exploration behavior; the effect could instead stem from smoother optimization or improved policy entropy. The claim should be more cautiously phrased.  
3. **Reward vs Return terminology:** Some figures and text conflate “reward” with “final return.” It should be episodic return instead of reward.
4. **Algorithmic clarity:**  
   - Algorithm 1 omits the replay-buffer storage step (`store(s_t, a_t, r_t, s_{t+1})`).  
   - The clipping in Actor-I vs Actor-II differs without clear justification.  
5. **Minor experimental details:**  
   - In Figure 2, the replay buffer sizes labeled “5w/10w/100w” likely mean “5k/10k/100k.”  
   - It would be helpful to explicitly state the default replay-buffer size used in main experiments (appears to be 100k).  
6. **Causal interpretation:** Some causal language (e.g., “BN enhances exploration”) should be softened to correlation or indirect influence.  


I think is a well-executed empirical study on a subtle but practically important issue in deep RL, so I'm leaning towards acceptance. My conrencs are addresabble and I'm willing to increase my score if the authors address them.

### Questions
- Why does Actor-I use different clipping parameters than Actor-II in Algorithm 1?  
- In Figure 2, what is the buffer size used in the first row, and what is the batch size used in the second row?
- Why is the “EE” configuration excluded instead of shown as a degenerate baseline? The authors quickly mentioned that "the absence of BN statistics updates degenerates the BN layer into a fixed affine transformation" without detailed explanation.

### Soundness
3

### Presentation
3

### Contribution
3
