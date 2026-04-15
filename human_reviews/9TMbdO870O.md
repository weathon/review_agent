# H2IL-MBOM: A Hierarchical World Model Integrating Intent and Latent Strategy as Opponent Modeling in Multi-UAV Game

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3

## Abstract
In the mixed cooperative-competitive scenario, the uncertain decisions of agents on both sides not only render learning non-stationary but also pose a threat to each other's security. Existing methods either predict policy beliefs based on opponents' interactive actions, goals, and rewards or predict trajectories and intents solely from local historical observations. However, the above private information is unavailable and these methods neglect the underlying dynamics of the environment and relationship between intentions, latent strategies, actions, and trajectories for both sides. To address these challenges, we propose a Hierarchical Interactive Intent-Latent-Strategy-Aware World Model based Opponent Model (H2IL-MBOM) and the Mutual Self-Observed Adversary Reasoning PPO (MSOAR-PPO) to enables both parties to dynamically and interactively predict multiple intentions and latent strategies, along with their trajectories based on self observation. Concretely, the high-level world model fuses related observations regarding opponents and multi-learnable intention queries to anticipate future intentions and trajectories of opponents and incorporate anticipated intentions into the low-level world model to infer how opponents' latent strategies react and their influence on the trajectories of cooperative agents. We validate the effectiveness of the method and demonstrate its superior performance through comparisons with state-of-the-art model-free reinforcement learning and opponent modeling methods in more challenging settings involving multi-agent close-range air-combat environments with missiles.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents H2IL-MBOM, a hierarchical model for opponent modeling in multi-agent reinforcement learning, particularly in air combat scenarios. H2IL-MBOM combines high-level intention inference with low-level strategy prediction to address non-stationary dynamics in mixed cooperative-competitive settings. Integrated into the PPO framework, this model achieves enhanced accuracy and interpretability, showing improved performance over baseline methods in simulations.

### Strengths
1.	The approach of modeling opponents through world models in air combat scenarios is innovative.

2.	H2IL-MBOM models opponents based on observational data, providing a useful approach for scenarios such as air combat, where direct access to the opponent's precise actions and states is unavailable.

3.	Comprehensive experiments in Gym-Jsbsim demonstrate the method’s significant performance advantages over model-free MARL and other opponent modeling methods, including ablation studies that confirm module effectiveness. Sufficient details of the experimental implementation are also provided.

4.	In the experimental section, Figure 3 and Appendix A.13 effectively demonstrate and validate that the proposed method can capture changes in opponent intentions in the air combat environment.

### Weaknesses
1.	The paper’s expression and presentation lack clarity; the authors provide numerous equations for various modules, making it difficult to smoothly understand the intent and overall functionality of the H2IL-MBOM framework. The extensive use of abbreviations also confuses readers. Figure 1, intended as an overview of H2IL-MBOM, includes excessive module details, which makes it challenging for readers to grasp the authors' main ideas. Consider breaking it down or omitting unnecessary details.

2.	Although applying the method of modeling opponents through world models in the air combat environment is innovative, I still hope the authors can conduct comparative experiments in more multi-agent adversarial environments, such as Google Football, and compare against additional baselines to demonstrate the advantages of the proposed method, especially since there is relatively less existing work in MARL for air combat environments.

3.	There are still some type errors, such as in the first paragraph of Section 3, where it says, "and using these these predictions along with observations to inform decision-makings."

4.	The authors do not provide a detailed analysis or validation of the effectiveness of the opponent model in the methods and experimental sections.

### Questions
1.	Is H2IL-MBOM equally effective in other multi-agent tasks and environments?

2.	Is there a rationale for the design of the action space, state space, and reward function used in the reinforcement learning (RL) framework?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces H2IL-MBOM, a hierarchical world model that integrates intent and latent strategy for opponent modeling in multi-UAV games. It addresses challenges in mixed cooperative-competitive scenarios by enabling dynamic prediction of opponents' intentions and strategies. The proposed MSOAR-PPO algorithm allows for real-time inference of adversaries' strategies and intentions, facilitating rapid adaptation to changes in opponents' behaviors. The method's effectiveness is demonstrated through comparisons with state-of-the-art methods in multi-agent air-combat simulations, showing superior performance and generalization ability. The paper concludes that H2IL-MBOM enhances decision-making in complex, dynamic environments by accurately capturing opponents' mental states and their evolving strategies.

### Strengths
- The proposed method demonstrates superior performance when compared to state-of-the-art model-free reinforcement learning and opponent modeling methods. It effectively captures the changing behavior patterns of opponents and exhibits strong generalization capabilities in multi-agent close-range air-combat environments with missiles.
- The H2IL-MBOM, coupled with the MSOAR-PPO algorithm, enables dynamic and interactive prediction of multiple intentions and latent strategies. This allows for real-time adaptation to changes in opponents' intentions and strategies, addressing the non-stationarity issue in multi-agent interactions and enhancing decision-making processes.

### Weaknesses
- Lack of novelty. Modeling others and world dynamics for multi-agent reinforcement learning has been widely explored in previous works[1,2,3]. It is necessary to justify the novelty of the proposed hierarchical framework. Although previous works do not apply to the multi-UAV game, I can not see any additional challenge introduced in the game. 
- Most of the baselines are out-of-date, e.g., MADDPG, MAPPO. It is necessary to compare stronger baselines with the SOTA opponent modeling methods that were introduced for general multi-agent games.
- Generalization. The generalization of the learned model to different numbers of agents/opponents and unseen behaviors during test time is not evaluated. The paper primarily focuses on air-combat scenarios. It is not clear how well the proposed methods would generalize to other types of multi-agent environments with different dynamics and objectives. 
- The hierarchical model proposed is complex, which could limit its scalability and applicability. 


References:
[1] Proactive Multi-Camera Collaboration for 3D Human Pose Estimation, ICLR 2023

[2] Fast Peer Adaptation with Context-aware Exploration, ICML 2024

[3] Greedy when sure and conservative when uncertain about the opponents, ICML 2022

### Questions
- How to extend the framework to handle the visual observation for real-world applications?
- Can you show some videos about the simulation and the learned policy?
- Is the model robust to the different scales of the population of agents, e.g. 10 vs. 10?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents a multi-agent model-based reinforcement learning framework for close-range air combat. Compared to prior work, this paper employs a more realistic observation model in which agents cannot observe private state information from other agents. The main contribution of the paper is a data-driven two-level latent variable model. The high-level model learns a latent space for "intentions," and the low-level model learns another for "strategies." The forward world model consists of models for intentions and strategies and how they affect future states/observations. These models are parameterized by Transformers, similar to prior work on TSSM.

The authors employ a self-play setting to evaluate the RL agent performance in a simulated air combat environment. The main results demonstrate that the proposed method achieves higher rewards than relevant model-free and model-based RL baselines. The results suggest that the novel hierarchical modeling approach helps more accurately predict the interleaving dynamics in a multi-agent environment.

### Strengths
- The main algorithm is a sophisticated approach to solving a challenging, practical multi-agent control problem.
- The authors include in-depth derivation and detailed algorithms in the appendix.
- The algorithm outperforms various model-free and model-based RL baselines.

### Weaknesses
The presentation of the paper needs improvement:
- The paper's key contribution is modeling the dynamics of "intentions" and "strategies," but I don't see a clear definition. Are they just two generic latent spaces the authors have assigned names to? What is a mental state?
- The paper is swamped with rather random-looking abbreviations. These don't flow well in sentences, making the method section difficult to follow.
- The experiment settings are not communicated clearly (see questions).
- The result figures are noisy. The authors should consider running multiple seeds to visualize the average trend. Also, the text in the figures is too small, and the captions are not informative at times.
- Equation 1 is outside the paper margin.

Overall, I often find it hard to distinguish if a statement is a motivation, a hypothesis, or some standard definition from prior work (e.g., section 3.1).

### Questions
- How are the plotted rewards computed? My understanding is that the authors use self-play during training. Then, how is the policy performance evaluated? Are the baseline methods compared on the same opponent team?
- Does each agent make decisions independently? I don't think the MDP formulation is appropriate here because not everything is observable. Also, the MDP seems to describe the entire simulation state but not from individual agents' point of view.

### Soundness
2

### Presentation
1

### Contribution
2
