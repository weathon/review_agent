# Adversarial Reinforcement Learning Framework for ESP Cheater Simulation

- Decision: Reject
- Scores: 8, 4, 4, 2

## Abstract
Extra-Sensory Perception (ESP) cheats, which reveal hidden in-game information such as enemy locations, are difficult to detect because their effects are not directly observable in player behavior. The lack of observable evidence makes it difficult to collect reliably labeled data, which is essential for training effective anti-cheat systems. Furthermore, cheaters often adapt their behavior by limiting or disguising their cheat usage, which further complicates detection and detector development. To address these challenges, we propose a simulation framework for controlled modeling of ESP cheaters, non-cheaters, and trajectory-based detectors. We model cheaters and non-cheaters as reinforcement learning agents with different levels of observability, while detectors classify their behavioral trajectories. Next, we formulate the interaction between the cheater and the detector as an adversarial game, allowing both players to co-adapt over time. To reflect realistic cheater strategies, we introduce a structured cheater model that dynamically switches between cheating and non-cheating behaviors based on detection risk. Experiments demonstrate that our framework successfully simulates adaptive cheater behaviors that strategically balance reward optimization and detection evasion. This work provides a controllable and extensible platform for studying adaptive cheating behaviors and developing effective cheat detectors.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents an approach/framework for the study of adaptive cheaters and the development of efficient cheat detectors in games. A simulation approach is adopted, with cheaters and non-cheaters modelled as RL agents and the cheat detector as a classifier, with the interactions modelled as an adversarial game. Experiments with two simple environments are conducted to demonstrate the framework and the results analysed.

### Strengths
This is a well written paper. The methodology is clearly laid out and well presented.

The experiments are comprehensive and thorough, and the analysis is considered and sound. The conclusion that the detector developed with adversarial training is a more robust classifier is a solid result, as well as the observation that the adversarially trained cheater can evade detection while still enjoying improved rewards over a non-cheater.

Though not a core area for me, this seems to be a solid contribution, with some clearly elucidated directions for future research too.

### Weaknesses
No weaknesses to note.

Minor: not sure AUROC was defined.

### Questions
Trivial point - for ease of reference, the baseline numbers from Table 1 might have been good to add as bars in Figure 4, so as to be able to get a better sense of the behaviour as lambda varies (and to more clearly see the excess AP performance for small lambda)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an adversarial framework to model cheaters and anti-cheater detection in computer games. It fuses standard RL with a minmax objective to train both models at the same time. Experimental analysis is done on two toy environments and several configurations are studied, including sensible baselines based on reward and episode length.

### Strengths
This paper proposes a relatively realistic model of cheating in games, which supports the cheater's dual objectives of maximizing reward while avoiding detection, and the mutual co-adaptation of cheater and detector. While it is not ground-breaking (being based on fairly well-established adversarial frameworks), it seems well-conceived.

The paper itself is clear, with only a few areas that could use improvement.

The experiments, while done in small settings, do focus on the main areas of interest: adaptive vs non-adaptive behaviour, comparing with proxy signals such as success and episode length, and sensitivity analysis of the coefficient trading off detection vs reward maximization.

### Weaknesses
The main weakness of the paper is in the experimental validation. While the analysis is sound, the conclusions are limited because only two very simple settings were studied. An analysis of more realistic (if small) games would be more conclusive.

The lack of such a comparison is probably explained by another weakness of the methodology: it requires white-box access to RL policies of cheaters and non-cheaters. This means that it cannot be easily adapted to existing games, where game developers can collect datasets of player behaviour, and it is not obvious which are cheaters; but due to the complexity of modern games, it is not a trivial endeavour to train effective RL policies for them.

Another aspect that is understudied is variability in player skill. Both cheater and non-cheater are assumed to play as well as possible (given the limits of training). It is relevant to study the spectrum of non-cheater skill -- especially as high-skill players may be labelled as cheaters, depending on the game. Cheater skill level is also relevant, as a player with poor skills who uses cheating may be easier to detect than a skilled one.

The related work section is extremely short, making it seem like there are more works that were left out. As just one example, Franzmeyer et al. "Illusory Attacks" (ICLR 2024) also seems to study attackers that attempt to conceal their interference, but the frameworks may be different (maximize a player's reward vs minimize a victim's reward).

Fig. 1: The difference between cheater and non-cheater is unclear - would be better to have them interact with a single environment, and having an extra arrow from the environment that represents unfair (ESP) observations. The feedback going to both cheater and detector from itself is also a confusing way to present it.

### Questions
I would like the authors to comment on the relevance of their contribution in the context of the difficulty of training good RL agents in realistic settings, and how it affects extensions to commercial games.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper constructs a simulation methodology that allows for reproducing ESP cheater behavior through adversarial reinforcement learning, including an adaptive cheat detector and adversarial training of cheaters.
The simulation combines existing techniques adapted to this problem setting, including environment formulation via POMDP/MDP, GAN/GAIL-style min-max training, reward shaping, PPO optimization, and trajectory classifier detection. Based on experiments, this study argues that detecting such adaptive cheaters requires a co-evolving detector.

### Strengths
- This research targets a realistic and socially significant problem, ESP cheating, that has not been intorduced by conventional adversarial RL or security RL approaches.
- The mathematical formulation of the proposed framework is well-organized.

### Weaknesses
- The framework of ESP cheaters versus detectors appears to be a new application, but it is essentially a form of adaptive RL-based evasion and bears strong similarities to existing RL-based adversarial attacks against detectors/classifiers.
- To the best of the reviewers' knowledge, the novelty in the design lies in:
(1) the detector being based on trajectory-based cheetah detection, and (2) introducing a complementary structure for cheetahs similar to MoE or a gating network. Both represent straightforward applications of existing techniques to this problem setting.

### Questions
The difference in observation where non-cheaters follow a POMDP while cheaters follow an MDP is critically important concerning the nature of the problem. How does this observational difference affect the difficulty of detection or convergence to equilibrium? To what extent does the information gap between partial observation and full observation bring about qualitative changes in the structure of the game? Can theoretical or empirical insights be provided on this point?

### Soundness
3

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
The paper proposes a simulation framework to model the adversarial interaction between an ESP cheater and a cheat detector. The interaction between an adaptive cheater and a detector is then formulated as a minimax game. The framework is tested on custom Gridworld and Blackjack environments.

### Strengths
Novel Architectural Choice: The Structured Cheater Model is a clever and novel contribution. By freezing the base policies and only learning the interpolation function $\omega$, the authors simplify the adversarial learning problem.

Well-Motivated Problem: The paper addresses a practical and difficult problem (detecting adaptive cheaters) where collecting labeled data is notoriously hard. The simulation-based approach is well-justified .

### Weaknesses
Disconnect Between Motivation and Experiments: This is the most significant weakness. The paper is motivated by complex, multiplayer First-Person Shooter (FPS) games where ESP cheats are a major issue. However, the experiments are conducted on simple, single-agent Gridworld and Blackjack environments. These environments fail to capture the strategic, interactive, and high-dimensional nature of the motivating problem. The authors acknowledge this as future work, but the gap is too large for the claims to be considered general.

Limited Novelty of the Base Framework: While the structured model is novel, the underlying adversarial RL framework (a minimax game solved with GDA and a non-saturating generator loss ) is a standard formulation, similar in principle to GANs and Generative Adversarial Imitation Learning (GAIL). The paper's novelty rests almost entirely on the structured architecture rather than a new learning paradigm

### Questions
Please see the weakness

### Soundness
2

### Presentation
3

### Contribution
2
