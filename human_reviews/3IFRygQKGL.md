# OptionZero: Planning with Learned Options

- Decision: Accept (Oral)
- Scores: 8, 6, 8, 8

## Abstract
Planning with options -- a sequence of primitive actions -- has been shown effective in reinforcement learning within complex environments. Previous studies have focused on planning with predefined options or learned options through expert demonstration data.
Inspired by MuZero, which learns superhuman heuristics without any human knowledge, we propose a novel approach, named *OptionZero*. OptionZero incorporates an *option network* into MuZero, providing autonomous discovery of options through self-play games. Furthermore, we modify the dynamics network to provide environment transitions when using options, allowing searching deeper under the same simulation constraints. Empirical experiments conducted in 26 Atari games demonstrate that OptionZero outperforms MuZero, achieving a 131.58% improvement in mean human-normalized score. Our behavior analysis shows that OptionZero not only learns options but also acquires strategic skills tailored to different game characteristics. Our findings show promising directions for discovering and using options in planning. Our code is available at https://rlg.iis.sinica.edu.tw/papers/optionzero.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the OptionZero framework, which incorporates options into MCTS and enables the automatic design of options through self-play, thereby avoiding cumbersome manual design. In Atari games, OptionZero achieves significant improvements compared to MuZero, achieving a 131.58% improvement in mean human-normalized score. It has shown promising directions for discovering and using options in planning.

### Strengths
1. This paper is well-written with a clear structure, making the research content easily comprehensible.

2. By introducing the OptionZero framework and leveraging self-play for automatic option design, this study paves the way for new approaches. The novelty is good.

3. The experimental results robustly support the effectiveness of the algorithm.

### Weaknesses
1. I am curious about whether the introduction of options has any impact on the theoretical optimality of MCTS.

2. Is it possible to demonstrate the advancement of the algorithm more directly by solely utilizing the learned network for action selection, without relying on MCTS?

3. What is the expected performance as the value of $l$ increases?

4. What are the differences in the running wall-clock time required for OptionZero compared to MuZero?

### Questions
Please refer to the weakness part.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce OptionZero, an advanced extension of the MuZero algorithm that autonomously identifies temporally extended actions, known as options, without the need for predefined options or expert-provided demonstrations. By incorporating an option network, OptionZero enhances planning efficiency by decreasing both decision-making frequency and the computational load required for complex environments. Evaluated on a suite of Atari games, OptionZero demonstrates notable improvements in human-normalized scores compared to MuZero, accompanied by a detailed analysis of how options are utilized across varying game states and scenarios

### Strengths
- Novel idea for autonomous and adaptable option discovery. The proposed method's ability to autonomously discover and tailor options to diverse game dynamics removes the need for predefined actions, making it highly adaptable across different environments.

- Convincing results for enhanced planning in RL.  By integrating an option network, OptionZero reduces decision frequency, enabling computational efficiency, particularly in visually complex tasks like Atari games.

- Strong Performance Gains: On Atari benchmarks, OptionZero achieves a 131.58% improvement over MuZero, a significant improvement compared to previous SOTA papers.

- Interesting ideas to adjust option lengths, balancing performance and training efficiency, particularly useful in tasks needing variable action sequences.

### Weaknesses
- Inconsistent Option Use Across Games: OptionZero's reliance on options appears to vary widely across Atari games. While longer options bring substantial gains in some games, they contribute less in others. This inconsistency suggests that the model’s option-based planning may struggle to generalize well across diverse, complex environments. The paper should discuss this limitation. 

- Challenges in Complex Action Spaces: In games with intricate action spaces, such as Bank Heist (Atari), OptionZero’s dynamic network encounters difficulty as option lengths increase, particularly with multi-step dependencies. This issue may restrict OptionZero’s application in environments where actions are highly combinatorial, relying instead on settings with more straightforward or predictable actions.

- Reduced Prediction Accuracy for Longer Options: The model’s prediction accuracy tends to decrease as options become longer, affecting planning quality where extended strategies are essential. I would recommend adding an experiment to study this effect and discuss potential limitations of the proposed method. 


- Limited Application Beyond Games: Although the model shows promise in game environments, the paper does not investigate its potential beyond Atari-like settings. I would appreciate seeing results on other domains, maybe robotic such as Gymnasium-Robotics. Under the current evaluation, the method seems limited to game-based scenarios.

### Questions
- How does the dynamics network handle complex action spaces, especially in games with highly varied option paths?"

- What are the specific computational costs of incorporating the option network, both in training and during MCTS simulations? Could the author discuss the associated overhead with the proposed method?

- Can the model be applied to environments beyond games with less predictable state transitions, and how would option discovery be affected? I would suggest adding studies in robotic environments.

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
This paper presents OptionZero, a novel approach that incorporates an option network into MuZero and allows the agent to learn temporary extended actions. The authors conducted empirical experiments and find OptionZero outperforms MuZero in teams of mean human-normalized scores on 26 Atari games.

### Strengths
The paper is well-written. The authors explain the use of options clearly with a toy example, demonstrating how options are used. The empirical results are also strong, achieving high mean normalized scores.

### Weaknesses
It's not clear the actual benefits options bring. In the intro, the paper claims options allow for "searching deeper", but the empirical analysis shows "deeper search is likely but not necessary for improving performance". While it's nice to have the option to do option, could the authors provide a more detailed analysis of options beyond a deeper search? 

The paper could also benefit more from discussions of 
1) the trade-offs between increased complexity and performance gains
2) how much tuning did the authors perform to make OptionZero work; were there failure cases/ideas during the development and how did the authors overcome them?

### Questions
1. Why select these 26 Atari games instead of using the standard 57 Atari games?
1. What is the hardware resource for conducting the experiments?
1. How long does training a single Atari game with OptionZero take?
1. How long does training a single Atari game with MuZero take?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The work proposes a revamped approach to the well-known options framework, which allows agents to take temporally extended actions as well as myopic ones. By combining a network which learns options with MCTS MuZero (which models transition dynamics) the authors propose a method to utilise options alongside single actions in self-play games.

### Strengths
The need for efficiency in decision making in RL is clear, as single-step actions are slow and computationally expensive (even more so in slow simulators). Thus, the problem addressed by OptionZero is clear and its existence is well-motivated. Additionally, since much prior work in options appear to be in manually defined and demonstration-based settings, the generalisability of OptionZero is a strong selling point.

Within fixed computational constraints, the idea of decreasing the frequency of queries to the network is a strong idea for the current state of RL. Also important is the notion of learning subroutines which the options network will identify as useful in different scenarios and not have to re-learn temporal relationships.

The flexibility to play options or primitive actions results in tailored reactions to scenarios, as an agent may need the fine-grained approach taken by traditional RL. The main results in Table 1 indicate the validity of the method, as using options provides a performance benefit more often than not, with longer option limits sometimes outperforming shorter ones.

### Weaknesses
It is unclear why options are outperformed by primitive actions in certain environments. The authors suggest that in environments with high combinatorial complexity, learning of the dynamics model may be difficult and thus options may simply produce more overhead than actual benefit. A more detailed analysis of these environments would be beneficial, for e.g. investigate whether there is a correlation between the stochastic branching factor of the environment and the performance of options.

Additionally, it seems that longer options may improve efficiency but not always increase performance when those options may be overextending in environments where more granular control is required. Have the authors considered implementing dynamic options lengths somehow? This may make the idea more viable, or at least a discussion on the complications of implementing that would be a good addition to the work.

### Questions
Clerical: In section 5.2 the $l_{1}$ option setting is mentioned as a baseline, but Table 1 compared the options to something called $l_{0}$. Do $l_{0}$ and $l_{1}$ refer to the same baseline? If so, using consistent notation will help make the results section more readable.

### Soundness
3

### Presentation
3

### Contribution
4
