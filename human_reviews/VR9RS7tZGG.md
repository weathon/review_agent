# Diffusion-Based Offline RL for Improved Decision-Making in Augmented Single ARC Task

- Decision: Reject
- Scores: 5, 3, 5, 3

## Abstract
Effective long-term strategies enable AI systems to navigate complex environments by making sequential decisions over extended horizons. Similarly, reinforcement learning (RL) agents optimize decisions across sequences to maximize rewards, even without immediate feedback. To verify that Latent Diffusion-Constrained Q-learning (LDCQ), a prominent diffusion-based offline RL method, demonstrates strong reasoning abilities in multi-step decision-making, we aimed to evaluate its performance on the Abstraction and Reasoning Corpus (ARC). However, applying offline RL methodologies to enhance strategic reasoning in AI for solving tasks in ARC is challenging due to the lack of sufficient experience data in the ARC training set. To address this limitation, we introduce an augmented offline RL dataset for ARC, called Synthesized Offline Learning Data for Abstraction and Reasoning (SOLAR), along with the SOLAR-Generator, which generates diverse trajectory data based on predefined rules. SOLAR enables the application of offline RL methods by offering sufficient experience data. We synthesized SOLAR for a simple task and used it to train an agent with the LDCQ method. Our experiments demonstrate the effectiveness of the offline RL approach on a simple ARC task, showing the agent's ability to make multi-step sequential decisions and correctly identify answer states. These results highlight the potential of the offline RL approach to enhance AI's strategic reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents a novel approach to addressing the challenges of strategic reasoning in abstract problem-solving using Latent Diffusion-Constrained Q-learning (LDCQ). The authors apply LDCQ, a prominent diffusion-based offline RL technique, to tasks from the Abstraction and Reasoning Corpus (ARC). Due to the limitations of the ARC dataset, they introduce Synthesized Offline Learning Data for Abstraction and Reasoning (SOLAR), which augments the original dataset with diverse trajectories for training offline RL agents. The paper shows that SOLAR, combined with LDCQ, enhances the agent’s ability to make multi-step sequential decisions and solve complex tasks that require abstract reasoning, highlighting the potential of diffusion-based offline RL to improve AI decision-making in environments requiring strategic and multi-step reasoning.

### Strengths
**Originality**: The application of diffusion-based offline RL to reasoning tasks, specifically through the introduction of LDCQ and the SOLAR dataset, is innovative and could open new avenues in RL for reasoning tasks.

**Quality**: The experiments are well-designed and support the paper’s claims. The comparisons between different approaches, including VAE and diffusion prior, are valuable.

**Clarity**: The structure and flow of the paper are generally clear, with useful visual aids and explanations of the ARC tasks and offline RL methodology.

**Significance**: If extended to more complex reasoning tasks, the proposed method could significantly improve how RL systems approach long-horizon decision-making problems, contributing to the broader field of AI reasoning.

### Weaknesses
**Data Augmentation Concerns**: While the SOLAR-Generator effectively augments training data, its reliance on hard-coded rules for action selection (Appendix Section B; specifically, Ln. 678-715)  might limit its applicability when tackling tasks that demand more dynamic or adaptive strategies. It'd be helpful for the authors to add more discussion about the limitations of the SOLAR-Generator's augmentation methods, and how it impacts scalability, transferability to other environments.

**Limited Comparisons**: The paper lacks sufficient comparative analysis with other offline RL approaches or state-of-the-art methods for ARC, making it difficult to gauge the improvement over prior methods/approaches quantitatively. Concretely, including and juxtaposing against any of the approaches from the ARCLE paper - for e.g.,  any sequential policy architecture  (like RNN policy of Vinyals et al. 2019) would greatly bolster the experimental results.

**Dataset Relevance**: SOLAR, while beneficial, is specifically tailored to ARC tasks (and the ARCLE environment), which may not generalize to other reasoning tasks or RL benchmarks (e.g. Meta-World, RLBench, CALVIN, NLE etc.). Extending the applicability of SOLAR to other domains would increase the paper’s impact.

### Questions
**Questions**

1. Data Augmentation Limitations: Could the authors elaborate more on how the deterministic nature of the SOLAR-Generator might affect generalization in more complex or varied task environments?

2. Could the authors provide more quantitative comparisons between LDCQ and other reinforcement learning techniques, especially in terms of training efficiency and performance?

3. Can the proposed LDCQ approach be generalized to other abstract reasoning benchmarks outside of ARC / ARCLE? What are the main challenges in doing so?

4. The LDCQ method relies on generating latent samples through diffusion models. Could the authors elaborate on the computational cost of this approach and how it scales with more complex tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper addresses the shortage of experience data necessary for training offline reinforcement learning (RL) methods on Abstraction and Reasoning Corpus (ARC) tasks. The motivation behind this work is to leverage ARC tasks to assess the reasoning capabilities of offline RL agents that incorporate diffusion models. In particular, the authors focus on evaluating the performance of Latent Diffusion-Constrained Q-learning (LDCQ). They introduced the Synthesized Offline Learning Data for Abstraction and Reasoning (SOLAR) dataset, generated by the SOLAR-Generator, which provides diverse trajectory data to ensure sufficient training experience for the learning agent. Experimental results show that the LDCQ method can correctly make multi-step sequential decisions to complete ARC tasks, thereby demonstrating reasoning capabilities.

### Strengths
- The paper is clear, well-organized, and visually informative.
- ARC tasks are challenging and suitable for demonstrating the strengths of RL methods.
- The LDCQ agent’s ability to omit unnecessary actions from the original data is notable (lines 424-430).

### Weaknesses
- The focus solely on evaluating diffusion-based offline RL, particularly LDCQ, seems narrow. Comparing traditional offline RL methods like CQL [1], IQL [2], or EDAC [3], as well as generative-guided agents such as Decision Transformers (DT) [4], would add depth and increase the significance of the work.
- The experiment is limited to a simple ARC setting. More complex ARC tasks should be investigated to create a major contribution. 
- The paper’s main contribution is unclear. It’s uncertain whether the primary aim is to introduce an augmented dataset or to assess diffusion-model-based offline RL agents.
   + If introducing a dataset is the goal, the focus could shift to complex or multi-task settings, and testing with diverse RL methods to show the usefulness of the dataset. Additionally, a more thorough introduction to ARC tasks would be beneficial.
   + If evaluating diffusion-based RL is the goal, it would be beneficial to include additional diffusion-based algorithms, like Synther [5], and test on more complex ARC tasks or in multi-task scenarios.


[1] Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems, 33, 1179-1191.

[2] Kostrikov, I., Nair, A., & Levine, S. (2021). Offline reinforcement learning with implicit q-learning. arXiv preprint arXiv:2110.06169.

[3] An, G., Moon, S., Kim, J. H., & Song, H. O. (2021). Uncertainty-based offline reinforcement learning with diversified q-ensemble. Advances in neural information processing systems, 34, 7436-7447.

[4] Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., ... & Mordatch, I. (2021). Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34, 15084-15097.

[5] Lu, C., Ball, P., Teh, Y. W., & Parker-Holder, J. (2024). Synthetic experience replay. Advances in Neural Information Processing Systems, 36.

### Questions
Why didn’t the authors use the dataset to evaluate other offline RL approaches? It’s unclear why LDCQ would be uniquely suited to the generated data.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper examines the ARC reasoning tasks. In this setting, agents are provided with demonstration examples, and must correctly manipulate a new grid using the inferred strategy. This paper takes an RL view to this challenge, using the ARLET environment. The approach is to synthesize additional data and trajectories, then use these trajectories to train a policy. The SOLAR-Generator approach is to synthesize data following ground-truth patterns. These trajectories are then filtered to ensure they are valid trajectories, then labelled with standard RL information (reward, termination, etc). This uses a pre-defined Grid Maker helper.

### Strengths
This paper provides a clean, well-described approach to solving a challenging reasoning setting. All design choices are explained clearly, and figures provide a visual intuition as to what is going on. The proposed SOLAR-Generator is well-reasoned and suitably achieves the desired effect. While many decisions are domain-specific, they remain simple and are well justified.

### Weaknesses
A persistent weakness in this paper is the conflation of implementation details with the main contribution. For example, the paper spends quite a large portion describing the diffusion-based RL policy, when the key takeaway from the paper is the synthetic data which enables solving ARC tasks. The two non-LDCQ baselines shown in 5.2 are lacking any Q signal at all. A simple and more informative baseline would be to train a simple non-diffusion policy on the discrete space to maximize Q.

The section 3.2 on describing the SOLAR generation method could benefit from using less abstraction, and show some concrete examples of data generated.

As the paper only focuses on this single setting, its potential future contribution is limited.

Additionally, there are no clear comparisons to past methods, making it hard to judge the effect of the proposed strategies.

### Questions
- Why is it necessary to train a latent representation of the action space, given that the action space is small and discrete, and as mentioned in the paper, a typical horizon for solving the task is only 5 steps?
- It would strengthen the paper to focus specifically on the SOLAR generation, and provide ablations on how performance scales with data, etc. The choice of policy network and sampling strategy (LDCQ) appears arbitrary, and distracts from the main contribution.
- A slight confusion when reading may clear up for future readers -- at which point is the agent conditioned on the demonstration examples when evaluated on a new task?
- Figure 3 can benefit from describing how the Q-network is trained. The core information is that it is trained via TD learning, which is missing in the figure. Figure three (b) is also counterintuitive, why does the model output a noised latent?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper develops an augmented offline RL dataset called SOLAR  (Synthesized Offline Learning Data for Abstraction and Reasoning), to mitigate insufficient data in Abstraction and Reasoning Corpus (ARC).

### Strengths
1. The SOLAR-Generator enables customization of augmented data by generating different trajectory data based on predefined rules.
2. The augmented dataset helps demonstrate the effectiveness of offline RL algorithms, particularly in enhancing reasoning abilities.

### Weaknesses
1. Limited novelty: This paper primarily focuses on evaluating existing methods rather than developing new algorithms or applications. Its main contribution is the generation of a synthesized dataset to augment limited data, which is commonly used in previous works, such as [A1] in reinforcement learning and [A2] in computer vision. The authors should clarify the unique contributions and specific challenges involved in designing the SOLAR-Generator for ARC.
2. Quality of Generated Data: The random generation of synthetic samples raises concerns about ensuring optimal or efficient trajectories. It's crucial to determine whether training data containing episodes that overshoot the target state could negatively impact the agent's performance.
3. Unclear level of Data Diversity: The diversity of the generated data is unclear. Experimental procedures suggest that data augmentation is based on the same input and target, with variations introduced at random steps. This approach may result in data that lacks sufficient variability, potentially limiting the model's ability to generalize.
4. Quantifying Improvement and Insufficient Experiments: The paper does not provide clear metrics or comparisons to baseline methods within the ARC framework, making it difficult to evaluate the actual benefits of using the augmented data. Including quantitative evaluations with various agents would enhance understanding of the effectiveness of the proposed approach.

[A1] Laskin, Misha, et al. "Reinforcement learning with augmented data." Advances in neural information processing systems 33 (2020): 19884-19895.

[A2] Yang, Suorong, et al. "Image data augmentation for deep learning: A survey." arXiv preprint arXiv:2204.08610 (2022).

### Questions
1. How does the SOLAR-Generator ensure the quality of the generated trajectories?
2. What specific performance improvements were observed when using the augmented dataset compared to standard training data?
3. To what extent does the generated data vary to provide meaningful diversity, and how does this diversity impact the model's generalization capabilities?

### Soundness
2

### Presentation
2

### Contribution
1
