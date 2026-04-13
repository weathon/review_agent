## Human Reviewer 1

### Summary
This paper highlights an often ignored discrepancy between the discrete-time RL framework and the real world. The world evolves while agents compute actions, limiting the types of problems agents can solve. Large models with high action inference and learning times conflict with intelligent agents' control over the environment. The paper examines this and explores asynchronous interaction paradigms for large models in high-frequency environments. It formalizes and tests the approach's benefits and limitations. Contributions include formalizing how time discretization induces a new learning problem, deriving regret bounds, proposing methods for staggered asynchronous inference, and conducting experiments on realtime games like Pokémon and Tetris.

### Strengths
1. Regret Decomposition seems to be rigorous in real-time reinforcement learning. This specific choice of time discretization is novel in Definition 1.
2. The staggered asynchronous learning and inference phase addresses  the poor scaling properties, highlighting that while the latter allows actions and learning at every step, actions are delayed and reflecting past states. It also emphasizes the importance of staggering processes to maintain regular intervals and shows implications for performance such as linear speedups.
3. Several Experiments clearly verify their theory.
4. The paper has great organization and is easy to follow.

### Weaknesses
* The experiment is recommended to be based on other base RL strategies rather than just DQN.
* Parallel updates should be considered as a baseline.
* Can staggered asynchronous inference be applied to multi-agent systems?

### Questions
I don't have other questions.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper addresses a key challenge in applying RL models in real-time environments: the need for rapid, continuous responses in settings where even minor delays can accumulate into significant regret. The authors analyze regret bounds and show that minimizing long-term regret is often impractical in conventional sequential RL settings but becomes achievable with sufficient asynchronous computing resources. They introduce new algorithms to stagger inference processes asynchronously, ensuring that actions can still be made consistently, even with larger models. Empirical results on Game Boy games like Pokémon and Tetris support these theoretical insights, showing that the proposed methods support much larger models than previously feasible in real-time environments. The findings suggest that performance relies more on the environment's predictability over the inference period than on rapid action frequency, making it possible to deploy large-scale models in real-time.

### Strengths
The paper presents an advancement in asynchronous interaction and learning within real-time RL, addressing the challenge of deploying large neural network models in environments where timing is crucial.

The authors define an induced delayed semi-MDP which connects the asynchronous nature of real-time environments to the traditional sequential interaction paradigm. This framework allows for a deeper analysis of the sub-optimality that arises from time discretization and computational delays. Furthermore, they propose a novel decomposition of real-time regret into three components: regret from learning, regret from inaction, and regret from delay. This decomposition sheds light on the limitations of sequential interaction and the advantages offered by asynchronous approaches. A notable innovation in the paper is the introduction of algorithms for staggered asynchronous inference, which enable the use of models with longer inference times without compromising the frequency of actions. This key development facilitates the deployment of larger models in time-sensitive settings.

In summary, this paper makes some contributions to the field of real-time RL. The authors present a novel problem formulation, and provide some empirical evidence to support their findings. The presentation and detailed information enhance reproducibility. Overall, the work has important implications for the deployment of large models in real-time settings and may lead to advancements in RL software and hardware design.

### Weaknesses
I have some questions and concerns about this work that I provide in the following.

- In Section 2, it is mentioned that the time between decisions only depends on the MDP. How does it **only** depend on the MDP? Since it denotes the time between decisions, doesn’t it depend on policy instead? The respective sentence doesn’t make sense.
- About taking actions from the behavior policy:
    - How to select the default behavior that is safe and performant at different time steps? What if the default behavior policy leads the MDP_async into OOD states wrt the learned policy?
    - How can we make sure that the behavior policy mimics the real world as it evolves, considering factors such as changing environmental dynamics, varying agent interactions, and the need for continual adaptation to new scenarios?
    - If we change noop with the worst or the best action, how the results and conclusions would change? How important it is to the core idea of the paper?

- In the Empirical Results Section, Questions 1 and 2 are not hyperlinked while Questions 3 and 4 are. I’d suggest making them consistent.
- The experimental section raises two slightly different questions about the performance of asynchronous interaction and learning in a real-time strategy (RTS) game with constant novelty (lines 346 and 424). The first question, stated in the general description, emphasizes evaluating the “speed of progress,” “action throughput,” and “maintaining learning performance.” In contrast, the experiment-specific question appears to narrow the focus to determining whether asynchronous methods can indeed enable faster progress despite longer model and inference times. While both questions relate to asynchronous methods' impact on learning and progression, their phrasing suggests slightly different evaluation criteria, which may confuse readers about the primary aim of the experiments. The first question frames the experiments more broadly, implying a focus on learning stability and throughput, whereas the second centers specifically on asynchronous methods' ability to handle extended inference times in a constantly learning environment.

- About lines 433 and 434: Does the performance improvement over sequential interaction need to be statistically significant? If yes, then how are the results in Figs. 9 and 10 justified when the improvement is only significant for |\theta|=100M? If no, then it would be helpful to include a discussion explaining why statistical significance is not required to interpret these results as meaningful. For instance, if the improvement trends observed with smaller |\theta| values still support the theoretical advantages outlined in Remarks 1 and 2, an explanation should clarify why these trends are practically relevant, even if not statistically significant. Such clarification would strengthen the interpretation of the scaling benefits claimed for asynchronous methods.
- Why are confidence intervals omitted from Figs. 7 and 8?
- The empirical evaluation is only conducted with DQN as the baseline method. To strengthen the evaluation, consider including more advanced baselines like QR-DQN as well. Comparing with such variants would provide a more comprehensive view of the asynchronous framework's performance across different reinforcement learning strategies.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper considers real-time reinforcement learning where the environment continually evolves as the agent performs inference, thereby introducing potential delays on various aspects. The paper proposes a regret decomposition into three terms---learning time, inaction time (i.e. default behaviour while policy is performing inference), and irreducible regret due to stochasticity of the environment with delays. The paper focuses on reducing the two former terms through asynchronous learning and inference. The paper conducts experiments on three environments with varying model sizes. The experiments demonstrate a positive (negative) correlation with model size and delays (performance) with the sequential formulation, and further demonstrate that the proposed asynchronous variant can reduce the performance degradation.

### Strengths
- The paper is mostly well written and the decomposition seems reasonable.
- The proposed method is simple and easily applicable to real-life systems.
- The experiments seem mostly convincing

### Weaknesses
- I would appreciate if the paper considers $\Delta_{delay}$ by artificially injecting noise to the transitions to see the theoretically described bound.
- While the decomposition is nice, I am also interested in seeing whether the regret bound is sublinear. I was wondering if the authors have considered theoretically quantifying these algorithms under this framework.

### Questions
- I believe the following questions are captured by $\Delta_{delay}$, please correct me if I am wrong:
	- How does technique like action chunking play into this? In those situations I believe $\beta$ will be the actions taken by the policy from state $s$ which will be sufficient with more deterministic environments?
	- Furthermore, how meaningful is the asynchronous inference if the inference time is too coarse such that $\pi(s_{t-1}) \neq \pi(s_t)$ assuming no inference delays?
- I am unsure how to interpret Figure 8, particularly why is $N^*_\mathcal{L}$ larger with smaller batch size? I would expect the inverse as it should be less time-consuming to update smaller batch sizes?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper addresses the challenges of employing large models for real-time reinforcement learning, where high inference latency prevents conventional sequential inference-and-learning processes from converging. The authors first formulate asynchronous interactive learning as a delayed semi-Markov Decision Process (semi-MDP) and analyze regret lower bounds by decomposing them into several theoretical components. Their analysis demonstrates that staggering multiple processes can enable effective learning while sequential learning cannot.

The paper then proposes two algorithms that stagger multiple processes for asynchronous RL by maintaining a constant action interval. On the evaluated game benchmarks of Pokemon Game Boy and Tetris, the proposed methods significantly outperform sequential methods, especially when the model size is large. Additionally, this paper analyzes the scaling properties of the number of inference processes needed.

### Strengths
- **Originality:** To the best of my knowledge, the decomposition of regret bounds into several components under real-time settings is interesting and novel. The proposed algorithms are well-motivated by this theoretical analysis.
- **Clarity:** The paper is well-written and easy to follow. The presentation is clear, including comprehensive details on problem formulation, theoretical proofs, and experimental settings with results.

### Weaknesses
- The paper's impact could be broadened by evaluating more experimental settings and environment domains to demonstrate the generality of the theory-motivated algorithms. For instance, including locomotion/control environments with continuous action spaces would provide stronger evidence for applying these methods to real-time robotics applications.
- On question 2: this paper could be more convincing, with more detailed discussions explaining why sequential learning methods fail as network size increases, for instance, including concrete examples demonstrating how large networks prioritize high-frequency actions could provide evidence to support the theoretical analysis empirically in a more grounded way.
- When comparing models of different scales, one potential extension is to consider variations in the number of layers, in addition to the number of channels, would provide a more comprehensive evaluation since both factors significantly affect inference time.
- Algorithm 2 would benefit from experiments/results in the settings with relatively high inference time variance. Though this is a minor point, such experiments would better justify the choice between two algorithms.

### Questions
Please see the above section.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
2