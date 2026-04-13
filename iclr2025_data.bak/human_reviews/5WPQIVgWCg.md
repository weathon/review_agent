## Human Reviewer 1

### Summary
This paper addresses the problem of satisficing exploration, inspired by the concept of satisficing in decision-making. The authors propose an algorithmic framework called SELECT, which leverages a learning oracle with sub-linear regret guarantees to iteratively identify and test potential satisficing arms. SELECT achieves constant regret in the realizable cases, and it maintains the original regret bound in the non-realizable cases. Finally, numerical experiments are conducted to demonstrate the algorithm's performance across various bandit settings.

### Strengths
1. It is commendable that the proposed algorithm can be applied to any learning oracle with sub-linear regret guarantees, making it adaptable to different bandit models.
2. The three-step algorithm design is well-constructed and clearly explained. Step 3 (LCB Tests) is particularly impressive. Roughly speaking, in the realizable cases, a certain round will continue indefinitely, while in the non-realizable cases, the algorithm will proceed round by round.

### Weaknesses
1. While the three-step design in each round is great, each round of SELECT runs independently, resembling the doubling trick, which is often criticized. In the non-realizable case, this may lead to suboptimal theoretical and practical performance.
2. The result that constant regret can be achieved in the realizable case is not surprising, given prior research like Garivier et al. (2019). Additionally, the study on the lower bound feels insufficient. For instance, in the case of finite-armed bandits, the lower bounds are limited to two-armed bandits. I encourage the authors to explore the lower bounds further.
3. A minor issue to note is the terminology "satisficing exploration." In the bandit literature, exploration refers to selecting arms with uncertain rewards to gather information, as opposed to exploitation, where arms are selected to maximize immediate rewards based on current knowledge. In this problem, there is indeed a tradeoff between exploration and exploitation. I believe the thresholding bandits problem (in the pure exploration setting) is a better model of satisficing exploration. The authors might consider clarifying this or adopting different terminology.

### Questions
1. For Condition 1, what happens when $\alpha < 1/2$?
2. Regarding the numerical results for finite-armed bandits, which algorithm is used as the learning oracle? Could you explain why SELECT outperforms Uniform UCB in Figure 4b?
3. Can $\gamma_i$ be multiplied by a constant? If so, are the empirical results sensitive to the choice of constant for both realizable and non-realizable cases?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper introduces SELECT, an algorithmic framework designed for satisficing exploration in bandit optimization. The primary objective of SELECT is to frequently identify arms with mean rewards exceeding a specified threshold, with its performance evaluated through satisficing regret, which measures the cumulative deficit of the chosen arm's mean reward compared to this threshold. SELECT operates by leveraging a learning oracle that provides a sub-linear regret upper bound. It iteratively identifies potential satisficing arms, collects data samples, and monitors the LCB of the arm’s mean reward against the threshold to determine if it qualifies as a satisficing arm. The algorithm guarantees constant satisficing regret in scenarios where a satisficing arm exists (realizable case) and matches the standard regret of the oracle in non-realizable situations. The framework is successfully instantiated across various bandit settings. Numerical experiments validate the efficacy of SELECT, demonstrating its ability to achieve constant regret.

### Strengths
- The paper successfully integrates standard bandit optimization algorithms into a new framework, achieving better results and providing a more direct application of bandit algorithms to the satisficing exploration problem.
- The proposed SELECT employs a unique approach by utilizing a learning oracle for bandit optimization, enabling it to sample candidate arms and monitor performance efficiently.
- The paper establishes that SELECT achieves a constant satisficing regret in realizable cases, independent of the satisficing gap, and inversely related to the exceeding gap. This feature allows it to maintain performance even when the satisficing gap varies.

### Weaknesses
- The main weakness is that the algorithm imposes stringent conditions (Condition 1) on the oracle algorithm, requiring sublinear regret for all time steps t. Most algorithms, including all oracles referenced in Section 5, only achieve sublinear regret when t is sufficiently large. If alpha approaches 1 when t is small, the theoretical regret bound could become excessively large.
- The algorithm involves hyperparameters that rely on the oracle algorithms. In scenarios where oracles are unavailable or when the sublinear oracles have unclear parameters, extending theoretical conclusions becomes challenging.
- The first step of each phase necessitates a bandit algorithm, which is crucial. However, the paper lacks a general discussion on how to select the appropriate oracle algorithm.
- While Remark 2 highlights the novelty of each step, an ablation study demonstrating the impact of each component would strengthen the paper. Currently, the experimental results do not robustly support the conclusions, as SELECT only outperforms all baselines in 3 out of 6 settings.

### Questions
- There appears to be an inconsistency in the paper regarding the baseline references to "Hajiabolhassan \& Ortner (2023)" in line 427 and "Michel et al. (2023)". Clarification is needed.
- The time horizon T may not be large enough for UCB-based algorithms. For instance, in Figure 3(b), as T further increases,  SELECT appears to perform worse than the other algorithms.
- The experimental results further indicate that Condition 1 is overly strict, as the oracles struggle to satisfy it. For example, in Figures 4(a) and 4(b), the regret appears to converge to T for Uniform UCB.

### Soundness
2

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
The paper tackles the problem of satisficing exploration in multi-armed bandits. Satisficing problem here represents finding an option which is above a preset threshold. The paper proposes a novel method SELECT which utilizes any existing bandit method with sub-linear regret guarantees and utilizes the same sample path trajectories to further provide a constant satisficing regret framework. 

The method implementation is split into three parts: 
1. Shadowing the sub-linear regret method's trajectory for a set number of rounds
2. Forced sampling of selected arm 
3. Comparing the lower confidence bound of the selected arm with the threshold value.

The paper provides regret guarantees based on the difference between the highest mean among all the options and the threshold value (denoted in the paper as $\Delta_S*$). The paper also provides a matching lower bound (up-to-logarithmic factors) to validate the performance of SELECT. 

The paper further goes on to provide examples of how SELECT can be used with different bandit frameworks, making the method quite applicable to a large set of setups. This is supplemented with experiments for the same, further strengthening the case of SELECT.

### Strengths
The following would contribute to the strengths of the paper:
- **Clear Writing**: The paper is well written, precise, and to-the-point. 

- **Justified Problem Setup**: The paper clearly explains the justification of the problem setup, literature surround it and solution of the problem with theoretical and experimental backing. 

- **Innovative Umbrella Solutions**: The novel proposed method SELECT can be appended to any sub-linear regret method and can provide constant satisfying performance guarantee to the respective application. This makes the algorithm quite applicable to a lot of varied problem setup. 

- **Theoretical performance guarantees**: The paper provides theoretical proof on both the regret upper-bound and shows the tightness to the fundamental lower-bound on the best performance possible on the satisficing problem. 

- **Example distinct setups**: The paper provides example problem setup in finite-armed bandits, concave bandits and Lipschitz bandits.  

- **Experiments**: The paper provides a synthetic implementation for all the example setups and showcases the promise of SELECT method.

### Weaknesses
There are very few obvious loopholes to the paper. Overall the paper is a complete work. A few paragraphs on the potential future works and possible extensions would be a good addition.

### Questions
Nothing to add here

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
4