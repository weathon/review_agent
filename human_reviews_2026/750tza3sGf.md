# BoreaRL: A Multi-Objective Reinforcement Learning Environment for Climate-Adaptive Boreal Forest Management

- Decision: Accept (Poster)
- Scores: 4, 4, 2, 4

## Abstract
Boreal forests store 30-40\% of terrestrial carbon, much in climate-vulnerable permafrost soils, making their management critical for climate mitigation. However, optimizing forest management for both carbon sequestration and permafrost preservation presents complex trade-offs that current tools cannot adequately address. We introduce BoreaRL, the first multi-objective reinforcement learning environment for climate-adaptive boreal forest management, featuring a physically-grounded simulator of coupled energy, carbon, and water fluxes. BoreaRL supports two training paradigms: site-specific mode for controlled studies and generalist mode for learning robust policies under environmental stochasticity. Through evaluation of multi-objective RL algorithms, we reveal a fundamental asymmetry in learning difficulty: carbon objectives are significantly easier to optimize than thaw (permafrost preservation) objectives, with thaw-focused policies showing minimal learning progress across both paradigms. In generalist settings, standard gradient-descent based preference-conditioned approaches fail, while a naive site selection approach achieves superior performance by strategically selecting training episodes. Analysis of learned strategies reveals distinct management philosophies, where carbon-focused policies favor aggressive high-density coniferous stands, while effective multi-objective policies balance species composition and density to protect permafrost while maintaining carbon gains. Our results demonstrate that robust climate-adaptive forest management remains challenging for current MORL methods, establishing BoreaRL as a valuable benchmark for developing more effective approaches. We open-source BoreaRL to accelerate research in multi-objective RL for climate applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a novel RL environment modelling the management of boreal forests. The environment tasks RL algorithms with two objectives: maximizing carbon sequestration and  minimizing permafrost thaw. The environment is built on a complex thermodynamic simulator that models the energy balance of different parts of the forest stand. The simulator also models the carbon cycle, water balance and various weather patterns. Finally, the environment allows for training only on specific site or from a site/weather sampled from some distribution.

To test their environment, the authors test both expected utility policy gradient (with different preference weights) and PPO. Experimentally, they observe that it is easier to optimize for carbon objectives than thaw objectives. To address the difficulty, they use curriculum learning and show that it outperforms other methods and learns to improve performance on thawing objectives.

### Strengths
- The paper is very well-written and motivated, it does a good job of presenting the subject as a valuable application area for ML.
- The work is novel and appears to address a gap in the literature.
- The environment involves a significant amount of work and attention to detail to model a large amount of aspects of the real world in the physical simulation.
- The authors test various RL methods on the environment and carefully analyze their performance.

### Weaknesses
My main concerns are as follows. I am willing to increase my score if these are at least somewhat addressed. Essentially, I need to be convinced that (a) the simulator is a reasonable model of real world dynamics (b) the forest management problem requires complex decision making that benefits from reinforcement learning methods.

- Looking at the appendix, the underlying simulator seems quite complex. I do not have the necessary expertise to judge how realistic it is. Could you point me to any evidence/papers to show this model is reasonable and grounded in real-world dynamics? The paper in general could benefit from additional justification of some of these decisions.
- You mention you believe the benchmark is far from saturated. To provide evidence of this, it would be helpful to provide additional baselines based on heuristics or simpler methods (e.g. what would be a reasonable baseline policy that a human would implement)? If the RL methods noticeably outperform this it would provide further evidence of the value of RL here.
	- The +100 density/0 density baselines are a step in the right direction but I'm guessing there are stronger baselines?
- RL performance plots would benefit from error bars if these are not too expensive to compute.

### Questions
- How close is the model of possible actions to the actual decisions humans need to make? It feels like there is a lack of granularity (though I am unfamiliar with the intricacies of forest management).
	- Similarly, the long time horizon between actions is interesting. I'm guessing most actions are taken over long time horizons for forest managements?
- You mention the use of numba and jit to improve performance. Are there any plans to create a Jax version of the environment (given the recent trend towards GPU simulation of RL environments)?
	- How expensive is the environment to run/training the PPO methods?

### Soundness
2

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
The paper proposes BoreaRL, a physics-based multi-objective RL environment that simulates how climate-adaptive boreal forest management decision impacts on the forest. It is a physically-grounded simulator. Its observation space include components such as ecological state, site climate parameters, historical information and episode-level site parameter context. Its action space include two discrete action: stand density change and species mix change. The reward function is a weighted sum of normalized net carbon change and thaw reward, with the weight set by the preference weight parameter. The paper experiments and compares with the following RL algorithms: Fixed-$\lambda$ EUPG, Variable-$\lambda$ EUPG, PPO Gated and Curriculum PPO.
Main contribution of the paper includes (1) a physics-grounded simulator that captures complex biogeophysical trade-offs (2) a modular MORL framework supporting site-specific and generalist training paradigms (3) benchmarking shows Curriculum PPO outperforms other algorithms (4) novel ecological insights for appropriate forest management.

### Strengths
The major strength is a physically-grounded, process-based and open-source simulator with a wide range of configurations. The authors show a high level of expertise in the boreal ecosystem covering three distinct models: energy, water, and carbon fluxes on a n-minute time step. It captures real biogeographical trade-offs, such as albedo, rain and snow patterns and insect disturbance. It is a valuable contribution to the domain of real-world environmental policy and management research.

### Weaknesses
Section 3.4: The paragraph ends with "The objective follows standard PPO:" seems unfinished.

Section B.2.1 The canopy energy balance equation should not be equal to 0.

**Analysis of Learning Objectives**: Despite applying multiple learning algorithms, the RL objective optimize for the same cumulative reward, so simply comparing “actions” (stand density change and conifer fraction) across algorithms without deeper causal insight doesn’t illuminate how or why they differ.

**Circular reasoning on fundamental asymmetry in learning difficulty:**: The author design a forest simulator which calculates annual carbon objective and thaw objective. The thaw objective is more complex than the carbon objective, and therefore the PPO algorithm struggles to optimize for the thaw objective. Instead of acknowledging the root cause of the difference in the learning difficulty is the reward shaping, the author claim this reveals something fundamental about the ecosystem management problem. But the difficulty is largely self-imposed through reward engineering choices.

**Novelty concerns**: Creating a simulator and running off-the-shelf RL agents—is relatively commonplace in applied RL research. The main contribution is domain-specific realism, not algorithmic innovation. In addition, there exists papers in the environmental science space that does something very similar.
- Overweg, H., Berghuijs, H.N.C., Athanasiadis, I.N. (2021). CropGym: a Reinforcement Learning Environment for Crop Management.
- Lapeyrolerie, M., Chapman, M.S., Norman, K.E.A., Boettiger, C. (2021). Deep Reinforcement Learning for Conservation Decisions.

### Questions
1. **Claim about superiority of Curriculum PPO**: The claim "In generalist settings, standard preference-conditioned approaches fail entirely, while a naive curriculum learning approach achieves superior performance by strategically selecting training episodes." is questionable. My confusion is two-folded.
     - To begin with, curriculum learning usually refers to gradually increasing task difficulty, shaping the distribution of episodes, selecting “easier” tasks first, which is not what the paper does. Therefore calling it "curriculum learning" is confusing. The implementation includes an episode selector network to decide whether to include a site-episode observation into training data, but claims it is for inference only. I am confused about at which step is the episode selector network trained.
    - The preference weight input in the observation space of Curriculum PPO. Therefore the Curriculum PPO is also preference-conditioned. Standard preference-conditioned approaches here just refers to gradient descent method. Therefore the claim boils down to "PPO with episode selection works better than gradient descent".

2. **Multi-objective balancing**: The idea of balancing between carbon and thaw objectives encourages studying Pareto-optimal behavior and decision-making under competing goals, which is both scientifically valuable and underexplored in RL benchmarks.

3. **Validity concerns**: The path from "trained policy" to "deployable forest management" is unclear.

4. Overall, the paper would be stronger if it either:
    - Went deeper on methodology: Why does curriculum help? Can we design MORL algorithms specifically for this type of asymmetric objective difficulty?
   - Went deeper on science: Validate against real forest data, partner with foresters, demonstrate actual deployment potential

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considers the application of multi-objective reinforcement learning (MORL) methods to the problem of boreal forest management. The main contributions are (1) formulating boreal forest management as two variants of a MORL problem based on physical simulations; (2) implementing these problems as a simulator and environment compatible with a common MORL benchmarking library; (3) experimentally comparing the performance of multiple different MORL approaches on these problems and discussing the results.

### Strengths
1. Interesting approach to a relevant application problem. This is the first work to propose the use of MORL for forest management.

2. Judging from a brief scan of the supplementary material, the underlying physical modelling appears thorough.

3. Implementation as a modular extension of an existing benchmarking library.

### Weaknesses
1. Not convinced by the focus of the paper. The authors mention that this problem has never been formulated as an MORL problem before, and not in such detail as a single-objective problem. As such, the modelling and formulation of the problem itself seems like a relevant contribution, yet little discussion is devoted to this in the main paper. More consideration should be given to the design of rewards, particularly as the validation experiments show that the thaw reward is difficult to optimise for even when used as the only objective. Different rewards could be explored and compared experimentally.

2. Following from the first comment, the comparison of different algorithms and design of a new solution approach seems premature. (Perhaps this work could even be split in two: one part focussed on the design of the problem and the description of the implementation, and another on improving algorithmic approaches.) Too much of the problem formulation is only discussed in the appendix.

3. Scalarised reward may not be the best metric to evaluate algorithmic performance, given that the reward functions appear to operate on different (effective) intervals. For example, the two solutions for $\lambda = 0.75$ and $\lambda = 1.0$ in Fig. 3(d) are both on the Pareto front, but one would return a higher scalarised reward than another. Multi-objective metrics (e.g. hypervolume, sparsity) would be more suitable for evaluation.

4. Some terms used in the paper appear to be undefined, e.g. λ-monotonicity, or misused, such as the Pareto front. Figures 3(d) and 6 (a-d), for example, are all described as showing Pareto fronts (i.e. the set of non-dominated solutions), but appear to draw no distinction between dominated and non-dominated solutions.

### Questions
Reward Design and Validation: Could you provide more detail on how the reward components (especially the thaw-related one) were designed and validated? Did you consider alternative reward formulations, and if so, what were the results?

Evaluation Metrics:  You mention λ-monotonicity and show scalarised rewards, but have you evaluated more standard multi-objective metrics, such as hypervolume or sparsity, to better capture the quality and diversity of Pareto fronts? If not, could you elaborate on why these were not used?

Definition and Interpretation of Pareto Front: In Figures such as 3(d), you label plots as Pareto fronts but they appear to include both dominated and non-dominated solutions. Can you clarify your definition of the Pareto front and explain how the solutions were identified as non-dominated?

Simulator and Environment Use: Given the level of detail in the simulator, how do you envision users might extend or adapt BoreaRL? For example, can users easily modify the reward components, action spaces, or add new ecological processes within your framework?

Generalizability of the Findings: The paper mainly focuses on preference-conditioned MORL. Have you tested or do you plan to test your benchmark environment with other types of multi-objective learners?

### Soundness
2

### Presentation
3

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
This paper introduces BoreaRL, a multi-objective RL environment for boreal forest management. BoreaRL consists of two components: (1) BoreaRL-Sim, a physics-based simulator running at tunable minute-scale resolution, and (2) BoreaRL-Env, a wrapper around BoreaRL-Sim that implements the mo-gymnasium API and accepts annual agent actions. The “multi-objective” aspect of BoreaRL refers to the dual objectives of carbon sequestration and permafrost preservation. Each episode has 50 time steps, representing 50 years, and the agent’s actions control the tree density as well as ratio between deciduous and coniferous tree species. Experiments across 4 different RL algorithms (fixed-weight EUPG, variable-weight EUPG, PPO Gated, Curriculum PPO) find that Curriculum PPO achieves the best pareto frontier in the multiobjective optimization.

### Strengths
**S1) Clarity of writing**

The paper is well-written and generally easy to follow. The motivation is clear, and the conclusions are straightforward to follow.

**S2) Novel multi-objective RL testbed**

The paper presents a novel, and real-world-motivated multi-objective RL problem. Furthermore, the tested algorithms show a large spread in performance, suggesting that the benchmark is useful for testing RL algorithm performance.

### Weaknesses
**W1) Questionable claim that permafrost preservation objective is hard to optimize**

A central conclusion from the paper is that the permafrost preservation objective is harder to optimize than the carbon objective. However, I am not fully convinced of this claim. To make this claim stronger, the paper should clarify the scale and range for each of the two reward terms $R_{carbon,t}$ and $R_{thaw,t}$. For instance, if $R_{thaw,t}$ has a range [0, 1] whereas $R_{carbon,t}$ as a range [0,100], then naturally we would expect it to be challenging for an RL algorithm to optimize the permafrost objective.

Also, is there any estimate of what the optimal return might be for the permafrost objective?

**W2) Conclusions can be made stronger with more repeated trials**

It is not clear to me which of the results reported are averaged over multiple runs, vs. a single result. For example, Figure 3b shows error bars, but I am confused by what the error bars are supposed to represent. (Is it showing standard deviation or standard error? Over how many random seeds? And do the random seeds only affect the test episodes, or was the RL algorithm trained multiple times with different random seeds?)

Also, the monotonicity plots (e.g., Figure 3d and elsewhere in the appendix) could benefit from runs with multiple seeds.

**W3) Importance of contribution to AI community is a bit unclear**

While I recognize that BoreaRL is “the first multi-objective reinforcement learning environment for climate-adaptive boreal forest management,” I am not entirely sure how to evaluate the importance of this contribution for an AI conference like ICLR. The paper evaluates several RL algorithms, but it is unclear how BoreaRL compares to other multi-objective RL benchmarks. Suppose some researcher discovers an optimal RL algorithm for BoreaRL; what would be the impact of that?

### Questions
Q1) Besides RL, what other tools do forest policy managers traditionally use? Is there any way to compare against such a non-RL baseline?

### Soundness
3

### Presentation
4

### Contribution
2
