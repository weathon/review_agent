# CARTS: Cooperative Reinforcement Learning for Traffic Signal Control and Carbon Emission Reduction

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 2, 8

## Abstract
Existing traffic signal control systems often rely on overly simplistic, rule-based approaches. Even reinforcement learning (RL)-based methods tend to be suboptimal and unstable due to the inherently local nature of control agents. To address the potential conflicts among these agents, we propose a cooperative architecture named CARTS (CooperAtive Reinforcement learning for Traffic Signal control). CARTS introduces multiple reward terms, weighted with an age-decaying mechanism, to optimize traffic signal control at a global scale. Our framework features two types of agents: local agents that focus on optimizing traffic flow at individual intersections, and a global agent that coordinates across intersections to enhance overall throughput. Importantly, the system is designed to reduce both vehicle waiting time and carbon emissions. We evaluated CARTS using real-world traffic data obtained from traffic cameras in an Asian country. Despite incorporating a global agent during training, CARTS remains decentralized at inference time, requiring no centralized coordination during deployment. Experimental results show that CARTS consistently outperforms state-of-the-art methods across all evaluated performance metrics. Moreover, CARTS effectively links carbon emission reduction with global agent coordination, providing an interpretable and practical approach to sustainable traffic signal control.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents CARTS, a cooperative multi-agent RL framework for traffic-signal control that introduces a global agent to coordinate local DDPG controllers during training—optimizing the site-wide total waiting time—while keeping inference fully decentralized (the global agent is removed at test time). CARTS does not just pick the next phase; it jointly learns the phase and its variable duration, moving beyond fixed action durations used in prior work. The method also integrates the HBEFA emission model available in SUMO to explicitly monitor and reduce CO/CO₂, leveraging the observation that waiting time is a primary driver of emissions in their setup. Experiments on a real-world morning-rush dataset (five consecutive intersections) and the SUMO RL benchmark show consistent gains over strong baselines (IDQN, IPPO, FMA2C, MPLight) on throughput, average wait, and environmental indicators; ablations further indicate that adding the global agent improves outcomes and that localized global windows (3×3 / 5×5) offer a practical path to scale while retaining benefits on fuel/CO/CO₂ across 16/49/169-intersection scenarios.

### Strengths
**Novel framework:** The introduction of a cooperative global-local structure is conceptually clear and practically valuable.

**Environmental perspective:** Explicitly integrates carbon emission modeling (via HBEFA in SUMO), which is novel in reinforcement-learning-based traffic control.

**Strong empirical validation:** Comprehensive experiments on real-world and benchmark datasets demonstrate consistent improvements.

**Scalable and interpretable:** The 3×3 and 5×5 local-global grids improve scalability while maintaining decentralization.

### Weaknesses
**Theoretical depth is limited:** No rigorous convergence or stability proofs beyond intuitive justification. There are no formal proofs, theorems, or mathematical analyses for CARTS's own convergence or stability (e.g., no bounds on policy improvement or analysis of the global agent's impact on training dynamics).

**Lack of statistical robustness:** Experimental tables (Tables 1–5) report only single-run results — **no mean, variance, or significance analysis** is provided. This undermines reproducibility and statistical credibility.

**Innovation boundary unclear:** Similar ideas have appeared in CoSLight (Ruan et al., 2024) and MARL coordination studies, novelty could be better clarified.

**Ablation and sensitivity:** No detailed sensitivity analysis for emission-related parameters (e.g., varying Mfuel or other constants).

**No analysis of inference-time distribution shift**:  CARTS relies on a global agent during training but is fully decentralized at inference. The paper repeatedly states this training/inference mismatch yet provides no stress tests under distribution shift (e.g., time-varying demand, incidents, lane closures) to show the learned local policies remain stable without the global coordinator. Please add controlled “before vs. after” experiments where the traffic distribution changes mid-episode and report performance deltas. 

**Scalability conclusions are not tied to compute/latency budgets:** The manuscript notes that processing all nodes is “impractical” and therefore restricts the global agent to nearby intersections (e.g., 3×3, 5×5; >10 intersections uses only neighbors). However, there is no quantification of GPU memory, wall-clock training time, or inference latency per intersection. Without these numbers it’s hard to assess real-time feasibility at city scale. Please report peak memory, training time/step, and per-decision latency across the 16/49/169-node setups, and discuss any batching/parallelization.

### Questions
1) How does the model perform under non-stationary traffic patterns (e.g., rush hour vs. sudden road closures)? The paper mentions non-stationary environments in related work (page 10: citing Abdoos et al. (2011) on "non-stationary environments"), but experiments use fixed datasets without explicit non-stationary tests. 

2) Can the authors provide more formal analysis of convergence or policy stability? The paper only sketches a generic RL value-function convergence argument (Banach fixed point, etc.) in the appendix.

3) Are the HBEFA emission parameters fixed or learned? How sensitive are results to these values? The manuscript lists the HBEFA-style equations and constants and argues that waiting time dominates emissions, but it does not specify the parameter settings**, **fleet composition, or any calibration/sensitivity of results to these choices. 

4) Does removing the global agent at inference lead to distributional shift or performance drop? The paper emphasizes that the global agent is used only during training and the system is fully decentralized at inference, and shows ablations “no global agent vs. with global agent” in training; however, it does not test non-stationary deployment or distribution shifts where the absence of the global coordinator might matter.

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
5

### Summary
The paper proposes CARTS, a cooperative reinforcement learning framework for intelligent traffic signal control. CARTS introduces both local agents (for individual intersections) and a global agent (for system-level coordination) to jointly optimize traffic flow efficiency and carbon emission reduction. During training, the global agent guides local policies toward globally coherent behavior, but inference remains fully decentralized. The method is built on the DDPG algorithm to enable continuous control of signal phase durations.

### Strengths
S1. The introduction of a global agent that coordinates local agents during training enhances overall traffic efficiency without requiring centralized control at deployment.

S2. CARTS supports continuous and adaptive signal phase durations through DDPG, improving responsiveness to real-time traffic fluctuations.

### Weaknesses
W1. The method shows limited novelty, as the idea of using a global agent to coordinate local training is similar to FedLight (DAC 2021).

W2. Using DDPG for continuous-time control introduces instability issues, including overestimation bias and high variance.

W3. The carbon emission modeling is superficial and does not influence the core algorithm design.

W4. The experimental evaluation is insufficient and omits many state-of-the-art RL-based TSC methods.

W5. The convergence proof is limited to classic single-agent DDPG and does not address multi-agent non-stationarity.

### Questions
Q1. While CARTS presents a cooperative architecture with a global agent to guide local intersections, this concept is not entirely new. Similar ideas have been explored in earlier works such as FedLight (DAC 2021), which also leverages a centralized coordinator to improve decentralized traffic signal control. The paper does not sufficiently clarify how its coordination mechanism differs conceptually or technically from these prior approaches, which reduces the methodological originality of the contribution.

Q2. CARTS adopts DDPG to learn variable signal durations, but DDPG is known to suffer from overestimation bias, high variance, and convergence instability in continuous control tasks. These problems are amplified in multi-agent settings, where each agent’s non-stationary policy continuously changes the environment seen by others. 

Q3. Although the paper emphasizes carbon emission reduction, this component is implemented only by calling the HBEFA emission model built into the SUMO simulator. The model merely computes CO and CO₂ emissions post hoc, without feeding back into the reward function or influencing the policy learning process. As a result, the emission analysis functions more as a descriptive add-on than an integrated optimization objective, limiting the contribution’s depth regarding sustainability.

Q4. The experiments benchmark CARTS against a small set of baselines (e.g., MA-DDPG, PPO, TD3, MPLight), but ignore numerous recent state-of-the-art methods in RL-based traffic signal control. Moreover, several settings and evaluation metrics are underreported, making it difficult to assess fairness and generalizability. A more comprehensive and standardized comparison is necessary to validate the claimed performance advantages.

Q5. The appendix provides a convergence analysis based on the Banach fixed-point theorem for single-agent DDPG, but this proof does not extend to the multi-agent, non-stationary environment considered in CARTS. Key aspects such as convergence rate, stability boundaries, and inter-agent coupling effects are ignored. Without a theoretical framework or empirical evidence supporting stability under multi-agent coordination, the claimed convergence guarantees remain unconvincing.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces CARTS (CooperAtive Reinforcement Learning for Traffic Signal Control), a novel cooperative reinforcement learning framework designed to optimize traffic signal control and reduce carbon emissions. Traditional traffic signal control systems often rely on overly simplistic, rule-based approaches. Even RL-based methods tend to be suboptimal and unstable due to the inherently local nature of control agents. To address this, the authors propose the CARTS framework, which incorporates multiple reward terms weighted with an age-decaying mechanism to optimize traffic signal control at a global scale. The framework includes two types of agents: local agents that focus on optimizing traffic flow at individual intersections, and a global agent that coordinates across intersections to improve overall throughput. Importantly, the system is designed to reduce both vehicle waiting times and carbon emissions. Experimental results using real-world traffic data from an Asian country show that, despite incorporating a global agent during training, CARTS remains decentralized during inference, requiring no centralized coordination during deployment. Results demonstrate that CARTS consistently outperforms state-of-the-art methods in all evaluated metrics. Moreover, CARTS effectively links carbon emission reduction with global agent coordination, providing an interpretable and practical approach to sustainable traffic signal control.

### Strengths
1.The paper addresses a significant real-world problem: optimizing traffic signal control while reducing carbon emissions. This is highly relevant and practical.

2.The introduction of the global agent in the CARTS framework effectively tackles issues arising from the decentralized nature of traditional RL methods in multi-agent traffic signal control.

3.The paper’s contributions lay a solid foundation for future research in adaptive traffic signal control and carbon emission reduction.

4.The experiments are well-structured, and the results show that CARTS outperforms current state-of-the-art methods on several performance metrics, confirming the robustness and scalability of the approach.

5.The paper is clearly organized and easy to follow, making the main points and contributions clear and accessible.

### Weaknesses
1.The explanation of related work is brief. While the paper cites several relevant RL methods, there is little discussion on how these methods compare with the proposed approach. A deeper comparison with existing methods would help clarify the unique contributions of CARTS.

2.The role of the global agent in training is highlighted, but there is limited discussion on why the system remains decentralized during inference. More explanation on how the transition from centralized training to decentralized inference works would strengthen the paper.

3.While the paper claims carbon emissions are reduced, more details on how the carbon reduction mechanism works in practice would be helpful. A more in-depth explanation of the environmental impact would make the claims more robust.

4.The paper presents results showing the superiority of CARTS but does not discuss the limitations or potential failures of the approach in detail. Providing more insight into any scenarios where CARTS might struggle would be valuable.

### Questions
1、In Section 4.2 of the article, it states: "In particular, the ϵ-greedy method will gradually reduce ϵ from 0.9 to 0.1 and a time decay mechanism is adopted to decay Wm G by the ratio (0.95)t in the t-th iteration." However, there is a lack of further explanation on how the decay rate is determined. How sensitive is the model to changes in the decay parameter? Does it affect the stability and performance of the model?

2、In Section 4.3, the article mentions "The duration of the traffic phase ranges from Dmin to Dmax seconds." What are the criteria for selecting Dmin and Dmax? Are these based on actual traffic data, or have the characteristics of the specific city's traffic environment (such as traffic flow and road network density) been taken into account? Do the selected parameter values consider the demand under different traffic conditions? Does the fixed green light duration range limit the exploration space for the agent, especially during the early stages of reinforcement learning? A too narrow duration range might cause the agent to converge to a more limited strategy, reducing the overall learning effectiveness.

3、In Section 4.5, the concept of carbon emissions is introduced, including the emissions of carbon monoxide and carbon dioxide during driving and when the vehicle is waiting. The article then states: "Since tstop during our experiments, the waiting time is the main variable affecting emissions." This transition is abrupt and lacks detailed explanation. The causal inference is not rigorous. To verify this point, the direct causal relationship between waiting time and carbon emissions must be clearly stated through data or experimental design, and how other confounding factors are eliminated should also be addressed.

4、In the second paragraph of Section 5 (EXPERIMENTAL RESULTS), "(¿10 intersections)" appears, which seems to be a formatting or encoding issue.

5、The simulation used SUMO and TSIS platforms. Could you clarify the assumptions regarding the integration of these simulation tools with carbon emission models (such as HBEFA)? In practical deployment, considering the simplifications made in the simulations, can the effectiveness of carbon emission reduction be reliably ensured?

6、Although CARTS performs well in simulations, how adaptable is it in real-world environments? Specifically, how does the system adjust in real-time to maintain effective traffic flow control in the case of unexpected events such as traffic accidents, road closures, and extreme weather?

### Soundness
3

### Presentation
3

### Contribution
3
