# City-Adaptive Testing of Autonomous Driving with Traffic Prediction and Scenario Fuzzing

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Autonomous Driving Systems (ADS) often struggle in complex urban environments because generic testing fails to capture city-specific traffic patterns and behaviors. To address this, we propose a city-adaptive testing framework that systematically evaluates ADS robustness by integrating spatiotemporal traffic prediction and multi-agent behavioral modeling. Our approach first introduces a novel traffic prediction model, called T-DDSTGCN, which combines graph and hypergraph representations to accurately forecast segment-level traffic speed and intersection turning probabilities. It achieves the best performance on both METR-LA and PEMS-BAY datasets, demonstrating its superior ability to capture spatiotemporal dependencies in traffic prediction tasks. Based on the predicted urban traffic flow, we construct diverse simulation scenarios enriched by a behavioral modeling framework called Primary Other Participants (POP), which simulates realistic motorcycle behavior using Level-K game theory and Social Value Orientation. To enhance scenario diversity, we further apply structured perturbations across traffic density, weather, and agent interactions. Our methodology is validated across 180 real-world urban scenarios on three industrial-scale simulation platforms, yielding 662 critical collision cases after multiple rounds of testing. We have conducted an initial manual screening of the 662 simulated accident scenarios, finding that 88.1\% of these accidents closely resemble real-world accident videos and reports. Furthermore, ablation studies highlight the critical role of human-like agent behavior in exposing ADS failures. Our findings suggest that incorporating traffic context and behavioral diversity into simulation testing is crucial for ensuring ADS safety and robustness in real-world deployments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a city-adapting testing framework: a spatiotemporal traffic predictor (T-DDSTGCN) to predict segment speeds; a turning-movement estimator (Speed2Turning); a behavior model for motorcycles (POP) leveraging Level-K game theory and Social Value Orientation; a scenario fuzzing method across traffic density, weather, and interactions. The paper conducts evaluations on METR-LA/PEMS-BAY datasets showing competitive traffic prediction performance of T-DDSTGCN. The fuzzing process results in 662 critical collisions, with 88.1% similar to real-world cases.

### Strengths
-paper is overall clear

-implemented a complete end-to-end test generation framework (prediction -> turning inference -> scene reconstruction -> POP agents -> fuzzing)

### Weaknesses
-T-DDSTGCN’s performance is quite close to that of SAGDFN (table 1)

-diversity of the generated 662 collisions is not analyzed in-depth. Although more collisions have been found, I wonder how many of them are truly new, unique, and valuable to downstream applications (e.g., improving the ADS).

### Questions
-is the performance difference between T-DDSTGCN and  SAGDFN statistically significant?

-how many of the generated collisions are unique and valuable to downstream applications?

-what are the root causes of the found collisions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a city-adaptive testing framework for autonomous driving simulation. It integrates (1) T-DDSTGCN, a dual dynamic spatio-temporal GCN for predicting road speeds and turning probabilities, (2) the POP model for realistic participant behavior based on game theory and social value orientation, and (3) a scenario fuzzing mechanism for diverse condition generation. Experiments on LA and SF data show improved realism and higher ADS failure exposure, demonstrating the framework’s effectiveness for city-level adaptive testing.

### Strengths
1. Clearly identifies an underexplored problem: "the lack of city-level adaptability in autonomous driving simulation", and proposes a coherent, data-driven solution.
2. The framework connects traffic prediction, behavioral modeling, and scenario fuzzing into a unified testing pipeline, bridging machine learning and simulation.
3. Experiments across multiple real-world cities and industrial-grade simulators (PanoSim, OasisSim, Apollo) support the claims, showing clear improvements in scenario realism and ADS failure exposure.
4. The system directly addresses real-world testing needs for autonomous vehicles, and the methodology is generalizable to other city-scale environments.

### Weaknesses
1. Predictive model necessity not fully justified. Seems unclear whether city-level forecasting provides substantial benefit over replaying historical data. In fact, the difficulty of collecting and recording large-scale urban traffic data is no longer high.
2. Restricted reproducibility. Key simulators (PanoSim, OasisSim) are commercial, limiting open validation.
3. The paper lacks detailed visualizations or case studies of complex driving interactions (e.g., POP-induced maneuvers, dense multi-agent scenarios), making it difficult to assess the qualitative realism of the generated traffic scenes.

### Questions
1. About the necessity of city-level forecasting: Please clarify the specific advantages of the T-DDSTGCN forecasting module compared to directly replaying or interpolating historical traffic data. Under what conditions does predictive modeling provide tangible benefits for city-scale simulation? Quantitative comparisons or ablations against a replay baseline would strengthen the justification.
2. On reproducibility and simulator access: Since two of the simulators (PanoSim, OasisSim) are commercial, can the authors provide code to replicate results on publicly available platforms (e.g., CARLA, Apollo)?
3. It would be very helpful to include more visualization results showing complex multi-agent behaviors, dense traffic interactions, and the effects of POP or fuzzing. Such visual evidence would make the framework’s realism and interpretability more convincing.

If the authors can convincingly address the concerns, I would be willing to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposes. Contrary to most prior works on traffic simulation, this work focuses more on the macro level 1. First, it uses a Graph Convolutional Network to predict traffic-flow speed and uses a rule-based turning probability prediction, where low-level is done by optimization-based game-theoretic planning. Although this work focuses on an important problem, key assumptions, contributions, and details are not clear ( the ego planning algorithms, the experiments setup, utility functions).

### Strengths
This paper focuses on two important problems: 1) Collision Scenario Testing  and 2) VRU, motorcycle modeling is rarely studied in previous work

### Weaknesses
- In 3.2 POP-ENHANCED SCENE SIMULATION, the assumption of planning using only the nearest motorcycle is overly restrictive: traffic scenarios often involve multiple motorcycles interacting over time, not just the closest one.
- Utility function is a large assumption about motorcycle behaviors. For example, can the proposed utility function handle motorcycles with diverse style such as swerving around the traffic, cut-in behaviors? What are the utility functions? This work also assumed that we know ADS’s reward function as well as the simulated agent's utility function.
- The Scenario fuzzing axises are not a new contribution: such as varying density, weather variations, positions.

### Questions
- My personal experience on SVO is that switching parameters does not vary the behavior a lot for multi-agent behavior.  Does this work focus only on two‐agent interactions (one motorcycle + one AV)? If so, how would SVO parameterization scale in more complex traffic scenes? This may limits the scalability of this work
- How does changing the weather actually alter the scenario dynamics?
- The assumption of predicting traffic flow speed, turning probability a valid one? What is the limitation of this formulation compared to previous works?
- How do you measure if the generated accident scenarios are realistic, what is the protocal for the human evaluation in this work?
- What is the AV or ADS system evaluated in this work?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents city-adaptive autonomous driving system testing framework that covers city-specific traffic patterns by spatial-temporal graph convolution network and behavioral modeling algorithm which simulates human-like maneuvers. Proposed Behavior modeling based on Game theory and Social value orientation supports realistic agent generation.

### Strengths
1. The proposed traffic speed prediction method outperforms baseline models by leveraging graph and hypergraphs
2. The paper introduces Level-K game theory into the POP model, enabling highly realistic and strategic behavioral simulations

### Weaknesses
1. The paper proposed T-DDSTGCN for traffic flow prediction. However, the main idea is derived from DDSTGCN [1], lacking novelty
2. If the proposed T-DDSTGCN is different from previous DDSTGCN, ablation for performance comparison need to be conducted.
3. The overall pipeline contains too much heuristics, such as speed differential for turning probability and generated motorcycle initially appearing only on branch roads 
4. Authors implement speed differential as variable for turning probability, but as authors mentioned in line 1166, other factors play significant influence on turning decisions
4. Figures and tables need to be improved for clarity and readability, especially for figure 2 and table 2


[1] Sun, Y., Jiang, X., Hu, Y., Duan, F., Guo, K., Wang, B., ... & Yin, B. (2022). Dual dynamic spatial-temporal graph convolution network for traffic prediction. IEEE Transactions on Intelligent Transportation Systems, 23(12), 23680-23693.

### Questions
1. Related to weakness [1], how does the propsed network T-DDSTGCN differ from previous DDSTGCN?
2. How did you decide if simulated accident scenarios closely resemble real-world accident with quantitative value? (88.1%) Is the number fair enough for city-adaptive traffic?
3. Could you explain the reason for utilizing DFS algorithm at line 294?
4. Could you clarify the specific reason for focusing the POP model on motorcycles? It seems the proposed method might also be applicable to other agents, like vehicles.

### Soundness
2

### Presentation
2

### Contribution
2
