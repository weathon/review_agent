# DriveE2E: Closed-Loop Benchmark for End-to-End Autonomous Driving through Real-to-Simulation

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Closed-loop evaluation is increasingly critical for end-to-end autonomous driving. Current closed-loop benchmarks using the CARLA simulator rely on manually configured traffic scenarios, which can diverge from real-world conditions, limiting their ability to reflect actual driving performance. To address these limitations, we introduce a simple yet challenging closed-loop evaluation framework that closely integrates real-world driving scenarios into the CARLA simulator with infrastructure cooperation. Our approach involves extracting 800 dynamic traffic scenarios selected from a comprehensive 100-hour video dataset captured by high-mounted infrastructure sensors, and creating static digital twin assets for 15 real-world intersections with consistent visual appearance. These digital twins accurately replicate the traffic and environmental characteristics of their real-world counterparts, enabling more realistic simulations in CARLA. This evaluation is challenging due to the diversity of driving behaviors, locations, weather conditions, and times of day at complex urban intersections. In addition, we provide a comprehensive closed-loop benchmark for evaluating end-to-end autonomous driving models. Code and dataset examples are in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a novel closed-loop evaluation benchmark of end-to-end autonomous driving agents by constructing digital twins of real-world intersections and replaying logs of the traffic participants. It uses infrastructure sensors for reconstruction and the CARLA simulator for simulation. The benchmark is comprehensive and consists of various driving behaviors and traffic participants.

### Strengths
1. Closed-loop simulation is important for benchmarking end-to-end driving models and constructing digital twins of real-world driving scenes can help with closed-loop testing and bridge the sim2real gap.
2. Using infrastructure-mounted sensors can help improve the accuracy and complexity of the reconstructed traffic scenario, improving the realism of the digital twins.
3. The proposed DriveE2E benchmark is comprehensive in the number of scenarios, agent behaviors, and traffic participant categories.

### Weaknesses
1. Although acknowledged in the limitations section, the log-replay simulation is a major weakness of this work for more realistic closed-loop evaluation, as it should be straightforward to incorporate behavior models for other agents. This could have been an advantage of using a driving simulator like CARLA, which offers easier agent management.
2. The reconstructed digital twin of real-world interactions lacks diversity and realism. The layout of the intersection is all four-way intersections, and the assets, such as bushes, trees, and road textures, all appear the same. 
3. The collected real-world data all come from a small region with only 15 intersections, which lacks diversity. Therefore, the results from such a benchmark may not be general and comprehensive for the driving agent's performance.
4. With the lack of photorealism of the image rendering of CARLA, the proposed DriveE2E benchmark is more suitable for modular testing of AD systems like downstream planning and decision making rather than evaluating end-to-end driving agents. End-to-end driving policies trained from such data may also exhibit a large sim2real gap.

### Questions
1. Can the authors provide some video results of the driving scenario visualizations? 
2. Can the authors provide more statistics on the assets used for creating the digital twins, or do they simply reuse the CARLA assets? Like how many different types of vehicles, pedestrians, etc? Are they fixed or randomly initialized in each scenario?
3. Can the authors elaborate more on the real-world data collection? For example, what types and how many infrastructure sensors are used? What is their perception range?
4. In Tab. 7, can the authors explain more about the element-wise similarity metrics used to evaluate the reconstruction fidelity?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces DriveE2E, a closed-loop end-to-end autonomous driving benchmark that integrates real-world driving scenarios into the CARLA simulator. The authors extract  800 dynamic traffic scenarios from real world data collected by infrastructure-mounted sensors
and create digital twin assets for 15 real-world intersections. Then the authors analyze the data distribution and evaluate several state-of-the-art end-to-end autonomous driving methods on the benchmark.

### Strengths
1. The paper proposes a practical real-to-simulation workflow to bulid driving scenarios in CARLA simulator with the real world data.
2. The proposed infrastructure cooperation is effective to collect extra information for simulator.

### Weaknesses
1. The scale of the digital twin build by the benchmark is insufficient. Only containing 15 intersections, the digital twin is much smaller than the towns already in CARLA and the diversity of surroundings is limited. Additonally, the benchmark does not contain other road structures such as T-junctions or roundabouts.
2. The sim-to-real gap are not evaluated qualitatively. Adding figures or videos of scenarios in real world and simulator at the same location in the same view can show the gap clearly. 
3. The simulation is only partly "closed-loop". The agents only replay the logs without interacting with the ego-vehicle, which hurts the realism of simulation.
4. How to make digital models of traffic agents is not mentioned. CARLA itself has a relatively small amount of vehicle models. Only selecting existing models for traffic agents in real world may lead to a large sim-to-real gap on the perception task of autonomous driving.

### Questions
1. Will the data processing code used to build the digital twins be open-sourced? These tools are helpful for the community to build more simulation scenarios with real world data.
2. It seems that the static intersection assets construction includes a lot of manual work. The high cost may make the digital twin construction unscalable. What is the typical time to build the digital twin of a single intersection manually? Are there any methods to reduce the manual construction time?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a real-to-sim digital-twin pipeline that ports real-world intersections into the CARLA simulator for closed-loop evaluation of autonomous driving systems. The authors reconstruct static environment assets (geometry/layout) to create digital twins of 15 real intersections, and use these scenes to evaluate multiple end-to-end driving baselines in fully closed-loop simulations. The goal is to enable repeatable, scenario-faithful testing in simulation while preserving key characteristics of real-world locations.

### Strengths
- `Well-motivated digital-twin evaluation: `Building digital twins for closed-loop assessment of end-to-end autonomous driving is timely and reasonable. The pipeline is clearly presented, and the multi-source asset creation—combining Blender tooling with OpenStreetMap, HD maps, and satellite imagery—is thoughtfully executed. The engineering effort is evident and appreciated.

- `Benchmarking value:` The paper provides extensive, apples-to-apples comparisons of popular end-to-end driving methods within the same digital-twin environments. This offers a strong reference baseline for the community and supports reproducibility and fair comparison.

### Weaknesses
- `Overlap with prior digital-twin platforms (ScenarioNet/MetaDrive):`
The core idea—building digital twins in a simulator for evaluation—appears closely related to prior work such as ScenarioNet/MetaDrive [1]. Please clarify the advantages over the previous work.

Reference:
[1] ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling

- `Visual domain gap between real and simulated scenes:`
While digital twins enable controllable evaluation, the appearance gap between real images and simulated renderings (cf. Fig. 2) is noticeable and may bias conclusions for on-vehicle deployment. Please quantify and discuss the impact.

- `Unexpected baseline ranking (UniAD outperforming MomAD in Table 3):`
The result contrasts with MomAD’s strong closed-loop performance on Bench2Drive. In my opinion, there are indeed possible factors that include differences in sensor suites/resolution, control frequency/latency, action spaces and low-level controllers, training data/domain, metric definitions (e.g., Driving Score variants), and route/weather distributions. I am wondering the reason behind this difference from the authors' aspects.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents DriveE2E, a Real2Sim-based closed-loop evaluation framework built upon the CARLA simulator. It features high-fidelity digital twins of 15 diverse urban intersections and 800 traffic scenarios generated from infrastructure sensor data.

### Strengths
The paper is well-written and clearly structured, making it easy to follow. It constructs high-fidelity digital twins of 15 urban intersections and selects 800 real-world traffic scenarios from over 100 hours of infrastructure sensor data. Meanwhile, it establishes a comprehensive closed-loop benchmark for end-to-end autonomous driving (E2EAD) by evaluating several classical baselines, demonstrating both technical completeness and practical relevance.

### Weaknesses
There are several points that require further clarification. The proposed Real2Sim framework is intended to address the unrealistic rendering issues in CARLA, yet the paper still relies on CARLA’s built-in assets for traffic participants, which seems inconsistent with that motivation. In addition, one of CARLA’s main advantages lies in its true closed-loop capability, where surrounding agents can dynamically respond to the ego vehicle’s behavior—unlike the log-replay mode adopted in this work. While this paper introduces a new scenario, it also employs a log replay method. Therefore, what are its advantages compared to frameworks like nuPlan or NAVSIM? This will determine the value of this work.

### Questions
Same to the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3
