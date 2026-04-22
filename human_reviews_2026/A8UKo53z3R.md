# WFDroneBench: A Benchmark for Sensor Placement and Drone Routing for Wildfire Detection

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Increasingly frequent and severe wildfires threaten ecosystems, public health, and infrastructure. Early detection is vital but limited by existing monitoring systems. Drones offer mobile, real-time coverage, but optimizing sensor placement and drone routing in dynamic fire zones remains challenging. To address this, we in- troduce WFDroneBench, an open-source Python benchmarking library for early wildfire detection that integrates machine-learned risk maps with optimization- based deployment strategies for sensors, charging stations, and drones. It evaluates risk maps, optimization strategies, and monitoring equipment using standardized metrics and realistic wildfire simulations. The framework supports benchmarking across predictive and decision-making components: machine learning researchers can assess risk models and operations research experts can compare routing strate- gies. WFDroneBench includes 7746 scenarios across 49 locations, built from historical ignitions, real-world wildfire risk maps, and simulated fire spread, along with two ground detector and four drone routing strategies. Our experiments show that risk-aware strategies – Team Orienteering Problem (TOP) and Max-Coverage – significantly outperform other baselines when risk maps are sufficiently accurate, with TOP achieving the fastest detection on the most difficult fires. We further find that risk-aware static infrastructure helps even under an imperfect riskmap and drone-based detection outperforms ground sensors. Finally, our results reveal two key open challenges: (i) detecting small fires rapidly and reliably, and (ii) improv- ing risk-map prediction, where the gap between ground-truth ignition patterns and available risk maps highlights a significant opportunity for ML innovation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors have a good idea in trying to develop a risk-based approach to determining wildfire fighting strategies, but this paper is trying to do too much. It was very hard to follow.

### Strengths
The motivation is relevant and significant.

### Weaknesses
*The paper was hard to follow, the authors need to focus either on 1) validating whether their WFDroneBench workflow supports four key stakeholder groups as they claim, or 2) their benchmarking (including validating their semi-synthetic wildfire benchmark dataset.) For either case, they must get feedback from actual firefighters that their approach is actually effective.

### Questions
What research questions were the experiments answering?

How can firefighters know that your risk estimation is in alignment with actual operations?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an open-source benchmark WFDroneBench for early fire detection, incorporating risk map, deployment strategies of ground equipment and drone routing strategies. WFDroneBench is modular and supports separate research on different system components. In addition, the paper curates a wildfire dataset containing 7746 scenarios. This paper evaluates and compares different wildfire detection strategies in its benchmark.

### Strengths
1. Good motivation. Proactive wildfire detection proves valuable in wildfire research.
2. The modular workflow allows the system’s components to support diverse research objectives, enhancing the extensibility of this benchmark.
3. The paper is well-written and provides a detailed introduction to the proposed benchmark.

### Weaknesses
1. The experiment results are inconsistent with the discussion conclusions:
* In Table 2, the UniCov drone routing strategy achieved the optimal detection rates and comparable detection times compared to TOP and Max-Cov, which contradicts the conclusion in line 413.
* In Table 2, compared with Max-Cov, UniCov achieved better detection rate and detection time simultaneously under the Rondom Sensor Strategy. The same experimental result is shown in sub-scenarios such as Slow Big and Fast Big, which contradicts the conclusion in line 418. 
* In Table 2, compared with Uni-Cov, a significant decline in the detection rates of TOP and Max-Cov is observed, especially in Slow Small and Fast Big scenarios. The negative impact on detection rates may outweigh the benefits of the risk map.
* When changing from the GT dynamic map to the BP risk map, the performance of TOP has declined, especially in the Fast Big scenario. This reult cannot prove robustness proposed in line 431.
* In Table 3, Max-Cov outperforms Uniform Coverage in detection rate, which contradicts the conclusion in line 436.
2. With respect to the proposed dataset, additional quantitative analyses of data distribution and data quality are recommended.

### Questions
It would be valuable to vary the numbers of ground stations, charging stations, and drones to conduct comparative experiments for exploring optimal resource allocation.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a new benchmarking framework, WFDroneBench, for early wildfire detction. It integrates a wide range of scenarios and locations based on historical ignition data and physics-based spread simulations. Hereby, the authors aim to provide a benchmark that supports the interests of multiple stakeholders and viewpoints, covering, e.g., the detection, drone allocation, and mechanical engineering decisions. In their experiments, they show the performance of different strategies in sensor placement and drone routing in wildfire detection.

### Strengths
- The paper is written in a very clear and concise way. It is easy to follow the proposed ideas and setup of the benchmark.
- WFDroneBench allows evaluating multiple important factors of the wildfire detection tasks, such as charging station and ground sensor placements, on top of the drone operations themselves. Hence, a holistic evaluation of strategies can be obtained.
- The benchmark is equipped with a selection of routing and sensor placement baselines to compare with.

### Weaknesses
- While the evaluations consider different strategies, only a single hardware setup is demonstrated. Showing additional drone hardware configurations would support the claims about meeting the needs of, e.g., mechanical engineers, as claimed in Figure 1.
- While the included strategies for drone routing and sensor placement act as a baseline (e.g., uniform coverage, brownian motion), the experiments section would be enhanced significantly by incorporating other, state-of-the-art drone routing approaches.
- The evaluation of the benchmark focuses on the rate of fires detected in under 12 hours (Table 2, 3) with varying sensor placement and routing. It seems like the proposed utility of the benchmark for policy makers and mechanical engineers seems underexplored by the manuscript.

### Questions
- In line 471, it is mentioned that environmental factors such as wind and temperature are not considered in the evaluation of the drones' performance. Since these factors impact the spread of wildfires, and may be useful for tasks such as routing, do the scenarios in the benchmark contain these informations, or is their inclusion entirely up to future work?

### Soundness
3

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
2

### Summary
This paper introduces WFDroneBench, an open-source benchmark and simulation framework that helps researchers and engineers test and compare wildfire detection strategies. The benchmark is extensively and useful, using real ignition data and realistic fire spread simulations. Experiments show that risk-aware routing strategies — especially TOP and Max-Coverage — detect fires faster and more effectively than random or uniform approaches. Drones outperform stationary sensors by a large margin. However, the success of these strategies depends heavily on having accurate and up-to-date risk maps.

### Strengths
The benchmark appears to be well executed and comprehensive.

1. Wildfires are a critical global issue, and early detection is both societally impactful and technically challenging. Thus a benchmark for this task is likely going to be very useful with the rise in global warming.
2. Glad to see that the benchmark would be completely open-source and has an associated library and toolkit, hopefully would be used to explore this clearly large problem space.
3. The experiments evaluate multiple baselines under consistent conditions, including both static and dynamic risk maps. The authors appear to have put effort in constructing reasonable baselines.

### Weaknesses
1. The work is primarily a benchmark, not a new algorithmic contribution. The routing formulations are not conceptually new. Reinforcement learning or imitation learning methods are only mentioned as future work. Including even one such baseline would make the benchmark more forward-looking.
2. The text occasionally reads more like a system report — it’s dense, with many details about parameters and datasets but less emphasis on insights.
3. Relevance to ICLR is low. It fits better in a systems, applied AI, or robotics venue than a core learning venue. Please comment on why this work is relevant for this venue.

### Questions
Please clarify my questions in weakness. In general, the paper is well executed but doesn't offer many insights into the problem itself and could use some experiments which illuminate open problems in this domain which would improve the benchmark.

### Soundness
3

### Presentation
2

### Contribution
2
