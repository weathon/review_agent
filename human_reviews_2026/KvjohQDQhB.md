# V2X-UniPool: Unifying Multimodal Perception and Knowledge Reasoning for Autonomous Driving

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Autonomous driving (AD) has achieved significant progress, yet single-vehicle perception remains constrained by sensing range and occlusions. Vehicle-to-Everything (V2X) communication addresses these limits by enabling collaboration across vehicles and infrastructure, but it also faces heterogeneity, synchronization, and latency constraints. Language models offer strong knowledge-driven reasoning and decision-making capabilities, but they are not inherently designed to process raw sensor streams and are prone to hallucination. We propose V2X-UniPool, the first framework that unifies V2X perception with language-based reasoning for knowledge-driven AD. It transforms multimodal V2X data into structured, language-based knowledge, organizes it in a time-indexed knowledge pool for temporally consistent reasoning, and employs Retrieval-Augmented Generation (RAG) to ground decisions in real-time context. Experiments on the real-world DAIR-V2X dataset show that V2X-UniPool achieves state-of-the-art planning accuracy and safety while reducing communication cost by more than 80\%, achieving the lowest overhead among evaluated methods. These results highlight the promise of bridging V2X perception and language reasoning to advance scalable and trustworthy driving. Our code is available at: https://anonymous.4open.science/r/V2X-UniPool-7326

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
V2X-UniPool unifies V2X perception with language-based reasoning by translating multimodal signals into a time-indexed knowledge pool and using RAG to fuse the retrieved context with on-board perception for planning. This results in impressive improvements on DAIR-V2X, cutting transmissions by >80% and reducing collisions to near-zero.

### Strengths
* V2X-UniPool bridges perception-centric V2X with language-centric planning via a time-indexed knowledge pool and RAG grounding, explicitly tackling heterogeneity, temporal desynchronization, and hallucination.
* The static/dynamic pool provides multi-resolution temporal semantics, and ablations across model architectures show consistent gains.

### Weaknesses
* The latency and network assumptions may be optimistic. There’s no robustness study for lower bandwidth, variable latency, or packet loss.
* Results are reported only on DAIR-V2X-Seq and in open-loop. the paper note end performance still depends on the chosen vehicle-side AD model and plan to pursue closed-loop validation, so real-world robustness and controller interaction effects remain untested.

### Questions
The more complete illustration based on the latency and network assumptions will be helpful.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces V2X-UniPool, a unified framework that bridges V2X perception with language-based reasoning for planning in autonomous driving. The system converts raw sensor streams from infrastructure and other various information into structured, language-based knowledge and organizes this knowledge into a knowledge pool with multiple categories. The empirical evaluation on the DAIR-V2X dataset demonstrates improvements in planning accuracy, safety, and communication cost over previous baselines.

### Strengths
1. The proposed framework, V2X-UniPool, using natural language as the only form to transmit information from the infrastructure to vehicles, significantly reduces the communication cost.
2. The design of Static/Dynamic, High Frequency/Low Frequency pool systematically organizes the information stored in the infrastructure.

### Weaknesses
### Major Weaknesses
1. **The performance gain compared with the previous SoTA (V2X-VLM) is not significant.**
- Compared with full configuration of V2X-VLM, the L2 error is much higher. Notably, while V2X intuitively should provide more performance gain in long-term planning (Just as the authors stated, V2X helps extend perception range), V2X-UniPool shows an even larger gap on 3.5s and 4.5s error.
- Compared with the no-fusion result from V2X-VLM, which achieves an average L2 of 1.49m, the 1.47m result is not convincing, especially when more external knowledge is given.
- Considered V2X-VLM directly predicts waypoints, V2X-UniPool predicts speed and curvature following the OpenEmma paradigm. The effect of this difference is unclear.
2. **Missing details in the knowledge pool component.**
- In section 1, section 3.2, and Figure 2, LiDAR input is seen as an essential part, but no further explanation on how this modality is integrated into the system.
- In section 3.2, from equations 3 and 4, it seems that the proposed two sub-pools cannot fully cover the dynamic pool described in Eq. 2.
3. **The ablation study is also not well designed to prove the effectiveness.**
- Multiple key components, including information retrieval, knowledge pool operations, build up a relatively complicated system. However, no detailed information is provided to show how those components contribute to the performance gain in Table 3.

### Minor Weaknesses
1. The VLM setup (probabilistic sampling or not, preset temperature) is unclear, which is important to the significance of the results in Table 3.
2. In section 3.3, Encode and Retrieve are not well defined.

### Questions
1. What's the performance of V2X-UniPool if the downstreamed AD model predicts waypoints directly? This helps to have a fair comparison with baselines.
2. Do you have qualitative results to show the actual improvement with V2X information integrated? Given that DAIR-V2X is collected at signalized intersections, do frequent conflicting interactions occur?
3. For the mapping from messages to specific pools, is it fixed during the design stage or dynamically changes on the fly? Will different pools be treated differently in the downstream tasks? How will this categorization affect the performance?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents V2X-UniPool. The core idea is to establish a "V2X as a Knowledge Service" model, where roadside infrastructure (RSUs) acts as a centralized processor. It fuses raw multimodal sensor data (LiDAR, cameras) into a structured, language-based "Knowledge Pool." Vehicles (users) then query this pool to retrieve a snapshot of their environment, which is fused with their local perception for final decision-making by a language model. The authors claim this approach addresses key V2X challenges like heterogeneity and high communication costs. Experiments on the DAIR-V2X dataset show an impressive reduction in communication overhead (>80%) and competitive planning accuracy.

### Strengths
- The concept of a centralized knowledge pool is innovative and well-motivated. It effectively abstracts away sensor heterogeneity and provides an interpretable interface for vehicles.
- The paper provides thorough experiments and ablation studies. The reported performance is strong.
- The methodology is described in detail with clear components for knowledge translation, pool construction (static/dynamic), and RAG-based retrieval.

### Weaknesses
- The paper's core motivation is that single-vehicle perception is limited and occluded, leading drivers/AI to blindly trust incomplete information and cause accidents. However, the proposed solution replaces one form of blind trust with another. It shifts trust from the vehicle's own sensors to the centralized V2X system. For example,  a person can maliciously place fake "construction ahead" sign that could fool the RSU's vision model, polluting the entire Knowledge Pool. This would cause all connected vehicles in the area to perform unnecessary and potentially dangerous maneuvers.

- The fusion process D = LM(Fuse(V, E)) is a black box. The paper does not test or explain how the system behaves when the local perception V (e.g., "I see a clear road") directly conflicts with the V2X knowledge E (e.g., "Infrastructure reports an obstacle"). 

- The proposed system hinges on a massive deployment of computationally powerful RSUs across entire cities.

### Questions
- How does the Fuse(V, E) function and the subsequent LLM reasoning handle fundamental conflicts between local perception V and infrastructure knowledge E? 

- How does the whole system handle adversarial attacks to the V2X system?

- Could you discuss the estimated cost and infrastructure requirements for deploying V2X-UniPool at a city scale?

### Soundness
3

### Presentation
3

### Contribution
2
