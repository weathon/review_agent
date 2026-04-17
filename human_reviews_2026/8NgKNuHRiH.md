# COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Cooperative perception enables autonomous agents to share encoded representations over wireless communication to enhance each other’s live situational awareness. However, the tension between the limited communication bandwidth and the rich sensor information hinders its practical deployment. Recent studies have explored selection strategies that share only a subset of features per frame while striving to keep the performance on par. Nevertheless, the bandwidth requirement still stresses current wireless technologies. To fundamentally ease the tension, we take a proactive approach, exploiting the temporal continuity to identify features that capture environment dynamics, while avoiding repetitive and redundant transmission of static information. By incorporating temporal awareness, agents are empowered to dynamically adapt the sharing quantity according to environment complexity. We instantiate this intuition into an adaptive selection framework, COOPERTRIM, which introduces a novel conformal temporal uncertainty metric to gauge feature relevance, and a data-driven mechanism to dynamically determine the sharing quantity. To evaluate COOPERTRIM, we take semantic segmentation and 3D detection as example tasks. Across multiple open-source cooperative segmentation and detection models, COOPERTRIM achieves up to 80.28% and 72.52% bandwidth reduction respectively while maintaining a comparable accuracy. Relative to other selection strategies, COOPERTRIM also improves IoU by as much as 45.54% with up to 72% less bandwidth. Combined with compression strategies, COOPERTRIM can further reduce bandwidth usage to as low as 1.46% without compromising IoU performance. Qualitative results show COOPERTRIM gracefully adapts to environmental dynamics, localization error, and communication latency, demonstrating flexibility and paving the way for real-world deployment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
# What the paper does
COOPERTRIM is an adaptive data-selection framework for cooperative perception that uses temporal uncertainty to decide what features to share and how much to share under bandwidth limits.
# Key idea
* Compute a conformal, temporal uncertainty signal by comparing current encoded features (F_t) with the previous fused features (F_{t-1}^{\text{fused}}); uncertainty indicates where collaboration helps most.
* Use an uncertainty-guided attention to score relevance per channel/region and apply adaptive thresholds to (i) select features and (ii) determine sharing quantity frame-by-frame.
* Train with an ε-greedy–inspired regimen to balance exploration/exploitation so the model learns robust selection under bandwidth constraints.

### Strengths
1. The temporally driven, uncertainty-aware communication scheme is conceptually clear and well structured: it measures discrepancies between the previous fused representation and the current features, applies conformal quantile thresholding to select candidates, and then uses attention with an adaptive mask cutoff to decide both what to transmit and how much—focusing bandwidth on high-value regions.
2. The $\epsilon$-greedy training schedule provides a practical stabilizer under bandwidth constraints: intermittent full-feature updates interleaved with predominantly masked updates smooth optimization and reduce variance, yielding stronger performance than standard-deviation–only uncertainty baselines and curriculum-style fine-tuning.
3. The method is readily portable: as a drop-in component for cooperative semantic segmentation backbones (e.g., CoBEVT, AttFuse, DiscoNet), it delivers consistent improvements at equal or lower communication budgets.

### Weaknesses
1.	Lack of comparison with asynchrony-robust methods (e.g., CoBEVFlow). While the task settings may differ, CoBEVFlow demonstrates that estimating BEV flow and propagating prior features can effectively counter temporal variation; this capability should be considered—either as a baseline or as a complementary design—when claiming advantages in time-varying scenes and realistic, asynchronous communications.
2.	Single-benchmark evaluation. Experiments are confined to OPV2V, which limits external validity. Broader evidence across datasets (e.g., DAIR-V2X, V2X-Sim, OPV2V-Async) and tasks beyond semantic segmentation would strengthen generality claims.
3.	“Conformal” is used primarily as quantile gating rather than as standard conformal prediction with finite-sample coverage guarantees. The paper lacks formal coverage analyses or mismatch bounds, so the terminology risks overstating the method’s theoretical assurances.
4.	Limited system- and communication-layer characterization. Reported metrics focus on bandwidth ratios/Mbps and IoU, with no measurements of end-to-end latency, packet loss/retransmissions, congestion behavior, or the computation/runtime overhead introduced by attention and masking under different hardware budgets. Deployment-level compute-communication trade-offs thus remain underexplored.

### Questions
1.	Benchmark against asynchrony-robust methods and/or integrate BEV flow.
Can you compare to CoBEVFlow or prepend a BEV-flow pre-alignment module, reporting IoU–bandwidth trade-offs under controlled time offsets (e.g., ±50/100/200 ms) on OPV2V/OPV2V-Async? Does your two-threshold policy still add gains beyond flow alone?
2.	Report system- and network-level metrics under realistic conditions.
Measure end-to-end latency (encode→select→transmit→align→fuse→decode), packet loss/retransmissions, and congestion behavior across link budgets (e.g., 3/6/12 Mbps) and loss rates (0–10%). Plot IoU–bandwidth–latency curves and characterize degradation/fallback under losses.
3.	Quantify computational overhead and deployment feasibility.
Detail added FLOPs/memory and per-frame latency from attention/masking on embedded automotive hardware (e.g., Jetson/SoC) and desktop GPUs. Compare compute-communication trade-offs against feature-compression/distillation baselines at equal accuracy.
4.	Establish cross-dataset and cross-task generalization.
Evaluate beyond OPV2V (e.g., DAIR-V2X, V2X-Sim, OPV2V-Async) and beyond semantic segmentation (detection/occupancy/tracking). Include fine-tuned and zero-shot transfers, reporting full IoU–bandwidth curves to substantiate external validity.

I would support acceptance provided the authors satisfactorily resolve all identified concerns.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an adaptive data selection framework called COOPERTRIM for cooperative perception in autonomous agents. The main idea is to exploit the temporal continuity of the environment to identify relevant features and avoid transmitting redundant or static information. This reduces the communication bandwidth required while maintaining comparable accuracy to existing selection strategies. The proposed framework uses a conformal temporal uncertainty metric to measure feature relevance and a data-driven mechanism to determine the amount of data shared. The evaluation shows significant bandwidth reduction and improved IoU compared to other selection strategies. However, there are some limitations, such as no ablation study on choosing optimal thresholds, only one simulated dataset was used, and the method has not been evaluated on real-world datasets. Additionally, the threshold-based method may not be robust in scenarios where only minor changes occur in the scene. The method is currently only evaluated on segmentation tasks and further evaluation on detection tasks would demonstrate its generalizability. Finally, the impact of the computation cost after introducing this data selection to collaboration perception models needs to be evaluated. Overall, the idea of using temporal uncertainty for data selection is interesting and the theoretical proof is sound, but further research is needed to address the limitations mentioned above.

### Strengths
1. The idea of data selection by using temporal uncertainty is interesting.
2. The theoretical proof is sound.
3. The reduction in communication bandwidth consumption in segmentation tasks is obvious.

### Weaknesses
1. No ablation studies on how to choose the optimal thresholds.
2. Only one simulated dataset is used; No real-world dataset is evaluated.
3. What is the mathematical expression of the distance function? What is the deep reason for using this distance function?
4. How robust is the threshold-based method? For example, in a certain scenario, maybe the overall scene doesn't change much, only a small object (e.g., a new pedestrian emerges), probably leading to a small temporal uncertainty, and how will the system take actions to this?
5. Although the method is advantageous for segmentation tasks, it should be better evaluated on detection tasks as well to demonstrate its generalizability. 
6. Apart from the bandwidth consumption, one important factor is how the computation costs change after introducing this data selection to the collaboration perception models. What is the processing speed (evaluated by FPS)? Can this method be used in a real-time driving system (Typically more than 100 FPS)?

### Questions
Please refer to the weaknesses.

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
4

### Summary
This paper presents CooperTrim, an adaptive, uncertainty-aware feature selection framework for cooperative perception. The framework leverages conformal prediction to estimate temporal uncertainty for feature relevance and employs a data-driven adaptive mechanism to select an appropriate quantity of shared features based on environmental complexity. CooperTrim is plugged in and extensively evaluated on semantic segmentation using several co-perception methods on the OPV2V dataset, demonstrating significant reductions in bandwidth usage without sacrificing accuracy.

### Strengths
**Compelling Motivation and Scope:** The paper focuses on the bandwidth-accuracy trade-offs in cooperative perception, arguing for temporally and contextually adaptive feature selection that is not covered by static or threshold-based approaches. 

**Effectiveness of Sub-modules:** CooperTrim's integration of conformal temporal uncertainty estimation with a cross-attention-based selection mechanism is well described, addressing both feature relevance and adaptivity. Detailed elaboration on training strategies provides hints on reproduction.

**Concrete Adaptivity Insights:** The claim of scene adaptivity is validated by the qualitative results given in Fig. 4.

### Weaknesses
**Major Weaknesses:**
1. Although more components are incorporated, the proposed method remains a threshold masking mechanism. The adaptivity claim needs more quantitative validation. For example, in Fig. 4 (left), a convincing result would be to show that the IoU curve is relatively stable, or at least more stable than the BW curve ("complexity" curve). The current result cannot prove that the adaptivity benefits the final results.
2. Robustness against localization error and latency is not discussed.

**Minor Weaknesses:**
1. The method is only tested on OPV2V, which is a relatively simple simulated dataset.
2. More network-efficient baselines are expected to be included. Presenting results for the object detection task can also help enhance comparisons with previous methods.

### Questions
1. Can the method adjust the bandwidth requirement during operation? Or a new model (a newly learned threshold generator) is needed to cope with a new network condition?
2. There will be a potential delayed response to a new traffic pattern as history information is used to generate the threshold. Are there any results on the influence of the temporal window size?
3. How well would CooperTrim's adaptivity generalize to other perception tasks (detection, tracking, etc.)? In those tasks, the ROI is much sparser.

See the weaknesses section for more details.

### Soundness
3

### Presentation
3

### Contribution
2
