# Decentralized Manufacturing Management Based on Federated Learning with Stacking Ensemble

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
We propose a new intelligent management system to overcome the limitations of privacy, security, communication efficiency, and real-time analysis of data generated in smart manufacturing environments. As the digital transformation of the manufacturing industry accelerates, the importance of data utilization has grown, but the existing centralized approach involves data leakage risk and network load issues. To overcome these limitations, we propose a three-layer federated learning architecture consisting of cloud–anchor–edge. In particular, the anchor layer applies a stacking ensemble technique that combines predictions from multiple models to accurately identify complex anomaly patterns that are difficult to detect with a single model and maximize the robustness of model predictions. Compared to the accuracy of 0.5585 achieved by a single 1D-CNN model, the model applying stacking to federated learning significantly improved performance to an accuracy of 0.7438. Furthermore, to address the continuously changing data distributions in manufacturing environments, we propose a data distribution change detection and edge reallocation mechanism to enhance system flexibility and adaptability. The proposed system demonstrates significantly faster inference times than centralized learning models, presenting it a powerful alternative that ensures data privacy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a Decentralized Manufacturing Management System (DMMS) that addresses data privacy, communication efficiency, and anomaly detection challenges in smart manufacturing. The system employs a three-layer federated learning architecture (cloud-anchor-edge) where edge devices train models locally, anchor nodes aggregate and apply stacking ensemble techniques, and the cloud coordinates overall operations. The key innovation is combining multiple anchor models trained via federated learning (FedProx) with a meta-learner (XGBoost) to achieve better anomaly detection accuracy. Additionally, the system includes an adaptive mechanism using Wasserstein distance to detect data distribution shifts and reallocate edges to optimal anchors. Experiments on AI-Hub partial discharge data demonstrate that FL+Stacking achieves 0.7438 accuracy (33% improvement over standard 1D-CNN FL at 0.5585) with 6.3× faster inference than centralized Random Forest (2.3882ms vs. 15.1691ms).

### Strengths
1. The problem of addressing real manufacturing constraints itself is essential.
2. Comprehensive experimental evaluation with strong empirical results

### Weaknesses
1. Severely limited evaluation scope undermines generalization claims: The paper evaluates only on a single dataset (AI-Hub partial discharge data) from one specific manufacturing scenario (electrical equipment anomaly detection). No validation is provided for other manufacturing tasks mentioned in the introduction (vibration analysis, temperature monitoring, pressure anomalies, predictive maintenance). This raises serious concerns about whether the proposed architecture and stacking approach generalize beyond this specific use case. The authors acknowledge this limitation in the conclusion but do not sufficiently justify why readers should believe the approach will work elsewhere. Additionally, the dataset has only 9 edges and 5 classes—scalability to larger, more realistic deployments remains unproven.

2. Insufficient justification and analysis of key design choices: Critical design decisions lack proper justification or ablation studies: (1) Why use 5 anchors specifically? No ablation on varying this number. (2) Why XGBoost as meta-learner? No comparison with alternatives like neural networks or other gradient boosting methods. (3) The Wasserstein distance threshold of 0.1 appears arbitrary without sensitivity analysis. (4) All anchor models use the same 1D-CNN architecture, which may limit diversity benefits in stacking—why not heterogeneous base learners? (5) The frozen encoder mechanism is poorly explained: when/how is it trained, why keep it frozen vs. adaptive, and what's the performance impact? These gaps make it difficult to understand what truly drives performance and how to configure the system for new deployments.

3. Missing critical cost and overhead analyses: While inference time is reported (2.3882ms), the paper omits crucial practical considerations: (1) Training time comparison between centralized and federated approaches—how much longer does FL+Stacking take to converge? (2) Communication overhead during aggregation and edge reallocation—the SYN-ACK-ACK protocol adds rounds of communication but costs are not quantified. (3) Memory requirements for storing multiple anchor models and frozen encoders on resource-constrained edge devices. (4) Frequency and computational cost of Wasserstein distance calculations for distribution shift detection. Without these analyses, practitioners cannot assess the true deployment costs. The claim of "real-time" performance needs more rigorous validation including end-to-end latency measurements.

### Questions
1. How does the stacking ensemble provide benefits when all base learners use identical 1D-CNN architectures? 
2. What are the communication costs and failure handling mechanisms in your system?
3. Can you provide evidence that your approach generalizes beyond partial discharge detection?

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
4

### Summary
The paper addresses the evolving landscape of smart manufacturing where real-time data analysis is crucial for innovations like quality control and predictive maintenance, yet sensitive data raises privacy and security concerns under regulations such as GDPR. Motivated by the drawbacks of centralized data processing, including vulnerability to breaches and high network loads, the authors aim to develop a decentralized system that balances data utility with protection. Key challenges include ensuring privacy and real-time performance in distributed environments while adapting to dynamic data distributions and complex anomalies. The proposed Decentralized Manufacturing Management System (DMMS) employs a three-layer federated learning architecture of cloud-anchor-edge, with the anchor layer using stacking ensembles to enhance anomaly detection accuracy from 0.5585 to 0.7438, and incorporates a Wasserstein distance-based mechanism for detecting distribution shifts and reallocating edges for adaptability.

### Strengths
1. The hierarchical structure distributes computational tasks effectively across layers. This allows edges to handle local training without data exposure while anchors aggregate models for specialized learning. Such design not only preserves privacy but also scales seamlessly with manufacturing expansions.

2. Stacking ensembles integrate diverse model predictions at the anchor level. They capture subtle anomaly patterns that single models overlook, leading to robust detection. Performance metrics demonstrate a 33% accuracy improvement over baseline federated learning approaches.

3. The adaptation mechanism monitors data shifts using Wasserstein distance. It triggers edge reallocation through communication protocols like SYN-ACK, maintaining model relevance. This ensures sustained effectiveness in dynamic industrial settings where equipment changes occur frequently.

### Weaknesses
1. Stacking ensembles introduce additional computational overhead at the anchor layer. This could strain resources in large-scale deployments with numerous edges. Optimization strategies might be needed to mitigate increased complexity.
 
2. The fixed Wasserstein distance threshold of 0.1 lacks justification. It may not adapt well to varying data environments. Empirical tuning across scenarios would improve reliability.
 
3. Assumption of anchors specializing in specific defects overlooks potential overlaps. This could lead to suboptimal reallocation during shifts. More flexible specialization mechanisms might enhance adaptability.
 
4. Experiments simulate environments without real-world deployment. Practical factors like network delays are underrepresented. Field trials would reveal unforeseen implementation challenges.
 
5. The experiment methods, models, datasets are outdated. Most of FL algorithms utilize Deep neural networks for evaluation.

6. Almost no baseline methods from related works are compared.

### Questions
See weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a three-tier cloud–anchor–edge federated learning (FL) system for anomaly detection in smart manufacturing. Edges train 1D-CNN models locally to preserve privacy; anchors aggregate client updates and then perform a stacking ensemble across multiple anchor models

### Strengths
Using a meta-learner over anchor predictions is a simple, effective way to leverage client heterogeneity

### Weaknesses
Hierarchical FL and server-side ensembles (including stacking) have prior art; the paper would benefit from a sharper positioning of what is new beyond combining them in this domain

The paper technical contribution is rather limited

Results are on a single dataset with heavy feature condensation (per-channel statistics), which weakens claims about real-time sequence modeling

The system is motivated for real factories, but deployment evidence is limited to a lab setup. An applied paper like this one may benefit from stronger empirical results

### Questions
Per-edge results show large variance.  Can you add personalized FL baselines or anchor-specialized adaptation to show improvements where stacking under-performs?


Any additional industrial time-series (vibration/temperature/pressure) or public datasets (e.g., MIMII, PUMP, NASA bearing) to validate broader utility?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
To address the challenges of privacy protection and dynamic data variation in large-scale distributed manufacturing, this paper proposes a Decentralized Manufacturing Management System (DMMS). By introducing a three-tier Cloud–Anchor–Edge architecture and integrating a stacking ensemble technique, the system aims to achieve three main objectives: data privacy preservation, high-precision anomaly detection, and dynamic adaptability.

### Strengths
1. The targeted problem and scenario are practical and highly relevant to real-world industrial applications. The proposed hierarchical Cloud–Anchor–Edge structure demonstrates a clear architectural innovation.
﻿
2. The experimental results are clearly presented with comprehensive evaluation metrics, providing an intuitive understanding of the system’s performance.

### Weaknesses
1. The paper lacks sufficient theoretical innovation or mathematical derivation. The proposed three-tier architecture and stacking ensemble appear to combine existing FL concepts without providing a formal convergence analysis or theoretical justification.
﻿
2. The experimental evaluation is limited to a single dataset, and the baselines do not include comparisons with existing hierarchical or clustered FL methods. This makes it difficult to evaluate the advancement of the proposed approach over established ones.
﻿
3. The paper lacks ablation studies to thoroughly investigate the contribution of each key component (e.g., the Cloud–Anchor–Edge structure, the stacking ensemble, and the Wasserstein-based adaptation mechanism). Such studies are essential to validate the effectiveness of each module.

### Questions
1. The related work section does not clearly highlight how this paper advances beyond existing approaches. Please provide a more detailed discussion of prior works and explicitly clarify the novelty and unique contributions of this study.
﻿
2. Could the authors conduct an ablation study to quantify the contribution of each major component (the stacking ensemble, the Wasserstein-based adaptation mechanism, and the anchor layer)?
﻿
3. The experiments currently focus on different model types but do not include comparisons with existing hierarchical or clustered FL approaches. Adding at least three representative baselines would better demonstrate the effectiveness and innovation of the proposed method.

### Soundness
1

### Presentation
2

### Contribution
2
