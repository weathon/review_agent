# Leveraging Prediction Inconsistency for Online Error Detection in Procedural Videos

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
An efficient and accurate system for detecting errors in procedural tasks is crucial for supporting human needs in daily life, as it can provide instant notifications and guide people to correct mistakes. In this paper, we address the challenge of real-time and online error detection in procedural task videos by leveraging inconsistencies in action detector predictions. We propose a DUal-Branch Action Detector (DUBAD) framework, which integrates both \textit{robust} and \textit{sensitive} actions detectors. The \textit{robust} action detectors generate accurate and stable action predictions, while the \textit{sensitive} detectors produce inconsistent predictions when errors occur. To achieve this, we design a temporal-aware dynamic weight module that enhances sensitivity to errors using affine transformations with input-dependent, constrained weights and biases. Furthermore, we train the action detectors with varying amounts of temporal information to amplify inconsistencies in prediction when action sequences deviate from the correct order. For videos containing multiple or diverse errors, we apply a majority voting scheme based on mismatches between robust and sensitive predictions. Extensive experiments on EgoPER, Assembly-101-O, and EPIC-Tent-O demonstrate that our method outperforms state-of-the-art approaches in online error detection, while maintaining real-time efficiency with a lightweight architecture.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduced DUal-Branch Action Detector (DUBAD) for online error detection in procedural videos. The system is designed to work with egocentric videos from wearable devices and can detect errors as they occur. DUBAD combines a robust action detector and a sensitive action detector. The robust action detector generates accurate action prediction and the sensitive action detector generates inconsistent predictions when errors occur. The proposed method was evaluated on EgoPER, Assembly-101-O and EPIC-Tent-O. The state-of-the-art performance demonstrates its effectiveness.

### Strengths
1. The proposed DUal-Branch Action Detector (DUBAD) combines robust and sensitive action detectors, trained with differing temporal receptive fields, to capture distinct error types. This dual design is an original contribution in the online error detection domain.
2. The method conceptualizes prediction disagreement as an implicit measure of uncertainty, which is an innovative way of bridging temporal modeling and online reliability estimation.
3. This paper introduces two modules — the Step Attention Module (SAM) for robust detection and the Temporal-Aware Dynamic (TAD) module for error-sensitive adaptation — represents a meaningful architectural advancement that extends beyond a simple ensemble strategy.

### Weaknesses
1. Training only on error-free videos (one-class classification) limits what the model can learn. The method requires dataset-specific tuning of multiple hyperparameters (s, l, β, m)
2. While achieving 24.4 FPS, this is slower than DTGL (37.0 FPS) despite similar lightweight goals. And feature extraction time (0.0261s) not counted as part of their method but is required for deployment
3. Limited explanation for why dual branches with different temporal receptive fields is the optimal design
4. Related work part is poorly written, missing majority of work in the field such as: 
[1] Xu, Mingze, et al. "Temporal recurrent networks for online action detection." Proceedings of the IEEE/CVF international conference on computer vision. 2019.
[2] Xu, Mingze, et al. "Long short-term transformer for online action detection." Advances in Neural Information Processing Systems 34 (2021): 1086-1099.
[3] Guo, Hongji, et al. "Uncertainty-based spatial-temporal attention for online action detection." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
etc.

### Questions
Different people can perform procedural action in different order, how to handle this when detection errors?

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
2

### Summary
This paper introduces DUBAD for online mistake detection in procedural videos. DUBAD addresses the limitations of existing methods that they cannot detect multiple errors in a video, or only focus on procedural errors while missing execution errors.

### Strengths
**Originality**:
- This paper is based on a clear motivation: online mistake detection needs to detect two types of errors and to perform detection continuously, rather than stopping after the first error is detected.

**Quality**:
- This paper is complete, from motivation to methodology to experiments.

**Clarity**:
- This paper clearly elaborates the limitations of existing methods and, based on this, proposes a targeted solution.

**Significance**:
- This paper addresses an important issue in online mistake detection and the experiments show that the proposed method achieves superior results.

### Weaknesses
- The most recent method used for comparison in the experiments was published in 2024. Is it possible to include newer methods in the comparison?
- I think one important reference [1] has been omitted, which also focuses on Procedural and Execution Mistakes. I hope the authors would discuss the difference of this paper's motivation from [1].

[1] Patsch, Constantin, et al. "MistSense: Versatile Online Detection of Procedural and Execution Mistakes." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

### Questions
- Given that many LLM-based models for online video understanding already exist (e.g., Videollm-online[2]), is it also possible to achieve real-time detection for LLM-based online mistake detection? If so, what is the necessity of continuing to use small models? If not, could the authors provide experimental data demonstrating that LLM-based methods have significantly higher latency than non-LLM methods?

[2] Chen, Joya, et al. "Videollm-online: Online video large language model for streaming video." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Soundness
3

### Presentation
4

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
The paper proposes DUBAD (DUal-Branch Action Detector), a framework for real-time online error detection in procedural videos. The key idea is to exploit prediction inconsistencies between robust and sensitive action detectors to identify both execution and procedural errors. The model combines two complementary modules — the Step Attention Module (SAM) for robustness and the Temporal-Aware Dynamic (TAD) module for sensitivity — each trained with different temporal receptive fields. The authors evaluate DUBAD on EgoPER, Assembly-101-O, and EPIC-Tent-O, showing that it achieves state-of-the-art F1 scores with real-time inference speed.

### Strengths
The proposed method achieves strong quantitative results across multiple benchmarks, outperforming prior online error detection methods such as PREGO and DTGL.

### Weaknesses
- Figure 1 is visually unclear (Too small texts!) and unintuitive; it is hard to follow how the different modules (SAM, TAD, robust/sensitive detectors) interact. The graphical elements are cluttered and do not effectively convey the main conceptual flow.

- The notation is overly complex and not well presented. For example, lowercase h denotes action prototypes, uppercase H denotes concatenated weight–bias vectors, and k appears both above the sigma operator and as a subscript to H in Equation (3), which creates confusion.

- The design philosophy of SAM and TAD is not clearly motivated or justified. They appear somewhat arbitrary, and it is difficult to grasp a coherent conceptual rationale for how these modules interact to produce “prediction inconsistency.”

- Table 3 (Ablation Study) uses a simple checkbox-style presentation that does not effectively demonstrate the causal contribution of each component. It is unclear whether the performance gains genuinely reflect the intended functionality of SAM and TAD or are due to confounding factors in training.

- Despite good performance numbers, the overall framework feels overly complicated, with numerous loss terms and modules that obscure interpretability. The complexity of the design risks overshadowing the conceptual clarity of the main contribution.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
1

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
The paper proposes DUBAD, a dual-branch online error-detection framework for procedural videos that explicitly leverages prediction inconsistency between a robust and a sensitive action detector, each trained with different temporal receptive fields. The robust branch uses a Step Attention Module (SAM) conditioned on action prototypes, while the sensitive branch adds a Temporal-Aware Dynamic (TAD) module to induce input-dependent affine transformations that amplify inconsistencies under errors. The system flags errors when ≥3 of 4 pairwise comparisons disagree. Experiments on EgoPER, Assembly-101-O, and EPIC-Tent-O show state-of-the-art F1 and real-time throughput with a lightweight architecture.

### Strengths
1.	The proposed DUBAD framework demonstrates good performance improvements over state-of-the-art baselines across three challenging and diverse datasets, validating the effectiveness of the proposed approach.
2.	The method is designed to be lightweight and efficient, achieving real-time processing speeds (24.4 FPS). This is an advantage for deployment on resource-constrained platforms, making it more practical than competing methods that rely on large, slow models.
3.	The paper provides a comprehensive set of ablation studies that systematically validate the contribution of each key component. This strengthens the claims and provides clear insight into what makes the framework effective.

### Weaknesses
1.	The framework's performance appears to be highly dependent on a number of key hyperparameters, particularly the short (s) and long (l) temporal window sizes, the TAD margin (β), and the causal filter window size (m). The optimal values for these parameters vary significantly across different datasets (as shown in Fig. 6 and Table 12), suggesting that extensive and careful tuning is required for new tasks. The lack of a principled method for selecting these parameters could limit the model's practical applicability.
2.	The error detection mechanism relies on a majority vote across four specific pairs of predictions. While the ablation study justifies the voting threshold, the choice of the four pairs themselves feels somewhat arbitrary. A clearer rationale for why these specific pairs are optimal for capturing inconsistencies would strengthen the paper's design principles.

### Questions
1.	Could you elaborate on the methodology for selecting the optimal values for the key hyperparameters (s, l, β, m)? Given their sensitivity, is there a more principled or automated approach to adapt the framework to a new dataset beyond an exhaustive grid search?
2.	Regarding the majority voting scheme, could you provide the rationale for selecting the specific four pairs? Have you analyzed the contribution of other potential pairs to the final detection performance?
3.	The model is trained only on "normal" (error-free) videos. How would the system handle novel but correct variations of a procedure that were not present in the training data? Is there a risk that such unseen but valid actions would be flagged as errors due to the induced prediction inconsistency?

### Soundness
2

### Presentation
3

### Contribution
2
