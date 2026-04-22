# Prompt-SSLC: A Unified Framework for Dual Prompt-Augmented Semi-Supervised Sequential Leader Clustering in On-the-Fly Category Discovery

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
On-the-fly Category Discovery (OCD) enables intelligent systems to perform real-time predictions while adapting to emerging categories in dynamic environments. We present Prompt-SSLC, a unified framework that integrates three synergistic components to balance stability and adaptability in streaming data scenarios. First, Semi-Supervised Sequential Leader Clustering (SSLC) dynamically updates prototypes to accommodate incoming data streams, ensuring flexibility in clustering. To enhance discriminability and mitigate prototype overlap, SSLC incorporates a Distance-Aware (DA) update mechanism that optimizes prototype distributions, maintaining inter-class separation as new data arrive. Second, dual prompting augments the foundation model: a *Task* Prompt guides category discovery, while an *Instance* Prompt dynamically recalibrates features to prevent drift toward previously learned classes without requiring retraining. Third, an Open-Set-Aware (OSA) classifier employs uncertainty estimation to identify and filter ambiguous samples, ensuring robust prototype updates. This cohesive integration of streaming clustering, feature recalibration, and uncertainty-aware filtering establishes a robust framework for OCD. Extensive experiments on generic and fine-grained benchmarks demonstrate that Prompt-SSLC achieves significant performance improvements, setting a new state-of-the-art for OCD.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper addresses the problem of On-the-fly Category Discovery (OCD), the task of continuously identifying new or emerging categories in data streams without halting or retraining the model. This is highly relevant in real-world dynamic environments where data distributions evolve over time (e.g., robotics, surveillance, or autonomous systems). The authors propose Prompt-SSLC, a unified framework combining semi-supervised clustering, prompt-based adaptation, and uncertainty-aware classification to achieve both stability (avoiding forgetting old categories) and adaptability (incorporating new ones).

### Strengths
- Combines three crucial paradigms: semi-supervised clustering, prompt-based adaptation, and uncertainty modeling into a unified pipeline for real-time category discovery.
- SSLC supports efficient updates on streaming data without requiring retraining, which is essential for on-the-fly operation.
- Using both task-level and instance-level prompts to guide discovery and recalibration is a creative extension of prompting to continual and open-set learning contexts.

### Weaknesses
- While SSLC and DA are conceptually sound, the paper lacks formal convergence or stability proofs for the streaming clustering process under non-stationary data.

-It is unclear how much each component (SSLC, dual prompting, OSA) individually contributes to performance improvements. A detailed ablation study would strengthen the empirical claims.

- The framework is demonstrated on visual benchmarks, but its adaptability to multimodal OCD and downstreaming tasks are not explored.

### Questions
1. Did the authors test the solution with the recent continual learning based solutions? 
2. How the solution be sensitive regarding the number of classes/clusters ? 
3. How are the Task Prompt and Instance Prompt jointly optimized during streaming updates?
4. How does the Distance-Aware (DA) mechanism handle catastrophic drift when the data distribution shifts significantly? 
5. The Open-Set-Aware (OSA) classifier relies on uncertainty estimation to filter ambiguous samples. Can you clarify what uncertainty metric is used (e.g., entropy, energy, or Bayesian variance), and how sensitive the prototype updates are to the chosen threshold in continuous data streams? 
6. Can the authors demonstrate the convergence of DA metric?

### Soundness
3

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
4

### Summary
The paper tackles On-the-fly Category Discovery (OCD) and proposes Prompt-SSLC, which combines (i) a semi-supervised sequential leader clustering algorithm with a distance-aware update, (ii) dual prompts (task + instance) plugged into a ViT backbone, and (iii) an open-set-aware classifier using MSP to route known vs. unknown samples. Experiments on both coarse- and fine-grained datasets show consistent improvements over existing OCD baselines.

### Strengths
Overall, the paper is well written and easy to follow. It addresses a timely and underexplored problem setting of truly streaming discovery without retraining at inference, which is interesting. The method is simple and modular, with each component (SSLC, dual prompts, OSA) serving an intuitive role, and the ablation studies show complementary and additive benefits.

### Weaknesses
- The novelty appears incremental. The method mainly combines existing components such as SLC, prompts, and MSP through heuristic coupling, with limited theoretical justification or analysis regarding the rotation update and its convergence or stability.
- The paper introduces a modified/constrained Hungarian matching and prototype-based assignment for old classes; this departs from standard GCD practice and may advantage the proposed approach. A clearer motivation and comparison with the standard protocol are needed.
- Thresholding and hyperparameters are not well specified. The method’s behavior depends heavily on parameters such as τ for the new-cluster radius, the rotation coefficient k, the top-k value for P_inst, and the OSA threshold (for example, the 95th percentile rule). The selection process and cross-dataset tuning are unclear, which raises concerns about fairness and reproducibility.

### Questions
- How is the rotation coefficient k chosen across datasets? Is there an adaptive scheme tied to local density or inter-prototype distances?
- What is the time/space complexity as the prototype pool grows (both for SSLC queries and constructing P_inst with top-k neighbors)? Any pruning/merging strategy?
- Please clarify τ selection and whether any validation on unlabeled data was used (risk of leakage).

### Soundness
3

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
The paper proposes Prompt-SSLC, a unified framework for the open-world problem of On-the-Fly Category Discovery where models must recognize known classes while dynamically discovering new ones in streaming data without retraining. The approach integrates Semi-Supervised Sequential Leader Clustering with a Distance-Aware prototype update, dual prompt learning to adapt vision foundation models, and an Open-Set-Aware classifier to detect unknown categories. Extensive experiments on both generic and fine-grained datasets show that Prompt-SSLC achieves soda performance, outperforming existing methods by a significant margin while maintaining efficiency and adaptability.

### Strengths
1. Unified solution to OCD: It proposes Prompt-SSLC, an uncommon and well-targeted combination of semi-supervised online clustering, dual prompting over a foundation ViT, and open-set routing addressing a sparsely explored yet practical setting (on-the-fly category discovery without retraining).
2. Dual prompting for OCD: It uses a Task Prompt (with partial label masking to simulate GCD) and an Instance Prompt (built from nearest prototypes) to adapt features on the fly without retraining.
3. Figures clearly convey the problem, the overall pipeline, and the effect of DA updates. Algorithm 1 clearly explains the streaming update logic.

### Weaknesses
1. It assigns the instance to the nearest prototype if its distance falls below a predefined radius (\tau). I wonder how to set this threshold and there is no analysis about it in the paper. Moreover, I think each class can have a different radius to represent the class boundary, but the authors set the shared, fixed threshold.

2. There are various semi-supervised clustering methods, but there is no analysis. Can you explain why SSLC is mostly suitable for OCD? Please present the results and rationale of it.

3. In the DA algorithm, is it enough to consider only the second-nearest prototype? I want to see its analysis or the visualization results like t-SNE.

4. Novelty of dual prompting: the idea of instance-aware and task-aware dual prompting is already proposed in DualPrompt [ECCV'22]. Can you compare the proposed method with DualPrompt?

5. Computation cost and scalability issues: Instance Prompt needs two passes and k-NN over a growing prototype pool is required. Please compare the cost and scalability analyses with baselines.

### Questions
1. Hyperparameter stability: How is the rotation strength “k” chosen, and how sensitive is the method to it?

2. Compute and scalability: Since the Instance Prompt requires two forward passes and nearest-prototype lookups, can you include throughput, FLOPs or etc?

3. (minor) Please proofread the paper, e.g., Open-Set-Aware (OSA) in line 177.

I am open to increasing the score if the authors answer the weaknesses and questions thoroughly.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper focuses on the challenge of novel class detection while dynamically discovering new categories in a data stream without retraining. The introduced prototype-based semi-supervised online clustering incrementally clusters incoming samples using “prototypes” with prompts aligning the model with discovery and feature refinements. Here, the prototypes are updated in a distance-aware methodology, which is the key contribution. The classifier, which is open-set aware, detects class instances and helps in orchestrating novel class identification. Empirical evaluation show significant improvements over existing on-the-fly detection approaches.

### Strengths
- The proposed framework integrates online prototype clustering (SSLC), prompt-based adaptation, and open-set routing. The distance-aware prototype updates and dual prompting strategy offers a lightweight solution for continual adaptation, avoiding catastrophic forgetting and heavy retraining.
- The paper tackles a challenging and underexplored setting where models must recognize known classes and dynamically create new ones from a continuous data stream without retraining.
- Evaluation is thorough with ablation studies, sensitivity analysis, and comparisons. These strongly validate the effectiveness

### Weaknesses
- While the paper focuses on on-the-fly category discovery, the conceptual boundary between OCD and related paradigms such as Novel Class Discovery and Generalized Category Discovery is not clearly articulated. The authors could better emphasize what specific challenges OCD introduces (e.g., strict online constraint, no replay, no retraining) and how these differ from batch-based novelty detection. This would strengthen the motivation and contribution.
- Since the proposed methods have several complementary components, it seems to make the system more complex and parameter-sensitive. The paper could be improved by providing a more systematic justification or ablation of the design choices. For example, consider why dual prompts outperform other PEFT methods, how distance thresholds are tuned. Furthermore, analyzing computational or latency cost for true online deployment would be useful.
- Although results are strong, the main metric used is accuracy gains. It would be better to add additional metrics  to measure time efficiency, memory usage, robustness to domain shift, or qualitative visualizations of evolving prototypes, thereby having the results more interpretable. Furthermore, it would also be useful to have error analysis, particularly when classes are incorrectly identified.

### Questions
- The paper uses shuffled static datasets. These datasets shown in Table 1, seem to be daily balanced between old and new classes. Wouldn’t it be more practical to have an imbalance dataset? Moreover, what would be the effect on accuracy if there are such class imbalances?
- In table 2, a few results between Prompt-SSLC and SSLC are very close. Are there results statistically significant? What are its confidence intervals?
- How is the proposed dynamic distance mechanism different from metric learning used in data stream research? 
For example: 
       - Lima, M., Neto, M., Silva Filho, T., & Fagundes, R. A. D. A. (2022). Learning under concept drift for regression—a systematic literature review. IEEE Access, 10, 45410-45429.
       - Kummert, J., Schulz, A., & Hammer, B. (2023). Metric Learning with Self-Adjusting Memory for Explaining Feature Drift. SN Computer Science, 4(4), 376.

### Soundness
3

### Presentation
3

### Contribution
3
