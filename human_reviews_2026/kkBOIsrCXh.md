# Embodied Navigation Foundation Model

- Avg Score: 8.00
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 10

## Abstract
Navigation is a fundamental capability in embodied AI, representing the intelligence required to perceive and interact within physical environments. To achieve such intelligence, recent advanced works leverage Vision-Language Models (VLMs), which demonstrate strong generalizability and possess a well-suited formulation for navigation. However, these approaches remain largely confined to narrow task settings and embodiment-specific architectures. In this work, we introduce a cross-embodiment and cross-task Navigation Foundation Model (NavFoM), trained on eight million navigation samples that encompass quadrupeds, drones, wheeled robots, and vehicles, and spanning diverse tasks such as vision-and-language navigation, object searching, target tracking, and autonomous driving. NavFoM employs a unified architecture that processes multimodal navigation inputs from varying camera configurations and navigation horizons. To accommodate diverse camera setups and temporal horizons, NavFoM incorporates identifier tokens that embed camera view information of embodiments and the temporal context of tasks.  Furthermore, to meet the demands of real-world deployment, NavFoM controls all observation tokens using a dynamically adjusted sampling strategy under a limited token length budget. Extensive evaluations on seven public benchmarks demonstrate that our model achieves state-of-the-art or highly competitive performance across different navigation tasks and embodiments without requiring task-specific fine-tuning. Additional real-world experiments further confirm the strong generalizability and practical applicability of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a cross-task, cross-embodiment navigation foundation model that ingests egocentric multi-view video plus language and outputs waypoint trajectories, trained on 8M navigation samples spanning quadrupeds, drones, wheeled robots, and vehicles. 

Its key innovations are Temporal-Viewpoint Indicator tokens to encode camera view and time, and a Budget-Aware Temporal Sampling scheme that keeps recent context while respecting a token budget.

### Strengths
* The paper introduce Temporal-Viewpoint Indicator, which provide explicit view and time encoding so the LLM can align frames across cameras, and Budget-Aware Temporal Sampling preserves recent context under a fixed token budget.
* Across seven public benchmarks and additional real-world robot trials, NavFoM attains state-of-the-art or highly competitive results without task-specific fine-tuning.

### Weaknesses
* The model fine-tunes only a single epoch, this suggests possible under-optimization given the 12.7M sample corpus.
* Multi-task training uses a fixed loss weight 𝜙 = 10 to scale navigation MSE against QA loss, the paper doesn’t probe sensitivity to this choice.

### Questions
An ablation study on the loss weights can help validate the effectiveness of the fixed hyperparameters.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents NavFoM, a unified foundation model for cross-embodiment and cross-task embodied navigation. The model is designed to handle diverse camera setups (single or multi-view) and temporal contexts using two key mechanisms: Temporal–Viewpoint Indicator (TVI) tokens that encode spatiotemporal context, and Budget-Aware Temporal Sampling (BATS) that efficiently selects observation tokens under computational constraints.

Overall, this is a well-motivated and timely contribution that advances the goal of general-purpose embodied AI by unifying perception, language, and control within a single framework.

### Strengths
1. NavFoM is trained on a large-scale, diverse dataset encompassing multiple embodiments and navigation-related tasks. This scale enables effective transfer and cross-task generalisation.

2. The introduction of TVI tokens and the BATS sampling scheme are well-motivated, domain-specific innovations that address the unique challenges of temporal-spatial consistency and token efficiency in navigation settings.

3. By using a single backbone to handle instruction-following, object search, and driving tasks, the paper takes a meaningful step toward a generalist embodied navigation model. The paper reports competitive or state-of-the-art performance across multiple benchmarks and demonstrates successful deployment on real robots.

### Weaknesses
1. The real-world experiments rely on remote inference using an RTX-4090 GPU, as the model is too large to be hosted on typical on-board computing hardware. While this approach is understandable given current hardware limitations, it limits the practical deployability of the proposed system. Exploring model compression, distillation, or quantisation strategies could make NavFoM more suitable for real-time, on-board navigation scenarios. The reviewer acknowledges that such challenges are common in large foundation models but believes they warrant explicit discussion in the paper.
2. 
* The performance gains reported in Table 1 are often modest relative to baselines, and the paper prioritises the success rate (SR) metric in the main text while other metrics receive minimal attention. A clearer explanation of the relative importance and interpretation of these metrics would improve transparency. 
* Additionally, for metric-related discussion the readers are referred to the appendix. The appendix provides only a brief description of the metrics, mainly full-form expansions without equations or methodological details. Including formal definitions and clarifying what each metric measures would help readers better assess the quantitative improvements.
3. The paper does not explicitly discuss limitations, failure modes, or robustness issues, which is especially important for large foundation models deployed in real-world environments. Acknowledging potential failure cases (e.g., under sensor noise, unseen environments, or dynamic obstacles) and discussing safety implications would provide a more balanced and credible presentation of the work.
4. While the paper presents ablations on TVI and BATS components, the analysis remains somewhat surface-level. The interactions between these modules, their computational overhead, and their contribution to overall model complexity are not extensively studied. A more detailed examination (for example, isolating how each mechanism influences learning dynamics or token efficiency) would strengthen the methodological understanding and reproducibility of the work.

### Questions
1. Figure 3:Could the authors clarify whether Figure 3 is an illustration of the proposed TVI tokenization or an empirical plot generated from a specific task? If it represents actual performance data, additional details on the experimental setup and interpretation would be appreciated.
2. Fine-tuning: Have the authors evaluated NavFoM under a fine-tuning regime on at least one representative navigation task? Reporting such results would help assess how well the foundation model adapts to task-specific settings and whether it can surpass specialised policies after adaptation.
3. Clarification on data proportions in Figure 7: The description of “x% others + specific data” in Figure 7 is somewhat ambiguous. Could the authors specify the absolute scale of “other” versus “specific” data used to construct each mixture? Without this information, it is difficult to interpret the proportional relationships or judge whether performance improvements stem from data diversity or data volume.
4. Interactions between BATS and TVI: While Figure 8 presents independent ablations of the BATS and TVI components, it would be valuable to investigate their combined or hybrid variants (e.g., Linear Probability Sampling + Individually Learned Special Tokens). Have the authors explored whether these mechanisms exhibit complementary effects when jointly optimised?
5. Relative contribution of TVI and BATS: From the reported ablations, TVI appears to contribute more significantly to performance consistency: models including TVI often perform comparably well even when alternative sampling schemes replace BATS. Is this interpretation accurate, and can the authors comment on why TVI might play a more critical role in the unified framework?
6. Evaluation in Table 4: In the NavTest results, was NavFoM fine-tuned on the NavSim dataset, or are these results based on zero-shot transfer? Please clarify how the comparisons to baseline models were normalised to ensure an equitable evaluation (e.g., identical data exposure, observation viewpoints, training epochs, or evaluation protocols).
7. Latency and on-board scalability: Given that navigation tasks often require real-time control, could the authors provide quantitative information about inference latency and computational scaling? Insights into how NavFoM performs under constrained hardware or whether lightweight variants (e.g., distilled or quantised versions) are feasible would help evaluate its deployability in practical robotic systems.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
NavFoM adopts a unified architecture that can handle multimodal inputs from different camera configurations and time spans, making it suitable for tasks across various embodiment types, such as quadruped robots, drones, wheeled robots, and vehicles. These tasks include vision-language navigation, object search, object tracking, and autonomous driving. To address multi-view and temporal contexts, the model introduces Temporal-View Indication (TVI) tokens, which help the model understand and adapt to different camera perspectives and time steps. Through an innovative Budget-Aware Temporal Sampling (BATS) strategy, NavFoM efficiently manages navigation history information with limited computational resources, balancing performance and inference speed. Moreover, NavFoM achieves state-of-the-art performance on multiple public benchmarks and demonstrates strong generalization capabilities across various embodiments and tasks, particularly excelling without the need for task-specific fine-tuning. Through practical testing on platforms such as humanoid robots, quadruped robots, drones, and wheeled robots, the paper validates the model's potential for real-world applications, showing that NavFoM can effectively handle complex navigation tasks with broad practical applicability.

### Strengths
1.Unified Architecture: The model processes multimodal navigation inputs from different camera configurations and task horizons. It uses temporal-viewpoint indicator tokens (TVI tokens) to accommodate diverse camera setups and time contexts in navigation tasks. This allows the model to handle various visual inputs dynamically.

2.Budget-Aware Temporal Sampling (BATS): This method addresses real-world deployment challenges, such as memory and inference speed, by dynamically sampling navigation history based on a token budget. This ensures the model balances performance with resource constraints.

3.Generalization Across Tasks and Embodiments: NavFoM achieves state-of-the-art (SOTA) performance across seven public benchmarks and in real-world experiments without requiring task-specific fine-tuning. The model demonstrates significant improvements, particularly in multi-camera settings, and shows robust performance in diverse environments such as robotics, drones, and autonomous vehicles.

### Weaknesses
A large language model based approach might fail to handle highly dynamic environments.

### Questions
1. The output of the model is two-dimensional. How can we control the drone to navigate in a three-dimensional space?
2. Will using the latest model, such as Qwen3 VL, lead to performance improvement?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The primary contribution of this research is the design and training of a unified navigation foundation model (NavFoM). This model was trained on a dataset comprising eight million navigation samples, encompassing diverse robotic platforms and varied navigation tasks. To handle the varying camera configurations across platforms and the temporal span of tasks, NavFoM incorporates identifier tokens to embed both view information and temporal context. Furthermore, to accommodate computational resource constraints in practical deployments, the model employs a sampling strategy that dynamically adjusts observation tokens within a limited token length budget. Evaluation results demonstrate that NavFoM achieves leading or highly competitive performance across seven public benchmarks without requiring task-specific fine-tuning. Real-world robotic experiments further validate its robust generalization capabilities and practical applicability.

### Strengths
1. The paper introduces the core concept of a “cross-vehicle, cross-task navigation foundation model,” demonstrating high innovation and foresight. It addresses the limitation in current embodied AI navigation research where models are often confined to specific vehicles or tasks, pointing toward a promising direction for constructing universal navigation agents.
2.The paper's technical solution for unifying multi-vehicle, multi-task inputs—using identifier tokens to embed camera view information and task temporal context—is remarkably clever and direct. The dynamic token sampling strategy also demonstrates deep consideration for practical deployment constraints.
3.Extensive testing across seven public benchmarks and real-world scenarios covering diverse robots and tasks provides robust validation of NavFoModel's effectiveness and generalization capabilities, lending strong credibility to the results.
4.The research does not confine itself to idealized laboratory settings but explicitly addresses and tackles challenges in real-world deployments, such as computational efficiency and adaptability to dynamic environments, thereby enhancing the practical value of the work.
5.The author provided detailed implementation specifics and stated that the source code would be open-sourced.

### Weaknesses
1.The paper would be significantly strengthened by adding a dedicated analysis of the model's failure modes (e.g., in long-horizon planning or perceptually challenging environments), which is vital for defining the approach's boundaries and guiding future research.​​ 
2.The paper emphasizes the model's practicality and proposes the BATS strategy to optimize computational overhead, but lacks detailed, quantitative analysis of deployment costs.

### Questions
see weekness

### Soundness
4

### Presentation
4

### Contribution
4
