## Human Reviewer 1

### Summary
This paper presents SigMap, a prompt-based architecture for cross-scenario wireless localization that integrates masked autoencoding with geographic and topological maps serving as soft prompts. The model introduces a cycle-adaptive masking mechanism designed to align with the cyclic nature of Channel State Information (CSI) signals, thereby improving feature learning during pretraining. Evaluated within simulated DeepMIMO environments, SigMap demonstrates strong generalization capability and achieves parameter-efficient few-shot adaptation. The approach aims to bridge the gap between environment-specific training and scalable localization across diverse wireless scenarios.

### Strengths
(1) The idea of using maps as prompts is both innovative and practical. By embedding spatial priors directly into the learning framework, the model can better understand geographic context without requiring explicit supervision or heavy parameterization. This approach provides a lightweight yet effective way to integrate domain knowledge into data-driven models.

(2) The proposed cycle-adaptive masking strategy effectively leverages the inherent periodic and structural characteristics of CSI signals. This allows the pretraining process to focus on more informative segments of the data, improving robustness and representation quality, especially when dealing with noisy or incomplete measurements.

(3) The demonstration of few-shot adaptation using a frozen backbone is impressive, as it highlights the model’s ability to generalize with minimal retraining. This efficiency in adapting to new environments or conditions suggests that SigMap could serve as a versatile foundation for scalable wireless localization systems, reducing computational and data requirements during deployment.

### Weaknesses
(1) The absence of real-world evaluation limits the impact of the results. Without validation on empirical datasets or publicly available benchmarks such as CSI-Bench, it is difficult to assess how well the approach generalizes beyond simulation. This gap weakens the practical relevance of the presented findings.

(2) The paper’s claim of developing a “foundation model” for wireless localization appears overstated. While the architecture shows potential for generalization within simulated settings, it lacks evidence of robustness across devices, propagation environments, or hardware variations, all of which are critical for real-world applicability.

(3) Although the system integrates several established components—masked autoencoders, vision transformers, and graph-based prompting—the overall architectural contribution feels incremental. The novelty lies more in the combination and application context rather than in introducing fundamentally new mechanisms or model designs.

(4) The work asserts interpretability through the use of map prompts but does not provide supporting analysis. Visual or quantitative evaluation of how the prompts influence model predictions would strengthen the paper’s interpretability claims and offer deeper insights into model behavior.

(5) The scalability of the proposed approach remains uncertain. The paper does not explore how the framework performs when applied to large-scale or densely connected map graphs, which are common in real-world urban deployments. Understanding such scalability constraints is important for practical use in complex environments.

(6) While the paper relies on ray-tracing–based wireless simulation, this approach—though widely used—offers limited novelty unless extended with advanced modeling such as diffuse scattering, dynamic environments, or hybrid physics–ML calibration. The current setup would benefit from stronger validation or augmentation to better capture real-world propagation complexity.

### Questions
Can the authors report scalability experiments by evaluating SigMap on larger or denser map graphs, or by simulating more complex urban propagation conditions, to objectively assess how the method performs in real-world large-scale deployments and justify its practical robustness?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
The paper proposes SigMap, a multimodal foundation-model framework for wireless localization featuring (1) a periodicity-aware adaptive masking pretraining scheme tailored to CSI, and (2) a “map-as-prompt” mechanism that encodes 3D maps as geometric prompts for parameter-efficient finetuning. Experiments on DeepMIMO (O1-3p5) show gains for single/multi-BS localization and some few-shot cross-scenario transfer.

### Strengths
I think how the authors use GNN to generate the Prompt  is a great innovation. It cleverly borrows the idea of Prompt-Tuning from LLMs, encoding 3D map information into lightweight soft prompts used to guide a large signal foundation model. This fundamentally solves the problem of model adaptation in new environments.

### Weaknesses
Weakness
1.	The introduction has too many paragraphs, although it compares with many existing works, the logic is not clear. It cannot effectively introduce the work done in this article from existing works. In addition, the shortcomings of many existing studies, such as the inability to capture high-dimensional features for description, are not sufficient to demonstrate the inadequacy of these work.
2.	The experimental setup is relatively single and the validation depth is insufficient. Evidence is mostly from a single ray-tracing world, and there is only one cross scenario experiment. The existing experiments are difficult to fully demonstrate the universal applicability of the proposed model. Apart from error metrics, are there any other experiments that demonstrate the effectiveness of the proposed model?
3.	How to better reflect the mentioned advantages limited labeled samples, efficient parameters, interpretability? For example, the model proposed parameters efficient, but the comparison of training time, memory usage, inference complexity is insufficient.
4.	Insufficient ablation experiments.

### Questions
All of the paper's training and testing are based on DeepMIMO, a simulation dataset. It lacks validation on data collected in the real world.
Real-world signals are filled with noise, dynamic interference, and complex propagation effects that simulators cannot fully replicate. It is a significant unknown whether the clean physical laws learned from the simulator can maintain high performance in a dirty real-world environment.
the current "Map-as-prompt" primarily encodes the environment's geometry by processing 3D coordinates  with a GNN. It does not encode material information. The model doesn't know if it's facing a concrete wall that absorbs"signals or a glass curtain wall that reflects them.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper proposes SIGMAP, a transformer backbone pre-trained with cycle-adaptive masked modeling on Channel State Information, then fine-tuned with a learned geographic prompt from a 3D map via a GNN. The paper claims three main contributions: (1) cycle-adaptive masking to break periodic shortcuts in CSI; (2) map-as-prompt conditioning using 3D geometry; (3) parameter-efficient adaptation with strong cross-scenario generalization.

The experiments demonstrate substantial improvements over other baselines, on both single- and Multi-BS localization, as well as generalization performance.

### Strengths
1. The self-adaptive masking and GNN map-as-prompt strategies are novel and meaningful combinations for indoor localization task. The experimental results show significant advantages over other baselines.

2. During fine-tuning, only prompt GNN and projection head are trained, while the backbone is kept frozen. This makes the model efficient and handy for deployment.

3. The algorithm achieves consistent metric gains in different tasks. And the improvements are substantial.

### Weaknesses
1. The paper asserts good generalization abilities, but it’s not intuitively clear why the algorithm achieves this. The model isn’t trained using meta-learning or transfer learning techniques. The paper also lacks of experimental comparisons to modern baselines that target at generalization in indoor localization, e.g., [1].

2. The paper doesn't mention how the quality or degradation of the 3D Map could adversely affect the performance of the model. Illustrations of the 3D Map used are needed. More ablation studies on the qualities of the 3D Map are desirable.


[1] Gao, Jun, et al. "MetaLoc: Learning to learn wireless localization." IEEE Journal on Selected Areas in Communications 41.12 (2023): 3831-3847.

### Questions
1. Could the authors explain why the model achieves good generalization abilities to new environments? Since the algorithm is not trained using meta-learning or transfer learning, I am curious about how the model learns to generalize.

2. Could the authors give an example of the 3D Map used in the paper? Can the authors discuss how the quality of the 3D map would affect the model’s performance?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3