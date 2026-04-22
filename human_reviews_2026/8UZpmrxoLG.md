# Astra: General Interactive World Model with Autoregressive Denoising

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Recent advances in diffusion transformers have empowered video generation models to generate high-quality video clips from texts or images. However, world models with the ability to predict long-horizon futures from past observations and actions remain underexplored, especially for general-purpose scenarios and various forms of actions. To bridge this gap, we introduce Astra, an interactive general world model that generates real-world futures for diverse scenarios (e.g., autonomous driving, robot grasping) with precise action interactions (e.g., camera motion, robot action). We propose an autoregressive denoising architecture and use temporal causal attention to aggregate past observations and support streaming outputs. We use a noise-augmented history memory to avoid over-reliance on past frames to balance responsiveness with temporal coherence. For precise action control, we introduce an action-aware adapter that directly injects action signals into the denoising process. We further develop a mixture of action experts that dynamically route heterogeneous action modalities, enhancing versatility across diverse real-world tasks such as exploration, manipulation, and camera control. Astra achieves interactive, consistent, and general long-term video prediction and supports various forms of interactions. Experiments across multiple datasets demonstrate the improvements of Astra in fidelity, long-range prediction, and action alignment over existing state-of-the-art world models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Astra, a new framework for building a general-purpose, interactive world model. The central problem it addresses is that existing video generation models (like diffusion transformers) can create high-fidelity clips but lack true interactivity—they cannot generate long, coherent videos that dynamically and precisely respond to external user actions (e.g., camera controls, robot actions, or vehicle movements).

### Strengths
The paper's primary strength lies in its proposal of a Astra framework, which skillfully combines the powerful generative capabilities of pre-trained diffusion models with an autoregressive, action-conditioned paradigm, effectively bridging the gap between high-fidelity video generation and real-time interactivity. The authors' core contributions are embodied in three innovations: (1) A lightweight ACT-Adapter for efficiently injecting action signals into the pre-trained model while preserving its knowledge; (2) An innovative "noise-augmented memory" strategy to overcome the "visual inertia" problem, forcing the model to prioritize responding to actions rather than simply repeating historical frames; (3) The introduction of a Mixture of Action Experts (MoAE) module, which enables flexible handling of heterogeneous action inputs from different domains, enhancing the model's generality. Experimental results validate the effectiveness of this design, with the model performing exceptionally well on the key "Instruction Following" metric.

### Weaknesses
**W1** The authors constructed a new benchmark, Astra-Bench, for evaluation. According to the paper, this benchmark is "comprising 20 held-out samples from each dataset". This scale is extremely small and likely insufficient to robustly evaluate the model's generalization capabilities, which could lead to biased evaluation results. 
**W2** The paper's title claims it is a "General Interactive World Model". It is questionable whether this is sufficient to support such a broad "general" claim. Out-of-domain scenarios with different physics or interaction types (e.g., fluid dynamics, complex object stacking, multi-agent interaction) are needed.
**W3** Can interaction modalities from different domains all be mapped to control signals of the same dimension? For example, the degrees of freedom (DoF) of an embodied agent are far greater than the degrees of freedom of autonomous driving.
**W4** The first embodied agent interaction in the appendix not only has blurry artifacts, but also exhibits incorrect affordances and false interactions, which are challenges that remain to be addressed.

Typos:
There is a clear spelling error in the title of Appendix A: "A MORE EPERIMENTAL DETAILS".

### Questions
See in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses a key limitation in current video generation and world modeling approaches which are lack of interactivity and long-horizon consistency. While diffusion-based models are able to generate high-fidelity videos from text or images, they often produce short, self-contained clips, fail to respond dynamically to user actions or control signals, and struggle with error accumulation in long rollouts. The paper addresses this by building a general-purpose, interactive world model that can simulate realistic futures across diverse domains (e.g., driving, robotics, exploration) while maintaining responsiveness to actions and temporal coherence. It proposes a lightweight module named as ACT-Adapter that injects action signals directly into the latent space of a pre-trained video diffusion backbone, a training strategy of noise-augmented history memory training which corrupts historical frames to reduce over-reliance on visual context and improve responsiveness., and mixture of action experts that handles multiple action modalities of camera, robot pose, and keyboard/mouse. Astra-bench is the benchmark suite used for evaluation which spans across multiple datasets to evaluate the visual quality and instruction-following performance.

### Strengths
1. The paper addresses the limitation of passive video generation by showing interactive world modeling where video synthesis is conditioned on external actions .
2. The framework proposes a single, general-purpose model by training on a diverse datasets of driving, robotics, exploration and handles heterogeneous action types via a Mixture of Action Experts.
3. The paper proposes a noisy memory training strategy which forces the model to reply on action signals and not over-rely on past visual information.

### Weaknesses
1. The ACT-Adapter seems be to showing a minimal performance improvement in Table 2. The ablation study in Table 2 shows it provides a score of 0.669 on Instruction Following, while a cross attn. adapter achieves 0.642, suggesting the performance gain of the new adapter is relatively small.
2. The comparison to baseline methods in Table 1 does not reflect a fair comparison.
a). Since Wan2.1 is the pre-trained backbone of Astra, it would be an ablation of the paper instead of baseline.
b). MatrixGame and YUME are described as domain-specific ("game-specific" and "walking-specific"). Since Astra is trained on a general mixture of data from multiple domains (driving, robotics, walking), the comparison of domain-specific with a general model does not seem to show a fair comparison. It creates a question that is it the data that is giving a good performance or is it the architecture and training strategy helping with the performance.

### Questions
There seems to be some contradiction in training details. Section 4.1 mentions that the number of target frames is fixed to 33. Appendix A.2 mentions that the number of target frames is set to 32. Can the authors please clarify this?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Astra, a framework for interactive world modeling that generates long and temporal coherence video sequences across diverse scenarios. Astra enhances a pre-trained video model with an light-weight action-aware adapter for precise action conditioning, a noise-augmented history memory during training to ensure long-term consistency, and a mixture of action experts to effectively handle diverse action inputs.

### Strengths
1. This paper uses a lightweight action-aware adapter for precise action conditioning.
2. Astra achieves good responsiveness and is able to generate long, temporally coherent video sequences by employing a noise-as-mask strategy during training.
3. Astra employs a mixture of action experts to effectively adapt to diverse scenarios and handle various types of action inputs.

### Weaknesses
1. Mixture of action experts idea is similar to [1, 2] and action-aware adapter is similar to [3, 4]. Please provide a conceptual comparison with these reference.
2. The paper does not thoroughly analyze the underlying reasons why the noise-as-mask strategy enables the generation of long, temporally coherent video sequences.
3. The paper does not explain why the router network performs so well across diverse scenarios and with various types of action inputs.

[1] Mixture of Action Expert Embeddings: Multi-Task ACT

[2] DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving

[3] Long-Context Autoregressive Video Modeling with Next-Frame Prediction

[4] Epona: Autoregressive Diffusion World Model for Autonomous Driving

### Questions
1. In Section 3.3, the explanation of the types of random noise and blur used is unclear. Please provide a more detailed description.
2. It was not properly analyzed the lightweight action-aware adapter model complexity. Please provide a more detailed description and comparison.
3. In Figure 6, compared to YUME, the results from Astra appear to exhibit some color shift. Could the authors explain the cause of this phenomenon?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, author proposes Astra, an interactive world model that extends pre-trained diffusion models for long-horizon, action-conditioned video prediction. 

The core contribution lies in three components in an auto-regressive denoising framework: (1) an action-aware adapter that injects action signals into the latent space of a pre-trained diffusion model, (2) a noise-augmented history memory mechanism that balances temporal consistency and action responsiveness, and (3) a mixture of action experts that routes heterogeneous action modalities to specialized experts. 

Astra is evaluated on self-proposed benchmark consisting of diverse datasets and demonstrates some improvements in long-range prediction stability compared to state-of-the-art models.

### Strengths
**Strength (1)**: The paper is well-organized. Authors explain the core ideas with clear diagrams and concrete algorithmic descriptions. 

**Strength (2)**: Astra is a single model across multi-modal action spaces, covering camera poses, keyboard/mouse inputs, and robot poses. 

**Strength (3)**: The proposed solutions exhibit several elegant and practical design choices:
- The action-free guidance mechanism offers a simple, original mechanism to amplify action effects without heavy architectural changes.
- The noise-augmented history memory is an elegant, parameter-free training strategy to reduce “visual inertia” and force the model to rely more on action signals, improving responsiveness without modifying the backbone.

**Strength (4)**: The authors conduct evaluations across multiple domains, including autonomous driving, egocentric, and robotic settings, showcasing reasonable coverage.

### Weaknesses
**Weakness (1)**: The definition and formulation of action signals are insufficiently specified. The paper does not clearly describe how different types of actions (e.g., camera poses, keyboard/mouse inputs, robot poses, etc.) are represented, parsed, and projected to the action encoder.

**Weakness (2)**: A comparable method, YUME [1], is not discussed in Section 2 (Related Work). The paper does not clearly articulate how Astra differs from or improves upon YUME, which weakens presentation.

**Weakness (3)**: Experimental validation of design choices is limited. For example, no ablation study isolates the contribution of the Mixture of Action Experts (MoAE). A comparison against a simpler variant without a gating network would clarify whether MoAE provides meaningful gains.

**Weakness (4)**: Quantitative comparisons with existing world modeling methods are lacking. Although authors cite several relevant works [2, 3, 4, 5], Astra is not evaluated against them, making it difficult to assess the model's relative performance and significance. 

**Weakness (5)**: Although Astra is positioned as a general interactive world model trained on a mixture of five datasets, all evaluations are conducted on held-out data drawn from these same domains. It remains unclear whether the model generalizes to unseen environments.

**Weakness (6)**: The paper combines pose tracking with human evaluation to assess "instruction following" in Astra-Bench, but the metric definition, aggregation procedure, and scoring protocol are not clearly specified. It is difficult to compare against future work. 

[1] Mao, Xiaofeng, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. "Yume: An interactive world generation model." arXiv preprint arXiv:2507.17744 (2025).

[2] Cen, Jun, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li et al. "WorldVLA: Towards Autoregressive Action World Model." arXiv preprint arXiv:2506.21539 (2025).

[3] Huang, Siqiao, Jialong Wu, Qixing Zhou, Shangchen Miao, and Mingsheng Long. "Vid2World: Crafting Video Diffusion Models to Interactive World Models." arXiv preprint arXiv:2505.14357 (2025).

[4] Bar, Amir, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. "Navigation world models." In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 15791-15801. 2025.

[5] Bruce, Jake, Michael D. Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai et al. "Genie: Generative interactive environments." In Forty-first International Conference on Machine Learning. 2024.

### Questions
**Question (1)**: In the evaluation of Table 1, Wan-2.1 [6] is used as a baseline. How is Wan-2.1 adapted to accept continuous action inputs during evaluation?

**Question (2)**: YUME [1] also extends Wan-2.1 [6] for interactive video prediction. Could the authors explain why Wan-2.1 is chosen as the base model instead of YUME?

**Question (3)**: Astra-Bench uses both MegaSaM [7] and human evaluations for “instruction following.” Could the authors clarify how the numerical values of "instruction following" in Tables 1 and 2 are computed?

**Question (4)**: All experiments train on a dataset mixture across five domains. There is no zero-shot evaluation on held-out domains. Does the model generalize to completely new environment unseen during training?

**Question (5)**: Authors claim that increasing the length of history improves temporal consistency but weakens responsiveness (Line 257). However, no supporting quantitative data are provided. Could the authors supply such evidence?

**Typographical Error**: “Eperimental” → “Experimental” (Line 594).

[6] Wan, Team, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen et al. "Wan: Open and advanced large-scale video generative models." arXiv preprint arXiv:2503.20314 (2025).

[7] Li, Zhengqi, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. "MegaSaM: Accurate, fast and robust structure and motion from casual dynamic videos." In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10486-10496. 2025.

### Soundness
2

### Presentation
2

### Contribution
3
