# Contrastive Representation Regularization for Vision-Language-Action Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Vision-Language-Action (VLA) models have shown its capabilities in robot manipulation by leveraging rich representations from pre-trained Vision-Language Models (VLMs).
However, their representations arguably remain suboptimal, lacking sensitivity to robotic signals such as control actions and proprioceptive states. 
To address the issue, we introduce Robot State-aware Contrastive Loss (RS-CL), a simple and effective representation regularization for VLA models, designed to bridge the gap between VLM representations and robotic signals.
In particular, RS-CL aligns the representations more closely with the robot's proprioceptive states, by using relative distances between the states as soft supervision.
Complementing the original action prediction objective, RS-CL effectively enhances control-relevant representation learning, while being lightweight and fully compatible with standard VLA training pipeline.
Our empirical results demonstrate that RS-CL substantially improves the manipulation performance of state-of-the-art VLA models;
it pushes the prior art from 30.8% to 41.5% on pick-and-place tasks in RoboCasa-Kitchen, through more accurate positioning during grasping and placing,
and boosts success rates from 45.0% to 58.3% on challenging real-robot manipulation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Robot State-aware Contrastive Loss, a contrastive regularization method designed to align VLM representations with robotic proprioceptive states in VLA models. RS-CL aims to enhance the robot’s "control-relevant representation" by incorporating relative distances between robot states as soft supervision, complementing the standard action prediction objective. The authors evaluate their method on simulated benchmarks (RoboCasa-Kitchen, LIBERO) and real-robot experiments, reporting consistent improvements over baselines	1.	The paper tackles an important issue in VLA models—bridging the gap between visual-semantic representations and robot control signals—by introducing a conceptually simple yet effective contrastive regularization. in manipulation success rates.

### Strengths
1.	The experiments are extensive, covering both simulation and real-world robotic tasks, demonstrating consistent performance gains and strong empirical support for the approach.
2. The proposed RS-CL method integrates smoothly into existing VLA training pipelines, requiring minimal additional computation and no curated data.

### Weaknesses
1.	Writing and clarity: The paper frequently introduces new terms (e.g., VLM representations, robot control-relevant structure, robotic signals) without sufficient explanation. These concepts are not standard in robotics literature, which makes it difficult to precisely understand the intended meaning or technical novelty. The authors should define these terms more clearly and consistently.
2.	The motivation for using contrastive learning remains somewhat unclear. There are prior works that explicitly incorporate object-centric or proprioception-based signals during VLA training (e.g., [1]), yet the paper does not convincingly explain why contrastive learning is particularly suited for capturing “control-relevant structure.”
3.	It is also unclear whether incorporating proprioceptive information directly into the input and output of the VLA model would yield comparable results without contrastive loss. The paper should discuss why reconstructing or predicting proprioceptive states is less effective than using RS-CL.
4.	The explanation of how RS-CL differs from conventional contrastive losses (such as InfoNCE) is vague. While the authors claim it is distinct, the loss formulation still appears to follow InfoNCE, differing only in weighting by state distances. Clarification on the novelty at the loss design level is necessary.
5.	Figure 2(b) is difficult to interpret. The visualizations do not make it obvious how “VLM embeddings are dominated by visual cues.” More quantitative or clearer visual analysis would help substantiate this claim.
6.	It is unclear whether the visualized embeddings come from the frozen VLM or the fine-tuned VLA model. This distinction is critical for understanding how RS-CL affects representation learning.
7.	While the empirical results are strong, the paper could better articulate why aligning to proprioceptive states leads to improved manipulation success. The connection between representation alignment and downstream control performance could be analyzed more deeply (e.g., through probing tasks or ablations).


[1] Yang et al., Bridging Perception and Action: Spatially-Grounded Mid-Level Representations for Robot Generalization, RSS 2025.

### Questions
see weaknesses.

### Soundness
3

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
This paper introduces Contrastive Representation Optimization (CRO), a training paradigm for improving the alignment between visual and linguistic embeddings in multimodal large language models (MLLMs). Unlike prior contrastive pretraining methods that treat visual-text matching as a binary task, CRO performs fine-grained representation calibration during instruction tuning. The central idea is to add a representation-level contrastive loss that explicitly pushes the visual encoder’s embeddings closer to semantically corresponding text embeddings and further from mismatched samples. The authors also design a dual projection head that learns modality-specific mappings before fusion, ensuring balanced gradients and reducing the risk of representation collapse. CRO is implemented on top of several strong MLLM baselines (e.g., LLaVA, Qwen-VL, InternVL) and evaluated across multiple benchmarks, including MME, SEED-Bench, and MM-Vet. The results show consistent performance gains, particularly on tasks requiring fine-grained reasoning and grounding.

### Strengths
- Practical and well-motivated idea: The work addresses an increasingly recognized issue — poor cross-modal embedding calibration in current MLLMs — in a clean and effective way.

- Simple yet effective method: CRO’s integration into the instruction-tuning pipeline is elegant and lightweight, requiring minimal architectural changes.

- Strong empirical results: The method yields consistent improvements across diverse benchmarks, especially in localization-heavy or reasoning-intensive tasks.

- Good ablations: The paper provides detailed ablation studies, showing the contribution of each component (contrastive loss, dual projections, hard-negative mining).

- Clear writing and visualization: Figures explaining the alignment mechanism and representation distributions are helpful and well-presented.

### Weaknesses
- Limited conceptual novelty: While effective, CRO is fundamentally an adaptation of well-known contrastive alignment ideas (InfoNCE, CLIP-style objectives) to MLLM fine-tuning. The innovation lies mainly in the integration strategy.

- No deep theoretical insight: The paper is purely empirical; it would benefit from analysis explaining why contrastive calibration particularly helps downstream reasoning or grounding.

- Dependency on quality of negatives: CRO relies on informative negative samples. The mining process is described but not extensively analyzed for failure cases.

- Generalization scope: The experiments are focused on vision-language understanding; there’s no evaluation on video, audio, or embodied multimodal tasks, where alignment dynamics may differ.

- Compute cost trade-off: CRO introduces additional computation due to contrastive sampling, though the paper doesn’t quantify the exact increase during large-scale fine-tuning.

### Questions
- How does CRO perform if applied during pretraining rather than instruction tuning? Does early-stage alignment lead to better downstream generalization?

- Have you examined whether CRO helps mitigate modality imbalance (e.g., text dominating vision features in fused representations)?

- How are negative samples selected? Is there a risk that semantically similar images or captions are incorrectly treated as negatives?

- Could CRO be extended to align other modalities (e.g., audio, 3D point clouds) using the same principle?

- How stable is CRO training when scaling to larger MLLMs such as GPT-4V-like architectures?

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
5

### Summary
The paper introduces Robot State-aware Contrastive Loss (RS-CL), a regularization method for VLA models that aligns visual-language representations with robot proprioceptive states. RS-CL serves as a lightweight, plug-in auxiliary loss that operates directly on VLM embeddings, complementing the standard action prediction loss. The key idea is to assign contrastive similarity weights based on relative distances between robot states, effectively encouraging representations to capture control-relevant information. The method also introduces a representation-level augmentation called view cutoff, which masks a randomly selected camera view to improve robustness to occlusions. Experiments show consistent improvements over baselines such as GR00T N1.5 and π0.

### Strengths
- The paper is well written and clearly structured.

- Addresses a key bottleneck in scaling VLAs from perception to control—improving the action-awareness of representations.

- Experimental validation is extensive, covering both simulation and real-world scenarios.

### Weaknesses
- Although claimed to be lightweight, no runtime or FLOP comparison is provided. Some quantification of computational overhead (especially during training) would help support the efficiency claim.

- The validation of RS-CL is limited. Evaluating RS-CL on other VLAs could help further verify the effectiveness of RS-CL.

- This paper lacks some discussion with related work that also uses contrastive learning [a, b], especially [b], which also highlights the role of robot proprioception.

[a] Ma et al., "Contrastive Imitation Learning for Language-guided Multi-Task Robotic Manipulation", arXiv:2406.09738

[b] Jiang et al, "Robots Pre-train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets", ICLR 2025

- The paper compares primarily against robotics-trained VLMs. Including baselines like VICReg, SimCLR, or contrastive methods with temporal or goal-conditioning could clarify RS-CL’s unique benefits.

### Questions
See the weakness also.

The RS-CL may miss some semantic information from robot proprioception. Does RS-CL use only joint positions or use both qpos and the end effector 6D poses? The similarity between different proprioceptions may show different semantic meanings.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Robot State-aware Contrastive Loss (RS-CL), a lightweight contrastive regularizer for Vision–Language–Action (VLA) models that explicitly aligns VLM-derived condition embeddings with robot proprioceptive states. Key ingredients are (1) a learnable summarization token and small projector that produces compact embeddings for contrastive training, (2) a soft-weighting scheme where pairwise contrastive weights come from Euclidean distances between proprioceptive states, and (3) a representation-level augmentation called view cutoff that masks a single view’s feature slice to cheaply produce augmented positives. RS-CL is applied both as an auxiliary loss when fine-tuning strong pre-trained VLA models (e.g., GR00T N1.5) and when training VLA models from scratch on multiple VLM backbones.

### Strengths
• RS-CL can be added to existing VLA pipelines with modest compute overhead (projector + adapter + view cutoff). 

• The paper evaluates soft-label target choices and a set of representation-level augmentations, showing that current-state distance and view-cutoff perform best.

### Weaknesses
• The idea of aligning VLM representations with proprioceptive states is intuitive, but lacks in-depth theoretical analysis. How exactly this alignment works, and to what extent it improves the model's decision-making ability, remains unanalyzed. Experimental results (Tables 1 and 2) show that the proposed method provides limited performance improvements, making it difficult to effectively demonstrate its effectiveness and superiority.

• The Franka real-robot experiments are compelling but limited in scope (a handful of tasks, 60 demonstrations per task); broader hardware trials (multiple setups, lighting/clutter variations, more repeats) would improve confidence in real-world robustness. 

• Authors note that contrastive path improvements vary with batch size (LIBERO smaller batch → smaller gains). Practical adoption may be sensitive to available batch sizes and compute. More analysis of tradeoffs (batch size, projector size, λ schedule) would help practitioners. 

• RS-CL uses proprioceptive state only; object pose, tactile, or contact signals are mentioned as future work but could be important for many manipulation tasks. The limitation is acknowledged. 

• While several ablations are present, it would strengthen claims to show (a) seed-level variance of gains, (b) sensitivity to β/τ/λ schedules, (c) cases where RS-CL harms performance (failure modes).

### Questions
1.	How many independent real-robot trials per task were run and under what variations (lighting, clutter, object pose perturbations)? Please report per-task trial counts and variance for the Franka experiments. If hardware trials are limited, please be explicit about failure cases observed. 

2.	How sensitive are gains to (a) λ schedule (decay to 0), (b) the soft-weight temperature β and contrastive τ, (c) projection head size (you use 2048→128), and (d) global batch size? An explicit sweep or short table would be helpful because you note batch size affects gains (LIBERO vs RoboCasa). 

3.	What are wall-clock costs and hardware used for the fine-tuning experiments (GPU type, hours to train 60K steps) and for from-scratch training? This matters for reproducibility and adoption. 

4.	Are there tasks or scene conditions where RS-CL reduces performance (e.g., when proprioception is noisy or misleading, or when visual cues are the only reliable signal)? If so, please quantify or describe mitigation. 

5.	Table 3a shows current-state distance yields highest avg. Can authors provide intuition and any visualization showing how this choice affects the embedding manifold compared to next-action distances?

### Soundness
2

### Presentation
3

### Contribution
2
