# BridgEAD: A Vision-Language Framework for Action Modeling in End-to-End Autonomous Driving

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Recently, Vision-Language Models (VLMs) have shown promising prospects in autonomous driving tasks by leveraging rich world knowledge. However, current methods still face significant challenges in aligning the semantic space with the action space and struggle to maintain robust performance in closed-loop evaluations and long-tail scenarios. To address these challenges, we propose BridgEAD in this paper, a novel Vision-Language-Action (VLA) framework for end-to-end autonomous driving that unifies action planning and semantic reasoning. It integrates multi-view visual inputs and historical context into an unmodified VLM backbone for driving scenario reasoning, and leverages a diffusion-based generative planner to further align multimodal scene representations with precise trajectories. We employ supervised fine-tuning for model training to enable end-to-end optimization, thereby endowing BridgEAD with visual question-answering and trajectory planning capabilities. Extensive experiments on multiple benchmarks, including nuScenes, NAVSIM, and Bench2Drive, demonstrate that BridgEAD achieves superior trajectory planning performance in both open-loop and closed-loop evaluations across challenging driving environments. Qualitative results further highlight BridgEAD’s strong semantic reasoning ability in driving-related question-answering tasks. We will make our code publicly available upon publication to support future research in this domain.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes to integrate a diffusion based generative planner module with a VLM backbone processing scene images and navigational text. They train the model end to end with supervised fine-tuning while aligning multi-modal scene representations with trajectories. They run evaluations on NuScenes, NAVSIM and Bench2Drive showing comparable performance with SOTA models.

### Strengths
1. VLM does not need any fine-tuning. Any public VLM can be used for this architecture in a plug and play manner.
2. Planning token is a learnable and continuous, without any action code-book or dictionary. This is more flexible as the planning token remains latent, without requiring any supervision or explicit interpretation.

### Weaknesses
1. Performance. In the tables presented closed and open loop performance is worse than other VLM based approaches. There are specific abilities where this approach performs better, but overall there is performance degradation.
2. Using diffusion for planner is expensive, there should be some discussion on latency of such method.
3. Use of generic VLM as frozen can also be seen as weakness, as these models can have limited understanding of driving maneuvers. It is worth experimenting with VLMs fine-tuned on driving data like OmniDrive.

### Questions
1. The outputs of language head and planner head are used for separate loss functions, there is no direct alignment between these outputs. So, it's possible that the language model producing driving decision inconsistent with the planner. Did the authors consider any mechanism to prevent this?
2. The authors claim the trajectories generated are kinematically feasible. How was this enforced? The conditioning of the diffusion model with planner token does not enforce vehicle kinematic constraints.

### Soundness
3

### Presentation
2

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
BridgEAD targets a pivotal gap in VLA-based autonomous driving—the "semantic-to-action gap"—by leveraging an unmodified VLM for semantic reasoning and a diffusion planner for trajectory generation, demonstrating practical value in its modular design and comprehensive evaluation across open/closed-loop settings. The paper’s coverage of diverse datasets (real-world nuScenes, simulation-based Bench2Drive) and ablation studies on trajectory generation methods provide partial support for its core claims. However, critical limitations undermine its impact: key technical details (e.g., kinematic constraint integration in the diffusion planner) are underspecified, evaluations lack dedicated validation for long-tail scenarios, and most importantly, the model fails to achieve state-of-the-art (SOTA) performance across key benchmarks, with results either matching or falling short of existing baselines. Clarifying technical mechanisms, expanding evaluation scope, and acknowledging performance boundaries would strengthen the work’s contribution to VLA-driven autonomous driving.

### Strengths
The paper clearly identifies the dual limitations of current end-to-end (E2E) and VLM-based autonomous driving: traditional E2E methods lack high-level semantic reasoning, limiting generalization in long-tail scenarios; existing VLM-based approaches struggle with aligning abstract semantic outputs to precise, kinematically feasible trajectories. BridgEAD’s split design—retaining the VLM’s semantic reasoning via unmodified integration while using a diffusion planner for trajectory specialization—directly addresses this semantic-to-action gap.
Plug-and-Play Modularity： By freezing the pre-trained VLM backbone and adding a separate diffusion planner, BridgEAD avoids the complexity of modifying VLM architectures. This design enables easy upgrades as foundation VLMs evolve and reduces deployment costs, while lightweight interfaces (linear projection for visual tokens, text tokenizer for ego states/instructions) facilitate efficient multimodal fusion .
Comprehensive Evaluation：Unlike many works that focus solely on open-loop evaluations, BridgEAD is validated on three distinct benchmarks (nuScenes, NAVSIM, Bench2Drive) covering real-world and simulated environments. It assesses both trajectory planning (e.g., L2 distance, collision rate in open-loop; driving score, success rate in closed-loop) and semantic reasoning (VQA via BLEU, ROUGE, CIDEr), providing a holistic view of its VLA capabilities

### Weaknesses
While the paper claims the diffusion planner generates "kinematically feasible trajectories," it provides no details on how vehicle dynamics (e.g., maximum acceleration, steering angle limits) are integrated. 

Despite framing long-tail scenario performance as a key motivation, the paper does not conduct dedicated evaluations for extreme conditions (e.g., heavy rain, severe occlusion) or dynamic multi-agent interactions (e.g., unprotected left turns, sudden pedestrian crossings). Bench2Drive’s 23 weather conditions and 44 scenarios are only reported in aggregate, with no analysis of performance degradation in low-probability, high-risk scenarios

The paper fails to clarify how the VLM’s semantic outputs, driving decisions guide the diffusion planner.

BridgEAD does not achieve SOTA results on any major benchmark.

### Questions
While the paper claims the diffusion planner generates "kinematically feasible trajectories," it provides no details on how vehicle dynamics (e.g., maximum acceleration, steering angle limits) are integrated. 

Despite framing long-tail scenario performance as a key motivation, the paper does not conduct dedicated evaluations for extreme conditions (e.g., heavy rain, severe occlusion) or dynamic multi-agent interactions (e.g., unprotected left turns, sudden pedestrian crossings). Bench2Drive’s 23 weather conditions and 44 scenarios are only reported in aggregate, with no analysis of performance degradation in low-probability, high-risk scenarios

The paper fails to clarify how the VLM’s semantic outputs, driving decisions guide the diffusion planner.

BridgEAD does not achieve SOTA results on any major benchmark.

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
This paper proposes BridgEAD, a Vision–Language–Action (VLA) framework to address end-to-end autonomous driving which combines semantic reasoning and trajectory planning. Unlike traditional end-to-end models that lack high-level reasoning, BridgEAD integrates:

- Multi-view visual inputs, navigation commands, and ego-vehicle states into an unmodified Vision-Language Model (VLM) backbone for scene understanding and reasoning.
- A diffusion-based generative planner to bridge the semantic-to-action gap, translating reasoning outputs into precise, kinematically feasible trajectories.
- Supervised fine-tuning for joint optimization of visual question answering (VQA) and trajectory planning.

### Strengths
- BridgEAD addresses the semantic-to-action gap effectively, and improving alignment between reasoning and control.
- BridgEAD can be applied in plug-and play option, where BridgEAD uses unmodified VLM backbones, making the approach portable and scalable as foundation models evolve.
- BridgEAD conducted a good ablation study, showing clear comparison of trajectory generation methods (DiT vs. action tokens vs. text numbers), highlighting the superiority of the proposed approach.

### Weaknesses
- Despite improvements using the diffusion-based approach still incurs higher inference latency compared to VLP, DiMA which distills VLM knowledge to simpler planners. Comparisons with these relevant methods like VLP and DiMA are missing.
- Comparison scope: Strong against VLM-based baselines, but limited discussion on reinforcement learning or hybrid approaches like Diffusion Planner.
- BridgEAD claims Generative Action Planner as novelty, but integration of diffusion planner or VAE based planner with VLM is explored in ORION. 
- BridgEAD Relies on short history (6 frames at 2 Hz); may struggle in highly dynamic or multi-agent scenarios. It would be better if authors can provide an comparison showing the BridgEAD' s temporal reasoning limitations.

- VLP, VLP: Vision Language Planning for Autonomous Driving, CVPR 2024.

- DiMA, Distilling Multi-modal Large Language Models for Autonomous Driving, CVPR 2025.

### Questions
- How does BridgEAD handle ambiguity or occlusions in multi-view inputs, especially for critical objects like pedestrians?
- Could temporal context extension (e.g., longer history or predictive features) improve performance in complex scenarios?
- What is the inference latency for closed-loop planning? Is real-time deployment feasible?
- How robust is the diffusion planner to distribution shifts (e.g., different cities, weather, or sensor setups)?
- What are the most common failure cases observed during evaluation, and how severe are they in terms of safety?

### Soundness
2

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
5

### Summary
This paper introduces BridgEAD, which is a vision-language based model for end-to-end driving.  BridgEAD uses language head to perform autoregressive prediction for reasoning and diffusion head for driving action prediction. It achieves good performance on both nuScenes and NAVSIM.

### Strengths
1. The idea that uses language autoregressive sampling for reasoning and diffusion model for action prediction is well-motivated and promising.
2. The paper is well-written with good clarity.

### Weaknesses
I have major concerns in the experiment results.
1. The authors ignored a new important E2E driving benchmark -- WOD-E2E[1]. This benchmark is designed for long-tailed E2E driving scenarios, which can test model's generalization capability.
2. BridgeEAD did not achieve the best performance at any benchmark. In NAVSIM, it is worse than TransDiffuser. In Bench2Drive, it is worse than AutoVLA. In nuScenes, it is worse than EMMA.
3. A lot of details are not covered by the paper, including but not limited to:
1) Which vlm model did the authors use? Is it Qwen?
2) What are the training configurations for the model?
3) How the latency looks like?

[1] WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios

### Questions
To change my opinion, the authors need to :
1) Show results in WOD-E2E.
2) Have a good explanation  of the inferior results in the several benchmarks.
3) Clarify the details.

### Soundness
3

### Presentation
3

### Contribution
2
