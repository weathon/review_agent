# Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
End-to-End (E2E) solutions have emerged as a mainstream approach for autonomous driving systems, with Vision-Language-Action (VLA) models representing a new paradigm that leverages pre-trained multimodal knowledge from Vision-Language Models (VLMs) to interpret and interact with complex real-world environments. However, these methods remain constrained by the limitations of imitation learning, which struggles to inherently encode physical rules during training. Existing approaches often rely on complex rule-based post-refinement, employ reinforcement learning that remains largely limited to simulation, or utilize diffusion guidance that requires computationally expensive gradient calculations. To address these challenges, we introduce ReflectDrive, a novel learning-based framework that integrates a reflection mechanism for safe trajectory generation via discrete diffusion. We first discretize the two-dimensional driving space to construct an action codebook, enabling the use of pre-trained Diffusion Language Models for planning tasks through fine-tuning. Central to our approach is a safety-aware reflection mechanism that performs iterative self-correction without gradient computation. Our method begins with goal-conditioned trajectory generation to model multi-modal driving behaviors. Based on this, we apply local search methods to identify unsafe tokens and determine feasible solutions, which then serve as safe anchors for inpainting-based regeneration. Evaluated on the NAVSIM benchmark, ReflectDrive demonstrates significant advantages in safety-critical trajectory generation, offering a scalable and reliable solution for autonomous driving systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes ReflectDrive, a VLA-based end-to-end planner that (1) discretizes 2D waypoints into a token codebook; (2) uses a discrete diffusion (masked-LM–style) model to inpaint trajectories; and (3) adds a reflection loop at inference: detect unsafe tokens via a safety scorer, do local discrete search to find a “safety anchor,” then re-inpaint around it—no gradients needed. Evaluated on NAVSIM with PDMS metrics (NC, DAC, TTC, Comfort, EP), the reflective loop substantially improves safety and progress, and with oracle agents it approaches human PDMS.

### Strengths
Pros:

1. **Clear, modular idea**: discrete tokenization → diffusion inpainting → gradient-free reflective editing. Nicely matches the structure of discrete diffusion (mask/remask/inpaint) and makes constraint injection tractable at inference. 
2. **Practical safety loop**: the local Manhattan search over tokens + re-inpainting is simple and fast, with most cases finishing in 1–3 reflection iterations. With five diffusion steps, remask-low-confidence decoding, K=3 goals, NMS=0.9 m, max 10 refinement steps. Concrete enough to re-implement.
3. **Ablations and attribution**: Tables in the experiments section isolate contributions of goal-conditioning vs safety-guided regeneration;

### Weaknesses
Cons: (There are no obviously novel concerns, but some questions)
1. A core theme of the paper is safety and rules. From the ablation in Table 2, Goal-Conditioned Generation (GCG) and Safety-Guided Regeneration (SGR) have the larger impact. So I infer that the “safety” you highlight mainly comes from this reflective module. If so, on the one hand, where does Discrete Diffusion deliver unique value? On the other hand, did you try other trajectory parameterizations combined with goal and safety-anchor guidance?
2. There is a lack of description of the training data for multi-modal behavioral modes. I’m curious how the training data influences behavioral multi-modality: if the data is concentrated on a small set of behaviors (i.e., collected from only a few drivers’ styles), how do you train a top-k set of diverse behaviors? How do you avoid trajectory mode collapse?
3. There is a lack of description of the training data for multi-modal behavioral modes. I’m curious how the training data influences behavioral multi-modality: if the data is concentrated on a small set of behaviors (i.e., collected from only a few drivers’ styles), how do you train a top-k set of diverse behaviors? How do you avoid trajectory mode collapse?
4. In the Introduction, the paper emphasizes that guidance needs gradients, leading to slow sampling, hyper-parameter sensitivity, and potential instability. It repeatedly claims one benefit of discrete diffusion is eliminating gradient computation to reduce cost, and around line 60 states that other deployment schemes suffer slow sampling due to gradients. However, generation-based planning also depends on the number of sampling steps and the stochastic process; you use step-skipping with only 5 steps, yet the paper appears to lack experimental analysis on computational cost and speed, while stressing that “discretization enables efficient search.” The Limitations section explicitly notes that no algorithmic/engineering acceleration was applied and there is still room for efficiency improvement. Additionally, around line 203 the paper says accuracy is affected by discretization resolution; as resolution increases to meet accuracy/quality, inference speed will inevitably be impacted, and that speed is tightly coupled with the large model’s forward pass. I hope the authors can provide analysis and results validating the significance of discrete diffusion, the inference speed, and the effect of resolution.
5.  The paper calls NAVSIM a “real-world autonomous driving benchmark,” but NAVSIM is a data-driven simulator; that nuance matters for claims of real-world safety. Please clarify wording and discuss implications (non-reactive agents, sim-to-real).
 6. Some baselines use Camera+LiDAR (e.g., Transfuser, Hydra-MDP, DiffusionDrive/GoalFlow) whereas ReflectDrive is camera-only in the main line. Mixed-modality comparisons are informative but not apples-to-apples; add camera-only versions of diffusion/flow baselines or provide a C&L variant of ReflectDrive.
7. Safety oracle realism. The reflection loop’s default scorer assumes constant-velocity surrounding agents, which the authors themselves link to sub-par TTC/NC vs the oracle variant; this can bias safety judgments during reflection. Please add a learned (or stronger rule-based) predictor to close the realism gap.
8. Reward/metric shaping risks. The Global/Local scorers combine hard safety gates with weighted PDMS-like terms; optimizing a test-metric proxy at inference risks over-fitting to evaluator quirks (e.g., binary TTC thresholding). Provide robustness checks or alternative scorers.
9. Missing discretization specifics. The core idea hinges on codebook resolution (Δg), grid size |A|, and search radius δ. The paper states spatial range  [−100,100 ], but not Δg nor the token vocabulary size actually used at inference; please report these and study sensitivity.
10. Latency and systems profile.  The method is “designed for real-time,” yet no ms/frame or hardware stats are given (diffusion steps + reflection iterations + scorer costs). Please provide latency/throughput/memory on a target GPU/SoC and discuss worst-case bounds.

### Questions
1. Codebook & search: What Δg and codebook size |A| do you use? How is δ chosen (and its runtime impact)? Any adaptive δ strategy during reflection?
2. Noise schedule: You mention cosine-like masking, S=5 inference steps. What is the exact masking schedule during training/inference (token-wise remask policy, “remask low-confidence” threshold)?
3. Please formalize how “at-fault” collisions are computed and how TTC is thresholded; include ablations of scorer weights, W_ep, W_ttc, W_c.
4. Goal proposal: Why K=3 and d_NMS=0.9 m? Show sensitivity (K=1/3/5/10 already partly given—add per-metric trends and qualitative examples).
5. Variance & failures: In the oscillation failures (boundary vs collision), did larger δ or more iterations fix them, or is it a structural issue of discrete grids? Any hybrid continuous snap-back after the discrete anchor?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents ReflectDrive, a novel framework that integrates a reflection mechanism for safe trajectory generation in autonomous driving using discrete diffusion models. The authors aim to enhance the safety and reliability of trajectory planning by combining discrete diffusion with goal-based trajectory planning. The framework introduces three scoring functions—Global Scorer, Safety Scorer, and Local Scorer—to evaluate the generated trajectories and ensure adherence to safety constraints.

### Strengths
1. **Safety Focus**: The emphasis on safety through the reflection mechanism and scoring functions demonstrates a commitment to developing reliable autonomous systems.
2. **Evaluation on Closed-loop Benchmarks**: The use of NAVISM benchmark for evaluation adds credibility to the findings and demonstrates practical applicability.
3. **Easy to follow**: The key idea is clear and the paper is easy to understand.

### Weaknesses
1. **Limited Novelty**: The primary contribution of the paper appears to be the combination of discrete diffusion models with goal-based trajectory planning. This concept closely resembles the existing GoalFlow work, where flow matching is replaced by discrete diffusion, and more scorers are used. The incremental nature of this contribution raises questions about its originality.
   
2. **Insufficient Ablation Studies**: While the paper proposes three distinct scoring functions, comprehensive ablation studies are lacking. Only two of the scorers are evaluated, which does not provide a complete understanding of the impact of each component on the overall performance. A thorough ablation analysis is necessary to validate the effectiveness of all proposed components.

3. **Lack of Detail in Safety-Guided Regeneration**: The section on Safety-Guided Regeneration is overly simplistic. It does not adequately address the success rate of the local search strategy or provide contingency plans for search failures. Additionally, the generalizability of this search method in extreme or unusual scenarios is not explored, which is critical for assessing its robustness.

### Questions
which scorer play the most important role? Is it possible to use single scorer?

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
This paper proposes ReflectDrive, a framework for end-to-end autonomous driving that leverages a Vision-Language-Action (VLA) model with a discrete diffusion planner. The core contributions are: 1) The discretization of the continuous driving space into an action codebook, which allows for the use of a discrete diffusion model for trajectory generation. 2) A reflection mechanism with a gradient-free, iterative inference-time process. This mechanism first generates multi-modal trajectories based on sampled goals and then uses a safety-guided regeneration step to correct unsafe waypoints identified by a safety scorer.

### Strengths
1. The application of discrete diffusion and an inpainting-based reflection mechanism to a VLA model for driving is a novel approach to integrating safety constraints.
2. The work directly tackles the critical and unsolved problem of enforcing hard safety constraints (like collision avoidance and drivable area compliance) within imitation learning-based frameworks.

### Weaknesses
1. The safety score is a very dedicated design that still falls into the rule-based refinement. More critically, obtaining these scores would inevitably employ some privileged information. For example, the DAC score will require the knowledge of whether the car will fall out of the drivable region, which is impossible to reliably obtain purely from sensor data in the real world. This reliance on an oracle-like safety scorer questions the practical applicability of the reflection mechanism. It will be nice to see ablation studies where some less reliable, learning-based rewards are involved.
2. To justify the effectiveness of the discrete action space, it would be beneficial to see an ablation study on different granularities of the discretization.

### Questions
1. What do the variables c and s represent in Equation (2)?
2. The safety scoring mechanism described in Appendix C (e.g., $m_{DAC}$, $m_{NC}$) seems to require ground-truth or privileged simulation information. How do you envision this safety-guided regeneration being applied in a real-world vehicle that relies only on on-board sensor data?

### Soundness
2

### Presentation
4

### Contribution
3
