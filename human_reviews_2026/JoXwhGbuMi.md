# ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Recent studies have explored leveraging the world knowledge and cognitive capabilities of Vision-Language Models (VLMs) to address the long-tail problem in end-to-end autonomous driving. However, existing methods typically formulate trajectory planning as a language modeling task, where physical actions are output in the language space, potentially leading to issues such as format-violating outputs, infeasible actions, and slow inference speeds. In this paper, we propose ReCogDrive, a novel **Re**inforced **Cog**nitive framework for end-to-end autonomous **Driv**ing, unifying driving understanding and planning by integrating an autoregressive model with a diffusion planner. First, to instill human driving cognition into the VLM, we introduce a hierarchical data pipeline that mimics the sequential cognitive process of human drivers through three stages: generation, refinement, and quality control. Building on this cognitive foundation, we then address the language-action mismatch by injecting the VLM's learned driving priors into a diffusion planner to efficiently generate continuous and stable trajectories. Furthermore, to enhance driving safety and reduce collisions, we introduce a Diffusion Group Relative Policy Optimization (DiffGRPO) stage, reinforcing the planner for enhanced safety and comfort. Extensive experiments on the NAVSIM and Bench2Drive benchmarks demonstrate that ReCogDrive achieves state-of-the-art performance. Additionally, qualitative results across diverse driving scenarios and DriveBench highlight the model's scene comprehension.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
ReCogDrive introduces a reinforced cognitive framework that integrates high-level semantic reasoning from a Vision-Language Model (VLM) with continuous control via a Diffusion Planner for end-to-end autonomous driving. Through a three-stage training pipeline—driving Q&A pretraining to bridge the domain gap, imitation learning for language-to-action mapping, and reinforcement learning in simulation for safety and generalization—the model achieves robust, human-like decision-making in rare scenarios.

### Strengths
1. Integration of cognition and control: Combines semantic reasoning from a VLM with continuous control from a diffusion planner, forming a full cognitive-to-action pipeline.

2. Three-stage training paradigm: Sequentially applies Q&A pretraining, imitation learning, and reinforcement learning to enhance understanding, controllability, and safety while reducing domain and generalization gaps.

### Weaknesses
1.The paper presents a "cognitive framework" that, upon inspection, is a modular integration of existing methods rather than a novel algorithmic solution. The "hierarchical data pipeline" (Sec 4.1) is a data curation strategy, not an algorithmic one. It offers no new mechanism for mitigating domain bias or improving multi-modal alignment. The use of a diffusion planner (Sec 4.2) to connect VLMs to continuous control is not new. This has been demonstrated by recent work (e.g., ORION [1]), making the contribution here incremental at best. The "DiffGRPO" (Sec 4.3) is not a new algorithm but a direct application of the existing GRPO framework to a diffusion planner. Framing this as a key innovation inflates the paper's novelty. Furthermore, this RL stage operates in a decoupled fashion, fine-tuning the action policy without any mechanism to enhance or feed back into the VLM's reasoning process.

2.The central premise of using a VLM is for its reasoning, yet the paper's own results show the VLM is relegated to a simple feature extractor. The paper's own ablation study (Table 3) is damning: the "Driving Pre-training" (the cognitive part) yields a trivial 1.7 PDMS gain. In contrast, the non-cognitive RL fine-tuning provides the main performance boost (4.3 PDMS). Most critically, the appendix (Table 10) confirms that adding Chain-of-Thought (CoT) reasoning provides "no gain" and even hurts performance (a 0.1 PDMS drop). This finding fatally undermines the paper's core "cognitive" premise.

3.The paper's strongest result (+4.3 PDMS) comes from an RL stage that is effectively "training on the test." The model is fine-tuned within the NAVSIM simulator using the benchmark's own evaluation metric (PDMS) as the reward signal. This methodology is highly questionable. This approach raises significant concerns about the fairness of the comparison to other methods and suggests the results are likely overfit to the NAVSIM benchmark, severely limiting any claims of generalizability.

4.The experimental comparison (Table 1) is weak. The paper fails to benchmark against key contemporary VLA-specific models (e.g., AutoVLA [2]), a notable omission in this fast-moving subfield. Instead, the comparisons are against general-purpose E2E models or VLM models. This makes it impossible to assess the paper's true contribution to VLA research.

5.Finally, the experimental setup is ambiguous in several key areas, hindering reproducibility. The exact training data mix is unclear. The paper mentions multiple VQA datasets (Appx. D) plus 775K newly generated pairs, but the final composition and weighting for the reported models are not specified. The Bench2Drive protocol (Table 2) is unexplained. It's impossible to tell if these are zero-shot results or if the model was fine-tuned, making the scores difficult to interpret. The tables themselves are sloppy. Table 3 uses "100" and "100.0" inconsistently in the same column. Table 4 introduces an "Err. (%)" metric without a clear definition. The small differences are highlighted in red, creating unnecessary confusion.

[1] ORION: A Holistic End-to-End Autonomous Driving Framework. ICCV 2025

[2] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning. NeurIPS 2025

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes ReCogDrive, a novel end-to-end autonomous driving framework that integrates a Vision-Language Model (VLM) for cognitive reasoning with a diffusion planner to generate continuous trajectories. The framework is further refined using a reinforcement learning stage called DiffGRPO to improve safety and comfort, achieving state-of-the-art results on the NAVSIM and Bench2Drive benchmarks.

### Strengths
1. Well-designed Framework: The core architecture that integrates a cognitive VLM, a diffusion planner, and an RL refinement stage is well designed. It provides an effective solution to the modality mismatch problem for VLM-based agents.
2. Strong Empirical Validation: The paper's claims are well-supported by strong empirical performance on two challenging benchmarks (NAVSIM and Bench2Drive), comprehensive ablation studies that isolate the gains from each component, and a methodical data curation pipeline.

### Weaknesses
**Major Weaknesses:**

1. Lack of Motivation for GRPO: The paper provides insufficient insight into the choice of Group Relative Policy Optimization (GRPO). It is not clear why GRPO is better suited for optimizing a diffusion policy in this context compared to other well-established RL algorithms (e.g., PPO, or a simpler policy gradient method like REINFORCE).
2. Unclear DiffGRPO Algorithm Design: The description of the DiffGRPO algorithm seems to imply a fundamental difference from typical GRPO. In GRPO, trajectories within one group are computed starting from the same initial state. This design choice is critical for the advantage calculation to be valid, and the paper needs to elaborate more on this.

**Minor Weaknesses:**

1. Unclear Methodology and Data Pipeline: The methodology, particularly regarding the specifics of the data pipeline (Section 4.1) and the diffusion planner architecture (Section 4.2), is explained at a high level and lacks detailed examples for full clarity.
2. Messy Symbol Use: The paper suffers from unclear and at times inconsistent mathematical notation (e.g., $E_{act}$, $E_{hist}$, $D_{act}$, $x_{hist}$ are undefined, and $r_{1...G}$ in Eq. 9 is ambiguous).

### Questions
1. Could you please provide a more precise definition of the problem setup? Specifically, what are the exact of the inputs $I_{cam}$, $L_{nav}$, and $S_{ego}$? Similarly, how is the output $V_{traj}$ defined? What is the rationale for predicting headings when waypoints (which imply direction) are already being generated?
2. In Section 4.2, what is the architectural motivation for using both the full VLM hidden states $F_h$ and the pooled global embedding $\bar{F}_h$? What distinct roles do these two forms of conditioning play?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ReCogDrive, to address challenges in end-to-end autonomous driving framework that integrates a Vision-Language Model (VLM) with a diffusion-based planner and a reinforcement learning (RL) module. 
The key contributions include:
Integrates the diffusion planner with VLM models that translates VLM outputs into continuous, stable driving trajectories, addressing the modality mismatch between language and action spaces in way to provide cognitive guidance and enabling precise trajectory generation.
A Diffusion Group Relative Policy Optimization (DiffGRPO) algorithm that fine-tunes the planner using RL to improve safety and comfort.

The model is evaluated on NAVSIM and Bench2Drive benchmarks, achieving state-of-the-art performance in both open-loop and closed-loop settings. It also demonstrates superior performance on DriveLM and DriveBench for visual question answering (VQA) and reasoning tasks.

### Strengths
- The integration of VLMs with diffusion models and reinforcement learning addresses key limitations in current end-to-end driving systems.
- The structured approach to data generation and refinement is scalable, and enables the creation of high-quality VQA datasets for autonomous driving.
- The use of a diffusion planner to bridge the gap between discrete language outputs and continuous control actions.
- The introduction of DiffGRPO is a thoughtful addition that enhances the planner’s ability to generate safer and more comfortable trajectories.
- The paper presents extensive experiments, ablation studies, and qualitative analyses across multiple benchmarks, demonstrating the robustness and generalization of the approach.

### Weaknesses
- Diffusion Planner: integration of diffusion planner with VLM is explored in ORION. Despite improvements using  the diffusion-based approach still incurs higher inference latency compared to VLP, DiMA which distills VLM knowledge to simpler planners. Comparisons with these relevant methods like ORION, VLP and DiMA are missing.
- Zero shot testing: The model is evaluated in simulation environments (NAVSIM, CARLA), but lacks real-world deployment or testing, which is crucial for autonomous driving applications. It would be if authors perform zero-shot testing like training on Bench2Drive and testing on nuScenes.
- Training Complexity: The system’s architecture is quite complex, involving multiple stages (VLM pretraining, diffusion planning, RL fine-tuning), which may pose challenges for reproducibility and deployment. Detail training procedures are missing.
- Optional Use of VLM Reasoning: While the VLM can provide high-level reasoning, the paper notes that such guidance did not significantly improve performance in current benchmarks, raising questions about the practical use VLM high-level reasoning. This also raises a question  whether ReCogDrive can be applied to UniAD, VAD like architectures without VLM model, and UniAD-ReCogDrive gain improvements or not on base UniAD performance.

- ORION, ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation, ICCV 2025
- VLP, VLP: Vision Language Planning for Autonomous Driving, CVPR 2024.
- DiMA, Distilling Multi-modal Large Language Models for Autonomous Driving, CVPR 2025.

### Questions
Please refer weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ReCogDrive, an end-to-end autonomous driving framework that combines a Vision-Language Model (VLM) with a diffusion-based planner and an RL fine-tuning stage (DiffGRPO). The key idea is to use a cognitive data pipeline to teach the VLM human-like reasoning skills, then guide continuous trajectory generation through diffusion and reinforcement learning. The experiments on NAVSIM and Bench2Drive look solid — the model consistently outperforms both VLM- and diffusion-based baselines and achieves strong closed-loop results.

### Strengths
The paper is well-written and easy to follow, with clear figures and good motivation.
I like the overall integration of cognitive reasoning (via VLM) and low-level control (via diffusion + RL).
Results are strong and consistent across benchmarks, and the ablation studies help justify each component.
Technically, the approach is sound.

### Weaknesses
**(a) Novelty Overlap / Incremental Concerns**

Several very recent works (e.g., Drive-R1 Li et al., 2025 and AlphaDrive Jiang et al., 2025) already explore reinforcement learning and reasoning within VLM-based driving.

Likewise, Gen-Drive (Huang et al., 2025) combines diffusion with RL for driving policy optimization.

Given these, the claim of “first to apply reinforcement learning to VLA models” may be over-stated, and the conceptual contribution, though strong in integration, is not entirely novel.

**(b) Ablation Limitations**

No comparison against other RL-enhanced planners such as under identical environments, limiting attribution of improvements to DiffGRPO specifically.

The reward design (PDMS) could be discussed in greater depth—its weighting and sensitivity are unspecified.

### Questions
1. Could the authors provide the weighting coefficients used for the PDMS reward and justify their selection?

2. Is the VLM backbone frozen during diffusion and RL training, or does it receive gradient updates through the planner?

3. What is the exact runtime efficiency (latency per frame) compared to baseline diffusion and text-based methods?

4. Can the authors show qualitative examples of failure modes or collisions to highlight what DiffGRPO specifically improves?

### Soundness
3

### Presentation
3

### Contribution
2
