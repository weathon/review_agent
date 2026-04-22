# SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning

- Avg Score: 4.40
- Decision: Reject
- Scores: 2, 6, 6, 4, 4

## Abstract
Pruning is a typical acceleration technique for compute-bound models by removing computation on unimportant values. Recently, it has been applied to accelerate Vision-Language-Action (VLA) model inference. However, existing methods focus on local information from the current action step and ignore the global context, leading to $>20$% success rate drop and limited speedup in some scenarios. In this paper, we point out **spatial-temporal consistency** in VLA tasks: input images in consecutive steps exhibit high similarity, and propose the key insight that token selection should combine local information with global context of the model. Based on this, we propose SpecPrune-VLA, a training-free, two-level pruning method with heuristic control. **(1) Action-Level Static Pruning**  We leverage global history and local attention to statically reduce visual tokens per action. **(2) Layer-Level Dynamic Pruning** We prune tokens adaptively per layer based on layer-wise importance. **(3) Lightweight Action-Aware Controller** We classify actions as coarse- or fine-grained by the speed of the end effector. Fine-grained actions are pruning-sensitive, so the controller adjusts pruning aggressiveness accordingly. Extensive experiments show that, compared to the high-performing VLA model OpenVLA-OFT, SpecPrune-VLA achieves up to **1.57$\times$** speedup in the LIBERO simulation benchmark across different hardware configurations, and an average speedup of **1.70$\times$** in real-world robotic tasks, with negligible degradation in task success rate.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose SpecVLA - an acceleration method for Vision-Language-Action Models. The acceleration is obtained by A)  Action-level static pruning: the authors compute an attention based score on the visual and text tokens from the middle and the last layer, then the top K visual tokens are chosen, based on the score values. The authors add top-k dynamic tokens, if the corresponding cosine similarity is greater than a threshold value. This combination of tokens is also carried forward to the next step, if two frames are quite similar. B) Layer pruning: The attention scores are modified to include a metric for token confidence. Attention entropy is calculated and is modified to consider a metric for layer confidence. A final importance score for the tokens is calculated to help dynamic pruning. C) Action aware controller: The model shifts tasks based on the rotational and translational velocity.

### Strengths
A> The method is training free

B> The method proposes a multi-level pruning strategy, thereby helping the user to choose the computational requirements.

C> The pipeline combines local and global context  and uses that to make the end to end pipeline efficient.

### Weaknesses
A> The novelty of the method is limited. The proposed contributions have appeared in other contexts earlier. For instance: Attention entropy in LLMs to ensure faster inference [1], choosing transformer layers and tokens based on context [2], calculating token importance score [3], combining local and global features for inference [4], using translation and rotation velocities to determine tasks [5]. 

B> In equation 4, the term “rank-based weight” is vague. It is not clear if it is a concatenation of the earlier scores or is a smaller network.

C> There are a lot of threshold and design choices that need clarifications to understand the optimal choice, for instance: the value of top-K tokens, velocity thresholds, dynamic token threshold. The appendix includes instances of tokens pruned - but it does not show a trade-off between performance and latency, to understand the optimal values.

D> The paper has limited experimental results - the pipeline has been evaluated on a single dataset, but multiple devices. Since there is an intersection of scope with a lot of other efficient transformer works, the paper lacks comparison with other contemporary efforts.


[1] Zhang, Zhisong, et al. "Attention Entropy is a Key Factor: An Analysis of Parallel Context Encoding with Full-attention-based Pre-trained Language Models." CoRR (2024).

[2] Meng, Lingchen, et al. "Adavit: Adaptive vision transformers for efficient image recognition." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[3] Xu, Xuwei, et al. "GTP-ViT: efficient Vision transformers via graph-based token propagation." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2024.

[4] Norouzi, Narges, et al. "Algm: Adaptive local-then-global token merging for efficient semantic segmentation with plain vision transformers." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[5] Qu, Delin, et al. "SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model." CoRR (2025).

### Questions
The paper is an interesting read. I have the following questions:

A>The paper focuses on different varieties of a single dataset: LIBERO. It would be great if the authors could explain the scalability of the approach. I understand from the website of the dataset - that every variety has 10 tasks. Is there a possibility that the accelerations are limited to the lower number of tasks?

B>How do you calculate “rank-based weight” in equation 4?

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
3

### Summary
This paper addresses the inference latency bottleneck in Vision-Language-Action (VLA) models, primarily due to the LLM backbone. The paper presents a well-motivated, comprehensive pruning strategy tailored to VLA models. The primary strengths are its training-free design and the novel action-aware controller, though the complexity and tuning of the multi-stage pruning process are minor weaknesses.

### Strengths
- The paper is built on clear insights specific to the robotics domain: VLA inference is compute-bound, visual information is temporally redundant across action steps, and not all actions are equally sensitive to pruning.

- The lightweight action-aware controller is practical. It uses end-effector velocity to heuristically differentiate between coarse- and fine-grained actions, reducing pruning during high-precision phases. The ablation study confirms that this controller is critical for recovering the success rate lost due to aggressive pruning.

- The method is evaluated thoroughly in both simulation on the LIBERO benchmark and on a physical robot. Demonstrating consistent speedups (1.46x-1.70x) with minimal success rate loss across multiple hardware platforms (NVIDIA A800 and RTX 3090)  strongly supports the claims of effectiveness and generalization.

### Weaknesses
- Concern about the reproducibility and generalizability of the pruning configuration. The method relies on several hyperparameters ($K_G$, $K_D$, $K_L$, $\beta$, $L_{prune}$, etc.), notably the overall pruning ratio $\alpha$, which is set manually for each of the four LIBERO task suites. It is unclear how $\alpha$ should be determined for a new, unseen task without iterative tuning.

- The complexity of the two-level pruning interaction might be high. Tokens are first pruned statically at the action level ($V_{retain} = V_{global} \cup V_{dynamic} \cup V_{local}$), and this retained set is then pruned dynamically at the layer level. This multi-stage filtering, combined with the action-aware controller, makes the final token count difficult to track and analyze.

- The velocity-based frame sampling strategy appears heuristic, with constants chosen empirically. While effective, the paper lacks analysis of how sensitive performance is to these hyperparameters or whether this strategy generalizes across robot morphologies or task distributions.

### Questions
- How sensitive is the performance to the choice of temporal window T in the dynamic frame comparison? Would an adaptive method (e.g., using visual change detection) yield better results? Also, the method reuses attention scores from layers 15 and 32 of the prior step. Why these specific layers? Is there a systematic way to select “global” layers, or is this also empirical?

- The ablation in Table 2 implies the controller adds 1.5ms (72.3ms vs 70.8ms). What is the total latency cost of calculating $V_{local}$ (requiring partial forward passes), $V_{dynamic}$ (patch similarity), and the layer-wise entropy scores?

- EfficientVLA is noted to be originally designed for **diffusion-based action heads**, yet it’s compared against OpenVLA-OFT (which uses an MLP head). While the authors acknowledge this, the performance gap (e.g., 24.4% SR drop in LIBERO-Long) may partly stem from architectural mismatch rather than pruning quality alone.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a pruning method for VLA models. The authors diagnose and show that current VLA acceleration is compute-bound, so it’s intuitive to prune tokens in LLM, thus redundant visual patches as nearby images in the query are often similar in background and importance. They also incorporate global information from the previous step to guide pruning and introduce an action-aware controller adapter. Experiments show that their approach achieves higher speedup than prior methods while achieving better task success rates.

### Strengths
1. The diagnosis is intuitive, especially the demonstration that now accelerations are bound by compute. Thus pruning image tokens is really intuitive, especially since there are redundancies in images.
2. The results are convincing as higher speedup and better success rate are shown in the experiment section.
3. The paper is well written, and the explanation of the method is clear.

### Weaknesses
1. The method mostly relies on hand-tuned heuristics (velocity thresholds, entropy weighting, pruning ratios), it would be great to see extended ablation studies.
2. Do experiments in VLA pruning usually not report success rates and confidence intervals? This can be hard to judge statistical significance.

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents SpecPrune-VLA, a training-free acceleration scheme for Vision-Language-Action (VLA) models that prunes visual tokens using action-level static pruning that reuses global attention statistics from the previous action step and supplements them with local early-layer attention and motion-based dynamic tokens. They also adopt payer-level dynamic pruning driven by rank-weighted attention and per-layer attention-entropy and a lightweight action-aware controller that reduces pruning during fine-grained manipulation and increases it during coarse motion.
The evaluation on LIBERO tasks shows 1.46$\times$ average end-to-end speedup over OpenVLA-OFT and 1.70$\times$ on real robot.

### Strengths
- The efficiency for VLA is essential, as it highly impacts the performance of real robot controlling systems.
- The proposed two-level pruning, combining global and local information, is well-motivated and beneficial.
- The coarse/fine-grained switching based on real scenarios is helpful to improve the performance.

### Weaknesses
- The paper lacks an analysis of the importance/confidence scores. How are the layer confidences distributed?
- Ablation in Tab.2 is incomplete. e.g., there is no ablation on global only vs local only, entropy-based layer confidence vs rank-only weighting.
- Some annotations are potentially confusing. For example,  line-227 $I_1,I_2$ might be clearer if written as $I_t,I_{t+1}$

### Questions
- Sec 3
  - How many tokens per image in total in Fig.3?
- Sec 4
  - In Eq.1, how are the layers selected?
  - In Eq.1, what does the subscript  $i$ in $V_i$ denote?
  - In $T=\lfloor b + k\cdot v \rfloor + 4$, both translational velocity and rotational velocity seem to be included in $v$. Are these two types of velocity handled in the same way, or is there a difference in how they are applied?
  - For sec.4.1.2, could you discuss the computational complexity and the overhead introduced by the method?
  - Have you tried different similarity metrics?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SpecPrune-VLA, a training-free two-level token pruning framework designed to accelerate inference in VLA models. It leverages global context reuse across consecutive action steps and local speculative attention to select important visual tokens, while a lightweight action-aware controller dynamically adjusts pruning aggressiveness based on robot end-effector velocity. Experiments on the LIBERO benchmark and real-world robotic platforms show that SpecPrune-VLA achieves up to 1.57× speedup (A800 GPU) and 1.7× real-world speedup with negligible task success rate degradation (<1%) compared to OpenVLA-OFT.

### Strengths
1. Improving the efficiency of VLA is an important research topic. 
2. The writing of the paper is quite clear and easy to follow.
3. The method is training-free, making it easily integrable into existing VLA pipelines.

### Weaknesses
1. The mathematical formulation of global/local fusion and the entropy-confidence weighting is heuristic.

2. Too many hyperparameters are involved in the proposed method. The pruning heavily depends on multiple manually tuned thresholds (α, τ, β, K values).

3. Experiments are limited to OpenVLA-OFT.

### Questions
1. How do you ensure that reusing global attention maps from previous steps does not propagate outdated or erroneous importance information, especially in tasks with dynamic camera motion or sudden object appearance?
2. Many parameters (τ, α, β, K-values, pruning thresholds) are hand-tuned. Have you tested the robustness of performance under different hyperparameter configurations? How sensitive is your approach to these values?
3. The velocity-based controller seems tailored to a specific robot arm setup. How general is this to other robotic systems or action spaces?
4. For 4.1.1, the attention from the 15th and 32nd layer are selected to indicate the performance. What is the rationale? Have you tried other layers?

### Soundness
2

### Presentation
2

### Contribution
2
