# Visual Jigsaw Post-Training Improves MLLMs

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Reinforcement learning based post-training has recently emerged as a powerful paradigm for enhancing the alignment and reasoning capabilities of multimodal large language models (MLLMs). While *vision-centric* post-training is crucial for enhancing MLLMs’ intrinsic understanding of visual signals, current post-training paradigms are predominantly *text-centric*, where dense visual inputs are only leveraged to extract sparse cues for text-based reasoning. There exist a few approaches in this direction, however, they often still rely on text as an intermediate mediator or introduce additional visual generative designs. In this work, we introduce **Visual Jigsaw**, a generic *self-supervised* post-training framework designed to strengthen visual understanding in MLLMs. Visual Jigsaw is formulated as a general ordering task: visual inputs are partitioned, shuffled, and the model must reconstruct the visual information by producing the correct permutation in natural language. This naturally aligns with reinforcement learning from verifiable rewards (RLVR), requires no additional visual generative components, and derives its supervisory signal automatically without any annotations. We instantiate Visual Jigsaw across three visual modalities, including images, videos, and 3D data. Extensive experiments demonstrate substantial improvements in fine-grained perception, temporal reasoning, and 3D spatial understanding. Our findings highlight the potential of self-supervised vision-centric tasks in post-training MLLMs and aim to inspire further research on vision-centric pretext designs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper considers Visual Jigsaw as a general self-supervised task for post-training. Specifically, it has three variants, including image jigsaw, 3D jigsaw (ranking from the closest to the farthest), and video jigsaw. Empirically, initialized from Qwen2.5-VL-7B-Instruct, the proposed method achieves improvements over baselines separately. Moreover, when incorporating with a text-centric reasoning MLLM (ThinkLite-VL), the proposed image jigsaw post-training task still brings improvements on vision-centric benchmarks and maintains the performance on reasoning-oriented benchmarks.

### Strengths
1. This paper is well-written and easy to follow.
2. The motivation of this paper is clear and reasonable.
3. Empirical studies are partially sufficient.

### Weaknesses
1. The proposed three types of jigsaw tasks are evaluated *separately*. It is strongly encouraged to combine them and find out whether they can work together.
2. Qualitative examples of the reasoning pathways of both (1) when solving the proposed jigsaw problems and (2) answering multiple-choice questions are missing. Moreover, the underlying reason why this pre-text task is beneficial for VQA seems to be missing. An analysis of the reasoning pathways might be helpful.

Some suggestions on the claims:
1. The claim "While these jigsaw-style approaches provide structural ordering signals, they have generally shown weaker performance compared to more dominant approaches" is actually wrong. There are actually some great works [1, 2, 3] that demonstrate jigsaw-like self-supervised pre-training tasks are at least comparable with contrastive learning and masked image modeling. Therefore, adding discussions on these works and clarifying the claim is important.
2. The claim"Current post-training paradigms are predominantly text-centric, where dense visual inputs are only leveraged to extract sparse cues for text-based reasoning" is also not that appropriate. Recent o3-like approaches are definitely not text-centric reasoning methods. Discussions should include both early approaches like CogCoM [4] and Dyfo [5], recent approaches like DeepEyes [6], VGR [7], Pixel-Reasoner [8], and TreeVGR [9]. The advantages of visual jigsaw compared with these o3-like approaches should be discussed.

**References**

[1] Position prediction as an effective pretraining strategy. ICML, 2022

[2] DropPos: Pre-Training Vision Transformers by Reconstructing Dropped Positions. NeurIPS, 2023.

[3] Location-aware self-supervised transformers. WACV, 2024.

[4] CogCoM: A Visual Language Model with Chain-of-Manipulations Reasoning. ICLR 2025.

[5] Dyfo: A training-free dynamic focus visual search for enhancing lmms in fine-grained visual understanding. CVPR 2025.

[6] DeepEyes: Incentivizing" Thinking with Images" via Reinforcement Learning. arXiv 2025.

[7] Vgr: Visual grounded reasoning. arXiv 2025.

[8] Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv 2025.

[9] Traceable evidence enhanced visual grounded reasoning: Evaluation and methodology. arXiv 2025.

### Questions
N/A

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
5

### Summary
This paper introduces a novel post-training task called Visual Jigsaw, which encourages MLLMs to learn the inherent order within visual inputs. An RL-based supervision strategy is employed to train the model. Experiments on image, video, and 3D benchmarks demonstrate that the proposed method significantly improves performance.

### Strengths
1.	The paper presents an effective post-training framework for enhancing MLLMs.

2.	The idea of leveraging a self-supervised objective to strengthen the visual understanding capability of MLLMs is novel and promising.

3.	The proposed approach achieves strong performance across diverse benchmarks, including image, video, and 3D tasks.

4.	The manuscript is well structured, clearly written, and easy to follow.

### Weaknesses
1.	The Visual Jigsaw task employs a relatively simple supervised objective. Although the experimental results across multiple benchmarks are promising, the underlying mechanism of how this form of supervision contributes to model improvement remains unclear. It would strengthen the paper to include an analysis of what is learned during post-training and how this supervision enhances visual comprehension.

2.	The experiments are conducted on only a single MLLM. Including additional baseline models would help demonstrate the generalizability and robustness of the proposed Visual Jigsaw framework.

3.	The Visual Jigsaw post-training is performed using specific individual datasets rather than the original MLLM training data. It would be helpful to clarify the rationale behind this choice and provide guidelines for selecting appropriate post-training data.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

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
This paper proposes Visual Jigsaw , a self-supervised post-training framework designed to enhance the intrinsic visual understanding of MLLMs. The core idea is to formulate a visual ordering task, the model is presented with shuffled parts of a visual input and must generate a text-based permutation to reconstruct the original order. This approach cleverly leverages RLVR, specifically using GRPO, to optimize the model. The authors instantiate this method across images, videos, and 3D data . Extensive experiments on a wide range of benchmarks show that this post-training stage significantly improves fine-grained perception, temporal reasoning, and 3D spatial understanding, demonstrating the method's generality and effectiveness.

### Strengths
The paper presents an elegant and effective idea. It repurposes a classic self-supervised task (jigsaw puzzles) into a post-training stage for modern MLLMs. The framework is simple, requires no architectural modifications or extra generative modules, and is broadly applicable.

### Weaknesses
The improvement effect of some benchmarks is not significant.

### Questions
1. Have you tried expanding on the harder puzzle tasks (like 4x4)?
2. Given that some improvements are modest, the results would be more credible if variance  of evaluation are reported.

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
4

### Summary
This paper introduces Visual Jigsaw, a self-supervised post-training framework designed to enhance vision-centric understanding in MLLMs. The method formulates a general visual ordering task, partitioning and shuffling images, videos, or 3D inputs and requires the model to predict the correct permutation via natural-language output without modifying architecture or adding visual generators. Experiments on Qwen2.5-VL show improvements across fine-grained perception, temporal reasoning, and 3D spatial understanding benchmarks.

### Strengths
- The jigsaw ordering task provides a simple yet effective way to reinforce visual perception in MLLMs, avoiding generative components.

- Applying the post-training principle to images, videos, and 3D data convincingly demonstrates versatility.

- Evaluation on various benchmarks covering visual perception, temporal reasoning, and 3D geometry provide good empirical support.

- SFT vs RL comparison, jigsaw-difficulty analysis, and transfer to reasoning models (ThinkLite-VL) strengthen the claims.

- The paper is well written and figures are intuitive.

### Weaknesses
- The paper has limited conceptual novelty. The idea extends classical self-supervised “jigsaw” pretexts (Noroozi & Favaro 2016) into RL post-training. The novelty mainly lies in adapting it to MLLM post-training rather than the task itself.

- Missing related works of reconstruction-based methods in MLLMs and discussion in paper: Recent works like X-Former[1] explicitly combines contrastive and masked-reconstruction objectives with frozen encoder and decoders to improve visual understanding during pre-training with less data. In contrast, Visual Jigsaw uses structural ordering without dense pixel reconstruction. The paper should discuss this design trade-off in detail, why post-training with ordering might achieve similar benefits, and whether the two paradigms (reconstruction vs ordering) could be complementary.

- The paper omits discussion on training overhead for Jigsaw compared to SFT.

- Missing comparison to SoTA for image and video benchmarks.

- The paper could further analyze reward sensitivity and scalibility (beyond 3×3 grid/6-clips).

- All results rely on Qwen2.5-VL; would be good to show on other model architectures like LLaVA/Blip to confirm generality

- Missing qualitative analysis. Visual reasoning traces (< think > outputs) could better substantiate the claim of improved visual understanding.



[1] X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs. Sirnam Swetha, Jinyu Yang, Tal Neiman, Mamshad Nayeem Rizve, Son Tran, Benjamin Yao, Trishul Chilimbi, Mubarak Shah. ECCV 2024

### Questions
- How sensitive are the improvements to the number of jigsaw pieces (K)? Does performance saturate or degrade with grid size (4×4, 5×5) grids or > 6 video clips?

- The reward combines partial correctness with a discount factor γ = 0.2. How sensitive are results to this value? Did the authors explore other reward formulations ?

- What is the total cost of post-training ?

### Soundness
3

### Presentation
3

### Contribution
2
