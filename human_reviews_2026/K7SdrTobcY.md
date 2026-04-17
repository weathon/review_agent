# VidBridge-R1: Bridging QA and Captioning for RL-based Video Understanding Models with Intermediate Proxy Tasks

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
The "Reason-Then-Respond" paradigm, enhanced by Reinforcement Learning, has shown great promise in advancing Multimodal Large Language Models. However, its application to the video domain has led to specialized models that excel at either question answering (QA) or captioning tasks, but struggle to master both. Naively combining reward signals from these tasks results in mutual performance degradation, which we attribute to a conflict between their opposing task natures. To address this challenge, we propose a novel training framework built upon two intermediate proxy tasks: DarkEventInfer, which presents videos with masked event segments, requiring models to infer the obscured content based on contextual video cues; and MixVidQA, which presents interleaved video sequences composed of two distinct clips, challenging models to isolate and reason about one while disregarding the other. These proxy tasks compel the model to simultaneously develop both holistic, divergent understanding and precise, convergent reasoning capabilities. Embodying this framework, we present VidBridge-R1, the first versatile video reasoning model that effectively bridges the paradigm conflict. Extensive experiments show that VidBridge-R1 achieves significant performance gains on both QA and captioning within one model, demonstrating the efficacy of our approach in fostering more generalizable and powerful video understanding models. All code, models, and data will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes two kinds of data augmentation methods for video reasoning tasks, namely DarkEventInfer and MixVidQA, enabling the model to predict the answer under the augmented video. Experiments show the effectiveness of the proposed method.

### Strengths
1. The idea sounds reasonable. Effective data augmentation methods are always needed.
2. The writing is clear.
3. The presented results are promising.

### Weaknesses
1. As an data augmentation method, whether the method is scalable is questionable.
2. Lack of detailed analysis, why the reasoning models show poor performance  on video general understanding and captioning tasks, but the model trained with the augmented reasoning data bring clear improvement. Is this a specific kind of overfitting or data leakage?

### Questions
Please refer to the weaknesses.

### Soundness
3

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
3

### Summary
The paper focuses on post-training a video LLM for both QA and captioning, arguing naively combining the reward signals for both tasks leads to suboptimal performance. Two proxy tasks are presented:
DarkEventInfer, which presents videos with masked event segments, forcing the model to use context to infer the missing content. 
MixVidQA, which interleaves sequences from two different videos and challenges the model to reason about one while ignoring the other. 
The resulting VidBridge-R1 model is trained with these proxy tasks alongside traditional QA and captioning tasks. Experiments show that VidBridge-R1 achieves sota performance across a various QA and captioning benchmarks. The paper also includes an ablation study and analysis of the training dynamics.

### Strengths
- Well motivated problem statement and intuitive proposals
- Sota performance on both captioning and QA tasks

### Weaknesses
- The ablation in Table 3 eludes important rows showing the benefit of the proposed tasks together with the caption task, as well as the row with VidMixQA and not DarkEventInfer
- Each task being based on different data makes it difficult to disentangle the benefits of the task vs the data

### Questions
- How much compute do the new tasks add compared to the standard ones?
- About the training dynamics study: the initial dip in captioning performance on Dream is striking. How sensitive is the final model performance to the sampling ratio between the four tasks during training? 
- Is there any other evidence than the downstream performance for the conflict between the QA and captioning task?

### Soundness
2

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
5

### Summary
This paper addresses the conflict between video QA and video captioning under reinforcement learning (RL)–based training within the "Reason-Then-Respond" paradigm. The authors argue that QA (convergent reasoning) and captioning (divergent generation) inherently pull model optimization in opposite directions, causing performance degradation when trained jointly. To mitigate this conflict, the paper introduces two proxy tasks: 1) DarkEventInfer (predict masked video segments based on context) and 2) MixVidQA (answer questions about one of two interleaved video clips). Integrating these proxy tasks, the authors propose VidBridge-R1, a video understanding model trained via GRPO RL without an SFT stage. Experiments across general video QA, video reasoning, and captioning benchmarks demonstrate improved overall performance compared with several existing RL-based video MLLMs.

### Strengths
1. Overall, the paper clearly identifies a meaningful optimization conflict between QA and captioning under RL training and motivates why naive multi-task RL leads to mutual degradation.
2. The two proxy tasks (DarkEventInfer and MixVidQA) are intuitively aligned with promoting both holistic contextual reasoning and selective information grounding, and their construction process is described with adequate clarity and filtering steps.
3. The experimental evaluation is comprehensive, covering general video QA, reasoning QA, captioning benchmarks, and held-out test sets for the proxy tasks.

### Weaknesses
1. While the proxy tasks appear to be effective, it is still not fully demonstrated "why" these particular tasks are optimal among possible bridging tasks. A brief analysis or comparison with alternative proxy formulations (e.g., temporal reordering tasks, masked key-frame inference) would provide more insights to the readers.
2. The experiments are mainly conducted on Qwen2.5-VL-7B-Instruct. It is encouraged to discuss how generalizable the method is to stronger or smaller video-language backbones, e.g., 2B or 32B levels.
3. The code of this project is not uploaded for review. And the paper does not contain the Reproducibility Statement, yielding strong concerns in the reproducibility issue and the actual contribution to the community. I believe that it would be hard to reproduce the results given the limited implementation details in the paper.

### Questions
Please refer to the weakness section for my questions.

### Soundness
3

### Presentation
3

### Contribution
3
