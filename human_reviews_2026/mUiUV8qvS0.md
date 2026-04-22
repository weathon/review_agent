# Token-level Inference-Time Alignment for Vision-Language Models

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2

## Abstract
Vision-Language Models (VLMs) have become essential backbones of modern multimodal intelligence, yet their outputs remain prone to hallucination-plausible text misaligned with visual inputs. Existing alignment approaches often rely on expensive fine-tuning with annotated preference data or sequence-level inference strategies  that provide only coarse, delayed feedback. To overcome these limitations, we present TITA (Token-level Inference-Time Alignment), a lightweight framework that freezes the base VLM and instead trains a reward model to approximate its distribution. During inference, implicit preference signals are extracted as log-probability ratios between the reward model and the target VLM, yielding dense autoregressive feedback. This formulation can be viewed as an inference-time variant of Direct Preference Optimization (DPO), providing token-level corrective signals without retraining the backbone. Extensive evaluations on LLaVA-1.5-7B and 13B show consistent gains across 12 benchmarks, with improvements of 8.6% on MMVet and 6.7% on POPE, indicating stronger general understanding and reduced hallucinations. Additional experiments on Qwen2.5-VL-7B and DeepSeek-VL2-27.5B show comparable gains, especially in hallucination reduction and VQA accuracy, while incurring negligible inference overhead. Code is available at: https://anonymous.4open.science/r/TITA-BEC6.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes TITA, a token-level inference-time alignment framework for large vision–language models (VLMs). Instead of relying on costly fine-tuning or sequence-level reward reranking, TITA learns a small autoregressive reward model to guide decoding at the token level, using log-probability ratios between the reward model and the base VLM as implicit preference signals. The approach provides dense feedback during generation and shows consistent hallucination reduction and general VQA improvement across LLaVA, Qwen2.5-VL, and DeepSeek-VL2 families with minimal inference overhead.

### Strengths
1. The idea of bringing Direct Preference Optimization into inference-time, at the token level, is both conceptually neat and practical. It bridges the gap between coarse sequence-level feedback and expensive retraining.
2. The experiments are broad (12 benchmarks, several VLM families) and show clear, consistent gains in hallucination suppression and visual reasoning accuracy with very low additional cost.
3.The figures and algorithm explanations are intuitive; the comparisons with prior training-time and inference-time alignment frameworks (Fact-RLHF, CSR, SeVa, Critic-V) are fair and informative.

### Weaknesses
1. It would be valuable to analyze how the reward model’s scale or quality influences performance — for example, comparing smaller versus larger reward models to verify robustness of token-level alignment.
2. While the paper shows cross-model adaptability (7B to 27B), it would be insightful to analyze how reward model quality affects alignment. For instance, does using a smaller or noisier reward model degrade token-level signals significantly?

### Questions
See weakness

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
4

### Summary
This paper proposes TITA, a lightweight inference-time alignment framework designed to efficiently suppress hallucinations in VLMs by providing dense, token-level feedback signals. The core idea of TITA is to combine a frozen base VLM with a trained lightweight reward model, using the log-probability ratio between them to guide the decoding process. It constructs preference data via a self-supervised multi-view fusion approach. Experiments demonstrate that TITA significantly reduces hallucinations (e.g., +6.7% on POPE) and improves VQA performance across models such as LLaVA, Qwen2.5-VL, and DeepSeek-VL2, while introducing negligible inference overhead.

### Strengths
(1) TITA innovatively transforms sequence-level rewards into token-level signals, addressing the issues of feedback delay and high computational cost in existing methods. By directly guiding the decoding process without the need for sequence re-ranking, it enables timely intervention against hallucinations with extremely low training cost.

(2) TITA is a plug-and-play method that does not modify the parameters of the base model, giving it strong generality and allowing it to be flexibly applied to VLMs of different scales and architectures.

### Weaknesses
(1) TITA relies on image augmentation and response fusion to generate the “winning” responses. This mechanism primarily captures the comprehensiveness of visual elements, which may make it difficult to learn deeper semantic or complex reasoning errors that cause hallucinations in VLMs. As a result, the reward model may be limited in capturing more sophisticated preference patterns.

(2) The proposed method is highly sensitive to the scaling factor lambda. As shown in Figure 3 of the paper, the performance peaks at λ  = 0.6 and drops rapidly afterward. This indicates that parameter tuning may be required when applying the method, potentially even across different tasks.

(3) The reward model is trained using a sequence level BT loss to learn overall preferences between a winner (yw) and a loser (yl). However, during inference, it is used to provide token-level guidance for next-token generation. This conversion from sequence-level preference to token-level guidance may theoretically introduce inconsistencies especially in long-sequence generation, where locally optimal token choices may not guarantee the best overall sequence quality.

### Questions
Does TITA-guided inference alter the model’s attention distribution? What is the quantitative relationship between this change and the reduction in hallucination rates?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces TITA, a test-time alignment framework designed to mitigate hallucinations in Vision-Language Models (VLMs). The method employs a fine-tuned, lightweight reward model to guide the decoding at the token level of the target VLM during inference. Compared with existing approaches, the authors claim that TITA achieves superior effectiveness and efficiency. Experimental results demonstrate that the proposed framework yields significant overall performance improvements over the baselines.

### Strengths
- Clear and Well-Structured: The paper is well-organized, with detailed explanations of the preliminary, intuition, and methodology.

- Superiority in Alignment: The experimental results demonstrate that the proposed method achieves the overall best performance on the general VQA and hallucination benchmarks compared to the baselines.

### Weaknesses
- The backbones used in the experiments are somewhat outdated, particularly since the main results presented in Table 2 are based on the LLaVA 1.5 series models. While I acknowledge that the authors also provide results using Qwen-2.5-VL and DeepSeek-VL2, a more comprehensive evaluation using such recent and stronger VLMs would strengthen the manuscript.

- As a highly competitive and rapidly evolving research area, VLM alignment should provide evaluation against up-to-date methods and backbones. However, the paper primarily compares its approach with relatively outdated baselines (from 2023–2024) and employs older backbone models. This is difficult for me to fully assess the effectiveness of the proposed method.

- The presentation accuracy could be further improved. For example, TITA is not the best-performing method on MMB when using LLaVA-1.5-7B. SeVa achieves higher performance in this setting.

- The core techniques incorporated in TITA are based on well-established principles from prior works. While the derivation is clear and well-presented, it does not introduce fundamentally new concepts but rather applies existing methods in a different context.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
