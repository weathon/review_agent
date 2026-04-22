# RRVF: Visual Reinforcement Learning with Reasoning, Rendering, and Visual Feedback

- Avg Score: 5.33
- Decision: Reject
- Scores: 8, 4, 4

## Abstract
Multimodal Large Language Models (MLLMs) exhibit impressive performance across various visual tasks. Subsequent investigations into enhancing their visual reasoning abilities have significantly expanded their performance envelope. However, a critical bottleneck in the advancement of MLLMs toward deep visual reasoning is their heavy reliance on curated image-text supervision. To solve this problem, we introduce a novel framework, “Reasoning-Rendering-Visual-Feedback” (RRVF), that enables MLLMs to learn complex visual reasoning from only raw images. This framework builds on the “Asymmetry of Verification” principle, i.e., verifying the rendered output against the source image is substantially easier than performing deep visual reasoning to generate a faithful, structured representation such as code. We demonstrate that this relative ease provides an ideal reward signal for optimization via Reinforcement Learning (RL), thereby reducing reliance on image-text supervision. RRVF implements a closed-loop iterative process encompassing reasoning, rendering, and visual feedback components, enabling the model to perform complex reasoning, including self-correction through multi-turn interactions. This process is optimized end-to-end using the GRPO algorithm. Extensive evaluations are conducted on image-to-code generation across two diverse domains: data charts and web interfaces. The RRVF-trained model not only outperforms existing similarly sized open-source MLLMs and supervised fine-tuning baselines but also exhibits superior generalization. Notably, the model outperforms the more advanced MLLM used to generate visual feedback during training.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the image-to-code problem via "Reasoning-Rendering-Visual-Feedback". This framework does not require manually labeled text instructions. It simply renders the generated code and uses an MLLM to verify the original image and the rendered image. Experiments under various protocols demonstrate that the proposed method brings significant improvements over baselines.

### Strengths
1. This paper is overall well-written and easy to follow.
2. Both the motivation and the solution are quite clear and reasonable.
3. Improvements are quite significant.

### Weaknesses
I only have one minor concern:

1. The generalization of this method. The trained model is excellent at code generation. How about other reasoning-related benchmarks, e.g, mathvista, mathverse, logicvisa, etc. Do these advanced code generation capabilities implicitly contribute to better general reasoning capabilities?

### Questions
N/A.

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
This paper proposes a reinforcement learning framework named "Reasoning-Rendering-Visual-Feedback" that only utilizes image data rather than image-text supervision. The framework only utilizes outcome reward by the GRPO algorithm. The training framework enables models to have multi-turn reasoning and tool-call abilities. The experiments show that the framework brings more stable improvement than SFT on Qwen2.5-VL-7B.

### Strengths
1. The paper designs the understanding of icon-type images as a rendering and verifying process, which is an interesting and reasonable idea.

2. The paper designs the training of multimodal reasoning as a multi-round rendering and verifying process, so that the model only needs to rely on images and does not require additional text annotations. This alleviates the dependence of multimodal reasoning training on text annotations.

3. The authors' experiments show that the proposed framework brings more significant advantages than supervised training (SFT).

### Weaknesses
1. The proposed framework requires multiple rounds of inference and tool calls during both training and inference. How long does training take? How much does inference time increase compared to not using tools?

2. The applicability of this framework in visual inference is limited. It can only be used in chart-to-code and web-to-code scenarios. The authors did not explore how this framework adapts to general image inference.

3. In Tables 1 and 2, did the authors only conduct in-the-domain experiments on the ChartMimic and Plot2Code benchmarks? I think generalization experiments should be added to demonstrate that reinforcement learning algorithms like GRPO, in addition to performance improvements, also enhance the model's generalization ability for SFT.

4. The authors only conducted SFT and GRPO experiments on Qwen2.5-VL-7B-Instruct, without experimenting with other baseline models, failing to demonstrate the framework's generality.

5. The authors did not conduct ablation experiments on the proposed framework. For example, using different maximum rounds, removing certain proposed components, etc., can validate the effectiveness of the entire framework.

### Questions
1. Refer to the issues raised in the weakness section.
2. How many steps were trained for the reinforcement learning fine-tuning? The curve shown in Figure 3 only shows 100 training steps, which seems to indicate the possibility of overfitting.

### Soundness
2

### Presentation
2

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
The paper proposes a novel paradigm for multimodal reasoning training by eliminating the need for paired image-text/code supervision. Instead, the model learns solely from raw images via iterative rendering and visual comparison. This formulation shifts the paradigm from supervised imitation learning to self-supervised verification-driven RL. The authors introduce a closed-loop reasoning framework that iteratively generates code, executes it, and incorporates visual feedback for refinement. This mechanism allows the model to progressively correct its outputs based on rendered results rather than relying solely on single-pass generation.

### Strengths
1. The paper is well-written and easy to follow.

2. The research direction is impactful because collecting high-quality program annotations for visual tasks is expensive and often subjective.

3. The model achieves a high code execution rate on ChartMimic and performs competitively on WebSight, without requiring paired text supervision. It also shows better performance than supervised fine-tuning and comparable open-source baselines.

### Weaknesses
1. The framework depends on multi-round generation and external tool execution during training and inference. This raises questions regarding computational cost, latency, and scalability. The paper does not report training time, tool-call frequency, or inference speed compared to standard single-pass models.

2. The method is only evaluated on chart-to-code and web-to-code settings, which are structured scenarios with clearly defined rendering engines. It remains unclear whether the approach can extend to broader visual inference tasks (e.g., general scene understanding, reasoning, VQA).

3. The method exhibits a strong reliance on reward shaping and prompt design. The format reward, tool-use reward, and their weighting require manual tuning, yet the paper does not provide systematic analysis of reward stability or robustness. Additionally, no ablation or sensitivity study is conducted on the reward components, making it difficult to assess how much each part contributes to the final performance.

### Questions
1. Training cost & efficiency: How long does training take, and how many RL steps are used? What is the total computational budget?

2. Inference overhead: How much slower is inference when using the iterative tool-based framework? Is there a version that can operate efficiently without tool calls at test time?

3. Generalization evaluation: Did you conduct any experiments on tasks outside chart/web code generation? Could you add out-of-domain tests to verify generalization benefits of GRPO over SFT?

### Soundness
2

### Presentation
3

### Contribution
2
