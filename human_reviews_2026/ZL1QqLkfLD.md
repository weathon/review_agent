# Multi-Modal Action Recognizer Bridges Human Motion Generation and Understanding

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Human action recognition and motion generation are two active research problems in human-centric computer vision, both aiming to align motion with textual semantics. However, most existing works study these two problems separately, without uncovering the bidirectional links between them, namely that motion generation requires semantic comprehension. This work investigates unified action recognition and motion generation by leveraging skeleton coordinates for both motion understanding and generation. We propose Coordinates-based Autoregressive Motion Diffusion (CoAMD), which synthesizes motion in a coarse-to-fine manner. As a core component of CoAMD, we design a Multi-modal Action Recognizer (MAR) that provides semantic guidance for motion generation. Our model can be applied to four important tasks, including skeleton-based action recognition, text-to-motion generation, text–motion retrieval, and motion editing. Extensive experiments on 13 benchmarks across these tasks demonstrate that our approach achieves state-of-the-art performance, highlighting its effectiveness and versatility for human motion modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Coordinates-based Autoregressive Motion Diffusion (CoAMD), a unified framework that bridges human motion generation and skeleton-based action recognition. The framework leverages a multi-modal motion representation, decomposing absolute joint coordinates into joint, bone, and motion streams. During autoregressive generation, the MAR computes semantic alignment scores for partially generated motions, and the gradients of these scores steer the diffusion model’s sampling trajectory to enhance text-motion alignment. Extensive experiments on text-to-motion generation, text–motion retrieval, and motion editing demonstrate that CoAMD achieves state-of-the-art performance.

### Strengths
The paper unifies motion generation and action recognition in a single framework, named CoAMD, where a powerful action recognizer provides real-time semantic guidance during the diffusion process. Extensive experiments on 13 benchmarks demonstrate that CoAMD achieves state-of-the-art performance.

### Weaknesses
Previous research has established unified models for motion generation and understanding, such as LMM[1] and MotionLLM[2]. Although these prior works primarily rely on RGB videos or images as input for motion understanding, RGB video data is more easily accessible and widely applicable in real-world scenarios compared to joint-based motion data. This prevalence of RGB-based approaches somewhat diminishes the novelty of this work.
[1] Zhang M, Jin D, Gu C, et al. Large motion model for unified multi-modal motion generation[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 397-421.
[2] Chen L H, Lu S, Zeng A, et al. Motionllm: Understanding human behaviors from human motions and videos[J]. arXiv preprint arXiv:2405.20340, 2024.

### Questions
1. Why is there a discrepancy between the performance of MoMask reported in this paper on the HumanML3D and KIT datasets and the results presented in the original MoMask paper?
2. While this work introduces a model that unifies motion generation and action recognition, the comparative analysis in text-to-motion generation is limited to specialized, task-specific baselines, lacking comparisons with contemporary unified frameworks (e.g., LMM, MotionAnything).
[1] Zhang M, Jin D, Gu C, et al. Large motion model for unified multi-modal motion generation[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 397-421.
[2] Zhang Z, Wang Y, Mao W, et al. Motion anything: Any to motion generation[J]. arXiv preprint arXiv:2503.06955, 2025.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This study proposes a Coordinates-based Autoregressive Motion Diffusion model (CoAMD), which achieves coarse-to-fine human motion synthesis guided by a Multi-modal Action Recognizer (MAR) at the semantic level. The model can be applied to four tasks: action recognition, text-to-motion generation, text–motion retrieval, and motion editing, and achieves state-of-the-art performance across 13 benchmarks.

### Strengths
1. Coarse-to-fine motion generation improves the quality of motion generation.
2. Excellent cross-task performance and strong generalization ability.

### Weaknesses
Major Issues:
1.The core idea of the paper lies in the mutual enhancement between generation and understanding tasks. However, this idea has already been adopted in several previous works [1][2], leading to insufficient novelty.
2.In the multimodal motion recognizer, compared with the approach that divides the human body into five parts and uses contrastive learning to align text and motion, what advantages does the proposed method offer? Is it actually a subset of such approaches? Furthermore, could the overlap among joint points, skeleton structures, and motion dynamics introduce interference in motion generation?
3.The masking strategy in COORDINATES-BASED AUTOREGRESSIVE MOTION DIFFUSION originates from MoMask, and has also been used in MMM, BAMM, and LaMP. What advantages does the proposed method have compared to these approaches?
4.The comparison methods in Table 1 are incomplete. The paper focuses on the mutual enhancement and fine-grained modeling between behavior generation and understanding, yet lacks comparison with related works such as StableMoFusion, LaMP, KinMo, MG-MotionLLM, UniMotion, and Motion-Agent.
5.There are concerns about the experimental results: in Table 2, the FID of MoMask is 0.372 & 0.204 in the original paper, but 0.523 in this paper. This discrepancy raises doubts about the correctness and reliability of the experimental results. If the MoMask results are accurate, the proposed CoAMD method would not achieve the best performance on the KIT-ML dataset.
6.In the section on limitations and future work, the paper fails to address the critical aspect of physical plausibility. Can the current method ensure the physical rationality of generated motions?

Minor Issues:

1.In the introduction, there are too many commas, making the sentences disjointed. Additionally, please include proper citations when discussing application scenarios.
2.In Figure 2, the encoder icon is incorrect; please clarify what “h” represents.
3.Highlight or bold the best experimental results for better readability.

References:
[1] Cycle-Consistent Learning for Joint Layout-to-Image Generation and Object Detection
[2] Dual Reciprocal Learning of Language-based Human Motion Understanding and Generation

### Questions
Major Issues:
1.The core idea of the paper lies in the mutual enhancement between generation and understanding tasks. However, this idea has already been adopted in several previous works [1][2], leading to insufficient novelty.
2.In the multimodal motion recognizer, compared with the approach that divides the human body into five parts and uses contrastive learning to align text and motion, what advantages does the proposed method offer? Is it actually a subset of such approaches? Furthermore, could the overlap among joint points, skeleton structures, and motion dynamics introduce interference in motion generation?
3.The masking strategy in COORDINATES-BASED AUTOREGRESSIVE MOTION DIFFUSION originates from MoMask, and has also been used in MMM, BAMM, and LaMP. What advantages does the proposed method have compared to these approaches?
4.The comparison methods in Table 1 are incomplete. The paper focuses on the mutual enhancement and fine-grained modeling between behavior generation and understanding, yet lacks comparison with related works such as StableMoFusion, LaMP, KinMo, MG-MotionLLM, UniMotion, and Motion-Agent.
5.There are concerns about the experimental results: in Table 2, the FID of MoMask is 0.372 & 0.204 in the original paper, but 0.523 in this paper. This discrepancy raises doubts about the correctness and reliability of the experimental results. If the MoMask results are accurate, the proposed CoAMD method would not achieve the best performance on the KIT-ML dataset.
6.In the section on limitations and future work, the paper fails to address the critical aspect of physical plausibility. Can the current method ensure the physical rationality of generated motions?

Minor Issues:

1.In the introduction, there are too many commas, making the sentences disjointed. Additionally, please include proper citations when discussing application scenarios.
2.In Figure 2, the encoder icon is incorrect; please clarify what “h” represents.
3.Highlight or bold the best experimental results for better readability.

References:
[1] Cycle-Consistent Learning for Joint Layout-to-Image Generation and Object Detection
[2] Dual Reciprocal Learning of Language-based Human Motion Understanding and Generation

### Soundness
3

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
3

### Summary
The paper argues that motion generation and recognition both build alignment between text and motion and can complement each other. Because of it, the paper proposed CoAMD, a motion generation model with a recognition model that provides guidance. Experiments were done on multiple datasets and different tasks to show the effectiveness and versatility of the proposed method.

### Strengths
The paper provides a new perspective of viewing action recognition and motion generation, and it builds a novel framework to combine models for the two tasks and reaches better performances.

### Weaknesses
1. The performance of baseline models in Table 1 is worse than that reported in the original papers. For example, in MoMask, the R-Presicion was 0.521, 0.713, 0.807 but what you listed was 0.493, 0.686, 0.784. I wonder where exactly these numbers come from?
2. You used LLM to extract core action verbs. Can you elaborate on the necessity of this step? An LLM is costly for such a task.
3. From the metrics you provided, the improvement of MAR seems to be very limited compared with your model without MAR. 
4. Minor issues: Equation 5 is inconsistent with Equation 11 in the appendix. The shape of the encoder in Figure 2 is inverted — it goes from narrow to wide, which is counterintuitive.

### Questions
Please address the questions raised in the weakness section.

### Soundness
3

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
3

### Summary
This paper proposes a unified framework called CoAMD that bridges skeleton-based action recognition and text-to-motion generation by leveraging a Multi-modal Action Recognizer (MAR) as an active semantic guide during motion synthesis. The method introduces a multi-modal motion representation using absolute joint coordinates decomposed into joint positions, bone vectors, and motion dynamics, which enhances both recognition and generation. CoAMD generates motion in a coarse-to-fine, masked autoregressive manner using a diffusion model conditioned on text and contextual information from a transformer, while at each generation step, gradients from the MAR—trained via contrastive learning on both fine-grained motion–text retrieval and high-level action classification—are used to steer the latent motion toward better semantic alignment. Extensive experiments across 13 benchmarks demonstrate state-of-the-art performance in skeleton-based action recognition, text-to-motion generation, motion–text retrieval, and motion editing, with ablation studies confirming the importance of both the multi-modal representation and iterative semantic guidance.

### Strengths
* It successfully bridges two traditionally separate tasks—skeleton-based action recognition and text-to-motion generation—within a single, coherent architecture, demonstrating that semantic understanding and motion synthesis are mutually reinforcing.
* Rather than treating the action recognizer as a passive evaluation tool, the paper uniquely employs the Multi-modal Action Recognizer (MAR) as an active, gradient-based semantic guide during the diffusion sampling process. This enables real-time correction and significantly improves text-motion alignment.
* By decomposing absolute skeleton coordinates into joint positions, bone vectors, and motion dynamics, the model captures richer spatio-temporal and kinematic information. This representation consistently boosts performance in both recognition and generation tasks.

### Weaknesses
* The proposed method is designed for single-agent motion and does not handle multi-person or interactive scenarios. Real-world applications often involve multiple agents with social or physical interactions, which this model cannot address.
* The iterative semantic guidance mechanism introduces additional computational cost during inference. At each autoregressive step, the model must decode motion, compute the semantic score via MAR, and backpropagate its gradient, slowing down generation compared to unguided or non-iterative approaches.

### Questions
The performance of the model largely depends on the clarity and structure of the input text prompts. Although MAR is trained on diverse HumanML3D descriptions, highly abstract, ambiguous, or poorly phrased instructions may still result in suboptimal motion synthesis. How does the model perform when provided with vague, abstract, or grammatically incorrect textual descriptions?

### Soundness
3

### Presentation
3

### Contribution
3
