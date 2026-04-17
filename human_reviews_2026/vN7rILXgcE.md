# Training-free Uncertainty Guidance for Complex Visual Tasks with MLLMs

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Multimodal Large Language Models (MLLMs) often struggle with fine-grained perception, such as identifying small objects in high-resolution images or finding key moments in long videos. Existing works typically rely on complicated, task-specific fine-tuning, which limits their generalizability and increases model complexity. In this work, we propose an effective, training-free framework that uses an MLLM's intrinsic uncertainty as a proactive guidance signal. Our core insight is that a model's output entropy decreases when presented with relevant visual information. We introduce a unified mechanism that scores candidate visual inputs by response uncertainty, enabling the model to autonomously focus on the most salient data. We apply this simple principle to three complex visual tasks: Visual Search, Long Video Understanding, and Temporal Grounding, allowing off-the-shelf MLLMs to achieve performance competitive with specialized, fine-tuned methods. Our work validates that harnessing intrinsic uncertainty is a powerful, general strategy for enhancing fine-grained multimodal performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a training-free uncertainty guided framework for MLLMs to improve the fine-grained perception of visual information.  The proposed method estimates uncertainty from entropy-based measures derived from multiple forward passes with different portions of the visual input. It then selects the visual input with the lowest uncertainty as input for the final answer. The proposed method is applied to three tasks, including fine-grained visual search with high-resolution images, long video VQA, and temporal grounding. Experimental results show the effectiveness of the proposed method.

### Strengths
1. The idea of using uncertainty to select where to focus is interesting.
2. The authors have implemented the framework on different multimodal tasks to validate its effectiveness.
2. The overall writing is clear.

### Weaknesses
1. The evaluated benchmarks primarily involve tasks with short, discrete answers (e.g., single words or multiple-choice options). However, in realistic MLLM applications, outputs often take the form of longer sentences or paragraphs. The proposed entropy-based uncertainty estimation has not been validated in such open-ended generation settings, leaving its effectiveness for real-world scenarios unclear.

2. The method assumes that the type of question (e.g., yes/no, multiple choice, open-ended) is known beforehand to compute entropy appropriately. In practice, however, this information is rarely available during inference, which limits the applicability and automation of the proposed uncertainty estimation pipeline.

3. The approach always selects the top-1 region based on uncertainty. This assumption may fail when questions require reasoning over multiple fine-grained visual regions. 


4. The method requires performing tens to hundreds of inference passes per sample to estimate uncertainty, which introduces substantial computational overhead. Compared with the base model or other lightweight techniques, the proposed approach is significantly less efficient, raising concerns about its practical scalability for large-scale or real-time applications.

5. Additional fine-grained benchmarks like MME-RealWorld should be added to include more diverse question types.

### Questions
1. What are the results on the two video-related tasks with the Qwen2.5-VL-7B baseline?
2. How would the proposed framework affect the performance of the general MLLM benchmarks that might not require very detailed fine-grained information?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes UG, a training-free uncertainty-guided framework that improves fine-grained perception for MLLMs. The key observation is that when the model attends to the most relevant visual region, its output uncertainty decreases. Leveraging this, UG uses the MLLM itself to score candidate image crops / video frames / temporal windows by uncertainty, selects the most informative ones, and then performs the final inference only on these selected regions. Experiments on high-resolution image QA, long-video QA, and temporal grounding show that UG achieves strong gains over prior methods, including several finetuned systems, without any model training. The paper argues that this demonstrates uncertainty as an effective guidance signal for multimodal reasoning.

### Strengths
- **Simple and general training-free approach**: The method requires no model updates and can be directly applied to existing MLLMs without additional training or supervision.

- **Strong empirical gains**: Demonstrates substantial improvements across high-resolution image QA, long-video understanding, and temporal grounding benchmarks.

- **Model-agnostic**: Works with multiple popular multimodal models, showing broad applicability and compatibility across architectures.

- **Practical for real use cases**: Particularly valuable in scenarios where retraining is infeasible.

### Weaknesses
- **Inference overhead**: The approach incurs substantial inference-time cost, since it requires running the MLLM on many candidate crops/frames/windows to compute uncertainty before the final prediction. This runtime overhead is significantly higher than naive baselines and also higher than prior methods that rely on lightweight selectors or external scorers, making scalability a concern for long videos and high‐resolution inputs.

- **Core motivation not fully validated**: The key hypothesis that relevant regions exhibit lower uncertainty has only been evaluated when zooming toward the correct location. The paper does not systematically analyze *mislocalized* or distractor regions, so the claim that uncertainty reliably guides search remains primarily empirical.

- **Inconsistent uncertainty formulation**: The method uses token entropy for image and video frame selection but switches to a yes/no logit margin (BRC) for temporal grounding. The latter behaves more like a decision confidence score rather than a general uncertainty estimate. A calibration or ablation comparing these metrics would strengthen conceptual consistency.

- **Locality assumption and relational limits**: While the approach improves some relational cases, the appendix also presents failure examples where relevant objects are spatially distant, suggesting the method fundamentally relies on localizable evidence. A more systematic analysis of relational scenes and potential multi-patch extensions would clarify this limitation.

- **Compute fairness perspective not fully addressed**: Some strong prior systems amortize selection cost via training lightweight selectors or using external models, whereas this method shifts all cost to inference.

- **Task description could be more self-contained**: While standard benchmarks are used, the manuscript would benefit from briefly summarizing the problem format and evaluation protocol for each task/dataset to aid readers unfamiliar with these benchmarks.

### Questions
- Please refer to the points listed under *Weaknesses*.
- I am also curious about the sensitivity of the uncertainty signal to prompt phrasing. Have you tried alternative templates or paraphrased prompts when computing entropy scores, and does the performance of UG remain stable under such variations?

### Soundness
2

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
4

### Summary
This paper proposes a training-free test-time-scaling method to enhance fine-grained multimodal performance. For input high-resolution images or long videos, their method first conducts sampling and gains different visual clues and reasoning paths, then they select the final answers based on uncertainty or 'Yes-No' confidence. They demonstrate effectiveness on three tasks: Visual Search, Long Video Understanding, and Temporal Grounding.

### Strengths
1. The experiments are comprehensive. They demonstarte effectiveness on different tasks (image / video) and models (LLaVA, InternVL).
2. The motivation is reasonable. The Figure 1 is necessary for the story.

### Weaknesses
1. The method is a little simple, and the novelty is limited. They generate multiple answers with different viusal enhancement versions, then select final answers based on generation probability.
2. The time cost is little large. Especially, for long video understanding, they treat each frame (or short window) as visual candiate, but maybe some questions need motion across frames, enumerating all possible video clips would be computationally expensive. Similarly, for video temporal grounding tasks, a 'Yes-No' score must be calculated for every second.

### Questions
1. See weakness.
2. If the teaser image can also demonstrate uncertainty-video(clip) relationship, it will be better.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a training-free method to enhance the ability of MLLMs to focus on small regions on the image or video when answering questions. The proposed method divides the input image or video into segments, computes the MLLM’s answer uncertainty on each segment, and uses the uncertainty scores to select the segments that minimize the MLLM’s uncertainty. The paper provides several experiments to show the effectiveness of the proposed method in answering questions about small objects on images, questions on long videos, and localizing events in videos.

### Strengths
The direction of training-free approaches to enhance MLLMs' visual performance seems interesting and valuable to me. The proposed method is simple and architecture-agnostic, so it is easy to apply to various MLLMs. The method seems effective in answering multi-choice questions about small objects and localizing events in videos, so it can find application in these settings, although its broader effect on general visual question answering and video understanding performance remains unclear.

### Weaknesses
1. The evidence provided in Figure 1 does not support the paper’s main hypothesis that “an MLLM’s intrinsic uncertainty can be actively minimized at inference time to guide it toward the correct answer” as claimed in Lines 106-133. The results in Figure 1 suggest that zooming-in on the correct image region will minimize the model’s uncertainty, but not the other way around which is the paper’s hypothesis (that minimizing the model’s uncertainty will zoom-in on the correct image region). Note that there are many examples that contradict the paper’s hypothesis, for example: Imagine the question “Is there any dog in this image?” on a picture of a dog, then searching for the image region that minimizes the model’s uncertainty could focus on a completely empty background region where the model can be very certain that there is no dog! The line of research on adversarial attacks provides various such examples for unreliability of optimizing softmax-induced uncertainty. The paper must provide relevant experiments to support its key hypothesis, or narrow it down.

2. The choices of hyper-parameters (size of crop, temporal segment, k in top-k) seem to be made on the test videos (per section 4.4), and are therefore are likely overfitting to the small test datasets. The correct approach is to pick them based on a clearly described and separated validation set, and then apply them on the test set.

3. The paper does not report confidence intervals for its results. Note that given that the test datasets are small (<=100 samples in each subset considered in Table 1), the intervals can be quite large, making some of the benefits statistically insignificant. For example, a two-sample z-test (95%, two-sided) shows that the benefits of the proposed method for LLaVA and Qwen in Table 1 are not significant on HR-4K overall. The same concerns apply to Tables 2-5.

4. The proposed method’s gains on small visual details can be causing losses on more general VQA datasets, but the paper reports on only two image datasets focused on small details. For example, how does the proposed method affect counting, relation understanding, and text reading in standard datasets such as VQAv2, GQA, and TextVQA?

5. For the results in Table 4, the paper seems to artificially limit the baseline MLLMs to 8 frames across the video (Lines 299-300). This is misleading and does not reflect the real-practice use of these MLLMs. For example, Qwen-VL has a default FPS of 2 (up to 768 frames per video), which means limiting it to only 8 frames is a drastic divergence from its default operating setting. The same applies to Table 5.

6. The paper skips several important details in the main paper that hinder the clarity of its results. I recommend describing the datasets in more details (their underlying task and their size and their limitations), the ideas behind the other competing methods in more details (and why they are limited compared to the proposed method), and most importantly, the details of how the proposed method is applied to each dataset (how the cropped image is processed by the MLLM, on which tokens the score is computed, what are the specialized prompts used for computing the scores, and what are the failure cases).

7. The paper also does not report time and computation overhead of its proposed method compared to baseline MLLMs.

### Questions
Please see my questions and suggestions in the list of weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2
