# FAD-TQ: Industrial Fine-grained Anomaly Detection with Thinking Quality

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Recent research in industrial anomaly detection (IAD) has shifted beyond binary classification and segmentation, increasingly focusing on process-level, interpretable reasoning about the type and cause of anomalies. While multimodal
large language models (MLLMs) have enabled this reformulation through visual
question answering, current anomaly detection methods still suffer from two major limitations: the limited capacity of reward functions to capture intricate complexities and the reliance on generating supervised fine-tuning (SFT) data. Hence,
we propose FAD-TQ, a lightweight reinforcement learning framework for finegrained anomaly detection with thinking quality. Built upon the Group Policy
Gradient paradigm, it eliminates the reference model and KL regularization to reduce rollout overhead and directly optimize the original reinforcement learning
objective. To enable fine-grained guidance over the reasoning process, we design a thinking quality reward composed of two components: an efficiency reward
that penalizes redundant reasoning, and a relevance reward that encourages taskaligned, coherent thought trajectories. Furthermore, we introduce MVTec-LOCOAD-Pair3C, a principled evaluation protocol built on the existing dataset. By
defining three decision types—normal, structural anomaly, and logical anomaly,
rather than binary classification. Extensive experiments demonstrate that FAD-TQ
improves interpretability, accuracy, streamlined reasoning and training efficiency
with reduced computational costs. It demonstrates the potential of using smallscale benchmarks to evaluate MLLM capabilities in IAD. We hope this framework
and evaluation protocol can serve as an example for future research on processlevel reasoning in anomaly detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims at achieving better reasoining capability of image anomaly detection. To achieve this, it split the MVTec-LOCO dataset with logical and structure anomalies to set up a anomaly-type-classified testbed, and adopt Group Policy Gradient paradigm with two reward setup to penalizes redundant reasoning and coherence. The proposed methods achieves better performance on the proposed benchmark.

### Strengths
1) This paper achieves better reasoning capability for image anomalies by adopting the Group Policy Gradient paradigm for MLLM post-training.

2) The decision to split anomaly categories is a sound approach for rigorously evaluating the model's fine-grained classification capabilities.

3) The proposed empirical reward designs are effective, according to the qualitative results presented.

### Weaknesses
My primary concern with this paper is the insufficient comparison to relevant baselines and the absence of evaluation on a much larger, established benchmark for anomaly reasoning, namely MMAD.

1) The paper's comparison is limited to several zero-shot LLMs. It lacks a comprehensive evaluation against a wider scope of multi-modal LLMs that have been specifically designed or fine-tuned for anomaly detection and reasoning tasks.

2) The core of the proposed method involves leveraging the Group Policy Gradient paradigm, a technique extensively studied in reinforcement learning and widely applied to LLM post-training. Applying this existing technique to the narrow domain of anomaly detection, without significant novel adaptation, makes the paper's contribution appear marginal.

3)  A comprehensive benchmark for multi-modal anomaly detection and reasoning, MMAD, already exists. It is strongly recommended that the authors evaluate their proposed method on this benchmark. A strong performance on a large-scale, recognized dataset like MMAD would provide a much more robust validation of the method's capabilities than the results on the small-scale, curated dataset currently used.

### Questions
Could you please clarify the classification protocol for samples in the MVTec-LOCO dataset that exhibit both structural and logical anomalies? Table 1 suggests that these two categories are mutually exclusive. It is unclear how cases with multiple anomaly types are handled—are they assigned to a single, dominant type, or categorized in another way?

### Soundness
2

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
This paper addresses efficiency issues in multimodal large language model-based Anomaly Detection (MLLM-AD) by proposing FAD-TQ. It removes the reference model and KL regularization to reduce computational overhead and introduces a thinking-quality reward combining efficiency and relevance components. The study also constructs the MVTec-LOCO-AD-Pair3C benchmark, framing IAD as a three-way classification task.

### Strengths
1. The work targets the efficiency problem in MLLM-AD research.
2. Experiments are comprehensively presented.

### Weaknesses
1. Generalization is questionable as training and evaluation are conducted on in-distribution data. The framework lacks validation on out-of-distribution scenarios.
2. The MVTec-LOCO-AD-Pair3C benchmark covers only five object categories, failing to comprehensively assess MLLM capabilities in diverse industrial settings. Additionally, the small test sample size raises concerns about the stability of the results, making the claimed performance advantages less credible due to potential fluctuations.
3. The proposed method lacks sufficient innovation. GPG and relevance reward are direct applications of existing reinforcement learning techniques. No novel mechanisms are introduced to address the unique challenges of AD tasks.

### Questions
The paper formulates industrial anomaly detection as a three-way classification task. Given the task’s relative simplicity compared to complex multimodal reasoning, could traditional supervised learning algorithms achieve competitive or even better performance?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the FAD TQ method, which fine tunes large multimodal models via reinforcement learning to perform image anomaly detection tasks while producing interpretable reasoning chains. FAD TQ leverages a Group Policy Gradient paradigm, eliminating the need for a reference model and KL divergence regularization, thereby directly optimizing the objective. The method introduces a two component reward function that balances reasoning chain efficiency and quality. Additionally, the authors propose a new evaluation method, MVTec LOCOAD Pair3C, based on existing datasets. Experimental results demonstrate that FAD TQ outperforms existing approaches in interpretability and accuracy.

### Strengths
S1.The use of Group Policy Gradient paradigm eliminates the reference model and KL divergence, which simplifies the RL process, reduces VRAM requirements, and speeds up training rollouts, making the approach more practical.
S2.Experimental results show that FAD-TQ improves accuracy by 16.12% over Qwen2.5 VL 3B and 17.05% over MiniCPM V 4, and achieves an 8.99% improvement over the GRPO training method, representing a substantial gain.

### Weaknesses
W1. The experimental comparison focuses only on accuracy; there is no evaluation of training cost or efficiency.
W2. The proposed FAD TQ model is trained solely on Qwen2.5 VL 3B, without testing the reinforcement learning method’s robustness on a wider range of base models.
W3. Only GRPO and GPG are used as baseline training methods; the comparison set is limited.

### Questions
1. Could experimental analysis be conducted to compare FAD TQ’s training cost against other reinforcement learning methods?
2. Could FAD TQ be trained on additional base models (e.g., Qwen3 VL Thinking with built in reasoning chains) to verify the robustness of the proposed training method?

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
This paper introduces FAD-TQ, a new reinforcement learning (RL) framework for detailed industrial anomaly detection. The authors want to solve two common problems when using Multimodal Large Language Models (MLLMs) for this task: the reward signals are weak, and they rely too much on supervised fine-tuning (SFT) data. To tackle this, they propose a Group Policy Gradient (GPG) method and a new reward function called "Thought Quality" (TQ). They also created a new benchmark, MVTec-LOCO-AD-Pair3C, to show how their method improves MLLM performance.

### Strengths
+ The GPG algorithm is an RL method that doesn't need a reference model. This is great because it cuts down on computing costs and directly optimizes the main goal.
+ How the paper rewards both the "thinking process" and the final result separately. This seems like a really smart and necessary approach, especially for a task like anomaly detection.

### Weaknesses
+ A major concern is the results on their new benchmark. Even with test data and powerful MLLMs, their method doesn't actually perform better than simpler, smaller models that just do a basic "yes/no" classification. This makes me seriously question if their complicated approach is even necessary.
+ The paper argues that GPG is a great way to unlock the full potential of MLLMs. However, they don't compare it against a standard SFT approach. Without that comparison, it's hard to tell if GPG is truly needed.
+ The paper tries to turn anomaly detection into a more detailed "anomaly classification." The benefit of doing this isn't clear. They should probably also include the overall anomaly detection accuracy (just a simple "is it an anomaly or not?") as a baseline metric to help us judge their results better.

### Questions
Please refer to the **weakness** section.

### Soundness
2

### Presentation
2

### Contribution
2
