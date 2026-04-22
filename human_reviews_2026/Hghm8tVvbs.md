# Label-free GUI Grounding via Confidence-guided Negative Reinforcement Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Graphical User Interface (GUI) grounding maps natural language instructions to precise interface locations for autonomous interaction. Current supervised fine-tuning and reinforcement learning approaches rely heavily on costly annotated data, creating a bottleneck for scaling GUI agents. We introduce a label-free training paradigm leveraging two key insights: (1) coordinate tokens in model outputs exhibit distinct confidence patterns that reliably identify correct predictions, and (2) in sparse GUI coordinate spaces, negative samples provide more reliable learning signals than potentially corrupted positive ones. We propose Confidence-Guided Reinforcement Learning (CRL), which uses coordinate-token confidence to select pseudo-labels from multiple samples and assigns distance-based rewards. We further develop Confidence-Guided Negative Reinforcement Learning (CNRL), which exclusively learns from negative samples. Without using any annotations, CNRL-7B achieves 92.1\% on ScreenSpot-V2, surpassing UI-TARS-72B (90.3\%) trained on 18.4M labels. On ScreenSpot-Pro, CNRL-7B reaches 33.8\%, improving 8.9\% absolute over the base model and exceeding GUI-R1-7B (31.0\%) trained on 3K labels. On challenging high-resolution benchmarks, CNRL consistently outperforms CRL by 1-1.5\%, demonstrating that learning what to avoid can be more effective than learning from uncertain positive examples. Our findings establish coordinate-token confidence as a powerful alternative to manual annotations for scalable GUI agent development.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This study addresses the bottleneck of existing methods in Graphical User Interface (GUI) grounding tasks, which rely on expensive annotated data, by proposing a label-free training paradigm. The core idea is to leverage the confidence patterns of coordinate tokens in the model's output (where the confidences of correct and incorrect predictions show a clear clustering separation) and the fact that in the sparse coordinate space of GUIs, negative samples are more reliable than potentially noisy positive samples. Based on this, the study designs Confidence-guided Reinforcement Learning (CRL) and Confidence-guided Negative Reinforcement Learning (CNRL). CRL selects pseudo-labels from multiple samples using coordinate token confidences and assigns rewards based on distance, while CNRL learns exclusively from negative samples. Experiments show that CNRL-7B, without any annotations, achieves 92.1% accuracy on ScreenSpot-V2 (surpassing the 90.3% of UI-TARS-72B, which was trained on 18.4M labels) and 33.8% on ScreenSpot-Pro (an 8.9% improvement over the base model and exceeding the 31.0% of GUI-R1-7B, which used 3K labels). This confirms that the confidence of coordinate tokens can serve as a substitute for human annotation, facilitating the scalable development of GUI agents.

### Strengths
1.The proposed label-free training paradigms (CRL, CNRL) can effectively address the issue of annotation dependency in GUI grounding.

2.The label-free training paradigm sounds like an interesting approach with the potential to scale to larger amounts of unlabeled internet data.

3.It achieves quite good results on several grounding datasets.

### Weaknesses
1.The method is trained directly on the test datasets. Although it doesn't see the ground truth labels, there is a potential risk of data leakage. How would the method perform if it were trained on open-source training datasets, such as uground-web, and then evaluated on these benchmarks?

2.I believe this method has the potential to be scaled to more data. If the authors could demonstrate its effectiveness when trained on large-scale data, it would significantly validate the method's scalability.

3.On the ScreenSpot-pro dataset, the performance of most methods on the leaderboard has already reached nearly 60%, while the method in this paper only achieves 33.8%. This, to some extent, may weaken the impact of the method. I would like to see the potential for this method to achieve higher performance on this dataset.

### Questions
please see weakness. If my concerns are addressed, I will consider raising my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses GUI grounding by introducing a coordinate-focused confidence signal. Instead of averaging token probabilities across the entire output, the method averages probabilities only over coordinate tokens (digits, separators), which reliably separates correct from incorrect predictions. Using the pseudo-label derived from this confidence, the model receives a binary distance-based reward marking responses within a threshold of the pseudo-label as positive. Policy optimization is performed using a GRPO-style objective under VLM-R1.

Using Qwen-2.5-VL (3B/7B) as the base within VLM-R1, a single-epoch label-free training with 16 samples per prompt improves accuracy across four benchmarks. Notably, CNRL-7B reaches 92.1% on ScreenSpot-V2, matching or surpassing heavyweight labeled baselines (e.g., UI-TARS-72B at 90.3% trained on 18.4M labels). On the more challenging ScreenSpot-Pro, CNRL-7B achieves 33.8%, >8.9% above the vanilla base and 1–1.5% above CRL; on UI-Vision, it attains 20.1%.

### Strengths
S1:The paper formalizes a simple, task-specific confidence signal and integrates it with a negative-only RL scheme. While negative-only RL has appeared in LLM reasoning, its application to sparse 2D coordinate spaces with a carefully designed reward is novel and well-suited.

S2: The C-Conf metric outperforms alternatives such as entropy, majority voting, KDE, center, and medoid. Threshold robustness favors CNRL, binary rewards perform better under noise, and positive-only learning underperforms, supporting the design choices.

S3: The method achieves competitive results without any labels compared to SFT/RL baselines trained on hundreds of thousands to tens of millions of labels. This demonstrates the potential to lower barriers for building GUI agents.

### Weaknesses
W1: Evaluation is restricted to grounding benchmarks. There is no end-to-end agent study (e.g., OSWorld, WAA) to demonstrate that grounding improvements translate into task success, which is increasingly expected in GUI agent research.

W2: The approach samples 16 responses per item to construct pseudo-labels, then down-samples to 8 for advantages. This could be computationally intensive.

W3: The “1 epoch” training setup under VLM-R1 does not explicitly detail which unlabeled images or instructions are used per benchmark (train/val/test splits, or extra unlabeled data).

### Questions
(1) Exactly which unlabeled data are used for training on each benchmark? Are they strictly train splits (never test)? For UI-Vision (with trajectories), which subsets are included?

(2) How does the grounding model perform when integrated into a GUI agent for action prediction or other live, end-to-end tasks?

### Soundness
3

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
4

### Summary
This paper tackles the challenge of GUI grounding, the task of mapping natural language instructions to UI element coordinates, without relying on expensive labeled data. The authors propose Confidence-Guided Reinforcement Learning (CRL) and Confidence-Guided Negative Reinforcement Learning (CNRL), two label-free training paradigms.

### Strengths
* Novel label-free paradigm:
The idea of leveraging internal model confidence (coordinate-token confidence) for self-supervised GUI grounding is interesting.

* Strong empirical performance:
Results are comprehensive and competitive across multiple benchmarks. Achieving parity or superiority over label-intensive models with zero annotations is impressive.

### Weaknesses
- **Insufficient evidence of the generalizability of the method**. The evaluations are conducted only on Qwen2.5-VL. I wonder how CNRL performs on other models (and model architectures), such as UI-TARS.

- What and how much data is CRL and CNRL trained on? Is it directly trained on the benchmark data? What if training on public training sets?

- (minor) In Table 2, it is better to note the base model of the baselines. For example, SeeClick is finetuned from Qwen2-VL and Aria-UI from Aria. It is not fair to compare these models directly according to the "Labels" column.


### Typos 
1. Repeated sentence in Figure 1's caption "CNRL zeros advantages for positive samples while preserving negative learning signals."

### Questions
1. In Equation 2, why is c_{coord} defined as the average of the token probabilities? How about using products, or a weighted score, given the different importance of the coordinate number digits?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a label-free paradigm for GUI grounding, where models predict UI element coordinates from screenshots and natural-language instructions without using any human annotations. The authors propose two main ideas:
1. Coordinate-token confidence: instead of averaging probabilities over all output tokens, they compute confidence only on numeric coordinate tokens, observing clear separation between correct and incorrect prediction
2. Confidence-guided Negative Reinforcement Learning (CNRL): a modification of GRPO where only negative samples contribute to policy updates, under the hypothesis that in sparse coordinate spaces, “what to avoid” is more reliable than potentially mislabeled positives

Using Qwen-2.5-VL (3B/7B), they train with multiple sampled predictions per instance and apply binary distance-based rewards. Across four benchmarks, the method achieves strong label-free performance. Ablations show coordinate-token confidence outperforms eight pseudo-label strategies by 2–12% and that CNRL is far more stable to reward-threshold changes.

### Strengths
1. Clear problem‑specific insight: Focusing on coordinate tokens yields consistently better pseudo‑labels than eight alternatives, and the advantage grows with the sampling budget (N = 2 -> 12), as shown in Fig. 4 and Table 7. This aligns with the intuition documented under Eq. (2) that coordinate tokens carry the discriminative uncertainty for this task.
2. Robust training signal at high resolution: CNRL is less sensitive to the radius $\tau$ (Table 3) and maintains higher reward accuracy on ScreenSpot‑Pro (Fig. 5), matching the claim that “learning what to avoid” is safer when true targets occupy a tiny fraction of the screen.
3. Competitive performance: On ScreenSpot‑V2, CNRL‑7B 92.1% rivals or exceeds models trained on 9.6M–18.4M labels; on ScreenSpot‑Pro and UI‑Vision, absolute gains over the base models are substantial (+8.9% and +5.1%, respectively).
4. Reproducibility: Concrete hyperparameters, sampling protocols, and prompts are given (even if some definitions are missing) in Table 8.

### Weaknesses
1. Training split protocol is under‑specified: Explicitly list per dataset which split(s) supply unlabeled training images and which are reserved for evaluation; add a leakage audit. This is crucial in a label‑free setup.
2. No pseudo‑label SFT / consistency baseline: Since a pseudo‑label is already selected, show a simple MLE or consistency baseline with identical pseudo‑labels and KL regularization. This will isolate the benefit from GRPO/CNRL vs. self‑training.

### Questions
1. How do you normalize the distance? If by diagonal, does $\tau$ transfer across resolutions? Could you report results with at least two normalizations to demonstrate invariance?
2. With the same pseudo‑labels and KL to a reference, how does MLE or EMA‑teacher consistency compare to CRL/CNRL at equal compute?
3. Did you try elliptical bands or IoU‑style soft rewards for elongated UI elements, given the failure modes in Figs. 10–12?
4. If you train label‑free on ScreenSpot‑V2 and evaluate on ScreenSpot‑Pro, what happens? Such a transfer result would strengthen the “label‑free” story.

### Soundness
3

### Presentation
3

### Contribution
3
