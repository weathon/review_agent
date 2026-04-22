# ReGUIDE: Data Efficient GUI Grounding via Spatial Reasoning and Search

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Recent advances in Multimodal Large Language Models (MLLMs) have enabled autonomous agents to interact with computers via Graphical User Interfaces (GUIs), where accurately localizing the coordinates of interface elements (e.g., buttons) is often required for fine-grained actions. However, this remains significantly challenging, leading prior works to rely on large-scale web datasets to improve the grounding accuracy. In this work, we propose Reasoning Graphical User Interface Grounding for Data Efficiency (ReGUIDE), a novel and effective framework for web grounding that enables MLLMs to learn data efficiently through self-generated reasoning and spatial-aware criticism. More specifically, ReGUIDE learns to (i) self-generate a language reasoning process for the localization via online reinforcement learning, and (ii) criticize the prediction using spatial priors that enforce equivariance under input transformations. At inference time, ReGUIDE further boosts performance through a test-time scaling strategy, which combines spatial search with coordinate aggregation. Our experiments demonstrate that ReGUIDE significantly advances web grounding performance across multiple benchmarks, outperforming baselines with substantially fewer training data points (e.g., only 0.2\% samples compared to the best open-sourced baselines).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents ReGUIDE, a data-efficient GUI grounding framework for MLLM-based web agents. It combines reasoning-guided self-generation and spatial-aware criticism to improve coordinate localization without large datasets. A test-time spatial scaling further enhances accuracy. Compared with exisiting grounding models, ReGUIDE perfroms well on extensive expermients results.

### Strengths
1. The paper provides an extensive and comprehensive evaluation of ReGUIDE across multiple benchmarks.
2. The writing and organization are clear, coherent, and well-structured.
3. The proposed training and testing time scaling strategies are effective, and the combination of global–local search with voting demonstrates strong performance.

### Weaknesses
1. The main concern with this work lies in whether the proposed model truly achieves state-of-the-art grounding performance. In fact, several prior studies have already reported superior results. However, the authors did not include these stronger baselines (e.g., [1–3]) in their benchmark comparisons.

2. The ensuing concern is that the related work does not encompass the latest developments.

3. The results presented in Table 7 seem somewhat counterintuitive. First, it is unclear why five data benchmarks are listed, as in reality there are only two. Second, regarding AC benchmark, it is puzzling that the High-level model outperforms the Low model. Finally, it remains unclear how the experimental setup allows web agents to achieve such high performance on mobile tasks. If additional fine-tuning was applied, the reported task completion rate still appears lower than that of some existing mobile agents.

4. The relationship between the effect of consistency fine-tuning and attention–grounding modeling remains unclear. The authors should further clarify how these changes contribute to improved grounding performance. 

5. The analysis of the natural Gaussian distribution of coordinate token likelihood appears to yield conclusions similar to those reported in GUI-G2. Therefore, it is strongly recommended that the authors compare their findings directly with GUI-G2, clarifying any differences in methodology, experimental setup, and quantitative outcomes. 

**Reference**

[1] GTA1: GUI Test-time Scaling Agent

[2] GUI-G2: GAUSSIAN REWARD MODELING FOR GUI GROUNDING

[3] Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning

### Questions
See above

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
This paper presents ReGUIDE, a data-efficient framework for GUI coordinate grounding that enables multimodal language models to accurately localize interface elements. ReGUIDE employs a two-stage training approach: (1) self-generating reasoning through online reinforcement learning using grounding accuracy as reward, and (2) enforcing spatial consistency between global and local image views using transformation-based priors. At inference, a test-time search strategy combines spatial cropping with KDE-based coordinate aggregation. Using only 20K samples (≈0.2% of UGround), ReGUIDE outperforms prior models, improving Qwen-2.5-VL-3B accuracy from 55.5% to 88.0% on ScreenSpot and from 23.9% to 44.5% on ScreenSpot-Pro, demonstrating significant data efficiency gains.

### Strengths
**1. Novel Integrated Data-Efficient Framework:**

The paper proposes a highly novel framework, *ReGUIDE*, that tackles the GUI grounding problem with exceptional data efficiency. Its originality lies in the synergistic integration of components across both training and inference: reinforcement learning for self-generated language reasoning, a subsequent training stage enforcing spatial consistency under transformations, and a final test-time spatial search with KDE-based aggregation. This comprehensive combination effectively addresses the spatial reasoning and localization precision limitations of prior approaches.

**2. Strong and Data-Efficient Performance:**

ReGUIDE achieves significant performance improvements, setting a new competitive standard across multiple challenging GUI grounding benchmarks, including ScreenSpot and ScreenSpot-Pro. Notably, it demonstrates remarkable data efficiency, reaching state-of-the-art results while using only a small fraction of the data (a 20K subset, or 0.2% of samples) compared to prior methods trained on millions of examples.

**3. Comprehensive Empirical Evaluation and Insights:**

The paper presents extensive and rigorous experiments, including systematic ablation studies that isolate the contribution of each training and inference component. Moreover, it validates practical utility through strong transfer performance on downstream agentic tasks such as Multimodal-Mind2Web and AndroidControl, further supporting the empirical soundness of the proposed approach.

**4. Clear Presentation and Reproducibility:**

The paper is well written and well organized, featuring clear figures and structured technical explanations. It provides valuable insights into the model’s internal mechanisms through attention map visualization and analysis of self-generated reasoning pathways. The inclusion of complete experimental details, hyperparameter settings, and the stated plan to release code enhances the clarity and reproducibility of the work.

### Weaknesses
**1. Lack of comparison with other RL-based methods:**  
For instance, *UI-AGILE-7B*, which also employs reinforcement learning with only 9K examples, achieves 48.7% accuracy on ScreenSpot-Pro when initialized from Qwen2.5-VL, which is comparable to or even surpasses ReGUIDE’s results under a smaller data regime. It would be helpful to discuss this comparison to clearly highlight the differences between the two approaches.

**2. Inference latency:**  
ReGUIDE’s performance gain largely depends on its two-stage test-time spatial search. While effective, this procedure increases inference time by approximately 2.0 to 2.2 times (Table 13), introducing a relatively high latency overhead for real-time or interactive GUI agents. This raises concerns about its practicality in latency-sensitive scenarios where fast responses are critical.

**3. Entanglement of test-time scaling and training effects:**  
The main results combine the gains from both the training components (RL + Consistency) and the inference enhancements (test-time scaling), making it difficult to isolate the true contribution of the proposed data-efficient training. Since other baselines are evaluated using single-pass inference, a fair comparison requires reporting ReGUIDE’s performance both with and without test-time scaling to clarify how much of the impro

### Questions
**1. On the SOTA claim and leaderboard comparison:**

The paper reports state-of-the-art results on ScreenSpot-Pro. However, several stronger 7B baselines have already been listed on the public leaderboard (https://gui-agent.github.io/grounding-leaderboard/), such as GUI-ARP-7B (60.8%), Holo1.5-7B (57.9%), GTA1-7B (55.5%), and GUI-Cursor-7B (56.5%).

It would be helpful if the authors could clarify how ReGUIDE compares with these concurrent approaches and discuss their methodological differences to better position the contribution of this work.

**2. On inference latency and deployability:**

The two-stage test-time search appears to increase inference time by about twofold. Could the authors share their perspective on how this additional latency might affect real-world deployment, particularly for GUI agents requiring real-time interaction? 

**3. On isolating training and inference gains:**

The core contribution is argued to be Data Efficient Grounding achieved through the proposed training paradigm. However, the reported SOTA results include the significant gains from the Test-Time Scaling. To ensure a transparent analysis of data efficiency and computational trade-offs, could the authors please report ReGUIDE's performance in the main results before and after applying the Test-Time Scaling? This separation would clearly isolate the performance improvement attributable solely to the data-efficient training methods.

**4. On handling high-resolution inputs:**

The Cropping stage contributes notably to the gains on the high-resolution ScreenSpot-Pro benchmark, suggesting that the base model may still struggle with large input resolutions. Could the authors elaborate on whether the Consistency training helps mitigate this issue by learning a more robust, scale-invariant representation internally, or whether the improvement mainly depends on the external inference-time Cropping process?

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
4

### Summary
This paper addresses the critical bottleneck of data inefficiency in training MLLMs for GUI grounding. The authors propose ReGUIDE, a novel framework that achieves state-of-the-art results using a very small fraction--0.2% of the data required by existing baselines. The method's novelty lies in its two-stage training process, which first uses online reinforcement learning to reward accurate coordinate prediction, compelling the model to self-generate its own language-based reasoning without explicit supervision. This is followed by another stage that enforces spatial consistency between global and cropped views of the image. The framework is further enhanced by an effective test-time scaling strategy that uses spatial search and Kernel Density Estimation to iteratively refine coordinate predictions. Experiments across multiple benchmarks, including high-resolution and agentic tasks, demonstrate significant performance gains, highlighting the method's effectiveness in learning robust spatial priors from minimal data.

### Strengths
- Data Efficiency. This is the paper's primary contribution and it is highly significant. Achieving nice performance while training on only 20k samples versus 10M is a major step forward for the field, making high-performance grounding more accessible.
- Effective Inference-Time Search. The test-time scaling strategy, which uses KDE for an initial vote, crop, and then vote again, is well-motivated and empirically powerful. The paper shows this is particularly effective for high-resolution images.

### Weaknesses
- Two very relevant work, Aria-UI (for GUI grounding SFT) and GTA-1 (for GUI RL training with GRPO) is not discussed nor compared in the paper.
- From Tab.5, the proposed two test-time scaling strategies play important role in model's performance with SS and SS-pro. The questions here would be:
1) since we may easily move the two strategies to existing models like UGround, will they benefit from it?
2) since without the two strategies, the proposed model generally performs on par with the baselines, and considering the authors only used 0.2% of the data, can scaling law in terms data still apply to ReGUIDE models? For example, if using 10x or 100x of the data, how would ReGUIDE benefit from it?

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
