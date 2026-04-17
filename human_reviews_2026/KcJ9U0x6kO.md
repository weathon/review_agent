# HAMLET: Switch Your Vision-Language-Action Model into a History-Aware Policy

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 4

## Abstract
Inherently, robotic manipulation tasks are history-dependent: leveraging past context could be beneficial. However, most existing Vision-Language-Action models (VLAs) have been designed without considering this aspect, i.e., they rely solely on the current observation, ignoring preceding context. In this paper, we propose HAMLET, a scalable framework to adapt VLAs to attend to the historical context during action prediction. Specifically, we introduce moment tokens that compactly encode perceptual information at each timestep. Their representations are initialized with time-contrastive learning, allowing them to better capture temporally distinctive aspects. Next, we employ a lightweight memory module that integrates the moment tokens across past timesteps into memory features, which are then leveraged for action prediction. Through empirical evaluation, we show that HAMLET successfully transforms a state-of-the-art VLA into a history-aware policy, especially demonstrating significant improvements on long-horizon tasks that require historical context. In particular, on top of GR00T N1.5, HAMLET achieves an average success rate of 76.4% on history-dependent real-world tasks, surpassing the baseline performance by 47.2%. Furthermore, HAMLET pushes prior art performance from 64.1% to 66.4% on RoboCasa Kitchen (100-demo setup) and from 95.6% to 97.6% on LIBERO, highlighting its effectiveness even under generic
robot-manipulation benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper observes that multi-step robotic manipulation tasks are often history-dependent, whereas most existing Vision-Language-Action (VLA) models rely only on the current observation. To address this, the authors propose HAMLET, a framework that transforms pretrained VLAs into history-aware policies by introducing moment tokens, compact temporal representations learned through time-contrastive learning, and a lightweight memory module that aggregates past tokens to enhance action prediction.

### Strengths
1. The experimental results are very impressive.

2. This is a simple yet highly effective approach.

3. The writing is clear and well-structured.

### Weaknesses
See questions.

### Questions
1. Using past information is a very common strategy, as many previous imitation learning methods have incorporated historical observations [1,2]. Therefore, this aspect of the work feels somewhat less novel. However, the impressive results largely compensate for this. I am genuinely curious—why does adding the query token and aggregating past information lead to such a significant improvement? It would be great if the authors could provide more analysis on this.

2. As discussed in [1] and several other works, incorporating past information can lead to the copycat problem. Did you encounter this issue in your experiments?

3. In the simulation experiments, the multi-frame setting performs worse than the baseline. Why is that the case? Intuitively, having access to more information should improve performance, shouldn’t it?

[1] Fighting Copycat Agents in Behavioral Cloning from Multiple Observations. NeurIPS 2023.

[2] LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. NeurIPS 2023.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces HAMLET, primarily addressing the memory challenges inherent in Vision-Language-Action (VLA) models. The authors propose the use of Memory Tokens combined with a time-contrastive learning objective applied to the output features. This approach aims to obtain a compressed representation of the VLA's current internal state. This compact feature can then be provided to an action expert, supplying essential historical information. The method is notably lightweight and efficient, achieving a significant reduction in memory overhead.

### Strengths
- The paper is easy to follow, and the figures are clear and highly informative.

- The motivation behind this work is compelling. Most current VLA models neglect the issue of memory, and their tested benchmarks are often simple, placing minimal demands on historical information. However, as the authors correctly illustrate, real-world tasks requiring memory are abundant. Addressing this class of problems is crucial for VLA deployment.

- The proposed method is simple yet effective, incurring a low training burden. Crucially, it can be easily applied to many existing pre-trained VLA models to significantly enhance their memory capabilities.

### Weaknesses
This paper is comprehensive and well-executed; I found no discernible weaknesses.

### Questions
The authors successfully demonstrate and test many real-world tasks that are heavily dependent on memory, where the performance improvement brought by HAMLET is expected. However, the method also yields performance gains on benchmarks like LIBERO and SIMPLER. Since these tasks are logically completable without reliance on historical information, why does HAMLET still provide a performance boost? Could the authors offer an analysis or hypothesis for this unexpected improvement on tasks that are ostensibly memory-agnostic?

### Soundness
4

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
3

### Summary
This article addresses the limitation of existing Vision-Language-Action models (VLAs) that rely solely on current observations while ignoring historical context. It proposes the HAMLET framework, which transforms VLAs into history-aware policies by compressing perceptual information at each timestep using **moment tokens** initialized via time-contrastive learning (TCL), and integrating these historical moment tokens to generate memory features through a lightweight Transformer memory module. The article conducts experiments in both real-world (three tabletop tasks) and simulation benchmarks (RoboCasa Kitchen, LIBERO, SimplerEnv-Bridge). Results show that HAMLET, built on GR00T N1.5, achieves an average success rate of 76.4% on history-dependent real-world tasks (surpassing the baseline by 47.2%), improves performance from 64.1% to 66.4% on RoboCasa Kitchen (100-demo setup), and from 95.6% to 97.7% on LIBERO. Additionally, it incurs only minimal computational overhead (1.02× latency and 1.96× peak memory usage at a history length of 4) and can be adapted to different VLA backbones such as GR00T N1.5 and CogACT, verifying the framework’s effectiveness and generalizability.

### Strengths
1. The article accurately identifies the core contradiction in robotic manipulation tasks—VLAs’ "single-frame dependency" versus the "history-dependent nature of tasks"—and addresses the pain point of high computational overhead in traditional multi-frame baselines. It directly responds to the demand for efficient history-aware policies in the field of robot learning, demonstrating clear research value.

2. The combined design of moment tokens and a lightweight memory module is ingenious. Moment tokens reduce redundant information storage through compression, while the Transformer-based memory module enables selective integration of historical information. Notably, the framework does not require large-scale retraining of pre-trained VLAs, meeting the practical "plug-and-play" requirement. The technical route is clear and innovative.

3. Experiments cover both real-world and simulation environments, including comparisons across multiple tasks, baselines, and VLA backbones. Further analyses—such as efficiency evaluations, ablation studies (on component contributions, token length, and memory architecture), and cross-dataset transfer experiments—validate the framework’s performance, efficiency, and generalizability. The detailed data and logical reasoning enhance the credibility of the conclusions1

### Weaknesses
1. Currently, HAMLET uses moment tokens (default length of 4) and a lightweight 2-layer Transformer memory module, which perform well in the article’s designed tabletop tasks and standard kitchen manipulation tasks. However, validation is lacking in more complex scenarios, such as multi-object interactions, dynamically disturbed environments, or long-horizon tasks (with a history length exceeding 8). The mentioned issue of "the lightweight historical feature compression network" is valid: while the lightweight design ensures efficiency, it may insufficiently capture historical information due to limited feature expression capabilities in complex scenarios. Additional experiments are needed to demonstrate the framework’s robustness in such contexts.

2. The article only verifies HAMLET’s performance on diffusion-based VLAs (GR00T N1.5, CogACT) and does not extend it to autoregressive VLAs (e.g., models proposed by Pertsch et al., 2025; Kim et al., 2024). It also fails to analyze the modification costs and performance differences when adapting the framework to different VLA types, limiting the breadth of arguments for the framework’s applicable scope.

3. the article mentions that TCL enhances the temporal discriminability of moment tokens but does not elaborate on how specific parameters of TCL—such as the intensity of augmentation methods and negative sample selection strategies—influence token quality. It also lacks visualizations or quantitative analyses comparing the feature expression capabilities of moment tokens under different initialization methods (e.g., random initialization, other self-supervised approaches), leading to insufficient justification for the design of this core component.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes HAMLET, a plug-and-play framework to add history-awareness to pre-trained VLAs without costly retraining. It introduces "moment tokens" to compress each timestep and a lightweight memory module to aggregate them. The tokens are notably initialized with Time-Contrastive Learning (TCL) to focus on dynamic cues. The method proves highly efficient and effective on medium-horizon tasks, outperforming naive multi-frame baselines.

### Strengths
- The framework is a highly efficient, plug-and-play module that demonstrates a significant advantage in computational cost (latency and memory) over naive multi-frame approaches.

- The method is validated extensively across multiple VLA backbones (GROOT N1.5, CogACT) and benchmarks (real-world, RoboCasa, LIBERO), proving its generalizability and effectiveness.

### Weaknesses
- The memory mechanism is a simple rolling-window Transformer, which is not scalable for truly long-horizon tasks as critical information will eventually be dropped from the fixed-size history, which does not consider the history before the rolling-window.
- The core methodological contribution of Time-Contrastive Learning (TCL) for token initialization provides only a marginal performance gain (0.6% in Table 5a) over a random initialization, questioning its overall necessity.
- The comparison against the 'Multi-frame' baseline is a weak strawman, as this naive frame-stacking approach is known to fail and interfere with current-state grounding; the paper lacks benchmarks against more sophisticated history integration techniques.
- The "plug-and-play" claim is misleading, as the framework requires a dedicated finetuning stage and a separate token initialization phase, which contradicts the common understanding of a plug-and-play module as a zero-shot, training-free component.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
