# MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn Reinforcement Learning

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Recent advances in video multimodal large language models (Video MLLMs) have significantly enhanced video understanding and multi-modal interaction capabilities. While most existing systems operate in a turn-based manner where the model can only reply after user turns, proactively deciding when to reply during video playback presents a promising yet challenging direction for real-time applications. In this work, we propose a novel text-to-text approach to proactive interaction, where the model autonomously determines whether to respond or remain silent at each turn based on dialogue history and visual context up to current frame of an streaming video. To overcome difficulties in previous methods such as manually tuning response decision thresholds and annotating precise reply times, we introduce a multi-turn RL based training method that encourages timely and accurate responses without requiring precise response time annotations. We train our model MMDuet2 on a dataset of 52k videos with two types of dialogues via SFT and RL. Experimental results demonstrate that MMDuet2 outperforms existing proactive Video MLLM baselines in response timing and quality, achieving state-of-the-art performance on the ProactiveVideoQA benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel video multimodal large language model (Video MLLM) called MMDuet2, designed to enhance the model's proactive interaction capabilities—that is, the ability to autonomously decide when and how to respond while watching streaming videos. To address the challenges of existing methods requiring manual threshold adjustment and precise timestamp annotation of responses, the authors propose an innovative multi-round reinforcement learning (RL) training method. This method encourages the model to make timely and accurate responses at the right time through a carefully designed reward mechanism, without the need for precise timestamp annotation. Furthermore, the authors constructed a large-scale proactive dialogue dataset containing 52k videos for training. Experimental results show that MMDuet2 achieves state-of-the-art performance on benchmarks such as ProactiveVideoQA, significantly outperforming existing models.

### Strengths
1. This paper innovatively introduces multi-round reinforcement learning, using a reward mechanism to teach the model to find the optimal response time, cleverly circumventing the challenge of precise time labeling.
2. The authors constructed a large-scale dataset containing 52k videos, providing a solid data foundation for training more robust active models.
3. Experimental results show that the MMDuet2 model trained with SFT+RL outperforms previous state-of-the-art models and our own model trained solely with SFT on authoritative active interaction benchmarks such as ProactiveVideoQA. This demonstrates the effectiveness of reinforcement learning methods.

### Weaknesses
1. Using the "NO REPLY" text token is a concise and universal approach, but it also means that if the model chooses not to respond, a complete generation process (generating both tokens) is still required, which limits its inference efficiency.
2. The study on the reward component is insufficient, and related ablation experiments are lacking. The total reward is a weighted sum of four components, but the paper only mentions "a certain hyperparameter search" providing a set of weights without conducting ablation studies. This fails to clearly explain the specific contribution of each reward or penalty to the model's performance.
3. The task is limited to question-and-answer type tasks, and from data construction to model evaluation, it relies heavily on the question-and-answer paradigm. Therefore, the model's generalization ability for non-question-and-answer type proactive response tasks, such as caption tasks, needs to be tested.
4. There is insufficient comparative testing with other active response models. Related work mentions some active response models, such as Dispider [2] and TimeChat-Online[3]. However, the main experiments did not directly compare these models.

[1] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18407–18418, 2024a.

[2] Rui Qian, Shuangrui Ding, Xiao wen Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. ArXiv, abs/2501.03218, 2025.

[3] Linli Yao, Yicheng Li, Yuancheng Wei, Lei Li, Shuhuai Ren, Yuanxin Liu, Kun Ouyang, Lean Wang, Shicheng Li, Sida Li, Lingpeng Kong, Qi Liu, Yuanxing Zhang, and Xu Sun. Timechatonline: 80% visual tokens are naturally redundant in streaming videos. ArXiv, abs/2504.17343, 2025.

### Questions
1. The dialogue template shown in Figure 2 of the paper appears to be standard practice for frame-by-frame streaming video understanding models. Furthermore, the core mechanism for generating "NO REPLY" seems functionally equivalent to predicting a specific EOS token (VideoLLM-online [1]). Therefore, aside from the specific implementation, what is the fundamental innovation of this "text-to-text approach"?
2. To better contextualize the performance of MMDuet2, would the authors consider providing direct empirical comparisons against other relevant models, such as Dispider [2] and TimeChat-Online [3]? At the same time, could you please provide an ablation study analyzing the individual impact of the four reward components ($r_{\text{PAUC}}$, $r_{\text{rep}}$, $r_{\text{inspan}}$, $r_{\text{pfx}}$) on the model's behavior?
3. The authors employed a strategy of placing the model's response at the end of its response time period during the SFT phase to avoid the model developing illusions before seeing the relevant event. However, the goal of the RL phase is to encourage the model to make the correct response as early as possible. Are these two goals contradictory during training?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MMDuet2, aiming to enhance the proactive interaction capabilities of Video Multimodal Large Language Models (Video MLLMs) through multi-turn Reinforcement Learning (RL), enabling the model to autonomously decide when to respond during video playback. The authors construct a new dataset of 52k videos and present a text-to-text RL approach with a reward mechanism, including PAUC.

### Strengths
1.The paper addresses proactive interaction, which is an important and challenging promblem for making Video MLLMs more natural and practical in real-time applications.

2.  The use of RL to overcome the difficulty of precise reply time annotation is a promising avenue, and the reward mechanism design  theoretically considers timeliness, accuracy, and redundancy.

3. The creation of a large-scale new dataset (52k videos with two dialogue types) provides a valuable resource for research in this field.

### Weaknesses
1. The central contribution of this paper lies in rl , which is explicitly designed to improve proactive interaction timing. However, the paper reports that during training on complex ego-centric video tasks, the model exhibited reward hacking behavior—generating large amounts of repetitive content. Although this issue is solved by early stopping, such manual action may show a instability in the reward design and optimization process.

2. The occurrence of reward hacking indicates that the learned policy may not genuinely capture proactive interaction behavior but instead exploits the reward function. The paper does not provide ablations or diagnostic analyses to clarify why this failure occurs, nor does it offer evidence that the method can generalize to more complex or long-duration scenarios without collapsing.

### Questions
1. Regarding the observed reward hacking issue, was there a deeper analysis of its root causes beyond "early stopping"? Were more sophisticated reward shaping, curiosity mechanisms, or RL algorithms attempted to enhance the model's robustness on complex tasks?

2. Please elaborate on how the largely "offline" QA and captioning datasets in the SFT stage were processed and transformed to effectively support RL training for proactive interaction, rather than merely enhancing general understanding?

3. The paper states that reducing the frame interval from 2 seconds to 1 second during inference significantly improves performance, even if the RL phase used a 2-second interval. Does this suggest that the model's decision-making process is highly sensitive to the temporal granularity of the input?

### Soundness
3

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
3

### Summary
This paper investigates the reward-based RL in online video LLM settings and propose MMDuet2 to autonomously determine whether to respond as soon as possible or remain silent in a proactive manner by RL training. This paper starts from curating online video llm training data, which is designed for multiturn proactivate dialogue, then design the specialized chat template for it and use SFT and RL to train the model in the off-policy and on-policy way for proactivate capabilities. Experimental results show that the proposed MMDuet2 can proactively answer user's queries in the online video streaming setting while maintain the ability to answer in offline video setting.

### Strengths
1) This paper is one of the first batch to investigate RL training (esp. GRPO) for proactively answering in online streaming video settings, which is of substantial novelty. Also, the paper investigate the reward, the key component of GRPO, and model it specifically for online video settings (PAUC).

2) The authors design a training dataset especially for online video streaming setting and corresponding chat template. Looking forward to the dataset open sourcing.

3) The MMDuet2 trained by SFT and GRPO outperform previous online methods while maintaining offline video understanding capabilities

### Weaknesses
There are no major technical concerns about this paper, but I want to address some minor points as follows:

1) As the key component to apply GRPO to online video settings, the ablations on rewards should be more addressed. Did authors try other rewards than PAUC? Please compare several reward formulations and discuss why PAUC is preferred.

2) The author is encouraged to report the actual inference speed and latency of the MMDuet2 to see if it is realtime in practical scenes.

Also, the organization of this paper needs to be improved further since the current version seems too flattened with every details being straight written without a main storyline.

### Questions
See weaknesses.

### Soundness
4

### Presentation
2

### Contribution
3
