# $AutoDrive\text{-}P^3$: Unified Chain of Perception–Prediction–Planning Thought via Reinforcement Fine-Tuning

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Vision-language models (VLMs) are increasingly being adopted for end-to-end autonomous driving systems due to their exceptional performance in handling long-tail scenarios. However, current VLM-based approaches suffer from two major limitations: 1) Some VLMs directly output planning results without chain-of-thought (CoT) reasoning, bypassing crucial perception and prediction stages which creates a significant domain gap and compromises decision-making capability; 2) Other VLMs can generate outputs for perception, prediction, and planning tasks but employ a fragmented decision-making approach where these modules operate seperately, leading to a significant lack of synergy that undermines true planning performance. To address these limitations, we propose ${AutoDrive\text{-}P^3}$, a novel framework that seamlessly integrates $\underline{\textbf{P}}$erception, $\underline{\textbf{P}}$rediction, and $\underline{\textbf{P}}$lanning through structured reasoning. We introduce the ${P^3\text{-}CoT}$ dataset to facilitate coherent reasoning and propose ${P^3\text{-}GRPO}$, a hierarchical reinforcement learning algorithm that provides progressive supervision across all three tasks. Specifically, ${AutoDrive\text{-}P^3}$ progressively generates CoT reasoning and answers for perception, prediction, and planning, where perception provides essential information for subsequent prediction and planning, while both perception and prediction collectively contribute to the final planning decisions, enabling safer and more interpretable autonomous driving. Additionally, to balance inference efficiency with performance, we introduce dual thinking modes: detailed thinking and fast thinking. Extensive experiments on both open-loop (nuScenes) and closed-loop (NAVSIMv1/v2) benchmarks demonstrate that our approach achieves state-of-the-art performance in planning tasks. Code is available at https://github.com/haha-yuki-haha/AutoDrive-P3, https://openi.pcl.ac.cn/OpenAIDriving/AutoDrive-P3.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes AutoDrive, a unified vision-language-action (VLA) model for autonomous driving. The key idea is to train a single model that can understand visual inputs, interpret instructions, and output low-level driving actions. The authors introduce a new training method called P³-GRPO (Perception–Planning–Policy Generalized Reinforcement Preference Optimization), which combines supervision from perception, planning, and control tasks in a joint optimization framework.

The model is evaluated on standard driving benchmarks and shows strong performance across all metrics. The collision rate in particular is much lower than prior systems like DriveVLM and DriveGPT, and the authors demonstrate that the proposed P³-GRPO method is the main driver behind these improvements.

### Strengths
The experimental results are really strong. The improvement in success rate and collision reduction is quite large compared to previous SOTA, which is impressive.

The P³-GRPO training framework seems like a solid contribution. It’s a clean idea that unifies different supervision signals under one preference optimization scheme, and it clearly helps the model learn better driving behaviors.

The paper is well written and easy to follow, with a clear structure and convincing ablations showing the effect of P³-GRPO.

### Weaknesses
The biggest missing piece is runtime analysis. There’s no mention of latency or inference speed anywhere in the paper. For a VLM-based driving model, this is a serious omission. These models are large and autoregressive, and it’s hard to tell whether the proposed system can actually run in real time.

There’s also no discussion of how the model could be deployed in practice. Prior work like DriveVLM already acknowledged this issue and proposed a hybrid setup: a lightweight driving system for high-frequency control and a heavy VLM that only runs in rare or complex cases. That kind of hybrid design makes practical sense, but this paper doesn’t touch on it at all.

Overall, while the performance is strong, the work feels a bit detached from deployment realities. It would be much stronger if the authors at least analyzed compute requirements or discussed how AutoDrive could fit into a practical driving stack.

### Questions
1. What’s the actual runtime per frame or per driving decision? Without this, it’s hard to judge feasibility.

2. Could the system be applied in a hybrid setup (like DriveVLM’s), where the large model runs only in low-frequency cases?

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
This paper introduces AutoDrive-P³, a VLM framework for autonomous driving that unifies perception, prediction, and planning through structured Chain-of-Thought (CoT) reasoning. It proposes the P³-CoT dataset and P³-GRPO, a hierarchical reinforcement learning method to supervise all three stages, aiming for improved synergy and planning performance.

### Strengths
The paper's significance lies in its explicit attempt to unify perception, prediction, and planning within a VLM using CoT, addressing the limitations of fragmented or planning-only approaches. The quality is demonstrated through the creation of the P³-CoT dataset, designed to facilitate this unified reasoning , and the novel P³-GRPO algorithm, which extends reinforcement learning supervision beyond just planning. The clarity is good, effectively outlining the problem and the proposed solution. The strong results on nuScenes and NAVSIM suggest the approach improves planning reliability and interpretability.

### Weaknesses
The reliance on Qwen2.5-VL-72B to generate the P³-CoT dataset introduces potential biases or limitations inherent in the generating model. While manually verified, the quality of the generated CoT is crucial and may propagate errors. Additionally, the hierarchical nature of P³-GRPO, while intended to create synergy, could potentially lead to error accumulation if early stages (perception, prediction) are poorly optimized, impacting the final planning output. The paper acknowledges hallucination issues and the need for real-world testing, which are important next steps.

### Questions
1. The P³-CoT dataset was generated using Qwen2.5-VL-72B. How was the quality and correctness of the generated CoT reasoning verified beyond just checking the final labels? Could biases from the generating model affect the downstream training, especially on a different series of models?
2. P³-GRPO applies hierarchical supervision. Could errors or suboptimal performance in the earlier perception and prediction reward stages negatively impact the final planning stage more than a planning-only reward might? How sensitive is the final planning performance to the accuracy of the perception/prediction rewards?
3. The reward weights for P³-GRPO were set in a 1:2:2:5 ratio (format:perception:prediction:planning). Could you elaborate on the process for determining these weights? Have other weighting schemes been explored?
4. The paper argues that fragmented VLM approaches lack synergy. Table 3 compares SFT-only, Planning-GRPO, and P³-GRPO. While P³-GRPO performs best, the Planning-GRPO version also shows improvement over SFT-only. Could you further discuss the specific benefits attributed solely to the hierarchical supervision (P³-GRPO) compared to just applying GRPO to planning?

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
5

### Summary
This paper presents AutoDrive-P3, a unified end-to-end autonomous driving framework that integrates Perception, Prediction, and Planning through structured chain-of-thought reasoning. It introduces a new P3-CoT dataset and a hierarchical reinforcement learning algorithm (P3-GRPO) to ensure coherent reasoning across tasks.

### Strengths
1. The paper is easy to read, and the core idea is clearly presented.

### Weaknesses
1. As mentioned in the Introduction, both perception and prediction are important for planning, which I agree with. However, as described in the Method section, there is no direct connection from perception to planning, which seems inconsistent with Figure 1. I wonder how the information or features are transferred from perception to planning. How does the perception-CoT contribute to the planning-CoT?

2. The authors should include a Related Works subsection to discuss prior work on VLM-based autonomous driving. There are many studies that use VLMs to enhance end-to-end autonomous driving across all three tasks. Moreover, different strategies exist for integrating VLMs into the pipeline. For example, [1] encodes the VLM into the pipeline and jointly fine-tunes it, while [2] uses the VLM to generate an auxiliary dataset to improve planning, a strategy that is similar to the proposed dataset generation approach.

[1] J. Hwang et al., “EMMA: End-to-End Multimodal Model for Autonomous Driving,” TMLR, 2025.

[2] Y. Xu et al., “VLM-AD: End-to-End Autonomous Driving through Vision-Language Model Supervision,” CoRL, 2025

3. The results on nuScenes are close to those of OmniDrive and OpenDriveVLA in L2. Some discussion on possible reasons for this would be helpful, especially since this work emphasizes the benefits of jointly considering all three tasks.

### Questions
1. In Section 4.1, the data labeling pipeline is introduced, with more details in the Appendix. I am curious about what the prompt questions look like for the three tasks. In many related works, task-specific questions are carefully designed to extract knowledge from VLMs, what is the strategy here?

2. How is the quality of the P^{3}-COT dataset ensured? Since the labels are generated by Qwen2.5-VL-72B, did the authors encounter imperfect labels during annotation? How was label quality evaluated? Were any quality-control methods such as cross-validation or human evaluation (e.g., questionnaires) applied?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
They introduce a chain of thought dataset that combines reasoning in Planning, Prediction and Perception. They aim to reduce the error Propagation gap of classical Systems, by conditioning Output on different reasoning Levels. They also contribute a unified GRPO Training Approach that aims to balance the learning contribution by each part of the System (Perception, Prediction, Planning).

### Strengths
They contribute a unification in VLM reasoning by conditioning Planning on Perception and Prediction. They additionally introduce the hierarchical GRPO Training Extension that is interesting and potentially/based on their results useful.

### Weaknesses
They justify their work with a broad Claim, that is not backed by any citation or Explanation: VLM-based AD NEEDS CoT. This cannot be stated as a simple fact. Additionally, based on the text and figure 2 (Backed by figure 5, 10, 11, 12, 13, 14) it does not seem to be the case, that the model really works with Videos. Perception CoT seems to be non-temporal, but based on a single frame. While this could be for visualization purposes and Explanation, an additional Statement or Explanation would be helpful. Otherwise it is not really understandable wether the Pipeline really conditions on temporal reasoning (which seems to be claimed) or not.

### Questions
1. Could you Elaborate on why a key Limitation of VLM-based E2E Systems is "Lack of Chain-of-Thought"? 
2. Could you clarify wether you are using temporal Video data your VLM conditions on, or you are aggregating Frames/ Just conditioning on one frame? 
3. How did you validate the CoT Dataset gt?
4. Can you Showcase that CoT reduces the error Propagation and the gap reduction really comes from using CoT and not from foundation model Training/ higher capacity of the model generally ?

### Soundness
2

### Presentation
4

### Contribution
3
