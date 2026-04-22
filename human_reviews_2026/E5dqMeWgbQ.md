# Look Less, Reason More: Rollout-Guided Adaptive Pixel-Space Reasoning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 4

## Abstract
Vision-Language Models (VLMs) excel at many multimodal tasks, yet they frequently struggle with tasks requiring precise understanding and handling of fine-grained visual elements. This is mainly due to information loss during image encoding or insufficient attention to critical regions. Recent work has shown promise by incorporating pixel-level visual information into the reasoning process, enabling VLMs to access high-resolution visual details during their thought process. However, this pixel-level information is often overused, leading to inefficiency and distraction from irrelevant visual details. To address these challenges, we propose the first framework for adaptive pixel reasoning that dynamically determines necessary pixel-level operations based on the input query. Specifically, we first apply operation-aware supervised fine-tuning to establish baseline competence in textual reasoning and visual operations, then design a novel rollout-guided reinforcement learning framework relying on feedback of the model's own responses, which enables the VLM to determine when pixel operations should be invoked based on query difficulty. Experiments on extensive multimodal reasoning benchmarks show that our model achieves superior performance while significantly reducing unnecessary visual operations. Impressively, our model achieves 73.4\% accuracy on HR-Bench 4K while maintaining a tool usage ratio of only 20.1\%, improving accuracy and simultaneously reducing tool usage by 66.5\% compared to the previous methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an adaptive pixel-space reasoning framework for vision-language models (VLMs) in multimodal reasoning tasks. The core innovation lies in using rollout-guided reinforcement learning (RGRL) to enable models to dynamically determine when to invoke pixel-level operations (e.g., zoom-in) rather than applying them indiscriminately. The method consists of two stages: (1) operation-aware supervised fine-tuning (SFT) to establish baseline competence, and (2) RGRL training that learns query-specific tool usage strategies through pixel necessity rollouts and adaptive rollouts. Experiments on five multimodal reasoning benchmarks validate the effectiveness, achieving 73.4% accuracy on HR-Bench 4K while maintaining only 20.1% tool usage ratio.

### Strengths
- Clear Problem Motivation: The paper effectively identifies the key issue with existing pixel-space reasoning methods—overuse or neglect of pixel-level operations—and provides strong justification for the necessity of adaptive strategies.

- Novel Method Design: The rollout-guided RL framework is cleverly designed, dynamically estimating tool necessity through pixel necessity rollouts (forced tool use/no-tool use) and then guiding the model in adaptive rollouts without requiring manual annotation.

- Well-Designed Reward Functions: The multi-dimensional reward design (instruction-following, adaptive tool-necessity alignment, rollout consistency) is comprehensive, encouraging both correctness and efficiency with clear mathematical constraints on reward coefficients (e.g., b₂ > b₃, c₃ > c₂).

- Comprehensive and Convincing Experiments: (1) Evaluation on 5 diverse benchmarks covering fine-grained perception and high-level reasoning. (2) Comparison with multiple strong baselines (including GPT-4o, Gemini series, Pixel Reasoner). (3) Reports both accuracy and tool usage ratio, demonstrating efficiency gains. 

- Clear Writing: Well-structured paper with formalized problem definition, clear method description, and intuitive visualizations.

### Weaknesses
- Insufficient Computational Cost Analysis: Although tool usage is reduced, RL training requires 16 rollouts per query (n₁=4, n₂=4, n₃=8), potentially incurring high training costs. No reporting of training time, GPU hours, or training cost comparison with baselines. While tool usage is reduced at inference, no comparison of actual inference speed (FPS or seconds/sample)

- Unexplored Hyperparameter Sensitivity: Multiple hyperparameters in reward functions (b₁, b₂, b₃, c₁, c₂, c₃, λ_instr, λ_align, γ). Lacks sensitivity analysis for these hyperparameters; unclear how to select values or whether task-specific tuning is needed.

- Selection Bias in Case Studies: Cases in appendix mainly show successes; failure case analysis missing. Unclear under what conditions adaptive strategy fails and model limitations

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a framework for adaptive pixel-space reasoning in Vision-Language Models (VLMs), aiming to address the problem of overusing or underusing pixel-level operations (e.g., zoom-in) in multimodal reasoning tasks. The approach consists of two training stages: (1) operation-aware supervised fine-tuning (SFT) to establish baseline competence in both textual chain-of-thought and pixel operations, and (2) rollout-guided reinforcement learning (RGRL) that learns when to invoke pixel operations based on query difficulty. The method generates multiple rollouts per query—some forced to use tools, some prohibited from using tools, and some adaptive—and uses the relative performance to estimate tool necessity and guide policy learning.

### Strengths
1. The paper identifies a genuine problem in current pixel-space reasoning approaches, the indiscriminate use of visual operations regardless of necessity. 
2. The rollout-guided RL framework is creative in its approach. Using "pixel necessity rollouts" (forced tool use vs. prohibited tool use) to implicitly estimate whether tools are beneficial for a given query is interesting. The reward design that considers both correctness and alignment with estimated necessity shows thoughtful engineering.

### Weaknesses
1. I think this paper miss some comparison against several important recent tool-augmented baselines in the pixel-space reasoning literature:
- Mini-O3: A recent model specifically designed for visual reasoning with tool use
- DeepEyes: Cited in related work but not compared experimentally despite being highly relevant
- Simple O3: Also cited but missing from comparisons

2. Several relevant recent works are not cited:
- Learning Only with Images (arXiv:2507.20766): Directly relevant to pixel-space reasoning
- PyVision (arXiv:2507.07998): Another tool-use framework for VLMs
- LATTE (EMNLP 2025): Related to adaptive reasoning in multimodal models
- ReVPT: A reinforcement learning approach for visual perception with tools

3. Perhaps most concerning, the paper only evaluates on fine-grained visual reasoning benchmarks (V*Bench, MMStar, HR-Bench, InfoVQA) and completely omits standard general-purpose VLM benchmarks such as: MMBench, MMMU, POPE, MMvet, etc. This omission raises a critical question: *Does the specialized training for adaptive pixel-space reasoning harm general VLM capabilities?*

### Questions
1. While the paper shows cases where the method succeeds, there is insufficient analysis of when and why the adaptive strategy fails. For instance:
- What happens when the pixel necessity estimation is incorrect?
- Are there systematic patterns in which types of queries lead to wrong tool-use decisions?
- How sensitive is the method to the hyperparameters in the reward functions (b1, b2, b3, c1, c2, c3)?

2. Can you provide analysis of which types of queries benefit most from tool use vs. pure textual reasoning? This would help understand what the model has learned about tool necessity.

### Soundness
2

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
This paper studies the problem of training VLMs to adaptively utilize pixel-level operations to accomplish tasks requiring understanding of fine-grained visual information. It proposes a framework to train VLMs to only invoke pixel-level operations when necessary, improving tool-use efficiency. Experiment results show good performance compared to models without tool-use and more efficient tool-use ratio compared to prior work.

### Strengths
1. The paper is written clearly and easy to understand, with good motivation and detailed introduction of related work and proposed method.

2. The method design is intuitive and experiment results show the effectiveness of the method.

3. I appreciate the qualitative case study which gives the intuition on why performing pixel-level operations more frequently is not necessary more beneficial (other than efficiency concern).

### Weaknesses
1. While the approach is intuitive, the overall proposed framework consists of different stages (first SFT and then different stages of RL for pixel necessity rollouts and RL for adaptive rollouts), where each stage has their own different hyperparameters (e.g., reward in Eq.6). Is there a principled way in determining the learning curriculum for different stages, and the hyperparameters?

2. It appears that the pixel-level operation considered in this work is only zoom-in? It would be more interesting if more pixel-level operations can be considered, e.g., segmentation, etc.

3. It seems that Pixel Reasoner is the only one main baseline considered. It is not clear to me why in Table 1 there are a lot of missing numbers for detailed comparisons with other baselines. This makes the comparisons less convincing and is my main concern on the soundness of the paper.

### Questions
1. Beyond visual search related tasks that can benefit from zoom-in operation, are there other visual reasoning tasks that may benefit from other pixel-level operations?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces an adaptive pixel-space reasoning framework for Vision-Language Models (VLMs) that dynamically decides when to invoke pixel-level operations (e.g., zoom-in) based on query complexity.  The method combines operation-aware supervised fine-tuning (SFT) with a rollout-guided reinforcement learning (RGRL) approach, which uses the model’s own rollouts to estimate the necessity of pixel-level tools.  The authors demonstrate state-of-the-art performance on five multimodal reasoning benchmarks while significantly reducing unnecessary tool usage.

### Strengths
- The method achieves superior performance across multiple benchmarks while reducing tool usage. 
- The idea of using rollout-guided RL to adaptively decide when to use pixel-level operations is well-executed.   The approach does not rely on external supervision or hand-crafted rules.
- Extensive experiments and ablation studies validate the contribution of each component (SFT, RGRL, reward design, etc.).

### Weaknesses
- The overall inference time under the proposed motivation is not clearly discussed. This makes the motivation for reducing tool operation time limited. 
- The qualitative results do not show the necessity and improvement of adopting fewer calls of the tool function. Why the model with less reasoning can lead to better results? The performance gain is not clearly explained and discussed.
- The method only considers zoom-in operations. It is unclear whether the framework generalizes to other tool operations.
- The technical contribution is weak in this paper. The authors only provide modifications to the reward function and rollout strategy, which is not technically abundant.

### Questions
Please refer to the weaknesses period. Due to the doubts and unclearness on the motivation and the claim of this submission, I lean towards borderline reject currently.

### Soundness
3

### Presentation
3

### Contribution
2
