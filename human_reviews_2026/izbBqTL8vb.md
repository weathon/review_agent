# Perception-Aware Policy Optimization for Multimodal Reasoning

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has proven to be a highly effective strategy for empowering Large Language Models (LLMs) with long chain-of-thought reasoning abilities. However, its design and optimizations remain tailored to purely textual domains, resulting in suboptimal performance when applied to multimodal reasoning tasks. In particular, we observe that a major source of error (67%) in current multimodal reasoning lies in the perception of visual inputs. To address this bottleneck, we propose PAPO, a novel policy gradient algorithm that encourages the model to generate visually grounded reasoning without external supervision. Specifically, we introduce the Implicit Perception Loss in the form of a KL divergence term, which maximizes the difference between two probability distributions over the same rollout sequence, conditioned on either the original or corrupted visual input. Notably, PAPO does not rely on any additional data annotation, reward models, or stronger teacher models, and can therefore be seamlessly integrated into mainstream RLVR algorithms such as GRPO and DAPO. To further enhance the training stability of PAPO, we introduce the Double Entropy Loss, which effectively regularizes the new KL objective without compromising performance. Despite its simplicity, PAPO yields significant overall improvements of 4.4%-17.5% on diverse multimodal benchmarks. The improvements are more pronounced, approaching 8.0%-19.1%, on tasks with high vision dependency. We also observe a substantial reduction of 30.5% in perception errors, indicating improved perceptual capabilities with PAPO. Overall, PAPO offers a new perspective on advancing multimodal RLVR via the optimization objective, moving beyond rollout or reward design and pointing toward deeper integration of perception and reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper identifies a critical bottleneck in large multimodal models (LMMs): most failures in complex reasoning tasks arise from visual perception errors rather than logical flaws. To address this, the authors propose Perception-Aware Policy Optimization (PAPO), a novel algorithm that extends Reinforcement Learning with Verifiable Rewards (RLVR) frameworks such as GRPO and DAPO.
PAPO introduces two key components:
Implicit Perception Loss, which encourages visually grounded responses by maximizing the KL divergence between output distributions conditioned on original versus corrupted visual inputs.
Double Entropy Loss, which regularizes the objective and improves training stability.
Experiments on eight multimodal reasoning benchmarks show a 4.4%–17.5% overall improvement and a 30.5% reduction in perception errors.

### Strengths
- Clear and Important Motivation: The work is well-grounded in a detailed error analysis showing that perception errors account for the majority of LMM failures. This provides strong motivation for the proposed approach.
- Novel and Efficient Design: PAPO’s use of an implicit “reverse KL” objective to enforce visual grounding is elegant and requires no extra annotations, reward models, or teacher models, making it an efficient drop-in replacement for existing RLVR methods.
- Comprehensive Experiments: The paper evaluates multiple benchmarks and model sizes, includes ablations on masking strategies and loss weighting, and provides in-depth analysis of failure modes and stability mechanisms.

### Weaknesses
- Flawed Core Assumption: The hypothesis that model confidence should always decrease when visual information is removed is not universally valid. Robust models should ignore irrelevant visual cues, meaning PAPO may incorrectly penalize robustness in cluttered or partially relevant visual scenes (Although their mask strategies seem good).
- High Computational Overhead: PAPO increases per-step training time by over 40% for the 7B model, while the accuracy gains are often modest, making the cost–benefit ratio questionable.
- Metric-Method Collusion Risk: The primary evaluation metric (accuracy@8) can be artificially inflated by low-entropy policies. PAPO’s entropy regularizer may bias results toward deterministic outputs rather than genuine reasoning improvements.
- Counterintuitive Ablation Results: The finding that random masking outperforms semantic masking suggests the method may be learning brittle correlations rather than true visual grounding, can authors make more comparisons on this problem.

### Questions
Please finish the concerns in the weakness

### Soundness
4

### Presentation
4

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
The paper proposes Perception-Aware Policy Optimization (PAPO), a new reinforcement learning framework for large multimodal models that integrates perceptual awareness directly into policy optimization. Unlike prior works, PAPO jointly learns both perception and reasoing within the core RL objective. The key idea is an implicit perception loss which encourages visually grounded reasoning without external supervision. A double entropy regularizer is introduced to stabilize training. PAPO serves as a plug-in-play replacement for GRPO/DAPO and achieves 4.4-17.5% average improvement across eight multimodal reasoning benchmarks, with up to 19.1% gain on vision-intensive tasks and 30.5% fewer perception-related errors. The method is simple, annotation-free and computationally efficient, showing faster convergence and more robust multimodal understanding

### Strengths
1. The paper introduces a novel idea of integrating perception awareness directly into the policy optimization objective, moving beyond traditional data- or reward-level modifications in multimodal RL.

2. The methodology is well-motivated, theoretically sound, and empirically validated. The authors provide thorough experiments across eight multimodal reasoning benchmarks, showing consistent and interpretable gains over strong baselines such as GRPO and DAPO. The analysis includes detailed ablations and error breakdowns, supporting the claimed contributions.

3. The paper is clearly written and logically organized. The motivation, derivation of the objective, and implementation details are easy to follow. Figures and examples effectively illustrate both intuition and quantitative results.

### Weaknesses
1. The experiments are conducted only on the Qwen2.5‑VL model (3B/7B) and do not include other pretrained models (e.g., InternVL).

2. The paper lacks a discussion of scalability. Specifically, whether the proposed method remains effective (and efficient) when applied to much larger models (e.g., ~30B-70B parameters).

3. The paper could porvide a deeper discussion of how the proposed method differs (in objective, reward design, computational cost, etc.) from those works would improve the clarity of novelty:

  [1] Wang, Jiaqi; Lin, Kevin Qinghong; Cheng, James; Shou, Mike Zheng. Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models.

  [2] Shen, Junhao; Zhao, Haiteng; Gu, Yuzhe; Gao, Songyang; Liu, Kuikun; Huang, Haian; Gao, Jianfei; Lin, Dahua; Zhang, Wenwei; Chen, Kai. Semi-off-Policy Reinforcement Learning for Vision-Language Slow-Thinking Reasoning.

  [3] Wei, Yana; Zhao, Liang; Sun, Jianjian; Lin, Kangheng; Yin, Jisheng; Hu, Jingcheng; Zhang, Yingmin; Weng, Zejia; Wang, Jia; Han, Chunrui; Peng, Yuang; Han, Qi; Ge, Zheng; Zhang, Xiangyu; Jiang, Daxin; Patel, Vishal M. Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning.

  [4] Liu, Ziyu; Sun, Zeyi; Zang, Yuhang; Dong, Xiaoyi; Cao, Yuhang; Duan, Haodong; Lin, Dahua; Wang, Jiaqi. Visual Reinforcement Fine-Tuning.

  [5] Tan, Huajie; Ji, Yuheng; Hao, Xiaoshuai; Lin, Minglan; Wang, Pengwei; Wang, Zhongyuan; Zhang, Shanghang. Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning.

### Questions
Could the authors clarify why maximizing the reverse KL divergence between rollouts with original and masked visual inputs improves perception–reasoning coupling? A brief theoretical or deeper empirical explanation would help justify this design choice.

Apart from this question, I have no further concerns.

### Soundness
4

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
This paper proposes Perception-Aware Policy Optimization (PAPO), an RLVR algorithm. It addresses dominant visual perception errors in multimodal LLMs by integrating an Implicit Perception Loss (KL divergence with masked visual inputs) into GRPO, fostering visually grounded reasoning. A Double Entropy Loss ensures training stability against "loss hacking." PAPO achieves significant performance boosts on multimodal benchmarks, notably without external data or reward models.

### Strengths
1.The paper convincingly identifies and empirically quantifies visual perception errors as a primary source of failure in multimodal reasoning, providing a strong foundation for its contribution.
2. PAPO's core idea of using an Implicit Perception Loss to intrinsically encourage visually grounded reasoning, alongside a robust Double Entropy Loss for stability, is novel, intuitive, and empirically proven to yield substantial performance gains.
3. PAPO achieves state-of-the-art improvements without relying on additional data curation, external reward models, or stronger teacher models, making it a highly practical and accessible method for enhancing multimodal LLMs.

### Weaknesses
1.I am concerned about the result where random masking empirically outperforms DINOv2-based semantic-aware masking. The authors' hypothesis that semantic-aware masking obscures entire salient regions, leading to indiscriminate attention, requires more robust empirical validation. This outcome, coupled with the observation that pixel-level noise is less effective at obscuring informative semantics, prompts a critical need for deeper analysis into the optimal balance between information retention and strategic perturbation. 
2.While the paper commendably demonstrates PAPO's effectiveness in reducing perception errors, particularly on high vision-dependency tasks, I am concerned about the potential implications for tasks with low visual dependency. Explicitly enforcing visual grounding via the Implicit Perception Loss in such scenarios might introduce unnecessary computational overhead or divert the model's attention away from pertinent textual information. I recommend that the authors conduct dedicated sub-dataset analyses on low-dependency tasks and/or perform a thorough bad-case analysis to understand PAPO's behavior in these contexts. Furthermore, while Figure 1 indicates a significant reduction in perception errors, it is imperative to analyze whether this reduction leads to a compensatory increase in other error categories (e.g., reasoning or calculation errors).

### Questions
Please refer to the Weaknesses section above for details.

### Soundness
3

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
- This paper introduces PAPO for multimodal reasoning to encourage visually grounded generation by modifying the core optimization objective, addressing the finding that perception errors on models trained with standard RLVR account for 67% of failures.
- PAPO incorporates an Implicit Perception Loss, a KL divergence term that maximizes the difference between output distributions conditioned on original versus masked visual inputs, combined with a Double Entropy Loss for regularization.
- The algorithm, which requires no additional data or reward models, is integrated into GRPO and DAPO frameworks and evaluated on several benchmarks, demonstrating consistent performance improvements and reduction in perception errors.

### Strengths
- This work addresses an important problem of perception error in multimodal reasoning, starting from a clear, evidence-based motivation to achieve practical performance gains.
- The method effectively combines known techniques, such as visual input masking and entropy regularization, with existing RLVR frameworks (GRPO/DAPO) in a simple yet potent way, yielding significant benefits without making the framework overly complex.
- The proposed methodology is validated by extensive analyses, covering hyperparameter impacts, masking strategies, and training dynamics, including a transparent investigation of the failure mode.

### Weaknesses
- My main concern is the incomplete experimental comparison to establish its state-of-the-art significance. The paper references a number of existing works that modify GRPO/DAPO for multimodal reasoning, yet the experiments in Table 1 only compare PAPO against the original GRPO and DAPO baselines. Without a comparative analysis against other contemporary SOTA variants under identical training conditions, experimental significance seems limited.
- The framework's generalizability is undermined by its exclusive reliance on the Qwen2.5 model family. While this limitation is acknowledged in Section 7, it is difficult to determine if this is a broadly effective approach or an architectural artifact of Qwen.
- The paper's readability and logical flow are hampered by its over-reliance on the appendix. Critical analyses that form the core of the method's justification are deferred from the main text. This forces the reader to constantly switch contexts to verify the paper's central claims on stability and robustness. For the self-containedness of the main paper, consider aggregating key experimental highlights from the Appendix as a concise figure.

### Questions
Please refer to the weaknesses section for detail. Relatively minor questions:
- Considering the benchmarks are largely skewed toward math and geometry, it would be interesting to see how an OCR-based masking strategy (low masking ratio but information-wise critical) would perform with PAPO.
- While the Double Entropy Loss plays a crucial role in stabilizing PAPO, its own sensitivity to hyperparameter choice is not explored.

### Soundness
3

### Presentation
2

### Contribution
2
