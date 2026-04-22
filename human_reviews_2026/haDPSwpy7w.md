# Beyond Monolithic Rewards: A Hybrid and Multi-Aspect Reward Optimization for MLLM Alignment

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 2

## Abstract
Aligning multimodal large language models (MLLMs) with human preferences often relies on single-signal, model-based reward methods. Such monolithic rewards often lack confidence calibration across domain-specific tasks, fail to capture diverse aspects of human preferences, and require extensive data annotation and reward model training. In this work, we propose a hybrid reward modeling framework that integrates complementary reward paradigms: (i) model-based rewards, where a learned reward model predicts scalar or vector scores from synthetic and human feedback, and (ii) rule-based rewards, where domain-specific heuristics provide explicit correctness signals with confidence. Beyond accuracy, we further incorporate multi-aspect rewards to enforce instruction adherence and introduce a generalized length-penalty reward to stabilize training and improve performance. The proposed framework provides a flexible and effective approach to aligning MLLMs through reinforcement learning policy optimization. Our experiments show consistent improvements across different multimodal benchmarks when applying hybrid and multi-aspect reward modeling. Our best performing model in the 3B family achieves an overall average improvement of 9.5% across general and math reasoning tasks. Focusing specifically on mathematical benchmarks, the model achieves a significant average improvement of 16%, highlighting its effectiveness in mathematical reasoning and problem solving.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes HARMO, a hybrid and multi-aspect reward optimization framework for GRPO-style training. The core idea is a reward engineering technique that uses hybrid rewards, including rule-based reward, reward model-based reward, as well as multi-aspect rewards, including format reward, and lenght-penalty reward. HARMO outperforms baslines on various benchmarks.

### Strengths
1. This paper presents a clear and practical hybrid reward design: the combination of verifiable rule-based and open-ended RRM reward, which addresses a known weakness of monolithic scalar rewards in multimodal settings.
2. This paper proposes a GRPO variant with modified advantage: modifying the advantage calculation to use only the group mean as a baseline, which is a concrete algorithmic choice to reduce difficulty-dependent bias.

### Weaknesses
1. The paper is limited in novelty. The core algorithm is a modest variant of GRPO: the primary novelty lies in reward engineering, which is a simple combination of rule-based and reward model-based rewards. The format reward is standard; the length penalty is clever but relies on group sampling and the existence of at least one “correct” sample in the group.
2. Eq. 3 implies equal weighting, but the component scales differ. Without explicit weights or normalization, training may be dominated by whichever term has larger magnitude. There is no sensitivity analysis of P_{max}, τ, or R{fmt} magnitudes.
3. Advantage normalization change: Removing std is motivated by difficulty-dependent bias, but there is no ablation on this specific choice within HARMO.
4. The baselines in the experiments are not clearly expressed. 
5. Open-ended scoring with cosine similarity to a single reference contradicts the claim that these tasks are not verifiable. It risks punishing diverse, equally good answers, especially in VQA with many acceptable phrasings.

### Questions
1. How are R^{hybrid}, R^λ, and R^{fmt} scaled or weighted relative to each other? 
2. How to differentiate which responses are verifiable and which are open-ended?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HARMO, a novel Hybrid and Multi-Aspect Reward Modeling Optimization framework for aligning MLLMs. It tackles the limitations of monolithic rewards by integrating hybrid accuracy rewards (combining rule-based verification for verifiable tasks and model-based feedback for open-ended tasks) and multi-aspect behavioral rewards (including a generalized length penalty to prevent brevity-based reward hacking and a format adherence reward). Experiments show HARMO significantly improves MLLM performance, particularly in mathematical reasoning.

### Strengths
1. This paper introduces a novel hybrid reward system and a highly effective generalized dynamic length penalty, directly addressing critical limitations of current MLLM alignment methods.
2. This is a high-quality research with meticulous methodology, comprehensive ablation studies, robust empirical validation across diverse benchmarks, and statistical rigor.
3. This paper is well-written, logically structured, with clear explanations of complex concepts, and supported by informative figures, tables, and illustrative case studies.
4. This paper makes a substantial contribution by tackling fundamental MLLM alignment challenges, leading to significant performance improvements in reasoning, and offering a practical, scalable framework for future research.

### Weaknesses
1. Supplementary material images have low resolution, higher resolution is recommended.

### Questions
1. Given that the proposed hybrid and multi-aspect reward components (rule-based, model-based, length, format) and loss function are largely modality-agnostic and could be directly applied to pure-text LLM, how does HARMO's methodology offer a strong design specifically for cross-modal alignment that differentiates it from a generically applied LLM alignment technique?
2. While individual reward components like rule-based, model-based, length penalties (e.g., DAPO[1]), and format adherence (e.g., GRPO) have been previously explored, please clarify the specific novelty or unique differentiating factors of HARMO's combined reward formulation and its integration strategy that contribute to its claimed efficacy for MLLM alignment.
3. Your modification to the GRPO advantage function removes the standard deviation for "more stable and unbiased learning" based on an argument of "difficulty-dependent bias." Could you provide a more rigorous mathematical argument or empirical analysis to substantiate this claim, especially regarding the theoretical implications for policy gradient estimation and training stability?
4. The Skywork7B RM is central to your hybrid reward. Could you provide more details on its training data, particularly regarding its alignment with human preferences on diverse MLLM tasks? Are there known biases or limitations of this RM that could impact the HARMO training process?
[1] Yu Q, Zhang Z, Zhu R, et al. Dapo: An open-source llm reinforcement learning system at scale.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors present a reward modeling framework, termed HARMO, for MLLMs alignment. The HARMO reward is a combination of rule-based, model-based, and behavioral rewards. Empirical evaluations on mathematical reasoning and VQA benchmarks demonstrate the effectiveness of the proposed reward model.

### Strengths
1. The idea of buidling a hybrid reward combining rule-based rewards and model-based rewards is insightful, as well as the finding of the strong bias towards shorter outputs for RL aligned models.

2. The hybrid reward yields significant performance gains over traditional RM-based baselines on a diverse set of benchmarks covering mathematical, general VQA, and OCR-based vision tasks.

### Weaknesses
1. It is not clear in Eq. (1) how to determine a response is verifiable and open-ended? When calcuating the cosine similarity, how to obtain the reference response? Then, a decision threshold is needed? Moreover, as I understand, determining whether a question-and-answer task is open-ended or verifiable primarily depends on the nature of the question, not the response.

2. In Eq. (3), why not use three weighting parameters to integrate the three reward components? How to guarantee a balance among them, giving the varying scales of their values?

3. To directly show the advantages of the HARMO reward model, I suggest that the authors can conduct experiments on some MM reward benchmarks, e.g., Multi-Modal Reward Bench (M. Yasunaga, arXiv:2502.14191v1, 2025), VL Reward Bench (L. Li, CVPR 2025) and MM-RLHF-Reward Bench (Y. Zhang, ICML, 2025).

4. The experimental results show that the HARMO significantly enhances math reasoning capability. Could you validate the method on more complex visual reasoning tasks in high-definition images? I suggest the author do some validation on the MME-RealWorld benchmark (Y. Zhang, ICLR 2025.)

### Questions
Please find the questions in the weaknesses part.

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
This paper proposes a hybrid reward optimization framework for aligning Multimodal Large Language Models (MLLMs). The framework combines model-based rewards, rule-based rewards to move beyond what the authors describe as "monolithic" single-signal rewards. The method is primarily evaluated on math and visual question-answering (VQA) tasks.

### Strengths
It's somewhat valuable to apply the rule-based + model-based rewards to a vision-language model on math/reasoning and VQA tasks and do the ablation study.

### Weaknesses
+ The primary issue with this paper is its lack of originality. The authors correctly identify that simple rule-based rewards are often limited to deterministic tasks and unsuitable for nuanced, open-ended feedback. However, the hybrid methodology they propose to solve this (combining different reward signals, including process-based feedback) is not new. A very similar paradigm was presented in the DeepSeek-R1 model, which was released substantially earlier (approx. 10 months ago) than this submission. The authors may mean DeepSeek-R1-**Zero** in line 52. Therefore, the core contribution of this paper appears to be an application of a pre-existing technique.
+ The paper claims to be about MLLM "alignment," which implies a broad improvement in helpfulness and safety across diverse tasks. However, the experiments and reward design are overwhelmingly focused on the math/reasoning domain.
+ The method fails to demonstrate a clear benefit. On the VQA tasks, the model's performance actually decreases slightly on docvqa and chartqa (7b) compared to the baseline. Also, this baseline is only an instruct-model (Qwen2.5-VL-instruct).

### Questions
The scores on lines 423 to 429 do not appear to match directly with those in Table 5. I guess the scores are relative improvements?

### Soundness
2

### Presentation
3

### Contribution
1
