# Listener-Augmented Thinking VLMs for Robust Text-to-Image Alignment

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Improving the generalization and robustness of reward models for visual preferences is crucial-especially under limited annotated data, where collection is costly and complex. While we show that reinforcement learning with verifiable rewards (RLVR) improves generalization, we identify a key failure mode: models often produce final answers that are inconsistent with their own chain-of-thought (CoT); when the same trace is shown to a frozen vision–language model (VLM), it frequently predicts a different choice, and accuracy drops as this inconsistency grows. To solve this, we propose a listener-augmented GRPO objective: a frozen listener re-evaluates the reasoner’s CoT (with the final answer masked) and provides a dense soft reward, combined with an exact-match term for unbiased supervision. We further introduce a probability-based pairwise scoring scheme: at training time it yields calibrated confidence rewards for GRPO, and at inference time it supports an anchor-based strategy that can avoid $\mathcal{O}(n^2)$ comparisons while still producing reliable win probabilities. Under this framework, we establish a clear hierarchy - \textbf{single-image scoring $<$ pairwise comparison $<$ RL with CoT $<$ RL with CoT + listener} - showing that conditioning on both images already boosts generalization, CoT improves further, and listener shaping provides the largest gains. Despite training on only 16\% of HPDv2, our approach achieves 67.4\% on the ImageReward test set and improves OOD accuracy by up to +6 pp on a modern 1.2M-vote benchmark, while reducing contradiction rates. Integrated into a Flow-GRPO text-to-image pipeline and evaluated with the latest HPSv3 metric on prompts from HPDv2 and T2I-CompBench, it yields consistent gains. We will release code and models to facilitate reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes listener-augmented GRPO for visual preference modeling in text-to-image alignment. The core finding is a “listener disagreement” failure mode: GRPO-trained reasoners often produce final answers that contradict their own CoT when re-evaluated by an independent (frozen) VLM. The method adds a soft, pre-answer listener reward—the listener reads the reasoner’s CoT with the final token masked and returns a calibrated probability of the correct choice—which is combined with an exact-match term and a formatting check in the GRPO objective.   Using only 16% of HPDv2, the approach reaches 67.4% on ImageReward test, improves OOD accuracy by up to +6pp on Rapidata-HSP, reduces contradiction rates, and yields consistent gains when used as a reward in a Flow-GRPO pipeline for Stable Diffusion 3.5-Medium, evaluated with HPSv3 on HPDv2 and T2I-CompBench prompts.

### Strengths
Clear failure-mode identification: The paper empirically surfaces and quantifies listener disagreement (misalignment between explanations and decisions), which is both intuitive and practically important for CoT-based preference models.

Simple, effective objective shaping: The soft listener reward is well-motivated (uses all data, avoids thresholds, improves credit assignment) and integrates cleanly into GRPO without extra labels or a value head.

Ablations and hierarchy: The paper articulates and validates a compelling hierarchy—single-image < pairwise < RL+CoT < RL+CoT+listener—seen across proxy and end-to-end tracks.

### Weaknesses
1. Listener choice & calibration:  The current setup fixes the listener to Qwen2.5-VL-7B-Instruct, but it remains unclear how sensitive the results are to the choice of listener backbone or model size. The findings may heavily depend on the specific listener used, which raises concerns about generalizability. 

If the listener essentially functions as an LLM-as-a-judge, it raises the question of whether there is any fundamental difference between the two paradigms. Do listener-based evaluations and LLM-as-judge settings differ in principle, or are they simply variants of the same idea under different names? If not, similar issues of bias and calibration are likely to persist. In that case, one may further ask: why not employ a stronger or more expert model (e.g., GPT-family models) as the listener to obtain more reliable judgments and better isolate the effect of the proposed method from limitations of the chosen model?

2. Statistical rigor & stability reporting: Most tables lack confidence intervals, multi-seed variance, and significance tests. Given the small but meaningful gains (e.g., 67.2→67.4%), uncertainty estimates are essential.

3.Modeling comparisons: The probability-based decision is effectively a binary softmax; comparisons to classical Bradley–Terry/Luce models, Plackett–Luce or DPO-style direct preference optimization under pairwise conditioning are limited.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Listener-Augmented GRPO, a reinforcement learning framework that improves the robustness and generalization of vision–language models for text-to-image preference alignment. The key idea is to use a frozen “listener” model to re-evaluate the reasoner’s chain-of-thought and provide a soft reward reflecting reasoning consistency. Trained with only 16% of HPDv2 data, the method achieves 67.4% on ImageReward, improves OOD accuracy by +6 points, and reduces reasoning–answer contradictions. Integrated into a Flow-GRPO pipeline, it yields consistent gains in generation quality, offering a simple and effective approach for scalable preference alignment.

### Strengths
1. This paper is well-structured, making it accessible to readers with varying levels of expertise in the field.

2. The idea of incorporating a listener model in the GRPO training pipeline is reasonable and well-motivated, and the results in this paper generally support it.

3. The authors reveal a clear hierarchy in human preference modeling with vlms: single-image < pairwise < RL with CoT < listener-augmented RL with CoT.

### Weaknesses
1. The overall presentation of the paper can be improved. In particular, the paper lacks an intuitive illustration of the proposed method (e.g., a pipeline diagram or an algorithm), which makes it difficult for readers to quickly grasp the overall methodology. Moreover, the figures in this manuscript could be substantially refined for better clarity and visual quality.

2. The benefits of the proposed method appear to be rather minor. The experimental results reported in the paper (e.g., Table 2) do not clearly demonstrate a significant improvement. Furthermore, the experiments on Flow-GRPO also do not show substantial gains in visual quality (e.g., cases 3, 4, and 5 in Figure 1). I would also like to ask the authors why SD 3.5 was used for the experiments, rather than a more advanced T2I model such as Flux (with improved visual quality), which has been widely adopted in Flow-GRPO and other diffusion GRPO methods like DanceGRPO, TempFlow-GRPO and MixGRPO.

3. The experiment part seems limited. The baselines compared are few, and the ablation studies are insufficient. For example, the coefficients in Equation 3 are set to 0.5, but the authors do not provide any discussion or justification for this choice.

### Questions
Please refer to the weaknesses mentioned above. Considering the relatively low level of completeness and clarity in the presentation of the current version, I find it difficult to vote for acceptance. From my point of view, the authors could carefully revise the manuscript.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a method called Listener-Augmented GRPO (Listener-Augmented Group Relative Policy Optimization) to enhance the robustness and accuracy of Vision-Language Models (VLMs) in text-to-image generation tasks. The approach involves adding a frozen listener model that re-evaluates the chain of thought (CoT) and provides a dense soft reward. This helps resolve the issue where traditional methods produce inconsistencies between the reasoning and the final answer. Despite being trained on a limited dataset, the method still achieves high accuracy on out-of-distribution datasets and reduces contradictions between reasoning and the final answer.

### Strengths
1. The paper identifies and rigorously quantifies a realistic and underexplored failure mode, listener inconsistency, providing clear problem framing and metrics.
2. It integrates frozen-listener soft scoring into GRPO, offering dense, stable, and label-free reward shaping; together with dual-image conditioning and structured JSON two-token decisions, this forms a replicable and efficient training recipe.

### Weaknesses
1. In Figure 1, the two dialogue answers are the same? It is recommended to add multiple examples.
2. There is a lack of sensitivity analysis for key hyperparameters (such as specific parameters in the Listener reward).
3. The experiment is not sufficient, and there is inadequate analysis of the dataset's diversity. The model's performance in different scenarios has not been fully explored.
4. The method in the paper still relies on reinforcement learning and chain-of-thought (CoT), which have already been widely applied in vision-language models. The innovation of the Listener mechanism is quite conservative and does not present a fundamental breakthrough.
5. There is insufficient comparison with existing cutting-edge methods, making the innovation seem limited.

### Questions
1. How does the choice and capability of the listener affect results? Would stronger or weaker VLM listeners change reward stability or bias, and is there an optimal “listener–reasoner capacity ratio”?
2. The reward combines format penalties, exact-match checks, and listener probabilities. How sensitive is performance to their weighting? Could adaptive or uncertainty-aware weighting further stabilize training?
3. Is there theoretical or statistical grounding for the anchor sampling consistency when n > 2? How do anchor number and sampling policy influence bias and variance under long-tailed or multimodal candidate quality distributions?
4. Since the listener reads the CoT to assess persuasiveness, how does performance vary with short, missing, or noisy CoTs, or under reflective prompting like self-ask?

### Soundness
2

### Presentation
2

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
The paper targets the challenging problem of improving the generalization and robustness of reward models (RM) for test-to-image problem. The main motivation is to improve the inconsistent prediction of RM against the COT reasoning. A listener model is introduced to provide a reward. The proposed model is easily reproduced and reasonable performance has been reported in the experimental section.

### Strengths
- The problem of inconsistent COT with the VLM final prediction is an important problem due to the limited performance of the existing VLM models. Thus, it would be helpful if the inconsistent result can be addressed by a listener model and will further improve the performance of the text-to-image models. 

- The idea is simple to implement and therefore it would be easy to reproduce.

### Weaknesses
- The experiments are not sufficient to validate the effectiveness of the proposed algorithm. First, as the main idea is to improve the flow-GRPO for the text-to-image problem. I would strongly suggest to provide more evaluations on the final results on the text-to-image benchmarks. For example, the wildly used benchmark like GenEval, DPG-Bench, and OneIG benchmark can be evaluated. Second, there would usually exist reward hacking during the RL training. How to justify the improvement of reward score is consistent with the visual improvement rather than the result due to the reward hacking. 

- As the paper is based on the assumption of inconsistent result between the COT trace and final prediction, will the issue still exists for the larger VLM models (Qwenvl2.5-72B) or the recent models (e.g., qwen3-vl)? If the improvement of the VLM models well solves the issue, we will not need the additional listener model. 

- Is the proposed algorithm can be generalized to other text-to-image backbones, for example, Flux or Qwen-image?

- The novelty of paper is limited. The involvement of a listener is interesting but is not enough for an ICLR paper.

### Questions
- Please mainly address the question in the weakness section. More specially, please provide the sufficient experiment evaluations on the text-to-image benchmark because the improvement of the reward model score does not usually leading to the improvement on the final text-to-image performance.

### Soundness
2

### Presentation
2

### Contribution
2
