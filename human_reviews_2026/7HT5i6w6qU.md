# TinyEye: Sharpening Visual Reasoning of Tiny Models with Offline Policy Optimization

- Decision: Reject
- Scores: 4, 4, 4, 4, 2

## Abstract
Multimodal reasoning with small vision–language models (VLMs) is increasingly important in real-world applications, yet their limited capacity makes optimization and alignment especially challenging. In this paper, we propose a holistic framework for offline policy optimization to sharpen the visual reasoning capabilities of small models. At its foundation is TinyEye-Data, a large-scale corpus of two million reasoning trajectories distilled from state-of-the-art VLMs across 68 verifiable tasks, which provides diverse and reliable binary supervision entirely in the offline setting. We instantiate the framework through a four-stage pipeline: (1) native-resolution warm-up for robust vision–language alignment, (2) instruction tuning on TinyEye-Data to establish a broad reasoning foundation, (3) annealed rejection sampling to mine hard cases and refine supervision, and (4) Discriminative Direct Preference Optimization (DDPO), a new margin-based objective that formulates policy learning as reward classification and resolves the likelihood displacement issues of DPO. Stages (3) and (4) together form the core of verifiable offline reinforcement learning, where rejection sampling refines signals and DDPO optimizes the policy against them. The resulting model, TinyEye-2B, achieves state-of-the-art results across diverse reasoning benchmarks, reaching 50.3% on MMMU, 55.2% on MathVerse, and 63.9% on HallBench, outperforming other models of comparable scale by significant margins.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a four-stage offline pipeline culminating in a core contribution, “Discriminative Direct Preference Optimization (DDPO),” which reframes DPO’s pairwise objective into a margin-based binary classification of response. DDPO is claimed to “resolve the likelihood displacement issue of DPO” and to “guarantee” monotonic reinforcement of positives while suppressing negatives under binary rewards. Empirically, DDPO is reported to beat a DPO baseline.

Other than that, the authors also proposed a TinyEye-Data dataset containing 2M reasoning trajectories from advanced VLMs. Combined with traditional VL alignment training (stage 1), SFT (stage 2) and rejection sampling (stage 3), the resulting model TinyEye-2B is able to achieve competitive results on various reasoning benchmarks.

Overall, I see limited technical contribution in this work and lean to recommend rejection pending author’s discussion.

### Strengths
1. Clear end-to-end pipeline. The four-stage offline recipe (native-resolution warm-up → large instruction-tuning → annealed rejection sampling → Stage-4 preference training) is well structured and practical for small VLMs.

2. Competitive results. DDPO shows meaningful gains over the stated DPO baseline on several benchmarks in the Stage-3 → Stage-4 transition.

3. Claimed mitigation of DPO’s “likelihood displacement.” The paper motivates a concrete pathology and offers a principled surrogate expected to avoid it.

### Weaknesses
1. While the DDPO seems to be the core contribution of this paper, we see mixed evidence of effectiveness vs. DPO on benchmarks. Table 4 shows DDPO isn’t uniformly superior: on MathVista, DPO (68.0) edges DDPO (67.9). The paper emphasizes DDPO’s strong HallBench gains (binary classification-like), which might reflect task-objective alignment rather than a general advantage. 

2. TinyEye-Instruct/Reason are said to be “compact and verifiable,” but the paper provides sparse details on sources, license, overlaps, or leakage control, which is important for fair comparisons and reproducibility.

### Questions
1. Can you provide targeted experiments that provoke likelihood displacement and show DDPO avoids it while DPO fails, holding all else equal (same pairs, same ref, same tuning)?

2. What $\alpha$ works best, how is it chosen, and how do outcomes change across $\alpha$?

3. Could you add DPO-Positive / SimPO / ORPO baselines, tuned comparably, to demonstrate DDPO’s advantages are not limited to default DPO?

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
4

### Summary
This paper addresses the critical challenge of insufficient visual reasoning capabilities in small vision-language models (VLMs) for real-world edge deployment, where existing post-training methods (Supervised Fine-Tuning/SFT, Direct Preference Optimization/DPO, Reinforcement Learning/RL) suffer from overfitting, likelihood displacement, and high computational costs. To solve this, the authors propose TinyEye, a unified offline policy optimization framework designed to enhance the reasoning performance of compact VLMs.

### Strengths
1.TinyEye-Data: A large-scale verifiable corpus of 2 million reasoning trajectories distilled from state-of-the-art teacher VLMs
2.Empirical Validation: The resulting 2B-parameter model, TinyEye-2B, achieves state-of-the-art performance across diverse benchmarks: 50.3% on MMMU (multimodal reasoning), 55.2% on MathVerse (math reasoning), and 63.9% on HallBench (general multimodal QA).

### Weaknesses
1. The proposed method in the paper is only tested on a 2B-parameter model. It lacks experiments on 7B-parameter models, which are commonly used in both research and practical applications. This makes it hard to tell if the method works effectively for 7B-parameter models too.  
2. The training process requires TinyEye-Data, which has 2 million reasoning trajectories. Using such a large amount of data may be inconvenient in real-world scenarios (e.g., situations where data is scarce). However, the paper does not discuss how to solve this problem.  
3. The paper does not study how different image compression qualities (e.g., how much an image is compressed) affect the model’s reasoning results. In practice, images are often compressed, so it is unclear whether the model can still perform well in reasoning when dealing with compressed images.

### Questions
1. Has the proposed method been compared with other RLHF methods such as GRPO on the same dataset?
2. Could you provide the impact of different data compression ratios on reasoning performance?
3. Could you show the performance of a 7B-parameter model on these benchmarks — for example, comparing with similar 7B models like *m2-Reasoning*, which used less than 2M training samples?

### Soundness
3

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
4

### Summary
This paper presents TinyEye, a visual–language reasoning framework designed for tiny multimodal models (around 2B parameters). The method introduces a four-stage offline training pipeline that includes a novel Discriminative Direct Preference Optimization (DDPO) to replace DPO for better stability under binary reward settings. The authors also build a large-scale, verifiable reasoning dataset (TinyEye-Data) distilled from multiple teacher models. Experiments demonstrate significant improvements in multimodal reasoning benchmarks compared to existing open-source models of similar scale.

### Strengths
- The paper is clearly written and well structured; the motivation is well grounded in the “likelihood displacement” problem of DPO.
- Empirical results are strong: TinyEye-2B outperforms competitive open-source models such as InternVL-3 and Qwen2.5-VL, even though these models (might) use much larger training resources.
- The authors provide clear empirical evidence and intuitive explanations showing that DDPO improves training stability and alignment for small models.

### Weaknesses
- Since the base LLM is Qwen3-1.7B, it would be fair to include a direct comparison with InternVL-3.5-2B, which also builds on Qwen3-1.7B. Qwen2.5-VL and InternVL-3 seem somewhat outdated as baselines.

- The full training pipeline (Stage 1–4) is quite complex, involving multiple data generation and filtering steps, which could limit scalability and reproducibility.

- It would strengthen the work to include a comparison with online RL-based methods such as GRPO under a controlled setup.

- As DDPO is model-agnostic, it would be useful to test whether it generalizes to other mainstream VLM families (e.g., Qwen-VL, InternVL series) to confirm its robustness.

### Questions
In Table 1, could you explain why the model performs so well on MathVision and WeMath? Is it because the training data includes similar tasks or distributions?

### Soundness
3

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
3

### Summary
The paper proposes TinyEye, a holistic offline policy optimization framework to improve visual reasoning in compact VLMs. It introduces TinyEye-Data (2M verifiable reasoning trajectories across 68 tasks), a four-stage training pipeline—(1) native-resolution warm-up for robust vision–language alignment, (2) large-scale instruction tuning over TinyEye-Instruct/Reason, (3) annealed rejection sampling with tandem repeat avoidance (TRAS) to mine hard cases and prevent collapse-like degeneracy, and (4) Discriminative Direct Preference Optimization (DDPO), a margin-based, sign-preserving objective tailored to binary rewards that avoids DPO’s likelihood displacement. The resulting 2B model achieves SOTA among small VLMs on multiple benchmarks (e.g., 50.3 MMMU val, 55.2 MathVerse, 63.9 HallBench), with ablations showing consistent gains from each stage and DDPO outperforming DPO, especially on binary-evaluated tasks.

### Strengths
Well-motivated offline pipeline for small models: coherent coupling of distillation, rejection sampling, and discriminative preference optimization that avoids expensive and unstable on-policy RL.

Verifiable, diverse data at scale: TinyEye-Data spans 179+93 datasets with task-specific verification (symbolic/numeric checks for math, VLM-as-judge for open-ended), plus multi-teacher distillation and pass@8 difficulty estimation.
Practical safeguards against collapse: tandem repeat detection and TRAS during sampling; shortest-chain aggregation to favor concise, effective reasoning.

Novel objective with theory: DDPO reframes DPO’s relative likelihood into binary reward classification with a sign-preserving margin; appendix provides derivation linking to GSPO and argues away DPO’s additive-shift degeneracy.

Strong empirical results for 2B scale: competitive or superior to 2B–4B baselines across multimodal reasoning, textual math, and general VQA; clear improvements at each stage; thinking vs. no-thinking ablation demonstrates CoT value.

### Weaknesses
Limited novelty: The proposed four-stage training pipeline offers practical value in engineering integration and feasibility, but its methodological originality appears limited from an academic standpoint. Each stage largely relies on combinations of existing paradigms and hyperparameter tuning, making it difficult to pinpoint substantive breakthroughs in theoretical framing, learning objectives, or training dynamics. Consequently, its potential to inspire and transfer to subsequent research remains to be further demonstrated.

Questionable fairness of distillation: The training data are heavily distilled from stronger teacher models. Although such a setup can be expected to improve performance, it also complicates attribution: the current experiments do not systematically compare different distillation configurations and methods, making it unclear whether gains primarily stem from teacher capability transfer or from the proposed training mechanisms and objectives. This undermines the strength of the methodological claims. It is advisable to include comparable baseline methods at the teacher/distillation level to demonstrate the non-triviality of the approach.

Insufficient baselines: The comparisons are concentrated in Stage 4 and are reasonably thorough against standard DPO, but the first three stages lack systematic, side-by-side evaluations against alternative alignment schemes, SFT/RFT recipes, or interchangeable components. At the algorithmic level, the work also omits equal-data, equal-compute comparisons with a broader set of offline preference/policy optimization methods (e.g., SimPO, ORPO, DPO-Positive, Alpha-DPO). This limits both the external validity of the conclusions and the clarity of attribution. A unified-protocol, multi-method comparison is recommended.

Large-scale inclusion of public benchmarks in training: TinyEye-Data incorporates a substantial number of widely used community benchmarks as training sources. While using benchmark data for proof-of-concept studies can be understandable, directly employing them as large-scale training corpora may erode these benchmarks’ credibility and validity as independent evaluation and comparison tools, thereby impacting the long-term benchmarking ecosystem. This practice is especially contentious in the context of training open-weight models.

Limitations of purely offline strategies: While end-to-end offline policy optimization improves stability and cost controllability, it also introduces typical issues of distributional mismatch and limited reachability [1]. For multimodal tasks, textual evidence in offline trajectories may not be reliably grounded by the current student model to the visual inputs, potentially exacerbating vision–language hallucinations and semantic mismatches [2]. The authors are encouraged to discuss these issues in the paper.

[1] Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models
[2] Semi-off-Policy Reinforcement Learning for Vision-Language Slow-Thinking Reasoning

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the authors focus on the training techniques of existing LVLMs, including the supervised fine-tuning, offline reinforcement learning, and online reinforcement learning, and claim the urgent need for a discriminative, stable, and efficient training method. Then, authors propose a unified offline policy optimization framework with corresponding datasets called TinyEyes. Experiment are conducted on a training-from-craft LVLM with the proposed method and data. The trained TinyEye-2B demonstrates competitive results on math, language, and general scenarios with learning LVLMs.

### Strengths
1.  A well-distilled 2M data including the instruction part and reasoning part are proposed, which may contribute to the community if open-sourced.
2. The overall results are competitive among LVLMs whose parameters are less than 4B.

### Weaknesses
1. The motivation is not clear.  This paper proposes to solve the problem of post-training for small models. While the shortcut problem of naive reinforcement learning exists for small models, the low-margin problem are common for all sizes of models. Meanwhile, the efficiency     claim should be further claimed, as the offline method like DPO requires a large amount of preference annotations.

2. Though four contributions are listed in the introduction, the core contribution is not highlighted. Most of the motivation and background part focus on the post-training techniques. But the first and second contribution mentioned are the overall pipeline to train a LVLM with processed data. 

3. The techniques novelty is limited. The explicit methods used in the proposed framework are widely used and explored in existing LVLMs. Similar procedure to the framework is also commonly observed in leading LVLMs like Keye, Mimo, Sail, etc.. The most important part in the paper is the DDPO. However, the core design of DDPO has limited and incremental novelty and is more like an engineering-level optimization.

4. The experiments can not support the claim. Or say what is the main claim for this paper? Table 4 only compares the DDPO with DPO without other recent studies in DPO.

### Questions
1. This paper may require careful re-organization to highlight the main claim and contribution to show the real problem to be solved. Most designs show limited relevance with small parameters models.

2. Though the trained TinyEye-2B model has competitive results, how to reach such performance is not clear besides the stage gains, more data or higher quality data or training objectives?

### Soundness
1

### Presentation
2

### Contribution
2
