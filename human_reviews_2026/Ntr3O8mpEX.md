# In Their Own Words: Reasoning Traces Tailored for Small Models Make Them Better Reasoners

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
Transferring reasoning capabilities from larger language models to smaller ones through supervised fine-tuning often fails counterintuitively, with performance degrading despite access to high-quality teacher demonstrations.
We identify that this failure stems from distributional misalignment: reasoning traces from larger models contain tokens that are low probability under the student's distribution, exceeding the internal representation capacity of smaller architectures and creating learning barriers rather than helpful guidance.
We propose Reverse Speculative Decoding (RSD), a mechanism for generating student-friendly reasoning traces in which the teacher model proposes candidate tokens but the student model determines acceptance based on its own probability distributions, filtering low probability tokens.
When applied to Qwen3-0.6B, direct distillation of s1K-1.1 reasoning trace data degrades average performance across major reasoning benchmarks by 20.5\%, while the same model trained on RSD-generated reasoning traces achieves meaningful improvements of 4.9\%.
Our analysis reveals that low probability tokens constitute the critical bottleneck in reasoning ability transfer.
However, cross-model experiments demonstrate that RSD traces are model-specific rather than universally applicable, indicating that distributional alignment must be tailored for each student architecture's unique internal representation.
Code and datasets are available at https://anonymous.4open.science/r/rsd.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a simple method to prevent performance degradation when distilling reasoning capabilities from larger teacher models to smaller student models. The simple method is to set a threshold to filter out the tokens in the reasoning traces with low probabilities and replace them with the student's generation.

While the problem is well-motivated and the idea is interesting, I have two major concerns regarding the methodological robustness and the novelty of the contribution.

### Strengths
* The problem is well-motivated 
* The paper is well-written and easy to understand

### Weaknesses
* The proposed method lacks sufficient empirical support and appears ad-hoc. This is seen from 2 perspectives: (1) The method's effectiveness is demonstrated on only a single dataset (s1K). This narrow evaluation makes it difficult to assess the generalizability of the findings. The threshold (0.01) seems to be a hyper-parameter that could be overfitted to this specific dataset's characteristics. (2) As the authors report in Sec 5.4, iteratively applying RSD leads to perf degradation, this suggests that RSD is not fundamentally improving the student's reasoning capability (i.e., I would treat the student at iter K as a new student I have never met and expect it it be improved).
* The novelty is unclear in the context of prior work. The core idea of aligning a teacher's output with a student's distribution to create more effective training data has been explored in "Reinforcement Learning Teachers of Test Time Scaling" (Cetin et al., 2025). In that work, the teacher is trained with a reward function that includes a KL term to explicitly minimize the difference between the teacher's and the student's distributions over the generated explanation tokens. This is fundamentally the same goal as RSD.

### Questions
1. Can the authors test their method on more datasets?
2. How is your method related to RLT (see weakness \#2)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper argues that strong reasoning traces from large "teacher" models can be suboptimal for LLM reasoning distillation in case the distributional misalignment between teacher and student is too large. Thus, the authors propose "RSD," a method to collect reasoning traces from large models by generating tokens and either accepting or rejecting them based on the student's probability, essentially guiding the teacher closer to the student distribution. The authors evaluate on AIME, GPQA, and MATH, showing that while distilling with the s1 dataset from [1] can hurt the average performance of reasoning/post-trained models, RSD can provide some improvements.

[1] Muennighoff, Niklas, et al. "s1: Simple test-time scaling." arXiv preprint arXiv:2501.19393 (2025).

### Strengths
- The concept that using reasoning traces that the student LLM struggles to understand is suboptimal is something that is intuitively logical and is consistent with prior results.
- I found the presentation level of the paper to be high, with the methodology and results being clearly presented.
- The authors share their anonymized code with the submission, which is great for research transparency and reproducibility.

### Weaknesses
**Major**
My main concerns are centered around the evaluation performed and the results:
1) Looking at the shared code, for all main experiments, the authors consider already post-trained versions of new models, with a focus on the Qwen3 family. This version of Qwen3 was already distilled with a much broader and higher quality closed-source of diverse reasoning data from the recent Qwen3 paper. In contrast, the open source S1 dataset is older and not really considered competitive even when compared to other open-source efforts such as [1]. To this end, I do not think it is surprising that the Qwen3-06B model's performance deteriorates. Thus, I am unsure how this provides any validation for the proposed hypothesis that low-probability tokens and not just "lower-quality" tokens are the cause of the observed phenomena. To really test the introduced hypothesis, I think the focus of the experiments should be models that have not already been subjected to large-scale reasoning post-training (e.g., Qwen2.5 instruct and/or Qwen3 base) together with broader and more competitive reasoning datasets (such as OpenThoughts [1]). While currently, the authors have done some preliminary investigations in Figure 4, I do not think that just varying the model axis with other competitive and/or reasoning models is enough. 
2) While the authors claim the hyperparameters are "following the s1 training recipe" [ln 243], their hyperparameters actually differ. For instance, the models from the original s1 paper seem to train for 5 epochs, while the authors of this paper report training for 15, potentially artificially increasing the gap caused by training on this relatively suboptimal, older dataset. I would appreciate it if the authors could explain this divergent choice, or whether I am perhaps misreading the s1 hyperparameters.
3) From Table 1, the best reported performance of the method after sweeping the percentile hyperparameter is only increased by a very small percentage on evaluations with very few problems (e.g., less than a single average question of improvement on AIME24 and AIME25). Other variations in hyperparameters seem to even hurt performance. These results do not seem statistically significant and suggest a considerable brittleness to hyperparameters. Moreover, the performances reported for Qwen3B-0.6 and all its post-training variations are visibly lower than what was attained in the Qwen3 paper (2.7 vs 10.7 for AIME24, 10.9 vs 15.7 for AIME25, 24.8 vs 27.9 for GPQA, 65.4 vs 77.6 for MATH500). I would encourage the authors to provide an explanation for this gap.


**Other**
1) The idea that models can benefit from their own current understanding, guided by teachers, and that teachers themselves can benefit from going towards the student's distribution is something very related to the StaR family of methods and other recent work on reasoning teachers [2, 3, 4, 5]. I think these are all quite relevant related works whose connection to RSD should be explained in the paper.

[1] https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M

[2] Zelikman, Eric, et al. "Star: Bootstrapping reasoning with reasoning." Advances in Neural Information Processing Systems 35 (2022): 15476-15488.

[3] Li, Xiaochuan, Zichun Yu, and Chenyan Xiong. "Montessori-instruct: Generate influential training data tailored for student learning." arXiv preprint arXiv:2410.14208 (2024).

[4] Guan, Xinyu, et al. "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking." arXiv preprint arXiv:2501.04519 (2025).

[5] Cetin, Edoardo, Tianyu Zhao, and Yujin Tang. "Reinforcement Learning Teachers of Test Time Scaling." arXiv preprint arXiv:2506.08388 (2025).

### Questions
Overall, I think the direction of providing better reasoning traces with distillation through a new decoding technique can be very valuable, and I appreciate the clear exposition the paper provided. However, I found the evaluation to be quite rushed, with empirical flaws and inconsistencies that I do not think provide concrete evidence of the proposed method's effectiveness. For these reasons, I do not think the paper is ready for acceptance at this stage.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the counterintuitive failure of transferring reasoning capabilities from large to small language models (LLMs/SLMs) via supervised fine-tuning (SFT). The authors attribute this degradation to "distributional misalignment," where teacher-generated reasoning traces contain tokens with extremely low probability under the student model's distribution, creating learning barriers. To resolve this, they propose Reverse Speculative Decoding (RSD), a novel mechanism where the student model dictates token acceptance based on its own probability threshold, generating model-specific, distributionally-aligned reasoning traces. Experiments with Qwen3-0.6B demonstrate that while direct distillation significantly degrades performance, training on RSD-generated traces yields a measurable performance gain.

### Strengths
（1）The paper provides a deep analysis of the counterintuitive failure in distilling reasoning capabilities to SLMs, identifying and verifying "distributional misalignment" as the core reason for performance degradation in models like Qwen3-0.6B.
（2）The proposed Reverse Speculative Decoding (RSD) method is conceptually sound and highly intuitive. By filtering out low-probability tokens from the teacher's reasoning trace before fine-tuning, the method successfully delivers a measurable performance boost.

### Weaknesses
（1）While $p_{th}=1\%$ is determined as the optimal value, the paper would benefit from a deeper analysis of how different $p_{th}$ values impact the semantic quality and reasoning logic correctness of the traces.
（2）Appendix Table 4 suggests that effective distillation requires tailored RSD traces for different student models. However, some models, like Qwen3-4B, lack customized experiments, leaving ambiguity about whether RSD is universally effective across all model architectures.

### Questions
（1）The paper indicates a need to customize RSD traces for different models. Could the authors confirm if performance gain is guaranteed for all models with tailored customization, or are certain architectures fundamentally unsuitable? Analysis of any failure cases would lead to deeper experimental conclusions.
（2）In Appendix Table 4, why does direct distillation of the teacher's reasoning trace yield a performance gain for the Qwen3-4B model without failure, unlike the 0.6B model? Is this related to the student model's size, and how does this observation align with the core "distributional misalignment" hypothesis?
（3）When the student model rejects the teacher's low-probability token, will sampling directly from the student's distribution potentially compromise the logical strictness or soundness of the teacher's intended reasoning path?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the counterintuitive phenomenon where small language models (SLMs) experience performance degradation when fine-tuned on high-quality reasoning traces from larger teacher models. The authors identify that this failure stems from distributional misalignment: teacher traces contain low-probability tokens that exceed the student's internal representation capacity. They propose Reverse Speculative Decoding (RSD), a mechanism that inverts traditional speculative decoding by having the teacher propose tokens while the student determines acceptance based on its own probability distribution (rejecting tokens below a threshold p_th).

### Strengths
- Novel inversion of speculative decoding paradigm (teacher-proposed, student-approved vs. standard student-proposed, teacher-approved)
- Comprehensive cross-model evaluation testing transferability both within-family (Qwen3 0.6B→1.7B, 4B) and inter-family (to LLaMA, Gemma, Phi)
- Well-motivated problem with clear empirical evidence (20.5% degradation from s1K-1.1)

### Weaknesses
- All experiments limited to mathematical reasoning only
- No systematic exploration of teacher model selection (different architectures, training methods, capabilities)
- Missing systematic analysis of which model properties (size, training data, architecture) predict RSD effectiveness

### Questions
- Is there a theoretical framework predicting which student-teacher pairs will benefit from RSD?
- What if you use the student's own successful rollouts as "teacher" traces?

### Soundness
3

### Presentation
3

### Contribution
3
