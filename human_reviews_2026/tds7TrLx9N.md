# Teachers That Listen: Adaptive Student-Aware Distillation for Reasoning

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Knowledge distillation is a standard approach to compress the capabilities of large language models  into smaller students. However, standard  distillation methods often produce suboptimal results due to a mismatch between teacher-generated rationales and the student's specific learning requirements. In this paper, we introduce the Adaptive student-aware Distillation for Reasoning (AdaptDistill), designed to bridge this gap by iteratively identifying the student's errors and allowing the teacher to refine its explanations according to the student's needs. Each iteration directly targets the student's learning deficiencies, motivating the teacher to provide tailored rationales that specifically address these weaknesses for better learning. Empirical evaluations on various challenging mathematical and commonsense reasoning tasks demonstrate that our adaptive distillation approach, AdaptDistill, significantly outperforms standard distillation methods, achieving significant performance gains. Our work fundamentally reframes knowledge distillation as an iterative teacher–student interaction, effectively leveraging dynamic refinement by the teacher for better knowledge distillation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The core idea of this work is to reframe knowledge distillation as an iterative teacher-student interaction process, aiming to bridge the gap between teacher-generated generic reasoning chains (Rationales) and the specific learning needs of the student model. In each iteration, the teacher first generates initial rationales. The student model learns from them, attempts the task, and reveals its learning difficulties and errors. Subsequently, the teacher refines and regenerates explanations specifically, based on the student model's error feedback and historical performance. The student model is ultimately fine-tuned on a curated dataset mixed with its own correct reasoning traces and the teacher's refined traces.Empirical results show that the AdaptDistill approach achieves significant performance gains compared to standard one-shot distillation methods on mathematical and common-sense reasoning tasks, including GSM8K and MATH.

### Strengths
1) The paper proposes a novel closed-loop adaptive distillation framework. By explicitly reintroducing the student model's error feedback into the teacher's generation step, it achieves customized guidance targeting individual student weaknesses, effectively addressing the mismatch between teacher output and student needs in standard distillation.
2) The method achieves a performance increase of up to 20% in accuracy on several challenging reasoning tasks compared to basic one-shot distillation baselines.
3) Experiments demonstrate that the method not only improves accuracy on tasks within the training distribution but also maintains and improves performance on Out-of-Domain (OOD) tasks (such as StrategyQA and TheoremQA), indicating an enhancement in the transferability of reasoning skills.

### Weaknesses
1) The core technical component of this method relies on the powerful teacher model (Llama-3.2-70B) performing the steps of "identifying learning gaps" and "generating customized, refined explanations" via Prompt Engineering. Although the framework concept is novel, the mechanism for "identifying learning gaps" and "refining explanations" lacks a quantifiable, learnable, modular design. Instead, it relies on the black-box reasoning capability of a Large Language Model (LLM) to act as an "in-context optimizer." This diminishes the purely technical innovation of the method itself and poses difficulties for future research in reproduction and improvement.
2) The paper's primary comparative baseline is "Standard One-Shot Knowledge Distillation" (i.e., performing CoT distillation in a single pass, following Shridhar et al. (2023)). This is overly simplistic given the current advancements in the knowledge distillation field. As you pointed out, the paper fails to compare against several recent and comparable methods. It lacks empirical comparison against recent State-of-the-Art (SOTA) works like DistilLLM and MiniLLM, which makes it difficult to conclusively prove AdaptDistill's leading position at the current technological frontier. While the paper mentions several iterative or adaptive distillation methods in the related work section (e.g., Wang et al. (2023), Adarsh et al. (2025), Agarwal et al. (2024)), it does not conduct direct performance comparisons between AdaptDistill and these iterative methods that are most similar in mechanism, making it difficult to fully demonstrate the superiority of its unique teacher feedback mechanism.
3) The validation set V used in the paper consists of only 20 samples. The authors justify this by the need to ensure the teacher model's context window can accommodate the historical records H for all iterations. However, using the historical performance of a set of only 20 samples to guide the teacher in targeted content generation over the entire training set $\mathcal{D}$ for multiple rounds may lead to biased guidance or cause the teacher's refinement process to overfit to these 20 samples, thereby affecting the final student model's generalization and robustness.

### Questions
1) Given that your method is iterative and adaptive reasoning distillation, please supplement the experimental results with performance comparisons against representative similar works, such as Wang et al. (2023) or Adarsh et al. (2025), which you cited in your related work. Please also explain why comparisons with DistilLLM Ko et al. (2024) and MiniLLM Gu et al. (2023), which are established or recent works in knowledge distillation, were omitted. If there are technical differences that make direct comparison infeasible, please clarify this in the paper.
2) The "identifying learning gaps" and "generating refined explanations" steps are crucial to AdaptDistill, primarily implemented via prompting the teacher model. Please publicly release the complete Prompt templates used to generate the gap information and the refined rationale in the Appendix. This is necessary to ensure the reproducibility of the experiments and allow readers to better understand the teacher model's decision mechanism.
3) Please provide a more in-depth discussion on the choice of using only 20 samples for the validation set. Furthermore, conduct an ablation study to compare the performance when using a larger validation set (e.g., 100 or 200 samples) under the constraint that the historical record H is limited to a small sliding window (e.g., only considering the 20 most recent samples instead of accumulating all history), to verify the robustness and representativeness of the current small validation set.

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
2

### Summary
The Adaptive student-aware Distillation for Reasoning (AdaptDistill) is designed to bridge this gap by iteratively identifying the student’s errors and allowing the teacher to refine its explanations according to the student’s needs.

### Strengths
1. The paper proposes an innovative solution to the distributional mismatch between the teacher’s rationales and the student’s learning bottlenecks, effectively enhancing instructional alignment.

2. The framework is rigorously defined and the experiments are carefully designed, ensuring the reliability and reproducibility of the results.

3. The proposed method shows strong potential to significantly improve the distillation process, especially for tasks involving complex reasoning.

### Weaknesses
- Lack of Ablation Studies: While the method shows strong results, the paper could benefit from more ablation studies comparing AdaptDistill with other state-of-the-art iterative distillation methods. This would provide a clearer picture of how AdaptDistill fares relative to similar approaches.
- Limited Task Variety: The experiments primarily focus on mathematical and commonsense reasoning tasks. While these are important, the paper could be strengthened by demonstrating the method’s effectiveness across a broader range of tasks or domains.
- Scalability Concerns: The iterative nature of the method requires multiple rounds of distillation, which can become computationally expensive. The paper could discuss strategies for making this process more efficient or scalable, particularly when applying it to larger datasets or models.
- Model Transferability: Although the paper tests AdaptDistill on different student models, further exploration into the transferability of the learned knowledge across different architectures would be valuable. It would also be useful to understand how AdaptDistill performs with varying model sizes.

### Questions
1. Could the authors provide more details on the scalability of AdaptDistill? How would it perform with much larger datasets or models?
2. How does AdaptDistill compare to other advanced distillation methods, such as reinforcement learning-based or self-guided distillation approaches?
3. What are the potential limitations of the iterative refinement process in terms of model convergence and overfitting after many iterations?
4. How would AdaptDistill perform on tasks beyond the domains tested, particularly on tasks that require high levels of generalization?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a distillation framework where the student is placed in the loop of data construction instead of passively imitating the teacher's CoT. For a pool of training questions, the student first attempts to solve them. If the student’s answer is judged correct (using an automatically gradable validation/evaluation setup), that example is directly added to the training set as a “student-solvable” instance. If the student’s answer is incorrect, the method prompts the teacher with information about the student’s observed mistakes (from a validation set) and asks the teacher to regenerate a reasoning trajectory that is tailored to the student’s current weaknesses. The student is then trained on this mixed set (student-solvable + teacher-regenerated-for-student), and the process is iterated.

### Strengths
1. Instead of assuming “teacher CoT = optimal supervision,” the paper explicitly conditions data generation on the current student’s performance. This is a reasonable correction to the common mismatch between long, teacher-style CoT and what a smaller student can actually learn.

2. The Iteration ablation shows progressive gains across iterations, which is good evidence that the loop is actually doing work, not just adding noise.

3. The paper is well-written and easy to follow.

### Weaknesses
1. Strong, partly unstated assumption on the teacher. 
The method implicitly assumes the teacher is strong and instruction-following enough to 1) interpret student errors (potentially noisy, coming from a validation-based diagnosis) and 2) rewrite a solution in a more student-friendly style. That’s a stronger assumption than vanilla CoT distillation, where the teacher only needs to solve the task. 

2. The paper mainly compares (base) vs standard distillation vs its own iterations 1, 2 and 3. It does not compare against closely related baselines (e.g., [1], [2], [3]) that also do error-/fault-/student-aware distillation or iterative, student-on-policy data collection. Without these, it’s hard to tell whether the gain comes from the specific “validation-conditioned prompting” the paper proposes, or just from doing iterative, student-aware distillation.

3. All main tasks are auto-gradable math/logic. This is the friendliest setting for the method, because correctness is easy to detect. It is unclear how the same loop would work for non-gradable or open-ended tasks (dialogue, safety, long-form QA) where validation cannot simply say correct/incorrect.

4. Since the student’s own success controls which data gets kept, there is a risk of data distribution collapsing around what the current student already finds learnable, unless the teacher’s regenerated data is sufficiently diverse and genuinely addresses the failure. The paper does not deeply analyze this risk.

5. The loop requires: student forward on (many) training items $\rightarrow$ judging $\rightarrow$ teacher regeneration for the failed ones $\rightarrow$ retraining. For small math benchmarks, this is fine; for larger, multi-domain corpora, the cost of per-failure teacher prompting could become substantial.


6. Because the paper reports iterations vs baseline but not “without error-conditioned prompting” or “with a weaker teacher,” it’s unclear which part of the pipeline (iteration, student-in-the-loop selection, or error-informed teacher prompting) contributes most.

[1] Li Z, Ji Y, Meng R, et al. Learning from committee: Reasoning distillation from a mixture of teachers with peer-review[J]. arXiv preprint arXiv:2410.03663, 2024.

[2] Wu Z, Li X, Liu Z, et al. Enhancing Long-Chain Reasoning Distillation through Error-Aware Self-Reflection[J]. arXiv preprint arXiv:2505.22131, 2025.

[3] Zhao X, Xu T, Wang X, et al. Boosting LLM Reasoning via Spontaneous Self-Correction[J]. arXiv preprint arXiv:2506.06923, 2025.

### Questions
1. When you say the teacher is “prompted by validation-set errors,” is the teacher given (a) each instance’s wrong student attempt and asked to fix it, or (b) an aggregated description of common student mistakes (e.g. “the student often skips steps / omits units / stops early”) that is then applied to new items? 

2. How sensitive is the method to the teacher’s capability? If you replace the teacher with a weaker model of the same family, does the student still benefit from the error-informed regeneration? This matters because your method delegates the hard part (understanding and rewriting mistakes) to the teacher.

3. There are recent works that also “let the teacher see the student’s mistake and generate a better rationale” or that do iterative, student-on-policy distillation. Why are these not included as baselines? Can you add at least one representative error-aware or verifier-/review-based distillation method under the same compute / teacher-token budget?

4. Did you evaluate your loop on tasks where correctness can’t be checked automatically and you have to use an LLM-as-judge? If yes, that seems to introduce a second strong model into the pipeline. Can you clarify whether this changes the method’s assumptions or makes it less practical?

5. How do you prevent the iterative process from overfitting to the current student’s local failure modes (e.g., always generating longer, more verbose CoT for everything)? Do you track diversity/length/structure drift of the regenerated rationales over iterations?

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
3

### Summary
This paper introduces AdaptDistill, an adaptive and iterative distillation framework for reasoning tasks. Unlike standard one-shot distillation, which ignores student-specific mistakes, AdaptDistill continuously refines the teacher’s rationales based on the student’s observed errors. The student is then updated using a curated mix of its own correct traces and the teacher’s improved explanations. Experiments on mathematical and commonsense reasoning benchmarks demonstrate consistent accuracy gains (up to +20%) across multiple student models. The method also improves out-of-domain generalization and outperforms longer standard training.

### Strengths
The paper is clearly written and well-motivated.

Knowledge distillation for reasoning is an important problem, especially for improving inference efficiency.

The proposed adaptive interaction between teacher and student is intuitively appealing and empirically effective.

### Weaknesses
1. The novelty of the approach is somewhat limited. The idea of iterative, student-aware feedback has been explored in several prior distillation frameworks where teachers provide targeted corrections based on student failures.

2. The paper lacks comparison with stronger or more recent state-of-the-art distillation baselines, which makes it difficult to fully assess the relative improvement.

### Questions
See Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
