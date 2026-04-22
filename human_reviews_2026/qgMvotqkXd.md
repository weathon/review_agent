# RefCritic: Training Long Chain-of-Thought Critic Models with Refinement Feedback

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
With the rapid advancement of Large Language Models (LLMs), developing effective critic modules for precise guidance has become crucial yet challenging. In this paper, we initially demonstrate that supervised fine-tuning for building critic modules (which is widely adopted in current solutions) fails to genuinely enhance models' critique abilities, producing superficial critiques with insufficient reflections and verifications. To unlock the unprecedented critique capabilities, we propose RefCritic, a long-chain-of-thought critic module based on reinforcement learning with dual rule-based rewards: (1) instance-level correctness of solution judgments and (2) refinement accuracies of the policy model based on critiques, aiming to generate high-quality evaluations with actionable feedback that effectively guides model refinement. We evaluate RefCritic on Qwen2.5-14B-Instruct and DeepSeek-R1-Distill-Qwen-14B across five benchmarks. On critique and refinement settings, RefCritic demonstrates consistent advantages across all benchmarks, e.g., 6.8\% and 7.2\% gains on AIME25 for the respective base models. Notably, under majority voting, policy models filtered by RefCritic show superior scaling with increased voting numbers. Moreover, despite training on solution-level supervision, RefCritic outperforms step-level supervised approaches on ProcessBench, a benchmark to identify erroneous steps in mathematical reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes RefCritic, a critic model trained to both (i) judge whether a solution is correct and (ii) provide feedback that effectively helps a policy model improve its answer. The system adopts a two-stage training pipeline: a cold-start SFT on filtered critique data, followed by RL (GRPO) with dual rewards that explicitly align critique generation with downstream solution improvement. Experiments using two 14B critic models trained on math data show consistent improvements in the generator’s self-refined accuracy and majority-vote performance. Further evaluation on ProcessBench demonstrates that RefCritic surpasses several existing baselines. The paper also reports out-of-domain gains on LiveCodeBench and GPQA, as well as the effectiveness of using the RefCritic-14B model to critique stronger models, with varying degrees of improvement.

### Strengths
1. **Goal-driven formulation.** The dual-reward scheme explicitly ties critique usefulness to the generator’s performance improvement, addressing a common gap in prior “LLM-as-critic” methods that focus only on correctness. The λ-scheduled reward balancing within the GRPO framework is conceptually simple, computationally efficient, and compatible with most existing RLVR approaches.
2. **Consistent empirical gains.** RefCritic achieves steady improvements in critique-based self-refinement Pass@1 and majority-vote scores across AIME24, AIME25, and OlympiadBench, with RL-finetuned models outperforming their SFT counterparts in most cases. These results validate the effectiveness of the dual-reward algorithm design.
3. **Generalizable performance.** The trained RefCritic-14B further enhances the performance of stronger external models, showing clear gains in 2 out of 3 tested cases. This suggests that the learned critique behavior is generalizable and can assist other models rather than overfitting to the generator-specific style seen in the training. Additional experiments on LiveCodeBench and GPQA confirm the cross-domain applicability of RefCritic beyond mathematical problems.

### Weaknesses
1. **[Significance and Novelty]** The idea of training a critic model using both correctness judgments and refinement outcomes has been previously explored, notably in [1]. Several findings and design choices in this paper, such as the limited benefit of SFT-only critics and the generalization to other generators and tasks, are also reported there. While RefCritic focuses on mathematical reasoning rather than coding, the overlap in methodology and conclusions reduces the originality and significance of this work. Clarifying key distinctions or introducing new experiment insights would strengthen the impact of this work.
2. **[Soundness]** (1) The paper omits direct comparison against existing critic models (e.g., those in Table 4) in the main results (Table 2), making it difficult to evaluate the advantage of RefCritic relative to prior methods. (2) In Table 4, the authors deviate from the evaluation protocol in [2], which asks the model to directly output the first erroneous step, by using Qwen2.5-14B-Instruct to extract the answer. This additional extraction step may introduce bias or error, and the derived results are not strictly comparable to those in the ProcessBench. (3) In Figure 3(a), the generator used (Qwen2.5-72B) appears to be a base model without instruction-following ability, which could disadvantage methods like self-critique that rely on instruction comprehension. Using an instruct-tuned generator would yield a fairer comparison.
3. **[Presentation]** Table 2 is somewhat confusing and could be reorganized for clarity. The meaning of entries such as “R1-Qwen-14B Maj” under “Pass_r@1” is not immediately clear. The λ = 0 and λ = 1 configurations could be moved to an ablation section, allowing the main table to focus on key results and improve readability.

[1] Teaching Language Models to Critique via Reinforcement Learning, ICML 2025.

[2] ProcessBench: Identifying Process Errors in Mathematical Reasoning, ACL 2025.

### Questions
1. Why are the critic models evaluated on *ProcessBench* (Table 4) not included for comparison in the main experiment (Table 2)? 
2. Have the authors explored multi-turn self-refinement using the trained RefCritic models? It would be interesting to see whether iterative critique–refinement cycles can further improve solution accuracy.

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
The paper proposes RefCritic, a reinforcement learning (RL)-based framework for training long chain-of-thought (CoT) critic models that provide actionable feedback to improve the reasoning of a policy language model. The core idea is to move beyond supervised fine-tuning (SFT) by introducing a dual-reward RL objective: instance-level correctness and refinement accuracy. The authors train RefCritic on mathematical reasoning tasks using models like Qwen2.5-14B and DeepSeek-R1-Distill-Qwen-14B, and evaluate on benchmarks including AIME24/25, OlympiadBench, and ProcessBench. Results show consistent gains in both refinement-after-critique (e.g., +6.8–7.2% Pass@1 on AIME25) and majority-vote filtering settings.

### Strengths
1. The paper is well-written and logically structured. The 
2. The experimental design and scale are huge
3. While the dual-reward concept is not entirely novel, the specific formulationis a meaningful operationalization.

### Weaknesses
1. The central idea, using refinement performance as a reward signal for training critics, has been explored in prior work. For instance, Training Language Models to Critique With Multi-agent Feedback also leverages feedback loops where critique quality is tied to downstream correction success. The paper would benefit from a more nuanced discussion of these related approaches in Section 2.
2. The method critically relies on binary, verifiable ground truth (e.g., mathematical answers, code execution) that help to provide the accuracy of responses and critiques. This limits its applicability to open-ended, subjective, or real-world tasks common scenarios. The out-of-distribution results on GPQA and LiveCodeBench are encouraging but still within “closed-answer” regimes. The paper does not address how RefCritic would function in settings without deterministic evaluation, which weakens its broader relevance.
3. The refinement reward requires sampling multiple refined solutions per critique during training, which is computationally expensive.

### Questions
No question

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
4

### Summary
This paper proposes a LLM critic model based on reinforcement learning (RL) with dual rule-based rewards: (1) instance-level correctness of solution judgments and (2) refinement accuracies of the policy model based on critiques, aiming to generate high-quality evaluations with actionable feedback that effectively guides model refinement. Experimental results on five benchmarks show the effectiveness of the proposed method.

### Strengths
1. This paper provides analyses on SFT-based critic models, which gives some meaningful insights.
2. Empirical results show the superior performance of the proposed method.
3. This paper is overall well-organized.

### Weaknesses
1. This paper misses an important line of work about critique generation for refinement, such as [1]. In my view, the proposed method is similar to [1], especially the design of R_r (Equation 6). The authors should clearly discuss the difference to highlight their core novelty.

2. Although the authors claim that their method is a long-chain-of-thought critic module, I do not find how this method can improve the long-chain-of-thought generation ability. Now, the methodological design is mainly aimed at better refinement.

3. The quality of generated critiques themselves should be assessed via automatic metrics or human evaluation.

4. The content of Line 262 should be after that of Line 249. The explanation of \lambda should be after the apperance.

[1] Training Language Model to Critique for Better Refinement. ACL 2025 Findings.

### Questions
I have included my questions in the weaknesses part.

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
4

### Summary
This paper proposes RefCritic, a training framework for long chain of thought critics that combines supervised fine-tuning with rule-based reinforcement learning. The critic is optimized with two rewards, one for accurate solution-level correctness judgment and the other for the accuracy gain when the policy solves the problem based on the critique. The goal is to make critiques not only label answers as right or wrong but also provide feedback that actually improves the policy. Experiments on Qwen2.5-14B-Instruct and DeepSeek-R1-Distill-Qwen-14B show consistent improvements on AIME24, AIME25 and Olympiad style benchmarks, for example, an absolute improvement of around 7 percentage points on AIME25 in terms of Passr@1. Although training only uses solution-level labels, RefCritic also outperforms several step-supervised baselines on step level error detection in ProcessBench.

### Strengths
1. The paper is logically clear and easy to follow.

2. The use of refinement effectiveness as a direct reward for the critic is conceptually clear and technically reasonable.

3. Experiments are extensive, including test time scaling, comparisons with multiple base models and several out-of-distribution benchmarks.

### Weaknesses
1. The introduction and main experiments lack systematic comparison with recent strong critic baselines such as DeepCritic [1] and RealCritic [2], so the position and advantage of RefCritic in this line of work remain unclear.

2. The paper does not report the computational cost of the dual reward RL training, and hence it is difficult to assess the cost effectiveness and scalability of the proposed approach to larger models or broader deployment.

3. The paper lacks qualitative cases that show critic outputs and policy answers before and after refinement, which makes it harder to see how RefCritic behaves in practice.

4. The ablation study on design choices seems limited. Beyond lambda, other important hyperparameters such as the number of critiques and refinements per example, the two-stage RL schedule, and the decoding and length settings are not analyzed, so it is unclear the sensitivity of the proposed method is to these choices.

[1] Yang W, Chen J, Lin Y, et al. DeepCritic: Deliberate Critique with Large Language Models[EB/OL]. arXiv:2505.00662, 2025.

[2] Tang Z, Li Z, Xiao Z, et al. RealCritic: Towards Effectiveness-Driven Evaluation of Language Model Critiques[EB/OL]. arXiv:2501.14492, 2025.

### Questions
1. The rewards Rc and λRr rely on a rule-based discriminator that checks whether the generated answer matches the golden answer. Could the authors clarify whether this discriminator is robust to mathematically equivalent but differently formatted expressions, and please also provide the sensitivity analysis for the results of this choice?

2 .On page 3, the footnote for Qwen2.5 14B Instruct says “we provide an empty thinking process and only use the content after ”¡/think¿””. This tag and quoting style look unusual compared to the usual </think> format. Could the authors clarify what exact tag and quotation you use here?

3. In Figure 1, the RL training reward includes a factor like (1 + 0.75) / (1 + λ). Could the authors explain the motivation for this normalization and in particular why the reward is divided by 1 + λ?

4. In Appendix B, some ablation results appear different from the corresponding main results. Could the authors clarify the differences in experimental setup and explain this discrepancy in detail?

### Soundness
2

### Presentation
3

### Contribution
2
