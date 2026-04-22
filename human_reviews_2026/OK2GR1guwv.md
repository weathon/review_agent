# Safety-Aligned Weights Are Not Enough: Refusal-Teacher-Guided Finetuning Enhances Safety and Downstream Performance under Harmful Finetuning Attacks

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2, 4

## Abstract
Recently, major AI providers such as Google and OpenAI have introduced Finetuning-as-a-Service (FaaS), which allows users to customize Large Language Models (LLMs) using their own data. However, this service is vulnerable to safety degradation when user data includes harmful prompts, a threat known as harmful finetuning attacks.
Prior works attempt to mitigate this issue by first constructing safety-aligned model and then finetuning the model on user data. However, we observe that the safety-aligned weights provide weak initialization for downstream task learning, leading to suboptimal safety-alignment and downstream task performance.
To address this, we propose a **Refusal-Teacher (Ref-Teacher)-guided finetuning framework**. 
Instead of finetuning a safety-aligned model on user data, our approach directly finetunes the base model under the guidance of a safety-aligned Ref-Teacher, which filters harmful prompts from user data and distills safety-alignment knowledge into the base model.
Extensive experiments demonstrate that our Ref-Teacher-guided finetuning strategy effectively minimizes harmful outputs and enhances finetuning accuracy for user-specific tasks, offering a practical solution for secure and reliable deployment of LLMs in FaaS.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses maintaining safety alignment in LLMs under Finetuning-as-a-Service (FaaS) against malicious finetuning attacks. It finds that the standard “safety-align then finetune” pipeline weakens performance, while joint finetuning causes gradient conflicts between safety and task goals. To fix this, the authors propose Refusal-Teacher (Ref-Teacher) finetuning. Results show improved safety and task accuracy across multiple settings.

### Strengths
1. The paper's motivation is clearly shown through the experiments in Section 4, which effectively frame the problem the proposed method aims to solve.
2. The experimental evaluation is comprehensive, covering a diverse range of datasets and settings.
3. The paper is well-written and easy to follow.

### Weaknesses
1. My main concern is the fairness of the comparison. The proposed method uses a data filtering step that the baselines lack, and this filter appears optimized for the evaluation tasks. Although Appendix C1 includes a related comparison with LLaMAGuard3-8B, a more direct evaluation applying the same trained data filter to the baseline methods is needed.
2. The method adds several components that likely increase computational cost and deployment complexity. The paper would benefit from a clear analysis of the time and memory overhead relative to the baselines.

### Questions
1. Could you provide a detailed analysis of the computational cost (e.g., training time, memory usage) of your method compared to the standard SFT and other finetuning-stage baselines?
2. The "Base -> SA + FT" method mentioned in Section 4 appears to be a strong baseline. What was the reasoning for not including it in the main comparison tables?

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
5

### Summary
This paper addresses the problem of maintaining LLM safety during downstream finetuning. The authors identify a key limitation in the standard two-stage pipeline where a safety-aligned model is first prepared and then finetuned on user data. They then argue that this approach leads to suboptimal downstream task performance due to ‘weak initialization’ and can still compromise safety. As an alternative, the authors propose a Refusal-Teacher (Ref-Teacher)-guided finetuning framework. Instead of finetuning a pre-aligned model, this framework directly finetunes the base LLM. The process is guided by a specially trained, frozen Ref-Teacher model. This teacher serves two functions: 1) it filters harmful prompts from the user's data using a learned *refusal feature*, and 2) it provides *alignment distillation* by generating soft refusal labels for a separate safety dataset, which helps the student model learn safety objectives with reduced gradient conflicts. The authors demonstrate that their framework consistently outperforms existing methods, achieving lower harmfulness scores while simultaneously attaining higher accuracy on downstream tasks.

### Strengths
- The paper is well-written and easy to follow. The motivation is laid out logically, building a clear case for why a new approach is needed.
- The proposed solution, to finetune the base model directly while carefully managing the safety/utility trade-off, is an effective response to this finding. This work provides a practical approach to a problem in Finetuning-as-a-service.

### Weaknesses
- The data filtering strategy is configured to maximize recall on harmful prompts, which may discard some harmless user data (a high false positive rate). It is unclear what the percentage of harmless data filtered out is across different tasks.
- The authors should experiment with other data filtering methods [1][2] for a more comprehensive comparison.
- [3][4] also studies this problem from the similarity perspective, which should also be discussed in the revision.

[1] Deep ignorance: Filtering pretraining data builds tamper-resistant safeguards into open-weight LLMs

[2] Pharmacist: Safety Alignment Data Curation for Large Language Models against Harmful Fine-tuning

[3] Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets

[4] When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment

### Questions
- Could the authors provide a brief analysis of the computational overhead of the ‘Teacher Preparation Stage’ compared to the standard ‘Alignment Stage’ used for the baselines.

- Could an adversary craft finetuning examples that are benign in their feature representation (low cosine similarity to the refusal feature) but still steer the model towards unsafe behavior on related prompts?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on the problem of fine-tuning degrading safety alignment in language models. The setting is fine-tuning-as-a-service, where a user wants to fine-tune a model on some task-specific data. The user is assumed adversarial such that a certain fraction of user data consists of harmful prompts and harmful responses. 

The paper proposes to first train, starting from a base model, a safety-aligned teacher model called Ref-Teacher. Then, in the fine-tuning stage, Ref-Teacher is used in two ways: 'refusal feature' from the teacher is used for classifying whether user prompts are harmful (and filtering out use samples classified as harmful); and using soft 'refusal labels' from the teacher for balancing loss on user data and KL-divergence loss on alignment data (called as alignment distillation). 

Experiments consider fine-tuning 3 base models on 4 tasks, and show that the proposed method outperforms several baselines on the metrics of fine-tune accuracy and harmful score.

### Strengths
* The problem of fine-tuning degrading safety alignment is practically relevant and has recently received a widespread attention.
* The idea of using signals from a safety-aligned teacher model is interesting.

### Weaknesses
* Quite a large number of solutions have been recently proposed for mitigating safety degradation after fine-tuning. Besides alignment stage defenses (baselines in the paper, such as Vaccine and Booster, fall under this), there are fine-tuning-stage defenses (e.g., SafeInstruct [Bianchi et al., 2024], VLGuard [Zong et al., 2024], constrained-SFT [Qi et al., 2024]), and post-fine-tuning defenses (e.g., SafeLoRA [Hsu et al., 2025], RESTA [Bharadwaj et al., 2024], SOMF [Yi et al., 2024],  Antidote [Huang et al., 2024]). The paper does not acknowledge the vast related work on this topic. While it is infeasible to compare against too many baselines, it is important to acknowledge the related work on this topic and provide qualitative comparisons. For fairness of comparison, it will be great if the authors can consider a couple of baselines from other setups (e.g., one from post-fine-tuning-stage and one from fine-tuning-stage) for comparison.

* One of my main concerns is that the paper takes an overly simplistic approach for baselines for preserving alignment after fine-tuning -- first alignment is performed by supervised fine-tuning (SFT) of a base model on alignment data and then task-specific fine-tuning is performed. In practice, users significantly prefer fine-tuning instruct models. Leading instruct models take a number of steps for alignment beyond simple SFT on alignment data including RLHF via preference tuning such as Direct Preference Optimization (DPO) or other online RL algorithms such as Proximal Policy Optimization (PPO). Consequently, when starting from an instruct model, adaptation to the user task is often easier and the safety degradation is typically less significant. Many prior works on the topic of fine-tuning degrading safety alignment consider the case of adapting instruct (or chat) models and the impact of safety alignment (e.g., constrained-SFT [Qi et al., 2024], SafeLoRA [Hsu et al., 2025]). The paper has limited experiments in Appendix C.3 when considering instruct models as Ref-Teacher, but lacks details on using instruct models as starting points for task specific fine-tuning.

* The paper does not conduct any ablation experiments to quantify the contributions of data filtering and alignment distillation. (More details in the Questions.)

References
1. T. Huang, G. Bhattacharya, P. Joshi, J. Kimball, L. Liu, "Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-Tuning", 2024
2. Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu,"Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation", 2024
3. Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Röttger, Dan Jurafsky, Tatsunori Hashimoto, and James Zou, "Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions", 2024
4. Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy Hospedales, "Safety fine-tuning at (almost) no cost: A baseline for vision large language models", 2024
5. Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, and Peter Henderson, "Safety alignment should be made more than just a few tokens deep", 2024
6. Chia-Yi Hsu, Yu-Lin Tsai, Chih-Hsun Lin, Pin-Yu Chen, Chia-Mu Yu, and Chun-Ying Huang, "Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models", 2025
7. Rishabh Bhardwaj, Do Duc Anh, and Soujanya Poria, "Language models are homer simpson! safety re-alignment of fine-tuned language models through task arithmetic", 2024
8. Xin Yi, Shunfan Zheng, Linlin Wang, Xiaoling Wang, and Liang He, "A safety realignment frame- work via subspace-oriented model fusion for large language models", 2024

### Questions
* The proposed method has two stages so-called alignment distillation and data filtering. It is not clear how much each stage contributes and how two stages help each other. Are there any ablation experiments quantifying the contribution of each stage? For instance, data filtering can be used in the conventional setup of filtering on top of an aligned model. How would it compare with the proposed method?
* In the experiment, FA is measured on downstream benchmarks by using specific number of samples (L359 on page 7). Why specific number of samples are chosen from these benchmarks in contrast to using the entire test set?
* In Algorithm 1, L260, the equation number (eq (2)) seems to be a typo.
* In Table 8, including BeaverTails seems a bit unfair since Ref-Teacher is trained on BeaverTails. Can the author give more details on the inclusion of BeaverTails?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Ref-Teacher fine-tuning method, where a base model is fine-tuned on downstream tasks along with safety data to preserve its safety alignment properties. The paper first makes the observation that directly fine-tuning the base model achieves both robust safety-alignment and good downstream task performance, but can also introduce "gradient conflicts" where the safety and task gradient directions may be at odds with each other, resulting in worse task performance compared to training on the task only. They then train a new teacher model called Ref-Teacher by filtering out harmful prompts via refusal alignment and training on a newly proposed regularized loss function. The Ref-Teacher is then used to train a distilled model.

### Strengths
1. The paper tackles a relevant, current and pervasive problem.
2. The experimental section shows clear improvements over state-of-the-art

### Weaknesses
1. The solution seems costly, and it is not clear if it is worth the benefits for LLMs where training is already prohibitively expensive.
2. The solution is non-intuitive and relatively more complex in terms of implementation vs other state-of-the-art ones
3. It requires changing the data itself, raising concerns about distributional shifts and making the generalizability questionable.
4. The depth of the novelty is not clear, and quite a few of the observations such as training on both safety and task datasets, gradient differences between them, etc. are already well known in practice.

### Questions
1. Why did the authors take the route of creating a Teacher model which both creates harmful/harmless prompts and is distilled from? Is it simply because empirically it gives better results or is there an intuition behind this? The state-of-the-art papers (such as RepNoise and Lisa) have very elegant solutions compared to the 3 to 5 step process that the authors have employed here. 
2. While the results from the experimental section show that they are better, the main question that the paper does not properly answer is why. Why must the Ref-Teacher be used for more effectively distinguishing harmful vs harmless prompts? Why not use a better model? 
3. What are the advantages of this compared to using the standard methods of detecting harmful prompts?
4. "we assume a setting where a pre-aligned model is unavailable" - this is a strong assumption. Can you please explain a practical scenario where this might be the case?
5. This solution seems to be a two-step solution - creating a teacher model to distill from, and also creating new filtered safety-targeted dataset to train on. The state-of-the-art frameworks compared against do not do training on a new safety-targeted dataset but use the base datasets. Is that not a concern since the solution seems dependent on the underlying data too to a certain extent? 
6. What are the resource costs of this solution in terms of number of extra FLOPs? Is the cost worth the performance gains and/or implementation complexities?

### Soundness
2

### Presentation
2

### Contribution
2
