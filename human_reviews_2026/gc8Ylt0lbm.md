# DynaGuard: A Dynamic Guardian Model With User-Defined Policies

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Guardian models play a crucial role in ensuring the safety and ethical behavior of user-facing AI applications by enforcing guardrails and detecting harmful content. While standard guardian models are limited to predefined, static harm categories, we introduce DynaGuard, a suite of dynamic guardian models offering novel flexibility by evaluating text based on user-defined policies, and DynaBench, a dataset for training and evaluating dynamic guardian models. Our models provide both rapid detection of policy violations and a chain-of-thought reasoning option that articulate and justify model outputs. Critically, DynaGuard not only surpasses static models in detection accuracy on traditional safety categories, but is competitive with frontier reasoning models on free-form policy violations, all in a fraction of the time. This makes DynaGuard an critical tool for language model guardrails.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces DynaGuard, a dynamic guardian LLM that (1) enforces user-defined policies rather than a fixed taxonomy, (2) produces interpretable rationales for its judgments, and (3) supports a token-efficient classification-style inference path for low latency. Training uses a mixed recipe—SFT on policy-conditioned, reasoned exemplars plus GRPO for reinforcement—and a new dataset, DynaBench, comprising 61.5k training samples paired with policies and a 543-example human-constructed test set. The model family (1.7B/4B/8B) reports strong F1 on WildGuardMix, HarmBench, Safe-RLHF, XS-Test, and especially its own DynaBench that particularly enforces and tests custom policies.

### Strengths
1. The paper introduces DynaBench, a policy-conditioned safety dataset that covers real-world adaptive policy settings such as corporate policies. This enables programmable moderation possible with in-context learning for applying enterprise or jurisdiction-specific rulebooks, making the benchmark directly applicable to customized real-world guard-model deployments.
2. The paper shows that pre-existing guards transfer poorly to policy-conditioned evaluation on DynaBench (Llama-Guard F1 13.1% even with zero-shot rule adaptation capabilties), underscoring limited flexibility for guard models to be applied in practical situations. The proposed model DynaGuard, trained to follow policies, closes much of this gap with F1 73.1% on DynaBench, demonstrating substantially better coverage for customized policies.

### Weaknesses
1. Although the paper shares the same core idea with Constitutional Classifiers [1] in defining policies and generating datasets based on the policies for training classifiers, it does not mention the work. Please position this work with regard to Constitutional Classifiers.
2. Most of the other baseline models are non-CoT, leaving Table 3 misleading since it mentions non-CoT only in DynaGuard and GuardReasoner. 
3. The core motivation of guard models is that it has a better compute/defense tradeoff than language models, thus making it reliable for real-world production. DynaGuard tends to use CoT for better scores, but does not take into account the increased computation. For example, HarmAug[2] shows that their model serves as an attractive practical candidate by comparing FLOPs, latency, peak memory, and even monetary cost with pre-existing guard models.
4. The model performance benchmark only reports F1 score, which is sensitive to thresholds, which is not reported in the paper. For more reliable results, reporting AUPRC as well would be a good alternative.

[1] Sharma, M., Tong, M., Mu, J., Wei, J., Kruthoff, J., Goodfriend, S., ... & Perez, E. (2025). Constitutional classifiers: Defending against universal jailbreaks across thousands of hours of red teaming. arXiv preprint arXiv:2501.18837.

[2] Lee, S., Seong, H., Lee, D. B., Kang, M., Chen, X., Wagner, D., ... & Hwang, S. J. (2024). Harmaug: Effective data augmentation for knowledge distillation of safety guard models. ICLR 2025.

### Questions
1. The CoT version and the non-CoT version of DynaGuard seems to be trained as a separate model. Is that correct?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a semi-synthetic data generation pipeline and a resulting dataset (called DynaBench) of guardrail relevant data. The dataset is split into:
- **Generated training set** of 40K guardrail policies, each accompanied by simulated chatbot conversations reflecting adherence and violation. 
- **An evaluation dataset** of 543 additional, hand-written examples.

Based on the introduced dataset, the authors post-train Qwen3 models into guardian models using a mix of SFT and GRPO. Finally, the authors demonstrate that the trained guardrail models outperform existing models on the introduced DynaBench while showing competitive performance on existing datasets.

### Strengths
- **Principled Dataset Construction Pipeline**: To ensure a wide diversity in the dataset, the authors leverage a hand-written set of attribute seeds for user and agent personas, as well as a curated bank of rules. Based on this, they use LLMs to generate persona profiles, paraphrase rules, generate multi-turn dialogues and finally to label the data.
- **Extensive Evaluation across Multiple Benchmarks**: The authors evaluate on a wide range of benchmarks showing improved accuracy scores, and present various training and model ablations, as well as closer examination of failure modes.
- **Effectiveness Ablation Across Model Families:** The author demonstrate the effectiveness of the generated data across multiple models (section 4.1.)

### Weaknesses
- **Limited Technical Novelty:** While it’s definitely a solid dataset contribution in the field of guardrails, the technical novelty of the proposed approach is limited. Similar dataset construction pipelines were already proposed in many different areas. To post-train the models, authors follow a relative straightforward post-training procedure.
- **DynaGuard Prompt is not provided**: Could you please provide the DynaGuard prompt that you also used for the API models? 
- **Prompt Impact:** Did the authors analyze the impact of the prompt (for example comparing a baseline prompt vs. DynaGuard prompt on the API models)? This is especially important as most baseline models have been trained on a fixed set of safety categories, which are also reflected in the prompt.

### Questions
- What's the prompt used for DynaGuard?
- Could you also add other baseline models to the analysis in Figure 3 (left part; accuracy over rules, tokens, turn and hops)?

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
2

### Summary
This paper introduce DynaGuard, a dynamic guardian model designed to perform safety and compliance for LLMs under user-defined policies. Unlike static Guardrail system, DynaGuard conditions on an explicit policy context provided by the user. The authors also release DynaBench, a new dataset containing both standard safety categories and custom policy scenarios. Experiments show that DynaGuard maintains strong performance on standard harms and substantially outperforms static guard models when evaluating against novel or user-specific policies.

### Strengths
1. This is a real deployment gap, this paper adapts safety filter for dynamic policies. 
2.  Provides a valuable benchmark for studying policy-driven moderation; could become a standard resource.1

### Weaknesses
1. Real-world policies are often long, ambiguous, or inconsistent; paper does not evaluate robustness to noisy or underspecified policies.
2. Unclear how well DynaGuard scales to highly domain-specific or legalistic policies beyond the dataset’s scope (e.g., finance regulations, medical privacy).
3. Out-of-domain evaluations can strengthen the paper.

### Questions
Please address weaknesses above.

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
2

### Summary
This paper introduces DynaBench, a comprehensive benchmark designed to evaluate the safety and alignment capabilities of guardian models across various failure modes and real-world scenarios, and proposes DynaGuard, a set of models optimized for safety through fine-grained guardrails. The authors demonstrate that DynaBench effectively captures diverse safety challenges, with high annotation consistency and meaningful differentiation among models. Experimental results show that DynaGuard models outperform existing open-weight safety models, achieving lower error rates and improved robustness across multiple failure modes, while maintaining competitive overall accuracy. These contributions advance the development of safer, more controllable language models by providing a valuable evaluation framework and specialized models tailored for safety-critical applications.

### Strengths
A novel dataset, DynaBench, for training and evaluation of guardian models is proposed. It contains multi-turn conversations and adversarial jailbreaking prompts. It reveals the weakness of existing guardian models.

A novel guardian model, DynaGuard, is proposed, which is trained using DynaBench and exhibits promising performance on several benchmarks. DynaGuard is claimed to be the first model addressing the four desired characteristics of guardian models listed in Table 1.

### Weaknesses
It was not clear to me how the DynaGuard is technically different from the existing guardian models. In Sections 1 and 2, the difference in (resulting) characteristics among guardian models are mentioned. However, the technical differences between guardian models such as the difference in training scheme, system models, etc., are not clearly mentionedin Sections 3.4 and 3.5. Therefore, it is not clear to me which parts contributed to the good performance of DynaGuard.

No statistical evaluation is provided for the results. If I understand correctly, the result is reported for a single training result for each method. Because the standard deviation or related variation metrics is not provided, I can not judge whether the difference is meaningful or not.

### Questions
Please answer to the points mentioned in the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
