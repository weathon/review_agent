# X-REASONER: Towards Generalizable Reasoning Across Modalities and Domains

- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Recent proprietary models (e.g., o3) have begun to demonstrate strong multimodal reasoning capabilities. Yet, most existing open-source research concentrates on training text-only reasoning models, with evaluations limited to mainly mathematical and general-domain tasks. Therefore, it remains unclear how to effectively extend reasoning capabilities beyond text input and general domains. This paper explores a fundamental research question: Is reasoning generalizable across modalities and domains? Our findings support an affirmative answer: General domain text-based post-training can enable such strong generalizable reasoning, which is even more effective than in-domain multimodal training. Leveraging this finding, we introduce X-REASONER, a vision-language model with reasoning post training solely from general-domain text for generalizable reasoning, using a two-stage approach: an initial supervised fine-tuning phase with distilled long chain-of-thoughts, followed by reinforcement learning with verifiable rewards. Experiments show that X-REASONER successfully transfers reasoning capabilities to both multimodal and out-of-domain settings, outperforming existing models trained with in-domain and multimodal data across various general and medical benchmarks (Figure 1). Additionally, we find that X-REASONER’s performance in specialized domains can be further enhanced through continued training on domain-specific text-only data. Building upon this, we introduce X-REASONER-MED, a medical-specialized variant that achieves SOTA (state-of-the-art)-level performance on numerous text-only and multimodal medical benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors build a two-stage text-only post-training pipeline on Qwen-2.5-VL-7B: first SFT using long CoT distillation, followed by reinforcement learning on math questions with verifiable rewards, yielding X-REASONER. Experiments show that this approach improves text tasks and, notably, transfers to multimodal benchmarks, surpassing several similarly sized models trained with multimodal supervision. Continuing text-only domain post-training in the medical vertical produces X-REASONER-MED, which achieves state-of-the-art results on multiple medical text and medical VLM benchmarks.

### Strengths
1. The paper proposes and validates a challenging hypothesis: text-only reasoning post-training in a general domain can induce cross-modality and cross-domain reasoning generalization, and in several cases outperforms direct multimodal/domain-specific supervision.
2. The paper is well structured; figures, tables, and writing are clear and easy to follow.
3. Without using multimodal supervision, the method attains strong performance on multimodal reasoning benchmarks such as MMMU, MMMU-Pro, and MathVista.

### Weaknesses
1. A key issue is the mismatch in data quality between the two compared training modes. For example, the OpenThoughts dataset is distilled from DeepSeek-R1, a very strong reasoning model, whereas some compared multimodal methods (e.g., MAmmoTH-VL2-7B) use training data that differ in text quality, quantity, and distribution. Such differences are likely decisive for performance and make the current comparisons unfair, casting doubt on the validity of the conclusions.
2. The reported results for Qwen-2.5-VL-7B are inconsistent with official numbers (e.g., official MMMU is 58.6, while this paper reports 53). Similar discrepancies appear for other methods and benchmarks. This inconsistency requires explanation, as it critically affects the strength of the experimental conclusions.
3. All results are based on Qwen-2.5-VL-7B. To strengthen the claim that “text-only post-training → cross-modality/cross-domain generalization,” key findings should be replicated on another 7B-class VL base model.
4. The conclusions would be more convincing with additional ablations and head-to-head controls, explicitly isolating the effects of data quality, data source.

### Questions
Same as the Weakness.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes X-Reasoner, a vision-language model trained solely on general-domain, text-only data using a two-stage SFT–RL (GRPO) pipeline. Despite this training setup, the model exhibits strong cross-domain and cross-modal generalization (particularly to the vision modality). Building on this, the authors further train a medical-specific variant, X-REASONER-MED, which achieves state-of-the-art performance on medical benchmarks. The paper conducts extensive experiments to substantiate these claims.

### Strengths
1. The writing is clear and accessible, with a well-structured and logically organized presentation.
2. The experiments are reasonably solid and supported by rich implementation details.

### Weaknesses
1. The paper primarily explores whether a model trained solely on general text can generalize across domains and modalities. While this is an interesting direction, I have some reservations about relying exclusively on text-only training for a VLM. From an architectural perspective, this setup may under-emphasize vision–language alignment and mainly benefit the LLM backbone. In parallel, recent work suggests that directly training on multimodal data is also an effective and perhaps more targeted path—for example, VL-Rethinker [1] and MiMo-VL [2] report strong gains.
2. Regarding scope, the cross-domain generalization claim is mainly evaluated in the medical domain. Expanding to additional domains would help solidify the conclusion.
3. On evaluation coverage, several recent baselines and benchmarks appear to be missing. It would strengthen the paper to include baselines such as VL-Rethinker-7B[1], Mimo-VL[2], ThinkLite-VL-7B[3], LLaVA-Critic-R1[4] and so on,  and to expand benchmarks (e.g., V*[5], “VLMs Are Blinds”[6], BLINK[7], EMMA[8], etc.).
4.The ablation in Takeaway 3.2.2, probing whether the model truly relies on visual information, is interesting but could be further strengthened. The filtering procedure and the filtering model might benefit from refinement [8]. In addition, evaluating on vision-centric reasoning suites (e.g., V* and “VLMs Are Blind”) could further corroborate the findings.

[1] https://arxiv.org/pdf/2504.08837
[2] https://arxiv.org/pdf/2506.03569
[3] https://arxiv.org/abs/2504.07934
[4] https://arxiv.org/pdf/2509.00676
[5] https://arxiv.org/abs/2312.14135
[6] https://arxiv.org/abs/2407.06581
[7] https://arxiv.org/abs/2404.12390
[8] https://arxiv.org/abs/2501.05444

### Questions
Please see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces X-REASONER, a vision-language model designed to enhance generalizable reasoning abilities across both modalities and domains. It explores whether reasoning capabilities can be generalized across these dimensions and demonstrates that training based on general-domain text is effective for reasoning generalization. The model is trained with a novel two-stage strategy and performs well in various tasks on domain-specific multimodal data, including complex medical reasoning benchmarks.

### Strengths
1. The paper presents a novel approach to generalizable reasoning, showing that general-domain text-based training can effectively improve reasoning capabilities across domains and modalities.Being trained only on text-based data, the model surpass other MLLMs with similar model size.
2. The model is trained with a novel two-stage strategy, combined with SFT+CoT and R, and author give detailed experiments results to demonstrate how to come up with such a strategy.

### Weaknesses
1. The models are only trained on 7B. It may be helpful to show robustness of the training strategy with some smller/bigger model size.
2. Some metrics in figures (like pass@5 in Fig 3 and 4) is not well introduced, which makes it complicated to understand the performance of the models in the first glance.

### Questions
1. The X-REASONER is trained on text-only data and get good results. I'd like to know whether this training strategy can be extended to multimodal datasets, which may further improve the model's ability.
2. While larger model size cost too much computation resources, can experiments on small model size (like 3B) be carried out to show the robustness of the method?
3. In Fig.4 the origin X-REASONER outperforms X-REASONER-Med on pass@5 metrics. Can authors explain the reasons in detail?

### Soundness
3

### Presentation
3

### Contribution
3
