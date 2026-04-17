# Sensitivity as a Shield: Inducing Sensitivity to Prevent Unauthorized Model Merging

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Training large language models (LLMs) from scratch is costly, driving interest in leveraging open-source LLMs for domain-specific tasks without additional training. Model merging has emerged as a solution to integrate knowledge from fine-tuned models efficiently, but it raises security concerns on unauthorized model merging. Existing approaches primarily focus on post-hoc mechanisms to detect malicious exploitation of released models. In contrast, we propose a novel paradigm: safeguarding models against unauthorized merging before misuse occurs. Specifically, after training a model with strong capabilities in a specific domain, we propose an unmergeable}method that preserves a model’s domain-specific performance while preventing malicious users from acquiring its capabilities through model merging. We identify the critical role of neuron-sensitive weight regions in enabling unmerging and propose two complementary operations, global and local sensitivity processing, to enforce protection. Both theoretical analysis and empirical evaluations demonstrate the effectiveness of our approach in maintaining task performance while making models resistant to unauthorized merging.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposed a method for making the model unmergeable that preserves a model’s domain-specific performance while preventing malicious users from acquiring its capabilities through model merging.

### Strengths
- The paper is well structured.

### Weaknesses
- What is model merging? There is no formal definition of such a concept. I would suggest that the authors provide a formal definition of model merging and build all their methods based on that insight. Currently, model merging is just a vague concept to me.
- Proposition 1: Does gradient mean sensitivity? What is the gradient with respect to? What are the inputs and outputs of the loss function (E.g., are there any input prompts $x$? Then why are they missing in the notation?)
- The intuitions oversimplified the problem. I can hardly understand why increasing sensitivity would make the model unmergeable. This also leads to the following question.
- Why can the global sensitivity increase the sensitivity but preserve the capability of the specific model? Are there any insights in the model design that contribute to this? A further in-depth analysis is necessary for the audience to fully understand why this works.
- In the experiments, the choice of model type is limited; only the Mistral model is evaluated. Try other model architectures like Llama and QWEN. And would it be possible to protect a model with architecture A by using another model with architecture B?
- Can you provide any theoretical proof that this method is effective in making the model unmergeable? i.e., proving that publishing model weights using (2) will be effective.

### Questions
- Would your method be effective when using quantization?

### Soundness
2

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
This paper proposes a defense mechanism against unauthorized model merging in Large Language Models (LLMs). While model merging efficiently integrates knowledge from fine-tuned models, it introduces significant security risks, including IP infringement, harmful responses, and data leakage.

In contrast to existing post-hoc detection methods, this work introduces a proactive defense paradigm to make models "unmergeable" before they are released. The core idea is to intentionally move the model's weights into a "sensitive region." This ensures that any attempt to merge the protected model with another will result in a significant degradation of its specialized capabilities.

The authors propose two complementary operations: Global Sensitivity Processing and Local Sensitivity Processing. The global operation uses poorly performing open-source models to define a universal perturbation direction. The local operation applies a fine-grained alternating add/subtract modification to neurons identified as highly related to the specific domain.

The authors provide theoretical analysis and empirical results showing that their combined method maintains the model's original task performance while effectively resisting multiple merging techniques (TA, TIES, DT). The paper also demonstrates that this protection is robust against adaptive attacks, such as supervised fine-tuning (SFT), which fails to recover the lost capabilities after a merge.

### Strengths
Originality: The paper addresses the novel and practical problem of proactive defense against unauthorized model merging. This shifts the focus from existing post-hoc detection methods to a pre-deployment safeguard, which is a new and valuable formulation.

Quality: The proposed method is technically sound, combining two complementary operations (global and local sensitivity processing) to achieve its goal. The claims are substantiated with empirical evidence across multiple models and state-of-the-art merging techniques (TA, TIES, DT). The authors also include important robustness checks, such as testing against adaptive fine-tuning attacks.

Clarity: The paper is clearly structured and well-written. It effectively explains the security risks of model merging, the intuition behind using "sensitivity" as a defense, and the steps of the proposed algorithm. The results are presented clearly and support the main claims.

Significance: This work is highly significant as it offers a practical, training-free solution to a pressing IP protection concern for model developers. The experimental results indicate the method is fair: it effectively prevents unauthorized merging while preserving the model's core domain-specific performance, ensuring the model remains useful for its intended purpose.

### Weaknesses
Utility-Protection Trade-off: The primary weakness is the inherent trade-off between protection and performance. The proposed "unmergeable" model (Combined method) has a lower accuracy (65.07 on GSM8K) than the original "specific model" (71.87). This means a model owner must release a slightly degraded model to the public to gain protection, creating a utility cost for all users.

Questionable Scalability: The experiments are limited to 7B and 8B models. It is uncertain how this method scales to significantly larger models (e.g., 32B, 70B, or larger). The complex weight dynamics in larger models might react differently to the sensitivity processing, potentially leading to a much larger performance drop (utility cost) to achieve the same level of protection. Validation on at least one larger-scale model is needed to demonstrate practical relevance.

### Questions
The Introduction effectively uses Meta's Llama license to motivate the problem. To further strengthen this motivation, it would be beneficial to add more examples if they are available. For instance, are there other major model deployers who have expressed similar concerns, or are there existing industry surveys that quantify the perceived risk of unauthorized model merging? Adding these references would help establish the broader significance of the problem.

### Soundness
3

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
5

### Summary
This paper addresses the issue of unauthorized model merging, where fine-tuned LLMs have their capabilities illicitly extracted and integrated into other models. The authors introduce a defense mechanism called Unmergeable Models, which applies a post-processing step to a fine-tuned model before release. This approach makes the model’s weights highly sensitive to merging perturbations, causing merged models to suffer significant performance degradation while preserving the original model’s utility.

### Strengths
1. The paper addresses the important problem of unauthorized model merging.
2. The proposed method is thoroughly evaluated across a variety of datasets and experimental settings.
3. The presentation is clear and easy to follow.

### Weaknesses
The main weakness is the questionable necessity of the proposed method. Model merging techniques are effective largely because the models are fine-tuned from a common pre-trained base. A simpler and more scalable defense would be to apply a fixed, secret permutation to the neurons within each layer as a post-processing step. While the authors mention a similar baseline (PaRaMS) in the appendix, their argument for dismissing it is unconvincing. A keyless, post-training permutation would be highly scalable, require no complex key management, and eliminate the risk of key leakage.

Furthermore, by guiding the model to a sharp minimum, the proposed method may compromise performance under practical acceleration techniques like quantization and low-precision inference.

Overall, the concept of creating "unmergeable" models is not new, and the proposed method seems overly complex and potentially detrimental compared to simpler, more robust alternatives.

### Questions
Could the authors provide a direct performance comparison against a simple, post-training neuron permutation baseline? This data is crucial to justify the added complexity of the proposed method.

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
The paper presents a method for modifying models such that it is hard to conduct wight merging using their open-source weights. The work shows that this is effective against merging weights from specialized models of the llama family and show some results that indicate that there is no performance degradation resulting from their method.

### Strengths
- Relevant problem of securing models against malicious use with little effort
- I.e. cannot protect against distillation from models with this method but that also has much higher compute needs as compared to weight merging

### Weaknesses
- Threat model is somewhat unclear/ not that well motivated: If the model is open-source, why should model merging not be ok? The authors seem to refer to open-weights models with specific licenses but not open-source models
- Writing is often not that clear, e.g. “if the model is sensitive on its domain performance against weight perturbation”
- Method relies on assumption that we are merging a specialized model. What about general-purpose models?

### Questions
- If the method practically makes the model little robust, doesn’t this have negative influence e.g. when hosting the model at a lower precision for inference?
- What about models that are not just finetuned for a specific domain but that are generally capable?
- How exactly are they different from Li et al. (2025) and Wei et al. (2024)?

### Soundness
2

### Presentation
2

### Contribution
2
