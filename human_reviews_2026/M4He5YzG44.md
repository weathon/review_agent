# A Lightweight Explainable Guardrail for Prompt Safety

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2

## Abstract
We propose a lightweight explainable guardrail (LEG) method for the classification of unsafe prompts. LEG uses a multi-task learning architecture to jointly learn a prompt classifier and an explanation classifier, where the latter labels prompt words that explain the safe/unsafe overall decision. LEG is trained using synthetic data for explainability, which is generated using a novel strategy that counteracts the confirmation biases of LLMs. Lastly, LEG's training process uses a novel loss that captures global explanation priors and combines cross-entropy and focal losses with uncertainty-based weighting. LEG obtains equivalent or better performance than the state-of-the-art for both prompt classification and explainability, both in-domain and out-of-domain on three datasets, despite the fact that its model size is considerably smaller than current approaches. If accepted, we will release all models and the annotated dataset publicly.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes LEG, a lightweight, explainable guardrail for unsafe-prompt detection. Unlike retraining via RLHF or DPO, LEG is modular, LLM-agnostic, and low-latency. It uses multi-task learning with a shared encoder: one head classifies prompts as safe or unsafe, another highlights words that justify the decision. Because token-level labels are scarce, the authors synthesize explanations with a strategy designed to counter confirmation bias. Training employs a joint loss encoding global explanation priors, combining cross-entropy and focal losses with uncertainty weighting. Experiments on three datasets report SOTA or near-SOTA classification and faithful explanations, plus efficiency gains over heavier guardrails in practice.

### Strengths
The paper emphasizes the necessity of faithfulness for unsafe-prompt detectors and explicitly evaluates whether token-level rationales truly support the model’s decisions.

A novel joint loss captures global explanation priors and combines cross-entropy and focal loss with uncertainty-based weighting, effectively balancing safety classification and explanation tagging.

The authors introduce a bias-countering synthetic explanation generation strategy to mitigate confirmation bias and supply token-level labels where human data are scarce.

Results cover both in-domain and out-of-domain settings on three datasets, reporting strong performance for prompt safety (safe/unsafe) and word-level explanation accuracy, supported by ablations of the joint loss.

### Weaknesses
The loss formulation in Equation 1 aggregates many terms without explicit balancing weights for most components, which risks scale-dominant terms steering optimization; adding learnable or tuned coefficients and reporting a sensitivity analysis would likely improve stability and performance.

According to Table 1, the Prompt baseline performs comparably to LEG and sometimes surpasses it, which weakens the claim that the carefully designed joint loss is the main source of improvement; statistical significance tests, effect sizes, and experiments isolating the loss’s contribution beyond the encoder are needed.

The ablation study in Table 6 yields dataset-dependent winners for different loss-term combinations, which suggests limited generalization of a single configuration; proposing a simplified default would strengthen confidence.

The latency comparison with Llama Guard 3 may be functionally uneven, because Llama Guard 3 (~57 ms) is already very fast and outputs fine-grained unsafe categories while LEG provides only a binary label; a fair comparison should match task granularity or extend LEG to multi-label predictions and evaluate end-to-end throughput.

### Questions
For the word-level label, is there a formal definition? A token that appears “harmful” in isolation may be non-harmful in other semantic contexts. Do you account for sentence-level semantics in the loss, and if so, how is the whole-sentence meaning incorporated?

Because the word labels are obtained from GPT-4o-mini, the supervision itself may not be fully faithful. After supervised learning, the faithfulness of the proposed method could therefore be upper-bounded by GPT-4o-mini. In addition, there is no baseline evaluation of GPT-4o-mini on the benchmarks, which makes the evaluation less comprehensive.

There exist explainable AI methods that produce reasons for a model decision, such as LRP (https://github.com/rachtibat/LRP-eXplains-Transformers). This approach yields token-level relevance scores with a single backpropagation pass over the guard model, covering all tokens—not only those deemed harmful—and providing both positive and negative contributions regardless of whether the final decision is safe or unsafe. This seems more faithful, whereas your method explains only when the decision is unsafe, outputs only “harmful” words, and does not indicate which tokens contribute more.

There is also a family of related methods like SHAP and its extensions that come with theoretical guarantees. How does your method compare against these baselines?

For the faithfulness evaluation in Section 5.3, is the metric standard? A more common approach is to mask words sequentially according to an importance metric and measure the resulting accuracy curve. In your case, you use classifier confidence as the metric; it would be helpful to compare the area under the accuracy curve (AUC) across methods.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new external, lightweight guardrail for prompt safety. It outperforms existing guardrails on prompt classification and can highlight components of the prompt that are potentially unsafe. Notably, it's much smaller than current LLM-based guardrails such as LlamaGuard and WildGuard.

### Strengths
The model design is original and novel. The quality of evaluation is relatively high, with reasonable baselines and benchmarks. Table 1 highlights strong prompt classification performance compared to existing guardrails, demonstrating the effectiveness of the proposed model. The writing is clear.

### Weaknesses
Overall I like the idea and enjoyed reading the paper. However, I see the following two weaknesses:

1. Since the loss function is impacted by the overall frequencies of tokens in safe vs unsafe contexts in the training set, the trained model might be sensitive to these keywords. For example, a prompt might contain typically unsafe keywords while being safe overall. Evaluation on such cases will provide more information about the trustworthiness of LEG. OR-Bench (https://arxiv.org/pdf/2405.20947?) might be a good resource for this.

2. Even though the explainability classification is human-interpretable, the mechanistic relationship between the identified keywords and the prompt classification result is not directly interpretable, despite their correlations shown in Table 3. Thus, LEG's prompt classification decisions aren't fully explainable. Since it's not feasible to address this within the conference timeline, please acknowledge it as a limitation beyond stating that "the generated explanations are faithful".

### Questions
1. An informative baseline to consider adding is a frontier/SOTA LM for zero-shot prompt classification and explainability classification - how does LEG compare to it?

2. Even though the three prompt datasets are not highly similar lexically, they might share topical similarity. For example, they might use instances that fall under a similar set of risk categories. Does LEG maintain robust generalization performance between risk categories, rather than between datasets?

3. Lines 197-199: how often did the LLM fail the confirmation bias test?

4. Is the base model for LEG pre-trained? Or is the model only trained on prompt safety datasets?

### Soundness
3

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
The paper proposes a lightweight explainable guardrail (LEG) method for the classification of unsafe prompts. LEG uses a multi-task learning architecture to jointly learn a prompt classifier and an explanation classifier, where the latter labels prompt words that explain the safe/unsafe overall decision. It demonstrates competitive performance with a small number of parameters.

### Strengths
- The loss design is reasonable and the ablation study verifies the effectiveness of each component.
- The small training and inference cost, due to the small base model.

### Weaknesses
- Strong baselines are missing. For example, the prompt guard model trained by meta, which has strong detection performance.
- Limited contribution. The prompt guard model trained by meta also has a small number of parameters (86M), so the contribution of this paper is limited.
- The evaluation of using prompted large language models is missing. For example, using GPT-4o may achieve a strong performance.

### Questions
Why do you choose DeBERTa as the base model instead of causal language model?

### Soundness
2

### Presentation
2

### Contribution
2
