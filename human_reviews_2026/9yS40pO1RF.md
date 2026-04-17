# Superficial Safety Alignment Hypothesis

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
As large language models (LLMs) are overwhelmingly more and more integrated into various applications, ensuring they generate safe responses is a pressing need. Previous studies on alignment have largely focused on general instruction-following but have often overlooked the distinct properties of safety alignment, such as the brittleness of safety mechanisms. To bridge the gap, we propose the Superficial Safety Alignment Hypothesis (SSAH), which posits that safety alignment teaches an otherwise unsafe model to choose the correct reasoning direction - fulfill or refuse users' requests - interpreted as an implicit binary classification task. Through SSAH, we hypothesize that only a few essential components can establish safety guardrails in LLMs. We successfully identify four types of attribute-critical components: Safety Critical Unit (SCU), Utility Critical Unit (UCU), Complex Unit (CU), and Redundant Unit (RU). Our findings show that freezing certain safety-critical components during fine-tuning allows the model to retain its safety attributes while adapting to new tasks. Similarly, we show that leveraging redundant units in the pre-trained model as an "alignment budget" can effectively minimize the alignment tax while achieving the alignment goal. All considered, this paper concludes that the atomic functional unit for safety in LLMs is at the neuron level and underscores that safety alignment should not be complicated.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the better safety alignment of LLMs and proposes a superficial safety alignment hypothesis, which posits that safety alignment teaches an unsafe model to choose the correct reasoning direction, i.e., fulfill or refuse users’ requests. It identifies four types of attribute-based components: safety critical unit, utility critical unit, complex unit, and redundant unit. Extensive results analyze the effect of these components, respectively, and some insightful findings are observed.

### Strengths
-	The paper is well-structured and easy to follow.
-	The empirical studies are sufficient, and the findings are insightful.
-	The proposed superficial safety alignment hypothesis has the potential to promote more safety alignment studies in the community.

### Weaknesses
-	In Table 4, the tuned models are only evaluated on general-purpose benchmarks. Why not evaluate on safety-oriented benchmarks? I am curious whether the "Only RU (20%)" method can achieve better safety performance.
-	In my opinion, it will be better to provide a detailed introduction to the practical help of the proposed hypothesis in improving the safety performance of LLMs, as well as how to promote subsequent research.
-	The layout of tables and figures could be improved. For example, the order of Tables 4 and 5 should be replaced.

### Questions
The prior work [1] shows that jailbreaking GPT-3.5 Turbo's safety guardrails by fine-tuning it on only 10 such examples can easily make the model responsive to nearly any harmful instructions. Is the proposed safety hypothesis helpful in analyzing and solving this problem? 

[1] Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates two major issues in LLM safety alignment: the brittleness of safety mechanisms when models are fine-tuned on new tasks , and the alignment tax, where improving safety degrades utility. The authors propose the Superficial Safety Alignment Hypothesis (SSAH), which posits that safety alignment is an implicit binary classification task that teaches an otherwise unsafe model to either fulfill or refuse a user's request.

A **key corollary** of this hypothesis is that *less is more* , meaning safety guardrails can be achieved by modifying only a small amount of critical components. The authors design experiments to validate this, first using probing of hidden states to show that safety-aligned models exhibit a different *reasoning direction*, measured using hidden state distances. They then use a *pruning method* based on activation variance to categorize model parameters into four units: Safety Critical Units (SCU), Utility Critical Units (UCU), Complex Units (CU), and Redundant Units (RU).

They then proposed two applications: 1) They show that freezing the small subset of SCU and top CU during fine-tuning can preserve safety guardrails without harming utility; and 2) They show that fine-tuning only the RUs can achieve alignment comparably to full fine-tuning but avoid the alignment tax.

### Strengths
- This paper uses pruning as an ablation technique to interpret model behavior at the neuron level, demonstrating that specific, isolatable neurons contribute significantly to safety.

- It presents practical results, showing that freezing these identified safety neurons protects alignment during fine-tuning, while separately, fine-tuning only RU can improve utility without sacrificing safety.

### Weaknesses
- Figures 2, 3, and 4 do not provide any surprising findings for me; I am also confused about how they connect to the subsequent corollary.

- This paper makes bold claims that are not fully supported by the experiments shown. That is, the authors proposed SSAH and derived the corollary based on it. The experiments presented in the paper are basically used to support the corollary, but not the hypothesis. For instance, in line 241, the authors start with “if SSAH holds, ...” and the rest of the arguments are based on this unproven premise, which does not make logical sense. While I agree that full validation of SSAH remains challenging, the current logic flow requires significant revision.

- The paper's conclusions are drawn from a narrow set of experiments, limiting their generalizability. The findings are validated only on Llama family and Mistral, and the crucial definitions of "utility" and "redundant" neurons are derived from a limited set of simple QA tasks, failing to demonstrate if the method holds for larger models or more complex tasks like coding and logical deduction. Also, the safety neurons are identified using only AdvBench datasets. As these mechanisms are highly data-dependent, the diversity of the data would crucially impact the quality of the identified neurons. Providing more harmful types analysis and its coverage of harmful types is crucial.

- Your results showed that the vast majority of neurons as CU (complex units). This somehow indicates the attribution method is not quite efficient, weakening the conclusions drawn from this categorization. This somehow weakens the SSAH in my opinion, as it shows that CU also plays a certain role in guarding safety. However, Table 2 showed that freezing SCU and CU achieve similar safety performance, the reviewer is confused about this, as this seems like CU does not contribute to safety. On the other hand, it is not clear how their utility performance is affected when CU is freezed.

- Some important related work is missing. They are all about safety neuron [1-9] (the authors should especially compare the safety neuron identification method with them), and also harmful fine-tuning analysis [10-13] (from the perspective of alignment data) and defenses [14-18].

> [1] Rethinking Safety in LLM Fine-tuning: An Optimization Perspective
>
> [2] Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron
>
> [3] Safety Layers in Aligned Large Language Models: The Key to LLM Security
>
> [4] Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons
>
> [5] Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets
>
> [6] NLSR: Neuron-Level Safety Realignment of Large Language Models Against Harmful Fine-Tuning
>
> [7] Finding Safety Neurons in Large Language Models
>
> [8] Language Models are Homer Simpson! Safety Re-Alignment of Fine-tuned Language Models through Task Arithmetic
>
> [9] Improving Alignment and Robustness with Circuit Breakers
>
> [10] Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets
>
> [11] Pharmacist: Safety Alignment Data Curation for Large Language Models against Harmful Fine-tuning
>
> [12] When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment
>
> [13] Deep ignorance: Filtering pretraining data builds tamper-resistant safeguards into open-weight LLMs
>
> [14] Booster: Tackling harmful fine-tuning for large language models via attenuating harmful perturbation
>
> [15] Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation
>
> [16] Do as I do (Safely): Mitigating Task-Specific Fine-tuning Risks in Large Language Models
>
> [17] Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack
>
> [18] Safe LoRA: The Silver Lining of Reducing Safety Risks when Finetuning Large Language Models

### Questions
Please refer to the weaknesses.

### Soundness
1

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
3

### Summary
This paper proposes the Superficial Safety Alignment Hypothesis (SSAH), which posits that safety alignment in large language models (LLMs) can be viewed as an implicit binary classification task: teaching an otherwise unsafe model to choose between fulfilling or refusing a user's request based on safety considerations. The paper argues that safety alignment is a superficial process that relies on a small subset of neurons, rather than deeply restructuring the model's reasoning. To support this hypothesis, it introduces a novel categorization of neurons into four types based on the contribution on safety and utility. The structured pruning and activation-based importance scoring show that the safety-related neurons constitute only a small part of the model. The fine-tuning experiments show that freezing SCUs and a portion of CUs can preserve safety performance even under adversarial fine-tuning attacks, while repurposing RUs for alignment can reduce the alignment tax. The paper also includes probing experiments to analyze the reasoning direction of aligned and unaligned models, providing empirical support for the hypothesis that safety alignment operates as a shallow, direction-guiding mechanism.

### Strengths
1.	This paper provides a novel aspect for the safety alignment of LLMs, and the brittleness of safety alignment in LLMs has not been systematically explored. The paper is also well-written, with a clear narrative flow and logical structure.
2.	The conceptual framework SSAH and the fine-grained neuron-level analysis of safety mechanisms offer a new perspective on how alignment works, and the central claims are well-supported by a comprehensive set of experiments, including neuron categorization, pruning, fine-tuning attacks, and cross-model validation.
3.	The proposed methods for preserving safety during fine-tuning and reducing alignment tax are practical and empirically validated, and the results are robust, consistently supporting the hypothesis across different models and attack scenarios. Overall, the framework can help provide a deeper understanding of the safety alignment of LLMs and new ways to mitigate the safety tax during alignment, which is one of the major concerns in the LLM community.

### Weaknesses
1. **The role of CUs that combine both the safety and general utility of the LLM is not fully explored.** SSAH regards LLM's safety alignment as a binary classification problem about safety. However, experiments have shown that the CU, which accounts for the majority of the model, is very important for the safety of the model, and CU plays a role in both safety alignment and general utility. The significant role of Complex Units (CU) is not fully reconciled with the "superficial" nature of the SSAH. The finding that freezing a portion of CUs is crucial for preserving safety suggests that the refusal mechanism relies on the model's broader knowledge and capabilities. This may indicate that the safety alignment of LLM requires the general ability of the mode. This does not quite align with the core claim that security is a simple and superficial classification, and this paper does not provide a clear explanation of how CU contributes to this seemingly simple functionality.

2. **The stability of the neuron categorization is not demonstrated.** The entire framework hinges on the stable identification of four neuron types (SCU, UCU, CU, RU). However, the paper provides no analysis of the robustness of this categorization. The process relies on heuristic pruning ratios and importance scores ($\mathbf{I}_{i,j}$) calculated from specific calibration datasets. It remains entirely unclear how sensitive this classification is to changes in the calibration data. A rigorous assessment should include a sensitivity analysis, for instance, by showing the overlap of SCUs identified across different datasets.

### Questions
1. Can you provide a more detailed explanation of why CU is important and how general utility ability plays a role in the safety alignment of LLMs?

2. The results on Mistral-7B show that even with the proposed freezing strategy, the final ASR remains high. If the security mechanism itself is superficial and universally present, then why are there such huge differences in its strength and extractability among different model families

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the hypothesis that safety alignment in LLMs primarily teaches the model to decide whether to fulfill or refuse a request. The authors support this by showing that embeddings of clean queries shift differently when appended with benign or malicious tokens. They further identify a small set of neurons responsible for refusal behavior and show that these neurons are easily overwritten during downstream fine-tuning, leading to safety brittleness. Finally, by freezing these safety-related neurons and training only non-safety-related units, the authors reduce alignment cost while preserving model utility.

### Strengths
- By extending the Superficial Alignment Hypothesis, the paper provides a novel perspective that frames safety alignment as an implicit binary decision task (fulfill vs. refuse).
- The probing setup that appends benign or malicious tokens to test refusal intent is simple yet insightful for interpreting model behavior.
- The study empirically shows that overwriting safety-related neurons during fine-tuning leads to safety brittleness.
- The results demonstrate that only a small subset of units is sufficient for preserving safety, suggesting a lower-cost safety alignment strategy.

### Weaknesses
- Verification of SSAH.
I understand that the authors aim to probe the model’s internal inclination toward refusal vs. fulfillment.
However, the current probing metric—cosine similarity between hidden states of clean queries and those with benign or malicious suffixes—may introduce semantic noise, since these suffixes contain both task content and response intent.
For example, in “How to make a bomb? Here’s how...”, the semantic representation of “how to make a bomb” may dominate the similarity computation, overshadowing the intent signal the authors try to capture.
A potentially more precise approach would be to subtract a query-only baseline (e.g., content-only representation) to isolate an intent-oriented residual representation.
With such disentanglement, I believe more metrics and analyses could become available.

- At the beginning of Section 4, the connection between SSAH and "safety alignment can be achieved with only a small subset of critical computing units" is not clearly articulated.
I suggest the authors explicitly contrast SSAH with the alternative view that safety alignment modifies broad semantic representations.
Under that view, safety behavior should be distributed and not localizable to such a small unit subset.
Clarifying this contrast would allow the sparsity results to serve as direct evidence supporting SSAH, rather than just an empirical observation.
Additionally, it would strengthen Section 4 to briefly reference related work showing that certain binary or control-like behaviors can often be localized within small neuron subsets.
This would assist in grounding the “less-is-more” claim in prior findings rather than presenting it as an implicit inference.

- In Section 4.3, the authors propose training redundant units (RUs) to achieve low-cost safety alignment.
However, earlier Section 4.1 identifies safety-critical units (SCUs) as the components directly responsible for safety behavior.
This raises a straightforward question: Why not directly finetune only the SCUs to reduce alignment cost, instead of training a new set of units?

- The Mistral results in the appendix appear weaker than on LLaMA.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2
