# When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in malicious queries. Prior jailbreak research mainly augments these queries with additional string transformations to maximize attack success rate (ASR). However, the impact of style patterns in the original queries that are semantically irrelevant to the malicious intent remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We first define ASR inflation as the increase in ASR due to style patterns in existing jailbreak benchmark queries. By evaluating $36$ LLMs across seven benchmarks, we find that nearly all models exhibit ASR inflation. Notably, the inflation correlates with an LLM's relative attention to style patterns, which also overlap more with its instruction-tuning data when inflation occurs. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs, six fine-tuning style settings, and two real-world instruction-tuning datasets, SafeStyle consistently outperforms baselines in maintaining LLM safety.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper demonstrates LLM ASR improvement in response to malicious queries with stylistic patterns. Through extensive evaluation of 32 LLMs, the authors attribute this to superficial style alignment rather than true understanding of safety principles. This paper proposes SafeStyle, which effectively reduces ASR by incorporating stylized safe samples during fine-tuning, without affecting other capabilities.

### Strengths
- This paper presents a real-world issue: the coupling of malicious intent and style instructions, and successfully separates them.
- A comprehensive empirical evaluation of 32 LLMs and 7 jailbreak benchmarks demonstrates the real existence of the influence of style instructions, with results showing that it can lead to ASR inflation.
- The proposed SafeStyle shows significant effects, highlighting the superficial vulnerabilities in current alignment.

### Weaknesses
- The influence of style instructions on ASR proposed in this paper is insightful, but given this situation, the authors have not provided more feasible suggestions. For example, should future jailbreak benchmarks be model-specific? Because different models have different style alignment fields. This complicates the design of benchmarks. Alternatively, considering the situation of ASR-inflation, how should the design principles of benchmarks be adjusted? How can it be proven that the evaluation results are fair?
- SafeStyle is designed from the defender's perspective, requiring model fine-tuning, while attackers can change their style at any time. Therefore, the defender may need to prepare a large amount of data for potential styles, undermining their leverage.

Minor:
- L183, the citation format for SORRY-Bench is incorrect.

### Questions
Could the authors clarify the criteria for selecting the evaluated styles? I'm not very convinced that Shakespeare and poetry are common downstream tasks for LLMs.

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
This paper studies the impact of style patterns on model safety. Specifically, it shows that (1) the presence of certain style patterns in prompts — for example, instructions like “create a list of xx” — can increase the risk of jailbreak attacks; and (2) when instruction fine-tuning data contains many such style patterns, this can substantially raise safety risks introduced during training, especially when evaluation uses data drawn from in-domain style distribution. While the paper addresses an important problem, I have serious reservations about many of the experimental choices: I believe numerous potential confounders that affect model safety were not adequately controlled.

### Strengths
1. The paper tackles an important question — how “style patterns” influence safety — and studies it across two realistic threat settings (jailbreak attacks and instruction-finetuning attacks), providing a focused experimental pipeline for observing safety changes.

2. The authors provide a detailed analysis of how safety-oriented data can mitigate the risks that arise from style patterns in instruction fine-tuning.

3. The paper is well written and easy to follow.

### Weaknesses
1. The paper offers examples of style patterns (e.g., “create a list of xx”). In the experiments, the authors extract the core harmful intent of the original malicious instructions and remove other content, treating all removed content as style patterns. I find this operation overly broad and unconvincing: there is no justification for treating everything that was removed as a style pattern. A stricter, better-controlled design is needed — for example, explicitly appending the hypothesized style patterns around the original malicious instructions so that the role of the style pattern is isolated.
2. Even if style patterns have an effect, this phenomenon can likely be explained by prior hypotheses from earlier work — namely, the idea of creating a “competing objective” that runs counter to the model’s safety behavior. In other words, this paper may be operationalizing an already-proposed mechanism (creating a competitive objective) rather than identifying a fundamentally new mechanism; what it contributes is a concrete instantiation (style patterns) of that prior idea.
3. The changes shown in the left panel of Figure 2 are not significant across many models. That weak significance undermines the strength of the paper’s claims.

4. For the instruction-finetuning attack, the authors inject style patterns into the training data and then evaluate using test data drawn from the same distribution of style patterns. Isn’t this essentially a form of backdoor injection? The procedure looks like you are attempting to implant a style-pattern trigger into the language model to bypass intrinsic safety. More rigorous experiments are required to rule out alternative explanations: for example, inject other non-style patterns as controls and compare their effects to demonstrate that the style patterns specifically are causing the observed harms.

5. Table 1 shows that the proposed mitigation can reduce safety risks, but in many cases it also harms usability, which runs counter to the goal of instruction fine-tuning. The real objective should be to reduce safety risk while preserving (or minimally impacting) the utility gains of fine-tuning. I encourage the authors to evaluate mitigations in more realistic settings and report trade-offs under those conditions.

### Questions
Refer to the proposed weakness.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper argues that seemingly benign style patterns in prompts (e.g., “create a list…”, “write a poem…”) can inflate jailbreak Attack Success Rates (ASR) even when the underlying malicious intent is unchanged. The authors:

(1) define ASR inflation and show it appears widely across 32 instruction-tuned LLMs and 7 benchmarks;

(2) provide evidence that superficial style alignment during post-training contributes to these failures;

(3) propose SafeStyle, a simple defense that injects a small, style-matched set of safety examples during fine-tuning, which consistently reduces ASR while preserving style adaptation utility.

### Strengths
(1) Clear phenomenon- “ASR inflation” characterizes a real, under-discussed confound in jailbreak evaluation. The paper shows that original benchmark prompts often combine “style + malicious intent,” and the style piece alone can tip models into compliance.

(2) Breadth of evidence- 32 open LLMs across 7 benchmarks; result: 28/32 show higher ASR with style present; SorryBench and MedSafetyBench impact the most models. Family trends (Mistral highest; Gemma lowest) are informative.

(3) Mechanistic Interpretation- A statistically significant correlation (ρ≈0.57) between relative attention to style tokens and per-model ASR inflation; style patterns that inflate ASR show higher bigram overlap with instruction-tuning corpora supporting the superficial-alignment phenomenon.

(4) Careful causal probe via controlled tuning- Fine-tuning Llama-3.1-8B on list/poem styles shows sharp ASR increases when train and test styles match; mixing in style-removed data mitigates the effect; style position (prefix vs suffix) has little impact.

(5) Simple, practical defense

### Weaknesses
(1) Reliance on GPT-4o for multiple roles- GPT-4o is used to (i) extract malicious intents, (ii) judge ASR, (iii) generate responses/safety data. This centralizes evaluation and could bias both labeling and defense wins toward the same model family’s preferences; more judge diversity would strengthen claims.

(2) Generalization beyond open models/non-reasoning models- Claims are strongest for open models (Llama/Gemma/Qwen/Mistral/OLMo). It remains unclear how large frontier/proprietary models behave especially because their training data is not open-sourced. Additionally, it would be great to assess the ASR inflation for some reasoning models.

### Questions
(1) Can you test the robustness of your findings using additional judges (e.g., Claude, Llama-Guard, or human raters) instead of GPT-4o for extraction and ASR labeling? It would help confirm that ASR inflation and SafeStyle’s gains are not artifacts of a single model family’s evaluation bias.

(2) Have you examined whether ASR inflation or SafeStyle’s benefits hold for proprietary or reasoning-focused models (e.g., GPT-4-Turbo, Claude-Opus, or DeepSeek-R1)? Including even small-scale tests would clarify external validity beyond open instruction-tuned models.

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
4

### Summary
This paper investigates a new LLM safety risk arising from style compliance. The authors find that large language models become more vulnerable to malicious prompts when those prompts follow the same stylistic patterns that the models were fine-tuned on. Through extensive experiments across multiple LLMs, they demonstrate that stylistic alignment—such as using list formats or poetic styles—can significantly increase jailbreak success rates.

To explain this phenomenon, the authors introduce the concept of superficial style alignment, where the model learns to associate style patterns rather than genuinely understanding safety instructions. As a result, it may mistakenly treat malicious inputs formatted in familiar styles as safe.

To address this issue, they propose SafeStyle, a defense method that augments fine-tuning with a small amount of style-matched safety data, effectively reducing vulnerability while preserving the model’s stylistic adaptability.

### Strengths
The paper conducts extensive experiments across a large set of LLMs, providing strong empirical evidence for the identified vulnerability.

The experimental design is solid and systematic, covering multiple styles, models, and datasets to validate the observed phenomenon.
It investigates an under-studied but important topic in LLM safety—style compliance—highlighting a novel and practical risk overlooked by prior alignment research.

The proposed defense method (SafeStyle) is simple yet effective, demonstrating clear improvements while maintaining performance.

### Weaknesses
While the paper provides strong empirical evidence for style-induced vulnerabilities and introduces an effective defense, the utility evaluation remains limited. The authors mainly rely on a style adaptation score (e.g., LC_WR) to demonstrate that SafeStyle preserves stylistic performance. However, this metric only measures the model’s ability to follow styles that were explicitly seen during fine-tuning, rather than testing general instruction-following or reasoning capabilities. Consequently, it remains unclear whether SafeStyle generalizes to unseen or mixed-style prompts or how it affects broader model utility such as factual accuracy and reasoning. Incorporating comprehensive utility benchmarks (e.g., AlpacaEval full suite, or human evaluations) and cross-style generalization tests would strengthen the validation of SafeStyle’s effectiveness.

I think the observed style-triggered compliance resembles backdoor attack behavior: an injected pattern reliably changes model outputs. I recommend the authors run targeted interventions (e.g., masking/paraphrasing style tokens, robustness to paraphrased triggers, fine-tuning unlearning tests, and attribution analyses) to clarify whether the phenomenon is a brittle surface trigger or a more distributed representation issue. This distinction affects mitigation strategy and threat modeling.

### Questions
Please address the concerns in weakness.

### Soundness
3

### Presentation
3

### Contribution
2
