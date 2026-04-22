# Refuse without Refusal: A Structural Analysis of Safety-Tuning Responses for Reducing False Refusals in Language Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 2, 8, 8

## Abstract
Striking a balance between helpfulness and safety remains a fundamental challenge in aligning large language models. To achieve this balance, models should refuse harmful prompts (e.g., "How do I shoot someone?") while remaining responsive to benign inputs—even those superficially resembling harmful prompts (e.g., "Where can I shoot a good photo?"). However, reliably distinguishing genuinely harmful requests from innocuous but superficially risky ones is challenging, often leading to false refusals. In this paper, we address the issue by decomposing a response in the safety-tuning dataset into two distinct components: (i) a boilerplate refusal statement, and (ii) a rationale explaining the refusal. Our experiments and analyses show that refusal statements impede accurate discrimination between harmful and benign prompts by inducing reliance on superficial cues. In contrast, training solely on rationales reduces false refusals without compromising overall task performance and only rarely compromising safety. Furthermore, applicability studies demonstrate that rationale-only benefits are also observed in in-context learning, and rationale-only fine-tuning remains compatible with existing approaches. The results emphasize the necessity of precisely curated, fine-grained safety supervision datasets and outline directions for constructing aligned agents that better reconcile helpfulness with safety.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles false refusals by decomposing safety-tuned responses into two components: (i) a boilerplate refusal statement and (ii) a rationale explaining the refusal. The key result is that the statement, not the rationale, drives over-refusal: removing or delaying the statement and training on rationales only markedly reduces false refusals on pseudo-harmful prompts while largely preserving safety on genuinely harmful inputs and core capabilities across multiple base models.

### Strengths
1. The decomposition into statement vs rationale is simple yet powerful. They show that a simple data-side correction, paired with in-context learning, it substantially reduces over-refusal.
2. The method is simple, easy to implement, and incurs minimal engineering overhead.

### Weaknesses
1. The core idea is intuitive and simple, yet too simple. Decomposing safety responses into a boilerplate refusal statement vs. an explanatory rationale and then removing the statement is essentially a data formatting/ablative supervision change rather than a new modeling principle, learning objective, or algorithm.
2. The contribution largely reads as data formatting for one failure mode rather than a broader safety-alignment advance. Conceptual novelty is modest without a new objective, theory, or model-side mechanism.
3. The study relies on a single LLM judge (Llama-3.3-70B-Instruct) with a custom prompt. Broader evaluation is encouraged (more powerful model like GPT 5). Moreover, experiments are limited to small base models, so scalability to larger instruction/RLHF-tuned systems remains untested.

Minor:

1. Can u analyze first-k tokens instead of first token?
1. Safety and over-refusal should be analyzed jointly: does a trade-off exist, or can the proposed method improve both?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

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
This work provides a valuable decomposition of safe responses by decoupling them into a refusal statementand a refusal rationale. Through a series of analyses, the authors convincingly demonstrate that the refusal statement's reliance on superficial cues is a key factor leading to erroneous over-refusal. This insight offers an important direction for constructing safety-alignment data in future research.

### Strengths
The paper's claim that the refusal statement's reliance on superficial cues leads to erroneous over-refusal is both insightful and intuitive. This point is convincingly substantiated by extensive experiments across four different models.

### Weaknesses
+ While the extensive experiments in this paper establish a correlation between the reliance on superficial cues in refusal statements and excessive refusal, they primarily demonstrate its role as a contributing factor rather than providing a definitive causal explanation. Further analysis would be needed to rule out alternative explanations and solidify this claim.
+ While the paper's observation suggests a viable path to mitigate over-refusal by having LLMs output only a request-specific rationale, this approach poses a pragmatic challenge regarding usability. In practice, for a response to be easily digestible, it is often preferable for an LLM to present a clear conclusion either before or after its detailed reasoning. Omitting a definitive refusal statement might compromise the clarity and directness expected in real-world applications.
+ While the paper's observation—that request-specific rationales are key to reducing erroneous refusals—is valuable, it introduces a significant practical challenge. The collection of high-quality training data containing such detailed, query-specific rationales is likely to be prohibitively expensive, posing a barrier to the widespread implementation of this solution.

### Questions
1. While I agree with the authors' intuition that refusal statements can over-rely on superficial cues, the evidence and analysis presented in the paper do not yet robustly establish this causal link. To strengthen this central claim, the authors should provide more compelling evidence, for instance, through a theoretical justification or more direct, extensive analytical experiments.

2. As noted earlier, while the request-specific rationale approach reduces erroneous over-refusal, it introduces challenges such as decreased readability and higher costs for data collection. The authors' discussion of these trade-offs appropriately addresses pragmatic challenges and is valuable for understanding how these findings might be translated into practice.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on the problem of false refusals in LLMs. The authors investigate this through a data-centric framework, where they decompose a refusal response into a templated statement and a rationale that explains why the request is unsafe. Through controlled finetuning across multiple model families, they find that refusal statements increase false refusals by making models rely on superficial cues, whereas rational-only supervision reduces false refusals while preserving safety and general-task performance. They further demonstrate the applicability of rationale-only supervision to in-context learning and inference-time mitigation settings.

### Strengths
1. **Controlled experimental design.**
- The authors carefully and structurally decompose refusal statements and rationales, and the experiments are systematically replicated across multiple model families and benchmarks.

2. **Thorough analysis.**
- Beyond aggregating metrics, the authors incorporate (in Section 6) entropy measurements, token-level attribution, template substitution tests, and human evaluation. These provide a rigorous understanding of how refusal statements shape false refusal patterns and support their claims robustly.

3. **Conceptual contribution and practical applicability.**
- After establishing the scientific understanding of the utility of rational-only supervision through the data-centric framework, the authors further apply this insight in the applicable setting of in-context learning and combination with inference-time mitigation methods (Section 7). This makes the contribution not only theoretically elegant but also practically impactful for real deployment scenarios.

### Weaknesses
**The refusal statement styles are limited.**
- The study primarily relies on a fairly uniform, boilerplate refusal statement format, but the statement can be more nuanced and varied in tone, structure, and context. I wonder how robust are the findings under synthetic variations that introduce stylistic and structural diversity.

### Questions
**Improving LLM safety is beyond relying on training data alone**. Many works use additional safety mechanisms such as refusal tokens [1] and post-hoc classifiers [2].  It would be interesting to see how rationale-only supervision integrates with other common alignment techniques.

[1] Jain, Neel, et al. "Refusal tokens: A simple way to calibrate refusals in large language models." arXiv preprint arXiv:2412.06748 (2024).

[2] Han, Seungju, et al. "Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms." Advances in Neural Information Processing Systems 37 (2024): 8093-8131.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies where false refusals come from - when the model refuses innocuous requests that superficially resemble harmful ones. Their core finding is that by decomposing the model response to harmful requests into the generic refusal statement and the rationale and feeding just the rationale to the model during SFT, it's possible to reduce the false refusal rate. They also find that the position of the refusal statement matters: namely it's better to put it at the end. They also find that the more specific the rationale the better as far as reducing false refusals. They also analyze the reasons for this, finding that the model maintains higher entropy over token space and is able to distribute attribution across tokens in the prompt when tuned on rationales, whereas when trained on refusals the model tends to attribute to only a single token in the prompt.

### Strengths
The paper contributes to the understanding of where false refusals come from, and has clear implications for the design of safety tuning datasets. Namely model responses to harmful queries should include highly specific rationale for why the query cannot be fulfilled, and can omit a generic refusal statement without compromising safety. In addition, the paper offers a theory of why this is the case in terms of token entropy and attributions.

### Weaknesses
Would have been nice to see results for different model sizes within a family. Also would be have been nice to see how safety / refusal rates for the pretrained models after the experimental interventions in the paper compare to those of the released instruction-tuned variants.

nit line 376 should be Rationale-Only models, not Rationale-Onlymodels

### Questions
When comparing request-specific versus generic rationales, were the results (in table 2) averaged over all refusal statement positions? Or was that comparison only for rationale-only conditions

### Soundness
4

### Presentation
3

### Contribution
4
