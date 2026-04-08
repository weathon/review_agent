## Human Reviewer 1

### Summary
Authors propose language confusion gate (LCG), which uses a 2-layer MLP to predict the language class of next generated tokens and mask the unwanted language tokens. Results show that language confusion behaviors are reduced by 10x by applying the proposed method.

### Strengths
1. Authors propose an effective fix on the problem of language confusion based on reliable observations.
2. Method is simple and intuitive.
3. That the method does not require modifying the base model alleviates the concern of the potential forgetting happening.

### Weaknesses
1. In section 5.4, there is slight performance degradation on Qwen series after adjusting for LCG (no degradation on GPT-OSS though). Further error analysis on this matter is needed. For example, some qualitative analysis on some reasoning traces: does LCG suppress the diversity/exploration in language models?
2. In light of the above, more experiments could be done evaluating general capabilities after applying LCG.

### Questions
1. Could authors measure if there is any effect on inference speed?
2. Is there potentially a cleaner way to obtain the expected class of the next token? For example rule-based/heuristics-based on logits/last hidden states. Would like to see such a simple baseline to be experimented with to justify a two-layer MLP is needed.
3. Does the incorporation of LCG affect the length of reasoning traces?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper addresses the language confusion issue in LLMs, where models inappropriately mix languages during generation. It proposes the LCG (Language Confusion Gate), a lightweight plug-in module that filters tokens during decoding without modifying the base LLM. Trained via norm-adjusted self-distillation, LCG predicts possible language families and applies masking when necessary. Experiments across various models show that LCG reduces language confusion while maintaining task performance.

### Strengths
1. The proposed LCG is a practical and lightweight solution, avoiding the need for model retraining and adding minimal computational overhead.

2. The norm-adjusted self-distillation method effectively mitigates the bias towards high-resource languages caused by token embedding norm imbalance.

3. The work distinguishes between harmful language confusion and legitimate code-switching, ensuring the model retains necessary multilingual expression capabilities.

### Weaknesses
1. LCG only classifies tokens into broad language families (CJ, Latin, Symbols, Low-Res), failing to resolve confusion between languages sharing the same script (e.g., Chinese vs. Japanese).

2. The evaluation’s reliance on rule-based detectors for language confusion may have limitations, especially in complex multilingual contexts.

3. The paper does not discuss the generalization of LCG to more low-resource languages beyond the tested ones.

### Questions
1. Could the token classification method be optimized to handle more fine-grained language distinctions rather than just broad families?

2. What is the specific computational overhead of LCG in large-scale inference?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper proposes the Language Confusion Gate (LCG), injecting a lightweight MLP layer into LLM to reduce unintended mixing of scripts/languages during LLM decoding. This is based on the observations (i) language confusion is rare and the correct-script token is usually ranked in the top-k, (ii) embedding-norm imbalance favors high-resource scripts. The MLP layer is trained to predict which language families should be permitted at each step and mask others in the logits LCG reduces both Chinese/Japanese and Latin confusion by up to an order of magnitude on translation and QA settings (FLORES+, INCLUDE) and on “thinking” models for code generation (HumanEval-XL), with negligible or no degradation in BLEU/accuracy/Pass@k. This method works for both thinking and non-thinking models.

### Strengths
1. The paper introduces a norm-adjusted self-distillation training signal for a language-family gate. I consider it as a very simple idea distinct from weight edits or reward finetuning.

2. The paper demonstrates empirical gains of the proposed method across open and closed models.

### Weaknesses
1. The paper calculates “code-switch rate” on FLORES-WITH-LATIN but does not include human judgments or llm-as-a-judge on the responses. It could offer more evaluation on whether the improved responses are also preferred by humans. 

2. Are there cases that the proposed method do not help to correct the language confusion and could even increase the problem? It would be helpful to showcase some negative cases as well.

### Questions
1. How do you perform the threshold calibration for the sigmoid function in the MLP? What is the criteria to determine that?
2. Do you try any multilingual tasks, where the model need to respond in at least two different languages? Does the proposed method help in that scenarios?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper introduces the Language Confusion Gate, which is a mechanism to prevent language confusion. Language confusion is the scenario in which a model mistakenly uses tokens from an inappropriate language - such as including Latin text in a language that does not typically have code switching (e.g. Chinese or Japanese). This paper distinguishes between "harmful confusion and acceptable code-switching." The authors carefully construct test scenarios by analyzing token distributions, properties of unicode, and task relevant exceptions such as english technical terms in otherwise non-english text. The authors construct a classifier trained on final hidden states, accounting for interesting observations about HRL token norms, and use this classifier to gate allowed tokens at model decoding. This method is effective at reducing harmful confusion, permitting intentional language mixing, and maintains task performance. I found the analysis of why language confusion happens (top logit distributions, insufficiency of greedy decoding, high norm bias) particularly interesting and insightful.

### Strengths
This paper identifies a rare, but important, failure mode for LLMs. They provide interesting analysis to motivate studying and solving this problem, then introduce several specific technical innovations to address it. 

S1. This paper is extremely well motivated. I was initial skeptical of the prevalence of the problem, but the paper expands on prior work by Marchisio et al. showing that the problem setting is realistic. Sections 2 and 3 are particularly effective in exploring why harmful language confusion happens (e.g. showing that the language confusion token is the top-1 choice 56% time). 

S2. Paper introduces high resource language (HRL) norm bias. The authors show that output tokens with very high norm are naturally higher in the distribution (230-241), and that HRL tokens are over-represented in the top 5% of token output norms. In 264-265 the authors are very precise about the claims, stating they do not claim that this phenomena accounts for all confusion and thus cannot be used as a direct signal. Rather they use this to normalize measurements when training the classifiers. 

S3. The classifier and gating mechanism are very carefully designed - section 4.3 lists deterministic rules of when the classifier can override the expected token prediction. These ensure that the masking does not prevent fluent outputs. 

S4. The evaluations are broad and relevant, including both translation focused tasks derived from FLORES and tasks suitable for reasoning models. The evaluations also carefully distinguish between harmful and intentional code mixing (e.g. WITH-LATIN scenario).

S5. Appendix A details the analysis of partial unicode characters present in model vocabulary - these are details often missed in other papers that study tokenization. The examples in the appendix and discussion of connections between token gating and speculative decoding are also interesting. 

Overall this is a very strong and well-executed paper.

### Weaknesses
None of these weaknesses should be taken as a reason to avoid publication.

- It would be interesting to see how experimental results change if the additional deterministic intervention rules were not used to further control the gating mechanism 

- If there exist a lightweight way to do experiments going beyond the CJ/Latin split, it would be quite interesting to see those results. Perhaps leveraging further unicode metadata? However this is not necessary, it would simply further strengthen the paper. 

- Further investigation of the high-norm tokens would also be interesting. Could these be tied to token frequencies in pretraining data, for models with known data (e.g. olmo models?). Also, the causality direction between high norms <> HRL tokens could be further investigated.

### Questions
- Could you provide statistics indicating how many tokens are disallowed at certain timestamps? 

Line 206 contains a typo

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4