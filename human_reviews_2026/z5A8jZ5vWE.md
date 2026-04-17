# Measuring Intent Comprehension in LLMs: A Variance Decomposition Framework

- Decision: Reject
- Scores: 4, 6, 2, 6, 2

## Abstract
People judge interactions with large language models (LLMs) as successful when outputs match what they want, not what they type. Yet LLMs are trained to predict the next token solely from text input, not underlying intent. Because written language is an imperfect proxy for intent, and correlations between phrasing and desired outcomes can break down in training data, models that rely too heavily on surface cues may respond inconsistently to semantically equivalent prompts. This makes it essential to evaluate whether LLMs can reliably infer user intent—especially in high-stakes settings where robustness and generalization are critical. We introduce a formal framework for assessing intent comprehension in LLMs: whether a model demonstrates robust understanding of user intent by producing consistent outputs across semantically equivalent prompts while differentiating between prompts with distinct intents. Our evaluation approach is based on a variance decomposition of model responses into three components: variability due to user intent, user articulation, and model uncertainty. Models that understand what users want, and are not overly sensitive to textual cues, should attribute most output variance to intent differences, rather than articulation style. Applying this framework across diverse domains, we find that larger models typically assign a greater share of variance to intent, indicating stronger comprehension of intent, although gains are uneven and often modest with increasing model size. These results motivate moving beyond accuracy-only benchmarks toward semantic diagnostics that directly assess whether models understand what users intend.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposed a method to evaluate if LLMs truly understand users' intents. It models how LLMs understand users' intent by analyzing how their responses vary across prompts with different underlying intents. The experiment results on open-source LLMs show that larger models have stronger intent understanding capability.

### Strengths
I really like the idea of modeling LLMs' intent understanding ability by measuring how robust their responses are across different prompts with varying intents behind them. This perspective is inspiring. And this works decouples model's response changes into three factors:
1. The model’s inherent uncertainty
2. The diversity of prompts
3. The variation in intents

And this work proposes a quantitative method to quantify the impact of each factor.

### Weaknesses
1. **Assumption of intent-response mapping**. This work assumes that prompts with different intents necessarily yield different responses. However, this assumption overlooks cases where distinct intents can share the same surface form or prompt (*false negatives*). Such cases are ignored in the proposed evaluation framework.
2. **Restrictions to Numeric Responses**: The framework appears to support only numerical responses, where the output of the model is a number. This design limits its applicability and generalizability to more diverse response forms, such as long text.
3. **Limited model coverage**: The evaluation only includes LLaMA and Gemma but does not examine proprietary models such as GPT or Gemini, which could provide more valuable and interesting insights given their popularity.
4. **Temperatur Design**: The authors state in the appendix that all models are tested with a temperature of 1. However, for LLaMA's temperature range is [0, 1], meaning that a value of 1 leads to extremely diverse responses (MU is high). In contrast, Gemma’s temperature range is [0, 2], where 1 represents moderate diversity (MU is lower). This discrepancy makes the comparison between the models seem unfair.

### Questions
1. I’d like to know how you would handle cases where different intents share similar or even identical prompts. How might such cases be incorporated into your current assumptions and framework?
2. In lines 320–323, I found it difficult to fully understand the recall formula defined as IS / (IS + AS + MU). While the definition of precision is clear, the recall formula seems to omit false negatives (my weakness #1). Could you please provide a clearer explanation or justification for this formulation?
3. In Figure 4, the meaning of the different colors is unclear. It would be helpful to include a brief description or legend in the caption to clarify their significance.
4. I’d like to understand why all models were set to a temperature of 1. Do you think this ensures a fair comparison?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a formal definition of intent comprehension in LLMs, which is the desired property that LLMs are sensitive to the *intent* of user requests (Intent Sensitivity; IS) without being overly sensitive to superficial differences in prompt phrasing (Articulation Sensitivity; AS). The third source of variation in model responses is termed model uncertainty (MU). The authors quantify these three concepts by using a variance decomposition of model responses and offer two descriptive metrics: (1) Meaningful Variance Share (MVS) that reflects the "precision" or the ratio of explainable variance, and (2) Intent Comprehension Index (ICI) which captures the "F1-score" i.e. the harmonic mean between the MVS and the IS. To evaluate this in practice, a set of 24 prompts with 10 intent-preserving paraphrases each was generated using GPT-4.1. Five open source LLMs were evaluated across Gemma and Llama families with varying sizes, generating 25 responses per paraphrased prompt. Findings indicate generally that IS increases with model size (as expected), but so does AS, which is not desirable.

### Strengths
- Paper is extremely well written, clear, and well-motivated. I especially found the conceptual framework very clear, the relaxation of intent comprehension to be realistic, and the explanation of the different metrics to be exceptional.
    - Small suggestion, the explanation of the MVS and ICI in lines 320--323 cleared a lot up for me, and I think it would make the whole section easier to follow if it was moved up earlier. Generally maybe splitting Section 3 with subsection headings could also make it even better.
- It addresses an important notion--intent comprehension--that is fundamental to understanding how LLMs behave and can provide insight into improving capabilities.
- The variance decomposition approach is well justified with respect to related literature. I didn't check the math, but the explanation seems very intuitive and well-defined.

### Weaknesses
1. The paper lacks details on the prompt generation and the validation of the LLM-generated prompts and paraphrases. In order to make sure that the intent of each prompt is maintained across paraphrases, a human validation would be necessary. 
2. Generally, the scope of the contributions is limited to just measuring the phenomenon and there is no investigation into why this occurs or how to improve performance. If the authors could do one or both of the below, it would greatly improve the scope of the contributions and utility of the paper insights:
    1. The ability of a model to understand user intent from prompts is likely heavily influenced by the post-training, i.e. the instruction finetuning and/or alignment finetuning. It would be really insightful to include experiments comparing base versus instruction tuned versions of each model (most if not all of the models in the paper have their counterparts open source). If full scale experiments are not feasible, at least a discussion on the differences in the post-training of Llama vs Gemma families and how this might affect the results could be helpful. 
    2. Another way to improve the contribution is maybe to finetune any of the models on the prompt variants to see if this improves the IS scores and decreases the AS.
3. There are lots of problems with the presentation of results and figures in this paper.
    1. Figure 1 is very tiny and the text is not readable. 
    2. Figure 2, the radar chart is a poor choice for visualizing the different models since the lines all overlap and the three metrics are related, not distinct. Consider switching to a stacked bar plot, which is clearer and better suited for the three metrics which all sum to 1.
    3. Figure 3, the line chart is also not a great choice here since the x-axis are categorical and not continuous, which makes it confusing to interpret. Please switch to a more appropriate plot, such as a grouped bar plot. Also, it is missing a y-axis label.
    4. Figure 4 is missing a legend, making it impossible to know which colours correspond to which metric. I'd suggest adding the legend onto the chart as well as mentioning in the figure caption.

### Questions
## Questions
1. I'm curious about the translation approach to generating paraphrases. Was this also done with GPT 4.1? Were the paraphrases validated in any way by humans to ensure the intent was preserved?
2. Why in the consistency definition 2.1 do you write $p_1,p_j\in\tau^{-1}(T)$ instead of $\tau(p_i)=\tau(p_j)$? The latter is clearer in my opinion and more consistent with the definition 2.2.
3. In line 145, why is the "response distribution $\pi$ over $\mathcal{A}$"? Isn't a model's output distribution usually over the input? The notation is bit confusing.

## Suggestions and typos
- The discussion mentions that the method can be extended beyond numerical outputs, which seems like it would be really helpful in practice! However, I did not check this since it's fully in the appendix and not in the main paper. Perhaps it could be moved up into the main paper if it does work.
- Throughout the paper, textual citations are used when they should actually be parenthetical citations. Please correct these (i.e. make sure to use `\citep{}`)
- Citations to all models are missing (Section 4)
- Line 075: "the 70B model..." doesn't make sense out of context since no model sizes have been mentioned yet. Perhaps "the largest model we tested"
-  Line 282 and 319 contain extra "."s
- Line 351 "GPT-based" is this GPT 4.1? It's important to be precise.
- Line 356 "section" needs to be "Section"
- Line 365 "figure" needs to be "Figure"

### Soundness
4

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
This paper introduces a framework to measure how well LLMs understand user intent beyond surface phrasing. Using a variance decomposition method to separate the effects of intent, phrasing, and uncertainty, the authors evaluate LLMs with cross-lingual paraphrases. Results show that larger models better capture intent but remain sensitive to linguistic variation, revealing trade-offs between semantic understanding and surface-level robustness.

### Strengths
1. The paper moves beyond correctness-based evaluation toward consistency-based evaluation, providing a new perspective for analyzing model understanding.

### Weaknesses
1. The framework depends on an idealized ground-truth intent mapping (τ(p)), which is rarely available in real-world data. This makes it difficult to apply at scale or to open-ended user interactions.
2. The decomposition metrics (IS, AS, MU, ICI) are mathematically sound but conceptually dense, making them less intuitive for practitioners. The interpretability of results beyond "higher ICI = better intent understanding" is limited.
3. Although the paper aims to separate intent from articulation, in practice, this distinction is often ambiguous. The evaluation might artificially enforce boundaries that do not reflect real human communication variability.
4. The “Sufficient Intent Comprehension” definition relies on a human or heuristic evaluator, introducing subjectivity and limiting full automation. Different evaluators could yield inconsistent measurements.

### Questions
1. The variance decomposition relies on estimating response distributions. How sensitive are IS, AS, and MU to sampling noise, prompt diversity, or domain imbalance?
2. The "sufficient intent comprehension" definition depends on an evaluator $\tilde{\tau}, V$. How robust are the findings to the choice or bias of this evaluator?
3. Since the focus is on consistency rather than correctness, how should one interpret a model that is consistently wrong but semantically coherent?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a formal framework to measure intent comprehension in LLMs. It moves beyond standard accuracy benchmarks by decomposing the variance of model responses into three components: Intent Sensitivity (IS), Articulation Sensitivity (AS), and Model Uncertainty (MU). The goal is to quantify whether models respond to the user's underlying purpose (high IS) rather than superficial changes in wording (low AS).

### Strengths
The paper tackles the critical and timely problem of distinguishing between a model's understanding of user intent and its reliance on surface-level textual cues.

The variance decomposition (IS, AS, MU) is a principled and novel method for quantifying this distinction. The resulting "Intent Comprehension Index" (ICI) is an intuitive summary metric.

The use of cross-lingual translation chains to generate a diverse set of semantically equivalent prompts (paraphrases) is a smart and scalable approach to test articulation sensitivity.

### Weaknesses
The findings are somewhat inconclusive. While larger models show higher IS, they also show higher AS, suggesting they may be overfitting to surface cues as well as intent. The overall improvement in the ICI metric is described as "modest" with increasing model scale.

The evaluation is confined to open-ended, guesstimation-style tasks that require a numerical answer. This is a significant limitation, as it's unclear how these findings or the framework itself would generalize to more common, non-numeric, or creative generation tasks.

The paper acknowledges that AS and IS are not fully separable, and the results bear this out. The framework doesn't seem to fully resolve the trade-off between being sensitive to meaningful intent changes and being sensitive to all changes in text.

### Questions
N.A.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a variance decomposition framework to evaluate whether large language models (LLMs) understand user intent. The authors first construct an experimental dataset comprising questions (tasks), intents, and back-translated paraphrased prompts across five domains to test model responses. Their evaluation approach decomposes the variance in model responses into three interpretable components: intent sensitivity, articulation sensitivity, and model uncertainty. Furthermore, they introduce two summary statistics, Meaningful Variability Share (MVS) and Intent Comprehension Index (ICI), that serve as metrics for assessing intent comprehension.

### Strengths
The paper presents a systematically designed experimental framework that decomposes response variance across models, domains, and tasks into interpretable components (intent sensitivity, articulation sensitivity, and model uncertainty) thereby providing a principled basis for analyzing how LLMs vary or remain consistent in their responses with respect to user intent and prompt articulation.

### Weaknesses
-	While the paper provides a useful measure of overall response consistency, it does not take into account the correctness of the responses. Understanding intent requires not only consistent responses across different articulations but also responses that are semantically relevant to the given questions. From this perspective, the proposed methods alone may not be sufficient to assess the model’s understanding of intent accurately.

-	While the paper presents extensive variance-based analyses across multiple model sizes, the reliability of its comparative conclusions remains questionable. In particular, the comparison between LLaMA 70B and Gemma models (12B, 27B) is not fully controlled, as it is unclear whether the observed differences in intent comprehension reflect parameter scaling effects or model-family differences.

-	The paper does not consider cases in which different intents may yield identical responses. In such cases, even if the model correctly understands the underlying intent, its Intent Sensitivity could be lower. Moreover, since the metric relies on responses that can be represented as numerical values to compute response variance, it may have limitations in evaluating whether the model truly comprehends questions.

### Questions
-	It would be helpful to include several paraphrased prompts (articulated questions) generated through back-translation in the Appendix.

-	The experiments could be more comprehensive by testing a wider range of temperature values, as evaluating only two settings appears insufficient.

-	The reliability of the experimental framework could be improved by verifying the constructed dataset, as it currently relies solely on GPT-4.1 for question and intent generation.

Typos & Minor Comments:

-	Each figure requires more detailed explanation; in particular, Figure 4 lacks a legend or color description.

-	The Appendix section should appear after the References.

-	The citation formats are inconsistent overall. When referring to a study within a sentence, the author’s name and publication year should appear in parentheses, unless it is used as part of the sentence.

-	In Figure 1, the sentence “What’s the price of traveling from Boston to Paris?” appears twice under different colors; the orange one should be taken from the Articulation Sensitivity section, and it should read “What would be the total cost for me to travel from Boston to Paris?”

-	In Figure 1, the term “Purpose Sensitivity” should be replaced with “Intent Sensitivity.”

-	Lines 494 and 620: the abbreviation PS should be unified consistently across the text.

-	Line 637: the origin or meaning of TU and PU should be clearly defined.

-	Line 318: there is an extra period that should be removed.

-	Line 664: “(see Appendix A)” likely refers to Appendix G.

-	Line 668: “Appendix B” should probably be Appendix H.

-	Footnote 5 (below line 755): the appendix link appears to be broken.

### Soundness
2

### Presentation
2

### Contribution
3
