# Detecting Motivated Reasoning in Internal Representations of Language Models

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Large language models (LLMs) sometimes produce chains-of-thought (CoT) that do not faithfully explain their internal reasoning. In particular, a biased context can cause a model to change its answer while rationalizing it without acknowledging its reliance on the bias, a form of unfaithful motivated reasoning. We investigate this phenomenon across families of LLMs on reasoning benchmarks and show that motivated reasoning is reflected in their internal representations. Training non-linear probes over the residual stream, we find that the bias is perfectly recoverable from representations at the end of CoT, even when the model neither adopts it nor mentions it. Focusing on such cases where the bias is not mentioned, we further show that probes can reliably (i) predict early in the CoT whether the model will ultimately follow a bias, and (ii) distinguish at the end of the CoT whether a bias-consistent answer is driven by the bias or would have been chosen regardless. These results demonstrate that internal representations reveal motivated reasoning beyond what is visible from CoT explanations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper uses probes to detect unfaithful reasoning in LLMs. They find that probes are able to detect unfaithful reasoning, including before it happens, using model internals.

### Strengths
* The motivation of this paper is great - unfaithful COT reasoning is a major problem with one of the most promising approaches for AI safety, COT monitoring. This approach, if sound, could help to flag when normal COT monitors which look at the model’s text output are potentially unreliable. I also think the motivation of being able to detect unfaithful reasoning without having to run counterfactuals is strong
* This paper uses a strong experimental set up, building off of one of the standard ways of evaluating and demostrating unfaithful COT.

### Weaknesses
* How the probe is trained is pretty unclear to me from the description, and it feels like the results could be widely varying in interesting-ness depending on how it was trained
* Less a weakness and more an area for further impact: I think there’s a huge opportunity to design a probe for unfaithful reasoning in general. If there’s any way to generalize this to that use case, I think this work could be hugely impactful. Then, looking at the results of what gets flagged would be quite interesting
* This isn’t a huge deal, but the paper is only 6 pages, so it could be worth extending the results or doing additional analysis, since there’s certainly a lot more to be done here.
* It’s not clear to me how much of the work is being done by having something model internals based (a probe) vs. just having a good dataset for training a classifier (which could just be a finetune of the whole model or some other model). It’d be helpful to know if the internals specifically are adding value over these other baselines, to know how much value the internals are adding. I’d guess the internals-based approach isn’t necessary (but I think the results are interesting if done by a probe or a full-param finetune)

### Questions
1. Can you expand in more detail on how the data for the probe was produced?
2. Can the method in the paper be extended to detecting unfaithful reasoning in general? If so, how would that be done?

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
4

### Summary
This paper focuses on the problem of faithful reasoning of large language models (LLMs). Specifically, it examines the internal representation of LLMs to analyze the influence of bias on the model's reasoning in its chain-of-thought. The empirical results show that bias remains consistently predictable/recoverable from representation at the end of CoT, and probes can help distinguish reliance on the bias reliably in advance.

### Strengths
1. The focused problem of the unfaithfulness of LLMs' reasoning and bias detection is important and of high significance.
2. The findings empirically show that the probe is a useful tool to detect the bias reliance in model internal representation.

### Weaknesses
1. The presentation is not good and clear enough that allow readers to easily catch the challenge and technical contribution of this work.
2. The bias term is not well-defined and there is limit intuitive example or illustration on what kind of context would be catergorized into bias, and the literature search is not sufficient for surveying previous related work on bias detection in nlp context.
3. The probe technique is not presented clearly to let the reader understand the detailed implementation.
4. The Figures are confusing and hard to read without specific experimental description on specific findings. Please consider to enrich the caption to add more intuitive explanation.
5.  Is there any baseline methods with CoT to show the basic detection performance?

Not about the main content:
1. As the authors claimed, they use LLM to implement some methods that they can verified, I'm not sure whether it should be categorised into LLM-engaged research and allowed by the submission.

### Questions
1. Please consider enhancing the representation to explicitly state the technical challenge of the existing research problem.
2. Please do a more thorough literature survey on the bias context detection problem in both the conventional NLP domain and LLM research area.
3. Please consider involving more models and other baselines to make the experiments convincing.

### Soundness
2

### Presentation
1

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
The paper investigates potential misalignment between Chain-of-Thought (CoT) and "internal computations" of LLMs in biased contexts. Using "RFM (Recursive Feature Machine) probes" on the "residual stream" of Qwen3-8B, Llama-3.1-8B, and Gemma-3-4B, the authors study three tasks: (i) "Bias recovery"—at the end of CoT, the hint is decodable from internal representations even when the output/CoT does not mention or follow it; (ii) "Retrospective detection"—distinguishing "motivated" from "coincident" cases that look identical from CoT text alone; and (iii) "Prospective detection"—predicting whether the model will follow the hint "before" generating CoT. The setting covers three hint types (sycophancy/consistency/metadata) and four datasets (MMLU, AQUA-RAT, ARC-Challenge, CommonsenseQA), with a four-way transition taxonomy (resistant/motivated/coincident/divergent).

### Strengths
1. Clear Motivation: CoT-only monitoring may not reveal **motivated reasoning**; probing internal states is positioned as a complementary signal for safety/evaluation.
2. Broad Experimental Coverage: Multiple models/datasets and three hint types suggest that the studied effects occur across settings; taxonomy distributions are provided.
3. Methodological Simplicity : External probes on the residual stream without LLM retraining; training protocol (per-layer/step probes; 80/20 split) is straightforward.

### Weaknesses
1. What do the probes capture? (interpretability/construct validity) RFM has relatively high capacity. Strong “warm-up” hint prediction near the CoT end could be influenced by **lexical/positional traces** of the hint rather than an internal representation of “intention.” Causal interventions (e.g., activation patching, position shuffling, explicit hint masking) would help test whether suppressing hint traces substantially reduces retrospective/prospective performance.
2. Probes appear trained per layer/step. **Cross-dataset/hint/model** transfer (e.g., train on MMLU and test on ARC-Challenge; train on sycophancy and test on metadata; train on one model and test on another) is not reported. Such results would indicate whether the method captures more general signals rather than distribution-specific patterns.
3. Since CoT-only cannot distinguish **motivated** vs. **coincident** by construction, a strong **CoT-only** classifier baseline (e.g., style/length features) and/or comparisons with recent faithfulness frameworks (e.g., FRODO (Paul et al., 2024); FUR (Tutek et al., 2025)) would contextualize the added value of internal-representation probing.
4. Figures emphasize “best layer” trends, but multi-seed means ± std, confidence intervals, or significance tests are not shown. Adding error bars for the main AUC/accuracy plots (e.g., Fig. 1–2 and the detection figures) would strengthen the statistical basis.
5. Providing RFM hyper-parameters, regularization, optimizer/schedule, epochs/early-stopping, random seeds, and split scripts (including whether splits are stratified) would facilitate independent reproduction.
6. “Motivated reasoning” can be broader than sycophancy/consistency/metadata biases. Clarifying how the four-way taxonomy connects to this term may aid interpretation.

### Questions
1. How do "linear probes" or shallow MLPs perform relative to RFM?
2. Have you tried "input masking/position shuffling" to assess whether probes primarily read "hint-token residue"?
3. Do you have "cross-dataset/hint/model" transfer results (e.g., train on one dataset/hint/model, test on another)?
4. At which exact point is the "prospective" representation extracted (post-question, pre-first CoT token)? A "layer × step" heat map would be informative.
5. How does the method compare to "CoT-only" baselines and to faithfulness frameworks such as "FRODO" (Paul et al., 2024) and "FUR" (Tutek et al., 2025), both cited in the related work?

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
This paper investigates the phenomenon of motivated reasoning in LLMs, focusing on cases where biased contexts influence the model’s reasoning and answers. The authors applied probes on internal activations and conducted study over different benchmarks.

### Strengths
This paper studies the faithfulness of LLMs' CoT reasoning, which is an important topic to discuss. It examines over different models and benchmarks, and provided lucid illustrations.

### Weaknesses
1. It is unclear how this work differs from previous works on LLM faithfulness, and there are already varieties of work demonstrating LLM's unfaithfulness from the perspective of internal activations including but not limited to [1][2][3]. Could you clarify your novelty upon these works?
2. The generality of the probe across different task domains is unclear.
3. The paper did not provide deeper analysis or explanation as why the bias happens. Is it related to the model architecture or training methodology?
4. The paper doesn't explicitly discuss how the detection methods can be applied as mitigation strategies.
5. More datasets and models should be incorporated for evaluation.
6. Interpretability analysis is limited. Empirical experiments are a bit rudimentary. At least some case studies should be provided.

The paper's main text is only of six pages. This itself is not a weakness but it's clear that much more experiments should be included before publication at top-tier conferences like ICLR. The current version should be submitted as a workshop paper or short paper instead.

[1] Discovering Latent Knowledge in Language Models Without Supervision (ICLR 2023)

[2] The Internal State of an LLM Knows When It’s Lying (EMNLP 2023 Findings)

[3] Overthinking the Truth: Understanding how Language Models Process False Demonstrations (ICLR 2024)

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
