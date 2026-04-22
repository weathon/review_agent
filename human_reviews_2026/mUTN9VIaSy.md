# Bi-directional Bias Attribution: Debiasing Large Language Models without Modifying Prompts

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
Large language models (LLMs) have demonstrated impressive capabilities across a wide range of natural language processing tasks. However, their outputs often exhibit social biases, raising fairness concerns. Existing debiasing methods, such as fine-tuning on additional datasets or prompt engineering, face scalability issues or compromise user experience in multi-turn interactions. To address these challenges, we propose a framework for detecting stereotype-inducing words and attributing neuron-level bias in LLMs, without the need for fine-tuning or prompt modification. Our framework first identifies stereotype-inducing adjectives and nouns via comparative analysis across demographic groups. We then attribute biased behavior to specific neurons using two attribution strategies based on integrated gradients. Finally, we mitigate bias by directly intervening on their activations at the projection layer. Experiments on three widely used LLMs demonstrate that our method effectively reduces bias while preserving overall model performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel bias mitigation method for large language models (LLMs) based on bi-directional bias attribution, which modifies neuron weights directly rather than relying on fine-tuning or prompt modification. The approach first identifies stereotype-inducing words, computes forward and backward attribution scores to locate neurons that strongly mediate biased associations, and then adjusts those neurons’ outgoing weights to neutralize bias effects. The method is efficient and interpretable, making it attractive for bias correction without retraining. Experiments across standard bias benchmarks demonstrate meaningful bias reduction while largely preserving model quality.

I think the main contribution lies in connecting interpretability techniques with bias mitigation, creating a new approach to fairness interventions. The reliance on manually identified stereotype words make the approach somewhat limiting and the theoretical basis for weight adjustment and its generalizability beyond the tested contexts remain unclear.

### Strengths
Novel combination of interpretability for bias mitigation: 
The paper presents a creative and technically meaningful idea: using bi-directional attribution to identify and correct bias-related neurons without fine-tuning or prompt engineering. While attribution-based interpretability methods are well known, directly leveraging them for bias mitigation is a fresh and promising direction that bridges the two important subfields. 

Training-free, efficient, and potentially generalizable method: 
The approach modifies existing model weights through targeted interventions rather than retraining, making it computationally light and practically attractive. This could be particularly useful for post-deployment fairness adjustments, where retraining is infeasible.

Mechanistic interpretability grounded framework: 
By analyzing neuron activations through forward and backward attributions, the method provides a more interpretable view of where and how biases emerge inside the model. Even if attribution faithfulness is imperfect, the attempt to connect bias to internal mechanisms rather than surface outputs is conceptually strong.

Solid empirical evaluation: 
The experiments are systematic, showing consistent bias reduction across multiple datasets while maintaining generation quality. The results demonstrate the feasibility of neuron-level interventions and position the method as a practical proof of concept.

### Weaknesses
Fragile theoretical grounding and interpretability assumptions: 
The method assumes that gradient-based attributions faithfully capture causal influence of neurons on biased behavior, but this assumption is only weakly supported. Attribution methods in large transformers are known to be noisy, context-sensitive, and often misaligned with causal importance. Without rigorous validation, it remains unclear whether the identified “bias neurons” are genuinely responsible for the effects being mitigated. In terms of interpretability, there is little empirical evidence that the neuron-level adjustments are truly interpretable in practice, for example, by visualizing which features are changed or explaining their semantic role.

Dependence on predefined stereotype word lists: 
The identification of bias-inducing neurons relies on manually or heuristically selected “stereotype words.” This injects subjective bias into the pipeline and may limit generalizability to less well-defined attributes or intersectional biases. The reliance on such lexicons also makes the approach difficult to scale beyond English or to subtle forms of representational bias.

Missing related work section:
The paper lacks a thorough review of prior interpretability-based bias or neuron-level editing approaches. The absence of this context makes it harder to evaluate the originality and significance of the proposed method.

Limited analysis of generalization and stability: 
The evaluation focuses on standard bias benchmarks and does not examine whether the neuron edits generalize to other models, domains, or unseen contexts. There is also no robustness analysis to test stability across different random seeds or attribution thresholds. Given that attribution methods can vary across runs, this omission weakens confidence in reproducibility.

### Questions
The attribution and interpretability portion: 
How confident are you that the forward and backward attribution signals identify neurons with genuine causal influence on biased behavior rather than merely correlated activation patterns?
Have you compared your neuron selection against causal intervention baselines (e.g., activation patching, representation ablation) to confirm that the edited neurons are indeed responsible for the observed bias?
Can you provide qualitative evidence (e.g., neuron visualizations, activation maps) demonstrating that your edits are interpretable and consistent across examples?

Reliance on stereotype word lists: 
Do you expect the method to generalize to settings where biased cues are more subtle, implicit, or multimodal?
How does the method behave on other social dimensions beyond those represented in your word lists (e.g., intersectional or cultural biases)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a debiasing method where they first identify specific neurones that get activated for stereotypical cues in the input. Next, a previously proposed intervention mechanism is applied to the detected neurones to mitigate social biases.

### Strengths
The paper considers an important problem -- how to mitigate social biases in LLMs. Considering much prior work that use prompt-based approaches or fine-tuning LLMs (or alignment) this paper proposes a different approach where a subset of neurones responsible for social biases are identified and then acted upon.

### Weaknesses
I do not understand why P(man | The doctor is likely a) is considered as a stereotypical inference in Definition 2. For example, it could indeed be an image of a male doctor shown to an LLM and the correct prediction would be it is a man. It does not cause any stereotypical bias against the disadvantaged group (i.e. females in this case). 
- The definitions of social bias types considered in the paper are not provided. For example, do you consider gender to be binary? This would affect how for example SFI is interpreted (what does the set D include). 
- The propose method assumes bias triggers/cues can be found at word level. However, this assumption is questionable. For example, a word such as "black" could trigger a demographic attribute or just indicate a colour. It is not clear how such word-level ambiguities are handled in the proposed method. See  (https://aclanthology.org/2022.acl-long.135/) for a discussion of sense-level bias evaluation.
- I do not think nouns and adjectives are the only triggers of stereotypes. See questions below for an example of a verb triggering social biases. It is not explained in the paper why the authors decided to limit their templates to nouns and adjectives. This looks like a severe limitation of the proposed method.
- Template-based approaches have coverage issues as explained in prior work (https://arxiv.org/abs/2210.04337, https://aclanthology.org/2025.findings-acl.1361/) Despite this, the proposed method still resort to using templates, which is problematic.
- This is not a weakness but a suggestion for improvement. The authors could include an example in the introduction to explain what they mean by "stereotypical cues" and how they are used for bias mitigation. The current description in the introduction is at a very high-level and it is nearly impossible to understand what is proposed without an example.

### Questions
- How is Integrated Gap Gradient defined for demographic attributes that take more than two values such as race?
- "First, in the lower layers of deep language models, the contribution of individual neuron activations to the final output tends to be marginal, as their influence is increasingly transformed and potentially suppressed by the model’s subsequent non-linear operations"... can you provide a reference or experimental evidence to support this claim?
- How does the proposed debiasing method handle intersectional biases?
- How does the proposed method handle ambiguities arising at word-level processing?
- Why are stereotypical cues limited to nouns and adjectives? For example, "female surgeons tend to kill more patients than male surgeons" shows a bias against females and is indicated by a verb.
- I do not understand the significance of the result of Theorem 1. Can you explain how the proposed method depends on this Theorem?
- IG and IG^2 are both previously proposed intervention methods. As I see it your work is simply using these methods but on a set of neurones identified using stereotype inducing nouns/adjectives. Could you explain whether there is any methodological novelty (any innovation that you had to do these previously proposed methods) when applying them to the current problem of social bias mitigation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of social bias in large language models (LLMs) by proposing a method for both bias analysis and mitigation. The authors first introduce a technique to identify stereotypical words associated with biased behavior. They then adapt the Integrated Gradients method to mitigate the detected bias within the model. The experimental results demonstrate strong performance in reducing bias, and the paper also provides theoretical insights that help explain the effectiveness of their approach.

### Strengths
With the widespread adoption of large language models, addressing their social impact—particularly bias—has become increasingly critical. This paper tackles this important issue by proposing a practical solution tailored for open-weight LLMs. In addition to empirical results, the paper offers theoretical analysis that sheds light on the underlying mechanisms of bias and its mitigation, which adds depth to the contribution.

### Weaknesses
Since this paper focuses on social bias, rigorous and meaningful evaluation is both crucial and challenging. I have two main concerns in this regard. First, the evaluation of the bias-related word selection process lacks clarity and quantitative justification. Second, the methodology used for evaluating bias in the language models themselves needs further elaboration and validation. (See detailed comments in the Questions section.)

### Questions
Q1. On the Definition of Fairness:
I understand that defining fairness in the context of text generation is inherently difficult. In line 95, for example, the input is "Her mother was very happy", and the two possible outputs are "because her son got a good grade" and "because his son got a good grade". Forcing the model to assign similar probabilities to these two outputs might not always be desirable, as it could harm overall generation quality. This kind of fairness constraint seems appropriate only for certain specific tasks and not as a general objective across all LLM outputs.

Q2. On Stereotype Cue Selection:
My main concern with the stereotype cue word selection is the lack of validation. There is a rich body of sociological literature that studies biased or stereotyped words, and many of those works emphasize that bias is highly context-dependent—what’s considered biased in one context might be neutral in another. Therefore, it’s important to compare and discuss your selected word list against existing lists in the literature. Some justification, especially through expert annotation, is needed to support your word selection.

Q3. On Bias Evaluation Benchmarks:
In the experimental section, the paper evaluates on only a limited set of bias benchmarks. But bias in NLP is notoriously hard to measure, and researchers typically run evaluations on a wide range of benchmarks to ensure robustness. For example, datasets like CrowS-Pairs, SEAT, and downstream tasks such as NLI or fairness-aware text classification are commonly used. I also suggest referring to this paper for more context and standard practices: https://arxiv.org/abs/2210.14975

Q4. On Benchmark Pitfalls and Best Practices:
Finally, I strongly recommend the authors read and reflect on the paper "Stereotyping Norwegian Salmon: An Inventory of Pitfalls in Fairness Benchmark Datasets" https://aclanthology.org/2021.acl-long.81.pdf. It highlights many common pitfalls in fairness evaluations, such as flawed assumptions, annotation bias, and lack of demographic nuance. Including a discussion of these issues would improve the quality and credibility of the evaluation in this paper.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a bi-directional bias attribution framework (BBA) for mitigating social bias in large language models (LLMs) without prompt modification or fine-tuning. BBA combines forward and backward integrated-gradient analyses to identify bias-related neurons and performs lightweight activation interventions at the projection layer input to suppress their influence. Extensive experiments showcase its effectiveness.

### Strengths
1. The method avoids the computational and maintenance costs of fine-tuning or prompt engineering, relying only on activation-level adjustments.

2. Integrated-gradient-based neuron attribution offers transparency and clear diagnostics of where biases emerge.

3. Experiments showcase the effectiveness of this method.

### Weaknesses
BBA assumes access to hidden activations and gradients, which is feasible for open-source LLMs (e.g., LLaMA, Mistral) but not available in closed-source systems such as GPT models. 
Although I think it is not very important limitation, we hope the authors can clearly locate this study as designed for **open-source models**.

### Questions
I hope the authors can discuss that how to extend this study to close-source LLMs. However, I still maintain the point that this study is good enough.

### Soundness
3

### Presentation
3

### Contribution
3
