# Revealing the Impacts of In-Context Learning on Gender Bias in Large Vision-Language Models

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 6, 4

## Abstract
In-context learning (ICL) has emerged as a flexible paradigm, enabling large vision-language models (LVLMs) to perform tasks by following patterns demonstrated in context exemplars. While prior work has focused on improving ICL performance across multimodal tasks, little attention has been paid to its potential to amplify societal stereotypes. This study aims to fill this gap by systematically investigating how ICL influences societal biases, with a focus on gender bias, in LVLMs. To this end, we propose a comprehensive evaluation framework comprising six ICL settings and evaluate four LVLMs across two tasks. Our findings indicate that ICL could amplify gender bias, while female-presenting in-context examples generally do not exacerbate bias and may even mitigate it. In contrast, similarity-based retrieval methods, originally designed to improve ICL performance, fail to consistently reduce gender bias in LVLMs. To mitigate gender bias through ICL, we propose a provisional approach that replaces natural in-context images with synthetic ones. This method achieves lower gender bias while maintaining stable performance on standard quality metrics. We further show that textual cues alone can influence the gender bias level of LVLMs through ICL, and that adding visual cues modulates this already strong textual signal. We advocate for the pre-deployment assessment of gender bias in the context for LVLMs and call for the advancement of ICL strategies to promote fairness on downstream applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the ability of in-context examples to affect the gender bias in vision-language model predictions. The authors focus on two tasks: image captioning with the COCOBIAS dataset and pronoun prediction with the VISOGENDER dataset. The study focuses on manipulating the construction of the examples between random, male-only, female-only, balanced, image-image similarity-based, and image-text similarity based. The results with QwenVL, MiniCPM-o 2.6, Qwen2.5-VL-7B, and Idefics-3-8B show that careful construction of the example set can mitigate gender bias, but the error analysis shows that very few of the examples are incorrectly predicted, raising concerns about the nature of this problem in modern VLMs.

### Strengths
S1: The paper is clearly written, with a clear structure. The experiments are well-motivated and described. This is an important and interesting problem to study.

### Weaknesses
W1: The analyses and supporting figures are too difficult to understand. It takes too much effort for the reader to figure out exactly how "MR_m - MR_f are more favorable under the FS setting than under the MS setting" (L304). Figures 2 and 3 need urgent attention to the y-axis labels, and the captions, reminding the reader about the meaning of RS, MS, FS, Bx, SIIR, and SITR.

W2: The analyses in this paper focus on rather simple next-token completion tasks that are exactly how the models would have been adapted for multimodal ability. The analysis could be improved by considering VQA-style tasks, for which binary gender data splits exist [1].

[1] Cabello et al. EMNLP 2023. Evaluating Bias and Fairness in Gender-Neutral Pretrained Vision-and-Language Models.

### Questions
Q1: What is the relationship netween the Bias Evaluation Metrics used in this paper and Bias Amplification?

Q2: Does mutlimodal ICL actually work in these models? The lack of variation in the classic measures in Table 1 suggests that QwenVL might be ignoring the in-context examples. 

[2] Wang and Russakovsky. ICML 2021. Directional bias amplification.

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
3

### Summary
The paper studies whether and how in-context learning (ICL) alters gender bias in large vision-language models (LVLMs). The authors design a six-setting ICL evaluation framework, measure four state-of-the-art LVLMs on image-captioning and VQA, and observe that (i) ICL can indeed amplify gender bias, (ii) placing female-presenting exemplars in the prompt often mitigates bias, (iii) similarity-based example-retrieval does not consistently help, and (iv) synthetic in-context images provide a practical mitigation knob. The work concludes with a recommendation that LVLM deployments should examine the bias of their prompts and that more fairness-aware ICL strategies are needed.

### Strengths
Important and timely question: while ICL is now the de-facto way to steer LVLMs, its influence on social bias has been largely unexplored.
Comprehensive evaluation design: six prompt settings, two tasks, four public LVLMs, and both amplification and mitigation analyses.
Interesting empirical findings: (a) ICL occasionally increases gender bias even when overall utility improves; (b) female exemplars are systematically more helpful than male ones; (c) retrieval-based exemplar selection, though popular for accuracy, does not cure bias.
Practical mitigation recipe: swapping natural contextual images for synthetic renders is simple yet shows measurable bias reduction without hurting standard metrics.
Clear call-to-action: the paper argues convincingly that prompt designers should audit bias and that future ICL research must consider fairness.

### Weaknesses
Limited definition of “gender bias”: the study focuses on binary male/female stereotypes; it ignores non-binary identities and intersectional dimensions (race, age, etc.).
Causal attribution is still weak: while correlations are shown, the mechanisms of amplification (e.g., attention patterns, token probabilities) are not deeply analysed.
Only two downstream tasks: conclusions may not generalise to other vision-language tasks such as visual reasoning or grounded dialogue.
Dependence on proprietary models: two of the four LVLMs have opaque training data and safety filters, making some observations hard to reproduce or explain.
Synthetic-image mitigation is evaluated with a small set of GAN-generated faces; effectiveness on more complex scenes or in real applications remains unclear.

### Questions
Did you control for content drift when replacing natural images with synthetic ones? Could changes in low-level image statistics, not gender attributes, drive the mitigation?
How sensitive are results to the number of in-context exemplars? Does bias amplification scale linearly with more biased examples?
Can your synthetic-image strategy be combined with similarity-based retrieval (i.e., retrieve similar images, then substitute them with synthetic gender-balanced versions)?
Did you examine other social biases (race, age, profession)? If so, are the trends similar; if not, do you foresee any obstacles to extending your framework?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper systematically investigates the impact of in-context learning (ICL) on gender bias in Large Vision-Language Models (LVLMs). The authors propose a comprehensive evaluation framework comprising six ICL settings (Random, Male-only, Female-only, Balanced, and two similarity-based retrieval methods). This framework is used to evaluate four different LVLMs across two tasks: image captioning (on COCOBIAS) and pronoun prediction (on VISOGENDER). The study finds that ICL can amplify existing gender biases, but this effect can be masked by model safety filters or low "reveal rates". A key finding is that female-presenting in-context examples tend to mitigate bias, whereas common similarity-based retrieval methods (designed for performance) fail to consistently reduce it and may even amplify it. As a provisional mitigation strategy, the paper proposes replacing natural in-context images with synthetic ones, which is shown to lower gender bias while maintaining stable task performance.

### Strengths
1. Important and Timely Problem: The paper tackles a critical and under-explored area. While ICL for LVLMs is a popular research topic, most work focuses on performance improvements, neglecting the potential for bias amplification. This study provides a much-needed analysis of the fairness implications.
2. Comprehensive Methodology: The proposed framework of six distinct example-selection strategies is a key strength. It allows for a systematic and controlled study of how the composition of in-context examples influences model behavior, moving from random baselines to attribute-specific (MS, FS) and performance-oriented (SIIR, SITR) settings.
3. Thorough Experimentation: The findings are supported by experiments on four modern LVLMs (QwenVL, MiniCPM, Qwen2.5VL, Idefics3) and two distinct tasks (generation and prediction). This demonstrates the robustness and generalizability of the conclusions.

### Weaknesses
1. Contradictory Claims on Prior Bias: The paper's explanation for why introducing female examples helps reduce bias relies on the claim that they "help counterbalance the LVLMs’ biased priors against females". This is directly contradicted by the paper's own zero-shot results on COCOBIAS, which found that three of the four models already "demonstrate a consistent bias toward the female category". This inconsistency undermines the core explanation for the paper's main finding.
2. Lack of Mechanistic Explanation: Related to the point above, the paper does a good job of showing what happens (FS mitigates bias) but struggles to provide a deep explanation of why it happens.
3. Limited Scope of ICL Factors: The study focuses exclusively on the composition of the demo examples. However, a large body of work on ICL has shown it to be highly sensitive to other factors, such as the order of the examples (Yang et.al). The "Balanced Sample" setting, for instance, interleaves examples, but it's unknown if the effect would hold if the order were changed (e.g., order the examples with the same label as the query in the beginning or closest to the query).
4. Visual vs. Textual Cues: By claiming ICL can amplify existing gender biases but not so much, this paper assumes that the models are reacting to the visual content of the in-context examples. However, recent research has suggested that multimodal models often pay little attention to visual context in ICL, relying heavily on textual cues instead (Chen et.al). Is it possible that the model is not using the images, and that's why the results show ICL has limited influence?
5. Practicality of Mitigation: The proposed mitigation (using synthetic images)  is interesting but its practical application is unclear. This approach adds a significant computational overhead (running a text-to-image model like SDM) for every set of in-context examples, which may not be feasible for real-time applications. The paper doesn't discuss this trade-off.

Reference

Yang, X., Peng, Y., Ma, H., Xu, S., Zhang, C., Han, Y., & Zhang, H. (2024). Lever LM: configuring in-context sequence to lever large vision language models. Advances in Neural Information Processing Systems, 37, 100341-100368.

Chen, S., Liu, J., Han, Z., Xia, Y., Cremers, D., Torr, P., ... & Gu, J. True Multimodal In-Context Learning Needs Attention to the Visual Context. In Second Conference on Language Modeling.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2
