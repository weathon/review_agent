# UDDETTS: Unifying Discrete and Dimensional Emotions for Controllable Emotional Text-to-Speech

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Recent large language models (LLMs) have made great progress in the field of text-to-speech (TTS), but they still face major challenges in synthesizing fine-grained emotional speech in an interpretable manner. Traditional methods rely on discrete emotion labels to control emotion categories and intensities, which cannot capture the complexity and continuity of human emotional perception and expression. The lack of large-scale emotional speech datasets with balanced emotion distributions and fine-grained emotional annotations often causes overfitting in synthesis models and impedes effective emotion control. To address these issues, we propose UDDETTS, a universal LLM framework unifying discrete and dimensional emotions for controllable emotional TTS. This model introduces the interpretable Arousal-Dominance-Valence (ADV) space for dimensional emotion description and supports emotion control driven by either discrete emotion labels or nonlinearly quantified ADV values. Furthermore, a semi-supervised training strategy is designed to comprehensively utilize diverse speech datasets with different types of emotional annotations to train the UDDETTS. Experiments show that UDDETTS achieves linear emotion control along three interpretable dimensions, and exhibits superior end-to-end emotional speech synthesis capabilities. Code and demos are available at: https://anonymous.4open.science/w/UDDETTS.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a controllable emotional TTS framework that unifies discrete emotion labels with a dimensional Arousal–Dominance–Valence (ADV) space. Contributions can be summarized as:
- a semi-supervised neural-codec LLM that can take either labels or quantized ADV tokens as control input;
- a nonlinear binning scheme that discretizes the ADV space into controllable units;
- an ADV predictor that infers pseudo-ADV from text for end-to-end, text-only emotional synthesis.

### Strengths
- Unifying discrete and dimensional emotion control in LLM-based TTS is well-motivated and really novel for fine-grained affect control.
- This paper presents solid and extensive experiments.
- This paper provides code at an anonymous link.

### Weaknesses
- ADV ranges are normalized to [1,7], and bins are chosen via a CLT-inspired heuristic plus clustering. It’s unclear how sensitive control linearity is to the chosen number of bins and cluster variability.
- Although the paper aims to achieve fine-grained and interpretable emotional control through the continuous (ADV) space, the amount of training data with ground-truth ADV annotations appears to be very limited. Most emotional datasets only provide discrete emotion labels, while ADV values are available for a small subset. As a result, the ADV predictor and the overall controllability of the system rely heavily on pseudo-ADV values inferred from semi-supervised learning rather than real annotated data. This raises concerns about the precision and reliability of the ADV mapping, especially for subtle or compound emotions. The scarcity of reliable ADV-labeled samples might constrain the model’s ability to learn accurate continuous emotion representations, which somewhat contradicts the paper’s goal of achieving fine-grained control in the ADV space.
- The system employs both an ADV predictor and a label predictor. However, the paper does not clearly explain how these two emotion sources interact or which one dominates when their predictions disagree. Since the final emotional output depends on the fusion of both, inconsistencies between the predicted ADV vectors and categorical labels could lead to unstable or conflicting emotional expressions. Moreover, no quantitative analysis (e.g., disagreement rate, calibration curve, or preference correlation) is provided to demonstrate whether the two predictors are aligned. This ambiguity raises concerns about the reliability and interpretability of the emotional control, which is central to the paper’s claimed contribution.

### Questions
- How robust are the nonlinear bin boundaries across different training splits?
- How often does the ADV predictor disagree with the LLM-predicted label, and which path dominates the final emotion?

### Soundness
2

### Presentation
4

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
This paper proposes UDDTTS, an LLM-based TTS using an ADV space to model emotional representations for expressive speech synthesis. While the idea is interesting, the paper lacks methodological clarity, strong experimental validation, and a comprehensive review of related LLM-based TTS work. The results also do not show clear advantages over existing methods.

### Strengths
The work introduces a potentially useful direction for controllable emotional TTS by modeling ADV in LLMs.

### Weaknesses
Your proposed UDDTTS does not outperform other approaches in terms of MOS, UTMOS, WER, SS, and STOI. I suggest further improving these metrics through more refined method design.

Methodology is not well written. Please define the symbols before using them. I am confused with the method design. 

The generated speech quality is not good with unclear pronunciations, which is not common in the existing TTS models. I am wondering if including ADV is the reason why the speech intelligence is getting worse. I would suggest improving the performance further with more advanced techniques.

The literature review on LLM-based TTS approaches is relatively limited, and a more comprehensive investigation is recommended.

I am not fully convinced by how you disentangle complex emotions in the ADV space while addressing sparsity and imbalance issues. Could you provide experimental evidence to support this claim?

It is unclear why AB preference tests were not included, as they are commonly used to assess perceptual differences in TTS quality.

### Questions
How did you prove that you capture the continuity of emotion distributions?

Having only 12 listeners for the subjective evaluation is insufficient for a comprehensive assessment of TTS models. For each listener, how many samples were evaluated? Were these samples randomly selected from the test set or manually chosen? 

How did you create and process spontaneous emotion datasets and elicited emotion datasets?

What does Z1 mean? 

Why do you assume that Xspk can effectively represent the speaker embedding while excluding emotional representations? Could you elaborate on this assumption and provide evidence or verification?

How do you demonstrate that your proposed speech tokenizer captures rich emotional information? What are the key differences between your tokenizer and CosyVoice’s speech tokenizer?

I am also unclear about the motivation, design, and working mechanism of the ADV predictor. Could you explain this in more detail? Is the speech signal considered in its process, or is it modeled solely based on textual emotion inputs? If speech is not incorporated, the emotional states between speech and text might differ. How did you address this issue?

Will you release your testests?

### Soundness
2

### Presentation
1

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
The paper proposes UDDETTS, a unified LLM-based framework for controllable emotional TTS that integrates both discrete emotion labels and dimensional emotions in the Arousal-Dominance-Valence (ADV) space. It addresses challenges including the sparsity and imbalance of emotional annotations, by introducing a semi-supervised training strategy and a nonlinear binning method for ADV quantization. The architecture includes a neural codec language model, an optimal-transport conditional flow matching (OT-CFM) module with an emotional mixture encoder, and a vocoder. An ADV predictor supports end-to-end synthesis from text alone. Trained on large-scale emotional and general speech datasets, UDDETTS demonstrates superior performance in label-controlled, ADV-controlled, and end-to-end TTS tasks, with linear control along ADV dimensions.

### Strengths
1. The paper proposes a LLM-based TTS framework to explicitly unify discrete and dimensional emotions, addressing a key limitation in prior work of emotional TTS.
 
2. Introducing the interpretable ADV space to LLM-based TTS is a meaningful step toward continuous, decoupled emotion control, addressing limitations of discrete-label methods. The nonlinear binning and semi-supervised fusion of annotations effectively tackle data imbalance and sparsity.

3. Evaluations across three tasks use diverse metrics (e.g., MOS, ES, SRC/KW) and show consistent improvements over baselines. The visualization in Figure 4 effectively shows that the proposed techniques (nonlinear binning, semi-supervised training) increase the coverage of the ADV space.

### Weaknesses
1. The novelty is limited. The work is built directly upon the architecture of models like Spark-TTS and CosyVoice. The addition of ADV control seems to be an incremental improvement rather than a novel framework.

2. The core components lack detailed explanation. For ADV quantizer, the nonlinear binning based on clustering is a potential key innovation, but its derivation and relationship to solving sparsity/imbalance are unclear in the main text.

3. The semi-supervised strategy for mixing spontaneous/elicited datasets with varying annotations is not sufficiently ablated. It's unclear if this fusion is mutually beneficial or merely a way to scale data volume.
 
4. The experiments are not sufficient enough. Further justifications are required. 
* The baselines were not trained on the same datasets, making it difficult to determine whether the performance is influenced by model architecture or training data.
* Comparisons with other dimensional emotion models (e.g., EmoSphere++) are missing. 
* The comparison against "description-based baselines" (Sec. 4.5) is potentially unfair, as these models are not designed for the specific prompt format used.
* Custom emotional texts (Table 7) lack details on design/validation for bias (e.g., inter-annotator agreement or diversity checks), risking overfitting to specific prompts.

### Questions
1. The nonlinear binning is central to handling sparsity. Can you provide a more intuitive explanation about how the clustering algorithm leads to a balanced and effective quantization? Why not alternatives like quantile-based or density-based methods? How sensitive is the coverage rate (89.35%) to bin count (m=14)?

2. Why is a separate RoBERTa-based regression model with MSE loss used instead of integrating ADV token prediction directly into the LLM (using CE loss like sparkTTS)? Was the above latter alternative approach explored, and if so, how did its performance compare?

3. For Sec. 4.5 comparisons, did you ensure baselines were prompted in alignment with their intended capabilities? The current setup may not fairly test them.

4. How do you demonstrate that mixing dataset types (spontaneous vs. elicited) via semi-supervised training can benefit this task effectively, rather than just adding data volume? For example, ablate training only on fully labeled data (D_{S,AL}) vs. the full setup.

### Soundness
3

### Presentation
2

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
This paper proposes UDDETTS, a LLM framework that unifies discrete and dimensional emotions for controllable emotional text-to-speech (TTS). The framework introduces an interpretable ADV space to describe dimensional emotions, supporting emotion control driven by discrete emotion labels or non-linearly quantized ADV values. Moreover, this paper designs asemi-supervised training strategy to fully utilize speech datasets with different emotion annotation types, experimental results show promising emotion controlablety for speech synthesis.

### Strengths
1. The paper tackles a critical and timely problem in expressive speech synthesis, contious and dimensional control is a clear and important direction for the field.
2. The semi-supervised learning strategy is an effective solution to extend the training to larger-scale dataset, while only part of the data is well labeled.

### Weaknesses
1. Although this article compares many different baselines, the reasonableness of the comparison is still not clear to me. A more reasonable comparison would be to add the adv prediction and control modules to the corresponding frameworks, which would better illustrate the universality of the article's contribution.
2. Some details are not very clear. For example, in Table 3, preference scores are given for two systems, but it is uncertain whether the same backbone is used for the corresponding systems, and it is also uncertain whether the baseline systems have been optimized with similar training methods using the same emotional data.
3. The article mentions some other control schemes, such as EmoSphere-TTS, but they are not shown in the experimental results.

### Questions
Besides the issues mentioned in the weakness,

1. How sensitive is the model's performance to the ratio of ADV-annotated data versus label-only data?
2. Has the accuracy of the ADV predictor been tested standalone?
3. Was any experiment conducted where the ADV predictor and the emotional mixture encoder were integrated into a different LLM-based framework, such as IndexTTS2 or Spark-TTS, to measure the performance lift?

### Soundness
3

### Presentation
3

### Contribution
3
