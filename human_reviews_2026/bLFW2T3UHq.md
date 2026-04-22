# Grounding or Guessing? Visual Signals for Detecting Hallucinations in Sign Language Translation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 2, 8

## Abstract
Hallucination, where models generate fluent text unsupported by visual evidence, remains a major flaw in vision–language models and is particularly critical in sign language translation (SLT). In SLT, meaning depends on precise grounding in video, and gloss-free models are especially vulnerable because they map continuous signer movements directly into natural language without intermediate gloss supervision that serves as alignment. We argue that hallucinations arise when models rely on language priors rather than visual input.
To capture this, we propose a token-level reliability measure that quantifies how much the decoder uses visual information. Our method combines feature-based sensitivity, which measures internal changes when video is masked, with counterfactual signals, which capture probability differences between clean and altered video inputs. These signals are aggregated into a sentence-level reliability score, providing a compact and interpretable measure of visual grounding. 
We evaluate the proposed measure on two SLT benchmarks (PHOENIX-2014T and CSL-Daily) with both gloss-based and gloss-free models. Our results show that reliability predicts hallucination rates, generalizes across datasets and architectures, and decreases under visual degradations. Beyond these quantitative trends, we also find that reliability distinguishes grounded tokens from guessed ones, allowing risk estimation without references; when combined with text-based signals (confidence, perplexity, or entropy), it further improves hallucination risk estimation.  Qualitative analysis highlights why gloss-free models are more susceptible to hallucinations. Taken together, our findings establish reliability as a practical and reusable tool for diagnosing hallucinations in SLT, and lay the groundwork for more robust hallucination detection in multimodal generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper tackles the problem of hallucination in gloss-free sign language translation. SLT is a challenging task, which has been facilitated by the use of large language models (LLMs) and vision language models (VLMs) in recent years. However, these models tend to hallucinate. Gloss-level supervision can reduce hallucinations, by grounding the model with sign-level supervision, but glosses are expensive to annotate and label (note from reviewer: and introduce other problems). The authors of this paper propose a novel method to detect hallucinations and ground models in visual inputs, which reduces hallucinations and improves the results on gloss-free sign language translation.

### Strengths
- The paper is, for the most part, well-written and pleasant to read.
- The paper is well situated in related work.
- This is important work, as it improves the reliability of SLT models, especially gloss-free ones (which is critical because glosses can be considered harmful for SLT.)

### Weaknesses
- The paper contains acronyms that are not defined, e.g., AP.
- The methodology is rather math heavy and can be challenging to follow at times. It would benefit from some intuitive explanations of the equations.
- There is no statement on whether the signing community was consulted for this work. The authors should make a clear statement on their hearing/signing status.
- Appendix C and D are empty (probably because they contain only figures.) This layout problem should be addressed.

### Questions
- If we remove the subclause from the first line of your abstract, it reads "Hallucination is particularly critical in SLT." That is not the point you are trying to make, so please rephrase this.
- I recommend adding a statement about the dangers of using glosses for sign language translation at the end of the first paragraph of your introduction. They should not be used for sign language translation purposes: see [1]. This will further strengthen your paper since you are tackling gloss-free SLT.
- Does your system have merit in domains other than SLT to detect hallucinations?

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
4

### Summary
This work investigates hallucination in sign language translation. The authors introduce a novel metric named reliability to measure the hallucination in SLT. The propose metric is based on two parts, i.e., internal model changes and output probability shifts. By adopting the CHAIR as the annotation on sentence-level, the proposed metric shows encouraging performance based on two SLT models on both benchmarks, i.e., PHOENIX-2014T and CSL-Daily.

### Strengths
It is the first work to investigate hallucination in sign language. The authors provide a novel metric to measure the reliability for SLT based on the visual input and text output. They also demonstrate the proposed reliability surpasses text-only baselines in detecting hallucinations.

### Weaknesses
1.The authors did not provide a clear analysis of the relationships about the three key concepts, i.e., sign language translation performance, the rate of hallucination and the proposed metric.

2.The key steps of the proposed metric is missing, such as how to calculate the weights. The appendix is also not completed, which is confusing. For specific issues, please refer to the descriptions in the Question section.

3.The authors claim that the hallucinations significantly impact the performance of SLT. However, there lacks data showing how hallucinations hurt translation quality. There is only limited comparative examples are given in the appendix, which is insufficient to support the main contributions.

4.Since the goal of mitigating hallucinations is to improve sign language translation performance, I think the authors should clarify the specific application directions of the proposed metric in current practical contexts. 

5.The authors attribute the main reason of hallucinations to the poor performance of visual encoder. I think this point has been observed in the previous study [1]. It enhances visual representation capabilities by incorporating small pre-trained language models. Please provide more details about how to distinguish it from existing research in the manuscript.

[1] Zhigang Chen, Benjia Zhou, Jun Li, Jun Wan, Zhen Lei, Ning Jiang, Quan Lu, and Guoqing Zhao. 2024. Factorized Learning Assisted with Large Language Model for Gloss-free Sign Language Translation. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 7071–7081, Torino, Italia. ELRA and ICCL.

### Questions
1.The authors provide the results on two SLT models. I suggest that the authors provide results on more open-source models. Additionally, the current method only considers cases where the input is video. Could it measures the different inputs, like poses?

2.All formulas in the manuscript lack recommended numbering.

3.There are notations in the equations lack descriptions. 

In Line 154, $\pi^{-1}$ lack descriptions. Which part of the model does “hidden state” represent?

In Line 164, $scale$ lack descriptions. For $s_t^{attn}$, does a large result also indicate higher reliability? Are the set “video” and “masked” same?

4.In the 4 note, the authors provide a brief description about the calculation method of CHAIR. I think the authors should provide a more detailed description.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This manuscript aims to evaluate "hallucination" in sign language translation (SLT), defined as the extent to which the decoder generates token-level predictions without relying on visual information. To assess hallucination in SLT accurately, the manuscript proposes feature-wise signals, such as changes in feature orientation and intensity shifts in attention. Additionally, it leverages prediction discrepancies under noisy conditions to evaluate hallucination. Extensive experiments demonstrate that the proposed reliability indicator correlates with the "CHAIR" phenomenon in SLT, achieving impressive detection performance.

### Strengths
- S1. This manuscript is well-motivated, addressing the crucial issue of evaluating and improving the utilization of visual information in sign language translation (SLT), a topic that has garnered significant attention in recent years. The manuscript aims to design an indicator to shed light on this challenge.
- S2. The manuscript provides a comprehensive set of indicators for assessing the utilization of visual information, considering both the feature space and the output space.
- S3. The manuscript conducts thoughtful and thorough experiments on hallucination evaluation and detection.

### Weaknesses
- W1. The definition of hallucination in SLT is unclear. While hallucination in visual-language models (VLMs) is introduced due to the ambiguity in input range and output space, SLT has clearly defined output results and visual-language correspondences. The errors in hallucination can be directly measured using metrics like word error rate (WER) or similar set prediction metrics. Therefore, the introduction of the hallucination concept in SLT seems unnecessary.
- W2. As a translation task, many words are predicted based on contextual information. What we are truly concerned with are the words that can only be predicted by visual information. However, this manuscript does not address the distinction between different types of tokens in this context.
- W3. The insufficient use of visual information is a well-known issue in the sign language understanding community, and addressing this challenge is critical. While this manuscript attempts to detect hallucination, it does not propose any solutions for resolving this problem.
- W4. The practical applicability of the proposed method is limited. As shown in Figure 1, the proposed metric shows some correlation with "CHAIR" in SLT, which is used to demonstrate the effectiveness of the indicator. However, why not directly use "CHAIR" as the hallucination indicator? Furthermore, the high variation in the proposed metric suggests its limited practical value.
- W5. While Table 1 demonstrates that the proposed method achieves high hallucination detection accuracy, the process for obtaining ground-truth labels for detection remains unclear.

### Questions
- As mentioned in the weaknesses, I doubt the necessity of introducing the concept of hallucination in SLT, as it may mislead future research. However, I remain open to opinions and experimental evidence that could demonstrate the significance of hallucination in SLT.
- I also encourage the author to improve the writing and organization of this manuscript, and provide more valuable take home messages.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a token-level reliability metric for hallucination detection in sign language translation (SLT), combining encoder sensitivity to input perturbations and internal state changes under counterfactual video masking. The metric is evaluated on PHOENIX-2014T and CSL-Daily using both gloss-based and gloss-free models.

### Strengths
-  Precisely identifies hallucination in SLT—especially in gloss-free settings—as a critical, under-addressed issue tied to visual grounding weakness.

- The token-level reliability score is architecture-independent, requiring no reference translations, making it broadly applicable across existing and future SLT systems.

- Combines encoder sensitivity (input robustness) and counterfactual masking (internal consistency) into a well-justified, interpretable proxy for visual grounding.

- Establishes a reusable evaluation tool that can become a standard metric in SLT research and development.

### Weaknesses
- Lacks testing on larger, more diverse, or continuous SLT benchmarks (e.g., How2Sign, OpenASL).  
- Tested on a small set of architectures (primarily transformer-based). 
- Unclear whether the proposed reliability score outperforms or complements prior art.

### Questions
Performance Impact on Gloss-Free SLT:
- The paper positions reliability as a diagnostic tool, yet provides no evidence that it leads to tangible improvements in gloss-free sign language translation. Can the authors demonstrate substantial gains (e.g., ≥1.0 BLEU)?  

Cross-Architecture Validation:
- Claims of model-agnosticism are unsubstantiated. Can the authors apply the reliability score to ≥3 diverse gloss-free architectures (e.g., SignBT, STMC, BLT, or non-autoregressive variants) and show:  Consistent hallucination prediction (correlation, AUC)?

### Soundness
3

### Presentation
3

### Contribution
3
