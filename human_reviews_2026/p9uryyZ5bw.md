# Unveiling Impact of Frequency Components on Membership Inference Attacks for Diffusion Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Diffusion models have achieved tremendous success in image generation, but they also raise significant concerns regarding privacy and copyright issues. Membership Inference Attacks (MIAs) are designed to ascertain whether specific data were utilized during a model's training phase. As current MIAs for diffusion models typically exploit the model's image prediction ability, we formalize them into a unified general paradigm which computes the membership score for membership identification. Under this paradigm, we empirically find that existing attacks overlook the inherent deficiency in how diffusion models process high-frequency information. Consequently, this deficiency leads to member data with more high-frequency content being misclassified as hold-out data, and hold-out data with less high-frequency content tend to be misclassified as member data. Moreover, we theoretically demonstrate that this deficiency reduces the membership advantage of attacks, thereby interfering with the effective discrimination of member data and hold-out data. Based on this insight, we propose a plug-and-play high-frequency filter module to mitigate the adverse effects of the deficiency, which can be seamlessly integrated into any attacks within this general paradigm without additional time costs. Extensive experiments corroborate that this module significantly improves the performance of baseline attacks across different datasets and models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work is an analysis of the deficiencies of Diffusion Models when modeling high frequencies in the context of membership inference attacks. The authors claim that filtering out high-frequencies from suspected images can improve attack results by decrising the number of membership FN and thus improving TPR. They show detailed analysis and experimental results depicting improvement as well as ablation showing the impact of the masking strength on MIAs metrics.

### Strengths
- showing the correlation between membership scores and high-frequency content (shown for MS-COCO)
- describing how decreasing FN improves TPR and impacts Youden Index J (redefined as membership advantage in the paper - to my understanding); Overall linking improvements made (TPR -> J) with the effective differentiation of member and hold-out data (better classification).
- showing empirically benefits from masking high frequencies:  MIAs metrics show indirectly that membership FN decrease outweighs TN decrease when the high-frequency masking is used.
- showing that claimed benefits hold when two types of defenses against MIAs are adopted.

### Weaknesses
W1. RW does not mention any of works that analyze frequency components (not in DM not anywhere else) which is crucial to tell that mentioned frequency deficiencies insights are largely known and how new contribution fits to current knowledge. Authors only list previous works, they do not link previous work to their contribution.

W2. The improvement would be easier to understand if standard FN, FP, TPR, FPR metrics would be used. It is an additional work for the reader to understand how they are connected to terms used in the paper, while the idea is simple. To my understanding masking high-frequencies should decrease FP (good thing, the goal described in the paper), but at the same time should also decrease TN (bad thing, not commented nor shown). I understand that in tested settings FN decrease outweighs TN decrease but I would ask for clarification or TN results here.

W3. Tables descriptions are lacking details about the setup/ are not clear. 
Table 1 - what are failed samples? Misclassified? Clarify the comparison what hold-out samples you compare with? Similar issues with Tab2, Tab5, Tab6.
 
W4. Results for 3 out of 4 grey-box methods mentions. Only partial results under one specific setup in the appendix for 4th method. Why?

W5. Other issues:
- the shown underlying problem with distribution shift is not novel for MIAs, yet no previous ideas for mitigations discussed in the paper.
- Contributions line 80 - std. dev of what? It is not specified by then.
- What is the definition of the term: error-based attacks? Are they attacks that use only reconstruction error? How general is the proposed General paradigm? It is created for error-based attacks and generalize only to error-based attacks? 
- In my opinion LLMs were used extensively, especially in the first part of the paper, with rather mixed effect on the clarity, with uplifted words, LLM-sentences and genAI small problems popping here and there (e.g. lines 49-50, 162, 183, maybe also 323: Laion-MI mentioned with no results?).

### Questions
- weaknesses.
- 5.3 lines 380-381 - What is the generality of this claim, what are the assumptions?
- Fig2 - PIA - there is x scale difference for plots, improvement is not evident. 
- What are the assumptions of Proposition 1? What is k?
- What does it mean in practice that s=0.1? 
- Limitations: what is pre-training setting?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper investigates the impact of frequency components on membership attacks on diffusion models. It provides a theoretical analysis to prove that filtering high-frequency components can boost membership attack performance. Experimental results on different datasets and models confirm the discovery.

### Strengths
- The paper investigates an overlooked component in membership attacks
- It provides a theoretical analysis to prove that filtering high-frequency components can boost attack performance
- The proposed approach consistently boost the attack performance across datasets and models

### Weaknesses
- In Proposition 1, why is "the standard deviation of membership scores" considered, instead of the mean scores?
- The high-frequency threshold $r_t$ is recommended to be in [3, 10]. 
  + Is it universal or resolution-dependent / dataset-dependent? An analysis of this is recommended to add.
  + This threshold is small, meaning most of the frequencies can be filtered out. For example, in Table 4, the configuration ($r_t = 3, s = 0$), e.g., removing all frequencies except the first three, could achieve ASR 93.70. This strong result is surprising and may indicate a weakness in the benchmark dataset.
- From Table 4, it seems $s = 0$ can provide competitive performance with other settings, while being simpler and probably more efficient. Why is the setting ($r_t = 5, s = 0.2$) selected as the default in Section 5?
- Section 5.6: The authors should investigate adaptive defenses that exploit knowledge of this high-frequency-based attack mechanism.

### Questions
- In Proposition 1, why is "the standard deviation of membership scores" considered, instead of the mean scores?
- The high-frequency threshold $r_t$ is recommended to be in [3, 10]. 
  + Is it universal or resolution-dependent / dataset-dependent? An analysis of this is recommended to add.
  + This threshold is small, meaning most of the frequencies can be filtered out. For example, in Table 4, the configuration ($r_t = 3, s = 0$), e.g., removing all frequencies except the first three, could achieve ASR 93.70. This strong result is surprising and may indicate a weakness in the benchmark dataset.
- From Table 4, it seems $s = 0$ can provide competitive performance with other settings, while being simpler and probably more efficient. Why is the setting ($r_t = 5, s = 0.2$) selected as the default in Section 5?
- Section 5.6: The authors should investigate adaptive defenses that exploit knowledge of this high-frequency-based attack mechanism.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the impact of frequency components on membership inference attacks against diffusion models. The authors identify a "high-frequency deficiency" in how diffusion models process information, which causes current error-based MIAs to misclassify member data with high-frequency content as hold-out data and vice versa. To address this, they propose a high-frequency filter module that can be integrated into existing attacks to mitigate this deficiency. Extensive experiments on multiple datasets show that the filter improves attack performance.

### Strengths
1. The paper identifies the previously overlooked "high-frequency deficiency" in diffusion models, which adds a new dimension to understanding MIA limitations.
1. The proposed high-frequency filter method is simple, light-weighted, and effective to improve the performance of several current MIA methods. 
2. The paper is easy to understand.

### Weaknesses
1. The study primarily relies on empirical observations of high-frequency signals' impact on MIA performance, lacking a rigorous theoretical exploration or analysis on why high-frequency components degrade attack efficacy, which can provide a more valuable insight for this phenomenon.
2. The ablation experiments in Section 5.4 show that suppressing all high-frequency signals harms performance, but the paper does not adequately explain why this occurs, and how to determine the optimal suppression threshold. 
3. The paper only uses three tradition error-based MIA baselines. Can the method directly filter the high-frequency signals of the test images themselves instead of intermediate steps, so that this method can be applied to more types of advanced baselines? If not, it would be good to give some explanations.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates MIAs for diffusion models, mainly from the perspective of the frequency analysis. In detail, the authors observe that existing MIA methods suffer from a "high-frequency deficiency", where diffusion models handle high-frequency information inefficiently, leading to misclassification of member and nonmember. They formalize MIAs into a general paradigm based on reconstruction errors and propose a plug-and-play high-frequency filter module to mitigate this deficiency. Extensive experiments demonstrate that the filter enhances baseline attack performance across diverse datasets and models.

### Strengths
+ The paper is well-structured and clearly presented, making its contributions easy to follow.

+ The frequency perspective introduced a novel and valuable lens to analyze and understand the effectiveness of existing membership inference attacks (MIA).

### Weaknesses
+ (Major) The core contribution of analyzing memorization through frequency components (structure/low-frequency vs details/high-frequency) appears to have significant overlap with prior work. Specifically, [1] investigates membership inference by separating structural and detailed information, which is conceptually similar to the low-frequency/high-frequency components discussed here. The manuscript lacks a clear statement on its contribution beyond this established approach.

+ (Major) The underlying mechanism is still not clear. The explanation of **why** high-freq components are handled with more variation by diffusion models is insufficient. The high-level explanation in the introduction (Line 59-62) is vague and lacks support. Key questions remain unanswered: Is there any direct evidence or established literature demonstrating this behavior? What is the precise relationship between the observed "high-freq deficiency" and the well-known concept of "spectral bias" in neural network [2] ?

+ (Major) The derivation and presentation of Proposition 1 are confusing. The condition $k^2>1+\frac{2 \Delta}{h^2_H}(...)$ seems to be presented as a premise, but the proof in Appendix C suggests it is actually a conclusion. The logic appears to be: if $\frac{\sigma_H'}{\sigma_M'}>\frac{\sigma_H}{\sigma_M}$, then we have $k^2>1+\frac{2 \Delta}{h^2_H}(...)$. Besides, why the assumption $k^2>1+\frac{2 \Delta}{h^2_H}(...)$ holds? Any empirical evidence?

+ (Major) To thoroughly demonstrate the method's effectiveness, especially in the critical low-confidence scenario, it is essential to provide ROC curves on a ​​log-scaled axis​​. Linear-scale ROC curves can make performance look artificially strong by compressing the low false-positive rate region, and a log scale is the standard for a more rigorous evaluation.

+ (Minor) The proof (Appendix B) showing that "error" equals "sample" is considered trivial. Furthermore, a very similar proof has already been provided by [3] (equations 7 and 14).

+ (Minor) The method to obtain the threshold used in the threat model is not described. Please specify how this threshold is determined.

[1] Unveiling structural memorization: Structural membership inference attack for text-to-image diffusion models. ACM MM 2024

[2] On the Spectral Bias of Neural Networks. ICML 2019

[3] Are Diffusion Models Vulnerable to Membership Inference Attacks. ICML 2023

### Questions
See the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
