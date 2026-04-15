# Test-Time Fairness and Robustness in Large Language Models

- Decision: Reject
- Scores: 6, 6, 6, 5

## Abstract
Frontier Large Language Models (LLMs) can be socially discriminatory or sensitive to spurious features of their inputs. Because only well-resourced corporations can train frontier LLMs, we need robust test-time strategies to control such biases. Existing solutions, which instruct the LLM to be fair or robust, rely on the model’s implicit understanding of bias. Causality provides a rich formalism through which we can be explicit about our debiasing requirements. Yet, as we show, a naive application of the standard causal debiasing strategy, counterfactual data augmentation, fails under standard assumptions to debias predictions at an individual level at test time. To address this, we develop a stratified notion of debiasing called stratified invariance, which can capture a range of debiasing requirements from population level to individual level through an additional measurement that stratifies the predictions. We present a complete observational test for stratified invariance. Finally, we introduce a data augmentation strategy that guarantees stratified invariance at test time under suitable assumptions, together with a prompting strategy that encourages stratified invariance in LLMs. We show that our prompting strategy, unlike implicit instructions, consistently reduces the bias of frontier LLMs across a suite of synthetic and real-world benchmarks without requiring additional data, finetuning or pre-training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies an important problem of ensuring better fairness and robustness of LLM during test time. It focuses on the notion of stratified invariance and advocates the adoption of a stratified data augmentation procedure at test time. The work further implements the procedure on LLMs through prompting (with specially designed role-playing prompts), and naming this strategy out-of-context (OOC) prompting. Extensive empirical validations are done to demonstrate that OOC improved the stratified invariance of LLM predictions and hence fairness in real-world datasets.

### Strengths
1. The motivations for the test-time fairness enhancement are well written, and the gaps in the current literature for counterfactual invariance are well discussed.
2. It provides a good analysis of the assumptions, and properly discusses them in the context of practical adoption.
3. The empirical experiments are well-designed to show the superior performance of OOC.

### Weaknesses
1. While the authors adequately discussed the assumptions when adopting stratified data augmentation using OCC in the context of LLM, there is no explicit discussion on the limitation of the method in theory and in practice. For example, what are the implications when the assumptions do not hold?

### Questions
1. Since the stratifying measurement or the adjustment set $S$ is important to the stratified invariance, can the authors clarify more in the paper how $S$ is typically chosen? I see in the experiments section that $S$ are usually the labels of the task, or empty set.
2. Could the authors clarify what does $S$ being an empty set mean? And how do they affect the context obfuscation/addition steps when it is an empty set?
3. It is assumed that the LLM can incorporate and generate a response containing the new context $z^+$ given the obfuscated input. Do you have empirical results on whether this is indeed the case? And how to test them practically, so that we can determine whether the proposed out-of-context will work with this LLM at test time?
4. It is unclear to me how to read Figure 2. More explanations should be provided.
5. The paper brings causal invariances to LLM inference. While it may be typical for causal invariances to consider classification tasks, it is not how typically LLMs are used in practice. How does the method extend to generative tasks for LLMs?

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
2

### Summary
This paper considers the problem of LLM debiasing at the test time. By making use of causal invariance, the authors proposed a novel stratified invariance notion to address the limitation of standard conterfactual data augmentation. Besides, an out-of-context prompting strategy, inspired by stratified invariance, was proposed to demonstrate that the bias of LLMs can be reduced for real-world benchmarks.

### Strengths
- Originality:
    - Unlike previous works that used safety instructions to implicitly address the bias issue, this work leverages the causal invariance framework that utilizes interventions to obtain a less biased result.
    - This work also developed a stratified invariance notion that is built on observational data (random generations).
    - A novel OOC strategy is introduced to debias LLM predictions.

- Quality: 
    - The theoretical definition and analysis are introduced for stratified invariance which makes the design of OOC in principle.
- Clarity:
    - The clarity could be improved.
- Significance:
    - The proposed method evaluated on stratified invariance bias shows significant improvement, but it is unclear on other evaluation metrics.

### Weaknesses
- The presentation of this paper could be substantially improved. I tried very hard to understand this paper, but many things still remain unclear. I will list a few here:
    - Line 127-138 are helpful for understanding but they only appeared in the method section. I suggest the author can elaborate the problem and objective further in the introduction. 
    - Motivation of applying causal invariance in LLM debiasing is unclear.
    - It would be better if in Sec. 3 or before, a complete example in LLMs where the introduced variable can have some correspondence. I can only find such correspondence in Sec. 5.1.
    - Notation abusing makes some concepts confusing: e.g., what does "a.s." & "d" over = mean? in distribution?;  
    - typos: e.g., "predictins" in line 215
- Why does OOC perform less significant on LLAMA-3-70B? Is it because the possible obfuscation instructions are of low quality?
- To measure if your method successfully remove/reduce the bias, except for the stratified invariance you introduced, are there other bias evaluation metrics that you can use to demonstrate?

### Questions
See questions above.

### Soundness
3

### Presentation
2

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
The paper considers the test-time evaluation of fairness for LLMs. In particular, the paper aims to address the potential issue of naive applications of certain causal debiasing strategies (e.g., in terms of counterfactual data augmentations operating on the individual level), and proposes Stratified Invariance (Definition 1). The idea is to incorporate additional measurements at test time, so that stratified predictors can be constructed (for bias evaluation purposes). Prompting template and empirical results are presented.

### Strengths
The strength of the paper comes from the clear presentation of the potential issue of directly applying certain causal fairness notions (especially ones that are related to counterfactual invariance) in the LLM context (Section 2), and the attempt to address this issue by proposing stratified invariance (Definition 1), which is a reasonable middle ground between the almost-sure-equality between potential outcomes (counterfactual invariance) and the distribution-level equality (referred to as intervention invariance in the paper). The theoretical presentation (Section 2) is relatively clear and not hard to follow, the OOC prompting design (Section 3) is guided by the theoretical analysis, and the empirical evaluations include how OOC improve stratified invariance, as well as how to approach counterfactual invariance through stratifications.

### Weaknesses
The paper can be improved by (1) considering recent LLM debiasing strategies that do not specifically "rely on model's implicit understanding of bias" (lines 47 -- 49), so that the addressing of the existing LLM literature can be more comprehensive; (2) including discussion on the inference overhead of the proposed pipeline.

(1) recent LLM debiasing strategies that do not specifically rely on model's implicit understanding of bias

The paper presents criticisms of the existing LLM debiasing strategies in terms of the reliance on model's implicit understanding of bias (lines 47 -- 49). However, this might not be the case for recent LLM debiasing strategies. For instance, Li et al. (2024) consider a possible causal modeling of how LLM decisions are modulated by prompts, and proposed prompting-based strategies to encourage fact-based reasoning where no social category (e.g., gender, race) appears.  These strategies do not rely on model's understanding of bias. Considering them would help make the addressing of existing very relevant literature more comprehensive.

(2) discussion on the inference overhead of the proposed pipeline

Since the proposed approach (OOC prompting) involves multiple inference followed by a majority vote, it seems that the inference cost goes up pretty quickly. It would be important to discuss the relation between the inference overhead introduced by the proposed approach and the effectiveness of debiasing.

#### Reference

Li, J., Tang, Z., Liu, X., Spirtes, P., Zhang, K., Leqi, L., & Liu, Y. (2024). Steering LLMs Towards Unbiased Responses: A Causality-Guided Debiasing Framework. arXiv preprint arXiv:2403.08743.

### Questions
As detailed in comments in the section above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposed stratified invariance, a stratified notion of debiasing, to capture a range of debiasing requirements from population level to individual level through an additional measurement that stratifies the predictions. The authors further propose Out-Of-Context (OOC) prompting, a zero-shot method that simulates stratified counterfactual data augmentations in LLM predictions.

### Strengths
The theoretical development of Stratified Invariance and Stratified Data Augmentation is interesting. Also, by experimenting on both synthetic and real-world datasets, the authors demonstrate the advantage of the proposed prompting strategy to boost stratified invariance in LLM predictions at test time.

### Weaknesses
While the paper’s introduction of "stratified invariance" is an interesting measure of fairness, it appears conceptually close to existing techniques in fair representation learning and causal fairness (e.g., statistical parity). It would be good if the authors could provide an in-depth discussion with other fairness metrics or write out the equations for comparison if this measurement is claimed as a novelty. It is also worth noting that the proposed metric and/or prompting strategy only works for decision tasks (i.e., the output is a discrete answer/decision). The paper also lacks discussions with other debiasing methods, particularly those leveraging causal inference, such as the causality-guided debiasing framework proposed by Li et al. (2024) and Si et al. (2023). The OOC prompting strategy is also very similar to the FACT strategy in Li et al. (2024): i.e., the transformation of the examples in Figure 1 is very similar to Figure 1 in  Li et al. (2024). It would also be nice to conduct comparative analyses with these existing prompting approaches to contextualize the contributions and highlight relative advantages. I am willing to increase my score if the authors could address these concerns.

- Li, Jingling, et al. "Steering LLMs Towards Unbiased Responses: A Causality-Guided Debiasing Framework." arXiv preprint arXiv:2403.08743 (2024).
- Si, Chenglei, et al. "Prompting gpt-3 to be reliable." arXiv preprint arXiv:2210.09150 (2022).

### Questions
1. Could the authors provide concrete examples to better illustrate Definition 2 and Definition 3? Specific examples could clarify these definitions and their practical implications, helping readers understand the core concepts more intuitively. A worked example applying these definitions to a simple fairness scenario would help illustrate how they capture different aspects of fairness.

2. Could the authors offer intuitive justifications for Lemma 1 and Theorem 1? An explanation or reasoning beyond formal proofs would help make these results more accessible and easier to interpret.

3. Could the authors conduct comparative analyses with existing prompting approaches to contextualize the contributions and highlight relative advantages (see weakness section)?

4. Could the authors provide an in-depth discussion on how this work aligns with or differs from other fairness metrics?

### Soundness
2

### Presentation
1

### Contribution
2
