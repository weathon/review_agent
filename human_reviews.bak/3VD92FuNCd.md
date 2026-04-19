# Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs

- Decision: Reject
- Scores: 8, 6, 3, 6

## Abstract
Past work has studied the effects of fine-tuning on large language models' (LLMs) overall performance on certain tasks. 
However, a way to quantitatively and systematically analyze its effect on individual outputs is still lacking.
In this work, we propose a new method for measuring the contribution that fine-tuning makes to individual LLM responses, assuming access to the original pre-trained model. 
We introduce and theoretically analyze an exact decomposition of any fine-tuned LLM into a pre-training component and a fine-tuning component.
Empirically, we find that one can steer model behavior and performance by up- or down-scaling the fine-tuning component during the forward pass.
Motivated by this finding and our theoretical analysis, we define the Tuning Contribution ($\mathrm{TuCo}$) in terms of the ratio of the magnitudes fine-tuning component and the pre-training component.
We find that three prominent adversarial attacks on LLMs circumvent safety measures in a way that reduces the Tuning Contribution, and that $\mathrm{TuCo}$ is consistently lower on prompts where the attacks succeed compared to ones where they don't. 
This suggests that attenuating the effect of fine-tuning on model outputs plays a role in the success of these attacks.
In summary, $\mathrm{TuCo}$ enables the quantitative study of how fine-tuning influences model behavior and safety, and vice versa.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on quantitatively analyzing the effect of fine-tuning on individual outputs of large language models (LLMs). To be specifical, this work introduces a decomposition of a fine-tuned model into a pre-training component and a fine-tuning component, through which it presents that the model behavior can be steered by up-/down-scaling the fine-tuning component during the forward pass. Based on that, this work proposes Tuning Contribution (TuCo) in terms of the ratio of magnitudes and investigate its utility on adversarial attacks for LLMs. Both empirical and theoretical results are provided to demonstrate the rationality of the proposed TuCo and provide in-depth insights into a quantitative study of how fine-tuning influences model behaviors.

### Strengths
1. This paper focuses on quantitatively investigating the effects of fine-tuning in individual prompts, which is, at least from the perspective of the reviewer, a new and novel research problem, and provides insights on understanding the model behavior and performance from a systematic concept framework.
2. This work provides a decomposition of a fine-tuned LLM as an embedding superposition of a pre-training component and a fine-tuning component, leveraging the residual architecture of Transformer LLM. It is reasonable and extendable for further analysis of model behavior understanding.
3. In general, the illustration is clear and provides an intuitive explanation of the decomposed two-component and the analytic framework, and the computation of Pre-prompt tuning contribution is also easy to understand.
4. Both theoretical analyses based on the generalized decomposition and the empirical results with jailbreak attack are provided to demonstrate the effectiveness of TuCo, and provide some further insights on understanding model behavior and for the safety of LLMs.

### Weaknesses
1. Although this work provides the canonical decomposition of a fine-tuned model with the theoretical results based on the gronwall bound, the current version provides limited implications behind the derived proposition, making it hard to understand the significance of the analytical results, and draw further insights on the analysis.
2. The computational cost of TuCo is not considered in experiments, and it would be better if the current version could incorporate another detection method to have an empirical comparison with TuCo for detection tasks, which can provide more convincing results on the effectiveness of TuCo.
3. I do not very understand why the decomposition can be regarded as exact decomposition, and is there any gap between the idealized setting stated at section 4.2 for the motivation? as the authors state it is informally motivated.

### Questions
1. Could the author explain or discuss more about the theoretical implications behind the proposition results?
2. Could the author also analyze the computational cost of TuCo and discuss why you only consider the magnitude of the fine-tuning component on the last token's hidden state as represented by the function $proj_n(\cdot)$?
3. please refer to the third point in the weakness.

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
The paper aims to quantifying the impact of fine-tuning on individual outputs of LLMs. The authors propose TuCo, a metric designed to measure the contribution of fine-tuning to an LLM's output for any given prompt. The key idea is to decompose the fine-tuned model's representations into two components: a) PTC: The output of the corresponding layer in the pre-trained model, and b) FTC: The difference between the outputs of the fine-tuned model and the pre-trained model at each layer. The paper provides a theoretical foundation for this decomposition and demonstrates that the relative magnitudes of the PTC and FTC can bound the discrepancy between the outputs of the pre-trained and fine-tuned models. Empirically, the authors validate their approach by: a) Scaling the FTC: Showing that adjusting the magnitude of the FTC can steer the model's behavior and performance on specific tasks. b) Analyzing Adversarial Attacks: Investigating how three types of jailbreak attacks affect TuCo. The findings suggest that these attacks reduce the TuCo, meaning they attenuate the effect of fine-tuning and exploit the model's pre-training behaviors to circumvent safety measures.

### Strengths
I believe that this paper has its own contribution. While the basic idea is simple, the authors show that it can truly reveal the behaviours of models, making it a very useful tool in understanding the consequences of fine-tuning in practice. The authors also provide some theoretical analysis to support their claim, further solidifying their findings. Moreover, the experimental analysis looks sound to me, and the results quite align with my intuitions.

### Weaknesses
1. I wonder if additional discussion about the difference between TuCo and robust fine-tuning (https://arxiv.org/abs/2109.01903) / task vectors (https://arxiv.org/abs/2212.04089) would be beneficial. It seems that the difference is that previous works typically attenuate the effects of fine-tuning by parameter scaling, while your work employs output scaling, especially for the section 5.1 - 5.2.

2. The authors mainly focus on the quantitive analysis in the main body of the paper. Considering that many of the adopted metrics for LLMs can be misleading, is it possible the authors further provide some qualitative analysis for the results, especially echoing Figs 3-4. For example, what the model output changes across different values of alpha. Is it possible that the improper choices of alpha will make model output random characters or nonsensical strings? 

3. Intuitively, I think the paper may have some interesting contributions to the community beyond the mentioned ones in the main content and conclusion. I wonder if the authors could discuss more about the potential usages and applications of TuCo in practice. 

4. I also found a small typo in the section page: Perez et al (2022) should changed to (Perez et al 2022)

### Questions
I appreciate the contribution of this paper, and I only have some minor questions mentioned in the box of Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This work studies how fine-tuning LLMs contributes to the individual response. The authors propose a decomposition of post-trained LLM into a pre-training component and fine-tuning component and define a Tuning Contribution of these two components. Empirical evaluation shows that TuCo is sensitive to language model inputs.

### Strengths
1. The interpretation of models remains a persistent and significant challenge in the field of deep learning.

2. Fine-tuning LLMs has become a prevalent practice. Elucidating the mechanisms of LLM fine-tuning could potentially enhance this process, thereby contributing to the broader understanding and application of these sophisticated models.

### Weaknesses
1. Overall the work is ad-hoc.
This study introduces and quantifies several metrics across diverse contexts. However, it appears to lack novel insights into LLM fine-tuning or practical guidelines. For the observed disparities in model outputs across various inputs (for example, among different languages, or harmful prompts with and without adversarial strings), because the outputs are different in those settings, it is not hard to define quantities that distinguish them. In addition, while Proposition 4.5 establishes a theoretical bound on these metrics, its practical application or utility within the study remains unclear.


2. The paper is not well-written. The study presents multiple definitions and evaluation frameworks; however, the organization appears arbitrary, lacking a cohesive and succinct narrative. Moreover, the introduction of a novel metric within the evaluation section deviates from conventional structure, potentially compromising the clarity and flow of the presented research.

### Questions
N/A

### Soundness
2

### Presentation
2

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
The paper presents a novel measurement of the relative contribution of fine-tuning on a sample derived from the difference between the effects of the pretrained model and the full pretrained model in each layer, and it shows that this metric can be used to identify jailbreaks and that intervening on it can be used to steer model behavior.

### Strengths
- The authors present a novel metric, TuCo, for identifying the relative contribution of fine-tuning on a given sample.
- The authors present evidence that TuCo is a useful tool in the analysis of model behavior. In particular, jailbreaks tend to decrease the contribution of fine-tuning as measured by TuCo, which obtains strong results in terms of discriminating between jailbroken and unmodified prompts.

### Weaknesses
- The gap between the formulation of Prop 4.5 and the definition of TuCo is not adequately explained: many alternative formulations are possible. In particular, it should be made clear why the proposed formulation is the right one.
- Simpler baselines are not considered:
    - For example, a simpler approach might take only differences between the final hidden states of the two models into account.
    - Such an output-only definition is equivalent to a variation of TuCo which takes the compositional structure of the pretrained model into account.
- Along these lines, it is unclear why it is better to view the differences between layers l of the two models in isolation, ignoring the compositional effect of the deviation between the two models. 
- While it suffices to represent the decomposition into PTC and FTC, it is unclear that the notation presented in 4.2 and 4.3 is a natural way to represent the decomposition of a model into circuits. In particular, the notation hides the compositional structure of the circuits in $C_1$ and necessitates that when taking composition into account, the circuits are no longer disjoint.
- Proposition C.1 (iii) appears to be incorrect: the proof claims that the equation on line 1002 holds for arbitrary disjoint $C_1$ and $C_2$. This appears to instead be a required assumption. For a trivial counterexample, consider scaling the components in $C_1$ by a constant factor and subtracting the difference from those of $C_2$.

### Questions
- Does a formulation which aligns more closely with Prop 4.5 have worse empirical performance?
- Why is PTC defined as the sum of $PTC(x^{FT}_s, s)$ rather than $PTC(x^{PT}_s, s)$?

Comments:
- If possible, the typesetting of Proposition 4.5 should be improved.
- On line 323, $PreCo(x)$ should be defined as $1 - TuCo(x)$

### Soundness
3

### Presentation
2

### Contribution
3
