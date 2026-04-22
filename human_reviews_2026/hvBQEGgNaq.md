# The Last Token is Enough: Lightweight Contrastive Decoding for Mitigating Hallucinations in Large Vision-Language Models

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Contrastive decoding is a mainstream approach for mitigating hallucinations, which actively induces hallucinations and compares them with the original output to calibrate the logits of the next token, thereby enhancing the response reliability of Large Vision-Language Models (LVLMs). Although contrastive decoding has shown promising effectiveness in suppressing hallucinations, the process of inducing hallucinations inevitably incurs additional computational overhead and considerable inference latency, an issue largely overlooked in existing studies.  To this end, we present LightCD, a lightweight and efficient contrastive decoding method that achieves better hallucination mitigation while maintaining high computational efficiency. Specifically, we observe that LVLMs tend to aggregate learned information into a small number of tokens, which play a critical role in supporting the final output. The last token, used for next-token prediction, leverages the attention mechanism to “see” and summarize this information in order to produce the final result. 
In light of this, we revisit the process of inducing hallucinations in the current contrastive decoding paradigm and propose processing solely on the last token in the final layer of LVLMs to replace full-sequence inference. LightCD introduces two key modules: Selective Attention Perturbation, which identifies and filters critical information from attention heads containing factual evidence; and a Residual-free Mechanism, which suppresses cross-layer information propagation to generate desired hallucinations.
Finally, the model completes contrastive decoding by subtracting the hallucination logits from the original token logits. Extensive experiments demonstrate that LightCD significantly outperforms existing methods in mitigating hallucinations while achieving a 2$\times$ increase in inference speed.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LightCD (Lightweight Contrastive Decoding), an efficient method to reduce hallucinations in large vision-language models by operating only on the last token, which summarizes most contextual information. It replaces full-sequence contrastive generation process with two lightweight modules: Selective Attention Perturbation to induce controlled hallucinations and a Residual-Free Mechanism to suppress cross-layer information. Experiments on LLaVA-1.5 and InstructBLIP show that LightCD achieves comparable or better hallucination mitigation than prior methods while reducing inference time, demonstrating its advantages.

### Strengths
1. Excessive computational overhead is a major limitation of contrastive decoding; therefore, exploring more efficient approaches in this direction is a valuable research effort.
2. The proposed method is well-motivated and reasonable, and the experimental results demonstrate its advantages in both effectiveness and efficiency.

### Weaknesses
1. The authors only conducted experiments on LLaVA-1.5 and InstructBLIP, which are relatively old and outdated models. It is recommended to include experiments on more advanced LVLMs, such as Qwen3-VL.
2. This work focuses solely on multimodal large language models. However, the proposed method does not appear to incorporate designs specifically tailored to the multimodal setting. Therefore, it should, in principle, be directly applicable to text-only LLMs as well. It is thus recommended to include experiments on text-based LLMs and text hallucination benchmarks to further validate the method’s generalization and versatility.
3. I found another work [a] that also focuses on addressing the high computational cost of contrastive decoding. The authors are encouraged to include a discussion and comparison of effectiveness and efficiency of their method with that work.
4. The performance of the proposed method fluctuates noticeably with different hyperparameter settings. For example, the accuracy drops by more than 3 points when the number of selected heads decreases from 28 to 16. I consider this sensitivity to be a limitation of the method.

Minor:
1. While the overall writing of the paper is fluent and easy to follow, some parts feel somewhat redundant. For example, the second paragraph of Section 4.1 repeats points that have already been sufficiently discussed in the previous section, and thus could be removed for a more concise presentation.

I will consider increasing my score if all issues are resolved.

[a] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models, ICCV 2025

### Questions
1. Are the optimal settings of k and $\lambda$ consistent across different LLMs (e.g., Qwen and LLaVA)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Though contrastive decoding has proven effective in mitigating hallucinations, it requires additional forward passes to obtain hallucinated logits, which increases computational cost. This paper proposes a framework called LightCD, which obtains hallucinated logits by modifying the attention weights of the last token in the final layer, thereby enhancing the efficiency. This method is motivated by the authors' observation that the last token in the final layer primarily attends to a specific subset of tokens.

### Strengths
1. The observation presented in this paper is interesting — the last token in the final layer often attends to a specific subset of tokens.
2. The research question is meaningful. Proposing a new contrastive decoding paradigm that avoids double inference is a reasonable and valuable direction.

### Weaknesses
1. The writing format should be improved. 
    - An abbreviation should only be used after its full form has been introduced. For example, in line 16, the term “LVLMs” appears without its full form being defined.
    - In line 38, the phrase `The emergence of vision-language models (LVLMs)` should be revised to `The emergence of large vision-language models (LVLMs)` to ensure consistency with the abbreviation **L**VLMs.
    - In line 392, the heading **Results on MME** should be placed on a new line for better formatting and readability.
    - Typo: In line 19, `mitigate` should be revised as `mitigates`.
    - In line 148, `While these approaches help suppress hallucinations, they also increased inference cost and latency`， the tense usage in this sentence is inconsistent. `increased`-->`increase`.
    - In line 167, the paper lacks a description of `H`. Although it is obviously the number of heads, it should be explicitly stated for clarity.
    - In line 172, it should be `Where $y_t$ denotes the token at step t`, instead of `... at t step`. The suggested version is also consistent with the description in line 173.
    - In Equation (4), it would be better to add a multiplication dot between $\alpha$ and $logit$, e.g., $\alpha$$\cdot$$logit_\theta$, for clearer mathematical expression.
    - In Table 1, it would be better to keep all numerical values with the same number of decimal places (e.g., two digits after the decimal point). For instance, “76.9” should be formatted as “76.90” for a more consistent presentation.
    - In line 356, the authors claim that one of the baselines is denoted as `“Regular”`. However, in Table 1, this baseline appears as `“sample”`. I assume the authors mistakenly wrote `“sample”` instead of `“Regular”`.
    - Caption formatting is inconsistent: except for Figure 4, all other figure and table captions end with a period. The caption of Figure 4 should also terminate with a period to maintain a consistent presentation.
2. There is some redundancy in the Introduction section. In the paragraph starting at line 82, the authors repeatedly mention that the extra inference time mainly arises from the additional inference required for contrastive decoding. This part could be better organized to improve clarity and conciseness.
3. The logic of the Introduction and the Method is not very clear. From the observed phenomenon to the proposed method, there should be a clearer logical transition that explains how the identified problem naturally motivates the proposed solution.
4. Figure 4 should be revised. The caption and numbers are too small.

### Questions
1. In line 296, the authors state that `“a lower entropy indicates a more concentrated attention distribution, suggesting that the head is more focused and thus more likely to capture critical information.”` However, it is unclear why a more concentrated (lower-entropy) attention distribution necessarily implies that the head captures more critical or meaningful information. Could the authors provide theoretical justification or empirical evidence to support this assumption?
2. In line 329, the authors state `However, during hallucination generation, such cross-layer information propagation suppresses the effectiveness of SAP and substantially degrades the quality of the generated hallucinations` Could the authors elaborate on why cross-layer information propagation would weaken SAP’s effectiveness and provide supporting analysis or experimental results to substantiate this statement?
3. Following the aforementioned question, the authors obtain the hallucinated logits without the residual connection. How can the effectiveness of these hallucinated logits be demonstrated to result from the proposed Attention Head Selection and Top-$\gamma$ Token Filtering, rather than from the disruption of normal information flow caused by removing the residual connection?
4. The datasets used in this paper are somewhat limited. Recent strong works typically evaluate on more diverse benchmarks rather than only these two datasets — for example, CHAIR, MMMU, and HallusionBench. Could the authors provide more numerical results on other hallucination benchmarks?

I am happy to increase my score if the weaknesses and questions are addressed.

### Soundness
3

### Presentation
1

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
The paper introduces LightCD, an improved Contrastive Decoding method designed to reduce inference latency. LightCD achieves this by identifying that hallucination induction can be efficiently performed by focusing only on the last token of the final Transformer layer. Specifically, it employs a Selective Attention Perturbation (SAP) module to suppress the most informative attention heads (those with lowest entropy) for the last token, and a Residual-Free Mechanism to prevent fact leakage, thereby efficiently generating "hallucination logits" in a single pass. The method requires only one complete forward pass, thus improving the computational efficiency of the CD process. Experimental results demonstrate improved accuracy on two specific hallucination benchmarks and clear efficiency gains.

### Strengths
Clear Motivation: The paper is well-motivated, clearly articulating the issue of computational inefficiency in existing CD methods. The The idea that the core information required for contrastive decoding resides primarily in the last token of the final layer is straightforward  and clear presented. The integration of SAP (targeting critical attention heads) and the Residual-Free Mechanism (preventing factual residual connections) offers a heuristic, lightweight solution.

Significance in Efficiency: Achieving the effect of contrastive decoding with only a single forward pass represents a substantial practical improvement for deploying CD in real-world applications, offering considerable computational cost savings.

### Weaknesses
Major Concerns

1. Primarily focus on the CD paradigm. The related work section and baseline comparisons are restricted almost exclusively to CD methods. To establish broader significance, the work should be contextualized against other major categories of inference-time/post-hoc hallucination mitigation techniques, such as Uncertainty Calibration and Rejection, Prompt Guardrails, etc. The current presentation gives the impression of limited novelty as an incremental advancement on CD.

2. Insufficient Experimental Validation.

- Experiments should be expanded to include more recent and diverse LVLM architectures with varying parameter sizes to confirm the generalization ability of LightCD.

- The comparison should be extended to include other non-CD inference-time/post-hoc baselines (e.g., prompting-based methods or uncertainty methods) to fully showcase LightCD's performance and efficiency.

- More results are claimed to include in the appendix. Nevertheless, the experiments in appendix lack proper comparisons with existing baselines.

- The manuscript appears to be rushed and lacks several necessary details, impacting clarity:
a. Authors introduce the two core modules (SAP and Residual-Free Mechanism) but fails to discuss how to use these modules to implement contrastive decoding (implicitly introduced in Section 3 as preliminaries (4) but lack of explicit description of the complete proposed method in Section 4). 
b. Section 5.1 (Datasets and Baselines) is incomplete, as the specific baseline methods used for comparison are not properly introduced here, appearing abruptly in later tables (finally found in Appendix).
c. The specific details regarding experimental settings are not clearly articulated in experiment section.

Minor Concerns

- Figures 2 and 4 suffer from excessively small font sizes, making them difficult to read. While the empirical study on the attention map (used for motivation) is insightful, the visual representation is not clear enough to strongly support the core design principle.

- Figure 4 lacks clear units or labels for the quantitative results being presented (e.g., what does the count represent?). Figure 6 is missing both axis titles and units, making interpretation impossible.

### Questions
No more specific question.

### Soundness
3

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
This paper introduces a lightweight and efficient contrastive decoding method aimed at mitigating hallucinations in LVLMs while reducing computational overhead and inference latency. Hallucinations in LVLMs refer to factually incorrect or irrelevant responses that deviate from the visual inputs. Traditional contrastive decoding methods have been effective in reducing hallucinations but often introduce significant computational costs and latency. LightCD not only mitigates hallucinations effectively but also achieves up to a 2x increase in inference speed.

### Strengths
1.	LightCD focuses only on the last token in the final layer of the LVLM during inference, instead of processing the entire sequence. This reduces the need for full-sequence inference, thereby improving efficiency.
2.	The introduction of the SAP and residual-free Mechanism modules provides interpretability to the hallucination mitigation process. Instead of treating decoding as a black box, the paper identifies which attention heads contain factual versus hallucinatory evidence and explicitly controls their contributions. This mechanistic transparency enhances both the scientific rigor and explainability of the proposed approach.

### Weaknesses
The Residual-Free Mechanism is introduced as a way to suppress cross-layer information propagation to enhance hallucination generation. However, this mechanism could potentially degrade the overall performance of the model for tasks requiring rich, multi-layered integration of knowledge. The impact of removing residual connections on non-hallucination tasks, e.g., factual reasoning, remains underexplored. An ablation study that isolates the effect of this mechanism on various tasks beyond hallucination mitigation would strengthen the method’s versatility claims.

### Questions
1. SAP focuses on perturbing the most relevant attention heads based on entropy measures. How this approach handles cases where the attention distribution is more evenly spread across different heads or when some heads are weakly relevant to the final output? How does SAP adapt in these situations, and does it always identify the most critical attention heads, or could there be cases where the perturbation might unintentionally suppress important information from other heads? 
2. Does LightCD ever result in over-suppressing information, leading to a loss in the richness or diversity of the generated content? In tasks requiring creative or diverse outputs, could excessive mitigation of hallucinations reduce the model's ability to generate varied or novel responses?

### Soundness
3

### Presentation
3

### Contribution
2
