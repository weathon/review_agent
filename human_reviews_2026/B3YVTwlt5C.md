# Towards Effective MLLM Jailbreaking Through Balanced On-Topicness and OOD-Intensity

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Multimodal large language models (MLLMs) are widely used in vision-language reasoning tasks. However, their vulnerability to adversarial prompts remains a serious concern, as safety mechanisms often fail to prevent the generation of harmful outputs. Although recent jailbreak strategies report high success rates, many responses classified as ``successful'' are actually benign, vague, or unrelated to the intended malicious goal. This mismatch suggests that current evaluation standards may overestimate the effectiveness of such attacks. To address this issue, we introduce a four-axis evaluation framework that considers input on-topicness, input out-of-distribution (OOD) intensity, output harmfulness, and output refusal rate. This framework identifies truly effective jailbreaks. In a substantial empirical study, we reveal a structural trade-off: highly on-topic prompts are frequently blocked by safety filters, whereas those overly OOD often evade detection but fail to produce harmful content. By contrast, prompts that balance relevance and novelty are more likely to evade filters and trigger dangerous outputs. Building on this insight, we present a recursive rewriting strategy called Balanced Structural Decomposition (BSD). The approach restructures malicious prompts into semantically aligned sub-tasks, while introducing subtle OOD signals and visual cues that make the inputs harder to detect. BSD was tested across 15 commercial and open-source MLLMs, where it consistently led to higher attack success rates, more harmful outputs, and fewer refusals. Compared to previous methods, it improves success rates by 67\% and harmfulness by 21\%, revealing a previously underappreciated weakness in current multimodal safety systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper focuses on addressing the vulnerability of Multimodal Large Language Models (MLLMs) to adversarial prompts and the inaccuracies in existing jailbreak evaluation standards. It points out that current jailbreak strategies often overestimate success rates, as many "successful" responses are benign, vague, or unrelated to malicious goals.
To solve this, the paper proposes a four-axis evaluation framework covering input on-topicness, input out-of-distribution (OOD) intensity, output harmfulness, and output refusal rate. Through empirical research, it reveals a structural trade-off: highly on-topic prompts are easily blocked by safety filters, while overly OOD prompts evade detection but fail to generate harmful content.
Based on this insight, the authors design a recursive rewriting strategy called Balanced Structural Decomposition (BSD). BSD decomposes malicious prompts into semantically consistent sub-tasks, adds subtle OOD signals and visual cues, and uses a neutral tone to present inputs. Experiments on 13 commercial and open-source MLLMs show that BSD outperforms existing methods, improving attack success rates by 67% and harmfulness by 21%, and also performs well against guard models.

### Strengths
- The proposed four-axis evaluation framework comprehensively captures both input and output characteristics of MLLM jailbreaks, addressing the defect of overestimating attack effectiveness in traditional binary evaluation and providing a more accurate and reliable benchmark tool for subsequent research.
- The BSD strategy innovatively balances the trade-off between on-topicness and OOD intensity. By recursively decomposing prompts and integrating visual cues, it effectively evades MLLM safety filters while ensuring the generation of harmful content, achieving breakthroughs in attack performance.
- The study conducts extensive experiments across 13 MLLMs (including commercial closed-source and open-source models) and two guard models, using three datasets (HADES, MMSafetyBench, and AdvBench-M). The large-scale and multi-dimensional experimental design enhances the generalizability and persuasiveness of the research results.

### Weaknesses
- The BSD strategy relies heavily on the quality of sub-task decomposition by the Qwen2.5-7B model. When facing overly obvious or complex malicious objectives, the decomposition fails to produce semantically diverse sub-tasks, leading to reduced jailbreak success rates and limiting the strategy's adaptability.
- The generation of descriptive images in BSD depends on the FLUX.1-schnell text-to-image model. The paper lacks an in-depth analysis of how image quality, style consistency, and semantic alignment with sub-tasks specifically affect jailbreak results, and there is insufficient verification of the necessity of visual cues.
- The study only evaluates the short-term effectiveness of jailbreak attacks but ignores the long-term impact of repeated use of BSD on MLLMs. It does not explore whether MLLMs can learn to identify and defend against such structured decomposition attacks, resulting in incomplete research on attack durability.
- This paper needs to add more advanced baselines for further comparison to highlight the superiority of the proposed attack.

### Questions
Comments:

1. Over-reliance on specific decomposition models without alternative mechanisms: The BSD strategy is highly dependent on Qwen2.5-7B for sub-task decomposition. When this model fails to decompose complex or obviously malicious prompts effectively, the entire jailbreak process breaks down. The paper does not propose alternative decomposition models or adaptive adjustment mechanisms to address this single-point failure risk, which affects the robustness of the strategy.

2. Insufficient validation of the role of visual cues: Although BSD integrates descriptive and distraction images, the paper only verifies the performance differences between FLUX-generated images, colored boxes, and random noise in ablation experiments. It does not quantitatively analyze how factors such as image semantic relevance to sub-tasks, visual complexity, and number of distraction images affect the jailbreak effect, making the role of visual cues in the strategy unclear.

3. Limited analysis of cross-dataset generalization differences: The paper tests BSD on three datasets (HADES, MMSafetyBench, AdvBench-M) but does not deeply analyze why the strategy has significant performance differences across datasets (e.g., lower ASR on AdvBench-M due to fewer samples). It also fails to explore whether the decomposition logic of BSD needs to be adjusted for different types of malicious prompts in different datasets, limiting the understanding of the strategy's cross-scenario adaptability.

4. Lack of in-depth comparison with state-of-the-art baselines: Although this paper has conducted in-depth comparisons with CS-DJ and FigStep, it is still not convincing enough. I recommend that the authors compare with the methods in the following references:

[1] Ma, Teng, et al. "Heuristic-induced multimodal risk distribution jailbreak attack for multimodal large language models." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

[2] Liu, Yi, et al. "Arondight: Red teaming large vision language models with auto-generated multi-modal jailbreak prompts." Proceedings of the 32nd ACM International Conference on Multimedia. 2024.

[3] Jeong, Joonhyun, et al. "Playing the fool: Jailbreaking llms and multimodal llms with out-of-distribution strategy." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

5. Minor issues
- The font size of the text in Figures 1, 8, 9 is too small to be read.
- "Stage 1: Width-first balancing via Width Balance Score." should be "Stage 1: Width-first balancing via width balance score."

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates a practical limitation of existing OOD jailbreak attacks where the responses classified as "successfully jailbroken" are frequently unrelated to the malicious intent while the on-topic attack prompts only elicit response rejection.
To effectively jailbreak MLLMs, the authors proposed balancing this trade-off between OOD-ness and On-topicness by iterative decomposition of attack prompts and depth refinement within a tree structure.

### Strengths
- The paper is tackling an unexplored but practical limitation of existing OOD attacks with qualitative analysis of relevance-novelty trade-off.
- The paper proposed a novel tree-based decomposition attack strategy to balance the trade-off.
- The paper is well-written and easy to follow.
- The comprehensive experimental results over closed and open-source models demonstrate the jailbreak effectiveness of proposed method with detailed qualitative analysis.

### Weaknesses
- One of my concerns is in the paper’s reliance on SBERT embeddings to quantify On-Topicness (OT) and Out-of-Distribution Intensity (OI). SBERT has known limitations in capturing subtle semantic variations—such as word order perturbations or nuanced paraphrases [R1] —which may undermine the reliability of these metrics. For instance, in Eq (2), when the short caption from MLLM is subtly but semantically altered where only minor lexical changes lead to a substantially different meaning—it is unclear whether the OI metric based on SBERT embeddings can reliably capture such distinctions.
- Also, when measuring OI in Eq (2), a safety-aligned MLLM may generate a safe summary rather than one semantically consistent with the harmful prompt $P_0$. This naturally increases the OI score even though the divergence stems from safety alignment rather than genuine distributional novelty. Consequently, the metric may conflate the model's safety-alignment with true out-of-distribution characteristics, limiting its interpretability as an indicator of OOD intensity.
- In Eq. (3), the formulation of the Harmfulness Score (HS) may produce misleading results. If the reference vector $h_{ref}$ ​contains uniformly high category scores while the response vector $h_r$ ​ is uniformly low, the ℓ1 ​distance term​ still increases—thereby raising the overall HS despite the response being less harmful.
- In Eq. (4), the notation $N$ is used without a clear definition. While it appears to represent the total number of evaluated responses, this seems not explicitly stated in the manuscript.
- In Eq. (5), the assumption that greater semantic dissimilarity among decomposed sub-tasks directly corresponds to higher OOD characteristics lacks clear justification. It is not evident that mutual dissimilarity between sub-tasks translates to genuine OOD behavior from the model’s perspective.
- The experimental comparison is limited to two baselines (FigStep, CS-DJ), which is too narrow to support broad claims. Including a broader set of recent baselines (also with recent victim models such as GPT-5) would make the results more convincing.
- The iterative decomposition (width) and refinement (depth) and image generation likely add significant computational overhead compared to the non-optimization methods such as FigStep.
[R1] https://arxiv.org/pdf/2309.03747

### Questions
See above weaknesses.

### Soundness
3

### Presentation
2

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
This paper addresses the challenge of effectively jailbreaking Multimodal Large Language Models (MLLMs). It argues that current evaluation methods are flawed, as they often misclassify benign or off-topic responses as successful attacks. To rectify this, the authors propose a new four-axis evaluation framework that assesses prompts based on On-Topicness (OT), Out-of-Distribution Intensity (OI), Harmfulness, and Refusal Rate. Through empirical analysis, the paper identifies a critical structural trade-off: highly on-topic prompts are more harmful but also more likely to be rejected, while extreme out-of-distribution (OOD) prompts evade filters but produce less harmful content. To exploit the optimal balance between relevance and novelty, the authors introduce Balanced Structural Decomposition (BSD), a recursive strategy that decomposes malicious instructions into semantically coherent sub-tasks paired with descriptive images. Evaluated on 13 MLLMs, BSD demonstrates superior performance by generating more harmful outputs with fewer refusals compared to baseline methods, highlighting a vulnerability in existing safety mechanisms that rely on surface-level filtering.

### Strengths
+ The introduction of the four-axis framework (On-Topicness, OOD-Intensity, Harmfulness, Refusal Rate) provides a more nuanced and comprehensive standard for evaluating MLLM jailbreaks, addressing the overestimation problem in prior work.
+ The paper makes a good contribution by empirically identifying and formalizing the fundamental trade-off between prompt relevance and novelty, offering a clear explanation for the limitations of existing attack strategies.
+ The proposed BSD strategy is a well-motivated and systematic approach that directly targets the identified OT/OI trade-off, demonstrating state-of-the-art attack performance across a wide range of models.

### Weaknesses
- The authors' findings do not significantly differ from previous work. For example, one of the main findings of this paper, that existing jailbreaks are generally ineffective, is consistent with the conclusions of Nikolić et al. (ICML’25). However, this paper fails to provide sufficient new insights or highlight the differences from previous work.
- While BSD integrates descriptive and distracting images, it does not analyze specific visual features (such as content relevance) or perform ablation experiments to determine how this affects the OT/OI balance or model behavior, making the role of visual effects difficult to understand.
- More advanced defenses need to be added for further evaluation. This paper only evaluates GuardReasoner-VL-3B/7B, which does not seem convincing.

### Questions
Q1: Regarding the overlap between your findings and previous work (e.g., Nikolić et al., ICML'25) on the ineffectiveness of existing jailbreaks, could you elaborate on why the paper does not include a detailed comparative analysis of the evaluation frameworks, attack mechanisms, or core conclusions between your work and Nikolić et al.'s study?
[1] Nikolić, Kristina, et al. "The Jailbreak Tax: How Useful are Your Jailbreak Outputs?." In Proc. of ICML, 2025.

Q2: For the visual components integrated into BSD (descriptive and distraction images), since the paper does not analyze specific visual features (e.g., semantic relevance between descriptive images and the original malicious prompt), do you have plans to supplement experiments that quantify how variations in visual content relevance affect the On-Topicness (OT) and OOD Intensity (OI) balance of the input?

Q3: The paper lacks ablation experiments to isolate the impact of visual cues on BSD’s performance. For example, why did you not design an ablation group that removes descriptive images or distraction images entirely, and compare its ASR, HS, and RR with the full BSD model? This would help clarify whether visual components are necessary for achieving the OT/OI balance.

Q4: When evaluating BSD against guard models, the paper only tests GuardReasoner-VL-3B and GuardReasoner-VL-7B. Why did you not include other mainstream MLLM guard models (e.g., LLaVA-Guard, VILA-Guard, or commercial guard systems like OpenAI’s Content Policy Enforcement) in the evaluation to verify the generalizability of BSD’s ability to bypass defenses?

Q5: The paper states that BSD’s sub-task decomposition relies exclusively on Qwen2.5-7B, which fails when malicious prompts are overly explicit or complex (resulting in overt intent and model rejection). Given this limitation, why did the study not evaluate alternative decomposition LLMs (e.g., smaller open-source models like LLaVA-7B or safety-aligned models) to test whether they could improve the robustness of the decomposition module?

Q6: For overly complex malicious prompts (e.g., multi-step illicit operations) that Qwen2.5-7B fails to decompose effectively, the paper only notes "reduced jailbreak success" but does not define clear criteria for "complexity" of prompts. Could you clarify how the paper quantifies prompt complexity, and whether this quantification was used to systematically test the decomposition module’s limits?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a systematic study of multimodal jailbreaks and introduces a quantitative framework (on-topicness, OOD-intensity, harmfulness, refusal) along with the Balanced Structural Decomposition (BSD) attack. BSD adaptively decomposes malicious prompts to balance semantic relevance and distributional novelty, achieving improvements in attack success.

### Strengths
- This paper defines and separates on-topicness and OOD-intensity, enabling structured, interpretable analysis of jailbreak mechanisms in multimodal LLMs.

- This paper proposes BSD, a recursive decomposition framework that operationalizes this balance and achieves improved attack success rates.

### Weaknesses
- The contribution is primarily heuristic and engineering-focused. BSD refines existing decomposition-based attacks rather than introducing new theoretical insights into model vulnerability.

- The BSD framework mainly extends prior decomposition-based attacks with additional heuristics (WBS, SDU) but lacks causal or mechanistic justification for why balancing on-topicness and OOD-intensity fundamentally increases vulnerability.

- The hierarchical search and threshold settings introduce many heuristic hyperparameters without sensitivity or stability analysis, making the method’s reproducibility and robustness across models and datasets uncertain.

### Questions
- How sensitive are the results to the embedding model choice and the specific OT/OI computation?

- Can the authors demonstrate that BSD captures intrinsic safety vulnerabilities rather than benchmark-specific weaknesses?

- Would the observed OT–OI balance remain effective under adaptive or adversarially retrained guard models?

### Soundness
2

### Presentation
2

### Contribution
2
