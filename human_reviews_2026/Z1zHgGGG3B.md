# Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Large Vision-Language Models (LVLMs) have demonstrated remarkable multimodal capabilities, but they inherit the tendency to hallucinate from their underlying language models. While visual contrastive decoding has been proposed to mitigate this issue, existing methods often apply generic visual augmentations that disregard the specific context provided by the text query, limiting their effectiveness. This study introduces a novel training-free decoding strategy that addresses these limitations, featuring two key contributions. First, a self-augmentation prompting strategy that leverages the intrinsic knowledge of the model to dynamically align semantics between the query and the visual augmentation. Second, an adaptive thresholding algorithm that adaptively adjusts next token candidate size based on the output sparsity, utilizing full information from the logit distribution. Extensive experiments across four LVLMs and seven benchmarks demonstrate that the proposed decoding significantly enhances factual consistency compared to state-of-the-art decoding methods. This work highlights the importance of integrating query-dependent augmentation and entropy-aware decoding for improving effective generation of LVLMs. The source code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles hallucination in Large Vision-Language Models (LVLMs) by improving visual contrastive decoding (VCD). It introduces SAVCD: (1)a self-augmentation selection (SAS) prompt that asks the LVLM to pick a semantically-disruptive augmentation for the current query, and (2) Sparsity Adaptive Truncation (SAT) — a thresholding rule that uses the entropy of the expert logit distribution to set the plausibility cutoff. At each step, SAVCD contrasts the expert and amateur logits, then removes tokens below the SAT threshold before sampling. Experiments on multiple LVLM families and 7 benchmarks show consistent gains over prior VCD methods.

### Strengths
1. The technical soundness of SAS and SAT is solid. Authors have conducted quantitative experiments and intuitive explanations to demonstrate its effectiveness.
2. The experiments, though tested on old models, can be rated as solid.
3. The paper is well-written: clear and easy to follow. The figures are carefully polished and informative.

### Weaknesses
1. The contributions sound incremental, given that VCD has already been widely researched. SAS and SAT make sense intuitively and experimentally, but might not be enough contribution for a top tier conference. However, I am not an expert on LLM Decoding, so I will carefully refer to other reviewers' comments on novelty and contribution before making my final decision.
2. The SAS Augmentation Choice seems arbitrary to me, as "Noise, Horizontal Flip, Vertical Flip, Random Crop, Random Mask and Color Inversion" only presents a subset of the image's properties. For instance, when asked about size, reachability, OCR or spatial relationship, chances are that these Augmentation choices will not work.
3. The experimental results don't show robust generalization across datasets and models: for instance, SAVCD in Qwen-VL or POPE-MSCOCO consistently underperform other methods. 
4. The experiments are conducted on rather old models like LLaVA-1.5, Qwen-VL, which may cast doubt on its effectiveness on its effect on contemporary models. The claim could be strengthened with an experiment on Qwen2.5-VL.

(minor) For Table 1: LLaVA-1.5-7B (POPE-MSCOCO) Acc. SOTA should be VACoDe, not SAVCD.

### Questions
I would appreciate it if the authors can share any information about applying it to newer models, if they had done experiments on that before. This would help me determine if this method has value for today's VLM development.

I am giving a slightly higher score, as I hope authors can address my concerns.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a training-free decoding strategy named SAVCD, aimed at mitigating hallucinations in Large Vision-Language Models (LVLMs). The method features two core contributions: 1) a Self-Augmentation Selection (SAS) prompting strategy that leverages the model's own knowledge to dynamically select the most relevant visual augmentation for constructing a meaningful "amateur" model in contrastive decoding; and 2) a Sparsity Adaptive Truncation (SAT) algorithm that dynamically adjusts the threshold for candidate tokens based on the entropy of the output logit distribution to improve factual consistency. Experimental results demonstrate that the proposed method outperforms existing decoding strategies across several benchmarks and models.

### Strengths
1.  The paper presents a complete and well-designed decoding framework. The experimental section is thorough, covering multiple models and benchmarks, and the results consistently show that the proposed SAVCD method outperforms baselines in improving factuality. This demonstrates its practical utility as a plug-and-play tool.
2.  The core idea of having the model self-select the most appropriate visual augmentation via prompt engineering to construct an "amateur" perspective for contrastive decoding is highly novel and interesting. It explores the possibility of using a model's own reasoning capabilities to guide its decoding process, offering a new perspective on mitigating LVLM hallucinations.
3.  As a purely inference-time strategy, SAVCD requires no additional training or architectural modifications. This gives it strong generality and a low barrier to adoption, making it an attractive feature for the community.

### Weaknesses
The main weakness of this paper lies in its methodology, which is built on an assumption that may not always hold, and the solution relies to some extent on elaborate engineering tricks, which may affect its robustness and generality.

1.  **Questionable Core Methodological Assumption**: The core of SAVCD, the SAS prompt, relies on a critical assumption: that a model prone to hallucination and reasoning failures can accurately interpret a complex prompt to select an optimal corrective strategy for itself. This introduces a degree of circular reasoning. If the model's foundational reasoning is strong enough to perform this selection task perfectly, the initial problem of hallucination might be less severe. This design makes the method's success partially dependent on the very problem it aims to solve.
2.  **Over-reliance on Prompt Engineering**: The method's effectiveness is heavily dependent on a meticulously crafted, complex prompt that includes extensive descriptions and examples. Such approaches are often brittle, being sensitive to changes in wording, structure, or even model versions. Furthermore, its scalability is limited; for instance, introducing new augmentation types would require manually modifying and debugging this complex prompt. This makes the method feel more like a highly-tuned heuristic for a specific set of tasks and models rather than a general, robust principle.

### Questions
1.  Regarding the robustness of SAS selection: What is the accuracy of the model's augmentation selection in the first step? Are there cases where the model selects a suboptimal or incorrect augmentation, leading to worse results than the baseline? Could the authors provide an analysis of such failure cases?
2.  Regarding the limitations of the SAT algorithm: The core of SAT is using entropy to measure model uncertainty. However, models can sometimes produce a completely wrong hallucination with very high confidence (i.e., very low entropy). In such "confidently wrong" scenarios, could SAT's stricter threshold inadvertently lock in this error, making it harder to correct?
3.  Regarding the generality of the prompt: The SAS prompt appears quite elaborate. How much would the performance degrade if this prompt were simplified (e.g., providing only the names and simple descriptions of augmentations, without the ICL examples)? This would help in understanding to what extent the performance gain comes from the method itself versus the elaborate prompt engineering.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Self-augmented Visual Contrastive Decoding, a training-free decoding strategy designed to mitigate hallucination problem in Large Vision-Language Models (LVLMs). SAVCD consists of two main components. First, it dynamically selects a query specific visual augmentation that produces maximally informative discrepancies between expert and amateur logits. Second, it applies contrastive decoding based on the self-augmented selection-based augmented image to penalize tokens that are likely to cause hallucination, without requiring additional training. The method significantly reduces hallucinations compared to existing baselines and achieves high computational efficiency

### Strengths
Strengths are summarized as follows:

(1) Conceptual Advancement: Query-Aware model-driven self-augmentation to reduce the hallucination effect of VLMs.

(2) Practical and Efficient mitigation without retraining method. 

(3) Comprehensive evaluation and well-written to follow.

### Weaknesses
(1) The performance of SAVCD remains dependent on the underlying LVLM, limiting its robustness when transferred to different architectures or weaker base models. It would strengthen the paper to include expedriments on multiple LVLMs with varying parmeter scales to show cross-model consistency. 

(2) The range of available visual augmentation is still constrained to a predfined candidate set, even though the method dynamically selects among them. 

(3) VaCoDe Misrepresentation; While the paper highlights a major distinction fro mVaCoDe, the description partially misrepresents VaCoDe's mechanism. Specifically, VaCoDe also selects augmentations dynamically from a fixed set based on output divergence, rather than using purely fixed perturbations. It would strengthen that clarifying the comparison in the main text between VaCoDe and SAVCD.

### Questions
Please refer to the Weaknesses part.

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
2

### Summary
This paper addresses the issue of hallucination in Large Vision-Language Models (LVLMs) to improve their generation quality. Previous works have employed contrastive decoding (CD) to enhance the output of Large Language Models (LLMs), an approach later extended to the visual domain as Visual Contrastive Decoding (VCD). Although promising, existing VCD methods often rely on generic visual augmentations that are agnostic to the specific context of the text query, limiting their effectiveness. To address this, this paper introduces a training-free decoding strategy with two key innovations: first, a self-augmentation prompting technique that leverages the model's intrinsic knowledge to dynamically align semantics between the query and visual content; and second, an adaptive thresholding algorithm that adjusts the candidate token set based on output sparsity, making full use of the logit distribution. Extensive experiments validate the method's effectiveness across four LVLMs and seven benchmarks.

### Strengths
This paper is well-organized with good writting. Importantly, the authors provide extensive experiments to validate the effectiveness of their method.

### Weaknesses
**Novelty Assessment**: As I am not deeply familiar with the contrastive decoding literature, I cannot authoritatively assess the novelty of the proposed method against all prior works in Visual Contrastive Decoding (VCD). The idea appears novel based on my reading, but a more thorough discussion of related work would help situate this contribution for a broader audience. I look forward to the discussion with other reviewers and the authors' response on this point.

**Model Generalization**: The experimental validation, while extensive, relies on a set of LVLMs that does not include several recent state-of-the-art models (e.g., InternVL3/3.5, Qwen-VL2/3). To strengthen the claim of generalizability, it would be valuable to see if the proposed SAVCD method's benefits hold on these newer architectures.

Note to the AC and Reviewers: Given my limited background in this specific sub-field, I will rely heavily on the discussion and author response to finalize my assessment, and I defer to the experts on the finer points of the related work.

### Questions
Please refer to my weakness

### Soundness
3

### Presentation
3

### Contribution
2
