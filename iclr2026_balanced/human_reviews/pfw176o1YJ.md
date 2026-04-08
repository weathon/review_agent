## Human Reviewer 1

### Summary
This paper investigates the emergence of visual priors in Large Language Models (LLMs) that are trained solely on text. these text-only models encode surprisingly rich visual understanding, which can be unlocked through lightweight multimodal adaptation. The authors conduct over controlled experiments that systematically vary model scale, pre-training data composition, and adaptation strategies to dissect the structure and origin of such priors. Their key findings include:

1. Visual priors decompose into perception and reasoning components
2. Reasoning priors are predominantly acquired from text sources involving code, math, and academic language
3. Perception priors emerge diffusely from broad web-crawl text and depend heavily on visual instruction tuning

Lastly, the paper introduces MLE-Bench, a benchmark focused on perception-specific evaluation.

### Strengths
1. The decomposition of visual priors into perception and reasoning is novel and theoretically impactful; it challenges assumptions of monolithic visual semantics in LLMs.
2. Extensive and well-structured experimental setup across five model sizes, varying data domains, and different visual encoders (MetaCLIP, DINOv2, MAE).
3. Offers practical insights for data-centric pre-training of future MLLMs—highly relevant to both industry and academic development.
4. The paper conducts over 100 controlled pretraining runs, systematically varying data mixtures, encoder choice, and instruction-tuning strategies.
5. The paper is well-written and structured logically.

### Weaknesses
1. The paper’s split of pretraining text into “reasoning” vs. “visual” buckets hinges on an automated pipeline: a 32B dense LLM is prompted to multi-label 1024-token segments into categories such as visual concept/attribute/relationship and code/math/science reasoning; those labels then drive mixture sweeps and downstream claims about where “priors” come from. That setup introduces several risks. 
First, it is model-dependent: is there consistency across other models? i.e. other than Qwen
Second, the error characteristics are unknown: there was no report of inter-annotator (human) checks are reported for the classifier, so mislabeling rates—especially on ambiguous text that mixes descriptive visual language with abstract reasoning—are unconstrained
My main concern here is whether the LLM-based labeling truly distinguishes “visual” from “reasoning” content, or if the observed gains simply reflect differences in data quality—i.e., one category inadvertently contains cleaner or more informative text, which could artificially boost general VQA performance.

2. The paper’s validation at scale is constrained and, in some cases, internally inconsistent, which makes it difficult to confidently generalize the findings beyond the mid-scale (30–50B token) regime. While the main conclusions are derived from extensive 30–50B-token experiments, the authors extend only two mixtures to 1T tokens, both using a 7B LLM: one “language-favorable” (mix 0) and one “balanced” (mix 6). Here, the results in Table 2 (mid-scale) show that mix 0 outperforms mix 6 on pure language benchmarks, as expected from its language-heavy composition. However, in Table 3 (1T-token scale), the trend reverses on several language tasks—mix 6 overtakes mix 0 despite having proportionally less “language” text. This reversal is puzzling and suggests a possible interaction between model scale, token budget, and mixture composition that is not fully explored. Could the authors elaborate more on this?

Compounding this, the 1T-token budget far exceeds the Chinchilla optimal region for a 7B model (≈150–200B tokens), placing both scaled-up models deep into the over-training regime where diminishing return are likely. That makes it difficult to attribute performance shifts solely to mixture content, rather than to scaling artifacts or excessive token exposure.

### Questions
1. The perception vs reasoning tasks showed weak or negative correlation (Figure 4). Can the authors clarify why such inverse relationships might occur?
2. since hallucination "stronger models do not necessarily guarantee fewer hallucinations in this blind VQA", did the authors measure hallucination rates systematically?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
## Summary

This paper is a pioneering work that uses a proxy visual task framework (P-SET) to quantify and dissect the **visual priors** that emerge in Large Language Models (LLMs) from **text-only** pre-training, decomposing them into separable perception and reasoning components. The study reveals that these priors follow unique scaling laws, where the reasoning prior scales poorly, providing the MLLM community with critical **descriptive insights** to guide future model alignment and pre-training strategies.

### Strengths
## Strengths

1.  **S1: Pioneering Dissection of Emergent Visual Knowledge.**
    This work represents a **pioneering effort** to systematically dissect and analyze the implicit visual knowledge (priors) that emerges purely from **text-only** pre-training in Large Language Models. By demonstrating the existence and growth of these priors, the paper provides a crucial, non-trivial insight into the mechanisms by which LLMs acquire world knowledge, laying the foundation for **mechanistic interpretability** studies in the multimodal domain.

2.  **S2: Novel Framework for Operationalizing and Separating Priors.**
    The paper introduces a **novel and comprehensive framework** using proxy visual tasks (perception vs. reasoning) to quantitatively *operationalize* and *separate* the distinct components of the visual prior. This methodology allows for the first time a clear, quantitative distinction between different types of emergent visual capabilities, moving beyond anecdotal evidence to **systematic, scalable measurement**.

3.  **S3: Key Scaling and Developmental Insights for MLLM Construction.**
    The work yields **critical, quantitative insights** into how the perception and reasoning priors scale with model size. The key finding that the **reasoning prior scales poorly** compared to the perception prior provides a valuable, data-driven hypothesis for MLLM researchers, guiding future efforts to strategically adjust data composition or architecture to unlock superior visual reasoning capabilities.

4.  **S4: Rigorous Construction of Proxy Visual Tasks (P-SET).**
    The paper's construction of the P-SET benchmark, specifically designed to isolate and measure perception and reasoning capabilities within the text modality, is highly rigorous. This set of tasks effectively serves as a crucial tool for **probing** the latent knowledge structure of LLMs, and the benchmark itself is a valuable public contribution to the field of MLLM analysis.

5.  **S5: Strong Empirical Evidence via Causal Intervention (Minimal Fine-Tuning).**
    The central hypothesis—that these priors exist and are beneficial—is supported by strong empirical evidence showing that LLMs with higher measured priors achieve better performance in MLLM tasks (e.g., VQA) after **minimal multimodal fine-tuning**. This suggests that the latent knowledge is indeed functional and ready to be "unlocked," validating the study's relevance to the efficiency of MLLM alignment.

### Weaknesses
## Weaknesses

1.  **W1: Lack of Mechanistic Evidence for the Separation of Priors.**
    The central claim is the separation of visual priors into "perception" and "reasoning" components. This distinction relies heavily on an **operational definition** (performance on specific proxy tasks) rather than **mechanistic evidence**.
    * The authors **fail to demonstrate** that these priors are truly **orthogonal or separable** in the LLM's internal representations (e.g., via representation similarity analysis, or targeted causal interventions on specific latent dimensions).
    * This lack of a solid **mechanistic foundation** suggests that the claimed separation might be an artifact of the task design itself rather than an inherent property of the emergent knowledge within the LLM.

2.  **W2: Fatal Causal Confounding of Textual Knowledge and Visual Priors.**
    The core, high-impact thesis is that visual priors emerge from **text-only** pre-training. However, the pre-training data contains vast amounts of **structured, descriptive visual language** (e.g., "the red square is to the left of the blue circle").
    * The paper fails to **rigorously rule out** that the LLM is merely encoding **linguistic knowledge about visual relationships** (i.e., the language semantics of relational phrases) rather than truly emergent **visual spatial knowledge**.
    * This **causal confounding** between linguistic encoding and visual priors significantly weakens the claim that LLMs learn to "see" before seeing an image, as the identified "prior" could simply be complex language fluency.

3.  **W3: Lack of Prescriptive Guidance for MLLM Design (The Analysis-Synthesis Gap).**
    The work is primarily **descriptive**, successfully identifying the existence and scaling trends of these priors (e.g., reasoning prior scales poorly). However, it fails to offer **actionable, prescriptive guidance**.
    * For instance, given the finding that the reasoning prior scales poorly, can the authors propose a **concrete, quantifiable text pre-training data optimization strategy** (e.g., adjusting the ratio of relational reasoning text) and empirically prove that this modification **more efficiently** boosts the reasoning capability?
    * This lack of **analysis-to-synthesis** guidance limits the practical value of this research for constructing or optimizing the pre-training stage of next-generation MLLMs.

### Questions
## Questions

1.  **Q1: Mechanistic Validation of Prior "Separability."**
    The core claim is the operational separation of visual priors (perception vs. reasoning) based on P-SET tasks. Can the authors provide evidence at the level of the LLM’s internal representations (e.g., via Representation Similarity Analysis, CCA, or causal intervention) to prove that these priors are **orthogonal or uncorrelated** in the latent space? Lacking this mechanistic evidence, how do we rule out that the separation is merely an artifact of the **proxy task design** (P-SET)?

2.  **Q2: Rigorously Excluding Textual Knowledge Confounding.**
    The paper asserts that visual priors emerge from text-only training, yet fails to **strictly exclude** that this is simply the encoding of **structured, descriptive visual language** found in the massive text corpus. Have the authors attempted to train a small LLM on a corpus **devoid of all relational and spatial language** to see if the reasoning prior still emerges? If this **linguistic confounding** is not ruled out, the causal foundation of the central thesis—that "LLMs learn to see before seeing"—is a fatal weakness.

3.  **Q3: Transferability Challenge of the P-SET Proxy Tasks.**
    The P-SET tasks are synthetic. While the paper shows a **correlation** between the prior and downstream MLLM performance, how **robust** is this correlation? Have the authors performed an ablation showing that an LLM with a higher P-SET score *consistently* yields superior downstream performance even when subjected to *different* minimal fine-tuning techniques (e.g., LoRA vs. full tuning)? If a high P-SET score does not reliably predict downstream gains, its value as an **efficient MLLM pre-training metric** is questionable.

4.  **Q4: Influence of Specific LLM Family on Scaling Laws.**
    The scaling laws are predominantly based on the Llama family or similar Transformer architectures. Have the authors attempted to replicate the scaling trends of these priors on **non-Transformer architectures** (e.g., Mamba or RetNet)? If these laws are **architecture-dependent** rather than a **universal property of language models**, doesn't the demystification of visual priors only serve as a description of a **specific model family** instead of a **general law of learning**?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 3

### Summary
Based on the common practice of finetuning Multimodal Large Language Models (MLLMs) from pretrained Large Language Models (LLMs), this paper analyzes how LLM pretraining influences downstream MLLM finetuning performance. It focus primarily on how LLM pretraining data construction affects the downstream MLLM performance. The results reveal several interesting findings including aspects of
- Scaling law of LLM pretraining on MLLM finetuning
- Source of visual prior
- Data arrangements
- Structure of visual priors

Guided by these findings, this paper pretrains a new LLM using a reorganized language dataset, achieving improved downstream performance for MLLM finetuning.

### Strengths
- Well-motivated problem with clear relevance to current MLLM construction practices.
- Sufficient evaluation breadth and depth, with extensive controlled studies across data mixtures and scales.
- In-depth findings that are coherent with the presented experiments and lead to a concrete, data-centric pretraining recipe.

### Weaknesses
- Overclaim in the abstract that "perform visual tasks without ever having seen an image". The paper does not actually demonstrate an MLLM that performs visual tasks without any visual training.
- Unclear vision-input processing in the default settings. It is not clear whether the default setup uses any vision encoders or just a MLP adapter. And how to process the vision inputs.
- Since stronger LLMs typically yield stronger MLLMs after adaptation, it remains ambiguous how much of the reported gains come from claimed "visual priors" versus general language capacity. The paper only reports perfomance on vision tasks, but lacks evaluation of the language capacity of each mdoel. So it is not very clear whether it is really benefits from visual prior or just stronger langauge capacities.

### Questions
How is model trained with "0B" data obtained in Figure 1? Why a model with no training data can also get non-trivial performance?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper presents a comprehensive study on how the vision-related capabilities of MLLMs—including visual perception and reasoning—emerge during text-only pretraining. To investigate this, the authors conduct extensive controlled experiments by training LLMs of different sizes under various data mixture settings, carefully varying the proportion of image-caption pairs of different types.

These experiments yield a set of valuable insights, revealing where the model’s perception and reasoning abilities originate from even in the absence of real visual input. The results highlight the importance of descriptive language data and reasoning data in shaping vision-related capabilities, and identify the optimal data mixture ratios for strong downstream VQA performance.

Leveraging these findings, the authors scale up the training to approximately 1 trillion tokens and train a larger model following the recommended data composition. The improved performance of this scaled model further validates the effectiveness of the insights derived from the ablation studies.

### Strengths
* The problem investigated in this paper is highly important and relevant to the MLLM community, yet has been largely overlooked in prior work. Understanding how vision-related capabilities emerge from text-only pretraining provides valuable insight into the foundations of multimodal learning.

* The conclusions drawn from the extensive experiments are both instructive and insightful. They offer practical guidance for future model development, particularly in designing effective data mixtures and scaling strategies for training more capable MLLMs.

* The paper is well-organized, clearly written, and easy to follow. The presentation is informative, with a logical flow that effectively conveys the key ideas and findings.

### Weaknesses
* All VQA questions are divided into several categories, such as General VQA, Knowledge-based VQA, and OCR & Chart VQA. However, these category definitions may appear abstract to readers, especially without concrete examples. Including one or two representative examples for each type in the main text or a table would significantly improve clarity and help readers better understand the distinctions between the tasks.

* I understand that the 9-page limit constrains the space available for presentation. Nevertheless, some conclusions in the supplementary materials also contain valuable insights that support the paper’s findings. While they may not fit in full, briefly referring to them in the main text would help draw readers’ attention to these contents and enhance the overall impact of the work.

### Questions
See Weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
3