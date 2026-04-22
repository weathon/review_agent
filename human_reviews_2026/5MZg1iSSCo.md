# The Coherence Trap: When MLLM-Crafted Narratives Exploit Manipulated Visual Contexts

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
The detection and grounding of multimedia manipulation has emerged as a critical challenge in combating AI-generated disinformation. While existing methods have made progress in recent years, we identify two fundamental limitations in current approaches: (1) Underestimation of MLLM-driven deception risk: prevailing techniques primarily address rule-based text manipulations, yet fail to account for sophisticated misinformation synthesized by multimodal large language models (MLLMs) that can dynamically generate semantically coherent, contextually plausible yet deceptive narratives conditioned on manipulated images; (2) Unrealistic misalignment artifacts: currently focused scenarios rely on artificially misaligned content that lacks semantic coherence, rendering them easily detectable. 
To address these gaps holistically, we propose a new adversarial pipeline that leverages MLLMs to generate high-risk disinformation. Our approach begins with constructing the MLLM-Driven Synthetic Multimodal (MDSM) dataset, where images are first altered using state-of-the-art editing techniques and then paired with MLLM-generated deceptive texts that maintain semantic consistency with the visual manipulations. 
Building upon this foundation, we present the **A**rtifact-aware **M**anipulation **D**iagnosis via MLLM (AMD) framework featuring two key innovations: Artifact Pre-perception Encoding strategy and Manipulation-Oriented Reasoning, to tame MLLMs for the MDSM problem. Comprehensive experiments validate our framework's superior generalization capabilities as a unified architecture for detecting MLLM-powered multimodal deceptions. In cross-domain testing on the MDSM dataset, AMD achieves the best average performance, with ACC, mAP, and mIoU scores of 88.18, 60.25, and 61.02, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper looks at investigating how MLLM driven misinformation in two modalities, namely text and image. They construct a dataset MDSM that simulates realistic multimodal manipulations where images are first manipulated and paired with MLLM-generated deceptive texts that maintain semantic consistency with the visual manipulations. Additionally this paper proposes a detection and grounding framework that outputs both coordinates of the manipulation and explanations (AMD).

### Strengths
* I believe this paper is relevant and has a few strengths to it. There are some interesting experiments that are done with both a zero-shot setting and training models on the MDSM dataset.

* Authors spent time trying to do LoRA finetuning on their dataset which I think was an important experiment the have included.

* Showcasing how other models like HAMMER and FKA-Owl which are prevalent in multi-modal manipulations was good to have and some discussion of the analysis

* Including a human evaluation of some of the manipulated images/text was good to include for this work to showcase how humans can be fooled by multi-modal models

### Weaknesses
* I believe that the authors should try and include more Open-Source multimodal models, for zero-shot evaluation in Table 2, currently the only model present is Qwen and no other popular models like Deepseek, LLaVa, Yi-VL.

[1] Deepseek llm: Scaling open-source language models with longtermism.
[2] Visual instruction tuning, Neurips 2023
[3] Yi: Open foundation models by 01.ai, 2024.

### Questions
* Did the authors use GPT-4o and Gemini-2.0 on all the images for the test set? Seems like quite a large test set for a paid model. How large is the test set?

* Why didn’t the authors include more Open-Source models in Table 2, currently there is basically one 1, since only variants of QWEN is included.

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
This paper focuses on the "coherence trap" phenomenon in multimodal misinformation detection. The authors first construct MDSM, a large-scale, diverse, and aligned benchmark for multimodal manipulation detection, comprising challenging samples from reputable news sources. To address this challenge, the authors propose AMD (Artifact-aware Manipulation Diagnosis), a novel framework built on Florence-2 that integrates artifact pre-perception encoding and manipulation-oriented reasoning. Experiments show that AMD outperforms existing methods on MDSM, demonstrating improved robustness in detecting coherently generated fake news.

### Strengths
1. The paper identifies and formalizes the "coherence trap," a highly relevant and critical issue in the era of advanced generative models, where the very coherence that makes AI-generated content useful also makes it dangerously deceptive.

2. The construction of MDSM is a significant contribution. Its scale, diversity of sources, and alignment between modalities make it a valuable resource for the research community.

3. The paper presents thorough experiments, including ablation studies and cross-domain evaluations, demonstrating the superior performance of AMD over state-of-the-art baselines on the MDSM benchmark.

### Weaknesses
1. While the paper formally defines and highlights the "coherence trap" as a critical challenge in multimodal misinformation detection, the underlying concept of detecting semantically aligned fake content is not entirely novel. Prior works, such as MMFakeBench, have already explored scenarios involving coherent image-text manipulations. 

2. The proposed AMD framework is built upon the powerful Florence-2 model and leverages its strong pre-trained multimodal understanding. The architectural innovation of AMD itself appears limited. 

3. While the paper shows AMD's overall success, a deeper analysis of when and why AMD fails (e.g., specific types of manipulations it struggles with, examples of false positives/negatives) would strengthen the work and provide more insight for future research.

[1]Liu X, Li Z, Li P P, et al. MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs. ICLR 2024

### Questions
1. While the "coherence trap" is well-motivated, similar aligned fakes have been studied in prior work (e.g., MMFakeBench). How does this work differ in problem formulation or threat model beyond dataset scale?

2. The AMD framework builds directly on Florence-2 with minimal architectural changes. To what extent do the gains come from the model design versus the strong pre-trained backbone?

3. What are the main failure modes of AMD? A deeper analysis of false positives/negatives or challenging manipulation types (e.g., subtle edits, coherent text-only fakes) would strengthen the paper.

4. How generalizable is AMD beyond news domains? The reliance on Florence-2’s world knowledge may limit performance on out-of-distribution content (more sophisticated, MLLM-generated manipulations or content generated by different large models).

### Soundness
2

### Presentation
2

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
The paper tackles multimodal misinformation detection under the claimed more realistic threat model, generating images that have been locally manipulated (via face swap or face attribute editing), and paired with text generated by a multimodal LLM (MLLM) that is semantically consistent with the manipulation. The authors propose MDSM, a 441k-sample dataset where faces in news images are edited and paired with coherent deceptive narratives. Then introduce AMD, a Florence-2–based model that can handle both (i) binary “fake vs real,” (ii) manipulation type, and (iii) manipulated-region bounding boxes tasks in text form.

### Strengths
1. The motivation is clear. The paper explicitly argues that prior work assumes crude cross-modal inconsistency, which makes detection too easy because the text and image obviously disagree. By contrast, MDSM uses an MLLM to generate fluent, contextually aligned fake narratives that match the manipulated visual identity. This is a meaningful direction.

2. AMD outputs manipulation decisions and the tampered region coordinates as a single textual answer instead of separate detection heads. This makes the downstream application easier.

3. They employ a cross-domain setting for evaluation, emphasizing the generalization ability.

### Weaknesses
1. The novelty claimed based on the flaw of previous works is not strong enough. Prior work like FKA-Owl is already an MLLM-style system that “incorporates more world knowledge to improve the model’s cross-domain performance,” explicitly targeting multimodal fake news scenarios. The paper acknowledges this but still claims current approaches “fail to account for sophisticated misinformation synthesized by MLLMs,” which is not persuasive enough. 

2. On the model side, AMD is essentially Florence-2 + DaViT with several augmented function modules. This is solid engineering, but conceptually similar to known frameworks in this area. 

3. The paper repeatedly stresses that MDSM “defines a more challenging and practical problem,” “high-risk disinformation,” and “semantically coherent and contextually plausible narratives,” and that previous benchmarks are “too simplistic to effectively deceive the public.” Are there any quantitative and empirical demonstrations of the claimed risk?

4. MDSM only keeps “human-centric” news with faces and named entities, then applies two manipulation types in the image domain, including Face Swap and Face Attribute editing. This is not a general-purpose “multimodal misinformation” benchmark. A multimodal celebrity face tampering benchmark is more appropriate.

5. In the cross-domain training setting, how could the data leakage risk be avoided? Would there be the same celebrity headshot in training and testing sets? Besides, the distribution of manipulations and the linguistic style of fabricated text are almost identical across “domains.” It looks closer to the same attacker pipeline, different news source name.

6. In the tables, you compare single mAP values across models: how exactly is multi-label type detection scored as AP? Is it per manipulation type and averaged? Macro/micro? The paper does not say.

7. Which parameters are actually trainable in APE? Only the Artifact Token + classifier Ca? Is there a two-stage schedule or joint multitask training with loss L? The description is not clear enough for training details.

8. In ablations, why do the authors never isolate TRP alone?

9. If the AMD decoder is effectively trained to emit the answer at the end of that rigid QA format, how robust is it to prompt perturbations? Could an adversary just reformulate the question?

### Questions
Please see the question above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces a novel forgery detection framework that targets semantically coherent multimodal content generated by modern Multimodal Large Language Models (MLLMs). The proposed framework leverages Artifact Pre-perception Encoding (APE) and Manipulation-Oriented Reasoning (MOR) to collaboratively analyze image-text manipulations through the reasoning capability of MLLMs. In addition, the authors construct an MLLM-Driven Synthetic Multimodal (MDSM) dataset, where image and text modifications are jointly guided by MLLMs to ensure semantic alignment.

### Strengths
1.The paper clearly highlights the risks posed by semantically coherent forgeries generated by modern Multimodal Large Language Models (MLLMs).
2.It presents a large-scale, semantically aligned multimodal dataset, effectively filling a crucial gap in resources for studying MLLM-driven misinformation.
3.The proposed framework integrates Artifact Pre-perception Encoding (APE) and Manipulation-Oriented Reasoning (MOR), leveraging the reasoning capabilities of MLLMs to collaboratively analyze image-text manipulations. By synergizing APE and MOR, it effectively adapts MLLMs for precise manipulation analysis.
4.The authors conduct comprehensive experiments on both the MDSM and DGM4 datasets, achieving state-of-the-art average cross-domain generalization performance.

### Weaknesses
1.The paper aims to tackle the challenge of forgery detection in semantically aligned image-text scenarios. However, its model design primarily focuses on visual forgeries while largely overlooking textual forgeries.
2.Limited interpretability: although some visualizations are provided, the paper does not clearly demonstrate which specific forgery cues the model captures, nor does it offer a human-understandable reasoning process behind its decisions.
3.The organization of Section 2.2 could be improved — the relationships among its subsections are not clearly articulated, making the overall flow somewhat difficult to follow.

### Questions
1.Could the authors elaborate on the choice to focus specifically on facial modifications and replacements? Have considerations been made regarding the potential extension of the framework to other types of visual or contextual manipulations, such as scene-level or object-level edits?
2.Would the authors be able to provide additional details about the dataset used to train the artifact-aware classification head? In particular, it would be helpful to know the scale of the data and the relative proportions of different manipulation types included during training.

### Soundness
2

### Presentation
3

### Contribution
2
