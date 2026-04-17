# Mitigating Object Hallucination in Large Vision-Language Models through Adversarial Contrastive Finetuning

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
In recent years, large vision-language models (LVLMs) have made remarkable progress across a variety of vision-language tasks. However, they remain prone to object hallucination like generating descriptions of nonexistent objects in images. To explore the internal mechanism of object hallucination, we collected normal and hallucinated image-text pairs and performed quantitative analysis based on cosine similarity and qualitative analysis based on smooth Grad-CAM. We found that LVLMs may cause hallucinations due to incorrect extraction of image features and mismatch between image and text features. Inspired by these findings, we propose an adversarial contrast fine-tuning (ACFT) method designed to enhance the alignment between visual and textual embedding and encourage the visual modality to focus on the correct image features, thus mitigating object hallucinations. The key approach involves automatically generating paired positive and negative examples using an adversarial hallucination attribute flipping (AHAF) method, followed by contrastive fine-tuning of the LVLM. Through extensive experiments, we show that ACFT achieves state-of-the-art performance on multiple benchmarks, e.g. outperforming existing approaches like VCD, OPERA and VTI, etc. on multiple benchmarks like POPE and MME.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Adversarial Contrastive Fine-tuning (ACFT) to mitigate object hallucination in LVLMs.
The method introduces Adversarial Hallucination Attribute Flipping (AHAF) to automatically generate aligned positive–negative image pairs, followed by contrastive fine-tuning that maximizes text–image embedding similarity for positives and minimizes it for negatives.

### Strengths
1. The logical organization of the paper is clear and coherent, making it easy to follow.
2. The motivation is well-articulated and grounded in empirical observation, effectively linking visual perturbations to hallucination behavior.

### Weaknesses
1. Limited benchmark coverage.

(a) The evaluation focuses only on binary benchmarks (POPE, MME). The paper should also consider generative hallucination evaluations such as AMBERA [3], MMHal-Bench [4], and ObjectHal [5], which assess open-ended generation quality rather than yes/no predictions.

(b) Without these benchmarks, it is difficult to verify whether ACFT generalizes beyond binary tasks.

2. Lack of fair comparison with fine-tuning baselines.

(a) The paper omits DPO-based hallucination mitigation approaches that achieve strong results with very limited data, e.g., CHiP-DPO [1], On-Policy DPO [2].

(b) These methods typically use only around 5k preference pairs yet reach or exceed state-of-the-art results, which questions the efficiency advantage claimed by ACFT.

3. Data efficiency and computational cost unclear.

(a) The paper mentions using “0.9% of COCO” but does not specify the actual number of samples or the corresponding GPU hours.

(b) Without a comparable-scale baseline (e.g., DPO fine-tuning with 0.9% of COCO pairs), the efficiency claim is not well supported.

4. Reproducibility concerns on VCD results.

(a) The reported VCD results on LLaVA-1.5-7B differ significantly from the original paper [6].

(b) The original work reported around +2% Acc and +4% F1 gains under adversarial settings, whereas this paper shows lower-than-baseline results **(w/ VCD < w/o VCD)**, suggesting inconsistent setups that may compromise the fairness and reproducibility of comparison.

5. Conceptual limitation in contrastive design.

(a) Positive/negative pairs are created only in the image domain, while text-side perturbations or multimodal feature contrasts are not explored.

(b) It remains unclear whether similar or better robustness could be achieved via token-level or text-level contrastive learning.


[1] Chip: Cross-modal hierarchical direct preference optimization for multimodal llms, ICLR 2025.

[2] Mitigating hallucinations in large vision-language models via dpo: On-policy data hold the key, CVPR 2025.

[3] Amber: An llm-free multi-dimensional benchmark for mllms hallucination evaluation,  2023.

[4] Aligning large multimodal models with factually augmented rlhf, ACL 2024.

[5] Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback, CVPR 2024.

[6] Mitigating object hallucinations in large vision-language models through visual contrastive decoding, CVPR 2024

### Questions
Please refer to the issues discussed in Weaknesses.
In particular, I would like the authors to clarify the following:

(a) Whether the experimental setup is fair and consistent, especially regarding the reported VCD [6] results.
(b) Whether, under a comparable fine-tuning data scale, the proposed method can achieve state-of-the-art performance similar to DPO-based approaches [1–2].
(c) Whether the method can maintain its advantage on generative hallucination benchmarks, such as AMBERA [3], MMHal-Bench [4], and ObjectHal [5].

If these points can be convincingly addressed, I would consider raising my overall rating.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the problem of object hallucination in Large Vision-Language Models. It identifies visual-textual misalignment and erroneous visual feature extraction as key causes, particularly in short-text outputs. To mitigate this, the authors propose a novel Adversarial Contrastive Fine-Tuning (ACFT) method, which uses adversarially generated image pairs to enhance visual-textual alignment. Experimental results on benchmarks like POPE and MME show that ACFT achieves state-of-the-art performance in reducing hallucinations without compromising general visual understanding.

### Strengths
The paper demonstrates strong originality by creatively combining adversarial training with contrastive learning in a novel framework (AHAF+ACFT). It formulates the hallucination problem from an under-explored image-prior perspective, specifically targeting the short-text-output scenario, which provides a fresh alternative to dominant language-prior approaches. 

The work is technically sound, evidenced by rigorous quantitative analysis (cosine similarity, benchmark results) and qualitative validation (Smooth Grad-CAM). Its significance is high: ACFT achieves SOTA performance while being highly efficient (using only 0.9% of COCO data), making robust hallucination mitigation more practical for real-world applications without added inference cost.

### Weaknesses
* ***Scope Limitation and Generalizability***.
The paper's exclusive focus on short-text outputs (Yes/No QA) is a significant limitation. While strategically chosen to argue for the image-prior perspective, it leaves the method's efficacy in more common, open-ended long-text generation tasks (e.g., captioning, detailed description) completely unverified. The claimed superiority over "language-prior" methods is thus narrowly contextualized, and the core contribution would be substantially strengthened by demonstrating that ACFT's benefits transfer to these standard hallucination evaluation settings.

* ***Lack of Cross-Dataset Validation***
The experimental validation is confined to two benchmarks (POPE and MME) which, despite their relevance, are structurally similar (both are VQA-style). The method's robustness would be more convincingly demonstrated by evaluating on a dataset featuring a different type of short-answer task (e.g., object counting in a non-VQA format) or from a different data distribution, to ensure the improvements are not overfitted to the specific QA format of the primary benchmarks.

### Questions
*   **Q1:** The AHAF method uses PGD to create negative samples. To what extent are the learned robust features specific to defending against PGD-style, norm-bound perturbations? Have you explored if the improvements hold against other types of distribution shifts or semantic perturbations that might more naturally cause hallucinations (e.g., occlusions, unusual object contexts)?
*   **Q2:** In line 42, what is "hallucination heads"?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the causes of object hallucination inVLM, proposing that it stems from the misalignment between visual and textual features through quantitative analysis usingcosine similarity between image and text embeddings and qualitative analysis using Grad-CAM. Based on this diagnosis, the paper introduces Adversarial Contrastive Finetuning, a method that first generates aligned positive and negative image pairs using an Adversarial Hallucination Attribute Flipping technique. It then uses these pairs for contrastive finetuning to improve visual-textual alignment . Experiments on LLaVA-1.5-7B and MiniGPT-4-13B improve performance on the POPE and MME-Existence benchmarks compared to the baselines.

### Strengths
1.The paper visually demonstrates through cosine similarity plots and Grad-CAM heatmaps that hallucinated image-text pairs tend to exhibit lower alignment and misdirected visual attention compared to non-hallucinated pairs.

2.The proposed method, involving the construction of negative image samples via AHAF and subsequent contrastive finetuning ACFT, is shown to improve the performance of the tested models LLaVA1.5 and MiniGPT-4 on the POPE and MME-Existence hallucination benchmarks.

### Weaknesses
1.The core ideas identifying misalignment between visual and text representations as a cause for hallucination , and employing contrastive learning to improve alignment and mitigate hallucination —may lack significant novelty, as similar concepts have been discussed in prior work HACL[1].

[1]Hallucination Augmented Contrastive Learning for Multimodal Large Language Model

2.A major concern lies in the choice of base models for experimentation. The paper evaluates its method on LLaVA1.5 and MiniGPT-4 , which are relatively old, approx. 2 years and significantly weaker compared to the current sota in VLMs. Given the rapid iteration cycles in VLM development, it remains unclear whether the proposed method would yield similar benefits for much more capable models like Qwen2.5VL or InternVL3. The lack of validation on contemporary models severely limits the practical relevance and generalizability of the findings.

### Questions
1.The evaluation of hallucination mitigation primarily relies on the POPE and MME-Existence benchmarks, which predominantly use binary or multiple-choice formats to assess object presence . Object hallucination, however, also manifests significantly in free-form text generation like image captioning. Could the authors provide results on caption-based hallucination benchmarks, such as CHAIR, to demonstrate the effectiveness of ACFT in reducing hallucinations in generated descriptive text, and quantify the performance improvement in that setting?

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
This paper introduces Adversarial Contrastive Fine-Tuning (ACFT) to mitigate object hallucination in Large Vision-Language Models (LVLMs). The authors first analyze the internal causes of hallucination via cosine similarity and Smooth Grad-CAM, observing that hallucinated samples exhibit weaker image–text alignment and diffuse visual attention. Based on these insights, they propose a two-stage framework:
(1) Adversarial Hallucination Attribute Flipping (AHAF) — a PGD-based adversarial procedure that generates positive/negative image pairs differing in final output generation.
(2) ACFT, which performs contrastive fine-tuning to maximize the similarity between text anchors and positive images while minimizing it for adversarial negatives.
Experiments on POPE and MME benchmarks using LLaVA-v1.5-7B and MiniGPT-4-13B show improved F1 and accuracy over baselines such as VCD, OPERA, VTI, and Woodpecker, without degrading general visual comprehension.

### Strengths
The paper’s main contribution is the idea of adversarially aligned positive-negative pairs for contrastive LVLM fine-tuning—uniting adversarial robustness and cross-modal alignment.

The methodology is detailed and mathematically rigorous, defining clear objectives for adversarial perturbation (Eq. 1–3) and the contrastive loss (Eq. 4–8).

Experiments are well-presented, including multiple baselines, two architectures, and ablations (with/without contrast loss, Grad-Cam).

The authors also show that the method uses only 0.9% of COCO for fine-tuning—a good efficiency advantage.

The paper is organized and easy to follow.

### Weaknesses
The central motivation—that low cosine similarity between image and text embeddings correlates with hallucination—remains empirically stated rather than conceptually justified. It is unclear why a higher similarity between the image feature and the corresponding text feature should necessarily indicate reduced hallucination. For instance, when the input question is semantically irrelevant to the image, one would still expect low similarity without this implying hallucination. 

The evaluation is limited to binary yes/no benchmarks (POPE and MME-Existence), which mainly assess short-text object *existence detection*. However, since the core problem concerns object hallucination—i.e., generating descriptions of nonexistent objects in images—additional description-level (or caption-level) benchmarks (e.g., CHAIR, CCEval) are necessary for a more comprehensive evaluation.

The baselines are largely non-fine-tuned or inference-time methods, while ACFT involves model fine-tuning. This makes the comparison potentially unfair. The paper should include results against fine-tuned or post-training alignment baselines to ensure a fair assessment of relative improvement.

The Grad-CAM attention maps presented in the ablation/visualization (Figure 5) do not show statistically significant or consistently noticeable improvements, as the paper claimed " tightly focused on the target objects."

Although the paper claims efficiency advantages (using only 0.9% of COCO), it provides no quantitative metrics such as training time, GPU hours, or per-batch cost. Since each batch involves iterative adversarial optimization (PGD steps), the computational overhead could be substantial. A cost–performance analysis is necessary to validate the claimed efficiency benefits.

### Questions
How does the similarity metric behave when the question or text prompt is semantically irrelevant to the image? Do LVLMs consistently hallucinate under such conditions, or does low similarity simply reflect unrelated semantics?

Since object hallucination often manifests at the caption or descriptive level, could the authors extend evaluation to captioning-based hallucination benchmarks (e.g., those using CHAIR or CCEval) to test the generality of ACFT beyond binary yes/no tasks?

Given that ACFT involves model fine-tuning, can the authors include comparisons against other fine-tuned or post-training alignment methods?

What is the runtime overhead of the framework?

### Soundness
2

### Presentation
3

### Contribution
2
