# BadConcepts: Backdooring VLMs with Visual Concepts

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Backdoor attacks embed hidden behaviors in models such that inputs with specific triggers cause adversary-chosen outputs while clean inputs remain unaffected. Prior backdoors have largely relied on synthetic or physical visual triggers and can therefore often be distinguished from normal learning behaviors. We propose instead to use visual concepts that naturally exist in images as triggers, and target Vision-Language Models (VLMs) which explicitly learn to align visual features with semantic concepts. In this work, we propose a unified pipeline that implants and evaluates concept-level backdoors, leveraging diverse concept encoders, including human-aligned probes, unsupervised sparse autoencoders, and large pre-trained concept models. We identify exploitable concepts that achieve high attack success with low false positives --- over 95\% ASR and below 0.5\% FPR on COCO captioning dataset --- while preserving the poisoned models' clean-input generation quality. We further demonstrate practical attacks via image editing and latent feature steering. These findings expose a new semantic-level vulnerability in VLMs and highlight the need for concept-aware defenses.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper explores using visual concepts as backdoor attack triggers for image captioning task on VLM. The proposed framework uses a visual concept encoder to compute the concept score of all images in the fine tuning dataset and ranks them. The top ranked images are selected and treated as backdoored samples to pair with the target text. The method is evaluated on LLava and multiple visual concept encoders.

### Strengths
1. The paper provides insights in exploring how visual concepts could be used backdoors for attacks on VLM.

2. The paper provides detailed analysis on the distribution of concept scores in the studied datasets, that help understand when the proposed attack could perform well.

3. The proposed method is evaluated on Youden’s J statistic and false positive rate, which is especially important in this case where the backdoor boundary could be ambiguous.

### Weaknesses
1. The proposed method requires access to all training data during the fine-tuning phase. How does the attack perform if it approximates the training dataset distribution by constructing it own dataset and rank among the local dataset?

2. It appears the attack depends highly on the overall training data distribution. Under one concept encoder, a concept may have different distributions (unimodal etc) in different dataset.

3. The proposed method is only evaluated one target model architecture. Cross architecture analysis could be important for concept encoders that rely on model internal representations.

4. The proposed method appears to be not suitable for clean label attack, which could make it vulnerable to inconsistency defenses.

### Questions
Q1. On line 76, the authors claim  
> At the same time, concepts as triggers
provide attackers with greater flexibility, as they can be chosen from a broad range of attributes in
the data domain and embedded into diverse scenes.

What is concepts as triggers compared to?

Q2. I am not sure if it is ideal to have a backdoor (snow) that is triggered if it is very obvious (people playing with snow), and not triggered when it is somewhat obvious although not obvious enough (there is a snow sign billboard). Can the author add discussions on that?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces ​​concept-level backdoor attacks​​ against Vision-Language Models (VLMs), where ​​naturally occurring visual concepts​​ (e.g., "snowy", "tennis", "red") serve as triggers. Unlike traditional backdoors that rely on synthetic or physical triggers (e.g., patches, adversarial noise), concept-based triggers are ​​inherently semantic and natural​​, making them harder to detect with existing defenses.

### Strengths
1. The paper is clearly written, and the overall framework is well illustrated through intuitive figures.

2. The proposed method effectively exposes the vulnerability of Vision-Language Models (VLMs) to backdoor attacks, which is an important topic for model safety and trustworthiness.

### Weaknesses
1. The fact that visual models can be implanted with backdoors has already been extensively studied. However, the paper lacks a clear motivation for using visual concepts as triggers. It remains unclear why this particular form of trigger is worth investigating — what are the unique challenges and real-world implications compared to existing types of triggers? The manuscript should elaborate on the scenarios in which such visual-concept-based backdoors are likely to occur in practice.

2. The problem setup appears relatively simple and can be addressed using standard fine-tuning techniques. The idea of directly fine-tuning an adapter model is a common practice, and similar category-specific backdoors have been previously explored in purely visual models. The paper does not sufficiently articulate what new challenges arise when extending these attacks to multimodal VLMs, nor does it clearly demonstrate the limitations of prior single-modality approaches in this context.

3. The experiments lack comparisons with strong baseline methods. Without these baselines, it is difficult to evaluate the actual advantages or novelty of the proposed approach.

4. The paper does not discuss how existing VLM backdoor defense techniques perform against the proposed attack. Such analysis would be important to understand the practical robustness of the method and its implications for real-world security.

### Questions
1. From a technical standpoint, how does the proposed attack differ from existing backdoor injection techniques used in unimodal visual models? What specific challenges arise due to the vision-language interaction in VLMs, and how does the proposed method address them?

2. Could the authors clarify the real-world threat model or application scenario that justifies this design choice?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a backdoor attack on VLMs that uses natural visual concepts as triggers instead of synthetic patches or adversarial perturbations. The method poisoned a small portion of the data with such concept and after training, any images with such target concept will yield poisoned results.

### Strengths
1. The paper is clearly written and the experimental setup is easy to follow. 
2. The paper provides a systematic study across multiple concept-selection methods (e.g., sparse autoencoders, concept classifiers), showing how different concept definitions affect attack success rate.

### Weaknesses
The novelty of the paper is the primary weakness. The core attack mechanism is not new. The proposed method is equivalent to a class-level targeted data poisoning or label-flipping attack, where the “class” is defined by a semantic concept. By replacing captions for images that strongly express a particular concept, the fine-tuning process shifts the model’s representation so that the entire semantic region associated with that concept becomes aligned with the attacker’s target output. This behavior has already been well-established in prior works [1–4], to name a few. The paper does not acknowledge or discuss these similarities, which makes it difficult to justify the claimed novelty.

[1] Jia, Jinyuan, Yupei Liu, and Neil Zhenqiang Gong. "Badencoder: Backdoor attacks to pre-trained encoders in self-supervised learning." IEEE S&P, 2022.
[2] Yang, Wenhan, Jingdong Gao, and Baharan Mirzasoleiman. "Better safe than sorry: Pre-training CLIP against targeted data poisoning and backdoor attacks." arXiv:2310.05862, 2023.
[3] Carlini, Nicholas, and Andreas Terzis. "Poisoning and backdooring contrastive learning." arXiv:2106.09667, 2021.
[4] Jha, Rishi, Jonathan Hayase, and Sewoong Oh. "Label poisoning is all you need." NeurIPS 2023.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes BadConcepts, a novel backdoor attack framework that uses naturally occurring visual concepts (e.g., “snowy”, “red”) as triggers in Vision-Language Models (VLMs), rather than synthetic or physical visual triggers. The method leverages diverse concept encoders to score images for a target concept, then poisons only the top-k% samples with a malicious output (e.g., “attack successful”). Experiments on LLaVA show that certain concepts achieve >95% attack success rate (ASR)  on COCO, while preserving clean-task captioning quality.

### Strengths
1. The paper introduces a new paradigm of concept-level backdoors, distinct from pixel or object-based triggers.
2. The proposed BadConcepts pipeline is clear and easy to understand.
3. Experiments demonstrate high attack success while preserving clean-input generation quality.
4. The manuscript is well-structured.

### Weaknesses
1. The paper provides limited empirical analysis of defenses against concept-level backdoors. It remains unclear how these attacks perform when evaluated against standard backdoor detection or mitigation methods.

2. The evaluation focuses primarily on image captioning, leaving other multimodal tasks such as visual question answering (VQA) unexplored.

3. The experiments are conducted on a limited set of architectures, and it is unclear whether concept-based backdoors can be adapted to  different VLM architectures.

4. The method section could provide more detailed explanations of the concept scoring process to improve clarity and reproducibility.

5. The proposed method appears to alter the model’s understanding of specific concepts (e.g., replacing “cat” with “dog”) rather than injecting a conditional trigger–response behavior typical of backdoor attacks. The authors should clarify how their approach differs from conventional data poisoning attacks, as this distinction is crucial for proper positioning within the backdoor literature.

6. The paper does not analyze how the backdoor behaves when triggered by semantically similar or correlated concepts, which may affect attack specificity.

### Questions
Please address the weakness above.

### Soundness
3

### Presentation
3

### Contribution
3
