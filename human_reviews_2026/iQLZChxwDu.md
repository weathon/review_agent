# Probing CLIP's Comprehension of 360-Degree Textual and Visual Semantics

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
The dream of instantly creating rich 360-degree panoramic worlds from text is rapidly becoming a reality, yet a crucial gap exists in our ability to reliably evaluate their semantic alignment. Contrastive Language-Image Pre-training (CLIP) models, standard AI evaluators, predominantly trained on perspective image-text pairs, face an open question regarding their understanding of the unique characteristics of 360-degree panoramic image-text pairs. This paper addresses this gap by first introducing two concepts: \emph{360-degree textual semantics}, semantic information conveyed by explicit format identifiers, and \emph{360-degree visual semantics}, invariant semantics under horizontal circular shifts. To probe CLIP's comprehension of these semantics, we then propose novel evaluation methodologies using keyword manipulation and horizontal circular shifts of varying magnitudes. Rigorous statistical analyses across popular CLIP configurations reveal that: (1) CLIP models effectively leverage explicit textual identifiers, demonstrating an understanding of 360-degree textual semantics; and (2) CLIP models fail to robustly preserve semantic alignment under horizontal circular shifts, indicating limited comprehension of 360-degree visual semantics. To address this limitation, we propose a LoRA-based fine-tuning framework that explicitly instills invariance to circular shifts. Our fine-tuned models exhibit improved comprehension of 360-degree visual semantics, though with a slight degradation in original semantic evaluation performance, highlighting a fundamental trade-off in adapting CLIP to 360-degree panoramic images.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper's title is "probing CLIP's comprehension of 350-degree textual and visual semantics". The paper presents several themes, including 360-dgeree panoramic image generation (040), accurate evaluation of semantic alignment between generated 360-degree panoramic images (Line 050), and CLIP models' comprehension ability in 360-degree panoramic image-text pairs (Line 072). These themes make the paper less focused. Nevertheless, the paper studies CLIP models in term of their ability to understand panoramic images. It adopts various statistical tools to investigate this matter. It horizontally shifts panoramic images and measures how consistent CLIP models output scores. It concludes that CLIP models can understand keywords such as "a 360 degree view of" and "panorama", but does not process robustness in producing consistent scores on horizontally shifted panoramic images. It adopts LoRA to adapt the visual encoder, enhancing this robustness.

### Strengths
Below are notable strengths of this paper.

- The variety of statistical tools are interesting.
- Technical design choices are generally well thought-out, e.g., the null hypotheses, dataset construction, etc.
- Using LoRA to empower CLIP models' robustness to horizontal-shift panorama is interesting.

### Weaknesses
Below are notable weaknesses of this paper.

- The focus of this paper is not clear. Line 011 and the first paragraph of Introduction leave an impression that this paper focuses on generating 360-degree panoramic images. However, Line 049 motivates evaluating semantic alignments between generated 360-degree panoramic images, leaving an impression that the work focuses on developing methods to evaluate generative models' performance in terms of semantic alignments of their generated images. Quite confusingly, Line 051 and Line 072 state that the the work particularly focuses on CLIP models and studies their understanding ability of panoramic images.

- As the paper is motivated for accurate evaluation of semantic alignment between generated 360-degree panoramic images, it is not convincing to only focus on CLIP (Line 073). Given the existing of powerful Multimodal Large Language Models (MLLMs), it is natural to ask whether these MLLMs can serve as the evaluator. 

- Line 064 states "raises questions about their applicability to evaluating 360-degree panoramic image-text pairs, which present fundamentally different characteristics". Did the authors justify that 360-degree panoramic image-text pairs "present fundamentally different characteristics" from CLIP's pretraining data? Do the CLIP's pretraining data contain panoramic images? This is an important point, as the analysis of CLIP's understanding ability on panoramic images can be greatly determined by whether the CLIP's pretraining dataset contains panoramic images and the amounts / portion of such images

- Line 035 and 068 mention "360x180". What does 180 mean?

- In Equation (1), it is unclear why multiplying the constant 100. It is unclear either why using a max operation. The paper does not explain these

- Table 1 presents some statistical testing results but does not explain what data are used in the test. Specifically, it states that it uses "two paired image-text datasets (360_real and 360_syn) but does not explain how they are constructed and what data they contain. (Okay, the paper talks about them in later text, Section 4.1. But the current presentation causes confusions, i.e., using the acronyms without defining them.)

- It is questionable whether the current design of beta (Line 240) is a good choice. It is based on the difference of CLIP scores between images and their left-right flips. To align with horizontal shifts in the study of 360-degree visual semantic, isn't it a better choice to properly horizontally shift images to derive beta?

- The paper uses LoRA to adapt CLIP models for gaining a robust understanding of understanding of 360-degree visual semantics. But the paper does not explore how it helps evaluate generated panoramic images, which is the theme of the paper (Line 049).

### Questions
The reviewer asks the authors to address each point in weaknesses listed above and does not repeat them in this Questions box.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper systematically investigates the semantic understanding capability of the pre-trained CLIP model regarding 360° panoramic images and related textual descriptions, particularly its limitations in circular/360-degree visual semantics. The authors find that horizontal circular shifts of panoramic images cause significant fluctuations in the CLIP score, indicating the model's lack of intrinsic understanding of the geometric property of such images. To address this issue, the authors propose a lightweight fine-tuning framework based on Low-Rank Adaptation (LoRA), designed to enhance the CLIP image encoder's perception of 360° visual semantics.

### Strengths
1. The problem is well-defined and the motivation is clear: The research addresses a practical limitation of CLIP in panoramic vision tasks, which is highly relevant. Through a series of methods and designs, the authors measure the model's sensitivity to both semantics (e.g., 360°-related cues in prompts) and visual features in panoramic images. The conclusion—that CLIP is more sensitive to semantic information but less capable in capturing and extracting panoramic vision features—is reasonable and logically sound.

2. The method design is clever: Using image augmentation and incorporating constraints effectively enhances CLIP's feature extraction capability for 360° panoramic images and improves semantic-visual alignment.

### Weaknesses
The main weakness of this work is the lack of demonstrated applicability in downstream scenarios and insufficient experimental results, which primarily rely on inspection. Firstly, providing more quantitative results, rather than the binary 0/1 outcomes from the inspection, would be more persuasive. Secondly, it would be significantly better to show positive results on downstream tasks, such as 360° image retrieval/generation or visual question answering. If show promising results, I will consider raise my score.

### Questions
1. The current experiments focus on indoor and cityscape panoramas (e.g., Laval). It is recommended to test the model's generalization on more diverse scenes, such as natural landscapes or dynamic environments.

2. The current method only fine-tunes the image encoder. Since 360° semantics also involve textual understanding, future work could explore jointly fine-tuning the text encoder or designing more fine-grained text prompts.

### Soundness
3

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
4

### Summary
This paper explores the CLIP models' comprehension abilities of 360-degree image's visual and textual semantics. To probe CLIP's such visual and textual semantics, the paper set up an evaluation method in two ways: keyword manipulation for textual cues, and horizontal circular shift of 360-degree images for visual cues. Based on these evaluations, first CLIP is able to exploit the textual cue such as "360-degree image" for a better alignment with the corresponding image. However, second, The invariance does not robustly hold with respect to the image's circular horizontal shift, indicating CLIP's limited comprehension of 360 degree visual semantics. To remedy this, authors propose to fine-tune the CLIP models by introducing LoRA adapters on visual encoders to introduce invariance to such horizontal shifts in the images. Experiments show the improve performance of understanding visual cues of 360-degree circular shifts across different CLIP models.

### Strengths
This paper investigates the CLIP's understanding capabilities of textual and visual semantics, which is underexplored in literature and derived a meaningful observations and solution. Specifically, the findings that (1) CLIP can exploit "360-degree image-specific" cues in the text prompt, rather than a generic prompt like "a photo of" for a better alignment to its image, and (2) CLIP's alignment especially lacks invariance to 360-degree image's circular horizontal shift would be valuable. All these presentation are clear in the draft and writing is well-organized.

### Weaknesses
While the motivation and observation presented in the paper are quite strong, but the major concern lies in the technical side. The devised solution to improve the CLIP's understanding of visual semantics appears too straightforward and leads to expected results; fine-tuning CLIP on shifted images can naturally enhances its robustness to such shifts during inference. In addition, although the method adopts fine-tuning with LoRA, it is important to include different fine-tuning methodologies as well, such as full fine-tuning of both encoders or each encoder individually, to measure the respective effectiveness. Further reasoning and analysis for the fine-tuning part are expected to strengthen its technical solidity of this work.

### Questions
Based on the weaknesses stated above, further analyses on the fine-tuning methodology and the fine-tuned models are expected. First, how the different fine-tuning strategies affect the image and text understanding capabilities in 360-degree images? Second, regarding robustness in visual semantics, can the fine-tuned models generalize to shift magnitudes unseen during training (e.g., when the horizontal circular shift applied at test time exc eeds the range observed during training)? In addition, one natural question is that does the scale of pre-training data of CLIP models affect the robustness of visual and textual semantics understanding after fine-tuning?

### Soundness
4

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
4

### Summary
This paper probes whether CLIP understands 360° panoramic semantics.

It defines two new notions:  textual semantics (explicit cues like “360 photo”) and visual semantics (invariance under circular shifts) — and tests them via statistical analysis.

Results show CLIP models rely on textual cuse but fail to maintain shift invariance.

A LoRA-based fine-tuning improves robustness but slightly degrades baseline performance
.

### Strengths
- This work clearly defines 360-degree textual semantics and 360-degree visual semantics, addressing a **novel and underexplored problem**.

- The presentation is clear and concise, making it easy to grasp the main ideas quickly.

- The proposed LoRA-based fine-tuning effectively instills shift invariance for 360-degree panoramic scenes.

- The paper provides comprehensive experiments and analyses to support its claims.

### Weaknesses
1. The overall method is complete and well studied. But the main concern is that the explored 360-degree visual setting represents a **relatively narrow scenario** and can be viewed as a special case of standard 2D images.

2. The proposed LoRA-based tuning is a commonly used technique, and thus the methodological novelty appears limited.

3. The paper relies on the original CLIP model, which is **somewhat outdated**; incorporating comparisons with more recent models such as SigLIP-V2 or Qwen-VL would strengthen the analysis.

4. While the experimental results are solid, they primarily serve as confirmatory findings rather than revealing deeper insights or unexpected behaviors.

### Questions
1. Would the same findings hold for SigLIP or multimodal LLMs (e.g., CLIP-based vision towers in LLaVA or Kosmos-2)?

2. Any visualization or interpretability on why CLIP fails under shifts?

### Soundness
2

### Presentation
3

### Contribution
2
