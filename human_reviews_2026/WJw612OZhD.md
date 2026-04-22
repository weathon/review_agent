# VisCoP: Visual Probing for Video Domain Adaptation of Vision Language Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Large Vision Language Models (VLMs) excel at general visual reasoning tasks, but their performance degrades sharply when deployed in novel domains with substantial distribution shifts compared to what was seen during pretraining. Existing approaches to adapt VLMs to novel target domains rely on finetuning standard VLM components. Depending on which components are finetuned, these approaches either limit the VLM’s ability to learn domain-specific features or lead to catastrophic forgetting of pre-existing capabilities. To address this, we introduce **Vis**ion **Co**ntextualized **P**robing (**VisCoP**), which augments the VLM's vision encoder with a compact set of learnable *visual probes*, enabling domain-specific features to be learned with only minimal updates to the pretrained VLM components. We evaluate VisCoP across three challenging domain adaptation scenarios: cross-view (exocentric → egocentric), cross-modal (RGB → depth), and cross-task (human understanding → robot control). Our experiments demonstrate that VisCoP consistently outperforms existing domain adaptation strategies, achieving superior performance on the target domain while better retaining capabilities from the source domain. We will release all code, models, and evaluation protocols to facilitate future research in VLM domain adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces VISCOP (Vision Contextualized Probing), a method to adapt vision-language models (VLMs) to new domains—like egocentric video, depth imagery, or robotic control—without fine-tuning the frozen vision encoder. Instead, VISCOP uses a small set of learnable "visual probes" that interact with intermediate features of the frozen encoder to extract domain-specific visual cues, allowing the model to retain its original capabilities while excelling in the new domain. Experiments across three challenging adaptation tasks show that VISCOP outperforms traditional fine-tuning approaches by achieving higher target-domain accuracy while avoiding catastrophic forgetting of source-domain knowledge.

### Strengths
- VISCOP introduces a novel, parameter-efficient mechanism for domain adaptation in VLMs by using learnable visual probes that interact with intermediate visual hiddens, avoiding catastrophic forgetting while enabling strong target-domain performance.
- The method achieves state-of-the-art results across three challenging domain adaptation scenarios—cross-view (exocentric to egocentric), cross-modal (RGB to depth), and cross-task (human action to robotic control).
- Empirical analyses, i.e., attention visualizations, t-SNE embeddings, and ablations on probe count and layer interaction, demonstrate that VISCOP captures domain-specific visual features effectively.

### Weaknesses
- While VISCOP avoids finetuning the vision encoder, the method’s dependency on a fixed, frozen encoder limits its ability to exploit domain-specific low-level features that might be better learned through encoder updates.
- The paper lacks ablation on probe initialization methods and alternative interaction module architectures beyond cross-attention.
- No analysis is provided on inference latency or memory overhead introduced by the visual probes.
- Real-world robotic evaluation uses a small, non-public dataset (xArm-Det) with limited documentation on object diversity, pose variation, and camera calibration, reducing reproducibility

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
VISCOP introduces cross attention based probing approach for domain adaptation of video understanding in VLMs. VISCOP augments a frozen vision encoder with a compact set of learnable visual probe modules that interact layer-wise with intermediate feature representation via cross attention. It helps to extract domain-specific visual attributes without modifying original model. The method evaluated three diverse domain adaptation scenarios and perform competitively on the target domain while better retaining source domain performance.

### Strengths
- **Originality and Significance**: The method of using layer-wise visual probes to selectively extract new, domain-specific visual features from a frozen encoder is an excellent, novel approach that effectively decouples new learning from pre-trained knowledge, thereby successfully mitigating catastrophic forgetting.

- **Empirical Performance**: VISCOP demonstrates strong, consistent performance gains across all three complex domain adaptation tasks (cross-view, cross-modal, cross-task). It successfully achieves the best balance of target and source domains.

- **Interpretabilty**: VISCOP provided diagnostic studies including attention visualization, ablation studies which illustrate how the probes learns distinct and domain relevant representation that base VLM frozen envoder failed to capture.

### Weaknesses
- **Data selection**: In Table1-3, base VLM performa well in many of the target domain; This raises concern 1) potential overlap of source domain (and target domain) with the pretraining data of VLM 2) Substantial distribution shift might be less severe than claimed. Could authors explain, how these datasets selected?

### Questions
- Can the authors discuss the robustness of this optimal probe count and initialization across the diverse domain shifts? 
- Does the optimal number of probes change significantly based on the severity of the distribution shift?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces VisCoP, a method for adapting Vision-Language Models (VLMs) to new video domains without catastrophic forgetting. In particular, from popular exocentric videos to egocentric videos and even for robot control and other visual modality. Then, the paper introduces a small set of learnable visual probes, which is used to learn domain-specific features, and preserve existing knowledges. In detail, without changing the output of pre-trained backbone, both the output from a probing network (cross attention) that corresponding to the probes are concatenated and fed into LLM together for downstream application, which seems pretty intuitive in method design. But still, the application task is indeed very important and challenging.

### Strengths
- The cross-domain application task for video analysis is very important, where the three application scenarios, particular for robot control, are comprehensive.
- Comprehensive evaluation and ablation studies on three domain transfer tasks.
- Both qualitative and quantitative analysis are provided.

### Weaknesses
- Despite comprehensive ablation studied, It looks like that the proposed method does not compare the other baselines clearly. As such, more comparison with (including repurposing any other methods, or methods that only focus on limited field transfer) is needed.
- Meanwhile, compared with prompting-based methods, the proposed methods clearly introduces more parameters and requires much more computation resource. However, a detailed explanation is still needed to let the audience check whether the surge in computation is still within a reasonable range. 
- The novelty of the method design is limited. The proposed method has been studied in the image or language fields. For generalization over videos, I was expecting some video-specific techniques that can help with proposed evaluation setup, either in form of training objective or network design. 
- Followingly, how do you guarantee that the outputs of probe will be exactly the domain-specific information for the new domains. How to you distinguish your methods from another baseline that just augments the network with more parameters? After all, I didn’t see clear optimization objective for the motivation mentioned in the introduction.

### Questions
Just out of curiosity, For line 402-403, “the resulting in 0% accuracy across all levels of VIMA-Bench” seems surprising. More explanation on the performance, metric definition is needed.

Please see my comments in weakness.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a finetune method for VLM  when trying to improve VLM on the target domain. The proposed module VisCop share a similar design spirit with Q-former, which uses a few queries to probe the features.

### Strengths
Strength:

- This paper is well-written and easy to follow. The motivation and problem setting are clear.

- On the evaluation benchmarks, the proposed method shows general effectiveness.

### Weaknesses
Questions:

- Does $Acc_{source}^{expert}$ indicate the performance of the model after being finetuned on target domain data?

- In table 1, are Egocentric Benchmarks and Exocentric Benchmarks all target domain benchmarks?

- In table 1, why only the third method suffers from performance degeneration on the source domain? In addition, does "✓" mean fully tuning? I am curious about the root cause of source performance degeneration on the source domain. Is it fully fine-tuning LLM on the target domain?

- In table 2, Lora-tune the LLM seems to be a bad choice since it causes about 10$%$ of performance drop. So why not just train the model with only Viscop?

- When testing the process model on the source domain, will it process the source features with VisCop?

- VisCop introduces residual vision features, so what is the performance of using a LORA to introduce vision features?


Weakness:

- The technical contribution of this paper is not significant. The experiment design could be clearer to show more information. For example, to isolate the design choice that causes severe source performance generation and target performance gain.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
2
