# Robust onion: Peeling Open Vocab Object Detectors Under Noise

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 4, 6, 8

## Abstract
The impact of real-world noise on Open Vocabulary Object Detectors (OV-ODs) is constrained by their architectural complexity and the scarcity of noise-annotated datasets. Our empirical analysis, Robust Onion, uses controlled synthetic visual degradations to mirror feature collapse of real-world noises and systematically peel apart OV-OD components to assess their robustness. Our findings include: Similar vision backbones show comparable robustness, driven by identical feature collapse at similar layers. Pretraining, architectural nuances, and captions contribute little to robustness. Robustness relies strongly on the image domain rather than on annotations, explaining the similar impact of COCO and LVIS on robustness (same images, different annotations), and how datasets like ODinW-13, with large, isolated objects, can give a misleading impression of high robustness. These insights point to potential research on cross-layer feature exchange and continual learning strategies to improve robustness efficiently. Our findings highlight critical directions for designing robust OV-ODs under challenging visual degradations

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the robustness of Open-Vocabulary Object Detectors (OVODs) under real-world noise and visual degradation.
The authors introduce Robust Onion, a systematic framework that “peels apart” different OVOD components to analyze their resilience to noise using controlled synthetic distortions. Through empirical analysis across multiple architectures and datasets, the paper finds that:

1. robustness is driven mainly by the image domain rather than annotations.

2. similar backbones exhibit comparable robustness due to shared feature collapse patterns.

3. pretraining details and captions contribute little to noise robustness.

4. common benchmarks such as ODinW-13 may give a misleading impression of robustness.

These insights highlight the need for new strategies such as cross-layer feature exchange or continual learning for building noise-tolerant OVODs.

### Strengths
- Comprehensive evaluation across multiple models and datasets under diverse visual distortions.

- Clear empirical dissection of factors (architecture, pretraining, annotations) affecting robustness.

- Rich analysis with quantitative and qualitative visualization results.

- Identifies key limitations of current benchmarks (e.g., ODinW-13) and provides valuable diagnostic insights.

### Weaknesses
- Lacks quantitative comparison against recent SOTA robust or noise-aware OVD methods.

- The motivation for studying robustness under visual noise could be better connected to real-world deployment scenarios.

- Analysis-heavy paper without a concrete methodological contribution or design proposal.

- No clear theoretical or mathematical formulation for the proposed “robust design direction.”

- Missing comparison with input-level denoising or data augmentation baselines.

### Questions
- How does Robust Onion compare quantitatively to recent noise-aware or robust OVD baselines beyond ODinW-13?

- What are the key real-world deployment scenarios where robustness under visual noise is most critical?

- Can the insights from this analysis lead to a concrete training or architectural strategy for improving OVD robustness?

- How does the proposed analysis differ in impact from simpler input-level methods such as denoising or augmentation?

### Soundness
2

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
This paper presents robustness analysis of open-vocabulary object detectors (OV-ODs) under common visual corruptions like pixelation, motion blur, and turbulence. The authors propose the Robust Onion framework, which isolates how different model components contribute to robustness. Evaluating six models (e.g., GLIP, MM-GDINO, GLEE) across COCO, LVIS, ODinW-13, and Wider Face, the study finds that backbone depth, not fine-tuning or caption supervision, dominates robustness behavior. It also introduces two lightweight continual learning strategies (LR-TK0+ and LR-TK0++) to improve robustness in zero-shot settings.

### Strengths
S1. The paper introduces a clear analytical framework to “peel” away layers of complexity and pinpoint which components of OV-OD models contribute to or detract from robustness. By systematically turning off or swapping certain components (e.g. using a frozen vs. fine-tuned backbone, or evaluating with vs. without caption-based training), the authors can attribute robustness (or lack thereof) to specific factors. This level of analysis moves beyond treating the model as a black box – it provides insightful breakdowns of how different stages (backbone, detector head, multimodal fusion, etc.) behave under noise. 

S2. The study delivers some important insights. One of them, the discovery that backbone depth/capacity is the primary driver of robustness (more so than fine-tuning or the richness of text supervision) is a interesting fact for further work.

S3. The paper makes a practical contribution by proposing simple fine-tuning strategies (LR-TK0+ and LR-TK0++) that yield noticeable robustness gains. These strategies are lightweight (avoiding full model retraining) and thus would be attractive for practitioners looking to harden existing OV detectors against noise. The fact that a gradual fine-tuning on noisy data (LR-TK0++) outperforms a naive augmentation approach is a useful takeaway for the field.

S4. Writing and structuring of the paper is easy to follow.

### Weaknesses
W1. Contribution - While the analysis is detailed, the paper’s contributions are primarily empirical. The lack of a strong algorithmic or theoretical innovation prevent the exact take home knowledge to advance the field of OVOD. The proposed robustness fixes (LR-TK0+/TK0++) are relatively simple fine-tuning heuristics rather than fundamentally new methods which means, the contributions in applied and theoretical research are limited. 

W2.  Comparisons with Prior Work: The paper does not explicitly situate itself against closely related robustness studies. For instance, Chhipa et al. (2024) [1] evaluates open-vocabulary detectors (OWL-ViT, YOLO-CLIP, Grounding DINO) under distribution shifts and corruptions finding significant performance drops as well. Similarly, in the broader object detection literature, there have been benchmarks for robustness to corruptions (e.g. COCO-C and BDD100K-C) introduced by Liu et al. (2024) [2]. These works revealed, for example, that even high-mAP detectors can be very brittle and that transformer-based detectors may handle corruptions better than older architecture. It is suggested to have combined reasonable analysis with such studies and provide clear insights your findings.

[1]  Chhipa, Prakash Chandra, et al. "Open-Vocabulary Object Detectors: Robustness Challenges Under Distribution Shifts." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024

[2] Liu, Jiawei, et al. "Benchmarking object detection robustness against real-world corruptions." International Journal of Computer Vision 132.10 (2024): 4398-4416.

W3. This paper propose fine-tuning strategies (LR-TK0 series) but do not compare them to alternative robustness interventions like standard data augmentation training or adversarial training. It’s mentioned that LR-TK0++ beats “random augmentation” (presumably LR-TK0+), but a stronger baseline could be full data augmentation during initial training (for instance, training the detector on corrupted images from scratch or heavy augmentation schedules). Evaluating such a baseline would show how far the proposed lightweight approach is from what one could achieve with more extensive retraining.

W4. The claim that backbone depth alone drives robustness might be somewhat oversimplified. There is a correlation in their results, but correlation does not guarantee causation. Deeper models often also differ in other aspects: e.g. architecture family (CNN vs Transformer), pre-training dataset size, or training strategies. It’s possible that the robustness comes from some of these factors (for example, transformer-based detectors might inherently be more robust to certain perturbation. I suggest to provide convincing argument for that.

W5. Scope of Noise Types: The study covers three synthetic noise types (pixelation, gaussian blur, turbulence). These mainly represent low-level distortions blurring or obscuring the image. While these are important, the robustness problem has other dimensions that the paper does not address – for example, illumination changes, weather effects (rain, fog).

### Questions
Please refer weakness section. I can change the score based on rebuttal's response.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Robust Onion, an extensive empirical analysis of Open-Vocabulary Object Detectors (OV-ODs) under visual noise and distortions. The work introduces a systematic framework to “peel apart” different components of OV-ODs (backbones, fusion modules, pretraining datasets, fine-tuning strategies, etc.) using controlled synthetic degradations that approximate real-world noise such as turbulence, motion blur, and pixelation. The authors evaluate six prominent models (GLIP, FIBER, MM-GDINO, GLEE, YOLO-World, and RegionCLIP) across multiple datasets (COCO, LVIS, ODinW-13). Key findings suggest that robustness is primarily driven by the vision backbone, particularly its depth and scale, while language features, annotations, and pretraining data contribute little. The analysis also highlights that robustness correlates with object size and domain rather than annotation type, and that prompt engineering or caption expressiveness has minimal effect. Finally, the authors propose LR-TK0+ and LR-TK0++, lightweight continual learning extensions designed to enhance robustness in zero-shot settings.

### Strengths
Provides a comprehensive empirical dissection of robustness factors in open-vocabulary object detection, covering multiple architectures, datasets, and controlled noise settings.

The experimental setup is methodologically clear and systematic, using synthetic degradations to emulate real-world noise with qualitative and quantitative alignment (as shown in Figures 1–3).

Offers important and actionable insights, such as the dominance of backbone features in determining robustness, limited effect of pretraining size or language inputs, and the misleading robustness impression given by ODinW-13 due to large-object bias.

Introduces lightweight continual learning strategies (LR-TK0+, LR-TK0++) that show measurable improvement on COCO and WiderFace without retraining full models, demonstrating practical applicability.

### Weaknesses
The paper does not clearly position itself in relation to previous robustness studies. For instance, Chhipa et al. (2024) [1] evaluated open-vocabulary detectors including OWL-ViT, YOLO-CLIP, and Grounding DINO under distribution shifts and common corruptions, reporting significant performance drops across models. Similarly, Liu et al. (2024) [2] introduced robustness benchmarks such as COCO-C and BDD100K-C in the broader object detection literature, showing that even high-mAP detectors can be fragile, while transformer-based architectures generally perform better under corruptions. It would strengthen the paper to connect its analysis with these prior works and clarify how its findings extend or differ from them.

Evaluation noise types are somewhat narrow, focusing primarily on pixelation, turbulence, and motion blur, with little exploration of other real-world distortions (e.g., rain, snow, fog).

[1] Chhipa, Prakash Chandra, et al. (2024) "Open-Vocabulary Object Detectors: Robustness Challenges Under Distribution Shifts." European Conference on Computer Vision.

[2] Liu, Jiawei, et al. "Benchmarking object detection robustness against real-world corruptions." International Journal of Computer Vision, 132 (10), 4398-4416. (2024)

### Questions
Please follow the strengths and weaknesses. 
I am open to adjusting scores.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work evaluates the effect of noise on open-vocabulary object detectors, similar to ImageNet-C and CIFAR-C for classification. The authors analayze numerous detectors across many axes. Interesting findings include robustness being correlated to backbone, and that robustness is more correlated to images than corresponding annotations; while pre-traiining  and captions matter little. They propose a solution, training spatial tokens, on corrupted inputs to improve robustness.

### Strengths
* The analysis is thorough and interesting. I reallt enjoyed reading all of Section 4, and I found the key findings, such as sensitivity to backbone, to be very interesting. 
* The presentation is mostly very good. 
* There was a need for a such a study in the literature; object detectors are related to yet different from classifiers, and so one might see different behaviour.

### Weaknesses
* Although the paper is mostly well written, Section 5 became difficult to parse. For example in line 450 "an existing approach for low-resolution clas- sification (preserves zero-shot)"; what does preserves zero-shot mean here?
* It seems like the solution in Section 5 trains on corrputions. This seems to be  training on the test domain, in which case this ceases to properly test generalization. 
* Some findings are not suprising; like larger objects being more robust. It's more difficult to corrupt the structure!

### Questions
* When visual backbones are shared, couldn't they have the same visual pre-training? So maybe it's not suprrising that they would be correlated in robustness.
* Will a benchmark be released?
* For classifiers there is a concept of expected calbiration error for measuring decreased confidence in ood examples. Is there a similar metric that could be shown here? It seems like this could be an good measure of 'graceful degradation'.

### Soundness
3

### Presentation
3

### Contribution
3
