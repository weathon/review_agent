# RED: Robust Event-Guided Motion Deblurring with Modality-Specific Disentangled Representation

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Event cameras provide sparse yet temporally high-resolution motion information, demonstrating great potential for motion deblurring. 
However, the delicate events are highly susceptible to noise. Although noise can be reduced by raising the threshold of Dynamic Vision Sensors (DVS), this inevitably causes under-reporting of events. Most existing event-guided deblurring methods overlook this practical trade-off. The modality-indiscriminate feature extraction and naive fusion treat images and events as statistically similar inputs, leading to unstable and mixed representations, especially when events are disrupted. To tackle these challenges, we propose a Robust Event-guided Deblurring (RED) network with modality-specific disentangled representation. First, we introduce a Robustness-Oriented Perturbation Strategy (RPS) that mimics various DVS thresholds, exposing RED to diverse under-reporting patterns and thereby fostering robustness under unknown conditions. To better exploit partially disrupted events, we design a Modality-specific Representation Mechanism (MRM) that disentangles the inputs into three complementary components: an image-semantic representation capturing structure and textures, an event-motion representation extracting fine-grained motion details, and a cross-modal representation modeling complementary interactions. Building on these reliable features, two interactive modules are presented to enhance motion-sensitive areas in blurry images and inject semantic context into under-reporting event representations.
Extensive experiments on synthetic and real-world datasets demonstrate RED consistently achieves state-of-the-art performance in terms of both accuracy and robustness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Considering the trade-off between noisy fake event and under-reporting of events, this paper proposes a robust event-guided deblurring network with modality-specific disentangled representation. It improves the network robustness to different DVS thresholds (diverse under-reporting patterns) with a perturbation strategy. Besides, it presents the blurry image and partially disrupted event data with modality-specific representations. It also introduces two interactive modules to enhance motion-sensitive areas in blurry images.

### Strengths
1, It proposes a robust event-guided motion deblurring framework, which is trained with various DVS thresholds.

2, It disentangles modality-specific representation for image and event data. These two modalities interacts and benefits each other with two modules: one for improving spatial details in images and the other one for enhancing semantic information in event encoding.

### Weaknesses
Overall, this paper is well-written and its designs are validated with various experiments. However, I have some concerns in terms of the core contributions.


1, The perturbation strategy looks like a data augmentation trick.

2, The novelty of disentangleed image and event representations is arguable. Previous methods already follow the three steps of image feature extraction, event feature extraction and cross-module aggregation.

3, The cross modality attention is the common cross attention. In addition, it is strange to compute the query and key from one modality, and obtain the value from another modality. Generally, key and value should come from one modality.

4, The abstract could be improved with more details on the modality-specific representations.

### Questions
Please see weakness.

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
This paper introduces a robustness-oriented perturbation strategy (RPS) for event-based vision networks, aiming to improve their resilience under event under-reporting or dropout conditions. RPS dynamically simulates changes in the DVS output threshold to enable the network to adapt to diverse dropout patterns.

To further address feature unreliability caused by indiscriminate fusion or the absence of weak events, the authors propose a modality-specific representation mechanism (MRM) that disentangles semantic, motion, and cross-modality features. Two specialized submodules enable “coadjutant interactions”:

1. Motion Saliency Enhancer Module (MSEM) – strengthens motion-related spatial details often lost in blur.

2. Event Semantic Engraver Module (ESEM) – transforms semantic cues from images into deep event embeddings, mitigating semantic degradation caused by sparse events.

The paper also provides a probabilistic analysis of noise aggregation, modeling photon arrival with a Poisson process and circuit noise as Gaussian. Through comprehensive ablation and cross-model studies on several benchmarks, the proposed approach shows clear improvements in PSNR and SSIM, confirming both robustness and generalization.

### Strengths
The RPS module introduces a practical, model-agnostic way to simulate and handle event dropouts, effectively bridging the gap between synthetic and real-world noise conditions.

RPS can be easily integrated into other frameworks (e.g., MAT, AHDINet) and consistently improves their performance without architectural redesign.

The disentanglement design via MRM, combined with MSEM and ESEM, leads to well-structured multimodal representation and enhanced reconstruction fidelity.

Ablation and cross-model experiments rigorously validate the contribution of each component, demonstrating robustness across under-reporting ratios.

The model achieves superior PSNR/SSIM, especially under high dropout rates (>0.2).

### Weaknesses
Computational Cost Not Reported – The extra cost introduced by RPS (e.g., FLOPs, inference delay) is not discussed, which is crucial for real-time applications.

Limited Perturbation Scope – The experiments focus solely on under-reporting; robustness against other real-world degradations (e.g., motion blur, temporal jitter, or sensor noise) is untested.

Weak Theoretical Analysis – The mechanism by which RPS and MRM improve robustness lacks a formal mathematical explanation or interpretability study.

Partial Feature Disentanglement – Although semantic and motion features are decoupled in design, MSEM and ESEM remain interdependent through attention coupling, suggesting only dominant (not complete) disentanglement.

### Questions
How much computational overhead (in FLOPs or runtime) does RPS introduce during training and inference?

Could the RPS concept be extended to handle other sensor degradations, such as background noise or temporal jitter?

How independent are the features extracted by MSEM and ESEM? Have you visualized their activation maps to confirm disentanglement?

Can the authors provide more intuition or theoretical justification on how dynamic threshold perturbation leads to feature stability?

Would combining RPS with other noise-injection or adversarial training methods yield further robustness gains?

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
4

### Summary
The paper deals with the issue of under-reporting of events for event-based motion deblurring task, and proposes a network named RED. The paper introduces Robustness-Oriented Perturbation Strategy (RPS) to enhance the robustness and adaptability of RED to real-world conditions. A Modality-specific Representation Mechanism(MRM) is designed to explicitly model semantic understanding, motion priors, and cross-modality correlations from blurry images and events.  Two interactive modules MSEM/ESEM are presented to enhance motion-sensitive areas in blurry images and inject semantic context into under-reporting event representations.

### Strengths
1. The topic of handling under-reporting events in various DVS thresholds is an interesting topic for event-based deblurring. Bad event streams may indeed influence the fusion of two modalities.
2. It seems that the method can generalize better than others under high under-reporting ratio.
3. The figures and tables in the paper are well-organized.

### Weaknesses
1. The structure of the manuscripts is not good. Section 3.3 is too short to introduce the proposed modules. It appears that these two modules may have been combined in a way that gives the impression of being designed primarily to fulfill the workload requirement, rather than for a clear technical motivation.
2. The novelty of the MRM, as well as MSEM and ESEM, is not satisfying. The attention operations in the MRM are common modules in the field of the Transformer. Besides, there is no need for the full names of these modules to be so complicated.
3. The results in Figure 6 are not satisfying. The RED brings no visual improvement compared to other methods on this figure.
4. In real scenarios, are there situations when the under-reporting issue such high as the paper's experiments conducted?

### Questions
As listed in the "Weaknesses".

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
3

### Summary
This paper addresses the unstable deblurring performance caused by the "noise-under-reporting" trade-off of event cameras, proposing a Robust Event-guided Deblurring (RED) network. The core innovations include: designing a Robustness-Oriented Perturbation Strategy (RPS) to simulate event under-reporting patterns under different DVS thresholds, enhancing the model's adaptability to real-world scenarios; proposing a Modality-specific Representation Mechanism (MRM) to disentangle image semantic and event motion features, avoiding mixed representations; introducing two bidirectional interaction modules (MSEM and ESEM) to achieve complementary fusion of motion priors and semantic information. Experiments show that RED achieves state-of-the-art performance on both synthetic and real-world datasets, and RPS can be used as a plug-in to improve the robustness of other models. This method effectively solves the core pain point of event-guided motion deblurring, providing a robust and practical solution for deblurring in high-dynamic scenarios.

### Strengths
1.This paper systematically addresses the "noise-under-reporting" trade-off of event cameras. The RPS strategy innovatively simulates the physical mechanism of real event acquisition, the MRM realizes the disentanglement of modal features, breaking through the limitation of mixed features in existing methods, and the design of bidirectional interaction modules is highly targeted.
2. The experimental design is comprehensive, covering multiple synthetic and real-world datasets, verifying the method's robustness under different under-reporting ratios. Ablation experiments detailedly validate the necessity of each component including RPS, MRM, and MSEM/ESEM. The plug-and-play generality of RPS is verified by integrating it into other models, with highly credible conclusions.
3. The paper has a coherent structure, progressing layer by layer from problem analysis, method design to experimental verification. Framework diagrams and performance curves intuitively show the method flow and advantages. Technical details (such as the probabilistic modeling of RPS and the attention mechanism of MRM) are elaborated in detail, and the literature review is comprehensive, facilitating the understanding of the research background.

### Weaknesses
1. Suboptimal computational efficiency: The parameter count (19.2M) and computational complexity (637.45G FLOPs) of RED are relatively high. Compared with lightweight models (e.g., EFNet: 7.73M parameters, 379.43G FLOPs), there is a lack of efficiency-performance trade-off analysis.
Insufficient depth of modal interaction mechanism: The feature fusion methods of MSEM and ESEM are relatively simple (such as element-wise multiplication and concatenation), and do not consider the dynamic adaptation between the degree of event under-reporting and image blur intensity. The cross-modal attention of MRM does not distinguish the differential adaptation of different motion types (e.g., rigid body motion, non-rigid body motion).
2.Incomplete coverage of extreme scenarios: Experiments do not involve extreme scenarios such as complete event loss, coexistence of strong noise and high under-reporting, and ultra-high-definition images (e.g., 4K). The adaptability of the method to different event camera models (different DVS threshold characteristics) is not evaluated.
3. Limited depth of core innovation: "Perturbation training to improve robustness" and "modality-specific feature extraction" are already mature ideas in the fields of image restoration and cross-modal fusion. The innovation of RED is more about the adaptive combination of existing ideas in event-guided deblurring tasks, lacking breakthrough paradigm innovation.

### Questions
1. What is the performance of RED in high-noise scenarios with low DVS thresholds? Can supplementary comparative experiments under different noise intensities be provided to illustrate its advantages in noise robustness compared to methods such as EFNet and AHDINet?

2. Can the feature fusion weights of MSEM and ESEM be dynamically adjusted according to the degree of event under-reporting and image blur intensity? How adaptable is MRM to different motion types (rigid/non-rigid), and is targeted optimization required?

### Soundness
3

### Presentation
3

### Contribution
3
