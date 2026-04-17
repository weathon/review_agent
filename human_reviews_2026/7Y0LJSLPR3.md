# Collaborative Hybrid Propagator for Temporal Misalignment in Audio-Visual Segmentation

- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Audio-visual video segmentation (AVVS) aims to generate pixel-level maps of sound-producing objects that accurately align with the corresponding audio. However, existing methods often face temporal misalignment, where audio cues and segmentation results are not temporally coordinated. Audio provides two critical pieces of information: i) target object-level details and ii) the timing of when objects start and stop producing sounds. Current methods focus more on object-level information but neglect the boundaries of audio semantic changes, leading to temporal misalignment. To address this issue, we propose a Collaborative Hybrid Propagator Framework~(Co-Prop). This framework includes two main steps: Preliminary Audio Boundary Anchoring and Frame-by-Frame Audio-Insert Propagation. To Anchor the audio boundary, we employ retrieval-assist prompts with Qwen large language models to identify control points of audio semantic changes. These control points split the audio into semantically consistent audio portions. After obtaining the control point lists, we propose the Audio Insertion Propagator to process each audio portion using a frame-by-frame audio insertion propagation and matching approach. We curated a compact dataset comprising diverse source conversion cases and devised a metric to assess alignment rates. Compared to traditional simultaneous processing methods, our approach reduces memory requirements and facilitates frame alignment. Experimental results demonstrate the effectiveness of our approach across three datasets and two backbones. Furthermore, our method can be integrated with existing AVVS approaches, offering plug-and-play functionality to enhance their performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studied the topic of audio-visual video segmentation that aims to generate pixel-lvel maps of sounding object that align with the corresponding audio. The author argue that the existing method often face temporal misalignment, where audio cues and segmentation results are not temporally coordinated.  Also, current methods mainly focus on object-level inforamtino but neglect the boundaries of audio semantic changes, leading to temporal misalignment. To address these problems. the paper propose a collaborative hybrid propagator framework (Co-Prop) to address the aforementioned problem by employing retrieval-assist prompts with Qwen LLM to identify control points of audio semantic changes. Additionally, an audio insertion propagator is implemented to process each audio portion using a frame-by-frame information propagation. The proposed method is evaluated on the AVSBench dataset and show compatitive results.

### Strengths
- The idea of tanalysis the audio to pinpoint key time points where the sound sources change is interesting.
- The paper conduct various experimental results to highligh the advantages of the proposed method.

### Weaknesses
- Overall, the presentation and writing quality of the paper are poor. The methodology section suffers from a lack of precise mathematical formulation, including a clear definition of the learning problem, the training objective, and the dimensionality of the involved tensors. This omission makes it difficult for readers to follow the technical explanation. Additionally, Figure 2 is poorly presented—particularly the audio-insert propagator module—which is hard to interpret due to unclear visual design. The methodology also omits key details, such as a visualization of the dataset, explaination of the alignment rate metrics and the specific version of the Qwen model used.

- The AVSBench benchmark used in the paper is out of date. For example [a,b], may out-perform the proposed method without using LLM.

- The paper lack qualitatiave video demonstrate of the prediction results.

- The reproducibility is limited due to the unavailability of the code.

- The major experimental results are based on the binary segmentation benchmarks (i.e., M3 and S4). To strengthen the evaluation, it would be beneficial to include experiments on the AVSS benchmark, which provides semantic labels and enables a more fine-grained assessment of model performance.


[a] Yang, Q., Nie, X., Li, T., Gao, P., Guo, Y., Zhen, C., Yan, P. and Xiang, S., 2024. Cooperation does matter: Exploring multi-order bilateral relations for audio-visual segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 27134-27143).
[b] Huang, S., Ling, R., Hui, T., Li, H., Zhou, X., Zhang, S., Liu, S., Hong, R. and Wang, M., 2025. Revisiting Audio-Visual Segmentation with Vision-Centric Transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 8352-8361).

### Questions
n/a

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an Audio-Visual Video Segmentation (AVVS) framework consisting of two main stages. In the first stage, the input audio is segmented into temporal segments based on objects, guided by a large language model with retrieval-assisted prompts. In the second stage, the model performs mask propagation from keyframes with audio-inserted feature fusion. The proposed method achieves state-of-the-art performance across three AVVS datasets and two backbones.

### Strengths
- The method demonstrates strong performance on multiple AVS datasets using two different backbones.
- The proposed approach is designed as a plug-and-play module, making it flexible and easily adaptable to existing frameworks

### Weaknesses
- The submission lacks line numbers for reference, and the writing template does not conform to the required format.
- The claim that "existing AVVS models decode all frame masks simultaneously" is inaccurate. In practice, most existing methods perform decoding within short temporal windows rather than all frames at once.
- The novelty is relatively limited. The approach combines two existing techniques: (1) retrieving key audio and video frames using a pretrained LLM (Qwen), and (2) applying a conventional audio-guided keyframe-based mask propagation method. The proposed Audio-Insert block is essentially a standard transformer-based audio-video fusion module.
- The method lacks discussion on handling multi-source sound scenarios (when multiple sound sources occur simultaneously). While the audio is segmented temporally, it is not separated by source, which may negatively impact audio segmentation and mask propagation quality.
- The RCPG module heavily depends on the LLM’s ability to identify accurate keyframes. If the keyframe selection is incorrect, the subsequent AIP module (which relies on mask propagation from those keyframes) is likely to fail.

### Questions
See weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The manuscript proposes a novel Co-Prop framework to alleviate temporal misalignment, where segmentation masks persist even after the sound has stopped. The method first leverages the Qwen multimodal large language model to convert audio into text and semantics, which are used to retrieve key frames where the sound source is present. It then trains an Audio-Insert Propagator that fuses the current frame’s audio features with video features and propagates masks to the remaining non-key frames. The proposed method outperforms several prior methods.

### Strengths
1). The paper formalises the Temporal Misalignment Issue in audio-visual segmentation, where masks persist even after the sound has stopped.
2). The writing is clear, the method is easy to follow, and the problem–solution flow is coherent.
3). The method delivers competitive results across several benchmarks.
4). Beyond standard MJ/MF metrics, the paper introduces a time-alignment–oriented metric (e.g., alignment rate) and a small Multi-source Onset Conversion (MOC) set, making improvements more interpretable.

### Weaknesses
1).The contribution feels incremental: similar pipeline-style “AI workflow” papers exist [1–3], and even within AVSS/AVVS there are methods that convert audio into textual/context cues and then filter or guide frames [1]. The paper should clarify what is fundamentally new beyond this recipe.

2). Results are not especially compelling and several strong baselines [A], [B], [C] are missing.  As a result, the central claim about the importance of “temporal misalignment” is not convincingly supported.

3).The method is built on an older propagation architecture. Recent vision foundation models (e.g., SAM 2) seem directly applicable to this framework, yet are neither compared nor discussed. At minimum, the paper should justify this choice or provide an integration/ablation.

4). The two-stage pipeline increases optimization and engineering burden. Please add details on curriculum, losses, stage scheduling, and typical failure modes; otherwise it’s hard to assess practicality.

5). Refer to reproducibility concerns. Code and data artifacts (key-frame selection, retrieval set, prompts) are not clearly stated to be released. Without these, reproducing the key results is difficult.

[1] Unleashing the temporal-spatial reasoning capacity of gpt for training-free audio and language referenced video object segmentation.  
[2] Open-Vocabulary Audio-Visual Semantic Segmentation
[3] Retrieval-Augmented Generation for AI-Generated Content: A Survey
[A] Stepping stones: A progressive training strategy for audiovisual semantic segmentation 
[B] Cooperation does matter: Exploring multi-order bilateral relations for audiovisual segmentation
[C] Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics

### Questions
1). Which Qwen multimodal large language model (e.g., model name/version, parameters) is used? Please add the exact citation and report key settings (prompt template, temperature, decoding, and whether ASR is involved).
2). The implementation details need more explanation: architecture of the Audio-Insert/propagation unit (attention type, Q/K/V sources, feature dims, fusion layer(s)), how masks are encoded/updated, loss terms and training schedule, initialization, and compute/memory cost.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper tackles a important problem in AVVS: temporal misalignment. The core idea is to preemptively segment the audio timeline and then propagate masks within these semantically consistent chunks. To address this, the authors propose Co-Prop, a two-stage framework. First, an Audio Boundary Anchoring stage uses a large language model with a retrieval-augmented prompt to identify "control points" in the audio where the sound source changes. Second, a Frame-by-Frame Audio-Insert Propagation stage uses a dedicated model to segment the first frame at each control point (the "keyframe") and then propagates this mask forward, injecting audio features at each step, until the next control point. The writing is relatively clear, and the results on the benchmarks are strong. However, I have significant reservations about the methodological choices and the experimental validation, which currently hold back what could otherwise be a very compelling contribution. The framework feels a bit like using a sledgehammer to crack a nut, potentially effective, but unnecessarily complex and fragile.

### Strengths
Focusing on temporal misalignment is spot-on. This is a real pain point in the field that most methods gloss over, and explicitly tackling it is a great direction. The idea of decoupling the problem into "find the audio segments" and then "propagate within them" is original and intuitive. Using an LLM to reason about audio semantics for boundary detection is a creative, if heavy-handed, approach.

### Weaknesses
The absence of key 2025 supervised works (like the CVPR 2025 RAVS paper) is a significant omission. Furthermore, including MoCA—an unsupervised, ultra-lightweight model—alongside supervised heavyweights without explicitly highlighting this fundamental difference feels like an unfair comparison setup. It leaves me wondering if the gains are truly state-of-the-art or just a result of a curated comparison. The entire framework is a chain of critical dependencies. If the LLM misses a control point or the Keyframe Processor botches the initial mask, the error is locked in and propagated faithfully by the AIP. The authors acknowledge this in the limitations, but it's a fundamental architectural weakness that limits robustness. Using a giant LLM like Qwen just to find audio boundaries seems excessive. The paper lacks an ablation against a simpler, dedicated audio event detection model. Without this, it's hard to justify the added complexity, computational cost, and dependency on a proprietary model.

### Questions
1. Could you please comment on the omission of other recent 2025 supervised works like RAVS? Additionally, could you clarify the rationale for including the unsupervised MoCA model in the main comparison table and discuss how its different learning paradigm affects 2. Could you provide the total parameter count for the trainable parts of Co-Prop (KPF + AIP)? Also, is the Qwen LLM used during training, or only at inference? This is crucial for a full understanding of the model's footprint.
3. You show that RCPG is better than cosine similarity, but is a full LLM truly needed? Have you experimented with a simpler, trainable audio event detection network to perform the same boundary detection task? A comparison here would greatly strengthen the argument for the LLM's role.
4. Given the pipeline's fragility, do you have any analysis of failure cases? For instance, how often does the RCPG module fail to identify a key boundary, and what is the resultant performance drop on that video segment?
5. The RCPG module's retrieval relies on a pre-processed training set. How would your method perform on a new dataset where such a curated control-point database is not available? Is the method fundamentally tied to the training domain?

### Soundness
3

### Presentation
2

### Contribution
2
