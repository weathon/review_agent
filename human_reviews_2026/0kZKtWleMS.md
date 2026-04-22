# Concepts in Motion: Temporal Bottlenecks for Interpretable Video Classification

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 4, 6, 8

## Abstract
Conceptual models such as Concept Bottleneck Models (CBMs) have driven substantial progress in improving interpretability for image classification by leveraging human-interpretable concepts. However, extending these models from static images to sequences of images, such as video data, introduces a significant challenge due to the temporal dependencies inherent in videos, which are essential for capturing actions and events. In this work, we introduce $\textbf{MoTIF}$ (Moving Temporal Interpretable Framework), an architectural design inspired by a transformer that adapts the concept bottleneck framework for video classification and handles sequences of arbitrary length. 
Within the video domain, concepts refer to semantic entities such as objects, attributes, or higher-level components (e.g., “bow,” “mount,” “shoot”) that reoccur across time—forming motifs collectively describing and explaining actions. Our design explicitly enables three complementary perspectives: global concept importance across the entire video, local concept relevance within specific windows, and temporal dependencies of a concept over time. Our results demonstrate that the concept-based modeling paradigm can be effectively transferred to video data, enabling a better understanding of concept contributions in temporal contexts while maintaining a competitive performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work presents MoTIF, an interpretable framework for video classification that adapts the concept bottleneck model (CBM) idea to temporal data. The method introduces a transformer-like architecture with per-concept temporal attention, enabling explanations at global, local, and temporal levels. The goal is to capture semantically meaningful concepts across frames and use them to explain and predict actions.

While the paper is well-motivated and technically clear, the approach faces representational constraints. By forcing all reasoning through a small set of predefined, frame-level concepts, MoTIF loses much of the expressive capacity that makes modern end2end, space-time video transformers successful. As a result, its significantly underperforms on challenging datasets such as Something-Something v2, where end-to-end models achieve far stronger results.

### Strengths
The paper is relatively well written and easy to follow.

The proposed approach is sound. 

The resulting model offers intuitive explainability of its action classification predictions. 

A reasonable ablation study is reported.

Experiments show that, at least for frame-level visual representations, adding a CBM head does not results in a drop in performance.

### Weaknesses
The key concerns I have regarding this work is that all the results are shown for frame-level visual representations that are fundamentally limited in their ability to capture spatiotemporal concepts in action classification. As a results, the strongest variant only achieves a classification accuracy of 30% on SSv2 (compared to 76% for the current sota method on this dataset). It is unclear if the proposed approach is applicable to modern space-time transformer architectures, which makes it hard to gauge its impact on the community. To address this issue, the authors need to report results with sota space-time transformer representations on non-toy datasets (e.g. on SSv2).

In the same vein, the sota methods' results need to be moved from the appendix to Table 1 to provide better context to the reader. 

Furthermore, variance in performance form concept sampling should be reported for every backbone it Table 1 directly. This would likely make some improvements over linear probing baseline not statistically significant at least on some datasets. 

As a final comment for Table 1, the authors need to report runtime for all the variants, as this is an important factor to consider.

Evidence of the fact that diagonal attention is not introducing performance penalties are not compelling, as only toy datasets (like hmdb) and weak architectures are ablated. It would be much stronger to show this on SSv2 with top-performing backbones.

### Questions
Please results with sota space-time transformer representations on non-toy datasets (e.g. on SSv2).

Please move sota results from the appendix to Table 1 and add runtime for all the variants as well as variance from concept sampling for your method.

Please show diagonal attention ablation on SSv2 with top-performing backbones

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
This paper introduces MoTIF (Moving Temporal Interpretable Framework), a new class of concept bottleneck models (CBMs) designed to provide interpretability for arbitrary-length video data. Unlike traditional CBMs that operate on static images or short clips, MoTIF extends the framework to handle long-term temporal dependencies, enabling interpretable reasoning across entire video sequences.

The model incorporates per-channel self-attention mechanisms to ensure that individual concept representations remain disentangled, preventing the mixing or interference of different semantic concepts across channels. Furthermore, MoTIF leverages a text-encoder-based grounding framework to tie each visual concept directly to natural language descriptions, linking model internals to human-understandable concepts. Additionally, MoTIF enables local-to-global interpretability: it allows fine-grained inspection of individual frame-level or concept-level activations, while also supporting high-level explanations of model behavior over full video trajectories.

### Strengths
- The paper tackles interpretability in the video domain, an area that remains significantly underexplored. Extending concept bottleneck models to temporal data represents an important and timely contribution.
- The proposed framework effectively supports probing at multiple levels of abstraction—from localized, concept-specific attention patterns to global, sequence-level explanations—offering a comprehensive view of model behavior.
- The paper is clearly written and well organized. The methodology, implementation details, and experimental setup are described with sufficient clarity and depth, making the work easy to follow and reproducible.

### Weaknesses
The core issue I have with the manuscript is the limited modeling of dynamics in the current version.

- Despite being framed as a video CBM, the method is largely appearance-centric. The concepts shown in the figures are predominantly static nouns or one-frame cues (e.g., bread/bagel in Breakfast; kayak/paddle in UCF101), and the explanation views mostly reweight per-window concept activations rather than modeling interactions that require multi-frame reasoning. The temporal module is per-channel self-attention that forbids cross-concept interactions, which may preserve attribution but weakens the capacity to capture genuinely dynamic, relational events (i.e., spatiotemporal object centric concepts, like a ball moving to the right). As a result, the paper does not convincingly demonstrate that the core challenge that makes video harder than images, *dynamics*, is addressed to a meaningful extent. If the main benefit is “interpretable CBMs that work on videos,” the paper should show that the model captures temporal structure that single frames cannot. As written, the strongest qualitative evidence highlights local/global concept scores and attention heatmaps, but the concepts themselves are mostly static and I am unconvinced that an image CBM would perform much worse or be less interpretable. 

- Breakfast, HMDB51, UCF101 can be handled to a large degree with static cues; the model performs well there. The only dataset in the suite that truly requires temporal reasoning, SSv2, is where performance drops most sharply relative to strong non-interpretable baselines. Including non-interpretable SOTA numbers alongside Table 1 would make this gap clear; SSv2 state of the art is ~80% top-1 (e.g., MVD ≈77%), far above the reported results. This undercuts the claim that the method meaningfully tackles dynamics.

- Because concepts are LLM-generated without a video-centric constraint, there is no evidence the concept set contains temporal concepts that image CBMs would miss. It would be informative to (i) construct explicitly temporal concepts and show they are used, and (ii) demonstrate video-centric wins that image-centric CBMs cannot achieve. Looking at the concept list in Table 9, SSv2 is the only dataset that I see containing concepts which require more than one frame to understand - and is the only dataset with results far below non-interpretable baselines. 

- The evaluation focuses on accuracy. There is no metric showing that the learned internal representations capture more dynamic information than a non-interpretable counterpart. Applying a dynamics-sensitivity analysis (e.g., using [a]) would directly test whether the model is learning temporal structure, not just aggregating static evidence over time.

Small things: 

- Figure 6 caption should say UCF-101, not SSv2. 
- Table 2 should compare against removing random concepts, and/or adding k concepts (insertion vs. deletion plots). 
- Could the reason for better performance be a larger number of parameters used in the diagonal attention with many concepts compared to standard attention?

[a] Kowal et al. Quantifying and Learning Static vs. Dynamic Information in Deep Spatiotemporal Networks. (TPAMI 2024)

### Questions
- Compare against an image-CBM, like the one from Rao et al., 2024 and see if it does worse on temporal data/concepts. You could use the concepts in SSv2 that are known to be dynamic (e.g., pretending to touch smth, pushing smth left to right vs. right to left) and compare image CBMs to them, as well as the overall performance across the datasets. 
- Alternatively, explicitly showing your method captures more dynamics than non-interpretable or image CBM baselines would be interesting (i.e., using [a] or frame shuffling experiments that show how a model is dependant on temporal ordering)
- Would it be possible to include explicitly temporal concepts (e.g., “starts to pour,” “moves backward”) in the concept generation process and assess whether MoTIF can meaningfully ground them? Again, showing MoTIF outperforms on these temporal concepts would be ideal.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces MoTIF (Moving Temporal Interpretable Framework), a novel architecture for interpretable video classification. The work extends the principles of Concept Bottleneck Models (CBMs), which have been successful for images, to the video domain. The core challenge in this extension is modeling temporal dependencies while maintaining the interpretability of individual concepts. MoTIF addresses this with a transformer-inspired architecture that employs a "per-channel temporal self-attention" mechanism. This mechanism intentionally prevents the mixing of concept channels, allowing the model to reason about the temporal evolution of each concept independently.

### Strengths
- The primary contribution is the adaptation of the concept bottleneck paradigm to sequential data like video. The proposed "diagonal attention" mechanism, which uses depthwise 1x1 convolutions to create per-concept QKV projections, is a clever and effective way to model temporal dynamics without sacrificing concept-level interpretability.
- The framework is explicitly designed to produce three distinct and useful types of explanations: global, local, and temporal. This provides a much richer and more comprehensive view of the model's decision-making process than a single vector of concept importance.
- The paper presents a comprehensive set of experiments across four diverse datasets and a range of modern vision-language backbones.

### Weaknesses
- A core design choice of MoTIF is to enforce strict concept isolation via diagonal attention to maintain interpretability. However, many complex actions are inherently compositional, defined by the interactions between different concepts over time, which MoTIF cannot capture.

### Questions
This paper presents a novel and well-executed contribution. 

- In Figure 5, what is the white circle?

### Soundness
2

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
3

### Summary
The paper introduces MoTIF (Moving Temporal Interpretable Framework), an architectural design that successfully adapts the Concept Bottleneck Model (CBM) framework for video classification, handling sequences of arbitrary length. CBMs previously struggled with videos due to inherent temporal dependencies and the variable length of video data. MoTIF addresses this by incorporating transformer-inspired blocks while strictly preserving concept separation.  The primary innovation is the Per-channel temporal self-attention (diagonal attention) mechanism, which isolates temporal reasoning for each concept channel using depthwise 1x1 convolutions. MoTIF explicitly models reoccurring semantic entities (motifs) and provides three complementary explanation views: global concept importance, local concept relevance within specific temporal windows, and temporal dependency maps visualized through attention heads. Experiments on four benchmarks (Breakfast Actions, HMDB51, UCF101, and Something-Something V2) demonstrate that MoTIF achieves competitive performance while offering fine-grained and faithful explanations of concept contributions in temporal contexts.

### Strengths
MoTIF successfully extends the CBM principle from static images to sequential data. This innovation bridges the gap between interpretability and high-capacity temporal architectures

The core mechanism, per-channel temporal self-attention (diagonal attention), maintains concept independence by avoiding cross-channel mixing in the QKV projections.  Ablation studies confirm that MoTIF preserves generalization while enforcing interpretability.

MoTIF offers three crucial and complementary explanation views: (i) global concept relevance (via LSE pooling), (ii) localized temporal explanations (using windowed attributions, identifying concepts active in decisive windows), and (iii) attention-based temporal maps that visualize how each concept channel focuses across time, exposing anchor moments or consistent presence.

MoTIF achieves competitive predictive performance, surpassing zero-shot and linear-probe baselines across all four datasets. Performance improves consistently with higher capacity CLIP-based backbones

### Weaknesses
The architecture deliberately trades computational efficiency for strict concept isolation; thus, when the number of concepts (C) is large, full MHA may be more efficient. Disabling diagonal attention significantly reduced GPU memory usage and epoch time for longer videos, such as those in the Breakfast Actions dataset

The architectural design suppresses cross-concept interactions to maintain interpretability. Consequently, stacking more blocks offers limited or no richer hierarchical abstractions, which deliberately restricts the model’s expressivity compared to conventional deep transformers

MoTIF maintains competitive accuracy given its interpretability constraint. However, on challenging datasets like SSv2, a performance gap persists when compared to strong non-interpretable baselines such as VideoMAE V2.

 Open challenges remain regarding key non-trivial design choices, such as selecting the optimal window size, which depends on action duration, speed, and alignment

### Questions
How do the authors ensure the practicality and deployment efficiency of MoTIF when working with very long video sequences and a rich concept vocabulary?

Given that performance can be sensitive to the concept set, what are the recommendations for extracting richer and more action-relevant concepts directly from video data, especially for abstract classes like those found in SSv2, rather than relying solely on external language model guidance?.

### Soundness
4

### Presentation
3

### Contribution
4
