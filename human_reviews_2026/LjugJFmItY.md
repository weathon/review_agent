# PredNext: Explicit Cross-View Temporal Prediction for Unsupervised Learning in Spiking Neural Networks

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Spiking Neural Networks (SNNs), with their temporal processing capabilities and biologically plausible dynamics, offer a natural platform for unsupervised representation learning. However, current unsupervised SNNs predominantly employ shallow architectures or localized plasticity rules, limiting their ability to model long-range temporal dependencies and maintain temporal feature consistency. This results in semantically unstable representations, thereby impeding the development of deep unsupervised SNNs for large-scale temporal video data. We propose PredNext, which explicitly models temporal relationships through cross-view future Step Prediction and Clip Prediction. This plug-and-play module seamlessly integrates with diverse self-supervised objectives. We firstly establish standard benchmarks for SNN self-supervised learning on UCF101, HMDB51, and MiniKinetics, which are substantially larger than conventional DVS datasets.  PredNext delivers significant performance improvements across different tasks and self-supervised methods. PredNext achieves performance comparable to ImageNet-pretrained supervised weights,  through unsupervised training solely on UCF101. Additional experiments demonstrate that PredNext, distinct from forced consistency constraints, substantially improves temporal feature consistency while enhancing network generalization capabilities. This work provides a effective  foundation for unsupervised deep SNNs on large-scale temporal video data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PredNext, a novel self-supervised learning framework for deep Spiking Neural Networks (SNNs) on video data. PredNext explicitly models temporal relationships through cross-view future feature prediction, incorporating both step-wise and clip-level prediction mechanisms to enhance semantic consistency and improve feature generalization. Designed as a modular plug-and-play component, PredNext can be seamlessly integrated with a variety of existing self-supervised methods (e.g., SimCLR, BYOL, SimSiam) without modifying backbone architectures. The authors further establish the first systematic benchmark for self-supervised SNNs on large-scale video datasets, including UCF101, HMDB51, and MiniKinetics. Experimental results demonstrate that PredNext consistently yields substantial performance gains and achieves performance comparable to supervised ImageNet pretraining, enabling effective unsupervised temporal representation learning for deep SNNs on complex video tasks.

### Strengths
1. Traditional SNNs rely solely on the autoregressive spatiotemporal dynamics of LIF neurons to capture temporal information, which recent studies have suggested is insufficient. This paper provides new insights into this issue, which is highly commendable.

2. This work contributes valuable and important baselines by adapting several classical unsupervised learning algorithms to SNNs. This represents a meaningful contribution to the community.

3. The authors conducted extensive experiments that convincingly demonstrate the effectiveness of the proposed method.

### Weaknesses
1. It is recommended to revise the introductory description around line 79. Specifically, the authors should emphasize that the temporal processing capability of SNNs originates from the intrinsic dynamics of spiking neurons rather than solely from the membrane potential. Moreover, not all neurons produce linear outputs below the threshold, so the current wording may be misleading.

2. Figure 2 appears overly complex and does not leverage color cues to guide the reader’s attention, which may hinder comprehension of the algorithm. It is very likely that readers will find this figure difficult to understand by the end of the Introduction. This issue may be alleviated by briefly explaining the roles of  $P_T$ and $P_C$ either within the figure or in close proximity to it.

3. Some items in Table 1 could be formatted in bold to highlight key information more effectively.

4. In Table 2, it is suggested to use checkmarks and crosses in a consistent font style or ensure they originate from the same LaTeX package for visual uniformity.

### Questions
1. In Line 3 of Algorithm 1, it appears that the notation suggests $x_i$=$x_j$. If the authors intend to express that the outputs of the data augmentation function differ due to randomness applied at different times, the current formulation may be misleading and could benefit from clarification.

2. In Line 4 of the algorithm, the symbol T seems to be used to represent the number of time steps. However, if I understand correctly, throughout the paper T denotes data augmentations (or is associated with the augmentation function), while N should represent the temporal length. Similar symbol inconsistencies appear in several places. If my interpretation is incorrect, I would appreciate the authors’ clarification.

3. Could the authors provide a theoretical analysis to support the problem formulation and the effectiveness of PredNext, instead of relying solely on empirical observations? Even a supplementary theoretical justification in the appendix would significantly strengthen the submission, especially considering ICLR’s expectations.

4. Could the authors more clearly emphasize the direct relationship between this work and SNNs? The current presentation seems to focus more on modeling spatiotemporal features between frames, rather than leveraging the intrinsic autoregressive temporal dynamics of spiking neurons within frames, which is typically a core property of SNNs.

5. The authors claim that previous work did not train unsupervised SNNs to deeper network depths. However, this is not entirely accurate, and the proposed method also does not extend beyond SEW-ResNet18 in depth. Do the authors have additional evidence or analysis supporting this claim? In particular, a statistical comparison of unsupervised methods applied to deeper SNNs or larger parameter scales would be crucial to validate the novelty emphasized in the manuscript.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes PredNext, a plug-and-play auxiliary module for unsupervised SNNs that adds two explicit temporal prediction heads: Step Prediction  and Clip Prediction , trained with cross-view targets. The method is integrated into standard SSL frameworks (SimCLR, MoCo, SimSiam, BYOL, Barlow Twins) and evaluated on UCF101, HMDB51, and miniKinetics. PredNext consistently improves SNN baselines on classification and retrieval and reports analyses of “temporal feature consistency,” arguing that explicit prediction improves consistency more productively than forced consistency constraint.

### Strengths
1) Even without deep familiarity with DVS video, I could follow the technical pipeline and reasoning; clarity and readability are clear strengths of this submission.

2) The cross-view PT/PC heads are simple and slot into multiple SSL recipes for SNNs. This idea is a practical contribution for the community.

3) The paper moves beyond small DVS benchmarks to UCF101, HMDB51, and miniKinetics, with coherent pretrain/finetune protocols.

4) Across SimCLR/MoCo/SimSiam/BYOL/Barlow Twins, PredNext improves top-1/5 and retrieval R@K; ablations illustrate the impact of PT vs. PC and step length.

5) The paper formalizes a consistency error and contrasts explicit prediction vs. direct consistency penalties, a useful negative result.

### Weaknesses
1) Fig. 1(b): The “distribution of video features in high-dimensional space” appears as a schematic; it’s unclear whether it represents t-SNE or merely an illustration. Define what **blue vs. red** denote, what **green arrows** represent, and  what the inter and intra- cluster distance encodes.  Fig. 2 remains visually cluttered. Arrows cross and re-enter modules, making the flow hard to follow. Please improve the labeling.
2) Please **number** all equations. In the “final optimization objective,” specify whether **α** is fixed across datasets or tuned per dataset. Provide an ablation varying **α** to assess the relative importance of each view’s learning target and report the setting you recommend.
3) **Positioning vs. strong ANN baselines is incomplete.** Table 3 has **no ANN SSL baselines** under the same spatiotemporal setup (e.g., ResNet-18/34  or a Video Transformer). Since the paper argues SNNs better capture long-range temporal dependencies/consistency, please add **ANN-SSL counterparts** trained under the identical clip lengths, strides, and augmentations. This will solidify the “SNN advantage” claim. 
4) **Scaling.** How do results change with **SEW-ResNet-34/50** and with a **spiking transformer** backbone?

### Questions
Please refer to the four questions included in **Weaknesses** for details.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The PredNext paper presents a novel module to enhance unsupervised representation learning in Spiking Neural Networks (SNNs), specifically for temporal video data. The core innovation is explicitly modeling temporal relationships by forcing the network to perform cross-view future feature prediction through two heads: Step Prediction and Clip Prediction. PredNext's predictive objective guides the network to learn stable, semantically rich features that are naturally consistent, a superior alternative to merely imposing forced consistency constraints, which the authors show degrades performance. The module ntegrates with existing self-supervised methods, and achieves significant performance gains on large-scale video benchmarks, even matching ImageNet-pretrained supervised weights.

### Strengths
The paper presents an original and well-executed contribution to unsupervised learning in spiking neural networks by introducing explicit cross-view temporal prediction to enhance temporal consistency. The idea is conceptually novel within the SNN context, bridging predictive coding principles with modern self-supervised methods like SimSiam and MoCo. The experiments are thorough, spanning multiple large-scale video datasets, and the reported gains are consistent and meaningful. The writing is clear, the methodology is reproducible, and the visualizations effectively support the claims. Overall, the work makes a significant and timely contribution by demonstrating that deep SNNs can achieve competitive unsupervised performance on complex temporal data, pushing the field beyond small-scale, biologically inspired setups.

### Weaknesses
The paper’s main weakness is that PredNext mostly improves short-term smoothness rather than learning true long-range temporal structure. Performance drops quickly when predicting beyond one step, showing limited temporal modeling capacity. 

The training objective is also sensitive to the loss weight ($\alpha$), but the paper provides no analysis of how this affects stability or generalization. Comparisons are limited to ANN-based self-supervised baselines, leaving out SNN-native or biologically inspired methods such as STDP-based learning. This makes it hard to judge the real advantage of PredNext for spiking systems. 

The two extra prediction heads add non-trivial computation during training, which could limit scalability.

Finally, the experiments focus only on visual data and do not test multimodal or event-based inputs, where SNNs typically excel.

### Questions
1. The performance appears to peak sharply at one-step prediction ($m=1$). Can the authors clarify whether this is a model limitation or a training stability issue? Have they explored architectures or loss designs that allow for longer temporal prediction without collapse?

2.  Since $\alpha$ directly controls the trade-off between self-supervised and predictive losses, how sensitive is PredNext to its value? Would an adaptive or scheduled weighting improve stability?

3. The paper notes that the prediction heads are only used during training, but could the authors quantify the actual increase in computation and memory footprint? Understanding this cost would help assess scalability to larger networks or datasets.

4. Why were biologically inspired unsupervised methods (e.g., STDP or predictive plasticity-based models) excluded from benchmarking? Including at least one such baseline would better contextualize PredNext’s contribution within SNN research.

5.  PredNext improves temporal consistency but risks over-smoothing. How do the authors ensure that important temporal variations (e.g., motion or action cues) are not suppressed during training?

6. Since SNNs are well-suited to event-driven data, do the authors expect PredNext to transfer effectively to modalities like DVS or multimodal fusion (e.g., vision + optical flow)? If so, how would the framework adapt?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper studies self‑supervised learning (SSL) for spiking neural networks  in video representation learning. The authors propose two pretext tasks tailored to spiking video encoders—Step Prediction and Clip Prediction and systematically benchmark several mainstream SSL paradigms adapted to SNNs on UCF‑101 and HMDB‑51. The work evaluates downstream performance on action classification and video retrieval, and provides an analysis of feature distribution and temporal consistency.

### Strengths
1. Presents (to my knowledge) one of the first systematic comparisons of multiple SSL families when ported to SNN video encoders.
2. The Step Prediction and Clip Prediction pretext tasks appear well‑aligned with SNN dynamics. Solid empirical evaluation across two standard video benchmarks with both classification and retrieval. The ablations indicate both pretext components contribute to the gains 
3. Analysis on the distribution and temporal consistency of video features in Fig 4 is interesting and helps explain how SSL affects time‑evolving SNN features.
4. Establishing baselines and guidance for SSL with SNNs is useful for the neuromorphic community and may reduce the barrier to entry for future work.

### Weaknesses
1. Absolute performance vs. ANN baselines. Your best SNN numbers (e.g., **72.2%** on UCF‑101 and **41.5%** on HMDB‑51) lag a modest but non‑trivial margin behind a supervised **ANN ResNet‑18** initialized from ImageNet (≈**76.3% / 48.9%** as reported in **[1]**). This gap matters for ICLR’s broader audience.
2. Compute, memory, and scalability are under‑specified. SNN-based models scale poorly on modern GPUs due to it's recurrent temporal dimension. Training SNNs over time steps often incurs high **memory and wall‑clock** costs. The paper currently omits **GPU hours** and **peak memory**, for your main settings (e.g., $b=256$, $T=16$). Actionable: Report GPU hours and peak memory per configuration; Add training throughput and inference latency. If efficiency is a motivation, consider reporting sparsity metrics (average spike rate) or proxy energy estimates on commodity hardware.

[1] Xiao et al, ReSpike: Residual Frames-based Hybrid Spiking Neural Networks for Efficient Action Recognition, Neuromorphic Computing and Engineering (2025)

### Questions
1. How many GPUs are required to train each main model? 
2. While the results of different SSL methods are present in Table 3 and Figure 4, there lack a detailed analysis and explaination on what causes the results to differ. Can you elaborate on the comparison between SSL methods?

### Soundness
3

### Presentation
3

### Contribution
3
