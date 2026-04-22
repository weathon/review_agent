# On the Quantization of Neural Video Codecs

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Full-precision floating-point neural image and video codecs pose significant challenges in power consumption, storage requirements, and cross-platform interoperability, particularly when deployed on resource-constrained devices. To address these issues, network quantization techniques have been extensively studied for neural image codecs. However, the quantization of neural video codecs remains largely unexplored. Unlike quantizing neural image codecs, quantizing neural videos codecs requires significantly more effort. Many coding components operate on temporally correlated data and often rely on features propagated from previous frames, introducing additional sensitivity to both cross-platform round-off errors and network quantization. This work presents the first systematic and algorithmic study of quantization effects across multiple neural video coding frameworks and temporal buffering strategies. Extensive analyzes are conducted to evaluate how various combinations of coding frameworks and temporal buffering strategies respond to various quantization schemes in terms of coding performance and computational complexity. This work offers actionable insights into the future development of neural video codecs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents the first systematic study on quantizing neural video codecs across multiple coding frameworks and temporal buffering strategies, introducing a mixed-precision scheme that slashes complexity by 53–87 % with only 2–4 % BD-rate loss.

### Strengths
1. The paper systematically investigates quantization in neural video codecs across diverse frameworks and buffering strategies. The experiments comprehensively cover component-wise analysis, multiple quantization methods, and hybrid buffering for fair evaluation.
2. The mixed-precision quantization scheme achieves up to 87 % bit-operation and 53 % model-size savings with only a 3.5 % BD-rate penalty, clearly demonstrating a practical performance–complexity trade-off.
3. The study ties codec quantization to cross-platform reproducibility and temporal drift prevention, providing engineering insights that bridge algorithmic design and system implementation.

### Weaknesses
1. The presentation is poor. The paper even lacks an “Abstract” heading on the first page, which is a serious formatting oversight. In addition, there are other presentation issues such as inconsistent font sizes in tables and uneven spacing around some section headings.
2. The quantization evaluation is performed in a floating-point simulation environment without hardware validation. While standard in algorithmic studies, this setup may not capture integer rounding and overflow behaviors on real accelerators, which are crucial for verifying cross-platform consistency and deployment efficiency.
3. The study focuses exclusively on the DCVC-FM backbone, leaving uncertainty about transferability to transformer-based or flow-matching codecs.
4. The paper reports theoretical complexity reductions (bit operations, memory, and model size) but does not include real-device latency or energy measurements. This limits the strength of its claims about practical efficiency.
5. The paper does not quantify how quantization-induced errors accumulate across long temporal dependencies. While short-sequence BD-rate analysis is provided, a study on long-term drift or stability would strengthen claims of robustness in real-time deployment.

### Questions
1. Have the authors analyzed temporal error accumulation theoretically or through long-term video sequences to validate stability?
2. How does the quantization behavior vary when applied to different backbone architectures such as DVC or FVC?
3. What are the actual runtime and power benefits on mobile or edge devices, beyond bit-operation and model-size reductions?
4. Have you considered using an automated bit-width allocation or mixed-precision search method (e.g., gradient-based or reinforcement learning approaches) to replace the current manually tuned configuration for better scalability?

### Soundness
3

### Presentation
1

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
This paper presents a detailed analysis of the effects of quantization on video compression models, taking into account the following factors: different coding frameworks (e.g., CC, MRC), temporal buffering strategies (Explicit, Implicit, Hybrid), quantization training  strategies (PTQ, QAT), and each decoder component. Based on this, the authors proposed a mixed-precision compression model built upon the MCR architecture, achieving a good rate–distortion–complexity trade-off.

### Strengths
1. The RD performance improvement of the proposed model MCR-Hybrid is obvious, as shown in Table 7.
2. The experiments in Section 4 demonstrate a substantial amount of work and provide insightful implications for the design of quantized models.

### Weaknesses
The article lacks some key experiments to demonstrate the model's effectiveness.
1. As cross-platform consistency is the most important functionality of INT-quantized compression models, the paper lacks validation in this aspect. Specifically, it remains unclear whether encoding and decoding on different GPU models would introduce reconstruction errors.
2. The comparison of actual encoding and decoding speed is missing in Table 7. Given that it is well known that Decoder BO, including MACs, sometimes fails to reflect the real-time computational cost [r1], such a comparison is necessary.
3. The analysis of decoder components in Section 4.2 lacks reference to complexity metrics. For Conclusion 1, could the inter-frame main decoder suffer the most from quantization simply because it has the highest complexity?

[r1] Zhaoyang Jia, Bin Li, Jiahao Li, Wenxuan Xie, Linfeng Qi, Houqiang Li, and Yan Lu. Towards practical real-time neural video compression, 2025.

### Questions
1. In the experiments, how are non-8-bit integer quantizations such as W14/W12/W10 and A14/A12/A10 actually implemented? Do they actually facilitate hardware deployment and deliver real speed-ups?
2. Are the training procedures described in Tables 18–20 complete? DCVC-FM is trained using sequences of up to 32 frames, whereas the authors only adopt a training strategy with up to 5 frames and achieve comparable performance. Can this shortened frame training strategy reproduce the performance of DCVC-FM?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of the quantization in encoder-decoder neural video codecs. It seeks to present a systematic evaluation of quantization in neural codecs, which have been less explored than for neural image codecs. This is a problem of practical importance due to the increasing use of this class of models and the need for memory-efficient architectures. The primary contribution of the paper is a large number of ablative experiments determining how different components of neural codecs perform under varying bit-widths under uniform post-training quantization (PTQ) and quantization aware training (QAT). A supplementary contribution is the exploration of mixed-precision quantization schemes, and the analysis of temporal buffering strategies under quantization. While the paper contains a large amount of experimentation it would benefit from improved focus and organization, which would improve the strength of the contribution as a systematic evaluation and allow for the practical impact for the paper to be increased.

### Strengths
* The paper addresses an important issue. The practical deployment of neural video codecs is increasing, and there is a need for systematic analysis of memory-efficient components in these architectures. While the use of quantization in neural video codecs is not novel (the authors highlight that decoders have frequently incorporated some quantization as part of the encoder or decoder, L128-140), systematic ablation of these features is rarely performed comprehensively and is a valuable contribution. 
* The paper is well presented - with clear language, well-presented tables, and clear figures. A large amount of effort has gone into providing clear visual presentation and complete experimental details - which is appreciated. 
* The paper contains a very large number of ablative experiments. These are conducted with a large number of configurations (PTQ, QAT; decoder configurations; min-max and MSE quantization; buffering strategies), across multiple datasets. The reported metrics (BD-Rate, PSNR, bpp) / rate-distortion curves in the Appendix are appropriate for video compression. 
* While there are some issues of clarity in the characterization of Implicit / Explicit / Hybrid temporal buffers, and the broad classification of residual coding strategies - examining quantization under different architecture classes is well motivated.

### Weaknesses
**Major** 
* The main issue in the paper is in terms of contribution and positioning. The paper positions itself primarily as a systematic study of quantization effects on neural video codecs, but also introduces some modifications based on mixed-precision quantization and temporal consistency (e.g. pg 5, L239-240, 'a hybrid approach that merges across explicit and implicit buffering strategies'). The paper may be better positioned by focusing the contribution on one aspect in detail (either a proposed new method or a systematic survey). This may additionally help focus the experiments, which while numerous are inconsistent with the contribution scope of a systematic study (e.g. L252, "non-uniform quantization falls outside our scope", and L255 "... generalizing our findings to the full spectrum of quantization methods and network architectures ... is not our intent.") or detailed evaluation of a new method. 
* The paper would benefit from including qualitative examples (e.g. comparison of the frames between the different codecs). This is common for video compression papers - and would support the qualitative comments in the paper (e.g. L349-350 "quantization errors in the motion decoder lead to motion errors ... increasing the bitrate" would be useful to show visually). 
* The quantization literature could be more complete. There are a large number of quantization works, so in order to position this as "the first systematic study of quantization effects on neural video codecs", it would be useful to put the quantization analysis in context. This is important as the quantization described is currently at an engineering level of different methods, rather than providing theoretical motivation or analysis. For example, reviewing the following may be useful to strengthen the theoretical background: 
    * Gholami et al. (2021), A Survey of Quantization Methods for Efficient Neural Network Inference, https://arxiv.org/abs/2103.13630 
    * Gersho and Gray (1992), Vector Quantization and Signal Compression https://link.springer.com/book/10.1007/978-1-4615-3626-0 
    * Jacob et al. (2017), Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, https://arxiv.org/abs/1712.05877 

**Minor** 
* There is a separate large line of research on the use of implicit neural representations for video compression (e.g. NeRV[1]), with a large amount of work looking specifically at quantization effects (e.g. Shi et al. (2025), On Quantizing Neural Representation for Variable-Rate Video Coding [2]). These are also occasionally referred to as neural video codecs. It may be worth including a sentence clarifying that the author's work exclusively looks at quantization effects for encoder-decoder hyperprior style codecs. 
* It would be useful to explicitly describe the weight / activation quantization bits being evaluated in the method section (the results jump into W8A10 [L309], which reads as arbitrary. The section "Sensitivity of Decoder Components to Quantization" [L309-319] feels like some sentences are presented out of order, making it difficult to draw clear conclusions from. 
* There are some survey papers which look at quantization of neural video codecs as components of broader analysis which may be useful to refer to (e.g. Gomes et al. (2025), "End-to-End Neural Video Compression: A Review", https://ieeexplore.ieee.org/document/10962175)

[1] Chen et al. (2021), NeRV: Neural Representations for Videos, https://openreview.net/forum?id=BbikqBWZTGB 

[2] Shi et al. (2025), On Quantizing Neural Representation for Variable-Rate Video Coding, https://openreview.net/forum?id=44cMlQSreK

### Questions
* The characterization of the temporal correlation frameworks (Figure 2 and 3) appears to be a broad classification of existing methods. This makes it difficult to understand which components are proposed as new contributions or analysis for this paper. It would be useful if the authors could clarify this (and how this relates to the additional hybrid approach in Fig 3c). 
* It would be useful to analyse the weight and activation histogram distribution before / following quantization to better understand the sensitivity of the different components to quantization. Evaluating the MSE between the quantized / non-quantized distributions can also help understand the relative importance of bit-widths (beyond the performance level evaluation presented in the paper). E.g. [1], [2]. 
* Table 5 caption - "X denotes the number of channels" (X doesn't appear used within the table). 
* There are a large number of 1-2 letter acronyms (RC, CC, MCR, CRC, I, P, M, H, MP) used for tables. Where space permits, it may be beneficial to use the full names to improve readability. 

[1] Han et al. (2016), Deep Compression: Compressing Deep Neural Networks With Pruning, Trained Quantization and Huffman Coding https://arxiv.org/pdf/1510.00149 (Figure 4) 

[2] Zhao et al. (2019), Improving Neural Network Quantization without Retraining using Outlier Channel Splitting, https://arxiv.org/pdf/1901.09504 (Figure 1)

### Soundness
2

### Presentation
3

### Contribution
2
