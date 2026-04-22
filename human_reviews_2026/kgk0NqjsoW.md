# DAVE: A VLM Vision Encoder for Document Understanding and Web Agents

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
While Vision–language models (VLMs) have demonstrated remarkable performance across multi-modal tasks, their choice of vision encoders presents a fundamental weakness: their low-level features lack the robust structural and spatial
information essential for document understanding and web agents. To bridge this
gap, we introduce DAVE, a vision encoder purpose-built for VLMs and tailored
for these tasks. Our training pipeline is designed to leverage abundant unlabeled
data to bypass the need for costly large-scale annotations for document and web
images. We begin with a self-supervised pretraining stage on unlabeled images,
followed by a supervised autoregressive pretraining stage, where the model learns
tasks like parsing and localization from limited, high-quality data. Within the supervised stage, we adopt two strategies to improve our encoder’s alignment with
both general visual knowledge and diverse document and web agentic tasks: (i)
We introduce a novel model-merging scheme, combining encoders trained with
different text decoders to ensure broad compatibility with different web agentic
architectures. (ii) We use ensemble training to fuse features from pretrained generalist encoders (e.g., SigLIP2) with our own document and web-specific representations. Extensive experiments on classic document tasks, VQAs, web localization, and agent-based benchmarks validate the effectiveness of our approach, establishing DAVE as a strong vision encoder for document and web applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces DAVE, a specialized vision encoder for VLMs targeting document understanding and web agent tasks. The authors propose a two-stage training approach: (1) self-supervised pretraining using MAE on unlabeled document and web images, and (2) supervised autoregressive pretraining with limited annotated data. Key innovations include a weight-merging scheme to create decoder-agnostic encoders and ensemble training to combine general visual features with domain-specific representations. Experiments demonstrate improvements over existing encoders on document, web, and agent benchmarks.

### Strengths
The focus on document and web understanding addresses a real limitation of current VLMs.
Testing across multiple benchmarks (document tasks, VQA, web agent tasks) demonstrates broad applicability.

### Weaknesses
The paper lacks deeper analysis of why this specific combination of techniques works. Why is ensemble training with SigLIP2 optimal? What are the theoretical guarantees of the weight merging approach?

While the combination is new, individual components are well-established, such as MIM pretraining and Visual-language alignment.

### Questions
1. How sensitive is the approach to the choice of generalist encoder? Would similar improvements be seen with other encoders?

2. More visual encoders should be compared, such as [1][2][3].
[1] Web-DINO: Scaling Visual Representation Learning
[2] A Token-level Text Image Foundation Model for Document Understanding
[3] RADIOv2.5: Improved Baselines for Agglomerative Vision Foundation Models

3. A similar idea to Vary, which uses two encoders to represent the text scenes.

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
The paper presents DAVE, a vision encoder tailored for document understanding and web agent tasks within vision-language models. It introduces a two-stage pretraining framework combining self-supervised masked autoencoding on large unlabeled datasets with a supervised autoregressive stage for tasks such as OCR, layout prediction, and web grounding. To enhance generalization across architectures, the authors employ a weight-space merging scheme that combines encoders trained with different text decoders, producing a decoder-agnostic encoder, and an ensemble training strategy that fuses general semantic features from SigLIP2 with DAVE’s specialized structural features.

### Strengths
* The paper is clearly written and easy to follow.
* The research question of vision encoder for document images is important.

### Weaknesses
* The contribution of the paper is limited. The training methods employed in the paper including self-supervised pre-training, model merging are well-established ideas such as MAE self-supervised learning [1], model soups [2].

* Although the experiments encompass a wide range of benchmarks (e.g., DocVQA, ChartQA, Mind2Web), the paper provides limited analysis of cross-domain generalization—such as performance on non-English documents—as well as robustness and interpretability aspects.

[1] Masked Autoencoders Are Scalable Vision Learners.

[2] Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.

### Questions
There are a few typos in the paper:
* Line 240: "Appendix X"
* Line 268: It should be "Qwen2.5-7B-Instruct."

### Soundness
2

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
5

### Summary
DAVE presents a compelling approach to developing a specialized vision encoder for document and web understanding, demonstrating strong empirical results. The innovative MAE reconstruction and decoder-agnostic merging scheme are notable contributions. However, the paper suffers from several weaknesses, primarily in providing insufficient experimental evidence or logical clarity to fully support certain design choices and claims. Addressing these points, particularly through more rigorous ablation studies and transparent comparative methodologies, would significantly strengthen the paper's impact and credibility.

### Strengths
1. The paper proposes a modification to the standard MAE objective by reconstructing raw pixel values directly, rather than normalized pixel values. This aims to stabilize training, particularly for document and web images which exhibit low inter-patch variance. While further empirical evidence directly linking this change to stability would strengthen the claim, the approach itself is a thoughtful adaptation to domain-specific data characteristics.
2.  A significant strength is the introduction of a distillation-based model merging scheme. This approach trains multiple encoder instances, each aligned with a different text decoder, and then learns optimal coefficients to combine their weights into a single, decoder-agnostic encoder. This method effectively addresses the challenge of encoder overfitting to specific decoders and enhances compatibility across diverse VLM architectures.
3.  The authors leverage a vast amount of data for both self-supervised and supervised pretraining, including large-scale unlabeled document/web images and curated annotated datasets. This extensive data utilization, combined with the proposed training strategies, leads to DAVE achieving impressive performance across a wide range of document, general VQA, and web agent benchmarks, often outperforming strong baselines like SigLIP 2.

### Weaknesses
1.  The core hypoththesis of this paper is that current VLM vision encoders "lack the robust structural and spatial information essential for document understanding and web agents." However, this assertion is made without references or experimental evidence. Given the existence of numerous VLMs works for document understanding, a more detailed explanation is needed to clarify why these models specifically lack the required powerful structural and spatial information. The observed performance gains of DAVE, while compelling, could also be attributed to its extensive and specialized training data rather than a fundamental architectural superiority in capturing these specific features, necessitating a more rigorous analysis.
2.  For document and web images, the authors argue that low inter-patch variance leads to training instability, and therefore they reconstruct from raw pixel values rather than normalized pixel values. However, the authors only show a difference in inter-patch variance distribution between ImageNet-like, document-like, and web-like images, without providing experimental evidence which directly demonstrating that this variance difference leads to training instability with the standard MAE reconstruction method. 
3.  DAVE combines a specialized encoder ($\phi_{spec}$) for low-level structural features with a generalist encoder ($\phi_{gen}$) for high-level semantics. However, if we  fine-tuning a strong, generalist encoder like SigLIP 2 (with DAVE's training protocol and data) would yield comparable results? 
4.   When comparing DAVE with other models like SigLIP 2, it's unclear whether the original decoder in SigLIP2 was replaced by the same LLM  decoder used in DAVE? 
5. In Table 3, DAVE’s performance on RICO-SCA is slightly lower than that of SigLIP 2. The authors explain this by stating that the task emphasizes semantic understanding, which DAVE was not designed to tackle. However, this explanation directly contradicts the motivation claimed by the authors for integrating SigLIP 2 into DAVE, namely to provide high-level semantic capabilities for better document understanding. This inconsistency needs to be addressed.
6. Figure 2, which illustrates the incremental impact of design choices, uses terms like "Scratch Decoder" and "Pretrained Decoder" without clear definitions. Furthermore, some data points in Figure 2 can not be found in the tables in the paper, making it difficult for readers to fully understand and verify the ablation results. 
7. In Table 4c, when comparing DAVE with multi-encoder setups (e.g., SigLIP 2 + DiT, PS, DP), it is not explicitly stated whether these models underwent the same extensive training pipeline as DAVE — including self-supervised pretraining, multi-decoder supervised training, and model merging. If these pre-trained models were merely concatenated and directly evaluated on the benchmarks without following the same training procedure, the comparison would be unfair. 
8. The comparison in Table 4d between DAVE and a fine-tuned SigLIP 2 is potentially unfair. The fine-tuned SigLIP 2 is described as being trained "only on the supervised data," implying it did not undergo DAVE's full two-stage pipeline, including self-supervised pretraining and multi-decoder merging.

### Questions
see Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The main contribution is a vision encoder architecture for document and web understanding evaluated on a broad array of benchmarks and reports improvements over strong baselines. The use of PlotQA, ChartQA, FinTabNet is well motivated. Key contribution is enabling data efficient training in specialized domains.
Compared to SigLIP2 and AIMv2 in particular, the submission shows that
1) On Document Understanding Benchmarks (OCRBenchm DocVQA, InfoVQA) the gains are meaningful and suggests that the proposed architecture provides a non trivial improvement in the document domain
2) On General VQA the results are a bit mixed which calls for more understanding of the regressions esp wrt to SigLiP2

Overall in terms of novelty and significance, compared to known encoder types and training strategies (contrastive + self-supervised for SigLIP2l autoregressive multimodal for AIMv2) the core contribution lies in demonstrating broad generalized across heterogenous benchmarks.

### Strengths
The breadth of evaluations and ablations is a key strength.  
[1] Thorough benchmarking on Document understanding, general VQA, and Web understanding benchmarks: We see evaluation across a wide variety of benchmarks and baseline models. Using both Qwen-2.5-7B-Instruct and Llama-3.2-3B-Instruct decoders further solidify the alignment on different text decoders
[2] Very strong improvements on Mind2Web (Table 2)
[3] Clean ablation setups are useful to justify the training setup. Table 4 c,d are esp interesting contribution comparing different multi encoder settings

### Weaknesses
[1] average perfomance is used across all results which might under or over estimate the performance. Having some best of N results using some confidence or majority voting would further help understand the variance of the model performance better.
[2] there might be potential domain bias, example overfitting to synthetic chart styles in PlotQA or financial domain layouts in FinTabNet. Might have been useful to detect and discuss such biases.

### Questions
[1] We need to have confidence intervals for the results esp when comparing SigLIP2 to DAVE using Qwen-2.5-7B-Instruct model in Table 1. 
[2] A little more discussion of regression on RealWorldQA and TextVQA would be useful to understand if regressions on general VQA capabilities are real
[3] we processed 500K PDF samples from arXiv with an OCR model (Cui et al., 2025), formulating them into additional recognition
and grounding tasks. What kinds of post processing was done on the data to ensure high quality examples?

### Soundness
3

### Presentation
3

### Contribution
3
