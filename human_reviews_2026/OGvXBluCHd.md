# HandReader: Advanced Techniques for Efficient Fingerspelling Recognition

- Decision: Reject
- Scores: 4, 0, 4, 6

## Abstract
Fingerspelling is a significant component of Sign Language (SL), allowing the interpretation of proper names, characterized by fast hand movements during signing. Although previous works on fingerspelling recognition have focused on processing the temporal dimension of videos, there remains room for improving the accuracy of these approaches. This paper introduces HandReader, a group of three architectures designed to address the fingerspelling recognition task. HandReader_RGB employs the novel Temporal Shift-Adaptive Module (TSAM) to process RGB features from videos of varying lengths while preserving important sequential information. HandReader_KP is built on the proposed Temporal Pose Encoder (TPE) operated on keypoints as tensors. Such keypoints composition in a batch allows the encoder to pass them through 2D and 3D convolution layers, utilizing temporal and spatial information and accumulating keypoints coordinates. We also introduce HandReader_RGB+KP — architecture with a joint encoder to benefit from RGB and keypoint modalities. Each HandReader model possesses distinct advantages and achieves state-of-the-art results on the ChicagoFSWild and ChicagoFSWild+ datasets. Moreover, the models demonstrate high performance on the first open dataset for Russian fingerspelling, Znaki, presented in this paper.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes three architectures for fingerspelling recognition, HandReader:

(1) HandReaderRGB introduces TSAM (Temporal Shift-Adaptive Module) into ResNet-34 to handle variable-length videos without cropping/padding, and uses a shift counter to avoid temporal overwriting on short sequences;

(2) HandReaderKP proposes TPE (Temporal Pose Encoder), which rearranges 54 hand/body keypoints per frame (x,y,z) into a (3,T,54) tensor and performs 2D→3D convolutions for “coordinate accumulation”;

(3) HandReaderRGB+KP sums the two feature streams and feeds them to BiGRU + CTC.
The authors also release Znaki (a Russian fingerspelling dataset) and report SOTA on ChicagoFSWild/+ and Znaki. The overall approach is engineering-oriented with simple and efficient modules.

### Strengths
Practical, simple solutions (TSAM/TPE) for real deployment pain points (variable length, low latency) with low engineering overhead.

Three tracks (RGB / KP / fusion) under a unified BiGRU+CTC training/decoding setup.

Znaki dataset shows strong quality control (multi-annotator verification, signer-disjoint splits), which can foster cross-language evaluation.

### Weaknesses
1) TSAM: No ablation for shift counter on/off, per-layer number of shifts, or other key hyperparameters; no direct comparison to “fixed-length TSM + mask/padding.”

2) Only “sum” is reported; no comparisons to concat/gating/cross-attention.

3) Efficiency and deployability: Missing a unified comparison against 3D-CNN, TimeSformer, TSM (fixed-clip) on FLOPs/Params/end-to-end latency/peak VRAM; preprocessing costs (hand-crop, MediaPipe) are not included.

4) Fairness of pipeline: RGB uses Shi (2019) cropping while keypoints are extracted from raw frames; the field-of-view mismatch may confound results.

### Questions
Please provide channel split ratios, shift directions/steps, per-layer shift counts, and the algorithmic box (or pseudocode) for the shift counter; report boundary handling (zero/replicate/circular).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
In this paper the authors present a new dataset for fingerspelling and russian and two architectures for fingerspeeling, relying on RGB or keypoint as input. The presentation is clear enough and the Znaki dataset is useful for the community of fingerspelling. The experimental results of the authors show that adding TSAM instead of TSM slightly improves the results on ChicagoFSWild dataset in terms of letter accuracy, and that the fusion of both models achieves competitive results w.r.t. previous methods such as Conformers. 
Overall apart from the Znaki dataset the paper is mostly engineering and reads more like a technical report.

### Strengths
The main strength of the paper seems to be the Znaki dataset for Russian fingerspelling. The collection of the dataset is well documented but it is not clear if the dataset will be made available to researchers.  
Also another small strength is the engineering behind making the TSM module adaptable to variable length sequences, although the actual gains are minor.

### Weaknesses
Overall the novelty of the paper is very weak. TSAM seems a small engineering trick behind making TSM adapt to variable length sequences and the ablation result in Table 7 of the supplementary seems a very small increase in terms of letter accuracy. Also we only have letter accuracy as metrics - no CER/WER, word accuracy, S/D/I breakdown, or confusion matrices.

Furthermore, another contribution stated is the combination of the keypoints as a three-dimension tensor, however this has been done multiple times before and is the standard approach for inputting keypoints in deep networks. 

W.r.t. the Znaki dataset, the authors did only train their own methods and did not show any comparisons with previous state of the art. Also in table 5 the authors added model size and performance metrics however this was not presented for any other previous method.

In Table 3, the results are confusion. Conformers were trained on both train datasets as it is said - what about HandReader? It is not clear. Also, for fair comparison the authors should pretrain the conformer using their pipeline.

There is not intuition w.r.t. to the previous methods -> what is the trick of handreader that really enabled it to obtain such performance ? It seems that even standard TSNs in Table 7 are able to beat a lot of previous sota methods.

### Questions
1) What are the results of a conformer using your exact pipeline for training ?
2) Did the authors try any other fusion methods apart from feature summation ? 
3) Regarding the claim "Such a method reduces VRAM usage by up to 50% during batch creation" we do not see any actual numbers or evidence to support this. 
4) What are the performance metrics of other methods ? Showcase a proper comparison.

In general, at its current state I would urge the authors to more restructure their paper around the new dataset and the results of previous sota methods on it. If they are going to claim contributions w.r.t. a new method more results/metrics are needed and clear comparisons with the state of the art. Currently TSAM only offers a slight 1% increase and we are not sure it is not noise -> maybe test on more datasets ?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a compositional pipeline to solve the task of fingerspelling recognition. They introduce an architecture, consisting of a modified Temporal Shift Adaptive Module and a Temporal Pose Encoder to process input RGB and keypoint sequences both separately and jointly. They validate their method on the ChicagoFSWild and ChicagoFSWild+ datasets and introduce a new Russian Fingerspelling dataset.

### Strengths
1. The authors achieve state-of-the-art results on both ChicagoFSWild and ChicagoFSWild+ benchmarks, which demonstrates the effectiveness of their input processing and pipeline. 
2. They also introduce a new dataset, containing a large number of high-resolution videos, which is helpful to the sign language community.

### Weaknesses
1.Some sections of the paper are not fully clear and easy to follow. 

2. The proposed method mainly leverages existing architectures to solve the task without significant modifications. For the RGB video processing the extension of the Temporal Shift-Adaptive Module (TSAM) for varying video lengths is a useful addition but is not in my opinion a major modification of the existing (TSAM) architecture. Also, organizing key points as tensors and using them as input to a convolutional encoder, to the best of my knowledge, has been extensively used in action and sign language recognition. 
2. The proposed approach is evaluated on a limited number of datasets. Its effectiveness could be evaluated further by extending it to more benchmarks.
3. Evaluating only on fingerspelling benchmarks limits the scope of this work. A more general approach that targets both widely used sign language datasets and fingerspelling might be more appropriate for this conference.

### Questions
1. It is somewhat surprising that the keypoint-only model performs similarly to the RGB-based model, as prior work typically reports stronger performance from RGB features. Could the authors explain this behavior or discuss possible reasons ?
2. Did the authors investigate how the quality or accuracy of the MediaPipe keypoint predictions affects overall model performance?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses fingerspelling recognition by introducing "HandReader," a set of three architectures. The core contributions are two novel encoders: 1) the Temporal Shift-Adaptive Module (TSAM), which extends TSM to effectively process variable-length RGB videos, and 2) the Temporal Pose Encoder (TPE), which uses 2D/3D convolutions to process keypoints. These models achieve state-of-the-art results on the ChicagoFSWild and ChicagoFSWild+ benchmarks. The paper also introduces "Znaki," a new, large-scale 37k-video dataset for Russian fingerspelling, detailing its rigorous collection and validation methodology.

### Strengths
A substantive assessment of the strengths of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage reviewers to be broad in their definitions of originality and significance. For example, originality may arise from a new definition or problem formulation, creative combinations of existing ideas, application to a new domain, or removing limitations from prior results.

1. The paper introduces TSAM, a novel module that effectively handles variable-length video inputs by adapting the TSM mechanism. It also proposes TPE, a new encoder that processes keypoints as tensors using 2D and 3D convolutions to capture spatio-temporal dynamics.
2. The HandReader models achieve new state-of-the-art results on both the ChicagoFSWild and ChicagoFSWild+ test sets, outperforming prior work in RGB-only, KP-only, and combined modalities.
3. The paper introduces "Znaki," a large-scale (37k videos) and rigorously validated dataset for Russian fingerspelling.

### Weaknesses
A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc.

1. While the HandReader_KP model is very fast on a CPU (3.9ms, Table 5), the best-performing RGB (51.4ms) and RGB+KP (55.2ms) models are considerably slower. This could limit real-time applicability on low-power devices.
2. The architectures are not fully end-to-end and depend on separate, pre-trained models for hand-cropping and keypoint extraction. This adds computational overhead and complexity.

### Questions
Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This is important for a productive rebuttal and discussion phase with the authors.

1. What is the computational efficiency trade-off of using TSAM versus the original TSM, given the less parallelizable, per-video logic?
2. The reliance on external preprocessing models is noted as a limitation. Have you explored any end-to-end alternatives that could integrate hand-localization or keypoint estimation directly into the architecture?

### Soundness
2

### Presentation
3

### Contribution
2
