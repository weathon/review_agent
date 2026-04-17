# TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
The exponential growth of video content necessitates effective video summarization to efficiently extract key information from long videos. However, current approaches struggle to fully comprehend complex videos, primarily because they employ static or modality-agnostic fusion strategies. These methods fail to account for the dynamic, frame-dependent variations in modality saliency inherent in video data. To overcome these limitations, we propose **TripleSumm**, a novel architecture that adaptively weights and fuses the contributions of visual, text, and audio modalities at the frame level. Furthermore, a significant bottleneck for research into multimodal video summarization has been the lack of comprehensive benchmarks. Addressing this bottleneck, we introduce **MoSu** (Most Replayed Multimodal Video Summarization), the first large-scale benchmark that provides all three modalities. Extensive experiments demonstrate that TripleSumm achieves state-of-the-art performance, outperforming existing methods by a significant margin on four benchmarks, including MoSu. Our code and dataset are available at https://github.com/smkim37/TripleSumm.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
TripleSumm introduces an adaptive tri-modal fusion architecture for video summarization that integrates visual, textual, and audio features at the frame level. The model interleaves two components: a Multi-Scale Temporal (MST) block, which captures temporal patterns with hierarchical windowed self-attention, and a Cross-Modal Fusion (CMF) block, which dynamically weighs modality importance using cross-attention. To support large-scale multimodal training, the authors also release MoSu, a new dataset of 52,678 videos featuring synchronized trimodal data and “Most Replayed” annotations from YouTube. TripleSumm achieves state-of-the-art results on four benchmarks (MoSu, Mr. HiSum, SumMe, and TVSum), outperforming prior unimodal and bimodal approaches by notable margins while remaining computationally efficient.

### Strengths
- The CMF block effectively learns frame-wise modality weighting, overcoming the common bias toward visual features in prior multimodal architectures.


- The MST block’s multi-scale attention enables both global context understanding and fine-grained temporal sensitivity.


- Extensive quantitative and qualitative evaluations across four datasets, plus thorough ablations (on modalities, window sizes, and fusion strategies), strengthen empirical claims.

### Weaknesses
- The adaptive fusion’s learning dynamics are empirically motivated but not theoretically analyzed (e.g., modality weighting stability, attention interpretability).


- Using “Most Replayed” statistics as pseudo-ground truth may encode platform or viewer biases, which are only partially mitigated.


- Despite its scale, MoSu is heavily drawn from YouTube-8M and English-transcribed videos, potentially limiting generalization to non-English or low-resource domains.


- The multi-scale temporal attention, though efficient per layer, could still pose latency issues for very long videos or real-time summarization.

### Questions
- How well would TripleSumm adapt to domains beyond YouTube-style videos, such as surveillance, documentaries, or multilingual news—where audio-text alignment and modality saliency differ significantly?

- How sensitive is performance to hyperparameters like window size, number of fusion layers, or modality embedding dimension? Would a learned or data-driven windowing schedule outperform the fixed [N,45,15,5] setup?

- Does the adaptive attention occasionally overfit to dominant modalities (e.g., always preferring audio in music videos)? Could regularization or entropy constraints improve stability?

- How do biases in the “Most Replayed” signal affect what the model learns to consider “important”? Has any qualitative audit been performed on viewer-driven biases (e.g., clickbait or thumbnail influence)?

- What is the real-time performance profile (e.g., inference time per video minute) compared to unimodal transformers, and could TripleSumm be deployed on edge devices or streaming platforms?

- Could the fusion and temporal mechanisms be generalized to other multimodal sequence tasks (e.g., event detection, captioning, or multimodal retrieval)? How would the model handle more than three modalities, such as motion or depth cues?

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
3

### Summary
TripleSumm proposes an adaptive multimodal video summarization framework that integrates visual, audio, and textual cues through a Multi-Scale Temporal Block and a Cross-Modal Fusion Block. It also builds a new large-scale trimodal video summarization dataset with aligned video, transcript, and audio data. Experiments show that the method effectively models both temporal and multimodal patterns, achieving strong summarization performance.

### Strengths
- S1: The separation between temporal modeling (MST) and cross-modal fusion (CMF) is elegant and easy to interpret.
- S2: The framework effectively captures both intra-modal temporal dependencies and inter-modal relationships.
- S3: The new dataset provides valuable resources for future research on multimodal summarization.

### Weaknesses
- W1: This paper shows relatively weak originality. The proposed model mainly focuses on optimizing multimodal (visual, text, audio) feature representations through standard attention operations, where self-attention is used to enhance intra-modal features and cross-attention is used for inter-modal fusion. The motivation and methodology closely resemble those of earlier works such as UMT [1] and CFSum [2], without demonstrating a clear conceptual or technical advancement beyond them.

- W2: This paper’s methodological contributions lie almost entirely in representation learning, without offering task-specific innovations for video summarization itself. The proposed feature enhancement techniques could be applied broadly to other video-related tasks such as action recognition, rather than being tailored to the summarization objective. Moreover, this work is not the first to integrate three modalities (visual, textual, and audio), as prior studies have already explored trimodal fusion for similar purposes.

- W3: The “multi-scale” temporal modeling relies on manually predefined window sizes (e.g., N, 45, 15, 5) rather than adaptive or learnable scales. This heuristic choice may not generalize well to videos of different lengths or dynamics, limiting flexibility and scalability.

- W4: The paper does not include detailed ablation studies isolating the effects of each module (e.g., MST vs. CMF, scale depth, or fusion token design).

[1] UMT: Unified Multi-modal Transformers for Joint Video Moment Retrieval and Highlight Detection

[2] CFSum: A Transformer-Based Multi-Modal Video Summarization Framework With Coarse-Fine Fusion

### Questions
- Q1: Is the reported performance overly dependent on the newly introduced MoSu dataset, whose annotation process (based on replay statistics) may introduce bias and limit reproducibility, potentially misaligning with true human summarization preferences?

- Q2: Is there any discussion of computational cost, such as runtime or FLOPs comparisons, to support the claimed efficiency of the method?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles multimodal video summarization by proposing TripleSumm, an architecture that performs frame-level adaptive fusion of three modalities. To address the widely acknowledged shortage of suitable evaluation resources in this area, the authors also introduce MoSu, a new large-scale benchmark providing synchronized tri-modal data. Experimental results show that TripleSumm yields clear state-of-the-art gains across four benchmarks, including MoSu, indicating the effectiveness of the fusion strategy and the value of the new dataset.

### Strengths
1. Well-motivated architecture design — TripleSumm performs adaptive, frame-level fusion of visual, audio, and text signals using specialized Modality and Temporal Blocks, enabling both fine-grained and long-range semantic capture.
2. Benchmark contribution — MoSu fills a critical gap by offering the first large-scale trimodal dataset for video summarization, which meaningfully advances evaluation and reproducibility in this field.
3. Strong empirical results — TripleSumm achieves state-of-the-art performance on four public benchmarks (including MoSu) with notable improvements and does so without excessive model size, demonstrating both effectiveness and efficiency.

### Weaknesses
1. Dataset Details and Quality Control
 The paper should include more comprehensive details about the proposed dataset, such as the average and variance of summary lengths, textual and audio statistics, and the distribution of video durations. Moreover, the authors need to justify why the generated summaries can be considered high-quality representations of the original videos under the current construction pipeline. It is strongly recommended that the summary quality be validated through human evaluation to ensure reliability.
2. Model Design Rationale
 Some architectural design choices lack clear justification. For instance, the model employs larger temporal window sizes in the early layers and smaller ones in the later layers. This design appears counterintuitive, as shallow layers typically lack the capacity to model long-range temporal dependencies. The authors should clarify the motivation behind this choice or consider the more conventional approach of using smaller windows in early layers and progressively larger ones in deeper layers.
3. Clarity of Figure 2
 Figure 2 is somewhat misleading. The main text states that the four modalities are processed independently by the MST, whereas the figure depicts them as being input jointly, suggesting possible cross-modal interaction within MST. The authors should revise the figure or clarify in the text how the modalities are actually handled to avoid confusion.
4. Generalization Evaluation
 To better demonstrate the generalization capability of the model and the utility of the proposed dataset, it is recommended to train the model on the new dataset and evaluate it on other benchmark datasets (with or without fine-tuning). Such experiments would significantly enhance the credibility and impact of the proposed dataset within the research community.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TripleSumm, a trimodal video summarization framework that adaptively fuses visual, textual, and audio information to better capture the dynamics of real-world videos. The authors argue that existing methods are limited by unimodal or bimodal inputs and cannot adapt when the dominant modality changes over time. TripleSumm introduces two key components: a Multi-scale Temporal (MST) block that progressively refines attention from global to local temporal contexts, and a Cross-modal Fusion (CMF) block that computes per-frame adaptive weights to determine the relative importance of each modality. To support large-scale multimodal learning, the authors also construct MoSu (Most Replayed Multimodal Video Summarization), the first large trimodal dataset containing over 52,000 YouTube videos with synchronized visual, text, and audio signals and “most replayed” statistics as pseudo-ground-truth for frame importance. Extensive experiments across four benchmarks (MoSu, Mr. HiSum, SumMe, and TVSum) demonstrate that TripleSumm achieves state-of-the-art results with strong generalization and efficiency, supported by comprehensive ablations validating its dynamic fusion design and multi-scale temporal modeling.

### Strengths
Clear, modular architecture that explicitly separates temporal refinement (MST) from cross-modal fusion (CMF).

 Strong, consistent gains across multiple datasets/metrics with a small model size.

 Careful ablation validating design choices (windowing schedule, dynamic fusion, modality combos).

### Weaknesses
Reliance on “Most Replayed” as ground truth, while pragmatic, can encode popularity/behavior biases; human alignment on MoSu isn’t quantified beyond transfers to SumMe/TVSum. It would be better to human-study agreement or correlation with editorial summaries.

 Selection uses standard KTS + knapsack; could the model be trained end-to-end with a differentiable or learning-to-select objective, and would that change results?

 While parameter-efficient, the inference cost with four MST blocks and CMF per frame isn’t fully benchmarked vs. strong baselines at equal latency. A runtime/memory comparison would help.

### Questions
For MoSu, non-English transcripts are machine-translated; for other sets, text is generated via image captioning. How sensitive are results to transcript quality and captioning choice? Any robustness analysis?

### Soundness
3

### Presentation
3

### Contribution
3
