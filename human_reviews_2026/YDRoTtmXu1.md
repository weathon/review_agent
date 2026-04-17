# When MLLMs Meet Compression Distortion: A Coding Paradigm Tailored to MLLMs

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
The increasing deployment of powerful Multimodal Large Language Models (MLLMs), typically hosted on cloud platforms, urgently requires effective compression techniques to efficiently transmit signal inputs (e.g., images, videos) from edge devices with minimal bandwidth usage. However, conventional image codecs are optimized for fidelity to serve the Human Visual System (HVS) and ill-suited for MLLMs, in which diverse downstream tasks are jointly considered. In this paper, we first systematically analyze the impact of compression artifacts on several mainstream MLLMs. We find that: Compression distortion unevenly impacts different-level image features, leading to varying effects on MLLMs' downstream tasks depending on their feature-level reliance. Motivated by this discovery, we propose an image Codec TAilored to MLLMs (CoTAM) designed to adaptively protect multi-level features and suit different demands of downstream tasks. The encoder leverages CLIP's shallow-layer attention to generate an importance map for bit allocation, preserving critical semantic regions. Concurrently, the decoder integrates a lightweight adapter with a multi-level loss function to ensure the faithful reconstruction both of low-level details and high-level semantic context for robust synthesis of cross-level features. Extensive experiments validate that our method achieves up to 35.99\% bitrate saving while maintaining the same performance on the MLLM tasks, outperforming previous SOTA neural codecs. The code is released at https://github.com/jmliu206/CoTAM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper notes that cloud-hosted Multimodal Large Language Models (MLLMs) urgently need efficient compression for edge-device image/video transmission with low bandwidth. Conventional codecs, optimized for Human Visual System (HVS) fidelity, are ill-suited for MLLMs’ diverse downstream tasks. Through analysis, the authors find compression distortion unevenly affects image features, impacting MLLMs based on their feature reliance. They propose CoTAM, an MLLM-tailored codec: its encoder uses CLIP’s shallow attention for bit-allocation (preserving key semantics), while its decoder employs a lightweight adapter and multi-level loss to reconstruct low-level details and high-level context. Experiments show CoTAM cuts bitrate by up to 35.99% while maintaining MLLM performance, outperforming prior SOTA neural codecs.

### Strengths
1. Clear and concise description of the proposed method.

2. It depicts the influence of compression distortion on the performance of MLLM, and analyzes the change between original and compressed image as shown in Figure 5.

3. It can achieve the better results within TextVQA, MME and VideoMME tasks.

4. "a disproportionate collapse of the cross-level representations that bridge low-level and high-level information", a interesting observation and may be the key to further improve the performance of MLLMs.

### Weaknesses
1. LLM backbones maybe matter, but this manuscript doesn't compare different LLM backbones.

2. The Equation about $D_{avg}$ and $D_{top1}$ lack the detailed explanation for $A, d and p_i$.

3. The average of $[CLS]$ attention scores lacks the motivation, and the detailed calculation of average score is not provided. Besides the other tokens in CLIP may work well even better.

4. The global attention of high-resolution images in Figure 6(c) is not appropriate since the ratio of downsample is not clear, and how to downsample the image is not to provided.

5. Traditional Codec methods are not compared with the proposed method though the $Adapter$ in Figure 6(b) is not in these traditional methods.

### Questions
1. The clarification of $[CLS]$ attention scores may improve the score of the paper.

2. The variety of MLLMs and Codecs may result in different analysis.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors address the problem of image compression for multimodal large language models (MLLMs) in a multi-task setting. They conduct a detailed analysis of how the image compression process impacts features at different levels and draw several interesting and meaningful conclusions. Building on these analyses, they design a new image compression architecture—based on existing work—that improves MLLM task performance.

### Strengths
Overall, the idea is interesting, the topic is timely, and the paper is generally well written and well structured.

### Weaknesses
I have several concerns regarding the depth and clarity of the analysis section:
- The proposed three-stage theory is intriguing, but the justification appears primarily driven by the observation in Figure 5(b). Although both Figures 5(a) and 5(b) exhibit a U-shaped curve, only Figure 5(b) shows a small plateau at the beginning that supports the notion of three distinct stages. a) Please clarify whether this plateau is the main empirical evidence for defining the three stages. b) The caption of Figure 5 refers to “Average Max Attention Distance,” whereas the text mentions “Distance to the Most-Attended Token.” Are these the same metric, or do they differ? If they differ, please explain the relationship between them. c) Is the precise separation between stages (e.g., between Stage 1 and Stage 2) dependent on the model architecture or the dataset used? Providing a more rigorous or generalizable explanation would strengthen the argument.

- More details should be given regarding the compression methods used. For example, for most compression algorithms—traditional or learning-based—using a fixed quality factor across different images typically results in different bitrates. a) In generating Figure 4, did the authors fix the quality parameter across images, or did they fix the bitrate to ensure comparability? b) The same question applies to subsequent experiments in the experiment section: were all methods evaluated under consistent bitrate conditions? c) Additionally, please clarify whether the observed phenomenon holds across different quality levels or if it appears only under specific compression strengths.

### Questions
N/A

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
4

### Summary
This paper designs an image compression codec specifically for Multimodal Large Language Models (MLLMs). The authors argue that existing codecs, optimized for either the Human Visual System (HVS) or narrow machine-vision tasks, are ill-suited for the diverse, general-purpose nature of MLLMs. The paper's contribution is twofold: 1. It first presents an analysis of how compression affects MLLMs, modeling the vision encoder's information flow as a "three-stage" process (Screening, Local Extraction, Global Integration). The key finding is that compression disproportionately degrades "cross-level features"—the synthesis of low-level details and high-level semantics—while robust low-level and coarse high-level features are less affected. 2. Proposed Method (CoTAM): Based on this analysis, the authors propose a codec, CoTAM, which features (a) a "Shallow CLIP Guidance" mechanism at the encoder to allocate more bits to semantically important regions (identified by shallow CLIP-layer attention) and (b) a "Multi-level Fidelity Decoder" that uses a lightweight adapter and a multi-level loss function (supervising both shallow and deep features) to protect the vulnerable cross-level representations. Experiments across a comprehensive suite of MLLM benchmarks (MME, TextVQA, POPE, etc.) show that CoTAM achieves significant bitrate savings (up to 36% BD-rate) over human-centric (ELIC) and machine-centric (Bridge-d1) codecs, all while maintaining MLLM task performance and incurring minimal computational overhead.

### Strengths
1. The paper tackles a problem of high practical and academic importance. As MLLMs become central to many applications, developing codecs that are "aware" of their specific perceptual needs is a critical research direction.

2. The core methodology is well-motivated. The idea of first analyzing the internal failure point of MLLMs under compression (the vulnerability of cross-level features) and then designing a codec (multi-level loss, semantic guidance) to specifically protect that failure point is logical and elegant.

3. The experimental validation is a major strength. The authors evaluate on a wide and diverse set of MLLM benchmarks, which is essential for "general-purpose" MLLM compression. The reported BD-rate savings of ~36% (Table 2) are substantial and clearly demonstrate the method's effectiveness over strong, relevant baselines.

### Weaknesses
Despite its strengths, the paper has several weaknesses that challenge its claims.

1. The "three-stage information flow" (Sec 2.2.1) is presented as a novel discovery. However, the U-shaped attention distance curve (Fig 5a/b) and the general progression from local, low-level features in shallow layers to global, semantic features in deep layers are well-established properties of Vision Transformers. This part of the analysis feels more like a re-characterization of known ViT properties in the context of compression, rather than a fundamental new insight into MLLMs.

2. The 'Shallow CLIP Guidance' (Sec 3.2) is implementation-specific, creating a dependency that the paper doesn't fully resolve. While the authors evaluate on MLLMs with SigLIP and InternViT encoders , they only demonstrate that CLIP guidance is better than no guidance on a simple ablation. The appendix (A.8) merely shows that these encoders have similar U-shaped attention curves; it does not validate that CLIP's shallow attention is an optimal, or even a valid, importance map for these architecturally different encoders. The study would be significantly strengthened by testing a 'matched-encoder' guidance, such as using a SigLIP-guided map for the SigLIP-based MLLM."

3. The guidance map is an 8x8 grid quantized to only three levels ($\mu \pm k\sigma$). For a 336x336 input, each guidance block covers a 42x42 pixel patch. This seems incredibly coarse. It is difficult to believe this crude map can effectively guide bit allocation for tasks requiring fine-grained detail or small object recognition. The paper needs to provide much stronger justification or analysis for this coarse-grained approach. Maybe add some fine-grained results.

4. I am confused by some experimental results. For example, Figures 7 and 8 show that Bridge d1 is even worse than  ELIC. Can the author explain how the method is tested and which encoder is used?

### Questions
Please see the weakness part. Overall, I find the paper satisfying and interesting. The paper would be further improved if the authors addressed the weaknesses above.

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
3

### Summary
This paper examines the impact of image compression on MLLMs, showing compression distortion harms cross-level features with low-level details with high-level semantics. Motivated by this insight, the authors propose CoTAM, a codec tailored for MLLMs that adaptively allocates bits based on shallow CLIP attention and reconstructs multi-level features via a lightweight adapter and multi-level loss. The encoder leverages semantic priors to prioritize important regions, while the decoder balances low-level fidelity and high-level semantics. CoTAM is compatible with both image and video MLLMs and supports high-resolution inputs through hierarchical guidance. Experiments show up to 36% bitrate savings without performance loss, outperforming both human-centric and machine-centric codecs.

### Strengths
I think the paper has following strengths:

1. Interesting finding and motivations: The paper provides analysis showing that compression artifacts disrupt cross-level features (merging low-level details with high-level semantics) rather than uniformly harming all visual cues. This finding is well-supported by attention-distance metrics and cosine-similarity experiments across multiple vision encoders. 

2. Good feasibility: The method wraps existing neural codecs with only a lightweight adapter (1 transformer block) plus a negligible 128-bit metadata overhead, making it easy to integrate into production pipelines without retraining.

### Weaknesses
However, I have some small concerns about the paper:

1. Task-Specific Bias in Evaluation: Most of the benchmark tasks are on high-level semantic understanding tasks, what if the author benchmarks some low-level vision-related tasks besides OCR? I'm curious about that.

2. I'm doubtful about the scalability part. CoTAM’s CLIP-guided encoder relies on shallow-layer attention maps (8×8 grid) for bit allocation. While this works for 336×336 inputs, its scalability to higher resolutions (e.g., 4K) or longer video sequences is unclear. The hierarchical guidance extension (Section 3.4) is only briefly tested, with no ablation on how local/global attention fusion scales computationally. A 128-bit overhead for 336×336 images may balloon for larger inputs.

### Questions
In general, I'm satified with the paper, but small concern remains. Please see the weaknesses section for details.

### Soundness
3

### Presentation
3

### Contribution
3
