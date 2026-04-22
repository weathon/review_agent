# Transformers from Compressed Representations

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Compressed file formats are the corner stone of efficient data storage and transmission, yet their potential for representation learning remains largely under-explored. We introduce \textbf{TEMPEST} (TransformErs froM comPressed rEpreSenTations), a method that exploits the inherent byte-stream structure of compressed files to design an effective tokenization and encoding strategy. By leveraging this compact encoding, a standard transformer can directly learn semantic representations from compressed data streams, bypassing the need for raw byte-level processing or full media decoding. Our proposal substantially reduces the number of tokens required for semantic classification, thereby lowering both computational complexity and memory usage. Through extensive experiments across diverse datasets, coding schemes, and modalities, we show that TEMPEST achieves accuracy competitive wit the state-of-the-art  while delivering efficiency gains in memory and compute.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses a specific challenge in low computational resource environments, where training multimedia (audio/image) classification models can be costly due to large data sequence lengths. Existing compression methods typically focus on partially decoded formats or byte-level operations and prioritize input modeling over classification. This study proposes learning directly from compressed representations at a natural block level instead of a byte level, as blocks better retain the semantic information of the data.


The authors utilize natural data blocks found in formats such as MP3, Opus, and JPEG. These blocks are embedded using a transformer network, generating embeddings for each block. Subsequently, these embeddings are compacted in sequence length using a MLP-Mixer-like network, forming the most compressed version, termed the Block Embedded Sequence. This sequence is regularized with a reconstruction objective for regularization purposes rather than as a final goal. The compressed sequence is then processed by a transformer encoder and a linear classifier to produce class predictions.


Experiments were conducted on audio formats in MP3 and Opus at varying bitrates (at least for MP3) and JPEG for images, using the ESC-50, SpeechCommandsV2, and AudioSet datasets for audio, and MNIST for images. The results demonstrate comparative performance to the Audio Spectrogram Transformer model, with reduced computational cost.

### Strengths
- The research demonstrates that compressed formats are viable for classification tasks, offering significantly reduced computational costs.
- Extensive ablation studies examine the impact of Block Embedded Sequence Length, Reconstruction Loss, Depth of the Block Embedding Network, bit rates, and other variables.

### Weaknesses
- The paper appears to focus on a relatively specialized and narrow problem area. Expanding the evaluation to include a broader range of tasks would enhance the understanding of the method's overall applicability and value. 
- Additionally, the block definitions seem tailored to specific inputs, raising questions about their scalability to other formats. A comprehensive analysis in these areas would be beneficial.

### Questions
Could you provide any insights or analyses on the performance of intermediate sequences in tasks beyond classification? How much informational content is retained within these intermediate sequences?

### Soundness
3

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
3

### Summary
The paper introduces TEMPEST, a novel transformer-based approach that directly learns semantic representations from compressed multimedia data (MP3, Opus, JPEG) without requiring full decoding. This is achieved by leveraging the inherent block structure of these formats for natural tokenization, employing a two-stage architecture that processes block-level embeddings followed by sequence-level aggregation. TEMPEST delivers significant efficiency gains, including a 3x reduction in token sequence length and an 11x smaller attention matrix compared to traditional baselines, while maintaining competitive accuracy across diverse datasets.

### Strengths
1. **Novel Conceptual Framework**: The paper introduces TEMPEST, a fundamentally new paradigm to multimedia processing by working directly with compressed formats, e.g., MP3 and JPEG. The authors also incorporate innovative training techniques like bit rate augmentation and multi-bit rate inference that further improve model performance, generalization, and robustness.


2. **Practical Utility**: The demonstrated efficiency gains have clear practical applications in real-world systems with 3x reduction in token sequence length (32 tokens/sec vs. 108 tokens/sec for AST) and 11x reduction in attention matrix size (1,024 elements vs. 11,664 elements).

### Weaknesses
1. **Limited Baseline Comparisons:** The current evaluation primarily benchmarks TEMPEST against AST. While AST is a direct and relevant baseline, it was published in 2021, and the field has seen significant advancements since then.  To fully contextualize TEMPEST's contributions and thoroughly assess its performance, it would be beneficial to include comparisons with more recent methods in the relevant domain. 

2. **Dataset Complexity:** For the image experiment, the evaluation largely relies on datasets such as MNIST, which are relatively simple and may not fully reflect the challenges of real-world scenarios. The authors should consider evaluating TEMPEST on more complex datasets to better demonstrate the method's effectiveness. For instance, despite potential computational resource constraints, exploring performance on CIFAR-10 or CIFAR-100 would offer stronger evidence of the method's robustness and generalizability.

3. **Out-of-Distribution Capability:** The current experiments mainly focus on classification tasks and do not explicitly address TEMPEST's performance under out-of-distribution conditions. However,  investigating how TEMPEST handles or identifies OOD samples would significantly strengthen the paper's claims regarding its practical utility.

4. **Format Dependency and Unification**: The paper mentions that TEMPEST is applicable across various data formats (e.g., MP3, JPEG). However, it appears that the method still exhibits a degree of format dependency, requiring separate training for each format. Could the authors elaborate on whether it is possible to train a single, unified model of TEMPEST that can simultaneously process and handle multiple distinct data formats without format-specific adjustments?

### Questions
The proposed TEMPEST appears to offer a promising approach for efficiently processing large-scale data. Given this potential, it would be beneficial for the authors to discuss how TEMPEST relates to and can potentially combine with existing "adaptive token methods" (e.g., those that dynamically adjust sequence length or token importance [1,2,3]) in the conclusion. 

[1] Duggal, Shivam, et al. "Adaptive length image tokenization via recurrent allocation." First Workshop on Scalable Optimization for Efficient and Adaptive Foundation Models. 2024.

[2] Wen, Tiansheng, et al. "Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation." Forty-second International Conference on Machine Learning.

[3] Hu, Wenbo, et al. "Matryoshka query transformer for large vision-language models." Advances in Neural Information Processing Systems 37 (2024): 50168

### Soundness
3

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
2

### Summary
This paper proposes a compressed-domain transformer framework for efficient semantic representation learning named TEMPEST. Specifically, TEMPEST includes two core modules: Block Embedding Network and Classification Network, regularized by a Block Reconstruction Network. The Block Embedding Network leverages the intrinsic block structure of various compressed file formats to tokenize byte streams into blocks, applies intra-block attention with lightweight transformers, and compresses them into embeddings. The Classification Network aggregates these block embeddings across an entire file using a standard transformer encoder with a [CLS] token for downstream classification. The Block Reconstruction Network guides the embedding process via a reconstruction loss, ensuring semantic fidelity and compactness without full media decoding.

The experimental evaluation in this paper assessed the performance of the proposed TEMPEST across audio and image domains, including three compressed formats and multiple datasets, comparing it with state-of-the-art baselines. The results indicate that the proposed TEMPEST achieves competitive or superior accuracy while reducing sequence length and attention matrix size, thereby providing computational and memory efficiency gains.

### Strengths
- Directly embedding features from compressed files instead of raw data avoids the additional storage and transmission overhead introduced by the decoding process, which is a meaningful and practical advantage for real-world applications.
- Experimental results demonstrate that TEMPEST can improve efficiency while maintaining competitive performance.

### Weaknesses
- The method relies heavily on structural characteristics of specific compression formats, which limits its generality. TEMPEST’s core idea is to use compression blocks rather than bytes as token units, which requires the compression format to have explicit and easily parsable minimal independent decoding units. The authors may need to explicitly clarify which compression formats are supported and whether extra engineering adaptation is required for different compression algorithms.
- The experiments primarily evaluate on MP3, Opus, and JPEG, lacking verification on other common formats such as H.264, HEVC, and FLAC. There is also no analysis of the relationship between different compression formats and model performance.
- Although TEMPEST shows advantages over AST in FLOPs and TPS, the design requires coupling between different sub-networks, making the method more complex, and lacking corresponding experimental analysis.
- The experiments are mostly conducted on small-scale datasets. It remains unverified whether compressed-domain training with TEMPEST on large-scale data (e.g., compressing ImageNet and training with TEMPEST) can match the multi-task generalization performance of raw-domain training.
- The audio tasks are limited to classification, with no evaluation on other task types such as detection, segmentation, or generation.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes TEMPEST, a tokenizer+encoder pipeline that learns directly from compressed file formats (CFFs) by treating blocks (e.g., MP3 frames, JPEG MCUs, Opus frames) as atomic units rather than bytes. Each compressed block is embedded with a lightweight intra-block transformer and channel mixer, then a vanilla transformer encoder aggregates block embeddings for classification. Across ESC-50, SC2, and AudioSet, TEMPEST is competitive with AST while using far fewer tokens per second and lower FLOPs.

### Strengths
1. The paper’s central idea—treating self-contained codec blocks as tokens—is clean and broadly appealing, because it avoids byte-boundary misalignment while preserving the semantics that matter.
2. Despite its simplicity, the approach delivers strong efficiency: it cuts sequence length and attention cost substantially yet remains competitive in accuracy, especially on data-limited regimes like ESC-50.
3. The small reconstruction head is a practical touch that consistently sharpens the block embeddings. I also appreciate the codec-aware perspective on augmentation: training and even inference with multiple bit rates act like multi-view learning and yield measurable gains.
4. Finally, the method shows encouraging performance with a lightweight architecture, suggesting the idea travels beyond a single codec or modality.

### Weaknesses
1. The approach assumes block-structured formats with discoverable boundaries. Not all compression schemes expose clean block markers (even JPEG MCU boundaries require approximation), which may limit generality and complicate deployment outside MP3/Opus/JPEG. 
2. TEMPEST trails AST on SC2 and AudioSet at ~65% FLOPs and much shorter sequences; the paper doesn’t explore accuracy–compute scaling laws (e.g., what happens if TEMPEST matches AST’s FLOPs, tokens, or parameters?).
3. Ablations show accuracy benefits from deeper block embedding encoders but also increased FLOPs (since Et runs over length-144 intra-block tokens). It’s unclear, at fixed FLOPs, whether adding capacity to Et (embedding) or Ec (classifier) yields better returns.

### Questions
1. How does TEMPEST scale in FLOPS compared to AST —does accuracy close the gap on SC2/AudioSet if you increase depth/width or raise?

2. Under a fixed FLOPs budget, where should capacity go for the best returns—into the block encoder E_t or the sequence encoder E_c.

3. For codecs without clean, explicit block markers, can you learn or detect boundaries reliably and how does boundary error impact accuracy?​

### Soundness
3

### Presentation
4

### Contribution
3
