# Variable-Length Audio Fingerprinting

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
Audio fingerprinting converts audio to much lower-dimensional representations, allowing distorted recordings to still be recognized as their originals through similar fingerprints. Existing deep learning approaches rigidly fingerprint fixed-length audio segments, thereby neglecting temporal dynamics during segmentation. To address limitations due to this rigidity, we propose Variable-Length Audio FingerPrinting (VLAFP), a novel method that supports variable-length fingerprinting. To the best of our knowledge, VLAFP is the first deep audio fingerprinting model capable of processing audio of variable length, for both training and testing. Our experiments show that VLAFP outperforms existing state-of-the-arts in live audio identification and audio retrieval across three real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a Transformer-based framework for the fingerprinting of variable-length audio fingerprinting. The approach in this paper dynamically segments audio based on spectral entropy and learns fingerprints through a dual-attention Transformer using supervised contrastive learning. Judging by the results reported in the paper, the approach successfully captures temporal and semantic coherence in variable-length signals, achieving state-of-the-art performance in both live audio identification and retrieval tasks on multiple datasets. Ablation experiments confirm the importance of the variable-length design and dual-attention structure for robust fingerprint generation under diverse distortions.

### Strengths
1. The approach proposed is reasonably novel, both in architecture and in the problem it solves.
2. The formulation of segmentation logic and the learning processes is theoretically sound and verifiable.
3. The work is clearly presented, and the appendices provide additional details that allows for better understanding (with one important caveat)
3. Robust fingerprinting of variable-length sequences (like audio), especially against resampling and speed adjustments, is of significant interest in signal processing-related tasks.

### Weaknesses
The segmentation threshold, being manually tuned rather than learned, might limit domain-transferability. The experiments on the effect of entropy thresholds also seem to be lacking in granularity.
The experiments reported seem to indicate 
Presentation of Variable Length Segmentation algorithm is contained in Appendix B rather than the main body of the manuscript. As appendices may not be published in the proceedings, leaving out such an important part may not be desirable.

### Questions
1. The entropy threshold for segmentation seems empirically derived, but the exploration and discussion could be a bit more thorough. Do the authors plan on further investigating alternative methods of segmentation (such as more semantically-related approaches)?
2. The authors may consider moving some of the discussions regarding the Variable Length Segmentation algorithm to the main body of the text.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an attention-based audio fingerprinting method capable of handling variable-length input audio.

### Strengths
The core idea leverages two attention mechanisms—inter-frame and frame-to-segment attention—to capture both local and global contextual information from audio frames, which is a reasonable design choice.

### Weaknesses
However, several significant issues undermine the paper’s contribution and rigor:

1. **Lack of Technical Precision**: The mathematical formulation is often imprecise. For instance, in Section 3.1, the use of an approximation symbol (≈) is misleading; the intent appears to be minimizing the distance between *z* and *z*′ , which should be explicitly stated as an optimization objective. Moreover, symbols are frequently introduced without definition—e.g., *H* and *s* appear abruptly in the Frame-to-Segment Cross-Attention section, and *z* is used in the loss function (Section 3.3) without prior explanation.
2. **Limited Novelty**: The proposed architecture does not introduce substantial methodological innovation beyond combining existing attention mechanisms in a straightforward manner.
3. **Insufficient Experimental Detail**: Key implementation details are missing. For example, it is unclear how the baseline models (Wav2Vec2, HuBERT, and AST) were adapted for the fingerprinting task—specifics regarding classifiers, optimizers, training protocols, etc., are omitted. This omission raises concerns about the fairness of the reported comparisons and may suggest an attempt to understate baseline performance.
4. **Unsubstantiated Claims**: The introduction identifies three specific limitations of prior work—Loss of Natural Boundaries, Redundant or Noisy Context, and Distortion Incompatibility—but the paper never empirically investigates how these issues affect performance or how the proposed method mitigates them. The conclusion that “VLAFP likely benefits most from variable-length segmentation” is presented as intuitive rather than evidence-based, weakening the motivation and contribution of the work.
5. **Missing Methodological Clarification**: The computation of the spectral entropy score—a key evaluation metric—is not described, making it difficult to interpret or reproduce the results.

Overall, while the problem setting is relevant, the current presentation lacks the technical rigor, novelty, and empirical validation required for publication in its present form.

### Questions
The above weaknesses.

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
4

### Summary
This paper proposes Variable-Length Audio FingerPrinting (VLAFP), a novel deep learning framework for audio fingerprinting that overcomes the limitations of fixed-length segmentation prevalent in existing methods. VLAFP introduces a transformer-based architecture with dual-attention mechanisms—self-attention to model inter-frame dependencies and cross-attention to aggregate frame-level features into segment-level embeddings. A key innovation is the variable-length segmentation strategy based on spectral entropy, which dynamically determines segment boundaries to preserve semantic coherence. The method is evaluated on two tasks—Commercial-Broadcast Retrieval (CBR) and Dummy-Target Retrieval (DTR)—across three real-world datasets (FMA, LibriSpeech, AudioSet), demonstrating superior performance over state-of-the-art baselines in terms of precision, recall, F1, and top-1 hit rate. The paper also includes comprehensive ablation studies and hyperparameter analysis, supporting the effectiveness of the proposed design choices.

### Strengths
1. Novelty of Variable-Length Processing: To the best of the reviewers’ knowledge, VLAFP is the first deep audio fingerprinting model capable of handling variable-length inputs during both training and inference. This addresses a critical limitation in existing methods and enables more natural and semantically meaningful segmentation.

2. Well-Motivated and Effective Segmentation Strategy: The spectral entropy-based segmentation is well-justified and grounded in signal processing principles. The method adaptively forms segments based on acoustic homogeneity, avoiding unnatural cuts across silence or phonetic boundaries, which is particularly beneficial for speech and music signals.

3. Strong and Consistent Empirical Results: VLAFP consistently outperforms both specialized audio fingerprinting models (NAFP, AMG) and general-purpose audio representation models (wav2vec2, HuBERT, AST) across multiple datasets and evaluation metrics. The improvements are significant, especially in the CBR task, with F1 scores increasing by over 5 points on FMA and LibriSpeech.

### Weaknesses
1. Computational Overhead at Inference: The paper notes that VLAFP has a longer inference time due to the overhead of handling variable-length segments. This could be a practical limitation for real-time applications, and the trade-off between accuracy and efficiency is not deeply analyzed.

2. Evaluation on Synthetic Distortions: The audio augmentations (time-stretching, background noise, impulse response) are synthetically applied. While standard in the field, real-world broadcast distortions may be more complex and heterogeneous, and the robustness of VLAFP in truly uncontrolled environments remains to be validated.

### Questions
1. The paper compares against wav2vec 2.0 and HuBERT but does not specify which model variants (e.g., Base, Large) were used. Could the authors clarify the exact versions and whether they were fine-tuned for the fingerprinting task? Additionally, would more recent self-supervised learning (SSL) models—such as WavLM or Whisper—yield better performance?

2. The paper states that HuBERT achieves near-perfect recall but very low precision on CBR, suggesting it indiscriminately classifies most segments as commercials. Could the authors provide a more in-depth analysis of why this occurs? Is it due to the model's training objective, the nature of its learned representations, or a mismatch between its pre-training task and the fingerprinting task?

3. The segmentation algorithm uses a z-score threshold θ. The results in Table 7 show that smaller θ values (leading to shorter segments) yield better performance. Can the authors elaborate on the trade-offs of using very short segments?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper extends the notion of fixed-length segment audio fingerprinting to **variable-length** segmentation. An audio recording is divided into variable-length segments determined by measures such as spectral entropy, and each segment is fed into a transformer-based encoder to obtain an embedding (fingerprint).
Training is performed by contrastive learning.
Experiments have been performed over speech, music, and acoustic events datasets.

### Strengths
### Originality
- The paper provides a clean extension of neural audio fingerprinting by replacing the convolutional encoder with a transformer encoder and by proposing a segment-level embedding refinement strategy.  
- The model is trained using a standard contrastive loss as used in the baseline Neural Audio FingerPrinting (NAFP) [Chen et al.]. 
- The idea of segmenting database recordings into variable-length segments based on spectral entropy is conceptually interesting and practically motivated.  
- While simple, the formulation is a logical step forward for audio fingerprinting systems.  

### Clarity
- The presentation is clear and easy to follow.  
- The description of the variable-length segmentation procedure and cross-attention refinement is well articulated.

### Significance
- The model introduces a new approach to neural audio fingerprinting and offers a straightforward way to handle variable-length segments.

### Weaknesses
### Conceptual 
- The paper is primarily empirical, with limited theoretical analysis.  
- The empirical analysis could be enhanced by including other tasks that rely on audio representation learning. Currently, the scope is narrowly focused on audio retrieval; broader implications for general-purpose audio representation learning are unexplored. 
- The method restricts segment lengths to a predefined range \([T_{\min}, T_{\max}]\), raising a question about how queries shorter than \(T_{\min}\) could be handled. Additionally, events shorter than $T_{\min}$ may be merged with neighboring frames, potentially reducing fingerprint specificity.

### Experiments
- Another important metric to compare is storage efficiency (how much storage space the fingerprints require).
- The proposed method has a high hit rate, but is slower to search (Table 4). Could this be a bottleneck for practical use? I have a question here: What is the necessity of loading and locating audio when we are searching in the fingerprint database (Sec. 5.3)?
- Fig. 5: How is the hit rate close to $100\%$ with $\theta=\infty$? I think it will lead to segments of fixed length without any overlap, and hence, any relative time delay between the query segment start time and the target segment start time will result in the embeddings not matching.

### Presentation
- What is spectral entropy? Variable-length audio segmentation is an essential part of the contribution, but has been discussed very superficially in the paper. 
- Fig. 6: In the comparison across different segmentation methods, it appears that the waveform method is better than spectral segmentation, although marginally, for many query durations. It would be beneficial if the paper also compared these methods in other ways, such as the nature of the segments obtained, to justify the choice of one method.
- At some places, the notations are written in a very non-standard way. E.g., in line 124, $A=A[1,2,...,N]$. 
- Minor typos. E.g., line 023: advertisers has.


### Overall
The contribution looks less on the ML side and more suitable for an audio-focused venue such as INTERSPEECH, ISMIR, ICASSP, or IEEE TASLP.

### Questions
- Please address the questions mentioned in the Weaknesses section.
- Could you provide additional experiments demonstrating the model’s potential as a general-purpose acoustic representation learner (e.g., on non-retrieval audio benchmarks)?  
- How does the method behave for queries shorter than the minimum segment length \(T_{\min}\)? Are they padded, merged, or discarded?

### Soundness
2

### Presentation
3

### Contribution
2
