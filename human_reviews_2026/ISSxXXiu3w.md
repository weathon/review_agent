# Chunk Based Speech Pre-training with High Resolution Finite Scalar Quantization

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Low latency speech human-machine communication is becoming increasingly necessary as speech technology advances quickly in the last decade.  One of the primary factors behind the advancement of speech technology is self-supervised learning. Most self-supervised learning algorithms are designed with full utterance assumption and compromises have to made if partial utterances are presented, which are common in the streaming applications.  In this work, we propose a chunk based self-supervised learning (Chunk SSL) algorithm as an unified  solution for both streaming and offline speech pre-training.  Chunk SSL is optimized with the masked prediction loss and 
an acoustic encoder is encouraged to restore indices of those masked speech frames with help from unmasked frames in the same chunk and preceding chunks.  A copy and append data augmentation approach is proposed to conduct efficient chunk based pre-training.
Chunk SSL utilizes a finite scalar quantization (FSQ) module to discretize input speech features and our study shows a high resolution FSQ codebook, i.e., a codebook with vocabulary size up to a few millions, is beneficial to transfer knowledge from the pre-training task to the downstream tasks.   A group masked prediction loss is employed during pre-training to alleviate the high memory and computation cost introduced by the large codebook. The proposed approach is examined in two speech to text tasks, i.e., speech recognition and speech translation. Experimental results on the \textsc{Librispeech} and \textsc{Must-C} datasets show that the proposed method could achieve
very competitive results for speech to text tasks at both streaming and offline modes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a chunk-based self-supervised learning (SSL) framework that masks both future chunk information and sampled frames within each chunk. The method is efficiently implemented through copy-and-append data augmentation techniques, enabling scalable training. In addition, the paper introduces a multi-label prediction scheme, inspired by audio codecs, to efficiently handle large vocabulary sizes. Training is conducted using dynamic chunk sizes, allowing the model to learn representations across varying temporal contexts, while inference leverages the chunk structure to balance latency and performance. Experimental results on LibriSpeech ASR and MuST-C speech translation tasks demonstrate the effectiveness of the proposed approach.

### Strengths
- The paper presents an efficient implementation of chunk-based SSL, which is particularly valuable for streaming ASR and ST applications where low-latency processing is critical.
- The paper also provides an efficient discretization by using multi-label classification.
- The proposed approach demonstrates strong experimental performance, outperforming existing SSL-based methods on both ASR and ST benchmarks.

### Weaknesses
* **Unclear masking implementation (potentially critical):**
  The manuscript’s description of the masking operation is ambiguous. The text appears to apply masks by multiplying masked positions by **0** rather than using large negative values before the softmax (or otherwise excluding them from the attention denominator). If true, this would leave nonzero attention weights on masked/future positions (the softmax denominator still includes those entries), allowing the model to attend to information that should be hidden. This is a fundamental concern that could invalidate the reported results. Please clarify precisely how masking is implemented (include equations), and provide an ablation that compares the current implementation with a standard masked-attention formulation (e.g., add −∞ or a large negative constant before softmax) to show the effect on performance and verify correctness.

* **Limited novelty and missing literature context for multi-label representations:**
  The multi-label / codec-like representation component echoes several recent works on discrete and multi-label audio representations (for example, MMM and other codec-inspired SSL approaches). The paper currently does not situate its contribution clearly within this literature nor convincingly establish novelty. Please expand the related-work discussion to cover recent multi-label and codec-based representation studies, and clarify what is new here (algorithmic change, efficiency, empirical regime, etc.). If novelty is incremental, the manuscript should present stronger empirical or analytical evidence that the proposed variant offers a meaningful advantage over prior methods.
  * Shi, Jiatong, et al. "MMM: Multi-Layer Multi-Residual Multi-Stream Discrete Speech Representation from Self-supervised Learning Model." Proc. Interspeech 2024. 2024.

* **Insufficient analysis of representation trade-offs (phoneme purity alone is inadequate):**
  The analysis relies heavily on phoneme-label purity as a measure of representational quality. However, phoneme purity alone favors larger codebook / vocabulary sizes and does not capture clustering compactness or redundancy trade-offs. The paper should include additional probing analyses based on **cluster purity**, and report trade-offs between vocabulary size, cluster purity, and downstream performance. Presenting such metrics (and plotting cluster/phoneme purity for each vocabulary size) would provide a more trustworthy and nuanced interpretation of why larger vocabularies help or hurt.

### Questions
My primary questions are already listed in the Weaknesses field. If these questions are addressed satisfactorily, I am open to revising my scores.

I also have a few minor questions and suggestions that could improve the clarity and comparability of the paper:

- Why was Conformer chosen as the backbone? Many SSL models (e.g., Wav2Vec 2.0, HuBERT, WavLM) are based on standard Transformers, which would allow for a clearer comparison.
- Similarly, regarding model size, it would improve comparability if the proposed method used a similar number of parameters as the baseline models.

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
4

### Summary
The paper proposes Chunk SSL, a chunk-wise self-supervised pretraining framework that aims to unify streaming and offline speech encoders. Core pieces: (1) Copy-and-Append Data Augmentation (CADA) to parallelize chunked masked-prediction pretraining; (2) Finite Scalar Quantization (FSQ) to discretize frames, with very large codebooks; and (3) a group masked-prediction loss to keep compute/memory manageable with high-resolution codebooks.

### Strengths
- The paper proposes an alternative streaming SSL recipe that aims to unify pretraining for both streaming and offline speech encoders within a single framework

### Weaknesses
- The method largely recombines familiar pieces—chunked SSL, discretization, and masking—with the main novelty in CADA and the FSQ + group-loss design.
- Given how widely SSL embeddings are used, the paper’s downstream evaluation feels narrow; broader tasks (e.g. speaker verification/diarization) would better demonstrate generality.
- Although effective, the approach depends on millions-level FSQ codebooks; as the vocabulary grows, optimization becomes harder and WER degrades—raising concerns about training stability and compute cost.
- Some implementation details for CADA are hard to follow; it would help to include a concise algorithm box/pseudocode.

### Questions
- I see clear gains for streaming ASR, but the offline WERs trail the comparative approach. Could you explain why? Intuitively, when the streaming constraints are relaxed, an effective method should at least match—ideally surpass—offline baselines.
- As a follow-up: if CADA is strictly equivalent to naïve chunk SSL (only enabling parallelization), does that imply the conventional naïve approach could also support streaming—and, if trained identically, would its offline performance remain competitive?
- Please also address the weaknesses noted above. I’m happy to raise my score if these concerns are satisfactorily resolved.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes an integrated framework for both non-streaming and streaming self-supervised speech training. By introducing a new CADA masking strategy and FSQ-based target prediction, the unified model achieves performance comparable to prior specialized models for either setting. However, additional ablation studies are required to justify the benefit and usage of CADA and FSQ. In particular, CADA might be replicable by simply applying both non-causal and block-wise causal attention masks during self-supervised training, without the need for copy and append. Likewise, a comparison for adopting FSQ over continuous targets would greatly justify its use.

### Strengths
1. The paper introduces a model capable of dynamically adjusting the chunk size during both training and inference, enabling users to flexibly control latency and reducing the mismatch between training and inference.

2. The proposed CADA approach might be a potential alternative to previous masking strategies for self-supervised prediction, e.g. Best-RQ and Wav2vec 2.0.

### Weaknesses
1. If the goal is to obtain fine-grained prediction targets, why not consider continuous representation could serve as a more efficient alternative, e.g. mel-filterbank features. The benefit of quantizing mel-filterbank features using FSQ for mask prediction is not clearly established. Since FSQ quantizes each channel independently, it is reminiscent of the approach in dMel: Speech Tokenization Made Simple. However, it seems plausible that continuous mel-filterbank features could suffice for training. A direct comparison between models predicting continuous mel-filterbank features and those using FSQ targets would be necessary to justify the added complexity of introducing an FSQ encoder.

2. As shown in Figure 2(b), it is unclear why the “1st chunk” is not appended as well. Assigning zero attention scores to the 'full context 1st chunk' to the 'appended 1st chunk' should work.

3. Following above, when using “offline training with infinite chunk size", Figure 2 shows no indication of the masked chunk being copied and appended for prediction, which does not seem right.

4. A more thorough comparison with simple joint causal and non-causal masking during training would strengthen the justification for using CADA. For example, training a Best-RQ-style alternative that includes both chunk-wise causal attention mask and non-causal attention masks within each batch would be a fair baseline. This could also justify CADA using twice the computation during training.

### Questions
1. What is the motivation for introducing an additional FSQ module for target prediction instead of directly using mel-filterbank features?

2. In the non-streaming training setup, how is masked prediction performed if the first chunk is not appended?

3. Have you explored training a unified model for both streaming and non-streaming scenarios without the copy and append operation?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a chunkwise pretraining scheme for speech self-supervised learning (SSL), aiming to enhance streaming capability for downstream tasks, including automatic speech recognition (ASR) and speech-to-text translation (ST). The authors develop a training scheme that allows efficient training of the chunkwise speech SSL by using the specialized "copy and append data augmentation (CADA)" technique. In addition, they employ a masked-prediction objective using finite scalar quantization (FSQ) codes as targets. They evaluate their Chunk SSL speech models on ASR and ST, reporting both offline (non-streaming) and online (streaming) performance.

### Strengths
[S1]: The motivation is clear. The chunkwise SSL could intuitively help on the streaming performance of the specified downstream tasks. 

[S2]: The method appears to be training-efficient.

### Weaknesses
[W1]: The presentation requires further refinement. I cannot fully grasp the precise meaning of some sentences. Even the context of the core design—CADA—is somewhat confusing. See [Q1].

[W2]: The completeness and soundness of the experiments are questionable. The authors propose CADA and FSQ on speech SSL, but I cannot infer how effective CADA or FSQ are individually from the provided results. See [Q2].

[W3]: The novelty is limited. The novelty mainly lies in CADA. However, I consider CADA as a technique for efficiently train the chunkwise SSL model, at the cost of increased memory usage due to the "copy-and-append" behavior. The idea of trading memory for time is not particularly novel or impressive. I also have a few questions regarding the core design choices; please see [Q3].

### Questions
**[Q1] Questions regarding the presentation:**

[Q1-1] In L74-76, *"We hypothesize that speech frames associated with a high resolution FSQ token are mainly mapped to an unique modeling unit in the downstream task and it makes the knowledge transfer easier from the pre-training stage to the fine-tuning stage."* 
What does this sentence mean? What is *"an unique modeling unit in the downstream task"*? Do you have any support on this?

[Q1-2] In L106-109, *"Inspired by the implementation of the chunk encoder with a look-ahead chunk for the streaming speech translation (Liu et al., 2021), we propose a copy and append data augmentation for the Chunk SSL, which reorganizes input data and changes the augmented data computation, but it is still strictly equivalent to the computation in the naive Chunk SSL algorithm."* 
I am unsure which part of Liu et al.’s work inspired you, and what exactly the ‘naive chunk SSL algorithm’ refers to (seems to be a critical baseline).

[Q1-3] In L115-118, *"Assuming the chunk size is 4 frames, the speech features are segmented into three base chunks (in cyan): “B1”,
“B2” and “B3”. Chunk “B2” and “B3” are copied and appended to the end of the utterance as “E1” and “E2” (in lime). They correspond to the extended chunks of “B1” and “B2” respectively. There is no extended chunk for the last chunk (“B3”)."* 
The statement that chunks “B2” and “B3” are copied and appended (extended) contradicts the claim that there is no extended chunk for the last chunk “B3.”

[Q1-4] Overall, I cannot quite get why we need to ‘copy and append,’ and how intuitively effective this approach can be, given that the *naive chunk SSL algorithm* is not well introduced or clearly defined as a baseline.

[Q1-5] In L135, what is X? You should define the transformation from X to X′ (the CADA-augmented utterance) more precisely and carefully. 

[Q1-6] Figure 2-(b) is confusing. In the main text that discuss it, why does "B1" pair with "E1" (L203)? Shouldn’t "B2" pair with "E1," according to Figure 1?

---

**[Q2] Questions about the soundness of experiments**

[Q2-1] In Table 1 and Table 2, only the Chunk SSL scores are reported. However, to demonstrate the effectiveness of the proposed method (CADA), the so-called *"naive Chunk SSL"* should be included as a baseline. 

[Q2-2] Following the previous question, how much benefit is gained from using FSQ?

[Q2-3] Why does Table 3 not include other SSL methods?

[Q2-4] Are all baselines in Tables 1 and 2 taken from the literature? If so, how do you ensure the consistency of experiments between the previous work and your work? Has BEST-RQ released its codebase?

---

**[Q3] Questions regarding the core design of CADA**

[Q3-1] Can you elaborate on the novelty of CADA? What do you think is beyond "technical training efficiency improvements"?

[Q3-2] What is "copy and append" mainly about? Is it about 1. enabling look ahead and 2. getting the target from the base output for the extended chunks?

[Q3-3] If it is all about the two aforementioned points, why not simply apply batch expansion? We can have the first half of the batch unmasked and have the look-ahead causal chunked attention masks, while the second half (also copied from the first half) partially masked and having the corresponding causal chunked mask that is equivalent to CADA's design. The batch-expanded version seems more natural to me, since it requires no length expansion for each utterance and avoids specialized convolution operations (split and concat in Figure 2-(b)).

### Soundness
2

### Presentation
1

### Contribution
2
