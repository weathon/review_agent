# Majority Bit-aware Watermarking for Large Language Models

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
The growing deployment of Large Language Models (LLMs) in real-world applications has raised concerns about their potential misuse in generating harmful or deceptive content. To address this issue, watermarking techniques have emerged as a promising solution by embedding identifiable binary messages into generated text for origin verification and misuse tracing. While recent efforts have explored multi-bit watermarking schemes capable of embedding rich information such as user identifiers, they typically suffer from the fundamental trade-off between text quality and decoding accuracy: to ensure reliable message decoding, they have to restrict the size of preferred token sets during encoding, yet such restrictions reduce the quality of the generated content. In this work, we propose MajorMark, a novel watermarking method that improves this trade-off through majority bit-aware encoding. MajorMark selects preferred token sets based on the majority bit of the message, enabling a larger and more flexible sampling of tokens. In contrast to prior methods that rely on token frequency analysis for decoding, MajorMark employs a clustering-based decoding strategy, which maintains high decoding accuracy even when the preferred token set is large, thus preserving both content quality and decoding accuracy. We further introduce MajorMark$^+$, which partitions the message into multiple blocks to independently encode and deterministically decode each block, thereby further enhancing the quality of watermarked text and improving decoding accuracy. Extensive experiments on state-of-the-art LLMs demonstrate that our methods significantly enhance both decoding accuracy and text generation quality, outperforming prior multi-bit watermarking baselines. The code of the proposed methods is available \href{https://anonymous.4open.science/r/MajorMark}{here} for review.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a multi-bit watermarking technique for large language models. The proposed approach selects tokens from a green list that covers the majority of the vocabulary, aiming to maintain the quality of generated text. Building on a majority-bit–aware watermarking framework, the authors introduce two methods: MajorMark and MajorMark+, with the latter offering a higher expected green-list ratio and better suitability for embedding longer messages. Experimental results demonstrate the effectiveness of the proposed watermarking scheme.

### Strengths
+ This is an interesting and important topic.
+ The paper is clearly written and easy to follow. The authors present two variants of their majority-bit–aware watermarking scheme, providing a structured and coherent discussion of their design and application.

### Weaknesses
- The paper claims that previous methods suffer from high computational complexity and that the proposed approaches are computationally efficient. However, the work lacks a detailed comparison or empirical analysis to substantiate this claim. In practice, the proposed methods appear to introduce additional computational costs due to multiple rounds of hashing, clustering-based decoding, and trial-and-error decoding processes.
- The decoding accuracy heavily depends on the specific embedded information. When the numbers of majority and minority bits are similar, decoding accuracy tends to be higher. In contrast, when the embedded bits are highly imbalanced (e.g., significantly more 1s than 0s), the variance in token distribution across shards diminishes, making decoding less effective. For instance, embedding 4 bits into 16 tokens with the message 1100 may yield shard distributions of {8, 8, 0, 0}, while 1110 could result in {5, 5, 6, 0}, illustrating that the decoding feature, variance in shard token counts, depends on the specific bit pattern.
- The proposed system cannot embed messages consisting entirely of 0s or 1s, or containing long substrings of identical bits. While the paper claims such occurrences are rare, this assumption may not hold when messages are short or divided into small blocks (as in MajorMark+). For example, when r=2 and b = 8, approximately 23.4% of the code space becomes infeasible, which is non-negligible.
- The system’s ability to distinguish between watermarked and non-watermarked text—crucial for LLM watermarking—is insufficiently evaluated. Although Appendix A.12 briefly discusses false positives, it omits key experimental details such as the threshold selection procedure and evaluation setup. Detection accuracy also seems to depend on the specific embedded message. Integrating and analyzing the detection method as part of the main algorithm would strengthen the work. In addition, this problem is a critical problem actively studied by the literature.
- The claim that a larger green-list ratio preserves text quality is overstated. While a larger ratio may help preserve fluency, it does not inherently ensure quality. The evaluation primarily relies on perplexity (PPL), which is limited in capturing semantic or stylistic fidelity. Additional text-quality metrics and qualitative comparisons would provide a more comprehensive assessment. Moreover, further justification is needed for using the top-5 hit rate as a key metric, as it could be trivially optimized by restricting watermarking to the top-5 tokens.
- Several experimental details are insufficiently described. For instance, when evaluating robustness against copy-paste and paraphrasing attacks, the ratio or extent of modified text should be explicitly reported.

### Questions
+ How is the embedded information selected during evaluation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an LLM watermarking method that leverages the majority bit and uses a clustering-based decoding strategy to improve the watermarking performance. The method recovers the embedded message by analyzing the occurrence of tokens across predefined vocabulary shards, enabling more accurate decoding.

### Strengths
1. The proposed method avoids the trade-offs as in prior works to improve the watermarking performance. 

2. Experimental results show the performance is superior to some baselines.

### Weaknesses
1. The method is mainly compared to two prior works. Recent and stronger baselines are missing, such as UPV, SIR, SimMark, SemStamp, etc. 

2. The method is largely a statistical refinement of existing token-level schemes, which limits its novelty.

3. Although the results show resistance against modification and paraphrasing attacks, it is unclear from a methodology perspective how the method can help improve the robustness. In addition, performance against stronger attackers also needs to be evaluated. 

4. Complexity and overhead are only briefly discussed, which need more comprehensive evaluations.

### Questions
1. How do the methods compare to more recent works as listed above? 

2. What is the computational complexity compared to prior works?

3. It also needs to be presented in greater detail how the majority bits are computed.

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
3

### Summary
This paper addresses the fundamental trade-off between text quality and decoding accuracy in multi-bit watermarking for Large Language Models (LLMs). The authors observe that existing methods suffer from this trade-off, which is governed by the size of the "green list" of preferred tokens. To overcome this, they propose MajorMark, a novel watermarking paradigm. The core idea is to construct the green list based on the majority bit λ of the message m, which theoretically guarantees a large green list ratio (γ ≥ 0.5) and thus preserves text quality. Crucially, MajorMark abandons traditional frequency-based decoding and instead employs a clustering-based strategy to recover the message by analyzing the distribution of token occurrences across vocabulary shards.
The paper further introduces MajorMark+, an enhanced version that partitions the message into blocks for encoding, leading to even better text quality. For decoding, MajorMark+ replaces clustering with a more robust deterministic decoding mechanism that enumerates possible majority bits and their counts, significantly improving decoding accuracy.

### Strengths
1.This paper reframes the multi-bit watermarking problem by decoupling the decoding process from the green list size. The "majority bit-aware" encoding and the subsequent distribution-based decoding (clustering/deterministic) represent a significant departure from prior art and a truly novel conceptual contribution.
2.This paper demonstrably achieves what it sets out to do: improve the trade-off between text quality and decoding accuracy. The experimental results across the board show that MajorMark+ in particular sets a new state-of-the-art, achieving lower perplexity (better quality) and higher bit accuracy simultaneously. This is a significant practical achievement.
3.This paper provides solid theoretical backing for its design choices, with formal proofs for the guaranteed green list size (γ ≥ 0.5) and its expected value. This combination of theoretical insight and empirical validation is the hallmark of a high-quality research paper.

### Weaknesses
1.While the paper correctly points out that the decoding complexity of MajorMark+ (b-r passes) is far more efficient than exponential methods, it is still a notable overhead compared to single-pass methods like MPAC. For a very long message b and small r, this could become a practical concern.
2.As shown in Figure 4, performance under strong paraphrase attacks degrades for all methods, including the proposed ones. While MajorMark and MajorMark+ are competitive, does the block-wise structure of MajorMark+ make it more vulnerable if a paraphraser alters the specific tokens used to select the block index?
3.The paper notes that extreme messages (all 0s or all 1s) lead to γ=1.0, rendering the watermark ineffective, and suggests disallowing them. While the practical impact is negligible for large b, this is a slight limitation in the universality of the codespace. For MajorMark+, this applies at the block level, which is a clever mitigation but still a constraint.

### Questions
1.Could you provide some empirical numbers on the actual wall-clock time required for decoding?
2.In MajorMark+, what's the reason for the performance drop under paraphrase attacks? Have you considered alternative, more robust ways of assigning tokens to blocks?
3.You propose a threshold on the standard deviation difference to detect unwatermarked text. How was this threshold (value of 2) chosen? Is it sensitive to the message length b or the text length T?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MajorMark and MajorMark+, majority-aware multi-bit watermarking methods for large language models that replace frequency-based decoding with clustering, achieving higher decoding accuracy and better text quality without tuning the green-list ratio γ.

### Strengths
1. It relies on green-list frequency decoding, which inherently struggles to balance text quality and decoding accuracy.
2. The adoption of multi-bit block encoding is appropriate but not conceptually novel.
3. The clustering-based decoding design is adaptive and somewhat innovative, yet its theoretical foundation remains limited.

### Weaknesses
1. The decoding stage relies heavily on unsupervised clustering (K-Means) over vocabulary shards. However, no justification is provided for its stability under diverse token distributions, nor for convergence guarantees or error bounds.
2. The claimed robustness of MajorMark under paraphrasing attacks is demonstrated only empirically, without a clear mechanism or theoretical explanation for why clustering-based decoding should retain signal stability after semantic rewrites.
3. Only claimed results of LLaMA2-7b, other experimented SOTA LLMs should be shown and claimed.

### Questions
1. Please show more accurate SOTA-Based LLM results
2. Could the authors formalize why cluster separability is expected in the latent frequency space, rather than merely observed empirically?

### Soundness
3

### Presentation
2

### Contribution
2
