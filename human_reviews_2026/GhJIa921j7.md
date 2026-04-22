# ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Modern language models (LMs) still rely on fixed, pre-defined subword tokenizations. Once a tokenizer is trained, the LM can only operate at this fixed level of granularity, which often leads to brittle and counterintuitive behaviors even in otherwise strong reasoning models. We introduce \textbf{ByteFlow Net}, a new architecture that removes tokenizers entirely and instead enables models to learn their own segmentation of raw byte streams into semantically meaningful units. Our approach is grounded in information theory: ByteFlow Net performs compression-driven segmentation based on coding rate of latent representation, allowing the model to dynamically evaluate the information cost of grouping bytes and decide chunk boundaries during processing. Unlike prior self-tokenizing methods that depend on brittle heuristics with human-designed inductive biases, ByteFlow Net adapts its internal representation granularity to the input itself. Experiments demonstrate that this compression-based chunking strategy yields substantial performance gains, with ByteFlow Net outperforming both BPE-based Transformers and previous byte-level architectures. These results suggest that end-to-end, tokenizer-free modeling is not only feasible but also more effective, opening a path toward more adaptive, robust, and information-grounded language models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes ByteFlow Net, a tokenizer-free language modeling framework that processes raw UTF-8 byte streams and dynamically learns hierarchical chunk boundaries through an information-theoretic coding-rate criterion. Compared to previous work, their method uses the rate-distortion criteria to determine the chunking boundary, which is new and novel.

### Strengths
The paper highlights tokenization rigidity as a key bottleneck for multilingual and structure-sensitive modeling, and motivates a tokenizer-free approach that adapts granularity both theoretically (information control) and practically (no preprocessing or vocabulary drift). Its framing as adaptive compression rather than tokenization replacement is elegant and insightful.

The use of coding rate—essentially a differentiable measure of local representational complexity—to determine chunk boundaries is elegant and grounded in information theory. Also see weakness and questions.

The writing is clear and easy to follow.  The authors also provided complexity analysis and training details for reproducibility.

### Weaknesses
While the information-theoretic formulation is novel, the paper’s framing of the criterion as a Shannon coding rate is, in my view, not entirely rigorous — the connection to classical source-coding arguments is more heuristic than formal.
Nevertheless, the proposed approach remains compelling, as the quantity introduced in Equation (11) effectively captures the sensitivity of the local representation to new information: it measures how much the embedding space expands or changes when a new byte is added.
In this sense, even if the theoretical grounding is loosely tied to Shannon’s framework, the method still functions as a principled sensitivity-based chunking rule that adaptively identifies informative boundaries in practice.

### Questions
For equation (12), I think using mutual information can give a better estimate. Is there any relation between the proposed metric and mutual information?

The author can try this new information metric as a potential future work. https://arxiv.org/pdf/2002.10689

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a new dynamic chunking method for replacing the BPE tokenizer with a byte-level, hierarchical model. The authors use a lossy code rate metric with top-k over the sequence to select tokens. They show that their segmentation method performs better than the competitors both on bits-per-byte and downstream performance. They also do ablation studies to justify their selection method.

### Strengths
- Principled method
- The paper is mostly well written and easy to read (except one part, see below)
- The method performs better than the baselines

### Weaknesses
- The method relies on top-k over sequence, which (1) leaks minimal information from the future, (2) it is unclear how to apply it in an autoregressive setting. It is unclear how their code rate-based approach can be generalized to an autoregressive setup. Maybe with an auxiliary predictor, like in Mixture-of-Depths [1], or a learned threshold using a PI controller like [2]. The authors do not discuss this limitation anywhere in the paper. 
- While the paper is mostly well written, eq. 15 is unclear. What is the definition of IndexedMatMul exactly? If it is "IndexedMatMul efficiently applies position-specific linear transformations...", then why does it need the output of chunk()? What is the argument of this chunk? How many matrices are there? Is T in eq 15 the max sequence length? Then these matrices add up to a huge number of parameters.
- The authors do not show the FLOP requirements of their model vs the baseline, so it is not possible to know if the performance improvement is solely because of the higher number of FLOPs spent by the model, or due to an advantage of the byte-level modelling.

[1] Raposo et al. 2024: Mixture-of-Depths: Dynamically allocating compute in transformer-based language models
[2] Kallini et al. 2024: MrT5: Dynamic Token Merging for Efficient Byte-level Language Models

### Questions
- L 104: "We introduce a new paradigm that replaces static tokenization with dynamic, learned segmentation.". This is not a new proposal from this paper, there were similar papers before, like Nawrot et al. 2023, cited by the authors. I propose to replace this with one that focuses on the principled chunking approach instead.
- What is \Delta in L161? Shouldn't this be V \in [0, 258)? 
- What is the isolated effect of Canon? While I agree with the fact that SWA is important, I don't see why an additional Canon layer is necessary, or what effect it would have if added to the BPE baseline.
- In the paragraph "Why Not Global Threshold" the OOM issue is easily solvable by adding an emergency hard-top-K that should never be hit during normal operation, but would prevent OOM. This would also prevent the information leakage and the missing autoregression issue.
- Tab 2: Why are the ByteFlow Net results underlined even when the baseline has higher accuracy?
- Tab 3: Why are the baseline results underlined when other methods have better performance?
- Fig 3. is not discussed in the main text, only mentioned in the reproducibility statement. What are the categories shown by the colors, and what is the conclusion from the figure?

### Soundness
2

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
This paper proposes ByteFlow Net, a tokenizer-free LM that works directly on UTF-8 bytes. Instead of relying on a fixed subword vocabulary, the model learns where to place boundaries by scoring each byte position with an information-theoretic quantity: the approximate lossy coding rate of the current latent representations, and then select the top-K highest information gain positions to a higher-level stream. The architecture has a clear hierarchy: a local byte encoder (SWA + Canon layers) to get contextual byte features, the coding-rate chunker to downsample, a deeper/wider global transformer to do the standard high-level modeling, and a symmetric decoder that brings things back to the byte level. On FineWeb-Edu-100B, the model outperforms both BPE-based LLaMA baselines and several recent byte/hierarchical models.

### Strengths
1. The paper tackles a real bottleneck: static tokenization is non-learnable, language-specific, and fixes the granularity. The proposed method uses coding rate as the segmentation signal, which seems a principled alternative. The connection to rate–distortion is neat.
2. The hierarchical encoder–decoder is well-motivated and aligned with efficient modeling strategies, and applying SWA + Canon layers provides a realistic path for scaling byte-level models efficiently. 
3. The empirical study shows performance improvement of the proposed method.

### Weaknesses
1. The paper does not report the actual cost of computing the coding-rate scores (the log-det–style term and its approximation). Since this has to run per sequence and per step, training-time overhead and distributed stability (e.g. with variable K, mixed-length batches) should be quantified. 
2. Its practical advantage over existing byte-level or tokenizer-based architectures remains small at current scales (less than 1.3B). Demonstrating competitive performance at more than 7B parameters or on multilingual corpora would strengthen its real-world relevance.
3. Figure 3 is underspecified. It is supposed to justify the manifold-preserving claim, but from the figure alone it’s hard to tell (i) what each point is, (ii) how many sequences are visualized, and (iii) whether differences are due to t-SNE randomness. As it stands, this is the weakest link in the “why it works” narrative.

### Questions
1. Can you give a side-by-side training-time comparison (tokens/s or bytes/s) with a standard BPE transformer (perhaps with sliding window attention), a hierarchical byte model with fixed chunking (stride / whitespace), and another dynamic chunker (e.g. cosine-based)? Right now it’s not clear what we pay in practice for using ByteFlow Net.
2. Can you report runtime and memory for the coding-rate module itself (forward + top-K selection), and compare it with other methods?
3. Can you clearify Figure 3? My understanding is: each dot is the contextualized representation of one byte position (after the local encoder), projected to 2-D by t-SNE; different colors correspond to different text segments from FineWeb-Edu; and you plot several such segments together to see whether the chunker keeps the original clustering. If that’s right, can you state it explicitly in the caption, and say how many segments/colors you used and how many points per color (is it the full sequence length T per segment?).

---

Minors:
Line 189: along -> alone
Line 097: resolution

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
A byte level llm, trained from scratch similar to the byte latent transformer but with a different tokenization boundary detection algorithm. Empirical results seem to support their work.

Note i will assign scores after the discussion, dont take them soo seriously now.

### Strengths
Intuition for their work is solid.

Strong abelations

### Weaknesses
I would like to see a more in depth discussion of computational overhead and how this could be improved

### Questions
I cant say i understand the tsne section can u explain the intuition again?

Is this algorithm hard ware friendly

Can u talk about optimization with such a morel.

### Soundness
2

### Presentation
2

### Contribution
2
