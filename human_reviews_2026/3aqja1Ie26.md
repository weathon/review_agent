# Learning Medium-Sensitivity Functions: A Case Study on QR Code Decoding

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
The hardness of learning a function that attains a target task relates to its input-sensitivity. For example, image classification tasks are input-insensitive as minor corruptions should not affect the classification results, whereas arithmetic and symbolic computation, the learning of which has been recently attracting interest, are highly input-sensitive as each input variable connects to the computation results. 
This study investigates the learning functions of medium sensitivity through learning-based Quick Response (QR) code decoding, which has both sensitivity to the change of plain texts and insensitivity to the bit flips. 
Our experiments reveal that Transformers can robustly decode QR codes, even beyond the theoretical error-correction limit, while remaining sensitive to single‑character changes in plain texts. We demonstrate that the robust decoding ability is derived from the regularity of natural language words. Transformers trained on English-based datasets learn to exploit it. Interestingly, this generalizes to words in different languages and to random alphabetical strings. To our knowledge, this study provides the first case study of learning medium-sensitivity functions and also suggests potential applications of learning-based QR code decoding that boost classical methods in combination.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies how Transformers perform on QR code decoding. The authors first train a model to identify the mask pattern and then use a Transformer to decode the QR data. They find that Transformers remain robust under bit-flip noise even without using error-correction codes, achieving performance beyond the theoretical limit. The authors suggest that this robustness comes from the model’s ability to capture language structures, as shown by the variation in decoding success across different languages. This implies that Transformers may correct some intentional misspellings in QR codes.

### Strengths
The paper explores an interesting and relatively unexplored task. The authors evaluate the model’s performance across different mask patterns and provide an insightful analysis of why Transformers can surpass classical decoding algorithms and even exceed theoretical performance limits.

### Weaknesses
The task is interesting, but I am concerned about the level of difficulty. Since Reed–Solomon doesn’t modify the original URL codewords, a Transformer is effectively learning to ignore irrelevant components (function patterns, err correction words) and to map 8-bit codewords back to characters. Its apparent error-correction ability comes from modeling language structure rather than true parity-based correction.

### Questions
(1) What's the architecture/size of the mask-classification model?
(2) Is a 12-layer Transformer unnecessarily large for retrieving a short URL, which typically consists of only a few words or tokens? A smaller or shallower model might achieve similar performance with far less computational cost.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a case study on learning medium-sensitivity functions through the task of QR code decoding. The authors construct several URL datasets and employ a standard Transformer architecture and training strategy. Experimental results show that Transformer-based QR code decoding achieves a high success rate, demonstrates robustness to various levels of corruption, and generalizes beyond English-rich data to other languages. This study provides insights into the potential of learning-based approaches for medium-sensitivity tasks.

### Strengths
1. The paper is well written and easy to follow. The motivation for studying medium-sensitivity functions is clearly articulated.

2. The ablation study is comprehensive, covering different mask patterns and corruption types, including flip and burst errors.

### Weaknesses
1. While QR code decoding is an interesting example and the ablation studies are detailed, the overall scope of the paper is narrow. It is unclear how the findings can be generalized to a broader set of medium-sensitivity tasks.

2. The evaluation is limited. The impact of model architecture is not explored, and the test set consists of only 1,000 samples, compared to 500,000 training samples. Increasing the diversity and size of the test set would help validate the generality of the results.

### Questions
1. Is it possible to use a hybrid input order? For example, Table 1 shows the effects of row and column ordering. Could a combined strategy, such as block-wise or diagonal ordering, be beneficial? Could the performance differences be attributed to positional embeddings?

2. It is notable that a learning-based method can achieve high success rates in QR code decoding. Are there other real-world applications of medium-sensitivity functions where such methods could be applied? Can the learned model generalize to these other tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This submission is about using transformers to decode QR codes. The overall takeaway is that the models learn to do this extremely well.

The work is pitched as the first study into learning what the authors term *medium-sensitivity* functions. This does not seem to be formally defined, but they compare with parity learning (high sensitivity) and image classification (which they call low sensitivity).

### Strengths
The experiments seem interesting and I expect this work will be turned into a solid contribution in the future.

### Weaknesses
The authors have missed key prior work. I found a blog post [1] and Github repo [2] on learning neural networks to decode QR codes specifically. 

I am not an expert, but there appears to be decades of work on using neural networks to decode error-correcting codes [3,4] and more recent work that includes training [5,6].

It seems that a major revision is required.

[1] [https://medium.com/@andrewromanenco/decoding-qr-codes-using-neural-networks-272f9e8ba635](https://medium.com/@andrewromanenco/decoding-qr-codes-using-neural-networks-272f9e8ba635)

[2] [https://github.com/Brainydaps/AI-QR-Code-Decoder](https://github.com/Brainydaps/AI-QR-Code-Decoder)

[3] Yuan, Jing, and C. S. Chen. "Neural net decoders for some block codes." IEE Proceedings I (Communications, Speech and Vision) 137.5 (1990): 309-314.

[4] Yuan, Jing, V. K. Bhargava, and Q. Wang. "An error correcting neural network." Conference Proceeding IEEE Pacific Rim Conference on Communications, Computers and Signal Processing. IEEE, 1989.

[5] Beery, Yair, David Burshtein, and Eliya Nachmani. "Deep learning decoding of error correcting codes." U.S. Patent Application No. 15/996,542.

[6] Choukroun, Yoni, and Lior Wolf. "Error correction code transformer." Advances in Neural Information Processing Systems 35 (2022): 38695-38705.

### Questions
Can you give a mathematical definition of "medium-sensitivity function"?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a rigorous empirical and theoretical study of learning medium-sensitivity functions through Transformer-based QR code decoding. It formalizes QR decoding as an intermediate sensitivity task—sensitive to text changes but robust to bit-level noise—and derives a novel analytical model for the Reed–Solomon success rate under corruption. Experiments demonstrate that Transformers not only exceed the theoretical error-correction limits but also generalize across languages and random strings, exploiting statistical regularities of natural words. 
This work bridges sensitivity theory and deep learning, revealing how attention models perform hybrid symbolic–robust computations.

### Strengths
- Introduces a novel conceptual framework of medium-sensitivity learning, formalizing an intermediate regime between robustness and semantic invariance in neural models.

- Provides a theoretically grounded link between Transformer behavior and Reed–Solomon coding limits, extending information-theoretic analysis to neural error correction.

- Presents quantitative evidence that Transformers can exceed classical decoding thresholds, revealing emergent redundancy exploitation beyond explicit code design.

- Demonstrates broad empirical validity across synthetic and linguistic datasets, supporting both the theoretical model and its generalization to real-world settings.

### Weaknesses
- The theory assumes idealized noise and independence, which may not reflect real Transformer behavior.

- Experiments are limited in scale and domain, so generalization beyond controlled settings is unclear.

- Computational cost and scalability are not discussed, leaving practical applicability uncertain.

### Questions
1. How does your medium-sensitivity framework connect to known notions of robustness or Lipschitz continuity, and can it be quantified by model capacity or gradients?

2. How does the mathematical formulation of medium-sensitivity generalize beyond the discrete symbol space—can the same definitions and bounds be extended to continuous vector spaces where distance and corruption are not count-based?

3. Can your Reed–Solomon analysis handle non-linear or data-dependent noise, and how would that change the theoretical limits?

4. What kinds of failure cases did you observe, and what do they reveal about the limits of medium-sensitivity learning?

### Soundness
3

### Presentation
4

### Contribution
3
