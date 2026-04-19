# CipherPrune:  Efficient and Scalable Private Transformer Inference

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 8

## Abstract
Private Transformer inference using cryptographic protocols offers promising solutions for privacy-preserving machine learning; however, it still faces significant runtime overhead (efficiency issues) and challenges in handling long-token inputs (scalability issues). We observe that the Transformer's operational complexity scales quadratically with the number of input tokens, making it essential to reduce the input token length. Notably, each token varies in importance, and many inputs contain redundant tokens. Additionally, prior private inference methods that rely on high-degree polynomial approximations for non-linear activations are computationally expensive. Therefore, reducing the polynomial degree for less important tokens can significantly accelerate private inference.  Building on these observations, we propose \textit{CipherPrune}, an efficient and scalable private inference framework that includes a secure encrypted token pruning protocol, a polynomial reduction protocol, and corresponding Transformer network optimizations. At the protocol level, encrypted token pruning adaptively removes unimportant tokens from encrypted inputs in a progressive, layer-wise manner. Additionally, encrypted polynomial reduction assigns lower-degree polynomials to less important tokens after pruning, enhancing efficiency without decryption. At the network level, we introduce protocol-aware network optimization via a gradient-based search to maximize pruning thresholds and polynomial reduction conditions while maintaining the desired accuracy. Our experiments demonstrate that CipherPrune reduces the execution overhead of private Transformer inference by approximately $6.1\times$ for 128-token inputs and $10.6\times$  for 512-token inputs, compared to previous methods, with only a marginal drop in accuracy. The code is publicly available at https://github.com/UCF-Lou-Lab-PET/cipher-prune-inference.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper introduces **CipherPrune**, a framework designed to enhance the efficiency and scalability of private inference in Transformer models, particularly BERT and GPT2, by implementing secure token pruning and polynomial reduction protocols. It addresses challenges related to runtime overhead and scalability by utilizing cryptographic techniques such as correlated oblivious transfer and secure multi-party computation, achieving significant speed improvements (up to 18.2x faster than existing methods) without compromising accuracy. Additionally, the framework incorporates crypto-aware fine-tuning to optimize pruning and approximation thresholds during training, ensuring efficient inference while maintaining data privacy.

### Strengths
- **Hybrid Cryptographic Approach**: The paper introduces a hybrid framework that combines homomorphic encryption for linear operations and secure multi-party computation (MPC) for non-linear operations, significantly improving the efficiency and scalability of private inference in Transformer models like BERT and GPT2.

- **Secure Token Pruning Protocol**: CipherPrune implements a secure token pruning protocol that evaluates token importance against a learned threshold, allowing for efficient pruning of unimportant tokens while preserving data privacy during inference.

- **Crypto-aware Fine-tuning**: The framework incorporates crypto-aware fine-tuning to optimize pruning and approximation thresholds during training, ensuring that the model maintains high performance and accuracy while facilitating efficient private inference.

### Weaknesses
I have a concern about the communication cost comparison between SoftMax and GELU here. 

From the manuscript, the pruned tokens are selected after attention so that the communication from SoftMax here is still significant because you need to do compare, multiply, and etc.

How does the communication cost from SoftMax compared to one that GELU introduce?

### Questions
see weakness

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
This paper introduces an innovative framework designed to address the efficiency and scalability challenges of private Transformer inference. The proposed CipherPrune framework incorporates secure encrypted token pruning and polynomial reduction protocols to optimize performance. By progressively pruning unimportant tokens and assigning lower-degree polynomial approximations to less important tokens, CipherPrune achieves notable reductions in execution time and communication costs without compromising accuracy.

### Strengths
1. The combination of encrypted token pruning and polynomial reduction is novel and well-executed, contributing to reduce PI costs and latency while maintaining model performance.

2. The proposed CipherPrune demonstrates substantial improvements in runtime and communication efficiency.

### Weaknesses
As token pruning may be less effective in tasks that rely heavily on retaining comprehensive token-level information, such as machine translation, I suggest that the authors explicitly outline which applications are most suitable for this method and where its benefits may be more limited. This clarification would enhance the practical understanding and applicability of the work.

### Questions
Can you specify which types of NLP tasks or applications are most suitable for the CipherPrune framework, and which tasks may see limited benefits from the proposed token pruning approach?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces *CipherPrune*, a framework designed to mitigate the communication and latency overheads associated with both nonlinear and linear operations (FLOPs) in private inference for transformer-based models. CipherPrune introduces a token pruning method that assesses input token redundancy, progressively removing unimportant tokens layer by layer across the network, thereby leveraging task-level token redundancy to achieve efficiency without sacrificing predictive performance.  

Additionally, for less important tokens CipherPrune employs low-degree polynomial approximation for GELU and Softmax, in contrast with the higher degree polynomial approximation for critical input tokens, further enhancing efficiency without compromising security guarantees or predictive performance.

### Strengths
$\bullet$ This work contributed to both the front-- protocol and network level-- and demonstrated the potential of a synergistic approach that achieves significant speedups and substantial communication reductions. 

$\bullet$ The authors identify limitations in plaintext-based token pruning for private inference and introduce ciphertext-specific pruning techniques that avoid the need for decryption at each network layer, addressing a key bottleneck in efficiency. 

$\bullet$ They effectively tackle the shortcomings of task-agnostic token pruning (such as word elimination) seen in prior work, demonstrating the advantages of a task-specific, layer-wise progressive token pruning strategy that optimizes both performance and efficiency. 

$\bullet$ The paper is well-written and clearly outlines the challenges, the limitations of previous approaches, and the authors' contributions, making it easy to follow and appreciate the impact of this work.

### Weaknesses
$\bullet$ Step 3 (Figure 4), prune and reduce, would increase the computational burden on the client side. A discussion on the resource burden (computation and memory) on the client should have been included, assuming the client has a low-end mobile device with limited compute and memory. 

$\bullet$ This work does not address the overhead introduced by LayerNorm, which can add substantial communication and latency costs depending on the protocols used. For instance, in CipherGPT [3], LayerNorm accounts for **22%** of the total latency and communication overhead. 

$\bullet$ One significant drawback of this paper is the absence of a detailed characterization of layerwise token redundancy. Without this, readers miss out on key insights into how redundancy varies across layers, which could have helped guide more efficient network design and optimization.

$\bullet$ Overall, the CipherPrune framework appears quite complex to understand, implement, and deploy. Its successful application requires expertise in both protocol design and network architecture, making it challenging for those without specialized knowledge in these areas. 



## Correction in the draft

$\bullet$ L#355 Pruning ratio larger than zero --> Pruning ratio greater than zero

$\bullet$ Figure 4 depicts the 2PC threat model for Post-LN configuration, and in the FFN block diagram, one linear layer (after the GELU layer) is missing. 

$\bullet$ Missing citations [1] and [2]

1.  Zimerman et al., Converting Transformers to Polynomial Form for Secure Inference Over Homomorphic Encryption,  ICML 2024

2.  Luo et al. SecFormer: Towards Fast and Accurate Privacy-Preserving Inference for Large Language Models, ACL 2024. 

3. Hou et al., CipherGPT: Secure Two-Party GPT Inference

### Questions
See the weaknesses

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
CipherPrune effectively removes unnecessary tokens by combining input-adaptive pruning based on importance scores with progressive layer-wise pruning. For the remaining tokens, it applies high-order and low-order polynomials according to their importance to maximize computational efficiency. This approach achieved 6.1x and 10.6x faster performance on inputs of 128 and 512 tokens, respectively, significantly improving inference speed without accuracy loss.

### Strengths
- Achieving a substantial improvement in speed, with a 6.1x increase for 128 tokens and a 10.6x increase for 512 tokens, without any reduction in accuracy, is remarkable.

- The application of different polynomial orders based on importance scores, rather than merely pruning, is an impressive idea.

- The paper clearly differentiates itself from existing studies by detailing each step, explaining what previous research overlooked, why these aspects are necessary, and how their work contributes to these areas.

- All experimental settings are well-documented, providing extensive end-to-end results and comparisons across multiple models.

This paper demonstrates advancements over BOLT in every aspect. The time complexity of pruning and the resulting speed have both improved, with no difference in accuracy compared to BOLT. The authors provided a thorough analysis of BOLT, detailing the shortcomings of its ideas and the specific areas they have enhanced. In addition to building on ideas from previous studies, they introduced an innovative approach by applying polynomials of different orders based on importance scores. Comparisons with various papers are well-presented in the appendix, and it’s impressive that this technique can be applied to most inference models, including 2PC and 3PC models. I believe this is an excellent paper.

### Weaknesses
- Although the paper claims no accuracy reduction, there is a minor decrease of around 0.2%.

- In terms of pruning, the only prior work compared is BOLT. (I’m unsure if there are other relevant papers, so it may be possible that only BOLT exists in this context.)

### Questions
See the weakness

### Soundness
4

### Presentation
4

### Contribution
4
