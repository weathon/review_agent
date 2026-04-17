# Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
Existing large language model (LLM)-based embeddings typically adopt an encoder-only
paradigm, treating LLMs as static feature extractors and overlooking their core gener-
ative strengths. We introduce GIRCSE (Generative Iterative Refinement for Contrastive
Sentence Embeddings), a novel framework that leverages autoregressive generation to iter-
atively refine semantic representations. By producing sequences of soft tokens optimized
under a contrastive objective, GIRCSE captures latent concepts and implicit semantics
that encoder-only methods often miss. To guide this process, we propose an Iterative
Contrastive Refinement (ICR) objective that encourages each refinement step to yield bet-
ter representations. Extensive experiments show that GIRCSE outperforms strong LLM-
based embedding baselines on the MTEB embedding benchmark. Moreover, GIRCSE ex-
hibits an emergent test-time scaling property: generating more tokens at inference steadily
improves embedding quality. Our results establish generative iterative refinement as a new
paradigm for representation learning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper raises a problem of existing LLM-based embeddings: their overlooking of LLMs' generalization capabilities in embedding learning. To address this issue, this paper proposes a novel framework that employs autoregressive generation to iteratively refine semantic representations, thereby enhancing embedding learning. Extensive experimental results demonstrate the effectiveness of the proposed approach.

### Strengths
- The proposed approach is novel and interesting. It effectively exploits the generative capacity of LLMs for embedding learning.
- The soft token generation mechanism is particularly elegant, enabling the iterative refinement process to remain differentiable while capturing richer semantic information from generated tokens.
- The experimental results demonstrate competitive performance compared to strong baselines across diverse evaluation tasks, validating the effectiveness of the proposed approach.
- The analysis of iteration steps (number of generated tokens) provides valuable insights into the trade-off between computational cost and performance, offering practical guidance.

### Weaknesses
- While the method incurs higher computational costs than existing LLM embeddings, the KV cache can alleviate the efficiency issue. 
- The paper does not provide code
- It would be better to also discuss the limitations
- The ablation study could be improved: While soft token generation is interesting and beneficial for optimization, the paper lacks a comparison between (a) directly encoding the concatenation of input and generated text, and (c) the proposed soft token approach.

### Questions
**Questions:**

- Could prompt engineering help steer LLMs to generate more useful tokens and potentially reduce the required number of iteration steps? It would be valuable to investigate whether task-specific or constrained prompts that guide the topic or style of generated tokens could improve both efficiency and effectiveness.
- How to set the soft token number properly? especially when using it in real-world applications.


**Suggestions:**
- The claim that "LLM-based embeddings typically adopt an encoder-only paradigm" may cause confusion, as many LLM-based embedding methods still employ LLMs in an autoregressive manner, thus maintaining a decoder-only architecture.
- Correct quotation marks to proper LaTeX formatting: "GP" → ``GP'', "TTS" → ``TTS''

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a generative framework that lets LLMs “speak embedding languages” instead of acting as static encoders. The key idea is to iteratively generate soft tokens that refine sentence representations under a new Iterative Contrastive Refinement (ICR) objective.

### Strengths
1. The core idea is creative and well-motivated
2. The paper is clearly written and supported by both quantitative and qualitative analyses.
3. provides a neat way to connect generation and contrastive learning.

### Weaknesses
1. The framework fixes the number of refinement steps (K) during training and inference; an adaptive stopping criterion based on semantic convergence could make the process more efficient and principled.
2. While the paper claims that soft tokens capture latent semantics, there is no direct evidence or visualization showing how these tokens differ structurally from standard hidden states.
3. It is unclear whether gradient flow through the softmax-weighted soft tokens remains stable in multi-step generation. Flat or peaked distributions could cause weak or unstable gradients, potentially limiting refinement quality.

### Questions
1. Given that the number of refinement steps (K) is fixed, have the authors considered adaptive stopping criteria based on semantic saturation?
2. What evidence supports the claim that soft tokens capture latent semantics beyond surface-level similarity? Could the authors visualize or cluster the soft-token space?
3. Is gradient flow stable through the soft token generation? Since each soft token is a weighted sum over all vocabulary embeddings, gradients must pass through the softmax distribution. In multi-step generation, could this cause gradients to become unstable or too weak (for example, when the softmax becomes too flat or too sharp)?

### Soundness
2

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
4

### Summary
This paper presents a novel framework to train that enables generative embedding models. Unlike naive next token prediction which leads to nondifferentiability, it incorporates soft-token generation which can be seen as the weighted average of possible next tokens. Furthermore, these soft-tokens are trained with contrastive objective which is regularized by the their new objective which ensures that newly generated soft-tokens lead to better performance. Overall, the framework adapts existing decoder models to generative text embedding models which eventually could lead to test-time scaling behaviour observed in recent reasoning models.

### Strengths
* The framework is novel, and idea of test-time scaling is relevant and interesting given the recent inference-time scaling efforts.
* Performance is robust across different datasets and models.
* Paper is well-written and easy to follow.

### Weaknesses
* A significant test-time scaling [methodology](https://jina.ai/news/scaling-test-time-compute-for-embedding-models/) for embedding model is not cited.
* Test-time scaling claims are ambitious, gains are marginal or non-existent. Even though methodology is interesting, I dont think generating more tokens leads to similar observations made in o1/s1 style reasoning works. Scores barely change, and decrease in the causal-eos is expected since they are not trained to generate new tokens. Comparison with a model like GritLM would be more accurate.

* 5-20 tokens are really low for test-time scaling, generating hundreds of tokens would be more interesting to draw any conclusions. Moreover, model is limited to 5 token generation during training. This is quite conservative and reduces the scope of paper.

*  Instruction following setup is not ideal. They are often used to measure models' generative performance. Popular benchmarks are MMLU, GSM8K, Alpaca, Human eval etc. Current evaluation setup is not diverse enough to capture instruction following or any other generative task performance. GritLM's setup is much more preferred and should be used in my opinion.









* There is a typo in line 153, it should be decoder instead of encoder.

### Questions
1) What is the performance of standart embeddings after your training methodology?
2) Why do you only train with soft tokens? Can't you use the standart embeddings as an additional objective?
3) You train with query-answer pair, have you tried using wikipedia 1 million datasets from SimCSE to train SimCSE/LLM2Vec style where we rely on augmentations?
4) What is the number of non-soft tokens, because even generating 20 soft tokens is small given the large document or query lengths?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper "Let LLMs Speak Embedding Languages..." aims to address the problem that algorithms generate representations from text using LLMs are usually static, in that they just make one fixed embedding from e.g. the last layer, last token embeddings. To address this they propose a method to iteratively refine the embeddings over several steps. The algorithm auto-regressively generates so-called "soft-tokens" which should give the model a richer ability to refine the representations. This has the drawback of requiring a fine-tuning of the language model but they show that you only need to use a limited amount of data to still achieve decent performance improvements. The primary empirical results show for popular models, (mistral, qwen) that this algorithm can reliably improve the learned representations.

### Strengths
I think the primary strengths of this work are as follows:
1. Originality. The iterative algorithm and soft-token idea is well founded and intuitive, and to the best of my knowledge is novel at least in this use case. 
2. Quality. This paper uses a very reasonable suite of evaluations and tests appropriate models that people are typically using in practice.
3. Clarity. I think the empirical results are presented well.
4. Significance. This algorithm is general, and I think it can be applied to many different architectures. The ability to scale this method in terms of the number of time steps in a test-time compute fashion is quite novel to the field of representation learning, and to my knowledge has not been as successful as this paper demonstrates in the past.

### Weaknesses
I would happily raise my score on the "presentation" category if these are addressed:
1. I thought that the presentation of the GIRCSE algorithm was not clear in the main text in Section 3.2 / 3.3. I did however find that the "Algorithm 1" pseudo-code description made this much more obvious what is going on. I would recommend moving this to the main text if space permits.
2. Figure 1 is not very clear, it looks more like what would be expected for a poster than a figure introducing the algorithm.

### Questions
1. Is the fine-tuning stage of this algorithm more computationally expensive then regular SFT on the same .2B tokens?

### Soundness
4

### Presentation
2

### Contribution
3
