# Lossless Vocabulary Reduction for Auto-Regressive Language Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Tokenization---the process of decomposing a given text into a sequence of subwords called *tokens*---is one of the key components in the development of language models. Particularly, auto-regressive language models generate texts token by token, i.e., by predicting the next-token distribution given the previous ones, and thus tokenization directly affects their efficiency in text generation. Since each language model has their own vocabulary as a set of possible tokens, they struggle to cooperate with each other at the level of next-token distributions such as model ensemble. In this paper, we establish a theoretical framework of lossless vocabulary reduction, which efficiently converts a given auto-regressive language model into the one with an arbitrarily small vocabulary without any loss in accuracy. This framework allows language models with different tokenization to cooperate with each other efficiently by reduction to their maximal common vocabulary. Specifically, we empirically demonstrate its applicability to model ensemble with different tokenization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a general framework for lossless vocabulary reduction (LVR) that converts any next‑token distribution over a vocabulary into an equivalent distribution over an arbitrary sub‑vocabulary. The key construct is nested tokenization defined by first tokenizing into $V$ and then retokenizing each token’s byte string into $V_{\text{sub}}$. The induced distribution $p_{V\to V_{\text{sub}}}$ is shown to be lossless at the text level. The authors propose lossless vocabulary reduction (LVR), by giving a computable recursion for the marginal probabilities via relative covering sets, and then presenting an efficient approximation reuses cached covers and probabilities and restricting the inner loop to top‑K tokens. On top of the analysis, the authors introduce ensemble via maximal common vocabulary (MCV), formed by intersecting vocabularies and restricting BPE merges accordingly. Experiments show that LVR preserves accuracy on GSM8K compared to the original model and substantially outperforms baselines. The method also helps to ensemble two models with different vocabulary.

### Strengths
1. **Rigorous theoretical analysis.** The paper precisely defines and states tokenization with an appropriate symbol framework. On top of the framework, the paper clearly proposes the LVR with rich theoretical demonstrations and proof.
2. **Clear demonstration.**  The binary toy example helps to understand “lossless text distribution but different greedy path” phenomenon.
3. **Experimental results.** On GSM8K, LVR closely matches or slightly exceeds the original model across different vocabulary settings, whereas Naive Restriction fails badly, demonstrating the effectiveness of the method.
4. **Successfully applied to model ensembling.** The proposed method successfully ensembled different language models with different vocabularies, showing application values for other research fields.

### Weaknesses
1. **Overclaim on "lossless".** Theorems guarantee exact text‑level equivalence, while Algorithm 2 introduces two approximations such as the top‑K truncation. There is no error bound tying those approximations to the final possible "loss" incurred. The result is that “lossless” in practice becomes “almost lossless,” but the gap is unquantified. 
2. **Limited model pairs compared.** Only one pair of model ensemble is examined in the paper. Results on more models pairs, tasks, or even working languages are recommended. 
3. **No perplexity or likelihood evaluation.** Since the claim is distributional equivalence, perplexity/negative log‑likelihood curves before/after reduction would be the most direct metric. The current focus on task accuracy is informative but indirect.

### Questions
Please refer to weaknesses

### Soundness
3

### Presentation
3

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
This paper presents a framework for lossless vocabulary reduction with detailed theoretical guarantees, showing that for an auto-regressive language model, the reduction does not lead to any performance loss, and it is more efficient compared to byte-level reduction methods.

### Strengths
The paper is theoretically grounded, with rigorous and well-structured proofs that justify the correctness of the proposed approach. Moreover, the method is motivated by a clear intuition and contributes a novel perspective to the problem of vocabulary reduction in language models.

### Weaknesses
1.The "An illustrative example" in Section 3.2, the example might not be presented in the most effective format. The use of a visual illustration or an alternative narrative style could potentially improve clarity and readability.

2.The authors only provide experimental results on vocabulary reduction over GSM8K and a single ensemble setup. This might be somewhat limited in terms of empirical evidence, and the experimental section could benefit from broader evaluations to more convincingly support the method’s effectiveness.

### Questions
1.In the experimental section, the authors present ensemble results using Qwen2.5-3B and Falcon-7B. Given this, could the authors also provide the individual vocabulary compression rates for Qwen2.5-3B and Falcon-7B, as well as a comparison of compression rates across different ensemble strategies?

2.Different models have different tokenizers, which affects not only ensemble learning but also knowledge distillation, as mentioned in the Introduction. Could the proposed method be applied to knowledge distillation as well? If so, could the authors provide a possible application idea?

3.A minor issue: the first half of the abstract overlaps too much with the first paragraph of the introduction.

### Soundness
3

### Presentation
2

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
This paper introduces Lossless Vocabulary Reduction (LVR), a method designed to reduce the size of subword vocabularies while maintaining full reversibility to the original tokenization. The key idea is to perform deterministic merging and re-encoding of redundant subwords, enabling significant vocabulary compression without retraining or modifying existing tokenizers.

### Strengths
1. The paper tackles a well-known bottleneck in NLP efficiency, oversized vocabularies that inflate embedding and softmax layers. A lossless solution that requires no retraining is both innovative and practically relevant.
2. The authors provide rigorous proofs ensuring bijective mappings between reduced and original vocabularies. This formal treatment gives confidence that the method is safe for use in production systems where reversibility is critical.
3. The method can be applied to any existing subword vocabulary without altering model architecture or retraining steps. This post hoc compatibility makes it very attractive for deployment in pretrained LMs.

### Weaknesses
1. The paper does not compare LVR with recently proposed learned vocabulary reduction or adaptive tokenization approaches, which may achieve similar goals via training-based optimization.
2. Experiments focus mainly on LM perplexity and simple classification benchmarks. More diverse downstream tasks, especially those sensitive to segmentation (e.g., NMT, summarization)—would make the validation more comprehensive.

### Questions
1. How does LVR affect inference efficiency (e.g., wall-clock time, FLOPs, or memory usage) given the longer token sequences?
2. Is there a practical limit to vocabulary reduction beyond which sequence length growth harms model throughput or accuracy?
3. How are special tokens ([CLS], [SEP], , etc.) handled — are they exempt from reduction or transformed within the same framework?
4. For large pretrained models (e.g., GPT, T5), how easily can LVR be retrofit into production pipelines, and are there compatibility concerns with cached tokenizers?

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
This paper introduces Lossless Vocabulary Reduction (LVR), a method to convert an autoregressive language model's next-token distribution into an equivalent distribution over an arbitrary sub-vocabulary $\mathcal{V}_{sub}$. The authors prove that this reduction is lossless and propose the Maximal Common Vocabulary (MCV) method for model ensembling, which is both computationally efficient and empirically effective.

### Strengths
S1: This paper presents a theoretical framework to losslessly reduce a language model to an arbitrary sub-vocabulary. This is a strong, novel, and highly significant theoretical contribution that generalizes prior work, which was limited to byte-level reduction.


S2: The proposed LVR framework, especially the MCV ensemble application, provides a principled and effective solution, achieving the theoretical correctness of byte-level reduction while offering superior generation efficiency.

S3: The authors conduct well-designed and clearly presented experiments, which directly support the paper's core claims.

### Weaknesses
W1:  The Algorithm 2 still requires iterating over the relative cover set $C_{\mathcal{V},\mathcal{V}_{sub}}$  and the top-K tokens of the original vocabulary $\mathcal{V}^{(K)}$. The authors should provide a more detailed complexity analysis (both Algorithm 1 and Algorithm 2), for example, discussing the size of the relative cover set and the overall computation complexity.


W2: The Algorithm 2 introduces a top-K approximation, which technically breaks the "lossless" guarantee of the core theory. While the Table 1 show the performance is "almost lossless", the sensitivity to the hyperparameter $K$ (set to 300 in experiments) should be discussed.


W3: Since the proposed Maximal Common Vocabulary (MCV) method can be regarded as a generalization of byte-level reduction, and its performance is largely determined by the size of the common vocabulary, the authors should provide a simple analysis the common-vocabulary sizes of tokenizers used in widely adopted LLMs. Such an analysis would provide an intuitive quantification of the extent to which MCV improves over byte-level reduction.


W4: Because the current MCV ensemble approach appears to be strongly dependent on the model family (i.e. the tokenizer), the authors should evaluate a more diverse set of model families and additional ensemble configurations to further validate the advantages of their method. Expanding the range of models and ensemble combinations is likely to produce more pronounced and convincing results.

### Questions
Q1:  Could you provide a more detailed analysis of the computational overhead of Algorithm 2? For example, What is the typical size of the relative cover set $C_{\mathcal{V},\mathcal{V}_{sub}}$ during generation? How does it grow with the sequence length $k$?

Q2: In Table 1, the Falcon3-7B model shows significant fluctuations in the reported metrics when the maximum token length is ≤ 2 bytes. Could the authors further analyze the factors driving these metric variations? Specifically, is the observed instability attributable to the approximation algorithm used?

### Soundness
4

### Presentation
3

### Contribution
3
