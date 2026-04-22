# LitePruner: A Lightweight Realtime Token Pruner before Large Language Models

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Tokenization is one of the core steps of the language model pipeline. However, the tokenizer yields more tokens for the same context in non-English languages, especially in low-resource languages due to the shared multilingual settings, which results in unexpected fairness problems in terms of token fees, response latency, and long context processing. In this paper, we study a real-time computing problem, attempting to reduce the total number of tokens per query but maintain decent performance in multilingual settings. We present a simple, training-free, CPU-based pruner model to reuse pre-trained weights from the first attention layer of small models to rank token importance, only delivering important tokens to the target larger models. This method is motivated by the fact that early layers in both small and large models latch onto similar shallow local signals due to similar tokenization algorithms (e.g., BPE) producing identical local signals. Massive in-context learning experiments on MGSM, Global-MMLU-Lite and ARC and RAG-based experiments on PubMedQA and MEMERAG show that our method can preserve decent performance for languages while reducing up to $30\%$ of the total number of tokens in both in-family and across-family model settings, where the pruner model and the target large model are in or not in the same model family. Our method is compatible with commercial LLM APIs and CPU-based, contributing to real-life applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes using the first attention layer of small pre-trained models to rank and prune input tokens before passing them to larger LLMs, motivated by tokenization disparities in multilingual settings. The method is training-free, CPU-based, and evaluated on multilingual ICL and RAG benchmarks with reported token reductions of up to 30%.

### Strengths
1. The paper addresses a real fairness issue where non-English users pay significantly more for LLM services due to tokenization disparities, and provides a practical deployment that is CPU-based, training-free, and compatible with commercial APIs.

2. The paper provides extensive experiments across multiple benchmarks, languages (high/medium/low-resource), and model families.

### Weaknesses
1. The paper claims that "Early layers in both small and large models show similar attention patterns due to similar tokenization", and provided evidence (Tables 6-7) of high cosine similarity between attention distributions.However, correlation does nto equate to causation. If would be great if the authors can show that tokens with low attention in the small model are actually redundant for the large model. High cosine similarity just means the OOD attention patterns correlate, and doesn't validate that these patterns predict what can be pruned.

2. Llama3-70B improves from 4.0% to 31.2%. This isn't noise removal, as the performance even after the improvement is lackluster.

### Questions
1. What happens with causal mask?

2. How do you compare to frequency-based baseline?

3. What explains the MGSM anomalies?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes LitePruner, a training-free, CPU-based token pruning method that reduces input token counts in multilingual settings while preserving downstream performance. It leverages relative attention weights (RAW) from the first attention layer of a small pretrained model to rank token importance and forwards only the top-k% tokens to the target large model. The authors validate effectiveness on multilingual ICL (MGSM, Global-MMLU-Lite, ARC) and RAG (PubMedQA, MEMERAG) benchmarks, demonstrating generalization across both in-family and across-family model settings.

### Strengths
LitePruner is practical. It requires no training, runs on CPU, and can be deployed before commercial APIs like GPT to save token costs. Experiments span multiple languages, model families (Llama3, Gemma2, Aya), and benchmarks (MGSM, ARC, MMLU, PubMedQA). Results show minimal performance drop at top-90% pruning, and even improvements in some low-resource languages, suggesting potential denoising effects. The authors also provide empirical support via RAD and cosine similarity, showing early-layer attention alignment between small and large models.

### Weaknesses
The paper provides insufficient theoretical justification for its core assumption that the first attention layer alone captures sufficient token importance, particularly for complex multilingual reasoning tasks such as MGSM with chain-of-thought prompting. While the authors present empirical correlations using RAD and cosine similarity, they do not explain why early-layer attention should generalize across tasks, languages, or model families. Methodologically, Algorithm 1 omits key implementation details; for example, it does not specify how positional encodings are adjusted after arbitrary token removal, which may affect the target model’s interpretation of sequence order. The method’s reliance on raw attention weights from the first layer also makes it incompatible with efficient operators such as Flash Attention, limiting its practical deployment efficiency despite the claimed CPU compatibility. In the RAG experiments, only documents are pruned while queries remain intact, but the paper offers no rationale for this asymmetric treatment, which could influence retrieval quality.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a training free framework to rank token importance, which prunes the context of the input and sends only a subset of important tokens to the target model using a computationally cheap model. It shows that this method can achieve competitive results for low-resource language tasks and save token budgets due to less efficient tokenization for these texts.

### Strengths
1. The author targets an important question and proposes a conceptually simple yet effective framework for solving the problem.
2. It has been tested on several benchmarks, making it very comprehensive. 
3. It leverages the clear motivation that the small and large models share similar attention patterns, which can be used as a proxy for token importance estimation.

### Weaknesses
1. Efficiently solving low-resource language tasks is a very interesting problem, which should be the focus of the paper as suggested in the abstract and intro. However, very little room has been left for discussing the specific characteristics of these tasks (e.g. tokenization fairness as mentioned by the author). A very straightforward baseline that can be potentially included is to see if a direct translation of the prompt from low to high resource language can improve efficiency and quality. 
2. There are many similar prior works which use the same method and has been well tested on many long context benchmarks (e.g. SpecPrefill https://arxiv.org/abs/2502.02789 being the most similar one). This should be cited and compared accordingly since many aspects of the paper shares the same insights and methodology as SpecPrefill (e.g. use a smaller draft model to prune important tokens, use attention as the surrogate, the handling of position ids, etc). Another similar line of work is called GemFilter, which uses the model’s own shallow layers as proxy. 
3. How does this method work in the multi-turn setting? This should be a potential pitfall for this method. If not, it should be discussed clearly. 
4. On line 258, the author mentions that “top-90% is still a common choice for all scenarios”. The reviewer thinks that keeping 90% would be a relatively high value for this method to break even the cost of pruning itself. This should be discussed more formally. Since the method should either 1) improve accuracy or 2) increase the efficiency.

### Questions
Listed above in weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the problem of tokenization inefficiency and fairness in multilingual large language models, where non-English languages produce disproportionately more tokens, leading to higher costs, slower inference, and shorter usable context. It proposes LitePruner, a training-free, CPU-based token pruning method that reuses the embedding and first attention layer of a small pre-trained model to rank token importance and remove less important tokens before sending input to the target large model.

The authors conduct two main sets of experiments. First, in-context learning (ICL) tests are performed on multilingual benchmarks MGSM, Global-MMLU-Lite, and Multilingual ARC, under both in-family (e.g., Llama3-1B for Llama3-70B, Gemma2-2B for Gemma2-27B) and across-family (e.g., Llama3-1B for GPT-4.1-nano, Gemma2-2B for Aya-expanse-8B) settings, using 3-, 5-, and 8-shot prompting. Second, retrieval-augmented generation (RAG) experiments are conducted on PubMedQA and MEMERAG, where documents are pruned by LitePruner before retrieval and evaluated with metrics such as Mean Reciprocal Rank (MRR), Faithfulness (FA), and Semantic Answer Similarity (SAS).

### Strengths
- The paper’s idea of performing pre-inference token pruning using the first attention layer of a small model is moderately original. Although adopt small model as a proxy of the attention score is not a novel idea, applying this method in permanent token pruning in LLMs is new.
- Experiments are conducted under multiple settings, covering both in-family and across-family settings across multiple model families (Llama 1/8B, Gemma, GPT-4.1-nano) and two task types (in-context learning and retrieval-augmented generation).

### Weaknesses
- The method relies on shared tokenization between the small and large models (e.g., Llama3-1B for Llama3-70B). However, in cross-family settings (e.g., Gemma for GPT-4.1-nano), tokenizers and embedding spaces differ substantially, which may undermine the assumption of attention pattern similarity.
- Some detailed hyperparameter choices (e.g., head averaging details, normalization methods, or handling of positional encodings after pruning) are not specified.
- The evaluation results are not consistently stable across datasets, and the underlying reasons are not sufficiently analyzed. For instance, the performance drops observed on MGSM and Global-MMLU-Lite are notably large and remain unexplained.
- In the in-family and cross-family experiments, different large language models are used as targets, making it difficult to directly assess how the quality of the small model influences the pruning results.
- No explicit ablation study is conducted to test the sensitivity or necessity of specific design choices in LitePruner.

### Questions
Please refer to the weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
2
