# Coupling RNN with LLM: Does Their Integration Improve Highly Order-Sensitive Language Understanding?

- Decision: Reject
- Scores: 2, 6, 4, 0

## Abstract
Pretrained large language models (LLMs) have demonstrated remarkable success across various language modeling tasks. However, in domain-specific applications, particularly those involving highly order-sensitive data, general LLMs exhibit limitations in achieving state-of-the-art performance. One notable issue is that the contextual embeddings by LLMs still lack a strong positional inductive bias, especially for long and highly ordered sequences, leading to the "lost in the middle" problem. In this work, we utilized the potential of sequential models (RNNs) with LLMs to address the issue and investigate whether RNN integration improves LLM performance. The LLM generates rich contextual embeddings using the attention mechanism of the Transformer. The RNN further processes the LLM embeddings to capture the contextual semantics of long and order-sensitive dependencies. The LLM-RNN model leverages the potential of both Transformer and recurrent structures to enhance performance in domain-specific tasks. We perform a wide range of experiments leveraging multiple types of LLMs (encoder-only, encoder-decoder, and decoder-only) and RNNs (GRU, LSTM, BiGRU, and BiLSTM) across diverse public and real-world datasets to investigate the potential (either positive or negative) of LLM-RNN models. The experimental results highlight the superiority of the LLM-RNN model, showing improvements in commonsense reasoning, code understanding, and biomedical reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work investigates whether integrating RNN layers with LLMs can improve performance on tasks involving high order sensitivity. The authors propose a hybrid LLM-RNN architecture, where pretrained LLMs, including encoder-only, decoder-only, and encoder-decoder variants, generate contextual embeddings that are subsequently refined through RNN layers (LSTM, GRU, BiLSTM, or BiGRU). Evaluation results indicate a performance increase with LLM-RNN models on the classification task.

### Strengths
1. The paper evaluates multiple LLM architectures (encoder-only, decoder-only, encoder-decoder) and RNN variants, providing a comprehensive assessment across different configurations.

2. The hyperparameters and all the experimental settings are provided, facilitating replication and following researches.

3. The paper is clearly written and easy to understand.

### Weaknesses
1. The paper lacks theoretical analysis and primarily presents an empirical study, offering little insight into why the integration improves performance.

2. The observed improvements are largely intuitive and expected, given that RNNs are naturally suited for modeling sequential dependencies.

3. The performance gains reported in Table 1 are limited—mostly under 1%. This raises questions about the practical significance of the proposed approach.

4. The coupling mechanism between the LLM and RNN is relatively shallow, simply passing LLM embeddings to an RNN without exploring deeper or more synergistic integration (e.g., incorporating recurrence within LLM layers).

5. While the paper mentions various tasks such as code understanding, commonsense reasoning, and biomedical text analysis, the evaluation is limited to classification. Since LLM embeddings are widely used for generative and reasoning tasks, it remains unclear whether the RNN outputs preserve the generative or semantic richness of the original LLM representations. -- If LLM-RNN does not match or exceed pure LLM performance on other core tasks such as reasoning or generation, then their practical usefulness would be quite limited.

### Questions
1. Beyond classification, have you tested the LLM-RNN model on other standard LLM tasks such as question answering, structured generation, or sequence completion?

2. Have you conducted any theoretical analysis to compare the pure LLM and the LLM-RNN models, for example examining how differences in message passing or internal representation dynamics might account for the observed results?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a novel approach that integrates RNNs with LLMs to enhance the performance of order-sensitive tasks. The authors propose coupling LLMs with RNNs, leveraging the sequential processing power of RNNs to complement the contextual embeddings produced by LLMs. Experiments show that the LLM-RNN model outperforms LLMs in order-sensitive tasks.

### Strengths
1. The paper proposes an interesting idea about the improvement of LLM and the idea is simple but useful. 

2. The paper conducts a thorough evaluation using various LLMs and RNN across multiple tasks.

3. This paper is good writing and easy to follow.

### Weaknesses
1. The Transformer incorporating RNN can solve order-sensitive tasks well with the advantage of RNN while the performance can also be affected by the disadvantage of RNN. For example, LSTM and other RNN models always forget some information with the encoding while the long context in LLM can make it worse.

2. Although the paper tests multiple RNN models, it would be helpful to explore how other advanced RNN could further enhance performance.

3. The authors should also give some ablations about the comparision between larger LLM and LLM with RNN.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies a simple hybrid: take hidden states from a pretrained LLM, project them, and feed them to different RNN models, followed by a linear head. The authors evaluate on datasets from different domains across tasks (sentiment, code understanding, and biomedical). Experiments indicate that the integration of LLM and RNN can achieve improvement versus the base pre-trained LLMs.

### Strengths
S1. This paper is well-organized and easy to follow.

S2. This paper spans three task families and multiple LLM families (e.g., BERT, GPT, DeepSeek) with four different RNN types, which help probe when RNN post-processing helps.

### Weaknesses
W1. The paper is basically a simple hybrid that feeds a pretrained LLM’s hidden states into an RNN for classification across several domains. The experiments show small gains, but there is little deeper theoretical insight so the work reads as a simple incremental study. 

W2. The claims are overstated. The text says “in all cases RNNs consistently enhanced performance,” yet the tables show counterexamples, for example GPT-2 on Sentiment140 drops from 81.18 to 80.97, and DeepSeek-Coder on CodeXGLUE defect detection also declines. The “consistent” claim does not hold.


W3. The motivation around order sensitivity and “lost in the middle” is weakly supported. Most benchmarks are short-context sentiment or code classification. There is no sequence-length stress test or long-context suite, and no controlled length-sweep analysis.


W4.​​ The study compares LLM vs. LLM+RNN but not against other lightweight sequence shapers on top of LLM states, for example a small TCN or an extra self-attention block with relative position bias. There is also no ablation on where to attach the RNN, final layer vs. intermediate layers, or whether freezing the LLM changes the conclusion.

### Questions
Please see Weaknesses.

Additional Questions:


Q1. Under a matched hyperparameter search for base LLMs and LLM+RNN variants, do the hybrids still outperform, and how does this change on long-context (length-binned) evaluations?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes a hybrid architecture where pre-trained language models (BERT, RoBERTa, GPT-2, CodeT5, DeepSeek) generate contextual embeddings that are then processed by RNN layers (GRU, LSTM, BiGRU, BiLSTM) before classification. The authors claim this helps capture sequential, order-sensitive dependencies that transformers miss.

**Methodology:**
- Freeze pre-trained model weights
- Extract token embeddings from the LLM
- Feed embeddings through RNN → FC classifier
- Test on three domains:
  - Sentiment analysis (IMDb, Twitter Airlines, Sentiment140)
  - Code Defect detection (CodeXGLUE + 3 custom datasets)
  - Biomedical reasoning Named entity recognition (NCBI disease dataset)

The main claim is RNNs refine LLM embeddings to better capture sequential order, improving performance on order-sensitive tasks.

### Strengths
Tests multiple model architectures (5 base models × 4 RNN types) across 7 datasets

### Weaknesses
### 1. Poor Writing Quality - Fails Basic Scientific Communication Standards

The paper's writing is incoherent and makes it hard to understand what the authors want to say. **This is a basic requirement for a scientific paper.**
**Concrete examples:**
- **2nd paragraph**: Rambles about fine-tuning, LoRA, knowledge graphs with no clear connection to the paper's thesis
- **3rd paragraph, Line 75-76**: Cites papers that finetune BERT/RoBERTa for sentiment classfication tasks to support LLM has limitation in sentiment analysis and code understanding. Why these two citations related to code understanding? They have no relation to code. Also, BERT/RoBERTa are not LLM at all. It makes no sense to claim LLM have trouble in sentiment analysis and code understanding, when previous show BERT/RoBERTa have trouble.
- **Figure 1**: Shows "embedding shift" and "saliency" metrics but never defines them - no explanation of how they're computed or what they represent
- The introduction reads like "random talking instead of academic paper" with no logical flow or clear problem motivation
- The introduction also mentioned "lost in the middle problem" while the full paper didn't touch that problem at all. Also, that work is about 100K+ context retrieval, not relevant to their short sequences. 
- Conflates "order sensitivity" with general sequential modeling without clear definition

### 2. Obsolete Approach with Questionable Motivation and Contradictory Results

**The investigation was already done 2018-2020:** Adding RNN layers on top of transformer embeddings is not novel. See ["Simple BERT Models for Relation Extraction and Semantic Role Labeling"](https://arxiv.org/abs/1904.05255) and related work from that era. 

**Modern LLMs don't work this way:** Current best practices use generation-based classification (prompting) rather than fine-tuning classification heads. The paper provides **no evidence** that their approach is better than modern methods, nor any comparison.

**Their own results contradict their claims:** In Table 3 , **DeepSeek-coder by itself without any RNN head works best**. This directly undermines their central thesis that RNNs improve order-sensitive understanding.

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
1
