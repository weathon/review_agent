# FoNE: Precise Single-Token Number Embeddings via Fourier Features

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Language models treat numbers in the same way as ordinary word tokens, which introduces two major issues: (1) embeddings of numerical tokens primarily reflect their frequency in text corpora rather than their inherent numerical properties, leading to frequency bias, and (2) numbers are often split into multiple tokens, forcing the model to aggregate these pieces to recover their values.  Inspired by the observation that pre-trained Large Language Models (LLMs) internally learn Fourier-like features for number tokens, we propose **Fo**urier **N**umber **E**mbedding **(FoNE)**, a novel method that directly maps numbers into the embedding space with their Fourier features. FoNE encodes each number as a single token with only two embedding dimensions per digit, effectively capturing numerical values without fragmentation. 
Compared to traditional subword and digit-wise embeddings, FoNE achieves higher accuracy on arithmetic tasks, requires significantly less training data, and offers more efficient training and inference. 
A $38$M-parameter Transformer trained from scratch with FoNE outperforms a fine-tuned Llama-3.2-1B model on addition, subtraction, and multiplication. FoNE is also the only method that achieves $100\\%$ accuracy on over 100,000 test examples across these tasks. On 6-digit decimal addition, FoNE needs 64$\times$ less data than subword and digit-wise embeddings to reach $\ge 99\\%$ accuracy, while using 3$\times$ and 6$\times$ fewer tokens per number, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a new number encoding for LLMs based on Fourier features. The embedding is the sum of a number token and the separately constructed Fourier number embedding (FoNE). Experiments on synthetic datasets with addition and multiplication tasks show the superiority of the method over other number representations such as a simple single-digit-encoding or the xval approach.

### Strengths
1. The approach is novel and makes sense. Using Fourier features is intuitively more suitable for numerical reasoning than standard number tokenization, and the encoding-decoding approach is elegant.
2. The paper is well written and the methodology is easy to follow.

### Weaknesses
1. The experiment is not sufficient to show the superiority of the approach. Why is the method not tested on standard benchmarks for mathematical reasoning, such as GSM8K or MATH? The synthetic datasets is very artificial since it only contains equations of fixed-digit numbers, no other text. However, number representations in LLMs are only relevant for data that consists of both text and numerical tasks. 
2. The method seems incompatible with existing models since it requires specific modifications to the tokenizer and an edit after the embedding, so it is not applicable for finetuning, only for training models from scratch. This makes it difficult to evaluate the effectiveness, and limits the applicability of the method.
3. The method is only compared to xval as a baseline approach, although there have been other papers on this topic, such as the Number Token Loss (NTL) [1]. In the discussion section, it is argued that NTL is not relevant since "regression produces continuous value", but NTL is only a loss function that does not impose any constraints on what the number looks like. It should be comparable. In general, the paper only reports accuracy as a metric, although the numeric value clearly matters and regression metrics such as RMSE would be suitable. 

[1] Zausinger, Jonas, et al. "Regress, Don't Guess--A Regression-like Loss on Number Tokens for Language Models." arXiv preprint arXiv:2411.02083 (2024).

### Questions
1. The main question is how the method compares to a simple transformer model on a standard numerical reasoning dataset.
2. m and n are the number of integer and decimal digits, but m and n seem to be set for each number, and not by the maximum m and n of the dataset. Doesn’t this lead to issues because the resulting FoNE vectors are not aligned, I.e. the third cell for 1000 would represent the tens digit but the third cell for 0.12 would be one  hundredths? 
3. In definition 3.6, why is it \phi(0, 10) to \phi(9,10)? I expected it to be dependent on i and not fixed to 10.
4. How efficient is the end to end training? The manuscript claims that this method is even more efficient than the baselines since fewer tokens are used, but I imagine that computing the FoNE representation and adding it to the embedding vector creates an overhead. How long does training one epoch take in comparison to the baselines?

### Soundness
2

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
4

### Summary
The paper introduces FoNE (Fourier Number Embedding), a novel method for representing numerical values in language models. FoNE maps each number directly into an embedding using sine and cosine functions with multiple periods (powers of 10). This creates a single-token representation that encodes both the magnitude and structure of numbers.

Empirical evaluations on synthetic arithmetic tasks (addition, subtraction, multiplication) show that a small Transformer (38M parameters) trained with FoNE achieves perfect or near-perfect accuracy while being more data- and parameter-efficient than subword, digit-wise, or XVAL embeddings. The paper also discusses integration with existing models, such as continual pretraining on LLMs, and provides theoretical justifications for why FoNE preserves numeracy through modular encoding.

### Strengths
[S1] Clear motivation and valuable conceptual connection. I think it's a valuable connection between the previous finding that LLMs internally learn Fourier-like features and the clear need for accurate number embeddings (fragmented multi-token representations and frequency-based biases).

[S2] The empricial results are strong. Across multiple arithmetic tasks, FoNE consistently outperforms established baselines such as digit-wise, subword, and XVAL embeddings. The method achieves up to 64x higher data efficiency and strong parameter efficiency, reaching perfect or near-perfect accuracy even with a 38 M-parameter model, surpassing fine-tuned large models like Llama-3.2-1B on the same tasks. The analysis also highlights faster training and inference, fewer tokens per number, and robustness across different architectures, which together underscore the practicality and scalability of the approach.

[S3] Theoretical grounding is solid. The paper provides clear and mathematically sound explanations for why FoNE preserves numeric precision (via modular recovery and periodic embeddings).

### Weaknesses
[W1] Benchmarks are still synthetic. All experiments are conducted on synthetic arithmetic datasets, which may not fully reflect the complexities of real-world numeric reasoning (e.g., financial QA, table reasoning), where additional challenges might present. E.g., numbers can take different forms (e.g., "eighteen", "XVIII", $18), or with units associated (18g vs 0.018kg), and contextual grounding might be essential. 

[W2] Model scale. The core experiments use very small Transformer model (38M) where text capabilities are not tested. It remains unclear whether FoNE can coordiate well with LLMs' semantic embedding space of pretrained text tokens without introducing instability. For example, there might be a misalignment between the embedidng spaces of texts and numbers; Another concern is that FoNE removed number semantics entirely: information such as "integers around 2000 is more likely to be a year number". The paper briefly argues MLP layers preserve semantics but it remains to be empirically validated. A demonstration of how FoNE interacts with pretrained embeddings or retains contextual numeric meaning would strengthen the paper’s impact.

### Questions
Interestingly, if one compare FoNE with the classic sinusoidal positional encoding (from Vaswani et al., 2017, Attention Is All You Need), they share very similar expressions (compared to Definition 3.2 in this paper, when switching $pos$ with the encoded number):

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

In its essence, every PE is a way to encode a number (position index). Would it be possible to modify other forms of PE into number embedding methods?

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
3

### Summary
This paper proposes a special method called Fourier number embedding to tokenize numbers in natural language, which is defined as a list of sinusoids. A special token [NUM] is used to handle its integration with other word embeddings in both the encoding and decoding stages. Experiments show the proposed methods are effective in doing arithmetic tasks.

### Strengths
1. The paper identifies a well-known limitation of LLMs, their weak representations and reasoning over numbers.
2. The proposed approach is clearly explained and easy to follow, and the experiments demonstrate meaningful gains.

### Weaknesses
1. There are probably not sufficient discussions and experiments on the performance of tasks involving floating point and other numbers that do not necessarily represent a value (e.g., year, flight number, music pitches, etc.).
2. The contribution of the paper is quite limited on the arithmetic task, which can be done in fact by a calculator, not an LLM. A further exploration can be related to how improved number representation enhances natural language inference, in some tasks related partially to numbers.

### Questions
Q1&2: see weaknesses.
Q3: In the evaluation (Figure 3), is there a strict control that samples in the test set are not in the training set?
Q4: How would the author explain the result in Table 8 (authors name it Table H in the appendices), where the arithmetic ability, though higher than that of the pre-trained model, is still lower than the ones in Figure 3? How to falsify the assumption that large models that understand languages to some extent (like GPT/LLama) cannot understand numbers well, regardless of the tokenization method?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new embedding scheme for numerical tokens, based on past empirical work showing that number embeddings often have fourier-like features. By directly encoding numbers in a pre-determined fourier-like scheme, they are able to dramatically improve an LLM's ability to do mathematical tasks.

### Strengths
The empirical results are super strong! They had the baselines I was looking for too (e.g., encoding each number as its own token).

### Weaknesses
I am curious about how this encoding scheme would impact training perplexity and stability on normal LLM training settings, where we have number prediction mixed in with regular language modeling loss. The paper talks about how this could work easily by using a special number token. I think for this work to really get adopted by a next-gen frontier or open-source model, it would require someone exploring sweeps of this at least at the small or medium pre-training scale.

I’m also interested how the authors think we should be handling cases of how messy tokenization can get in practice, e.g., real numbers look like everything from $1.000 to $1,000 to one thousand and 99/100 to 5263–36181 etc.

### Questions
See weaknesses!

### Soundness
3

### Presentation
3

### Contribution
3
