# Bit Cipher — A Simple yet Powerful Word Representation System that Integrates Efficiently with Language-Models

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
While Large Language Models (LLMs) become ever more dominant, classic pre-trained word embeddings sustain their relevance through computational efficiency and nuanced linguistic interpretation. Drawing from recent studies demonstrating that the convergence of GloVe and word2vec optimizations _all_ tend towards log-co-occurrence matrix variants, we construct a novel word representation system called  _**Bit-cipher**_ that eliminates the need of backpropagation while leveraging contextual information and hyper-efficient dimensionality reduction techniques based on unigram frequency, providing strong interpretability, alongside efficiency. 
We use the bit-cipher algorithm to train word vectors via a two-step process that critically relies on a hyperparameter---_bits_---that controls the vector dimension. While the first step trains the bit-cipher, the second utilizes it under two different aggregation modes---_summation_ or _concatenation_---to produce contextually rich representations from word co-occurrences. 
We extend our investigation into bit-cipher's efficacy, performing probing experiments on part-of-speech (POS) tagging and named entity recognition (NER) to assess its competitiveness with classic embeddings like word2vec and GloVe. Additionally, we explore its applicability in LM training and fine-tuning. By replacing embedding layers with cipher embeddings, our experiments illustrate the notable efficiency of cipher in accelerating the training process and attaining better optima compared to conventional training paradigms. In fine-tuning experiments, training cipher embeddings on target datasets and replacing the embedding layer of the LMs to be fine-tuned negates the need for extensive model adjustments, offering a highly efficient transfer learning alternative. Experiments on the integration of bit-cipher embedding layers with Roberta, T5, and OPT, prior to or as a substitute for fine-tuning, showcase a promising enhancement to transfer learning, allowing rapid model convergence while preserving competitive performance.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new method Bit-cipher to learn word representations efficiently while leveraging contextual information. It evaluation the efficacy of Bit-cipher on POS and NER tasks. The results of Bit-cipher is competitive to the two classic word embedding methods: word2vec and Glove. The paper also demonstrates the efficiency of Bit-cipher for LM training and fine-tuning with several   experiments intergrating LMs and Bit-cipher.

### Strengths
1. The idea of using bit-cipher to represent words is interesting.
2. Integration of word embeddings and LM is an important perspective to evaluate the proposed method.

### Weaknesses
1. The motivation of this paper is not strong enough. It lacks an explanation about the reason why bit-cipher achieves such performance. The advantanges of bit-cipher compared to the classic word embeddings are not clear. Only experimental results are provided. More theoretical proofs and straightfoward intuition should be provided.
2. The definition of bit-cipher is not clear. Section 3.1 is hard to follow. $\mathcal V_1^b$ is not introduced before using.
3. Probing experiments are conducted on only two downstream tasks (i.e., NER and POS). The results are only from one dataset for each task. The experiments are not convincing.
4. Again, to demonstrate the effeciency of Bit-cipher for LM integration, the experiments are not enough. For instance, it reach the claim "This approach enables models to converge more rapidly compared to traditional methods" without any comparison to classic word embeddings.

### Questions
1. What is the basic assumption of bit-cipher for learning word representations? What are the advantages of bit-cipher over classic methods?
2. Are there any theoretical proofs to support the advantages of the proposed method?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes Bit-cipher, a word-embedding technique that eliminates the need of backpropagation while leveraging
contextual information and hyper-efficient dimensionality reduction techniques
based on unigram frequency, providing strong interpretability, alongside efficiency.
Experiments illustrate the notable efficiency of cipher in accelerating
the training process and attaining better optima compared to conventional training
paradigms.

### Strengths
* The proposed method is novel.
* This paper provides counter-intuitive results that a simple embedding algorithm could yield competitive performance.

### Weaknesses
* This paper is quite hard to follow, especially section 3.1 and 3.2. Many notations in those chapters are used without any clarification. I cannot figure out the method until I read section 3.4, which provides a concrete example of building cipher embeddings. Unless authors make great improvements on presentation in the next version, I lean to reject this work.
* In LLM finetuning experiments, I did not see comparation between cipher embeddings and Word2Vec or Glove.

### Questions
See Weakness

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a new word embedding approach called "bit-cipher". The authors test the approach in NER and POS, and use it for language models. The topic seems old in the large language model era, but this is fine. My main concern is that how the approach contributes to the community given that we already have Word2vec, Glove and many other variants; as a reviewer, I did not see a clear advantage of "bit-cipher" over existing methods.

### Strengths
- The topic is fundamental in NLP
-  The authors tried to use the proposed method in some modern language models like OPT and T5

### Weaknesses
- **sum** and **cat** seem vert common for NLP.
- The evaluated tasks like POS and NER, it may not aligned with the performance of downstream tasks.
-  The performance comparison should be compared with more careful way, such as aligning it with pre-trained corpora and parameter scales to have a apple-to-apple comparison.

### Questions
- What is the advantage of "bit-cipher" over Word2vec, Glove and many other variants?  what is the siginificance of "bit-cipher"? If it is insignificant,  people in the LLM era might not learn anything from it.  Please clarify the siginificance.
- For benchmarking, is it possible to consider some downstream tasks which are sensitive to word embeddings? The paper https://arxiv.org/pdf/1507.05523.pdf might help.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
