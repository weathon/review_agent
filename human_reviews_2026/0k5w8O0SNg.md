# Cartridges: Lightweight and general-purpose long context representations via self-study

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Large language models are often used to answer queries grounded in large text corpora (e.g. codebases, legal documents, or chat histories) by placing the entire corpus in the context window and leveraging in-context learning (ICL). Although current models support contexts of 100K-10M tokens, this setup is costly to serve because the memory consumption of the KV cache scales with input length. We explore an alternative: training a smaller KV cache offline on each corpus. At inference time, we load this trained KV cache, which we call a Cartridge, and decode a response. Critically, the cost of training a Cartridge can be amortized across all the queries referencing the same corpus. However, we find that the naive approach of training the Cartridge with next-token prediction on the corpus is not competitive with ICL. Instead, we propose self-study, a training recipe in which we generate synthetic conversations about the corpus and train the Cartridge with a context-distillation objective. We find that Cartridges trained with self-study replicate the functionality of ICL, while being significantly cheaper to serve. On challenging long-context benchmarks, Cartridges trained with self-study match ICL performance while using 38.6x less memory and enabling 26.4x higher throughput. Self-study also extends the model's effective context length (e.g. from 128k to 484k tokens on MTOB) and surprisingly, leads to Cartridges that can be composed at inference time without retraining.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Cartridges, a memory-efficient, high-throughput, and general-purpose solution for processing long contexts in LLMs, promising to replicate the functionality of in-context learning (ICL). The authors argue that existing LLMs rely on costly KV caches for long-context processing or ICL. While KV cache compression can reduce memory, its performance collapses at high compression ratios.  

To this end, they propose Cartridges, a lightweight, parametric cache that is trained offline to represent the text corpus. Direct optimization with the next-token prediction objective yields poor generalization, so they introduce self-study, training on synthetic conversational histories with a context-distillation objective.  

On tasks such as NIAH, LongHealth, and MTOB, Cartridges achieves strong accuracy while reducing memory by 38.6× on average and boosting throughput by 26.4×, significantly outperforming state-of-the-art cache-compression baselines.

### Strengths
1. Long context is a critical research area for current LLMs, especially when it comes to both the ability to enhance the LLMs' own capabilities by processing ultra-long context as well as their efficiency.
2. Cartridge, as an architectural innovation, breaks away from the traditional KV-cache optimization paradigm, and its self-study training can also inspire future long-context training research.
3. The proposed method delivers extreme efficiency gains (38.6× memory reduction and 26.4× throughput boost) on very long contexts (up to 484k tokens) while still maintaining superior performance.

### Weaknesses
1. I would appreciate a better organization of the related work, instead of scattering it across Section 2.1, Appendix B, and Appendix E. From my perspective, it is a paper relatively focused on long-context efficiency, so it should be more concentrated on the comparison between cache optimization and linear attention. The key theoretical insights that appear only in Appendix B and E should also be highlighted in the main text. Although linear attention does require pre-training from scratch and non-trivial model conversions, a direct discussion of their trade-offs, whether in approach difference or in experimental performance, is essential for clarifying your methods.
2. The authors should more clearly distinguish Cartridges from prefix-tuning methods, for both learn a parametric cache, so a detailed explanation of their methodological distinctions and an experimental comparison are needed.
3. The paper reports results only in the Llama Series. Because Cartridges is presented as a general paradigm, additional experiments on Mistral and Qwen would strengthen the claim.
4. The authors claim that Cartridges can replicate the functionality of ICL; this should be backed better with evaluation on LongICLBench[1] or similar scalability tests in [2]. Likewise, their stated ability to support ultra-long contexts should be corroborated with results on RULER [3].


[1] LongICLBench: Long-context LLMs Struggle with Long In-context Learning https://arxiv.org/abs/2404.02060

[2] Many-Shot In-Context Learning https://arxiv.org/abs/2404.11018

[3] RULER: What's the Real Context Size of Your Long-Context Language Models? https://arxiv.org/abs/2404.06654

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

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
The work attacks the problem of high computational cost and reduced throughput of LLMs queried repeatedly to analyse a large (common) corpus of in-context documents.

The authors defined a cartridge - a sequence of pairs of pxd matrices that represent KV caches at every layer of the language model at hand. Here, p is the hyperparameter of the size of a cartridge, and d is the dimensionality of the model.

They suggest treating a cartridge as training parameters to represent information about a large common prefix. Given a common prefix, they suggest:
1. Generate a training dataset. In a loop, do:

1.a Retrieve a random chunk from the prefix

1.b. Seed a generic seed prompt from template categories (structuring, summarization, question, use case, creative)

1.c Generate a synthetic user query using the context of prefix chunk  + seed prompt

1.d Generate a synthetic assistant answer using the context of prefix chunk + synthetic user query

2. Construct a dataset of prefix chunk + synthetic user query + synthetic assistant answer
3. By minimizing the KL divergence between next-token predicted distributions produced by the model conditioned with the prefix chunk and the model conditioned with the cartridge over the parameters in the cartridge, we recover the optimal cartridge

Evaluations of LLaMa3-8B and Qwen3-4B on NIAH, LONGHEALTH, MTOB, and QASPER show very little quality degradation when using the cartridge instead of the long common prefix, despite ∼38.6× less memory and ∼26.4× higher throughput.

### Strengths
- The paper makes a clear, practical contribution. The proposed method has a high potential. Using self-study with chunking, the approach handles corpora beyond the model’s window. Cartridges can be concatenated at inference time. They are easy to serve with the existing infrastructure.
- The paper offers useful design ablations, including parameterization and initialization strategies.

### Weaknesses
The reproducibility of this paper is a potential weak point. Authors don’t provide code to reproduce their results, which may significantly diminish confidence and the potential impact.

Empirical results lack evaluations on real-world heterogeneous benchmarks, such as those with code or multimodal documents.

Operational procedures of versioning and updating cartridges are not discussed.

### Questions
The authors did not compare the QA performance of their cartridge system with the most obvious candidate, which is the RAG. Comparison with other prompt-compression methods (Lexico, Minicache, PALU, and KVPress) is also not provided. What would be the trade-off in accuracy and throughput?

Composition mechanism: The paper briefly notes that multiple CARTRIDGES can be composed at inference time, but the mechanism is unclear. Are there limitations in compositional depth or order sensitivity?


Evaluation breadth: The appendix lists several potential applications (e.g., summarization, retrieval, reasoning), yet the main results only cover QA and translation tasks. Were there any qualitative checks on these other use cases, even if not reported? If not, could you comment on any limitations that prevented such evaluations?

### Soundness
3

### Presentation
3

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
This paper proposed a new method called self-study, which learns to store the document knowledge in the learnable KV cache parameters using a distillation loss; the authors propose to generate data augmentations based on the document to avoid training on the same document multiple times. Experiments show that self-study matches the performance of ICL with additional spend on training compute. In addition, this method enables the model to handle documents beyond its context length.

### Strengths
1. The paper proposed a new method called self-study, which learns to store the document knowledge in the learnable KV cache parameters. The method amortize the inference compute to training compute, which is useful in many real-world applications.
2. The authors carefully studied the effect of different initialization strategies, self-study compute, diversity of seed data augmentation prompts, etc., showing solid investigation on the factors that can impact the performance.
3. The work also studied different parameterization methods for the self-study, such as prefix-tuning and LoRA. The experimental ablations are comprehensive.

### Weaknesses
1. The Figures have very small font and is very hard to read.
2. The paper didn't compare with existing baselines on memory layer and active reading, nor discussing these works in the related work section, e.g., [1][2].
3. There is missing one strong baseline that a summarizer is used to condense the document in the text space before feeding long document to context.

[1] https://arxiv.org/abs/2412.09764
[2] https://arxiv.org/abs/2508.09494

### Questions
1. See Weaknesses. Could you provide comparison with the missing baselines?
2. What are the applications that you think will need such techniques the most--Do you envision the popular applications will have different parameters for different documents? In what case, the information loss due to compression will impact the model performance?

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
This work explores the challenge of efficiently serving large language models (LLMs) for tasks requiring access to extensive text corpora.
Traditional approaches rely on placing entire corpora in the model's context window, which is memory-intensive due to the scaling of key-value (KV) cache with input length.
The authors propose an alternative: pre-training a smaller, corpus-specific KV cache called a Cartridge offline, which can be reused across multiple queries referencing that corpus.
While naive next-token prediction training does not yield competitive results compared to standard in-context learning (ICL), the authors introduce a "self-study" method, generating synthetic conversations about the corpus and using context-distillation objectives, to train Cartridges effectively.
The experimental results show that self-study-trained Cartridges match ICL performance on long-context benchmarks while dramatically reducing memory usage and increasing throughput.
Additionally, this approach extends effective context length and allows for composability of Cartridges without retraining.

### Strengths
1. The self-study technique leverages synthetic data generation and distillation objectives to further improve context compression, enabling high-quality representation learning beyond naive next-token prediction.
2. The experimental results show that Cartridge significantly reduces memory consumption and increases throughput by over 26x compared to traditional ICL methods. Despite resource savings, Cartridge achieves comparable performance as full-context ICL on challenging benchmarks.
3. The writing clearly demonstrates the paper's motivation, implementation, and analysis.

### Weaknesses
1. Effectiveness relies heavily on generating high-quality synthetic conversations. There should have been some analysis on the data quality and cost of synthetic data.
2. It would be better to compare with some naive methods that have a similar idea with Cartridge, e.g., existing prompt compression methods + SFT with your synthetic data.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
