# On the Theoretical Limitations of Embedding-Based Retrieval

- Decision: Accept (Poster)
- Scores: 4, 8, 4

## Abstract
Vector embeddings have been tasked with an ever-increasing set of retrieval tasks over the years, with a nascent rise in using them for reasoning, instruction-following, coding, and more. These new benchmarks push embeddings to work for any query and any notion of relevance that could be given. While prior works have pointed out theoretical limitations of vector embeddings, there is a common assumption that these difficulties are exclusively due to unrealistic queries, and those that are not can be overcome with better training data and larger models. In this work, we demonstrate that we may encounter these theoretical limitations in realistic settings with extremely simple queries. We connect known results in learning theory, showing that the number of top-k subsets of documents capable of being returned as the result of some query is limited by the dimension of the embedding. We empirically show that this holds true even if we directly optimize on the test set with free parameterized embeddings. We then create a realistic dataset called LIMIT that stress tests models based on these theoretical results, and observe that even state-of-the-art models fail on this dataset despite the simple nature of the task. Our work shows the limits of embedding models under the existing single vector paradigm and calls for future research to develop methods that can resolve this fundamental limitation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
With the ubiquitous use of embedding models, the authors look at theoretical limits for use of vector embeddings which cannot be overcome with increase of training data. They setup their mathematical propositions and hypothesize from the same about limitations of dense embedding models for retrieval tasks. They create a new dataset and use that to experiment. Their results indicate that the scores can be extremely poor for their settings even after training.

### Strengths
It is important in today's time with the ubiquitous use of embedding models, to carefully understand their limitations. The authors take both a theoretical and experimental approach to this very important problem.

1. The mathematical  formulation of the problem is strong.
2. The authors create a new dataset LIMIT  and use that for their work.
3. The authors establish that for tasks defined in LIMIT dense embedding models perform extremely poorly.

### Weaknesses
The experimental results are interesting. However, one must ask a pertitent question - are the authors being too pessimistic in their overall claims by creating a dataset and choosing a task which are incompatible with each other? Fundamentally, they have a dataset which is not semantic based and is a word matched based and they have an approach via dense vector embedding search which is unsuitable for this. This mismatch of choice clearly shows by the significant superiority of colbert and the near perfect match of BM25. This indicates experimenting with a wrong approach for a problem - like the cliched "if the only tool you have is a hammer .." . 

The fact that BM25 is hard to outperform on keyword heavy tasks is well researched in the literature [1-4] are examples from a quick search. I am sure a more focussed search will bring up more results.

So the question that one wonders after reading the paper - is are the claims in the abstract "Our work shows the limits of
embedding models under the existing single vector paradigm and calls for future research to develop new techniques that can resolve this fundamental limitation" are substantiated only by misplaced mapping of experiment and dataset. Their claim on LIMIT being a realistic dataset is valid but it is NOT a dataset well suited for the approach chosen. If this paper purpose was to establish to the practioner / reseacher that do not use embedding models blindly for any task it will be a very well experimented result despite the citations below. However the authors claims are far more stronger which therefore need a stronger and better experimental paradigm. 

Their extremely poor numbers of Recall are also an indication of a wrong technique to a problem and is potentially misleading to a reader who may not be as dilligent - one cannot deny the success of embedding models (the authors do not) but their results leave one wondering what went wrong. 

[1] BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models — Thakur et al., 2021.

[2] Match Your Words! / A Thorough Examination on Lexical Matching in Neural IR — Formal (T. Formal), 2021/2022.

[3] Salient Phrase Aware Dense Retrieval — “Can a Dense Retriever Imitate a Sparse One?” — Chen et al., 2021/2022 (SPAR).

[4] A Thorough Examination on Zero-shot Dense Retrieval](https://aclanthology.org/2023.findings-emnlp.1057/) (Ren et al., Findings 2023)

Minor comment:-

I note here details on writing and clarity.

1. the d in figure 2 is obtained by the hp-tuning. d is also used in definition of rank_{rt}. Line 166 defines d to be the dimension. I understand they are related but the d in figure 2 is effectively the din line 166.  maybe a different variable would have been easier.
2. line 356 refers to appendix C but this does not exist in the pdf.
3. "1000 queries to maintain statistical significance while still being fast to evaluate" - statistical significance is a bit loosely used.
4. "For example, in the full setting models struggle to reach even 20% recall@100 and in the 46 document version models cannot solve the task even with recall@20" - I cannot find two sets of results.
5. Line 457 mentions a Figure 5 which is not present in the document.
6. I struggled deciphering the colours in figure 3 and which model is colbert and which is BM25.

### Questions
1. I understand experimentally it is not possible to extend the experiments in Figure 2 beyond the numbers given, but are there any theoretical or empirical reasons to expect the equation can be extended to hundreds of times from about 50 to 4096?
2. Will LIMIT be/is publicly available ?
3. please address the concerns in the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a significant theoretical and empirical challenge to the standard single-vector embedding paradigm for information retrieval. The authors argue that as embedding models are tasked with increasingly complex, instruction-based retrieval scenarios, they are running into a fundamental theoretical limit, not just a practical one that can be solved by larger models or more data. The core theoretical contribution is to formally connect the representational capacity of an embedding model to the sign-rank of the query-document relevance (qrel) matrix. The authors prove that for any fixed embedding dimension d, there is a hard upper bound on the number of unique top-k relevant document sets a model can represent, meaning some combinations of relevant documents are impossible to retrieve, regardless of the query.

To validate this theory, the paper introduces two key empirical results. First, it demonstrates using a "free embedding" optimization (a best-case scenario without natural language constraints) that this dimensional limitation is real and predictable. Second, it introduces LIMIT, a new benchmark dataset designed to be a simple, natural language instantiation of a task with a high-sign-rank qrel matrix. Experiments show that even state-of-the-art SOTA embedding models fail catastrophically on LIMIT. The work concludes that the community must reconsider the fundamental limitations of the current dense retrieval paradigm for future instruction-following tasks

### Strengths
1. The primary strength of this paper is its formal theoretical grounding combined with a rigorous empirical validation. Moving the discussion from a vague sense of "difficulty" to a concrete mathematical limitation (by connecting embedding capacity to the sign-rank of the qrel matrix) is a novel and important contribution for the field.
2. The LIMIT dataset is a significant contribution. It is not just another retrieval benchmark but a cleverly designed probe specifically built to empirically validate the paper's theoretical claims.
3. The paper is well-written, clear, and addresses a fundamental assumption in the IR/NLP community.

### Weaknesses
1. While the simplicity of the LIMIT dataset is a strength for the theoretical argument, it is also a potential weakness. The synthetic nature of the task ("who likes X?") and the explicit construction of an all-combinations qrel matrix make it unclear how often such high-sign-rank relevance structures appear in organic, real-world benchmarks (e.g., MS MARCO, BEIR). The paper claims instruction-following will create these, but doesn't demonstrate that existing complex benchmarks already suffer from this problem.
2. The theoretical argument is based on the inability to represent all possible $\binom{n}{k}$ top-k combinations. In practice, a model only needs to represent the subset of combinations that can be expressed by plausible natural language queries. This set is likely much smaller than the theoretical maximum. The paper doesn't fully bridge the gap between "all possible" and "all plausible" combinations.

### Questions
N/A

### Soundness
4

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
5

### Summary
This paper provides fundamental limitation analysis of embedding models for IR, and shows that the such analysis results hold for many dataset instantiation.

### Strengths
The paper is well-written and easy to follow.
The experiments are well designed.
It is necessary to provide in-depth analysis for embedding models used for IR

### Weaknesses
See the below questions.

### Questions
The main goal of information retrieval is to improve the performance of a ranked list of documents in response to given a query. It is unclear how such theory analysis can help to boost the ranking performance. The motivations of finding the minimum embedding dimension are not convincing.

The row-wise order-preserving rank of A, as defined in Definition 1, is strictly constrained, where for **all** i, j, and k, they need to be subject to: if $A_{ij} > A_{ik}$ then $B_{ij}>$B_{ik}$. Can we relax the constraint? Let’s say: such that for xx present of i, j, k, if $A_{ij} > A_{ik}$ then $B_{ij}>$B_{ik}$?

In some IR task, the relevance scores can be set to be, e.g., 0 (non-relevant), 1 (relevant), 2 (highly relevant), i.e., using graded scores rather than binary scores to denote the relevance. Accordingly, in Definition 2, can we set A to be $A\in \{0, 1, 2\}^{m\times n}$ for instance?

 Are there any other limitations that the authors can provide theory analysis when applying embedding space for neural IR techniques?

What are the main differences between previous work that works on embedding dimension compression for embeddings used in IR tasks and the analysis the authors provide here?

Some writing/descriptions are not self-included and confusing. E.g., in the introduction section, the authors claim “Since embedding models use vector representations in geometric space, there exists well-studies fields of mathematical research (Papadimitriou & Sipser, 1982) that xx x”. The concept “mathematical research” is not self-included in this sentence.

Is the dataset publicly available for other researchers used in the experiments?

### Soundness
4

### Presentation
4

### Contribution
2
