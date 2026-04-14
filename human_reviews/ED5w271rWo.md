# Banyan: Improved Representation Learning with Explicit Structure

- Decision: Reject
- Scores: 8, 6, 5, 3

## Abstract
We present Banyan, a model that efficiently learns semantic representations by leveraging an inductive bias towards explicit hierarchical structure. Although typical transformer-based models excel at scale, they struggle in low-resource settings. Recent work on models exploiting explicit structure has shown promise as efficient learners in resource-constrained environments. However, these models have yet to demonstrate truly competitive performance. Banyan bridges this gap, significantly improving upon prior structured models and providing, for the first time, a viable alternative to transformer embeddings for under-represented languages. We achieve these improvements through two key innovations 1) A novel entangled tree structure that resolves multiple constituent structures into a single shared one, explicitly incorporating global context. 2) Diagonalized message passing functions that increase the influence of the inductive bias. Our final model has just 14 non-embedding parameters yet is competitive with baselines many orders of magnitude larger. Banyan outperforms its structured predecessors and competes with large unstructured models across various semantic tasks in multiple languages. Notably, it excels in low-resource settings, highlighting its potential for efficient and interpretable NLP in resource-constrained environments. These results underscore the value of appropriate inductive biases in capturing semantic relationships and open new avenues for efficient, interpretable NLP models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces a new recursive graph neural network for learning text representations in low-resource languages: Banyan (like the tree). This model extends previous work by building nested trees over sequences that share the same tokens. In Banyan the same tokens will have the same tree node, even if it comes from different sequences. For scalability reasons, the trees are constructed from a batch of samples rather than from an entire dataset. Embeddings are learned from a simplified message passing algorithm that traverse the trees both in bottom-up and top-down directions.
Having nested trees provides multiple advantages, notably the reduction of duplicated nodes and multiple context representations within the same node (in downward embeddings).
These advantages translate to strong semantic representations in both English (when compared to RoBERTa) and lower-resourced languages (when compared to XLM-R, Llama 3.1, Mistral).

### Strengths
This paper introduces a novel recursive model that learns textual representations and its learning mechanism.
The proposed architecture is novel and seems promising as it yields good results when compared to other more classical methods.
In addition, the proposed method is very efficient: it requires very little training and has only 14 non-embedding parameters.

### Weaknesses
The task being tackled is not clear from the abstract or the introduction. The motivation is well described (learning powerful text representations for under-resourced languages), but the task used to evaluate the Banyan model (Semantic Textual Similarity - STS task) is not described. The paper could gain clarity by mentioning the task earlier.

Evaluation is based on cosine similarity between sentences compared to human judgment. The paper would attract greater significance if the representations from the proposed model were tested in targeted applications such as sentiment classification or machine translation.

### Questions
Previous approaches used 1 tree per sequence. Banyan uses one tree for all sequences sharing the same tokens in a batch.
Some analysis of scale would be nice to complement Figure 4 in the paper. For instance, how many trees do you usually have in an entire batch? How many (sub-)sequences are represented in 1 tree?

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
The paper presents Banyan, an efficient and lightweight framework for representation learning on low resource languages. The method has been evaluated in common English tests as well as a series of low-resource languages. This method demonstrates great performance as well as remarkable efficiency. While the technical innovations are impressive, there are still several remaining concerns.

### Strengths
This proposed method is technically solid, and makes substantial improvement compared with existing methods.
The proposed method is highly efficient and lightweight, especially when compared with Large language models.

### Weaknesses
Some evaluation parts still need further validation. The details are described in questions

### Questions
1. Regarding low resource language evaluation, the authors only used four languages, while there are many other languages in the released semeval dataset, is there any cherry-picking on the languages evaluated ? How will the model perform on other low resource languages ?
2. The authors trained Banyan on datasets from Leipzig Corpora Collection, is there any overlap between the training corpora and the testing dataset ? and if the baseline methods such as XLM-R are also pre-trained on these corpora, how will the models perform ?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
In this work, the authors present a method called "Banyan" that can learn the semantic representations with hierarchical structure. It has two main improvements (entangled tree structure and diagonalized message passing functions) comparing to SELF-STRAE. According to their experimental results, Banyan achieves competitive performance with transformer based baselines. It also shows the low cost and efficiency.

### Strengths
1. The proposed method is a novel approach for semantic representation. Tree structure is inject into the representation.

2. According to the experiments results (Table 1 and Table 2),  Banyan achieves competitive performance with transformer based baselines on sentence level and work level. The results are also much better than previous baseline (Self-StrAE).

3. There are a clear ablations study on the effect of different modeling changes. It is helpful for others to understand the method.

### Weaknesses
1. For this proposed semantic representations, both the structure and representation are learned. The first  embedding tokens are used to determine which tokens to merge and in what order. The initial token embeddings have a large impact on the proposed method. It is a little unclear about this part (are they randomly initialized? Are they updated during model training? etc) 

2. Only 14 non-embedding parameters are used in the proposed method. It could also limit the ability the proposed model. It would be great if the proposed method can be used in the transformer-based embedding in the future.

### Questions
There are composition and decomposition (corresponding up and down embeddings), what are the embedding used for sentence-level and word-level?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper presents a strategy of representation learning by utilizing structural information discovered during learning. The proposed work is built upon a prior work with two specific pieces of improvement on building the structures and propagating the information during representation learning. Empirical evaluation was performed on multiple NLP tasks.

### Strengths
- The proposed new strategies for constructing tree structures are intuitive
- The proposed framework is parameter efficient and easy to learn.

### Weaknesses
- The writing of this paper can be significantly improved. There are some inconsistent statements and inconsistent terminology, which should be easily fixed after proofreading. For example, the paper uses both "under resourced languages" and "under represented languages", which I think it should be "low-resource languages". 
- There are some unsupported claims, for example, in the first section, the claim "It is thought that this principle lets humans generation ..." should be supported with references from prior work. 
- There are also some technical details that are unclear in the paper. For example, in line 204 - 207, what are criteria of entangling different trees together, and what are the benefits of entangling subtrees together? 
- There is a minor issue with the citation format; it should be, for example, (Tai et al., 2015).
- There are some existing works along the line of upward and downward along a tree structure to learn representation, which are not discussed in this paper. For example, Ji and Eisenstein, One Vector is Not Enough: Entity-Augmented Distributed Semantics for Discourse Relations, 2015. 
- The experiments are not sufficient; for example, some recent large language models are not included in the comparison.

### Questions
Please refer to the comments in the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
