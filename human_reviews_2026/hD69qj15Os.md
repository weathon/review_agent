# Cross-Tokenizer Likelihood Scoring Algorithms for Language Model Distillation

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 2

## Abstract
Computing next-token likelihood ratios between two language models (LMs) is a standard task in training paradigms such as knowledge distillation. Since this requires both models to share the same probability space, it becomes challenging when the teacher and student LMs use different tokenizers, for instance, when edge-device deployment necessitates a smaller vocabulary size to lower memory overhead. This work addresses this vocabulary misalignment problem by uncovering an implicit recursive structure in the commonly deployed Byte-Pair Encoding (BPE) algorithm and utilizing it to create a probabilistic framework for \textit{cross-tokenizer likelihood scoring}. Our method enables sequence likelihood evaluation for vocabularies different from the teacher model native tokenizer, addressing two specific scenarios: when the student vocabulary is a subset of the teacher vocabulary, and the general case where it is arbitrary. In the subset regime, our framework computes exact likelihoods and provides next-token probabilities for sequential sampling with only $\mathcal{O}(1)$ model evaluations per token. When used for distillation, this yields up to a 12% reduction in memory footprint for the Qwen2.5-1.5B model while also improving baseline performance up to 4\% on the evaluated tasks. For the general case, we introduce a rigorous lossless procedure that leverages BPE recursive structure, complemented by a fast approximation that keeps large-vocabulary settings practical. Applied to GSM8K mathematical reasoning distillation, our method improves accuracy by over 2%  the current state of the art. Code: https://github.com/truongbuu/cross-tokenizer-scoring

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces cross-tokenizer likelihood scoring algorithms that enable language models trained with one tokenizer to compute exact or approximate likelihoods for sequences encoded with a different tokenizer by exploiting the recursive structure of Byte-Pair Encoding. The authors prove the merits of their methods through comparing against baselines on the knowledge distillation task.

### Strengths
1. This paper is addressing an important research question: how to compute next-token likelihoods and perform knowledge distillation when teacher and student models use different tokenizers (i.e., resolve vocabulary misalignment). 

2. The paper is well written.

3. The paper’s idea is novel.

I don't have the background to evaluate more fine-grained part of the methodology proposed in the paper.

### Weaknesses
I don't have the background to evaluate more fine-grained part of the methodology proposed in the paper. I will focus on evaluating the baselines.

In the experiment, the paper doesn't include the baselines for other methods that try to address the knowledge distillation problem with teacher and student sharing different tokenizers (e.g. those methods that align teacher and student on the level of embedding). I understand that the paper claims that it focuses on probability conversion on the tokenizer directly. However, in my opinion, unless the paper also show that their method is helpful in addressing some tasks other than knowledge distillation, other knowledge distillation baselines may need to be included. This is especially so when considering the fact this paper can only deal with the BPE tokenizer, while other aligning methods may deal with arbitrary tokenizers and models for the knowledge distillation task.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Cross-Tokenizer Likelihood (CTL), a novel probabilistic alignment method addressing inconsistencies between multilingual tokenizers.
By minimizing per-token log-likelihood gaps across languages, CTL improves cross-lingual representation consistency and translation faithfulness.
The method is simple, elegant, and easily integrable into existing multilingual models (Qwen2.5-7B, XGLM-4.5B).

### Strengths
1. Tackles a long-standing issue—tokenizer mismatch—using a mathematically grounded likelihood objective rather than architecture tricks.

2. Improves multilingual alignment and reduces tokenization bias, especially for low-resource or morphologically rich languages.

3. The CTL layer is training-agnostic and introduces negligible computational overhead.

4. Consistent improvements across translation, code-switching, and QA tasks; simplicity and reproducibility make it valuable for practitioners.

5. The loss function is interpretable and differentiable, connecting probabilistic alignment with linguistic intuitions.

### Weaknesses
Only three downstream tasks (translation, QA, code-switching). Additional domains such as summarization or retrieval would strengthen generality.

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
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a method to exactly convert BPE-vocab LM models to any subset vocabulary(eg. Byte-level) in O(1) model evaluations. Furthermore, the authors propose a (somewhat expensive) method to approximately up-convert the byte-level vocab LM to an obtain probabilities for any other BPE vocab. The authors show that their proposed vocab conversion achieves low approximation error in token probabilities, and can be used for cross-model distillation and vocab pruning.

### Strengths
1. The proposed method computes exact sub-token probabilities in O(1) model evaluations for BPE vocabs
1. The proposed method is "training-free" on the teacher - requiring no training of new LM-heads, projections, etc
1. The authors successfully utilize their method to distill across models and for vocabulary trimming.

### Weaknesses
1. The proposed method has extremely large overhead for cross-tokenization distillation - large number of beam-search (6-8 beams) upto maximum length 10 for calculating every token probability.
1. Only 1 baseline method is compared against (ALM) - other methods for cross-model distillation should also be compared.
1. Empirical evaluations are extremely limited.

### Questions
1. For Figure 3 (Section 6.1), can the authors share the effective LM loss (probability of ground truth token) for the original model and their re-converted model, and the LM loss of other smaller original Qwen models (no conversion needed) on the same samples? This can more directly show how much performance/"effective model size" is being lost in this conversion.
1. In Table 3, for vocabulary trimming, can the authors also train the original (full vocab) model with the same warmup and distillation process? The vocabulary reduced models surprisingly achieve a "higher" score than the original models, while would imply the training process is significantly improving the model. Without these original scores, there is no way to judge the effectiveness of this conversion.
1. For the beam search approximation in C1,  can the authors compare the quality of the predicted probabilities as the beam size is varied?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the issue of cross-tokenization, which is of paramount importance in the context of LLM distillation. The issue arises from the vocabulary misalignment problem, often caused by different tokenizers used in various language models. The paper introduces a new approach to tackling this issue, known as cross-tokenizer scoring or cross-tokenizer conversion, which builds strongly on the structure of the BPE algorithm widely used in current tokenizers. 
The paper introduces the following contributions :
+ An analysis of the sequential structure of the byte-pair encoding and the introduction of the notion of relative alphabets.
+ Cross-tokenizer scoring algorithms
+ Experimental validation on two tasks: cross-tokenizer distillation and vocabulary trimming.

### Strengths
**originality**
+ The idea of cross-tokenizer conversion for LLM Knowledge distillation is original. 
+ The notion of relative alphabets.


 **significance**
+ The questions being asked are important since LLM distillation, assuming different vocabulary, is a very common practical use case.

### Weaknesses
+ The proposed approach is primarily limited to BPE tokenization algorithms. While it is true that many current tokenizers are built on BPE, it is not always the case. 
+ The clarity of the paper is clearly a big weakness of the paper. In particular, the proposed formalization is difficult to follow due to a lack of motivation or descriptions of the intuitions behind the concepts. For instance, section 4.2 is mainly a succession of definitions. In addition, some of the proposed definitions are generalizations of existing ones, such as relative cover encoding. Why is the notion of cover encoding important? A detailed positioning of the proposed approach with respect to the work of [Phan et al,. 2025] is also missing. There are certainly some very good ideas in this paper, but the format makes it very difficult to read and understand. It would have been interesting, for example, to consider a diagram highlighting the general framework of the proposed approach, particularly the concept of relative alphabets on which everything is based.
+ Experimental validation also does not allow us to highlight this notion of relative vocabularies and how it impacts the targeted tasks: LLM KD and vocabulary trimming.

### Questions
+ What about cross-tokenization outside the BPE algorithm? 
+ Would it be possible to describe more explicitly what contributions the approach makes in relation to the work of [Phan et al., 2025.]? 
+ In term of practical applications, what brings the concept of relative vocabularies ? 
+ How can it be used more concretely in a KD context, for example? 
+ Why was the standard experimental protocol of [Minixhofer et al. 2025] not followed in the experimental validation?

### Soundness
2

### Presentation
2

### Contribution
2
