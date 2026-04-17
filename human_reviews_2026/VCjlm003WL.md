# Characterizing Pattern Matching and Its Limits on Compositional Task Structures

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Despite impressive capabilities, LLMs' successes often rely on pattern-matching behaviors, yet these are also linked to OOD generalization failures in compositional tasks.
However, behavioral studies commonly employ task setups that allow multiple generalization sources (e.g., algebraic invariances, structural repetition), obscuring a precise and testable account of how well LLMs perform generalization through pattern matching and their limitations.
To address this ambiguity, we first formalize pattern matching as functional equivalence, i.e., identifying pairs of subsequences of inputs that consistently lead to identical results when the rest of the input is held constant.
Then, we systematically study how decoder-only Transformer and Mamba behave in controlled tasks with compositional structures that isolate this mechanism.
Our formalism yields predictive and quantitative insights:
(1) Instance-wise success of pattern matching is well predicted by the number of contexts witnessing the relevant functional equivalence.
We prove a tight sample complexity bound of learning a two-hop structure by identifying the exponent of the data scaling law for perfect in-domain generalization.
Our empirical results align with the theoretical prediction, under 20× parameter scaling and across architectures.
(3) Path ambiguity is a structural barrier: when a variable influences the output via multiple paths, models fail to form unified intermediate state representations, impairing accuracy and interpretability.
(4) Chain-of-Thought reduces data requirements yet does not resolve path ambiguity.
Hence, we provide a predictive, falsifiable boundary for pattern matching and a foundational diagnostic for disentangling mixed generalization mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work presents a formalization of pattern matching by quantifying evidence for functional equivalence. The authors systematically study how strength of functional equivalence in the data leads to different kinds of success and failures in compositional generalization tasks. The formalization is accompanied by empirical evidence with both Transformers and mamba architectures on data scaling, learned internal representation of functional equivalence, and the role of CoT on a set of synthetic compositional generalization tasks. The results yield important insights on the limitations of compositional generalization supported by pattern-matching behavior, including generalizing to in-frequent data and multi-hop problems.

### Strengths
- This work makes a significant contribution in providing a formalization to the boundary of pattern matching and compositional generalization.
- The paper is dense but I find it a good read. I appreciate that the authors studied a range of important issues beyond the base setting within the task suite, including testing two architectures, data/model scaling, interpretability, tasks with different compositional structures, and CoT.
- The results make several interesting implications for the capabilities of larger models.

### Weaknesses
- I think it would be good if the implications for specific LLM failures are discussed in more detail, as well as more discussions of whether certain aspects of the formalization would/would not apply given how natural language data may differ in properties studied in the synthetic settings. Though I understand this could be in part due to the space limit.

### Questions
- For clarification, how is k computed if there is functional equivalence between more than two subsequences?
- I'm curious what might happen if the models are trained on a mixture of different compositional tasks, or a mixture of compositional/non-compositional tasks? IMO this potentially captures the nuanced structures in natural languages, in which certain aspects are strongly compositional and others less so. Without explicit task cues, how would this change the learned strategy? Would functional equivalence be learned earlier or later? Would models still develop some form of context/task-dependent functional equivalence?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper characterizes “pattern matching” in sequence-to-sequence tasks as the identification of particular equivalences between pairs of certain input variables in sequences (subsequences of the input sequence), when all else is held equal. They investigate the ability of models to learn this kind of information by creating a synthetic domain where no other information exists, specifically by framing the problem as the composition of binary functions of the input, each of which is a random lookup table. In general, they find that transformers appear to be using this kind of pattern recognition when trained on this kind of data; but only when these equivalences appear several times. In general, in situations where such an equivalence is found, evidence can be found in intermediate layers for clustering. They develop a scaling law for the data needed to perform this kind of pattern matching. Additionally, they discover that having a non-tree-structured computation graph (where the same input variable can affect the output via multiple paths), causes pattern recognition to become much harder.

### Strengths
The domain setup seems to eliminate other potential sources of information cleanly. The definition of functional equivalence and specifically k-equivalence are simple and naturalistic definitions.

The large sweep over a variety of dataset sizes is also helpful for determining the role of data access.

### Weaknesses
The abstract and first paragraph of the introduction do not make it clear enough that “pattern matching" is undesirable. The first sentence could be read as “pattern matching" performed by LLMs as being too surface level. This reading recontextualizes later uses of the term to be neutral rather than negative, confusing such a reader. It should be made more clear that “pattern matching" specifically is being used to exclusively refer to undesirably syntactic/surface level heuristics.

"Functional equivalence, i.e., substituting input fragments observed to result in identical outputs in shared contexts” (used in the abstract and introduction lede) should be rephrased as, by itself, this is too vague to communicate what it means to someone who has not read the paper.. Perhaps something like “functional equivalence, i.e., identifying pairs of subsequences of inputs that consistently lead to identical results when the rest of the input is held constant.”

“Tightly ordered by” is used several times, but as far as I can tell is not a standard term. I think a term like “well predicted by” would be more standard and thus easier to read.

### Questions
Pattern matching is generally defined in terms of substitution rules (i.e., $a * (b + c) \to a * b + a * c$. Is the notion of pattern matching you define equivalent to unbounded substitution rules (with an unlimited number of variables on each side of the $\to$)? Is it a generalization? Is it a subset? A discussion of this would help ground your definition in the context of existing notions of pattern matching.

What is the purpose of holding out 30% of inputs and having a test condition where you add a new one in? In my understanding, this task is impossible to perform above chance on, as the primitives are defined as random tables.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper is about when LLMs can learn many-to-one (or, equivalently, non-injective) maps. A many-to-one map is a function such that two different inputs yield the same output. All maps are deterministic. In their experiments, an LLM is trained on many examples of such functions (sometimes composed with each other) and is asked to produce outputs on unseen inputs. For example, if we see $f_1(a) = f_1(b) = c$ and $f_2(f_1(b), d) = e$, we can infer that $f_2(f_1(a), d) = e$ as well, without having seen this particular example before. 

They use many-to-one maps as a formalization of what pattern matching means in LLMs. They then create quantifiable tests for when pattern matching is or is not happening. They also explore how data requirements scale with the complexity of the task, and how the amount of computation necessary to learn pattern matching scales with the size of the input vocabulary. Each section of the paper is summarized as follows:

- Their first experiments show how the number of times an invariance is observed affects the test accuracy on the many-to-one maps. Despite the fact that just two observations of an invariance is sufficient to implement an algorithm that solves the task perfectly, the LLM only begins to generalize well until after more examples of the invariance are seen (3+, depending on the amount of training).
- They next develop a metric for measuring how well internal representations capture shared states. In particular, one might hope that if two function inputs lead to the same output, the internal representations would be similar after a few layers. Similarly, if two function inputs lead to different outputs, they internal representations should be different at all points. They capture this by measuring cosine similarity between inputs that induce the same output and cosine between inputs that induce different outputs and then taking the difference between these two quantities. They then show that the models trained on their many-to-one task exhibit high values on the metric.
- They next consider how much data should be needed to learn the task, for a given input vocabulary. They show that, depending on how many examples are needed to confirm an invariance, the number of examples needed scales as polynomial in the size of the domain, with exponent between 2 and 2.5. By scaling the input size, they verify empirically that these scaling laws hold.
- When the function composition structure is more complex (the computation graph is not a tree, and is instead some general directed graph), they show the LLMs do not perform well. This is because it is harder to generalize when the same token is used for multiple different arguments to a function.
- They also train the models to output intermediate computations and find that it leads to more data-efficient generalization. However, when learning with general graphs rather than trees there is still no generalization.

### Strengths
- At a high level, thinking about generalization in terms of many-to-one functions seems like it clearly captures a kind of task-level generalization. Completing the task correctly requires non-trivial logical reasoning. The task has the nice properties that (1) it is possible to get 100% accuracy when correctly applying logical reasoning / a graph algorithm and (2) the LLM never sees the exact problem instance it is evaluated on.
- The empirical results in the paper strongly support the narrative presented. The setups of the experiments are thoughtful, and the analysis is suitable for the questions addressed in the paper. I would say the first results (that more examples of an invariance are helpful) is not very surprising, but I learned a lot from the second through fourth analyses (scaling laws, non-tree tasks, and CoT).

### Weaknesses
- I found the exposition introducing the problem to be a bit confusing. It wasn’t clear to me whether pattern matching is a desirable or undesirable property of transformers (is it capturing overfitting or generalizing?) The abstract suggests that surface-level pattern matching is bad, but perhaps that deeper pattern matching (which survives multiple logical steps) is a good thing.
- I am also confused about why it is interesting to understand pattern matching in LLMs. I’m not sure how the toy problem is supposed to generalize to other tasks, or shed insight into broad phenomena around LLM reasoning. The introduction did not really have examples of pattern matching in real-world LLM uses, so this added to the confusion. There were many citations supposedly about pattern matching, but not any explanation of how we are supposed to imagine pattern matching works beyond the toy problems in this paper itself. I believe there could be a good motivation for studying this problem, but I don’t get it from this paper, and I’d like the authors to explain.
- I think there could be more discussion of what makes the non-tree case hard. Is there not a way to reduce the non-tree task to the tree task? Or is generalization impossible because of the same token is used for multiple arguments.
- Another minor critique I have is about terminology: “functional equivalence” to me suggests that two functions are the same. I think “functional invariance” is a clearer term?

### Questions
What are the class of real-world problems or tasks that being good at pattern matching helps solve? Is it supposed to capture logical reasoning? Why this particular logical reasoning task versus another (like syllogisms or SAT problems)?

### Soundness
4

### Presentation
2

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
This paper formulates functional equivalence as the condition where two sets of tokens produce the same output in the same context. The experimental results demonstrate that the model's performance and internal representation of these equivalences get stronger as more of these functional equivalences occur in the training data. The paper also analyzes the scaling law (required training data size depends on the vocabulary size), the effect of ambiguous composition, and the same phenomenon in CoT training.

### Strengths
1. This paper studies an important problem of the compositional generalization of language models.

2. The experiments include various settings of practical relevance.

### Weaknesses
1. The results are limited to small synthetic task structures.

2. The settings require a deterministic function and strict functional equivalence, which may be too restrictive in a real-world NLP dataset.

### Questions
1. If the task structure is more complicated, would observing robust functional equivalence still be enough for compositional generalization? Wouldn't task complexity challenge the model's understanding of the task, resulting in training failure?

2. Is there any equivalent empirical observation where compositional generalization requires robust observation of functional equivalence in a real-world NLP dataset?

3. Would the formulation in the paper generalize to stochastic functions and approximate functional equivalence?

### Soundness
3

### Presentation
2

### Contribution
2
