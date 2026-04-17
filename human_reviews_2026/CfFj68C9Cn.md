# Learning to Recall with Transformers Beyond Orthogonal Embeddings

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Modern large language models (LLMs) excel at tasks that require storing and retrieving knowledge, such as factual recall and question answering. Transformers are central to this capability because they can encode information during training and retrieve it at inference. Existing theoretical analyses typically study transformers under idealized assumptions such as infinite data or orthogonal embeddings. In realistic settings, however, models are trained on finite datasets with non-orthogonal (random) embeddings. We address this gap by analyzing a single-layer transformer with random embeddings trained with (empirical) gradient descent on a simple token-retrieval task, where the model must identify an informative token within a length-$L$ sequence and learn a one-to-one mapping from tokens to labels. Our analysis tracks the ``early phase'' of gradient descent and yields explicit formulas for the model’s storage capacity---revealing a multiplicative dependence between sample size $N$, embedding dimension $d$, and sequence length $L$. We validate these scalings numerically and further complement them with a lower bound for the underlying statistical problem, demonstrating that this multiplicative scaling is intrinsic under non-orthogonal embeddings. Code to reproduce all experiments is publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work establishes a learnability statement on a specific factual recall task. Within a sequence, there is a token position which is marked, and the goal is to retrieve this token with the attention mechanism and then use the MLP to recall the correct label of the input, which is assigned randomly. This task is analysed in the first few steps of learning dynamics in the large limit of tokens, and under various (scaling) parameters such as embedding dimension, sample complexity and hidden layer size. The authors provide both a learnability criterion under these hyperparameters and a statistical lower bound based on the accessibility of attention queries that holds for more general algorithms (but is not very strong or specific). 
The main takeaway is that denoising the correct item and memorisation are more difficult for longer sequences, where a larger embedding size and/or more samples are needed to reach a similar performance as for small sequences.

### Strengths
- The presentation and clarity of the writing make it easy to follow the work.
- The dynamical analysis of the first three steps already gives a good insight into the general behaviour and the type of solution that is learned to fulfil the recall task.
- The illustration of Thm1 in terms of comparing the signal with the noise at various network components is helpful and intuitive.
- The empirical findings validate the theoretical results closely, even when the theory does not hold anymore for longer time dynamics.
- This work offers plenty of interesting directions to follow up on for the community.

### Weaknesses
- The model is not very close to large-scale models, but it has some resemblance to factual recall tasks and comes with the advantage of being analyzable.
- see questions

### Questions
- Do you have any insights in your model on how strictly non-orthogonal vectors would do, and how close to non-orthogonality your embeddings actually are? Since they are random and not learned, I would expect that they are still "quite" orthogonal. What would change in your experiments if you could actually learn them too, and therefore impose better-than-random, learned, possibly non-orthogonal (or more orthogonal...) structure? Do you have an intuition for that? I think an empirical ablation like this would help contextualise the result a bit better.


Some suggestions/clarifications on intuition:
- In the description of the problem setting, you say "can then recall the correct label via an associative memory mechanism." (L.126). Later on, L.144, you say the goal is to "learn the target function (permutation)". I understand that the VxV matrix is essentially the random input class->output class mapping, but it took me a moment to see that this is where the memorisation comes in; you might want to make this more explicit.
- Do you have an intuition about how increasing the sequence length L and the vocabulary length makes the task harder? Longer lengths make detection harder, and larger V memorisation - but if intuitively the two functionalities are located in different models (L.125), why do they both scale linearly in all of the other parameters, and not independently?
- Do you observe that $W_{KQ}$ learns $z_{trig}$?
- L.219 "during the second gradient step," -> "after the first step" (?) (the object reference is a bit ambiguous)
- What do the different alpha (transparency) values of the regressed information mean? It is there in almost all figures.

### Soundness
4

### Presentation
4

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
This paper investigates the gradient-based learning of the Transformer architecture for identifying informative tokens within the context. The theoretical result provides conditions on hyperparameters for successful learning, and the empirical result validates this.

### Strengths
1. This paper addresses an important problem of the learning dynamics for Transformer models.

2. The paper complements its theoretical predictions through experiments.

### Weaknesses
1. The results in the paper are limited to a simple toy model (easy task, single layer, 3 simplified gradient steps on a fixed batch). Although additional experiments exist beyond this toy setting, it remains unclear how the results will generalize to more realistic settings.

2. The setting assumes that the input contains a trigger embedding that marks the informative token, which makes this toy setting less practical. It is difficult to imagine a realistic task where this assumption holds.

### Questions
1. What are the useful implications of the theoretical results? Specifically, what is the takeaway for practitioners who train Transformer language models? What is the main contribution to the deep learning theory or language model theory that is not available from prior works?

2. Does adding an MLP layer on top of the attention layer hurt? It seems that this makes the theoretical condition for successful learning more difficult to satisfy.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work presents a theoretical analysis of the training dynamics of transformers on a factual recall task. Specifically, it considers two model architectures: (1) a single attention-only layer and (2) one attention layer followed by an MLP layer. The study examines how these transformers learn to map an output token to an informative position within a sequence. The authors quantify model accuracy after three steps of gradient descent and establish learnability guarantees as functions of vocabulary size, sequence length, model size, and the number of training samples. From an empirical perspective, the results provide practical guidance on selecting the hidden dimension and sample size, while also elucidating the role of the MLP component in recall tasks.

### Strengths
+ The paper is clearly written, with precise notations and formal language, and includes detailed technical assumptions.

+ The theoretical results appear sound and solid. To the best of my knowledge, this work provides the first theoretical analysis of training transformers to perform recall tasks, particularly in the regime where the embedding dimension can be smaller than the vocabulary size.

+ The theoretical findings are further validated through numerical experiments conducted in controlled settings, which support the correctness of the theory and demonstrate its potential generalization to more practical scenarios.

### Weaknesses
- Some assumptions may require further justification. For instance, in Assumption 1, it is not entirely clear why it is necessary to set $L = V^c, c \in (0, 1)$? Under this assumption, the sequence length is necessarily smaller than the vocabulary size, and the motivation for this specific scaling is not well explained.

- It is also suggested that the authors present their results in a more comparative manner and provide readers with additional background context. My understanding is that the main contribution of this work lies in analyzing the learning dynamics of transformers on factual recall tasks in the regime where $d < V$. However, it is somewhat difficult to determine whether the theoretical settings in this paper are consistent with prior works. For example, regarding the choice of model architectures, learning tasks, training algorithms, assumptions, and the final results. Without such contextualization, it becomes challenging for the general audience to assess the true contribution of the paper and identify whether the new result stems from new assumptions or from novel proof techniques.

### Questions
It appears that the considered factual recall task can be learned exactly after just three steps of gradient descent, provided that the embedding dimension or the sample size is sufficiently large. Could the authors offer some intuition for why this occurs? For example, even in the orthogonal regime, $d > V$, why can the model achieve perfect recall accuracy with only three-step GD?

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
This paper presents a theoretical framework for understanding how transformers acquire factual recall when trained on finite data with non-orthogonal embeddings. Focusing on single-layer attention and attention-plus-MLP architectures, the authors derive precise conditions under which gradient descent can identify and retrieve informative tokens within a sequence. They show that learning efficiency follows a multiplicative relationship among vocabulary size, sequence length, embedding dimension, MLP width, and sample size. Unlike prior analyses, this work establishes learnability even when the vocabulary size is larger than the embedding dimension. Empirical results corroborate the theoretical scaling laws and demonstrate relevance to practical settings.

### Strengths
This work provides a clear problem formulation and rigorous proofs supporting its theoretical claims. The technical results are overall a good contribution . Beyond theory, it offers numerical experiments that closely match the analytical predictions, showing the precision and tightness of the derived results. The findings also yield practical insights into transformer training, key architectural parameter choice, the importance of the MLP component.

### Weaknesses
- The clarity of some parts needs improvement (see Questions). It is recommended to include a brief intuition or proof sketch to better convey the key insights underlying the proof. The current presentation may be less accessible to a broader ML audience.
-  The gap between theory and practice remains substantial. The definition of the factual recall task requires further clarification. In the current setup, the model identifies a marked token within a sequence and maps it to another token through a ground-truth permutation $\Pi^*$. However, this formulation appears disconnected from practical scenarios, particularly regarding how the key information is represented or “marked” in the token features. It also seems misaligned with the setting described in the cited work [1].
- As for the theoretical aspects, it is recommended to include a quantitative summary of prior works. For example, specifying what learnability results/bounds have been established in those studies that analyze orthogonal embeddings.

[1] Nichani et al., Understanding Factual Recall in Transformers via Associative Memories

### Questions
- The clarity of some assumptions needs to be improved. For example, what do $\eta \approx 0$ and $\gamma \gg 0$ specifically mean? Why are these conditions not expressed using big-O notation?
- Similarly, it may be non-standard to write $\Omega(V \log V) \le N$ and $V \ge \Omega(1)$. When using big-O or related asymptotic notations, should readers assume that these are taken with respect to $V$?
- In addition, how did the authors name or interpret each term in Eq. (6)? It is recommended to provide readers with more intuition or explanation for these terms to enhance understanding.

### Soundness
4

### Presentation
4

### Contribution
4
