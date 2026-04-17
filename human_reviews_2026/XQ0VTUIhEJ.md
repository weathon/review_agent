# Whitened Self-Attention

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Self-attention in Transformer architectures is formulated as a function of the
pairwise contributions between target vectors and their context vectors. This con-
struction implicitly assumes ternary and higher order relationships are negligible.
It further treats the context vectors as though they can be processed individually,
as if mutually independent of one another. This model contradicts, however, our
understanding of language: that the meaning of words is influenced by complex
interdependencies. We introduce Whitened Self-Attention, a theoretically motivated
and novel enhancement that optimally accounts for inter-token correlations, and
based on several covariance modeling assumptions, we derive a computationally
feasible implementation for it. Experiments with a small GPT architecture show
that whitened self-attention reduces perplexity by 19.3%, achieves the same mean
cross-entropy loss in 37 times fewer training iterations, and, after hyperparameter
optimization, reduces training time by 91%. Our approach shows significant poten-
tial for scaling and for improving the performance and generalization of large-scale
language models. Moreover, as whitening decorrelates input sequences, it will
affect the structure of the resulting trained attention and feedforward weight matri-
ces. This will have an effect on their singular value decompositions, and should, in
turn, influence the results of the many studies on the mechanistic interpretability of Transformers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a modified self-attention mechanism motivated by whitening the input features of the self-attention. The authors claim that empirically the it improves over normal self-attention on a 2-layer GPT model.

### Strengths
The idea is straightforward and simple to understand. The motivation (whitening) is also less discussed for self-attention so this study could potentially add research diversity to the field.

### Weaknesses
For the technical correctness:
 - I am not sure why on line 161, you can assume the $L_{i}$ and $M_{I}$ converges to a steady state values, and also, why you can replace every recursion with your assumed steady state $L_{\infty}$, $M_{\infty}$. Please elaborate more on these, this doesn't look trivial to me.
 - Following the above point, after the simplification made with the assumption of the authors, I am not sure how is the model trained/encouraged to whiten the input sequence. For equation 9, without any regularization loss for learning the $L_{\infty}$, $M_{\infty}$ matrices, how do you encourage them to converge to the ground truth $L$ that whitens the input? The authors should elaborate more on this.

For the experiments:
 - I don't think Figure 4 well supports the authors claim about the functionality of the whitening filter. It is true that from (a), it shows decreased correlation as training progresses. However, it should be compared with a baseline without the proposed whitening filter. Since the token embedding and the layer norm are trained, it could be that they naturally learn to be de-correlated with training.
 - From Table 3, it seems like the author did run experiments on 7M models. Why not also post the results of MCE for the 7M model? Now I am not sure if the model scales well given the super small model size.
 - From the experiments (Figure 3), it is unclear to me if the model overfits or underfits without the training set MCE. Thus it is also unclear to me if the methods helps generalization, or it helps the model better fits the dataset. E.g., if the training MCE curves are the similar for both methods, then the proposed methods improves majorly generalization. The authors should clarify this.

### Questions
See the above weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes WSA, a theoretically motivated variant of Transformer self-attention that accounts for correlations among context vectors. The authors argue that standard attention treats context vectors as independent, an assumption that contradicts linguistic dependencies, and derive a whitening transformation based on covariance modeling. The core idea is approximating whitening via a recursive formulation (Eq. 9) that can be trained jointly with the model. Experiments with a 2-layer GPT-like model trained on the collected works of Dickens demonstrate perplexity reduction over standard self-attention, significant reduction in compute time to reach comparable validation loss, and that whitening improves decorrelation metrics (Fig. 4a) and stationarity of early-layer inputs (Fig. 4b). The authors argue this supports both theoretical and empirical efficiency improvements and could have implications for interpretability studies of trained Transformers.

### Strengths
- Mathematically principled derivation connecting self-attention to optimal linear estimation under correlated contexts (Secs. 3-4).

- Clear explanation of bias from correlated context vectors, intuitively shown in Figure 1.

- Empirical evidence that whitening improves convergence speed and decorrelation metrics, yet on toy scale experiments (Fig. 4a, Tabs. 1-2).

- Transparent discussion of limitations and future work (Sec. 7), including computational scalability via prefix-sum parallelization.

### Weaknesses
-  The experimental validation is extremely narrow, as all results are on a character-level Dickens corpus using a 1.6-1.9M parameter GPT with 2 decoder blocks. This corpus has short range dependencies and very limited vocabulary (93 tokens), which aligns well with the model’s covariance assumptions. It is unclear whether the same improvements would hold for tokenized or multilingual corpora with complex co-occurrence structures. Hence, while results are strong within this setup, the generalization to realistic language modeling conditions is questionable.

-  The whitening is derived assuming first-order stationarity and modeled with a block tridiagonal covariance. Figure 4b shows this assumption breaks down for the second decoder block, where non-stationarity increases drastically. This empirical evidence implies the whitening process fails in deeper layers, hence its impact to realistic multi-layer Transformers is again questionable.

- Even with block simplifications, recursive whitening introduces additional parameters and potential gradient instability. The authors acknowledge whitening recursion grows with sequence length and is not GPU-friendly, and suggest prefix-sum parallelization (Sec. 7), but this is prospective, not demonstrated. The method’s scalability to long contexts (e.g., N > 2048) remains speculative.

- The comparison is limited to vanilla SA with no comparison to existing orthogonalization or other-related methods. For example, orthogonal attention regularization (Xiao et al., ICML 2024).

Minor:
- Empirical results are heavily overplotted (Figure 3) without error bars or multiple runs.
- Table 1 and Figure 3 show large apparent speedups (“91% reduction”), but training time comparisons are based on different batch sizes and parameter counts (1.62 M vs 1.88 M weights, 256 vs 128 batch). Per-iteration cost is actually far higher for WSA, and the runtime savings come primarily from faster convergence under this specific loss surface. Without reporting total FLOPs or wall-clock time normalized by model size, claims of “91% reduction in training time” might be misleading.

### Questions
Please see the weaknesses section. Also, could whitening be applied selectively to early layers only, given non-stationarity later?

While theoretically interesting, this paper’s empirical evaluation is too limited. WSA could become an impactful line of work with broader validation, realistic datasets, proper baselines, and statistically rigorous reporting. At present, it falls short of ICLR’s standards.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents Whitened Self-Attention (WSA), a modification to the standard self-attention (SA) mechanism in Transformer architectures. The authors' core thesis is that standard self-attention is suboptimal as it consider context vectors as independent. The authors argue that these vectors are highly correlated, and standard attention's pairwise, weighted-sum approach (as seen in Equation 1) leads to "double-counting" of information and a biased, inefficient representation.

The authors propose to apply a whitening transform to the context vectors before they are fed into the attention mechanism. This transform decorrelates the vectors, making them stochastically orthogonal, which allows the subsequent weighted-sum operation to be an optimal linear estimator.

Recognizing that computing the full whitening matrix is computationally infeasible for large sequences, the authors derive a practical implementation.

Experiments on a small GPT model trained on a character-level dataset show promising results int terms of performance and efficiency.

### Strengths
Promising experimental results: The authors do a great job at motivating the community to further explore their methodology through demonstrating strong results. The performance improvements are not trivial indeed - a 91% reduction in compute time to reach an equivalent loss is great.
Pragmatic and realistic derivation: The authors manage to nicely bridge the gap between the theoretically motivated whitening (but computationally impossible) solution and a practical, implementable one. The step-by-step simplification from the full covariance matrix is logical and well-explained.
Interesting analysis of the whitening process: It is appreciated that the authors did not only present results from the Whitening filter, but also empirically validated how the whitening impacts outputs through training.

### Weaknesses
Scale of experiments very limited: While I full agree with the statement of the authors in the discussion regarding the importance of having controlled experiments in smaller setting, this particular setup is far from representative of the conditions under which GPT architectures demonstrate their characteristic behavior. The use of a ~1 M parameter model and a single corpus (Dickens) makes it impossible to infer how the proposed whitening mechanism would behave at scale. Given that Transformer performance, optimization dynamics, and attention patterns fundamentally change with model size, the presented results should be interpreted as conceptual evidence rather than empirical validation of scalability.
The writing would benefit from better structuring: I found the result section particularly convoluted to parse, with numerous results presented in succession, spread accross multiple tables. Furthermore, methodology of the experiments (e.g. batch size, number of parameters) was also merged with results and analysis - sometimes within the same paragraph. I recommend the use of subsections and perhaps focusing on the most important results and leaving the rest for a supplementary material.

### Questions
I am particularly confused by the choice of character-level tokenization - particularly in the context of the paper which aims to whiten (i.e. decorrelating) the input vectors. Character-level sequences inherently exhibit lower semantic correlation than word- or subword-level embeddings, which makes them a poor test bed for studying correlation structures in attention. Wouldn’t the results be more meaningful if demonstrated with more standard tokenization or embedding strategies, where inter-token correlations play a much greater role? Since it is such an uncommon choice (and combined with the already small scale of the experiments) I would expect either additional experiments with standard tokenization or a stronger justification for why character-level data is appropriate for evaluating the proposed method.

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper is about improving attention in transformer models. The main observation is that token embeddings are temporally correlated, and the authors posit that removing these correlations would improve training efficiency. They achieve this by applying a temporal "whitening" transformation right before each attention layer.

The authors train a very small transformer model to evaluate their approach. The results show a dramatic improvement in validation perplexity. However, more experimental work to establish confidence in the method is required before it is published at a top tier conference.

### Strengths
* The experimental results show very strong perplexity improvements.

* The exposition is clear, and a good discussion of extensions and limitations in Sec 7. The authors' perspective of whitening is interesting.

### Weaknesses
* The transformer model is too simplistic. I believe the model size of 2m params is too small to give confidence in the results. In addition, there is two model properties that might confound the results: 1. baseline model only has two layers, while the proposed model is very sequential since tokens are processed sequentially, possibly making its effective depth larger. 2. the single-character tokenizer is too weak, and there is a possibility that the proposed technique gives strong improvement because of that (this is actually suggested by the authors' results that the whitening is strong in layer 1 but not layer 2). The authors should evaluate using deeper models and real tokenizers.

* I did not find the theoretical or empirical motivation of why tokens need to be whitened. It would help to have some experiments to prove that this is an issue. Also the authors make multiple logical leaps on the covariance structure (e.g. tridiagonal structure) that are not backed by data.

* I did not see any ablations.

### Questions
Q: Did the authors tune the learning rate of the baseline?

### Soundness
2

### Presentation
2

### Contribution
2
