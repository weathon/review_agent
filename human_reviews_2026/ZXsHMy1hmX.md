# Nonparametric Teaching of Attention Learners

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Attention learners, neural networks built on the attention mechanism, e.g., transformers, excel at learning the implicit relationships that relate sequences to their corresponding properties, e.g., mapping a given sequence of tokens to the probability of the next token. However, the learning process tends to be costly. To address this, we present a novel paradigm named **Atte**ntion **N**eural **T**eaching (AtteNT) that reinterprets the learning process through a nonparametric teaching perspective. Specifically, the latter provides a theoretical framework for teaching mappings that are implicitly defined (i.e., nonparametric) via example selection. Such an implicit mapping is embodied through a dense set of sequence-property pairs, with the AtteNT teacher selecting a subset to accelerate convergence in attention learner training. By analytically investigating the role of attention on parameter-based gradient descent during training, and recasting the evolution of attention learners, shaped by parameter updates, through functional gradient descent in nonparametric teaching, we show *for the first time* that teaching attention learners is consistent with teaching importance-adaptive nonparametric learners. These new findings readily commit AtteNT to enhancing learning efficiency of attention learners. Specifically, we observe training time reductions of 13.01% for LLMs and 20.58% for ViTs, spanning both fine-tuning and training-from-scratch regimes. Crucially, these gains are achieved without compromising accuracy; in fact, performance is consistently preserved and often enhanced across a diverse set of downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the high training cost of attention learners (e.g., Transformers, LLMs, ViTs)—neural networks leveraging the attention mechanism to learn implicit sequence-to-property mappings—by proposing a novel paradigm called Attention Neural Teaching (AtteNT).

### Strengths
- Prior nonparametric teaching research was restricted to multilayer perceptron (MLP)-based learners and failed to account for the attention mechanism—an essential component of modern neural networks (e.g., Transformers, ViTs). This paper removes this critical limitation by systematically investigating the role of attention in parameter and function spaces, proving that attention learners’ training dynamics align with nonparametric teaching principles .

- It introduces the Attention Neural Tangent Kernel (ANTK) and proves (Theorem 3) that dynamic ANTK converges to the "importance-adaptive canonical kernel" in nonparametric teaching. This is the first work to establish consistency between parameter-space gradient descent (used by attention models) and functional gradient descent (used by nonparametric teaching), resolving a fundamental mismatch that prevented prior application of nonparametric teaching to attention learners .

-  The AtteNT paradigm is not a trivial adaptation of nonparametric teaching—instead, it leverages attention’s "importance-adaptive updates" (element-specific weights derived from Q/K/V interactions) to design a greedy sequence-selection strategy.

### Weaknesses
- Theorem 3 only proves asymptotic convergence (i.e., when training iterations approach infinity), but real-world attention model training uses finite iterations (e.g., LLM fine-tuning typically runs 5–100 epochs, ViT pre-training 800 epochs as in Table 2). The paper does not:

  - Derive a quantitative bound on convergence speed (e.g., how many iterations \(t_{\epsilon}\) are needed for ANTK to be within \(\epsilon\) of the canonical kernel);

  -  Validate via experiments how ANTK evolves over practical iteration counts (e.g., a plot of ANTK similarity to the canonical kernel across 800 ViT training epochs).

- The paper fixes the sample selection ratio (fraction of sequences chosen by AtteNT) at 70% for LLMs (Section C.1) and uses unspecified "dynamic ratios" for ViTs (Section C.2), but provides no analysis of how this ratio impacts performance. For example:
   - Would a 50% ratio further reduce training time but hurt accuracy?
   - Would an 80% ratio retain accuracy but minimize efficiency gains?

- The paper’s ablation studies (Table 3) only compare AtteNT’s "Random/Soft/Hard" selection strategies (all derived from nonparametric teaching) but fail to contrast AtteNT with established sample selection methods for attention models. This makes it impossible to determine if AtteNT’s gains stem from its nonparametric teaching foundation or from generic greedy selection.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors suggest that attention models can be made more sample efficient by selecting a subset of the training data, that maximises the functional gradient (approximated by the L2 norm between the target and the current estimate). This idea of greedy selection is grounded in non-parametric teaching, where the best possible training set is derived for achieving a target function with a non-parametric method. In this case, the authors connect the NTK of attention to an instance-importance weighted sequence kernel rate. They show that this connection eventually justifies the greedy data selection strategy. They validate their results empirically, showing that this strategy works for a variety of smaller language and vision models while maintaining accuracy.

### Strengths
- Interesting combination of non-parametric theory with parametric learning.
- The results of the final method seem convincing for the given examples.
- Extension of existing method.

### Weaknesses
- I had a quick look at the code and it is not well-documented, one would have to invest some time to understand how to reproduce their results.
- I find the paper generally difficult to parse, but perhaps this is just my lack of background. In particular, Section 4.1 could be improved by giving some intuition and describing under which circumstances the kernel would not be adaptive in terms of $\omega_j$, but would require higher-order importance weights.
- For Theorem 3, how important are your assumptions (large scaling, small learning rate)? Is there a way to verify if or when the kernel limit holds?
- Similarly, in an experiment, if possible, can you actually verify that you stay in the kernel regime holds (by computing the theoretical kernel and comparing it with the empirical one)? Or does the training diverge from it in practice, and does this correspond to the case where the adaptive selection strategy is useful?
- For the experiment section, it would be useful if you discussed in how far the architectures also still fall under the setting you discussed previously, or how they diverge from it. Also how the setting compares with finetuning.
- What is the variability in the accuracy and runtime results (e.g. Table 1) - since you refer to average values, are you already starting several seeds?

### Questions
- see weaknesses
- The Table 3 caption could be completed with a short description of the experiment.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper interprets the parametric attention learning process from a nonparametric teaching perspective, which motivates an AtteNT algorithm that uses sample selection to accelerate the attention learning process. Specifically, the paper shows that the dynamic ANTK in parametric gradient descent converges to the importance-adaptive canonical kernel of functional gradient descent, bridging the parametric attention learning and nonparametric teaching. Theoretical results show that applying the canonical kernel can lead to sufficient loss reduction. The paper thereby proposes an AtteNT algorithm to improve the attention learning efficiency via sample selection. Experiments in both LLM and CV scenarios show the improved performance and reduced training time after applying AtteNT.

### Strengths
1. This paper makes a contribution to accelerate the attention learning, which is an important problem in LLM and CV model learning. 

2. The motivation of the AtteNT method has theoretical justification, and the experimental results further validate the effectiveness of the method.

3. The experiments are conducted over multiple LLMs and vision models, ablation study is included.

### Weaknesses
1. Some experimental settings are unclear to me. Since the paper claims the reduction of training time, which metric is used to compute the training time with and without AtteNT? Specifically, how to decide when to terminate the training for each method? Is the sequence selection time included in the model with AtteNT?

2. In Table 1 and Table 2, adding AtteNT also improves the learning performance. Why AtteNT can lead to such performance improvement is not sufficiently explained in the paper. The theoretical analysis shows the connection between the parametric and functional gradient descent, but does not include the extra benefit of using non-parametric teaching. It could be helpful to have some explanation on why adding AtteNT can lead to even better performance than traditional learning.

### Questions
1. Will the number $m$ of selected samples influence the performance?

### Soundness
3

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
This paper introduces AtteNT, designed to improve the efficiency of training attention-based models. The authors reinterpret the standard training process through the lens of nonparametric machine teaching, where a teacher strategically selects a subset of training examples to accelerate the learner's convergence. The core theoretical contribution is establishing a formal connection between the parameter-based gradient descent used for attention networks and functional gradient descent in an RKHS, showing that the Attention Neural Tangent Kernel (ANTK) asymptotically converges to a canonical kernel used in nonparametric teaching. This insight justifies a greedy data selection algorithm that prioritizes samples with the highest error.

### Strengths
1. The primary strength of this paper is the novel and elegant theoretical bridge it builds between attention learning and nonparametric teaching.
2. The theoretical insights are translated into a simple, intuitive, and practical algorithm. The idea of focusing on samples with the highest error (i.e., the hardest examples) is a well-known concept, but this paper provides a nice theoretical justification for it.
3. While the theoretical analysis is limited to a single layer but the authors show that empirically their approach and intuition works on more realistic scenarios of vision and text in scale.

### Weaknesses
1. From a practical point, additional to attention layers, the models that are used also include complexities like residual connections, layer normalization, and multiple non-linear blocks. The paper does not address how the theoretical findings generalize from the simple model to these complex ones.

2. The setup assumes a noise-less case. Eq 21 and its use in Algorithm 1 seem to be sensitive to this assumption. It is not clear or discussed how the analysis for nonparametric teaching performs with the noisy data which is more practical. A greedy data selection based on maximum of error could lead the model to choose these noisy samples, as they would consistently produce high loss. The paper does not discuss the robustness of AtteNT to label noise.

### Questions
1. There are multiple places that authors mention the convexity of loss. To make it more clear, does loss need to be convex with respect to $f_\theta$ or with respect to $\theta$? The first one seems a reasonable assumption while the second one is very limiting.

2. Another clarification question, in Table 3, there is no effect of AtteNT, is that right? Also, it was not clear from the main text what ratio and interval are referring to.

3. A small typo: Eq 17, I believe it should be $K_{\theta_t}$.

### Soundness
3

### Presentation
3

### Contribution
3
