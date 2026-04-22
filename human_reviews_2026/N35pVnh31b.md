# Small Models, Smarter Learning: The Power of Joint Task Training

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
The ability of a model to learn a task depends critically on both task difficulty and
model size. We study this relationship for compositional operations, focusing on
nested ListOps and extending beyond arithmetic to permutation groups, with the
goal of determining how task difficulty sets the minimum parameter requirements
for small transformer models. We vary task difficulty by introducing new operations
or combinations of operations into the training data. We find that while operations
such as modular addition or permutation group products are difficult in isolation,
joint training with other operations, including product, maximum, or auxiliary
sub-block operations, reduces the parameter requirements by factors of 2 to 7.
Analysis of learned embeddings using PCA reveals that when joint training helps it
is usually accompanied by an increase in highly regular structures in the embedding
of inputs. These results suggest that joint training leads to qualitatively different
learning trajectories than learning operations in isolation, with shared number
representations supporting difficult tasks such as addition. Our findings further
demonstrate the importance of training curriculum on the emergence of abilities in
language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work investigates how task difficulty and model size jointly determine a transformer’s ability to learn compositional operations such as modular arithmetic and permutation groups. The authors find that joint training with additional or auxiliary operations can dramatically reduce the parameter requirements, compared to training on isolated tasks. The authors further attribute this improvement to better learned embedding via PCA analysis. They also observed that the training trajectories are quantitatively different.

### Strengths
1. The paper presents a very clear setup and is well written.
2. The use of datasets with controllable difficulty provides a strong foundation for conducting rigorous and systematic research.

### Weaknesses
My main concern is the limited analysis of model size. The authors constrain the model to a single head and model depth one with recursive reuse, but prior work shows that recursion under a tight parameter budget can severely restrict model capacity even with a similar FLOPs [1], and that the number of heads matters for modular arithmetic datasets [2]. This does not invalidate the authors’ contributions, but a more thorough study of model size, especially head count and depth, would be important.


1. https://arxiv.org/pdf/2507.10524

2. https://arxiv.org/pdf/2502.10390

### Questions
I was wondering whether the authors have examined in detail how the mixture of datasets contributes mechanistically to learning. While the paper shows that the embeddings exhibit clear structure, most of the interpretability analysis focuses solely on the embedding level. It would be helpful if the authors could further analyze how the learned features evolve with recurrent depth, whether certain layers or depths specialize in specific tasks. More importantly, I am curious whether any shared computational circuits emerge as the dataset's complexity increases.

Why do the authors present results primarily for non-prime moduli in most plots? Wouldn’t prime moduli represent a more challenging setting, where qualitatively different behaviors might emerge?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates if small models can learn difficult tasks through a synthetic (extension of) ListOps dataset. They discover that joint training with multiple tasks can often lead to efficient learning. They also study embedding patterns to give a 'white-box' explanation for their models.

### Strengths
1) The paper studies the effect of joint task learning with multiple experiments, which cover a lot of different aspects of the learning problem (eg. prime v/s non-prime moduli, shuffling of order for ADD, etc.)
2) Interpretability done with embedding vectors was useful. In particular, the restricted embedding hypothesis was a strong evidence for the claims made about the utility of joint training.
3) Experiments on permutation groups present an interesting addition, with a lot of scope for future works.

### Weaknesses
1) There is some literature which talks about compositional and/or multi-step mathematical reasoning, the paper was missing references to these [1, 2, 3]. Although the current paper has several important experiments which were missing or not considered in the papers that are mentioned below, it will be useful for the authors to devote some space to discussing these differences in the main text. 
2) The last section on 'Discussion and Limitations' doesn't really discuss limitations. For example, these results on synthetic datasets may not transfer immediately to realistic data. The restricted embedding hypothesis, while useful for interp, may not be suitable for practical purposes. 
3) Finally, these results on mathematical operations may not necessarily translate to large scale given the widely known problem of arithmetical reasoning in LLMs. This could be alleviated by performing experiments with increasing context length to the maximum limit as allowed by the authors' compute budget.


[1] A. Abedsoltan, H. Zhang, K. Wen, H. Lin, J. Zhang and M. Belkin, “Task Generalization With AutoRegressive Compositional Structure: Can Learning From $D$ Tasks Generalize to $D^T$ Tasks?,” arXiv preprint arXiv:2502.08991, Feb. 2025.

[2] T. Wang and W. Lu, “Learning Multi-Step Reasoning by Solving Arithmetic Tasks,” arXiv preprint arXiv:2306.01707, Jun. 2023. 

[3] W. You, S. Yin, X. Zhao, Z. Ji, G. Zhong and J. Bai, “MuMath: Multi-perspective Data Augmentation for Mathematical Reasoning in Large Language Models,” arXiv preprint arXiv:2405.07551, May 2024.

### Questions
1) Could you also state that in appendices A.2 contain results for prime moduli (in the first para of Section 3)?
2) Have you tried any experiments where you leverage the ability of the model to learn multiple tasks by then focussing on each of the tasks individually? An example could be to see if further finetuning is necessary for a model to learn just ADD once it has learnt MAX+MED+ADD?
3) Another experiment in the same vein might be to try to finetune models to learn slightly OOD tasks (eg. PROD which is ADD in log space).
4) Finally, if time permits, another interesting experiment to try would be to see what happens if the tasks are presented in a continual learning fashion? Is that as useful as joint learning?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper studies how small transformer models learn algorithmic tasks, particularly compositional mathematical operations such as those in the ListOps. The authors explore how task difficulty, model size, and training curriculum interact to influence learning efficiency and the emergence of abilities in small models.

### Strengths
Quality and Originality: The paper presents a systematic empirical exploration of how joint task training impacts the learning thresholds of small transformer models and attempts to reframe the understanding of scaling laws.

### Weaknesses
1. Lack of Novelty. Results are not surprising and authors need to discuss the multi-task learning (i.e., joint training can improve the performance is a common wisdom) and curriculum learning literature. Authors in the introduction discuss scaling laws but they do not provide the exact formulation of laws that contain curriculum learning factors. No precise alternative of KC complexity as well.

2. Small-scale experiments do not support motivations and hurt significance: Authors only conduct experiments on a small scale, not widely used Transformers architecture, and I cannot verify if this can be generalized to larger architectures. Also, based on the text part of Model Architecture, it's hard to reproduce the used model, and the authors should provide more details there (for example, providing a figure or model signature with specific layers will be very helpful to understand your experiments). Larger scale (with more parameters) experiments seem important to verify anything related to "scaling laws".

3. Potentially Trivial Experiments: Not sure if basic arithmetic + permutation group computations (although controllable) are important targets for LLM analysis. When the authors say EASY/HARD tasks, I find it surprising that some of these synthetic tasks are hard for these transformer models and before the Methodology section, I strongly recommend the authors provide preliminary background knowledge to explain the context.

4. Presentation and clarity could be significantly improved. There are too many references to the Appendix without a careful explanation of details about what these experiments are in the main text (For example, line 80 Appendix K) and till line 80 it's hard to understand what is this randomized sum table and its difference with pure SUM.  For permutation groups operations, like OP, it will be much better to provide a concrete example in the main body. Also please fix the presentation of Figure 2 since some texts now overlap.

### Questions
1. How well do these findings transfer to natural language or real-world multi-task learning scenarios? Demonstrating such transfer would strengthen the broader relevance of the conclusions.
2. Can the authors formalize or quantify how task combinations interact to better predict which combinations yield synergy versus interference?
3. Can the authors situate their findings within the broader context of several key work of scaling law theory and at least think about how to incorporate the missing parts like curriculums?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates how joint task training can reduce the parameter requirements for learning arithmetic tasks. The authors also provide a preliminary mechanistic explanation suggesting that joint training leads to more structured embeddings.

### Strengths
The paper presents several interesting phenomena

- The embeddings of jointly trained models separate even and odd numbers on ADD and PROD tasks. 
- The experiment on shuffled SUM suggests that the benefit of joint training emerges only when the easy and hard tasks share underlying numerical properties (Figure 3).
- The transfer learning experiment demonstrates that pretraining on simpler tasks and then transferring to harder ones is an effective curriculum. (Figure 5)

### Weaknesses
- The novelty and significance of the work may be limited.
  - The general observation that joint training or curriculum learning benefits language models, including for arithmetic tasks, has been reported previously. The most related prior work I know is [1]; it would be helpful for the authors to clarify how their approach and findings differ from that work.
  - While the finding that joint training leads to more structured embeddings is interesting, the paper does not analyze how these embeddings form or how they influence the parameter requirements of learning the task.
- It would be helpful if the authors could more explicitly summarize the main takeaways, epspecially how the findings on synthetic tasks might transfer to more realistic scenarios.
- The transfer learning experiment in Section 3.1 does not convincingly support the “embedding-restriction” hypothesis to me, as the results do not directly demonstrate that the effective search space is reduced.

- The paper would benefit from including at least a minimal related-work section in the main text for better contextualization.
- Minor comments (do not affect the score):
  - l161 "sizewe" -> "size we".
  - The interchangeable use of *SUM* and *ADD* could be confusing; I recommend using one consistently.
  - Figure 2a: the two subplots overlap.
  - Figure 2 caption (line 266): “(c)” should likely be “(b)”.
  - l348 "The also show".

### Questions
- Why did the authors choose to use a recurrent version of the Transformer instead of a standard multi-block architecture? Would the results remain consistent under a conventional architecture? It would be useful to clarify this choice more explicitly.

- How did the authors determine the performance for a fixed number of parameters? As discussed in Section 4, even after convergence, grokking may occur for arithmetic tasks.
- The jointly trained embeddings reportedly separate even and odd numbers. Do the authors have an explanation for why such parity patterns emerge in the ADD and PROD tasks, given that parity does not appear to play a direct role in either task?

### Soundness
3

### Presentation
2

### Contribution
2
