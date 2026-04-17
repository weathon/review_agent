# Understanding the Mechanisms of Fast Hyperparameter Transfer

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
The growing scale of deep learning models has rendered exhaustive hyperparameter (HP) optimization prohibitively expensive. A promising solution is the use of scale-aware HPs, which can enable direct transfer of optimal settings from small-scale grid searches to large models with minimal performance loss. Such approaches are useful when the optimal settings converge "fast" enough with scale. While approaches like the Maximal Update Parameterization ($\mu$P) have empirically displayed fast transfer when scaling model width, a deeper conceptual understanding of the mechanisms that enable this is still missing. Our work establishes a systematic conceptual framework for analyzing fast HP transfer across different synthetic and practical scenarios. In synthetic settings, we present various quantitative examples where transfer either offers a provable computational advantage or fails even under $\mu$P. 
  We then propose a key property that enables the fast transfer often observed in practice: through a novel decomposition of the optimization trajectory, we identify one component that rapidly converges with model width and determines the optimal HPs, and the other that continues to improve the loss with increased width but has negligible impact on HP choice. We conjecture that this decomposition elucidates the key mechanisms behind fast transfer and empirically validate it in practical settings such as LLM training.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper develops a formal framework for understanding fast hyperparameter transfer. Fast HP transfer means that hyperparameters tuned on a smaller model (less width) transfer to bigger models (more width). They give theoretical reasoning why this works, for example, in muP. They decompose the loss into top-k components and residual loss. The top-k components determine the optimal HPs and converge quickly across widths, which explains the fast HP transfer. This is validated through experiments on transformers and MLPs.

### Strengths
In summary:

* The paper is well-written and nicely put into the context of the related work. 
* The framework of HP transfer is novel and provides interesting novel insights and theory on the topic.
* The idea of the loss decomposition seems to be new as well and quite useful, possibly also for other applications. 
* The paper gives theoretical and empirical reasoning for why the HP transfer works.
* They also verify their theory on real-world models in their experiments, next to analytical ones.
* Understanding HP transfer is relevant given the trend to very large models, which can be tuned more computationally efficiently with HP transfer.

### Weaknesses
* It would be nice if the authors introduced the concept of muP in the introduction, since your paper is based on it. For someone not being super familiar with the topic, it needed quite a bit of effort to get into the topic.
* Multi-Fidelity optimization is also a type of HP transfer. I would appreciate a discussion on the relation to multi-fidelity as a fairly established approach.
* Please double-check for introducing abbreviations before using them.

### Questions
* How is k for top-k chosen in practice? What are the implications of it?
* What effect would a negative change in loss have?
* For which other HPs other than learning rate does your framework hold? Is your framework also applicable to other scale factors like depth, data, or number of training epochs?

### Soundness
3

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
2

### Summary
The work introduces a conceptual framework to study the transfer properties of hyperparameters of neural networks from low to large scales. A decomposition of the loss suggests a connection of certain optimization statistics to the convergence properties of optimal hyperparameter settings.

### Strengths
* Understanding the mechanisms behind hyperparameter transfers from low to large scales is an important issue to study as it can inform future algorithmic directions
* Assumptions are clearly stated
* Existing literature on analysis of hyperparameter transfer across scales and related analysis methods is well covered

### Weaknesses
* Unclear if there are any direct practical implications of the introduced conceptual framework
* In line with previous work, the conceptual framework targets a very specific choice of hyperparameters and optimization settings.
* Code to reproduce the experiments is not provided, although some details on the trainings and fixed hyperaparamter settings are provided. What are the specifics of the "Llama-style" transformer architecture you used?

### Questions
* The authors focus on a very specific choice of hyperparameters. They state "however our framework can be used more broadly for reasoning about scale-aware HPs", could you elaborate on that? Where are the limitations of your framework?

* What scale in terms of # of trainable parameters do your width choices correspond to?

Minor comments
* Numbering all equations would be good to facilitate discussions
* Axis in Figures descriptions are very small

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper explores the gap in understanding of what determines the fast convergence rate of zero-shot hyperparameter transfer at the infinite-width limit using $\mu$Parameterization, as seen empirically. 
The authors present a novel decomposition of the loss trajectory through a linearization scheme over an EMA of the original trajectory. 
They show that the sum of the top-k eigenvalues of the gradient matrix captures the loss change in the top-k directions of maximum change in loss.
This measure can be seen to be invariant across width-scaling, whereas the other residual components appear to reflect scaling with width.
The authors posit that this indicates the existence of a subspace of the trajectory (as captured by EMA smoothing), such that tuning for those top-k components at a width $n$ can guarantee fast hyperparameter transfer as $n \rightarrow  
\infty$.

### Strengths
* Novel framework and formulation to try and explain the practical gains of an important theoretical finding in $\mu$Parameterization.

### Weaknesses
* Hard to follow the notations in Section 4, especially with the incremental evolution of $\phi$ and its usage.

* Hard to see the practical implications of the finding, especially with the requirement of adjusting $k$ for any new task. Unless the only goal was to *explain* why fast HP transfer shows up in practice, the claim in abstract for validating this for LLMs might need to be revised.

* The understanding of the *main* paper on its own is a bit hard, with adequate parsing of the Appendix details quite crucial to sometimes connect points made in the main paper.

### Questions
Below are the enumerated questions and suggestions.

Please note that the rating will be increased contingent on the points below being addressed/clarified.

1\. The authors could consider explaining in brief the comparative scaling of the hyperparameter value compared to the performance metric with width, as the meaning of "fast transfer", in the abstract.

2\. L60-63: The authors could consider referring to the relevant sections.

3\. L116: In comparison to previous literature, is one of the contribution of this paper, to show that the top-k eigenvalues of the gradient matrix can adequately capture invariance across appropriate width-scaling?

4\. L172:173: Does this specifically apply to only certain types of hyperparameters? Or only the abcd-parameterization parameters that are scaled in width?

5\. L266-269: Could the authors explain why this happens and its implications?

6\. L255-266: *Top-$\kappa$ strong convexity*, here both variables appear same, $\phi_{n}^{\kappa_n}$ . If $\kappa(\nu)$ is an integer related to width-$n$, then what does the super- and sub-script denote?

7\. Figure 4.b): Can this be explained more please? Especially the role or choice of `Step 7185`. What is the blue dot line, in $LR(n) = \nu^{*}(n)$?

8\. Figure 7 (right): Could different colours be used here?

9\. L460-464: Is flatness of the loss residual the issue here or the loss gap at $n \rightarrow \infty$?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
1

### Summary
Scaling-aware hyperparameters propose hyperparameter tuning with a small-scale model (e.g., small width/depth) and then directly use the tuned hyperparameter (after modification with certain rules) for training a large-scale model. Recently, Maximal Update Parameter ($\mu P$) proposed a theoretical framework for scaling-aware hyperparameter tuning when scaling model width, but in the mean-field regime only. However, such fast-hyperparameter transfer phenomenon also works with finite width/data as well, but there is no prior explanation/justification for this. This is the main motivation of this work. 

To explain fast hyperparameter transfer in width-scaling, the authors propose a new decomposition of the optimization trajectory into two components: (1) a component that converges quickly with model width and determines the optimal hyperparameters, and (2) a component that improves the loss when the width is increased but has a negligible impact on HP choice. The authors then hypothesize that this one might explain the hyperparameter transfer with width scaling, and then validate their finding with empirical experiments.

### Strengths
The paper is very well-written and easy to follow. I hardly see any typos. The motivation is well-presented and very clear. Though I am not really an expert in this field, I can follow the paper very easily. I think it is a good paper.

### Weaknesses
First things first, since I am not an expert in this field, my opinion might be biased and not fairly evaluate the contribution of this paper. 

1. One might argue that the paper only focuses on fast hyperparameter transfer in terms of width. In practice, one might want to know about fast hyperparameter transfer when scaling the depth instead. However, I would not consider it an issue, since it is the weakness of the Tensor Program series in general. Besides, I think that the setting is good enough, since we can simply fix the depth, train with minimum width, and it should be fast enough. 

Since I am not an expert, I can only give support for this paper with a low confidence score. I will leave the decision to the AC and other expert reviewers.

### Questions
See weaknesses

### Soundness
3

### Presentation
4

### Contribution
3
