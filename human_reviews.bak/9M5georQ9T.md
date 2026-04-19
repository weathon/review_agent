# Optimizing Attention with Mirror Descent: Generalized Max-Margin Token Selection

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 6

## Abstract
Attention mechanisms have revolutionized numerous domains of artificial intelligence, including natural language processing and computer vision, by enabling models to selectively focus on relevant parts of the input data. Building on recent results characterizing the optimization dynamics of gradient descent (GD) and the structural properties of its preferred solutions in attention-based models, this paper explores the convergence properties and implicit bias of a family of mirror descent (MD) algorithms designed for softmax attention mechanisms, with the potential function chosen as the $p$-th power of the $\ell_p$-norm. Specifically, we show the directional convergence of these algorithms to a generalized hard-margin SVM with an $\ell_p$-norm objective when applied to a classification problem using a one-layer softmax attention model. Our theoretical results demonstrate that these algorithms not only converge directionally to the generalized max-margin solutions but also do so at a rate comparable to that of traditional GD in simpler models, despite the highly nonlinear and nonconvex nature of the present problem. Additionally, we delve into the joint optimization dynamics of the key-query matrix and the decoder, establishing conditions under which this complex joint optimization converges to their respective hard-margin SVM solutions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper examines how mirror descent (MD) algorithms approach convergence and exhibit bias when optimizing attention mechanisms in softmax attention, with potential functions using $\ell_p$-norms as a basis for analysis. Key theoretical findings include demonstrating convergence towards hard margin Support Vector Machine (SVM) solutions and achieving convergence rates similar to gradient descent, in more straightforward models. The research builds upon findings related to descent by expanding the scope to encompass a wider range of optimization algorithms and shedding light on properties related to token selection.

### Strengths
1. The paper has a strong theoretical foundation, it provides rigorous mathematical analysis and proofs for the convergence properties of mirror descent in attention optimization, extending previous work on gradient descent to a more general framework.
2. This paper provides a novel algorithmic insight, the introduction of $\ell_p$-AttGD generalizes both $\ell_p$-GD and attention GD, offering new perspectives on attention optimization and token selection.
3. This work provides a complete theoretical treatment of the optimization dynamics by examining both fixed-head and joint optimization scenarios.

### Weaknesses
1. The empirical evaluation is limited, the paper includes experiments, and they are primarily focused on synthetic data and a single real-world dataset (Stanford Large Movie Review Dataset). More diverse real-world applications would strengthen the practical implications.
2. Theoretical results are highly dependent on assumption, Theorem 2, Theorem 3, and Theorem 4, rely on specific assumptions about initialization and step sizes, which may limit their practical applicability.
3. The paper does not thoroughly address the computational overhead of implementing mirror descent compared to standard gradient descent.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper investigates the optimization dynamics of mirror descent (MD) for training attention mechanisms, specifically focusing on $l_p$-AttGD. The authors claim that $l_p$-AttGD converges directionally to a generalized hard-margin SVM with an $l_p$ norm objective when applied to binary classification with a single-layer attention model. Some experiments on synthetic and real data are presented.

### Strengths
The paper attempts to provide a theoretical analysis of mirror descent for attention training, extending prior work focused on gradient descent. It derives convergence results to a generalized hard-margin SVM and establishes convergence rates. The use of $l_p$ norms offers a degree of generality in the theoretical analysis.

### Weaknesses
1. The core idea of connecting attention optimization to SVM-like objectives is not new and has been explored in prior work, notably in "A Primal-Dual Framework for Transformers and Neural Networks" by Nguyen et al. and related papers. These prior works establish the fundamental link between attention and SVMs, including the optimization perspective. While this paper extends the analysis to mirror descent, the incremental contribution feels minimal and lacks motivation. Other core ideas of analyzing the implicit bias of GD/MD algorithms for softmax attention is already present in the cited works (Tarzanagh et al., 2023, 2024; Vasudeva et al., 2024a; Sheen et al., 2024) without fundamentally changing the nature of the problem or leading to significantly different insights.

2. The synthetic experiments are too simplistic and lack the complexity needed to represent realistic attention training scenarios. The real-data experiments, while showing some improvements in generalization, are insufficient to support the claims. The demonstration of improved token selection is based on a handful of examples (Figure 8) without any rigorous quantitative evaluation. Comparisons with GD do not include other commonly used optimization algorithms like Adam, making it impossible to judge the relative merits of MD for attention training.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes to study the dynamics of a simplified one-layer attention network. The implicit bias of gradient descent for such a problem is already known, this paper proposes to study what happens when the training algorithm is mirror descent with p-norms. The paper extends previous results known for gradient descent to this setting, and then demonstrate that using mirror descent to train a transformer on a sentiment analysis dataset yields better generalization.

### Strengths
- The paper is well written 
- It is quite easy to understand
- The fact that MD achieves sparser weights and leads to better generalization is interesting.

### Weaknesses
- The motivation is quite unclear. “A broader understanding of general descent algorithms, including the mirror descent (MD) family and their token selection properties, is essential.” why? In practice, nobody trains attention layers with mirror descent. The observation that it works better than gradient descent is not very strong in my opinion, because in practice transformers cannot be trained with gradient descent either. In the experiments, it would be worthwhile to compare the proposed methods to the widely used adam algorithm. Also, providing train losses values for the real data experiment would be enlightening.
- There are many unclear things about global and optimal tokens (def.3). First, the fact that there is a feasible solution is very important, precise conditions for it to happen must be put in the paper. The authors simply state that “under mild overparameterization d ≥ max{T − 1, n}, the problem is almost always feasible (Tarzanagh et al., 2023, Theorem 1).” These conditions must be put in the paper. Also, it is unclear whether the statement is “for all $\alpha_i$, the problem is feasible” or “the problem is feasible for at least one set of $\alpha_i$.
- I cannot understand why Theorem 1 does not require those conditions: clearly, $W^{opt}_{mm}$ must exist for it to hold, and there is nothing in the assumptions guaranteeing it.
- It is never properly stated in the paper if “globally optimal tokens” are also “locally optimal”. It seems not to be the case, since globally optimal tokens depend on $v$ while locally optimal tokens do not. This also causes issues in thm 1: globally optimal tokens do not seem to be always feasible, so W^{opt}_{mm} is not always defined, while the lp-AttRp always has a solution. For instance, it could very well have a global solution of fixed norm.

### Questions
- Can the authors clarify the questions above regarding feasibility? 
- Can the authors compare the training curves with that of adam?
- The model sizes do not seem to be specified. The architectural details should be put in the appendix


**Minor remarks and typos**

- L68: ERM refers to an equation much later in the manuscript
- L71: W(k) has not bee introduced thus far
- L113: the softmax acts here on a matrix, but later in (2) it acts on vectors.
- L139: (Blair, 1985)
- L165: (Tarzanagh et al. (2024; 2023)).  
- L165: the sense of nearby is unclear here.
- L169: it would be good to specify to which set the indices belong. Generally, it would be clearer to indicate in which space lives each newly introduced variable.
- L176: “the αi component has a significantly higher probability” : this statement should also involve the norm of W, right? Because if all values in the pre-softmax vector are high, then the probability vector will be almost constant.
- Eq 4b: typo W -> W’
- L280: $exp(2)$ could be $\exp(2)$
- In eq.5, it would be good to recall that $D_\psi$ depends on $p$.
- All figures should be in pdf format
- “In the Appendix G, we show that W and v generated by ℓp-JointRP converge to their respective…” it would strengthen that section to put it in the main text.
- Fig4: a log scale would make things clearer, especially highlighting the practical convergence rates.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel approach to optimizing attention mechanisms using Mirror Descent (MD), specifically focusing on a generalized max-margin token selection strategy for softmax attention models. The authors propose a family of MD algorithms, termed $\ell_{p}$-AttGD, which generalize traditional gradient descent by using the $p$-th power of the $\ell_p$-norm as the potential function. The main contributions include proving the convergence of these algorithms to generalized max-margin Support Vector Machine (SVM) solutions for optimizing the attention mechanism, both for fixed and jointly optimized parameters (key-query matrix and decoder).

### Strengths
1. Theoretical Contributions: The paper provides a solid theoretical foundation for understanding the convergence properties and implicit bias of MD in attention models. The extension to $\ell_p$-norm objectives adds flexibility in modeling and opens up new avenues for optimizing attention mechanisms.

2. Generalization of Attention Optimization: The approach generalizes previous work on attention models by using MD with a broad class of potential functions, allowing a deeper exploration of the optimization landscape beyond vanilla gradient descent.

### Weaknesses
1. As far as I know, mirror descent is not a popular optimization algorithm for training deep learning models. I agree that a simplified model (e.g., the one layer model considered in this paper and previous work) could provide valuable insights on understanding transformers, but it is not clear what is the role/implication the $\ell_p$-norm for deep learning. If possible, it would helpful if the authors could highlight a few practical mirror descent-based optimizers in the revision.

2. For Line431, for the $\ell_{1,1}$/$\ell_3$ optimizer, the authors provide code for implementing the $\ell_{1,1}$/$\ell_3$ optimizer for optimizing the transformer. In term of efficiency, would the $\ell_{1,1}$/$\ell_3$ optimizer be as efficient as popular optimizers like AdamW [LH2019]? Could the authors provide comparison with AdamW in Table 1?

[LH2019] Decoupled Weight Decay Regularization. Ilya Loshchilov, Frank Hutter.

### Questions
1. What does the comparison token $z_i$ mean in practice (Line 55)? For example, for a real-world application problem, how to set the comparison token $z_i$ given the input $X$ and label $y$?

### Soundness
3

### Presentation
3

### Contribution
3
