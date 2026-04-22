# FACT: a first-principles alternative to the Neural Feature Ansatz for how networks learn representations

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 4

## Abstract
It is a central challenge in deep learning to understand how neural networks learn representations. A leading approach is the Neural Feature Ansatz (NFA) (Radhakrishnan et al., 2024), a conjectured mechanism for how feature learning occurs. Although the NFA is empirically validated, it is an educated guess and lacks a theoretical basis, and thus it is unclear when it might fail, and how to improve it. In this paper, we take a first-principles approach to understanding why this observation holds, and when it does not. We use first-order optimality conditions to derive the Features at Convergence Theorem (FACT), an alternative to the NFA that (a) obtains greater agreement with learned features at convergence, (b) explains why the NFA holds in most settings, and (c) captures essential feature learning phenomena in neural networks such as grokking behavior in modular arithmetic and phase transitions in learning sparse parities, similarly to the NFA. Thus, our results unify theoretical first-order optimality analyses of neural networks with the empirically-driven NFA literature, and provide a principled alternative that provably and empirically holds at convergence.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present FACT, a self-consistent condition for L2-regularized neural networks. This condition provides a more theoretically grounded alternative to the NFA conjecture, establishing a connection to (locally) minimum-norm solutions. The authors demonstrate both empirically and theoretically that FACT aligns more closely with the behavior of trained fully connected networks than existing NFA-based approaches.

On the empirical side, the authors compare FACT-RFM with NFA-RFM, showing that the two exhibit similar phenomena, such as phase transitions and grokking, across various tasks. Theoretically, they prove a case, under a minimum-norm objective, where the NFA conjecture fails while FACT continues to hold.

### Strengths
The main contribution of this work is to extend the empirically motivated NFA conjecture to a more theoretically grounded framework based on (local) minimum-norm conditions. This perspective helps bridge the gap between the NFA line of research and the broader literature on L2-regularized solutions, highlighting RFM methods as an interesting alternative to gradient-based training.

The paper is clearly written, and the experiments are illustrative. To the best of my knowledge, the proofs are correct. Overall, this is a solid submission with only minor weaknesses.

### Weaknesses
The paper could benefit from a stronger early narrative explaining why FACT is a meaningful measure — specifically, how its foundation in (local) minimum-norm behavior motivates the framework. Although Sections 5 and 6 provide this perspective more clearly, the earlier parts of the paper introduce it less directly. For instance, establishing the connection earlier would help clarify how the low-rank bias and neural collapse phenomena [1] in Figure 2 relate to L2 regularization [2-3], and likewise how the grokking behavior in Figure 5 connects to L2-driven dynamics [4].

As a minor comment, the acronym “FACT” has appeared previously in the literature [5]. In addition, the name emphasizes self-consistency at convergence but does not reflect the central role of L2 regularization, which is arguably more important.

 1. X. Y. Han et al., 2022 Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Central Path
 2. T. Galanti et al., 2024 SGD and Weight Decay Secretly Minimize the Rank of Your Neural Network
 3. E. Zangrando et al.,2024 Neural Rank Collapse: Weight Decay and Small Within-Class Variability Yield Low-Rank Bias 
 4. Lyu et al., 2024 Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking
 5. Yubin Qin et al., 2023 FACT: FFN-Attention Co-optimized Transformer Architecture with Eager Correlation Prediction

### Questions
How different is the minimum-norm solution from the solution obtained by FACT-RFM in practice? Are they generally equivalent, or are there theoretical or empirical cases the authors are aware of where they diverge?

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
4

### Summary
The authors propose a principled derivation of first order optimality conditions of neural network matrices at convergence. They show that this explains many of the same phenomena as neural feature ansatz.

### Strengths
The paper presents a rigorous and principled theoretical derivation for previously empirically observed (or conjectured) phenomena around neural feature ansatz.

They prove their key theorem which gives a nice intuition for why this occurs.

They provide solid experiments that support the theory.

They have an insightful limitations figure that shows how their theory only holds at convergence (as opposed to NFA)

### Weaknesses
There is a whole field of literature on identifiability and representation learning (see here for a start: https://link.springer.com/article/10.1007/s10463-023-00884-4) that studies the representations learned by neural networks at convergence. I expect the authors to add those references and discuss the relation of their work to that field.

Fig2 caption: define what it means to train until 'interpolation'

lines 198-200: unclear argument, the following sentences are just algebraic manipulations

Many of the experimental setups read like: 'we found it works for these hyperparameters' – it would be great to see some ablation studies, e.g., how do think still/no longer work with different learning rate, dimensionality, weight decay etc.

236: 'akin to feature leaning in neural networks' – I think this is a key overstatement (also in the NFA literature), all that is happening here is learned linear filtering on the inputs. This is very different from *nonlinear* feature learning in neural networks. Please tone this down.

262 please write out FACT again below this equation, I think that would show how this is almost tautologically true, that FACT-RFM performs better because it just looks like a gradient step of W w.r.t. to the loss

### Questions
Please take all questions below as call for action to make your paper more clear:

Why is eNFA called equivariant? Please explain in paper.

On the first page, NFA is the gradient with respect to inputs, on page 3 it is w.r.t. to hidden activations. Which of the two is it?

206: how sensitive is this to different values of weight decay?

207: what correlation are you measuring? since NFA and AGOP are only proportional I would expect Spearman. However, for FACT, it is an equality so I would expect Pearson or even variance explained; Also, why look at sqrt? This can be avoided with rank correlation.

fig5: how exactly does NFA or FACT explain grokking? or is this just an observation about behavior of the system? fig 8 shows that FACT only holds at convergence… what do we learn here?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper derives an alternative to the Neural Feature Ansatz by investigating first-order optimality conditions of a loss function with L2 weight penalty. The alternative, FACT, performs similarly to the NFA in applications, but correlates better to solutions found in neural networks. The paper further shows a synthetic dataset for which NFA predictions fail completely while FACT predictions remain accurate. Overall these results show that FACT is a useful and theoretically principled alternative to NFA.

### Strengths
Derives an expression, FACT from first order conditions which is reminiscent of the NFA, thereby providing a theoretical basis for the method (the justification of NFA by contrast being empirical). This is a great contribution.

The experiments show that FACT behaves similarly to NFA in applications to tabular data, sparse parity, and grokking. These results show that FACT is approximately as useful as NFA.

The result showing that NFA predictions can break down while FACT predictions remain is particularly nice in highlighting the contribution of the work.

Overall the paper is clear and easy to read, with well made figures.

### Weaknesses
While the proposed FACT method may have a theoretical justification, it does not clearly improve on the NFA method in the applications considered here. The paper could be strengthened by including a case where FACT outperforms NFA on more than a synthetic task. But I do not view this as an essential addition to the paper.

### Questions
Are there any intuitions for why NFA correlates well with weights throughout learning but FACT does not?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces the Features at Convergence Theorem (FACT), a theoretical alternative to the Neural Feature Ansatz, the idea that a layer’s feature covariance correlates with the average gradient outer product. FACT is derived from first-order optimality conditions that must hold at convergence, and thus is more theoretically grounded. Empirical tests show that FACT reproduces key learning behaviours — including grokking, phase transitions, and sparse parity learning — similar to NFA. FACT matches NFA’s predictions in most settings but remains valid where NFA fails.

### Strengths
- Clarity and simplicity. The core identity (FACT) follows directly from first‑order optimality with L2 weight decay, relating a layer’s feature Gram $W^\top W$ to an empirical average of loss‑gradients times activations. It’s architecture‑agnostic under the mild requirement that the model depends on a weight matrix multiplication (Equation 2.1 and Theorem 3.1). This gives a simple target that any trained model with weight decay must satisfy.
- Forward and backward views. Besides the forward statement for $W^T W$, they derive a backward counterpart for $WW^T$ (bFACT), giving traction on both right and left singular spaces of a layer (Remark 3.4).
- Empirical alignment at convergence across realistic vision setups. In 5‑layer ReLU MLPs on MNIST/CIFAR‑10, Pearson correlation between the two sides of FACT and the learned $W^T W$ rises to $\sim1$ at convergence (Figure 2), often exceeding NFA and eNFA. This makes FACT a strong descriptor of terminal‑phase features in practice.

### Weaknesses
- FACT's correlation with $W^T W$ is typically low during most of training and spikes only near interpolation (Figure 8). This makes FACT relatively uninformative about how features emerge, arguably the purpose of studying these quantities.
- Mathematical novelty is modest. The main theorem is essentially the stationarity condition written in feature‑centric form.
Applies to the terminal phase only.
Kernel analysis is clarifying but incremental. The inner‑product kernel derivations neatly align the two updates $\tau$ vs $k'$, but the algebra is standard once you adopt the kernelised view. It explains why NFA often works without yielding new predictive dynamics (Section 5).
- Breadth of architectures/losses is limited. Core validations focus on MLPs with MSE; connections to cross‑entropy (with weight decay), convs/transformers, and decoupled weight decay (AdamW) are not systematically explored. eNFA is included but comparisons remain relatively narrow (Sections 3-4).

### Questions
1. FACT’s correlation with (W^\top W) is low through most of training and spikes only near interpolation (Figure 8). What intermediate quantity (e.g. a the $\sqrt{\text{FACT}\cdot\text{FACT}^T} $ you note) best tracks feature emergence earlier, and can you motivate it theoretically rather than empirically?
2.    Since FACT follows from first-order optimality, can you predict when alignment will occur, not just that it must at convergence? Any falsifiable prediction for the timing of grokking/phase transitions beyond endpoint agreement?
3. Given that Theorem 3.1 is a stationarity rewrite, what’s the new predictive leverage beyond endpoint consistency, for example,  conditions under which FACT will disagree with NFA before convergence in natural (non-adversarial) settings?

### Soundness
3

### Presentation
3

### Contribution
2
