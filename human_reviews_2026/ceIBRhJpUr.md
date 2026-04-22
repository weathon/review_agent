# $\mathbf{Li_2}$: A Framework on Dynamics of Feature Emergence and Delayed Generalization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
While the phenomenon of grokking, i.e., delayed generalization, has been studied extensively, it remains an open problem whether there is a mathematical framework that characterizes what kind of features will emerge, how and in which conditions it happens, and is still closely connected with the gradient dynamics of the training, for complex structured inputs. We propose a novel framework, named \ours{}, that captures three key stages for the grokking behavior of 2-layer nonlinear networks: (I) \underline{\textbf{L}}azy learning, (II) \underline{\textbf{i}}ndependent feature learning and (III) \underline{\textbf{i}}nteractive feature learning. At the lazy learning stage, top layer overfits to random hidden representation and the model appears to memorize. During lazy learning, the \emph{backpropagated gradient} $G_F$ from the top layer carries information about the target label, with a specific structure that enables each hidden node to learn their representation \emph{independently}. Interestingly, the independent dynamics follows exactly the \emph{gradient ascent} of an energy function $\mathcal{E}$, and its local maxima are precisely the emerging features. We study whether these local-optima induced features are generalizable, their representation power, and how they change on sample size, in group arithmetic tasks. When hidden nodes start to interact in the later stage of learning, we provably show how $G_F$ changes to focus on missing features that need to be learned. Our study sheds lights on roles played by key hyperparameters such as weight decay, learning rate and sample sizes in grokking, leads to provable scaling laws of feature emergence, memorization and generalization, and reveals the underlying cause why recent optimizers such as Muon can be effective, from the first principles of gradient dynamics. Our analysis can be extended to multi-layer architectures.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a three-stage framework to theoretically study grokking using two-layer neural network with MSE loss and quadratic activation functions. It find connections between stage II and the optimization of an energy function. It rigorously analyzes the role of learning rate, weight decay, sample size, and proposes a scaling law of generalization/memorization boundary. It also theoretically analyzes the benefits of Muon optimizer in the feature learning regime.

### Strengths
1. The first work to rigorously analyze the role of learning rate, weight decay, and width on grokking dynamics.
2. Interesting and novel analysis of the interactive feature learning regime.
3. First theoretical analysis of Muon in the feature learning regime.
4. The theoretical results of two types of memorization is intriguing.

### Weaknesses
1. MSE loss is uncommon in the training of deep learning models, though it's fine for the ease of theoretical analysis.
2. The theory needs a nonzero weight decay to provably show the three phases, while in practice, weight decay is unnecessary for grokking to occur.
3. Several assumptions need to be justified.

### Questions
1. In stage I, why the activations F are mostly unchanged? Any empirical observations to support such assumption? I guess it needs assumptions on the learning rate, weight decay, and initialization scale of V.
2. In Lemma 1, why can we assume W always follow normal distribution at each step? Besides that, W is assumed to follow normal, but in the proof (line 714-716), you assume w_i follow N(0, I), which is stronger.
3. In Lemma 1, why can we assume $<x_i, x_{i'}> = \rho$ ? Does this assumption hold in any synthetic tasks that show grokking?
4. In Theorem 5, what are focused and spreading memorization? What's the difference between memorization and overfitting?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work proposes a 3-stage framework to understand the emergence of feature learning in large-width two-layer neural networks, named **Li$_{2}$**. This consists of:

1. Lazy regime: first-layer weights are effectively random. The second layer fits the data using random features. The back-prop term to the hidden layer, $GF$, carries little usable structure and, without weight decay, can vanish at the lazy fixed point (ridge solution).
2. Independent: With weight decay \(\eta>0\), \(GF\) acquires target structure. Under some assumptions and large width \(K\), it simplifies to
$$
GF = \frac{\eta}{(Kc_1+\eta)(nc_2+\eta)}\tilde Y\tilde Y^{\top}F,
$$
so each neuron follows gradient **ascent** on the single-neuron energy
$$
E(w)=\frac{1}{2}\big\|\tilde Y^{\top}\sigma(Xw)\big\|_2^2,
$$
i.e., neurons learn useful directions independently.
3. Interactive: As several features are learned, neuron–neuron interactions reshape $GF$: similar features repel, and the signal is steered toward missing directions. 

This is discussed explicitly in a modular arithmetic task and quadratic activation $\sigma(x)=x^{2}$. Using group-theoretical tools, it is shwon that local maxima of $E$ align with group irreducible representations which for this task coincide with Fourier modes, yielding closed-form descriptions of learned features and their attained energies $E^\star$. 

Finally, an extension to the multi-layer case is also discussed.

### Strengths
The paper is clearly written and provides a fairly generic picture of the mechanism for feature learning in two-layer neural networks, which is task independent and rely only on an investigation of the gradients.

### Weaknesses
The results for each section rely on different assumptions, which makes **Li$_{2}$** look more like different patches of results rather than a unified picture. Some of the observations appearing in this work have also appeared in other works in the feature learning literature, and more throughout comparison is lacking. I expand on these two points below.

### Questions
1. Different results in this work appear to rely on different assumptions about $n,d,K,M$, and it is not immediately clear whether these are mutually consistent. For instance, Lemma 1 assumes that $K$ is sufficiently large and that $x_i^\top x_j=\rho$, which is only possible when $n<d$ unless the $x_i$ are degenerate. By contrast, Theorem 2 and Corollary 1 take $d=2M$ and $n=M^2$, which is consistent with Lemma 1 only for $M<2$. Similarly, Theorem 4 requires $n\gtrsim d_k^{,2} M \log(\delta/\delta)$ via Matrix Bernstein. Overall, I found it confusing to determine whether the regimes considered for the different phases of Li$_2$ hold simultaneously.

2. Feature learning for two-layer neural networks has been studied extensively in recent years, with several works arriving at a picture closely related to **$Li$_2$**. For example:

- One line of work systematically analyzes one-pass SGD dynamics in teacher–student settings (a 2LNN learning another, not necessarily identical, 2LNN). These studies show that the dynamics decompose into plateau / saddle-to-saddle phases: first-layer weights move within a neighborhood of zero (the *mediocrity phase*), then individual neurons correlate with target directions independently before finally coalescing (the *specialization phase*); see Saad and Solla (1995), Goldt et al. (2019), and Arnaboldi et al. (2023). While these works differ technically (finite-width networks, one-pass SGD), the overall mechanism is closely related, with a key difference being the absence of an overfitting phase under one-pass SGD.

- Another closely related line of work considers feature learning during the first few steps of GD with aggressive learning rates (Damian, 2022; Ba et al., 2022; Dandi et al., 2024). The energy in Eq. (8), often called the ``Correlation loss,’’ is a common approximation in this literature because it yields exact weak-recovery thresholds for the initial gradient steps. A notable observation is that, after a single aggressive step, the gradient can be asymptotically characterized—depending on sample complexity—by a rank-one matrix that correlates with the target, enabling the network to express nonlinear components with limited data. Is this the same mechanism as in Theorem 3?

- More recently, Montanari and Urbani (2025) studied a related teacher–student setting for full-batch GD on a wide 2LNN learning a single neuron, and identified three timescales: a lazy timescale, a generalization timescale, and an overfitting timescale (motivating early stopping). (There is no specialization here because the target has a single neuron.) In Li$_2$, overfitting is associated with the lazy phase. How can this be reconciled with Montanari and Urbani (2025)? How should we understand the benefits of early stopping within the Li$_2$ framework?

**References**

- (Saad and Solla 1995) Dynamics of On-Line Gradient Descent Learning for Multilayer Neural Networks. NeurIPS 1995.
- (Goldt et al. 2019) Dynamics of stochastic gradient descent for two-layer neural networks in the teacher-student setup. NeurIPS 2019
- (Arnaboldi et al. 2023) From high-dimensional & mean-field dynamics to dimensionless ODEs: A unifying approach to SGD in two-layers networks. COLT 2023
- (Damian 2022) Neural Networks can Learn Representations with Gradient Descent. COLT 2022.
- (Ba et al. 2022) High-dimensional Asymptotics of Feature Learning: How One Gradient Step Improves the Representation. NeurIPS 2022.
- (Dandi et al. 2024) How Two-Layer Neural Networks Learn, One (Giant) Step at a Time. JMLR 2024
- (Montanari and Urbani 2025) Dynamical Decoupling of Generalization and Overfitting in Large Two-Layer Networks. arXiv 2025

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper analyzes grokking dynamics in the presence of regularization. They identify three phases in the learning process - lazy learning regime and independent and interactive feature learning regime, and prove various theorems which govern different aspects of the training dynamics.

### Strengths
1) The theorems proved are mathematically rigorous, with detailed proofs in the appendices. 
2) The scaling law analysis in Section 5.4 will be useful since it gives a first-principles derivation of the scaling phenomenon.
3) Extensions to modern optimizers and deeper networks were adequately discussed.

### Weaknesses
1) My main concern is that the key observation (that there is a lazy and rich learning regime) was already reported in [1], who study this in the context of polynomial regression. While the current paper offers mathematically rigorous proofs, the distinction with [1] needs to be more elaborately discussed in the main text.
2) The mathematical framework presented displays the three stages, but fails to offer more insight on what drives these transitions or when do these transitions. For example, based on lines 471-475, it seems like it's the (inverse) learning rate which sets the scale for when these transitions occur, but it would be great if this could be explained in more detail.
3) How does one relate the top-down modulation in Sec 6 to the continuous feature learning observed in [2]?
4) Discussion on limitations not found in main text.


[1] Tanishq Kumar, Blake Bordelon, Samuel J. Gershman, and Cengiz Pehlevan. Grokking as the transition from lazy to rich training dynamics, 2024. 
[2] Gromov, A. (2023). Grokking modular arithmetic. arXiv preprint arXiv:2301.02679.

### Questions
1) Line 105-106 : I suggest modifying "grokking mostly happens ... regularization" to "grokking is accelerated ... regularization"
2) Why are the axes cut off till epoch 300 in Fig 2 last column?
3) Since the discussion on Thm 3 involves optimization in the complex domain, were any follow-up experiments done with complex weights?
4) The existence of maxima for the ascent functions is interesting (Thms 4-5). What can be said about the network's ability to find these maxima?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes **Li²**, a mathematical framework to explain grokking (delayed generalization) in two-layer nonlinear networks via the structure of the back-propagated gradient matrix (G_F). Training is decomposed into three phases: **(I) Lazy learning**, where (G_F) is effectively random and the top layer overfits random hidden representations; **(II) Independent feature learning**, where each hidden unit learns independently because each column of (G_F) depends only on its own activation—weight decay injects label signal, the dynamics become exact gradient ascent on an energy (E), and the local maxima of (E) coincide with the emergent features; and **(III) Interactive feature learning**, where hidden units begin to interact and (G_F) reorients toward missing features that must be acquired for generalization. On group-arithmetic tasks, the authors analyze when these energy-induced features generalize, their representational power, and how they vary with sample size. The framework yields **provable scaling laws** for memorization and generalization as functions of weight decay, learning rate, and data size, and offers a first-principles explanation for the effectiveness of optimizers such as **Muon**. The analysis is argued to extend to deeper architectures.

### Strengths
1. The paper is overall solid and offers a detailed study of the dynamics of two-layer neural networks.
2. It provides experiments that validate the theoretical results.

### Weaknesses
1. The writing feels very rushed: many symbols are undefined or unexplained, making the paper hard to follow and overly dense.
2. The theoretical setup is quite restricted; for example, a projection function is deliberately designed so that the hidden layer receives gradients that are random noise.
3. It is unclear whether the group (arithmetic) example pertains only to Stage II or also extends to Stage III.
4. The theoretical analysis of Muon appears disconnected from the main body of the paper, and the setup and assumptions of Theorem 8 are entirely unclear.
5. The relationships among the three stages are not well articulated; the analysis reads like heuristic case-by-case treatment rather than a genuinely unified three-stage dynamics analysis.
6. The abstract mentions scaling laws, but it is not evident where in the paper this is actually developed.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
