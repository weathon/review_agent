# Scaling Direct Feedback Learning with Jacobian Alignment Guarantees

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8, 2

## Abstract
Deep neural networks rely on backpropagation (BP) for optimization, but its strictly sequential backward pass hinders parallelism and scalability. Direct Feedback Alignment (DFA) has been proposed as a promising approach for parallel learning of deep neural networks, relying on fixed random projections to enable layer-wise parallel updates, but fails on deep convolutional networks, and performs poorly on modern transformer architectures. We introduce GrAPE (Gradient-Aligned Projected Error), a hybrid feedback-alignment method that (i) estimates rank-1 Jacobians via forward-mode JVPs and (ii) aligns each layer’s feedback matrix by minimizing a local cosine-alignment loss. To curb drift in very deep models, GrAPE performs infrequent BP anchor steps on a single mini-batch, preserving mostly parallel updates. We show that the forward-gradient estimator has strictly positive expected cosine with the true Jacobian. We relate this estimator-level guarantee to a standard stochastic-approximation result under a positive expected-cosine condition on the update direction, providing theoretical support for GrAPE’s alignment objective. Empirically, GrAPE consistently outperforms prior alternatives to BP, enabling the training of modern architectures, closing a large fraction of the gap to BP while retaining layer-parallel updates for the vast majority of steps.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces GrAPE, a hybrid direct-feedback method that learns layerwise feedback matrices using forward-mode Jacobian–vector products (JVPs).

For each layer~$\ell$, it forms a rank-1 Jacobian estimate $\hat J_\ell = (J_\ell p)\,p^\top$ from a random perturbation $p$, and then aligns a trainable feedback matrix $B_\ell$ to $\hat J_\ell$ via a cosine-similarity objective.

The network parameters are updated in a DFA-style manner using the output error routed through $B_\ell$, with infrequent backpropagation calibration steps to stabilize training in deep/wide models.

The analysis establishes positive expected alignment (with variance decreasing by batching) and invokes Zoutendijk-type convergence to stationarity under Wolfe/strong-Wolfe step-size assumptions and descent directions.

Empirically, the method scales to modern CNNs and Transformers, narrows the gap to BP relative to DFA, and leaves the feedback-refinement step layer-local and parallelizable in principle.

### Strengths
Clear hybrid design: Learns layerwise feedbacks via forward-mode JVPs and performs DFA-style updates, combining DFA's simplicity with forward-gradient signal quality.

Theoretical support for feedback learning: Establishes positive expected alignment between the rank-1 Jacobian estimate and the true Jacobian, with variance reduction via batching, justifying alignment of $B_\ell$ to $\hat J_\ell$ with Zoutendijk-style arguments.

Pragmatic stability via sparse BP: Proposes a novel occasional backpropagation calibration that improves stability/quality in deep or wide networks while retaining layer-parallel updates for the vast majority of steps.

Parallelizable in principle: The feedback-refinement step is layer-local, enabling parallel execution across layers (implementation permitting).

Strong empirical results: On CNNs and Transformers, GrAPE narrows the gap to BP and consistently outperforms standard DFA under comparable settings.

### Weaknesses
Convergence is heuristic rather than theoretical: The paper invokes Zoutendijk under Wolfe/Goldstein step-size assumptions and descent directions, but the actual update is stochastic and the per-layer cosine is only \emph{positive in expectation}, with the bound scaling like $1/n_\ell$, which can be very small in wide layers.
Eq.~(5) is labeled a ``\emph{sufficient local condition},'' yet there is no theorem showing local convergence. The claim should be softened or supported by a formal (e.g., high-probability) descent lemma.

Compounding noises: 
(i) The rank-1 JVP estimator $\hat J_\ell=(J_\ell p)p^\top$ aligns with $J_\ell$ only in expectation and can yield negative per-batch cosine unless perturbation count/batch size is large;
(ii) the feedback $B_\ell$ is then aligned to $\hat J_\ell$, compounding estimation error.
The paper would be substantially improved by providing lower bounds on $\cos\big(B_\ell, J_\ell\big)$.

Parallelism not demonstrated:
Although feedback refinement is parallelizable in principle, the implementation serializes these updates and reports no wall-clock speedups. Concrete parallel benchmarks (speed/memory vs.\ BP) would be appreciated.

BP calibration is under-discussed:
While the paper ablates the calibration period $T$, the role of BP calibration remains under-theorized and under-explained.
Since calibration is pivotal to performance (often $T=1$), more discussion about the role of BP calibration and clarification o;n its theoretical role would be appreciated.

### Questions
Formal convergence theorem:
Currently, Zoutendijk and the Goldstein/strong Wolfe conditions are used heuristically to motivate the alignment loss and feedback refinement, but there is no formal convergence guarantee for the actual stochastic update.
Since the expected positive cosine is derived between $J_\ell$ and $\hat J_\ell$, while the algorithm maximizes cosine between $B_\ell$ and $\hat J_\ell$, could you please provide a \emph{formal theorem} that yields convergence to stationarity for GrAPE.
Concretely, specify conditions under which the full update direction is a descent direction with high probability (e.g., $\nabla f(\theta_k)^\top \Delta\theta_k < 0$), and state the resulting convergence claim (stationarity) and how it depends on width $n_\ell$, perturbation/batch count $B$, etc.

Alignment guarantees between $B_\ell$ and $J_\ell$
Is it possible to provide a lower bound on the alignment between the learned feedback and the true Jacobian,\cos(B_\ell,J_\ell)?
As supporting evidence, could you also please add diagnostics:
  (i) an empirical descent test: the frequency of $\nabla f(\theta_k)^\top \Delta\theta_k < 0$ under GrAPE vs.\ DFA; and
  (ii) alignment to the true Jacobian: plots of $\cos(B_\ell,J_\ell)$ over training.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors address the problem of computing an efficient descent direction for training deep neural networks. They propose GrAPE, an adaptation of direct feedback alignment techniques in which the feedback matrices are updated to improve the alignment between the computed descent direction and the true gradient. The practical performance of the method is compared against backpropagation, FA, DFA, DRTP, and PEPITA across several network architectures.

### Strengths
* **S1**: I find the paper globally clear and well written, except in one point (see **W2**).
* **S2**: The paper leverages Zoutendijk’s theorem to propose a modified version of feedback alignment that yields better descent directions.
* **S3**: The method is evaluated on a wide range of architecture and achieves good performance, closing the gap with backpropagation.

### Weaknesses
* **W1**: The code for the experiments is not provided, hindering reproducibility.
* **W2**: By reading 3.4, I understand that the calibration step happens once every $T$ epoch, and thus at most once in one epoch. However, the way Algorithm 1 is presented suggests that when $\mathrm{epoch}\, \mathrm{mod}\, T = 0$, a calibration step is performed for each minibatch. Could the authors clarify this point?
* **W3**: The JVP estimator has variance proportional to layers' dimension. This limits the interest of GrAPE compared to BP for wide neural networks because of the potential need of frequent BP calibration.

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces GraPE (Gradient-Aligned Projected Error), a hybrid feedback-alignment method that enables layer-parallel training of deep neural networks. The GraPE algorithm builds on the ideas of (direct) feedback alignment, where the error is back-propagated to the layers of a neural network via random feadback connections. The core innovation of GraPE is to use relatively cheap, forward-mode Jacobian-vector products to estimate rank-1 Jacobians, which are then used to align each layer's fixed random feedback matrix via a local cosine-alignment loss. In very deep models, GraPE also performs infrequent backpropagation (BP) steps on single mini-batches. The authors provide theoretical guarantees of positive expected alignment and convergence using Zoutendijk-style arguments. Empirically, GraPE significantly outperforms vanilla FA or DFA and similar BP alternatives and, with occasional BP calibration, closes most of the performance gap with standard BP on models like VGG-16, ResNet-20/56, and Transformers, while retaining layer-parallel updates for the vast majority of training steps.

### Strengths
- This work achieves a major step forward of alternative learning algorithms to BP: making feedback-alignment training viable for modern, deep architectures (CNNs, ResNets, Transformers), a problem where prior methods (DFA, FA) have consistently failed. 
- The key idea of using forward-gradient estimates to align fixed feedback matrices is novel and neat. It creatively combines ideas from two previously separate lines of work (random feedback and forward-mode differentiation) into a single algorithm.
- The paper is well-written: the arguments are clear, and the authors performed extensive tests of their algorithm in a reproducible experimental setup using the BioTorch library.

### Weaknesses
- Perplexingly, the authors do not address the most obvious benefit of their algorithm: the wall clock speed-up due to parallelisation. There is a tentative FLOPs analysis  in appendix XC, but the actual runtime benefit remains unclear. The authors seem to explain this gap in the paper by the standard libraries they use, but leaving this test out substantially weakens the point of the paper.

### Questions
- Could the authors provide any preliminary estimates or a more detailed discussion of the expected wall-clock time compared to standard BP, even with a serialised implementation? I would gladly increase my score if this issue is addressed.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces GrAPE (Gradient-Aligned Projected Error), a learning algorithm designed to scale Direct Feedback Alignment (DFA) to modern deep architectures such as CNNs and Transformers. It uses the follwing udeas:
- Forward-mode Jacobian estimation: rank-1 Jacobian approximations obtained via forward-mode automatic differentiation ) are used to align each layer’s feedback matrix through a local cosine-similarity loss.
- Sparse backpropagation calibration: occasional full BP updates on a single mini-batch serve to re-anchor weights and mitigate drift in very deep networks.

The paper provides a theoretical analysis showing that the expected cosine alignment between the estimated and true Jacobians is positive, and invokes Zoutendijk-style convergence arguments to guarantee convergence to stationary points. This is a very interesting developemn in the context of DFA. Empirically, GrAPE achieves substantial improvements over DFA, DRTP, and other BP-free baselines on CIFAR and WikiText benchmarks, and approaches full backpropagation performance on VGG, ResNet, and Transformer models when using one BP calibration per epoch.

Overall, the work provides both a conceptually clean and empirically convincing bridge between feedback alignment and optimization theory, showing that layer-parallel learning with forward-only signals can be made competitive with BP.

### Strengths
This is a strong and timely contribution to the literature on biologically inspired and parallel learning algorithms. The combination of feedback alignment, forward-mode differentiation, and provable convergence is original and well executed. The empirical scope (from LeNet to Transformers) convincingly demonstrates the generality of the approach. The work provides theoretical clarity missing from previous DFA/forward-gradient studies and could serve as a reference point for future algorithmic or hardware developments.
- Novel combination of feedback alignment and forward-mode gradient estimation, with clear theoretical grounding.
- First rigorous convergence argument for DFA-like methods based on Zoutendijk’s theorem.
- Empirical scalability: works on VGG, ResNet, and Transformer models, closing much of the gap to BP.
- Reproducibility and rigor: transparent experimental setup, use of BioTorch, and fair baselines.
- Balanced discussion of limitations and potential extensions.
= Provides a conceptual bridge between biologically plausible local learning rules and modern optimization.

### Weaknesses
i) Lack of demonstrated computational gains.
Although GrAPE is designed to enable layer-parallel updates, the current implementation remains fully serialized. No wall-clock or energy improvements are reported, and the FLOP analysis in Appendix C is only theoretical. It would be useful to include even small-scale timing or memory comparisons to assess the practical benefit.

ii) Dependence on frequent BP calibration.
Most deep-network experiments require a full backpropagation step every epoch (T = 1) to reach competitive performance. This limits how “BP-free” the method truly is, and raises questions about whether GrAPE’s main advantage lies in its theoretical framework or in genuine computational savings.

iii) Variance of the forward-mode estimator.
The rank-1 Jacobian approximation introduces variance that grows with layer width. While the issue is discussed, no practical mitigation (such as averaging multiple perturbations or using low-variance directions) is explored experimentally.

iv) Limited discussion of hardware relevance.
Given the recent success of hardware-accelerated feedback learning (e.g., optical DFA systems, Wang et al., 2024), it would strengthen the paper to discuss how GrAPE could map onto such platforms, or at least estimate realistic parallel speed-ups under non-ideal hardware constraints.

### Questions
i) There has been recent progress on optical implementations of DFA (e.g., Wang et al., 2024, arXiv:2409.12965) showing impressive throughput and energy efficiency at scale. It would be interesting to hear the authors’ thoughts on how GrAPE might relate to or benefit from such hardware developments. Could its adaptive feedback and occasional BP calibration be implemented efficiently in optical or neuromorphic systems, and would the JVP alignment step introduce additional cost or complexity?

ii) The paper provides a clear FLOP-based comparison between BP and GrAPE, but no timing or memory results. Could the authors share any empirical measurements, even on small networks, to verify that the theoretical analysis matches practical runtime or memory usage?

iii) How robust is GrAPE to the choice of calibration interval T? In particular, could an adaptive strategy that monitors cosine alignment offer a better trade-off between accuracy and calibration cost? Relatedly, have the authors tried using multiple perturbation directions per layer to reduce variance, and if so, how does that affect parallelism and runtime?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes GrAPE (Gradient-Aligned Projected Error), a feedback-based learning rule designed to relax the strict sequential dependence of backpropagation (BP). Building on Direct Feedback Alignment (DFA), GrAPE updates feedback matrices by minimizing a cosine loss that aligns them with rank-1 Jacobian estimates computed via forward-mode Jacobian–vector products (JVPs). This alignment is theoretically shown to produce descent directions in expectation. To mitigate drift, the method introduces BP calibration, a full BP step performed every $T$ epochs. Experiments on CNNs  and a Transformer  show that GrAPE outperforms DFA and approaches BP performance, especially when calibrated frequently (e.g., $T=1$).

### Strengths
The paper addresses an important question: can we train deep networks without the strictly sequential backward pass of BP? Its motivation, enabling layer-parallel or biologically plausible learning, is timely. Using JVPs to align feedback matrices offers an interesting bridge between forward-mode differentiation and DFA-style local updates, supported by a simple convergence argument. The empirical evaluation is broad, spanning CNNs and Transformers, and shows that learned feedback alignment can indeed reduce the performance gap with BP. The inclusion of a calibration mechanism acknowledges practical instability and provides a workable hybrid between local and global learning.

### Weaknesses
Despite its conceptual appeal, the practical advantages over BP remain unclear. The paper claims that GrAPE reduces computational cost, but forward-mode JVPs roughly double the forward-pass work, and the “BP calibration” step reintroduces full sequential backpropagation, often every epoch. No runtime, memory, or parallel-efficiency results are reported, so the efficiency claims remain speculative. Moreover, it is not clear when parallel updates become practically important if the memory requirements of the method remain the same as those of BP.

The contribution of alignment itself is also not novel: prior work such as Deep Learning Without Weight Transport [1] and the more recent SVD-Space Feedback Alignment [2] have already shown that feedback matrix alignment can restore BP-level performance. In particular, [2] explores a local cosine loss function for aligning the singular vectors of the feedback matrices with those of the forward path, yielding better gradient descent directions; this connection should be discussed further. GrAPE differs mainly in how it estimates the Jacobian (via JVPs), but the conceptual insight, that alignment matters, is already well established. Similarly, other local learning rules such as LLS [3] and Local Error Signals [4] achieve BP-like performance without relying on periodic BP corrections.

Finally, the theoretical section is terse: the derivation of Eq. (4) and the connection to Zoutendijk’s theorem are only sketched, leaving unclear the precise assumptions under which the convergence argument holds.

[1] Akrout, M., Wilson, C., Humphreys, P., Lillicrap, T. and Tweed, D.B., 2019. Deep learning without weight transport. Advances in neural information processing systems, 32.

[2] Roy, A. et al. (2025) ‘Unlocking SVD-Space for Feedback Aligned Local Training’. Available at: https://openreview.net/forum?id=8Agcic0csh.

[3] Apolinario, M.P., Roy, A. and Roy, K., 2025, February. Lls: local learning rule for deep neural networks inspired by neural activity synchronization. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (pp. 7807-7816). IEEE.

[4] Nøkland, A. and Eidnes, L.H., 2019, May. Training neural networks with local error signals. In International conference on machine learning (pp. 4839-4850). PMLR.

### Questions
• Could the authors provide a more explicit derivation of Eq. (4) and clarify the assumptions under which the expected positive cosine alignment holds?

•  How much wall-clock speed or memory reduction does GrAPE actually provide compared to BP or DFA, particularly when T=1?

•  Since the strongest results rely on frequent BP calibration, is the approach truly parallelizable in practice? How would these updates be scheduled on real hardware?

•  How does GrAPE differ technically and empirically from prior feedback-alignment methods such as Deep Learning Without Weight Transport [1] and SVD-Space Feedback Alignment [2]?

•  Could the authors discuss more explicitly how GrAPE compares to other local learning rules such as LLS [3] and Local Error Signals [4], which achieve near-BP performance without periodic BP corrections?

•  What are the specific deployment scenarios where GrAPE would offer meaningful advantages over BP, given that BP calibration reintroduces global synchronization

### Soundness
2

### Presentation
2

### Contribution
2
