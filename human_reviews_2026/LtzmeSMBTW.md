# The Curious Case of In-Training Compression of State Space Models

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
State Space Models (SSMs), developed to tackle long sequence modeling tasks efficiently, offer both parallelizable training and fast inference. At their core are recurrent dynamical systems that maintain a hidden state, with update costs scaling with the state dimension. A key design challenge is striking the right balance between maximizing expressivity and limiting this computational burden. Control theory, and more specifically Hankel singular value analysis, provides a potent framework for the measure of energy for each state, as well as the balanced truncation of the original system down to a smaller representation with performance guarantees. Leveraging the eigenvalue stability properties of Hankel matrices, we apply this lens to SSMs $\textit{during training}$, where only dimensions of high influence are identified and preserved. Our approach, CompreSSM, applies to Linear Time-Invariant SSMs such as Linear Recurrent Units, but is also extendable to selective models. Experiments show that in-training reduction significantly accelerates optimization while preserving expressivity, with compressed models retaining task-critical structure lost by models trained directly at smaller dimension. In other words, SSMs that begin large and shrink during training achieve computational efficiency while maintaining higher performance. Project code is available at https://github.com/camail-official/compressm.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an in-training compression technique of state-space models. It is based on application of the balanced truncation method adaptively during training. Empirical results on the long-range arena tasks show that the proposed method improves the model in certain cases.

### Strengths
* Accelerating the training speed of SSMs is an important task, and the proposed method shows the benefits of the method over directly training a smaller model from scratch.
* The method is grounded in rigorous analysis of LTI systems and the balanced truncation method from the ROM literature, which has been shown reliability.

### Weaknesses
* The idea proposed in the paper is very incremental. It looks like a naive application of the balanced truncation method at the training stage. The Weyl-type perturbation analysis is not new either. The proposed method can be totally viewed as a multi-stage fine-tuning process, where post-training compression and fine-tuning are applied alternately.
* The paper proposes an in-training compression method, but there is only one experiment that shows the training-time benefit in Figure 3b on a specific dataset. More importantly, reducing the latent dimension of SSMs does not significantly reduce the training speed (at least on LRA). That said, Figure 3b is visually misleading because the left end starts from somewhere between 0.55 and 0.6 instead of 0. More benefits would come from test time, especially if you plan to do it recurrently, but this is not the advantage of the proposed method over post-training compression.
* The accuracy of the proposed method is comparison with the baseline is also not compelling enough. Among the LRA tasks, only CIFAR and ListOps have shown some significant benefits. Also note that for ListOps, it is known that a small state dimension is totally capable of achieving SOTA performance. (For S4D, the SOTA performance comes from $n = 4$ and works well even when $n = 1$.)
* The paper would benefit from a discussion of the post-compression weights, and how it lands in the training loss landscape. This is a more interesting problem than the Weyl-type perturbation analysis. Note that the model is optimized via gradient descent; although compressing the system does not change its impulse response a lot, the parameters can be totally changed. This introduces more subtlety in CompreSSM.

### Questions
1. You choose a fixed $\tau$ throughout. If I am given a task, how would I go for a $\tau$ that achieves a time--accuracy balance that I expect? I think this is a crucial question, and especially given that you do not want to do any additional hyperparameter tuning (otherwise, why not just setting $\tau = 0$ and training only one model?).
2. Relatedly, have you experimented with having a scheduling for $\tau$, instead of fixing it throughout the training stage?
3. The "Baseline" in Table 1 is not well-described in the section. How did you train the "Baseline" model? Is it trained by fixing $n$ to be the rounded average "State dim" for all LTI systems in an SSM and train it from scratch?
4. Controllability and observability are not generally assumed in SSMs. How would you reconcile the fact that these two assumptions may not generally hold?
5. Hankel singular values are very sensitive to initialization and parameterization. For example, this work (https://openreview.net/forum?id=RZwtbg3qYD) shows that without careful initialization, you will start with fast-decaying singular values, and without careful parameterization, LTI systems are driven to low-degree ones during training. These are mainly due to the diagonal elements of $\mathbf{A}$ that are far apart from the imaginary axis. Is it possible to use that as a criterion for "when to truncate"?
6. Have you tried other ROM techniques other than balanced truncation? For example, those based on $\mathcal{H}_2$ optimization, moment-matching, or rational approximation?
7. You mentioned that the work can be adapted to time-variance (or selective) case; however, for selective SSMs, e.g., Mamba, the parameterization is completely different, and $\mathbf{B}$, $\mathbf{C}$, and even $\Delta$ are computed via more sophisticated mechanisms. This would make compression harder. Do you have a simple approach to that? If not, then better avoid overclaiming.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose a complexity reduction strategy to accelerate SSM training. The idea is to remove non-essential dimensions of the state space by looking at the eigenvalues of the Hankel matrix during training. The pruned models perform better than a baseline where the dimensionality of the state space is fixed from the start.

### Strengths
- The idea reminds me of successful techniques for imposing sparsity when learning NNs.
- SSMs have become popular as surrogates of transformers. The possibility of compressing them during training in a principled way may help find a trade-off between computational costs and flexibility.

### Weaknesses
- While reducing the complexity of the trained model, the approach does not seem to imply complexity gains during the training phase.
- The analysis focuses on linear and time-invariant models, which are often outperformed by transformers on several tasks. It is unclear which parts of the idea can be applied to models with input-dependent matrices, such as Mamba-2.
- The method involves several hyperparameters, which may be expensive to tune.
- Experiments do not include alternative regularisation approaches.

### Questions
- Is this the first time Hankel-based regularisation techniques are applied to SSMs during training? What is the difference between the proposed method and more straightforward strategies, such as a simple $L_2$ regularisation of the matrices?
- How is the *normalised training time* mentioned in the figures defined?
- Would it make sense to include a simple alternative regularisation method as a second baseline?
- Is the truncation irreversible? Did you observe suboptimality issues associated with the greedy nature of the proposed method?
- Training complexity gains can be achieved if the reduction is irreversible and happens early enough. How is the latter compatible with the observation that there should be *"enough time in between successive reduction steps for the model to recover from pruning"*? How is the trade-off handled in practice?

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
3

### Summary
This paper proposes a method for compressing State Space Models (SSMs) during training rather than after training. The key innovation is using balanced truncation from control theory to identify and remove unimportant state dimensions while the model is being trained. The authors leverage Hankel Singular Values (HSVs) to measure the importance of each state dimension. Based on this information, they truncate low-importance dimensions in training, achieving computational speedups while maintaining or improving model performance compared to models trained directly at smaller dimensions.

### Strengths
- Novel application of control theory to SSM compression during training, with good theoretical grounding using Hankel Singular Values and Weyl's theorem to justify the approach
- The method is architecture-agnostic and works with different SSM variants
- Empirical validation showing that dominant HSVs are rank-preserving during training
- Achieves both training speedup over full dimension and better performance than smaller dimensions
- Well-written paper with clear mathematical exposition

### Weaknesses
1. Only tested on LRU architecture, not on other SSMs like Mamba, S4, etc.
2. No comparison with other compression techniques like knowledge distillation, quantization, or pruning
3. Method requires correlation between state dimension and model performance (acknowledged in the paper)
4. Needs sufficient training steps between reduction steps for model recovery (like pruning and QAT)
5. Missing ablation studies on key hyperparameters (reduction frequency, threshold values)
6. No analysis of what information is preserved/lost during truncation

### Questions
1. Step 5 of the algorithm (Section 3.1): What fraction values work well in practice? The paper doesn't provide specific guidance
2. Can you provide examples where there's no clear correlation between state dimension and model performance? I cannot think of such applications
3. Are you evaluating Sequential MNIST (SMNIST)? The paper mentions MNIST results but doesn't clarify if it's SMNIST
4. How does the computational cost of computing Gramians scale with state dimension?
5. Have you considered inverting the x-axis in Figure 3 to show "training time speedup" instead of "normalized training time"? This would make the Pareto front point to the top-right, which is more intuitive

### Soundness
3

### Presentation
3

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
The paper introduces CompreSSM, a generalized in-training compression framework for state space models to accelerate training while maintaining accuracy. The method uses Hankel singular value analysis to identify and prune low-energy state dimensions during training (inspired by Model Order Reduction and balanced truncation from control theory). The authors show that the rank ordering of Hankel singular values remains relatively stable throughout training. The authors use six datasets to provide empirical evidence for its ability to maintain or even improve accuracy as compared to a less efficient baseline training.

### Strengths
The authors provide good theoretical grounding for CompreSSM, bringing concepts from Hankel singular value analysis and truncation; they address a very relevant topic with a well-outlined algorithm.

Authors provide reasonable justification of the in-training truncation and its validity. Analysis of Hankel singular value trajectories supports their analysis.

The authors provide convincing results across six different datasets and five random seeds, while describing the experimental setup in great detail facilitating reproducibility.

### Weaknesses
If I understand correctly, empirical evidence is centered only around LRU, without other Deep SSMs or so-called "selective" (input dependent) architectures. 

The authors should provide additional analysis or detail on the computational complexity of the Algorithm (3.1) and overhead/clock time of each reduction step, more details into the recovery of the model after each reduction step (mentioned in 4.2 as part of the results, but not analyzed as far as I could tell) and how this translated into end-to-end training cost and efficiency on GPUs versus the vanilla method.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
3
