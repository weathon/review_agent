# Trapped by simplicity: When Transformers fail to learn from noisy features

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Noise is ubiquitous in data used to train large language models, but it is not well understood whether these models are able to correctly generalize to inputs generated without noise. Here, we study noise-robust learning: are transformers trained on data with noisy features able to find a target function that correctly predicts labels for noiseless features? We show that transformers succeed at noise-robust learning for a selection of $k$-sparse parity and majority functions, compared to LSTMs which fail at this task for even modest feature noise. However, we find that transformers typically fail at noise-robust learning of random $k$-juntas, especially when the boolean sensitivity of the optimal solution is smaller than that of the target function. We argue that this failure is due to a combination of two factors: transformers' bias toward simpler functions, combined with an observation that the empirically optimal function for noise-robust learning has lower sensitivity than the target function. We test this hypothesis by exploiting transformers' simplicity bias to trap them in an incorrect solution, but show that transformers can escape this trap by training with an additional loss term penalizing high-sensitivity solutions. Overall, we find that transformers are particularly ineffective for learning boolean functions in the presence of feature noise.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the capability of transformers on noise-robust learning. Transformers are compared to LSTMs for this task. Main contributions of this paper include the following.
1. For sparse parity and majority functions at high rates of feature noise, transformers perform significantly better than LSTMs, and the latter fails even for low levels of feature noise.
2. They observed that transformers do not have satisfactory performance for the task of random k-juntas.
3. They proposed an explanation related to sensitivity of optimal solution and of target function.
4. Solution: implementing penalty for high-sensitivity solutions

### Strengths
1. The problem formulation is novel by adding a perspective on robustness. Noise-robust learning is valuable but yet relatively under explored.

2. The paper has sound theoretical analysis and the theoretical results provide significant insights. It deploys various mathematical tools efficiently, including Boolean analysis, information theory, and learning theory.

3. Experiments, although relatively small-scaled, has a clear target on the conjecture and provides valuable support

4. The paper is in general well-written and the logic flows smoothly

### Weaknesses
My main concern is on the applicability and scope of this study. The investigated problems (parity and junta) are binary-input problems with rigid mathematical structures. This fact provides simplicity for analysis, but at the same time they are restricted because real-world data and noises are much more complicated.

### Questions
My question is highly relevant with what I wrote in the weaknesses section: Does the theoretical insights obtained from your study have any implications on the robustness of transformers for more complicated tasks

### Soundness
3

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
2

### Summary
The paper studies noise-robust learning: whether models trained only on inputs corrupted by feature noise can still learn the underlying noiseless target function. Using Boolean tasks, the authors:

* Formalize the Bayes-optimal predictor under iid bit-flip noise as $f_N^*(x)=\mathrm{sign}(T_{1-2p} f(x))$, where $T_\rho$ is the standard noise operator. This links the problem to noisy-channel coding and bounds performance via conditional entropy. 
* Show empirically that transformers (encoder-only SANs) often succeed at noise-robust learning for sparse parity and odd-length sparse majority functions, while LSTMs generally fail even at modest noise rates. (Figure 1)
* Demonstrate that transformers typically fail on random k‑juntas, even when they achieve near‑optimal accuracy on the noisy validation distribution; failure correlates with the gap between the sensitivity of $f$ and that of the optimal noisy predictor $f_N^*$. (Figure 2)
* Argue the failure mechanism combines a simplicity bias in transformers toward low‑sensitivity functions and the empirical observation that $f_N^*$ tends to have lower sensitivity than $f$. (Conjecture 1)
* Construct a controlled "trap" function where $f$ and $f\_N^\*$ have similar optimal noisy accuracy but very different sensitivities; transformers converge to the simpler $f\_N^\*$. Adding a differentiable sensitivity penalty helps "escape" in a narrow range of penalty weights, but when $f\_N^\*$ is much better than $f\_N^\*$, the penalty does not help.

### Strengths
* Casting training-on-noisy-features as a noisy-channel problem, with $f_N^*$ characterized by the noise operator and performance tied to $H(Y|Z)$, gives a precise target for comparison. 
* Extensive hyperparameter sweeps and repeated trials (300 per condition) improve the reliability of the conclusions. 
* The use of total influence $I[f]$ to quantify simplicity connects to prior theory and cleanly explains why models can perform well on noisy validation yet fail on noiseless evaluation. 
* The “trap function” isolates simplicity bias from other confounders: with nearly equal noisy optimal errors for $f$ and $f_N^*$, any preference for $f_N^*$ reveals the inductive bias. Figure 8 show SANs drifting toward the trap unless regularized. 
* The discussion connects Boolean results to real LLM settings where training data are noisy/stochastic but downstream tasks are noise‑sensitive (e.g., arithmetic), cautioning that minimizing conditional entropy on noisy features might impede learning fine‑grained rules.

### Weaknesses
* Narrow noise model and data distribution. All inputs are uniformly random bitstrings with iid symmetric bit‑flip noise and memoryless corruption. Real text has structured distributions and correlated, non‑binary errors (insertions, deletions, paraphrases). The paper acknowledges this but leaves generality uncertain. 
* Task scope. Results hinge on Boolean functions; while parity/majority and k‑juntas are classic, evidence that the same mechanisms dominate in natural language or code remains indirect. No experiments on tokenized sequences, algorithmic datasets, or synthetic “text‑like” corruptions are provided. 
* The sensitivity penalty that helps in the trap requires a narrow range of $\lambda$ and does not generalize when $f_N^*$ dominates $f$. This makes it difficult to apply similar ideas in practice
* Conjecture 1—$I[f_N^*] \le I[f]$—is well‑motivated and tested on small n and random samples, but remains unproven. Many conclusions rely on it conceptually; a counterexample would undercut the narrative.

### Questions
1. How sensitive are the main findings to non‑iid or structured noise (e.g., burst errors, deletions/insertions, or noise correlated with specific positions)? Could the channel view extend to these cases and does the simplicity gap persist? 
2. What happens when inputs are non‑uniform (e.g., biased bit marginals or low‑entropy substructures), closer to natural text statistics? Does the prevalence of self‑predicting functions change under such distributions? 
3. Can the authors provide explicit parameter counts to the models? It is a bit troublesome to compute them from the dimensions provided in the appendix. Does model size/capacity have any effect on the biases observed?
4. Is there a practical diagnostic to detect, during training, when a model is converging to $f_N^*$ rather than $f$? The training‑trace figures suggest this might be possible.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work analyzes the ability of transformers and LSTMs in learning boolean functions, in the feature noise setting (with no label noise). The motivation for this lies in the prevalence of noise in LLM training data, and boolean functions serve as a sandbox for understanding this phenomenon. Through extensive simulations, it is shown that for the majority function and certain sparse parity functions, transformers succeed with learning the optimal function while LSTMs fail. On the other hand, for most other random boolean functions, both of these architectures fail to learn in the feature noise setting, although for the case of transformers, this can be mitigated with a modified regularized loss function.

### Strengths
- The motivation for the theoretical setup is clear (e.g. inspiration from modern day LLM training).
- Extensive experiments are provided for demonstrating when transformers can and cannot learn boolean functions (e.g. which $k$ in a $k$-sparse parity in which learning with feature noise is possible, as well as majority functions, and other $k$-juntas).
- The analysis of the sensitivity of the optimal predictor under feature noise versus the teacher is an interesting perspective, and a conjecture regarding this is proposed, backed by simulations.

### Weaknesses
- The analysis for parity and majority seem quite straightforward, and it would be interesting if there was more theoretical analysis on progress towards the conjecture.
- For the LSTM model, it would be useful to have some theoretical analysis of this setting for learning boolean functions too.
- Perhaps an example of a realistic setting of feature noise in training transformers would be useful, and I believe this would better motivate this paper.

### Questions
- What are some of the main bottlenecks preventing the authors from making the conjecture rigorous?

### Soundness
3

### Presentation
2

### Contribution
3
