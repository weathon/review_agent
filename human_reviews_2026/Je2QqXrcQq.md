# R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8

## Abstract
A central challenge in image-based Model-Based Reinforcement Learning (MBRL) is to learn representations that distill essential information from irrelevant visual details. While promising, reconstruction-based methods often waste capacity on large task-irrelevant regions. Decoder-free methods instead learn robust representations by leveraging Data Augmentation (DA), but reliance on such external regularizers limits versatility. We propose R2-Dreamer, a decoder-free MBRL framework with a self-supervised objective that serves as an internal regularizer, preventing representation collapse without resorting to DA. The core of our method is a \emph{redundancy-reduction} objective inspired by Barlow Twins, which can be easily integrated into existing frameworks. On DeepMind Control Suite and Meta-World, R2-Dreamer is competitive with strong baselines such as DreamerV3 and TD-MPC2 while training 1.59$\times$ faster than DreamerV3, and yields substantial gains on DMC-Subtle with tiny task-relevant objects. These results suggest that an effective internal regularizer can enable versatile, high-performance decoder-free MBRL. Code is available at https://github.com/NM512/r2dreamer.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper  presents a decoder-free agent that introduces a self-supervised objective acting as an internal regularizer to prevent collapse. Experiments on DMC and DMC-Subtle validate the effectiveness of the proposed method.

### Strengths
- This paper proposes a decoder-free MBRL agent that adopt internal regularizer to avoid reconstructing observations from a latent state.
- This paper introduces DMC-Subtle, a modified DMC benchmark where task-critical objects’ sizes are significantly reduced, demanding a higher level of representational precision.
- The proposed approach outperforms baselines in DMC and DMC-Subtle.

### Weaknesses
- There is no comparison with competitive baselines such as TD-MPC2 [1].
- The proposed DMC-Subtle benchmark seems somewhat questionable or not well-justified.
- More challenging benchmarks are needed to better validate the effectiveness of the proposed algorithm.
- The DMC tasks used are relatively simple. How does the method perform on more complex tasks like *DMC Dog* or *DMC Humanoid*?
- As shown in Figure 3, the improvement appears to be marginal.
- There is no comparison with VAI [2], which is a baseline that adopts unsupervised visual attention.

References:

[1] Hansen et al. "TD-MPC2: Scalable, Robust World Models for Continuous Control", ICLR, 2024.

[2] Wang et al. "Unsupervised Visual Attention and Invariance for Reinforcement Learning", CVPR, 2021.

### Questions
Have the authors tried conducting experiments on environments with distractors, such as Distracting Control Suite [1]?

Reference:

[1] Stone et al. "The Distracting Control Suite — A Challenging Benchmark for Reinforcement Learning from Pixels", arXiv, 2021.

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
4

### Summary
This paper proposes R2-Dreamer, an enhanced variant of DreamerV3 that introduces an internal regularizer to improve latent state representation learning. Specifically, the authors replace the traditional image reconstruction loss with a Barlow Twins loss, aiming to encourage more informative and less redundant latent features. Theoretical analysis shows that this new objective is equivalent to optimizing a variational bound on an extended Sequential Information Bottleneck. Experimental results demonstrate that the proposed method achieves more robust latent representations and improved computational efficiency.

### Strengths
1. The paper clearly identifies a key limitation of DreamerV3: its latent representations can be overly influenced by reconstructing input images, leading to a focus on irrelevant pixels rather than task-relevant features. By replacing the reconstruction loss with a self-supervised objective defined in the latent space, the proposed method encourages more compact and task-relevant representations.
2. The paper provides solid theoretical analysis to explain the impact of the new learning objective and includes extensive experimental validation. Results on the challenging DMC-Subtle benchmark demonstrate the superiority of the proposed representation. Moreover, the ablation studies convincingly show both the necessity of the proposed objective and the redundancy of data augmentation under this framework.
3. The proposed approach is conceptually simple yet empirically effective, making it a strong candidate for a new baseline in model-based reinforcement learning.

### Weaknesses
1. The experimental environments are not sufficiently diverse. The paper evaluates only on DMC-Subtle, whereas DreamerV3 has also been tested on other benchmarks such as Atari and DMLab. Including experiments on these additional environments would make the evaluation more comprehensive and the conclusions more convincing.

### Questions
1. Minor: Why are the ablation results of R2-Dreamer+DA different between Figure 7 and Figure 10? In Figure 7, the performance of the DA version shows a significant drop, while in Figure 10, the drop is much smaller. This discrepancy seems inconsistent with the claim that data augmentation is harmful for precision-demanding tasks.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a data-augmentation-free learning method for RSSM-based decoder-free MBRL. The core of this method is is a feature redundancy reduction objective inspired by Barlow Twins and therefore has no need for pixel-wise reconstruction or data augmentation. Emprical study shows the superior performance on standard DMC benchmark and more chanlleging DMC-Subtle benchmark.

### Strengths
1. The paper is clearly written, allowing readers to follow the main arguments.

2. It provides a comprehensive experiments and ablations to demonstrate the effectiveness of their proposed new representation learning paradigm.

3. Releasing codebase is good, which can facilitate future research.

### Weaknesses
1. The authors didn't compare TD-MPC2 in their experiments, which is a strong state-of-the-art baseline for decoder-free methods.
2. Though authors evaluated different methods on many tasks on the DMC benchmark, I think evaluating only on these locomotion tasks is kind of not comprehesive and it would be better to evaluate on other types tasks like Meta-World.
3. One of claims in this paper is that their method doesn't need hand-engineered data augmentation like other decoder-free MBRL methods. But I question whether this is really a significant problem. For example, TD-MPC2 only uses very simple random shift augmentation, which I don't think it is troublesome for algo implementation. So I'm confused that why the data augmentation for decoder-free MBRL methods is a drawback.

### Questions
1. Could the authors compare their method with state-of-the-art decoder-free MBRL methods like TD-MPC2?
2. Could the authors evaluate different methods on other benchmarks like Meta-World?
3. Could authors explain why data augmentation is a problem?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes R2-Dreamer, a Model-Based Reinforcement Learning (MBRL) agent based on the DreamerV3 architecture. It addresses two main limitations in current image-based world models: the computational expense and potential for task-irrelevant overfitting in decoder-based models, and the reliance on brittle, task-specific Data Augmentation (DA) in existing decoder-free models. R2-Dreamer replaces the reconstruction objective with a redundancy-reduction objective inspired by Barlow Twins, applied between the image embeddings and the projected latent states of the RSSM.

### Strengths
1. Principled approach to removing DA: The paper successfully identifies a major pain point in current decoder-free methods—the reliance on heuristic DA. Proposing an information-theoretic internal regularizer (redundancy reduction) as a replacement is a sound and theoretically motivated direction, nicely grounded in the Sequential Information Bottleneck framework in Appendix A2.
2. Strong empirical validation of the core hypothesis: The ablation study (Figure 6) provides compelling evidence. It shows that while DreamerPro collapses without DA, R2-Dreamer maintains performance, proving that the proposed $\mathcal{L}_{BT}$ effectively prevents collapse without external heuristics
3. Effective stress-testing (DMC-Subtle): DMC-Subtle cleanly isolates failure modes of reconstruction-based (wasted capacity on background) and DA-based (distortion of small features) methods, highlighting the specific advantages of R2-Dreamer in precision-demanding tasks
4. Computational Efficiency: The reported speedups are significant and practically important for scaling MBRL.

### Weaknesses
1. Batch size concerns for Barlow Twins: Redundancy reduction objectives often require large batch sizes for stable covariance estimation. The paper uses standard Dreamer batching ($B=16, T=64 \implies N=1024$)8. While this appears sufficient for DMC, it raises concerns about stability in higher-dimensional or more diverse visual environments where 1024 samples might not sufficiently estimate the cross-correlation matrix.
2. Ambiguity in Encoder Training: The pseudocode (Algorithm 1) indicates that the image embeddings $e$ are detached before entering the $\mathcal{L}_{BT}$ loss9. If accurate, this means the image encoder does not receive gradients from the primary representation learning objective, relying solely on gradients flowing back from the RSSM via $KL$ terms, which is highly unusual for this class of SSL objectives.

### Questions
1. Clarification on Gradients: In Algorithm 1, you have e = ...detach(). Does this mean your image encoder $f_\phi(x_t)$ receives NO gradients from $\mathcal{L}_{BT}$? If so, what is the primary learning signal for the encoder? Is it solely the $KL(q(z_t|h_t, e_t) || p(z_t|h_t))$ term?
2. Batch Size Sensitivity: Have you evaluated the sensitivity of R2-Dreamer to the batch size (specifically the total samples $B \times T$ used for the correlation matrix)? Barlow Twins often degrades rapidly with small batches.
3. Complex Backgrounds: How does R2-Dreamer perform when the background is not just static (like DMC) but dynamic and irrelevant (e.g., "distracting control" suite)? This would be a stronger test of the claim that it avoids wasting capacity on irrelevant details compared to reconstruction.

### Soundness
4

### Presentation
4

### Contribution
4
