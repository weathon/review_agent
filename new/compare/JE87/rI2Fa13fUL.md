# Review

## Summary
This paper proposes a unified framework for understanding diffusion-based policies and consistency-based policies. Building on this framework, the authors introduce a new method, Generative Trajectory Policies (GTPs), which leverages a surrogate score function to improve computational efficiency and incorporates advantage weighting to enable policy improvement. Experimental results demonstrate that GTP achieves state-of-the-art performance across multiple tasks in the D4RL benchmark.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
- The paper is well-structured and clearly written.
- The authors provide a thorough review of prior work.
- The proposed method achieves state-of-the-art performance on several tasks in the D4RL benchmark.

## Weaknesses
- The theoretical results appear somewhat limited. Theorem 1 holds only under the assumption that $t = \tau_0 > \tau_1 > \ldots > \tau_K = u$, which implies that the time step $h$ between adjacent points in the trajectory must be equal. This condition is difficult to satisfy in practice, as the time steps in a trajectory are typically not uniform. Additionally, Theorem 2 is a well-known result in offline RL, and the advantage-weighted objective has been widely used in previous works. Although the authors provide a citation, it would be clearer to acknowledge this directly in the main text.
- The proposed method is closely related to prior work, particularly CTMs and C-AC, and the distinction between GTP and these methods is not fully clear. The authors should provide a more detailed comparison to clarify the novelty of their approach.

## Questions
- Could the authors clarify the key differences between GTP and CTM? From my understanding, the primary distinction lies in the use of a surrogate score function, which is also used in C-AC. It would be helpful if the authors could elaborate on any additional contributions beyond these existing techniques.
- Could the authors provide more details on the training process? Specifically, how are the time steps $t$, $\tau$, and $u$ selected during training? Are they sampled from a uniform distribution, and do the time intervals between adjacent steps need to be equal? Additionally, how is the surrogate score function $\tilde{f}(x_t, t)$ implemented during training?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4