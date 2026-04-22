# Learning to Dissipate Energy in Oscillatory State-Space Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
State-space models (SSMs) are a class of networks for sequence learning that benefit from fixed state size and linear complexity with respect to sequence length, contrasting the quadratic scaling of typical attention mechanisms.  Inspired from observations in neuroscience, Linear Oscillatory State-Space models (LinOSS) are a recently proposed class of SSMs constructed from layers of discretized forced harmonic oscillators.  Although these models perform competitively, leveraging fast parallel scans over diagonal recurrent matrices and achieving state-of-the-art performance on tasks with sequence length up to 50k, LinOSS models rely on rigid energy dissipation (``forgetting'') mechanisms that are inherently coupled to the time scale of state evolution.  As forgetting is a crucial mechanism for long-range reasoning, we demonstrate the representational limitations of these models and introduce Damped Linear Oscillatory State-Space models (D-LinOSS), a more general class of oscillatory SSMs that learn to dissipate latent state energy on arbitrary time scales.  We analyze the spectral distribution of the model's recurrent matrices and prove that the SSM layers exhibit stable dynamics under a simple, flexible parameterization. Without additional complexity, D-LinOSS consistently outperforms previous LinOSS methods on long-range learning tasks, achieves faster convergence, and relinquishes the need for multiple discretization schemes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the Damped Linear Oscillatory State-Space Model (D-LinOSS), an extension of the Linear Oscillatory State-Space Model (LinOSS), which models sequence data through layers of discretized forced harmonic oscillators. While previous LinOSS models coupled frequency and damping, restricting energy dissipation to a single time scale, D-LinOSS introduces learnable damping parameters that allow the system to adaptively dissipate energy on arbitrary time scales.

### Strengths
This paper presents a solid and well-motivated contribution to the theory and design of oscillatory state-space models. By introducing learnable damping into the Linear Oscillatory State-Space Model (LinOSS), the proposed D-LinOSS enhances the model’s ability to control forgetting and energy dissipation over arbitrary time scales. This addresses a key limitation of previous LinOSS variants, where frequency and damping were rigidly coupled, restricting flexibility in long-range dynamics. The work is theoretically rigorous, providing a detailed spectral analysis and stability proof that clarifies how the new parameterization expands the set of representable stable systems.

On the empirical side, the paper supports its theoretical insights with well-designed numerical experiments on synthetic and real-world datasets such as PPG-DaLiA and the Weather forecasting benchmark. These results, though modest in scale, consistently demonstrate that D-LinOSS achieves faster convergence, improved stability, and slightly higher accuracy compared to prior oscillatory SSMs. Moreover, the authors’ discussion of universality—showing that D-LinOSS preserves the universal approximation property of LinOSS—adds conceptual depth and connects the work to broader theoretical frameworks in sequence modeling. Overall, the paper offers a meaningful and well-executed advancement in understanding and improving the forgetting mechanisms of state-space models.

### Weaknesses
The main limitation of the paper lies in its empirical evaluation, which appears relatively underdeveloped given the strength of its theoretical contributions. The experimental results seem under-trained and insufficiently tuned, particularly for the baseline comparisons—there is little evidence that the baselines were optimized to their best performance. This makes it difficult to assess the true effectiveness of the proposed model.

Furthermore, the study is limited to small-scale numerical experiments, leaving open the question of whether the claimed properties—especially those related to stability and forgetting—would generalize to larger or more complex datasets. Without experiments at scale or more diverse benchmarks, the practical impact of the proposed approach remains uncertain. To strengthen the paper, the authors could include carefully tuned baselines, ablation studies, and larger-scale experiments that better validate the theoretical claims in real-world scenarios.

### Questions
Can you provide the detail about the tuning of the numerical experiments provided in the paper?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work revisits a very well celebrated work from last year and identifies a shortcoming in the set of equations describing the linear oscillatory state space model (LinOSS). Namely, the set of equations did not involve a damping term, which the authors in this work introduce. They show that with the addition of this term enables faster, efficient, and effective training across various benchmarks.

### Strengths
- The addition of the dampening term is a trivial, yet needed, addition that is well motivated and, in this reviewer's opinion, should have been part of the LinOSS to begin with. In this sense, authors have identified an important gap.

- Benchmarks are comprehensive and convincing that the addition of the dampening term is broadly useful; though more is needed to support the specific claims (see below).

### Weaknesses
- The scientific novelty/contribution is quite limited. The main model, i.e. LinOSS, in my opinion is a very important direction that future research should build on. However, the addition here is trivial, and does not necessarily lead to new novel insights. To me, this paper feels like a comment on the original paper as opposed to having enough novelty for deserving a paper of its own.

- I find the claim "reducing the hyperparameter search space by 50%" to be slightly misleading. Practically, one should have just simulated an oscillatory model as accurately as possible. In that sense, the choice between implicit or implicit-explicit integration method should not have been presented as an hyperparameter by the original work, since the former is effectively introducing decay that does not exist in real continuous dynamics. In my opinion, the fact that explicitly modeling decay corrects for this "spurious capability" resulting from incorrect discretization is a much more interesting contribution as opposed to "taking away" one hyperparameter. Thus, I would like to recommend emphasizing this contribution to the scientific methodology as opposed to reduction in hyperparameter count.

### Questions
- Could you please use the same symbols as the LinOSS paper for Eq. 1? It seems the variables x and y are switched, which leads to confusion on the part of the reader.

- It seems in Tables 2 and 3 that LinOSS-Im is more or less at the same level as D-LinOSS? The same may be true for Table 4, though no error bars are reported (for a good reason due to data scarcity, so not authors' fault...).  In that sense, can a devil's advocate say learning to decay is not that important as long as some amount of decay is built-in? For instance, could you please add a baseline in which the decay terms are set equal to some value (defined by the task requirements, or some hyperparameter)? Will D-LinOSS still beat this baseline?

- Looking at Figure 2, it seems that increasing the sequence length makes D-LinOSS more or less the same accuracy and speed as LinOSS-Im? Can you show training for longer sequences? Moreover, similar to above, can you fix the G entries and add this new model as a baseline too?

Overall, my current assessment is as follows:  On the empirical side, more experiments are needed to fully understand when learning the decay timescales is necessary. Theoretically, this work makes an important addition to a very influential model that was given an oral presentation in last year's ICLR conference. However, this addition almost feels like a correction, as opposed to introducing a new capability of the model. In that sense, what makes this model very exciting is already published and very well celebrated by the community. In my assessment, this work could be published as a comment on the original work, for instance in TMLR; but does not satisfy the significance and novelty requirements of ICLR.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a new continuous-time recurrent network by extending Linear Oscillatory State-Space models (LinOSS). Specifically, the proposed Damped LinOSS (D-LinOSS) adds a damping term to the continous-time forced second-order ODEs underlying each layer. This damping term enables learnable energy dissipation, allowing the discrete-time model to represent a wider range of stable dynamics than previous discretzied LinOSS models. On average, D-LinOSS outperforms LinOSS (both the IM and IMEX discretizations) and reaches State-of-the-Art performance on the evaluated benchmark tasks.

### Strengths
- The empirical results of D-LinOSS on several real-world datasets are significant, improving State-of-the-Art as well as the previous LinOSS models. 
- The addition of damping to D-LinOSS offers clear practical and theoretical advantages over LinOSS models, enabling eigenvalues that cover the entire unit disc, thus uncoupling frequency and energy dissipation. Table 1 and Figure 3 are particularly compelling illustrations of these advantages.

### Weaknesses
- Please clearly define (and if necessary, differentiate) the terms: energy dissipation, forgetting, damping, and their relation to real and imaginary eigenvalue components, eigenvalue magnitude, and oscillation frequency. Particularly energy dissipation, damping, and forgetting are often used loosely and somewhat interchangeably, which can be confusing. It may help to add a section defining important vocabulary at the beginning of Section 2. 
- The proof of bijectivity in Proposition A.3 is not sufficiently elaborated. Particularly, more details on the exact derivation of the inverse function from eigenvalues to A, G would help convince that it is the true inverse. In addition, classical injectivity-surjectivity arguments would aid in comprehension. 
- Figure 1 is somewhat difficult to parse. Using unnormalized time and different colors for each “frequency” as well as clearly labelling the z axis could go a long way. In addition, the caption should more closely explain the figure and directly reference it. Finally, it might be worth considering including Figure 3 in Figure 1 (or at least more prominently placing Figure 3 since it’s quite easy to understand and clarifies the main contribution of adding learnable damping). 
- The phrase "reducing the hyperparameter search space by 50%” is a bit misleading (unclear whether you are referring to the number of hyperparameters which are eliminated or the entire search space covered by all hyperparameters...). Maybe rephrase this to clearly state that you are just eliminating the need to choose between LinOSS-IM and LinOSS-IMEX. 
- Even though the empirical results are strong, their interpretation is missing. Please elaborate on possible reasons why D-LinOSS is better at the UEA Motor task while lagging on the Heartbeat task. 

If these weaknesses are sufficiently addressed, I am happy to increase my rating score.

### Questions
- What is meant by the trainability of the timestep parameters in line 124? In the original LinOSS, the timestep is not trainable and fixed. 
- Could you add some more intuition on why LinOSS-IMEX and LinOSS-IM can be universal while still failing to model exponential decay? 
- One of the main reasons to choose LinOSS-IMEX over LinOSS-IM is that it can model conservative systems. How does D-LinOSS compare to LinOSS-IMEX when modelling conservative systems? What solution does D-LinOSS reach (i.e., does it actually learn to set G=0, or does it find some other solution for conservative systems)? 
- Figure 3 gives a clear explanation of why D-LinOSS is better than LinOSS, theoretically. If we plot the actual eigenvalues after training on a real-world task (of your choice), is this behavior verified? This would give more body to the answer to “A natural question is whether or not a larger set of reachable eigenvalues is empirically useful” of line 250. 
- Figure 2 confirms the faster convergence on the synthetic task. Is this benefit also observed on the real-world datasets? A similar figure in the appendix would be a great addition.

### Soundness
3

### Presentation
2

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
In their paper, the authors build upon LinOSS and extend the previous physics-inspired oscillatory state-space model by introducing learnable damping. They identify limitations in existing oscillatory SSMs and theoretically motivate learnable damping. The authors perform a small synthetic experiment to learn exponentially decaying functions, and show that D-LinOSS outperforms LinOSS across eight different real-world learning tasks. The authors derive theoretical proofs of the improved representational capacity of D-LinOSS, while maintaining computational efficiency.

### Strengths
- The authors clearly outline the contributions of their paper, highlighting the limitations and improvements over existing oscillatory state-space models (LinOSS).
- The authors conduct rigorous theoretical analysis of their method and provide a link between the new parameterization to improved representational capacity. They prove that D-LinOSS layers span the full unit disk in the complex plane. 
- The authors provide a controlled synthetic experiment (learning exponential decay) to justify D-LinOSS and an empirical evaluation on more complex, real-world learning tasks.
- The authors present thorough empirical results outperforming LinOSS across eight datasets with details on hyperparameters, good reproducibility.
- Figure 1 is very well-designed.

### Weaknesses
- Most reported improvements in the presented empirical results are marginal. I would like to see more ablation on model size/parameter count to be sure of the practical significance of D-LinOSS. 
- While D-LinOSS is well-motivated, the contribution feels like a simple extension of LinOSS. The contribution feels incremental in the broader landscape without additional analysis on the interpretation of learned weights.

### Questions
Are the learned G values interpretable? Do there exist any links to the underlying physical motivation? Can you show qualitative evidence that learned damping captures long-term versus short-term dependencies?

### Soundness
4

### Presentation
4

### Contribution
2
