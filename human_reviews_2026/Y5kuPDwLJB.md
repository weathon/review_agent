# The Geometry of Grokking: Norm Minimization on the Zero-Loss Manifold

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2

## Abstract
Grokking is a phenomenon in neural networks, where full generalization occurs only after a substantial delay after complete memorization of the training data. Previous research has linked this delayed generalization to representation learning driven by weight decay, but the precise underlying dynamics remain elusive. In this paper, we argue that post-memorization learning can be understood through the lens of constrained optimization: gradient descent effectively minimizes the weight norm on the zero-loss manifold. We formally prove this in the limit of infinitesimally small learning rates and weight decay coefficients. To further dissect this regime, we introduce an approximation that decouples the learning dynamics of a subset of parameters from the rest of the network. Applying this framework, we derive a closed-form expression for the post-memorization dynamics of the first layer in a two-layer network. Experiments confirm that simulating the training process using our predicted gradients reproduces both the delayed generalization and representation learning characteristic of grokking.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper aims to establish a better theoretical understanding of the grokking phenomenon. They argue that gradient descent first reaches the zero-loss manifold (i.e., achieves perfect memorization) and then stays close to this manifold while decreasing the parameters’ norm, which eventually leads to generalization. Also, they provide a slightly modified learning dynamics where the post-memorization dynamics can be analyzed. They also provide some empirical results to support their claim.

### Strengths
Obtaining a better theoretical understanding of grokking is an interesting question that has attracted interest in recent years.  Explaining grokking by analyzing the dynamics after memorization is a natural approach, and, as the paper argues,  weight decay seems to play a key role in the post-memorization dynamics.

### Weaknesses
The bottom line is that I don’t think the contributions of the paper are significant enough. 

In Section 4, the main contributions are Theorems 4.9 and 4.13. Theorem 4.9 provides a nice observation, namely, that with a small enough weight decay, once gradient flow reaches the zero-loss manifold, it will stay close to it. Then, Theorem 4.13 establishes that, under some assumptions, around the zero-loss manifold, the gradient of the unregularized loss will be roughly orthogonal to the manifold. Hence, the movement on the manifold seems to be mostly controlled by the weight decay. Theorem 4.13 considers the direction of the gradient of the unregularized loss, but when $\theta$ is close to the zero-loss manifold, the magnitude of the gradient is anyway very small, and hence, the weight decay becomes dominant. Also, the regularized loss might be small even if the distance to the manifold is large, and then the theorem does not apply. I view these results as nice observations, but not as a significant step towards a better understanding of grokking. 

In Sections 5 and 6, the authors analyze a modified dynamics. In the case of two-layer networks, it corresponds to training the output layer much faster than the hidden layer. They obtain a closed-form expression of the output layer (as a function of the hidden layer) and of the dynamics of the hidden layer, and then show empirically (in Section 7) that delayed generalization also occurs under this regime. I think that it is an intriguing regime, but I don’t really understand how this discussion helps us understand grokking.

I did not verify the technical details.

### Questions
In Theorem 4.13, there is an assumption that  $proj_{\mathcal{Z}}$ $(S \cap \mathcal{Z}) \subseteq S$, but by definition if $x \in S \cap \mathcal{Z}$ then proj$_{\mathcal{Z}}(x) = x \in S \cap \mathcal{Z} \subseteq S$. I there a typo here?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The manuscript claims that post-memorization dynamics is driven by weight decay. Explicitly, that after memorization the learning trajectory is confined to the zero-loss manifold and weight decay minimizes the weight norm subject to the zero loss constraint. The authors claim that this is cause of grokking.

### Strengths
The general idea -- that in the presence of weight decay dynamics on the zero loss manifold are determined by the weight decay term -- is (trivially) correct.

### Weaknesses
The actual theorems seem trivial to me, and the presentation is over complicated. If one assumes that the dynamics has reached the zero loss manifold, then by definition further dynamics are not controlled by loss gradients (which vanish) but only by the regularization, if present. However, this does not answer two things which are crucial for the proposed mechanism to work:
1. why should the dynamics reach the zero loss manifold at all, in the presence of regularization?
2. Assuming it does,  why/when/under which conditions/etc does should weight norm minimization on this manifold should lead to better generalization?

Sec 5.2 is poorly written and hand wavy, and contains unsupported claims and trivial mistakes. For example, the claim that "it is highly unlikely that our trajectory will pass through the same θ1 twice" is loose and ill defined, and does not imply Eq. 9

In addition, the manuscript is over complicated with its non-standard definitions ("available direction"=tangent vector etc).

### Questions
none

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors prove that once a grokking model reaches the zero-loss set, its subsequent trajectory can be described as minimizing the weight norm subject to the constraint of remaining on the zero-loss set (under a set of suitable assumptions). They show empirically that a toy model that simulates these constrained dynamics behaves qualitatively similarly to grokking models (exhibiting the characteristic phenomena of delayed generalization and circular representation learning).

### Strengths
**Crisp, elegant result:** The overall takeaway is quite elegant and easy to understand. 

**Very well-written**: This was an exceptionally clear paper. I was really impressed with how easy this was to read. Similarly, the figures were wonderfully designed.

**Interesting methods**: The separation of dynamical systems into slow variables and fast variables that can be integrated out is a pillar of dynamical systems analysis. Applying it in this setting to understand learning on the zero-loss set is quite creative and interesting. I think this is likely to be a productive set of tools to apply to study other toy models and interesting dynamical phenomena. I want to see this paper published for this reason.

### Weaknesses
**1. Missing empirical validation.** After solving for the closed-form expression of the post-memorization dynamics of (part of) a grokking model, the authors show that simulating this training process reproduces two of the behaviors associated with grokking: delayed generalization and circular representation learning. But, if I understand correctly, this does not establish the central theoretical claim (lines 113-118):

> While the previous example illustrates our theoretical framework, it does not capture its full generality. It is unsurprising that applying weight decay encourages a reduction in norm. However, the central claim of this paper is significantly stronger: we argue that the learning dynamics under weight decay do not merely follow some norm-decreasing direction, but rather evolve along the direction *that optimally minimizes the norm, subject to remaining on the zero-loss manifold*.

The authors have not established that delayed generalization and representation learning occur *only* in the case where the training trajectory optimally minimizes the norm subject to remaining on the zero-loss manifold. These two observed phenomena may still be compatible with many other (suboptimal) learning trajectories. 

On a closer reading, I believe that most of the current writing does not make any false claims about what these experimental results imply. For example, the following lines do not need to be changed:
- Lines 022–023: "confirm that simulating the training process ... reproduces both the delayed generalization and representation learning..."
- Lines 427–428: "validate that our approximate learning dynamics reproduce the phenomena observed during standard training"

However, there are a few instances that can be interpreted too broadly and thus should be rewritten: 
- Line 062: "we validate our theoretical insights"
- Line 375: "we empirically validate our combined theoretical insights" 
- Lines 468–470: "formally established that the learning dynamics ... approximate as the minimization of the weight norm within the zero-loss set."

Minimum correction: In addition to addressing the above overstatements, the main body should explicitly comment on the remaining discrepancy between theoretical predictions and empirical confirmation. Omitting this would implicitly suggest stronger empirical confirmation than is currently demonstrated. 

Full correction: Convincingly establishing that the true training trajectory actually does follow the zero-loss-constrained norm-minimization path requires additional experiments. For example, you could test the alignment (via cosine similarity) between actual updates and predicted updates under training. Alternatively, you could argue for the necessity of the optimal direction if you can show that other learning trajectories within the zero-loss set do not produce these behaviors (or perhaps much more slowly). These are just suggestions. I'm sure the authors can come up with better ways to validate this. 

**2. False assumption about negligible singularities.** The manuscript claims that singular points "form a null set" and that the "probability of encountering a singularity during standard training is exactly zero" (lines 157–158), and then assumes the trajectory never passes through them (Assumption 4.7). There is an additional assumption about a constructed set not containing singularities in Theorem 4.13. I believe that the current conditions are insufficient for these statements to hold and that they are partially false as written.

2.1 Assumption 4.7: 
> We assume that our training trajectory does not pass through singular points. This is motivated by the fact that singular points form a null set, i.e. a set of Lebesgue measure zero.

The first problem is that, as written, smoothness alone does not imply this set is measure zero. That requires invoking an additional non-degeneracy condition (that there is at least some parameter with full rank) to argue that the rank-deficient locus is "thin." I think these concerns would be resolved by including some wording about "standard genericity conditions." 

However, I believe there may still be a problem in that standard architectural degeneracies (e.g., $ReLU(\alpha x) = \alpha ReLU(x)$ for $\alpha> 0$) would invalidate these conditions. Since these degeneracies also hold off of the measure-zero set, I do not believe they pose a substantial challenge to the theoretical argument and can be easily eliminated.

The second problem is that the stated assumption does not follow from the provided motivation. The fact that singular points form a null set is sufficient to establish that *before* training (under a random initialization), the probability of encountering a singularity is vanishing. However, the probability *after* training can, in general, be non-zero that a 1D training path crosses such measure-zero sets at isolated times. This claim can be strengthened by arguing theoretically that the path remains in a compact subset bounded away from the singularities and/or with empirical evidence showing that, e.g., the smallest singular value of $\nabla \mathcal F$ is non-zero.

2.2 Theorem 4.13: 
I believe you need a stronger statement than the absence of singularities (line 673) and that you need an additional statement that says something about keeping a uniform gap between singularities and the training trajectory. From my understanding, if the training trajectory is allowed to get arbitrarily close to singularities, the positive constant $\lambda_\text{inf}$ introduced in lines  721–723, can get arbitrarily small, which renders the resulting bound in Theorem 4.13 vacuous. (That the training trajectory can get arbitrarily close to singularities is no surprise. After all, the entire paper rests on the condition that the training trajectory gets close to the measure-zero zero-loss set.)

Minimum correction: 
My first concern (2.1) would be resolved by including some wording about "standard genericity conditions", clarification from the authors about my question on standard degeneracies, and weakening the statement about the loss trajectory avoiding singularities or providing additional theoretical/empirical evidence that the trajectory does not pass near singularities. 

My second concern (2.2) would be resolved by including an additional assumption on the gap between singularities and the training trajectory, or perhaps simply clarification from the authors on this point. I am less confident with (2.2) than with (2.1). 

Full correction: 
I am quite confident that the authors' general attitude towards singularities being negligible is false. Neural networks have incredibly rich degeneracies, and a lot of well-established theory (notably Watanabe's singular learning theory) emphasizes the key role these singularities play in learning dynamics. If the core theoretical derivation of the paper could be strengthened to drop the assumptions of negligible singularities (which seems very difficult to me), then this would resolve my remaining concerns.  

I recommend acceptance conditional on (i) softening the empirical‑validation language and acknowledging the remaining gap, and (ii) qualifying the singularity statements (ideally with a uniform‑gap assumption or an empirical singular‑value check). Addressing both minimum fixes moves me to accept; adding one of the maximum fixes would push me towards a strong accept.

### Questions
Do you see any connections to existing work within the literature on inductive biases on margin-maximization? 

See the questions I've interspersed among my concerns under weaknesses.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper analyzes the grokking phenomenon and argues that it can be explained through the lens of constrained optimization: fast memorization corresponds to fast convergence to the zero-loss manifold, then weight decays drives the model to minimize the weights' norm and thus (slowly) travel on the zero-loss manifold, towards a generalizing solution. Going further, the paper shows that this dynamic can be analyzed by decoupling the embedding layer from the rest of the network.

The theoretical results are crystallized in Thm 4.9, showing that optimization remains close to the zero-loss set Z, once reached, Thm 4.13, showing that gradient of the loss is orthogonal to Z in the vicinity of Z, suggesting that the trajectory is only driven by weight decay when close to Z / in Z, and Thm 5.1 which is essentially a technical tool allowing the authors to analyze the training dynamics of a subset of the parameters in isolation. This last theorem is used to provide an analysis of the first layer dynamic in a two layer network.

Experiments are provided to validate the insights of Thm 4.9 and 4.13 and the simulated dynamics of the 2 layer network.

### Strengths
- The paper is overall well written and easy to follow.
- The topic is definitely timely and relevant and the analysis is interesting. In particular, the approach taken to study isolated dynamics could be relevant in other settings.

### Weaknesses
- Some very relevant papers are missing for the literature review. In particular, Q1 investigated in this paper has already been quite thoroughly addressed in [1]. [2] also shows that this behaviour extends to other form of regularization, beyond weight decay. While the technical tools used in this submission to support the hypothesis that grokking is an artifact of regularization (memorization induced by data fitting term followed by slow convergence to generalizing solution driven by regularization) seems at first sight different from the one used in [1] and [2], this potentially affects significantly the novelty of this submission (at least the parts related to Q1, i.e. Section 3 & 4, Thms 4.9&4.13). It is important that the authors clarify during the rebuttal the position of the current submission compared to [1] and [2], and to check references in [1] and [2] as there may be additional relevant papers that were missed in the literature review.  [My low score could be updated during rebuttal depending on how this point is addressed by the authors].

- Overall, the results presented in Thm 4.9 and 4.13 are interesting but somehow not too surprising: 
   - Thm 4.9 boils down to observing that the regularization term can be taken to be arbitrary small to make the minimum arbitrary close to any point in Z, and thus constraining the dynamics to stay arbitrary close to the zero-loss set.
   - Thm 4.13 states that gradients points towards the zero-loss set in its vicinity, which as far as I see is quite expected. I am not entirely convinced that this theoretical result is strong enough to derive the conclusion the authors get to in Rmk 4.14.

- Section 6 is interesting but seems to be more hastily written than the previous ones. It is difficult to get to the high-level message that this section wants to convey. It shows an interesting derivation, but lacks commenting on its relevance in the current scope from a high-level point. 

[1] Lyu et al., "Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking." ICLR 2023

[2] Notsawo et al., "Grokking Beyond the Euclidean Norm of Model Parameters." ICML 2024

### Questions
- Please situate your work w.r.t. to refs [1] and [2]. In particular, how does the analysis presented in [1] already supports the conclusions of the analysis presented in Section 3 and 4? 

- Can you clarify how Thm 4.13 goes further than simply observing that gradient is zero in Z, hence dynamics are only driven by weight decay? 

#### Minor points / questions

- Thm 4.13: why do we need the assumption $\mathrm{proj}_Z(S\cap Z) \subset S$? Can you give an example where this would not be satisfied? 

- Thm 4.13: Why are 2 constants, C and x_0,needed? Couldn't you "absorb" the constant C in x_0? I.e. by defining $\tau = Cx_0$, we can get $\tau dist_Z(\theta)$ as the rhs of ineq (6), no? 

- Eq 19 and Eq 20: both $(H^TH)^{-1}$ and $(HH^T)^{-1}$ appear in these equations but only one of the two is invertible! (unless H is square...).

- Eq 20: Why doesn't terms related to the loss L appear in this equation? It seems to me that $L(W_1,\phi(W_1))$ is not necessarily $0$ for all choices of $W_1$...

### Soundness
3

### Presentation
3

### Contribution
2
