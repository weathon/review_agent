# Extended Flow Matching  : a Method of Conditional Generation with Generalized Continuity Equation

- Decision: Reject
- Scores: 5, 5, 6, 6, 3

## Abstract
Conditional generative modeling (CGM), which approximates the conditional probability distribution of data given a condition, holds significant promise for generating new data across diverse representations.
While CGM is crucial for generating images, video, and text, its application to scientific computing, such as molecular generation and physical simulations, is also highly anticipated.
A key challenge in applying CGM to scientific fields is the sparseness of available data conditions, which requires extrapolation beyond observed conditions.
This paper proposes the Extended Flow Matching (EFM) framework to address this challenge.
EFM achieves smooth transitions in distributions when departing from observed conditions, avoiding the unfavorable changes seen in existing flow matching (FM) methods.
By introducing a flow with respect to the conditional axis, EFM ensures that the conditional distribution changes gradually with the condition.
Specifically, we apply an extended Monge--Kantorovich theory to conditional generative models, creating a framework for learning matrix fields in a generalized continuity equation instead of vector fields.
Furthermore, by combining the concept of Dirichlet energy on Wasserstein spaces with Multi-Marginal Optimal Transport (MMOT), we derive an algorithm called MMOT-EFM.
This algorithm controls the rate of change of the generated conditional distribution.
Our proposed method outperforms existing methods in molecular generation tasks where conditions are sparsely observed.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces extended flow matching a new flow matching based method, that is designed for conditional generation. For this, the authors make use of the generalized continuity equation by Lavenant. The authors show that their proposed loss indeed has the correct gradients, i.e., regresses onto the true velocity field of the generalized continuity equation. The algorithm consists "learning" an interpolation via kernel regression (which is needed since "straight paths" are not the only viable solution anymore), and then regressing onto a flow matching loss where the is now matrix-valued. This is a generalization of the usual inverse problems framework of flow matching. Further, the authors showcase the effiacy of their algorithm via a toy example and conditional molecular generation.

### Strengths
I find the motivation very clear: Sometimes we already know the posteriors for several conditions (for instance in molecular dynamics, where we obtain some posterior samples via MCMC), and want to "smartly" interpolate between the conditions, i.e., learn a generative model which walks along "generalized "geodesics in the space of probability measures. I also like that the authors were very rigorous in their theorems and motivation for the developed algorithm.

### Weaknesses
However, the glaring weakness is that there is not clear cut numerical use case shown. I would like to see a not toyish example where we actually need several conditions and the transport between them. Usually, in the classical inverse problems works there is an implicit geodesic path taken where $y_t = t y + (1-t)y$, since one does not need to alter the condition if posterior sampling is the ultimate goal. If one wants to do style transfer (which seems to be the second motivation of this paper), then one can simply use a conditional FM network which receives the two conditions (source and target) as inputs. Therefore, while theoretically neat I am not convinced of why the generalized continuity equation and a network which moves efficiently also in the condition space, is advantageous. The authors can convince me by providing a clear example where either i) the classical conditional algorithms are not applicable or ii) this approach significantly outperforms the other flow matching models. 

I also have some smaller concerns. 

1) The scaling in $N_c$ and condition dimension seems to be bad. can you provide the run times for the molecular example also for the baselines? it only says in the appendix that they were completed within 4 hours, but I expect the baselines to train much quicker. Also latent space of a VAE is pretty low dimensional. Please provide training your conditional flow matching model on MNIST (no VAEs..), where the condition space is not discrete (i.e., for instance inpainting). Even if this does not fit your motivation, I would like to see the results in such a more standard example and this would improve my confidence in the scalability. 

2) Appendix D5 and F are empty (or almost empty). 

3) you do not seem to provide any code. I find the algorithm description to be not perfectly clear, there I would very strongly suggest that you at least publish code for the toy example. 

4) I believe that the example 7.1 is meaningless. You construct a random example with sparse conditions. Then you show, that your algorithm performs better on the OOD. But basically you can construct an inverse problem which aligns with your in distribution posteriors and does anything else on the OOD data. Of course I am aware that your point is that your algorithm is minimizing the Dirichlet energy and you measure the distribution induced by this. However, it is not clear to me if this is the theoretically optimal thing to do (wrt to Wasserstein). I am guessing that your algorithm computes something like Wasserstein barycenters weighted by some distance to the known conditions? Please clarify why the minimization of the generalized Dirichlet energy should yield theoretically sound posteriors. 

5) The manuscript is sloppy at times when discussing related work. "The authors in (Wildberger et al., 2023; Atanackovic et al., 2024) developed FM-based models to estimate the posterior distribution when the prior distribution p(c) of conditions is known. In contrast, our approach tackles situations where the conditions can only be sparsely observed, and the prior distribution is unknown."

The prior distribution p(c) is not known in (Wildberger et al, 2023). They are only able to sample from the joint distributions (c,x), but this does not mean that you can evaluate it. Further, their algorithm can very easily be adapted to the setting you described. If one has posterior samples for sparse conditions $c_i$ one can simply do the joint training over $(x_{i,j}, c_i)$.

6) when style transfer is one of the main modes of motivation, I would also like to see an example of it. 

Overall, I appreciate the idea and think that it has merits, but the execution prevents me from accepting it in the current form. I would love to see a practical example, where the main motivation of your algorithm becomes clear. Furthermore, providing a more standard inverse problem on MNIST (with no encoder/decoder) and a continuous condition space would show me that your algorithm at least somewhat scales.  If these problems are discussed/solved, then I am willing to raise my score.

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
To achieve extrapolation beyond observed conditions, the authors proposed Extended Flow Matching (EFM) framework that is developed upon the Conditional generative modeling. Specifically, the authors introduced a novel algorithm called MMOT-EFM derived from the Multi-Marginal Optimal Transport (MMOT). In the experiments, the authors showed improved MAE over compared FM-based methods.

### Strengths
1. The Extended Flow Matching sounds novel and the authors show the newly introduced conditional components in Fig.1, which is quite intuitive.

2. I like the well-structured theoretical discussion from FM to EFM, this can help domain experts grasp the main contribution and difference between the existing OT-CFM and the proposed MMOT-EFM

### Weaknesses
1. I feel concerned about the experimental design. For instance, the authors introduce a rather usual setting (Appendix 1300-1306). Though it aligns well with the synthetic point cloud experiments, it is quite different from the common practice [1]. 

[1] Ketata M A, Gao N, Sommer J, et al. Lift Your Molecules: Molecular Graph Generation in Latent Euclidean Space[J]. arXiv preprint arXiv:2406.10513, 2024.

2. I think critical experiments against highly related OT-CFM methods are missing in this version.  

Alexander Tong, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Kilian
Fatras, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models
with minibatch optimal transport. arXiv preprint 2302.00482, 2023b.

### Questions
1. Could you please justify the ZINC-250k experimental design?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an extension to flow matching to conditional generation on unknown (but related) conditions using a flow on both the data space and the condition space. A variant of this based on multi-marginal optimal transport is proposed as an extension to optimal transport conditional flow matching. 2D and conditional molecular generation experiments are performed showing conditional generation.

### Strengths
* Understanding how to extend current generative models to more general conditionals (especially unobserved conditionals) is an important problem particularly in the sciences. 
* I enjoyed the symmetry of the presentation of first standard flow matching and OT-CFM settings followed by EFM and MMOT-EFM settings. Table 1 is great to understand the difference to OT-CFM. 
* To the best of my knowledge the theory is correct and answers some of my questions on how one might generalize flow matching to condition-augmented spaces.

### Weaknesses
* It would be great to make clearer to the reader how this method extends to unseen conditions. I think lines 402-405 kind of get at this, but I would have loved to see more emphasis on this point. It is very easy to design a conditional generative model that technically extends to unseen conditions, but it is much more difficult to enforce that that model extends in a reasonable way. EFM has the potential to guide that extension and I would love to see that point explored further.
* The algorithm is not yet useful in real applications. While the authors also acknowledge this, it’s still a large limitation of the impact of this work. The molecule experiment is extremely limited in terms of comparisons to existing work and overall training setup.
* Much of the theoretical statements are direct extensions from prior work.

### Questions
When is MMOT-EFM and EFM in general expected to work better than COT-FM / Bayesian-FM? I know there is a short explanation on the differences in assumptions but it is difficult for me to translate what is gained when making a piecewise continuous assumption on p(x|c) vs. a measurability assumption. It’s not clear to me how this compares to these prior works in general.

Small comments that don’t affect the score: 
There appears to be an unfinished section D.5 in the appendix. 
GG-EFM isn’t defined in the main text. 
I didn’t understand the distinction between p_c and p_{0,c} line 170. 
Typo on line 311 “ton he”
Shr\”odinger to Schr\”dinger line 425
The source points in Figure 4 b and c (and corresponding appendix figs) are essentially invisible (grey against a grey background). It would be **really nice** to fix this. 


### Overall
I think this work presents an interesting idea with promise to understand how these models generalize to unseen conditions. However, this is not explored theoretically. In addition the current method does not scale to practical settings at the moment. I think further investigation as to when the assumptions behind this method make sense relative to other methods would greatly strengthen this work. A better understanding of how this relates to prior literature and when this method is preferable would likely change my opinion of this work.

### Soundness
3

### Presentation
3

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
Flow matching can generate different data distribution given different desired property conditions. The authors proposed the extended flow matching (EFM) which introduces a continuous mapping from a joint continuous space of time and property conditions to corresponding data distribution, which enables smooth shifts between different conditions. The authors also extended optimal transport to Multi-Marginal Optimal Transport (MMOT) for multiple property conditions. They validated their method on a 2D toy model and conditional molecular generation.

### Strengths
The theory of integrating property conditions and time in flow matching is highly innovative, and the authors developed MMOT to perform optimal transport within this space.

### Weaknesses
The experimental evidence is insufficient.

### Questions
Major:

1.	Could the authors explain or give an intuition about the regression in MMOT (Eq. 3.4)?

2.	Could the authors show the extrapolation ability of their methods in a more realistic application of EFM, e.g. style transfer of images?

Minor:

1.	At the end of Line 311, “focus on the” is misspelled as “focus ton he”.

2.	“ConvHull” should be explained.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose extended flow matching (EFM) for conditional sampling and style transfer using flow matching. EFM consists of 

1. learning a field which also uses the conditioning vector $c$ as input, which the authors call a matrix field. 
2. The authors then integrate the learned field $u(x, t, c)$ along different paths $\gamma: [0, t] \rightarrow [0, 1] \times C$, where $C$ is the set of conditioning vectors. 
    1. For instance, for conditional generation the authors propose integrating along the path $\gamma(t) = (t, c)$, which reduces to conditional flow matching. 
    2. For style transfer, the authors integrate along the path $\gamma(t) = (1, (1-t) c_1 + t c_2)$. Since integrating along $\gamma(t)$ can be out of domain for models learned trained just on pairs $x, c \sim p(x, c)$, the authors propose a learning algorithm such that the field $u_\theta$ also observes such paths during training. 

The authors propose learning such a field $u$ using optimal transport:

1. the authors propose learning an optimal plan similar to [Lipman et al 2023]
2. instead of using linear interpolation between different points on a path, the authors extend the set of paths to include functions belonging to an RKHS.

### Strengths
The authors identify an interesting problem: observing the conditioning vector in a number of domains can be hard or expensive. The proposal of integrating along paths between different marginals is also interesting, a similar proposal is studied in [Albergo et al 2023].

### Weaknesses
1. While the motivation of EFM was to provide ensure that the learned network $u(x, t, c)$ is smooth with respect to the conditioning vector $c$, the authors do not address how imposing smoothness can allow extrapolation to conditioning vectors not seen during training. 
2. Could the authors explain why the multi-marginal optimal transport approach allows for extrapolating to conditioning vectors not seen during training?
3. The authors should also consider including other works that learn multi-marginal flow models? For instance, [Albergo et al 2023] propose learning multi-marginal flows and present a learning algorithm for optimizing the paths such that the transport cost in $W_2$ metric is minimized. 
4. [Albergo et al 2023] also propose a much more general algorithm for including paths between samples from an arbitrary number of marginal distributions, available during training. 
5. The experiments section can be improved by adding extra text explaining the results and the figures, particularly in figure 4.


[Albergo et al 2023]  Albergo, M.S., Boffi, N.M., Lindsey, M. and Vanden-Eijnden, E., 2023. Multimarginal generative modeling with stochastic interpolants. arXiv preprint arXiv:2310.03695.

### Questions
1. Can the authors consider providing definitions before introducing a new notation in the text?
2. What is the effect of defining $\pi$ using plans built using batched samples? Would the vector/matrix field learned change as a function of the batch size? 
3. What kernels do the author use for the RKHS used to construct paths?
4. In lines 212-214 and lines 220-222, can the authors clarify the output of $u$?
5. the discussion about the weak assumption of measurability and continuity of $p(x|c)$ with respect to $c$ requires clarification, particularly since piece-wise continuous functions are measurable as well.

### Soundness
1

### Presentation
1

### Contribution
1
