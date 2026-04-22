# Continuum Transformers Perform In-Context Learning by Operator Gradient Descent

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Transformers robustly exhibit the ability to perform in-context learning, whereby their predictive accuracy on a task can increase not by parameter updates but merely with the placement of training samples in their context windows. Recent works have shown that transformers achieve this by implementing gradient descent in their forward passes. Such results, however, are restricted to standard transformer architectures, which handle finite-dimensional inputs. In the space of PDE surrogate modeling, a generalization of transformers to handle infinite-dimensional function inputs, known as "continuum transformers", has been proposed and similarly observed to exhibit in-context learning. Despite impressive empirical performance, such in-context learning has yet to be theoretically characterized. We herein demonstrate that continuum transformers perform in-context operator learning by performing gradient descent in an operator RKHS. We demonstrate this using novel proof strategies that leverage a generalized representer theorem for Hilbert spaces and gradient flows over the space of functionals on a Hilbert space. We further show the operator learned in context is the Bayes Optimal Predictor in the infinite depth limit of the transformer. We then provide empirical validations of this result and demonstrate that the parameters under which such gradient descent is performed are recovered through pre-training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies in context learning abilities of infinite-dimensional generalizations of the transformer architecture, called continuum transformer and previously introduced by Calvello et al. The authors extend previous finite-dimensional work by Cheng et al. showing that transformers implement gradient descent in-context to infinite dimensional settings for operator learning. The main result is that continuum transformers perform in-context learning by performing gradient descent in an RKHS, under a careful choice of the architecture. The authors conclude by providing numerical experiments on simple operator learning tasks that validate their theoretical findings.

### Strengths
1. Despite the technicalities, the paper is well written and rigorous.
2. The topic is timely and relevant, especially as transformer-based architectures are becoming increasingly popular in the field of operator learning with the apparition of foundation models.
3. The main theoretical results are interesting and bring new insights on the in-context learning abilities of transformer architectures in infinite-dimensional settings, and may lead to more practical considerations regarding the choice of the architecture.
4. The numerical experiments, although simple (linear operators), illustrate the theory.

### Weaknesses
1. Given the formulation of Thm 3.1, it is quite difficult to interpret the result in practical terms. I am wondering whether, instead of framing it as an existence result for the architecture, it would be more natural to introduce the kernel induced by the architecture. The result also makes a lot of assumptions on the weight matrices. Can the authors comment on these assumptions and whether they imply that one should fix these weights in practice during training?
2. The two assumptions 3.4-3.5 seems quite specific, it would be helpful to have some discussion on how reasonable they are in practice.
3. It seems counter-intuitive to me that one does not need to make any assumption on the operator $\tilde{G}$ that one tries to approximate. Could the authors comment on this?
4. The numerical experiments are performed for a very synthetic task (Laplacian kernel), and would benefit from a more challenging operator learning task (e.g. a time-dependent PDE, or nonlinear problem is the theory extends to nonlinear operators).

### Questions
1. In Eq. (1), what measure do you take on the space of functions U?
2. In Eq. (9), shouldn't one consider a relative loss by normalizing by $\|u\|_{n+1}$? Does the analysis extend to this case?
3. I don't think $\mathcal{O}$ is defined before line 209
4. Theorem 3.1: I can't find the definition of $\tilde{H}$ in the main text before Eq. (7). I see it defined in Eq. 17 but it would be helpful to include it in the main text given that it is crucial for the theorem.
5. I guess Prop 3.3 implicitly assumes that the underlying operator is linear so that the conditional distribution of the output is Gaussian? Is it not a strong assumption to assume that the Gaussian kernel is exactly the same as the kernel induced by the architecture?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a rigorous theoretical analysis of continuum transformers, the function-space analog of standard transformers, when applied to operator learning problems, such as those arising in PDE modeling. The authors show that:

- If the query, key, and value operators are chosen appropriately, each layer of a continuum transformer performs one exact step of gradient descent in an operator-valued RKHS

- As the number of layers tends to infinity, the transformer’s output converges to the Best Linear Unbiased Predictor, equivalent to the Bayes-optimal estimator of a Hilbert-space Gaussian process.

- When the model is trained end-to-end, its parameters naturally converge to the same fixed-point configuration that realizes the gradient-descent dynamics

Empirical studies are limited but targeted:

- The authors validate the gradient-descent interpretation when the attention kernel matches the data-generating kernel.

- They train a 250-layer continuum transformer on synthetic Gaussian-process data to verify the convergence of learned query/key operators toward the predicted fixed-point structure.

The paper is primarily theoretical but provides convincing toy experiments supporting the analytical results.

### Strengths
- Originality: Extends the “transformer = gradient descent” theory from vector spaces to operator-valued RKHSs, a nontrivial and conceptually elegant generalization.

- The mathematical treatment is rigorous and self-contained, including a generalized Representer Theorem for operator-valued kernels.

- The results provide a clear mechanistic interpretation of in-context learning for continuum transformers.

- Toy experiments (BLUP matching and operator-alignment studies) convincingly illustrate the theorems.

- The work strengthens the theoretical foundation of operator learning and may guide better architectures for PDEs

### Weaknesses
- Lack of practical tokenization discussion.

It remains unclear how tokenization is defined in PDE-learning settings. While a “next-token prediction” formulation is appropriate for fixed time-stepping tasks (e.g., 6-hour weather forecasts), such an in-context learning paradigm does not naturally extend to continuous-in-time scenarios where predictions are required at arbitrary temporal resolutions. Moreover, most PDE and computer-vision problems require some form of spatial tokenization to capture local correlations across the domain (e.g., patchifying a sample function). In this functional context, the notion of “next-patch prediction” becomes conceptually ambiguous and difficult to formalize.

- Limited empirical scope.

All experiments are synthetic. The theory’s applicability to real PDE problems (e.g., weather trajectory generation or high-resolution dynamics) is not demonstrated.

- Scalability concerns.

It is unclear how the proposed theory would extend to realistic, large-scale problems, such as generating full weather trajectories. In such settings, deploying a 250-layer network (as used in Section 5.3) would be computationally impractical. It would therefore be valuable to analyze how the average similarity score evolves as the model depth increases, from shallow configurations (1–10 layers) through intermediate (10–50) to deep regimes (50–250). My expectation is that the network would not reach the exact fixed point predicted by Theorem 3.6, but would likely approach it asymptotically.

- Dense notation.

The presentation could benefit from more intuition and conceptual figures summarizing main theorems and propositions.

- Lack of clarity on computational and memory complexity.

The continuum formulation replaces matrix multiplications with integral operators, but the paper does not discuss the resulting computational or memory scaling. It is unclear whether the operator-valued attention can be implemented efficiently in discretized settings or whether it incurs quadratic cost in the number of spatial degrees of freedom. A short discussion comparing computational requirements with standard transformer or neural-operator architectures would strengthen the paper’s practical relevance.

### Questions
- How do you envision “tokens” being defined in PDE settings? Are they spatial patches, basis coefficients, or entire functional evaluations?

- Could your analysis extend to models queried at arbitrary times (e.g., continuous-in-time diffusion or flow prediction)?
	​
- Would the cosine-similarity convergence in §5.3 persist for shallower models? A quantitative study varying the number of layers could clarify practical limits.

- Do you expect the same theoretical behavior when training on PDE datasets like Navier–Stokes or Darcy flow, where discretization and boundary conditions break some of the theoretical assumptions?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper lifts the in-context learning approach from finite dimensional transformers to continuum transformers acting on function spaces. Replacing softmax with a PSD operator valued kernel, each layer implements a gradient descent step for the in-context least squares objective in an Operator Valued Reproducing Kernel Hilbert space (OVRKHS) and depth corresponds to more steps. Under a $\kappa$-Gaussian distribution and matched $\kappa$ attention kernel, in the infinite depth limit the method equals a Bayes optimal predictor (BLUP/kriging). A pre-training analysis, under strong assumptions, shows that whitening Q/K and considering a residual injecting V forms  stationary solutions, suggesting that the model can learn parameters in a GD manner. The experiments validate the theory for a simple test.

### Strengths
- The paper presents a non-trivial extension of in context leaning to an operator valued, continuous, setting. The math is elegant have technical substance and can be used for further analysis.

- Considering operator valued kernels for this problem and their representer theorem is elegant and makes defining each layer as one gradient descent step clean. The analysis for the fixed point pre training and the connection to BLUP is nice despite the strong assumptions.

- The attention, the values and how each part of the method is constructed is clearly stated. 

Overall, I find this paper to be mathematically elegant and insightful.

### Weaknesses
- The Bayes optimality/BLUP limit require the data distribution to be $\kappa$-Gaussian and also an exact kernel match. In practice, the input functions come from non Gaussian distributions and the kernels might be misspecified. 
    
-  Rotational symmetry and equivariance are strong assumptions. Correct me if I am wrong but I believe that a global invariance in whitened coordinates might not hold due to anisotropies or boundary conditions. 

- All the proofs are performed in the function space, however the implementations in practice are discretised. Maybe there should be an analysis linking the continuum to the discrete implementation and what errors this induces. 

- Operator valued kernels can be expensive. The paper bypasses that using a separable form trick, scalar similarity * cheap operator, and tiny prompts however a discussion should be presented on estimating rich kernels and larger contexts. 

- The theory requires a PSD and symmetric kernel which standard softmax is not. So, what the authors use is not technically an attention kernel because it is not row normalized. This is not that clear. I suggest to the authors to use the kernel construction considered in “Learning Operators with Coupled Attention” where they consider a PSD/symmetric/universal kernel that is also normalized. The authors there consider a scalar kernel I believe that this can be adapted to your method in a straightforward manner (see Micchelli “On learning vector-valued functions” and Caponnetto “Universal Multi-task Kernels”).

 - The empirical validation is very limited. You consider no PDE benchmarks, no baselines, ICON or FNO and a meta-learner, no ablations for kernel mismatch or symmetry braking.

### Questions
- How is the projection error bounded when considering a truncated spectral basis?

- Is there a way to relax the whitening assumption by maybe considering a spectral normalisation of W_q, W_k and an adaptive step size? 

- How sensitive is the fixed point analysis to broken symmetries from non-periodic BCs, irregular meshes etc? 

- Please add at least one PDE example and compare to an FNO+meta learning or ICON approach.

### Soundness
3

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
4

### Summary
This work extends the line of work showing that Transformers perform gradient descent in-context during inference to continuum Transformers (defined on function spaces) and operator learning. The main theoretical contributions are that continuum Transformers can implement operator gradient descent in a reproducing kernel Hilbert space (RKHS) of operators and that in the infinite-depth limit, the model converges to the Bayes-optimal (kriging) predictor. The authors prove this by providing a weight construction implementing operator gradient descent. Furthermore, they prove that their choice of parameters also represents a stationary point of the training loss under gradient flow over the space of functionals. They empirically validate their theoretical claims on operator regression tasks.

### Strengths
- The paper is novel and investigates an important question, probing the fundamentals of continuum Transformers for in-context operator learning. This is a problem that may be of interest to folks in the in-context learning theory and operator learning/PDE foundation model communities.
- The paper is well-written and clear. The theoretical results are interesting and well done, and the authors validate their theoretical results empirically.

### Weaknesses
As far as I understand, the work does not provide any analysis of how many functional gradient descent steps / number of continuum Transformer layers it would take to achieve convergence on the in-context operator learning problem. In my opinion, the work as it stands already provides enough results to recommend acceptance, but I would be curious to know if such results are possible (see questions).

Minor:
- The authors could more clearly emphasize the significance and limitation of the theoretical claim in Section 3.4 / Theorem 3.6. As stated, the theorem establishes that the proposed parameter configuration constitutes a stationary point (fixed point) of the training dynamics in the continuum limit, but it does not imply that this configuration is the unique or global minimizer of the loss, nor that it corresponds to the explicit solution of the gradient flow. I understand this type of result is standard in recent in-context learning theory (and I believe the result itself is valuable to the community), highlighting it explicitly would help readers interpret the scope of the claim.
For example, lines 61 and 273 could note that the parameters form a critical point, not necessarily a global minimum.
- I think the writing would be slightly clearer if the authors allocated a specific subsection in the paper for discussion of the novelty/importance of the proof techniques, rather than scattering such comments throughout.

### Questions
- As far as I understand, the theoretical results characterize the continuum transformer’s behavior in the infinite-depth setting, but do not provide any results or intuition about the rate of convergence. Would it be possible to extend the authors' analysis to provide convergence rate results (e.g. how many steps/layers are required for the in-context operator to approach the Bayes-optimal predictor)? Or alternatively, have the authors observed any empirical scaling trends relating depth to approximation quality?
- In Figure 2, the authors attempt to provide empirical validation of their fixed-point claim by evaluating the cosine similarity of the key/query operators across layers during training as a proxy. I'm wondering if it would be possible to use a similar proxy to visualize the rate of convergence of the in-context operator prediction as the model evaluates layer-by-layer? For example, by evaluating the cosine similarity or relative error between the predicted operator at layer $\ell$ and at layer $\ell-1$, or between the layer $\ell$ prediction and the Bayes-optimal operator? Such a diagnostic might more directly visualize the model prediction's evolution under the proposed functional GD.
- Is it possible to extend this analysis of infinite-dimensional prefix-ICL to the infinite-dimensional causal-ICL setting, where given $n$ ICL examples, the model is supervised on all $n-1$ subproblems (given $i-1$ examples, predict the $i$th, for all $i \leq n$) in parallel? (e.g. as in [1])

1. Bai et al., 2023. Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection

### Soundness
4

### Presentation
4

### Contribution
3
