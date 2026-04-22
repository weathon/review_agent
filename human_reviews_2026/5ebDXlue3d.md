# Almost Bayesian: Dynamics of SGD Through Singular Learning Theory

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
The nature of the relationship between Bayesian sampling and stochastic gradient descent in neural networks has been a long-standing open question in the theory of deep learning. We shed light on this question by modeling the long runtime behaviour of SGD as diffusion on porous media. Using singular learning theory, we show that the late stage dynamics are strongly impacted by the degeneracies of the loss surface. From this we are able to show that under reasonable choices of hyperparameters for vanilla SGD, the local steady state distribution of SGD (if it exists) is effectively a tempered version of the Bayesian posterior over the weights which accounts for local accessibility constraints.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies the dynamics of SGD algorithms using as tools the singular learning theory. The dynamics of SGD can be approximately modeled using a Langevin SDE, with the exception that the dynamics are super-diffusive during early training and sub-diffusive afterwards, which the authors model using the time fractional Fokker-Planck equation. To capture the local geometry, the local learning coefficient (LLC) and the spectral dimension are utilized, with the LLC essentially acting as a mass dimension and the spectrial dimension determining the reachable area. The resulting stationary state of the Fokker-Planck equation was derived, demonstrating the connection to the Bayes posterior.

### Strengths
1. The theoretical analysis is rigorous.
2. Empirical evidence verifies some theoretical claims.

### Weaknesses
1. One of the main claim of the paper is the connection of the stationary distribution of SGD dynamics to the Bayes posterior, yet this was not empirically validated.
2. I found limited background provided on singular learning theory; I think this would be beneficial for readers not that familar with singular learning theory.

### Questions
I am not familar with the topic of this paper, as such I do not have strong opinions on it. Please see the Weaknesses listed above.

Minor:

Line 065: to related -> to relate

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper analyses the behaviour of SGD for NN optimization using concepts from singular learning theory. The key result is a connection between fractal dimension and local behavior of SGD as quantified by a local learning coefficient (LLC), and another one is a link between the diffusive process induced by SGD and a tempered version of the Bayesian posterior, with higher concentration for parts of the parameter space that are easier to reach from an initial condition. The theory is validated using relatively simple empirical experiments that validate the theories related to fractal dimension and LLC.

### Strengths
The research question is important: SGD is very widely used and the previous theoretical analysis of its properties relies on simplifying assumptions, some of which are relaxed here, and hence there is clear need for improved understanding. Even though the paper uses somewhat heavy theoretical machinery that may not be accessible for all readers, the key results are understandable for the broader community. I am not an expert with this range of theoretical tools myself, and hence cannot directly evaluate the significance of the contribution, but I find the concrete results convincing.

The empirical experimentation is well justified, with clear explanation of why the experiment is conducted like it was. It is on a small scale, but sufficient for shedding light on the theory.

### Weaknesses
It is somewhat difficult to grasp the importance of the spectral dimension, as it is not directly linked to quantities people usually consider. In effect, one needs to be already familiar with Lau et al. (2024); this is not directly a limitation as it is valuable to build on the LLC measure, but the paper would work better as a stand-alone piece if LLC was explained and illustrated a bit more detail here.

The empirical validation is limited to confirming the bounds of Lemma 3.4 and Corollary 3.2 that are both about the spectral dimension. As someone who found Corollary 3.1 a more tangible contact point I would have liked to see something brought from Appendix C to the main paper. I understand the choice though, as that result is more difficult to communicate.

### Questions
What can a practitioner learn about your work? Would it be useful to track the spectral dimension in some scenarios and use the information to guide some design choices? What would we gain over tracking LLC? I am very much willing to adapt my score once understanding better the practical significance of the results, but would like to remark that the paper could well be published even as a purely theoretical paper.

The estimated LLC is effectively constant for spectral dimensions between 0 and 15 in both Figure 2 and 3. Why is this? Is the lower bound overly loose for low spectral dimensions, or is this near-constant behavior explained by something else?

### Soundness
4

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
2

### Summary
The paper models long-time diffusion dynamics of SGD using the fractional Fokker–Planck equation (FFPE), which can describe sub-diffusive behavior induced by degenerate loss landscapes on a fractal geometry. The authors derived the specific solution of the steady-state distribution of FPPE based on a defined effective diffusion coefficient. They connected it to a tempered Bayesian posterior, modulated by local accessibility constraints determined by the local learning coefficient (LLC). Moreover, they demonstrated that the spectral dimension of SGD is bounded by the fractal dimension, and empirically verified this on MNIST.

### Strengths
1. This paper employs fractional SDE and spectral dimension analysis in a coherent framework, which introduces a novel insight to bridge the gap between SGD and Bayesian methods. All lemmas and corollaries are provided with detailed proofs.
2. The idea of modeling anomalous diffusion using the FFPE basis grounded in fractal geometry is reasonable and effective.

### Weaknesses
1. The experimental validation only utilizes the small MNIST and CIFAR datasets without larger-scale input, which limits the empirical generality of the proposed inequality $d_{s} \leq \lambda(w(t))$.
2. Experiments use only vanilla SGD without momentum or adaptive variants (Adam, RMSProp). It is unclear whether the proposed fractional diffusion framework can be extended to these more practical optimizers.
3. There is no ablation study about how hyperparameters (e.g., batch size, weight decay, epochs) affect the spectral dimension or LLC evolution.

### Questions
1. To assess the empirical fidelity of the “almost Bayesian” claim, can you quantify how closely the steady-state distribution $p_{s}(w)$ matches a true Bayesian posterior of SGLD samples as in Figure 5 (e.g., via KL divergence / Wasserstein distance)?
2. Have you attempted to compute LLC or spectral dimensions on larger or more complex networks (e.g., CNNs on ImageNet)?
3. The fractional Fokker–Planck framework only focuses on sub-diffusion. How would the model extend to super-diffusive or Lévy-like early dynamics, and would this modify the steady-state form?
4. Can you provide any quantitative evidence (e.g., LLC temporal stability plots) supporting the near-stability hypothesis 3.1 during late training?

### Soundness
2

### Presentation
2

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
This paper proposes a theoretical connection between stochastic gradient descent (SGD) dynamics and almost Bayesian behavior, by analyzing the steady-state of SGD through the lens of fractional diffusion and spectral geometry. The authors introduce the concept of local learning capacity (LLC) as an analog of an inverse temperature, and relate it to the spectral dimension of the loss landscape via a family of scaling laws. They provide a mathematical derivation of these relations and test them empirically on toy data and small-scale MNIST experiments.

### Strengths
- Ambitious conceptual link between stochastic dynamics, spectral geometry, and Bayesian ideas
- The LLC notion is at least intuitively appealing and might motivate future work on quantifying local "temperature" or effective capacity in SGD
- Some of the derivations are mathematically nontrivial and show that the authors know the literature on stochastic processes and singular learning theory
- The paper is generally readable and nicely typeset, with good figures (especially Fig. 1 for intuition)

### Weaknesses
- **Steady-state assumption**: many derivations rely on SGD reaching or approximating a steady-state distribution. The authors mention that this may not actually exist or be reachable in practice, but then they still use it for all the main results. This feels like a big logical gap that should be discussed much more clearly.
- **Notation and definitions**: this part is honestly confusing.  
  - V(ε) is first defined as a function of ε (Eq. 5), but later it takes a ball B(w*, ε) as argument (Eq. 7). It’s unclear if that’s the same V or a new one.  
  - The dependence on ρ and r is dropped mid-derivation.  
  - “walker dimension” vs. “walk dimension”: these seem to be the same thing, just inconsistent terminology.  
  - Definition 3.3 really reads like a derived result (it relates quantities already defined before), so calling it a definition doesn’t make sense. It should be a theorem or corollary, or at least justified.
- **Assumptions**: The “near-stability hypothesis” (Hyp. 3.1) is just stated but never tested. Likewise, the use of the Alexander–Orbach relation assumes homogeneity that is unlikely to hold for neural loss landscapes. There’s no discussion of what happens if it fails.
- **Experiments are extremely limited**:  
  - Only a toy Moons example (few hundred points) and small MNIST subset are shown.  
  - No error bars, no statistical tests, no correlation coefficients... so the claim that LLC correlates with the spectral dimension is just a visual impression.  
  - The authors even note that the bound is loose, so it’s unclear if the theoretical quantities actually predict anything.  
  - There are no ablations or sensitivity studies (learning rate, batch size, the coarse-graining scale ξ, etc.).
- **Practical relevance unclear**: given the weak and small-scale experiments, it’s hard to see what we should take away practically. The theory could still be interesting, but it would help to discuss possible uses or ways to estimate LLC in large-scale setups.
- **Minor stuff**: lots of typos (e.g., “Kullback-Liebler”, “whcih”) and inconsistent citation style ((double parentheses)) all over the place. Would be good to fix that.

### Questions
- Can you clarify the dependence of V on ρ and r in Eqs. 5–7? It currently looks like it depends on those implicitly, but the notation hides it. Also, in Eq. 6, you say λ is defined for an arbitrary choice of a – does that mean λ is independent of a? If yes, can you show why?
- Why is Definition 3.3 a definition and not a theorem? It seems to express a relation between previously defined quantities, so it’s not really definitional. Please clarify or justify.
- The Alexander–Orbach relation is usually derived for homogeneous fractal media. Under what assumptions does it hold here, and how do you know those hold for your loss landscapes?
- How exactly is the scale ξ chosen for D_ξ in Eq. 13? Is it a fixed number, dataset dependent, or tuned? Have you checked sensitivity to it?
- You introduce the “near-stability hypothesis” (Hyp. 3.1). Have you tried to test this empirically (e.g., by measuring distances to nearby minima or metastable states)?
- Could you add at least some quantitative evaluation for the LLC–spectral-dimension relation? e.g., correlation coefficients, confidence intervals, or plots across architectures or datasets. Right now, it’s impossible to tell how strong the relation is.
- Please fix the citation formatting (use \citep and \citet correctly) and typos throughout.

### Soundness
2

### Presentation
2

### Contribution
3
