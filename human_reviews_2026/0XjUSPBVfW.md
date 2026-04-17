# Natural gradient Bayesian sampling automatically emerges in canonical cortical circuits

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Accumulating evidence suggests the canonical cortical circuit, consisting of excitatory (E) and diverse classes of inhibitory (I) interneurons, implements Bayesian posterior sampling. However, most of the identified circuits' sampling algorithms are simpler than the nonlinear circuit dynamics, suggesting complex circuits may implement more advanced algorithms. Through comprehensive theoretical analyses, we discover the canonical circuit innately implements natural gradient Bayesian sampling, which is an advanced sampling algorithm that adaptively adjusts the sampling step size based on the local geometry of stimulus posteriors measured by Fisher information. Specifically, the nonlinear circuit dynamics can implement natural gradient Langevin and Hamiltonian sampling of uni- and multi-variate stimulus posteriors, and these algorithms can be switched by interneurons. We also find that the non-equilibrium circuit dynamics when transitioning from the resting to evoked state can further accelerate natural gradient sampling, and analytically identify the neural circuit's annealing strategy. Remarkably, we identify the approximated computational strategies employed in the circuit dynamics, which even resemble the ones widely used in machine learning. Our work provides an overarching connection between canonical circuit dynamics and advanced sampling algorithms, deepening our understanding of the circuit algorithms of Bayesian sampling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a theoretical analysis of how canonical cortical circuits, composed of excitatory (E) and inhibitory (PV, SOM) neurons, can implement natural gradient Bayesian sampling. The core contribution is the finding that, under a uniform prior, the bump height of the E-neuron population response encodes the Fisher Information (FI) of the posterior. This mechanism dynamically controls the sampling time constant of the bump's position, allowing the circuit to biologically implement Natural Gradient Langevin Sampling (NGLS). The authors further demonstrate that incorporating SOM neurons, which introduce a momentum term, upgrades the circuit to perform Natural Gradient Hamiltonian Sampling (NGHS). The work also connects other computational strategies, such as annealing and regularization, to the emergent properties of the E-I circuit dynamics.

### Strengths
1. The paper provides a comprehensive theoretical bridge between the dynamics of canonical E-I circuits and advanced Bayesian sampling algorithms.

2. It demonstrates how distinct neuron populations and their specific interactions can mechanistically implement components of these advanced sampling strategies.

3. It also offers novel insights into the computational role of non-equilibrium dynamics as an innate annealing strategy that can accelerate sampling.

4. The theoretical claims are well-supported by numerical simulations.

### Weaknesses
A major concern is that the work's core findings appear to be an incremental step over Sale & Zhang (2024). The authors acknowledge their work builds on this previous study , which had already established the circuit's Langevin and Hamiltonian sampling properties. The main new finding is that the E-neuron response height encodes the FI, thus upgrading the circuit to NGLS/NGHS. However, this critical finding is true only when the prior is uniform. This assumption limits the generality and remains a constraint on the work's overall impact.

The writing and clarity of the paper can be further improved. Some notations are a little bit confusing or not adequately explained in the main text. (E.g., $\xi$ in Eq. (1a), $p(s)$ in "We will leave the subjective prior p(s)...", and some inconsistent notations between main texts and appendix.) This makes it difficult to fully understand the theoretical analysis. Besides, although the figures are generally neat and beautiful, legends in Figure 2E&F&K are confusing. Different colors used in these figures are not explained.

### Questions
See the weakness section for my major concerns.

1. Could the authors please elaborate on the significance of their findings beyond Sale & Zhang (2024)? Given that the key NGLS/NGHS finding is constrained to a uniform prior, how much of the circuit's behavior is already captured by LS/HS models from the previous work?

2. How would the proposed mechanism (E bump height encodes FI) break down if a non-uniform prior is introduced? Would the circuit still perform some approximation of NGLS, or would the direct link between bump height and FI be lost?

### Soundness
4

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
There has in recent years been much interest in sampling-based probabilistic inference in neural networks as a model for various probabilistic computations in the brain. This paper builds on a line of work by Wen-Hao Zhang and collaborators, in particular [Sale and Zhang, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/f463d31ed2fdd7b0ec585c041ec1baa8-Paper-Conference.pdf), that aims to show that the dynamics of the location of a bump attractor state in a mean-field-style model for the canonical cortical microcircuit can sample interesting distributions. Its contribution is to try to show that the dynamics of the bump can approximate not just naive Langevin or Hamiltonian Langevin dynamics (as is done in Sale and Zhang), but natural gradient Hamiltonian Langevin dynamics, which can enable faster convergence to the target distribution.

### Strengths
This paper is timely, and its headline premise - that an efficient sampling algorithm naturally emerges from a standard model for cortical circuits - is clearly of interest to a broad computational neuroscience audience.

### Weaknesses
The biggest weakness of the paper is that, contrary to what the title and the abstract would lead the reader to believe, the authors do not actually show that their model implements full Riemannian Langevin dynamics. Rather, they show that the circuit can approximate the *diagonal* of the inverse of the Fisher information matrix, rather than the full matrix inverse. The authors try to present this as a feature rather than a bug by saying that the diagonal approximation is commonly used in machine learning, but I fundamentally think that it is not appropriate to say that the network is performing emergent Riemannian MCMC in that case (I would also note that most of the cited ML applications involve non-Gaussian distributions with non-constant Fisher information matrices, for which estimating the Fisher is much more difficult than in the Gaussian case). This is a crucial conceptual feature of Riemannian MCMC that the authors miss even when they introduce the idea of using natural gradients around equation (5): not only the size, but the structure, of the sampling noise is adapted to the geometry of the problem. This discrepancy also raises the need for some more experimental tests, which I detail under **Questions**. 

On the whole, I do not see this paper as being a sufficient conceptual advance relative to Sale and Zhang - which already shows evidence for Hamiltonian-like acceleration - given that the result is restricted to problem-adapted step sizes rather than adaptation of the structure of the metric, i.e., of the principal axes of variance of the sampling noise. I thus disagree with the overall tone ("remarkably", "unprecedentedly", etc).

### Questions
- What is the relationship of this manuscript to https://openreview.net/forum?id=BpBW4gJofo, also submitted to ICLR, with which it overlaps substantially? Even the figures look substantially similar, and the model is, so far as I can tell, precisely the same. Compare, for instance, eq.  (15) of this submission with eq. (17) of that submission. There the presentation is in terms of "adaptive step sizes" rather than approximate natural gradient, but the setting and overall scope of results are fundamentally the same. Some clarification here is required. 

- Given that the circuit implements a diagonal approximation to the Fisher matrix, I think a few more experimental probes are required. Figure 3A shows convergence rates in KL over time, but I think it is critical to test how this depends on the dimensionality and structure of the target distribution. I would expect the gap between the full Riemannian MCMC and the diagonal approximation to grow as the problem dimensionality increases and also as the different dimensions become more correlated. Looking at the cited paper by Masset and colleagues (NeurIPS 2022) that is the main related work to which the authors compare their results, those authors showed that both of those factors lead to slowdown both in linear rate networks and in spiking networks that approximate the rate dynamics. 

- The decomposition in eq. (15) seems non-standard relative to the usual way of decomposing the dynamics of a multivariate OU process into symmetric and skew-symmetric parts, as is done in the cited work of Ma, Fox, and colleagues (NeurIPS 2015). Could you also present the dynamics in that form? 

- How does the accuracy of sampling depend on the number of neurons used? From a neuroscience perspective, this is important as it determines the minimum size of each circuit module in the setup for sampling multi-dimensional distributions that the authors present. So far as I can tell, this is not probed systematically.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper extends the work of Sale & Zhang (2024) on Bayesian sampling in canonical cortical circuits to show that these circuits naturally implement natural gradient (NG) sampling algorithms. The authors propose that the canonical circuit, consisting of excitatory neurons and parvalbumin/somatostatin interneurons, can implement both natural gradient Langevin and Hamiltonian sampling. The key mechanism is that the total activity of E neurons (bump height $U_E$​) monotonically increases with the posterior's Fisher information, which automatically adjusts the sampling step size based on the local geometry of the posterior distribution. The authors demonstrate through theoretical analysis that the circuit implements NG Langevin sampling in the reduced E+PV circuit, adding SOM neurons enables NG Hamiltonian sampling, non-equilibrium dynamics during the transition from resting to evoked states further accelerates sampling through an intrinsic annealing strategy, and coupled circuits can sample multivariate posteriors using diagonal Fisher information matrix approximations analogous to techniques in machine learning.

### Strengths
1. Originality: the paper makes a theoretical contribution by identifying that canonical cortical circuits naturally implement natural gradient sampling, extending beyond the naive Langevin and Hamiltonian sampling identified in Sale & Zhang (2024). The connection between E neuron bump height and Fisher information provides a novel functional interpretation of circuit dynamics. The identification of computational approximations in the circuit (regularization via recurrent connections, diagonal FIM approximation) that parallel ML techniques is conceptually interesting.
2. Quality: the mathematical analysis is rigorous and follows established methods for analyzing continuous attractor networks. The authors provide detailed perturbative analysis, eigenmode decomposition, and explicit mappings between circuit dynamics and NG sampling algorithms. The derivations connect circuit parameters (bump height $U_E$​) to Fisher information and sampling step sizes in a principled way.
3. Clarity: the paper is generally well-structured. The progression from naive sampling to NG sampling is clearly explained, and the figures are effective in illustrating the main concepts, in particular showing how bump height scales with Fisher information and how this determines sampling time constants. The supplementary materials provide thorough derivations.
4. Significance: the potential unification of natural gradient sampling within the canonical circuit architecture is conceptually appealing. The model makes concrete predictions about the relationship between E neuron population activity, Fisher information, and sampling efficiency. The connection to ML approximation strategies (regularization, diagonal FIM) bridges neuroscience to machine learning.

### Weaknesses
1. I am not sure about the computational necessity of sampling in the circuit model if not to encode posterior uncertainty. The authors state that "a single snapshot of $r_F$ parametrically conveys the whole stimulus likelihood" (Eq. 8), meaning a population vector readout is sufficient. Given this, it's unclear what computational advantage NG sampling provides over simpler population coding schemes like probabilistic population codes (PPC), which can also perform Bayesian inference with linear readouts but without the complexity of maintaining sampling dynamics. Since the posterior uncertainty is entirely determined by the feedforward input rate $r_F$ (which controls likelihood precision $\Lambda$), the neural variability does not represent posterior uncertainty in the way that sampling-based models typically propose. For the Gaussian likelihoods and uniform priors assumed in this framework, deterministic inference methods (like direct computation of the posterior mean and variance) would be exact and more efficient. The authors do not provide quantitative comparisons of computational costs, convergence speed, or accuracy against such alternatives.
2. Moreover, restrictive assumptions limit the generality of this approach. The framework relies heavily on several assumptions. First, the model assumes Gaussian feedforward tuning curves (Eq. 1e) leading to Gaussian likelihoods (Eq. 8). However, real sensory likelihoods are often non-Gaussian and multimodal, and one of the purported strengths of sampling-based approaches is that they can represent arbitrary distributions. The authors do not address how the circuit would handle non-Gaussian inference problems. Second, the model assumes a 1D ring attractor, and its specific eigenmode structure is essential for the perturbation analysis. While the authors show a 2D extension (Fig. A4) that couples two ring attractors, this is still a rather restrictive latent structure which presumably not all canonical circuits possess, and how to scale to higher-dimensional feature spaces without this specific structure remains unclear. Finally, the analysis uses uniform priors throughout. This eliminates one of the key computational challenges of Bayesian inference, which is to show that the prior can reflect the statistics of its inputs. The authors do not demonstrate that the circuit can implement informative non-uniform priors or flexibly switch between different prior distributions.
3. The circuit already receives critical information that undermines claims of adaptive inference. Specifically: the feedforward input $r_F$ directly encodes the likelihood mean $\mu_z$ and precision $\Lambda$ (Eq. 8), meaning the posterior parameters are essentially pre-computed and fed into the circuit rather than inferred. Moreover, while the authors claim the circuit "adaptively adjusts the sampling step size," this adaptation is entirely driven by the feedforward input intensity $R_F$, which is externally provided rather than computed by the circuit itself. The recurrent weights ($w_{EE}$, $w_{SE}$, etc.) are also precisely tuned to match the required sampling parameters (Eqs. 11, Fig. 4E), but the authors do not mention how these precise weight configurations would be acquired or whether synaptic plasticity could maintain them.
4. The authors briefly mention that deterministic inference circuits require "complicated nonlinear functions," but this claim is specific to their problem structure (Gaussian distributions, linear-Gaussian dynamics). However, they do not provide quantitative comparisons of computational costs relative to deterministic approaches, convergence speed relative to standard (non-NG) sampling, accuracy trade-offs between their approach and alternatives, and performance relative to the single previous natural gradient sampling circuit study (Masset et al., 2022). The claim that NG sampling provides advantages over naive sampling is not substantiated with systematic quantitative comparisons.
5. There is limited biological justification for key mechanisms in the model. While the circuit architecture is based on known connectivity patterns, the assumption that recurrent E weights ($w_{EE}$) act as a regularization parameter (analogous to $\alpha$ in Eq. 5) is mathematically convenient but lacks biological justification. How would the circuit "know" to set this weight to prevent numerical instabilities in Fisher information inversion? Moreover, the non-equilibrium annealing strategy is presented as an "emergent property," but the functional advantage of this particular annealing schedule over other possible dynamics is not demonstrated.

### Questions
1. How would the circuit handle non-Gaussian likelihoods, which are common in real sensory processing? Can you provide numerical experiments or extensions showing the framework handles cases where the Gaussian assumption breaks down?
2. Given that the likelihood can be read out with a population vector (linear decoder) from $r_F$, what specific computational advantages does NG sampling provide over probabilistic population codes (PPC) or direct computation of posterior parameters? Can you provide quantitative comparisons (e.g. in terms of inference accuracy or speed, or computational cost)?
3. For Gaussian likelihoods and uniform priors, the posterior is also Gaussian with analytically computable parameters. What is the computational advantage of approximating this via sampling when exact solutions are available? Under what conditions would sampling-based inference be preferred?
4. How would the circuit acquire the precise weight configurations required for NG sampling (e.g., relationships in Eqs. 11, 14, Fig. 4E)? Do you propose these synaptic weights are learned, and if so, through what learning rule?
5. The claim that recurrent E input acts as regularization (like $\alpha$ in Eq. 5) is interesting, but how does the biological circuit "know" to set $w_{EE}$ to prevent numerical instabilities in Fisher information inversion? What mechanisms would maintain this relationship as environmental statistics change?
6. In multivariate cases, the circuit uses diagonal FIM approximation rather than full FIM. Can you quantify the loss in performance relative to full FIM?
7. How does the NG sampling in your circuit compare to naive Langevin/Hamiltonian sampling, deterministic inference, and the NG sampling circuit in Masset et al. (2022)? I would appreciate specific metrics like convergence time, sample efficiency, and accuracy.
8. The non-equilibrium annealing is claimed to accelerate sampling (Fig. 2K), but by how much compared to equilibrium sampling? How does this depend on circuit parameters?
9. How sensitive is the NG sampling to mismatches between the assumed circuit parameters and the true parameters? For example, what happens when $w_{EE}$ deviates from the value required for proper regularization, or when Fisher information is estimated incorrectly?
10. Can the circuit handle time-varying Fisher information (e.g., if the feedforward input statistics change over time)? How quickly can the circuit adapt its sampling strategy to new posterior geometries?

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
5

### Summary
This paper analyses the dynamics of E-I circuits (including two classes of I neurons, PV and SOM) with ring architectures, and through extensive mathematical derivations and also numerical simulations shows that they perform particular forms of sampling (natural gradient Langevin and Hamiltonian) from simple (mostly 1D Gaussian) distributions.

### Strengths
The fundamental goal of relating neural circuit dynamics to (sampling-based) inference is interesting, and there is a lot of nice ideas for working out such a relationship (e.g. the overall magnitude of neural activities acting as regulator of step size, or the putative role of SOM neurons in sampling). The mathematical analyses are extensive, and the paper is generally well written.

### Weaknesses
1. There are a number of seemingly arbitrary choices in model construction:

- Why is PV vs. SOM cell-mediated inhibition taken to be global, divisive (in firing rates), and instantaneous vs. local, subtractive (in synaptic inputs), and finite time-scale (referring to both the time constants of the I cells, and their effects on E cells), respectively? (E.g. if anything, I would have thought PV inhibition is more local than SOM is.)

- Why are only E but not I cell synaptic inputs noisy?

2. Overall, the circuit is shown to be able to sample from a 1D Gaussian (or the 1D marginals of a 2D Gaussian, see below). Sampling becomes particularly useful for high dimensional joint posteriors, and with more complicated distributions. It's unclear if such more challenging forms of sampling can be solved by this approach (although the Discussion does mention some future directions in this regard). Specifically, the Gaussian represents a special case for NG Langevin sampling, in which the FI is constant, and as such, equivalent to just changing the (effective) time constant. So it's unclear if the circuit actually continues to accurately approximate NG Langevin in the more general case, when the FI changes as a function of the sample (again, I acknowledge the mentioning of this in the Discussion). The setup also seems to require that the latent variable is directly encoded in a Gaussian-Poisson population code, so that the likelihood is a Gaussian, whose precision scales linearly with the input firing rates. Again, it is unclear how this can be generalized to cases of practical interest, in which neither of these assumption will hold generally.

3. Based on the derivations, in order for the circuit to implement NG Langevin sampling, U_E should scale linearly with FI (Eq.11). Yet, in the simulations, when w_EE is sufficiently large, there is a strong threshold nonlinearity in the relationship between the two (Fig.2C). Furthermore, the authors suggest that U_EE (controlled by w_EE) acts as a regularizer "improving the numerical stability in inverting the FI when it is small or ill-conditioned". However, the empirical relationship between U_E and FI (Fig.2C) seems to be such that specifically at the small FI values, where regularization is supposed to be useful, there is no difference between small and large w_EE values. Indeed, the results shown in Fig.2F barely show any advantage for a larger w_EE in the small FI regime (and it's somewhat conspicuous why w_EE/w_c is not shown). All this leaves it unclear what the role of w_EE is in sampling, and whether it really implements the kind of regularization the paper proclaims. This is a problem, because this seems to be the only function suggested for recurrent excitation in the circuit, and otherwise a purely feedforward circuit seems best (see e.g. Fig.2E).

4. It is unclear what's the advantage of the "non-equilibrium" "annealing" strategy of the circuit at stimulus onset, shown in Fig.2I-K. First, large differences in the steady state bump heights (at different values of w_EE; Fig.2J) translate to minimal differences in sampling speed gains (Fig.2K). In fact, if anything, it seems that the case when the bump height barely grows at stimulus onset (w_EE/w_c=0) results in slightly faster sampling. I couldn't find what U_E the "equilibrium NG" sampler used, but I suspect it used the large value corresponding to the stimulus being on. What about an "equilibrium NG" sampler that always uses the low U_E corresponding to the stimulus being off? If I am right, there is no special advantage to the "non-equilibrium" "annealing" the authors focus on — there is simply an advantage of using low U_E as long as possible. 

I also found the terminology here somewhat fanciful (in that it made the effects that are described here sound more fancy than they really are). The term "annealing", in the context of sampling, typically refers to procedures that make the sampler sample progressively different target distributions (or an optimizer to optimize progressively different objective functions, as in simulated annealing). This is not the case here — the sampler samples from the same posterior distribution throughout the period of "annealing", just with different time constants. The term "non-equilibrium" dynamics usually refers to autonomous dynamical systems. This is also not the case here — the autonomous dynamics of the system here itself changes during the "non-equilibrium" phase of the experiment because the inputs to the system change. 

5. The bivariate setup is a little confusing. The main text makes it sound as if the combined circuit sampled from the joint posterior, and a paragraph and Fig.A2A-B is devoted to explaining the properties of the correlated prior the circuit implements. In turn, such a correlated prior is interesting because it also makes the posterior correlated (especially because the likelihoods are independent). However, then Fig.3 (and its caption) emphasizes how each module samples the corresponding marginal posterior. Indeed, I found no demonstration that the joint posterior is correctly sampled (unless the precisions shown in Fig.A2D are somehow related to joint precisions). But then how is this more useful than two decoupled circuits, already covered by the preceding sections?

A more minor point is that Eq.13 suggest that the prior implemented by cross-module weights is a bivariate Gaussian. But then the text states that it stores "an associative (correlational) stimulus prior with each marginal uniform" — it would be useful to point out that this is consistent with a bivariate Gaussian as a degenerate case. 

6. Taking together the concerns above, there is something slightly odd about what the proposed neural circuit achieves computationally in the end: it samples from a likelihood that is already parametrically represented (in a very easily readable form) in the input. If my interpretation of the bivariate case is correct (point 5), there is no combination with a nonuniform prior, or with some other likelihood. If this is the case, then there seems to be no additional benefit to the operation of this circuit compared to what its input already provides. 

7. The role of attractors in the circuit dynamics is confusing. The starting point for all the mathematical analysis is based on the existence of attractor states in the network. However, if these are truly attractor states, then they persist even in the absence of a stimulus — this is a problem because if the stimulus disappears (or more generally, changes), we would not want the circuit to maintain its representation of the posterior that was based on the (previous) stimulus. Indeed, based on Table 1, it seems that recurrent weights were chosen to be smaller than w_c (0.8w_c), which is "the smallest value that allows the network to maintain persistent activity even when there is no feedforward input" — i.e. the network is *not* in the parameter regime in which it has attractors. 

So, it is unclear, whether the presented networks do or do not have attractor states, and in the former case, how they can usefully perform sampling, and in the latter case, how the mathematical derivations serve their understanding.

***

Minor issues:

l.59: "linear dynamics of Langevin and Hamiltonian samplings [...] used in machine learning (ML) research" Is it really true that parctical sampling algorithms used in ML research use linear dynamics? 
l.205: "We will leave the subjective prior p(s) unspecific": "p(s)" → "p(z)"
l.215: "is resulted from" → "results from"
l.258: "we investigate the how the" → "we investigate how the"
l.260: "It is because" What is because?
l.269: "the circuit with fixed weights flexibly sampling likelihoods": "sampling → "samples"
l.308: "sampling step size will gradually decreases": "decreases" → "decrease"
l.350: "denots" → "denotes"
l.357: "To ease of understanding": "To" → "For"
l.410: "satisfys" → "satisfies"
l.465: "flexibly" → "flexibility"

Fig.2: the orange-red colors are barely distinguishable
Fig.2F: what's the black dashed line?
Fig.2H: what are the solid vs dashed lines?
Fig.A2: what is Lambda_s (I couldn't find its definition)?

### Questions
1. Is there any biological evidence to back up the modeling choices mentioned in point 1 above?

2. Is there any evidence that the network is able to sample from non-Gaussian posteriors, and in cases in which the likelihood is not given by a Gaussian-Poisson population code?

3. Is there any robust evidence (beyond what is currently shown in Fig.2C) that non-zero w_EE has a useful functional role in the circuit?

4. Is there any evidence that "annealing", rather than just a generally low value for U_E, is specifically useful for sampling after stimulus onset?

5. Is there any evidence that the samples produced by the network faithfully represent the correlations under a joint posterior distributions? If so, how can they usefully perform sampling, and if not, how do the mathematical derivations serve their understanding?


6. Is there a way to demonstrate that the network represents a posterior which cannot be trivially decoded already from its input?

7. Are there attractors in the intrinsic dynamics of the network?

### Soundness
3

### Presentation
3

### Contribution
3
