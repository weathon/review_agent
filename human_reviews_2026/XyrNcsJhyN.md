# Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Modern neural networks exhibit a striking property: basins of attraction in the loss landscape are often connected by low-loss paths, yet optimization dynamics generally remain confined to a single convex basin and rarely explore intermediate points. We resolve this paradox by identifying entropic barriers arising from the interplay between curvature variations along these paths and noise in optimization dynamics. Empirically, we find that curvature systematically rises away from minima, producing effective forces that bias noisy dynamics back toward the endpoints — even when the loss remains nearly flat. These barriers persist longer than energetic barriers, shaping the late-time localization of solutions in parameter space. Our results highlight the role of curvature-induced entropic forces in governing both connectivity and confinement in deep learning landscapes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper describes an entropic mechanism by which stochastic optimization is biased towards flatter minima. The proposed mechanism is that flatter minima take up larger portion of weight space, and since training is noisy, entropic considerations would predict that the posterior distribution of network parameters would be localized in low-curvature zones.  This extends a line of research about the implicit bias of various optimizers, the connection between local curvature and generalization properties, and the connectivity of the low loss manifold of deep networks.

### Strengths
I really loved the central idea of the paper. It offers a simple conceptual model and provides concrete and convincing numerical demonstrations. The question it addresses is timely and important, and the answer is at once original, simple and insightful, even if not rigorously proven. The presentation is relatively clear, though I think much of it is wrongly phrased, see below. I think this is an important contribution in this sub field, that will likely generate interesting follow up works.

### Weaknesses
Despite the praises above, my final recommedation is a weak reject because I think the manuscript has some fundamental flaws, and proofreading is more than sloppy. However, as I wrote in the Strengths section above, I think the central idea is (probably) valid and interesting and that this manuscript would be a good contribution to the field once the flaws are fixed. I'd be very happy to upgrade my assessment pending the authors response.

1. The main weakness I see is in the framing. The terminology used is at times inaccurate and at times plainly wrong and contains what I think are fundamental mistakes in interpertation. However, the general idea is, I think, valid. The manuscript would be much stronger (and my grade would be higher) if this were corrected. I'll try to as explicit as i can with this point:  
  - The main disagreement I have with the framing has to do with how the authors distinguish energy and entropy in DYNAMICAL terms.  For example, the opening statement of Sec. 2 is that "the dynamics of a system are governed not only by energetic forces—derived from gradients of an energy or potential—but also by entropic forces, arising from thermal fluctuations". Similarly, the authors write that "entropy causes optimization to climb up the loss" (line 138) or that "entropic forces dynamically confine models" (line 72) or similar statements scattered throughout the manuscript.  
 This is a delicate point: The basic tenet of stat mech is that the Boltzmann distribution arises naturally *even if the underlying dynamics is purely deterministic and energy driven*, the canonical example for which is the micro-canonical ensemble, (sorry for the pun). When the authors write that the system is "dynamIcally confined" to high entropy regions, they use this term exactly in the opposite meaning of what "dynamical confinement" would mean in the stat mech literature -- namely places that have non-vanishing Boltzman weight and that therefore one should be able to probe if dynamics were ergodic. But one cannot probe them because of dynamic (=energetic, not entropic) constraints.
 * Explicitly: in the stat mech jargon, dynamical confinements pertain to *energy* barriers, while entropic confinements pertain to *free energy*. This is a crucial difference, which is central to the authors' main point.  
(a trivial textbook example: consider an ideal gas in a container. It is *perfectly dynamically possible* for the gas to be concentrated in one half of the container, but such configurations have very low entropy (high free energy) and therefore it is *thermodynamically* forbidden, while being *dynamically allowed*)

- A similar mistake happens when the authors write that "it is possible for entropic forces to be the stronger than energetic forces, leading to a scenario where entropy causes optimization to climb the loss." (line 138). However, this is generically the case. It is *always true* that "entropic forces" drive dynamics up the energy landscape (driving them down the energy landscape would result in negative temperature which is not strictly forbidden by thermodynamic, but is quite an esoteric condition which I think is irrelevant here). Even in the toy example given by the authors of a valley with a varying curvature, $V=g(x)y^2$, the basic fact that $\langle x^2 \rangle\ne0$ stems from the fact that at finite temperatures (hence finite entropy) the dynamics drive the system up the quadratic well.

I think both these mistakes, and similar ones, stem from the same basic misunderstanding. This framing should be fixed.

2. The main result of the paper is Fig. 2 and the curvature variation along the MEP. However, I suspect that the custom SGD the authors use might affect this result. At the very least, the authors should present how the curvature along MEP looks like if one uses the method of Frankle /Draxler directly, and show how their result changes with $k$ (defined in appendix A1, and, as far as I understand, is confusingly not the same $k$ as in Fig. 4). 

3. There are typos, missing references, and two paragraphs are repeated verbatim twice in Sec. 4.2. Appendix A2 terminates in the middle of a sentence (!). This is, frankly, not serious. I know well how submission to these conferences work, but this almost insulting for the reviewers and ACs and all the people who took the time to read your paper seriously and comment on it constructively. The minimum you could do is read it before you submit.

### Questions
I wrote my questions in the weaknesses box.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores the paradox that neural network minima are connected by low-loss paths, but SGD confines itself to individual solutions. The authors propose that entropic barriers, following from the interplay between curvature of the landscape and noise in the optimization, explain the confinement. They run experiments on ResNet architectures trained on CIFAR-10 to demonstrate that the curvature systematically increases along low energy paths between minima. This creates effective forces that bias SGD back to the low curvature endpoints. They show that the magnitude of these forces inversely scales with batch size and argue that these forces persist longer than energetic barriers, becoming dominant in later stages of training.

### Strengths
- The paper's motivation is well presented and the introduction/literature review clearly set up the problem.
- The paper connects two well-known phenomena: mode connectivity and the low curvature bias of gradient-based optimizers. These two nicely merge in the paper's analysis, giving an intuitive and convincing picture that the proposed explanation is actually behind the  phenomenon observed. 
- I think that the experimental setups are interesting and well-designed. I particularly like the idea of initializing in the middle of the path and looking at the relaxation when performing gradient descent projected on such curve.

### Weaknesses
1. My main concern with this work is that, while the ideas are very interesting, the experimental validation is too limited in scope to fully validate it. Given that the paper makes the strong claim that these energy barriers are a "univeral" phenomenon, the experiments should definitely be performed on more than two architectures (Wide ResNet-16-4 and ResNet-20) and one dataset (CIFAR 10) like the paper currently does.

2. Overall, the feeling I get from reading the paper is that a lot of space is used for things that are not central to the argument, like the details about curvature measures and their computation (which could have been an appendix) and the toy model that, in its current form, doesn't really help in conveying the idea (more below).  This space, in my opinion, would have been better spent by expanding the experimental setup.

3. It is not clear how the analytical toy model helps in understanding the phenomenon. First, the derivation feels disconnected. It is not clear how the authors go from Eq. 2 to Eq. 3. Expanding in an appendix would help but the main text section should still be adjusted to be more readable. Furthermore, I don't understand what we are getting from the analytical part of toy model described in page 2. Unless I am missing something, the main message i.e. "regions with smaller curvature $g(y)$ (flatter directions in $x$) contribute larger entropy and are statistically favored, even if the original energy $V(x, y)$ is minimized elsewhere" doesn't seem to follow directly from the analytics. It is however clear enough from Fig. 1B.

4. The claim that "entropic barriers persist longer than energy barriers", which is based on Figure 4D seems a bit overreaching given that it results from only one set of experiment.

### Minor weaknesses

1. I think that the sentence at L249, stating that "symmetries do not have any affect on optimization dynamics." is incorrect. There are plenty of works that study the relationship between symmetries and optimization dynamics. To mention only a few, see [1-3].
2. "delta correlated Gaussian noise" undefined at L091
3. In eq. 2 $g_y$ is the derivative of $g$ w.r.t. $y$?
4. Paragraph duplicated at line 357
5. Figure 2 caption has a ??
6. Page 4 has score(?) after equation 5
7. Missing space at line 56



- [1]  Du, Simon S., Wei Hu, and Jason D. Lee. "Algorithmic regularization in learning deep homogeneous models: Layers are automatically balanced." Advances in neural information processing systems 31 (2018).
- [2] Kunin, Daniel, et al. "Neural mechanics: Symmetry and broken conservation laws in deep learning dynamics." arXiv preprint arXiv:2012.04728 (2020).
- [3] Zhao, Bo, et al. "Symmetries, flat minima, and the conserved quantities of gradient flow." arXiv preprint arXiv:2210.17216 (2022).

### Questions
1. Why does starting from 0.7 relaxes the minimum in relative position 0 in Fig.3A? Does it converge to the one in position 1 for higher initializations? It would be interesting to have a more complete profile of what's going on for different starting points (see comments above).
2. What do you think would happen if you initialized in the middle of a path and then perform SGD without projecting to the path?
3. Do the curvature bumps persist when regularization is added to the loss?

### Soundness
2

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
This paper provides a dual view on why neural network optimization remains localized near certain minima despite low-loss connecting paths between different solutions. The authors investigate the training dynamics using the AutoNEB algorithm (Draxler) and show that while energetic barriers (arising from the loss) remain low along these paths, entropic barriers emerge from the interaction between curvature variations and stochastic gradient descent noise, manifested as a "curvature bump" away from minima. They demonstrate that these curvature-induced entropic barriers persist longer than traditional energetic barriers and play a key role in determining which regions of parameter space are dynamically accessible during training.

### Strengths
- The paper provides an interesting and fresh perspective on how optimization is biased towards certain solutions, driving interesting connections to the implicit bias of SGD (though the relationship to the literature could be expanded here)  
- The paper is sound. They use trace and max. Eigenvalue of the Hessian and using the SVD of the Fisher, providing robust evidence that curvature systematically increases along connecting paths.  
- The observations relating to the spawning experiments in Frankle are particularly insightful. I would encourage the authors to expand on this connection and engage deeply with the literature on the chaotic early phase of neural networks (e.g. Fort et al, 2020, Altıntaş et al, 2025). The controlled noise setting in Altıntaş could be an interesting addition to this work.

### Weaknesses
- While the core findings are convincing, the work would benefit significantly from additional ablation studies. Given that the experimental setting (ResNet-20, Wide ResNet-16-4) is not prohibitively expensive, several natural extensions would strengthen the paper:  
- How do entropic barriers manifest when learning rates are reduced? Do they become weaker (less noise)?  
- Algorithms like K-FAC, Shampoo, or natural gradient descent explicitly account for curvature. Do these methods navigate entropic barriers differently?   
- I think an experiment in at least Cifar-100 scale would be needed.

### Questions
- I believe the pivots in the AutoNEB procedure are initialized equidistantly along a linear interpolation between minima (following Draxler et al.), but the subsequent relaxation and potential addition/removal of pivots means they no longer maintain equal spacing. Could the authors clarify: how does the relative position translate into the relative metric position between the two points?  
- The existence of entropic barriers along low-loss paths raises intriguing questions about weight averaging techniques like Stochastic Weight Averaging (SWA), model soups, and linear mode connectivity ensembles. If curvature increases along connecting paths between minima, what does this imply for intermediate points obtained by weight averaging?

Minor comments

- The phrase "bottom of the loss landscape" sounds somewhat unnatural, you could consider using more precise terminology like "low-loss regions," "basins of attraction,"  
- Missing refs on Lines 208, 233  
- First two paragraphs of 4.2 repeated twice verbatim   
- MEP 1-2 could be clarified in the figure

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Minibatch training introduces stochasticity, which is also modulated by the discrete step size.
This paper studies so-called *entropic forces*, an implicit bias that progressively nudges optimization toward flatter regions of the loss landscape, strengthening as update noise increases.
Seen through the lens of mode connectivity, this driving force is proposed as an dynamic barrier between disconnected minima regions.
Indeed, by sampling low-loss paths between minima with AutoNEB, they observe that curvature rises away from the endpoints.
When models are initialized slightly off these endpoints, path constrained dynamics pull them back, revealing a curvature-driven “gatekeeping” effect that intensifies with smaller batch sizes.
Reusing a setup of linear mode connectivity, the authors further argue that the relative strength of the entropic force grows in the later stages with respect to the loss.

### Strengths
This paper presents an interesting study curvature-induced biases in neural network optimization.
It offers valuable insights into how stochasticity combined with geometry can influence the parameter dynamics, mostly towards the end of training when the loss' role is lessened.
The paper contributes to explain why certain solutions, though equivalent from the point of view of the train loss, are preferred and therefore challenges our understanding of the implicit biases of optimizers.
The results could have implications for late-stage optimization dynamics, low-loss manifolds of solutions, stability within the zero-loss regime, and connections to phenomena such as double descent or grokking.
An original point of the paper lies in the fine-grained, directional analysis of local curvature effects beyond more general arguments given in previous literature (e.g. Edge of Stability, catapult).
The work also raises an interesting question on overfitting, that could be countered by such entropic forces and hence foster better generalization.
In that sense it might also shed more light on why flatness measures have been successful despite not being principled.

### Weaknesses
While the paper is conceptually interesting and seems technically sound, it suffers from awkward or overly elliptical phrasing, despite no apparent space constraints (see questions and minors below for specifics).
This gives the impression that the manuscript was not thoroughly reread, which also somewhat undermines confidence in the results.

The paper sometimes seems to overreach:
- line 155: The reference to “the constant loss non-linear path” overstates uniqueness. In addition, identifying paths with the Auto NEB algorithm means incorporating any bias towards particular properties these paths might have, and this is not discussed. (see my question about it below)
- line 320 & 406: the claim that the entropic forces are responsible for the localization of the parameter when the *splitting epoch* is high appears is not fully supported.
Figure 4 only suggests a relative increase in the importance of entropic forces, not that they overtake loss dynamics. A clearer analysis of the trade-off between energetic and entropic components would strengthen this point.

Interpretation of Fig 2 c.: while earlier works have shown that intermediate models can outperform endpoints (e.g. Model Fusion via Optimal Transport by Singh and Jaggi or Git Re-Basin: Merging Models modulo Permutation Symmetries by Ainsworth et al), the abrupt loss drop in Figure 2(c) seems unusual. The explanation given by the authors line 260 is not very convincing as it should also applies in earlier works published since the work by Draxler et al. (see also my related question below)

Experimental details are reported but code is currently not provided.

Limitations include:
1. not discussing quantitatively the tradeoff between entropic forces and loss which makes it difficult to quantify the effect of entropic forces on a typical training
1. while the batchsize point of view is interesting, it would have been great to also investigate the learning rate since it is mentioned as the other source of stochasticity in section 2
1. while making broader connections and mentioning potential applications, the paper focuses only on the mode connectivity/linear mode connectivity setup

## Minor and additional feedback
- duplicated paragraphs in 4.
- line 621 trimmed or spurious full stop before "in practice" ?
- line 233 equation ??
- fig 2: plots inverted w.r.t. caption
- line 93: $\eta$ and $B$ are not introduced
- line 100: isn't there a missing $T$ in the variance formula ?
- line 70: when the loss is near $0$ is a bit imprecise, is it "when the loss is near zero along the path of solutions"
- line 100 and equation 2: I think it would be better to adopt a consistent notation for the variance
- the whole toy model example is a bit too elliptical (also while $g_y$ is understandable is it not clearly defined and adds to the reader disambiguation effort)
- line 208 missing link (?)
- The sign of $V_{eff}$ might be wrong in equation $3$ since it leads to $P(y)\propto \text{exp(ln }g(y))$ and so the dynamics would be attracted towards high curvature region instead of repulsed.
- $m$ of figure 3B is called $\rho$ in the main text 
- abstract: "yet optimization dynamics remain confined to one solution" different parameters can be sampled at the end of training so this is a bit unclear
- line 130: 'entropic' is spelled 'entorpic'
- line 249: 'affect' is written instead of 'effect'
- line 596: "on along" is redundant

### Questions
1. **Subsection 4.1** you mention that the loss drop may result from the elastic coupling between pivots. Could you elaborate on how this coupling contributes to a further decrease in loss?
1. **Figure 3 setup**: what behavior do you observe if the parameters are not projected back onto the MEP during training? This question seems important to assess the scope of entropic forces in a real training process for example.
1. **Figure 2 curvature plateau** – Around the midpoint of the MEP, curvature metrics appear nearly constant, with even a local minimum of $\lambda_{\text{max}}$ on Fig 2B. . Should we interpret this as a region where entropic forces vanish, leaving the parameters unbiased toward the endpoints? Have you tested initializations in this region?
1. **AutoNEB bias** AutoNEB minimizes both loss and path length: "the spring energy grows quadratically with the total length of the path" (quote from Drawler et al). Could this introduce a bias toward paths that wrap around high-loss regions or exhibit higher curvature? How sensitive do you believe your findings are to the specific properties of the AutoNEB algorithm?

### Soundness
2

### Presentation
1

### Contribution
3
