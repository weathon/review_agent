# Gaussian Ensemble Belief Propagation for Efficient Inference in High-Dimensional, Black-box Systems

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Efficient inference in high-dimensional models is a central challenge in machine learning.
We introduce the Gaussian Ensemble Belief Propagation (GEnBP) algorithm, which combines the strengths of the Ensemble Kalman Filter (EnKF) and Gaussian Belief Propagation (GaBP) to address this challenge.
GEnBP updates ensembles of prior samples into posterior samples by passing low-rank local messages over the edges of a graphical model, enabling efficient handling of high-dimensional states, parameters, and complex, noisy, black-box generative processes.
By utilizing local message passing within a graphical model structure, GEnBP effectively manages complex dependency structures and remains computationally efficient even when the ensemble size is much smaller than the inference dimension --- a common scenario in spatiotemporal modeling, image processing, and physical model inversion.
We demonstrate that GEnBP can be applied to various problem structures, including data assimilation, system identification, and hierarchical models, and show through experiments that it outperforms existing belief propagation methods in terms of accuracy and computational efficiency.

Supporting code is available at https://github.com/danmackinlay/GEnBP}{github.com/danmackinlay/GEnBP

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Efficient inference is challenging in graphical models with high-dimensional variables. 
The authors introduce an inference algorithm based on Gaussian Ensemble Belief Propagation (GEnBP) and the Ensemble Kalman Filter (EnKF), combining strengths of both algorithms to arrive at an algorithm which is competitive in accuracy, in scalability with respect to the dimensionality of the variables, and which can be applied to a wide range of inference problem structures.

### Strengths
The authors present a novel methodology with well motivated approximations and algorithmic structure, where important connections and details are highlighted. The methodology contains contributions and explanations that are valuable also on their own. Graphical illustrations and notation is used in an effective manner to improve clarity, and the text is well structured.

### Weaknesses
Although I find the text to be well structured in terms of its content, the text contains many language errors, forgotten punctuation, and incorrect figure references. I have listed some errors I encountered below which the authors may find helpful for improvement of the manuscript.  

List errors/suggestions style and language:
- Line 22: no spaces around “–”. 
- Line 48: why “seems” novel and not “is” novel, if that is the case? 
- Line 68: “table” is not capitalised.  This is also the case in many other places. Note that it should be capitalised since “Table 1” is a name.   
- Line 71: “section” not capitalised.  This is also the case in many other places. 
- Line 72: missing punctuation after “approximations”.
- Line 81-95: In my opinion, excessive use of bolds. Is it needed more than say a couple of times here, or even at all?
- Line 83: How about “a novel message-passing method for inference in high-dimensional graphical models”?
- Line 91: “Rank” is (incorrectly) capitalised even though the list is written as a sentence. 
- Line 93: Same as above but for “Compute”. 
- Line 99: How about “We will now introduce our notation and essential concepts”?
- Line 100: “appendix” is not capitalized. This is also the case in many other places. 
- Line 118: How about “We will assume that queries are always ancestral variables, i.e. that …”?
- Line 134: “equation” is not capitalized.  This is also the case in many other places. 
- Line 146: Missing word 'with'. “Edges connect each factor node j with each variable node…”? 
- Line 173: Unnecessary abbreviations, “we’ve” and “we’re”.
- Line 180: observed is misspelt.
- Line 184: “figure” not capitalised.  This is also the case in many other places. 
- Line 191: Missing punctuation before “The following”.
- Line 202: “definition” not capitalised. This is also the case in many other places. 
- Line 227: Double punctuation. 
- Line 261: Missing punctuation before “Marginalization”. 
- Line 294: Missing punctuation before “Throughout”. 
- Line 312: Missing punctuation before “Here”. 
- Line 314: Misspelt equivalently. 
- Line 323: Incorrect grammar, “See appendix I for a comparison with a naive attempt to do without the ensemble”.
- Line 325: Missing punctuation before “Then”. 
- Line 336: Missing punctuation after “high dimensions”. 
- Line 353: “algorithm” not capitalised. This is also the case in many other places. 
- Line 376: Incorrect grammar: “For a variable with K neighbors, in the worst case M = KN , when the cost becomes…” and missing punctuation. 
- Line 381: How about “In many practical applications it holds that N ≪ D, resulting in significant computational savings compared to GaBP which scales poorly with D.”. 
- Line 385: How about “better suited”?
- Line 404: Incorrect grammar “While GEnBP scales favorably with D, but unfavourably with respect to the node degree K, scaling as O(K 3 ) for some operations.”


The presentation of the experiments is unclear at times, please see Questions.

### Questions
Regarding Section 4.1:
- Figure 2 is never referenced. Although I assume the first paragraph describes observations made from it? 
- Why is the log likelihood only shown for GaBP in the case of the lowest dimensionality setting used? 
- I struggle to reach the following interpretation based on Figure 2: “We see that the Laplace approximation … but its posterior likelihood, while similar to GaBP, is even less stable, and both are inferior to GEnBP”:
- - For dimensionalities greater than around 100, it seems GEnBP is underperforming Laplace in terms of log likelihood (while the results provides no comparison to GaBP)? 
- - Why is GaBP and Laplace interpreted as less stable than GEnBP based on this figure? 

Section 4.2:
- Figure 3 is from the text implied to contain results using Langevin MC, but it does not --- only Figure 4 does. 
- The results show (and the text says) that GaBP performs better in terms of log likelihood for high viscosity settings. What do you suspect is the explanation for this? 

Overall, I think the paper is good and makes valuable contributions. However, I think the manuscript needs an overhaul to fix the language errors, missing punctuation, and figure references before it is ready for publication. If this is done and the experiments section is clarified I intend to raise my evaluation score.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a generalisation of the ensemble Kalman filter to make inference on arbitrary graphical models using belief propagation  (BP) with ensembles. By relying on ensembles rather than keeping track of the full moments, the method is able to scale to problems where the state dimension of the nodes are high dimensional, e.g. weather forecasting. The authors propose several numerical tricks that make this possible while retaining scalability. This is tested against a standard Gaussian BP on some examples, showcasing its computational efficiency and performance improvements, offered by its improved handling of nonlinear conditional dependence between nodes.

### Strengths
The paper is generally well-written and well-motivated. The methodology presented, while seemingly a small step beyond standard BP, is not at all trivial to execute, requiring various tricks to make it work in practice. In particular, I find the efficient computation of the factor-to-variable message and the ensemble conformation step to be an interesting contribution that is essential to retain the ensemble-based representation of the beliefs. The experiments showcase interesting settings where this technique could be applied, such as latent forcing identification in fluid models. In addition, it shows clear benefits of the approach over vanilla BP, with orders of magnitude increase in computation speed and performance gains.

### Weaknesses
While the paper is clearly written for the most part, there are some parts that I find need more explanation. See my questions below for details. I also find that the experiments can be improved with further baseline comparisons. Is GaBP really the only baseline that can solve the system identification problems in the experiments? One may probably also consider methods like the integrated nested Laplace approximation (INLA), or particle MC methods (e.g. the particle marginal Metropolis-Hastings algorithm).

### Questions
- It would be great if the authors could please clarify why we need a particle representation of the beliefs when we can just make computations with the DLR representation? My understanding is that in EnKF, it is used to propagate the particles by the nonlinear dynamics model and update them using Matheron's rule, but what does this correspond to in the general BP framework (in terms of steps (5)-(7))?
- Do we need tricks like covariance localisation and inflation in this general BP setting? These are almost always used to make EnKF work robustly in high dimensions.
- In Figure 2, why are some results missing for GaBP and Laplace method in the log-likelihood comparison? Was it producing NaNs due to overconfident predictions?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces the GEnBP algorithm, a method that combines Ensemble Kalman Filter (EnKF) and Gaussian Belief Propagation (GaBP) to efficiently perform inference in high-dimensional systems. GEnBP updates ensembles of prior samples into posterior samples using a message-passing strategy on a graphical model, making it suitable for applications with complex dependencies and high-dimensional variables, such as in geospatial and physical models.

### Strengths
The paper is well-structured, making it easy to follow the ideas presented. The proposed method can be considered to be novel in its approach to addressing the computational challenges of GaBP in high-dimensional systems.

### Weaknesses
While GEnBP demonstrates strong performance in high-dimensional contexts, its complexity scales disproportionately with the node degree in graphical models. This presents a significant limitation in highly connected graphs, where methods such as Forney factorization have been proposed by the authors as partial solutions. However, the effectiveness of these methods remains uncertain. Can some simple numerical results or insightful analyses be added to showcase this potential solution?

Additionally, a notable drawback is the absence of comparative analysis with other related works, both those employing and not employing EnKF, for efficient inference in high-dimensional, black-box systems. Key related works include:

  - Chen Y, Sanz-Alonso D, Willett R. "Autodifferentiable Ensemble Kalman Filters," *SIAM Journal on Mathematics of Data Science*, 2022; 4(2):801-833.

  - Chen Y, Sanz-Alonso D, Willett R. "Reduced-order Autodifferentiable Ensemble Kalman Filters," *Inverse Problems*, 2023; 39(12):124001.

  - Lin Z, Sun Y, Yin F, Thiéry A. "Ensemble Kalman Filtering-Aided Variational Inference for Gaussian Process State-Space Models." *arXiv preprint arXiv:2312.05910*, 2023.

  - Girin L, Leglaive S, Bie X, Diard J, Hueber T, Alameda-Pineda X. "Dynamical variational autoencoders: A comprehensive review." *arXiv preprint arXiv:2008.12595*, 2020.

Including a discussion or comparison on system identification and data assimilation from these (and other existing) works would enhance the comprehensiveness of the proposed method's positioning.

### Questions
1. Figure 2 is not cited in the text, which may confuse readers
2. Consider put Figure 1 in the experimental section
3. The paper highlights potential applications in fields like geospatial prediction and physical model inversion, which may involve real-time data. How does GEnBP handle streaming data, and what would be required to extend it to real-time applications with time-varying dependencies?

### Soundness
3

### Presentation
3

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
The authors introduce a novel method named Gaussian Ensemble Belief Propagation (GEnBP). They combine EnKF and GaBP together, in order to improve the efficiency in high-dimensional and non-linear cases. The method uses low-rank message-passing algorithms.

### Strengths
The authors try to use the strength of GaBP to solve (or I would like to say mitigate) the bad performance of EnKP in high-dim and non-linear problems. In the given examples, the new method works well. The computational complexity reduces largely for the dimensions of the data. It seems that the paper demonstrates superiority over existing methods in various benchmarks (but the figures are hard to read). I believe this method is potential for real-world applications for example in geospatial modeling.

### Weaknesses
1. Most important question for me: the scalability of the method to extremely large models, like weather simulations, remains to be fully demonstrated. Computational complexity still depends heavily on the degree of nodes in the graphical model, which means you must restrict the scale of the model. However, models in climate research always be extremely huge. How to choose N, how to choose K? If too small, can the model catches the nonlinearity well?
2. EnKF is somehow popular in climate or geophysical models, because of its computational complexity. But GaBP, in my impression, is a little bit outdated. For the nonlinear cases, we may use variational inference, or particle filter. Have you compared with them?
3. The innovation: GaBP mostly for low-dim problem. I am not sure it can help to improve the EnKF itself. More importantly, please see Bickson et al (2008) for the similar idea.
4. Even for small model, for example, Laplace seems better after consideration of trade-off. And Fig 2(c) seems not complete.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
