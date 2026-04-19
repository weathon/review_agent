# Quantitative Approximation for Neural Operators in Nonlinear Parabolic Equations

- Decision: Accept (Poster)
- Scores: 5, 6, 8, 5

## Abstract
Neural operators serve as universal approximators for general continuous operators. In this paper, we derive the approximation rate of solution operators for the nonlinear parabolic partial differential equations (PDEs), contributing to the quantitative approximation theorem for solution operators of nonlinear PDEs. Our results show that neural operators can efficiently approximate these solution operators without the exponential growth in model complexity, thus strengthening the theoretical foundation of neural operators. A key insight in our proof is to transfer PDEs into the corresponding integral equations via Duahamel's principle, and to leverage the similarity between neural operators and Picard’s iteration—a classical algorithm for solving PDEs. This approach is potentially generalizable beyond parabolic PDEs to a class of PDEs which can be solved by Picard's iteration.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies the approximation property of the neural operator for representing the solution operator for nonlinear parabolic equations. The key idea is that, the solution of the equation, written as the Duhamel’s integral, can be obtained from Picard iteration, with exponential convergence. The authors use a deep neural operator to approximate the Picard iteration process, where each layer approximates one step of Picard iteration. Overall, the paper is well written with clear explanation for all the ideas.

### Strengths
Same as summary.

### Weaknesses
Need for clarification on ReLU approximation. Detail in Questions.

### Questions
My primary concern is with the ReLU network approximation, specifically in the correspondence between the neural operator structure defined on page 6 and Lemma 2. In Eq. (24), you define the mapping, but it remains unclear how this mapping aligns with the neural operator structure described earlier. Please explain how the neural operator structure in page 6 result in eq (24). A clearer explanation or justification is needed to ensure consistency between these sections.

Additionally, I noticed that the notation $\Phi$ represents one step (either exact or approximated) of the iteration. However, on page 6, within each iteration, the activation is only used once, which is a shallow neural network with a single hidden layer. The reference to Guhring et al. (2020, Theorem 4.1) pertains to deep neural networks, which might not directly apply to this context. I suggest finding a more appropriate reference for the approximation result, specifically one that addresses shallow networks.
Another minor question is that, I think your remark “the curse of dimensionality that occurs in conventional neural network approximations does not appear here” in page 8 does not make sense. As addressed by yourself, the curse of dimensionality is hidden in Assumption 3, the representation of the Green’s function of the linear equation.

Another minor question is that, I think your remark “the curse of dimensionality that occurs in conventional neural network approximations does not appear here” in page 8 does not make sense. As addressed by yourself, the curse of dimensionality is hidden in Assumption 3, the representation of the Green’s function of the linear equation.

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
4

### Summary
The paper defines a neural operator based on Picard iteration and provides a quantitative approximation theorem for learning the solution operator of a general class of non-linear parabolic PDEs. The paper starts with a local well-posedness result for the considered PDEs, then introduces the considered neural operators, states the quantitative approximation theorem, provides a sketch of the proof and discusses extensions to longer time-horizons. The supplementary material contains proofs and shows how the result can be applied to Fourier Neural Operators (FNOs) and Haar Wavelet Neural Operators (WNOs).

### Strengths
Overall, the paper is well-written and structured. The results hold in quite general settings: Assumption 1 on the differential operator $\mathcal{L}$ of the PDE is rather general and includes, e.g., Laplace operators with different types of boundary conditions. Assumption 2 on non-linearity seems more restrictive and mainly limited to power-type non-linearities, but the obtained results are useful even in the case $F=0$, which is also covered by the setting. Assumption 3 is crucial to make the approach work, based on an expansion of the underlying Green's function. In the supplementary material the paper specializes the setting to certain types of Fourier neural operators and Haar wavelet neural operators, demonstrating the usefulness of the result.

### Weaknesses
The setting is formulated very abstractly, making reading potentially difficult for readers less familiar with general PDE theory. The paper does not discuss how the constructed neural operators could be implemented. In relation to existing neural operators (FNOs, WNOs) it remains unclear in what generality they are covered. Relation to existing literature is not always clear. I make these points more specific in the questions posed below.

Generally, I think this could be a valuable contribution, enhancing our understanding of mathematical foundations of neural operators. However, currently there seems to be a mistake in the proof of Theorem 1, affecting the construction of the neural operators and potentially making implementation infeasible. Consequently, at this point I need to recommend rejection, but I would be willing to increase my rating if this issue can be addressed convincingly. 

### Problem in the proof of Lemma 5

The main issue I currently see is that there appears to be a problem in the proof / formulation of the neural operators. The issue concerns the use of the time horizon $T$ instead of time variable $t$. In the definition of $\Phi_N$ in lines 264-269, in the first line the time integral is from $0$ to $t$ (the variable of the function to be approximated). In the next line, the integral with respect to $\tau$ is replaced by the inner product $\langle \psi_m, F(u) \rangle$. However, this means you are taking the integral up to $T$, not just up to $t$. 
The same problem can be found in the proof of Lemma 5 in the Appendix (lines 1245-1248), which does not seem to be correct in its current form. It seems that, to address this, the neural operator architecture would need to be modified, by making also the inner products $\langle \psi_m, u \rangle$ in the definition of the operator $K^{(\ell)}_N$ depend on the time variable $t$.
However, this would make a crucial difference from the perspective of computational effort required to evaluate the neural operator. In the current notation, to evaluate  $K^{(\ell)}_N$  only one inner product needs to be evaluated (the same for all times $t$). It seems that to fix the issue mentioned above, for each time $t$ a different inner product would need to be evaluated, which may be computationally infeasible (as at any given hidden layer, you are computing inner products with output functions from previous hidden layers). Here a thorough discussion is required.
Even without this issue, a more thorough discussion on implementation would be needed: The approach crucially relies on evaluating inner products, by definition of  $K^{(\ell)}_N[u]$ . To compute such inner products (between the hidden layer functions and the basis functions $\psi_m$) in practice, additional discretization is required at each layer; for each function evaluation.

### Questions
### Additional points requiring clarification
1.	The paper mentions that Theorem 1 avoids the curse of dimensionality. However, this is not entirely correct: it seems that in the statement of the theorem, the involved constant $C$ may still grow exponentially in $d$?
2.	Theorem 1 requires $N$ to be chosen in such a way that $C_G(N) \leq \varepsilon$. Is it possible to quantitatively relate $N$ and $\varepsilon$, e.g., in the FNO case?
3.	P.3: You state that under suitable conditions on $\mathcal{L}$ and $F$, the problems (P) and (P’) are equivalent, if the function $u$ is sufficiently smooth. Could you give some precise references here?
4.	Appendix A does not fully clarify why the considered operators satisfy Assumption 1. For example, A.1 states a bound on the Green function, but does not make explicit why this bound implies Assumption 1. For readers less familiar with PDE theory it would be useful to make explicit (at least for one of the sections A.1, A.2, …) why this bound implies that Assumption 1 is satisfied. 
5.	Similar results to Proposition 1 are well-known in the literature. Can you relate Proposition 1 to the literature more precisely, outlining differences to existing results in the literature? 
6.	Fourier neural operators: why do you only consider the case $d=1$ here? 
7.	Could you clarify if WNOs with Haar wavelets, as defined in Appendix C.2, have been used in previous papers?
8.	What is the role of $\tilde{D}$ in Appendix C.1? Can’t we just set $\tilde{D}=D$?


### Additional comments:

-P.4, l182: it should be operator, not operato,r

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors derive the approximation rate of solution operators for the class of nonlinear parabolic partial differential equations (PDEs), contributing to the quantitative approximation theorem for solution operators of nonlinear PDEs. They show that neural operators can efficiently approximate these solution operators without the exponential growth in model complexity, thus strengthening the theoretical foundation of neural operators. An innovative link between Picard iteration and neural operators is found and the authors also leverage solid PDE theory (using Duhamel's principle). This work provides a very good theoretical foundation for neural operators (and generalizes previous works on Neural operators such as Fourier Neural Operators, etc.).

### Strengths
This is a highly rigorous and technical paper. The details are presented clearly and I am able to understand the authors' motivation and thought process in developing the theory. I think this is a great theoretical contribution that has solid foundations in PDE theory to better understand Neural Operators (of which there are varieties like Fourier, Wavelet, etc.) and unifies all of them in this fairly decent class of PDEs. I hope to see extensions of this work (that the authors mentioned there would be) to more general classes of PDEs but this is a significant start. Using neural operators as solvers for PDEs has a wide array of applications, particularly in scientific computing and other "AI4Science" domains. What is interesting is the method and analysis they use to bypass, by appropriately selecting the basis functions within the neural operator, the exponential growth in model complexity. They also do not seem to have too many unreasonable assumptions (this I'm not highly confident about).  Error rates with respect to truncation N make sense. There is a lot of scope for future explorations with different architectures and PDEs.

### Weaknesses
1. The paper could provide a bit more intuition and background on Neural Operators to be a bit more self-contained. This could be presented in the Supplementary.
2. There are no numerical experiments (but for this paper I do not think this is a big weakness given that I believe the main contribution of theory is solid). Nevertheless, it would be nice to see in future (if it is even possible to implement all these different basis expansions, etc.)
3. I do not think it is easy to find a Green's function or design one, which I see as the main drawback of the paper (if there is to be a practical extension for Deep Learning). Perhaps it is not that hard to find one for the linear parabolic PDE. (addressed in the Questions below).
4. Minor: typo on line 182 "elliptic operato,r " 
I cannot think of any more now, but it depends on my questions below as well as I may not have understood everything.

### Questions
1. I do not think it is easy to find a Green's function or design one, which I see as the main drawback of the paper. Perhaps it is not that hard for the linear parabolic PDE. (addressed in the Questions below). How can one go about finding these (for specific cases or in general) or is this mainly constructive? Or do you only need to characterize the regularity of the Green function and how can you do that, if that is the case?
2.  Could you clarify if the error being log ((1/eps)^{-1}) implies exponential dependency?
3 (a). How can I interpret K^{(l)}_N? W are weights, b are biases, but I am not sure what K (the rank) is (unless it is specific to the Neural Operator?)
(b) Can you clarify (or add in the paper to make it more self-contained (see point 1 in Weakness) what is the rank of a neural operator? I do not believe it is defined in your paper (I checked other sources to understand). 
4. It seems the first 2 sentences of your Discussion are contradictory: you mention it is a limitation that your method only applies to parabolic PDEs, but the next sentence says your approach is not restricted to parabolic PDEs. Perhaps you could clarify that it is not truly a limitation then, and is just a first step?
5. For the long-time solution, how reasonable is it that u(T) again falls into  Ball(L_\infty, R) ? Does that limit the expressiveness of the solution of the PDE?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This research focuses on a new method for approximating the solution operators of nonlinear parabolic PDEs using neural operators. The authors aim to bridge the gap between theoretical understanding of neural operators and their practical application as PDE solvers by developing a quantitative approximation theorem. This theorem demonstrates that neural operators, specifically designed with a structure inspired by Picard's iteration, can approximate solutions of parabolic PDEs efficiently without experiencing the exponential growth in model complexity often associated with general operator learning. The authors provide examples of how their method can be applied to Fourier neural operators (FNOs) and wavelet neural operators (WNOs), highlighting the potential for this approach to solve a wide range of nonlinear PDEs.

### Strengths
1. The paper presents a quantitative approximation theorem for approximating solution operators of nonlinear parabolic PDEs using neural operators.
2. The authors show that the depth and number of neurons in their neural operators do not grow exponentially with the desired accuracy, potentially mitigating the "curse of parametric complexity."
3. The proof is constructive, leveraging the connection between Picard's iteration and the forward propagation through the layers of the neural operator.
4. The proposed framework may be potentially generalizable to other nonlinear PDEs solvable by Picard's iteration, including the Navier-Stokes, nonlinear Schrodinger, and nonlinear wave equations.

### Weaknesses
1. The theoretical results are not supported by numerical experiments, making it difficult to assess the practical performance and efficiency of the proposed neural operator architecture.
2. The paper focuses on a specific class of parabolic PDEs amenable to analysis via Banach's fixed point theorem. The applicability to more general PDEs, particularly the Navier-Stokes equation mentioned in the abstract, is not demonstrated.
3. The paper does not adequately address the challenge of handling periodic boundary conditions, a crucial aspect for applying FNOs.
4. The paper does not discuss the role of the trace operator and how to ensure sufficient regularity for the boundary conditions when defining the neural operator $\Gamma$.
5. The assumptions on the operator L and the nonlinearity F are quite strong, potentially limiting the scope of applicability.

### Questions
1. Can the authors provide numerical experiments to validate the theoretical results and compare the performance of their neural operator with traditional PDE solvers or other neural operator architectures?
2. The dual of $L^\infty$ is more complex than that of $L^1$. The authors should consider revising or removing this aspect throughout the paper.
3. Could the authors address how the proposed method handles periodic boundary conditions, which are essential for applying FNOs? Additionally, could they discuss the role of the trace operator and how to ensure sufficient regularity for the boundary conditions when defining the neural operator $\Gamma$?
4. Can the authors elaborate on how the proposed framework can be extended to handle more general PDEs, specifically the Navier-Stokes equations mentioned in the abstract with their specific projection operator and boundary conditions?
5. In PDE theory, extending local solutions to global solutions is a challenging and significant issue. Although the paper acknowledges the challenge of extending local solutions to global ones, it does not provide concrete solutions or insights. Could the authors elaborate on this challenge and clarify the practical relevance of Corollary 1, especially regarding its error estimate for long-time solutions?
6. How does the choice of basis functions $(\phi$ and $\psi)$ in the neural operator affect the approximation error rates? Could the authors provide specific examples to illustrate these rates for different basis functions?
7. Regularity of the solution space is crucial in numerical approximation. However, this paper does not incorporate this aspect into an error analysis. Could the authors characterize the impact of function regularity on the error estimate?

### Soundness
3

### Presentation
4

### Contribution
3
