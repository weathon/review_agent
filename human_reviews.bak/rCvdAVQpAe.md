# Physics-Informed Deep B-Spline Networks

- Decision: Reject
- Scores: 8, 6, 6, 5

## Abstract
Physics-informed machine learning provides an approach to combing data and governing physics laws for solving complex partial differential equations (PDEs). However, efficiently solving PDEs with varying parameters and changing initial conditions and boundary conditions (ICBCs) remains an open challenge. We propose a hybrid framework that uses a neural network to learn B-spline control points to approximate solutions to PDEs with varying system and ICBC parameters. The proposed network can be trained efficiently as one can directly specify ICBCs without imposing losses, calculate physics-informed loss functions through analytical formulas, and requires only learning the weights of B-spline functions as opposed to both weights and basis as in traditional neural operator learning methods. We show theoretical guarantees that the proposed B-spline networks are universal approximators of arbitrary dimensional PDEs under certain conditions. We also demonstrate in experiments that the proposed B-spline network can solve problems with discontinuous ICBCs and outperforms existing methods, and is able to learn solutions of 3D heat equations with diverse initial conditions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a physics framework for learning B-spline control points in systems with varying initial and boundary conditions. The approach allows for using analytical expressions as physics informed losses since B-spline function gradients can be expressed as analytical expressions. The proposed approach is used for estimating high dimensional surfaces described by families of ODEs/PDEs. The authors provide theoretical bounds and empirical evidence for their claims.

### Strengths
Originality: This paper has good mathematical intuition and a novel way to estimate the solution of PDEs using B-splines. This approach makes it easier to use PINN loses using analytical expressions improving efficiency over autograd.  The use of control points allows for any Dirichlet boundary conditions and initial conditions to be directly assigned. 

Quality: The method is very clearly described, and the authors provide a good mathematical background. They clearly state the difference between their proposed approach and KAN based approaches. The paper provides theoretical bounds and empirical evidence. 

Significance: This work has a lot of potential in the field of scientific ML, as the authors show significant speedup over Physics informed ML techniques, and a reasonable improvement in accuracy of predictions.

### Weaknesses
While the authors provide a good theoretical basis for B-spline based PINNs, the experimental section is quite small. They evaluate on 2 settings. However, it would be good to showcase the performance on other standard evaluation benchmarks (Navier-Stokes, Darcy Flow, Burgers, Kolmogorov flow etc.) 

The use of splines to train PINNs is not new. [1] uses Hermite splines to approximate PDEs. [2], [3] proposes using neural networks to learn the coefficients of B-spline functions. It would be good to show the differences between the proposed approach and these works. 

[1] Wandel, Nils, et al. "Spline-pinn: Approaching pdes without data using fast, physics-informed hermite-spline cnns." Proceedings of the AAAI conference on artificial intelligence. Vol. 36. No. 8. 2022.

[2] Doległo, Kamil, et al. "Deep neural networks for smooth approximation of physics with higher order and continuity B-spline base functions." arXiv preprint arXiv:2201.00904 (2022).

[3] Zhu, Xuedong, et al. "A Best-Fitting B-Spline Neural Network Approach to the Prediction of Advection–Diffusion Physical Fields with Absorption and Source Terms." Entropy 26.7 (2024): 577.

### Questions
1.	How does the proposed approach compare against [1], [2], [3]?

[1] Wandel, Nils, et al. "Spline-pinn: Approaching pdes without data using fast, physics-informed hermite-spline cnns." Proceedings of the AAAI conference on artificial intelligence. Vol. 36. No. 8. 2022.

[2] Doległo, Kamil, et al. "Deep neural networks for smooth approximation of physics with higher order and continuity B-spline base functions." arXiv preprint arXiv:2201.00904 (2022).

[3] Zhu, Xuedong, et al. "A Best-Fitting B-Spline Neural Network Approach to the Prediction of Advection–Diffusion Physical Fields with Absorption and Source Terms." Entropy 26.7 (2024): 577.

### Soundness
4

### Presentation
4

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
This paper proposes PI-DBSN, a direct neural network solution to learn to solve parametric PDEs with different initial or boundary conditions. PI-DBSN directly learns the weight of B-spline basis functions, by training under both data loss and physics-informed loss. Authors address that there are two-fold advantages of learning spline interpolants. On the one hand, universal approximation theorem holds for any solution in $L^2$ domain. On the other hand, it can solve problems under multiple configurations of initial or boundary conditions and the geometry of the domain has the potential to be dynamically change based on its parameterization.

### Strengths
This paper has an unique viewpoint as it trains a neural network which maps both physical parameters in PDE setup and parametric domain into weights of B-spline basis functions. Manipulating an NN solver to adapt to geometry of domain is a challenging topic. Moreover, an extensive discussion of related works supports its novelty and differentiate this paper from many previous researches. Authors provide a through analysis over universal approximation theorem as well.

### Weaknesses
While I am expected to see some further discussions of how to learn a PDE solution under different configurations of IC/BCs by fitting spline interpolants, the design of experiments hinder a reader to appreciate PI-DBSN.  Authors show performance for a convection-diffusion equation in space-time domain and a 3D heat equation, but on a regular box-domain. Even for this simple setup I didn't see superiority of applying PI-DBSN, rather than using PINN, let alone several NO approaches was not compared. In second case where a 3D heat equation was learnt, author even did not compare with any other methods, and $10^{-3}$ scale of residual error is not a wow number.

Though not required due to extensive work of experiments, I wish to, at least hear from authors, if complex geometric domain, such as $L$-shape problem introduced in Doleglo et al. (2022), or hyper-elastic problems setup in [Geo-FNO](https://arxiv.org/pdf/2207.05209) can be computed via PI-DBSN. Under these setups, is there any potential challenges or modifications needed to apply PI-DBSN to irregular geometries, and how the spline basis representation might need to be adapted.

Besides simple setup in experiments, several statements in this paper is inappropriate. Please consider revise them. For instance:


*    line 080: The term "Dirichlet initial and boundary conditions" is incorrect. It should be "Dirichlet boundary conditions and initial conditions".
*    line 112-113: It claims Neural Operator solution "do not account for varying initial and boundary conditions". Nevertheless, in original FNO paper, even for Burger's euqation, initial conditions are drawn from random samples. 
*    line 246-247: "initial condition is given by $s(0,t)=0, \forall t$" is an obvious typo. It should be  $s(x,0)=0, \forall x$. Even corrected, I don't see why a constant boundary shows superiority of spline basis representations. A correct and complete statement shall point out that: even under this simple boundary setup, prior works like PINN would fail to comply their solutions with respect to this particular BC. 
*    Line 338: "3D Fokker-Planck Equation" is inconsistent with section title "Heat Dispersion in a 3D box" (line 418), and setup "3D heat equation" (line 420), even though they are related.
*    All norms in loss functions $\mathcal{L}$ are of form $\lvert \cdot \rvert$ (line 223-229). Is this $l^2$ norm or $L^2$ norm or absolute value? Please specify.

### Questions
Some other questions, if answered, can improve the quality of this paper:

*    Figure 3 indicates PINN attains average better performance. It is also indicated in line 387-388 that prediction MSE of PI-DBSN is worse compared with PINN. Can you provide more details of network setup, and explain "similar NN configuration" stated in line 364-365. How many parameters of NN in each setup and how computation time would change if one set up same or similar number of parameters in each experiment.
*    What is the residual error in convection-diffusion setups?
*    How does a plot of number of NN parameters v.s. MSE. looks like? In table  2 only the relationship between number of control points v.s. MSE is reported. 
*    Are all experiments conducted over noiseless setup? Is PI-DBSN robust against noisy observation? 
*    What is the performance of PI-DBSN over Burger's equation and Navier-Stokes equation using the same setup in FNO paper? Are there any specific reasons that the comparison ruled out FNO branch of method?
*    How is physics model loss being evaluated. Namely, what is the quadrature rule when computing $\lvert\mathcal{F}_i(s,x,u)\rvert^2$?
*    Is there any ablation study on how to choose $\omega_p$ and $\omega_d$ to attain better performance? In line 230, authors claim both are set to $1$.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a novel Physics-Informed Deep B-Spline Network framework for efficiently solving partial differential equations (PDEs) with varying system parameters, initial conditions, and boundary conditions (ICBCs). Traditional physics-informed neural networks (PINNs) often struggle with discontinuous conditions and can be computationally demanding, as they typically require simultaneous learning of both weights and basis functions. The proposed method addresses these challenges by using a neural network to learn B-spline control points, which approximate the PDE solutions. This approach allows direct specification of ICBCs without imposing additional losses, reduces computational complexity by only requiring the learning of B-spline weights, and incorporates analytical formulas for calculating physics-informed loss functions. The paper further establishes theoretical guarantees that the B-spline networks serve as universal approximators for arbitrary-dimensional PDEs under specific conditions. Empirical results demonstrate that the B-spline network outperforms traditional methods, effectively handling discontinuous ICBCs and diverse initial conditions. This framework contributes a robust, efficient tool for solving high-dimensional PDEs, with applications in fields where complex physical systems require adaptive modeling.

### Strengths
The proposed Physics-Informed Deep B-Spline Network framework showcases originality in its formulation and application. By leveraging B-splines within a neural network architecture, the authors address a common challenge in physics-informed machine learning: the need for efficient, accurate solutions to partial differential equations (PDEs) with varying parameters and changing initial and boundary conditions (ICBCs). This approach creatively combines established techniques in both spline theory and neural networks to improve upon traditional physics-informed neural networks (PINNs), which typically require simultaneous learning of weights and basis functions. The novelty here also extends to how the B-spline networks are designed to handle discontinuous ICBCs—an area where many neural network approaches struggle. The method’s theoretical guarantee as a universal approximator further demonstrates its innovative edge, providing a rigorous mathematical foundation that strengthens its potential utility across a wide range of applications.

### Weaknesses
1. Limited Experiments
2. Scalability and Computational Complexity Analysis is not sufficient
3. Limited Discussion on Practical Constraints and Limitations
4. Lack of Comparison with Other Recent Advances

### Questions
1. Can you show more results on other PDEs, such as NS-Equation and so on?
2. Please compare with recent developed methods instead of original PINNs.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
In this paper, the authors present a hybrid framework which uses learning based B-spline model to approximate the PDEs solutions with different initial and boundary condtions. The proposed method only requires learning the weights of B-spline functions, which enables directly specifying IC/BC and calculating physics-informed loss through analytical formulas. It is demonstrated that the proposed method are universal appoxmiators of arbitrary dimensional PDEs on certain scenarios. The method outperforms the benchamrks on 3D heat equations.

### Strengths
The idea of using B-spline basis functions for PINN sounds novel. There is also a theoretical analysis to assisting the claim that PI-DBSN is a universal approximator under certain conditions. The advantages of analytically calculated gradients look sound, which can help stabilize and accelerate training process.

### Weaknesses
- In the experiments part, for 3D heat equation, it seems that there is no quantitative results for for the accuracy. It is unclear how the proposed method perform comparing to benchmarks especially PINN on this case. 
- In Figure 3, though PI-DBSN converges faster than PINN, the final training loss is much higher than the averaged loss of PINN. In addition, the PI-DBSN almost converge in less than 500 epochs. Does the training process of PI-DBSN not converge properly? Or the test case is too simple so that the ability of the proposed method is not fully tested?
- The authors claim that "we can directly specify Dirichlet initial and boundary conditions through the control points tensor without imposing loss functions, which helps with learning extreme and complex ICBCs". It seems there is no test case with such a extreme and complex ICBCs in the paper. Therefore it is unclear how the proposed method shows advantages on ICBC. Would you minding providing such a case that benchmarks such as PINN and PI-DeepONet fail while the proposed method works?
- It seems that all the performance comparisons are all run on CPU, would you minding providing some results on GPU? If it is hard to get a GPU version, how do you think the results will diff from that on CPU? As GPU might be more friendly for methods with larger NNs for example PINN.
- What is the scale/problem size of the test cases? Can the proposed method scale to much larger test cases?
- The current test cases have relatively simple geometry i.e., rectangular domains, is it possible to apply the proposed method to scenarios with more complex boundary geometries (e.g., arbitrary geometry shape )? or non-convex domain (such as flow past a cylinder)

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
