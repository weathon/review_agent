# Efficient Differentiable Contact Model with Long-range Influence

- Decision: Accept (Poster)
- Scores: 2, 4, 8, 8

## Abstract
With the maturation of differentiable physics, its role in various downstream applications—such as model-predictive control, robotic design optimization, and neural PDE solvers—has become increasingly important. However, the derivative information provided by differentiable simulators can exhibit abrupt changes or vanish altogether, impeding the convergence of gradient-based optimizers. In this work, we demonstrate that such erratic gradient behavior is closely tied to the design of contact models. We further introduce a set of properties that a contact model must satisfy to ensure well-behaved gradient information. Lastly, we present a practical contact model for differentiable rigid-body simulators that satisfies all of these properties while maintaining computational efficiency. Our experiments show that, even from simple initializations, our contact model can discover complex, contact-rich control signals, enabling the successful execution of a range of downstream locomotion and manipulation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors formulate four properties a contact model must possess in order to have well-behaved gradients. Those properties are: 1. no interpenetration between contacting bodies, 2. smoothness, 3. unilateral normal forces, 4. non-vanishing gradients. The authors then propose a contact formulation based on potential functions for tedrahedal meshes that satisfies these properties. To increase the computational efficiency of their model, the authors propose to blend between different potentials, i.e., between the exact potential and an approximation based on bounding spheres. The authors evaluate their contact model for different control tasks in simulation and compare to two other contact models and randomized smoothing. The results show that their model provides more informative gradients and enables successful optimization.

### Strengths
-The paper rigurously motivates the proposed properties. 

-The method (including the performance enhancements) is mathematically described in great detail. 

-The results clearly show the gradient quality of the proposed contact model. 

-The contact formulation using hierarchical potential blending is novel and computationally efficient.

### Weaknesses
-The claimed contribution of non-vanishing gradients has been studied several times before and is not novel. E.g. see [1,2]

-The baselines the contact model is compared to are limited and not representative of the relevant models in robotics. E.g. see [3,4]

-The supplementary videos should be presented with more effort, one could at least edit them together into a single video.

-My biggest concern with this paper is that physical accuracy is completely neglected. In the beginning, the authors acknowledge that there is a trade off between physical accuracy and well-behaved gradient information. However, they never investigate this trade off and solely focus on gradient quality in the experiments. In my opinion, very simple, smooth contact models can provide the same gradient quality achieved here with much less computation. The advantage of the proposed model could be increased physical realism but without experiments this is hard to judge. The impact and utility of the proposed method are very unclear. 


[1] Turpin, Dylan, et al. "Grasp’d: Differentiable contact-rich grasp synthesis for multi-fingered hands." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

[2] Schwarke, Clemens, et al. "Learning Deployable Locomotion Control via Differentiable Simulation." 9th Annual Conference on Robot Learning. 2025.

[3] Todorov, Emanuel, Tom Erez, and Yuval Tassa. "Mujoco: A physics engine for model-based control." 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 2012.

[4] Howell, Taylor A., et al. "Dojo: A differentiable simulator for robotics." arXiv preprint arXiv:2203.00806 9.2 (2022): 4.

### Questions
Thank you for the interesting paper! Maybe I misunderstand something here, but I was confused that the rod overcomes the box to push from the other side in the Push experiment. In my mind, even with contact gradients from a distance, the gradients from the side of the box that is closer to the rod in the beginning should always be stronger than those of the further side. What does the gradient look like if the contact surfaces are "behind" each other?

### Soundness
3

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
This paper argues that the choice of contact models for resolving collisions plays a significant role in the quality of gradients computed by differentiable rigid body simulators. They establish necessary properties that a well-behaved contact model should possess, i.e. log-barrier, smoothness, non-prehensile and non-vanishing. The authors then propose a practical contact model which follows all of these properties, and further modify it to improve upon efficiency, while ensuring that the modified contact model is still well-behaved. The proposed method is demonstrated on a variety of gradient based optimization tasks, each of which involves learning a sequence of control signals on controlled objects, which when interacting with other target objects through the proposed contact model, result in a desired motion through simulation. Time complexity analysis is also provided for select tasks.

### Strengths
•	The theoretical contributions of the paper are quite dense and robust, and they cover all aspects of the proposed methodology in detail, with relevant proofs in the appendix. 
	•	The tasks in the provided demos are quite varied and the corresponding results appear to be convincing.

### Weaknesses
- The exposition can be heavy in notation sometimes, making it hard to understand. 
- Although the authors provide theoretical comparisons with other contact models, it would be even better if the comparison can be extended to the demos in the evaluation section. This might provide valuable insights on the importance of each of the properties described in Section 3.
- The authors might want to provide an overview of the methodology in terms of an algorithm pseudocode and/or implementation details, for exposition and reproducibility.  

NOTE:   I reviewed an earlier version of this paper.  Most of the text including typos in figures and equations are exactly the same as last time.   I'd like to ask the authors to please at least correct these typos from previous reviews.

### Questions
•	Typo: In Figure 1, the caption should describe a hexagon, while it currently describes a pentagon
	•	In Section 4, in the expression for computing the contact potential between two triangles, should the RHS be (min L) instead of (argmin L)? 
	•	What do you use for the value of mu, the coefficient of the contact model P? 
	•	When the set of vertices are far enough, i..e. at distance d_2 in the notation of the paper, can we simply use the closed form centered potential as in eqn. (5), instead of blending it with the exact potential at d_1? This might be feasible considering the centered potential satisfies all the properties of a well-behaved contact potential as argued in the proof of Corollary 5.2. The general question here is: what is the motivation behind using blending?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Section 2 provides a brief overview on differential simulators in the field of computer graphics. In Section 3, the authors discuss several desirable properties for gradient-based optimization of differentiable simulators. In particular, the authors point out that a contact potential should **approach infinity when the signed distance approaches zero** (aka follow a "log-barrier"), be **twice-differentiable** (aka "smooth"), **only create pushing forces** (aka "non-prehensible"), and **provide non-zero gradients** even if geometries do not touch ("non-vanishing"). In **Section 4**, the authors extend the contact potential to fulfill several desirable properties for gradient-based optimization of differentiable simulators. In particular, the authors modify the contact potential function of (Liang et al., 2024; Ye et al., 2024) to have global support. However, computing the proposed potential function
P is not efficient as it requires computing the contact potential between each pair of
disjoint triangles **increasing computation quadratically with the number of triangles**. Therefore, the authors suggest in **Section 5**  to interpolate between two potential function as a function of signed distance. If the objects are close the previously derived potential function is used while if the objects are far apart then a faster to compute potential is evaluated that uses a bounding sphere hierarchy to make  the algorithm computational tractable. In **Section 6** the authors show numerous optimizations of different simulation scenes involving simulator gradients.

### Strengths
The method developed for smoothly interpolating between different force potentials is quite clever and will definitely inspire follow up work in the field. The derivation of the potential functions appears to be sound.

- Figure 2 is quite nice.
- Nice experiments (though "only" in simulation).

### Weaknesses
- **C1: The notation is a bit convoluted and the writing could be clearer.**
  Some suggestions:
	- You could simply define index $i$ to belong to vertices I and $j$ to vertices J, then you have $f_i$ instead of $f_{i \in I}^{I \cup J}$ and $x_i \in x_{i \in I}$. Also you could define $\mathcal{P}^{i \cup J}$ instead of $\mathcal{P}$ (as it is clear that the potential function acts between sets of vertices).
	- I am not sure what Definition 3.1 is defining and why it required the use of a "Definition" statement.
	- "Lemma 4.1" seems also unnecessary. Also why define properties through numbers (e.g. Property 3.1, 3.2, 3.3, 3.4)?
     - "P(x)=\infty iff x \in C_obs" seems imprecise/wrong.
	
- **C2: Adding naive "global support" of the contact potential degrades simulation realism.** If you extend the force potential to provide global support then you apply contact forces in the simulation between object that do not touch. While these forces are small, they may notably alter the simulation. For simulation of real-world scene, extending global support could introduce a reality gap and adversely effect control synthesis. That said, in the videos supplied with the supplementary material,  I could not observe such artifacts. That said, the arxiv paper [Hard Contacts with Soft Gradients: Refining Differentiable Simulators for Learning and Control](https://arxiv.org/abs/2506.14186) might be interesting to the authors (this is an arxiv paper so there is no need to discuss it in the author's paper). In principle, the authors could use straight-through estimation to only use the contact potential with global support for the gradient computation.

### Questions
- Q1: The paper https://github.com/taichi-dev/difftaichi (not cited by the authors) discusses how time-discretization causes gradient errors for stiff contact simulations. How does increasing contact stiffness or integration times affect the gradient quality of the proposed simulator?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a novel differentiable contact model for rigid-body simulation that guarantees both physical realism and well-behaved gradients. The authors identify key properties - log-barrier, smoothness, non-prehensility, and non-vanishing gradients as necessary for reliable gradient-based optimization in differentiable physics. They propose a new contact potential formulation that satisfies all these properties and introduce a Bounding Sphere Hierarchy (BSH) mechanism for computational efficiency, reducing the cost of evaluating contact interactions. Extensive experiments across contact-rich tasks demonstrate that this model produces smoother gradients, faster convergence, and improved stability compared to existing baselines.

### Strengths
1. **Strong theoretical foundation** - the work is grounded in rigorous mathematical analysis, formally defining the necessary conditions for a “well-behaved” contact potential. The authors’ proofs (Appendix A.1–A.3) are precise and significantly strengthen the paper’s technical depth.
2. **Novel and timely contribution** - differentiable contact modeling remains a bottleneck in differentiable simulation. Introducing a model that satisfies both smoothness and non-vanishing gradient properties is highly relevant, especially as differentiable simulators gain traction in RL and robotics (see Werling et al., 2021; Xu et al., 2022; Ye et al., 2024).
3. **Thorough experimental validation** - the experiments are well-designed and comprehensive, spanning both manipulation and locomotion tasks. The inclusion of both trivial and random initialization demonstrates robustness.
4. **High-quality writing and presentation**  -the manuscript is clear, well-organized, and logically develops from theoretical properties to practical implementation.
5. **Impact potential** - the long-range gradient property directly addresses a key failure mode in current differentiable physics pipelines (gradient vanishing in contact-free configurations), with broad implications for differentiable MPC, co-design, and world-model-based RL.

### Weaknesses
1. **Terminology ambiguity** - the use of “log-barrier” as a property (Property 3.1) is somewhat confusing. In optimization literature, log-barrier refers to a method, not an inherent property. Reframing this as a “barrier potential” or “barrier-form property” would avoid conceptual conflation.
2. **Referencing and clarity**
    * Propositions 3.3a and 3.3b are not explicitly labeled in Table 1, which may confuse readers unfamiliar with the notation.
    * The color scheme in Figure 2 is not self-explanatory; a legend or caption clarification would improve readability.
    * Figure 3 is visually appealing but does little to elucidate the core algorithmic contribution; it could be condensed or better explained.
3. **Accessibility for newcomers** - since the proposed formulation heavily builds on Liang et al. (2024) and Ye et al. (2024), a brief, self-contained overview of those works would make this paper more approachable.
4. **Presentation details** - several figures (especially in Sections 5–6) are too small to read comfortably. As a general guideline, the figure text size should match the main body font.
5. **Limited domain variety** - while the manipulation and locomotion experiments are strong, an articulated manipulation case (e.g., as in Srinivasan et al., 2024) would better demonstrate real-world robotic applicability.

### Questions
1. What does "GD" baseline refer to in Figures 4,6,7, and 8?
2. Given Figure 5, could the authors quantify how their method scales to mesh complexities in realistic robots (>10k triangles)? Is RL or trajectory optimization still feasible?
3. The method currently applies to rigid-body systems. Do the authors foresee any principled way to extend the non-vanishing gradient property to deformable or soft-body interactions?
4. The hierarchical blending (Eq. 8) relies on ϵ > 0 to guarantee theorems. How sensitive are results to this choice in practice?

### Soundness
4

### Presentation
4

### Contribution
4
