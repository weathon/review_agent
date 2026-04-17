# CHyLL: Learning Continuous Neural Representations of Hybrid Systems

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 6

## Abstract
Learning the flows of hybrid systems that have both continuous and discrete time
dynamics is challenging. The existing method learns the dynamics in each discrete
mode, which suffers from the combination of mode switching and discontinuities
in the flows. In this work, we propose CHyLL (Continuous Hybrid System
Learning in Latent Space), which learns a continuous neural representation of a
hybrid system without trajectory segmentation, event functions, or mode switching.
The key insight of CHyLL is that the reset map glues the state space at the
guard surface, reformulating the state space as a piecewise smooth quotient manifold
where the flow becomes spatially continuous. Building upon these insights
and the embedding theorems grounded in differential topology, CHyLL concurrently
learns a singularity-free neural embedding in a higher-dimensional space
and the continuous flow in it. We showcase that CHyLL can accurately predict
the flow of hybrid systems with superior accuracy and identify the topological invariants
of the hybrid systems. Finally, we apply CHyLL to the stochastic optimal
control problem.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The authors propose chyLL, a method to learn continuous representations of hybrid systems in latent space. The method ensures the representations become continuous in a higher-dimensional latent space, then fit a latent ode there and decode back, avoiding mode labels, event functions, or segmentation. The contribution is the gluing/conformal/anticollapse losses to learn this manifold from time-series alone, with demos (bouncing ball, torus/klein bottle topology checks, and a control task).

### Strengths
The paper is clear and well-motivated, and the pipeline and objectives are easy to follow. 
The route explored is worthwhile as a method to modeling hybrid dynamics that generalizes across systems. 
The proposal of learning a glued quotient manifold for hybrids via gluing + conformal losses is nice, and the training setup seems sensible (e.g. curriculum, anti-collapse, LM projection).

### Weaknesses
1. Although latent encoding for ODEs is not a new avenue, the quotient/gluing idea is a nice addition, but as the authors hint at in the paper, learning the correct 'glued' space might be hard in principle, and the lack of guarantees can be concerning. 

2. The experimental scope is a little narrow, with toy problems/examples, and only a few comparative methods. The results for the ball juggling with MPPI experiment also show a deep Koopman baseline achieving a lower mean tracking cost, without much of a detailed explanation. 

3. It's hard to comment on the scalability and robustness of the approach since there are no results under sensor noise/partial observability, many-guard/mode systems, or higher-dimensional robotics experiments.

### Questions
2. The authors should expand further on the limitation of the proposed method, and its failure mode, with respect to existing literature.

1. Can you provide any diagnostic or bound (e.g., jump norms across detected guards, encoder/decoder Jacobian conditioning near resets) that indicate the learned latent is truly continuous, or conditions under which it fails?

2. How sensitive is performance to the gluing/conformal/anti-collapse weights and latent dimension? It would be valuable to include a sweep or at least failure modes when turning each off.

3. Can you expand further on why deep Koopman wins on mean cost, and whether CHyLL improves with different horizon/MPPI settings or controller?

4. How do you think the method would behave under more realistic sensor noise or partial observability?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
CHyLL presents a groundbreaking method for learning hybrid system dynamics directly from time-series data, eliminating the need for mode switching or event detection. Its core innovation lies in reformulating the discontinuous state space into a continuous quotient manifold—the hybrifold—by topologically gluing guard surfaces via reset maps. A dual-phase training strategy separately optimizes the continuous encoder/flow and the discontinuous decoder. Evaluations on systems like the bouncing ball and torus show CHyLL's superior prediction accuracy and its ability to identify correct topological invariants.

### Strengths
This article is quite abstract, drawing on profound mathematical theories to guide the methodology design. It addresses a very practical problem and has achieved promising results in the selected experimental examples.

### Weaknesses
1. Although this article cites many theorems and employs sophisticated mathematical frameworks, its contributions are primarily concentrated on methodological design, with the theoretical contributions not being sufficiently sound. Perhaps the authors could further incorporate theoretical analysis or proofs regarding their methods (such as the design of the loss Eq. 4).
2. The experiments in the paper all use simulated data from simple examples. There is an absence of benchmarking on publicly available real-world datasets.

### Questions
1. Although profound mathematical theorems are cited in the paper, I have concerns regarding their rigor. For instance, Theorem 1 states that the manifold is piecewise smooth, while Theorem 2 requires the manifold to be C^r—what is the relationship between these two conditions? Additionally, I seem to be unable to find the precise definition of 'piecewise smooth'.
2. Could there be performance evaluations on some real-world public datasets?

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
4

### Summary
The paper studies the problem of learning hybrid dynamical systems, that is the systems combining continuous flows and discrete mode switches. Diversly from existing approaches, authors propose method CHyLL (Continuous Hybrid System Learning in Latent Space) that   learns directly from time-series data avoiding major scalability bottlenecks of explicit segmentation or event detection. The central insight is topological: hybrid systems’ discontinuities arising from mode switching can be “glued” to form a piecewise smooth quotient manifold, on which the overall flow becomes spatially continuous. CHyLL operationalizes this known theoretical result, by: (1) learning a singularity-free neural embedding of the quotient manifold in a higher-dimensional latent space, (2) concurrently learning the continuous flow within this embedded space, and (3) enabling prediction and control of hybrid dynamics without explicitly modeling discrete modes or resets. Experiments show that CHyLL can reconstruct hybrid system flows with high accuracy, recover topological invariants, and be applied to stochastic optimal control problems.

### Strengths
(1) I find paper generally well written, besides few minor issues listed bellow. I like the clarity of motivation, transparency of novelties and  appreciate the balance in presenting both intuition and technical complexity.

(2) Up to my knowledge, CHyLL appears genuinely novel in its topological formulation of hybrid system learning. The reinterpretation of mode switches as gluing operations forming a quotient manifold and the learned embedding that makes hybrid flows continuous is non-standard path. While some ideas overlap with Neural ODE extensions and manifold-learning methods, no prior work explicitly connects hybrid system topology, latent manifold embedding, and continuous neural flow learning in a single approach.

(3) I believe that making the embedding theorem (Simic et al. 2005) from differential topology operational within Neural ODEs can inspire new methods that up to known were not able to successfully tackle hybrid systems.

(4) Experimental setup is generally appropriate.

### Weaknesses
(1) Presentation:
- Section 3: I find that introduction of main concepts like guards and resets should be smoother. Before jumping to formal Definitions, authors can use Figure 2 to introduce these concept first informally to build the intuition. e.g. in simple terms, what is the role of $q$.
- Section 2: I feel that related work section would be easier to parse after the intuition and notation on hybrid systems is current Section 3.
- Levenberg-Marquardt: Due to unclear Experimental conclusions (lack of std),  this aspect can be either avoided or emphasised. Computational overhead of using it should be discussed. 

(2) Experiments: Standard deviations across trials is missing in experiments of Section 6.2. This makes it hard to conclude which version of the proposed method (w/o LM) is better performing, and diminishes the impact of the conclusions. 
 
(3) Minor:
- should read "data" in line 182
- "be" lacking in line 189
-  notation should be ${\cal L}_c(\theta)$ in line 299

While my current score reflects the above weaknesses, I am happy to revise it if the rebuttal is successful.

### Questions
Given the good performance of DynamicAE of Lusch et al. 2018 in experiment of Section 6.3, what do authors think of adding additional  discussion on combining CHyLL with Koopman-based method. Namely, instead of using parametrising the vector filed and using Eq. (5) to evolve the latent dynamics, Koopman approach would learn linear evolution. Also, since DynamicAE is known to fail in modelling evolution in longer time-horizon, one can think of combining the CHyLL with representation learning for Koopman/Transfer operators, e.g. of references bellow, to learn appropriate representations of hybrid systems. 

Han et al. Deep learning of Koopman representation for control, IEEE CDC2020
Kostic et al. Learning invariant representations of time-homogeneous stochastic dynamical systems, ICLR2024
Kostic et al. Neural conditional probability for uncertainty quantification, NeurIPS2024
Jeong et al. Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems, arXiv preprint 2025

### Soundness
4

### Presentation
3

### Contribution
4
