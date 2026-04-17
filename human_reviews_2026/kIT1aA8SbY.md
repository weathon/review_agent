# Stability in Training PINNs for Stiff PDEs: Why Initial Conditions Matter

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Training Physics-Informed Neural Networks (PINNs) on stiff time-dependent PDEs remains highly unstable. Through rigorous ablation studies, we identify a surprisingly critical factor: the enforcement of initial conditions.  We present the first systematic ablation of two core strategies, hard initial-condition constraints and adaptive loss weighting. Across challenging benchmarks (sharp transitions, higher-order derivatives, coupled systems, and high frequency modes), we find that exact enforcement of initial conditions (ICs) is not optional but essential. Our study demonstrates that stability and efficiency in PINN training fundamentally depend on ICs, paving the way toward more reliable PINN solvers in stiff regimes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the stability challenges encountered when training PINN on time-dependent PDEs. The authors conduct a systematic ablation study focusing on hard initial-condition constraints (HC-PINN). Their central finding and claim is that the exact enforcement of IC, achieved through a hard-constraint transformation integrated into the network architecture, is "not optional but essential" for achieving stability and efficiency in stiff PDE regimes. A theoretical perspective using the NTKs is presented to explain how the hard constraint mitigates spectral bias.

### Strengths
The paper provides a rigorous and systematic ablation study that is valuable for practitioners in the PINN community. The empirical results clearly demonstrate the practical benefit and robustness of the hard-constraint approach compared to soft constraints with adaptive weighting on the subset of stiff PDEs

### Weaknesses
- Lack of technical novelty. The core technique of using a transformation to enforce boundary and initial conditions exactly is a well established method in PDE solving with neural networks, predating much of the PINN literature. The paper's contribution is primarily a theoretical analysis and an engineering validation of this known technique on a specific problem class.

- Hardness of experiment cases. The chosen subset of PDEs used to demonstrate HC-PINN is limited to one-dimensional problems that can be efficiently solved using numerical schemes (<< the cost of training a PINN). While these problems have been extensively used to benchmark PINNs in ML literatures, their practical importance is limited to make them beyond 'toy examples'.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents the first systematic ablation study of stabilization strategies for Physics-Informed Neural Networks (PINNs) on stiff time-dependent PDEs. The authors argue exact enforcement of initial conditions is crucial for training stability and they propose a hard-constraint PINN (HC-PINN) formulation that embeds initial and boundary conditions directly into the network parameterization. A theoretical Neural Tangent Kernel (NTK) analysis and benchmarks on seven stiff PDEs show that HC-PINNs achieve up to $10^3 \times$ lower error and greater stability than standard PINN variants.

### Strengths
- The paper investigates an important problem, enforcing hard constraints, which is of general interest to the PINNs community. While enforcement of hard constraints is a techniques previously implemented across the PINN literature, a thorough ablation study of the effectiveness of this technique has not been previously conducted, to my knowledge.
- The experiments are generally thorough, with evaluations on seven stiff PDE systems and detailed ablations in the appendix.
- The method is supported by theoretical analysis in an NTK setting.

### Weaknesses
- I'm slightly concerned about the novelty of the contribution. I believe the rigorous ablation studies of hard constraint enforcement is a novel and important contribution for the PINN community. However, the work also seems to claim their HC-PINN (specifically their approach as defined in Section 2) as a novel contribution. These techniques have been present in the literature, for example see the reparameterizations in [1, 2, 3]. I believe the work could benefit from clarifying that the main contribution is the thorough ablation study + theoretical analysis.
  - As another example, Theorem 1 appears to be a standard result: it simply requires $\phi$ and $\psi$ to have all the time and spatial derivatives the PDE originally demanded of $u$.

- The experimental results do not seem to compare to state-of-the-art PINN methods. See questions for more details.

- The writing clarity of the paper could be improved, especially in the theoretical sections (2 and 3).
  - I found the loss definitions around lines 175-182 quite difficult to parse, especially the two different BC Loss definitions (periodic vs. non-periodic. Adding a bit of description in text may improve the clarity of this section.
  - Equation 4 seems to contain several typos. It should read $\phi(0, x) = 0$ and $\phi(t, x) = 0$ accordingly. Again, adding descriptions here would improve clarity.
  - Theorem 1's label "well-conditioning" seems to be misleading. It seems to be a proof of well-posedness, including ensuring non-singularity near $t=0$.
  - I found Theorem 2 quite confusing to parse. The authors should more clearly specify that $h$ represents the hidden-layer outputs and clarify what "the two losses" (lines 277-282) are (I'm assuming they correspond to the equations on lines 171 and the hard-constrained version defined through Section 2.1).

- The paper seems to make several claims that are somewhat vague and not well-supported by evidence. For example, in Section 3.2, lines 265-269, the authors argue HC-PINN achieves superior performance because
  - it "implicitly enforce[s] the time direction into the training of PINNs, reducing the spectral bias from training with multi-objective loss".
  - the hard constraint transformation works "similarly like an implicit time integrator".
  
  It would be useful to clarify how the method is similar to an implicit time integrator, and for the spectral bias, to define it precisely and to provide evidence for this claim with an experiment.

1. Lu et al. 2021. Physics-informed neural networks with hard constraints for inverse design
2. Dong et al. 2021. A Method for Representing Periodic Functions and Enforcing Exactly Periodic Boundary Conditions with Deep Neural Networks
3. Wang et al. 2023. An Expert's Guide to Training Physics-informed Neural Networks

### Questions
- The authors mention in Table 1 that the methods are all trained for 50k training iterations instead of the 300k iterations of the original papers [1, 2]. Is there a particular reason why 50k iterations were used instead of 300k? Do the HC-PINNs continue to outperform the baselines after 300k iterations?
- Did the authors try running their ablations using more state-of-the-art PINN implementations, such as [3, 4]? In particular, [3] uses causal training and obtains better results than the reported baselines for 1D AC and KS, and [4] finds several order-of-magnitude improvement by using an improved optimizer. I'm wondering if the findings about the benefits of hard constraints hold up when using these improved training techniques.

1. Wang et al. 2022. Respecting causality is all you need for training physics-informed neural networks
2. Anagnostopoulos et al. 2024. Residual-based attention in physics-informed neural networks
3. Wang et al. 2023. An Expert's Guide to Training Physics-informed Neural Networks
4. Kiyani et al. 2025. Optimizing the Optimizer for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks

### Soundness
2

### Presentation
1

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
The authors study the impact of different components on the training of PINNs. The authors identify a major bottleneck in the stable training of stiff PDEs which is the enforcement of initial conditions. Based on this observation they develop a technique to integrate the hard constraints into the PINN formulation itself. Two major approaches to mitigate the stability issue are discussed which include incorporating hard constraints and adaptive loss weighing strategies. Through extensive experiments on a varied set of equations (covering different types of systems) it has been shown that HC-PINN is shown to be superior to baseline PINNs.

### Strengths
S1) Along with experimental results a theoretical analysis of the hard constraint formulation using Neural Tangent Kernel (NTK) is provided. This enables the PINN to capture the exact ICs and hence helps reduce the spectral bias.

S2) The proposed technique can be combined with other existing advanced PINN variants such as time marching PINNs, causal PINNs, RBA PINNs and curriculum training.

S3) Detailed experimental evidence on a wide range of PDEs including comparison with other PINN variants. Very elaborate ablation studies have been carried out on all the 7 PDEs

### Weaknesses
W1) Detailed ablations and explanations on the choice of the decay rate $C$ have not been included. Two specific values for the decay rate have been used.

W2) As per my understanding only periodic boundary conditions have been studied, how does this framework extend to other types of boundary conditions. Is this a limitation or only periodic conditions have been chosen to be studied in this work.

### Questions
Q1) Figure 3 shows a sharp jump at a particular threshold value of $m$, why is that so, could there be an explanation for this?

Q2) Related to W1, how should a choice of C be made, how to physically interpret it?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes to modify the training of PINNs by incorporating initial conditions and/or boundary conditions as hard constraints on the neural network model of the solution, instead of as an additional loss term. This results in learned solutions that always exactly satisfy the initial/boundary conditions. Numerical experiments show that HC-PINN results in improved error compared to standard PINNs. Some theoretical results on the conditioning of the NTK of HC-PINNs are also provided.

### Strengths
* The paper tackles an important problem in PINN training, namely the hard enforcement of initial/boundary conditions with a simple and easy to implement modification of the models.
* Numerical results show that there is an improvement in error using HC-PINNs.
* The technique extends to standard IC/BC as well as periodic BCs.
* Theoretical results on NTK conditioning are provided.

### Weaknesses
* Numerical evaluation is done only on 2D settings. 
* While the final learned solutions have improved error, they are still far from machine precision, which is more important and relevant for PDE applications.
* Experimental tuning is limited (I believe only one architecture is studied without much tuning of optimization parameters).
* Presentation and exposition is very poor. For example:
  - Line 200 does not make sense, $\psi$ is both 0 and $u_0$. No conditions are given on $\phi$. This is clearly a typo but it is a very confusing one, and it greatly impacts the clarity of presentation. 
  - Theorem 2 is very confusing, and there is no clear statement in the theorem. The presentation would be vastly improved by encapsulating assumptions and intermediate results into lemmas, and making the final theorem a very simple and precise statement.

### Questions
1. Can you summarize in simpler terms what the quantitative result of Theorem 2 actually means?
2. How well would this method generalize to higher dimensions?
3. Could this method be combined with other approaches to achieve high precision results?

### Soundness
2

### Presentation
1

### Contribution
2
