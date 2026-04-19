# Non-Equilibrium Dynamics of Hybrid Continuous-Discrete Ground-State Sampling

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 6

## Abstract
We propose a general framework for a hybrid continuous-discrete algorithm that integrates continuous-time deterministic dynamics with Metropolis-Hastings (MH) steps to combine search dynamics that either preserve or break detailed balance. Our purpose is to study the non-equilibrium dynamics that leads to the ground state of rugged energy landscapes in this general setting. Our results show that MH-driven dynamics reach ``easy'' ground states more quickly, indicating a stronger bias toward these solutions in algorithms using reversible transition probabilities. To validate this, we construct a set of Ising problem instances with a controllable bias in the energy landscape that makes certain degenerate solutions more accessible than others. The constructed hybrid algorithm demonstrates significant improvements in convergence and ground-state sampling accuracy, achieving a 100x speedup on GPU compared to simulated annealing, making it well-suited for large-scale applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a novel hybrid continuous-discrete algorithm that combines deterministic continuous dynamics with Metropolis-Hastings (MH) steps for ground-state sampling in non-equilibrium dynamics.

### Strengths
1. The combination of chaotic dynamics with an MH step to ensure convergence to a target distribution represents an innovative approach to hybrid sampling.
2. The hybrid algorithm shows potential for improving ground-state sampling performance in combinatorial optimization
3. The authors address parallelization for efficient computation on GPUs
4. The discussion on combining chaotic dynamics with probabilistic methods provides a useful context for researchers working at the intersection of machine learning and statistical physics.

### Weaknesses
While the authors present an interesting optimization algorithm, the clarity of the writing is a major concern. The main ideas are difficult to follow in the current presentation. I would encourage the authors to reconsider their notation and improve their writing to convey their ideas more effectively to readers.


**Major concerns**
1. The momentum and pre-conditioning typically serve different roles in optimization: momentum accumulates past gradients, and pre-conditioning captures curvature information. I do not think the connection demonstrated in Section 3.2 is trivial. A detailed explanation is needed in the main text to connect them.
2. The paper lacks theoretical guarantees regarding the convergence or performance of the proposed algorithm.
3. Since the algorithm incorporates a momentum variable, it would be more consistent to account for this momentum within the MH step (Equation (7)), rather than applying it solely to $\boldsymbol{\sigma}$.

**Other suggestions**
1. The paper lacks clear definitions for essential variables (e.g., $\boldsymbol{u}$ and $\boldsymbol{e}$ in Equation (1)).
2. The time variable $t$ is used ambiguously. It denotes continuous evolution in Equation (1) but has discrete updates in Equations (4)-(5).
3. Although the authors appear to focus on ground-state sampling, the formulation provided in Section 3.2 is more oriented toward sampling from a Gibbs measure rather than explicitly defining the ground-state sampling.
4. Ensure that all variables and abbreviations are clearly defined. For example, abbreviations such as SA, HNN, and CACm used in Table 1 should be explicitly explained in the main text.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents two variants of the CAC (chaotic amplitude control) algorithm, namely CAC with momentum (CACm) and Metropolis-Hastings CAC with momentum (HMCACm). CACm is a deterministic continuous-time dynamical model for combinatorial optimization, and HMCACm is a Metropolis-Hastings adjusted version of CACm with the Boltzmann distribution as the theoretical equilibrium. MHCACm can be regarded as a unified framework that generalizes many existing methods, including simulated annealing, Hopfield neural networks, analog iterative machines, and CAC(m). Numerical results show that this method illustrates faster relaxation time on NP-hard problems. In particular, MHCACm exhibits excellent performance in sampling from easier ground states, which may be relevant to training over-parametrized neural networks.

### Strengths
- State-of-the-art algorithms exploiting relaxation to a continuous state form discrete combinatorial optimization do not sample fairly from the Boltzmann distribution due to the lack of detailed balance. The MHCACm algorithm fills in this conceptual gap by adding a Metropolis-Hasting step. This new design ensures that MHCACm samples fairly from a discrete distribution while iterating over the relaxed (continuous) search space. 
- The numerical results look strong. Table 3 shows that MHCACm has a success probability higher than dSBM, another well-known Ising solver based on GPU. 
- MHCACm is well-suited for large-scale deployment on GPU because its computational bottleneck is matrix-vector multiplication.

### Weaknesses
- Little theoretical justification for the effectiveness of MHCACm is provided. Specifically, it is not clear why a fair sampling strategy in the CAC framework would lead to a better performance in an optimization (ground state finding) problem. Relaxation to the ground state does not necessarily need to go through a detailed-balance algorithmic path. It would be nice to discuss how the Metropolis-Hastings step interacts with the CAC dynamics to potentially improve optimization performance.
- No empirical results for real-world optimization problems. While the performance of MHCACm has been benchmarked over the dWPE instances and GSET, these test instances are highly artificial and may not reflect the performance of the algorithm in a practical setting (e.g., quadratic assignment problems, portfolio optimization problems, etc.).

### Questions
- The paper only reports the TTS in the experiments on dWPE instances (section 4.4). It would be more transparent if the success probability and runtime data could be provided as well, as it is not clear whether the advantage comes from a higher success probability or a shorter wall-clock runtime due to GPU parallelization. 
- Can you elaborate more on the "dual-primal Lagrangian approach" and its difference from CAC?
- Some minor typos: e.g., line 122 "in order (to) benchmark this algorithm's ability".

### Soundness
3

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
The paper proposes a class of hybrid continuous-discrete algorithms by integrating continuous dynamics with Metropolis-Hastings steps. The paper also constructs a set of Ising problems with a tunable parameter to trade off between easy ground states and hard degenerate ground states, in order to experiment with the bias of different algorithms. The proposed class of algorithms are also fast solvers that achieve a great amount of acceleration on GPU due to a parallelizable structure.

### Strengths
* The paper writing is nice and structured, with a comprehensive literature review and detailed problem set-up.
* The paper looks technically sound with solid mathematical proof.
* The proposed algorithm is evaluated on multiple tasks and compared with various other benchmark methods, showing competitive performance.

### Weaknesses
* I am not very familiar with the literature, but seems that the tasks of ground-state sampling are not formally defined in the paper, as well as the idea of non-equilibrium dynamics.
* The connection between ground-state sampling and deep learning optimization/generalization mentioned in the paper is interesting, but the discussion is very limited. 
* For numerical experiments, the definition of TTS is hard to comprehend. Does smaller TTS indicate better algorithmic performance?

### Questions
* Based on the algorithmic design in this paper, is there any insight we can draw on what an ideal optimizer for deep neural nets should look like?
* Can you elaborate more on the numerical performance of CACm and MHCACm? From the charts, the two performances seem to be close to each other.

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
Overall, this paper proposes a promising hybrid continuous-discrete sampling framework that demonstrates clear benefits in convergence speed and sampling accuracy for rugged energy landscapes. This paper could benefit from additional theoretical insights and comparisons with established methods.

### Strengths
(1)The proposed method effectively leverages the Metropolis-Hastings method within a continuous-discrete framework, enhancing sampling efficiency for ground-state discovery in challenging discrete landscapes. This approach demonstrates practical advantages, such as notable improvements in convergence speed and sampling accuracy on GPU architectures.

(2) The method’s focus on non-equilibrium dynamics and its capacity to identify accessible ground states faster than traditional approaches offer a valuable contribution to optimization in rugged energy landscapes.

### Weaknesses
(1) Although the paper demonstrates the practical benefits of the hybrid continuous-discrete approach, the theoretical understanding of the sampling properties of the MHCACm algorithm remains unaddressed. I wonder if the authors could provide a discussion on potential directions for analytical proof of the sampling capabilities of MHCACm, such as convergence rates or mixing times. 

(2) I would like to suggest the authors include more comparison with other prominent sampling algorithms using collective variables, for example, 'Sampling metastable systems using collective variables and Jarzynski–Crooks paths' by G. Stoltz et al. 

In particular, I am curious  to see how the use of collective variables in that work relates to or differs from the proposed method of this paper, and if the authors can combine both approaches.

### Questions
(1)  While the method achieves a 100x speedup over simulated annealing on GPUs, a discussion of any limitations or computational trade-offs encountered in specific scenarios (such as highly multimodal landscapes) would be beneficial, for instance, I wonder if the authors could provide specific examples of problem types or landscapes where their method may face challenges.

(2) I wonder if the authors could provide additional insights into how MHCACm scales with increased problem complexity, for instance, if the authors could demonstrate how the empirical scaling results and the performance of proposed algorithm changes with increasing complexity for a range of benchmark examples.

(3)  The paper implies that the method’s bias towards “easy” ground states is advantageous, but this effect could also limit the algorithm’s ability to reach more challenging or rare ground states. I wonder if the authors could provide quantitative results on the algorithm's performance in finding both "easy" and "hard" ground states across different problem instances. 

(4) Additionally, I wonder if the authors could discuss potential modifications to the algorithm that could help balance the performance of exploration of both easy and hard ground states.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new algorithm that combines chaotic search and Metropolis-Hastings. The goal seems to solve optimization problems in discrete non-convex energy landscapes. The proposed algorithm is tested on several combinatorial optimization tasks.

### Strengths
- The empirical results include multiple baselines and comparisons in terms of different metrics.
- The visualization in Fig.1 clearly shows the main algorithmic idea.

### Weaknesses
- The problem that this paper aims to solve is vague. Is the goal to develop an algorithm that samples better in non-convex energy landscapes, or for optimization in discrete landscapes? Or is the goal to understand non-equilibrium dynamics in non-convex energy landscapes? Similarly, the motivation of the proposed algorithm which combines chaotic search with Metropolis-Hastings is not well-explained.
- The novelty of the proposed algorithm is unclear. Is the algorithm a straightforward combination of chaotic search and MH? If not, what is the challenge, and how does the paper solve the challenge? 
- The empirical improvement is not consistent. For example, Fig.2 shows that CACm is better than proposed method also the variance of the proposed is significantly larger than the baselines.
- The runtime comparison only considers simulated annealing. It will be better to include other baselines as well.
- The paper compared the standard Gibbs with gradient which is developed for combinatorial optimization. It will be more convincing to compare with gradient-based discrete MCMC that is developed for CO, such as [1].

[1] Revisiting sampling for combinatorial optimization, ICML 2023

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
2
