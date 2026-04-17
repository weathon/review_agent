# Optimal Control under Multiplicative and Internal Noise with Model Mismatch

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Natural agents interact with their environment through noisy and continuous sensorimotor loops. Stochastic optimal control provides a principled framework for this problem, but existing analytical solutions are restricted to linear dynamics with Gaussian observations and additive noise. They cannot address scenarios with multiplicative noise in control or observations, and with internal noise affecting estimation — features central to biological and robotic systems.
We provide a provably convergent algorithm that computes fixed-point controller–filter solutions for linear dynamics with quadratic costs under multiplicative and internal noise. Our method overcomes the limitations of prior analytical approaches and improves the efficiency of state-of-the-art gradient-based methods by more than three orders of magnitude in realistic tasks. Importantly, it also optimizes internal dynamics, relaxing the classical assumption that internal models must match external dynamics. Allowing such model mismatch yields substantially better performance under internal noise.
In sum, we provide the first full solution to stochastic optimal linear control with multiplicative and internal noise, covering both matched and mismatched internal models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a method for filtering of a latent state under, both, multiplicative and internal noises.  This framework extends the traditional linear-Gaussian assumption that makes Kalman filtering optimal.  The method, M-Match, provably converges to a fixed point of the linear-quadratic multiplicative internal (LQMI) model.  The authors further extend the methodology to M-Mis, which allows for model mismatch between the internal dynamics and the extrinsic dynamics in the environment.  The authors demonstrate utility of, both, M-Match and M-Mis along with a baseline (Todorov, 2005) in a 3D reaching task.  Additional experiments are presented in the appendix.

### Strengths
The paper reportedly presents the first analytical and convergent solution for stochastic linear control under multiplicative and internal noise.  The framework is fairly generalizable as it accounts for multiplicative and internal noise as well as model mismatch.  This work substantively builds on the numerical gradient descent method of Damiani et al. (2024) to converge up to three orders of magnitude more quickly.

### Weaknesses
The M-Mis methodology is presented tersely and appears speculative.  Much of the mathematical justification for this method is deferred to the extensive appendix.  Indeed, the entirety of the coordinate descent algorithm, and convergence guarantees, are religated to Appendix A.2.7 and A.3.2.  Due to the relatively brief coverage in the main text, the authors might consider removing the model mismatch component of this work for a more concise submission.

The experimental results provided in the main text are insufficient to support claims made in the paper, and to establish generality of the approach.  Much like in the previous comment, a substantial amount of experimentation is deferred to the appendix.  Even the 3D reaching task, presented in the main text, is not adequately described until the appendix.  Perhaps a journal is more appropriate given that the material in the appendix is critical to the submission.  

The paper fails to adequately contextualize the contributions with respect to prior work.  The authors include a very brief prior work section that only includes two citations (Todorov, 2005; Damiani et al., 2024).  This brevity is despite a long history of research in stochastic optimal control, which has previously considered multiplicative noise models.  Moreover, the authors fail to compare to Damiani et al., 2024 in the results reported in the main text.

### Questions
A brief search finds Wu et al., "Kalman filtering with multiplicative and additive noises." WCICA (2016), which seems quite relevant. How does the present model of multiplicative noise (i.e. ignoring internal noise) and associated M-Match algorithm compare to this prior work.

### Soundness
3

### Presentation
2

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
This work provides a gradient descent algorithm that approximates the solutions for linear dynamics with quadratic costs under multiplicative and internal noise. The theoretical proof section provides performance improvement results. A simple simulation case aims to verify the superiority of sample efficiency over traditional methods.

### Strengths
Detailed appendix.

### Weaknesses
1. Before proposing numerical algorithms, there was a lack of theoretical analysis on optimizing the landscape for the original problem. 
2. The process of proving algorithm theory does not provide a clear proof of convergence. 
3. The comparison method and testing benchmark are relatively simple.

### Questions
1. The paper assumes Gaussian noise distributions. How does the performance of the algorithm change when the noise is non-Gaussian? 
2. The paper mentions the similarity between the proposed framework and reinforcement learning approaches. How do the two approaches differ in terms of their control performance, and computational complexity?
3. While the paper focuses on the coordinate-descent algorithm, it would be interesting to compare its performance and efficiency with other analytical approaches, such as the Kalman-Bucy filter or other iterative methods.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper deals with stochastic optimal control for linear systems with multiplicative and internal noise (LQMI) and presents a provably convergent coordinate descent algorithm that significantly outperforms existing numerical methods. The authors extend this approach to distinguish internal dynamics from external dynamics (“model mismatch”) and argue that this leads to better performance in the presence of internal noise and greater biological realism.

### Strengths
- The coordinate descent algorithm with convergence guarantees (Theorem 1) is a real step forward. The proof that the orthogonality principle only applies when there is zero internal noise (Theorem 2) formally eliminates the ambiguities from earlier work. 
- The speed advantage of over three orders of magnitude compared to gradient-based methods is impressive and of practical importance
- The idea that internal representations do not have to match external dynamics challenges a central assumption of optimal control and could be relevant for neuroscience.

### Weaknesses
## Major
- The restriction to linear dynamics with quadratic costs is limiting. Although the authors acknowledge this, they do not sufficiently address how serious this limitation is. Many interesting control problems are inherently nonlinear. 
- All experiments relate to relatively simple tasks (grasping movements, playful neural networks). The grasping task with one joint has been studied since Todorov 2005. The 3D grasping task is hardly any more complex. The neural steering task is entirely synthetic.
- The work shows that M-Mis helps with internal noise, but mainly in one task (Fig. 1). How does one decide between M-Match and M-Mis in inverse optimal control? Both fit the data, but M-Mis has more parameters.

## Minor
- Section 4 headlines "neural population steering", but does not evaluate the algorithm on that task, but directly refers to the appendix. A summary of the results on this task should be added to that section.
- Eq. 9 & 12 are hard to grasp and contain many indices and transposes. Simplification or a more intuitive presentation would improve comprehensibility.
- Related work: The current related work section is very limited. The work could be better positioned in relation to:
    - iLQG/DDP methods for nonlinear control
    - Model-based RL with mismatched dynamics
    - Work on internal models in neuroscience beyond citations
- A.5: It is good to see that the use of LLM is acknowledged, although “writing assistant” is vague and should be clarified.

### Questions
For inverse optimal control, how would you choose between M-Match and M-Mis given behavioral data alone?

### Soundness
3

### Presentation
2

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
This paper tackles stochastic optimal control in systems with complex multiplicative and internal noise, a scenario often simplified in prior work. The authors introduce a provably convergent coordinate-descent algorithm that is over three orders of magnitude faster than existing numerical methods for this problem. They also propose a novel "Model Mismatch" framework, which relaxes the assumption that an agent's internal model must match external dynamics. This "mismatch" allows the system to achieve substantially better performance, especially under high internal noise, by treating the internal variable as an abstract representation for control rather than a pure state estimator. The primary limitation of this new analytical approach remains its confinement to systems with linear dynamics and quadratic costs.

### Strengths
* The framework includes multiplicative noise and removes restrictive assumptions like the separated inference and control and the orthogonality principle.
* The coordinate-descent algorithm is well-motivated and analytically derived; it achieves substantial computational gains over previous gradient-based methods.
* It introduces a "Model Mismatch" framework, relaxing the common and restrictive assumption that an agent's internal model must perfectly match the external world's dynamics. The authors show that this flexibility leads to substantially better performance, particularly in the presence of internal noise.

### Weaknesses
* The paper is math-heavy and difficult to parse; the clarity could be improved by condensing sections 3.1&3.2 to highlight the key derivation steps, main theorem, and algorithmic intuition, while moving detailed proofs to the appendix.
* The motivations for the multiplicative noise and internal noise are unclear. It can be better illustrated with examples that help readers understand why these noise structures are central to real-world control and biological systems.
* The empirical evaluation is limited to synthetic reaching tasks and toy neural-control settings; there is little demonstration that the method scales to real neural data or application scenarios like robotics.
* The authors suggest that time-varying dynamics could approximate nonlinearities, but the evaluation did not include nonlinear dynamics that are prevalent in neuroscience and robotics; thus, the applicability of the approach is questionable.

### Questions
* What are the concrete scenarios or empirical phenomena that require modeling multiplicative noise? For instance, are there known biological or robotic systems where control- or signal-dependent noise has been experimentally observed?
* A central result is that internal noise biases the internal representations away from the task-optimal (Kalman-like) estimate. What do these new representations encode? If they diverge from accurate estimation, what computational advantage or robustness do they confer?
* Does the model predict measurable behavioral or neural signatures, such as altered variability, control gain modulation, or shifts in sensory weighting, that could be tested experimentally?

### Soundness
4

### Presentation
2

### Contribution
3
