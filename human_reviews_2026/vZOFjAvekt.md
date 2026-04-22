# Test-time Generalization for Physics through Neural Operator Splitting

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Neural operators have shown promise in learning solution maps of partial differential equations (PDEs), but they often struggle to generalize when test inputs lie outside the training distribution, such as novel initial conditions, unseen PDE coefficients or unseen physics. Prior works addresse this limitation with large scale multi physics pretraining followed by fine tuning, but this still requires examples from the new dynamics, falling short of true zero shot generalization. In this work, we propose a method to enhance generalization at test-time, i.e, without modifying pretrained weights. Building on DISCO, which provides a dictionary of neural operators trained across different dynamics, we introduce a neural operator splitting strategy that, at test time, searches over compositions of training operators to approximate unseen dynamics. On challenging out-of-distribution tasks including parameter extrapolation and novel combinations of physics phenomena, our approach achieves state-of-the-art zero shot generalization results, while being able to recover the underlying PDE parameters. These results underscore test-time computation as a key avenue for building flexible, compositional, and generalizable neural operators.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a test-time adaptation approach for neural surrogate models to improve the out-of-distribution (OOD) performance. The authors combine pretrained operators and a beam search method, and a test-time scaling law is provided. There are two OOD scenarios considered in this paper, i.e., parameter extrapolation and operator composition. Several numerical experiments have shown the superiority of the proposed test-time computation framework compared to baseline models. 

Contributions:

- The authors are tackling the critical OOD problem in scientific machine learning. 

- The test-time adaptation in scientific machine learning is new.

### Strengths
- The test-time generalization in scientific ML is under-explored. This paper investigates a critical topic.

- This paper is well-written. The motivation and formulation of test-time scaling are well presented.

### Weaknesses
This is a good topic, but I have a few concerns regarding the experiments part. 

- First, I think the OOD scenarios can be broader. Apart from the parameter extrapolation and operator composition, the authors might also consider unseen initial conditionals, boundary conditions, geometries, etc. Please refer to the unisolver paper [1]. 

- Second, it would be good to have a more explicit discussion of computational overhead. On Page 2, the authors also claimed that “This test-time strategy comes at a higher computational cost, but enables to better adapt when faced with unseen dynamics.” It is good to see the performance improvement in the paper, but it would also be useful to see how much extra computation this actually requires. I think including some numbers or plots on runtime or resource use would give readers a better sense of the tradeoff between performance and efficiency. 

- Third, the tested PDEs can be broader. The Navier-Stokes equations seem to be one of the standard benchmark datasets that people will test. The authors might also consider testing on more diverse PDEs. 

---

**Refs:**

[1] Zhou, Hang, et al. "Unisolver: PDE-Conditional Transformers Towards Universal Neural PDE Solvers." Forty-second International Conference on Machine Learning.

### Questions
- Could you clarify the motivation for using LoRA? How large is your model? If it is relatively small, why was full model fine-tuning not considered as an alternative?

- How does this method scale to 3D PDEs?

- In the Abstract, it is better to cite the DISCO paper there to avoid confusion.  

- On Page 9, line 440, there is a citation issue for Figure ??.

### Soundness
2

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
This paper introduces a test-time adaptation strategy for neural PDE surrogates that enables zero-shot generalization to out-of-distribution dynamics. The method builds on DISCO, extracting a dictionary of neural operators from training trajectories, then uses beam search with operator splitting at test time to compose these operators and approximate unseen dynamics. The approach is evaluated on parameter extrapolation and physics composition tasks across three benchmarks. Results demonstrate significant improvements over baselines.

### Strengths
- The core idea of combining a learned dictionary of neural operators with classical operator splitting at test time is, to my knowledge, highly novel. This is a well-motivated approach of bringing forth test-time computation to the realm of PDE surrogate modeling, drawing clever parallels to LLM inference techniques (beam search, best-of-N sampling).
- A valuable feature of this formulation is the interpretability aspect. By analyzing the selected operator combinations, we can perform zero-shot parameter estimation.
- The benchmarks performed demonstrate strong performance achieving order of magnitude improvements over the baselines considered.

### Weaknesses
- While operator splitting has theoretical foundations for classical numerical methods, there's no analysis of when/why it works for learned neural operators. What's the role of the approximation error of the individual operators? Have you tried to study the convergence behavior of these splitting schemes (Lie or Strang)? How does the approximation error of the individual neural operators interact with the splitting error of the numerical scheme?
- While computational complexity is stated, actual runtime comparisons with baselines are absent. How does test-time compute compare to simply fine-tuning?
-  Training only on single-operator dynamics seems like a constraint. How would the method perform if training included some operator combinations? Additionally, it seems like the framework is restricted to purely additive composition. Can the method ever isolate a pure diffusion operator if its dictionary only contained reaction-diffusion and reaction operators?

### Questions
Please address weaknesses. Additionally:
- While the method is impressive with the reaction-diffusion case that's demonstrated, can you possibly extend this to other cases? I'd like to see this being extended to other problems of interest. If the model were trained on a dictionary containing operators for the Euler equations (inviscid flow) and a separate set of operators for viscous diffusion, could the test-time search successfully discover this composition to approximate solutions to the full Navier-Stokes equations?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an approach for Physics based on test time splitting. The primary idea is to create a library of neural operators using NODE and then combine them during test time.

### Strengths
The problem that the paper is trying to solve is relevant and timely, and will have a significant impact. The paper is general is well written.

### Weaknesses
Despite the fact that the problem statement is extremely relevant, there are several problems as highlighted below:
(a) The literature review is incomplete. There are works that has previously attempted to solve this problem. For example, ICON (https://www.pnas.org/doi/10.1073/pnas.2310142120), NCWNO (https://www.sciencedirect.com/science/article/pii/S0010465525003844). In fact the idea of combining previously learned solution is something NCWNO has explored previously (although the strategy of combining is slightly different). There also exists Poseidon, which is also in the same space.
(b) As the literature review is incomplete, so is the benchmarking in results section.
(c) The example selected are too simple. Solving such simple problem is not convincing.
(d) The fact that the final operator is a lienar combination of two operators (from dictionary) is somewhat limiting in my opinion.

### Questions
a) Why the final operator was considered to be a linear combination of learned operator?
b) In scientific computing, the objective is to predict given the boundary condition and initial condition. While I acknowledge that many previous work has considered time step data as input, this is not that useful as those data will not be available unless a numerical simulator is used. Will the proposed approach work in case we only give initial and boundary condition as input during testing. It seems to me it wont as forming the loss for selecting (through Beam search) the operators from the dictionary will not be possible in such cases.

### Soundness
2

### Presentation
3

### Contribution
2
