# Deep Stochastic Mechanics

- Decision: Reject
- Scores: 6, 8, 8, 6

## Abstract
This paper introduces a novel deep-learning-based approach for numerical simulation of a time-evolving Schrödinger equation inspired by stochastic mechanics and generative diffusion models. Unlike existing approaches, which exhibit computational complexity that scales exponentially in the problem dimension, our method allows us to adapt to the latent low-dimensional structure of the wave function by sampling from the Markovian diffusion. Depending on the latent dimension, our method may have far lower computational complexity in higher dimensions. Moreover, we propose novel equations for stochastic quantum mechanics, resulting in linear computational complexity with respect to the number of dimensions. Numerical simulations verify our theoretical findings and show a significant advantage of our method compared to other deep-learning-based approaches used for quantum mechanics.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work introduces an idea based on stochastic mechanics to provide approximate solutions to the time-dependent Schroedinger's equation, and neural parameterizations of various quantities involved.

### Strengths
The idea is very innovative and I especially liked the use of Nelsonian dynamics.

### Weaknesses
The approach is limited to very simple/small systems (harmonic oscillators and interacting distinguishable particles of only a handful of particles).  

The discussion is also quite limited, and bypasses a vast amount of literature in the field, including time-dependent variational wave functions based on neural network parameterizations. 

For example: 

"Another family of approaches, FermiNet (Pfau et al., 2020) or PauliNet (Hermann et al., 2020),
reformulates the problem (1) as maximization of an energy functional that depends on the solution of
the stationary Schrodinger equation. This approach sidesteps the curse of dimensionality but cannot ¨
be adapted to the time-dependent wave function setting considered in this paper." 

The approach that reformulates (1) as a static, variational problem does not date to 2020 and it is as old as quantum mechanics itself. 
Also, neural variational parameterizations of the wave function are routinely used to solve the time-dependent Schroedinger's equation. These approaches are based on Dirac and Frenkel's time-dependent variational principle, and have been used in several works, starting e.g. from https://www.science.org/doi/10.1126/science.aag2302, https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503 and many more.

### Questions
1) The approach is shown for very small systems, and in cases that are solved easily with many other approaches (including, for example, using a discrete basis and using the time-dependent variational principle with a neural network wave function). Do the authors have a sense of the scaling in terms of number of particles/ can they show a case that goes beyond what can be simulated exactly? 

2) A grid-based discretization seems to be used in the paper, at least in Figure 1 d. Can the authors clarify how does the discretization enter their algorithm?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors propose a stochastic formulation of the Schrödinger equation. The proposed formulation can be used to simulate quantum mechanics with high efficiency. 
A key difference between this new formulation and the one proposed by Nelson is that the new method employs the gradient of the divergence operator to facilitate the neural computations.
They also prove theoretically that the loss function used to train neural networks is upper bounded by the L2 distance between the true process that samples from the quantum density and an approximate process which the neural network tries to predict.
Experimental results show that the proposed method is superior to the baseline method PINNs.

### Strengths
1. The new stochastic formulation of the Schrödinger equation provides an efficient way for quantum mechanics simulation by utilizing the power of neural computation.
2. Training loss of the neural networks for learning stochastic process is bounded with theoretical guarantees.
3. The O(Nd) computational complexity looks very promissing and opens the door for large-scale quantum simulation.

### Weaknesses
1. Seems the neural network employed in this study is a single layer neural network. I am wondering how a single layer neural network could learn dynamics of a complicated wave function. Also the illustrations in the experiment look pretty simple. The authors are encouraged to tackle more complicated cases using the proposed method.
2. The O(Nd) computational complexity need more elaboration. Is it the computational complexity of training the neural networks on a single trajectory? For learning a stochastic process, we may need to sample many trajectories  in order to learn hidden low dimensional representations of process. I am not sure whether it is fair comparisons with other methods as listed in the table 1.

### Questions
1. For training losses defined in eq. 11 to eq. 15, because they need integration operation, I am wondering in practice how the integration is done and what is the window length of the integration during the training process at each iteration/epoch.
2. In page 5, the authors mentioned for each iteration, they will sample new trajectories using eq. 7. How do we handle the cold start problem at the very beginning of the training process. I mean at the beginning of the training process, neural networks have not learned dynamics of the stochastic process. So the trajectories we sampled may be invalid or they will mis-guide the learning of the target stochastic process.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a neural-network-based solver for simulating the time evolution of quantum systems, called Deep Stochastic Mechanics (DSM). DSM is based on Nelson’s formulation of stochastic quantum mechanics where the quantum system evolution is characterized by a stochastic process. The authors captured the correspondence between Nelson’s formulation and a diffusion process, where a partial differentiation equation lies at the heart. Then they propose to approximate the solution to the PDE with neural networks. A training process employs errors of the PDE in the time-evolution and initial points as the loss functions to approximate the solution. Empirical studies of this method are also conducted on several typical quantum systems.

### Strengths
The paper’s approach to simulating quantum systems is very intriguing. It provides insights into the connection between quantum mechanics and diffusion processes and exploits it with neural networks which recently have shown advantages in dealing with diffusion processes. It ends in a stochastic NN-based simulator with low training cost and with theoretical guarantees on the quality of the solution. Since the training process is based on trajectories of the stochastic process, it is naturally adaptive to the structure of the solution and boosts the precision because of the focus on the solution space. 

The experiments demonstrated the performance of DSM to surpass prior methods, tackling the most challenging problem in simulating quantum systems: the curse of dimensionality. It is suggested that NNs have the capability of exploiting the low latent dimension of the simulated quantum system, making use of their advantages in extracting low-dimensional features in high-dimensional data. This observation seems the most interesting part of this approach, which, to the best of my knowledge, is not exploited in other approaches. It is quite possible that a large number of quantum systems effectively have low latent dimensions, and DSM may be well-suited to provide sampling from these systems. 

Although the materials may not be easily accessible to the general ML audience, the well-organized presentation is likely to deliver the core ideas of this paper to a broader community. The inspiring innovations of this paper make it worthy to be published in ICLR.

### Weaknesses
Although the experiments cover several interesting cases where DSM works well, the limitation of DSM is not fully discussed in the paper. I would like to understand what cases will fail DSM. For example, does it perform worse on systems with complicated interactions or large latent space? I believe such examples may better illustrate the limitations and the suitable scenarios of DSM.

### Questions
Additionally, I wonder how the performance of DSM through stochastic quantum mechanics compares to other (NN-based) PDE solvers via Shrodinger’s formulation. Is it possible to exploit the latent space within Shrodinger’s picture, either with NNs or not?

A minor point: in Section 5.1, it is claimed that "Table 2 shows ... and the training time for ...", while the training time is not found in Table 2. Where is it?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a new approach to simulating time evolution in quantum systems by learning the gradient of the modulus and phase of the wavefunction over time. This is done by using a loss function that ensures the networks parametrizing these gradients satisfy the Schroedinger equation. The main novelty is to evaluate this loss on a batch coming from the samples of a stochastic process that describes the evolution of the modulus squared of the wavefunction. Numerical experiments for low dimensions show a better behavior than PINN.
My score is between 6 and 8, leaning towards 6. If more evidence for favourable scaling wrt traditional methods is provided, I can increase my score.

### Strengths
- Novel and interesting approach to simulating time evolution in quantum mechanics using deep learning
- Favourable O(d) scaling compared to PINNs which requires O(d^2) to compute the Jacobian
- Evaluation of the loss on trajectories coming from the stochastic process governing the modulus squared of the wavefunction, compared to PINNs which require to specify collocation points.
- Experiments show better performance than PINNs

### Weaknesses
- The framework does not seem to allow estimation of non-diagonal observables, eg momentum
- While I understand the complexity argument for evaluating the loss in O(d), it is unclear to me whether this method can scale better than traditional approaches. In particular, experiment of figure 4 studies the harmonic oscillator for d=1..9. However, if I understand the setting, the Hamiltonian is separable in the dimensions and so the problem has a natural scaling linear in d. I am therefore unsure about the promise for a more favourable scaling than traditional methods.
- The use of stochastic formulation of quantum mechanics for quantum mechanics simulation is not necessarily novel, however I have not seen references to it. I am not an expert in this field, but one example I found is [Quantum Dynamics with Trajectories, Robert E. Wyatt].

Minor:
- below eq 5, one of the two u's should be v.

### Questions
- Can you benchmark the method against traditional solvers for an interacting problem in higher dimension and show a more favourable scaling?
- How would one estimate the expectation of a non-diagonal operator?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
