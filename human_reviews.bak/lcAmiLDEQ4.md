# MindFlayer: Efficient Asynchronous Parallel SGD in the Presence of Heterogeneous and Random Worker Compute Times

- Decision: Reject
- Scores: 5, 5, 3, 3

## Abstract
We study the problem of minimizing the expectation of smooth nonconvex functions with the help of several parallel workers whose role is to compute stochastic gradients. In particular, we focus on the challenging situation where the workers' compute times are arbitrarily heterogeneous and random. In the simpler regime characterized by arbitrarily heterogeneous but deterministic compute times, Tyurin and Richt'{a}rik (NeurIPS 2023) recently designed the first theoretically optimal asynchronous SGD method, called Rennala SGD, in terms of a novel complexity notion called time complexity. The starting point of our work is the observation that Rennala SGD can have arbitrarily bad performance in the presence of random compute times -- a setting it was not designed to handle. To advance our understanding of stochastic optimization in this challenging regime, we propose a new asynchronous SGD method, for which we coin the name MindFlayer SGD. Our theory and empirical results demonstrate the superiority of MindFlayer SGD over existing baselines, including Rennala SGD, in cases when the noise is heavy tailed.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a parallel SGD computation scheme when each local work has random compute times. The authors show that the proposed method is a generalization of Rennala SGD and is thus optimal in deterministic worker time regime. When randomness exists, the authors empirically verify that the proposed method is better than Rennala SGD and ASGD for different settings. Theory contribution involves single device justification of superiority of MindFlayer, derived convergence result and time complexity of MindFlayer.

### Strengths
This paper first time studies parallel SGD setting where workers have random reply times. This setting is practically important. Presentation is clear and easy to follow, with motivating single device example. Theorems are core (convergence and time complexity), accompanied with digestions. Experiments are illustrative and effectively demonstrate the superiority of the proposed method over baselines.

### Weaknesses
I personally recognize the main value of current work to be the setting being considered, where worker has random reply time. The authors claim they are first to consider this type of setting, and I do think it is of practical importance. The authors may consider addressing the following limitations:

1. The proposed method seems theoretically intuitive but may have practical limitations. Algorithm 1 and Algorithm 2 look more like an experiment setup instead of a true practical algorithm. For example, distribution of $\eta_i$ and thus also $p_i$ are hard to evaluate in practice. Compared to ASGD and Rennala SGD, Algorithm 1 involves more parameters such as $B_i,p_i$ that users need to decide on. Experiments are carried out where $\eta_i$  is of certain ideal theoretic distribution, which is rarely encountered in practice. Therefore, empirically estimation of good choices of $B_i,p_i$ and even $\mathcal{J}_i$ remain a problem.

2. The setting being considered is interesting but may rise other practical issues. For example, the proposed method chooses to initialize another query whenever the current gradient computation takes too lone (quantified by $>t$), I feel this would significantly increase number of communication rounds between server and workers compared to baseline ASGD and Rennala SGD, which may form bottleneck in certain settings.

3. Figure 3 is hard to read, legends should be added to demonstrate what are dashed green line/ dotted purple line/ solid yellow line/ shaded grey region. The title includes no such information as well. 

Minor writing typos:
1. line 33, shouldn't the codomain of $f$ be $\mathbb{R}$ instead of $\mathbb{R}^d$?
2. line 5 in Algorithm 1 and line 308 of text, the authors seem to intend to refer Algorithm 2 while wrongly typed Algorithm 4? If Algorithm 4 is intended, it should be moved to main content.
3. line 346, it says "convergence of Rennala SGD" while I feel it's for  MindFlayer instead of Rennala SGD?
4. in Figure 3 title, "section" missed i
5. line 464, "the time complexity of Rennala SGD" should capitalize first letter

### Questions
see limitations above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposed a new asynchronous SGD method for minimizing smooth and nonconvex functions, called MindFlayer SGD, which is specifically designed to operate under conditions of heterogeneous and random compute times. Through theoretical analysis and empirical evaluation, the authors show that MindFlayer SGD outperforms existing methods, including Rennala SGD, especially in scenarios with heavy-tailed noise. In addition, the paper also introduced Vecna SGD in the heterogeneous regime with applications in the distributed optimization and federated learning

### Strengths
1. **Well-Organized Structure**: The paper is structured clearly, allowing readers to follow the development of ideas seamlessly. It includes a comprehensive literature review that contextualizes the research within the broader field of stochastic optimization, highlighting the significance of addressing the challenges posed by heterogeneous and random compute times. The authors present the MindFlayer SGD algorithm with detailed explanations of its design and motivation. This clarity enhances the reader's understanding of how the algorithm functions and its advantages over existing methods.

2. **Strong Supporting Evidence**: The claims made in the paper are robustly supported by both theoretical proofs and empirical results. The theoretical framework provides a solid foundation for the proposed algorithm, while the experimental findings demonstrate its superior performance in various scenarios, particularly in the presence of heavy-tailed noise.

3. **Sufficient Implementation and Proof Details**: The paper provides adequate details regarding the implementation of MindFlayer SGD and the associated proofs. This transparency allows for reproducibility and facilitates further research, enabling other researchers to build on the authors' findings effectively.

### Weaknesses
**Implicit Assumption of Utilizing Privileged Information $p_i$ in Algorithm Design**. In the real world, the distributions of compute time for different works are usually unknown. The proposed algorithms implicitly leverage this information to get the expected batch size $B$. I am not sure if this is required in the proof. The paper fails to discuss this limitation and provide a workaround without using information $p_i$

### Questions
## Major
1. Implicit Use of Privileged Information  $p_i$: Could the authors clarify this assumption further? Why is it necessary? Is there a possibility of incorporating an online estimate of $p_i$ within the algorithm?
2. Decentralized Setting: The paper primarily addresses a centralized setting. Are there any potential extensions to accommodate decentralized training?


## Minor
1. Method 4 in Algorithm 1 should be referred to as Algorithm 2
2. Algorithms 2 and 4 could be misleading. I don't think we can realistically sample compute times in the real world.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper explores the challenges and solutions of SGD in environments where compute times of parallel workers are irregular and unpredictable. Prior work introduced Rennala SGD for situations with heterogeneous but fixed compute times, optimizing time complexity. However, this method falters when applied to scenarios where worker compute times vary randomly. The authors introduce MindFlayer SGD, a new asynchronous SGD method designed to handle random compute times and demonstrate its effectiveness through both theoretical analysis and empirical data, showing that it outperforms existing methods like Rennala SGD, especially in environments with heavy-tailed noise distributions.

### Strengths
1. MindFlayer SGD provides a more realistic approach to asynchronous SGD in modern, heterogeneous computing environments.
2. The paper provides a solid theoretical framework for the proposed method, including proofs of its efficiency and effectiveness over traditional methods

### Weaknesses
1. As described in Algortihms 1 and 2, a client needs to wait for other clients even after completing all of its trials, and each server’s update (Algorithm 1 Line 12) must aggregates gradients from all clients. Given these characteristics, MindFlayer SGD seems to function more like a synchronous algorithm rather than an asynchronous one.
2. The generalized computation model modestly extends the fixed computation model used in Rennala SGD, which may not present significant technical challenges.
3. The impact of various hyperparameters involved in MindFlayer SGD is not deeply discussed Particularly, the parameters $t_i$ and $B_i$ are difficult to choose in practice, which plays a central role in MindFlayer SGD and could be crucial for practitioners aiming to adapt the method to specific applications.
4. The experiments compare MindFlayer SGD solely with Rennala SGD and vanilla ASGD, and the test problems may be overly simple. Including comparisons with a wider array of contemporary asynchronous methods on more complex machine learning problems could enhance the argument for MindFlayer SGD's superiority.

### Questions
1. In line 470, th authors use L-BFGS-B to obtain optimal $t$. How to obtain the unknown parameters, e.g., $p_j$, in the optimization problem? Is the optimality theoretically guaranteed? What is the computational cost incurred?
2. In Algorithm 1 Line 5, Line 307, and Line 323, shoud the “Method 4” or “Algorithm 4” be changed to ”Algorithm 2“?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes MindFlayer, a variant of asynchronous SGD that specifically deal with the presence of random compute times. Theoretical analysis and empirical results are provided accordingly.

### Strengths
1. This paper proposes MindFlayer, a variant of asynchronous SGD that specifically deal with the presence of random compute times. 

2. Theoretical analysis and empirical results are provided accordingly.

### Weaknesses
Major issues:

1. For the algorithm itself, the contribution of MindFlayer SGD is incremental since it's mostly based on Rennala SGD, as the authors stated in this paper by themselves: "For MindFlayer SGD, each iteration, on average, receives only Bp gradients, making it effectively a scaled-down version of Rennala SGD." (Lin 241-242)

2. Although it is stated that MindFlayer SGD is designed for dealing with the complexities/challenges of real-world distributed learning environments and applications, the experiment settings are far from those for real-world applications (especially in 2024). The only non-convex problem used in the experiment is an extremely simple two-layer neural network on the MNSIT dataset, which is far from convincing if the authors want to show that the proposed algorithm could be used in real-world distributed training and applications. 
Regardless of the real-world settings, most of the experiments are on convex problems, which cannot even be used to verify the theoretical results, since the theoretical analysis is based on smooth but non-convex settings. 

3. I cannot find the hyperparameters (basically just learning rates) used in the experiments, or in what range the hyperparameters are tuned. Actually, it is known whether the baselines are fairly tuned. For example, in Figure 5, the curve of ASGD seems to be a flat line (or it simply gets worse from the very beginning). However, if the learning rate is well tuned, a small enough learning rate should make the grad norm or loss of ASGD going down even a little bit. Thus, the overall experiment settings and the hyperparameter tuning is questionable.

Minor issues:

1. In several algorithm, for example Algorithm 1, does "Method 4" actually mean Algorithm 4 in the appendix? If so, please make the naming consistent across the entire paper.
2. Separating the main algorithm (Algorithm 1 and Algorithm 4) in the main paper and the appendix is not friendly to the readers. Especially there is still space remaining in the main paper.
3. Putting experiment description and the corresponding figures (Figure 4,5) separately in main paper and the appendix is also unfriendly to the readers.

### Questions
1. How are the hyperparameters tuned in the experiments?

2. For non-convex problems, smaller gradient norm doesn't always implies smaller loss value. And eventually we train models for smaller loss, not for smaller gradient norms. Could the loss curve also be added for the experiment of the neural network on MNIST?

### Soundness
3

### Presentation
1

### Contribution
2
