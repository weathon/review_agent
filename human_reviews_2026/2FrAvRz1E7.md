# InputDSA: Demixing, then comparing recurrent and externally driven dynamics

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
In control problems and basic scientific modeling, it is important to compare observations with dynamical simulations. 
For example, comparing two neural systems can shed light on the nature of emergent computations in the brain and deep neural networks. Recently, Ostrow et al. (2023) introduced Dynamical Similarity Analysis (DSA), a method to measure the similarity of two systems based on their recurrent dynamics rather than geometry or topology. However, DSA does not consider how inputs affect the dynamics, meaning that two similar systems, if driven differently, may be classified as different. Because real-world dynamical systems are rarely autonomous, it is important to account for the effects of input drive. To this end, we introduce a novel metric for comparing both intrinsic (recurrent) and input-driven dynamics, called InputDSA (iDSA). InputDSA extends the DSA framework by estimating and comparing both input and intrinsic dynamic operators using a variant of Dynamic Mode Decomposition with control (DMDc) based on subspace identification. We demonstrate that InputDSA can successfully compare partially observed, input-driven systems from noisy data. We show that when the true inputs are unknown, surrogate inputs can be substituted without a major deterioration in similarity estimates. We apply InputDSA on Recurrent Neural Networks (RNNs) trained with Deep Reinforcement Learning, identifying that high-performing networks are dynamically similar to one another, while low-performing networks are more diverse. Lastly, we apply InputDSA to neural data recorded from rats performing a cognitive task, demonstrating that it identifies a transition from input-driven evidence accumulation to intrinsically-driven decision-making. Our work demonstrates that InputDSA is a robust and efficient method for comparing intrinsic dynamics and the effect of external input on dynamical systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes InputDSA, an extension of DSA which considers inputs when defining a distance measure between data produced by two dynamical systems. To accomplish this, subspace DMDc is proposed which leverages results from prior works (N4SID) for subspace identification. A variety of synthetic and real data experiments are considered to show the utility of InputDSA.

### Strengths
- The text is overall easy to follow and well-written.
- Does a good job at pointing out a technical gap, that DSA and its extensions have not explicitly considered differences in the effect of external inputs.
- The experimentation is very diverse and does a good job at demonstrating practical use cases for computational neuroscience.
- Good description of hyperparameters in the appendix.

### Weaknesses
- The baselines seem insufficient in experiments 3.2, 4.1, and 4.2 since they primarily compare against only DSA which does not learn an input operator. As a result, it is unclear whether the practical performance improvements for each scenario is due to the proposed SubspaceDMDc or due to explicitly considering input in the metric formulation or both. 
- Synthetic experiments do not consider process noise (noise over time added to the latent state) which is an important characteristic of neural signals due to their intrinsic stochasticity. Could the authors provide experiments that clarify whether InputDSA is also robust to increasing levels of process noise or not? A simple example would be experiment 3.1 where we also consider $x_{t+1} = A(x_t + gF {\rm tanh} (x_t)) + B(u_t + {\rm tanh}(u_t)) + \nu_t$ where $\nu_t \sim \mathcal{N}(0,\sigma^2)$. 


Minor
- Naming the approach “InputDSA” (used in Fig 2 caption) and the distance between input matrices $B$ as “input DSA” (second term in eq. 6 used in Fig 2d,e, and f title) may cause some confusion. 
- $\alpha$ is used twice which may cause some confusion. In line 437 as the exponent of the power law, and in equation 6 as a balancing coefficient.

### Questions
- In Figure 2d, the scale between input DSA for DMDc and SubspaceDMDc is very different. How do we determine what magnitude of effect is required to conclude that there is a difference between two systems?
- How does the distance metric change for each experiment for different values of $\alpha$? Is this an important hyperparameter to tune or can we always use $\alpha=0.5$? When would we set $\alpha \neq 0.5$?
- How does the window size affect the InputDSA distance metric? Does window size need to be carefully selected to get reliable results?
- How do you select the rank of the latent space? Is the InputDSA still reliable when there is a mismatch between true and estimated rank?
- Is the InputDSA metric differentiable? What is does the computational cost of computing InputDSA scale with larger latent spaces and longer time series?
- In experiment 3.2, is the inputDSA metric also robust to phase changes of the input?
- is it always the case that we want to consider rotated state dynamics as similar?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes InputDSA (iDSA), extending Dynamical Similarity Analysis (DSA) to non-autonomous systems by explicitly modeling both (i) intrinsic/recurrent dynamics and (ii) input-to-state mappings. Estimation uses a new Subspace DMDc (a subspace-identification variant of DMD with control) to mitigate the partial-observation bias that makes vanilla DMDc fail. Applications include synthetic systems, RL-trained RNN agents, and neural data show shifts from input-driven to intrinsic dynamics.

### Strengths
1. This paper solves a clear problem gap - ‘Most dynamics-similarity methods ignore inputs.’
2. The proposed method has partial-observation robustness. Subspace DMDc directly addresses the bias where inputs instantaneously affect observed states and, via unobserved states, which is an excellent point for neural data.

### Weaknesses
1. Since InputDSA mixes intrinsic and input dynamics (via alpha in equation 6), it would be useful to show how sensitive results are to alpha. In other words, how to choose alpha when applying this method? 
2. The paper should clarify how the delay parameter d is determined or selected in practice.
3. The results lack statistical uncertainty estimates on results, such as statistical uncertainty test on silhouette scores.

### Questions
Please see weakness.

### Soundness
4

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
4

### Summary
The paper proposed a new method, inputDSA, to compare between observed trajectories of two dynamical systems. It is an extension to DSA that addresses the important point of external input. The intuition behind both methods is that lifting the observations to a highD space can make the dynamics linear (which is correct for infinite dimensions via Koopman theory). Comparison in the linear domain is relatively easy. Because neuroscience applications typically have external input, and this can completely bias the comparison process, the current work shows how to incorporate inputs into the framework. This has the added benefit of making the algorithm more efficient. The authors demonstrate the method on several synthetic examples, and on experimental neural data.

### Strengths
This is an important problem. With the availability of highD neural recordings, inference of dynamical systems from data is increasingly done using many methods. DSA is more of a comparison then direct inference, but it contributes to this field. External input is a crucial aspect of many of these methods.

The paper is clearly written, explains the problem, the approach and the results. 

The focus on partial observation is highly relevant to possible applications, and done systematically.

### Weaknesses
The interaction between A and B are almost not discussed. Appendix J touches upon it, but the results in the main text mostly show input similarity and state similarity. For instance, in the neural data example, Figure 5F shows angles within A or B. But an interesting question is whether the input arrives to directions that “matter” to the intrinsic dynamics. Can that be measured using eigenvectors of A and their relation to B? 

The limited effect of wrong inputs on state estimation. Results such as Figure 3D show that getting the input completely wrong has little effect on the quality of intrinsic (A) estimation. This seems somewhat contradictory to the original motivation of the paper.

### Questions
The scope of applicability is not clearly defined. When should this fail? What are the conditions (reachability, full rank) – and how can you verify this. For instance, how do you know it works for your examples?

Eq 1. This is not a symmetric measure. For instance, the dimensions can be different. Might be useful to discuss what happens if internal/input dimensions differ between systems.

Eq 10 – what is the motivation for this form of dynamics?

What is the meaning of the numerical value of scores? The original DSA paper used embedding in a lowD space, whereas you mostly use silhouette. What is the motivation for this choice?

Fig 2F – Partial observability degrades state estimation (top panel). From Figure 2A, one can think that having access to the true input is a way to rescue this estimation, because we have indirect information of the unobserved units. There is a small increase in the state silhouette score, but I wonder how much can be rescued. For instance, if the input is more than 1-dimensional – does this help? 
No reference to Fig 3AB from main text

Fig 3D – perhaps use bars, violin, or anything else that does not connect the dots, when the X axis is categorical.

Fig 3D – why is the intrinsic estimation unaffected by wrong (random) inputs? Are there inputs (e.g. anti-correlated) that will hurt the intrinsic correlation?

Line 338 typo? “crucial to how inputs”

Plume tracking: “This implies that inputs excite more dimensions of the RNN in Top agents (Fig. 4E), thereby allowing them to more effectively incorporate recent information to inform action selection in the future.” – I’m not sure I follow the logic here. Why should high-D input be more effective in incorporating information? Shouldn’t this be linked to the relation between these directions and those of the intrinsic dynamics? If these higher dimensions are orthogonal to internal dynamics, for instance, then it should just be noise.

Neural data: Aligning to the ntc point. If I understand correctly, this uses the result from a different pipeline (identifying this point), and then does a PSTH using this shift. Perhaps worth emphasizing.

Neural activity: why the need to use PCA and isomap for preprocessing? Does using Isomap change any assumptions about the dynamical system? For instance, it could be that the discarded dimensions are important for the closure of the dynamics.

Line 436: power law fit. Dynamics is usually dominated by the slowest mode. Why use this method of fitting? While several slow modes can interact, the very fast modes are not expected to be relevant here – but they are used in the fit. Also – integration (pre) is expected to have line attractor like dynamics, or very slow intrinsics dynamics. Isn’t the result opposite to this intuition? If so, do you know why?

Line 442: reflected – reflecting

Appendix K tables are in the next page, which is confusing.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces InputDSA, which aims to extend Dynamical Similarity Analysis (DSA) to systems with external inputs. The method fits, for each system, a linear controlled latent state-space model h_{t+1}=Ah_t+Bu_t via a subspace ID-style procedure, and then defines a similarity metric that compares the recurrent term A (“intrinsic dynamics”) and the input term B (“input-driven dynamics”) across systems after alignment. The problem is important, and the core idea, explicitly separating internal dynamics from input drive and comparing both, is novel and interesting.

### Strengths
The synthetic experiments are well designed and give evidence that under good excitation conditions and with partial observability, the proposed estimator can recover similar recurrent structure vs similar input structure. The applications to trained agents and to neural population data are very appealing and show how the method could validate and generate hypotheses about “what part of the dynamics actually matters.”

### Weaknesses
There are some serious concerns with the clarity and interpretability of the current manuscript:

1.	The main text never clearly defines what state is being modeled. In practice, the method learns a latent linear model (A,B) on top of lifted observations via Subspace DMDc. But the paper (especially schematic Fig. 1 and Sec. 2) blurs together true system state, observed activity, and this identified latent state, and then talks about A and B as if they were the system’s real intrinsic vs input-driven dynamics. As written, it’s not clear what is actually being compared from the main text alone.

2.	Many essential details of the identification procedure (how A,B are estimated, model order selection, assumptions needed for demixing) are only in the appendix. From the main body, it is difficult to understand the method and pipeline.

3.	More important: the application to neural data is not well justified, considering the fact that inputs in neural tasks are usually unknown/partial. In real neural recordings, the “inputs” we can measure are usually low-dimensional task variables, not persistently exciting signals. The paper acknowledges this and presents a robustness test with surrogate/noisy inputs in Sec. 3.2, and that test is done in an RNN control setting with richer continuous inputs. However, moving to neural data in Sec. 4.2 and the claimed interpretation “the circuit becomes intrinsically driven after commitment, whether the A/B split is still meaningful becomes unclear since the inputs here are pretty abstract and simple."

Overall, I think the idea of this paper is very interesting, and the paper brings some valuable contributions. But the paper in its current form oversells its interpretability on neural data, and the core method is under-specified in the main text. I would like to see a revision that: (i) clearly defines the latent state and is explicit that InputDSA compares identified models; (ii) makes the system ID procedure understandable without digging too much through the appendix; and (iii) tones down the neural interpretation in light of partial/weak inputs.

### Questions
See Weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2
