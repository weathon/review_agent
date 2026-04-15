# Deep Linear Hawkes Processes

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
Marked temporal point processes (MTPPs) are used to model sequences of different types of events with irregular arrival times, with broad applications ranging from healthcare and social networks to finance. We address shortcomings in existing point process models by drawing connections between modern deep state-space models (SSMs) and linear Hawkes processes (LHPs), culminating in an MTPP we call the _deep linear Hawkes process_ (DLHP). The DLHP modifies the linear differential equations in deep SSMs to be stochastic jump differential equations, akin to LHPs. After discretizing, the resulting recurrence can be implemented efficiently using a parallel scan. This brings both linear scaling and parallelism to MTPP models. This contrasts with attention-based MTPPs, which scale quadratically, and RNN-based MTPPs, which do not parallelize across the sequence length. We show empirically that DLHPs match or outperform existing models across a broad range of metrics on eight real-world datasets. Our proposed DLHP model is the first instance of the unique architectural capabilities of SSMs being leveraged to construct a new class of MTPP models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel marked temporal point process (MTPP) model called DLHP by drawing connections between linear Hawkes processes (LHPs) and deep state-space models (SSMs). Based on the SDE form of LHPs, DLHP models the intensity function by utilizing parameterization and parallelization techniques within SSMs. This approach achieves linear scalability and enables parallel processing, addressing the shortcomings of existing MTPP models. Experimental results demonstrate the effectiveness of DLHP.

### Strengths
1. Modeling the intensity function with the proposed latent linear Hawkes (LLH) layer is a novel and promising idea. The proposed model has the advantages of linear scalability and parallelism, addressing the shortcomings of existing marked TPPs.
2. The background and the proposed method are presented clearly. The paper is well-written and easy to follow.
3. Experiments are conducted to evaluate the model's predictive performance in terms of log-likelihood and computational efficiency.

### Weaknesses
1. While log-likelihood is an important metric for evaluating TPP model performance, this paper does not include experiments on the RMSE of time prediction—a metric commonly used in most baseline studies, such as RMTPP, SAHP, and THP.
2. Performing intensity recovery only on an inhomogeneous Poisson process is insufficient; it is also necessary to include recovery for other traditional TPPs, such as the Hawkes process.

Minor: The per-event log-likelihood in line 359 needs to be divided by the number of events N.

### Questions
1. If the background intensity is directly extended to be a function of time t, can the existence of a positive solution for Eq.(9) still be ensured?
2. Is it possible to use other numerical methods, such as the Euler method, to discretize the proposed LLH layer?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This research paper introduces the Deep Linear Hawkes Process (DLHP), a novel approach to modeling Marked Temporal Point Processes (MTPPs) that bridges the gap between deep state-space models and MTPPs. The benefits of state space models (SSMs) compared to attention-based model were emphasized. The innovation lies in modifying stochastic jump differential equations that describes the intensity function $\lambda$ of MTPP to become linear differential equations in deep state-space models. Its applicability was tested on various benchmark dataset. Notably, this work marks the first successful adaptation of state-space model architectural features to create a new category of MTPP models.

### Strengths
The paper introduces a new family of MTPP models that leverages SSM layers. Also, the stochastic jump differential equation was translated in terms of SSM. By stacking multiple layers more flexible intensity function can be generated, while input dependent characteristic enabled more complex behavior.

### Weaknesses
Only NLL was used as an evaluation metric, whereas RMSE and accuracy is commonly used in prior works. Also, as a method for modeling a stochastic process, simulation study can help showing the effectiveness of the method.

### Questions
- In Fig. 4, THP and IFTPP shows faster runtime when implemented in JAX. Is DLHP still faster when they are all implemented using JAX?
- Demonstration of the simulation study might be helpful to see wether the discretization of DLHP does not hurt the expressivity.

### Soundness
3

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
5

### Summary
This paper presents an interesting idea of using deep state-space models (SSMs) to model marked temporal point processes. The authors draw a connection between the SSMs and linear Hawkes processes via their stochastic differential equations, and develop a tailored model that enjoys comparable or superior performance against neural marked TPP baselines. The overall presentation is clear and easy to follow.

### Strengths
The idea of combining the deep SSMs with marked temporal point processes is interesting. The methodology/model architecture is well presented, easy for the readers to understand how these models are integrated together, via techniques like diagoalization, discretization, input-dependent parameters, etc. Meanwhile, the new model enjoys a better computational efficiency than other baselines when implemented with parallel scan.

### Weaknesses
Though I enjoyed reading the paper, there are still a few aspects of the paper that I think can be improved or answered by the authors:

*Methodology*:

1. I am not very convinced about the use of the Hawkes-process-like mechanism in each layer. I understand the connection between a linear Hawkes process and the SSM via the stochastic differential equation, but what is the need/meaning/interpretation of using this connection in the model? Why use $AB$ instead of one matrix of size $P\times H$ (or use $E\alpha$ instead of one $P*K$ matrix) as the parameters in Equation 10? If the idea is to inherit or imitate the parameter structure in the linear Hawkes process, this structure of parameters is overwritten/ignored since there are multiple layers and non-linear transformations stacked together.

2. On the other hand, if the authors do want to imitate the parameter structure of the linear Hawkes process, then the diagonalization would assume no influence between different marks, which is unrealistic in practice. And there is also no interpretation of the learned $\Lambda$ and $\alpha$ in the experiments.

*Experiments*:

3. I would suggest that the authors conduct more synthetic data experiments and results to validate the model’s ability to recover the ground truth. Only one simple Poisson process example is not enough, and the real data experiments are not straightforward in terms of interpreting the model's effectiveness.

4. No standard deviation was reported for the metrics in the tables/figures. The results of DLHP look like a random win against baselines given no reported standard deviation, especially only on real-world datasets with no ground truth.

5. It would be better if there was a rigorous computational complexity analysis of the model and baselines since the authors claim that it is one of the main strengths of using SSMs in modeling marked TPPs.

### Questions
1. The main text says $u_t^{(1)} = 0$ (line 231), but in figure 2 it says $u_t^{(1)}$ is the event embedding. I am confused about this.

2. Do the authors have any insights on how to choose the model architecture, given the performance discrepancy of the model using forward and backward mechanisms?

3. What do the authors mean by 1.4 times improvement from baselines in line 92? If it refers to the average rank of the DLHP, does it mean a lower number is better?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new marked temporal point process (MTPP) model known as the deep linear Hawkes process (DLHP), which is developed by integrating deep state-space models (SSMs) and linear Hawkes processes (LHPs). Through the use of diagonalization and discretization techniques, the DLHP enables efficient implementation of closed-form update. The model has been shown to surpass current methods in terms of performance on a variety of benchmark tasks, excelling in metrics such as log-likelihood and runtime across different sequence lengths. A key contribution of the research is pioneering the use of SSMs' unique architecture to develop a new class of MTPP models, positioning the DLHP as a competitive model in applications.

### Strengths
1. This work stands out for being the first effort in harnessing the distinctive architectural features of deep state-space models (SSMs) to develop marked temporal point process (MTPP) models. 
2. Efficient Implementation: the DLHP has demonstrated robustness, performance, computational efficiency, and extensibility in experiments. These qualities seem to make the DLHP a competitive contender as an out-of-the-box model for a variety of applications. 
3. Clear presentation: This paper presents its idea clearly.

### Weaknesses
1. As the authors clarified, DLHP has a higher implementation complexity, and the model architectures for DLHP tends to be more complex compared to other models. Does the enhancement in performance contributes to this increased complexity (since increased model complexity may lead to higher model likelihood value)? It would be better to provide necessary analysis regarding this issue and add some experiments for illustration if possible.
2. The proposed method sacrifices some of the interpretability of latent dynamics and parameters that are characteristic of LHPs, which could impact the model's transparency and understanding, and limit its usability for high-stakes applications where understanding event interactions is crucial. Would you discuss possible solutions or ideas regarding this issue? The motivation stated in line 201-214 of page 4 may provide some clues and could be discussed more deeply.

### Questions
The authors may also pay attention to the paper titled with “Mamba Hawkes Process”（https://arxiv.org/abs/2407.05302）and do a detailed comparison with it, also to further confirm the statement "the first instance of the unique architectural capabilities of SSMs being leveraged to construct a new class of MTPP models" in the paper.

See the weaknesses for other questions.

### Soundness
3

### Presentation
3

### Contribution
3
