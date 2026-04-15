# Unbalanced Diffusion Schrödinger Bridge

- Decision: Reject
- Scores: 5, 5, 5, 3

## Abstract
_Schrödinger bridges_ (SBs) provide an elegant framework for modeling the temporal evolution of populations in physical, chemical, or biological systems. Such natural processes are commonly subject to changes in population size over time due to the emergence of new species or birth and death events. However, existing neural parameterizations of SBs such as _diffusion Schrödinger bridges_ ( DSBs) are restricted to settings in which the endpoints of the stochastic process are both _probability measures_ and assume _conservation of mass_ constraints. To address this limitation, we introduce _unbalanced_ DSBs which model the temporal evolution of marginals with arbitrary finite mass. This is achieved by deriving the time reversal of _stochastic differential equations_ (SDEs) with killing and birth terms. We present two novel algorithmic schemes that comprise a scalable objective function for training unbalanced DSBs and provide a theoretical analysis alongside challenging applications on predicting heterogeneous molecular single-cell responses to various cancer drugs and simulating the emergence and spread of new viral variants.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the extension of the Schrödinger Bridge problem to the unbalanced setting where two measures of different mass can be considered. To solve the Unbalanced Schrödinger Bridge problem, the authors want to extend the Iterative Proportional Fitting (IPF) procedure and discretize it. To do so, they derive a time-reversal formula with killing and birth terms which are used to adapt the IPF algorithm to the defined problem. Then they explain how to sample from their algorithm and empirically study it on synthetic and single-cell dynamic datasets.

### Strengths
i)The introduction and the motivations are clear.

ii) The authors provide an extensive theoretical framework of their proposed method. They first derived an optimization problem and then studied the optimality conditions and then the algorithm to solve it. Finally, they explain how to sample from it.

iii) The authors show how their method performs on a synthetic and a real-world dataset.

### Weaknesses
i) The paper is mathematically heavy and I find its clarity poor. A lot of important mathematical quantities are assumed to be known by the reader as well as some of their properties. For instance:
1. The generator (Eq. 3) is not derived naturally and it is hard to understand what its purpose is at first.
2. The decomposition of the generator K is briefly mentioned in Section 4 and the properties of the decomposed parts are assumed to be understood
3. The DSB method [1] is described rapidly in the related work and it is hard to make the connections between the balanced and unbalanced cases on some concepts 
4. Generally because of the lack of presentation of the generator and the brief introduction on DSB, I find it hard to follow and understand the extension of the IPF algorithm (ie the full Section 4).

ii) I find the experiments not extensive enough. The authors only compared their method with one competitor. They could have also considered the following Schrödinger Bridge problem approaches (that deal with the balanced case) [1,2,3] and if they cannot be applied to this dataset, they could have considered other single-cell datasets [4].

[1] Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling, De Bortoli et al.
[2] Diffusion Schrödinger Bridge Matching, Shi et al.
[3] Simulation-free Schrödinger bridges via score and flow matching, Tong et al
[4] TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics: Tong et al.

### Questions
What happens when the measures have the same mass? Do you recover the original IPF and Schrödinger Bridge?

Can your method be applied to other single-cell dynamics datasets such as [4]? If so, how does your method compare to other methods?

I think the submission's clarity could be improved a lot by giving more details on the Diffusion Schrödinger Bridge in the related work and linking the different concepts to their unbalanced extension in Sections 4 and 5. Currently, the knowledge and understanding of the DSB method are assumed to the reader, making it hard for non-experts to understand the paper and its theory.

Due to the lack of clarity and the lack of competitors and datasets, I feel that this paper should go under a major revision to be accepted at ICLR. Therefore, my current vote is to reject the paper. While I find the theory to be strong and interesting, it might be possible that the paper's scope is too theoretical for a machine-learning venue like ICLR.

###################
Edit post rebuttal:
###################
Thank you for the rebuttal. I agree that this is a mathematical paper. In my opinion, it deserves a major rewriting to make it accessible to a larger audience and to non-experts in order for them to understand how to apply the proposed method to real-world problems. 

Thank you for the novel experiments. I still think that it would be valuable to the paper to consider other single-cell datasets like [4]. 

Due to the mentioned weaknesses above, I keep my current score.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.

### Summary
The main contribution of this paper is to extend the Diffusion Schrodinger Bridge (DSB) to the unbalanced case, by introducing time reversal of birth and death of measure in the process. Based on the time-reversal, the authors introduced the iterative proportional fitting scheme, the sampling scheme and finally the entire algorithm

### Strengths
This paper is written clearly and mathematically sound. The conditions and statements of every theorem are written rigorously and the logic between different sections are also flawless. It is good for readers to follow the math.

### Weaknesses
Disclaimer: I am not familiar with Schrodinger bridge, let alone Diffusion Schrodinger bridge, so my comments might be biased. Please ignore my questions If the authors or the chair found my questions too naive.

I'm curious about what are the uniqueness of SB/DSB/UDSB methods? Is there any real applications where SB/DSB/UDSB works well and other methods does not work? The authors conducted one synthetic experiment and one cellular dynamic experiments comparing with other works. It is good for those who familiar with previous SB/DSB works, but I believe most of the ICLR audiences are not familiar with that. So it will be good for the authors to compare with some non-SB/DSB methods which are more well-known, in order to prove the advantage of SB/DSB/UDSB methods.

### Questions
Same as "weaknesses" section

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors introduce unbalanced diffusion schrodinger bridges a generalization of DSB to the case where mass can vary with time. They introduce a theory of DSB with a forward killing and reverse birth SDEs for an unbalanced diffusion schrodinger bridge. This extends Liu et al. 2022 to the unbalanced setting. A heuristic estimation of $\Psi$ is introduced which avoids estimation of $\log \varphi_t$. These two methods are then applied on a toy example and a single-cell trajectory inference task.

### Strengths
- Extends DSB theory to the unbalanced SB problem, and presents a theoretically sound numerical method to approximate solutions to this problem. I found the formulation quite appealing overall. To the best of my knowledge this is an original significant theory that will be useful at least in the subfield of learning cell dynamics.
- Attacks a difficult and important problem in cell modelling.
- The theory is to the best of my knowledge correct, and the exposition of the method is fairly clear although the notation is quite heavy.

### Weaknesses
- Experimental evidence: Despite presenting a very interesting theory. Results are presented on one small single-cell dataset. It is difficult to tell how and when to apply UDSB given it is only on a single example with what seems like strong prior knowledge on birth and killing rates. Furthermore, there is only a single baseline for the cell dynamics interpolation task when many specialized methods exist for this task. For me this is the single weakest part of the paper. I would like to see additional comparisons to cell dynamics methods, particularly those based on branching SDEs.
- Placement with regard to existing work: I believe the authors missed work on Trajectory Inference [1,2,3]. Comparison to these methods which also account for cell growth and death in an SDE formulation would greatly strengthen this work and provide context on other ways this problem has been addressed.
- UDSB requires known killing and / or birth rates which may limit in practical applications. It seems like these may be difficult to tune in practice, but the freedom to set them is nonetheless quite interesting.
- Some of the experimental setting is quite unclear at least to me (see questions below).

[1] Lavenant, H., Zhang, S., Kim, Y., & Schiebinger, G. (2021). Towards a mathematical theory of trajectory inference.

[2] Lénaïc Chizat, Stephen Zhang, Matthieu Heitz, and Geoffrey Schiebinger. Trajectory Inference via Mean-field Langevin in Path Space. NeurIPS 2022.

[3] Elias Ventre, Aden Forrow, Nitya Gadhiwala, Parijat Chakraborty, Omer Angel, and Geoffrey Schiebinger. Trajectory Inference for a branching SDE model of cell differentiation. 2023.

### Questions
- I don’t understand why the reconstruction quality at the end timepoint should be any better. This is an IID generative modeling task. Is DSB underfit here?
- In addition, I don’t understand how the data is partitioned into training and test for this experiment. If the middle timepoint is left out, how is it known that the observed mass increases 35% but then decreases to -25% from the original total mass? The appendix just states a 80/20 test split.
- The requirement that $k$ is non-negative is not a standard requirement in unbalanced dynamic OT. This seems to be necessary in the theory, but is not a requirement in practice. I did not understand why. Could the authors explain this / clarify in the text?
- More of a comment than a question, “simulating virus spread” is mentioned once in the intro, in the conclusion with an extended experiment in the supplement. I think either this should appear in the main text or not be included. I don’t really understand why this is not in the experiments.

----
Post rebuttal Edit:

I thank the authors for taking time to respond with additional clarifications. My concerns remain re: general applicability and experimental validation, therefore my score remain the same. 

I think additional validation would improve this work. I suggest the Root dataset from here: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009466

as it has outside observations of approximate cell count via imaging.

I think it would be great to show tangible benefits on datasets without this type of prior knowledge too.

### Soundness
3 good

### Presentation
2 fair

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper considers a generalisation of the Schrödinger Bridge problem: initial and final marginals of the process are not probability measures. The authors derive that the time-reversal of diffusions with killing terms and show they correspond to diffusions with birth, and vice versa. Based on this result the authors updated the IPF algorithm and demonstrated its performance using biomedical data.

### Strengths
- The considered problem statement is important as it naturally generalises the Di usion Schrodinger Bridge problem

- The authors derived some fundamental result about the time-reversal of diffusions with killing terms

### Weaknesses
- The experimental results are not very convincing. Even taking std into account still differences between results (for Ours and Ours - no death/births) are not that big. Is this difference really important for a considered downstream task? Ok, MMD is slightly smaller, and so what? Can we really better predict, e.g., responses to cancer drugs in practice?

- Moreover, the dimensionality of considered data is rather limited. Although, the authors consider an important applied problem, it is not clear whether the proposed method is efficient for more high-dimensional tasks

- The authors do not discuss computational complexity and robustness of the method. How does capacity of NNs influence the results?

- The theoretical derivations look OK, but the practical implementation is very difficult to understand, follow and reproduce.

- I have some expertise in diffusion processes, but the text is difficult to follow for non-experts in diffusion processes. Some important propositions, used by the authors while proving their main results, can be provided for completeness in the appendix

- The description of the algorithm and of its each step are difficult to follow in the appendix

### Questions
- In Sec. 5, page 6, the authors introduce a TD loss. Why is it needed? Can the algorithm work without it? Any ablation study on how this loss influences the final quality of the results?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
