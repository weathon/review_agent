# On the interplay between learning and memory in deep state space models

- Decision: Reject
- Scores: 3, 3, 6

## Abstract
Deep state-space models (SSMs) have emerged as a powerful deep learning architecture for sequence modeling, but the theory of how these models learn long-term dependencies lags the practice. To explain how parameterization and the number of layers affect a model's expressiveness, we study the properties of deep $\textit{linear}$ SSMs, i.e., linearly coupled stacks of linear time-invariant systems. We show that such systems share timescales across layers, and we provide novel analysis on the role of linear feedforward connections in regularizing these temporal dependencies. In practice, SSMs can struggle with an explosion of the hidden state variance when learning long-term dependencies. We expand our theoretical understanding of this problem for deep SSMs and provide new intuitions on how this problem may be resolved by increasing the number of layers. Finally, we confirm our theoretical results in a teacher-student framework and show the effects of model parameterization on learning convergence.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper investigates memory in deep state-space models. However, the analysis is limited to linear state-space models, which presents a significant gap compared to models commonly used in practice. This opens up a potential research direction to explore whether the phenomena observed in this paper are specific to linear models or hold more generally for nonlinear cases.

### Strengths
This paper provides a detailed analysis of deep state-space models, using the theory of linear state-space models to demonstrate the performance of recurrent models. It includes a thorough review of related work, and the topic of identifying the advantages of state-space models in learning long-term memory is indeed important. **However, I would expect the paper to engage more deeply with prior research rather than focusing primarily on synthetic examples, which may not offer much inspiration for theoretical advancements or practical applications**.

### Weaknesses
1. **Lack of Reference to Relevant Literature**: The paper’s calculation of the autocorrelation function of linear state-space models should reference time series literature (e.g., ARMA models). It doesn’t add new insight and lacks generalization to nonlinear models, leaving questions on how the results extend.
2. **Limited Scale of Phenomenon Studied**: The scope of the paper is too narrow, focusing on a small-scale phenomenon, making comparisons with larger, well-established models (like S5 and Mamba) less meaningful.
3. **Insufficient Depth in Experiments (Figure 4)**: The experiments only use 1-3 layers of LRU, which is far from practical applications. More layers (up to 12) should be tested to understand the true behavior of the model as depth increases. Given the layers used in Figure 2, it is believed that extending the layers up to 12 is feasible within the available computational budget.
4. **Unclear Key Message for Different Audiences**: The paper doesn’t clearly separate the key takeaways for theoretical researchers and practitioners, which weakens its impact for both groups.

### Questions
1. It is difficult to discern the key contributions of this paper in relation to the existing literature, such as Recurrent Neural Networks: Vanishing and Exploding Gradients Are Not the End of the Story (https://arxiv.org/abs/2405.21064). There are a lot of similarity in the key message. It is important to clearly demonstrate how your work advances the recent literature, as the key contributions of this paper are currently hard to identify. 
2. Another issue is the significant duplication of content in the appendix for Lemma 1 (from the arxiv paper Recurrent Neural Networks: Vanishing and Exploding Gradients Are Not the End of the Story). If the proof has been adapted from previous work, it may be more appropriate to cite the original source instead.
3. In appendix Figure 5, the current argument is hard to show adding depth allows a deep SSMs to capture long-term dependencies with smaller eigenvalues across layers. Maybe there should be a bit more discussion.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies deep linear SSMs, and investigates several aspects of model memory and how it depends on stacking layers. For example, it is demonstrated that the autocorrelation function of a input stationary sequence can be computed as it is passed through the layers in the simple case of 1D state spaces, where one can observe that the hidden state variance explode exponentially. Another characterisation of memory is given by the group delay, which is shown again to increase as the depth of the model increases. Lastly, in the contrary direction the authors show that a narrow deep linear SSM is equivalent to a shallow and wide SSM. Some numerical tests were performed which confirms the equivalence, while suggesting there is some difference in optimisation.

### Strengths
The problem studied is an interesting one, namely how does deep SSMs differ from shallow ones. This is also timely as while the theory of shallow SSMs are relatively well understood, the advantages of deep SSMs are less studied.

### Weaknesses
There are a number of weaknesses to this paper, mainly on the implication of the results presented. The current paper reads like a collection of observations and simplified analysis, but with very little cohesion and overall message. Some of these issues are outlined in "questions" below. I believe that the paper in the current from is too preliminary, and lacks a concrete message to both theoreticians and practitioners.

The paper also suffers from clarify issues, e.g.:
1. Group delay explanation is not very clear as it is only loosely defined in words. If you are using the standard definition of group and phase delay, it would be useful to formally introduce them.
2. Equation (5) contains typos - $k$ should be $l$ or vice versa
3. Sec 3.3: Eq (6) describes a general $K$-layer SSM but the statement before it says $K=2$. This is rather confusing. I do not think the authors meant to take $K=2$ here and after.

### Questions
1. Fig 1C and associated discussion: The variance explosion is because of the following: for a stationary process, if the filter is not normalised (e.g. by $\ell_1$ norm), then the output will scale depending on the filter. However, in practice this would rarely be a problem because most deep architectures (deep SSM, transformers) have a layer norm or similar layer to scale this problem away. Can the authors comment on the significance of this observation and why it is interesting in practical architectures with normalisation?
2. The authors show (and this is straightforward) that a 1-dimensional hidden state deep linear SSM can be represented by a shallow but wide one. The equivalence makes sense, but then the authors claim that deep SSM have certain properties that are different from shallow ones. These properties are not due to optimisation, but the actual architecture under certain choices of the trainable parameters (e.g. exploding variance). This appears contradictory, since they are equivalent, there is also a choice of the shallow SSM that reproduces the exact behavior, and vice versa. This need to be explained.
3. Results discussed in lines 309-321 on page 6: the authors say that the results of the learned eigenvalues are "consistent" with their theoretical predictions. Can you pin-point exactly which theoretical results are consistent with these and why? The only consistency that I can see is that shallow and deep linear SSMs are equivalent, but this is not surprising or interesting. Moreover, the training dynamics is random, and so should the initial condition. How representative of these results are of the general case over multiple repeats?
4. The implication of the form of $P$ and $P^{-1}$ is not very clear at the end of the section 3. In fact, it does not resolve why the autocorrelations explosion is not contradicted by the equivalence of the models shown in (8).
5. Most diagonal SSM models are complex valued, not real valued. How do the results on page 7 change when they are complex numbers instead?
6. What is the practical importance of the findings of the variance explosion, the group delay, and the equivalence result on training/designing SSMs, that we currently do not know?

### Soundness
2

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
4

### Summary
This paper analyzes various sources of modeling memory in a Linear State Space Model. The authors take three different perspectives on how the width of the SSM layers and the depth of the model affect the learning capabilities of linear SSM under the simplified settings when B and C matrices are constant. 

The first perspective is that of memory and autocorrelation function, where the authors show that the correlation between inputs further apart in the time increases as the number of layers increases, showing that depth is beneficial when it comes to modeling memory.

The second perspective is that of learning the group delay response, where the authors show that for an impulse input, the response gets delayed as the the depth increases, as long as the input is appropriately normalized between the layers.

Finally, the authors study the case of a single layer (multidimensional) SSM that approximates a multi-layer single dimension SSM. In this case the authors show that since the $A$ matrix can be diagonalized, the resulting eigenvectors when multipled by $B$ appropriately delay the input signal.

### Strengths
The paper is quite insightful and uses simple and effective setups/experiments to explain how SSMs can model long term dependencies. It is also easy to follow barring few places where some of the terms introduced are not explained properly (I have pointed them in the weakness/questions section.)

Few things that I believe are important theoretical observations: 
- If any layer learns a long term dependency by setting $\lambda \approx 1$ then some other layer in the network needs to learn a $\lambda$ value close to $0$ in order to ensure that the state variance does not explode.
- The effects of the eigenvectors on the input B matrix when considering a linear SSM is very insightful!!

The experiments are well designed, and show the resulting and expected behavior of the models very well!

### Weaknesses
- Some of the key terms in the paper are not well defined in the pain section of the paper. 
	- In section 3.1, the authors do not define what R^{(k)}(\Delta) is, and it particular what $\Delta$ is. Judging by the derivation in the appendix and the previous work by Zucchet and Orvieto (2024) I believe that it is $E[X_{t+\Delta}X_{\Delta}]$ and that this quantity is independent of $t$. 
- The details in the experiment sections are a bit unclear. 
	- First of all, what is the motivation of the student teacher setup in the experiments. Is it mostly to generate $u_t$ and $y_t$ pairs. Should we assume that the teacher is a (untrained) planted model (i.e., with fixed A, B and C) and the job is mostly to give the subsequent target for the $u_t$? 
- What is the key motivation of the student teacher setup in the experiments. Is it mostly to generate $u_t$ and consequent output pairs to teach 
- In the experiments, the depth of the teacher considered is (maximum 3), and the students are even shallower. It would be nice to see the analysis done on more deeper networks, especially since the effects of depth is key in the theoretical results.
- The authors show results on LRU, some insight into what the model is and how the dynamics are different from the linear dynamics considered in the paper would be really appreciated!

### Questions
- What is the exact task that they authors are trying to solve for in their experiments. Should I just think of it as next step prediction, where are the $y_t$ coming from. Is the teacher trained/planted, and if it is trained where is $y_t$ coming from? 
- In the experiments the authors choose the input to be uncorrelated white noise. How would the results change if the inputs were correlated somehow?
- Can the authors elaborate on the setups for the experiments in Figures 1 and 2, the inputs are clear, but not sure what the targets are to elicit the behavior of the layers. 
- some questions regarding proposition 2:
	- What is the point of introducing proposition 2? 
	- Unless I am mistaken, it seems that in the proposition A is considered to be the same throughout different layers in the network (i.e, A is constant). However the final assumption and the derivation considered in the paper assumes that A^{(k)} changes with the layers. Am I missing something. The setup is a bit unclear, since in Section B of the appendix the authors assume what the form of H(z) is directly. 
	- Would appreciate some clarification here!

### Soundness
3

### Presentation
3

### Contribution
3
