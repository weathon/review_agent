# Identifying Feedforward and Feedback Controllable Subspaces of Neural Population Dynamics

- Decision: Reject
- Scores: 4, 8, 4, 2, 2

## Abstract
There is overwhelming evidence that cognition, perception, and action rely on feedback control. In neuroscience control is traditionally considered in the context of the brain controlling the body (i.e., the plant) dynamics, here we propose that neural population dynamics themselves should be controllable by, e.g., the activity of other brain areas. However, if and how neural population dynamics are amenable to different control strategies is poorly understood, in large part because machine learning methods to directly assess controllability in neural population dynamics are lacking. To address this gap, we developed a novel dimensionality reduction method, Feedback Controllability Components Analysis (FCCA), that identifies subspaces of linear dynamical systems that are most feedback controllable based on a new measure of feedback controllability. We further show that PCA identifies subspaces of linear dynamical systems that maximize a measure of feedforward controllability. As such, FCCA and PCA are data-driven methods to identify subspaces of neural population data (approximated as linear dynamical systems) that are most feedback and feedforward controllable respectively, and are thus natural contrasts for hypothesis testing. We developed new theory proving that non-normality of underlying dynamics determines the divergence between FCCA and PCA solutions, and confirmed this in numerical simulations of diverse linear and non-linear dynamical systems. To evaluate the degree to which different control strategies extract unsupervised subspaces relevant for task variables, we applied FCCA to diverse neural population recordings, and find that feedback controllable dynamics are geometrically distinct from PCA subspaces and are better predictors of animal behavior. These methods provide a novel approach towards analyzing neural population dynamics from a control theoretic perspective, and indicate that feedback controllable subspaces are important for behavior, providing insight into principles of neural computation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents FCCA, a method to identify subspaces of linear dynamical systems that are most feedback controllable for computational neuroscience. The authors relate feedforward and feedback controllability, then derives FCCA based on this relationship. Furthermore, they demonstrates that FCCA subspace compliments PCA for non-normal systems and is more predictive of behavior.

### Strengths
- Well-written and easy to follow even though the material is quite technical.
- Theoretical results appear to be correct.
- Highlights the importance of considering feedback control in computational neuroscience analysis.
- The insights on PCA and LQG and their relationship to control theory were interesting!

### Weaknesses
- **Insufficient Related Works**: There is no related works section and as a result it is unclear how the current work compares against existing methods that identify feedback controllability. Paragraph 2 in intro seems to be the most similar thing to a related works section. However, it is not properly cited. For example, in line 58, “prior works in network controllability” does not point to specific references.  Similarly, in line 61, the authors mention that current feedback control methods “are nascent”, but does not point to an explicit method or approach. Can the authors clarify what methods/citations are currently used to identify feedback control, even if they are simple or naive approaches? What about other measure of feedback control that are not related to identifying a subspace? Are there established feedback control measures in fields outside of computational neuroscience that could be referenced or compared? 
- **Motivation could be improved**: Line 80-81 mentions that the “goal of our work is provide insight into underlying principles of neural computation”. However, they do not clearly define what kind of insights they hope to extract through the identification of a feedback subspace. Can the authors elaborate on what kinds of insights a practitioner could hope to gain through a tool like FCCA? It would be very helpful to provide concrete examples of what kinds of analysis could be enabled that are previously not possible. Building on this, can the authors construct synthetic or real experiments that demonstrate how FCCA can “provide insights” successfully? 
- **insufficient Baselines**: In section 3 and 4, the authors compare performance between PCA and FCCA subspace, but do not provide comparisons against any other baselines. As a result it is not clear how FCCA compares to more recent literature. For instance, it might be helpful to provide comparisons against open loop dimensionality reduction methods to highlight how FCCA succeeds when existing methods might fail. 
- **Limited reproducibility**: 
    - I recommend an algorithm block sketched out starting from data, and ending with FCCA outputs. 
    - I also recommend a more detailed description of how non-normal synthetic data is generated. How are these matrices constrained to follow Dale’s law? What are the parameters tuned for manipulating non-normality? 


Minor, but affect presentation quality:
- Line 60: “Asses” -> “assess”
- Line 98: “controllabiliity" -> “controllability”
- Line 188: remove "w"
- Fig 2: what is cLDS? It is not described anywhere. Is there a formal definition somewhere?
- Line 304: What is FBC?
- Line 336, 404, 439: “appendix” is not referring to a specific part in the appendix.

### Questions
Please see Weaknesses section for majority of questions. Here are some additional questions:
- What is the computational complexity of this algorithm? Is it practical to compute for very long time series? Such as time series from large scale continuous recordings?
- What does it mean when PCA and FCCA subspaces are equivalent in the context of computational neuroscience?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors introduce FFCA, a linear dimensionality reduction approach that optimizes for feedback controllability instead of explained variance (PCA). They show that this method identifies different subspaces compared to PCA when the underlying dynamics are non-normal. They then derive a method of estimating this measure directly from observed neural activity, and apply it to various neural datasets, showing that the method yields neural subspaces that are consistently better predictors of behavior.

### Strengths
- FCCA has a clear interpretation as optimizing for a particular control-theoretic measure, and enjoys many of the same properties as PCA (simple/interpretable in the sense that subspaces are linear, and minimal hyperparameter overhead).
- Validation of significantly better prediction of downstream behavior relative to PCA across four neural datasets is convincing, and leads me to believe that this measure would be of broad interest/use to the neuroscience community. This is the first general-purpose, unsupervised linear dimensionality reduction approach that seems to be specifically suited towards identifying task-relevant dynamics that I'm aware of. As shown here (and in other works), task-relevant and high-variance components of activity need not coincide, an observation that I find to be currently underappreciated. 
- The derivation and theoretical analysis of FCCA appears to be sound.

### Weaknesses
- If I understand correctly, Eq. (13) gives an empirical estimator of FCCA from observed neural activity under the data model that the observations arise from an LDS driven by white noise. There should be a way to test how reliable of an estimator this is in a simple toy setup with known ground truth. Such a setup would also give the freedom to assess how estimation quality is affected by small deviations from this data model (e.g. driven by correlated noise instead of white noise). 
- The authors should comment on the computational efficiency of this method, and whether it can be prohibitive in certain cases. While PCA requires simply computing an SVD of the data matrix, this measure seems to require performing non-convex optimization over a cost function that is itself defined in terms of $T$ (lagged) covariance matrices. For practitioners: would it ever make sense to first fit a LDS to neural data and then posthoc apply the optimization procedure with estimated A and B in hand, simply for sake of computational efficiency?

### Questions
1. There should be analogous curves to Fig. 3 for the other neural datasets evaluated on, no? It would be good to put them in the appendix for completeness. 
2. In Fig. A3, for nonlinear experiments, is $A$ here the linearized dynamics/Jacobian, or just the weights? For switching networks, is this an average over the different $A$'s? It is surprising that the dependence on non-normality seem to break down completely when the underlying dynamics are nonlinear.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper describes a new dimensionality reduction method FCCA that seeks to find a subspace where simple controllers can control neural dynamics. They demonstrate their technique in simulation and on simple neural datasets.

### Strengths
Overall well-written and well-structured paper. It covers an important concept in neuroscience with applications to brain-computer devices. Math is well-described and reasoned through to give some intuition.

### Weaknesses
- Datasets chosen seemed very limited compared to the generality of the approach. Why is the number of recording sessions and not e.g. the dimensionality of the data a useful metric for Table 1? Many of these recording sessions are designed to be very similar across sessions. The monkey reaching tasks are overall very simple, and the prediction values obtained here (at highest dimension) seemed very comparable. 

- Inset to Fig 3 is meaningless without y axis values. 

- No direct measure of controllability was used making it hard to evaluate the predictive utility of FCCA over PCA.

### Questions
- If FCCA minimizes equation 13, can the authors show that? Is there a comparable metric for PCA as an alternative approach?

- Is there a metric for controllability related to eqn 13 that could be used in real datasets?

- If the goal is to "evaluate the task relevance of feedback vs. feedforward controllable subspaces, not to optimize decoding accuracy per se" (line 424) then is there a better way to quantify that goal?

- What is the complexity of the controller for each of these situations? If it is minimized (controller of low dimension), what is the actual value here? How low? How useful?

- How would this compare to other subspaces e.g. demixed PCA or others used in motor neuroscience or BCI explicitly for real-time control?

- When comparing angles between subspaces, shouldn't rotations and translations be accounted for? Or is this already invariant? 

- How guaranteed are solutions to find FCCA (given non-convexity)? Using L-BFGS and data-driven methods means solutions could be locally optimal and not globally so.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new dimensionality reduction method, "FCCA (Feedback Controllability Components Analysis)," to identify feedback controllable subspaces in neural population dynamics. It shows an index for "feedforward controllability" subspaces using the conventional controllability Gramian and introduces the classical PCA method which optimizes it. In contrast, it introduces a new index for "feedback controllability" subspaces using the Riccati equations for the control system and the observer system, and proposes the FCCA method to optimaize this index. Furthermore, it theoretically derives sufficient conditions under which the "feedforward controllability" subspace and the "feedback controllability" subspace coincide, and confirms this result using artificial data. Finally, FCCA is applied to multiple real neural recording datasets, such as rat hippocampus and monkey motor (M1) and somatosensory (S1) cortices, to examine whether the subspaces from FCCA and PCA can predict behavior using a decoder model. This study provides a novel data-driven method for analyzing neural population dynamics from a control-theoretic perspective and suggests that feedback controllable subspaces play an important role in behavioral performance.

### Strengths
The strengths of this study are as follows:

1. **Introduction of a control-theory-based index for neural population dynamics:** Although the importance of feedback control in brain function is widely recognized, indices for subspaces suitable for feedback control from neural population data have been lacking. The index proposed in this study is an important and unique perspective that addresses this problem.

2. **Proposal of an index invariant to state transformation:** The index of feedback controllability proposed in this study is shown to be invariant under state transformation in Theorem 1. Therefore, it is an important index that can be uniquely defined solely from the input-output relationship.

3. **Mathematical derivation of the relationship with existing methods:** Theorem 2 mathematically shows the sufficient conditions under which the proposed index (7) and the classical dimensionality reduction method (4) coincide. This theorem suggests that the non-normality of matrix A is the cause of the difference between the two, and this result is confirmed with artificial data.

4. **Demonstration of effectiveness in real data:** The fact that the proposed method was applied not to a single dataset but to multiple neural recording datasets from different brain regions and tasks, such as rat hippocampus (navigation task) and monkey M1/S1 (various reaching tasks), and consistently demonstrated its effectiveness is highly commendable.

### Weaknesses
While this study has many strengths, several significant concerns remain regarding the clarity of its theoretical foundations, the interpretability of the method, and the validity of its empirical evaluation.

First, many aspects of the theoretical definition and interpretation of the proposed method are unclear. A clear definition of "feedback controllability," which is central to this study, is not provided. The interpretation of the index used, $Tr(PQ)$ (or $Tr(\tilde{P}Q)$), is also difficult. While $P$ and $Q$ respectively have meaning as metrics for the feedback system and observer system as solutions to the Riccati equations, their product $PQ$ is not generally positive definite, and it is not explained what mathematical meaning it holds beyond being a mere "combination of indices."

Second, the validity of the FCCA method's derivation is difficult to verify. In particular, the explanation of the relationship between the Riccati equation solutions ($Q, \tilde{P}$) and the MMSE (Minimum Mean Square Error) prediction error covariance, as described in l.253-l.256, is insufficient. Furthermore, although the minimization of $Tr(PQ)$ was discussed up to section 2.2, the actual FCCA method (eq. 11) uses $Tr(\tilde{P}Q)$. There is a lack of discussion regarding the theoretical justification for this, and how much difference exists between $Tr(PQ)$ and $Tr(\tilde{P}Q)$.

Third, there is a problem with validation. The study's conclusions are based solely on a comparison between two specific dimensionality reduction methods (PCA and FCCA) under the strong assumption of a linear system. Considering that neural dynamics are inherently nonlinear, the generality of this conclusion is highly questionable. To truly demonstrate the superiority of the proposed method, a comparative study with a more diverse range of dimensionality reduction techniques is essential.

### Questions
### Main

1. How is "feedback controllability" clearly defined?
I understand that $Tr(PQ)$ serves as an measure of that subspace. However, similar to the definition of standard controllability (i.e., the existence of an input that can transition the state from any initial state to any final state), does "feedback controllability" also have such a definition?

2. I'm having trouble understanding the interpretation of the matrix product $PQ$.
The positive definite matrices $P$ and $Q$ each serve as parameters of the Lyapunov function for the internal system or observer, and have meaning as metrics of their respective spaces.
However, the matrix product $PQ$ does not seem to have such a meaning and is not generally positive definite.
While $PQ$ is invariant under bijective mappings of the state space, and its Trace represents a combination of indices for the subspaces of the control system and observer, does the matrix product $PQ$ itself have any direct meaning beyond this?

3. It is known that the solution to the Riccati equation is equivalent to the eigenvectors of the Hamiltonian matrix. Therefore, when deriving analysis based on the Riccati equation, interpretations using the Hamiltonian matrix are also commonly investigated. How is this handled in the current study?

4. I could not understand how the explanation in l.253-l.256 was derived.
Please explain the relationship between $Q, \tilde{P}$ (the solutions to the Riccati equations) and the error covariance of MMSE prediction.

5. Up to section 2.2, the estimation of $Tr(PQ)$ was discussed. Why does the actual FCCA method (13) use $\tilde{P}$ instead of $P$? Also, what is the extent of the difference between $Tr(PQ)$ and $Tr(\tilde{P}Q)$?

6. Pseudocode should be introduced for the Feedforward method (4), the FCCA method (13), and the decoder model used in Chapter 4 to clarify their respective input-output relationships.

7. I understand that the purpose, as stated in l.424, is to evaluate the difference between feedback vs. feedforward controllable subspaces.
However, I believe it is problematic to evaluate neural dynamics, which are difficult to describe as a linear system in practice, using only two specific dimensionality reduction methods that both presuppose a linear model.
Therefore, I think a comparison with more diverse dimensionality reduction methods is necessary.

### Minor

8. l.122, Equation (2): Stating that $dw(t)$ follows a Gaussian distribution is not accurate. For continuous-time white noise, it should be defined by $E[dw(t)] = 0$ and $E[dw(t)dw(t+\tau)^\top] = \delta(\tau) I$.

9. l.194: Should this be $PQ \rightarrow (T^\top)^{-1} PQ T^\top$ instead of $PQ \rightarrow (T^\top)^{-1} QP T^\top$? If so, subsequent proofs may need correction.

10. In Equation (13), the order is changed from $PQ$ to $QP$. Although the value does not change due to the Trace operation, it would be better to unify the notation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Feedback Controllability Components Analysis (FCCA), a dimensionality reduction technique for identifying subspaces of neural population activity that are most feedback controllable under a linear dynamical system (LDS) framework. The method builds on control-theoretic principles and contrasts with PCA, which the authors argue identifies subspaces maximizing feedforward controllability. Theoretical results link the divergence between FCCA and PCA to the non-normality of system dynamics. Empirical evaluations data recordings suggest that CCA subspaces are geometrically distinct from PCA subspaces and better predict animal behavior. The authors propose that feedback controllability provides a principled, interpretable, and behaviorally relevant perspective on neural dynamics.

### Strengths
1. The paper’s attempt to connect feedback control theory with dimensionality reduction is interesting and could, in principle, provide new insights into neural dynamics.
2. FCCA extends the control-theoretic concept of controllability to subspace identification in data, and the formulation is mathematically coherent.
3. The link between system non-normality and differences between PCA and FCCA is insightful.

### Weaknesses
1. While the paper claims to study “neural population dynamics,” the empirical systems analyzed are simple linear differential equations (LDEs) with no demonstrated correspondence to realistic neural dynamics. The analyses do not convincingly establish that the LDEs framework meaningfully captures nonlinear, high-dimensional neural processes such as attractor dynamics, oscillations, or state-dependent nonlinear feedbacks observed in real populations.
2. The method assumes the neural system can be adequately represented by a linear dynamical system, but the paper does not sufficiently justify this simplification. Classical control theory makes clear that notions of controllability and invariant subspaces are defined only for linear or locally linearized systems. This assumption may be invalid for strongly nonlinear neural activity, where local linearization can fail to capture relevant dynamics. More sophisticated methods for nonlinear controllability  (Lie-algebraic rank conditions) or Koopman operator–based linearizations (Brunton et al., 2015) exist and are not discussed.
3. The paper does not sufficiently contextualize its contribution within decades of work on controllable subspace identification, controlled invariant subspaces, and geometric control theory. Techniques for decomposing state-space into controllable/uncontrollable components (e.g., Kalman decomposition) are standard. Moreover, recent work in structured and sparse controllability and network control(e.g., structural controllability theory, Lin 1974) already addresses many aspects of identifying low-dimensional controllable subspaces. The paper does not clearly differentiate FCCA from these established approaches.
4. The mathematical presentation is sometimes confusing with the use of $u(t)$ and replacing it with Gaussian process. Consistent use of the notations would help readers follow the presentation better. The intuition behind its objective function, particularly how feedback controllability differs from standard controllability metrics, could be made clearer. 
5. The empirical evidence is limited.

### Questions
1. How sensitive is FCCA to inaccuracies in the estimated linear dynamical model?
2. Could FCCA be extended to nonlinear state-space models (e.g., using Koopman operators, local linearizations, or neural-network-based control models)?
3. How does FCCA relate to existing dynamical analysis methods such as jPCA, LFADS-based linearizations, or geometric control approaches?
4. What is the physiological interpretation of “feedback controllability”? Can it be linked to measurable cortical feedback pathways?
5. Will FCCA generalize to non-biological dynamical systems, such as artificial neural networks or complex engineered systems? A toy example of linearized nonlinear system around an operating point can be used to illustrate the methods effectiveness and limitations better.

### Soundness
3

### Presentation
1

### Contribution
2
