# Universal Learning of Nonlinear Dynamics

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
We study the fundamental problem of learning a marginally stable unknown nonlinear dynamical system. We describe an algorithm for this problem, based on the technique of spectral filtering, which learns a mapping from past observations to the next based on a spectral representation of the system. Using techniques from online convex optimization, we prove vanishing prediction error for any nonlinear dynamical system that has finitely many marginally stable modes, with rates governed by a novel quantitative control-theoretic notion of learnability. The main technical component of our method is a new spectral filtering algorithm for linear dynamical systems, which incorporates past observations and applies to general noisy and marginally stable systems. This significantly generalizes the original spectral filtering algorithm to both asymmetric dynamics as well as incorporating noise correction, and is of independent interest.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Based on  spectral filtering, this paper proposes an algorithm to improperly learn  a marginally stable unknown nonlinear dynamical
system. The algorithm can predict future outputs   from past observations with  vanishing prediction error for any nonlinear dynamical system that has finitely many marginally stable modes. Experimental results on next-token prediction numerically demonstrate its effectiveness over several methods.

### Strengths
1: The paper generalizes the original spectral filtering algorithm designed for  linear dynamical systems with the symmetric transition matrix  to both asymmetric dynamics as well as incorporating noise correction. 

2: The authors introduce a simple construction to approximate any bounded, Lipschitz nonlinear system with a high-dimensional LDS and further provide a novel analytical framework to  derive learning guarantees in nonlinear settings using linear methods.

### Weaknesses
1: The authors overlook a substantial body of related work on learning nonlinear dynamical systems of the form (1.1). For example, many recent studies have investigated the use of Gaussian processes to approximate both the transition and observation functions to learn such systems [1-3].

2: For the identification of linear dynamical systems mentioned in Appendix A, many methods can be used to learn system matrices, such as subspace state-space system identification; however, these methods are not discussed by the authors. 

3: The paper is difficult to follow due to a lack of explanation of technical details. For instance, while the proposed algorithm is based on spectral filtering, the authors do not provide a detailed description of this technique in the paper.

[1] Fan X, Bonilla E V, O’Kane T, Sisson S A. Free-form variational inference for Gaussian process state-space models, International Conference on Machine Learning. 2023 : 9603-9622.

[2] Frigola R, Chen Y, Rasmussen C E. Variational Gaussian process state-space models. Advances in neural information processing systems, 27, 2014.

[3] Turner R, Deisenroth M, Rasmussen C. State-space inference and learning with Gaussian processes, Proceedings of the thirteenth international conference on artificial intelligence and statistics. 2010 : 868 – 875.

### Questions
1: In line 149, could the authors explain how to fold the terms $D$ and $\xi_t$ into the input of the previous iteration? 

2: How to derive the Algorithm 1?

3: In Algorithm 1, how to determine $h,m$  and $\eta_t$. Moreover, it would be helpful to clarify whether these parameters influence the algorithm’s performance. In addition, what is $\Pi_\mathcal{k}$ in line 208.

4: Could the authors elaborate on how their proposed algorithm extends or improves spectral filtering approaches, and clarify the specific distinctions from prior methods.

5: In line 196, the authors define that $\Theta^t=(J^t,M^t,P^t,N^t)$. Hence, what does $\Theta(\cdot)$ mean in line 222. The notation is somewhat confusing. 

6: In experiments, the baseline methods used for comparison are relatively simple. The authors are encouraged to include additional learning methods for nonlinear dynamical systems mentioned in Weakness for a more comprehensive evaluation.

7: The proposed algorithm seems to perform prediction only on the collected output samples, rather than truly forecasting future outputs beyond the observed data.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a novel method to learn an LDS observer for marginally stable nonlinear systems with finitely many modes. The proposed approach leverages Spectral filtering theory for marginally stable LDS, yielding an efficient algorithm based on online convex optimization.

### Strengths
1. Learning an one-step output predictor in the online learning setup is an interesting topic. 

2. The spectral filtering method yields a convex approximation of the long-horizon behavior of a marginally stable LDS with relatively small set of parameters. This feature offers an efficient online least square regression for problems that require long-horizon trajectory information.

### Weaknesses
1. **Overstated Claims** 

The title and several statements (e.g., the first sentence of the abstract) substantially overstate the scope of the work relative to the specific technical problem it addresses. The paper is **not** about learning a full dynamical system. Instead, it treats **state estimation + one-step prediction** as a black-box problem and learns an adaptive online (time-varying) least-squares regression (LSR) model from the input–output trajectory. This is distinct from the conventional notion of *learning nonlinear dynamics*, which entails the ability to perform multi-step prediction once an accurate state estimate is obtained from past data. In the proposed approach, the model depends on the true system output $y_{t+1}$ to predict $\hat{y}_{t+2}$. If the true $y$ is replaced with the observer output $\hat{y}$, then the prediction error would increase dramatically. 

2. **Overly strong assumption on the nonlinear system**

The paper relies on a strong assumption that the underlying data-generating process (the nonlinear system) is stable. However, the chaotic examples in Section 7 do not satisfy this assumption; yet their trajectories can still be estimated well from output data. In nonlinear observer theory, assuming strong stability conditions such as $\mathrm{Lip}(f)\leq 1$ is often overly restrictive. The control literature provides alternative frameworks for relaxing this assumption by using the notion of differential observability, roughly speaking, 
$$\sum_{t=0}^{T}| y_{t}^{a}-y_{t}^b | \leq \gamma |x_0^a-x_0^b|$$ for some $\gamma>0$ and $h$ is Lipschitz-bounded.  

For such systems, **contracting** (incrementally exponentially stable) observers can be designed [1]. Moreover, a recent work [2] shows that contracting systems can be embedded into stable LDS with the same dimension (not infinite-dimensional). These works help explain why the proposed method may appear to work for chaotic systems, even though they are not marginally stable in the sense assumed by the paper.

3. **Restrictive assumption: finitely many modes**

Nonlinear systems often exhibit *continuous modes*, rather than finitely many discrete modes. Hence, the assumption of finitely many marginally stable modes is quite restrictive and limits the generality of the theoretical claims. 

4. **Potential training instability**

The proposed algorithm may suffer from training stability issues. For instance, if the underlying dynamics operate in a specific mode, the learned LDS may experience mode collapse. When the operational mode changes abruptly due to external inputs or disturbances, the training process could become unstable. The authors briefly acknowledge some sort of instability in a footnote. To address this issue, it may need the notion of **persistent excitation** (PE) from the control literature, which is often used to show the stability of online LSR for adaptive control. Roughly speaking, the external input or internal dynamics excites a wide range of modes. 

5. **lack of experiments on noisy cases**

In Table 1, the authors claim that OSF can handle adversarial noise. However, the illustrative examples are noise-free, and the theoretical framework itself assumes the absence of measurement noise.

6. **Inaccurate statements on system identification**

The claims about classical system identification (Sys-ID) in Table 1 are inaccurate. System identification has been studied extensively for nearly six decades, and the related work discussed in the paper represents only a small subset of the existing literature, particularly from the ML community. As far as I know, there are many Sys-ID approaches which can handle at least two or three of those requirements listed in Table 1.

[1] W. Lohmiller and J.J.E. Slotine. On Contraction Analysis for Nonlinear Systems. Automatica, 1998. 

[2] B. Yi and I. Manchester. On the equivalence of contraction and Koopman approaches for nonlinear stability and control. IEEE-TAC, 2023.

### Questions
1. Below Eq. (2.1), the authors wrote "We assume that $D,\xi_t=0$ as these terms can be folded into the input of the previous iteration". I do not understand it. How can one get rid of measurement noise without loss of generality?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies the problem of learning a marginally stable unknown nonlinear dynamical
system. The authors show that, under suitable assumptions, such systems can be approximated by a high-dimension linear dynamicl systems. Then, the authors design a predictor, using techniques from spectral filtering, with vanishing regret for nonlinear dynamical systems with finitely many marginally stable modes. Along the way, the authors identify a complexity measure of nonlinear dynamical systems that governs learnability. Finally, the authors complement their theoretical bounds with experiments on the Lorenz systems.

### Strengths
- This paper is really well-written. I actually really enjoyed the prose, and the authors do a great job describing the significance of their results and providing the reader with intuition.
- The problem studied is relevant and of interest to the broader ML community

### Weaknesses
Unfortunately, there are several issues that need to be addressed. First, the paper does not conform to the ICLR style guides as the margins are too small. More importantly, I do not think the main text is very understandable, as key quantities/results are left undefined. For example:
- A "Luenberger program" is repeatedly discussed, but never formally defined in the main text
- The system's "robust observability constant $Q_{\star}$ is repeatedly referenced, but never defined in the main text
- The key ideas of spectral filtering are not introduced 
- Key assumptions are not fully defined (i.e. see Assumption 3.1 references a Lemma in the Appendix)
- Key algorithmic pieces are not fully specified (see Line 2 of Algorithm 1)
- Key definitions are missing from the main text (i.e. see $Q_{\star}$ in Theorem 4.1).
- Proper descriptions of the plots are missing. What are the shaded areas in Figure 1? 
- SFeDMD is claimed to be an algorithm by the authors, but not properly defined anywhere in the main text. 

In my opinion, the main text of a paper should be as self-contained as possible, which I do not believe is the case with this paper. I do not doubt the contributions of this paper. However, I feel that this paper needs a significant rewrite of its main text in order fit the usual standards of a top-tier ML conference paper.

### Questions
See Weaknesses.

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
4

### Summary
This paper studies the problem of learning output predictor for nonlinear systems. The authors propose algorithms based on the idea of approximating a nonlinear dynamical system using high-dimensional linear dynamical system (LDS). Under certain assumptions, the authors show that their algorithm $\sqrt{T}$-regret for general nonlinear systems and the regret bounds become sharper when specializing to linear systems. The authors also show that a quantity ``$Q^{\star}$” plays a key role in their regret analysis and results, which corresponds to the LDSs used to approximate the nonlinear system.

### Strengths
1.	The authors provide comprehensive theoretical analysis for the proposed algorithm.
2.	The algorithm design and particularly the regret analysis seem to contain significant novelty.
3.	Existing works typically consider learning the output of linear systems. Extending it to nonlinear systems is a meaningful step.

### Weaknesses
1.	The assumptions made in the paper are pretty strong (e.g., Assumption 3.1), which require further justifications. In particular, the bounded state and output assumption can make the claim that the regret results hold for marginally-stable system vacuous. (Please see the questions 1-3 below) 
2.	The proposed algorithm (Algorithm 1) may not be constructive, e.g., it is not fully clear how the input parameters can be (please see the question 3 below).
3.	The regret bounds provided in Theorem 4.1 depend exponentially on certain problem parameters.

### Questions
1.	The authors assume that $x_t$ and $y_t$ are upper bounded by $R$. While the authors claim that their regret results (including that for nonlinear system and linear system) hold for marginally-stable systems, the regret bounds scale as the upper bound $R$ and $R$ can in turn scale polynomially as $T$ when the underlying system becomes marginally stable. Thus, for marginally-stable systems, the overall regret upper bounds are no longer sublinear in $T$.
2.	How to justify Assumption 3.1(iii)? Since Assumption 3.1(iii) is required for the $(A^{\prime},C^{\prime})$, given the original nonlinear system, how to check whether this assumption holds? In particular, from Lemma C.5 in the Appendix, it is not fully clear how $A^{\prime}$ and $C^{\prime}$ may be explicitly constructed from the original nonlinear system.
3.	While the algorithm proposed in this paper works for learning the output of asymmetric LDSs, the $A$ matrix needs to be diagonalizable and normal, which is still not the general LDS. This point may need to be made explicit in the abstract of the paper.
4.	In the statement of Theorem 4.1, the input parameters $D$ and step size $\eta_t$ need to be chosen based on the quantity $Q^{\star}$. Is there a constructive/explicit way to obtain the value of $Q^{\star}$. In Definition C.7, the value of $Q^{\star}$ is given by an optimization problem which does not seem to yield an explicit solution. Additionally, what is $\gamma$ in the statement of Theorem 4.1, is it also an input parameter to Algorithm 1? The authors provide upper bounds on $Q^{\star}$. Will using the upper bounds enough for setting the aforementioned input parameters?
5.	In Line 301, the authors mention a class of marginally stable bounded nonlinear predictors, what is this class and could the authors provide a formal definition for it? Also, the authors mention at some places in the paper (e.g., in the abstract) the terminology marginally-stable nonlinear system, what is the formal definition for this?
6.	Please correct the double quotation marks in the paper.

### Soundness
3

### Presentation
2

### Contribution
3
