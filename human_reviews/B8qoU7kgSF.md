# Generalization  Bounds for  Neural Ordinary Differential Equations and Residual Neural Networks

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 3, 5, 3

## Abstract
Neural ordinary differential equations (neural ODEs) represent a widely-used
class of deep learning models characterized by continuous depth. Understand-
ing the generalization error bound is important to evaluate how well a model is
expected to perform on new, unseen data. Earlier works in this direction involved
considering the linear case on the dynamics function (a function that models the
evolution of state variables) of Neural ODE Marion (2024). Other related work
is on bound for Neural Controlled ODE Bleistein & Guilloux (2023) that de-
pends on the sampling gap. We consider a class of neural ordinary differential
equations (ODEs) with a general nonlinear function for time-dependent and time-
independent cases which is Lipschitz with respect to state variables. We observed
that the solution of the neural ODEs would be of bound variations if we assume
that the dynamics function of Neural ODEs is Lipschitz continuous with respect
to the hidden state. We derive a generalization bound for the time-dependent
and time-independent Neural ODEs.Using the fact that Neural ODEs are limiting
cases of time-dependent Neural ODEs we obtained a bound for the residual neural
networks. We showed the effect of overparameterization and domain bound in the
generalization error bound. This is the first time, the generalization bound for the
Neural ODE with a more general non-linear function has been found.

## Human Reviews

## Human Reviewer 1

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
The authors claim to prove new generalization bounds for neural ODEs and residual neural networks. However, these claims are largely unsupported since their work does not significantly improve on Marion (2023) and Bleistein and Guilloux (2024). Some lemmas and proofs are directly borrowed along with notations from these two works without sufficient citations, which might be considered as a case of light plagiarism. The title is almost identical to the work of Marion (2023).

### Strengths
I do not believe that this paper has significant strengths.

### Weaknesses
**Section 2.** The related work is close to insufficient and misses several recent contributions to the field, such as Marion, Wu et al. (2024) and Chen (2024). 

**Section 3.** This section compiles a list of definitions and Lemmas without sufficiently motivating their introduction. I would suggest a major rewriting of this section in order to guide the reader through the proofs. 
* Also, Lemma 3.8 does not exist in Bartlett 2017b, and I believe that the Lemma as stated is wrong: there should not be a factor $1/\sqrt{n}$ in the integral, but rather a factor $1/\varepsilon$. See Bartlett 2017b Lemma A.5. 

**Section 4.** 
* The learning setup is unclear. The authors write "Let z be the solution of Neural ODE with x as the initial condition and let y be the true solution of the true differential equation learned by Neural ODE given by equation (4.3)" (l. 221-223): it is unclear what is meant by the "true solution". Do the authors assume a generative model for the data ? In this case, it should be introduced.
* The authors write that the empirical risk "cannot be optimized, since we do not have access to the continuous data." (l. 229 - 230). The authors have not introduced any form of continuous data, nor do they explain why the empirical risk cannot be optimized.
* This section seems to plagiarize Bleistein & Guilloux (2024) section 3.2, who consider a generative model where a continuous function, which is only observed at a discrete set of sampling times, generates the outcome through an unknown neural ODE. I believe that the authors have carelessly copied this text, hence introducing the two confusing sentences mentioned above which make no sens in their setting as it stands. 

**Section 5.**    

* The main contribution of this part seems to be an adaptation of Proposition 2 of Marion (2023) to the general non-linear case. The results in the section strongly resemble the results of Bleistein and Guilloux (2024) --- see Lemma 3.3. These results should be in my opinion at least ackowledged in the main text. 
* Settings concerns about plagiarism aside, the result cited here is directly implied by the results from Bleistein & Guilloux (2024), who establish generalization bounds for neural CDEs of the form  $ dz(t)  = \mathbf{G}(z(t))dx(t)$ , where $ \mathbf{G} $ is a generic neural network. Indeed, by setting $ x(t) = t $, one recovers a generic neural ODE. The authors of the aforementioned paper highlight the proximity between both models in Figure 2 of their paper.  
* In the abstract, the authors claim to have "showed the effect of overparameterization and domain bound in the generalization error bound". This is a strong overstatement, since the type of arguments used by the authors only work in the case where $n$ is taken to be sufficiently large to obtain concentration ; even if in this case the number of parameters exceeds the number of observations, these bounds become vacuous in this setting, since the bound presented in Theorem 5.9 does not tend to $0$ when $d$ grows at the same rate than $n$. Hence these bounds say nothing about the overparametrized regime, in which it is typically observed that neural network achieve good prediction performance even if they completely overfit the training data (see for instance Bartlett 2019).  

**Section 6.**
* Both Marion (2023) and Bleistein and Guilloux (2024) invoke discretization based arguments to go from continuous neural-ODE like architectures to discrete ResNet-type architectures. I do not see such an argument here, and am hence unconvinced by the soundness of Theorem 6.1. In particular, the authors simply write that "a neural ODE with an euler solver and $\Delta t= 1$ replicated the ResNet updates, it follows that the solution space of ResNets is contained within the solution space of Neural ODEs." (l. 366-368).

**Section 7.** 

* The authors claim to perform these experiments on neural ODEs. Given the previous approximations in the paper and the strong similarities of experiments displayed in figures 2 and 3 with the experimental section of Marion (2023), I strongly suspect that these experiments are carried out on ResNets rather than **continuous** neural ODEs. 
* Writing that experiment 1 validates Theorem 5.9 (l. 385) is an overstatement: the authors show (without any confidence intervals) on a purely synthetic dataset that the generalization error increases with the number of hidden units. However, the details on the model are insufficient (what is exactly meant by the number of hidden units ?). Also, since no precision is given on the training data, it is unclear whether the model operates in an overparametrized or underparametrized setting. 
* The experiment displayed in Figure 3 is directly copied from Marion (2023). This article should at least be acknowledged here in my opinion.  
* The experiment displayed in Figure 2 is novel and investigates the effect of penalizing the loss of a neural ODE with the bound of the solution, hence favoring solutions with a low euclidian norm. However, the experiment is not conclusive due to the high variance and the little variability of the mean for every choice of regularization. 
* Figures should be included in a vectorized format (PNG or PDF).   
* A experimental appendix should be added, that includes a detailed overview of the experiments.

**References**

Bartlett, Peter et al., "Spectrally-normalized margin bounds for neural networks", Neurips 2017. 

Bartlett, Peter et al., "Benign overfitting in Linear Regression", Proceedings of the National Academy of Sciences, 2020.

Bleistein, Linus and Guilloux, Agathe, "On the Generalization and Approximation Capacities of Neural Controlled Differential Equations", ICLR 2024.

Chen, Yihang et al., Generalization of Scaled Deep ResNets in the Mean-Field Regime, ICLR 2024. 

Marion, Pierre, "Generalization bounds for neural ordinary differential equations and deep residual networks", Neurips 2023.

Marion, Pierre, Wu, Yu-Han, et al., Implicit regularization of deep residual networks towards neural ODEs, ICLR 2024.

### Questions
I believe that this paper is largely insufficient for publication as it stands due to a lack of novelty, and often teeters on the brink of plagiarism. I strongly encourage the authors not to submit this work at the moment and to read the ethics requirements of ICLR 2025. Many points can be improved (see above). I list a few questions bellow. 

* Can you provide extensive details on the experiments carried out ? In particular, I would appreciate a mathematical formulation of the model use to generate your data and architectural details. Also, please carefully check that the experiments are run with neural ODEs instead of ResNets. 
* Please provide more mathematical details on your neural ODE to ResNet conversion (Theorem 6.1).

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
Generalization bound of neural ODEs whose vector field is an MLP

### Strengths
1. give an bound estimation of $z(t)$ based on the Lipschitz constant of an MLP $f(z)$
2. follow the similar technique as [1] to derive the generalization bound of $\dot z= f(z) $ 

[1] Bartlett et, al. Spectrally-normalized margin bounds for neural networks, NeurIPS 2017.

### Weaknesses
1. The bound estimation of $\|z(t)\|$ is very loose due to the Gronwall's inequality $u(t)\leq \alpha(t)+\int_a^t\beta(s)u(s)ds$. In this case, $\beta$ is the Lipschitz bound $\mathrm{Lip}(f)$ and $\alpha$ is a bound related to $\mathrm{Lip}(f)$ and bias norm. Thus, the downstream analysis of generalization bound could be very conservative. 
2. $\mathrm{Lip}(f)$ is estimated using the product of spectral norm bounds, which is again very loose. The SOTA estimation is based on some semidefinite programming formulation, see [2].
3. The assumption of globally Lipschitz $f$ is quite strong as the popular transformer architecture is only locally Lipschitz. 
4. The experimental results are quite weak, lacking of extensive comparison study. 
5. The presentation is poor. A few (not all) examples are listed as follows:

- The mapping $z(t)\rightarrow y$ is not defined. 
- Assumption 2 involves $A_i(t)$ and $b_i(t)$ which are introduced later in Section 5.
- A right ) is missing in 5.4.
- [1] appears twice in the reference.

[2] P. Pauli et. al. Novel quadratic constraints for extending lipsdp beyond slope-restricted activations, ICLR 2024.

### Questions
1. There exist many generalization bound analysis of residual networks. Can you provide some comparison studies with Thm. 6.1?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper provides generalization bounds for neural ODEs. It extends the class of neural networks used as the dynamics function. The bound is applied also to residual neural networks. Numerical experiments are used to show the effect of hyperparameters on the generalization gap.

### Strengths
* The results are applicable to a much larger class of neural ODEs than prior work, such as Marion's paper which is only for when f depends linearly on the parameters. 
* The prior work is well explained. Important lemmas from the prior work as used.
* I did not find errors in any of the proofs themselves.

### Weaknesses
* Some of the notation is conflicting/confusing. Please see the questions. 
* The prior work of Bleistein and Guilloux, and Marion is referenced. However, the bounds derived in these paper and in this paper are not compared.
* The main theorems 5.9 and 6.1 have only an outline of the proof, and there is not a full proof in the appendix.
* The numerically illustrations are missing details. Please see the questions.

### Questions
* On lines 89 and 93, the initial condition is given by $z(0)=\phi_{\theta(t)}(u)$. Why does the initial condition depend on the parameters at time t instead of time 0?
* On line 168, should it be "if" instead of "then"? Is line 169 an assumption of the lemma?
* In Assumption 3 (line 236), the outcome y is said to be in R. Is y a single number (say the solution at a final time) or is y a function of time? Same question for Assumption 4 and the loss function. Does this theory only apply for one-dimensional ODEs?
* Why is the risk a function of f in Definition 3.9, but is a function of z in section 4?
* Why is f a function of z, t, and theta in Assumption 1, but only a function of z and theta in equation (4.3)?
* In the numerical illustrations, what are the  regularization loss functions for either case?
* In the numerical illustration, how is the synethic data generated? Is it from numerical solutions of an ODE? If so, what ODE?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This submission proposes generalization bounds for neural ODEs and residual neural networks (the latter being a discretization of the first). 
The first 5 pages of the paper are dedicated to related works and preliminary results. The main results are in Theorems 5.9 and 6.9, generalizing results from Marion (2024), where the author only consider linear parametrization of the residuals (while still having non linear residuals with respect to the activations). Experiments on synthetic data are conducted.

### Strengths
The paper studies the interesting problem of deriving generalization bounds for residual architectures which are at the core of most successful deep learning methods.

### Weaknesses
In my opinion, this paper is not suitable for acceptance at ICLR. It appears incomplete and lacking in polish. Below are specific points:

- The paper shares almost the exact same title as Marion (2024), with only the term "deep" removed. This is inappropriate.
- The obtained generalization bounds in Theorems 5.9 and 6.1) are not commented on and, most importantly, are not compared with existing ones from Marion (2024).
- The second experiment (Fig. 2) appears very similar to experiments shown in Marion (2024) (Figs. 1 and 2), yet Marion’s work is not cited here.
- The paper references only around 20 prior works, which is insufficient. A broader acknowledgment of previous studies is necessary (see the references cited by Marion (2024) as a comparison).
- The bibliography is poorly presented and lacks formatting consistency.
- There are no experimental details provided. I looked in the appendix, but none were included.


Overall, the paper appears to have been submitted without adequate proofreading (see, for instance, the last sentence of the abstract). In addition:

- In Assumption 1, there is an unexpected dependence on time—this should be clarified.
- The symbol $z$ in line 227 is the same as the notation for the ODE solution. This is confusing. 
- Line 230: The expression involving the $\arg\min$ is difficult to understand. The $\arg\min$ is taken over $\theta$, yet it is denoted as a function $f$ (that is itself parametrized by $\theta$). Furthermore, $\arg\min$ is applied to $\theta(t) \in \theta(t)$, which is extremely unclear and problematic.
- Line 235: Typo present.
- In Lemma 5.1, it would be helpful to explicitly state that Assumption 1 is being used. 
- In Lemma 5.1, the structure is confusing. It is hard to tell what is an assumption and what is a result. Key definitions (e.g., for $f$) are also missing. Can you clarify ? 
- Line 265: The presentation here lacks rigor. Can you please specify the assumptions?
- There are multiple typos in the use of parentheses in lines 262, 267, and 272, which is unacceptable.
- Multiple typos are present in the experimental section (e.g., lines 423 and 429).

### Questions
- How different are your experimental results from those of Marion (2024)? 
- Why is there sometimes an additional t argument within the parametrized function in the neural ODES? 
- Both theorems provide the same bounds, which requires clarification—how is this possible?
- Please also see the questions and remarks in the previous section.

### Soundness
1

### Presentation
1

### Contribution
2
