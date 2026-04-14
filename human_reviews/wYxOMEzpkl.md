# A Solvable Attention for Neural Scaling Laws

- Decision: Accept (Poster)
- Scores: 8, 8, 5, 6

## Abstract
Transformers and many other deep learning models are empirically shown to predictably enhance their performance as a power law in training time, model size, or the number of training data points, which is termed as the neural scaling law. This paper studies this intriguing phenomenon particularly for the transformer architecture in theoretical setups. Specifically, we propose a framework for linear self-attention, the underpinning block of transformer without softmax, to learn in an in-context manner, where the corresponding learning dynamics is modeled as a non-linear ordinary differential equation (ODE) system. Furthermore, we establish a procedure to derive a tractable approximate solution for this ODE system by reformulating it as a *Riccati equation*, which allows us to precisely characterize neural scaling laws for linear self-attention with training time, model size, data size, and the optimal compute. In addition, we reveal that the linear self-attention shares similar neural scaling laws with several other architectures when the context sequence length of the in-context learning is fixed, otherwise it would exhibit a different scaling law of training time.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This manuscript studies the dynamics of learning a particular structured in-context regression task with linear attention. Its main results are approximate exponents for the decay of error with time, data, and model size.

**Edit following author response: I have raised my score from 5 to 8**

### Strengths
I found this to be a generally interesting paper, and its topic should be of broad interest. I do have some concerns that preclude my recommending publication in its current form, but I think these should be addressable.

### Weaknesses
Here I list some concerns; their order should not be ascribed any meaning. 

- The title is inaccurate and vague, since the authors are not the first to study linear attention, and their solutions are approximate. I would suggest switching to something more descriptive, and would strongly suggest mentioning "linear attention" in the title. This critique extends to the abstract, which is similarly vague. 

- In that vein, please refer to "linear attention" or "softmax attention" rather than calling everything "self-attention"; this will make the paper less confusing to read. 

- The review of models for neural scaling laws in Lines 44-51 is inadequate. The authors should cite the original work on source-capacity conditions in kernel regression by Caponnetto and de Vito (from 2007!), and more of the substantial literature on scaling laws in random feature models that precedes the paper of Maloney et al. Suitable references are reviewed, for instance, in the cited work of Bordelon et al. or in the expository work of Atanasov et al. (https://arxiv.org/abs/2405.00592). The authors should also cite Paquette et al (https://arxiv.org/abs/2405.15074), which is contemporaneous with Bordelon et al 2024. 

- The authors don't discuss the growing literature on in-context learning of linear regression, which is closely related to their analysis. First, a more extensive comparison to Zhang et al (which I mention is now published as https://jmlr.org/papers/v25/23-1042.html) is required. Next, it could be interesting to compare the results presented here to those of Lu et al https://arxiv.org/abs/2405.11751, who studied the asymptotic behavior of linear attention trained using ridge regression, i.e., to convergence. 

- The paper would be much improved if the authors could relate their data model to other tasks and to natural data. At the end of the day, you're doing gradient flow on the MSE, so at least in certain limits you should be able to replace the specific data model with a Gaussian covariate model of matched moments. Would you get the same scaling laws if you studied in-context regression under source-capacity conditions, i.e., the generalization of the task studied in Lu et al to structured covariates and targets? 

- The main analytical results of the paper are approximations, and those approximations aren't clearly organized. For instance, the statement of Theorem 3.1 references Assumption 3.1 rather than directly stating that its result is an expansion at large $\psi_s$. It would be clearer to state this directly. Also, though the authors provide reasonably compelling numerical evidence that their approximations produce reasonable results, it would be better to make this mathematically precise, i.e., to give explicit error bounds. Otherwise, please state your results as Results rather than Theorems.

### Questions
- The discussion in lines 64-71 is not very accurate, as the cited paper of Bordelon et al studies scaling laws for *random feature models*, not neural networks that can learn features. 

- Why do you bother to write (6) in terms of a generic loss $\ell$ rather than just writing the MSE, which is used throughout? This creates unnecessary overhead for the reader. 

- Unless I'm mistaken, all experiments use linear attention. The claims regarding architecture (in)-dependence of certain scalings would be much stronger if you could add experiments with softmax attention.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper establishes a theoretical framework to study the gradient flow learning dynamics of a single linear attention layer on a synthetic multi-task regression task.
It provides closed-form solutions for the dominant ODE dynamics based on perturbation theory and uses them to derive neural scaling laws w.r.t. training time, data set size, model size, and compute budget.

### Strengths
- **S1: Presentation & Contribution** Although the paper's content is relatively dense and introduces a lot of symbols, the presentation is very clear and, in my opinion, strikes a very good balance between mathematical detail and prose explanations.
  Specifically, I appreciated the author's clear outlines before executing the mathematical steps, which helped me understand many of the details, and convinced me that the contribution is non-trivial.

- **S2: Generality** I believe the synthetic multi-task sparse regression task, as well as the derived solutions of the dominant ODE dynamics can be of interest to study other phenomena in transformers, not only neural scaling laws.
  Therefore, the presented framework seems useful for future works.

### Weaknesses
The following are mostly minor concerns which, if addressed, would strengthen my confidence.

- **W1 (minor): Gap to 'practical' attention** The paper studies a relatively simplistic attention layer without softmax, skip connections, MLPs, depth, etc.
  I think this is okay, because the contribution is to derive closed-form expressions for the dynamics, which is likely intractable when including these components.
  It would be interesting to further investigate how useful the theoretical predictions are in a setting that more closely resembles a practitioner's setup.
  Specifically:
  1. Theoretically: How does the parameterization $W_K^\top W_Q$ (used in practise), rather than $W_{KQ}$, affect the dynamics, i.e. how would the presented results change?
  2. Empirically: How well does the theoretical prediction (linear attention) match the simulation results in Figs. 3/4 when using softmax attention?
  3. Empirically: Many attention-based models are trained with AdamW. How well does the theoretical prediction (gradient flow) match the simulation results in Figs. 3/4 when training with AdamW+small learning rate?

- **W2 (minor): presentation/editing suggestions**
  - It would be great if the authors could provide some practical recommendations based on the theoretical insights into neural scaling laws in Section 4.
  - It would be great if the authors could highlight more connections and differences with related works throughout the text, e.g. other works studying the learning dynamics of MLPs and CNNs (the author may want to cite https://proceedings.mlr.press/v202/pinson23a/pinson23a.pdf), or contrasting the neural scaling laws in Tables 1,2 with other empirical or theoretical results.
  - In the paragraph above Theorem 3.1, could you highlight which set of equations have the same form as the Riccati equation?
    I also believe it would be good to extract some of the currently inlined math into environments to facilitate looking up symbols, specifically on page 6.
  - Typos:
    - Line 108: Why not just use $a \approx b$ to indicate approximately equal?
      Is this notation used in the main text?
    - Equation 2: I think it should be $\mathbb{R} \times \mathbb{R}^d$
    - Equation 7: Can you say what $H_s$ is?
      It looks like it contains the second moments of the data and labels.
    - L269: 'can be rather exact' sounds weird; do you mean 'approximate, tractable solution'
    - L285: Maybe use '^T' instead of '\dot' for consistency with L253.
    - L297: $\epsilon$ should be $\epsilon_s$.
    - L331: To me it sounds weird to say the solution is 'considerably exact'.
      Maybe replace with 'is a good approximation'.
    - L379/381: 'with respect' is missing 'to'
    - Figs. 3d/4b/4f: Add a legend for each curve.
    - L536: 'a variety properties' misses an 'of'

### Questions
Please see the questions from W1. Additional minor questions:

4. L387: Why does the compute not scale with $N$, the data set size?
   I believe that computing a gradient should scale linearly in $N$.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a theoretical analysis of power laws in in-context learning using Transformers. Specifically, it introduces a task setup called multitask sparse feature regression and approximates the learning dynamics of self-attention trained by gradient descent on this setup through perturbation analysis. The authors verify that the test loss follows scaling laws with respect to training time, model size, the amount of training data, and the optimal compute budget.

### Strengths
- This paper demonstrates a commendable effort in addressing the challenging theoretical analysis of neural scaling laws in in-context learning, which is both complex and impactful.
- It provides a visually clear figure explaining the multitask sparse feature regression task, along with an accessible proof outline that enhances reader comprehension.
- The paper establishes an interesting connection between in-context learning dynamics and the Riccati equation. It is particularly intriguing that $g_s^0h_s^0$ emerges as a conserved quantity.

### Weaknesses
- **The problem setup appears somewhat artificial.** While the gradient descent analysis for multitask sparse feature regression is impressive, the problem setting itself seems contrived and may lack applicability. For instance:
  + The probability of task selection $\mathcal{P}(S=s)$, task strength $\Lambda_s$, and context sequence length $\psi_s$ are all exponentially proportional to $s$. The assumption that these three factors are coordinated (e.g., larger $\mathcal{P}(S=s)$ implies larger $\Lambda_s$ and $\psi_s$) seems limiting for practical applications.
  + In this multitask sparse feature regression task, only $\phi(s,\hat{\boldsymbol{x}})$ is necessary to predict $\hat{y}$, making $\phi(s,\boldsymbol{x}^{(1)}),\dots,\phi(s,\boldsymbol{x}^{(\psi_s)})$ effectively irrelevant. This raises doubt as to whether this setup truly qualifies as in-context learning. A token-wise feed-forward network would likely suffice, giving the impression that self-attention has been unnecessarily added to complicate the problem.
- **Several typos and ambiguities in presentation.** A few observations are listed below:
  + In eq. (2), $\mathbb{R}^d \times \mathbb{R}$ should be $\mathbb{R} \times \mathbb{R}^d$.
  + In Section 2.2, $f:\mathbb{R}^{(\mathcal{N}_S+1) \times (\psi_s+1)} \mapsto \mathbb{R}$,(page 4, line 182), the $s$ is incorrectly capitalized, and $\mapsto$ should be $\to$.
  + In Section 2.3, $\mathbb{R}^{\mathcal{N}_s \times (\psi_s+1)}$ should be $\mathbb{R}^{(\mathcal{N}_s+1) \times (\psi_s+1)}$.
  + Below eq. (7), the phrase “does not depend on $n$, where ...” is misleading, as eq. (7) depends on the task type $s^{(n)}$; rephrasing this may clarify the intent.
  + In the second line of eq. (26), it seems that $\mathcal{O}(\epsilon_s^2)$ is unnecessary, whereas it may be needed in eq. (28).

### Questions
- $f_s^0$ depends on the training data, so is it ok not to take the expectation over the training data in the definition of Test Loss (eq. (14))?
- In Section 2.2 on the generation of in-context data, is it correct to assume that the data sequence $\boldsymbol{x}^{(1)},\dots,\boldsymbol{x}^{(\psi_s)}$ is sampled independently and identically from $\mathcal{P}_X$?
- Apologies if mistaken, but in eq. (31), is $\frac{\psi_s \\#_s }{N}$ missing? Wouldn’t this result in $a_s = \psi_s^2 \\#_s (1+\Lambda_s^2) / N$?
- When transforming the summation to an integral in eq. (62), is $f_s^0$ defined for non-integer $s \in \mathbb{R} \setminus \mathbb{N}$? Similarly, does $\mathcal{P}(S=s)$ in eq. (64) have the same consideration?

### Soundness
2

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
4

### Summary
The paper performs a theoretical study on the problem of in-context learning for attention models, where a number of $(x,y)$ pairs compose the input string to the model. Compared to a standard transformer, the paper simplifies the architecture by analyzing a single self-attention layer. Thus, the learnable parameters are a matrix corresponding to the merged queries a keys' parameters, and the value's parameter matrix. The data distribution is composed of a power-law distribution over a number of tasks, a sparse feature extractor, and a "task strength" parameter that weights the sparse feature extractor to generate the labels. The sequence length and the task strength also follow a power law distribution. 

The paper shows that the training dynamics of this model can be solved in closed form, assuming the sequence length is sufficiently large for all tasks, and keeping the zeroth order term of the expansion at large sequence length. The authors then use these closed-form solutions to get the test loss and devise neural scaling laws in various scenarios by varying the variables (model size, time, number of data points). They also cover the compute-optimal setting.

### Strengths
1. The exposition of the theory is overall clear. I particularly appreciated the "Procedure sketch", which guides the reader through the proof technique, from the non-tractability of the original problem to perturbation analysis and the several changes of variables involved. Also, I appreciate the structure and attention to detail of the appendix. It can be seen that the authors put considerable effort into making the proofs clear and well-organized.

2. The results in Section 4.1. recover well-known scaling laws previously observed both empirically and theoretically on simplified models. The results in Section 4.2 are more specific to the in-context setting and the toy model proposed and seem entirely new to me. 

3. The theory is supported by a sufficient suite of experiments.

### Weaknesses
In many cases, it is unclear which set of assumptions are due to reasonable empirical observations in in-context learning and neural scaling laws, and which ones are mainly there to obtain tractable calculations (and if so, why so). For instance:

1. Sparsity. Why the feature extractor can only be {-1, 0, 1}. No justification is provided as to why this should be realistic. In fact, it seems precisely constructed to have the expansion of $H$ as the sum of two matrices, with the second one having the sequence length as a multiplicative prefactor. 

2. Why should the task strength follow a power law distribution?

3. Why the distribution of the tasks should also be a power law?

I would appreciate it if the authors stated the purpose of the assumptions more explicitly. More broadly, this task seems specifically designed to get closed-form solutions and it is thus unclear how they can be re-used in future works.

Another issue is that the analysis is the *sequential* scaling of large context length limit, and then the other quantities (time, number of samples, model size). Assumption 3.1 on large context length allows the authors to use perturbation analysis to get the zero-th order correction of the expansion at large context length (which results in Theorem 3.1), and essentially drop all the higher order terms. However, in Section 4.2 they study the *joint* behavior of sequence length with the other scaling quantities. How do the authors know that other terms of the expansion would not be relevant if the joint limit were taken? 

Other issues:
1. $f^0_s(t)$, $f^1_s(t)$ is defined in Theorem 3.1 but referred to earlier (line 303). 
2. Why notation changes from d to D in Section 4.
3. Why approx in Eq. 15. 
4. I feel this paper on the asymptotic theory of in-context learning should be cited [1].

[1] Asymptotic theory of in-context learning by linear attention (https://arxiv.org/abs/2405.11751)

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
