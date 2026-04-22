# Information-Theoretic Bayesian Optimization for Bilevel Optimization Problems

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
A bilevel optimization problem consists of two optimization problems nested as an upper- and a lower-level problem, in which the optimality of the lower-level problem defines a constraint for the upper-level problem. This paper considers Bayesian optimization (BO) for the case that both the upper- and lower-levels involve expensive black-box functions. Because of its nested structure, bilevel optimization has a complex problem definition and, compared with other standard extensions of BO such as multi-objective or constraint settings, it has not been widely studied. We propose an information-theoretic approach that considers the information gain of both the upper- and lower-optimal solutions and values. This enables us to define a unified criterion that measures the benefit for both level problems, simultaneously. Further, we also show a practical lower bound based approach to evaluating the information gain. We empirically demonstrate the effectiveness of our proposed method through several benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Bilevel Optimization via Lower-bound based Joint Entropy Search (BLJES), a bilevel Bayesian optimization method grounded in an information-theoretic framework.
BLJES simultaneously considers the information gain regarding both the upper-level and lower-level optima (points and values), thereby defining a unified acquisition criterion that evaluates the overall benefit across both levels.

Because exact maximization of this two-level information gain is computationally intractable, the authors derive a variational lower-bound approximation for it and show that this formulation can be viewed as a natural extension of single-level information-theoretic Bayesian optimization.

The effectiveness of the proposed method is empirically demonstrated through experiments using sample-path functions from Gaussian processes as well as several benchmark optimization functions.

### Strengths
While BILBO can only be applied to Bayesian optimization on discrete domains by design, this paper derives the gradient of the BLJES objective function, enabling its application to continuous domains as well. This constitutes a qualitative and practical advantage of BLJES over BILBO.
Also BILBO shows superior performance comparede to BILBO in the experimental results reported in the paper.

### Weaknesses
Compared with BILBO, the following weak points can be identified:

- Lack of theoretical performance guarantees.

- As I understand it, BLJES, when extended to constrained bilevel optimization, evaluates the information gain under feasibility conditions by approximating the ``predictive distribution given the optimal value'' with a truncated variational distribution, following the extension proposed by Takeno et al. (2022b).
In other words, during the evaluation of the acquisition function, the predictive distribution is truncated by the feasibility constraint. However, this paper does not provide theoretical guarantees on zero-violation assurance.
Although the experiments show good performance on constrained benchmarks, BLJES does not include a safety mechanism (e.g., trust region) that ensures all samples are taken strictly within the feasible set. It lacks an explicit guarantee for high-probability constraint satisfaction so some users would worry about using BLJES.

- It is better to provide source code and dataset for improving the reproducibility of the work.

### Questions
- In information-theoretic approaches, is it fundamentally impossible to provide some form of theoretical guarantee, such as a no-regret bound?
While the experiments presented in this paper convincingly demonstrate strong empirical performance, having a theoretical guarantee would make the method more trustworthy for practical use.

- I can also imagine applications where it is important to guarantee constraint satisfaction with high probability.
Is it possible to ensure this—similarly to BILBO—by combining BLJES with an additional trust-region–style feasibility filter or other operational safety mechanism as a front-end module, so that the method can satisfy constraints with high probability? If not, why?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses bilevel optimization problems where both the upper and lower level objective functions are expensive black-box functions.  The authors are the first to introduce information-theoretic principles to this domain, proposing a new acquisition function named BLJES (Bilevel optimization via Lower-bound based Joint Entropy Search). The core idea is to shift the optimization goal from directly seeking higher function values to maximizing the bilevel information gain regarding the optimal solutions $(x^\*, \theta^\*)$ and their corresponding optimal values $(f^\*, g^\*)$ for both levels. This unified decision criterion balances exploration across both levels of the problem. As direct computation of this information gain (which is the mutual information) is intractable, the paper derives a computable variational lower bound. A key technical contribution is the creative extension of the "truncation approximation" concept, which is mature in single-level BO, to the complex structure of bilevel problems. Furthermore, the paper demonstrates the flexibility of this information-theoretic framework, showing its natural extension to the decoupled setting where observations can be separated and to more complex scenarios involving expensive black-box constraints. Experimental results show that BLJES achieves competitive performance against the current state-of-the-art method, BILBO, on various synthetic datasets and benchmark problems.

### Strengths
- This work is the first to establish a rigorous information-theoretic framework for bilevel Bayesian optimization. The proposed bilevel information gain concept offers a novel perspective for tackling this complex problem.

- The paper successfully transforms a theoretically ideal yet computationally intractable objective (mutual information) into an engineering-feasible algorithm through a series of well-founded approximations, including the variational lower bound, truncation approximation, and Monte Carlo sampling. The extension of the truncation approximation from the single-level to the bilevel case appears intuitively sound.

### Weaknesses
- A significant shortcoming of this paper is the absence of theoretical analysis for the proposed algorithm. The authors note in the introduction that the SOTA comparison, BILBO, possesses a theoretical regret guarantee, yet this paper provides no equivalent theoretical support for BLJES.  Although the experimental section uses bilevel simple regret as an evaluation metric and demonstrates good empirical convergence, this is not a substitute for formal theoretical proof. The absence of theoretical results such as a regret bound leaves the algorithm’s sample complexity and convergence behavior under finite budgets uncharacterized.  This is a considerable theoretical gap for a Bayesian optimization method, where sample efficiency is central to its purpose.

- All experiments are conducted in extremely low-dimensional spaces (both upper and lower dimensions $d_X, d_\Theta \leq 2$).  This severely limits the generality of the results. It is well known that BO methods suffer from the curse of dimensionality, and the nested nature of bilevel optimization likely worsens this challenge.  Thus, success on low-dimensional synthetic problems does not demonstrate scalability or robustness of BLJES in realistic, high-dimensional cases.

-  Although one experiment includes a simulator-based energy market problem, the vast majority still rely on synthetic functions sampled from GP priors and standard academic benchmarks.  To convincingly demonstrate the method’s practical value, validation on real-world black-box problems (e.g., engineering design, hyperparameter tuning) is needed.  The current experimental setup feels thin and does not fully exhibit the method’s ability to handle complex, noisy, or constrained real-world tasks.

### Questions
The core methodology relies on a variational lower bound to approximate the intractable mutual information. The quality of this bound, and the validity of maximizing it as a proxy for the true information gain, critically depends on how well the chosen variational distribution $q$ approximates the true posterior $p$. The paper mainly justifies this approach by citing its empirical success in prior work, but does not provide a theoretical analysis of the bound's tightness in this more complex bilevel optimization setting.

Can the authors provide more theoretical results regarding the tightness of this bound? For example, is it possible to theoretically analyze the error induced by this approximation, or under what conditions this lower bound becomes a closer approximation to the true mutual information? Providing analysis in this area would significantly strengthen the theoretical completeness of the proposed method.

### Soundness
3

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
3

### Summary
This paper proposes an information-theoretic framework for Bayesian Optimization (BO) in bilevel optimization problems, where both the upper- and lower-level objectives are expensive black-box functions. The authors introduce a new acquisition criterion based on the mutual information (MI) between potential observations and the optimal solutions and values of both levels, termed bilevel information gain. Because the exact evaluation of this MI is intractable, the paper derives a variational lower bound combined with a truncation-based Gaussian approximation to obtain a computationally feasible expression. This approximation extends prior single-level entropy-based BO methods (e.g., Joint Entropy Search) to the bilevel setting. Moreover, the formulation naturally accommodates decoupled observation settings, where upper- and lower-level outputs can be obtained separately, and can be extended to constrained bilevel problems. The proposed method provides the first unified information-theoretic formulation for bilevel BO. The authors present analytical derivations for the lower-bound approximation, demonstrate flexibility under various experimental setups, and validate the approach on Gaussian process sample paths and benchmark test functions. The presentation is mathematically detailed and the empirical results show generally strong performance across tasks. Overall, this work offers a novel and principled direction for handling the intractable mutual information forms arising in bilevel BO and for reducing the computational burden of evaluating complex conditional Gaussian distributions.

### Strengths
This paper presents several strengths across originality, quality, and significance.
First, the work introduces a new (MI) gain metric that jointly considers both upper and lower levels in bilevel optimization, addressing a clear gap in existing research. The proposed approach provides a principled way to quantify shared information between levels, supported by a mathematically grounded derivation. By applying a variational lower bound and a truncated Gaussian approximation, the method achieves a computationally tractable form of the otherwise intractable MI expression. Furthermore, the framework is flexible enough to handle decoupled settings, where one level’s observation may be missing, and can also be extended to constrained bilevel problems.
Second, the presentation is clear and well-structured, with solid mathematical explanations and step-by-step reasoning. The authors provide thorough explanations and clarifications for each design choice, helping readers understand the rationale behind the modeling and approximation steps. The logical flow from theoretical formulation to implementation is coherent and easy to follow.
Third, the benchmark results are impressive and superior in most of the tasks. This shows a wide range of applicable cases where fewer drawbacks are seen.
Finally, this work is not only supported by convincing experimental results on benchmark problems but also offers a general mathematical framework that can inspire further research on information-theoretic Bayesian optimization for complex hierarchical problems.

### Weaknesses
Despite the solid strengths, there are a few aspects that could be improved.
First, the discussion section does not sufficiently address the limitations of the proposed approach. The paper focuses on the formulation and theoretical derivations but provides little reflection on potential weaknesses or failure cases. The practical use cases are also not deeply explored, leaving readers to infer the applicable scenarios and boundary conditions from the mathematical formulations.


Second, the paper could strengthen its mathematical justification regarding error propagation and convergence.
Third, the presentation is dense with formulas and algebraic derivations but lacks intuitive visual illustrations. A few schematic diagrams or graphical examples would greatly help readers quickly grasp the conceptual flow and the intuition behind the bilevel information gain framework at the beginning.

### Questions
There are several questions to help clarify the concerns outlined below:
1. The proposed method heavily relies on approximation techniques, including the variational lower bound and truncated Gaussian assumptions.
Is there any benchmark or ablation study for each of these approximation components?
It would be valuable to include small-scale demonstrations comparing each approximation against ground-truth results or showing quantitative errors.
For instance, while sampling from the Gaussian process posterior distribution $\Omega_B = \{\, y_f(x, \theta),\; y_g(x, \theta),\; f^*,\; g^*,\; x^*,\; \theta^* \,\}$ is indeed challenging, even a simple toy example could help clarify how the random Fourier feature (RFF) approximation behaves and how it affects the estimation of the bilevel mutual information.
2. The above issue may lead to another concern: since multiple approximations are stacked together, compounding approximation errors could occur.
Is there any theoretical assurance of convergence or empirical evidence demonstrating that these cumulative errors remain bounded or do not destabilize the optimization process?
3. In some experimental results, such as Fig. 2 (a), (c) and Fig. 3 (a), the regret curve of BLJES shows a sharp drop within the first 20 iterations, while the BILBO baseline occasionally exhibits similar behavior on other benchmarks.
This appears to be a consistent pattern of the optimizer rather than random variation.
Is there an explanation or intuition for why this phenomenon occurs across certain benchmarks?
I am happy to raise the score if my questions are addressed satisfactorily.
There are further suggestions for improving the presentation:
1. The limitations of this approach should be discussed. This helps the paper show the applicable range of the problem.
2. Visualization illustrations may improve the general ideas. Full of mathematical formulas requires more effort from viewers.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to study an open problem in bi-level optimisation that both the upper level and the lower level involve expensive black-box functions. The authors design a new and interesting method to solve it. They theoretically and empirically verify the proposed method.

### Strengths
The problem is well defined. The theory looks solid and rigorous, though I did not get time to check all details in the proofs.

### Weaknesses
The proposed problem is not well motivated. I would like to ask:

- What applications are aligned with the setting that both the upper level and the lower level involve expensive black-box functions?
- Do existing methods work in this setting? If no, why? What would happen if they are applied here?

The experiments are not strong enough. Only one relatively small real dataset is used - more datasets are required. For ablation study, the authors only present results on number of samplings. No code is provided.

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
2
