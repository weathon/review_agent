# Cost-aware Stopping for Bayesian Optimization

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
In automated machine learning, scientific discovery, and other applications of Bayesian optimization, deciding when to stop evaluating expensive black-box functions in a cost-aware manner is an important but underexplored practical consideration. A natural performance metric for this purpose is the cost-adjusted simple regret, which captures the trade-off between solution quality and cumulative evaluation cost. While several heuristic or adaptive stopping rules have been proposed, they lack guarantees ensuring stopping before incurring excessive function evaluation costs. We propose a principled cost-aware stopping rule for Bayesian optimization that adapts to varying evaluation costs without heuristic tuning. Our rule is grounded in a theoretical connection to state-of-the-art cost-aware acquisition functions, namely the Pandora's Box Gittins Index (PBGI) and log expected improvement per cost (LogEIPC). We prove a theoretical guarantee bounding the expected cost-adjusted simple regret incurred by our stopping rule when paired with either acquisition function. Across synthetic and empirical tasks, including hyperparameter optimization and neural architecture size search, pairing our stopping rule with PBGI or LogEIPC usually matches or outperforms other acquisition-function--stopping-rule pairs in terms of cost-adjusted simple regret.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the optimal stopping criterion problem—that is, determining when to stop the iterations in Bayesian optimization.
This is an extremely important issue in Bayesian optimization, whose main objective is to perform efficient exploration.
In particular, the paper makes a novel contribution by explicitly incorporating the evaluation cost of the unknown function into both the acquisition function and the corresponding stopping rule.

### Strengths
- It is useful that the paper provides a “conservative yet practically operable” guarantees.

- Although only a limited number of existing methods explicitly take evaluation costs into account, this paper discusses the combination of acquisition functions and stopping criteria and evaluates them experimentally, which adds meaningful insight to the field.

- It is also commendable that the paper provides a theoretically grounded method for scaling between cost and objective values (i.e., for setting the parameter \lambda).
This allows principled adjustment of the trade-off, rather than relying on purely heuristic tuning.

### Weaknesses
- There is no guarantee of termination within a finite number of iterations.

- The Bayesian optimality of the “index + stopping” rule in the Pandora’s Box problem relies on the assumptions of independence and discreteness, and this work cannot inherit the propety; the proposed method provides no strict guarantee of optimal stopping or regret rate in the Gaussian Process (GP) setting.

### Questions
Theorem 2 shows that, with the proposed stopping rule (combined with PBGI or LogEIPC), the expected cost-adjusted simple regret never exceeds that of the trivial baseline that stops immediately after the first evaluation. This provides a “safety-side” guarantee—one never wastes additional evaluations unnecessarily—which is novel among Bayesian-optimization stopping rules that explicitly incorporate cost. However, since the method does not guarantee finite-time stopping(which is not so a big requirement for a stopping criterion), it remains unclear how meaningful this no-worse guarantee is in practice. Without an assurance of eventual termination or bounds on stopping time, the result may be theoretically weak: it ensures safety in expectation but not operational reliability. In my view, the implications of Theorem 2 therefore require further discussion, as I am not yet fully convinced of its practical significance.

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
This work explores adaptive stopping of bayesian optimization where obtaining data points occurs a cost. The main contribution is a cost-aware stopping rule with theoretical guarantees.

### Strengths
* Adaptive stopping of Bayesian optimization is an important problem with practical relevance. This work is the first to study this problem in the cost-aware setting, which most closely resembles many practical settings.
* The proposed stopping rule comes with theoretical guarantees
* The work tackles a range of different cost-aware settings, such as budget-constrained and cost-per-sample, as well as settings where evaluation costs are or are not known in advance.
* A comprehensive set of baselines is considered, including "Hindsight", which provides a lower bound on achievable performance.
* Existing methodology and their relation to this work is well covered
* Clear structure, writing, and well-crafted figures.

### Weaknesses
* The number and type of benchmarks considered are limited. Additional and more different benchmarks would be helpful. This is my main issue with this work.
* A summary plot showing, e.g., average ranks or average normalized regret would be helpful for the overall evaluation. This would also allow you to display more different evaluation settings in the main paper.
* Unclear what effect does the debounce strategy ("requiring the stopping rule to consistently indicate stopping over several consecutive iterations before stopping optimization.") have? Would other approaches benefit from this, too?
* Code to reproduce experiments or an implementation of the algorithm is not provided. Some experimental details and the main hyperparameter settings are given.

### Questions
* How were the specific LCBench tasks chosen?
* Minor suggestion: Do not write "is defined in (5)", but "is defined in Equation (5)"

### Soundness
2

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
3

### Summary
The paper proposes the PBGI/LogEIPC stopping rule for cost-aware Bayesian Optimization, which indicates when BO should be stopped to achieve the best trade-off between optimal values found and cost required. The proposed stopping rule is derived from the Pandora Box Gittins Index (PBGI) acquisition function. The authors also provide a connection between PBGI-based stopping rule with a stopping rule derived from Log Expected Improvement Per Cost (LogEIPC) acquisition function, demonstrating the equivalence of the two forms, hence the name PBGI/LogEIPC stopping rule. The stopping rule is theoretically guaranteed to maintain a bounded simple regret. The proposed stopping rule is evaluated against other stopping rules when applying to different acquisition functions (PBGI, EIPC, LCB and TS), on various synthetic and real-world benchmark problems.

### Strengths
-	The paper is well-written when explaining its methodology.
-	The work presents a stopping rule that is able to apply to a general BO algorithm.
-	There is a theoretical analysis to support the work.

### Weaknesses
1.	My primary concern lies in the novelty of the work. The concept of the proposed PBGI/LogEIPC stopping rule appears to be essentially identical to the PBGI acquisition function introduced by Xie et al. (2024). In particular, the stopping rule directly uses the same decision criterion that motivated the PBGI acquisition function—namely, determining whether to evaluate the new potential candidate point or to stop and accept the current best observation (see Sec. 3.2, Xie et al., 2024). This work largely reformulates this existing principle, expressing the same condition in the form of an explicit stopping rule rather than an acquisition function.
2.	Furthermore, the empirical results indicate that the proposed stopping rule performs well primarily when used together with PBGI, but not when combined with other acquisition functions (AFs). This outcome suggests that the rule is specifically tailored to the PBGI AF and does not generalize effectively to other AFs in BO. As such, the paper does not substantially extend the prior work, since Xie et al. (2024) already evaluated the PBGI framework, which implicitly captures this stopping behaviour as part of its policy. Therefore, the contribution of the current work appears incremental, mainly reiterating a property already discussed in the original PBGI study rather than introducing a broadly applicable stopping strategy.

### Questions
1.	As I might have missed some details, can the authors explain more about the novelty of this work? 
2.	In the PBGI work, there are many acquisition functions to benchmark, e.g., KG, MES, MSEI, etc. Can the proposed stopping rule work with them? How was the performance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a stopping criterion for Bayesian optimization to enable efficient exploration in black-box function optimization problems where the evaluation cost varies across input points. The authors aim to suppress costly function evaluations while maintaining strong optimization performance, and they construct a new stopping rule based on existing acquisition functions used in cost-sensitive Bayesian optimization, PBGI and its equivalent LogEIPC. The proposed criterion adaptively determines whether to terminate the optimization process according to changes in the acquisition function value, thereby preventing unnecessary and expensive explorations in practice. Furthermore, numerical experiments on simulation and benchmark problems demonstrate that the proposed stopping method successfully avoids redundant high-cost evaluations while achieving optimization performance comparable to or better than existing approaches.

### Strengths
- This study is significant in that it clearly formalizes the previously ambiguous stopping condition in cost-sensitive Bayesian optimization, where the evaluation cost varies across input points. The proposed stopping criterion determines when to terminate the optimization based on the theoretical properties of the acquisition function, thereby preventing excessive exploration and unnecessary computational costs while achieving efficient and stable optimization.

- The proposed method is also notable for its application to AutoML tasks in a cloud environment, where its effectiveness is validated under practical conditions involving variable evaluation costs. Through these experiments, the study demonstrates that the proposed approach is beneficial for large-scale and high-cost real-world optimization problems, showing not only its theoretical contribution but also its high practical value.

### Weaknesses
- The meaning and derivation process of Equation (5) are not clearly explained in the text, making it extremely difficult to comprehend. Although the study relies heavily on the work of Xie et al., it lacks a self-contained explanation within the paper, which is problematic. Additional clarification should be provided so that readers can understand the content without referring to external sources.

- The theoretical analysis is based on an overly simplified assumption of a function with a constant mean, which does not adequately capture the complexity of real-world Bayesian optimization. This simplification undermines the generality and practical significance of the analytical results, creating a gap between the theoretical discussion and realistic applications.

- The proposed framework is highly dependent on the approach of Xie et al., and it shows limited applicability to acquisition functions other than PBGI and LogEIPC. Consequently, the generality and extensibility of the method are restricted, making it insufficient to claim its effectiveness across Bayesian optimization methods in general.

### Questions
- The definition and meaning of Equation (5) are extremely unclear. It is not evident whether \alpha_t^{\mathrm{PBGI}} represents the acquisition function itself or some other quantity. Moreover, the interpretation of the term involving (g) is insufficiently explained. In the upper condition part, the scales of EI and the cost c(x) appear to differ, making the interpretation of the entire equation ambiguous. Additionally, in the lower condition part, it is unclear whether f(x) denotes the objective function itself. If so, I completely do not understand why it can be used as acquisition function.

- The theoretical analysis in this paper focuses on a function with a constant mean, but this assumption seems far away from realistic Bayesian optimization settings. It should be clarified to what extent this simplified analysis holds practical meaning and validity for real-world optimization problems, and how the authors view the generalizability of their theoretical results.

- The proposed stopping criterion appears to be specifically designed for PBGI (or LogEIPC). It is unclear whether the insights and theoretical findings of this study can be applied, in some way, to other acquisition functions. If so, the applicable scope and required conditions should be explicitly discussed.

### Soundness
1

### Presentation
1

### Contribution
2
