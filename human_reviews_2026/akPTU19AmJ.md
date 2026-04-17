# Sobolev Training of End-to-End Optimization Proxies

- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Optimization proxies—machine-learning models trained to approximate
the solution mapping of parametric optimization problems in a single
forward pass—offer dramatic reductions in inference time compared to
traditional iterative solvers.  This work investigates the integration
of solver sensitivities into such end-to-end proxies via a
Sobolev–training paradigm and does so in \emph{two distinct settings}:
(i) \emph{fully supervised} proxies, where exact solver outputs and
sensitivities are available, and (ii) \emph{self-supervised} proxies
that rely only on the objective and constraint structure of the
underlying optimization problem.  
By augmenting the standard training loss
with directional-derivative information extracted from the solver, the
proxy aligns both its predicted solutions \emph{and} local derivatives
with those of the optimizer.  Under Lipschitz-continuity
assumptions on the true solution mapping, matching
first-order sensitivities is shown to yield uniform approximation error
proportional to the training-set covering radius.  
Empirically, different impacts are observed in each studied setting.  
On three large Alternating Current
Optimal Power Flow benchmarks, supervised Sobolev training cuts mean-squared error
by up to 56 \% and the median worst-case constraint violation by up to
400 \% while keeping the optimality gap below 0.22 \%.  
For a mean–variance
portfolio task trained without labeled solutions, self-supervised Sobolev training
halves the average optimality gap in the medium-risk region
(i.e. standard deviation above $10\%$ of budget) and matches the baseline elsewhere.
Together, these results highlight Sobolev training—whether supervised or
self-supervised—as a path to fast, reliable surrogates for
safety-critical, large-scale optimization workloads.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper proposes Sobolev Training of End-to-End Optimization Proxies: neural networks that amortize parametric optimization by approximating the solution operator. The key idea is to supervise both values and solver sensitivities (first-order derivatives) via a sparsity-masked Sobolev loss.

### Strengths
Learning reliable optimization proxies with good gradients is highly relevant for nested decision pipelines and time-critical applications. The sparsity-masked Jacobian supervision is a practical and nontrivial twist.

### Weaknesses
The choice of mask density (e.g., 5–25%, often ~95% sparsity) is tuned empirically. There’s limited guidance or theory for which Jacobian entries to supervise beyond random sparsification.

The theoretical guarantees assume exact, dense Jacobians, but experiments use noisy, sparse, and occasionally invalid sensitivities.

Theoretically, matching derivatives should reduce sample complexity, but the paper provides no data-efficiency curves (performance vs. number of training samples).

The baseline coverage seems limited.

### Questions
See Weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Sobolev Training of End-to-End Optimization Proxies, a framework for learning differentiable surrogate models that approximate the solution map of parametric optimization problems. Instead of training solely on optimal decisions, the proposed method incorporates the Sobolev loss, which matches both function values and gradients, allowing the mapping function to achieve better convergence to the original landscape. In theory, the paper formalizes conditions under which the optimal solution mapping is differentiable and proves that it yields a quadratic improvement in approximation error. The experiments include AC Optimal Power Flow and Mean–Variance Portfolio Optimization, which demonstrate improved feasibility and generalization, although the numerical gains are sometimes modest.

### Strengths
The paper is conceptually original in combining Sobolev training with the learning of end-to-end optimization proxies. This bridges a clear gap between differentiable optimization and learning-based surrogate modeling. The idea of using solver sensitivities or implicit KKT derivatives as gradient supervision is elegant and well-motivated.

On the technical side, the theoretical analysis is rigorous and clearly articulated. The authors carefully state the regularity assumptions (LICQ, SOSC) and derive a clean result showing a quadratic improvement in approximation error for value-plus-gradient training. The connection between the Sobolev loss and the smoothness of the solution map is intuitive and theoretically grounded.

The sparse Jacobian masking and projection layer are practical engineering contributions that make Sobolev training computationally feasible.

Overall, the paper is clear, technically sound, and relevant to the growing field of learning to optimize.

### Weaknesses
While the paper is theoretically sound and conceptually elegant, its main limitation lies in the modest empirical improvement. The quantitative gains over value-based baselines are small, and the claimed advantage in improving feasibility is not clearly demonstrated. The authors emphasize feasibility as a key benefit, yet the current metrics only partially capture this aspect. If feasibility improvement is indeed the main contribution, the paper would benefit from more comprehensive metrics and visualizations, such as the proportion of feasible instances or the number of violated constraints. At present, the reported metrics have limited explanatory power.

Furthermore, if feasibility is considered the main contribution of the paper, the evaluation should include comparisons with other feasibility-oriented approaches, such as projection-based methods. In the Mean–Variance Portfolio Optimization experiment, the authors additionally apply a projection layer to enforce feasibility but do not report results without this layer or provide any ablation analysis. As a result, it is difficult to disentangle whether the observed improvements come from the Sobolev loss itself or from the feasibility projection. This omission weakens the central claim that Sobolev training inherently improves feasibility.

The experimental design is confusing and inconsistent across tasks. AC-OPF and portfolio optimization each have both supervised and self-supervised variants, but these are split between the main text and appendices without a unified comparison. A summary table comparing all four settings (task × training × projection × mask) would greatly clarify the experimental logic. Moreover, portfolio experiments include a feasibility projection layer while AC-OPF does not, making cross-task comparisons uneven.

### Questions
1. For self-supervised learning, how are the sensitivities or partial Jacobians obtained without ground-truth optimal labels?
2. nested-AD and higher-order derivatives are mentioned as very costly. Please report wall-clock and GPU memory vs. value-only training, and any stabilization tricks (such as mask).
3. How is the binary mask constructed?
4. Provide an ablation over mask density (e.g., 5/10/25/50%) and show its effect on efficiency and performance.
5. In the portfolio optimization task, both baselines use a projection layer, while the AC-OPF experiments do not. Could you report results without the projection layer and clarify why this asymmetry exists across tasks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a Sobolev training framework for learning neural optimization proxies that jointly match solution values and the solver’s geometric sensitivities. To improve scalability, a sparse masking strategy is proposed to reduce the memory overhead of full Jacobian supervision and mitigate gradient conflicts in large-scale constrained learning. The authors derive uniform approximation bounds for value-only, Jacobian-only, and joint Sobolev training, and demonstrate that the proposed method outperforms value-only matching in both supervised and self-supervised settings.

### Strengths
- Incorporating solver sensitivities into the loss provides a conceptually intuitive way to teach the network about local geometric structure. The proposed sparse masking strategy is simple, effective, and well-motivated: it reduces memory usage, mitigates gradient conflicts, and ablation studies demonstrate that high sparsity levels (keeping only 5–25% of entries) yield the best performance.

- Experimental results show consistent reductions in mean squared error and, notably, in worst-case infeasibility and outliers, which is particularly valuable for safety-critical applications.

- The paper is clearly written, well-organized, and self-contained.

### Weaknesses
- While the paper provides bounds for value-only, Jacobian-only, and joint Sobolev training individually, there is no theoretical guarantee that joint Sobolev training consistently achieves a smaller error gap than value-only training.

- Although Sobolev training substantially improves MSE and infeasibility metrics compared to the benchmark, the opposite trend is observed for the optimality gap. This suggests that Sobolev may produce solutions that are close to optimal in Euclidean space but not necessarily in the function space (e.g., in terms of surface or contour).

### Questions
- What is the core novelty of this work compared to traditional Sobolev training? Is the primary contribution the masking mechanism M? If the main innovation is limited to the masking strategy, it may be considered incremental, and its novelty could be questioned. It would be helpful to clarify how this approach meaningfully extends or improves upon prior Sobolev-based methods.

- Incorporating the gradient of the proxy into the loss function may necessitate computing the Hessian during model updates. This can become computationally expensive, particularly when the proxy has a complex architecture with a large number of parameters. The issue may be further exacerbated when the underlying solver itself is complex, as representing its behavior accurately would require an even larger and more expressive proxy.

- Empirically, the benchmark (Bench) achieves a significantly better optimality gap while potentially requiring lower training cost compared to Sobolev training. Given this, it is unclear under what circumstances one would prefer Sobolev over Bench in practice.

- Minor points: Figure 2 is not referenced in the main text. Additionally, in lines 308 and 322, the term “Model” should perhaps be replaced with “Figure.”

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
3

### Summary
The paper proposes a Sobolev training framework for solution prediction for optimization problems. Typically, when one trains a neural network to predict the optimal solution, it uses the solution value to guide the training. The framework studied in this paper also considers the accuracy of predicting the Jacobians of the solution, therefore the blackbox solution prediction function match both the solution and the local geometry of the solution space. 

The framework contributions some novel techniques such as Sparse Sobolev loss, which makes training more efficient and tractable. It also provides theoretical analysis on the error bounds.

Empirically, the framework is evaluated in two real-world domains to demonstrate its effectiveness.

### Strengths
1. This is one of the first works in ML for optimization that uses Solobev training techniques. I consider this as an innovation for the research field. The sparse masking technique is novel and practically impactful.
2. Theoretical analysis shows that it helps with reducing sample complexity and improve generalization.
3. Empirical evaluation on two real-world domain demonstrates the effectiveness of the framework.

### Weaknesses
1. Jacobian computation in the framework produces extra overhead during training. The authors discussed how the overhead could be reduced, but it is still unclear how it compares to the benchmark method.
2. The results for portfolio optimization is not promising from a practical point of view. Since most funds / asset management firms in real world operates under tight risk in their portfolio management. Overall, portfolio optimization is not the best use case for studying optimization proxies, since rebalance frequencies are typically quite low when the covariance matrix is assumed to be estimated over long horizons. 
3. Some experimental setups are not described in details such that the results could be reproducible. For example, how do you generate the instances for the MPS application? What are the hyperparameters used in training?

### Questions
1. How much overhead does your framework produce compared to the baseline during training?
2. What are the problem sizes of the MPS instances?

### Soundness
3

### Presentation
3

### Contribution
3
