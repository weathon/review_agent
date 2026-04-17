# Combinatorial Bandit Bayesian Optimization for Tensor Outputs

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6, 4

## Abstract
Bayesian optimization (BO) has been widely used to optimize expensive and black-box functions across various domains. However, existing BO methods have not addressed tensor-output functions. To fill this gap, we propose a novel tensor-output BO framework. Specifically, we first introduce a tensor-output Gaussian process (TOGP) with two classes of tensor-output kernels as a surrogate model of the tensor-output function, which can effectively capture the structural dependencies within the tensor. Based on it, we develop an upper confidence bound (UCB) acquisition function to select query points. Furthermore, we introduce a more practical and challenging problem setting, termed combinatorial bandit Bayesian optimization (CBBO), where only a subset of the tensor outputs can be selected to contribute to the objective. To tackle this, we propose a tensor-output CBBO method, which extends TOGP to handle partially observed tensor outputs, and accordingly design a novel combinatorial multi-arm bandit-UCB2 (CMAB-UCB2) criterion to sequentially select both the query points and the output subset. We establish theoretical regret bounds for both methods, guaranteeing sublinear regret. Extensive experiments on synthetic and real-world datasets demonstrate the superiority of our methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper introduces Tensor-Output Bayesian Optimization, a framework for optimizing black-box functions with tensor-valued outputs using a tensor-output Gaussian process. It proposes both separable and non-separable tensor-output kernels that capture input-dependent cross-mode correlations, and develops a UCB acquisition tailored to scalarized tensor objectives. The work further extends to a combinatorial setting where only a subset of tensor entries contributes to the objective, using a partially observed TOGP and a CMAB-UCB2 criterion for joint input and subset selection. Theoretical sublinear regret bounds are provided, and experiments on synthetic and real tasks show consistent improvements over vectorized multi-output GP baselines.

### Strengths
Modeling tensor outputs directly with non-separable, input-dependent kernels is a clear step beyond vectorized MOBO, enabling richer cross-mode structure to be exploited. The CBBO formulation and the PTOGP-based acquisition rule are principled. The regret analyses leverage information gain and cover both full and partial observation regimes, giving the method theoretical footing. Empirical results are broad, with synthetic and multiple real datasets, and show strong gains over standard multi-task and multivariate GP baselines. The appendices give thorough algorithmic and complexity details.

### Weaknesses
The computational complexity $O(n^3 T^3 log(n))$ of TOBO is cubic in both n **and** tensor size, which may severely limit practical scale. 

The scalarization operators $L_f$ and $H_f$ are central in practice, but their choices, sensitivity, and domain-specific instantiations are only lightly discussed. 

Baselines omit state-of-the-art tensor GP surrogates with Kronecker structure and non-separable multi-output kernels beyond the selected MTGP/MVGP/MLGP such as proposed in "Scalable High-Order Gaussian Process Regression" (line 91).

### Questions
- How are $L_f$ and $H_f$ instantiated across tasks, and how sensitive are results to these choices or to non-Lipschitz scalarizations encountered in practice?
- How are tensor ranks chosen for the core tensors (line 208), and what is the performance/overfitting trade-off as ranks vary under separable vs non-separable kernels?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates optimization problems involving tensor-output objective functions, where each function evaluation returns a multi-dimensional tensor instead of a vector, as in standard multi-output settings. The authors introduce a Tensor-Output Gaussian Process (TOGP) that captures both dependencies across inputs and structured correlations within and between tensor modes of outputs. The proposed TOGP generalizes and introduces two types of kernels: a separable kernel, which assumes constant correlations across tensor modes, and a non-separable kernel, which allows correlations within and across tensor modes to vary with the inputs. 

The authors further apply the TOGP to Bayesian Optimization (BO) in two scenarios. In the fully observed setting (TOBO), all tensor entries are available at each iteration, and optimization proceeds via an Upper Confidence Bound acquisition function derived from the TOGP posterior. In the partially observed setting (TOCBBO), only a subset of tensor entries is observed per iteration, transforming the problem into a Combinatorial Bandit Bayesian Optimization (CBBO) framework. Here, the algorithm must select both the next input and the subset of tensor entries (the super-arm) to observe. The paper provides theoretical guarantees, including proofs of kernel validity and sublinear regret bounds for both TOBO and TOCBBO. Extensive evaluations include three synthetic and four real-world benchmark problems to compare against many multi-output GP baselines.

### Strengths
-	The idea is sound, which generalizes the well-known LMC model in multi-output (vector-output) GP into the tensor-input GPs.
-	The paper is clearly written, with theoretical proof for the kernel validity and regret bound under certain assumptions.
-	The empirical performance shows that TOGP has potential in modelling tensor-output functions, when compared to vector-output GPs.

### Weaknesses
Some experiments are missing: 
- Lack of single-objective BO baselines: Since the target problem settings rely on a scalarization operator (L_f or H_f in the CBBO case) to convert tensor outputs into scalar objectives, standard single-objective BO methods (e.g., single-task GP combined with a UCB acquisition function) should, in principle, operate normally. The paper would be stronger if it compared TOGP-based optimization with such single-task GP baselines to demonstrate the benefit of modelling complex tensor dependencies rather than relying on a simpler approach.
- Lack of single-objective BO baselines in the Bandit BO setting: Similarly, in the Bandit BO setting, the presence of the scalarization operator (L_f or H_f) allows single-objective BO algorithms to be applied directly. Prior works (e.g., Nguyen et al., 2020; Ru et al., 2020; Huang et al., 2022) already use this setup effectively, and including these as baselines would provide a fairer and more comprehensive comparison for TOCBBO.
- Lack of multi-objective BO baselines: Because TOGP inherently models multiple correlated outputs, it would be valuable to assess its performance under a multi-objective optimization framework. Comparing against the SOTA MOBO methods using metrics such as the hypervolume indicator (e.g., Maddox et al., 2021; Daulton et al., 2022) would further complement the paper.

### Questions
1. What is the runtime trade-off of the proposed TOGP? Since TOGP models complex interactions across multiple tensor modes, it likely incurs higher computational costs than both single-task GPs and multi-output GPs based on the LMC, due to the larger number of hyperparameters involved. Additionally, given that the problem settings already require a scalarization operator (L_f or H_f) to map tensor outputs to scalar objectives, which allows single-task GP to work just fine, it would be useful to analyze the accuracy–efficiency trade-off - that is, how much performance gain the tensor-output model provides compared to the reduced computational cost relative to a simpler single-task GP baseline.
2. Does the proposed TOBO algorithm account for the diversity of solutions? Given that the target problems have multiple outputs, some of which may be conflicting, can TOBO handle such cases, identify a set of solutions that satisfy different objectives or trade-offs? This is a common consideration in MOBO literature, where the goal is not only to find the best single solution but also to capture the Pareto diversity among competing objectives.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents two variants of GP-UCB for BO of tensor-valued objectives. Both model the objective using a tensor-output GP (TOGP) with the goal of capturing additional structure present in the tensor representation, with a kernel that is somewhat novel to capture correlation between modes. The first algorithm uses scalarization - basically a weighted sum of the ouputs - while the second does the same on a subset of outputs (partial observations) at each round. Convergence rates are analysed to show sub-linear regret, and some experimental results are presented.

### Strengths
The major strength/novelty of this paper appears to be the TOGP model, in particular the kernel used. And the kernel used for this GP is novel, at least to my understanding.

That said, at least the first algorithm presented does not seem particularly novel to me: it is basically a scalarizing multi-objective BO algorithm. Each element of the tensor represents an objective, and a linearization operator takes the place of scalarization. The second algorithm is arguably more novel, as it assumes partial observations of the tensor components at each iteration; however, past this, it again boils down to a scalarized MOBO formulation.

Experiments seem to have improved since last time I reviewed this paper, and experimental result seem sound, so I am inclined toward accepting.

### Weaknesses
See above

### Questions
Relevant questions were answered last time I reviewed this paper.

### Soundness
3

### Presentation
2

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
- The paper introduces Tensor-Output Gaussian Processes (TOGPs) that model structured tensor-valued outputs with novel separable and non-separable kernels capturing input-dependent correlations.
- Building on TOGP, the authors propose Tensor-Output Bayesian Optimization (TOBO) using a UCB acquisition function for efficient optimization of expensive black-box tensor-output functions.
- They further extend the framework to a Combinatorial Bandit Bayesian Optimization (TOCBBO) setting where only a subset of tensor outputs can be observed.
- Both TOBO and TOCBBO are theoretically analyzed, with sublinear regret bounds.
- Extensive synthetic and real-world experiments demonstrate that the proposed methods outperform several baselines in predictive accuracy and optimization effectiveness.

### Strengths
- The manuscript is well written with polished language that enhances readability and effectively communicates complex technical ideas.
- It introduces a novel Tensor-Output Gaussian Process (TOGP) framework with separable and non-separable kernels.
- The framework extends Bayesian Optimization to handle tensor-valued functions.

### Weaknesses
- The paper's main novelty lies in the Gaussian Process modeling, while the Bayesian Optimization layer mainly adapts standard UCB and CMAB-UCB2 frameworks without introducing a fundamentally new acquisition strategy.
- The experimental evaluation emphasizes predictive accuracy metrics (MSE, MAE, NLL) more than optimization-oriented ones, creating a mismatch with the paper’s stated BO focus.
- The paper lacks runtime or computational efficiency results, reporting only asymptotic complexity analyses without empirical timing comparisons.

### Questions
- The paper's primary contribution appears to lie in the Gaussian Process modeling rather than in the Bayesian Optimization framework itself. Could the authors clarify whether the main contribution is intended to advance GP modeling or BO methodology?
- Given that ICLR allows for a revision, the authors might be able to revise or extend the work to more clearly strengthen its contribution to the BO literature?
- While theoretical computational complexities are analyzed, no empirical runtime or efficiency results are provided. Could the authors report or estimate the practical computational overhead of TOGP and TOCBBO compared with existing multi-output GP or BO methods?
- How does the improved predictive accuracy of TOGP quantitatively translate into better optimization results such as faster regret decay or fewer evaluations? Have the authors conducted any ablation or correlation analysis to demonstrate this link?
- Could the authors consider extending their framework to more meaningful tensor structures, such as spatiotemporal tensors that capture both spatial and temporal dependencies?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel Bayesian optimization framework for systems with tensor-valued outputs, a setting previously unaddressed in the BO literature. The core idea involves a Tensor-output Gaussian Process (TOGP) with two new kernel classes that capture complex, input-dependent correlations within the tensor structure, avoiding the limitations of simple vectorization. The authors also introduce a more challenging Combinatorial Bandit BO (CBBO) setting where only a subset of tensor elements contributes to the objective, and propose TOCBBO to handle it via a CMAB-UCB2 acquisition function that decouples continuous input selection from combinatorial super-arm selection. Theoretical guarantees are provided for the sub-linear regret of both methods. Extensive synthetic and real-world experiments demonstrate superior performance over baselines that use standard multi-output GPs.

### Strengths
1. This paper introduces the first Bayesian optimization framework specifically designed for tensor-output systems, developing novel tensor-output Gaussian processes that capture complex structural dependencies through specialized kernel designs.
2. This paper provides strong theoretical guarantees, establishing sublinear regret bounds for both the standard tensor-output setting and the more challenging combinatorial bandit scenario with partial observations.
3. The proposed methods show robust performance and consistently outperform existing approaches across diverse real-world applications.

### Weaknesses
1. This paper does not provide computational complexity comparisons with existing methods. Additionally, the high O(n^3T^3) complexity of TOGP itself represents a significant limitation, as strategies for scaling to large tensor outputs are not discussed.
2. The theoretical analysis relies on the assumption that the true function is a sample from the proposed TOGP. There is little discussion of whether this holds in practice or how to assess its validity in real-world applications.
3. A more detailed explanation of Equation (3) in line 161 would be helpful. For instance, it would be useful to clarify whether the kernel function K(x,x′) is designed to capture both similarities in the input space X and correlations among the internal elements of the output tensor.
4. One concern is that the modeling and algorithm design are claimed to target tensor-type data, yet the actual formulations (e.g., Equations (4), (5), and (8)) are carried out at the matrix level. Would this process compromise or distort the intrinsic tensor structure?

### Questions
See the weaknesses part above.

### Soundness
2

### Presentation
2

### Contribution
2
