# Physics-Informed Inductive Biases for Voltage Prediction in Distribution Grids

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Voltage prediction in distribution grids is a critical yet difficult task for maintaining power system stability. Machine learning approaches, particularly Graph Neural Networks (GNNs), offer significant speedups but suffer from poor generalization when trained on limited or incomplete data. In this work, we systematically investigate the role of inductive biases in improving a model's ability to reliably learn power flow. Specifically, we evaluate three physics-informed strategies: (i) power-flow-constrained loss functions, (ii) complex-valued neural networks, and (iii) residual-based task reformulation. Using the ENGAGE dataset, which spans multiple low- and medium-voltage grid configurations, we conduct controlled experiments to isolate the effect of each inductive bias and assess both standard predictive performance and out-of-distribution generalization. Our study provides practical insights into which model assumptions most effectively guide learning for reliable and efficient voltage prediction in modern distribution networks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper studies three ways to include “physics” into GNN voltage predictors for distribution grids: (i) fine-tuning with a physics loss (power balance equation), (ii) using complex-valued GNNs to represent voltages naturally, and (iii) predicting residuals relative to a baseline solver. The authors evaluate these on a test suite derived from SimBench and report that physics-loss helps magnitude prediction, where complex nets give higher angle accuracy, and residuals are mixed. They claim these are “inductive biases” that improve OOD generalization.

### Strengths
The papers focus is on OOD problem, which is very relevant for the distribution grids as topological changes occur frequently due to line switchings in distribution grids. Plus, ENGAGE dataset is claimed to be having multi-grid testbed rather than the single feeder which represents practical conditions.  Writeup is also reasonably clear.

### Weaknesses
Major Weaknesses: 
1. **No conceptual novelty:** all three ways to put physics into the problem are already known as cited by authors. Presenting them together is more of an engineering comparison, not a new inductive bias theory or a fundamentally new method. The paper’s framing as discovering a new “inductive bias” is kind of confusing and looks a bit overstating the claim; they are conflate architecture choice (a true inductive bias reduction thing) with training regularizers and output reparameterizations (which are optimization/design choices). 
2. **Methodological Error:** DC power flow assumes voltage magnitudes are fixed (typically 1.0 p.u.) and linearizes only active power as a function of voltage angles (See DC Power Flow Revisied by B. Stott et.al.). As it does not model or predict voltage magnitudes, using DCPF as a baseline for magnitude RMSE is therefore meaningless (it either trivially treats |V|=1 and measures deviation from 1.0, or it applies an ad-hoc, unsupported hack to derive magnitudes). There is a very very rich theory of power flow approximations which authors overlooked completely (Dan Molzahn et.al A Survey of Relaxations and Approximations of the Power Flow Equations). Authors must justify or remove that comparisons as they are not valid. 

The statement : "a commonly used solving method for voltage prediction: DC Power Flow" is incorrect. 

**2(a)**. Further, as per the description of ENGAGE in paper, DCPF cannot be applied even to predict the voltage angles on authors test cases.  For low-voltage and many distribution feeders the R/X ratio is high, so the common DC assumptions (negligible R, small angle differences, and that reactive flows can be ignored) break down. Using DC approximations as a performance yardstick for distribution/low-voltage problems is therefore inappropriate unless the authors explicitly justify or quantify the DC error in those regimes. Distribution literature offers linear models suited to voltage magnitude estimation (DistFlow / LinDistFlow), which are a more appropriate classical baseline. The authors reference that linear methods are unreliable in distribution grids, but do not quantify or show the DC vs AC discrepancy across the ENGAGE grids. 

3. **Missing Baselines:**  There is a very large volume of work on power flow predictions. The paper compares against a “baseline GNN” and its variants, but omits credible DNN baselines and Non-DNN Gaussian Process regression (and sparse GPs),  and very importantly linearized analytic surrogates (LinDistFlow, parameterized linear models).  GPs in particular are an established surrogate for voltage mapping and probabilistic power flow; they often perform strongly in low-data regimes and give calibrated uncertainties — similar to the setting the authors claim to improve. (See B. Tan et. al. Gaussian processes in power systems: Techniques, applications, and future works).

### Questions
1. How does complex-valued DNN scale? What are the sizes of networks in the ENGAGE Dataset along with other electrical properties like R/X ratio? 
2. Can formal justification be provided that how all three components presented reduce the hypothesis space or inductive bias? 
3. How do you distinguish between improved optimization (better loss landscape) and genuine inductive bias (systematic generalization) in your results? 
4. Why do you fine-tune for only 20 epochs with the physics-informed loss? Was this empirically optimal? How about using Primal Dual kind of methods? (I know they are generally designed for OPF but the constraint of power balance is same). 
5. Can you test on PGLib cases? As your method doesn't take any network specific assumption, will it work for large scale mesh grids? 
6. How exactly did you compute voltage-magnitude RMSE for the DC power-flow baseline? Is DCPF formulation applicable to your LV grid test cases? 
7. Some reported errors are <10⁻³ p.u. or <0.002°. Are such differences meaningful given sensor precision and modeling uncertainty? What error is a good error for power flow problem?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how different physics-informed inductive biases influence the performance and generalization of graph-based models for voltage prediction in electrical distribution grids.  The authors focus on three types of inductive biases, Power-flow-constrained loss functions, Complex-valued neural networks, Residual-based task reformulation. The study measures both in-distribution and out-of-distribution (OOD) generalization via leave-one-grid-out experiments. Results show that the physics-informed loss variant achieves the most stable voltage-magnitude prediction and lowest variability across unseen topologies.

### Strengths
- The paper provides a well-controlled setup: all models share identical GNN architectures, training schedules, and datasets, allowing fair isolation of each inductive bias’s effect.
- The paper is well organized and provides concrete implementation details, including formulas for the power-flow loss, CVNN layer definitions, and hyperparameters.

### Weaknesses
- The study mainly benchmarks three established strategies. None introduces a new algorithmic formulation, architecture, or theoretical insight. The contribution is primarily empirical rather than methodological.
- The study evaluates all inductive biases on a single GraphConv backbone. However, power-flow learning performance can vary substantially across graph architectures (e.g., GCN, GAT, Graph Transformers). Without cross-architecture validation, it is difficult to attribute improvements to the inductive bias itself rather than to properties of GraphConv.
- Experimental results should perform a more comprehensive comparison, like model capacity, convergence speed, and inference efficiency. Compared to different archs, even traditional methods.
- While the paper claims to study “physics-informed inductive biases,” the bias is imposed only at the loss level or representation level, not within the GNN’s message-passing structure. There is no attempt to embed Kirchhoff’s laws, admittance matrices, or nodal couplings directly into the propagation mechanism.
- The ENGAGE dataset is derived solely from SimBench; it is unclear how many unique topologies exist or how dissimilar the held-out grids truly are. A stronger test would involve training on small IEEE systems (14-bus, 39-bus) and evaluating on larger ones (57-, 118-bus)** to demonstrate scale generalization.
- The paper would benefit greatly from schematic diagrams showing the overall experimental setup, the flow of each inductive bias, or a qualitative visualization of predicted vs. true voltage profiles.
- Large sections of the paper read as tutorial material. Yet the study lacks a strong discussion on how the proposed findings translate to real distribution-grid operations

### Questions
Personally, the paper primarily benchmarks three established inductive-bias strategies, Please authors clarify what the core methodological contribution or new insight is beyond this comparative study.

### Soundness
2

### Presentation
3

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
This paper evaluates the impact of three different inductive biases for machine learning and AC power flow: physics-informed losses, complex-valued neural networks, and residual predictions. Experiments are conducted on several low- and medium-voltage distribution grids.

### Strengths
* The paper considers distribution grids, which have received less attention than transmission grids in the literature. The distinction is relevant, as distribution grids often display different topological structure (e.g radial vs meshed) and different behavior (e.g. 3-phased unbalanced, large deviations from nominal voltage) compared to transmission grid
* In a literature of mostly isolated works, this paper identifies three building blocks and evaluates each one's impact on training and performance.
* Of the 3 inductive biases, complex-valued neural-networks is the one I have seen the least used in the literature on ML for AC power flows

### Weaknesses
* The paper provides very few mathematical details (there are a total of 7 equations in the paper), which would have been especially valuable for understanding the experiment setup
* The various inductive biases are not new, however the paper does a good job of motivating and evaluating each of them
* The numerical experiments only consider one variant per inductive bias, which leaves the question of whether combining them may result in improved performance. Additional experiments that combine them would be valuable

### Questions
* Please provide more information on numerical experiment settings
* Please conduct additional experiments that combine two or three inductive biases
* Please provide a more comprehensive mathematical framework, including: statement of power flow equations, description of input and output features, and any information related to experiment settings

### Soundness
3

### Presentation
3

### Contribution
2
