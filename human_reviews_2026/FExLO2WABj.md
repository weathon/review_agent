# FedEM: A Privacy-Preserving Framework for Concurrent Utility Preservation in Federated Learning

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Federated Learning (FL) enables collaborative model training across distributed clients without sharing local data, thus reducing privacy risks in decentralized systems.  However, the exposure of gradients during training can lead to significant privacy leakage, particularly under gradient inversion attacks.  To address this issue, we propose Federated Error Minimization (FedEM), an input-level defense framework that injects learnable perturbations into client data and jointly optimizes both the model and the perturbation generator.  Unlike traditional Differential Privacy methods that modify gradients, FedEM achieves a stricter privacy-utility trade-off by perturbing inputs directly. We validate the effectiveness of FedEM through extensive experiments on benchmark datasets.  For example, on MNIST, FedEM achieves only a 0.08\% decrease in accuracy compared to FedSGD, while significantly improving privacy metrics, with MSE improved by 46.2\% and SSIM reduced by 69.3\%.  These results demonstrate that FedEM effectively mitigates gradient leakage attacks with minimal utility loss, providing a robust and scalable solution for privacy-preserving federated learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Federated Error Minimization (FedEM), a framework to defend against gradient leakage attacks (GLAs) in Federated Learning (FL). The core idea is to move away from traditional defenses that perturb gradients, such as those based on Differential Privacy, and instead introduce learnable perturbations directly into the clients' input data. The framework is inspired by error minimization attacks and formulates the defense as a joint optimization problem, where both the model parameters and the input perturbations are optimized to minimize the training loss. The authors conduct experiments on several image and text datasets, showing that FedEM can mitigate GLAs while maintaining higher model utility compared to several baseline methods.

### Strengths
- The paper proposes a defense mechanism that operates on the input data level rather than the more common gradient level.

- The paper tackles the issue of gradient leakage attacks in federated learning, which is a critical and highly relevant challenge in the field of privacy-preserving machine learning.

### Weaknesses
- The central idea of the paper is to use a min-min optimization to generate input perturbations. It is a direct application of the Error Minimization Attack (EMA) framework. The primary contribution seems to be repurposing this attack mechanism as a privacy defense. The technical leap here feels incremental rather than foundational. The work does not introduce a fundamentally new mechanism or insight.

- The proposed client-side training involves a loop where the perturbation vector and a temporary local model are updated for N steps for each batch. This introduces significant computational overhead compared to standard FL training. The impact on the system's feasibility is not discussed.

- In the experiments section, the authors should plot the full privacy-utility trade-off curves to show the effectiveness of the method.

- The defense is primarily evaluated against one main attack (Invert-Grad), and its robustness against a wider range of adaptive attacks is not explored.

- Table 1 should include the results of no defense mechanisms, along with the results of having defense mechanisms.

- In Table 1, the performance of FedEM is on par with the other methods, with any differences falling within a margin of 0.001 or 0.01. This suggests there is no significant distinction in their effectiveness.

### Questions
See comments.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FedEM, a federated learning defense against gradient leakage attacks that perturbs client data directly rather than modifying gradients. The method jointly optimizes model parameters and learnable perturbations through dual-step optimization. Experiments on image datasets (MNIST, CIFAR-10/100, Tiny-ImageNet) and text tasks (CoLA, SST-2) show superior privacy-utility trade-offs compared to differential privacy baselines, achieving minimal accuracy loss (0.08% on MNIST) while significantly improving privacy metrics (46.2% MSE improvement, 69.3% SSIM reduction).

### Strengths
**1. Novel Defense Paradigm:** Input-space perturbation offers a fresh alternative to gradient-space defenses, with creative adaptation of error minimization concepts to privacy protection.

**2. Strong Empirical Results:** Consistently achieves better privacy-utility trade-offs than DP baselines across diverse datasets. Results on complex benchmarks (CIFAR-100, Tiny-ImageNet) demonstrate robustness.

**3. Comprehensive Evaluation:** Thorough experiments including multiple datasets, extensive ablations on perturbation magnitude (Tables 11-13), scalability to 50 clients, five privacy metrics, and generalization to text domains.

**4. Practical Insights:** Addresses perturbation lower bounds (Section 4.7), convergence behavior, and provides useful visualizations of perturbation evolution.

### Weaknesses
**1. Limited Theoretical Novelty:** Theorem 1 follows standard convergence analysis and Lemmas 1-2 are borrowed from Zhang et al. (2024). The paper provides only empirical privacy metrics without formal guarantees such as differential privacy bounds or reconstruction error guarantees.

**2. Computational Overhead Underexplored:** The method requires N=15 inner steps per batch, significantly increasing computational cost. While Figure 6 shows time increases, the paper lacks wall-clock time comparisons with baselines, memory overhead analysis, and practical guidance on selecting N for different scenarios.

**3. Incomplete Evaluation:** Privacy metrics focus primarily on the first training round with limited later-round analysis. The evaluation lacks adaptive attacks where adversaries know the defense, comparisons with gradient compression methods and unlearnable examples, statistical significance testing for most results, and realistic large-scale settings (default 4 clients).

**4. Clarity Issues:** Algorithm 1 is dense and the relationship between θ and θ_u needs clearer explanation. The claim of "stricter privacy-utility trade-off" in the abstract is not formally defined, and details on perturbation initialization and cross-round persistence are missing.

**5. Missing Analysis:** The paper provides limited characterization of what makes learned perturbations effective for privacy protection. There is no analysis of perturbation transferability across clients or systematic sensitivity analysis for key hyperparameters (αu, ρ_min, ρ_max).

### Questions
1. I'm concerned about adaptive attacks. What happens if an adversary knows you're using FedEM and specifically optimizes their reconstruction attack to account for the perturbations? Have you tested against such attacks?
2. The paper provides empirical privacy metrics, but can you offer any formal privacy guarantees? For instance, could you bound the reconstruction error or provide differential privacy guarantees under certain assumptions?
3. Most privacy results are from the first training round (E1). I'm curious whether privacy degrades as training progresses and the model becomes more accurate. What do the privacy metrics look like at rounds 10, 20, or 30?
4. It's not entirely clear from Algorithm 1—are the perturbations δk reinitialized at the start of each round, or do they persist and evolve across rounds? If they persist, how does this affect privacy guarantees over multiple rounds?
5. How sensitive is FedEM to the choice of hyperparameters? Specifically, how should practitioners choose N (perturbation steps), αu (perturbation learning rate), and the bounds [ρ_min, ρ_max] for their specific use case?
6. All experiments assume equal data partitioning across clients. How does FedEM perform under more realistic non-IID settings where clients have heterogeneous data distributions?
7. You mention connections to unlearnable examples, which also use data perturbations. Could you clarify the relationship and perhaps provide an empirical comparison? What are the key differences in how perturbations are learned?

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
This paper introduces Federated Error Minimization (FedEM), a defense framework for federated learning against gradient inversion attacks. The main difference that sets FedEM apart from prior defenses based on Differential Privacy (DP) is that instead of perturbing the gradients, FedEM introduces perturbations directly to each client's input. Each client learns their optimal perturbation and model in parallel, where the perturbations are restricted to have their L2 norms in a predefined interval. The authors evaluate FedEM experiments on benchmark datasets (including both text and image baed tasks) and compare its performance in terms of utility and privacy to other methods. They also provide a theoretical guarantee for FedEM convergence (to a neighborhood of stationary points) under smoothness and bounded variance assumptions. The main argument of the paper is that FedEm achieves better privacy-utility trade-off compared to prior methods, as demonstrated by the experiments.

### Strengths
1) The paper is very-well written and easy to comprehend. This is in part due to the main idea behind FedEM being relatively easy to convey. Still, the authors do a great job explaining what sets FedEM apart from the prior methods. The sections of the paper are organized and nicely flow from one to another. 

2) The experiments in the paper, as the authors correctly claim, are pretty extensive and covers a variety of contexts to demonstrate their robustness, including different client sizes (4 vs. 50), different attacks (Inverting-Grad v. GIAS) and different parameters for FedEM and other methods (sec 4.5 & 4.7). Several question I had while reading each experiment were subsequently answered by later experiments, which is always a positive sign. 

3) Despite limitations imposed by the page limit, the authors still manage to fit in important discussions that build intuition on what FedEM actually does. For example, Figure 4 and the discussion pertaining to it (end of section 4.2) is very helpful.

### Weaknesses
1) Despite generally outperforming the DP-based approaches in utility and privacy, the marginal contribution of FedEM in each benchmark is still relatively low. While the author's efforts in including experiments with different datasets and parameters deserve praise, the fact that FedEM achieves only a slightly better performance-privacy tradeoff than the other methods brings doubt into whether this trend would be continued in other settings. Overall, the narrative provided in the introduction presented the shortcomings of encryption- and DP- based techniques as the computational unscability and degrading model performance, but FedEM is not substantially different from the second category of methods. In this sense, the contributions of the method are relatively limited. While gradual improvements are of course important, finding at least one setting (in addition to those presented) in which FedEM overwhelmingly outperforms anything else would have strengthened the authors' point. 

2) Overall, there is almost no discussion about the limitations/shortcomings of FedEM. For example, taking the gradient with respect to the perturbations can become increasingly difficulty with increasing input complexity, which may introduce complications in settings with higher-dimensional input formats that perhaps do not arise in the other methods. Further, while the method is tested against classical attacks, can there be new attacks designed specifically for FedEM? (E.g., if the adversary knows the minimum/maximum perturbation employed by the clients, can they incorporate this to their optimization objective, equation (2), accordingly?). Finally, the fact that relationship between privacy strength and noise magnitude is not strictly monotonic can give rise to complications when optimizing for the noise radius, whereas with a more straightforward relationship (as observed in other methods), finding the desired point in the accuracy-privacy trade-off can be more straightforward. While the method of course cannot defeat all such limitations, the paper should at least discuss some of them. 

3) Some claims in the paper are more anecdotal and not justified by any results/citations. E.g., when talking about membership/property inference attacks: "However, with the development of defense strategies and a reassessment of their practical impact, recent research has shifted toward a more direct and severe threat: gradient leakage attacks". 


Minor:
- In algorithm 1, why us theta_u part of the input?  
- While potentially very helpful, Figure 1 can be a bit improved. Currently it is not super easy to read / its organization is a bit confusing
- Client heterogeneity is not defined in Theorem 1. While it is stated that all assumptions are in appendix E, it would be nice to have the formal terms that appear in the theorem defined/explained in the main body. 
- $\rho_u^{max2}$ looks a bit weird in Theorem 1. Consider replacing it with $(\rho_u^{max})^2$
- Having FedSGD in Table 1 would be helpful since it is the first result being presented, the improvement in privacy / loss in utility compared to no defense is one of the first questions that come up. Similarly for Table 4. 
- The axis names and numbers in Figures 2 and 5 are hard to read without zooming in. 
- While noise radius is used to refer to the __max__ perturbation norm (at least according to section 3.4), it is not clear why this is the case. 
- In section 4.2: "To ensure a fair comparison under high-utility settings, we set the privacy budgets or noise scales of each baseline as follows:" It is not clear why the following parameters achieve a "fair comparison".

### Questions
1) How does the server initialize perturbations for each client in algorithm 1? If it is just random, why are these perturbations not just initialized on the client level? Doesn't having the center send them to the client risks being intercepted and therefore can give rise to new types of attacks? 

2) Since the relationship between privacy strength and noise magnitude is not strictly monotonic, how would one go about optimizing the radius for noise?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FedEM (Federated Error Minimization), a privacy-preserving federated learning framework that defends against gradient leakage attacks by injecting learnable perturbations directly into client data rather than adding noise to gradients. The method employs a dual-step optimization mechanism where each client alternates between updating a local perturbation vector delta_k and a local perturbation model theta_u over N inner steps per batch. The authors reformulate the federated optimization objective to incorporate perturbation constraints (rho_min_u leq norm delta_k leq rho_max_u), provide convergence guarantees under smoothness assumptions, and evaluate the approach on multiple image and text datasets against gradient inversion attacks. Experiments show that FedEM achieves competitive utility while providing stronger empirical resistance to reconstruction attacks compared to differential privacy baselines.

### Strengths
The paper introduces an input-level perturbation approach that differs from gradient-level noise injection methods, providing an interesting alternative viewpoint on privacy-utility tradeoffs in federated learning.
The evaluation spans multiple modalities (images and text), datasets (MNIST, FashionMNIST, CIFAR-10, CIFAR-100, Tiny-ImageNet, CoLA, SST-2), and attack methods (Inverting Gradients, GIAS, LAMP), demonstrating broad applicability.
The paper provides formal convergence guarantees (Theorem 1) for both convex and non-convex settings, establishing that the method reaches a neighborhood of stationary points.
The appendix contains thorough investigations of perturbation magnitude effects, lower bound constraints, scalability to 50 clients, and randomness analysis across multiple seeds.

### Weaknesses
W1- The paper does not provide formal (epsilon, delta)-differential privacy guarantees. Instead, it relies on Lemma 2 from Zhang et al. (2024) using a non-standard metric epsilon_p (Equation 9) that measures empirical reconstruction quality. When comparing FedEM against LDP baselines (DP-GAS, DP-LAP, LDPM) in Table 1, this creates an asymmetry: the baselines provide rigorous information-theoretic privacy guarantees while FedEM provides heuristic protection against observed attacks. The paper would benefit from either establishing formal DP guarantees or clarifying how practitioners should reason about privacy protection in the absence of worst-case guarantees.
W2- The experimental setup raises questions about fair comparison. Algorithm 1 suggests per-batch server aggregation (Line 6 iterates over batches, Line 16 aggregates inside the batch loop), but it is unclear whether baseline methods also aggregate per-batch or use standard per-epoch updates. If aggregation frequencies differ, FedEM may receive more server updates per stated "round" than baselines. 
W3- Additionally, Table 1 compares methods under different privacy parameters: DP methods use noise scale 1/255, PPFA uses epsilon equals 0.995, LDPM uses sigma equals 0.0005, and FedEM uses radius 8/255. Converting these to comparable epsilon values would strengthen the evaluation.
W4- The connection between the stated optimization objective (Equation 3) and Algorithm 1 requires clarification. The objective formulates a min-min problem over theta and delta_k, but the algorithm introduces an auxiliary variable theta_u that is updated locally (Lines 9-12) yet discarded when computing the actual gradient sent to the server (Line 14 evaluates at global theta, not theta_u). Since theta_u is reset every batch (Line 8) and never communicated, its role in solving the stated objective is unclear. Additionally, Theorem 1 shows convergence to a neighborhood with error floor O(rho_max_u squared), though the practical implications of this convergence gap could be discussed more thoroughly.
W5- Several design choices would benefit from additional justification. Line 4 specifies that the server initializes perturbations delta_k, but the rationale for centralized versus local initialization is not provided. The constraint rho_min_u leq norm delta_k leq rho_max_u imposes a minimum perturbation magnitude, which differs from standard adversarial training approaches, and Table 6 indicates this affects accuracy. Finally, the computational overhead from N equals 15 inner steps per batch is acknowledged in Figure 6, but more detailed analysis (wall-clock time, FLOPs) comparing against baselines would help assess the practical trade-offs for resource-constrained federated environments.

### Questions
Q1. Does the server aggregate gradients per batch (as Algorithm 1 Line 16 suggests) or per epoch? Do baseline methods use the same aggregation frequency? How many total server updates occur per "round" for FedEM versus baselines?
Q2. Can the authors provide formal (epsilon, delta)-DP guarantees for FedEM? If not, how should practitioners reason about privacy protection when DP guarantees are not available? How does the privacy protection compare quantitatively to LDP methods under equivalent privacy budgets?
Q3. What are the accuracy results when all methods are constrained to the same number of server aggregations (not rounds, but actual communication events)? This would provide a fair comparison.
Q4.  Why is the local perturbation model theta_u necessary when gradients are evaluated at the global theta? What would happen if clients simply optimized delta_k directly using the global model without maintaining theta_u? Can you clarify the connection between Equation 3 and Algorithm 1 Line 14?
Q5. Why must the server initialize delta_k rather than having clients sample their own random initialization? What is the distribution used for initialization? Is this design choice necessary for correctness or is it arbitrary?
Q6. Can you provide the equivalent epsilon values (under standard DP accounting) for all methods in Table 1 to enable fair comparison? Specifically, what epsilon does FedEM with radius equals 8 divided by 255 correspond to?
Q7. What is the empirical or theoretical justification for requiring rho_min_u greater than 0? Table 6 shows accuracy degradation with higher rho_min_u - is this trade-off necessary, and under what conditions?

### Soundness
3

### Presentation
4

### Contribution
2
