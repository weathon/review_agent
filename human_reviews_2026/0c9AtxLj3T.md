# PEARL-Prox: Proximal Algorithm for Resolving Player Drift in Multiplayer Federated Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Recently, Yoon et al. (2025) introduced multiplayer federated learning (MpFL), a novel federated learning framework capable of formulating the strategically behaving, rational clients. 
In MpFL, the clients are modeled as players of a multiplayer game with individual objectives, aiming to seek an equilibrium.
While Per-Player Local Stochastic Gradient Descent (PEARL-SGD) algorithm has been proposed as a counterpart of Local SGD in the MpFL setup, it exhibits the *player drift* phenomenon&mdash;excessive local updates by individual players lead to divergence of the global dynamics. In this work, we formalize the concept of player drift and propose the *Per-Player Local Proximal Algorithm (PEARL-Prox)* to resolve it.
PEARL-Prox lets each player optimize a regularized objective with high accuracy, ensuring convergence to the equilibrium while enabling the players to exploit their local compute budgets.
Consequently, PEARL-Prox offers a significantly improved communication complexity of $\mathcal{O}\left(\log\epsilon^{-1}\right)$ compared to the $\Omega\left(\epsilon^{-1/2}\right)$ complexity of PEARL-SGD under the same theoretical assumptions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces an algorithm named PEARL-Prox which is the proximal counterpart of the PEARL-SGD algorithm used in the multiplayer federated learning problem. The authors define the conception of player drift and analyze the proposed algorithm in the exact / inexact setting under assumptions such as smoothness and monotonicity and provide the corresponding convergence guarantee. The obtained communication complexity is compared to that of the PEARL-SGD. Numerical experiments are provided to validate the claims made in the paper.

### Strengths
(i) The paper overall is well-written and the logic is easy to follow. 

(ii) The authors introduce and analyze the PEARL-Prox algorithm, which is novel, convergence guarantees are provided with theoretical intuitions.

(iii) The concept of player drift is introduced to help better understand the convergence in this case.

### Weaknesses
(i) The claimed better communication complexity of the PEARL-Prox algorithm seems to rely on selecting $\lambda$ optimally. However, the selection of $\lambda$ depends on $L_{\max}$ and the condition number $\kappa$, which is hard or impossible to compute. This also applies to Corollary 4.8 when we are use SGD as a subroutine, and we need to select the local step size. In addition, the comparison happens when $\tau$ (the local steps) goes to infinity, which is not practical. Therefore I do not believe that the comparison between those complexities are well justified. I believe this limitation is critical because the major contribution of the paper ties to the claim of improved complexity.

(ii) The proposed algorithm is novel only in the sense that it is applied in the MpFL setting, which is mainly theoretical. It is generally well known that the proximal framework can be used to mitigate the drift in FL, and compared to FedAvg or gradient based algorithms, we are essentially adding a regularizer to control how far each client can proceed in each step. In this sense, it is not unexpected that switching from PEARL-SGD to Prox yields such benefits, given the previous examples such as FedProx, ProxSkip and so on.

(iii) I do not get the motivation why we are considering MpFL setting, it seems to be very related to the Personalized Federated Learning framework where proximal algorithms are known to work very well. Is there a real world problem that the setting applies to while simpler frameworks can not? It seems to me that only unnecessary mathematical complications are added to the framework, but the practical problems can already be represented by simper framework. 

(iv) The empirical evidence seems to be very limited. I understand that the authors mentioned that right now the MpFL problems are in a theoretical stage, but only experiments on quadratics are provided which makes it very unconvincing. It would be much better if the authors can provide some sort of deep learning experiments where the proper tuning of step sizes are discussed, so that we know that the benefits are real.

(v) The set of assumptions seem to be very restrictive overall, the joint $F$ needs to be quasi-strongly monotone and star-cocoercive. In practice I do not see how they can be satisfied. The synthetic experiments are constructed in a way that those conditions are met, but as I said, it is important to figure out whether the benefits are true in general, otherwise the gain would be quite restrictive.

### Questions
(i) What is the meaning of the solution in MpFL? I understand that it is a Nash equilibrium, but does it have any special meaning in FL systems? It seems to me a bit strange that clients in FL systems pursue conflicting goals.

(ii) How do we tune the step size of the proximal operator, and how do we select the subroutine in practice? 

(iii) What is the main technical challenge in the analysis, the extension from gradient framework to proximal framework seems to be pretty straightforward, what makes it difficult/novel in this case?

### Soundness
2

### Presentation
3

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
The paper tackles “player drift” in Multiplayer Federated Learning (MpFL). It introduces PEARL-Prox, which has each player solve a locally regularized objective between communication rounds. The authors provide convergence guarantees, and verify the theoretical results via numerical experiments, demonstrating the effectiveness and practicality of the proposed PEARL-Prox in handling player drift.

### Strengths
This paper introduces PEARL-Prox, a proximal-type algorithm designed for Multiplayer Federated Learning (MpFL), where multiple clients (players) optimize their own utility functions instead of sharing a single global objective.

PEARL-Prox lets each player optimize a regularized objective with high accuracy, ensuring convergence to the equilibrium while enabling the players to exploit their local compute budgets.

The authors provide convergence guarantees, and verify the theoretical results via numerical experiments, demonstrating the effectiveness and practicality of the proposed PEARL-Prox in handling player drift.

### Weaknesses
1. The proposed PEARL-Prox algorithm strongly resembles the classical FedProx method. The only substantial change is that PEARL-Prox is formulated under a multi-player objective, where each client minimizes its own loss. However, the paper does not clearly articulate how this modification leads to fundamentally new algorithmic behavior, rather than a straightforward reinterpretation of FedProx under a Nash-equilibrium setting.
2. Limited experimental scope: No real-world MpFL scenario is tested, and only a single scalar metric is plotted. No tables of quantitative results or ablations appear; thus the empirical evidence rests entirely on figures.
3. Apart from PEARL-SGD, no comparison with classical FL drift-mitigation methods adapted to MpFL is shown. 
4. Strong assumptions: The paper does not discuss how restrictive these are or whether PEARL-Prox could still behave well when they fail (e.g., Assumption 4.2.) .
5. Theorem 4.4 requires $\lambda>\tfrac12(\ell+2L_{\max}\sqrt\kappa)$, but guidance on estimating $\ell$ or $\kappa$ is missing. Besides, while the paper repeatedly emphasizes the importance of λ, its practical selection strategy is never clearly specified or theoretically justified.

### Questions
1. How does PEARL-Prox differ fundamentally from FedProx beyond the change to a multi-player objective formulation? 
2. Why does the experimental evaluation omit real-world multi-player federated learning (MpFL) settings?
3. Could the authors provide quantitative results or ablation studies, beyond the single plotted metric, to substantiate the empirical claims?
4. Why are classical FL drift-mitigation algorithms (e.g., FedProx, FedDyn, SCAFFOLD) not adapted and compared under the MpFL setup, aside from PEARL-SGD? 
5. How restrictive are the theoretical assumptions (e.g., Assumption 4.2)? Does PEARL-Prox remain stable or effective when these assumptions are partially violated in practice?
6. How should λ be selected in implementation, and is there theoretical or empirical evidence supporting this choice?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper consider the framework of multiplayer federated learning (MpFL), where the goal is to find a Nash equilibrium of the game. A communication-efficient algorithm, PEARL-Prox, is proposed to mitigate the heterogeneity effect introduced by different client data in order to speedup the convergence, compared to PEARL-SGD. Convergence analysis and numerical simulations are provided.

### Strengths
1. The motivation, mitigating divergence due to excessive local computation under competitive objectives, is clear.

2. The convergence analyses are detailed. Both linear convergence for exact proximal computation and convergence to a bounded neighborhood for inexact computation are provided.

### Weaknesses
1. The idea of PEARL-Prox has limited novelty. While authors stress the difference from FedProx, the idea of adding a proximal term to mitigate divergence is conceptually similar. Novelty lies mostly in the game-theoretic reinterpretation rather than algorithmic mechanics.

2. The convergence analyses are straightforward. The techniques and ideas of analyses are similar to gradient play in game-theory literature, which undermines its technical difficulty.

3. The claim that PERL-Prox improves convergence rate to a linear rate is questionable. I suspect the claimed linear convergence is practically impossible as the sub minimization problem can never be solved exactly. In order words, if allowing inexactness, the convergence rate still should be $O(\epsilon^{-1/2})$, which is caused by the variance of sampling process.

4. All experiments are synthetic quadratic games and no real-world federated or multi-agent datasets are tested. Lacks evaluation of scalability (large n or high-dimensional d), robustness to noise, or comparison to strong FL baselines like FedProx/FedAvg under realistic conditions.

### Questions
1. Compared to FedProx, what is the novelty of the proposed algorithm (conceptually and technically)?

2. What makes the convergence analysis different from gradient play?

3. What is the convergence rate of PEARL-Prox with inexactness? Is it still linear if diminishing stepsizes are applied? Please show the rate in the order of $\epsilon$.

4. How does PEARL-Prox perform if more practical conditions are considered, e.g., non-convexity, asynchrony, partial participation, etc?

### Soundness
2

### Presentation
2

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
This paper proposes and studies a novel algorithm for the Multiplayer FL framework, an emergent framework that goes beyond the usual FL setting and aim to find personalized models that fit an equilibrium where all models are optimal given the values of all other models.
The proposed algorithm, coined PEARL-Prox, reduces the number of communication rounds required by allowing to perform more local training steps between communication rounds.
It is based on proximal updates, that are performed locally by clients.
When parameters are properly set, the local updated (even if performed approximately) allows for many local training steps while still guaranteeing fast convergence: this allows to reduce the number of communucations required in the multiplayer FL setting.
Numerical examples illustrate the theory on $n$-player games where objective functions are quadratics.

### Strengths
1. A new method for tackling client heterogeneity in the novel framework of multiplayer FL is proposed, which extend the range of applicability of this type of framework, extending the available algorithmic toolbox.
2. The proposed method is studied theoretically, with guarantees that show a reduction in the number of communication from $O(1/\epsilon^{-1/2})$ to $O(\log(1/\epsilon))$.
3. One strength of the method is that it can use any optimizer locally, since the theoretical results solely rely on the fact that a local optimization problem is approximately solved locally.

### Weaknesses
1. Although it makes sense to propose algorithm in novel frameworks, the proposed algorithm seems to be very closely related to algorithms like FedProx [1,2] or variants like 5GCS [3]. The authors made a significant effort to claim that the setting is fundamentally different from FL, but strong similarity show in assumptions 4.1-4.3 and subsequent proofs. Most results seem to be direct translations of existing results in federated learning.
2. The examples of "closely related frameworks" in FL look somewhat artificial. The interpretation in term of multi-agent reinforcement learning is vague and seems incorrect, as the authors claims this would require extending the framework to "multi-objective games".
3. At the end of page 4, it seems that authors claim that the player drift phenomenon is somewhat unexpected: it seems that this is very close to the client drift phenomenon, and that it arises in very similar situations.
4. Experiments look somewhat artificial, and only exhibit marginal gains w.r.t. algorithms that use only one local step. This does not look as spectacular as what the theoretical claims state.

[1] Federated Optimization in Heterogeneous Networks, Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith, MLSys, 2020.

{2] On Convergence of FedProx: Local Dissimilarity Invariant Bounds, Non-smoothness and Beyond, Xiao-Tong Yuan, Ping Li, NeurIPS 2022.

[2] Can 5th Generation Local Training Methods Support Client Sampling? Yes!, Michał Grudzien, Grigory Malinovsky, Peter Richtárik, AISTATS 2023.

### Questions
1. What are the actual challenges in extending results from the classical FL setting to multiplayer FL?
2. Are there other classical numerical benchmarks for multiplayer FL where the proposed method could be studied?
3. It seems that player drift has fundamental involves fundamental differences with the classical client drift: what are these fundamental differences?

### Soundness
4

### Presentation
3

### Contribution
2
