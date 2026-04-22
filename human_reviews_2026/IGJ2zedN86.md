# Federated Equilibrium Solutions for Generalized Method of Moments applied to Instrumental Variable Analysis

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Instrumental variables (IV) analysis is an important applied tool for areas such as healthcare and consumer economics. For IV analysis in high-dimensional settings, the Generalized Method of Moments (GMM) using deep neural networks offers an efficient approach. With non-i.i.d. data sourced from scattered decentralized clients, federated learning is a popular paradigm for training the models while promising data privacy. However, to our knowledge, no federated algorithm for either GMM or IV analysis exists to date. In this work, we introduce federated IV analysis (FedIV) via federated GMM (FedGMM). We formulate FedGMM as a federated zero-sum game defined by a non-convex non-concave minimax optimization problem. We characterize the solutions to the federated game using Stackelberg equilibrium and show that it satisfies client-local equilibria up to a heterogeneity bias. Thereby, we show that the consistency of the federated GMM estimator across clients closely depends on the heterogeneity bias. Our experiments demonstrate that the federated framework for IV analysis efficiently recovers the consistent GMM estimators for low and high-dimensional data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The authors introduce FedIV, a framework for federated instrumental variable (IV) analysis via a federated version of generalized method of moments (FedGMM). They also introduce FedDeepGMM to solve FedIV, which is a federated adaptation of DeepGMM from Bennett et al. (2019). The authors formulate the federated GMM problem as a non-convex, non-concave minimax optimization, modeled as a federated zero-sum game. They characterize equilibrium solutions using Stackelberg equilibria. Experiments on synthetic and image datasets (FeMNIST, CIFAR10) demonstrate convergence and consistency of the proposed method, even under non-i.i.d. client data distributions.

### Strengths
* The paper proposes federated GMM and federated IV analysis.
* The paper provides a theoretical framework that connects federated minimax optimization with Stackelberg equilibria.
* Extends classical GMM consistency results (Hansen, 1982; Bennett et al., 2019) to the federated setting, quantifying how heterogeneity affects consistency.
* Experiments demonstrate empirical convergence and estimation quality comparable to centralized DeepGMM, matching theoretical claims.

### Weaknesses
* While experiments confirm convergence, they are mostly illustrative. The scope of the experiments is quite limited, and there is no comparison with other FL causal methods such as the cited work of Xiong et al. (2023).
* The notation is not clear and even confusing sometimes, especially the probabilistic parts of the theory.
* Some gaps exist in the theoretical results. Especially, the proofs of Lemma 1 and Theorem 2 which are directly adapted from Bennett et al. (2019) to the federated setting.
* Justifications for some steps in several proofs are not explicitly given.

### Questions
1. Why didn't the authors compare their method with other existing FL causal methods such as Xiong et al. (2023)? At least under the conditions where these methods apply.
2. The paper doesn't specify which random variables the expected values are taken over, either in the main paper or in the proofs in the appendix. For example in Equation 4, is the expected value taken over $\epsilon_i$? What is $\mathbb{E}_{n_i}$ in Equation 6, and how is it equal to the empirical average?
3. In Lemma 1, the inverse of $C_{\tilde{\theta}}$ is taken without properly discussing that it is possible to do so. Also, defining the dual norm $\lVert \cdot \rVert_{*}$ based on $C_{\tilde{\theta}}^{-1}$ would require $C_{\tilde{\theta}}$ to be positive semi-definite. I couldn't find in the paper the proof of that or any assumption about it.
4. In the proof of theorem 1, line 911, does $\lambda_{\mathrm{max}}(\cdot)$ mean the largest eigenvalue in magnitude or largest eigenvalue as a real number? If it is the latter, then the equation in 911 does not necessarily hold unless the matrix is positive semi-definite. Am I missing something or misunderstanding the notation?
5. In the proof of Theorem 2, why does the inequality at line 1088 hold when you replace the expected value with the empirical average? Also, why does the inequality at line 1091 hold when you introduce the $\epsilon_k$? It could be much clearer if each step in the proof is clearly justified. 
6. There is no proof of proposition 1 and the authors only cite Jin et al. (2020), but it should be stated how the eigenvalues of the Jacobian behave asymptotically ($\gamma \rightarrow \infty$).
7. Theorem 3 analyzes the deterministic continuous-time FedGDA flow but Algorithm 1 is based on discrete steps. The flow equation is obtained from the discrete equations ($\eta \rightarrow 0$) but not the other way around. The paper glosses over this.

I believe that the presentation needs serious enhancement overall, as my comments and questions imply. It would be easier if all equations are properly numbered so that the authors or even the reviewers can easily refer to them.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates Instrumental Variable (IV) analysis within a federated learning context. The authors state that there are currently no algorithms for federated Generalized Method of Moments or IV analysis. To address this, the paper introduces federated IV analysis by way of federated GMM (FEDGMM). This FEDGMM is formulated as a federated zero-sum game, which is a non-convex non-concave minimax optimization problem. The paper theoretically characterizes the solution to this federated game using Stackelberg equilibrium. It shows that this solution satisfies client-local equilibria up to a heterogeneity bias. The consistency of the federated GMM estimator across clients is then shown to depend on this bias.

### Strengths
1.	The paper addresses a novel and highly important problem. Applying IV analysis and GMM estimation in a privacy-preserving, federated setting opens up new possibilities for causal inference in sensitive domains like healthcare and economics, where data is decentralized and cannot be pooled.
2.	The paper frames the federated GMM problem as a non-convex non-concave minimax game. The subsequent analysis uses the concept of Stackelberg equilibrium to characterize the properties of the game's solution.

### Weaknesses
1.	The entire theoretical framework and the algorithm are analyzed under a "full client participation" setting (Line 194). This is a major limitation and is unrealistic for most practical FL systems, where client sampling (partial participation) is a defining and non-negotiable characteristic.
2.	The paper does not propose a new federated algorithm. The algorithm used for the analysis (FEDGDA) is a standard, synchronous federated adaptation of gradient descent-ascent. The contribution is thus limited to analyzing a standard algorithm in a new problem setting, rather than designing an algorithm that is robust to the specific challenges of this federated game (e.g., mitigating the heterogeneity bias, handling partial participation, or reducing communication).
3.	The experimental setup is insufficient. The experiments only compare the federated algorithms against their centralized counterparts, lacking comparisons to simpler federated alternatives. Furthermore, the validation relies on synthetic or simple image datasets (FEMNIST, CIFAR10) with artificial IV structures. The study should be extended to include more complex models and real-world datasets to demonstrate practical applicability.

### Questions
Please refer to _Weaknesses_.

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
This paper introduces FEDIV, the first framework for federated instrumental variable (IV) analysis, and proposes a federated version of the Deep Generalized Method of Moments (DEEPGMM) algorithm, termed FEDDEEPGMM. The method formulates the federated GMM estimation problem as a non-convex non-concave minimax optimization—a federated zero-sum game—and characterizes its solution through Stackelberg (local minimax) equilibria.

Theoretical results establish that under bounded gradient and Hessian dissimilarity across clients, FEDDEEPGMM converges to an approximate federated equilibrium that ensures client-level consistency of GMM estimators, up to a heterogeneity bias. The paper also analyzes the relationship between the stable points of the federated gradient descent-ascent (FEDGDA) flow and the local minimax equilibria, showing their equivalence under smoothness conditions. Experiments on both synthetic (1D regression) and high-dimensional (FEMNIST, CIFAR-10) datasets demonstrate that FEDDEEPGMM achieves comparable or better estimation accuracy than centralized DEEPGMM baselines, even under non-i.i.d. data.

### Strengths
The paper is the first to explicitly tackle federated instrumental variable analysis, bridging causal inference, GMM estimation, and federated learning. Framing the problem as a federated minimax game is conceptually original and mathematically elegant.

The derivation of E-approximate federated equilibria and theorems establishing connections between equilibrium solutions, consistency, and heterogeneity are technically solid and carefully motivated. The proofs adapt and extend the theory of local minimax optimization (Jin et al., 2020) to the federated context.

### Weaknesses
1. While the technical contribution is significant, the motivation is weakly communicated. The paper assumes familiarity with both federated learning and econometric GMM/IV analysis, offering limited intuition on why federated IV analysis matters or what real-world settings require it (e.g., distributed healthcare causal studies). A short example motivating the use case (like the one in the introduction) could be more tightly tied to the technical results.

2. The main methodological component, FEDDEEPGMM, is a straightforward federated adaptation of DEEPGMM using existing FEDGDA optimization schemes. While the theoretical characterization is new, the algorithmic structure itself does not introduce a fundamentally new federated optimization mechanism.

3. The experiments mainly reproduce DEEPGMM setups under federated conditions. However, they do not deeply analyze the heterogeneity bias—the paper’s key theoretical insight. There is no quantitative or visual exploration of how bias or consistency deteriorates as client heterogeneity increases. This omission weakens the empirical validation of the central theoretical claim.

4. The computational and communication cost of running deep minimax optimization in a federated environment is non-trivial. The paper does not report runtime, convergence speed, or communication overhead compared to baselines such as FedAvg or SCAFFOLD.

5. The manuscript is highly mathematical, with minimal narrative or conceptual explanations. This makes it difficult for non-theoretical readers to grasp the significance of the results or their connection to causal inference practice.

### Questions
1. Could the authors provide quantitative experiments on how the heterogeneity bias (as defined in Theorem 1) affects the consistency of estimators? For instance, varying the Dirichlet α parameter and plotting estimation error vs. heterogeneity.

2. The current equilibrium analysis focuses on pure strategies; could the authors elaborate on the practical implications of the open problem of mixed-strategy equilibria in federated settings?

3. How sensitive is FEDDEEPGMM to communication frequency or local iteration count (R)? Some insight into optimization stability would be helpful.

4. Could the authors provide a simplified algorithmic summary or intuition—perhaps a diagram—of how the theoretical results (equilibrium → consistency) translate into practical algorithm behavior?

5. In the experiments, why do some high-dimensional cases (e.g., CIFAR10z) show large variance or poor performance? Is this related to curvature mismatch or gradient dissimilarity?

### Soundness
2

### Presentation
2

### Contribution
2
