# EA-PS: Estimated Attack Effectiveness based Poisoning Defense in Federated Learning under Parameter Constraint Strategy

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Federated learning is vulnerable to poisoning attacks due to the characteristics of its learning paradigm. There are a number of server-based and client-based poisoning defense methods to mitigate the impact of the attack. However, when facing persistent attacks with long-lasting attack effects, defense methods fail to guarantee robust and stable performance. In this paper, we propose a client-side defense method, EA-PS, which can be effectively combined with server-side methods to address the above issues. The key idea of EA-PS is to constrain the perturbation range of local parameters while minimizing the impact of attacks. To theoretically guarantee the performance and robustness of EA-PS, we prove that our methods have an efficiency guarantee with a lower upper bound, a robustness guarantee with a smaller certified radius, and a larger convergence upper bound. Experimental results show that, compared with other client-side defense methods combined with different server-side defense methods under both IID and non-IID data distributions, EA-PS reduces more performance degradation, achieves lower attack success rates, and has more stable defense performance with smaller variance. Our code can be found at https://anonymous.4open.science/r/EA-SP-6BC9.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a client-side defense method called EA-PS to mitigate persistent poisoning and backdoor attacks in federated learning. The method builds upon LeadFL by introducing a new optimization objective that minimizes the change in attack impact between training rounds, and adds a parameter constraint strategy to stabilize model updates. Theoretical analyses claim lower optimization upper bounds, a smaller certified radius, and better convergence guarantees, and experiments show the proposed defenses can lower the attack variances.

### Strengths
- Introduces a novel and theoretically motivated parameter constraint strategy to enhance defense stability.
- Demonstrates that EA-PS can be easily combined with existing server-side defenses, improving flexibility.

### Weaknesses
- The reviewer is confused about why the BA presented in the paper is surprisingly high while the authors still state the proposed defense is effective. Although EA-PS stabilizes defense performance across rounds, its backdoor accuracy remains high in many settings, indicating that the method may not fully neutralize persistent attacks. 
- From Figure 1, the reviewer did not find that EA-PS- is always better than FL-WBC and LeadFL.
- The authors are suggested to explain long-lasting attack effects. It seems that they use backdoor attacks to represent this effect.
- The conducted backdoor attacks are relatively simple, but more advanced attacks, such as adaptive ones and distributed triggers, should be investigated.

### Questions
- Why does the BA appear to be high in some cases?
- What is the effect of a long-lasting attack?
- The reviewer is not confident about some statements in the manuscript. For example, some descriptions from Figure 1 are not straightforward. It is suggested to double-check related content.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes EA-PS, a client-side defense for federated learning. It minimizes the change in attack effect across rounds and adds a parameter constraint to keep updates in a controlled space. The idea is clear, and the experiments look solid across datasets and attack types.

### Strengths
Overall very solid work: well-motivated, simple to plug in, and results show good robustness and stability.

### Weaknesses
Some tables look cramped due to space limits, which hurts readability a bit.

### Questions
The paper states: *“We derive a lower theoretical upper bound of the enhanced objective function to prove the efficiency of EA-PS.”*
Could you also add a small empirical check to back the “efficiency” claim? For example, show (i) convergence curves under equal time/compute budgets, or (ii) a quick cost–benefit view (runtime/communication vs. BA reduction and stability). That would make the efficiency claim more convincing.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes EA-PS, a client-side defense against poisoning attacks in federated learning. The method optimizes a temporal attack-impact difference $A_t - A_{t-1}$ and enforces a parameter constraint $AB = \lambda B$ with an adaptive mapping $B$ and a regularizer over $\beta$ to stabilize local updates. The constrained objective integrates these components, followed by gradient trimming. The authors provide theoretical analyses on convergence and certified robustness.

### Strengths
+ The paper clearly focuses on persistent poisoning and instability in existing client defenses. Figure 1 effectively illustrates the instability of BA across different methods and settings.
+ The formulation combines an attack-effectiveness objective with a parameter constraint and an adaptive mapping, with implementation achieved through gradient clipping.
+ Theoretical results include a convergence bound (Theorem 5.1) and a certified-robustness argument (Theorem 5.2).

### Weaknesses
- The intuition for preferring $A_t - A_{t-1}$ over $A_t$ is insufficiently developed in the main text. The proof sketch of Theorem 4.1 is mostly algebraic and does not clearly explain why the resulting “lower upper bound” would translate into empirical stability in non-convex FL.
- It remains unclear whether $B$ is client-specific or global, how $B$ and $\lambda$ are estimated or updated in each training round, and what the computational or communication overhead is when enforcing $AB = \lambda B$. The practical impact of the approximation $\lambda \simeq (A_t + A_{t-1}) / 2$ on theoretical guarantees is not discussed.
- The datasets used (CIFAR-10, FEMNIST, and FashionMNIST) are too simple, limiting the external validity for realistic and heterogeneous FL scenarios. The models are lightweight, so scalability to deeper CNNs or transformer architectures has not been demonstrated.
- While classic server-side methods are included, newer 2024–2025 defenses are not reported as main baselines. Although AlignIns and other recent works are cited, the experimental section still focuses on older methods, which reduces the competitiveness of the evaluation.
- Table 1 is dense and difficult to interpret. Some reported variances are large and not explained. Confidence intervals and significance tests are missing, despite strong claims such as “BA improved by up to 14.9%” and “variance reduced by 40%”.
- Convergence and certified-robustness statements are provided, but the connection between their assumptions and non-IID deep models is not sufficiently discussed. Important factors such as coordinate-wise Lipschitz conditions and bias introduced by clipping and mapping are not analyzed.
- The notation is inconsistent (for example, $A_t$ vs. At). Figure captions are brief, and the transition from Section 4 to Section 5 is abrupt, particularly around Eq. 11 and the discussion on gradient trimming.

### Questions
1.Why should minimizing $A_t - A_{t-1}$ (rather than $A_t$) improve stability under persistent attacks beyond the algebraic bound in Theorem 4.1? Please add empirical results showing how this objective correlates with BA variance reduction.

2.Is $B$ global or client-specific? How is $B$ initialized and updated, and what is the runtime or communication overhead of enforcing $AB = \lambda B$ in each round?

3.How sensitive are the theoretical guarantees and empirical results to the approximation $\lambda \simeq (A_t + A_{t-1}) / 2$?

4.Can you provide results using deeper models and larger or more heterogeneous datasets, and compare them against recent defenses to demonstrate state-of-the-art performance?

5.Please report mean ± standard deviation across the five runs, include significance tests for major claims, and present an analysis of computational and communication overhead.

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
3

### Summary
This paper proposes a client-side Federated Learning (FL) defense method named EA-PS, aimed at addressing poisoning attacks, particularly those with persistent effects. Compared to existing server-side and client-side defense methods (like LeadFL), EA-PS aims to improve defense robustness and stability. The method introduces two core innovations: 1) a new optimization objective function that minimizes the temporal change in the attack impact coefficient ( $A_t - A_{t-1}$) rather than its absolute value; 2) a parameter constraint strategy ( $AB = \\lambda B$) to enhance stability by limiting the parameter perturbation range. The authors provide theoretical analysis, claiming the new objective has a superior optimization upper bound, and analyze convergence and robustness (certified radius). Experiments on CIFAR10 and FEMNIST demonstrate that EA-PS outperforms baselines in reducing backdoor accuracy (BA), especially in reducing BA variance.

### Strengths
**++ Significance of the Problem Definition:** The paper correctly identifies a critical, often-overlooked issue in poisoning defense research: the instability of defense performance (i.e., high variance in backdoor accuracy across different rounds or runs), as shown in Figure 1. Many prior works focus only on average performance, but high variance makes a defense unreliable in practice. EA-PS explicitly targets stability as one of its core design goals.

**++ Methodological Novelty:** Shifting the defense objective from minimizing instantaneous attack impact (like $A_t$ in LeadFL) to minimizing the change in attack impact ( $A_t - A_{t-1}$ ) is a novel perspective for better handling persistent attacks. Furthermore, combining this objective with a parameter constraint strategy to control the stability of the optimization space is an innovative client-side defense design.

**++ Solid Experimental Evaluation:** The experimental design is comprehensive. The authors evaluate EA-PS under various server-side defenses (FedAvg, MultiKrum, Bulyan, etc.), multiple attack types (pixel, spectrum, label flipping, etc.), and both IID and non-IID data settings. More importantly, the ablation studies using LeadFL+ (constraint only) and EA-PS (objective only) clearly demonstrate the individual contributions of the two core components, strongly supporting the effectiveness of their combination (especially the parameter constraint) in reducing variance.

### Weaknesses
\-- **Clarity and Rigor of Theoretical Analysis are Questionable.** The paper's theoretical contributions (Theorems 4.1, 5.1, 5.2) are highly ambiguous in their presentation, and potentially contradictory, which weakens the reliability of their theoretical foundation. See C1.

\-- **Complexity and Overhead of the Method are Severely Understated.** EA-PS introduces significant computational complexity, especially the Hessian-based calculation of $A_t$ and the optimization of multiple new hyperparameters. The paper's discussion of this overhead is insufficient, casting doubt on its practicality for large-scale models. See C2.

\-- **Motivation for the Parameter Constraint Strategy is Unclear.** The paper fails to clearly explain why the specific form $AB = \\lambda B$ was chosen as the constraint to achieve stability. The intuition and mathematical principles behind it are poorly articulated. See C3.

### Questions
- **C1: Regarding Theoretical Rigor:**
  - For _Theorem 5.1 (Convergence):_ The paper claims EA-PS has a "larger convergence upper bound" and presents this as a trade-off (sacrificing convergence speed for robustness). However, in the abstract and conclusion, this is presented as a positive "guarantee." This phrasing is extremely confusing. In convergence analysis, a "larger" upper bound typically means worse convergence (a weaker bound). Then why a large convergence upper bound can be stated as a 'Convergence Guarantee'? The authors need to clarify this contradiction.
  - For _Theorem 5.2 (Robustness):_ The paper claims EA-PS has a "smaller certified radius." In the robustness certification literature (e.g., randomized smoothing), a larger certified radius implies stronger robustness (i.e., the model is provably robust within a larger perturbation radius). The authors' conclusion seems to contradict standard understanding. If a different definition is used, it must be explicitly stated. The ambiguity of this core robustness claim severely damages the paper's credibility.
  - For _Theorem 4.1 (Optimization Bound):_ The proof of $A_t \\le A_{\\hat{t}}$ in Appendix B.10 (esp. the derivations from (30) to (34) and (33) to (35)) is very obscure and seems to rely on many unstated assumptions and simplifications. For example, the step $A_{\\overline{t}}-A_{t}=(t+1)\\epsilon$ is derived too hastily. The mathematical rigor of this proof is questionable.
- **C2: Regarding Complexity and Practicality:**
  - On _Hessian Matrix Calculation:_ Algorithm 1 relies on the calculation of $A_t$, which (per Equation 5) is based on the Hessian matrix $H_{t,e}^k$. For modern deep networks, computing the full Hessian is computationally and memory-wise prohibitive. The paper mentions using $\\tilde{H}\_{t,e}^{k}$ (Eq 12) but does not specify how this is approximated (e.g., Hessian-vector products, Fisher Information Matrix, or other methods). If it cannot be approximated efficiently, the method is not feasible in practice.
  - On _Hyper-parameter Sensitivity:_ EA-PS introduces multiple new hyper-parameters ( $\\alpha, \\beta, \\gamma, \\lambda$ ) and an adaptive spatial mapping $B$. The tuning experiments in Figures 2 and 3 show that BA and variance are quite sensitive to these parameters (especially $\\alpha$ and $\\lambda$). The paper lacks a discussion on how these parameters could be tuned efficiently in a real-world application.
  - On _Time Overhead:_ The paper acknowledges higher time overhead in the limitations section (Table 21) but downplays it. Given the Hessian calculation problem (C2.1), the actual overhead is likely much higher than LeadFL. A more thorough analysis of computational and (potentially) communication overhead is needed.
- **C3: Regarding Motivation of the Constraint Strategy:**
  - The paper jumps from the need to "ensure stability" to the $AB = \\lambda B$ constraint (Eq 6). The motivation "map the optimized manifold space... into the unit space I... by converting the spatial constraints... into the base (rank) constraint" is extremely cryptic.
  - $A$ is the attack impact coefficient matrix, $B$ is the "spatial mapping," and $\\lambda$ is the "linear decision rule" ratio. Why does this an eigenvalue-like form "ensure stability" and "alleviate untargeted attacks"? What does $B$ physically represent (e.g., an eigenbasis of $A$)? The authors need to provide a much clearer, more intuitive physical explanation to support this core mechanism.

### Soundness
4

### Presentation
3

### Contribution
3
