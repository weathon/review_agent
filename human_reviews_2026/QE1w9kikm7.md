# Theoretical Analysis of Relative Errors in Gradient Computations for Adversarial Attacks with CE Loss

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Gradient-based adversarial attacks using the Cross-Entropy (CE) loss often overestimate robustness due to relative errors in gradient computation induced by floating-point arithmetic. Empirical methods like MIFPE mitigate this by scaling logits with a factor $ c = T / \Delta_{\text{detach}} $ where $ T = 1 $, significantly improving evaluation accuracy. However, a theoretical understanding of these errors remains limited.
To bridge this gap, we pioneer the first rigorous theoretical analysis of floating-point errors in CE-based gradient attacks, systematically dissecting relative errors across four distinct scenarios: (i) unsuccessful untargeted attacks, (ii) successful untargeted attacks, (iii) unsuccessful targeted attacks, and (iv) successful targeted attacks. This foundational study uncovers novel patterns in numerical instability and derives the optimal scaling factor $T = t^\* $ that minimizes error impact in each scenario. Notably, our analysis reveals that $ t^\* $ closely approximates 1 in unsuccessful untargeted attacks, providing a theoretical justification for MIFPE's empirical choice and addressing prior optimality gaps.
To validate the correctness of our theoretical derivations, we refine MIFPE by incorporating $ T = t^\* $ into the Theoretical MIFPE (T-MIFPE) loss function, which further reduces floating-point-induced errors. Comprehensive experiments validate our theory.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on floating-point errors in CE-based gradient attacks, which dissects relative errors across four distinct scenarios: (i) unsuccessful untargeted attacks, (ii) successful untargeted attacks, (iii) unsuccessful targeted attacks, and (iv) successful targeted attacks. To this end, this paper uncovers patterns in numerical instability and derives the optimal scaling factor that minimizes error impact in each scenario.

### Strengths
The topic this paper focused on,  the floating-point errors in CE-based gradient attacks, is very novel and interesting.

### Weaknesses
1. The template of this paper should be ICLR 2026, rather than ICLR 2025.

2. The motivation is not clearly explained, which make this paper difficult to understand. For example, authors do not explain how the model robustness is overestimated, without any experimental evidence. Besides, the floating-point error is also not well-defined. Authors should mathematically present its definition.

3. A lot of symbols are not defined. For example, what is the definition of $z_{pi_1}$ and $z_{pi_2}$? What is the definition of $c$ in Eq. (5)? What is the motivation or intuition of using CE(cz,y)?

4. The assumption for $\partial_{\hat x}(z_{pi_1}-z_{pi_2})$ is not experimentally verified. It may not hold in all model architectures or training regimes. Authors should experimental verify the correctness of this assuption before using it.

5. Could $t*$ be chosen in a simplified manner for practical usage?

6. It is unclear whether defenses could adapt to T-MIFPE or whether the observed improvements hold under adaptive attack strategies. Please clarify it.

7. While T-MIFPE consistently improves over MIFPE, the gains are small (e.g., 0.01–0.34% in robust accuracy). However, this gain is so small, which may limit the practical use of this theory.

### Questions
Please refer to weakness. Authors should improve their writing.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper targets the floating-point arithmetic issue that causes the over-estimate robustness in gradient-based adversarial attacks using CE loss. Previous works find that scaling logits by a factor $c=T/\Delta_{\texttt{detach}}$ where $T=1$ can improve robustness evaluation, but lack theoretical justification. Therefore, this paper provides the first formal analysis of this aspect across four distinct scenarios, and refines MIFPE to further validate the theorem. Experiments confirm the analytical correctness.

### Strengths
- The motivation is clear, and the paper provides the first theoretical analysis of the floating-point issue and why scaling logits by a factor can improve the estimation.
- The analysis covers four typical attack settings (target/untarget and successful/unsuccessful).
- Experiments further validate the correctness of the theory.

### Weaknesses
- Lack of some details.
- The notations and equations need some explanations for better clarity and readability.

### Questions
- Is $t^\ast$ computed averaged per batch?
- Would other loss functions, such as CW, also present similar issues? Could the framework extend to these cases?
- What precision mode is used for experiments?

### Soundness
3

### Presentation
2

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
A previous work, MIFPE, presented a technique to improve gradient-based adversarial attacks with CE loss by mitigating numerical underflow errors. This is achieved by rescaling the logits with a factor that depends on a parameter T, whose empirically estimated value is 1.
This paper extends MIFPE with a theoretical analysis aimed at obtaining an optimal value for the T parameter. Based on their findings, the authors propose a method to dynamically adjust T at each iteration of the attack. Experiments show that this strategy consistently improves attack performance compared to using a fixed T=1, although the improvement is very small, as the optimal computed value for T is often close to 1.

### Strengths
- The paper is well-written and easy to follow
- The addressed topic is relevant, as reliably estimating robustness against adversarial examples is still an open problem
- The experimental findings confirm the theoretical basis, and the proposed strategy in some settings even improves AutoAttack (which is considered a state-of-the-art method)

### Weaknesses
- The paper contribution, although theoretically sound and empirically proven, is mainly limited to estimating an optimal value for an already existing method. The main issue (numerical underflow in CE loss) and the solution (MIFPE) were presented in that previous work, where additionally their authors already tried to provide a basic theoretical justification and an empirical estimate of T (which aligns with the findings provided in this paper).
- The absolute runtime overhead of the additional T parameter estimation at each attack iteration is reported. However, this is not informative for assessing the overall impact on the attack performance. You should relate this value to the normal attack runtime, for instance, by reporting the relative runtime increase for each iteration and for the entire attack process.

Minor issues:
- In both the Introduction and Related Work sections, some symbols ($z, \pi, \Delta_{detached}$) appear without explaining their meaning, which is then described in the Theory Analysis section. You should either explicitly state their meaning as they appear or postpone them.
- In Sect. 2.1, you formalize the input vector as an image with C, W, and H dimensions. I believe that this can be generalized to consider any input dimension, as the approach should be applied to other application domains than images.
- In the Related Work section, the statement "extensive empirical evidence has revealed their significant limitation in overestimating model robustness" should report a supporting reference (for instance, [a] or even the already cited [b]).
- In Fig. 2, the last caption words mention "gray vertical dashed lines", but I guess you meant red.

[a] Carlini, N., Athalye, A., Papernot, N., Brendel, W., Rauber, J., Tsipras, D., Goodfellow, I.J., Ma̧dry, A., & Kurakin, A. (2019). On Evaluating Adversarial Robustness. ArXiv, abs/1902.06705.

[b] Carlini, N., & Wagner, D.A. (2016). Towards Evaluating the Robustness of Neural Networks. 2017 IEEE Symposium on Security and Privacy (SP), 39-57.

### Questions
- I think it would be very interesting to individually report the APGD-CE performance from the AutoAttack results, to analyze the improvements related to fixing the underflow errors in CE loss. Could you please show these results?
- I am also interested in understanding whether using the T-MIFPE approach inside the APGD-CE loss can further improve the attack performance by combining the automatic restarts and step size improvements of APGD with the mitigation of CE numerical errors. I suppose it is sufficient to modify a few lines in the autoattack implementation. Could you please provide that?
- In the appendix, last page, you state that "Each experimental run [...] was executed under $\ell _\infty$  - or $\ell _2$ -bounded threat models". However, in the paper, I only see results for the former. Is it a refusal, or did you actually run experiments for the latter as well? If so, what are your findings for this setting? I think it should be relevant to at least discuss them.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents the first systematic theoretical analysis of floating-point–induced relative errors in gradient computations for cross-entropy–based adversarial attacks. The authors classify attacks into four cases: successful and unsuccessful, targeted and untargeted. They derive the optimal scaling factor that minimizes these errors and propose a theoretically grounded loss function named T-MIFPE. Experiments on CIFAR-10, CIFAR-100, and ImageNet demonstrate consistent yet modest improvements over MIFPE, confirming the validity of the theoretical analysis.

### Strengths
1. Provides a theoretical treatment of floating-point–induced gradient errors across multiple attack scenarios.
2. Experiments across multiple datasets and models consistently support theoretical findings.
3. Offers a generalizable framework for analyzing numerical stability in adversarial attacks.

### Weaknesses
1.Experimental improvements are minimal (mostly ~0.1%), which may limit practical impact despite theoretical justification.
2. Some theoretical assumptions (e.g., independence between gradient terms and scaling factor) are not thoroughly discussed.
3. The analysis is restricted to CE-based attacks; more complex losses or adaptive attacks remain unexplored.
4. The paper is difficult to follow.

### Questions
1. Does the theoretical framework consider potential dependence between the scaling factor and gradient terms?
2. Could the approach generalize to other loss functions such as C&W or DLR?
3. Since experiments use a fixed random seed, have you tested result stability across multiple initializations?

### Soundness
3

### Presentation
2

### Contribution
2
