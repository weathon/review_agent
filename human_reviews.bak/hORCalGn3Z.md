# Communication-Efficient Gradient Descent-Accent Methods for Distributed Variational Inequalities: Unified Analysis and Local Updates

- Decision: Accept (poster)
- Scores: 6, 5, 6, 8

## Abstract
Distributed and federated learning algorithms and techniques associated primarily with minimization problems. However, with the increase of minimax optimization and variational inequality problems in machine learning, the necessity of designing efficient distributed/federated learning approaches for these problems is becoming more apparent. In this paper, we provide a unified convergence analysis of communication-efficient local training methods for distributed variational inequality problems (VIPs). Our approach is based on a general key assumption on the stochastic estimates that allows us to propose and analyze several novel local training algorithms under a single framework for solving a class of structured non-monotone VIPs. We present the first local gradient descent-accent algorithms with provable improved communication complexity for solving distributed variational inequalities on heterogeneous data. The general algorithmic framework recovers state-of-the-art algorithms and their sharp convergence guarantees when the setting is specialized to minimization or minimax optimization problems. Finally, we demonstrate the strong performance of the proposed algorithms compared to state-of-the-art methods when solving federated minimax optimization problems.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies federated minimax optimization problems. 

The authors proposed Proxskip-GDA and Proskip-L-SVRGDA which generalize the recent advance of Proxskip (Mishchenko et.al 2022) on the minimax (variational inequalities) problems and establish the new state-of-art in terms of communication complexity in both deterministic and stochastic setting.  

However, the analysis and design of the proposed methods are very similar to the ones for minization problems, including the deterministic setting (Mishchenko et.al 2022) and the stochastic setting (Malinovsky et.al 2022). The author need to clarify the differences between their methods and the previous Proxskip framework (especially the hardness when generalize these methods from minimization to minimax).

### Strengths
The convergence rates in this paper are significant. They cover the settings of both stochastic and deterministic and establish the new SOTA.

### Weaknesses
My main concern about this paper is its novelty. The generalization of proxskip framwork (Mishchenko et.al 2022) into minimax optimization seems direct. The variance reduction variant is interesting, but the technique is also similar to the work for minimization problems (Malinovsky et.al 2022).

For the experimental part, the author should also compare their methods with Sun et.al 2022 which firstly establish the linear rate under similar settings.

### Questions
Please refer to the weakness part.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies minimax optimization and variational inequality problems in a distributed setting. It proposes and analyzes several novel local training algorithms under a single framework for solving a class of structured non-monotone VIPs.

### Strengths
1. The minimax problem covers a variety of crucial tasks within the field of machine learning, and it becomes imperative to investigate minimax problems in a distributed context.

2. It proposes a single framework for solving a class of structured non-monotone VIPs

### Weaknesses
1. The main concern is the motivation behind this paper is not clear. Why do we need to design communication-efficient federated learning algorithms suitable for multi-player game formulations


2. The experimental analysis is very limited.
 
3. The main contributions of this paper are theoretical analysis. It would be better if this paper were submitted to other conferences such as COLT, and AISTAT.

### Questions
1.  minimax optimization is important. But do GAN, adversarial training, robust optimization and multi-agent reinforcement learning tasks match the study in this paper?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper suggests a family of algorithms for decentralized and federated learning settings of problems of solving variational inequalities. This algorithms share the same framework which is based on ProxSkip optimisation method. Concise convergence guarantees are obtained for these methods with general assumptions on problem, and obtained convergence rates expectably improve previously existed algorithms, because achieve acceleration and variance reduction.

### Strengths
All the proofs are concise and easy to follow, which is a metodical merit of the paper. Though the improvements achieved are not surprising, the way they were achieved is demonstrative enough.

### Weaknesses
Experiments seem not comprehensive. It would be interesting for authors to consider more complicated variational inequality problems than random quadratic and least squares, for example, particular test matrix games like policeman and burglar problem https://www2.isye.gatech.edu/~nemirovs/BrazilTransparenciesJuly4.pdf, and advanced practical problems like GAN training. Also, figures with comparison of the methods have onlu convergence curves without shadows showing standard deviation of function values from run to run, which is required due to stochasticity of the algorithms.

### Questions
1) Is usage of colors in Appendix C motivated?
2) Only strongly-monotone co-coercitive case is considered in the paper. Can authors report on the convergence guarantees in non-strongly-monotone case?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper provides a unified analysis of algorithms for general regularized VIPs and distributed VIPs. The proposed algorithms improve communication complexity and have strong performance compared to state-of-the-art algorithms. The paper's main contributions include the development of a new communication-efficient algorithm for solving VIPs, theoretical analysis of the algorithm's convergence properties, and experimental results demonstrating the algorithm's effectiveness on a range of problems.

### Strengths
Originality: The proposed algorithms are inspired by ProxSkip algorithm with a probability and a control variate. As far as I'm concerned, the originality of this paper is not strong, but the strength lies in the framework that recovers state-of-the-art algorithms and their sharp convergence guarantees. 

Quality: The paper's theoretical analysis of the proposed algorithm's convergence properties is rigorous and well-supported. The paper's experimental results demonstrate the algorithm's effectiveness on a range of problems.

Clarity: The paper is well-written and organized, with clear explanations of the key concepts and results. 

Significance: This paper has the potential to advance the field of distributed/federated learning and have practical implications for solving variational inequality problems in machine learning.

### Weaknesses
1. $prox_{\gamma R}(v)$ in Equation (4) should be $prox_{\gamma R}(x)$.

2. This paper does not mention the challenges or difficulties in algorithm design or theorem proofs. More explanations may help to clarify this paper’s originality.

### Questions
1. Theorem 3.1 shows that ProxSkip-VIP converges to the neighborhood of the solution, while the first experiment shows that the proposed variance-reduced method converges to the exact solution. Please add some words to explain it.

2. Is the choice of the probability ($p=\sqrt{\gamma\mu}$) because of the purpose of analysis? What impact will the change of $p$ have on the performance of the proposed algorithms.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
