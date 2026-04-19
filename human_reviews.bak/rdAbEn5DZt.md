# Joint Gradient Balancing for Data Ordering in Finite-Sum Multi-Objective Optimization

- Decision: Accept (Spotlight)
- Scores: 6, 8, 8

## Abstract
In finite-sum optimization problems, the sample orders for parameter updates can significantly influence the convergence rate of optimization algorithms. While numerous sample ordering techniques have been proposed in the context of single-objective optimization, the problem of sample ordering in finite-sum multi-objective optimization has not been thoroughly explored. To address this gap, we propose a sample ordering method called JoGBa, which finds the sample orders for multiple objectives by jointly performing online vector balancing on the gradients of all objectives. Our theoretical analysis demonstrates that this approach outperforms the standard baseline of random ordering and accelerates the convergence rate for the MGDA algorithm. Empirical evaluation across various datasets with different multi-objective optimization algorithms further demonstrates that JoGBa can achieve faster convergence and superior final performance than other data ordering strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors in this paper modify the classical method in finite-sum multi-objective optimization such as MGDA algorithm by developing a sample ordering rule. JoGBa jointly balances gradients from multiple objectives during optimization. It is demonstrated for both theoretical analysis and empirical results, the proposed sample ordering outperforms random ordering.

### Strengths
The main idea of this paper is clearly presented through the incorporation of the MGDA algorithm with a novel sample ordering method. Figure 1 offers a great visualization and improves the audience's understanding. The numerical results demonstrate the practical superiority of the proposed algorithm.

### Weaknesses
1. line 65 typo "oerdering"
4. for better presentation, the balancing routine could be written as an algorithm instead of a definition
5. The authors need to better explain the balancing routine
6. The algorithm is complicated, the authors should provide with more motivation, remarks, and discussion about the design of algorithm. 
7. Please provide a detailed analysis of the time and space complexity of solving the balancing problem, or to compare its complexity to existing methods.
8. typo at the end of line 19 of Algorithm 1 
9. typo in Assumption 3.3 
10. In the statement of Theorem 3.6 and 3.9, the definitions for $w$ and $\sigma$ are missing. 
11. As a remark of Theorem 3.9, the authors might want to do discuss about the step size (depending on T). As T going large, the step size is diminishing. But why the converge rate is similar compare to Theorem 3.6 where a fixed step size is used? Please provide a detailed explanation of this apparent discrepancy, possibly including a comparison of the convergence behaviors under different step size regimes.
12. Please provide interpretations of the third and fourth terms in Theorem 3.9 in the context of the algorithm's behavior or performance. You could also suggest they discuss how these terms relate to key aspects of the method. 
13. There is a gap between (2) and (13), so as Theorem 3.9 and its proof. I'm convinced with the proof and the theorem statement. 
14. The authors need to discuss about the computation cost for the balance problem in the numerical section to make a fair comparison.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a novel sample order approach for multi-objective optimization problems. The method is inspired by the online vector balancing problem which tries to make the average of the centered gradient as close as to zero. Compared with different existing sample order approaches, the author shows that the proposed algorithm can achieve faster convergence. The experimental results on the NYUv2 and QM9 data sets also support the theory because the proposed algorithm outperforms different baseline methods.

### Strengths
Novel Approach to Multi-Objective Optimization: Inspired by the online vector balancing problem, The JoGBa method introduces a unique framework for ordering samples in multi-objective optimization by jointly balancing gradients across objectives . This is a new contribution to the multi-objective optimization field.

Theoretical Convergence Rate: The paper provides a thorough theoretical analysis. The convergence proofs are grounded in established assumptions, lending credibility to the claims.

Extensive Empirical Validation: JoGBa is validated on multiple datasets (e.g., NYUv2 and QM9) across diverse multi-objective optimization algorithms, such as MGDA, PCGrad, and Nash-MTL. This empirical diversity strengthens the generalizability of the findings.

### Weaknesses
Improved Convergence and Performance not clear: The authors claims that the method consistently outperforms existing data ordering strategies and dynamic weighting approaches. However, it seems the convergence rate for different algorithms are similar based on Theorem 3.6 and Theorem 3.7.

Presentation and Clarity: Certain sections, especially the theoretical parts and Algorithm 1, could be more reader-friendly. A more structured breakdown of steps and implications of theoretical results would enhance readability, making the technical details accessible to a broader audience.

Intuitions not sufficient: the online vector balancing problem is the core of balancing the gradient, but the authors fail to give enough intuition and explanation about how to implement the method across different objectives.

### Questions
1.	In line 189, the stale mean should be $\nu_{t+1}$. You write it as $m_{t+1}$. Please double check the notation.

2.	In Line 226, a necessary and sufficient condition for $\lambda$ not $x$.

3.	In Line 654, “beginning” instead of “begining”.

4.	In Definition 3.1, the authors mentioned online vector balancing which is the core of how to balance the gradient. If possible, please try to explain more clearly about the intuition (e.g., online vector balancing problem tries to make the average of the centered gradient as close as to “zero”). This will make the reader have a smooth reading experience.

5.	In Theorem 3.6 and Theorem 3.7, it seems the convergence rate of random sample ordering and the proposed algorithm are the same. Though in Theorem 3.6, the convergence rate is “=” while in Theorem 3.7, it is “<=”, one cannot conclude that the proposed algorithm converges faster than random sample ordering. The experimental results also show that the proposed algorithm converges just a little bit faster than other methods. Please try to explain clearly about the advantage of the proposed algorithm (maybe add some remarks/comments to give comparisons of different algorithms both theoretically and experimentally).  

6.	If possible, let the reader know that the paper has a clear set up of the experiments in the appendix. This could help practitioners better understand the method’s sensitivity and optimize it for specific tasks.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes a novel framework JoGBa for sample ordering in finite-sum multi-objective optimization problem, and demonstrate its empirical effectiveness on various MOO methods.

### Strengths
Data ordering for better optimization performance (that can especially outperform random shuffling) is important, and I think bring this concept to multi-objective optimization is an interesting and important contribution.

The paper theoretically analyze the convergence of MGDA for finite-sum MOO problem, with both random shuffling and JoGBa ordering.

Empirical results look promising.

### Weaknesses
For audience like me that is kind of familiar with MOO but not sample ordering, I think the presentation of algorithm 1 and figure 1 can be improved. Specifically, 
(1) I cannot directly see from Figure 1, how the JoGBa approach (Fig 1 Right) is different from data ordering on each objective separately (Fig 1 Middle). In other words, Figure might not be illustrative enough.
(2) Algorithm 1 contains numerous superscripts and subscripts, making it challenging to interpret (which is acceptable if it prioritizes rigor). However, a clearer description might help—for example, one or two sentences explaining how JoGBa differs from ordering data for each objective separately. From the current description in lines 176 to 178, "The sample ordering is then determined based on the results of solving the balancing problem (routine Balancing) with the gradient on each objective," it’s still unclear (at least for me) how this approach differs from the one illustrated in the middle of Figure 1.

I believe the paper will be a clear accept for me, if the authors can clarify my concern or improve the presentation of the main methodology as mentioned above.

### Questions
See the question in previous section.

### Soundness
3

### Presentation
3

### Contribution
4
