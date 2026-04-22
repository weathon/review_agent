# On the Expressive Power of GNNs for Boolean Satisfiability

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Machine learning approaches to solving Boolean Satisfiability (SAT) aim to replace handcrafted heuristics with learning-based models. Graph Neural Networks have emerged as the main architecture for SAT solving, due to the natural graph representation of Boolean formulas. We analyze the expressive power of GNNs for SAT solving through the lens of the Weisfeiler-Leman (WL) test. As our main result, we prove that the full WL hierarchy cannot, in general, distinguish between satisfiable and unsatisfiable instances. We show that indistinguishability under higher-order WL carries over to practical limitations for WL-bounded solvers that set variables sequentially. We further study the expressivity required for several important families of SAT instances, including regular, random and planar instances. To quantify expressivity needs in practice, we conduct experiments on random instances from the G4SAT benchmark and industrial instances from the 2024 SAT competition. Our results suggest that while random instances are largely distinguishable, industrial instances often require more expressivity to predict a satisfying assignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper analyzes the expressive capabilities of Graph Neural Networks (GNNs) in the context of SAT solving. The authors particularly look at the Literal-clause graphs with negation connections (LCNs), one of the standard representation used in GNNs for SAT solving. The authors then theoretically prove which families of problems GNNs can and cannot distinguish, using the Weisfeiler-Lehman (WL) test, a well known expressivity analysis technique for GNNs. With experiments, the authors assess whether WL-powerful architectures are, in principle, capable of predicting satisfying assignments on a wide range of benchmark instances including randomly generated and competition benchmark instances. The results show that while almost all random instances are WL-distinguishable, many SAT competition instances are indistinguishable, showing the limitations of GNNs for industrial SAT solving.

### Strengths
- The theoretical results imply that with LCNs, any Message Passing Neural Network based models cannot fully distinguish between SAT and UNSAT instances for certain families of problems such as 3-SAT, even when given partial assignment information.
- The authors conduct experimental results to determine whether instances in a benchmark set are WL-distinguishable or not, with the experiments being done on a wide range of benchmarks sets.
- The paper provides insights into the limitations of GNNs for SAT solving, which is an important topic given the recent interest in using machine learning for combinatorial optimization problems.

### Weaknesses
- The theoretical results are limited to LCNs, due to the limitations of expressivity analysis with WL tests as stated in the Appendix. While I understand LCNs to be a common approach of using GNNs for SAT solving such as in NeuroSAT, VCGs are also widely used such as in [1], and the paper would greatly benefit from discussing their insights, other than just briefly stating that it is not possible to perform expressivity analysis on them. I personally think that as VCGs incorporate polarity information directly into the graph edges, it allows them to distinguish graphs for formulas such as those in Figure 2.
- The random instances from G4SATBench may not be as informative as ones from the SAT competition. To my understanding, this comes down to whether the benchmark was able to randomly produce any problem that is difficult to solve. If performing evaluations on random instances were the objective, the random track from the 2018 SAT competition may be better suited in this regard.
- I see no reason to limit the experimental results to the 2024 SAT competition benchmarks. Including benchmarks from previous years would only strengthen the experimental contributions.

[1] Yolcu, E., & Póczos, B. (2019). Learning Local Search Heuristics for Boolean Satisfiability. In H. M. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché‑Buc, E. B. Fox, & R. Garnett (Eds.), Advances in Neural Information Processing Systems 32 (NeurIPS 2019) (pp. 7990–8001).


Minor issues:
- The graph visualization in Figure 1 does not match up with the formula given. I believe that C6 should be C1, if we assign clause numbers in order of appearance.
- Figure 2 seems to be in the middle of the references section.
- Citations are not formatted correctly; they are missing parentheses. For example "variables Heule et al. (2024)" at L123 should be "variables (Heule et al., 2024)".
- In L132 and L137, $\chi^\ell(v) := (\chi^{\ell-1}(v), \{ \{ \chi^{\ell-1}(v) : w \in N(v) \} \})$ should be written as $\chi^\ell(v) := (\chi^{\ell-1}(v), \{ \{ \chi^{\ell-1}(w) : w \in N(v) \} \})$ (the $v$ should be $w$ in the inner set).
- In L412, it says "the only family with some formulas that could not be solved by WL is the k-clique", but Table 4 clearly shows k-vercov as another family that WL could not fully solve.

### Questions
- Is it possible at all to extend the WL-test to allow for expressivity analysis of VCGs?
- Are there other suitable graph representations of SAT problems that would be better than LCNs in terms of expressivity? What would the best representation look like, if any?
- Why were harder instances (hard+, hard++) provided only for 3-SAT? Is it not possible to generate harder instances for other problems as well?
- The experimental analysis were done on SAT instances only. Would it be possible to extend the analysis to UNSAT instances?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The power of GNNs can be understood by studying WL-testing, which is basically a form of graph isomorphism testing. On a high level, GNNs cannot distinguish some graphs from each other and this is reliant on how the weisfeiler lehman kernel behaves on these graphs. The paper shows that even if GNNs were "maxed out" on the WL power hierarchy, there would be SAT instances that would be difficult for them since it is difficult for the WL process itself (this is actually not surprising, given that SAT is a hard problem) but more interestingly, it creates these hard instances constructively, which is better than showing that they exist. Empirically, it then studies real SAT instances. Probably as expected, random SAT instances can fall to WL , and contest SAT instances don't fall as easily.

### Strengths
The paper's contribution is mostly theoretical. One of the best aspects is that they actually construct the hard family of instances, which is likely to be useful for further work more than a simple proof of existence. Also, although the paper is about GNNs on paper, most of the results are about the WL kernel's suitability for SAT, which is technically a broader problem. In reality, the paper is about finding the "blind spots" of the WL kernel as a method.

### Weaknesses
The empirical study is fairly surface level and misses the mark. We already expect that random instances are easy, contest instances are harder, and hard-by-construction instances are computationally hard. But the main problem is as follows. The power of a GNN is only upper bounded by the power of a WL test. Showing how well the WL test performs on various SAT instances, then, is meaningful only in one direction - if it fails, certainly we don't expect GNNs to succeed, but if it succeeds or performs in an intermediate manner, we *cannot actually confirm* that GNNs will follow ! Thus, the construction of the hard instances is meaningful "back to GNNs" though WL is being studied, but the studies on empirical "actually existing" instances are statements that only mean something wrt the WL kernel, not GNNs themselves.

### Questions
It seems to me that the paper is actually about the suitability of the WL kernel to SAT, and can be thought of better that way. Do you agree ? From a GNN point of view, the empirical section is actually not meaningful (as pointed out under weaknesses)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper analyzes the expressive limits of GNNs on SAT problems using a literal–clause–negation (LCN) graph representation. It proves that even the full Weisfeiler–Lehman (WL) hierarchy cannot, in general, distinguish satisfiable from unsatisfiable 3-SAT formulas, via a parity-based construction. The authors further show that this limitation persists in sequential variable-assignment settings, linking theory to neural SAT solvers. They complement the negative results with positive findings (e.g., PlanarSAT identifiable by 4-WL) and empirical WL-color diagnostics on random and industrial benchmarks. The work provides a clear, rigorous theoretical contribution, though experiments remain illustrative rather than learning-based.

### Strengths
The presentation is clear and well-organized with logical progression from simple examples to full results, making complex theoretical concepts accessible while providing complete proofs in appendices. The analysis is nuanced, including both impossibility and positive results (e.g., PlanarSAT separability, random instances).
The authors are transparent about the scope of their results, clearly acknowledging that their experiments test only necessary conditions for expressivity and do not demonstrate actual learning performance.
The main theoretical contribution (Theorem 5.3) is rigorous, proving that even the full n-WL hierarchy cannot distinguish SAT/UNSAT formulas in general, and this limitation transfers to practical sequential solvers (Corollary 5.5), showing that even after Θ(n) variable assignments, the indistinguishability persists - meaningfully connecting fundamental expressivity limits to realistic GNN-based SAT solving approaches. The link to Tseitin formulas from proof complexity is noteworthy.

### Weaknesses
The paper lacks any trained GNN experiments, making it unclear whether the theoretical limitations observed actually translate into performance gaps in practice.  The experiments are purely diagnostic and do not involve any trained GNNs, leaving the practical impact of the theoretical limits untested.
 
Lacks an explanation for why industrial instances require higher expressivity than random ones. The paper does not discuss possible ways to overcome the identified expressivity limits, leaving the reader without guidance on how future GNN architectures might address these shortcomings.

### Questions
Given that the experiments are WL-diagnostic rather than learned, do you plan to test whether these expressivity limits manifest in actual trained GNN solvers?

Given that even n-WL cannot distinguish SAT/UNSAT in general, what architectural modifications beyond standard MPNNs (e.g., augmentations, auxiliary features, hybrid approaches) do you believe could address these expressivity bottlenecks while remaining scalable?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper analyzes the expressiveness power of graph neural networks (GNNs) for the Boolean constraint satisfiability problems. The analysis is based on the observation that the power of GNNs is bounded by Weisfeiler-Leman (WL) test, which is designed to test graph isomorphism. Due to such fundamental limitation, GNNs are proved to be insufficient to solve NP-Complete problems. Furthermore, empirical evaluations are performed on the G4SAT benchmark and SAT Competition 2024.

### Strengths
- this paper consolidates many known theoretical results about GNNs and WL test as well as empirical datasets
- background knowledge about SAT solving, GNNs, graph isomorphism, and WL tests are carefully illustrated

### Weaknesses
- the proposed analysis is not novel, since all reported results are either well-known or immediately implied by previous works. Particularly, Xu et al. (2019) has pointed out that GNNs are bounded by the WL-test, which cannot even solve the graph isomorphism problems that are not as strong as the NP-Complete problems like Boolean satisfiability. 
- the presented theoretical analysis is largely rephrasing previous known results therefore does not contribute new insights
- there are some empirical observations regarding the number iterations for WL to converge, however, which does not provide meaning guidance on solving satisfiability problem with GNNs

### Questions
Besides consolidating previous known results, which is a meaningful contribution in the perspective of literature survey, what new insights either theoretical or empirical does this paper provide?

### Soundness
3

### Presentation
3

### Contribution
1
