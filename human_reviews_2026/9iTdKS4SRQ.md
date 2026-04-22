# GIT-BO: High-Dimensional Bayesian Optimization with Tabular Foundation Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Bayesian optimization (BO) struggles in high dimensions, where Gaussian-process surrogates demand heavy retraining and brittle assumptions, slowing progress on real engineering and design problems. We introduce GIT-BO, a Gradient-Informed BO framework that couples TabPFN v2, a tabular foundation model that performs zero-shot Bayesian inference in context, with an active-subspace mechanism computed from the model’s own predictive-mean gradients. This aligns exploration to an intrinsic low-dimensional subspace via a Fisher-information estimate and selects queries with a UCB acquisition, requiring no online retraining. Across 60 problem variants spanning 20 benchmarks—nine scalable synthetic families and ten real-world tasks (e.g., power systems, Rover, MOPTA08, Mazda)—up to 500 dimensions, GIT-BO delivers a stronger performance–time trade-off than state-of-the-art GP-based methods (SAASBO, TuRBO, Vanilla BO, BAxUS), ranking highest in performance and with runtime advantages that grow with dimensionality. Limitations include memory footprint and dependence on the capacity of the underlying TFM.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GIT-BO, a gradient-informed BO framework that aims to address the challenges of high-dimensional BO. The core idea is to integrate TabPFN v2, a tabular foundation model, as a surrogate model with a gradient-informed active subspace mechanism. This approach leverages the predictive-mean gradients from TabPFN v2 to identify a low-dimensional active subspace, aligning exploration to this subspace and using a UCB acquisition function without requiring online retraining. The authors claim that GIT-BO delivers a strong performance-time trade-off across 60 problem variants, including synthetic and real-world tasks, outperforming state-of-the-art GP-based methods.

### Strengths
- The core idea of integrating TabPFN v2 with gradient-informed active subspaces is an intuitive approach for handling the curse of dimensionality
- The paper highlights the potential for significant runtime advantages (orders-of-magnitude speedups) compared to GP-based methods, which is a crucial for real-world applications
- The paper presents extensive and thorough experimental results across a wide range of synthetic and real-world benchmarks

### Weaknesses
- All methods start with 200 LHS samples. For 500 total iterations, this means 40% of the budget is spent on initialization. Why such a large initialization? This heavily favors methods that converge quickly (like GIT-BO with fast TabPFN inference) over methods that need more iterations to refine their models (like SAASBO-NUT).
- Appendix B.1 (Figure 6) shows that vanilla TabPFN v2 without GI subspace performs poorly (8.6× worse regret). But this comparison is misleading. Vanilla TabPFN v2 uses random candidate sampling in the full 500D space, which is obviously inefficient. A fairer baseline would be TabPFN + random subspace projection or TabPFN + trust regions. Without these comparisons, it's unclear whether the gains come from gradient-informed subspace discovery specifically, or just from any dimensionality reduction paired with TabPFN.
- Appendix C (Figure 10) shows that GP+GI subspace performs poorly compared to TabPFN+GI. The authors hypothesize this is due to "poorly estimated GP hyperparameters" and "unreliable gradient estimates." However, in my eyes this suggests that the success is driven by TabPFN's quality as a surrogate, not by the GI subspace mechanism itself. If GI subspaces are theoretically principled (per Assumption 3), why do they fail with GPs? This casts doubt on the generality of the approach
- Beyond invoking "No Free Lunch," the paper does not deeply investigate why GIT-BO fails on Rover, Styblinski-Tang, etc. Are these problems outside TabPFN's pre-training distribution? Do they lack low-dimensional structure? A failure mode analysis would strengthen the contribution.

### Questions
- Can you provide empirical evidence that TabPFN's posterior approximation error $\epsilon_\text{approx}(t)$ actually vanishes (or remains small) on your benchmark tasks?
- The $\Phi$-Sobolev certificate in Li et al. (2024) applies to likelihood-informed subspaces in inverse problems, where the Fisher matrix is computed from a known likelihood model. In GIT-BO, you compute $H$ from TabPFN's predictive mean gradients, which are outputs of a learned, black-box transformer. Why is it valid to apply the same certificate?
- Why fix $r = 10$ for all problems? The authors mentioned in Appendix B.2 that adaptive variance-based selection (92.5%, 95%) achieves better average ranks than $r=10$, yet the main results use $r=10$

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GIT-BO, a Bayesian optimization (BO) algorithm for high-dimensional black-box problems. GIT-BO uses prior fitted networks (PFNs) as the surrogate model, which allows for fast inference, where the popular Gaussian process surrogate has cubic inference complexity. While TabPFNv2, a tabular foundation model, already allows for modeling problems with up to 500 dimensions, the authors argue that additional steps are necessary to perform BO efficiently on high-dimensional problems. To this end, they use the gradient information provided by the PFN to estimate an active subspace. Like other subspace-based approaches for high-dimensional Bayesian optimization, they then sample points in the active subspace and project them to the full-dimensional space to evaluate the acquisition function and choose the next point to evaluate.

By comparing to several state-of-the-art algorithms for high-dimensional BO on various benchmarks, the authors provide extensive empirical evidence for the effectiveness of the proposed approach.

### Strengths
The paper addresses a relevant problem. High-dimensional BO has received considerable attention in the past and is an active field of research. The paper is the first method that uses PFNs, which is an interesting surrogate model due to its in-context capabilities, for high-dimensional BO.
The approach is well-motivated, and the paper is well-written. The storyline is clear, and the paper features an extensive empirical evaluation that shows the benefits of the approach. The evaluation is open about the limitations of GIT-BO and the performance of other state-of-the-art methods. The paper identifies the superior inference time of GIT-BO as a key strength, which is reasonable.
The paper provides an appendix with extensive ablation studies and additional experiments. The appendix complements the paper and preempts many questions I had when reading the paper.

### Weaknesses
The main concerns I have with this paper are the large performance degradations upon minor modifications of the algorithm. For instance, Figure 6 shows that the GIT-BO with expected improvement instead of the upper confidence bound performs considerably worse, worse than TabPFN without the gradient-informed subspace. Similarly, Figure 9 shows that the technique for sampling candidates in the low-dimensional subspace is crucial for performance. I wouldn’t expect such a big impact from these choices, and it makes me question the generalizability of the method.

### Questions
-	GIT-BO currently uses a search space of fixed dimensionality. Do you think the method would benefit from expanding subspaces, similar to BAxUS?
-	The reference point is centered on the centroid of the observed data. What is the impact of that choice? What happens if you center it on the incumbent observation, and why is this choice necessary?
-	Why does EI perform so poorly in Figure 6?
-	What is the difference between the sampling schemes in section B.4? What’s the difference between random sampling in the subspace and uniform sampling? And do you have any insight into why one performs better than the other?

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
5

### Summary
The paper presents a novel method GIT-BO that utilises the TabPFN foundational model as a surrogate, and uses its gradients to identify a low-dimensional subspace over which to conduct the acquisition optimisation. Authors benchmark the time and sample complexity of their method against other baselines and also conduct a number of ablations showing the importance of each of the components of the final algorithm.

### Strengths
- The proposed method is faster (in terms of wall-clock time, while maintaining performance comparable to the existing state of art
- The authors conduct plenty of interesting ablations, showing the importance of each of the components used in the final algorithm 
- The additional theory in the appendix, while not particularly novel and easily following from preceding work, is still a nice addition for completeness

### Weaknesses
- Since authors emphasise the importance of time-complexity, as opposed to pure sample complexity as it is typically done in BO literature, it would be nice to demonstrate a problem setting, where we actually care about time-complexity (e.g. high-throughput BO), as in most classical BO problems, sample complexity is paramount, whereas wallclock time is of secondary importance
- It seems to be authors focused on a relative low-data regime, where fitting a GP is still relatively fast, it would be more interesting to see what happens when the size of the dataset is much bigger, where technically fitting a GP should be slow, yet TabPFN should still be fast

### Questions
- In Appendix B1, Figure 6, you show that replacing UCB with EI in your method makes the performance drop drastically on Levy and Rosenbrock. This is quite unprecedented. Can you provide some explanation as to why that happens?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces GIT-BO, a gradient-informed Bayesian optimization framework that leverages TabPFN v2, a tabular foundation model, to perform high-dimensional Bayesian optimization without surrogate retraining. The key idea is to extract predictive-mean gradients from the frozen TabPFN model, identify a low-dimensional active subspace via Fisher information, and perform UCB-based acquisition within this subspace. The method is evaluated on 60 high-dimensional benchmark problems (up to 500D), demonstrating strong performance and runtime advantages over GP-based baselines like SAASBO, TuRBO, and BAXUS.

### Strengths
++ This method combines frozen tabular foundation models (TFMs) with gradient-informed subspace discovery, and provides a novel fusion of amortized inference and classical dimension reduction.

++ Comprehensive benchmarking against SOTA methods, with rigorous statistical ranking and runtime analysis.

### Weaknesses
-- The performance depends on the pre-trained foundation model. The frozen TFM may not adapt well to functions outside its pre-training distribution, leading to poor performance on certain tasks. 
Additionally, no fine-tuning or domain adaptation is performed, which limits generalization to highly specialized or out-of-distribution objectives.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
