## Human Reviewer 1

### Summary
This paper proposes Symmetry-Aware Bayesian Optimization (SABO), a framework for exploiting known or learnable symmetries in black-box optimization problems to improve sample efficiency. Traditional approaches, such as Brown et al. (NeurIPS 2024), incorporate known invariances into Gaussian Processes through orbit-averaged kernels. This paper introduces a max-alignment view that aligns input pairs via the most similar symmetry transformations.
To address the fact that the resulting kernel may violate positive semi-definiteness, the authors introduce the projected-max kernel, which applies a PSD projection based on spectral (eigenvalue) analysis to restore validity. 
Overall, the paper contributes a more general method for incorporating symmetry structure into BO, extending symmetry exploitation beyond fixed and known invariance groups to settings where symmetries can be learned or adaptively aligned.

### Strengths
The paper presents an original and conceptually appealing extension of symmetry-exploiting Bayesian optimization methods, offering a practical and theoretically grounded approach that goes beyond prior invariant-kernel formulations. By introducing the max-alignment view and the projected-max kernel, the authors propose a framework allowing Bayesian optimization to leverage approximate or continuous symmetries without requiring exact group knowledge. The integration of spectral analysis and PSD projection is technically sound and shows thoughtful consideration of kernel stability, while the empirical results convincingly demonstrate the benefits of sharper, symmetry-aware modeling.

### Weaknesses
This paper's presentation significantly limits accessibility and clarity, especially for a broader audience. It does not clearly explain why PSD matters and what is the motivation of spectral analysis. Similarly, although the paper includes justification of the max-alignment view, the explanation remains largely heuristic. It does not provide an analytical argument or formal characterization of when or why this leads to better Bayesian optimization performance. As a result, the motivation still feels conceptually underdeveloped and difficult to follow for readers without deep kernel-theoretic background.

### Questions
Repeat from the weakness section:
1. Why PSD projection matters?
2. Why consider a spectral analysis?
3. The authors can explain more about how kernel validity relates to PSD properties, and why “valid covariance” matters in GP inference, and what specific advantages “max alignment” brings in the Bayesian optimization setting (e.g., in terms of acquisition behavior, uncertainty modeling, or convergence properties).

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
The paper introduces a Gaussian process kernel that handles symmetries (invariance over a group action), for the purpose of Bayesian optimization. One approach is to average the kernel over the symmetries. This paper proposes instead taking a max, which is not necessarily PSD, so then a straightforward PSD projection is included as well.

The paper does a theoretical analysis that provides strong motivation for the use of the max rather than the average over the group for the kernel. Empirical results show that this new kernel performs much better than baseline approaches for handling symmetries. Spectral analysis shows that existing generalization bounds do not predict the improved performance, and the discussion raises ideas for why that might be.

### Strengths
* The paper is very clear and is well written.

* The paper addresses an important issue (the detrimental effect of symmetries on GP modeling with standard kernels, and thus on BO performance). The core idea to take the max instead of the average over group orbits is not novel in the general case, but its application to Bayesian optimization is novel, and the paper provides a significant amount of useful analysis to go along with this that make it, in my mind, a very original work.

* I especially appreciated the strong theoretical motivation in Section 3.

* The paper seems particularly appropriate for ICLR as it is about representation learning with symmetries.

### Weaknesses
In Section 2.2 the paper describes data augmentation as "a popular way to enforce symmetry." I certainly appreciate that it is intractable in many settings, but by virtue of being popular, it should be compared to in this work. In Ackley2d it could be done directly, since |G| is only 8, and so with 50 total iterations, it becomes a lot of data but still feasible. On other problems, the popular strategy would be to randomly sample a few (still 8 perhaps?) items from G and just augment with those. These are pretty easy to implement and would strengthen the results of the paper significantly. It should come paired with including results showing wall times; the paper suggests that k_avg and k_max will have similar wall time, but it does not actually compare them both to k_b, and so it isn't clear if increasing the amount of data by 8x will make k_b slower than k_max and k_avg.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
The authors claim to contribute:
- Introduce a PSD projection of the max-alignment kernel over a symmetry group to avoid typical kernel averaging strategies, which can dilute high-similarity alignments when the symmetry group is large. This kernel is a valid GP kernel and doesn’t change the asymptotic cost of GP inference, so can be used for BO when the objective function is invariant under a known symmetry group.
- Their max kernel behaves as one expects on toy problems and does really well on standard BO test cases in comparison to the average kernel, which is the closest invariant kernel in the literature so far. It also outperforms standard kernels.

### Strengths
- Paper is overall well written, clear, and concise. It outlines relevant related work and explains why their method might be preferred to each, situating their method well within the literature. Exposition of method is clear.
- Method seems general—can be used with any base kernel and be PSD thanks to their construction. Can be used with any symmetry group, just need to maximize over orbits. This operation scales quadratically with the size of the symmetry group but this is a problem for previous invariant kernels too (e.g., average kernels).
- Clear 1-D examples showcase how the max kernel more accurately models the objective than the average kernel.
- Regret is impressive compared to base and average kernels.
- Compares findings with spectral analyses and finds a gap, and suggests new ways to explain this gap, which may lead nicely into future work.

### Weaknesses
- I would appreciate more real-world BO test cases that are group invariant beside the one WLAN problem, to demonstrate that this is actually an important innovation for applications people care about. More argument about the relevance of this area in the intro, e.g., examples, could also help.


General feedback for improving paper: I think the intro is a little abrupt and a smoother motivation of why we care about max alignment before introducing notation and the PSD projection of the max kernel would help the reader.


I strongly recommend acceptance — the paper is well-written, introduces a max-alignment kernel which is not in the BO literature yet (to best of my knowledge) by approximating it with a projection to make it PSD. This leads to really good results on both toy, interpretable settings and on standard (group invariant) BO test functions. It seems clear that this work would be interesting to the BO community.

### Questions
I’m not an expert; why is it difficult to identify a suitable fundamental domain to restrict the search space to, if one knows the symmetry group? Is there an easy example to demonstrate this? It would be good to explain this to readers less familiar with symmetry (which includes much of the BO community).

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3