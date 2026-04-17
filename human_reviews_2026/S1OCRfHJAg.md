# Frequency-Domain Better than Time-Domain for Causal Structure Recovery in Dynamical Systems on Networks

- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
Learning causal effects from data is a fundamental and well-studied problem across science, especially when the cause-effect relationship is static in nature. However, causal effect is less explored when there are dynamical dependencies, i.e., when dependencies exist between entities across time. In general, it is not possible to reconstruct the causal graph from data alone. The conventional static causal structure recovery algorithms employ tests such as the Fischer-z test and the chi-square test to assess the conditional independence (CI) of data which forms the basis for recovering Markov Equivalent Graphs (MEGs) wherein causal structure can be recovered partially. For data that are dynamically related, multivariate least square estimation, based on Wiener Filters (WFs) relying on second order statistics for estimating a data stream from other streams, provides a means of recovering influence structures of the directed network underlying the data. Here, WF based projections can be determined in time-domain or in frequency-domain; the question this article sets out to answer is which is better? Here, we obtain concentration bounds on the accuracy of the WF estimation in both time and frequency-based approaches. Exploiting the computation speed of Fast Fourier Transform (FFT), we establish that the frequency domain provides distinct advantages. Moreover, frequency domain projections involve complex numbers; we establish that the phase properties of the resulting estimates can be effectively leveraged for better recovery of the MEG in a large class of networks; the time-domain has no analogue of phase. Thus we report the "Wiener-Phase" algorithm provides the best accuracy as well as computational advantages. We validate the theoretical analysis with numerical results. Performance comparison with state of the art algorithms are also provided. Further, the proposed algorithms are validated on a real field dataset known as the "river-runoff" dataset collected from the online repository of CauseMe, and on measurement data from transistor based circuits.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The work compares time- versus frequency-domain estimation of multivariate Wiener filters for causal structure recovery under a linear dynamical influence model, proves computational gains via FFT, provides finite-sample concentration bounds for both domains, and proposes Wiener-PC (CI testing via Wiener coefficients) and Wiener-Phase (phase/imaginary-part–based skeleton and collider discovery), validated on synthetic and real river-runoff data to support the claims.

### Strengths
- S1: Uses the zero/nonzero support of multivariate Wiener coefficients as a principled CI surrogate, with FFT yielding explicit complexity gains over time-domain least squares, and finite-sample bounds provided in both domains.

- S2: Introduces Wiener-Phase, which leverages the imaginary part/phase of frequency-domain Wiener filters to recover the skeleton and identify colliders under phase assumptions, with strong empirical scalability.

- S3: Provides broad empirical evidence on synthetic node networks and a real-world dataset, showing superior accuracy and runtime trade-offs versus multiple baselines.

### Weaknesses
- W1: Key guarantees for Wiener‑Phase rely on structural assumptions (e.g., equal phase of all incoming edges and a nonzero imaginary part of transfer functions), which are somewhat strict.  

- W2: The approach hinges on heuristic thresholding of Wiener coefficients, tuned via sparsity/degree constraints rather than statistically calibrated procedures, which may make results sensitive to sampling and frequency discretization.  

- W3: Baseline coverage omits widely used frequency-domain causal discovery families such as spectral Granger variants, making it harder to position the proposed methods relative to established spectral measures beyond time-domain Granger.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates causal structure recovery in linear dynamical systems using Wiener Filter (WF) based approaches. The authors compare time-domain (TD) and frequency-domain (FD) methods for computing WF coefficients and propose two algorithms: Wiener-PC (extending PC algorithm with WF-based conditional independence tests) and Wiener-Phase (exploiting phase properties of WF for efficient MEG reconstruction). The work provides theoretical analysis including concentration bounds and sample complexity for both approaches, demonstrating computational advantages of FD methods.

### Strengths
1: The Wiener-Phase algorithm cleverly exploits the phase properties of frequency-domain WF coefficients to efficiently identify colliders and reconstruct MEGs, which has no analogue in time-domain approaches.

2: The evaluation includes synthetic data with controlled network structures, scalability analysis up to 50 nodes, and validation on real-world river-runoff dataset, showing consistent advantages of FD approaches.

### Weaknesses
1: The Wiener-Phase algorithm relies on two strong assumptions (phase alignment of incoming edges and non-zero imaginary components) that significantly limit its applicability. While the authors claim these apply to "a large class of networks," the conditions are quite specific and may not hold in many real-world scenarios.

2: The comparison primarily focuses on traditional static methods (Fisher-Z, chi-square tests) and Granger causality, missing more recent state-of-the-art causal discovery methods for time series data.

3: I'm unclear on where the distinction between static and dynamic approaches is specifically presented in your algorithm, and your specific contribution is unclear.

4: While the paper shows results for up to 50 nodes, the computational complexity analysis suggests the methods may not scale well to truly large networks due to the combinatorial nature of PC-based approaches. Please increase the credibility and workload.

### Questions
1: How can practitioners verify whether Assumptions 1 and 2 hold for their specific applications? What happens to algorithm performance when these assumptions are violated beyond the limited robustness study provided?

2: Given the O(n^(q+3)) complexity of Wiener-PC, how does the method perform on networks with hundreds or thousands of nodes? What are the practical limits of the approach?

3: While Lemma A.1 suggests any frequency can be used, how does the choice of frequency affect performance in practice, especially in noisy or finite-sample settings?

### Soundness
2

### Presentation
2

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
The authors tackle the problem of causal structure learning for systems where causal dependencies are non-static and dynamical in nature. They  provide a theoretical framework based on Wiener Filters (WFs) and Fast Fourier Transform (FFT) to introduce a novel method for deciphering non-static causal dependencies of dynamical systems from time-series data. They empirically validate their proposed approach on synthetic toy systems of 20 variables (nodes) and a real data (river-runoff) setting. This work provides a new perspective on causal structure learning of dynamical systems using frequency domain WFs.

### Strengths
- The authors introduce a novel, computationally efficient, method for causal structure recovery of non-static / non-stationary systems (systems with dynamic causal dependencies).
- The paper provides a novel way of thinking about causal structure learning of dynamical causal dependencies from time-series data; i.e. causal structure learning in the frequency domain. This insight leads to the aforementioned improvements in computational efficiency.

### Weaknesses
- Although in general this work is reasonably well written, and the authors did a good job at presenting the many nuanced details present in this work, this paper can still be hard to follow at times. 
- The baselines in the empirical experiments seem to primarily be methods which operate in the static (stationary) setting. It would be useful to compare to some existing works that approach the problem of dynamic and non-stationary dependencies. To give some examples: [1, 2].
- Similarly, empirical experiments are quite limited. Although I believe this work presents comprehensive theoretical backing for their approach, which provides a reasonable degree of contribution already, the authors could consider some of the datasets/systems from [1, 2] to add to their  experiments section. Or perhaps, simulating data in some of the example systems the authors provide in Appendix C.6 could also be considered for addition experiments. I believe additional empirical experiment would help further strengthen this work.

### Questions
- In practical applications, how large does $N$ need to be to ensure reasonable recovery of the MEG? 
- On lines 224-225: "For the settings with large number of samples (large $T$) or with longer delays (large $L$) improvements are significant". What are the practical settings where we would apply the proposed method and where there is sufficiently large $L, T$ such that it would yield competitive performance?
- Is the proposed approach practical for high dimensional systems (large $n$)?
- For the synthetic experiments in section 6, are the dynamical / non-stationary dependencies of the ground truth process/DAG random? Could the authors provide some additional clarification here?
- Lines 422-424: "Bounds on sparsity metrics such as average, maximum, and minimum degree are typically available ...". In practical applications and scientific applications, this is not necessarily know apriori, and thus, is not a reasonable assumption to make. Could the authors provide some examples supporting this claim? 
- Could the authors provide further details for how the Granger baseline is implemented? Further, have the authors also considered a very simple correlation baselines? i.e. compute the correlation between samples across adjacent time-points and use this as the estimated DAG?
- The CS metrics seems like a weird choice. Why not consider something like average prevision / area under the average precision curve? You can compute this without thresholding, and it would provide a metric that captures some sort of notion of accuracy across "all thresholds".


Minor comments:

- I would define the acronym FFT (Fast Fourier Transform) earlier in the text (in fact I don't think its defined anywhere currently).
- Notation for $T$ slightly over-loaded. It is used to indicate transpose and in the number of time-series samples. 

References:

[1] Huang et al. "Causal discovery from heterogeneous/nonstationary data." JMLR. 2020.

[2] Fujiwara et al. "Causal discovery for non-stationary non-linear time series data using just-in-time modeling." Conference on Causal Learning and Reasoning. PMLR, 2023.

### Soundness
3

### Presentation
2

### Contribution
2
