# Local SGD and Federated Averaging Through the Lens of Time Complexity

- Avg Score: 4.80
- Decision: Reject
- Scores: 8, 6, 4, 2, 4

## Abstract
We revisit the classical Local SGD and Federated Averaging (FedAvg) methods for distributed optimization and federated learning. While prior work has primarily focused on iteration complexity, we analyze these methods through the lens of time complexity, taking into account both computation and communication costs. Our analysis reveals that, despite its favorable iteration complexity, the time complexity of canonical Local SGD is provably worse than that of Minibatch SGD and Hero SGD (locally executed SGD). We introduce a corrected variant, Dual Local SGD, and further improve it by increasing the local step sizes, leading to a new method called Decaying Local SGD. Our analysis shows that these modifications, together with Hero SGD, are optimal in the nonconvex setting (up to logarithmic factors), closing the time complexity gap. Finally, we use these insights to improve the theory of a number of other asynchronous and local methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies Local SGD and its variants. It begins by showing a lower bound for the canonical Local SGD algorithm. Then, the authors proposed several new variants of Local SGD to avoid the term in the rate of the canonical Local SGD. Finally, they propose Decaying Local SGD with new stepsize rules.

### Strengths
The paper first establishes a lower bound on Local SGD of order $\Omega (\epsilon ^{-3/2})$. This term seems to appear in most analyses of Local SGD, and it is now justified by this lower bound result. 

The authors then investigated two stepsizes and dual Local SGD, and proposed a $\sqrt {n}$-times larger global aggregating stepsize in Local SGD. Though the idea of using a different, larger global stepsize is not new and already appears for instance in (Karimireddy et al., 2020), the analysis in this paper is the refined version over (Yang et al. (2021) and Karimireddy et al. (2020)). 

They finally proposed a new Decaying Local SGD method, with a new stepsize rule. This can lead to much faster convergence in some cases, and can also be generalized to asynchronous and tree-graph problems. 

The paper addresses several non-trivial technical issues of Local SGD. The writing is clean and easy to follow. The results are described in a reasonable way. 

I checked the proof of the lower bound of Local SGD, and I looked at the analysis of Dual Local SGD. I did not spot any proof mistake.

### Weaknesses
The only weakness is the experimental results seem to be less convincing. I understand that Local SGD is a very robust baseline method. But I still expect the Dual Local SGD can be faster under the conditioning that the additional $O(\epsilon ^{-3/2})$ term dominates. 
It would be great to show in some experiments that, at least in some conditionings, the Local SGD algorithm really gets slowed down in practice due to this term. This would also help to better motivate the entire story of the paper. Otherwise, now the paper looks like addressing a merely theoretical issue.

### Questions
Is it reasonable to add a Minibatch SGD in the experiment plots? The algorithm proposed in this paper has the same theoretical rate as Minibatch SGD, but it would be great to know how they compare to each other in practice. 

This is not really a question but a small suggestion. It seems clearer to rewrite the lower bound of (Tyurin and Richtárik, 2024) in Table 2 as $\Omega \left( \min \left\\{ \tau \frac {L \Delta} {\epsilon}, h \frac {L \sigma^2 \Delta} {\epsilon^2} \right\\} + h \left( \frac {L \Delta} {\epsilon} + \frac {L \sigma^2 \Delta} {n \epsilon^2} \right) \right)$. Now, it seems clearer that the red term in your lower bound is the average over the minimum terms in their algorithmic-independent lower bound.

### Soundness
2

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
This paper shows that while Local SGD (FedAvg) seems efficient in theory, it’s actually slower in real time than simpler methods like Minibatch or single-worker SGD once communication costs are included. The authors fix this with Dual Local SGD and Decaying Local SGD, which use better scaling and adaptive step sizes to reach optimal time efficiency. Experiments on MNIST and CIFAR-10 confirm these tweaks make training faster and more stable.

### Strengths
1. This paper analyzes Local SGD and FedAvg through time complexity, not just iteration count, offering a more realistic view of efficiency.

2. Challenges the common belief that local updates always save time, providing a clearer picture for real-world federated training.

### Weaknesses
1. In real distributed or federated systems, communication often overlaps with computation, and network latency can vary significantly. This makes the proposed time complexity framework mathematically elegant but not fully representative of real wall-clock behavior.

2. All experiments are conducted on small benchmarks (toy problem, MNIST, CIFAR-10) with simulated workers. The paper does not demonstrate wall-clock speedup or test on realistic federated environments (e.g., cross-device, mobile, or edge networks).

3.The “decaying local step” rule assumes that increasing local step sizes never harms stability up to logarithmic factors, but this is not proved for strongly nonconvex objectives. The bound holds only asymptotically with carefully tuned constants. In practice, large step sizes can cause divergence, contradicting the theoretical “never worse” claim.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work studied local SGD in terms of time complexity. It shows that under a certain computation time framework, canonical Local SGD can be strictly suboptimal in time complexity compared to Minibatch SGD and HeroSGD. To overcome the issue, the paper proposed Dual Local SGD and proved they achieve optimal complexities (up to log factors) under the same model. Experiments demonstrate the effectiveness of the proposed algorithms.

### Strengths
1. Time complexity perspective is a meaningful metric in practice, which is more useful in algorithm design.
2. The observation that Local SGD is strictly worse (under the claimed computation framework) is interesting.
3. The proposed fix for Local SGD is simple and elegant, with a solid theoretical result supporting their effectiveness.
4. The writing of the work is clear in general, the flow of the story is easy to follow.

### Weaknesses
1. Assumption 1.1 characterized a fixed computation model, which is not fully verified yet. For example, here the gradient computation (for local SGD) always take only one sample, for small batch scenarios, some fixed time cost (for things like kernel launch) may take effect or even dominate the computation. Similarly for the communication, there should be some existing models in that community (like Postal model or other latency models) which characterized the communication/broadcast time more accurately. It would strengthen the paper if the authors could clarify the intended scope of this model.
2. Confusing lower bound results. From the proof, the results are a combination of lower iteration complexity bounds and Assumption 1.1 (lower iteration bound $\times$ execution time), but if I understand it correctly, Assumption 1.1 highlights "at most" for each execution, that means 'execution time' $\leq$ something, which I think is confusing to imply a lower bound (bound $\geq$ something). So to improve the clarity, I think Assumption 1.1 can be further clarified, e.g., the execution time are just exactly $h$ (and $\tau$).
3. The experiment part seems to still use the "rounds" in the $x$-axis, but the work highlight time complexity, maybe it is better to attach more results in terms of wallclock time. Also the experiment does not echo the observation that Local SGD is strictly worse than Minibatch-SGD and Hero-SGD, it would strengthen the work if there are corresponding empirical results.
4. Writing. Missing $O(\cdot)$ (and $\Omega) notations for many expressions throughout the work.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper reinterprets existing convergence analyses of Local SGD in a time complexity model, aiming to demonstrate that it cannot beat the time complexity of simple baselines, such as mini-batch SGD and single-machine SGD. The paper then proceeds to analyze a two-step-size version of Local SGD, and shows that it can match the convergence guarantee of mini-batch SGD in the homogeneous non-convex setting. Finally, it analyzes another step-size schedule for SGD with two step-sizes, showing a similar convergence rate. Experiments on MNIST and CIFAR-10 are performed to compare Local SGD and its two-step-size variants.

### Strengths
The point of view of time complexity is indeed interesting, as it can potentially highlight that regimes where we previously thought Local SGD could beat the baselines actually do not exist once we account for computation and communication costs. Thus, morally, the paper is asking interesting questions.

### Weaknesses
Some of the proofs in the paper are incorrect, and even the ones that are correct are either not entirely novel or not particularly surprising. 

- Theorems 2.3 and 2.5 are wrong. First of all, since these results are derived from upper bounds on Local SGD's convergence, they do not represent time complexity lower bounds. Thus, the theorem statements themselves are wrong. The paper acknowledges this in the preceding text, but still chooses to write the incorrect version of these theorems. Furthermore, the proof of Theorem 2.5 is incorrect because it overlooks specific terms in the upper bounds when deriving the theorem. If the results being used were lower bounds, this would have been fine, but these results are upper bounds, and hence ignoring some terms would not give a correct picture: ignoring these terms makes the upper bounds invalid. Since these are the data heterogeneity terms, this is further potentially misleading in the heterogeneous setting. 

- In the convex setting, i.e., when comparing Theorems 2.1 and 2.2, we do not have the above issue, as the authors do look at the lower bound due to Glasgow et al. for Local SGD. Still, I am not convinced that conclusion is correct in all regimes as the authors claim. Please answer my question below regarding this. 

- The paper is inconsistent about stating and discussing the resuls for Hero SGD/single machine SGD. For instance, in Algorithm 3 we look at an algorithm with $R$ iterations, as opposed to $KR$ iterations which would be the correct baseline to compare to. But when discussing the algorithm's guarantee in Table 1, the authors mix the different terms from these two algorithms: the optimization term seems to follow Algorithm 3, while the noise term for Hero SGD seems to suggest $KR$ iterations. This is most probably a typo or misunderstanding about which single machine baseline is the correct comparison. Surprisingly in Theorem 2.2, the authors revert back to the correct baselines, with $KR$ iterations. This is all very confusing, and would likely mislead an uninformed reader. I suggest describing in the pseducode only the variant with $KR$ iterations, because this equalizes the compute each machine actually does in the parallel algorithms (Local SGD and mini-batch SGD). 

- Morally, we don't need to look at time complexity to conclude that in the homogeneous setting Local SGD has a limited utility. The seminal result of [Woodworth et al.](https://arxiv.org/abs/2102.01583) already shows that accelerated mini-batch SGD and accelerated single machine SGD together are min-max optimal. Yes, this leaves open the question whether a similar dichotomy can be shown for non accelerated algorithms, but it is not a very interesting gap. Even if in principle, unaccelerated Local SGD could match the min-max optimal rate, that would still not explain its practical effectiveness. More than anything, their lower bound doesn't show that Local SGD is a bad algorithm, but that rather the homogeneous convex smooth setting is not "rich" enough to actually benefit from local update steps. Infact, this is the motivation for studying mildly heterogeneous settings, where the benefit of collaboration is not limited to just variance reduction in the noise term, but collaboration can genuinely help clients explore the loss landscape. The results of [Patel et al.](https://arxiv.org/abs/2405.11667) highlight this, as they show that the dichotomy pointed out by Woodworth et al. does not really hold in the heterogeneous setting, because single machine SGD is no longer a reasonable baseline.  Finally, this highlights that several important references are missing in this discussion. I would encourage the authors to re-write with a more through inspection of what the current results imply in the accereated and non-accelerated settings. The thesis by [Patel](https://arxiv.org/abs/2507.00195) does contain many useful missing references. 

-  Theorem 3.1 is not interesting. It makes a point, which has been underlined again and again (by some of the works cited in the paper itself): Local SGD with two step-sizes can recover mini-batch SGD. It is not any more insghtful to show this for a specific choice of step-sizes. Granted, the step-sizes for which Theorem 3.1 shows this are not extreme, but even the [SCAFFOLD paper's Theorem 1](https://arxiv.org/pdf/1910.06378) already shows this. What is the novelty here? What would be interesting is if the authors could identify a theoretical setting, where Local SGD with two step-sizes not only matches but beats both mini-batch and single-machine SGD. Similarly, the fact that there is a gap between Local SGD with one and two step-sizes was also formalized again by the lower bound instances of [Patel et al.](https://arxiv.org/pdf/2405.11667) (see their Remark 5). 

- It is unclear what is the point of Theorem 4.1, it attains the same convergence rate as Theorem 3.1. If the point is to show that a bigger step-size schedule does not hurt or helps in some way, then clearly the theoretical setting considered in the paper is not rich enough to demonstrate that. Empricially, previous works have already investigated this quite extensively, for instance [Charles and Konecny](https://arxiv.org/abs/2007.00878) do this on several benchmarks. So what would be interesting is to consider some theoretical setting, where such a scheme could have provable optimization benefits.

### Questions
While the proofs of Theorems 2.1 and 2.2 look correct to me, I am not convinced that the time complexities of mini-batch SGD/single machine SGD are always at least as good as Local SGD's time complexity. If this point were being made for accelerated algorithms, I would be convinced, given the results of Woodworth et al. in the oracle complexity model. However, for non-accelerated methods, we know that there exists a regime where the convergence rate of Local SGD is strictly better than both mini-batch and single-machine SGD. If such a regime indeed exists, there must be at least some regime of $\tau$ and $h$, where even the time complexities of Local SGD are strictly better. What am I missing? Can the authors give an explicit proof that the complexity in 2.1 can never strictly improve over the one in Theorem 2.2? This is only alluded to in the contributions section, but not explicitly shown. 

For context, about which regime I am looking at, note that when $\frac{LB}{\sqrt{KR}} < \sigma < \frac{LB\sqrt{MK}}{\sqrt{R}}$, then the dominant term in the convergence rate of the three algorithms is as follows:
- Single Machine SGD: $\frac{\sigma B}{\sqrt{KR}}$ which is strictly bigger than $\frac{LB^2}{KR}$ due to the first restriction on the range of $\sigma$ above;
- Mini-batch SGD: $\frac{LB^2}{R}$ which is strictly bigger than $\frac{\sigma B}{\sqrt{MKR}}$ due to the second restriction on the range of $\sigma$ above;
- Local SGD: $\frac{LB^2}{KR} + \frac{\sigma B}{\sqrt{MKR}}$, which, based on the best two bullets, is strictly better than both the previous algorithms.

Due to this, in the above regime, there must exist some regime of $\tau, h, \epsilon$ where Local SGD's time complexity is also better. If I am missing something, I am happy to be corrected.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper re-evaluates Local SGD / FedAvg through time complexity (wall-clock) rather than iteration complexity. Under a simple compute/comm model (per-grad cost $h$, per-sync cost $\tau$, it argues canonical Local SGD is never better—and can be strictly worse—than Minibatch SGD or “Hero” (single-worker) SGD. It then proposes (i) Dual Local SGD (two step sizes, correcting the global aggregation scaling) and (ii) Decaying Local SGD (larger, decaying local steps), and shows these match the optimal time complexity (up to log factors) achieved by Minibatch/Hero in nonconvex settings.

### Strengths
- The paper is generally well written and well structured; algorithms and tables are clear. The Contributions section and Table 2 succinctly convey the main message.

- Reframing the discussion on Local SGD in terms of time complexity—with explicit compute and communication costs ($h$, $\tau$) exposes a gap between iteration-rate folklore and wall-time behavior.

### Weaknesses
- The claimed contribution regarding the “correct” scaling for Canonical Local SGD—namely $1/\sqrt{n}$ rather than $1/n$—does not seem particularly novel. Several prior works (e.g., Khaled et al., 2020; Woodworth et al., 2020) already employ step sizes $\eta = \Theta(\sqrt{n})$, which, when combined with $1/n$ averaging, effectively yields an overall $O(1/\sqrt{n}​)$ scaling.

- The stated time-complexity lower bounds for Local SGD (Theorems 2.1, 2.3, 2.5, 3.1) become trivial in the edge case where the per-gradient compute cost $h \rightarrow 0$ (i.e., computation is negligible relative to communication), collapsing to zero. 

- Much of the analysis appears to closely follow existing literature (e.g., Tyurin & Richtárik, 2023; Tyurin & Sivtsov, 2025), which makes the technical novelty feel incremental.

- The numerical section offers limited insight. The main toy example is a one-dimensional strongly convex problem, yet Figure 1 shows very slow convergence even with a logarithmic y-axis, and the methods converge to different objective values. Figure 2 on standard ML tasks exhibits similar issues.

### Questions
- Theorem 2.5 presents results for both Local SGD and SCAFFOLD. Because SCAFFOLD uses control variates, its rates typically avoid explicit dependence on data heterogeneity, whereas Local SGD usually does depend on this. In your result, this dependence is not visible. Could you clarify: (i) whether you absorb heterogeneity into other constants, and (ii) how the heterogeneity parameter would appear if stated explicitly? A version of the theorem that makes the dependence explicit would help.

- In Theorem 3.1, the phrase “when combined with Hero SGD” is ambiguous. Do you mean a hybrid algorithm that interleaves Local SGD with a single-worker (“Hero”) phase, a fallback schedule, or simply a comparison/baseline in the bound? Please spell out the precise algorithmic composition and schedule implied by “combined”.

### Soundness
3

### Presentation
3

### Contribution
2
