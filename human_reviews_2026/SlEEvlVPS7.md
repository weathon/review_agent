# Bridging the Gap Between Homogeneous and Heterogeneous Asynchronous Optimization is Surprisingly Difficult

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Modern large-scale machine learning tasks often require multiple workers, devices, CPUs, or GPUs to compute stochastic gradients in parallel and asynchronously to train model weights. Theoretical results typically distinguish between two settings: (i) the homogeneous setting, where all workers have access to the same data distribution, and (ii) the heterogeneous setting, where each worker operates on different data distributions. Known optimal time complexities in these settings reveal a significant gap, with far more pessimistic guarantees in the heterogeneous case. In this work, we investigate whether these pessimistic optimal time complexities can be overcome under different assumptions. Surprisingly, we show that improvement is provably impossible under widely used first- and second-order similarity assumptions for a broad family of algorithms. We then turn to the interpolation regime and demonstrate that the weak interpolation assumption alone is also insufficient. Finally, we introduce a minimal combination of irreducible assumptions, strong interpolation and the local Polyak-Lojasiewicz condition, to derive a new time complexity bound that matches the best-known result in the homogeneous setting, without requiring identical data distributions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores the gap between time complexities under homogeneous and heterogeneous worker losses for a  class of asynchronous optimization algorithms. These methods distribute the gradient computations across workers until “a sufficient number” of gradients have been computed and then update the iterate, but they differ in how the next iterate is computed. In this paper, the authors investigate under which assumptions one can close the gap between the lower bounds for homogeneous and heterogeneous worker losses.

I find the paper interesting and important when it comes to expanding our theoretical understanding of the time complexities of the particular classes of methods under consideration. However, the scope is limited (weighted gradient methods) and the analysis does not make any advances on the algorithmic side (it simply finds conditions under which it is possible to get the desired compute-time dependence, relying on a mixing already been proposed in the literature).

### Strengths
1. The authors address a well-defined question (under what assumption can a weighted gradient method of the type studied in the paper attain optimal worker time dependence thus far only associated with homogeneous worker losses) and present a sound a convincing argument with constructive “difficult functions” for each considered scenario that shows that the lower bounds cannot be attained. 
2. The overview of results in the area is comprehensive and clear. 
3. The final result is clear: under a set of strong conditions, Rennala SGD attains the desired  compute time dependence, and no other weighting can improve its performance.

### Weaknesses
The paper would have been easier to read with at least the very basic notation (e.g. \tilde{\Theta}) in the main document. The fact that the author frames a lot of the discussion around Renala vs Malania SGD, instead of the underlying ideas of the methods (equally weighted gradients vs importance sampled gradients/unbiased worker gradient estiamtes) makes it more difficult to keep up a good intuition while reading the paper. 

I am also not fully convinced by the framing of the paper. It would be good if the authors could provide more intuition about what they are trying to do and why? Because now I come away from reading the paper with the feeling that this is more of a “robustness analysis” of Rennala SGD (how much can we wiggle the assumptions from homogeneity and still get the same compute time dependence in the convergence result). Clearly, this is not the way the authors wants us to read the paper (cf. the very last paragraph of Section 1), but I think they need to make a slightly stronger argument for why their result is more than this. For example, it is clear that they want to attain the harmonic mean dependence on worker compute times, but why this particular form? Presumably, you would like to avoid waiting for the slowest worker, while being able to use very fast compute nodes to reduce variance and progress to the next iterate. But is there something fundamental with this dependence? Second, in general, there should be little hope of not waiting for the slowest worker, unless it is not needed (strong interpolation), so the results are not very surprising. It is not clear to me why the authors were hoping that gradient similarity or weak interpolation would help.

### Questions
1. The class of methods is rather limited. What is the reason to only consider weighted SGD and no other techniques?
2. Can you elaborate on what is fundamental with the  harmonic mean compute-time dependence in the heterogeneous setting?
3. Can you elaborate on why you would expect that gradient similarlity or weak interpolation would help?
4. In your model for heterogeneity, you assume that the variances of worker subgradients are identical. When is that reasonable in practical ML settings? What would happen if you allowed them to be different?
5. For the simulations, I could not find how you generated the \tau_i’s, and how they vary in magnitudes and time (you emphasize in the introduction that your setting includes non-constant computation times)
6. It seems that Theorem E.4 is new. Is that true? Why was it relegated to an appendix?

### Soundness
4

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
This paper considers the problem of distributed optimization where n workers/clients are working in parallel asynchronously to optimize an objective function and find the minima, drawing motivation from distributed and federated ML. Each worker has their own function f_i, dataset D_i, and are able to compute their local stochastic gradient on f_i(x,D_i). The global objective is a sum of the expected values of the local objectives. The paper specifically revisit the mathematical analysis of asynchronous SGD under the "heterogenous" case, i.e., where each client has a different data distribution, attempting to improve the time complexity under alternate mathematical assumptions.
They first show that the existing pessimistic bounds are indeed tight and cannot be improved under certain classes of assumptions considered. Then finally, they introduce an alternate assumption under which the guarantees for the "heterogenous" case matches the "homogenous" case. Primarily a theoretical paper with several takeaways, no experiments.

### Strengths
-- The paper first introduces a weighted SGD as a general way of writing the distributed SGD solution. The two specific cases Rennala SGD (optimal in homogenous) and Malenia SGD (optimal in heterogenous) come out as special cases of the general weighted SGD solution. This is helpful since now the weighted SGD can be directly analyzed.

-- First, they show that convergence if Malenia SGD is challenging. Even first order or second order similarity between the functions across all the clients do not necessarily help. I greatly appreciate this detailed analysis of a negative result, highlighting the mathematical insights on why it doesn't work.

-- Then they consider an alternate set of interpolation assumptions under which it is possible to improve the time complexity of Malenia SGD. 

-- The takeaways of the theoretical results are quite nice and insightful.

### Weaknesses
-- It is not clear how the new set of assumptions compare with the previous assumptions. Which assumptions are weaker or stronger? Is it possible to give intuition on what these new assumptions mean, perhaps using a toy example on the functions f_i?

-- For instance, are there toy functions/ classes of functions that satisfy the old assumptions but not the new? Vice versa, are there functions that satisfy the new assumptions but not the old?

-- Theorem 3.1 gradient condition: Do you need an expectation $\Delta f(x;\xi)=\Delta f(x)$

-- The table on page 7 with the contributions is nice. But the first column can be clearer about homogenous/heterogenous case to make the contributions clearer.

-- Missed several related works on asynchronous SGD
[1] Xiangru Lian, Yijun Huang, Yuncheng Li, and Ji Liu. Asynchronous parallel stochastic gradient for nonconvex optimization. In Advances in Neural Information Processing Systems, pages 2737–2745, 2015.
[2] Wei Zhang, Suyog Gupta, Xiangru Lian, and Ji Liu. Staleness-aware async-sgd for distributed deep learning. In International Joint Conference on Artificial Intelligence, pages 2350–2356. AAAI Press, 2016.
[3] Dutta, S., Joshi, G., Ghosh, S., Dube, P., & Nagpurkar, P. Slow and stale gradients can win the race: Error-runtime trade-offs in distributed SGD. In International conference on artificial intelligence and statistics (pp. 803-812) 2018.

-- Also see the citations therein

-- Abstract is too domain-oriented with specific jargon and is not very accessible for a broader audience.

-- Though the contribution is theoretical, and experimental results are not necessary - But I was curious if an asynchronous algorithm is implemented on say a strongly convex loss function or one that satisfies the necessary assumptions, how well do these bounds align with practice? Are there empirical studies on Rennala SGD (optimal in homogenous) and Malenia SGD (optimal in heterogenous) and how good do the time complexity bounds align with practice?

### Questions
Could you compare the two sets of assumptions, and what do they mean intuitively for real functions? 

Is one stronger or weaker?

What are some toy functions that satisfy one but not the other?

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
In this work, the authors study the complexity of heterogeneous asynchronous optimization. In particular, given workers with non-constant computation time of gradients that is bounded for worker $i$ by $\tau_i$, the authors investigate the optimal complexity achievable with an asynchronous method that they call Weighted SGD. In particular, two previous methods for asynchronous optimization known as Rennala SGD and Malenia SGD are special cases of Weighted SGD. The authors prove a series of negative results for this method under various combinations of function/data similary/interpolation and convexity/PL-condition as well as smoothness.

### Strengths
1. The work is presented really well. I found it very accessible and in fact an enjoyable read. The main takeaway messages are nicely emphasized and all assumptions are clearly stated.
2. Asynchronous optimization is a sensible approach to distributed systems, especially when hardware failures are possible, which can be very important given the never ending growth of large-scale clusters.

### Weaknesses
1. I'd say the main weakness of the paper is the significance. The studided methods were proposed and analyzed in prior literature. The non-convergence result is mostly trivial. Moreover, the results for first-order and second-order similarity are only applicable to Weighted SGD, which in itself is not a particularly interesting method.

2. I'm also concerned about how the results for the similary are presented. The authors wrote:
> One might expect that when both $\delta\_1$ or $\delta\_2$ are small, it would be possible to exploit the similarity and design a method with smaller variance and better dependence on $\{\tau\_i\}$. Surprisingly, it is not the case

But the authors don't actually show that no other method exists in this setting, they only present a counterexample to Weighted SGD with fixed weights. I find this kind of presentation rather misleading, in particular the way it is written in Takeaway 3 and Takeaway 4.

3. I also feel the topic of the paper is drifting from the practical methods. Malenia SGD requires each worker to submit at least one gradient, which is not something we would use in practice when a node failure is possible. It makes me question if the methods presented in this paper are truly possibly useful. I suspect that in practice it is much better to proceed without weighting for all nodes to submit at least a gradient.

### Questions
1. Do the authors believe Malenia SGD is a practical method? What happens if there is a failure?

2. More generally, what kind of setting are the authors motivated by? It doesn't appear that we have data heterogeneity in distributed computation on clusters as we can simply duplicate the data.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the difference in time complexity between heterogeneous and homogeneous settings. It aims to improve results in heterogeneous cases, narrowing the gap and aligning the complexity more closely with that of homogeneous problems. The authors introduce a unified framework called Loretta SGD for asynchronous algorithms, which generalizes Malenia SGD and Rennala SGD. They demonstrate that, under a strong interpolation regime and a local PL condition, convex functions with local smoothness can achieve results that match the performance observed in homogeneous settings.

### Strengths
The paper is solid, and the proofs and proposed theorems are logically sound and accurate. The convergence bound and complexity orders are carefully justified under appropriate assumptions, making the theoretical contributions reliable and well-supported.

The submission presents a derivation of the convergence results, aiming to find assumptions under which the performance of Rennala and Melania SGD coincide for homogenous and heterogeneous cases. They identify the PL condition, and strong interpolation would help, and thus derive the convergence in terms of E.

The paper is well-written, but its flow could be improved by adding a simpler discussion after each theorem because it leaves the reader confused about the main idea.

### Weaknesses
However, the submission has limited significance. Empirical results on quadratic optimization demonstrate a performance gap between Rennala SGD and Malenia SGD without interpolation; however, Rennala SGD does not converge to a minimum of the quadratic optimization problem. In the interpolation regime, the gap decreases, and Rennala SGD outperforms Malenia SGD, especially when devices have different computation times. Results on the CIFAR-10 dataset with Resnet-18 are also included. However, experiments with larger datasets and more data heterogeneity are missing. What happens when both variants of Loretta SGD are used on larger image and text datasets? What about the performance gap in those cases?

The submission is incremental compared to existing results. The paper examines how to reduce the gap between the convergence rates achieved by enhancing Malenia SGD's reliance on the arithmetic mean of computation times versus the harmonic mean. It shows that the assumptions of first- and second-order gradient similarity are insufficient to improve this dependence. They then apply the assumptions of weak interpolation and the PL-condition, but still do not achieve an improvement. Finally, the paper concludes that only under the strong interpolation regime, and when local functions satisfy the PL condition, does the time complexity improve.

### Questions
The reviewer identifies the paper from a previous submission, and last time, a query was also raised about the weak interpolation assumption. I am not convinced by the definition of weak interpolation in Assumption 4.1. Interpolation typically means that the model fits the data at each client, meaning the loss or gradient at each sample is zero. I believe the paper refers to the statement, “Formally, interpolation requires that the gradient with respect to each point converges to zero at the optimum, implying that if the function $f$ is minimized at $w*$ and thus $f(w*) = 0$, then for all functions $f_i$ we have $f_i(w*) =0$ in [1]. However, this definition aligns with the strong interpolation assumption presented in Assumption 5.6, rather than the weak interpolation assumption. There is no prior work that discusses a concept called weak interpolation; see the definition of interpolation in [2]. Therefore, I do not understand the need for this additional result with weak interpolation, especially since there is no clear intuition as to why this statement is valid. Interpolation indicates that all functions share common minimizers. If weak interpolation is assumed, then the reverse, namely, that if $x^*$ is a stationary point of each $f_i$, then it is also a stationary point of $f$, is trivially satisfied. Hence, weak interpolation automatically implies strong interpolation. Using other techniques, such as tuning $\gamma$ or incorporating correction terms, can’t we try to bridge the gap? Any discussion on that would make the conversation more well-rounded. Currently, the primary motivation is to determine how to make Rennala SGD work for heterogeneous cases by considering various theoretical assumptions. Does parameter tuning have any effect on this?

**REFERENCES**

[1] Sharan Vaswani, Aaron Mishkin, Issam Laradji, Mark Schmidt, Gauthier Gidel, and Simon Lacoste-Julien. Painless stochastic gradient: Interpolation, line-search, and convergence rates. NeurIPS 2019.

[2] Guillaume Garrigos and Robert Mansel Gower. “Handbook of Convergence Theorems for (Stochastic) Gradient Methods.” (2023).

### Soundness
3

### Presentation
3

### Contribution
3
