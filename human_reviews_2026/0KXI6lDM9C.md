# Proving the Limited Scalability of Centralized Distributed Optimization via a New Lower Bound Construction

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
We consider centralized distributed optimization in the classical federated learning setup, where $n$ workers jointly find an $\varepsilon$-stationary point of an $L$-smooth, $d$-dimensional nonconvex function $f$, having access only to unbiased stochastic gradients with variance $\sigma^2$. Each worker requires at most $h$ seconds to compute a stochastic gradient, and the communication times from the server to the workers and from the workers to the server are $\tau_{\textnormal{s}}$ and $\tau_{\textnormal{w}}$ seconds per coordinate, respectively. One of the main motivations for distributed optimization is to achieve scalability with respect to $n$. For instance, it is well known that the distributed version of \algname{SGD} has a variance-dependent runtime term $\frac{h \sigma^2 L \Delta}{n \varepsilon^2},$ which improves with the number of workers $n,$ where $\Delta := f(x^0) - f^*,$ and $x^0 \in \mathbb{R}^d$ is the starting point. Similarly, using unbiased sparsification compressors, it is possible to reduce \emph{both} the variance-dependent runtime term and the communication runtime term from $\tau_{\textnormal{w}} d \frac{L \Delta}{\varepsilon}$ to $\frac{\tau_{\textnormal{w}} d L \Delta}{n \varepsilon} + \sqrt{\frac{\tau_{\textnormal{w}} d h \sigma^2}{n \varepsilon}} \cdot \frac{L \Delta}{\varepsilon},$ which also benefits from increasing $n.$ However, once we account for the communication from the server to the workers $\tau_{\textnormal{s}}$, we prove that it becomes infeasible to design a method using unbiased random sparsification compressors that scales both the server-side communication runtime term $\tau_{\textnormal{s}} d \frac{L \Delta}{\varepsilon}$ and the variance-dependent runtime term $\frac{h \sigma^2 L \Delta}{\varepsilon^2},$ better than poly-logarithmically in $n$, even in the homogeneous (i.i.d.) case, where all workers access the same function or distribution. Indeed, when $\tau_{\textnormal{s}} \simeq \tau_{\textnormal{w}},$ our lower bound is $\tilde{\Omega}(\min[h (\frac{\sigma^2}{n \varepsilon} + 1) \frac{L \Delta}{\varepsilon} + {\tau_{\textnormal{s}} d \frac{L \Delta}{\varepsilon}},\; h \frac{L \Delta}{\varepsilon} + {h \frac{\sigma^2 L \Delta}{\varepsilon^2}}]).$ To establish this result, we construct a new ``worst-case'' function and develop a new lower bound framework that reduces the analysis to the concentration of a random sum, for which we prove a concentration bound. These results reveal fundamental limitations in scaling distributed optimization, even under the homogeneous (i.i.d.) assumption.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the fundamental limitations of centralized distributed optimization within the federated learning framework. It establishes a new lower bound on the time complexity for finding an ε-stationary point of smooth non-convex functions. A key aspect of the work is the explicit inclusion of bidirectional communication costs, particularly the server-to-worker  communication time. The paper reveals fundamental limitations in scaling distributed optimization.

### Strengths
1.The explicit consideration of bidirectional communication costs (τs and τw) is a powerful and novel contribution, reflecting a more realistic model of distributed systems.
2.The paper makes a significant theoretical contribution by constructing a new lower bound that clarifies the scalability limits of a specific class of optimization algorithms.
3.The construction of a novel "worst-case" function and the associated proof framework is a strong technical achievement.

### Weaknesses
1.The paper's discussion on the practical implications of its pessimistic results for unbiased compressors is too brief. It would be significantly strengthened by a more detailed exploration of how this points to the potential superiority of biased compressors in real-world scenarios.
2.The practical takeaways for system designers and practitioners are not stated explicitly enough. The paper should offer clearer guidance on how these theoretical results should inform practical decisions (e.g., algorithm selection based on network characteristics).
3.The scope is limited to unbiased compressors, while many state-of-the-art methods use biased ones. The paper should better position itself regarding this gap between its theoretical setting and common practice.
4.The notation is dense. A table of key symbols and their definitions in the appendix would improve the paper's readability.

### Questions
1.Given the pessimistic results for unbiased compressors when τs > 0, does this imply that biased compressors (e.g., TopK) are a fundamentally better choice in practical scenarios with communication bottlenecks?
2.Can the current analysis framework be extended to provide lower bounds for biased compressors? What are the main challenges in doing so, or would a completely new approach be required?
3.How should a practitioner interpret these results when designing a distributed system? For instance, in a network with symmetric communication costs (τs ≈ τw), does your lower bound suggest that investing in the studied forms of compression is futile and methods like Batch Synchronized SGD are preferable?

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
4

### Summary
This paper presents new lower bounds for centralized distributed non-convex optimization of smooth functions. This is an active area, where convergence guarantees for finding approximate stationary points using first-order stochastic oracles have been studied extensively in the past several years. The key difference in this paper's setting is that, instead of examining oracle, communication, or iteration complexity, the paper focuses on time complexity in a model where each machine has the same computation speed and potentially different up (to the server) and down (from the server) communication times. The paper considers the homogeneous setting, for which it is more challenging to establish lower bounds, as one can not benefit from the usual round complexity constructions, which force machines to depend on each other for growing the coordinate span of their model. 

The most remarkable, and perhaps astonishing, result that the paper shows is that when the up and down communication times are comparable, the optimal time complexity is attained by the best of (uncompressed) mini-batch SGD and single-machine SGD. Given the attention this research area has received, a plethora of algorithms have been devised and analyzed, including compression algorithms. However, the fact that one of the two simplest algorithms is min-max optimal provides a firm foundation for the optimization theory in this area. Notably, this result is comparable to the seminal work of [Woodworth et al.](https://arxiv.org/abs/2102.01583), which also reveals the same dichotomy for homogeneous smooth convex optimization, but without considering computation and communication times. 

Additionally, the paper presents a lower bound in the case where up and down communication require different times. In that setting, the bound is still matched by either one of the most natural compression algorithms, mini-batch QSGD or single machine SGD. The hard instance for the lower bound follows the serial hard instance due to Carmon et al., which in turn follows the chain function of Nesterov. The authors make the original hard instance more complex by altering the process of discovering the next coordinate. 

Overall, besides minor writing issues, I don't have any significant concerns with the paper and recommend accepting it. Once the authors address the writing issues, I am happy to increase my score.

### Strengths
The paper is well-written; it provides a comprehensive survey of the current landscape of results and algorithms, ultimately highlighting the gap in existing rates when it comes to time complexity, thereby motivating the central question of the paper. The paper also does a good job of summarizing the technical preliminaries and clearly expresses the novel technical idea. As mentioned above, the dichotomy between mini-batch and single machine SGD is a somewhat surprising result in the non-convex setting. It provides clarity on existing gaps and guides future research.

### Weaknesses
1. I believe the paper can do a more thorough job of reviewing the relevant literature. For instance, the morally most similar paper due to [Woodworth et al.](https://arxiv.org/abs/2102.01583) is not discussed. Similarly, several other interesting related threads are not discussed, such as multi-point oracles and variance reduction, which present natural improvements in the non-convex setting and where, in the oracle complexity model, (almost) min-max rates [are known](https://openreview.net/forum?id=SNElc7QmMDe). 
2. While the lower bound of the paper is surprising, I believe it is essential to highlight that the lower bound itself does not indicate or explain the "limited scalability of centralized distributed optimization". Firstly, the lower bound considers a specific setting of homogeneous non-convex optimization. It is very well possible that under additional assumptions, such as higher-order smoothness and/or some notion of quasi-convexity, we might expect different results and optimal algorithms. I would encourage the authors to read the future directions presented by [Woodworth et al.](https://arxiv.org/abs/2102.01583), which indeed motivated many exciting follow-up results. I recommend that the authors temper the conclusions to be drawn from the results, and instead use these to either motivate why other settings might be interesting to study or show empirical evidence that centralized distributed optimization has indeed plateaued (the latter is very unlikely to be the case).
3. It is essential to underline that while in optimization theory the variance reduction due to averaging across the clients is usually the only justification for collaboration, that is a very narrow view. In a sense, the model under which such rates are provided is actually too restrictive to demonstrate the other benefits of collaboration. For instance, in the same vein as the hard instance of this paper, collaboration can help clients discover parts of the loss landscape that are inaccessible to individual clients. This effect does not appear in the stochastic optimization setup where fresh samples are used, and there is no distribution shift between the training and test distributions. Both of these effects are unavoidable in practice. Similarly, there is ample evidence that collaboration can have a regularization effect. This has been studied in the context of local update algorithms ([1](https://openreview.net/forum?id=dOoPSZFDiRB), [Section 4.3](https://arxiv.org/pdf/2507.00195)), but it may also be true in general for other forms of collaboration. 

Overall, in the last two points, I aim to emphasize that it is crucial to note that the paper examines a specific theoretical model, which does provide some evidence for what we might observe in real experiments. However, ultimately, that is a particular model, which can not capture all aspects of empirical training. 

4. I believe the presentation and definition of distributed zero respecting algorithms (whole of section 2.1) can be made much more crisp by introducing an oracle model. I recommend that the authors check the definitions in [Patel et al.](https://openreview.net/forum?id=SNElc7QmMDe) and other similar papers. Protocol 1 is quite confusing, and I especially don't like how the authors present two parallel for loops sequentially. I believe it is ambiguous to say what it means to calculate the next point, based on some information. This can be made more rigorous by considering a coordinate span, like the papers in this area do.

### Questions
1. Because section 2.1 is not super clear, does the class of algorithms considered here allow simultaneous queries (multiple queries at the same random seed) or not? What about local update algorithms? Can the queries be adaptive to the information on each machine locally or not? I would like the authors to comment on the results of [Patel et al.](https://openreview.net/pdf?id=SNElc7QmMDe), which may provide better upper bounds when simultaneous queries are permitted. 
2. In the vein of the previous question, can the authors use their hard instance to give better lower bounds in the setting when computation and communication times are not critical, and we are looking at oracle complexity? Specifically, I want to understand whether the new technique can actually demonstrate that the convergence rate of SCAFFOLD and MB-SGD is tight in the oracle complexity sense. For more context, see [Table 1](https://openreview.net/pdf?id=SNElc7QmMDe) of Patel et al. and note that there is no lower bound for the single query model (their Theorem 3.2 can not demarcate between single and multiple queries).   
3. How is the result in this paper morally different from [Woodworth et al.](https://arxiv.org/abs/2102.01583)? Should I view the consequence of the result as just the time-complexity version of their result? If so, can the authors comment on whether they believe non-convexity is actually critical to their construction? If not, is it possible to get a similar dichotomy between mini-batch and single-machine SGD in the convex setting as well? That would complete the picture in the homogeneous setting.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents lower bounds for centralized federated learning under $L$-smoothness, where communication time is important. It contains a new construction of lower bounds with theoretical guarantees.

 I think this paper may be accepted.

### Strengths
1) This paper is well written, and all the results are clarified.

2) This paper provides a new construction for lower bounds.

3) The result of this paper provides theoretical evidence and a quality comparison with previous works.

### Weaknesses
**Typos:** 

1). p. 5 line 228. It seems “quartic” $\to$ “quadratic”

2). P. 7 line 377. I think, $F_T \to F_{T, K, a}$

3). P. 22 line 1146. $j \to k$

### Questions
I checked all the theorems, except for key Lemma D.1. Therefore, I don't have any major or minor comments, and for this reason, I marked a low level of confidence.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In the paper the authors discuss the influence of both worker to server (w2s) and server to worker (s2w) communications on the overall complexity of solving the distributed optimization problem. The authors propose a new worst-case function, that is required to prove the lower bounds for this setup.

### Strengths
1)Addressing not only w2s, but also s2w communications is important and not usually done in optimization.

2)Modifying the worst-case function and using novel techniques to prove the lower bounds.

### Weaknesses
1)The class of zero-respecting protocols is quite restricting, as it does not contain sketchings, dithered quantizations, and so on. 

2)Theorems 4.2 and E.1 seems to be proven only for RandK compressors, rather than for any unbiased zero-respecting.

### Questions
1)Theorem 4.2 contains only $\tau_s$, but no $\tau_w$. Theorem E.1, on the contrary, contains $\tau_w$, but no $\tau_s$. Could the authors clarify, how they correctly combine lower bound, that involve both s2w and w2s communications?

2)Could authors specify more, how does the heterogeneous setup fail under the consideration of s2w communications?

### Soundness
3

### Presentation
2

### Contribution
3
