# Dynamical properties of dense associative memory

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 2, 10, 8, 8

## Abstract
Dense associative memory, a fundamental instance of modern Hopfield networks, can store a large number of memory patterns as equilibrium states of recurrent networks. While the stationary-state storage capacity has been investigated, its dynamical properties have not yet been discussed. In this paper, we analyze the dynamics using an exact approach based on generating functional analysis. We show results on convergence properties of memory retrieval, such as the convergence time and the size of the attraction basins. Our analysis enables a quantitative evaluation of the convergence time and the storage capacity of dense associative memory, which is useful for model design. Unlike the traditional Hopfield model, the retrieval of a pattern does not act as additional noise to itself, suggesting that the structure of modern networks makes recall more robust. Furthermore, the methodology addressed here can be applied to other energy-based models, and thus has the potential to contribute to the design of future architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The work provides an analysis of the dynamical properties of the Dense Associative Memory model (DAM), focusing on the high-order polynomial function ($n\geq 3$) from [Krotov and Hopfield (2016)](https://arxiv.org/abs/1606.01164) using generating functional analysis (GFA). The analysis of the work enables a quantitative evaluation of the convergence time and the storage capacity of DAM, shedding further light on why memory retrieval with DAM is more robust than the Classical Hopfield model.

### Strengths
1. An interesting, new work which studies the dynamical aspects of DAM in attempt to provide another perspective on the storage of capacity of these models.

2. The approximated or calculated dynamics are shown to be close to the simulated dynamics, illustrated in Fig. (2).

### Weaknesses
1. Limited experimentation. As far as I am aware, the experimentation is focused on the case of the polynomial order $n = 3$ while the mathematical derivations are focused on the case of $n \geq 3$. 

2. The experiments are not conducted on natural datasets (e.g., CIFAR10, ImageNet, etc.) or a small subset of them. In such datasets, there exists a strong need to understand how the diversity of pattern-features affect the dynamical regimes of DAMs, which are governed by their memories, mathematically.

3. What about asynchronous update? This work focuses on parallel update, which technically had been shown by [Cheung et al. (1987)](https://opg.optica.org/ao/abstract.cfm?uri=ao-26-22-4808) that it does not guarantee convergence for $n = 2$, and it is not the preferred update in [Krotov and Hopfield (2016)].

### Questions
Overall, I think the work is interesting. But the presentation can be written much better, and it requires a lot more experimentation to help explain its mathematical contributions. See the weakness section above for my concerns/questions.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
I feel like I'm slightly cheating here, since I'm somewhat plagiarizing the last paragraph of the paper. But that was an almost perfect summary, and it's more or less what I wanted to say anyway. At least I didn't use an LLM. ;)

The authors provide an analysis of the time evolution of statistical quantities (means, covariances, etc.) of the dense associative memory. For that they used the generating functional analysis, which is typically exact in the limit N --> infinity. (I'm guessing there are some cases where it breaks down, but I don't know what they are.) They veriﬁed the theoretical predictions with numerical experiments.

### Strengths
The calculation itself (which I only spot-checked) was highly nontrivial. While the technique itself is not knew, to my knowledge this is the first time it was applied to the dense associattive memory. I can also verify the statement by the author that their results cannot be captured by the standard signal-noise analysis (or at least not in my hands).

The results shed new light on the dynamics for cubic and higher nonlinearities and higher, and some interesting phenomena emerged -- such as suppression of crosstalk due to the recalled pattern, and a complete characterization of the basin of attraction.

The authors verified their theoretica predictions with numerical simulations, and agreement was good even though they used only 512 neurons. This is impressive, given that deviations usually scale as 1/sqrt(N), which in their case is 5%.

### Weaknesses
I didn't see any weaknesses. Unless you count a possible typo: it looks to me like theta_i is in a different place in Eq. 13 versus Eq. 6. But maybe that's on purpose?

### Questions
No questions.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Using Generating Functional Analysis (GFA), the authors derived an asymptotically exact solution that quantitatively characterises convergence time and the size of attraction basins, aspects previously under-investigated. The analysis provides novel insight into the model's robustness, showing that unlike the traditional model, pattern retrieval does not introduce additional self-noise.

### Strengths
1. The work provides an asymptotically exact analysis of the dense associative memory's dynamics in the large-system limit using GFA.
2. The analysis yields explicit and quantitative results on convergence properties, such as the convergence time and the size of the attraction basins.
3. The modern/dense model is shown to be more robust than the traditional model because the noise variance does not depend on the overlap and does not increase as the overlap grows.

### Weaknesses
1. Rather than providing full proofs, proof sketches are provided. I can see and understand most of the results, however it would be ideal to have full, rigorous proofs provided.
2. There are some minor typos, e.g., "confirms" -> "confirmed" in L393. I suggest the authors proofread carefully.

### Questions
1. Could the authors' methods be simply extended to related formulations or models, e.g., Burns & Fukai "Simplicial Hopfield networks" ICLR 2023. (Probably it works similarly to the proof sketch of Gardner's model.)

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper provides the first asymptotically exact dynamical analysis of dense associative memory (DAM) introduced by Krotov & Hopfield (2016), using the Generating Functional Analysis (GFA) framework.
While previous works mostly studied stationary-state storage capacity, this work studies on dynamical properties like recall convergence time, attraction basin region and self-noise behavior.
The main results is DAMs exhibit retarded self-interaction that stabilizes dynamics and prevents the recall process from generating additional self-noise. This makes memory retrieval more robust and expands the basin of attraction.

### Strengths
Overall, the paper is of high quality, I believe it provides a really good analysis on the whole recall process which previous studies did not fully explore.

1. The motivation is clear
2. The mathematical derivation looks correct to me
3. The results are novel to my best knowledge

### Weaknesses
1. The term "retarded self-interaction" lacks intuition to me, can the authors provide details on how do we interpret this term biologically?
2. It would be interesting to see whether the framework used in the paper can be easily expanded to different non-linear activation such as the exponential function.

### Questions
1. Will the argument $M=O(N^{n−1})$ become $M = O(e^N)$, if we switch the activation function to e^x?
2. Can the authors further discuss the limitations of this work, and potential future directions?
3. I suggest the authors to provide a table to compare existing results and their result.
4. Can the authors explain why the existing framework under control theory not work here? Based on my understanding, previous studies shows that the DAM system typically has exponential convergence rate as long as its stable. How does the result from this paper differ from this conclusion?

### Soundness
3

### Presentation
3

### Contribution
3
