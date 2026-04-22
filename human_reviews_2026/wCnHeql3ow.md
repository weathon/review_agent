# Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Grokking is the phenomenon whereby, unlike the training performance which peaks very early on during training, the test/generalization performance of a model stagnates over arbitrarily many epochs and then suddenly jumps to usually close to perfect levels. In practice, it is desirable to reduce the length of such plateaus, that is to make the learning process "grok" faster. In this work, we provide new insights into grokking. First, we show both empirically and theoretically that grokking can be induced by asymmetric speeds of (stochastic) gradient descent, along different principal (i.e singular directions) of the gradients. We then propose a simple modification that normalizes the gradients so that dynamics along all the principal directions evolves at exactly the same speed. Then, we establish that this modified method, which we call egalitarian gradient descent (EGD) and can be seen as a carefully modified form of natural gradient descent, groks much faster. In fact, in some cases the stagnation is completely removed. Finally, we empirically show that on classical arithmetic problems like modular addition and sparse parity problem which this stagnation has been widely observed and intensively studied, that our proposed method removes the plateaus.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the phenomenon of grokking—where neural networks plateau for long periods in generalization performance before abruptly improving—by connecting the length of the plateau to the spectral properties of the gradient during training. The authors propose Egalitarian Gradient Descent (EGD), a modification to classic gradient descent that normalizes the gradient along all principal directions, theoretically eliminating slow convergence due to ill-conditioned spectra. Extensive empirical evaluations are provided across toy problems and algorithmic tasks, demonstrating that EGD can dramatically reduce (or eliminate) the grokking plateau compared to vanilla SGD and other methods.

### Strengths
1. Clear theoretical insight and connection to spectrum: The paper gives a transparent analytic treatment for a simplified grokking scenario, precisely quantifying how ill-conditioning in the gradient’s principal directions translates into stagnation in test accuracy (see Section 3; Theorem 1 and Corollary 1). This directly links observable grokking delays to quantifiable optimization issues, moving beyond anecdotal claims.

 2. Simple, principled method: EGD is conceptually straightforward, introduces no free hyperparameters, is optimizer-agnostic, and is presented as a direct, closed-form modification to the gradient for each layer (Section 4). Compared to competing approaches like Grokfast, EGD incorporates no history/memory buffers, making it lightweight for integration.

 3. Sound mathematical formalism: Derivations (e.g., equations (11)-(12), as well as the detailed proofs in Appendix A) are logically consistent, and the singular-value “flattening” interpretation of the update is rigorous.

 4. Strong empirical backing: Results in Figures 1–3 show that EGD consistently accelerates grokking across modular addition, modular multiplication, and sparse parity datasets—tasks where the phenomenon has been robustly documented. EGD typically leads to immediate generalization, while SGD and even “column normalization” lag—this is evident in Figure 1 and Figure 2.

 5. Insightful spectral diagnostics: Figure 4 explicitly illustrates how delays in grokking correspond to highly skewed gradient spectra, making the theoretical contribution tangible and informative for the reader.

### Weaknesses
1. Scalability and Computational Cost: The method requires per-step SVD computation on the gradient matrix for each layer, an operation that grows rapidly with layer width. While the authors claim (Page 7) that randomized or approximate SVDs “work just fine”, there is no benchmark or ablation quantifying the tradeoff, or on what network scales/width SVD ceases to be efficient. This undermines claims about practical usability in modern architectures.

2. Lack of Ablation and Sensitivity Analysis: The experimental section has limited sensitivity checks. For example, EGD is briefly compared to a “column normalization” heuristic, but not to common natural gradient or second-order optimizers (despite Section 4.1’s close connection). There is no ablation quantifying: (i) the effect of running EGD on only top-$k$ directions vs. full-rank, or (ii) across various batch sizes/optimizers. This makes it hard to parse where the real gains come from.

3. Missing Rigorous Quantitative Comparisons with Strong Baselines: Grokfast (Lee et al., 2024) and other recent grokking accelerators are discussed theoretically (Section 4.2), but quantitative experimental comparisons are limited. The paper could benefit from more comprehensive results tables directly benchmarking EGD against these methods across identical tasks.

4. Overstated Generality and Claims Beyond the Evidence: In several places (e.g., Section 6, Conclusion; Abstract; and throughout the Introduction), the paper implies EGD could be universally beneficial and “plug-and-play” for any neural network. The only evidence is for the tightly controlled algorithmic family, and the real computational cost is glossed over. The claims regarding optimizer-agnosticism and hyperparameter-freeness require much more stress-testing.

5. Limited Theoretical Depth on Nonlinear/Deep Networks: Section 4 describes the transition from linear to nonlinear settings, but almost all the theoretical analysis is confined to the analytically tractable linear case, or at best layer-wise argument. No generalization, convergence, or stability claim is rigorously proven for deep, nonlinear architectures—the regime of real interest.

### Questions
1. Applicability/Scalability Beyond Toy Tasks: What is the maximal network width, depth, and dataset size on which you have successfully deployed EGD, and how does the runtime/throughput compare with vanilla SGD and other strong baselines? Can you provide concrete experiments or ablations addressing actual wall-clock cost?

 2. Necessity of Full-Rank SVD: Is the full SVD required for each gradient step in all layers, or would it suffice to approximate just the top-$k$ singular values/vectors, perhaps by sketching methods? (Experiments addressing this could open EGD to wider scaling.)

 3. Direct Empirical Comparison to Grokfast: Your only comparative discussion with Grokfast is qualitative. Can you provide synchronized benchmarks (same datasets, architectures, and hyperparameters) for EGD and Grokfast, so the empirical strengths and limitations are evident?

 4. Failure Modes or Adverse Side Effects: Can EGD cause instability, exploding/vanishing gradients, or harm convergence in high-noise regimes or complicated real-world architectures? Any evidence of problems in practical runs would be useful.

 5. Potential for Generalization Analysis in Deep Networks: Is there any theory in progress on whether the spectral-flattening argument for “egalitarian” updates extends robustly to multi-layer, highly nonlinear settings?

### Soundness
3

### Presentation
3

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
This paper introduces Egalitarian Gradient Descent (EGD) as an optimization to reduce and or mitigate the grokking phenomenon. Theoretical rationale is provided as to why grokking occurs within training practices using stochastic gradient descent (SGD). They show that the leading singular directions dominate the optimisation process, which they show plays a role in grokking. Therefore, they propose the EGD, which sets all the singular directions of the gradient update to 1; thus, all singular directions are optimised equally, hence the name Egalitarian. They highlight that this method can be viewed as a simplified version of Grokfast and enjoys the same benefits while not requiring hyperparameter sweeps. The results of the method are shown on the sparse party problem and modular arithmetic tasks.

### Strengths
- The paper is very well written, reads well, and is self-contained. 

- It is positioned very well within the literature exploring grokking and methods to mitigate/reduce grokking. 

- It provides a strong theoretical rationale behind the method presented.

- Provides insights into the occurrence of grokking more generally.

### Weaknesses
- In Figures 1 and 2, it is not clear where the train accuracy lines are for the EGD method. Is this due to them going up in tandem, or has this data been omitted? 

- Figures 1 and 2 range from -0.5 to 1, resulting in a large portion of the figures being clear; this should be reduced to make the data easier to read. It would also be useful to provide the grid to make reading off the figures easier. 

- Increasing the number of tasked benchmarked would increase the impactfulness to the overall community, especially tasks where grokking is induced, such as on MNIST in [1]. 

- Although the method is fairly simple, as it requires performing SVD on the gradient to receive $U$ and $V^T$, it would be good to provide the codebase to enable the work to be easily replicated. 

- The `Limitations and Future Work` section reads more like future work. The limitations of the work are not clearly represented. This should be more aptly named Future Work. 

- The Bibliography is missing links to papers. 

- Typo/Spelling Mistake: Line 375 occured should be `occurred`


[1] Liu, Z., Michaud, E.J. and Tegmark, M., 2022. Omnigrok: Grokking beyond algorithmic data. arXiv preprint arXiv:2210.01117.

### Questions
1.  Line `374` states In practice, we turn off EGD and switch it for vanilla (S)GD once we detect grokking has occured $^1$ where the footnote states This is detected by monitoring validation loss.  When does one exactly turn off EGD? At what point do you consider grokking detected? Is it once the model starts to generalise? Does this not introduce a hyperparameter?

2. For the sparsity parity problem, Figure 3. The method does not fully resolve grokking for Parity(n=100, k=3) ReLU NN(width=100). Is there a rationale behind why here the model training and test accuracy do not increase in tandem, as was shown in the modular arithmetic tasks? 

3. Can you provide the results of this method on Grokking-Induced MNIST and IMBD dataset as done in [1]

[1] Liu, Z., Michaud, E.J. and Tegmark, M., 2022. Omnigrok: Grokking beyond algorithmic data. arXiv preprint arXiv:2210.01117.

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
This article studies the grokking phenomenon based on new insights on simple classification problems. A theoretical analysis is provided on a binary classification problem, showing how grokking can arise due to ill-conditioning in optimization. Then a modified gradient descent method (egalitarian GD) is proposed to adjust the gradient direction to avoid stagnation during training. Numerical results on sparse parity and modular addition also confirm the effectiveness of the insights.

### Strengths
-	The theoretical picture of the grokking phenomenon is interesting. 
-	The proposed EGD algorithm works well in practice. 
-	The article is well written, with a quite complete literature review.

### Weaknesses
-  The theoretical example is somehow biased and too low-dimensional, which makes it hard to explain practical results.
- There is no theoretical convergence analysis on the proposed EGD algorithm.

### Questions
- Can you explain why you choose a different training and test data distribution in Section 3 (fig 5)? This is not a standard machine learning setup.
- Are you considering the same training and test distributions in the numerical results in Section 5? If so, I would suggest to modify the theory in Section 3 to make it consistent with the setup in Section 5. 
- It is unclear how the G is F in  eq. 11 is estimated from mini-batch samples. This practical aspect should be discussed to better understand the numerical results.

### Soundness
2

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
5

### Summary
The paper proposes a buffer-free gradient preconditioning method for accelerating grokking phenomenon, reducing the gap between memorization and generalization from spectral analysis. The authors have provided theoretical results that justify their egalitarian gradient descent (EGD) algorithm assuming two-parameter model. Based on the insights, the algorithm is tested on the typical examples where grokking appears, demonstrating the acceleration.

### Strengths
- The idea is simple and elegantly addressed, with easy theory and clear empirical examples.
- Experiments show that at least among the given examples, the proposed method effectively reduces the grokking gap.
- The experimental details are sufficient and easily reproducible.

### Weaknesses
Some points prevent me from giving a higher score. I will raise my score based on the discussion.

- Since the effect of grokking has demonstrated in various other settings than modular arithmetic such as in [Omnigrok](https://arxiv.org/abs/2210.01117) (MNIST, Transformer, Graph CNN, LSTM) or [Kumar et al. (2024)](https://openreview.net/forum?id=vt5mnLVIVo) (MNIST), more experiments can be performed to further justify the clear effectiveness of the proposed EGD.
- Theoretical justification only exists for toy examples. I believe it should be easy to extend the theoretical results to show that the insights are extendable to larger, more general models such as linear layers or two-layer MLPs.
- The proposed algorithm seems to be similar to RMSProp. Further discussion between the existing gradient descent variants to the proposed method should be carried on. Moreover, how does the proposed egalitarian gradient descent interacts with “momentum” variable which typically appear in the machine learning context?
- The connection between FIM, NGD, and the proposed EGD can be further strengthened by showing how do individual FIM-based and NGD-based algorithms behave under grokking environment. This will further justify the proposed method for accelerating grokking, and imply richer connection between FIM and NGD involved in this problem.

### Questions
- The box in line 350-355 seems to swallow additional lines after the equation.
- Some works such as [Kumar et al. (2024)](https://openreview.net/forum?id=vt5mnLVIVo) and [Grokfast](https://arxiv.org/abs/2405.20233) point out that accelerated grokking-induced solution space can be different from the non-accelerated grokking-induced solution space. In other words, the final reaching ground in the parameter space can be differ greatly. Does this proposed method results in the same solution manifold or in the greatly different solution manifold?

### Soundness
3

### Presentation
3

### Contribution
3
