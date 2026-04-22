# Improved state mixing in higher-order and block diagonal linear recurrent networks

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 8, 4

## Abstract
Linear recurrent networks (LRNNs) and linear state space models (SSMs) promise computational and memory efficiency on long-sequence modeling tasks, yet their diagonal state transitions limit expressivity. Dense and/or nonlinear architectures (e.g., LSTMs) on the other hand are provably more expressive, but computationally costly. 
Here, we explore how expressivity in LRNNs can be increased via richer state mixing across time and channels while maintaining competitive efficiency. Specifically, we introduce two structured LRNN architectures: (i) Higher-order Linear Recurrent Units (H-LRU), which generalize first-order recurrence to $m$-th order, mixing multiple past states, and (ii) Block-Diagonal LRUs (BD-LRU), which enable dense intra-block channel mixing. Per-channel (H-LRU) / per-row (BD-LRU) L1-normalization of selective gates stabilizes training and allows for scaling window/block sizes. In synthetic sequence-modeling benchmarks (compression, selective copying, associative recall), H-LRU is found to be the most parameter-efficient in compression, while the performance of BD-LRU matches or exceeds those of linear SSMs (Mamba), low-rank LRNNs (DeltaNet) and LSTM baselines. In permutation composition tasks ($S_3$-$S_5$), BD-LRU is found to efficiently solve these tasks at moderate block sizes, outperforming both linear and non-linear baselines. A parallel-scan implementation of the proposed architectures keeps the throughput competitive with diagonal LRNNs for moderate orders (H-LRU) and block sizes (BD-LRU), while preserving the efficiency that motivated LRNNs. 
These results indicate that the structure of state mixing rather than width alone shapes expressivity of LRNNs, offering a practical route to closing the efficiency–expressivity gap in linear sequence models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Background: traditional recurrent NNs (e.g. LSTMs) rely on fully dense transition matrices.  Linear recurrent NNs (and state space models) are a class of neural networks in which the transition matrix is a diagonal matrix, and they are much faster to compute over long sequences.  

The authors introduce and evaluate two new classes of recurrent neural networks (RNNs).  Higher-order linear recurrence (H-LRU) can refer not just to the hidden state at the previous time step, but to prior time steps as well.  Block-diagonal recurrence (BD-LRU) uses a state transition matrix that is block-diagonal, rather than diagonal, and thus offers a compromise between linear RNNs and fully dense RNNs.  

The authors show that H-LRUs are actually a type of BD-LRU with a particular block structure.  They then show that BD-LRUs can be tamed by use of a normalization scheme, which normalizes the transition matrix to avoid exploding or vanishing activations and gradients. 

The authors compare their RNNs with various block sizes against LSTMs, Mamba, and Deltanet, on several synthetic tasks.

### Strengths
For the most part the paper is clear and well-written, and the mathematics are well-presented. 

The normalization technique in equation (6) seems valuable.

### Weaknesses
The concepts are not hard, but the writing style is rather dense and difficult to slog through.  

The major weakness of the paper is (1) extremely small scale, and (2) reliance solely on toy synthetic problems.  

In general, I think that synthetic tasks are actually quite useful -- they allow you to evaluate precisely which tasks a given NN architecture is good at.  However, any proposed architecture should *also* be evaluated on real-world data sets, such as one of the many language-modeling datasets. 

This problem is exacerbated by the extremely small scale of the NNs involved.  The authors test block sizes for the BD-LRU of between 1 and 5, which is miniscule.  Modern TPU and GPU hardware is typically efficient only for matmuls of dimension 128 or greater.  

The final issue is that the actual performance of BD-LRU is bizarre.  In Figure 1, it is simultaneously the best performing and the worst-performing model.  Moreover the performance does not scale with block size as one would expect -- the points seem like a random scatter plot, which looks to me like an instability of the technique.  At the very least, this issue needs to be explained, and the authors do not do so. 

The parallel scan algorithm also needs to be properly explained.  A citation to an old paper on prefix sums is not sufficient for equation (8), which is clearly not a simple sum.

### Questions
What is your explanation for the results in Figure 1?

In figure 2, where is relu?  Why does relu fail so abysmally?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The author introduces that LRNNs face a fundamental tradeoff: diagonal architectures are computationally efficient but expressively limited, while dense architectures are more expressive but computationally expensive. This paper explores structured approaches to bridge this efficiency-expressivity gap.

The author proposed Higher-order Linear Recurrent Units, which mix multiple past hidden states and Block-Diagonal LRUs, where the recurrent weight matrix is block instead of pure full or diagonal. The author notice and proposed a method to solve the stability issue. The author provides experiments to demonstrate the effectiveness of the proposed architecture.

### Strengths
The presentation of the architecture is clear, and easy to follow. The notations and formulas are well defined which makes the reading smooth.
The proposed method is novel which has not been proposed before.
There are extensive experiments to evaluate the proposed architecture.

### Weaknesses
See Questions

### Questions
The author introduces that LRNNs face a fundamental tradeoff: diagonal architectures are computationally efficient but expressively limited, while dense architectures are more expressive but computationally expensive. This paper explores structured approaches to bridge this efficiency-expressivity gap.

The author proposed Higher-order Linear Recurrent Units, which mix multiple past hidden states and Block-Diagonal LRUs, where the recurrent weight matrix is block instead of pure full or diagonal. The author notice and proposed a method to solve the stability issue. The author provides experiments to demonstrate the effectiveness of the proposed architecture.



The presentation of the architecture is clear, and easy to follow. The notations and formulas are well defined which makes the reading smooth.
The proposed method is novel which has not been proposed before.


1. Line 13-14: "Dense and/or nonlinear architectures (e.g., LSTMs) on the other hand are provably more expressive"
    - For architecture with nonlieanrity, no matter the recurrent matrix is full or diagonal, the architecture should have universal approximation property. Why the author state that dense architectures are more expressiv, what is the "expressive" here refers to?

2. Line 876 - 917 Appendix F discuss the expressivity. 
    - Firstly, without non linearity it is obvious that diagonal matrices is contained in the block diagonal matrices is contained in the full matrices. 
    - Consider a target is in the form $y=C A^n B$, say if A is diagonalizable, then whether the model uses a diagonal or dense matrices does not matter. And in reality a random matrix $A$ has probability 1 to be diagonalizable in complex space. So there expensiveness are very "near" except for a measure zero set of targets.

3. Are there any new trade-off arises in H-LRU? The $m$ determines the window size the hidden state looks back. When increasing this $m$, the memory can be reduced. but as a side effect, the temporal information is also lost. Consider the extreme case where $m$ equals to the sequence length. The the recurrent model would collapse to something like a feedforward network, where the input dimension is the sequence length. This makes is able to look at the entire sequence at once but also lost all the temporal information. Thus, the question is that is there a fixed good $m$ that balances this tradeoff. 

4. For BD-LRU, the $m$ determines the number of block and it is fixed. The question of how $m$ should be determined is also a question. If there is no algorithm that can determin this $m$ without actually train the model, I think the significance of the results is much lower.


6. If there are no reparameterzation involved in the implementation as shown in Appendix G, I doubt whether the model can have better performance than S4 or S4D models.  The paper says that the BD-LRU and H-LRU is more expressive than diagonal models, but experiments does not has diagonal models involved.

7. From the experiment the H-LRU does not seems have better results than other architectures. some cases worse than LSTM.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies linear RNNs with a more expressive transition than the common diagonal form. Higher-order LRUs (H-LRU) mix the last m hidden states, and block-diagonal LRUs (BD-LRU) use a dense transition matrix with m×m blocks.

They introduce a normalisation procedure for well-behaved training when scaling m, avoiding exploding/vanishing gradients.

On several synthetic sequence tasks, increasing m improves accuracy, especially for BD-LRUs. The biggest gains often appear when moving from m=1 to m=2. The authors attribute this to access to complex eigenvalues only when m>1.

Importantly, they provide an efficient higher-order parallel scan implementation for these non-diagonal LRUs.

### Strengths
The motivation is clear. Diagonal LRNNs are efficient but expressively limited. Structured non-diagonal mixing can close the gap while retaining much of the efficiency.

The normalisation is well motivated and supported by the ablation in Figure 2.

Several benchmarks support the move to H-LRUs and BD-LRUs. The jump from m=1 to m=2 is notable, plausibly due to complex eigenvalues (as the authors note).

Permutation tasks show an advantage for higher m as task complexity increases, especially for BD-LRU.

### Weaknesses
The use of H-LRU and BD-LRU themselves is not novel, which is why the efficient implementation and normalization are so important.

The main text briefly states how block-diagonal structure reduces the cost of the parallel scan; more detail is deferred to the appendix/code. A short sketch in the main text would help.

### Questions
If I understand correctly, BD-LRU and H-LRU are equivalent at m=1. In Figure 1 it would be good to highlight the m=1 case separately. Are the m=1 dots overlapping for H- and BD-LRU?

It might be too much to include full derivations in the main text, but since it is central to the study, could you briefly flesh out how the block-diagonal structure reduces the parallel-scan time complexity (the “hopscan” approach) in the main text, and point to the detailed appendix/code?

I would be curious to see the performance of across m for a benchmarks of real language data, such as wikitext-103. Would it be possible to include something like this?

Optional curiosity: have you considered a diagonal transition with complex entries as a baseline?

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
3

### Summary
This paper introduces two Linear RNN gating parameterizations that enhance expressivity while retaining efficiency. The first one, Higher-order Linear Recurrent Units (H-LRU), generalizes first-order recurrence to m-th order, enabling temporal mixing across multiple past hidden states. The second one, Block-Diagonal Linear Recurrent Units (BD-LRU), replaces purely diagonal state-transition matrices with a block-diagonal structure, permitting richer intra-block channel mixing. Both architectures incorporate selective gating and L1-normalization of the transition blocks (per-channel for H-LRU, per-row for BD-LRU) to ensure stability. The authors also provide a theoretical guarantee (Proposition 1) that under this L1-normalized gating, the hidden-state norm remains bounded. 

Empirical results show that both H-LRU and BD-LRU achieve strong performance on the MAD benchmark tasks (compression, associative recall and compression). Moreover, BD-LRU matches or exceeds recent linear baselines (Mamba, DeltaNet, DeltaProduct) and even LSTM on the permutation tasks that require state tracking. Finally, the authors provide a parallel-scan implementation that is shown to yield throughput comparable to diagonal LRNNs for moderate block sizes.

### Strengths
1. The proposed architectural parameterizations (H-LRU, BD-LRU) are conceptually elegant and clearly motivated. The authors show that higher-order recurrence and block structure are natural ways to increase mixing without resorting to full dense transitions. Both architectures incorporate input-dependent selective gates with L1-normalization (per-channel or per-row), ensuring forward-pass stability and bounded dynamics.

2. The authors provide a theoretical justification (Proposition 1) that this normalization guarantees dynamical stability and normalized hidden-state evolution, showcasing that these architectures are well-behaved and stable by construction.

3. Overall, the paper is well-organized. The progression from model definition (Sec 2) to normalization (Sec 3) and experiments (Sec 4, 5) is nicely structured. Figures and tables presenting results are clear too.

4. The results are strong on the considered synthetic tasks. BD-LRU, in particular, shows significant gains over existing linear RNNs (Mamba, DeltaNet and DeltaProduct) and LSTMs also on state-tracking tasks like permutation tasks (Table 2).

5. The proposed parallel-scan algorithm demonstrates that block-diagonal recurrences can achieve throughput competitive with diagonal ones (Figure 4), addressing a practical concern for deployment.

### Weaknesses
1. **Limited evaluations**: The tasks considered in the empirical evaluations are only synthetic ones. While the use of the MAD benchmark is very useful, demonstrating competitive performance on large-scale language or long-sequence datasets (e.g., LRA, language modeling). This makes it hard to assess if the structural benefits of the proposed architectures transfer to more practical usage. The authors acknowledge this as future work though.

2. The core contributions are about the structure of state mixing. Yet, the authors do not provide empirical analysis on the actual learned matrices and their properties, e.g., their eigenvalue spectra. This would provide crucial insight into what kind of mixing the model is learning (e.g., is BD-LRU learning negative eigenvalues?) and why it succeeds on state-tracking (permutation task).

3. The discussion mentions FLOPs and throughput but lacks detailed runtime comparisons across block sizes, hidden dims and sequence lengths. It’s unclear how block‐size m scales in practice (for m large).

4. The paper provides key code snippets in Appendix G, however, the full code is not yet released.

### Questions
1. To what extent can the BD-LRU architecture support eigenvalues outside the positive range (i.e., negative or complex eigenvalues) in the state transition matrix, as discussed by Grazzi et al. 2025? How does the L1 norm, block size m or order in H-LRU affect the eigenspectrum?

2. Outside the permutation tasks, have you also evaluated on regular language tasks such as parity or modular counting, where such eigenvalues were shown to be necessary for state-tracking?

3. Can you also provide latency/throughput for varying block sizes (e.g., m=2, 4, 8, 16) and higher sequence lengths to better assess the runtime trade-off for the parallel-scan implementation?


-- References --

*Gu, A., Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. arXiv 2023*

*Orvieto, A., Smith, S.L., Gu, A., Fernando, A., Gulcehre, C., Pascanu, R., De, S.. Resurrecting recurrent neural networks for long sequences, in: International Conference on Machine Learning 2023*

*Gu, A., Goel, K., Ré, C.. Efficiently modeling long sequences with structured state spaces. arXiv 2021*

*J. Siems, T. Carstensen, A. Zela, F. Hutter, M. Pontil, R. Grazzi. DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products. In NeurIPS 2025.*

*R. Grazzi, J. Siems, A. Zela, J.K.Franke, F. Hutter, M. Pontil. Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues. In ICLR 2025*

*M. Poli, A. W. Thomas, E. Nguyen, et al. Mechanistic Design and Scaling of Hybrid Architectures. arXiv 2024*

### Soundness
2

### Presentation
4

### Contribution
3
