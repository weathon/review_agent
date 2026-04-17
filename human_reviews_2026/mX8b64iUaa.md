# ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models

- Decision: Accept (Oral)
- Scores: 6, 8, 6, 6

## Abstract
Recurrent Neural Networks (RNNs) laid the foundation for sequence modeling, but their intrinsic sequential nature restricts parallel computation, creating a fundamental barrier to scaling. This has led to the dominance of parallelizable architectures like Transformers and, more recently, State Space Models (SSMs). While SSMs achieve efficient parallelization through structured linear recurrences, this linearity constraint limits their expressive power and precludes modeling complex, nonlinear sequence-wise dependencies.
To address this, we present ParaRNN, a framework that breaks the sequence-parallelization barrier for nonlinear RNNs. Building on prior work, we cast the sequence of nonlinear recurrence relationships as a single system of equations, which we solve in parallel using Newton's iterations combined with custom parallel reductions. Our implementation achieves speedups of up to $665\times$ over na\"ive sequential application, allowing training nonlinear RNNs at unprecedented scales. To showcase this, we apply ParaRNN to adaptations of LSTM and GRU architectures, successfully training models of 7B parameters that attain perplexity comparable to similarly-sized Transformers and Mamba2 architectures.
To accelerate research in efficient sequence modeling, we release the ParaRNN codebase as an open-source framework for automatic training-parallelization of nonlinear RNNs, enabling researchers and practitioners to explore new nonlinear RNN models at scale.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper adds to the body of literature on efficient recurrent neural networks (RNNs). Although RNNs have potential for inference efficiency, particularly at long sequence lengths, the training of RNNs has been difficult due to the inherently sequential nature of the forward pass, unlike a Transformer forward pass which can be trivially parallelized.

This paper offers a new approach to parallelizing the training of RNNs, FlashRNN. The whole forward pass is solved in parallel via a fixed-point iterative method. This allows the efficient training of non-linear RNN cells, albeit the method is only tractable if the RNN cell has a Jacobian with a particular block-diagonal structure. The paper provides efficient implementation of the method in CUDA and scales FlashRNN to the 7B parameter scale.

### Strengths
+ The overall approach is a good contribution, and as far as I am aware this is the first work to allow training of non-linear RNNs via an exact parallelization method with no numerical instability.
+ The extensive implementation of a custom CUDA kernel which can be applied to a range of RNNs is a great contribution to the open-source community, and allows comparison with other architectures at a large scale.
+ The extensive computational results are quite impressive, including training a 7B parameter model from scratch in order to compare the performance of FlashRNN at language modelling tasks.
+ The paper is up-front and clear about the limitations of FlashRNN in section 2.1

### Weaknesses
+ The conceptual contributions of the paper could be argued to be somewhat limited. As far as I can tell (although I'm not an expert in this exact subject), the parallel solution of the sequence model by forward substitution is not novel and is described in Gonzales et al, Lim et al, etc. It is a bit hard to tell which of section 2 is novel and which is from previous work--we know that at least some of section 2 is previous work from the description, but it's not clear if it all is previous work. It seems like the conceptual contribution of this work could be characterized as following Gonzales et al, but simplifying the RNN cell instead of trying to solve the recurrence for general RNNs with a clever method.

+ The paper doesn't empirically justify the diagonalization of the RNN mixer in equation (9). While it's true that diagonalized matrices are often used in SSMs, the motivation for this work is that SSMs are not necessarily expressive enough. Therefore the justification for using a diagonalized matrix is not internally consistent. I think the authors could perhaps add extra ablations on the table 1 experiments where they use a non-diagonal GRU/LSTM (trained sequentially), to demonstrate that there is not much difference between diagonalized and non-diagonalized RNNs. 

+ [minor] The paper is missing some earlier work on linear RNNs and parallelizations of linear RNNs, particularly [1,2,3]


  [1] Bradbury, James, et al. "Quasi-recurrent neural networks." arXiv preprint arXiv:1611.01576 (2016).

  [2] Martin, Eric, and Chris Cundy. "Parallelizing linear recurrent neural nets over sequence length." ICLR 2018

  [3] Qin, Zhen et al, Hierarchically Gated Recurrent Neural Network for Sequence Modeling, NeurIPS 2023

### Questions
+ Could you demonstrate that the diagonalization in equation (9) does not significantly affect the expressivity of the RNN layer?
+ Could you elaborate on why it is important to have a nonlinearity in the sequence mixer specifically, given the use of multiple layers with feature-mixing MLPs in all modern architectures? It is not clear to me, given the strong results of Mamba on most tasks (when equipped with MLPs, not just evaluated as a single layer).

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes FlashRNN, which parallelizes nonlinear RNN application by recasting the full sequence recurrence as a single nonlinear system (Eq. (2)) and solving it with Newton iterations whose linear subproblems have a block-bidiagonal structure solvable via parallel scan.

### Strengths
1. Novelty:  Prior work typically applies Newton/scan to given RNNs (e.g., Lim; Gonzalez). This paper instead redesigns LSTM/GRU so their Jacobians are diagonal or 2×2 block-diagonal, which makes each Newton step’s linear system amenable to efficient parallel reduction without runtime Jacobian approximations.
2. Scale of Experiments: This is the most exciting part. The paper successfully trains 7B-parameter FlashGRU/FlashLSTM models and reports their downstream task results. This proves that FlashRNN is not just a toy model. Moreover, the efficient CUDA implementation further highlights the significance of this contribution.

### Weaknesses
N/A

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
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces FlashRNN, a framework that parallelizes training-time application of nonlinear RNNs by recasting the unrolled recurrence as a single all-at-once system solved with Newton iterations and a custom parallel reduction (prefix-scan) solver over a block bi-diagonal linearization. Implementations span (i) pure PyTorch, (ii) CUDA-accelerated reduction, and (iii) a fully fused CUDA kernel. 

The authors adapt GRU and LSTM (FlashGRU/FlashLSTM) to make their Jacobians diagonal or 2×2 block-diagonal, enabling efficient reductions. They report large speedups over naïve sequential application versus traditional RNNs and show competitive perplexity and downstream results vs. Mamba and a Transformer baseline at 7B parameters.

### Strengths
The proposed idea casting nonlinear RNN application into a Newton+scan routine with a specialized CUDA solver is well motivated and clearly explained.

Thoughtful GPU hierarchy design, e.g. Appendix D2.

The author promised code release which would encourage the community try the proposed method.

### Weaknesses
It's unclear to me how the the diagonal (GRU) and block-diagonal (LSTM) Jacobians limitation been overcomes. Any study shows those limitation not matters at scale?

I'd suggest add more opensource baseline in Table 2.

It's unclear numerical stability when further scale up the model.

It would be interesting to see more ablation on newton iterations, how does the memory footprint change as context length change, and how far the context length can be pushed.

### Questions
See the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces *FlashRNN*, a framework that reformulates the entire recurrent computation in nonlinear RNNs over a sequence as a *single nonlinear system of equations*, which can then be solved in parallel via Newton iterations and prefix-scan parallel reductions.  To make this practical, the authors constrain the RNN Jacobians' structure (e.g., diagonal or block-diagonal) so that the prefix-scan remains efficient and take care of the channel mixing via a MLP, similarly as done in SSMs. They further provide a CUDA implementation that automatically parallelizes user-defined RNN cells and demonstrate training of 7B-parameter nonlinear RNNs (FlashGRU, FlashLSTM) with competitive perplexity to Transformers and Mamba2 while achieving significant (up to 665×) speedups over naïve sequential RNNs.

### Strengths
1. Tackling the core sequential bottleneck of nonlinear RNNs is a long-standing challenge, so the motivation of the paper is clear. The provided solution, i.e. recasting recurrence evaluation as a global nonlinear system of equations and applying Newton iterations combined with parallel prefix reductions, is conceptually clean and theoretically grounded from previous work.

2. The authors release **FlashRNN**, a PyTorch + CUDA framework that generalizes to arbitrary RNN cells, lowering adoption barriers and fostering community experimentation.

3. The authors train the efficient versions of GRUs and LSTMs up to 7B parameters with substantial wall-clock speedups. Few works have shown nonlinear RNNs trained at this scale and doing so has potential to reshape how sequence models are trained.

4. The experimental (including runtime) analysis is thorough. The authors provide profiling, ablations, and comparisons to Transformers and SSMs, which add credibility. 

5. The authors openly discusses convergence challenges, overheads, and structural constraints in the Jacobian.

### Weaknesses
1. As the authors write, imposing diagonal or block-diagonal Jacobians simplifies parallelization but may severely limit expressivity due to channel mixing. The solution taken is similar as in SSMs, i.e. using downstream MLPs to "restore" expressivity. I believe this needs stronger empirical support on more state-tracking tasks where xLSTM performs relatively well.

2. The authors assume that a small, fixed number of Newton iterations (e.g., 3) suffice. However, this is not guaranteed across other recurrent architectures not studied in this paper or different tasks (e.g., time series forecasting, speech, RL, etc.). Where there any scenarios where convergence fails or oscillates? Otherwise, formal guarantees on truncated Newton convergence or stability under limited precision would benefit the paper.

3. Newton iterations can be sensitive to numerical precision.

4. While the authors briefly claim reduced memory scaling relative to BPTT, the paper lacks a quantitative analysis of GPU memory usage across model sizes.

### Questions
1. How difficult is it for practitioners to define a new RNN cell compatible with FlashRNN’s structural constraints?  

2. Have you tried experiments with FlashRNN in mixed or low precision? 

3. Do you plan and how easy would it be to integrate fused CUDA kernels from [Pöppel et al. (2024)](https://arxiv.org/pdf/2412.07752) in FlashRNN?

4. Would the quasi-Newton approximations as in [Gonzalez et al. (2024)](https://arxiv.org/pdf/2407.19115) work also at scale instead of imposing the diagonal matrix structure?

5. How do GPU memory and throughput scale with hidden size?

### Soundness
3

### Presentation
4

### Contribution
4
