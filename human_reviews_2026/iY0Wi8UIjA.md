# SLIM: Structure-aware Low-rank Inference Model

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
This paper introduces a new method for the low-rank compression of large language models. Existing techniques typically compress the weights individually, overlooking the internal dependencies within a transformer block. To address this limitation, we formulate a joint optimization problem to find the optimal low-rank weights for an entire transformer block, thereby minimizing the output reconstruction error. Our formulation allows the incorporation of key architectural elements, including residual connections and normalizations. We then introduce SLIM, an efficient algorithm to solve this optimization problem. Experimental results demonstrate that our method consistently achieves task accuracy improvements of over 5\% compared to existing techniques across a range of compression ratios and model families.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper takes the propagation of low-rank approximation error in transformer and proposes to sequentially update each weight matrix.

Specifically, this paper shows each step reduces to solving a regularized rank-constrained linear regression problem with a closed-form solution and also shows that a gradient-descent–based refinement can further enhance the performance.

### Strengths
1. This paper brings an insight into the low-rank compression for neural networks that have sequential structures. It introduces a sequential optimization at the transformer block level.
2. SLIM is compatible with many other compression techniques, such as pruning and quantization.
3. The gain obtained on pretrained model can be maintained on instruction following fine-tuning.

### Weaknesses
1. The value of the close-form solution is undermined by the gradient-based solution. See Questions 5 and 6.
2. Experiment setups are unclear, which undermines the reproducibility. See Questions 1, 2, and 3,

### Questions
1. What sample data did you use for the low-rank approximation? How did you choose it and why did you use it?
2. What is the initialization of the low-rank weights in gradient-based optimization of Equation (4)? Is it random, truncated SVD, or the close-form solution?
3. For the 10 epochs, did you repeat {optimize Block 1, ..., optimize Block N} for 10 times or optimize Block 1 for 10 times and then optimize Block 2 for 10 times, ...?
4. What is the MSE of SLIM (no OPT)?  
5. What is the computational complexity of the gradient-based optimization of Equation (4)? It should be $O(d^2)$?
6. If the complexity of gradient-based algorithm is $O(d^2)$, what is the value or insight of the close form derivation given the close-form algorithm is much more complex and performs worse than the gradient-based algorithm?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies low rank compression of LLMs. They propose a new method based on reduced rank regression, and provide numerical experiments for different LLMs with up to 7B parameters. The proposed method performs well compared to the baselines such as SVD-LLM.

### Strengths
The numerical experiments appear to be reasonable. There is a diverse choice of models and benchmarks, though some ablations are missing.

### Weaknesses
I think the paper has very limited technical contribution.

- Handling of residuals and normalization: The paper's main claim is that it performs compression based on the whole transformer block instead of a layerwise scheme. However, this claim is not really supported by the paper. In particular, under the sequential compression that this paper studies, the compression happens for one layer at a time, which is just layerwise. Additionally, the residual connections end up as a bias term that does not change the problem fundamentally, and the normalization terms are just approximated out of existence, I believe simplifying it to the choice of $\eta$.

- The effect of $\eta$ is not studied as far as I can tell. Given that normalization is just tuning $\eta$, I think some investigation is required here.

- Comparison with pruning methods: I would like to ask the authors to compare with additional pruning methods, given that LLM-Pruner has memory issues. For example, https://arxiv.org/abs/2406.07831.

### Questions
- Could the author please provide the ablations requested above?

- What happens in terms of quality if one just uses reduced rank regression directly without considering normalization and residual connections? Could the authors please provide some numerical results?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper optimizes the traditional low-rank decomposition methods by jointly optimizing it across the entire transformer block. The author constructs a structure-aware optimization algorithm that consider the residual connections and RMS Norms in the optimization. Experiments across different models and datasets shows SLIM can outperform the baselines over different settings.

### Strengths
- This paper offers a novel perspective that jointly optimize the SVD across after an entire transformer block. This idea is intuitive and clearly presented in the paper.
- Wide ablation studies and analysis. Experiments have shown great potentials for this methods to be applied together with other methods, such as quantization. 
- Structure-aware methods. As shown in the ablation, this methods respects the residual connections and RMS-Norm of the transformer model, which is novel and critical.

### Weaknesses
- Only Wikitext-2 experiment on other model families. Wikitext-2 is a very easy task, while you have included strong models that can solve complicated math and reasoning tasks. I wonder whether you can provide more experiments on other more challenging tasks.
- The greedy algorithm undermines the goal of join optimization and how far are we from achieving the global optimization>

### Questions
- Can't we just align the f(x) where f(x) is each sub-components? I feel like optimizing across an entire layer is equivalent to optimizing across each sub-layer?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes SLIM (Structure-aware Low-rank Inference Model), a new method for compressing large language models by performing joint low-rank optimization across entire transformer blocks instead of compressing each weight matrix independently. SLIM formulates a block-level objective that minimizes output reconstruction error while explicitly accounting for residual connections and normalization layers such as RMSNorm.

### Strengths
1. The proposed joint optimization framework explicitly incorporates key architectural elements such as residual connections and normalization.
2. The method achieves consistent and significant accuracy improvements (over 5%) across multiple LLMs and benchmarks under equal compression ratios.

### Weaknesses
1. The main weakness of this paper is that its primary contribution—joint optimization of low-rank weights across transformer blocks—has already been explored in prior works [1] and [2]. Both studies consider the cumulative effects of cascading reconstruction errors, and the formulations are highly similar: Equation 6 in [1] and Equation 12 in [2] are nearly identical to Equation 6 in this paper. Furthermore, [1] provides a closed-form solution for low-rank compression, where Equation 11 in this paper closely parallels Equations 4 and 5 in [1]. It seems the authors were not aware of these earlier works, as they are not discussed in the article, which makes the only new contribution the inclusion of residual connections and normalization layers.
2. Lines 237–241 should be revised, as this paper is not the first to introduce the concept of block-level joint optimization for low-rank compression.
3. More advanced low rank compression method [1,3] should be compared.


[1] Zhao, Jialin, Yingtao Zhang, and Carlo Vittorio Cannistraci. "Pivoting Factorization: A Compact Meta Low-Rank Representation of Sparsity for Efficient Inference in Large Language Models." Forty-second International Conference on Machine Learning.

[2] Wei, Jiateng, et al. "Structured optimal brain pruning for large language models." Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.

[3] Wang, Jingcun, et al. "Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression." The Thirteenth International Conference on Learning Representations.

### Questions
Check above.

### Soundness
3

### Presentation
3

### Contribution
1
