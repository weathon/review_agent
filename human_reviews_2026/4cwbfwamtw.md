# ARA: Adaptive Rank Allocation for Efficient Large Language Model SVD Compression

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 6

## Abstract
In the field of large language model (LLM) compression, singular value decomposition (SVD) is a widely studied and adopted low-rank decomposition technique. Since SVD operates exclusively on linear modules, and these modules in LLMs are separated by nonlinear components, SVD can only be applied independently to each linear module. Under a global compression ratio constraint, determining the appropriate rank for different linear modules becomes a critical problem. Existing approaches, such as heuristic algorithms and mask-based training, have made progress in addressing this challenge. However, these methods still suffer from several limitations: heuristic algorithms explore the solution space within restricted regions, while mask-based training struggles to efficiently capture the relationship between singular value spectra and trainable parameters. More importantly, current methods overlook the key property that the gain function is non-smooth at a compression ratio of 1, which often leads the training process to suboptimal local minima. To address these issues, we propose an Adaptive Rank Allocation (ARA) method. Specifically, (1) ARA introduces a dedicated mask design that enables efficient mapping and updating between retained ranks and trainable parameters; and (2) it employs an additional loss function to guide parameter selection toward globally optimal solutions. Experimental results demonstrate that ARA achieves state-of-the-art performance. On the LLaMA2-7B model with a 80\% compression ratio, ARA reduces perplexity on WikiText2 from 8.38 to 6.42 and improves average zero-shot task accuracy by 9.72 percentage points compared with uniform compression. These results highlight the effectiveness of our method for rank allocation in SVD-based LLM compression.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Adaptive Rank Allocation (ARA), a method for allocating different compression ranks across linear modules in LLM compression via SVD. The key contributions are: (1) a staircase mask design that enables efficient rank allocation with monotonicity and global update influence, (2) a guidance loss function that handles the discontinuity at compression ratio = 1 by allowing modules to retain full-rank matrices when SVD would be parameter-inefficient, and (3) strong empirical results showing significant improvements over baselines. I have some concerns about the experiment and relationship between this paper and previous work.

### Strengths
1. Strong Empirical Result.
2. Novel Treatment of R≥1 Case: The guidance loss Lg and dynamic computational flow (Equation 8) address a genuine problem that prior mask-based methods overlook. This is a valuable contribution.
3. Improved Mask Design: The staircase binary matrix parameterization ensures monotonicity while avoiding vanishing gradients (unlike tanh-based masks) and maintaining global receptive field (unlike Gumbel-Sigmoid masks). The technical design is sound.

### Weaknesses
1. Missing Baselines and Citations. 
The paper should cite the following two papers in both literature review and experimental comparisons:
1) TFWSVD—EMNLP 2022, titled "Numerical Optimizations for Weighted Low-rank Estimation on Language Model
2)  RankDyna— EMNLP 2023 findings, Dynamic Low-rank Estimation for Transformer-based Language Models

TFWSVD is the follow-up of FWSVD, in a more accurate way. More importantly, that paper proposes a Fisher information variance metric φ(W) that predicts when SVD will fail for a given matrix, which is conceptually very similar to what ARA's guidance loss is trying to achieve. Matrices with high φ(W) should be the ones where ARA's guidance loss activates and chooses full rank. 
Second, RankDyna performs dynamic rank selection for SVD compression of LLMs. The key difference is that RankDyna learns ranks during fine-tuning while ARA learns them separately on calibration data. The computation cost may be close, please consider add it as a baseline. 

2.   Technical concerns
The straight-through estimator used to backpropagate through the binary mask introduces gradient bias that isn't discussed. The paper doesn't analyze training stability, convergence properties, or sensitivity to initialization. 
3. Experiment concerns.
The paper mentions: "ARA rescales the compression ratios of all modules proportionally after training to precisely meet the target." Should be discussed more.

### Questions
1. Is the guidance loss learning the same thing that φ(W) (proposed in TFWSVD) already measures?
2. If not, can we consider some analysis based on φ(W)? For example: φ(W)-Based Threshold for Activation, φ(W)-Weighted Guidance Loss, φ(W)-Based Initialization?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Adaptive Rank Allocation (ARA), a method that learns layer-wise ranks for large-language-model SVD compression. ARA employs a trainable monotonic mask generated by a simplex-constrained vector and a staircase binary matrix; a guidance loss automatically switches layers to their full-rank weights when low-rank decomposition is parameter-inefficient. Trained on small calibration data with straight-through estimation, ARA produces a discrete binary mask that exactly meets a target compression ratio. Tests on LLaMA-2 and Qwen at 60% and 80% compression report lower perplexity and higher zero-shot accuracy than uniform truncation and several allocation baselines.

### Strengths
- Converts the discrete rank-selection problem into a continuous, simplex-constrained optimisation that is easy to train with standard back-propagation.
- The proposed staircase-mapping mask ensures monotonicity with respect to singular values, preserving the theoretical optimality of the Eckart–Young theorem and avoiding the instability of Gumbel-Sigmoid or tanh masks.
- The framework remains orthogonal to pruning and quantization, and can be combined with both for further efficiency gains.

### Weaknesses
- The monotonic mask enforces that singular values with larger magnitudes are always prioritized. As a result, ARA cannot invert or locally re-weight the importance of individual singular directions—only adjust the global cutoff boundary. This restricts its ability to capture task-critical, low-energy directions that matter more to downstream performance than to reconstruction error.
- The optimization objective remains dominated by spectral energy preservation and cross-entropy loss. It does not directly model task sensitivity or gradient-based importance, leading to suboptimal trade-offs when “energy ≠ task importance”. 
- The experimental design compares different allocation strategies using only the data-aware SVD algorithm from SVD-LLM. While this ensures consistency, it fails to account for the algorithmic contributions of competing methods, potentially disadvantaging approaches that integrate allocation with novel compression techniques. A more equitable comparison would involve contrasting SVD+ARA against methods like Dobi-SVD that incorporate both algorithmic innovations and allocation strategies.
- The largest tested model (14B parameters) falls short of demonstrating scalability to larger-scale models (e.g., 70B), which are more representative of real-world deployment scenarios.

### Questions
- The evaluation is limited to LLaMA and Qwen architectures, which share substantial structural similarities. It remains unclear whether the method generalizes to diverse model architectures, particularly MoE models.
- It is unclear whether this method can be extended to other low-rank decomposition algorithms besides SVD.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes ARA, an adaptive rank allocation scheme for SVD-based compression of LLM linear layers. It learns per-layer binary masks over singular values with a monotone “staircase” parametrization, uses a straight-through estimator during training, and adds a full-rank guidance loss to decide when not to decompose a layer (i.e., revert to the dense matrix) under a global compression target.

### Strengths
- A simple, monotone mask construction that avoids vanishing gradients and enforces preference for larger singular values; the STE makes training match inference.

- Competitive numbers across multiple models and tasks, with tables showing consistent improvements over prior SVD rank-allocation methods.

### Weaknesses
- Incremental conceptual novelty. The core ideas that monotone masks over singular spectra and an auxiliary term that favors keeping full rank when low-rank is inefficient—extend well-known SVD truncation and mask-learning practices. The “guidance loss” formalizes a heuristic already implicit in prior pipelines that skip SVD when k(m+n)>mn. The theoretical treatment does not go beyond standard truncation analyses. 

- Claims target “efficient LLMs,” yet there is no end-to-end latency/throughput study on real inference stacks (e.g., vLLM/SGLang). The work optimizes perplexity and zero-shot scores, but does not show serving wins or memory bandwidth reductions for prefill/decoding under realistic batching.

- The method requires input-aware SVD and mask training with extra losses; experiments train for multiple epochs on a small calibration split (first shard of C4, 256×512 tokens) with several new hyperparameters, which risks overfitting and weakens external validity. Ablations are narrow and do not test sensitivity to these choices. 

- The objective uses a soft global-rate penalty and a post-hoc rescaling to hit the target ratio, which can change layer allocations after training.

### Questions
- Can you present end-to-end serving results (latency, tokens/sec, memory traffic) in vLLM or SGLang with continuous batching under mixed sequence lengths, comparing dense vs SVD-ARA models at the same accuracy?

- Does the post-training rescaling of layer rates change the final allocation materially? If so, why is training not run with the final rates to avoid train–test mismatch?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper focuses on the rank allocation problem in SVD-based LLM compression. Because SVD is applied per linear module, deciding how many singular values to keep under a global compression budget is an important issue. The paper proposes ARA, which learns a monotonic, globally influenced mask over singular values and adds a full-rank guidance loss so the model can decide to keep a layer dense when low-rank is actually worse. Calibration training is done on 256 C4 samples and evaluated on LLaMA2 and Qwen models. The proposed method shows large gains over baseline methods at 80% and 60% compression.

### Strengths
* The rank-selection problem itself is important and worth studying.
* The scenario discussed in Section 3.3 where compression actually offers no gain and and the method should fall back to the full-rank weight is also an important but sometimes under discussed topic.
* The paper also shows that the proposed approach is orthogonal to other techniques such as quantization.

### Weaknesses
* The method section is not very well presented. I would suggest to improve the clarity for Section 3.2 and consder bringing Algorithm 1 into the main text or explicitly referencing it in the main text.
* The paper keeps saying prior mask methods get stuck in local minima and ARA “reaches the global optimum” (Fig. 2), but what they actually do is add a simple guidance loss term, this does not represent a general optimality result. This statement should consider to be toned down.
* There is no efficiency report in the paper. Since the loss incorporates multiple terms, it would be better to include a wall-clock comparison with baseline methods.

### Questions
* The current calibration uses a random subset of the C4 dataset. How robust is this design choice? In particular, how much do the results change if a different subset is sampled or if a different calibration dataset is used?

### Soundness
3

### Presentation
2

### Contribution
3
