# Projected Compression: Trainable Projections for Efficient Transformer Compression

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 2

## Abstract
Large language models have steadily increased in size to achieve improved performance; however, this growth has also led to greater inference time and computational demands. Consequently, there is rising interest in model size reduction methods. To address this issue, we propose \textbf{Projected Compression}, a novel model compression technique, that reduces model weights by utilizing projection modules. Specifically, we first train additional projection weights and preserve access to all the original model parameters. Subsequently, these projections are combined into a lower-dimensional product matrix, resulting in a reduced-size standard Transformer-based model. Unlike alternative approaches that require additional computational overhead, our method matches the per-token computation cost of training a compressed model. Experimental results show that Projected Compression performs especially well with increasing compression rates as high as 90\% compared to other compression methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed Projected Compression (PC), making pruning mask tunable during training and offering a potential to recover and leverage all information of original parameters. Furthermore, authors proved that this approach increased minimal computational cost during continue training, and showed the comparison to hard pruning with retraining.

### Strengths
1. method is simple and effective
2. training cost is well estimated

### Weaknesses
1. Unclear end-to-end improvement - The baseline (HPR with activation-based importance) is relatively weak. The improvements shown in Tables 1/2 are marginal (typically <0.02 loss difference), which is unconvincing given that PC introduces additional trainable parameters (P1, P2).
2. The paper claims PC can "recover and leverage information from parameters that would have been permanently removed,"(line141).  but provides no analysis or visualization showing: 
-  Whether trained P1, P2 actually reactivate initially "pruned" dimensions
- What projection matrices learn compared to initialization
 - Whether the method truly benefits from accessing all original parameters
3. discussions of gradient/PCA based method are missing, where also put more effort on masking. 

[1] [ICML2024] LoRAP

[2] [ICLR2024] sheared llama

[3] [ICML2025] SlimLLM

### Questions
1. Could you provide evidence that PC retains or recovers more important information after training? 
     

 2. How does PC perform compared to other gradient-based structured pruning methods such as Sheared LLaMA? 
     

3. In Table 3, PC shows strong sensitivity to initialization methods (random: 3.16 vs activation-based: 2.94). If the trainable projections can be optimized during training, why doesn't random initialization converge to comparable performance? Does this suggest limited optimization capacity? 
     
4. Without the auxiliary weights Wa, what are the fundamental advantages of PC's learnable projections over static importance-based masks? Can you provide ablation results comparing PC with and without $W_a$?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces the "Projected Compression" (PC) technique, which compresses a weight matrix by pre- and post-multiplying by trainable projection matrices, resulting in a smaller weight matrix. Unlike PEFT methods like LORA, the PC method has as its goal is permanent reduction in model size while maintaining performance.

### Strengths
The training process is efficient, allowing for more rapid learning of the projection matrix weights than for the original large weight matrix entries.

The projection approach better incorporates the original frozen weight matrix information in defining the compressed network than does importance-directed weight pruning of the large network.

The performance of the PC method is significantly better than pruning at high compression rates.

### Weaknesses
A minor quibble, but the equations in the paper should be numbered for easy reference.

I would argue with the interpretation that "Projected Compression allows each token representation to interact with the full capacity of the base model during training, even though only a compressed projection is used during inference. " and "It reinforces the intuition that PC does not discard any information a priori but instead passes token activations through learnable subspaces of the frozen parameter space".
The reduction to a subspace means that information IS lost. Even though the subspace is defined by the larger space, all that one has after compression is access to a set of linear combinations of the original weights. Information is lost. All you can hope for is that the information that is lost is not important for the task at hand, which is the same rationale used for weight pruning.
Note that I am not saying that information loss is bad - network compression requires this - just that the interpretation given in the paper that the compressed network still allows "interaction with the full capacity of the base model during training" is suspect. For one, there is an infinity of different large weight matrices (even ones with random entries) that give the same compressed weight matrix, just with different projection matrices. At best, the involvement of the original weight matrix in defining the compressed weight matrix is that of providing a good initialization to the training. To see this, consider just choosing the large weight matrix to be full rank but with random entries. You can still find the projection matrices that will give the same result after training as when using the original weight matrix. So the only potential advantage would be in the speed and quality of the training. To show this, experiments would need to be done to compare training with the original weights to training with random weights. I expect that the original frozen weights would provide a better initialization than the random weights, but this needs to be shown.

Comparison should be made to training the compressed weight matrix from scratch (equivalent to training a smaller model). Normally this does not work as well as start with a large model and then pruning, and so can provide a useful baseline. It also emphasizes the actual benefit of the PC approach as providing a better initialization of the compressed weight matrix.

A drawback of this method as compared with pruning techniques is that a standard back-propagation training phase is required, which can require a large number of iterations of forward and backward passes, resulting in a heavy computational burden. Pruning is typically done via computing weight importance measures based only on single feed-forward passes of the network. No measures of computational expense are provided in the paper.

### Questions
What would be the difference if one trained the compression process starting from a randomly initialized large weight matrix instead of the pre-trained large weight matrix?

Were the projection matrices only trained using the benchmark data sets? The original weight matrices were presumably trained with very large pretext datasets, so using only problem-specific datasets to train the projectors means that one is doing fine-tuning only, the the projectors would need to be retrained for every down-stream problem, adding to the effective computational burden.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Projected Compression (PC), a model compression technique that learns projection matrices to create a compressed weight matrix. Experiments show that PC outperforms traditional hard pruning, especially at high compression rates like 90%.

### Strengths
1. LLM weight compression is a critical research direction.
2. Learning projection matrices seems new in this line of research.

### Weaknesses
1. The evaluations are insufficient. Only evaluating loss is not a convincing way to evaluate the impact of compression.
2. Also, the compared baselines are limited to Hard Pruning with Retraining (HPR); the paper would be stronger if it compared PC against other modern compression methods.
3. The paper is slightly ambiguous about the final deployed model. It should explicitly confirm that the final artifact is a standard transformer using only the computed $W_C$ weights, with $W$, $P_1$, and $P_2$ discarded.

### Questions
See weaknesses.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Projected Compression (PC) that compress models by training two trainable low-rank matrices attached to the frozen weight matrices. This method ensures the access to the original model weights during the training/compression to optimize the compression processes. This paper shows the effectiveness of its methods by conducting experiments on the pretrained models and Llama3-1B model that it consistently outperforms token pruning retraining on different compression ratios.

### Strengths
- The writing of this paper is well presented with formulas and graphs showing the workflow of this training algorithm.
- Consistently outperforms the baselines under different settings with limitations discussed about memories

### Weaknesses
- *Novelty is limited.* I didn't understand why we need this kind of compression method. A commonly used baseline for the model compressions is directly SVD the original weight matrices to two low rank matrices and train the separate low rank matrices directly. However, I didn't see comparisons over this naive way of compression or any experimental results showing author's method is better than this baseline.
- *The baselines are limited.* There have been many works explores pruning, low rank approximations, as well as other compression methods, such as pruning. In this paper, I only see the prune with retraining with no clear explainations. Authors should compare with more different SoTA methods
- *The improvement over current baselines is limited as well.* From Table 1 and Table 2, the difference between proposed method and the baseline is pretty minimal. 
- *Lack of Analysis of convergence speed.*

### Questions
My central question of this paper is why we want to design such kind of low rank projection training. If we decompose the original weight matrices into low rank matrices using SVD. The decomposed matrices still have access to the knowledge contained in the original weights. This method should also have more efficient training compared to the current paradigm

### Soundness
2

### Presentation
3

### Contribution
2
