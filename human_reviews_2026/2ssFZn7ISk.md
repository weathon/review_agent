# Exchangeability in Neural Networks and its Application to Dynamic Pruning

- Decision: Reject
- Scores: 2, 6, 0, 2

## Abstract
Modern neural networks (NN) contain an ever-growing number of parameters, substantially increasing the memory and computational cost of inference. Researchers have explored various ways to reduce the inference cost of NNs by reducing the model size before deployment and dynamically pruning the inference computation at runtime. In this work, we present ExPrune, a general, dynamic pruning optimization that enables multi-granularity partial computation on a per-input basis. ExPrune requires no change to the model architecture or the training algorithm. ExPrune is based on our theoretical results that the relationship between certain model parameters and intermediate values can be described by a statistical property called \textit{exchangeability}. By identifying exchangeable parameters and values in the model, we are able to first partially evaluate the network, analyze the statistics of the partial results, and make pruning decisions on the fly.
Because ExPrune is theory grounded, it generalizes across model architectures in different problem domains. We evaluate ExPrune on one computer vision models, one graph model and one language model. ExPrune provides 10.98--17.33\% reduction in FLOPs with negligible accuracy drop and 21.61--27.16\% reduction in FLOPs with at most 1\% accuracy drop. We also demonstrate that ExPrune composes with static magnitude pruning. On models that have been aggressively statically pruned, ExPrune still provides additional 10.24--11.11\% reduction in FLOPs with negligible accuracy drop and 13.91--14.39\% reduction in FLOPs with at most 1\% accuracy drop.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces EXPRUNE, a novel dynamic pruning method grounded in the statistical concept of exchangeability. The authors formalize how certain neural network parameters and intermediate values possess exchangeable distributions across trained models (viewed as samples from a distribution over random initializations). By identifying exchangeable parameter groups, EXPRUNE enables partial computation followed by statistical confidence checks to prune remaining operations on a per-input basis.

### Strengths
1. The term "Exchangeable parameters" looks novel in the area of pruning.

2. Both CNNs and transformers are considered.

### Weaknesses
1. Some statements are not clear. An example is Figure 4. What's the meaning of different colors? Why does activation B only have a common color with weight W', but not W?

2.  Limited experiments. 

2.1 No ImageNet experiments, which is the standard benchmark for evaluating CNN pruning methods

2.2 Lack of comparison with recent strong baselines in structured pruning and dynamic inference

2.3 No wall-clock time measurements. The paper only reports FLOPs, which is an imperfect proxy for actual speedup. Section 6 (line 435-442) acknowledges hardware acceleration challenges but provides no real timing data. Without wall-clock measurements, it's impossible to assess practical value

3 Prohibitive hyperparameter tuning cost.

3.1 Line 361: 2000 Optuna trials per model on the validation set represents enormous computational cost.

3.2 Figure 3 (line 072) requires training **500 models** from different random initializations to demonstrate exchangeability. This massive training cost is not accounted for when claiming "efficiency" gains. In practice, users have only **ONE trained model**, making the theoretical exchangeability properties impossible to verify

4. Line 749 specifies vastly different threshold ranges for different architectures: [-30,0] for CNNs, [-1,0] for GCN, [-0.005,0] for OPT. These ranges appear to be manually defined without theoretical guarantees or principled derivation

5. So many other dynamic pruning methods exist (layer-wise early exit, token pruning for transformers), but are not compared.

6. The code link at line 488 does not work properly. Except for the README file, all other files show “The requested file is not found.”

### Questions
See weaknesses

### Soundness
2

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
3

### Summary
This paper presents EXPRUNE, a novel, general dynamic pruning optimization designed to reduce the computational cost of neural network inference on a per-input basis. The method is grounded in the authors' theoretical finding that certain model parameters and intermediate values exhibit a statistical property called exchangeability, which implies they are identically distributed and have symmetric interdependence. EXPRUNE leverages this by partially evaluating a network computation, analyzing the statistics of the partial result, and making an on-the-fly decision to prune the rest of the computation if a confident prediction can be made (e.g., the final sum will be negative, resulting in a zero output from ReLU). A key advantage is that EXPRUNE requires no changes to the model architecture or training algorithm, allowing it to generalize across vision, graph, and language models. The evaluation shows it can reduce FLOPs by 10.98-17.33% with negligible accuracy drop and also composes with static pruning to provide additional performance gains.

### Strengths
1. For me, the paper is the first to model the relationship between neural network parameters and intermediate values using the statistical concept of exchangeability. The theoretical insight enables a novel dynamic pruning mechanism by using partial computation as a statistical sample to predict the outcome of the full computation.

2. The method is tested on three distinct and important domains: computer vision (ResNet18-BN), graph property prediction (GCN), and language (OPT), demonstrating the generality of the approach.

3. The paper is well-written and structured logically, making it easy to follow.

### Weaknesses
1. The central performance metric is FLOPs, which is acknowledged as a "proxy for inference efficiency". However, this is a weak proxy for a dynamic pruning algorithm. A stronger evaluation would include wall-clock time or memory-access metrics on target hardware.

2. The paper claims EXPRUNE is a "general, dynamic pruning optimization" that "generalizes across model architectures". While the inclusion of CNNs, GNNs, and a Transformer is good, the bulk of the evaluation relies on the ReLU activation function. Can the core idea be adapted to modern activations like GELU/SiLU?

3. The method looks sensitive to layer-specific hyperparameters. These are found using a 2000-trial optimization search with Optuna, which is a very costly and complex tuning process.

Minor:

1. Some notations are reused. For example, $b$ represents input/output in Section 2 and the model's bias parameter in Section 3.

### Questions
Why select the Wald's test, given that “the assumptions of Wald's test are not strictly met by all exchangeable sequences”?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The article proposes a dynamic pruning algorithm. The algorithm stands on a key concept called "exchangeability" that checks the presence of exchangeable model parameters and intermediate values. Exchangeability enables the method to partially evaluate
the network, and prune some computation on the fly.  The proposed method is architecture independent, thereby facilitating convenient adaptation and migration. The approach has been evaluated on image, graph and question answering tasks.

### Strengths
The main strength of the paper is that it is architecture independent. Thus, it is easily portable, likely to enable hassle free adaptation on new tasks. The exchangeability idea seems interesting (although  I am not fully convinced based on the given explanation). Finally, dynamic pruning has potential to reduce computation cost significantly for very large models, unlike static models.

### Weaknesses
In my opinion the paper has the following major issues: 

1. Even though exchangeability sounds a good idea, I am bit skeptical about it relevance in the context of neural nets training. Once the training starts it is difficult to parameters are going to behave jointly and things get worse with the model size. Thus, I believe the assumptions underlying the  exchangeability for neural nets is hard to prove, even though, theoretically, in isolation "exchangeability" is reasonable. 

2. Dynamic pruning has the problem of "resurrection". It is very difficult to predict, while the training is going on, which parameters are going to contribute when. Some parameters may look exchangeable with others or may look useless, but they may contribute later. The proposed approach is potentially detrimental for this. 

3. I thin the experimental results are very weak in many aspects: i) choice of datasets, ii) choice of baselines, iii) experimental setup.  A work like this requires solid experimental evidence. Unfortunately the paper does not have it. It does not validate the claims made in the paper.

### Questions
In the weaknesses section, I have detailed the main issue. I do not any additional question that has significant importance.

### Soundness
2

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
This paper develops a theoretical framework based on statistical exchangeability to reason about the distribution of neural network parameters and activations from initialization and throughout training. This framework is then used to motivate EXPRUNE, a dynamic weight pruning method that does not require model retraining and has general applicability.

In its current form the paper should be rejected because:
  1. The methodology is largely similar to previous methods with the main contributions being a new theoretical motivation using statistical exchangeability. However the argument for why exchangeability provides greater theoretical understanding than previous approaches is not made clearly or convincing.
  2. The overall experimental insight into performance of the method is lacking.
  3. Several important baselines are not considered.

### Strengths
The extension of statistical exchangeability to the distribution of neural network parameters is interesting and novel.

The top-1 prediction head method is novel and impactful. For transformer language models suitable for the edge deployment, the decoding head can be disproportionately FLOP intensive due to large vocabulary size relative to embedding dimension.

### Weaknesses
A few relevant prior work in dynamic pruning are not included in the literature review.
[1] uses traditional activation and weight magnitude pruning at test time to dynamically prune LLM weights.
[5] first performs the operation in low precision to predict negative outputs that will be dynamically pruned.
[6] performs the operation in decreasing order of significant bits.

There are also connections to the broader dynamic pruning/sparsity literature such as [11, 12, 13, 14] as well as other pruning methods that are not training free that are not made directly. This could be an opportunity to highlight that EXPRUNE is training and calibration free.

The experimental section does not have enough substance.
The paper only includes one major experiment conducted on 3 different model types (CNN, GCN, Transformer LLM) as well as on a statically pruned CNN.
Only an unoptimised baseline is used with the addition of SnaPEA for the first CNN experiment.
The paper would greatly benefit from a significantly expanded set of experiments, baselines and ablations.

In the main experiment, after a set of suitable hyperparameters is identified on the validation set. The subset containing the first 5 pareto slices is plotted. This is test set contamination.

### Questions
- Throughout the paper it is highlighted that the method does not require a specialized model or training algorithm, however it is not explicitly stated that retraining or calibration also is not required. A reader would benefit if this is emphasised early.
- Why was computation ordering not tested?
- Why was "ReLU Strikes Back" inference methodology (exploiting input sparsity) not used as a naive baseline with the normally trained ReLU models?
- Why is EXPRUNE not composed with exploiting input sparsity?
- With regards to baselines, direct comparison to other neuron level dynamic pruning methods such as [1, 4, 5, 6] would make the papers position in the literature more clear and highlight any unique benefits a statistical exchangeability based approach brings.
- The error free exact mode of SnaPEA was used in the experiments, however, a lossy predictive mode also exists. Why was it not used to show the FLOP vs accuracy trade off of the SnaPEA method? 
- Does EXPRUNE also compose with activation aware static pruning methods? [2, 3, 7, 8, 9] What further insights are there to the importance of dynamic vs static activation information when pruning? 
- Ideally there would be additional experiments demonstrating exchangeability in a practical setting would help bridge the exchangeability theory to the EXPRUNE method.
- Statistics of early ReLU classification for different computation order, check interval or other variables would all provide further insight in to the workings of the method and provide value to the reader.
- How does performance and early prediction ability vary across scales? [10]

I would raise my score if the experimental section was significantly expanded and more in depth and if I was convinced of a more fundamental connection between the statistical exchangeability framework and the EXPRUNE methodology.

Additional questions unrelated to the score:
- Is the computation order random or static?
- Can the top-1 prediction head be generalized to allow early sampling?

References:

[1] Koike-Akino et al. μ-MoE: Test-Time Pruning as Micro-Grained Mixture-of-Experts https://arxiv.org/abs/2505.18451

[2] Sun et al. A Simple and Effective Pruning Approach for Large Language Models https://arxiv.org/abs/2306.11695

[3] Frantar et al. SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot https://arxiv.org/abs/2310.04564

[4] Chen et al. CompRRAE: RRAM-based Convolutional Neural Network Accelerator with Reduced Computations through a Runtime Activation Estimation https://arxiv.org/abs/1906.03180

[5] Suresh et al. Early Prediction of DNN Activation Using Hierarchical Computations https://doi.org/10.3390/math9233130

[6] Ibrahim et al. DSLOT-NN: Digit-Serial Left-to-Right Neural Network Accelerator https://arxiv.org/abs/2309.06019

[7] Liu et al. AWP: Activation-Aware Weight Pruning and Quantization with Projected Gradient Descent https://arxiv.org/abs/2506.10205

[8] Hussien et al. Small Contributions, Small Networks: Efficient Neural Network Pruning Based on Relative Importance https://arxiv.org/abs/2410.16151

[9] Bhuiyan et al. Z-Pruner: Post-Training Pruning of Large Language Models for Efficiency without Retraining https://arxiv.org/abs/2508.15828

[10] https://huggingface.co/SparseLLM

[11] Yuan et al. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention https://arxiv.org/abs/2502.11089

[12] Shazeer et al. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer https://arxiv.org/abs/1701.06538

[13] Raposo et al. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models https://arxiv.org/abs/2404.02258

[14] Elhoushi et al. LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding https://arxiv.org/abs/2404.16710

### Soundness
2

### Presentation
2

### Contribution
2
