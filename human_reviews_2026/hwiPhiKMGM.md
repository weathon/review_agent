# Enjoy Your Layer Normalization with the Computation Efficiency of RMSNorm

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Layer normalization (LN) is a milestone technique in deep learning and has been widely adopted across various network architectures. However, LN introduces additional computational costs in the inference process. This issue has been addressed by its counterpart, RMSNorm, which removes the centering operation. This paper explores how to retain the theoretical advantages of LN while achieving the computational efficiency of RMSNorm. We first propose a general framework to determine whether an LN in any DNN can be equivalently replaced with RMSNorm. We introduce the methodology for removing the centering operation of LN after a linear layer with mathematical equivalence, by proposing column-based wight centering (CBWC) on linear layer. We further define the foldable LN---i.e., that can be replaced by RMSNorm without altering model behavior after applying constraints onto certain layers, and introduce zero-mean graph to analyze whether any LN in arbitrary given neural network is foldable. We present an algorithm that automatically detects foldable LNs and show that most LNs in currently widely used architectures are foldable, which provides a straightforward benefit in reducing the computational costs during inference. Additionally, we conduct extensive experiments to show that 'CBWC+RMSNorm' achieves performance comparable to vanilla LN, while improving efficiency during training, even in cases where the LN is not foldable.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper's central idea is to propose a new layer normalization method which attempts to retain the preformence of traditional LN (centering+scaling) but with the computational savings of RMSNorm (which essentially skips the centering)
They propose CCC (Column-Centered Constraint) and CBWC (Column-Based Weight Centering) to force the linear layer before LN to produce zero-mean outputs, so LN’s centering becomes redundant and LN can be replaced by RMSNorm without changing behavior (for inference; training equivalence is broken by dropout). They generalize via a zero-mean graph to define foldable LNs and provide a detection algorithm. Empirically they report 50–80% norm-op speedups compared to LN, translating to  2–12% end-to-end model inference gains on several mid-scale models, with comparable accuracy LN on many tasks, albeit mostly lower.

### Strengths
Overall the paper is clear and mostly straightforward on what it sets out to do and the mathmatical framing is rigirous, here are some specefic points worth mentionong:

1-Simple, general approach: CCC (constraint) + CBWC (reparam) give a clean LN→RMSNorm swap wherever the graph is foldable, the formalization is clear and the problem is directly addressed.

2-Graph-level criterion & tool: “Foldable LN” + zero-mean graph + Algorithm 1 to auto-detect eligible LNs across architectures is usful although it would be good to have all the code shared together.

3-Latency motivation is real, for LLMs any savings are practically good to have, and the fact that RMSNorm was popular, even with its modest savings, show that. So further savings are good, and in this paper they quantify LN’s share (around 7–12% of wall-time) and  show some that model-level wins ( around 2–12%) are plausible in low-batch inference, although modest.

4-Empirical picture is mostly stable: CBWC+RMS ≈ LN on AG News and SWIN (ImageNet-100) and > RMSNorm; translation is slightly worse than LN but > RMS.

### Weaknesses
1- Experimental scope is too narrow for an efficiency paper, the results are scattered across a few mid-scale tasks (Multi30K, AG News, ImageNet-100) and modest model sizes with no clear presentation of the hyperparameters. For a paper whose core claim is efficiency and generality, there is no systematic study across a broad model zoo, hardware configurations, or kernel baselines. The study feels scenario-specific rather than comprehensive. If gains are modest, they must at least be universal and well-quantified. Moving some of the results fro Appendix H can help a bit.

2-The main benefit is small end-to-end inference speedups (~2–12%), yet: there is no throughput evaluation (only latency sketches), no comparison to vendor-fused or PyTorch-native optimized LN kernels, and strangely, the paper uses its own CUDA RMSNorm instead of PyTorch’s, creating implementation confounds and making fairness unclear.
If the paper argues about deployment-level efficiency, the methodology needs to be much more standardized and more gains should be shown, and also the need for gains, however modest, should be argued better in a practical sense, for example energy consumption at large scale deployment comes to mind.

3- Training-time “equivalence” is not rigorous. They explicitly admit dropout breaks the zero-mean property, so training claims are empirical, and some tasks show accuracy regressions vs LN which is expected in a way but should be clearly communicated as a trade-off with an argument for why the trade-off makes sense for a user.

4-Expressiveness/constraint risk not fully explored. CCC removes one DoF per column (no mean-shift outputs); interactions with bias, weight decay, and optimizers are not deeply analyze, at least not in the main text. 

5- Presentation quality (figures/tables). Many captions are one-liners that don’t explain the setup/findings; some axes/labels are missing or unclear, reducing interpretability. This hurts reproducibility and clarity of the empirical case and overall harms the flow of the paper.

6-No code is provided, and the paper uses a custom CUDA RMSNorm kernel instead of standard PyTorch implementations. For an efficiency-focused paper, open code is essential to verify fairness, foldability detection, and wall-clock improvements. Without it, the claimed speedups are not independently verifiable.

Overall, the practical robustness, empirical validation, and readability fall short of top-tier standards. It contributes a clean theoretical tool (CBWC + foldable LN framework), but the gains are incremental, the experiments modest, and the exposition dense.

### Questions
1-Why compare against your own RMSNorm CUDA kernel instead of standard PyTorch or fused vendor kernels?

2-Can you report throughput, not just latency, across multiple batch sizes?

3-What is the real deployment benefit on hardware with already-fused LN (A100, H100, TensorRT-LLM, etc.)?

4-How unstable are results with dropout, GELU, and nonlinear blocks in deeper networks? 

5-Do you think CCC constraints measurably reduce representational capacity in more challenging settings, and how would you ingestigate that more properly?

6-How significant are the observed efficiency gains (2–12%) in real inference pipelines where LN kernels are already optimized or fused?

7-How scalable and reliable is Algorithm 1, the foldable-LN detection method, when applied to very large computational graphs such as those in billion-parameter Transformers?

8-Why were the experiments limited to small and medium-scale models, and how would the framework perform on large-scale LLMs or vision transformers?

9-Are there quantitative ablations isolating the effects of CBWC, RMSNorm replacement, and their combination on performance and stability?

10-Could similar "folding" techniques be generalized to other normalization types (GroupNorm, InstanceNorm) or normalization-free architectures?

11-Given the modest speedup versus implementation complexity, in what practical scenarios is the proposed framework worth adopting?

### Soundness
2

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
5

### Summary
This paper explores methods to improve the computational efficiency of LayerNorm to make it as efficient as RMSNorm. For linear layers followed by LayerNorm, the re-centering operation can be folded into the weight parameters (i.e., column-wise recentering, or CBWC). Theoretically, CBWC+RMSNorm is equivalent to LayerNorm. The authors proposed algorithms to automatically detect foldable LayerNorm, and found most norms in modern architectures are foldable. Experiments on a range of models and tasks show the promising performance of CBWC+RMSNorm.

### Strengths
* The proposed method offers another perspective to improve the efficiency of LayerNorm.
* CBWC+RMSNorm achieves comparable running efficiency to RMSNorm; such efficiency could be interesting to the research community.
* The experiments are based on a set of tasks and models.

### Weaknesses
* The proposed method is not general, i.e. it’s not a drop-in replacement to LayerNorm. 
* The experiments are based on small-scale dataset and models, which is less convincing in the era of LLMs.

### Questions
1. An important pre-condition of this study is that LayerNorm could outperform RMSNorm. The analysis in Figure 4 is very important and interesting. Could you please show how the input norm changes across layers? 
2. Also, the paper reported that RMSNorm often performs worse than LayerNorm, but in the original RMSNorm paper and the Pre-CRMSNorm paper, RMSNorm performs comparable to LayerNorm. Is there any chance that such a difference is caused by sub-optimal optimization for RMSNorm?
3. The translation experiments should use larger-scale datasets, such WMT En-De or En-Fr, to make the results convincing.
4. To demonstrate the effectiveness, it’s important to show the scaling ability of the new method over different model sizes so that readers can have a better understanding on training stability, performance and running efficiency. The current experiments mainly focus on small models which are not that convincing.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores a method to bridge the gap between Layer Normalization (LN) and RMSNorm, enabling models to benefit from the stability of LN while enjoying the computational efficiency of RMSNorm. The authors note that the mean subtraction step in Layer Normalization can be eliminated without degrading performance if the input vectors are zero-mean. Based on this observation, they propose two techniques: Column Centered Weight Transformation (CCWT) and Column Based Weight Centering (CBWC). CCWT allows for the conversion of pretrained LN-based models to RMSNorm during inference by adjusting the weights of adjacent linear layers to ensure that the inputs to the normalization layers are zero-mean. CBWC is a training-time strategy where linear weights are constrained so that their outputs are zero-mean, which enables RMSNorm to serve as a drop-in replacement for Layer Normalization during training. The authors theoretically and empirically demonstrate that RMSNorm, when combined with CBWC or CCWT, behaves equivalently to Layer Normalization, achieving similar performance with lower computational costs, particularly during inference.

### Strengths
1. The paper defines and establishes the conditions under which any Layer Normalization (LN) in deep neural networks can be transformed into RMSNorm. This framework allows for both training and inference to proceed without the need for the centering step used in LN.
2. This approach reduces inference overhead by eliminating the mean subtraction step found in Layer Normalization.

### Weaknesses
1. The weight transformation and constraints may complicate training workflows.
2. Replacing LN entirely may risk instability in edge cases, especially if zero-mean constraint isn't strictly met.

### Questions
1. It appears that the proposed method is applicable to any Layer Normalization (LN) deep networks. Could the method be tested on Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), or Recurrent Neural Networks (RNNs) to further validate its effectiveness?
2. Column-Based Weight Centering is used to enable RMSNorm to replace LN during training. However, could this replacement constrain the learning dynamics? Might it limit weight flexibility and cause potential side effects? I would appreciate some discussion on this topic.
3. Regarding Column-Based Weight Centering, could the authors provide a detailed analysis of the runtime or training overhead? This would help in gaining a better understanding of its implications.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a method that centers the linear layer weights so that the output has naturally mean zero (column-based wight centering, or CBWC), so that RMSNorm and LayerNorm are equivalent, and one can use CBWC + RMSNorm to achieve the representation power of LN. It also proposes a graph framework to analyze which LNs can be turned into this ("foldable") in a given neural architecture. Experiments show acceleration over LN and better accuracy over RMSNorm can be achieved.

### Strengths
1. The goal of combining the theoretical advantages of LayerNorm with the computational efficiency of RMSNorm is clearly motivated and addresses a relevant practical problem.

2. The paper provides experimental evidence across different modalities (vision, language, and multimodal tasks), demonstrating that the proposed framework is not limited to a single domain.

3. Theoretical analysis is thorough, providing clear mathematical conditions and proofs for when LN can be safely replaced by RMSNorm.

### Weaknesses
1. The experiment section is relatively weak and limited in scope. Only very few comparison data points are reported, and they are on small-scale settings.  Most results are small-scale (e.g., CIFAR-10, MNIST, ImageNet-100) or conducted on relatively small models, with only superficial measurements of acceleration (for example, the whole model acceleration is not measured, only estimated). There is no large-scale validation on modern foundation models or a comprehensive analysis across diverse architectures and tasks. In some cases, only plots are provided, with no clear quantitative number comparisons. 

2. In practice, the proposed method may be cumbersome to apply. It is not a simple drop-in replacement for LayerNorm, since one must first determine which LNs are foldable and apply CBWC accordingly. This adds nontrivial implementation overhead and limits the method’s practicality for large, existing architectures.

3. The paper’s presentation could be significantly improved. Key ideas are not clearly separated or motivated, and the logical connections between theoretical sections and practical usage (e.g., training vs. inference) are often implicit.

4. The comparison of RMSNorm vs. LN with empirical evidence is not sufficient. Based on my knowledge, they do not have many performance differences in LLMs to begin with. The motivation part could be improved by having a comparison between RMSNorm and LN, both in accuracy and speed. 

Part of review is revised with LLM assistance.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
