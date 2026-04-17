# Test-Time Training Done Right

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6, 4

## Abstract
Test-Time Training (TTT) models context dependencies by adapting part of the model's weights (often referred to as fast weights) at inference time. This adapted fast weight, similar to recurrent states in RNNs, stores temporary memories of past tokens in the current sequence. Existing TTT methods have struggled to demonstrate effectiveness in handling long-sequence data, due to their computational inefficiency on modern GPUs. The TTT layers in many of these approaches operate with extremely low FLOPs utilization (often below 5%) because they deliberately apply small online mini-batch sizes (e.g., updating fast weights every 16 or 64 tokens). Moreover, a small mini-batch implies fine-grained block-wise causal dependencies in the data, making them unsuitable for data beyond 1D ordered sequences, like sets or N-dimensional grids such as images or videos. In contrast, we pursue the opposite direction by proposing an extremely large chunk update, ranging from 2K to 1M tokens across tasks of varying modalities, which we refer to as Large Chunk Test-Time Training (LaCT). This approach improves hardware utilization by orders of magnitude, and more importantly, facilitates scaling of nonlinear state size (up to 40% of model parameter size), hence substantially improving state capacity, all without requiring cumbersome and error-prone custom kernel implementations. It also allows easy integration of sophisticated optimizers like Muon for online memory updates. We validate our approach across diverse data modalities and tasks, including novel view synthesis from image sets, language models, and auto-regressive video diffusion models. Our approach can scale up to 14-billion-parameter auto-regressive video diffusion models handling sequences of up to 56K tokens. In our longest sequence experiment, we perform novel view synthesis with more than one million context length. Our results highlight the computational and performance benefits of large-chunk test-time training, paving the way for more efficient and scalable long-context sequence modeling. We hope that this work will inspire and accelerate new research in the field of long-context modeling and test-time training. See visual results on project website \url{https://tianyuanzhang.com/projects/ttt-done-right/}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the problem of low FLOP utilization with TTT layers. They proposes Large-Chunk Test-Time Training (LaCT): updating fast weights over very large chunks compared to prior work to reach high FLOP utilization. As updates of fast weights treats token in the current chunk as sets, TTT layers are paired with sliding window attention to recover locality. The method is validated on novel view synthesis (NVS, up to ~1M tokens), language modeling (per-position loss & RULER/S-NIAH retrieval), and autoregressive video diffusion showing competitive performance with linear time models and Transformers.

### Strengths
1. The suggested solution of combining large chunks with SWA is effective, simple and easy to implement.
2. Evaluation is very robust spanning multiple modalities and tasks, comparing to relevant baselines.

### Weaknesses
1. Combining linear time sequence models with attention is not new and has been shown to improve the performance, see e.g., [1], although not motivated from low throughput.
2. The method strongly relies on large chunk sizes yet the effect of chunk sizes on performance (while maintaining high throughput) is not clear from the results.

[1] An Empirical Study of Mamba-based Language Models

### Questions
1. Is it possible to ablate the effect of the chunk size on performance?
2. Can you clarify the end-to-end complexity where the chunk size equals the full sequence length?

### Soundness
3

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
This paper proposes Large-Chunk Test-Time Training (LaCT), a novel test-time training (TTT) method that updates “fast weights” in large chunks rather than small batches. TTT dramatically improves GPU utilization and scalability of nonlinear state size without requiring custom CUDA kernels.  

LaCT integrates windowed attention for local structure, TTT for long-range context, and is flexibly compatible with N-dimensional data. The method is validated on novel view synthesis, language modeling, and autoregressive video diffusion.

### Strengths
- **Extremely thorough experimentation**: LaCT is tested on three different downstream tasks across two different modalities (text, image/video).  Throughput, scaling, and efficiency experiments are also provided to exhibit the feasibility of LaCT.
- **Clear writing and strong figures**: Lots of preliminaries and related works are provided, as well as a clear linear flow of the paper introducing the research problem and solution (poor TTT GPU utilization caused by small-chunk updated).
- **Scalability and practicality**: Runs efficiently in PyTorch with strong FLOP utilization, making it generally adaptable - especially given the lack of  reliance on custom CUDA kernels.

### Weaknesses
- **Core results deferred to appendix**: Most quantitative comparisons and ablations are in the appendix, leaving the main paper heavy on discussion. For example, the experiments section provides very surface-level details regarding datasets, training details, etc. which are pushed down to the appendix. This makes the actual quantitative impact of LaCT on downstream datasets less apparent. For example, Tables 3 and 4 in the appendix highlight the near identical performance between full attention and LaCT, however without any discussion highlighting that LaCT is more efficient, the takeaway is unclear.  
- **Sparse dataset/training details for autoregressive video diffusion**: The authors very briefly mention using a an internal, proprietary video dataset with minimal transparency, limiting reproducibility.  
- **Limited comparison to alternative TTT methods**: Comparisons against other TTT methods (e.g., Titans, Atlas, which are cited in the main paper under Behrouz) are not provided. While they may be much too inefficient to compare to the scale and downstream tasks of LaCT, explicit quantitative results highlighting this point would be useful.

### Questions
1. How does LaCT compare against previous TTT methods, given its much higher propensity for scalability?
2. Why was LaCT trained on proprietary data when any video diffusion dataset could have sufficed? 

Please address question in the Weaknesses section as well. Overall while the paper is quite dense, the method is interesting, applicable, and thoroughly ablated and tested.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses a critical inefficiency in existing Test-Time Training (TTT) methods when applied to long-sequence data. The authors identify that conventional TTT approaches, which utilize very small online mini-batch sizes for frequent weight updates, suffer from extremely low FLOPs utilization on modern GPUs. To overcome this, the paper proposes a novel paradigm called Large Chunk Test-Time Training (LaCT). The core idea is to move in the opposite direction of prior work by using extremely large chunks of tokens (from 2K to 1M) as the basic unit for updating the model's "fast weights". The authors claim that this approach: (1) Improves hardware utilization by orders of magnitude, leading to significant computational efficiency gains. (2) Enables the effective scaling of large, non-linear fast weights, thereby substantially increasing the model's state capacity. (3) Can be implemented with just a few dozen lines of native PyTorch code, obviating the need for cumbersome and error-prone custom kernel implementations and thus democratizing research in this area. (4) Facilitates the easy integration of more sophisticated test-time optimizers, such as Muon.
The authors embed LaCT within a hybrid architecture that also uses windowed attention to capture local dependencies. The effectiveness of this framework is demonstrated on three diverse and challenging tasks: novel view synthesis, language modeling, and autoregressive video diffusion.

### Strengths
1. This paper correctly identifies a critical bottleneck in existing TTT methods when applied to long-sequence data, and proposes the LaCT using massive chunks for updates instead of tiny ones.

2. This paper shows thorough and rigorous Experiments. (1) Validation on novel view synthesis (image sets), language modeling (1D sequences), and video generation (high-dimensional sequences) powerfully demonstrates the method's generality. (2) The scale of the experiments, such as handling context lengths of over 1M tokens and scaling to a 14B parameter video model, is remarkable and convincingly showcases the method's scalability. (3) The appendix contains ablation studies that systematically analyze the impact of key design choices, including state size, optimizer choice, linear vs. non-linear fast weights, and chunk-wise vs. per-token recurrence. This analysis greatly strengthens the credibility of the paper's conclusions.

3. The paper is well-organized and clearly written. The figures are highly effective to provide the summary of LaCT's advantages, the basic framework and the design choices.

### Weaknesses
1. LaCT fundamentally treats tokens within a large chunk as an unordered set. While the use of windowed attention mitigates this by handling local structure, the inherent limitations of this design choice are not fully explored. For tasks that are extremely sensitive to fine-grained causal ordering, LaCT might struggle. 
2. According to the experimental results in the appendix, LaCT's performance on the VBench benchmark for autoregressive video generation does not show a significant advantage over other AR methods.

### Questions
1. Did the authors observe any specific failure cases in the experiments that could be attributed to the lack of strict causality within a chunk?

2. How sensitive is LaCT's performance to the chunk size? In practice, how would you advise balancing an increase in chunk size (for more compute intensity) against an increase in state size (for more model capacity and memory)?

### Soundness
3

### Presentation
4

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
This paper revisits Test-Time Training (TTT) for long-context modeling and introduces Large Chunk Test-Time Training (LaCT), which updates fast weights over large token chunks instead of small mini-batches. The method aims to improve GPU utilization, support scaling of non-linear fast-weight networks, and integrate window attention for local dependency modeling together with the Muon optimizer for stable fast-weight updates. The authors evaluate LaCT on three modalities: novel view synthesis, language modeling, and autoregressive video diffusion.

### Strengths
1. The proposed approach is general and can be readily adapted to a variety of domains and applications.

2. The evaluation across three domains validates its scalability, with the view synthesis task demonstrating near-transformer performance at much higher efficiency.

### Weaknesses
There are some minor weaknesses that do not affect my rating but do affect readability. Some parts of the paper are hard to follow, especially for readers unfamiliar with TTT, as several key details and related work are placed in the appendix. A few figures are incorrectly rendered with shading artifacts, which should be fixed.

### Questions
1. The authors report 40B-token training for the 760M model and 60B tokens for the 3B model. Have the authors considered scaling experiments with additional training data (~500B-1T) to better evaluate the method’s true language modeling capability? Given the limited token counts, the model’s performance on ARC-c is nearly at random-guess level, which raises questions about the significance of these results.

2. Additionally, given the training setup and results in Table 9, would the 760M model outperform the 3B model if both were trained on the same number of tokens? The current configuration, where the larger model is exposed to only slightly more data, raises the question of whether the observed results reflect true scaling behavior or simply data inefficiency.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this work the author presents LaCT, Large Chunk Test Time Training. This work tries to solve the problem by TTT frameworks where they operate under extreme low FLOPSs (<5%) utilization on the modern GPUs due to low mini-batches (16 or 64 tokens). This work produces large chunk updates consisting 2k-1M tokens across different tasks. The authors validates their claim on three different tasks a) novel view synthesis (NVS), b) language modeling c) auto-regressive video diffusion.

### Strengths
1. This particular method enables large chunk updates enabling higher FLOPs utilization. 
2. The authors claims to implement the LaCT layer simply using PyTorch without custom CUDA kernels.
3. The authors present metrics highlighting the throughput.
4. The empirical results shows the validity of the method on various tasks requiring long context updates.

### Weaknesses
1. One of the major concerns for me is how is the work different from the Fast Weight Programmer and Expressive RNNs. In both the cases the objective is to store the short term memory as the fast weights, next the hidden state in Expressive RNNs [A] basically are used as the fast weights similar to this work. Next the synaptic weight modifications in Expressive RNNs uses a network that uses gradient descent while the update in this particular work is by calculating a gradient $ g $  from a self-supervised loss $ \mathcal{L} $ over the entire chunk. Thus both this work uses update by a learning algorithm Although this particular work uses MUON as an optimizer.
2.  Can the authors clarify that this method is not compared against the Mamba based LM models, as well as the use of the other architectures that handles long context. The authors also might try to use memory-augmented attentions.
3. The positional embedding interaction with the fast weights seems ambiguous, authors need to clarify this for the fast weight layer.
4. It is not very clear how does the authors address the alignment problem, of TTT. The self-supervised loss is not perfectly aligned with the main task, the gradient can be harmful.

Minor Concerns:
1. The paper needs improvement in the writing, it is quite difficult to follow the flow of the paper. 
2. The paper should have compared against other methods such as the current state  RNNs etc. 
3. Few of the abbreviations are assumed to be known, the authors should explicitly state them eg: SMs in Section 2.2.

[A]. https://arxiv.org/pdf/2407.04620

### Questions
Most of the major weaknesses presented are questions. If the authors can clarify these questions, then I will gladly increase my rating.

Why the related work and many other ablations presented in the appendix?

### Soundness
3

### Presentation
2

### Contribution
2
