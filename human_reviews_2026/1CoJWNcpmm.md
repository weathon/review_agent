# You Do Not Fully Utilize Transformer's Representation Capacity

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
In contrast to RNNs, which compress their history into a single hidden state, Transformers can attend to all past tokens directly. However, standard Transformers rely solely on the hidden state from the previous layer to represent the entire context. We show that this design choice induces representation collapse and degrades performance. To address this issue, we introduce Layer-Integrated Memory (LIMe), a lightweight extension that leverages existing key–value buffers and learns per-head, per-layer routing weights to integrate representations from all previous layers with negligible overhead. Through extensive experiments—including language modeling, synthetic reasoning benchmarks, and very deep architectures—LIMe consistently achieves faster convergence, lower perplexity per FLOP, and substantial accuracy improvements on synthetic tasks while preserving higher value–vector entropy and improved token separability. Finally, our analysis of the learned routing weights reveals systematic reuse of both local and long-distance features, demonstrating how LIMe mitigates collapse, unlocks richer representations without increasing hidden-state size, and points to promising directions for future research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies "representation collapse" as a key weakness in standard Transformer decoders, where the reliance on a single residual stream from the immediately preceding layer forces the model to compress all prior information, leading to a loss of feature diversity in deeper layers. To address this, the authors propose LIMe, a lightweight architectural modification. LIMe allows each attention head at every layer to compute its KV representations by routing and mixing the KV buffers from all preceding layers, not just the current one. This is achieved by learning a per-head, per-layer routing matrix that weights the contributions of past layers.

### Strengths
- The primary strength of LIMe is its elegance and low overhead. By reusing existing KV buffers, it adds multi-layer information flow with almost no additional memory and a negligible computational cost (especially when GQA is used). This makes it a very practical and "drop-in" friendly modification.

-  The paper does an excellent job of clearly identifying a specific problem (representation collapse) and proposing a solution (LIMe) that directly targets it.

### Weaknesses
- The authors correctly identify in the limitations that the vanilla implementation of the router has an $\mathcal{O}(L^2)$ asymptotic complexity (where $L$ is the number of layers), as each layer's router must process keys from all $L-1$ previous layers. This is fine for the 16-layer models in the main paper, but it will become a significant computational bottleneck for scaling to very deep models (e.g., $L=100+$). The heuristic ablations in Appendix F (e.g., last-j or first-j) all show worse performance, suggesting a difficult trade-off between performance and scalability.
- The method's core idea, accessing all previous KV caches, creates a practical implementation challenge for large-scale training. In a standard pipeline parallel setup, this would require significant communication across pipeline stages (GPUs), as later layers would need to fetch KV caches from all earlier GPUs. The authors acknowledge this and their preliminary test shows a ~7.8% latency overhead. This practical hurdle might deter adoption for training SOTA-scale models, as it requires "non-trivial engineering effort" to optimize.

### Questions
Please refer to my weakness part.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper starts from the observation that in standard transformer networks, there is a single residual stream, meaning that the representations from all the previous layers are compressed into a single hidden state. This single hidden state is then used as the input of the next layer. This can lead to *representation collapse*, which is the phenomenon where different tokens become undistinguishable. Hence, this paper propose a new mechanism to address this issue, called LIMe. The idea is that each layer can attend to the representations of *all* previous layers, instead of just the immediante previous one. In practice, this is done by modifying the way keys and values are computed. Instead of just using the keys and values computed from the input of the current layer, the keys and values from all previous layers are linearly combined, using trainable weights. Said otherwise, the keys and values of used in the attention of layer L are obtained by doing a linear combination of the keys and values of all the heads of the previous layers. The weight of this linear combination are fixed trainable parameters.

The proposed method is then empirically evaluated on different language modeling tasks. First, a LLaMa like model, with 1B parameters is trained on 50B tokens, and evaluated on downstream NLP tasks such as QNLI, WiC or ARC (easy/challenge). Here the experiments show that LIMe obtain better performance than the standard transformer architecture, as well as other approaches such as DenseFormer or HyperConnections. Then the model is compared to the standard transfomer on GSM8k or synthetic tasks such as arithmetic expression evaluation, again showing that LIMe performs better than the baseline. There are also ablations studying the *representation collapse* showing that LIMe is less prone to representation collapse than standard transformers.

### Strengths
I am a bit of the fence regarding this paper.

In terms of strengths, I believe that the proposed idea in the paper is simple and elegant. The paper is clearly written and easy to follow. The experimental evaluations are convincing.

### Weaknesses
My main concern with the paper is its relation to previous work, and especially its significance with respect to these.

First, I believe that the paper does not make a great job discussing the difference with previous work such as DenseFormer, or Value Residual Learning. More precisely, I think that the idea of combining the representations from multiple previous layers instead of just using the representation from the previous layer is not new. The contributions of the paper are thus mostly about details of how this idea is implemented in practice, and the paper could do a better job at discussing these. Moreover, I believe that the baseline considered in the paper (DenseFormer, HyperConnection) have multiple variant considered in the original papers, and the details of which one is used are missing. Finally, I am a bit surprised that the baseline (such as DenseFormer) does not seem to improve compared to the standard transformer, which goes against the claim of the original paper.

Another minor concern is the additional runtime required by the method, as it needs to read significantly more activations from memory compared to the standard transformer. 

**Additional references**

*MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections.* Da Xiao, Qingye Meng, Shengping Li, Xingyuan Yuan. 2025.

*Value Residual Learning.* Zhanchao Zhou, Tianyi Wu, Zhiyun Jiang, Fares Obeid, Zhenzhong Lan. 2024

*LAUREL: Learned Augmented Residual Layer.* Gaurav Menghani, Ravi Kumar, Sanjiv Kumar. 2024

*DeepCrossAttention: Supercharging Transformer Residual Connections.* Mike Heddes, Adel Javanmard, Kyriakos Axiotis, Gang Fu, MohammadHossein Bateni, Vahab Mirrokni. 2025

### Questions
Which variant of DenseFormer and HyperConnection did you use?

Did you re-implement the baselines yourself or use existing code?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Layer-Integrated Memory (LIMe), which allows each attention head to access Key-Value representations from all previous layers through learned routing weights.

### Strengths
Comprehensive experimental design: The evaluation spans multiple dimensions: language modeling perplexity, mathematical reasoning on GSM8K, and synthetic tasks with controlled difficulty levels. The representation collapse analysis combines entropy measurements, linear separability tests, and grammatical probing to validate the core hypothesis from different angles. The routing weight analysis provides interpretability by revealing which layer representations the model prefers to access. This is the most lovely part of this paper.

### Weaknesses
1. Limited novelty over prior work. The core mechanism of using learned weights to aggregate multi-layer representations appears in Transparent Attention (Bapna et al., EMNLP 2018), which uses trainable softmax-normalized weights to combine encoder layer outputs in NMT decoder cross-attention. The mathematical formulation resembles that prior work, with the main difference being application to decoder-only self-attention. More recently, Hyper-Connections (Zhu et al., Sept 2024) addresses representation collapse through multi-stream connections with learned routing, sharing similar motivation. The paper does not clearly articulate what architectural insight LIMe provides beyond adapting these known techniques to decoder-only models with efficient KV buffer reuse.
2. Unclear computational cost analysis. The paper claims "negligible overhead" yet mentions O(L**2) routing complexity in limitations. For a 64-layer model, each layer must route over 64 previous layer KV pairs, but the paper does not provide memory bandwidth analysis for this case. The pipeline parallelism overhead of 7.8% contradicts the "negligible" claim for production scenarios.

### Questions
N/A

### Soundness
4

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
The paper suggests adding a weighted average after the standard key projection. The average is taken over all the key representations of the current token in the current layer and head as well as the previous ones (over $i * h$ vectors in the $i$-th layer with a model having $h$ kv heads). The same is done for the values (but not the queries). The coefficient of weighted average is shared between keys and values. Results show improvement over baseline (as well as DenseFormer and HyperConnections) on downstream tasks. Particularly, there is a signficant boost in accuracy over Arithmetic Expression Task which is attributed to the ability to store more information needed for reasoning. Additionally the authors show that the representation remains linearly separable even in later layers which is not true about the baseline.

### Strengths
While the method shares similarity with existing methods such as DenseFormer, the correct placement of weighted averages is important and in addition to superior performance on the experiments, yields side-benefits such as the ability to re-use the KV cache. The authors report additional investigative results such as the analysis done on the learned router weights.

### Weaknesses
In Section 5.1, it would be very helpful to have the random baseline for each task. In particular, that results that are reported for several of tasks seem near-chance (e.g. WiC). There is also no confidence intervals reported which makes it very hard to determine the significance of the improvements. Overall this makes me question the efficacy of the method in general language modeling.

It is confusing to refer to LLaMA in Table 1. Based on my understanding, this is only a model with the same base architecture as LLaMA models where as a LLaMa baseline suggests the pre-trained models. I strongly suggest to make this clear since based on my understanding you are training everything from scratch.

I have asked additional questions below. Overall, I am uncertain about the intepretation of the provided results and whether they can currently clearly establish the effectiveness of the proposed method.

### Questions
1. When doing value classification (e.g. in Fig. 2b) is the rest of the model frozen?

2. Did you consider using a per-dimension (instead of per-head) weighted average? Was there any difference in performance? Alternatively, is it important to average across heads or is it enough to average over the same head across different layers? 

3. DenseFormer does a similar mixing as the proposed method. Still, the results for DenseFormer are sometimes even worse than the baseline. Also, Denseformer paper reports reasonable improvements over the baseline. Why similar consistent improvements are not observed in these new set of experiments?

### Soundness
3

### Presentation
2

### Contribution
3
