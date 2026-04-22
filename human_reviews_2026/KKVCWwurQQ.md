# Why Should the Server Do It All?: A Scalable, Versatile, and Model-Agnostic Framework for Server-Light DNN Inference over Massively Distributed Clients via Training-Free Intermediate Feature Compression

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Modern DNNs often rely on edge–cloud model partitioning (MP), but widely used schemes fix shallow, static split points that underutilize edge compute and concentrate latency and energy on the server. The problem is exacerbated in autoregressive (AR) LLM inference, where per‑token forward passes repeatedly generate bulky intermediate features (IFs). We introduce SLICER, a retraining‑free, architecture‑agnostic framework that compresses IFs to reduce both communication and server load in split computing. SLICER combines (i) asymmetric top‑K filtering (ATKF) to sparsify low‑magnitude activations, (ii) magnitude‑splitting (MS) to group the remaining non‑zeros into equal‑cardinality blocks, and (iii) adaptive bit quantization (ABQ) that selects per‑block bitwidths under a distortion budget. Across standard vision and LLM workloads (e.g., ImageNet/COCO; HellaSwag, PIQA, ARC‑E/C, GSM8K, HumanEval), SLICER reduces uplink volume by up to 10× and server GPU time by up to 4.4×, while keeping task quality within ~0–3 pp of baseline. In multi‑device settings and AR LLMs, SLICER scales by shifting meaningful compute to the edge and lowering bits‑per‑token and server time per token, stabilizing per‑step traffic. The codec attaches to off‑the‑shelf models without retraining or architectural changes, offering a plug‑and‑play path to scalable, low‑latency distributed inference. Code is provided in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SLICER, a training-free, model-agnostic IF codec that attaches to modern DNN 
models and reduces uplink volume and server load. The contributions are: 
(i) Asymmetric Top-K Filtering (ATKF) to enforce exact sparsity, 
(ii) Magnitude-Splitting (MS) into equal-cardinality blocks, and 
(iii) Adaptive Bit Quantization (ABQ) with an integer-only distortion guard. 
Reported gains include up to 10x uplink reduction and 4.4x server time speedup with little accuracy drop across vision and LLM tasks.

### Strengths
1. It identifies a key problem in edge-cloud DNN inference and proposes a novel, training-free IF compression framework.

2. The proposed techniques (ATKF, MS, ABQ) are well-motivated and effectively reduce uplink volume and server load.

3. Scales in multi-device settings and is model-agnostic, making it versatile for various applications.

### Weaknesses
1. There is no analysis on the long token length setting for LLMs with SLICER, which maybe impact by the proposed ATKF method.

2. There is no device-side profiling for computation and memory overhead.

3. Some varibales seems arbitrary, such as the choice of equal block size in MS.

### Questions
Overall, I think this paper address an important problem in the realm of edge-cloud computing, and the idea is interesting. I have a few comments and questions as follow.
Citation Error: There should be a citation for Llama 2 in the section 4.1.

1. How does quantization/sparsity noise in IFs propagate over long token generations in LLMs?

2. What are measured codec compute/energy costs on representative edge SoCs, and how do they scale with IF size/depth? 

3. Does ATKF random tie-breaking induce nondeterminism affecting reproducibility or accuracy? 

4.  The hierarchical search adjusts sparsity/split under budgets, but stability behavior under fluctuating networks isn’t deeply analyzed.

5. Are features compressed per layer, per token, or globally across the model? How does the granularity influence both latency and reconstruction fidelity?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Model partitioning in client–server architectures for DNN inference typically relies on static split points, which can overload the server and underutilize edge devices. This issue is especially critical for large language models (LLMs), where autoregressive generation produces large intermediate features (IFs). The paper introduces SLICER, an architecture-agnostic framework that splits models and compresses IFs to reduce server load and communication costs in split computing. SLICER employs a three-stage pipeline (Asymmetric Top-K Filtering (ATKF), Magnitude Splitting (MS), and Adaptive Bit Quantization (ABQ)) to determine split points that balance latency, memory, and bandwidth constraints. The experiments demonstrate the framework’s effectiveness across different domains.

### Strengths
+ The method is validated on multiple domains, showing consistent performance improvements.
+ The authors have released the implementation, promoting transparency and facilitating future research.

### Weaknesses
- The paper does not compare SLICER against prior split-computing or communication-aware inference frameworks, such as BottleFit [1] and Frankensplit [2]. Including these baselines would better suit SLICER’s contributions.
- The experiments use only a single client device (an NVIDIA Jetson AGX Orin) in the client–server pipeline. This setup is too limited to substantiate claims about scalability and generalization to multiple clients.
- The hyperparameter selection process (e.g., sparsity level s, distortion threshold δ, block count M, asymmetry factor λ) is not well explained, limiting reproducibility and interpretability.
- The discussion focuses solely on single-point splitting. It would be beneficial to explore the implications and challenges of multi-point partitioning.
- Although multi-client scenarios are briefly mentioned, the paper does not discuss whether synchronization mechanisms are required when multiple front-end devices are involved.

Reference:

[1] Matsubara, Yoshitomo, Davide Callegaro, Sameer Singh, Marco Levorato, and Francesco Restuccia. "Bottlefit: Learning compressed representations in deep neural networks for effective and efficient split computing." In 2022 IEEE 23rd International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM), pp. 337-346. IEEE, 2022.

[2] Furutanpey, Alireza, Philipp Raith, and Schahram Dustdar. "Frankensplit: Efficient neural feature compression with shallow variational bottleneck injection for mobile edge computing." IEEE Transactions on Mobile Computing 23, no. 12 (2024): 10770-10786.

### Questions
1. Since SLICER is a partitioning framework, would it be feasible to include direct comparisons with existing split-computing frameworks such as BottleFit or Frankensplit?
2. The experimental setup involves only one client device. Could the authors expand the evaluation to include multiple or heterogeneous edge devices to assess scalability better?
3. How were key hyperparameters (e.g., s, δ, M, λ) selected? Were they tuned empirically, or is there a mechanism to adaptively determine them in practice?
4. The paper discusses partitioning at a single split point. Is it possible to extend SLICER to support multiple split points within a DNN? If so, how might this impact performance and communication costs?
5. The “Hierarchical Search Policy” (lines 324–327) is mentioned briefly. Could the authors provide a more detailed explanation of how this policy operates and influences split-point determination?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes SLICER, a training-free, model-agnostic framework for server-light split computing that compresses intermediate features (IFs) to reduce both uplink traffic and server load. The codec combines Asymmetric Top-K Filtering (ATKF) for exact sparsity control, Magnitude-Splitting (MS) into equal-cardinality blocks, and Adaptive Bit Quantization (ABQ) with a lightweight integer mismatch guard. The method includes a predictive, constraint-aware configuration policy for both single-shot and autoregressive (AR) inference.

### Strengths
+The framework seems that it training-free and plug-and-play.
+ The formulation selects (w,ℓ) and compression knobs θ to meet latency/memory caps and stabilize bits-per-token and server time/token—addressing a real bottleneck for multi-client LLM services. 
+ Results span CNNs/transformers, LLM AR decoding, and diffusion; multi-device scaling shows server time reductions up to 4.4× and sustained backend throughput as clients grow; vision achieves state-of-the-art BPP for IFs with negligible accuracy loss.

### Weaknesses
- The concrete edge device in this paper used is never mentioned in this paper. As such, the latency measurement is unfair since latency depends on the computation ability of the concrete device.

- The evaluation largely abstracts the backend queue and uses a parametric wireless model; end-to-end wall-clock (with control-plane signaling for grid/codec metadata, CSR indices, and potential RANS entropy coding) is not dissected across diverse networks and straggler patterns. More real-network and asynchronous/many-client studies would strengthen the scaling claims. 

- Comparisons against LLM baselines could be deeper.

### Questions
- Can you proofread the paper before the submission pls? lots of typos here,i.e., line 336 '?' and 'multi device' in the conclusion 

- What is the control-plane overhead (bytes and latency) for metadata?

### Soundness
2

### Presentation
2

### Contribution
3
