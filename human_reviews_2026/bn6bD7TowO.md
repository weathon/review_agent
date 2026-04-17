# Topology Matters in RTL Circuit Representation Learning

- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6

## Abstract
Representation learning for register transfer level (RTL) circuits is fundamental to enabling accurate performance, power, and area (PPA) prediction, efficient circuit generation, and retrieval in automated chip design. Unlike general programming languages, RTL is inherently a structured dataflow graph where semantics are intrinsically bound to the topology from a hardware view. However, existing language-model-based approaches ignore the nature of RTL circuits and fail to capture topology-sensitive properties, leading to incomplete representation and limited performance for diverse downstream tasks. To address this, we introduce TopoRTL, a novel framework that explicitly learns topological differences across RTL circuits and preserves the behavior information. First, we decompose RTL designs into register cones and construct dual modalities initialized with behavior-aware tokenizers. Second, we design three topology-aware positional encodings and leverage attention mechanisms to enable the model to distinguish topological variations among register cones and RTL designs. Finally, we introduce a topology-guided cross-modal alignment strategy, employing contrastive learning over interleaved modality pairs under topological constraints to enforce semantic consistency and achieve superior modality alignment. 
Experiments demonstrate that explicit topological modeling is critical to improving RTL representation quality, and TopoRTL significantly outperforms existing methods across multiple downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces TopoRTL, a novel framework for representation learning of RTL circuits that explicitly models both behavioral semantics and topological structures. Existing LLM-based or text-only RTL methods (e.g., CodeV, DeepRTL) treat Verilog code as pure text, overlooking the inherent dataflow topology of circuits. TopoRTL addresses this limitation through three key components: behavior-aware dual-modal tokenizers, which capturing both graph-based (structural) and textual (behavioral) modalities; topology-aware positional encodings, which integrating bit-width, path length, and graph density into Transformer attention to represent structural hierarchy and timing characteristics; topology-guided cross-modal alignment, which ensuring semantic consistency between the two modalities under topological constraints. Experiments on PPA prediction and natural language circuit retrieval show substantial gains over prior baselines such as CodeV, Qwen3, and CircuitFusion, with fewer parameters. The results confirm that explicit topology modeling is crucial for RTL representation learning.

### Strengths
1.	The paper introduces a novel perspective on RTL representation learning by emphasizing the importance of topological structure in addition to behavioral semantics. The idea of integrating bit-width, signal-path, and graph-density encodings into Transformer attention is new in the RTL/EDA domain. Furthermore, the cross-modal alignment strategy between topology and textual modalities represents a creative extension of multimodal learning techniques into hardware design contexts.
2.	The methodological pipeline—from register cone extraction to dual-modal embedding and topology-guided alignment—is technically sound and carefully justified. Experiments are extensive and clearly show that TopoRTL achieves consistent improvements across PPA prediction and circuit retrieval benchmarks, outperforming both large-scale LLMs (e.g., CodeV, Qwen3) and graph-based methods.
3.	The paper is well organized, with a clear motivation and coherent narrative flow. Figures (especially Fig. 1–3) effectively illustrate the problem and model design. The appendices provide detailed implementation and dataset descriptions, enhancing reproducibility.
4.	This work bridges two rapidly growing research areas—representation learning and EDA automation—demonstrating that explicitly modeling topology leads to superior circuit representations. The framework could meaningfully impact early-stage chip design optimization and inspire future work on structure-aware learning across code and graph modalities.

### Weaknesses
1.	The t-SNE visualization (Figure 4) shows separation but doesn’t analyze how topology encoding affects cluster formation. A more explicit comparison of embedding similarities (e.g., through cosine distances between functionally equivalent circuits) could strengthen the argument.
2.	The summary modality relies on GPT-OSS-120B to generate textual circuit descriptions. The paper doesn’t analyze how summary quality affects downstream performance or whether the model could generalize with human-written or noisy summaries.

### Questions
1.	During training, how sensitive is the model to the quality of text summaries? If summaries are partially inaccurate, does the topology encoder still preserve robustness?
2.	Could the proposed topology-aware encoding generalize beyond RTL (e.g., to gate-level netlists or analog blocks)?
3.	What is the quantitative contribution of each topology encoding (bit-width, path, density) when combined pairwise, rather than removed individually?
4.	Could TopoRTL be integrated into synthesis or verification tools for real-time PPA estimation?

### Soundness
3

### Presentation
3

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
This paper proposes TopoRTL, an RTL-stage circuit representation learning framework that incorporates topological information while preserving functional behavior. The RTL is partitioned into register cones and represented in both graph and summary modalities. These two representations are then encoded and fused through a multimodal learning architecture. Experimental results on downstream prediction tasks show that the learned embeddings improve the accuracy of RTL design quality estimation.

### Strengths
1. The paper addresses an important problem in AI for EDA: circuit representation learning, and effectively leverages the multimodal nature of RTL format. The writing is clear and the methodology is easy to follow.
2. The proposed approach incorporates topology-aware positional encodings within a transformer-based architecture to model RTL graphs. The use of bit-width encoding and max-path/density encodings is well-motivated and aligns naturally with RTL structural characteristics.
3. The experimental results demonstrate that fusing the graph and summary modalities leads to improved performance, validating the effectiveness of the multimodal representation learning strategy.

### Weaknesses
1. The main concern is that the proposed multimodal RTL representation learning framework appears highly similar to CircuitFusion [ICLR’25]. Although the authors provide an empirical comparison, the paper does not sufficiently clarify the methodological differences or improvements. In the related work section, the discussion focuses primarily on behavior-based and topology-based models, but does not address prior multimodal fusion approaches. Moreover, key components such as register-cone decomposition, graph and summary modalities, contrastive/masked pre-training objectives, and multimodal fusion design closely resemble those of CircuitFusion. A deeper analysis of novelty is needed.
2. For the topology-aware graph transformer, the paper lacks comparison with recent graph transformer architectures widely explored in the AI/graph community (e.g., Graphormer, SGFormer, GPS, SAN). It is unclear whether the proposed topology encodings provide advantages over standard positional or structural encodings used in these models.
3. The downstream evaluation focuses on high-level RTL quality prediction, but does not explore whether the proposed topology encoding can capture structural variations that affect post-synthesis PPA. Since synthesis outcomes are highly topology-dependent, examining whether TopoRTL embeddings can reflect such differences would provide stronger evidence of the model’s effectiveness. Some discussion or experiments in this direction would be beneficial.
4. Additionally, the evaluation is limited to PPA prediction tasks. There also exist functional verification tasks where representation quality matters, such as verification coverage prediction (e.g., Design2Vec [NeurIPS’21]). Demonstrating that the embeddings are useful across both functional verification and PPA prediction tasks would make the contribution more comprehensive and significantly strengthen the impact.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes a topology-aware RTL circuit representation method. They analyze the topology and functional behavior to design the structure ofthe  network, which provides meaningful insights. Finally, their algorithm shows better performance than CircuitFusion and other LLM models.

### Strengths
1. Good performance compared with competitive works, e.g., CircuiFusion and other LLM models.
2. Meaningful insights about explicitly encoding the topological information in RTL.

### Weaknesses
1. The scale of designs is relatively small, where only 7,576 sub-circuits are extracted from 115 RTL designs. Meanwhile, I cannot find a difference in model performance between large designs (>200 registers) and small designs (<30).
2.  The prediction ability of the trained model seems to depend on PDK, which will affect the generalization ability.

### Questions
1. In Table 1, I find the performance on timing-related metrics (e.g., Slack, TNS, and WNS) is better than that on Area and Power. Can authors explain the reason for these results? I thought area and power would be easier to predict because they do not depend on topology.
2. Can authors explain the generalization ability of their works if the PDK is changed?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors designed a method that integrates two modalities of circuit with considerations for circuit topology. The authors used behavior-aware dual-modal tokenizers with three positional encodings (bit-width, max-path, graph-density) and adopted grahp transformer to encode circuits to representations.

### Strengths
* The paper is well written, easy to follow.
* The authors tackled an important problem that the field needed to address.
* The authors performed extensive experiments with good results.

### Weaknesses
* In table1, the proposed model consistently outperforms baselines in Area, Power and WNS while not in Slack and TNS. The authors explained why their model shows good performance in Area, Power and WNS, however, they do not explained the possible reason behind relatively lowever performance in Slack and TNS.
* It appears that the experiments were conducted only once and the performance was reported based on that single run. To ensure that the model’s performance is not dependent on a specific random seed but is statistically meaningful, it is necessary to repeat the experiments multiple times and report the mean and standard deviation of the performance.

### Questions
See the 'Weakness' part.

### Soundness
3

### Presentation
3

### Contribution
3
