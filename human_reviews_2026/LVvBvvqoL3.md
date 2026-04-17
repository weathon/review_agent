# Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning

- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Estimating the quality of register transfer level (RTL) designs is crucial in the electronic design automation (EDA) workflow, as it enables instant feedback on key performance metrics like area and delay without the need for time-consuming logic synthesis. While recent approaches have leveraged large language models (LLMs) to derive embeddings from RTL code and achieved promising results, they overlook the structural semantics essential for accurate quality estimation. In contrast, the control data flow graph (CDFG) view exposes the design's structural characteristics more explicitly, offering richer cues for representation learning. In this work, we introduce StructRTL, a novel structure-aware graph self-supervised learning framework for improved RTL design quality estimation. By learning structure-informed representations from CDFGs, StructRTL significantly outperforms prior art on various quality estimation tasks. To further boost performance, we incorporate a knowledge distillation strategy that transfers low-level insights from post-mapping netlists into the CDFG-based predictor. Experimental results demonstrate that StructRTL establishes new state-of-the-art results, highlighting the effectiveness of combining structural learning with cross-stage supervision.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces StructRTL, a structure-aware graph self-supervised learning framework. Unlike token-based LLM embeddings, StructRTL operates on Control Data Flow Graphs (CDFGs) to explicitly capture structural semantics. It employs two self-supervised tasks, structure-aware masked node modeling and edge prediction, and further incorporates knowledge distillation from post-mapping netlists to enhance RTL quality estimation. Experimental results demonstrate notable improvements over both LLM-based and graph-based baselines in predicting area and delay.

### Strengths
1. This work presents a successful demonstration of tailoring architectural design and pretraining strategies for the RTL domain by leveraging CDFG inputs, a Transformer backbone, and dedicated pretraining tasks.

2. It conducts comprehensive experiments, including ablation studies and knowledge distillation analyses, showing consistent improvements over prior methods.

3. The open-source release of the complete pipeline and the constructed dataset could be a useful contribution to the research community.

### Weaknesses
1. The work demonstrates limited technical novelty, as it primarily integrates existing techniques such as graph masked autoencoding and knowledge distillation into the RTL quality estimation context without introducing fundamentally new learning principles.

2. It remains unclear whether the proposed framework is inherently specific to the RTL domain or if its structural learning principles can generalize to broader code or graph representation learning tasks. The paper lacks a discussion of transferable insights or contributions to the general ML community.

3. The ablation study is basic and does not adequately explain why each pretraining component contributes to the observed improvements, nor how the learned structural representations differ qualitatively from token-based embeddings.

4. The experiments focus solely on area and delay prediction, without exploring other hardware-related metrics or evaluating the method’s generalization to unseen hardware designs.

5. The contributions are largely application-driven within the EDA field, offering limited theoretical or methodological advances that would engage the broader representation learning or ML research community.

### Questions
My questions have been included in the weakness section.  I'm willing to adjust my scores if my concerns are properly addressed.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work focuses on representation learning of the RTL (register transfer level) representation of digital electronic design with the goal of design quality estimation. This StructRTL approach utilizes the control data flow graph (CDFG) of the RTL code which is obtainable from compiler tools. It uses a graph neural network (GNN) + transformer encoder. The three pretraining tasks that make this approach work by creating strong representations is 1) masked node modeling 2) edge predictions 3) knowledge distillation of fine-grained circuit synthesis insights.

### Strengths
* Very strong experimental results.
* Pretraining tasks are sounds and essential for representation learning, leading to the strong downstream task result of overall quality estimation.

### Weaknesses
* This work implements various ideas well but is ultimately heavily inspired by prior work. Graph representation (GraphMAE, MaskGAE) and fine-grained knowledge distillation (VeriDistill).

### Questions
* What is your insight on data scale. In the software domain, when trained on vast amounts of data, LLMs seem to perform strongly with just token representation of code. If there was large scale data available for RTL code, would the gap between token based representation and StructRTL be slimmer?

### Soundness
4

### Presentation
4

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
StructRTL is a  framework for predicting RTL hardware design quality (area/delay) without requiring slow logic synthesis. The method uses Control Data Flow Graphs (CDFGs) with graph neural networks and incorporates knowledge distillation from post-synthesis netlists. It combines GNN processing with Transformer encoding and employs two self-supervised pretraining tasks. The approach achieves state-of-the-art results on the OpenABC-D dataset, significantly outperforming existing methods including traditional ML models, graph-based approaches, and LLM-based methods.

### Strengths
a. State-of-the-Art Performance: Clear and substantial outperformance across multiple quality metrics (area R²=0.8676, delay R²=0.8872) on a large, modern dataset, providing strong evidence for the structure-aware approach's superiority.

b. Rigorous Experimental Validation: Comprehensive ablation studies systematically validate each architectural component (GNN backbone, pretraining tasks, positional embeddings), proving their individual contributions are critical and non-redundant.

c. Effective Technical Integration: Successfully combines advanced techniques (graph learning, self-supervision, knowledge distillation) into a cohesive pipeline that meaningfully incorporates low-level physical design insights.

d. Practical Impact: Demonstrates massive speedup over traditional synthesis-based methods while maintaining accuracy, showing clear real-world applicability for accelerating hardware design workflows.

### Weaknesses
a. Limited Generalization Assessment: All experiments confined to OpenABC-D dataset; no evaluation on other RTL benchmarks or real-world proprietary designs raises questions about performance on diverse circuits outside the training distribution.

b. Insufficient Knowledge Distillation Analysis: The source of the teacher model's superiority isn't deeply explored; missing ablation comparing netlist-trained vs CDFG-trained teachers to validate claims about "low-level insights."

c. Indirect LLM Comparison: Post-hoc comparison to LLM methods like VeriDistill; would benefit from more direct comparison such as fine-tuning the same LLM on CDFG representations for stronger evidence.

d. Pipeline Complexity Concerns: Multi-stage pipeline (CDFG construction, GNN processing, Transformer encoding) lacks discussion of computational costs and training complexity, potentially creating adoption barriers compared to simpler approaches.

### Questions
Q1.	Generalization: How does StructRTL perform on other RTL benchmarks beyond OpenABC-D? Have you tested on proprietary or more complex designs?

Q2. 	Knowledge Distillation Deep Dive: Can you provide ablations comparing different teacher model configurations (netlist-based vs CDFG-based) to better isolate the value of low-level insights?

Q3.	Direct LLM Comparison: Would fine-tuning existing LLMs on CDFG representations provide a more direct comparison to validate the structural approach's superiority?

Q4.	Computational Analysis: What are the training time, memory requirements, and computational costs of the full pipeline compared to simpler baseline methods?

Q5	Scalability: How does performance scale with larger, more complex designs? Are there practical limits to the CDFG approach?

### Soundness
2

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
This paper proposes StructRTL, a structure-aware graph self-supervised learning framework for estimating quality metrics (area and delay) of Register Transfer Level (RTL) hardware designs. The method operates on control data flow graphs (CDFGs) and uses two pretraining tasks namely structure-aware masked node modeling and edge prediction to train a model for RTL quality prediction. Additionally, knowledge distillation from post-mapping netlists is incorporated to improve performance.

### Strengths
1. Correctly predicting design metrics at an earlier stage in the design cycle is very useful since it can reduce the overall design cycle time.
2. Using CDFGs is well motivated.
3. The paper is generally well-written and easy to follow.

### Weaknesses
1. while larger than some prior work, however the current dataset size is still small.
2. 80% of the designs have less than 600 nodes which raises concerns about whether the method scales to real life designs and whether the current dataset truly represents real-world complexity.
3. Proposed knowledge distillation requires running synthesis to get the netlists. Scalability will become an issue for big designs.

### Questions
1. What kind of designs did you use for creating the dataset? Please specify the design types.
2. How does the technique perform for combinational circuits and sequential circuits? How much difference do you see in the performance?

### Soundness
2

### Presentation
2

### Contribution
2
