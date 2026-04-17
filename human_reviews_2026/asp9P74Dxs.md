# Graph Foundation Models: Bridging Language Model Paradigms and Graph Optimization

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
The pretrain-transfer paradigm, which underpins the success of large language models (LLMs), has demonstrated the immense power of creating foundation models that learn generalizable representations from vast datasets. However, extending this paradigm to Operations Research (OR) problems on graph structures remains challenging due to the fundamental conflict between the statistical flexibility of language and the strict combinatorial constraints of graphs. To bridge this gap, we introduce the Graph Foundation Model (GFM), the first framework capable of solving all distance-based optimization problems on graph structures. By introducing the LLM-like self-supervised pre-training paradigm on the paths generated from random walks in the graph, GFM is compelled to internalize the graph’s complex topological and combinatorial rules, where the connectivity of the structure itself can be treated as the supervisory signal. Unlike existing neural methods that learn complex and task-specific solving policies, our approach leverages the pre-trained GFM as a foundational model of the graph’s intrinsic structure, which in turn enables a simple generative heuristic to tackle a diverse range of optimization challenges effectively. Comprehensive experiments on networks ranging from 20 to 893 nodes demonstrate that GFM achieves competitive performance against specialized solvers across a variety of distinct optimization task classes, while maintaining significantly faster inference times. Our work establishes a new paradigm of adapting the pretrain-transfer framework to graph optimization, opening the door for applying foundation model innovations to operations research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the Graph Foundation Model (GFM), a framework that transfers the pretrain–transfer paradigm of large language models (LLMs) to graph-structured data for operations research (OR) and graph optimization problems.

### Strengths
1. This work clearly defines the model architecture, training curriculum, and decoding strategies for constrained graph tasks.

2. It introduces a foundation model perspective for graph optimization, moving beyond task-specific architectures.

3. It evaluates across multiple scales and graph problems (SP, Graphic-TSP, TP-SOD, TP-DOD)

### Weaknesses
1. Reported scalability only reaches ~900 nodes, and further validation or discussion on larger networks is needed.

2. It’s unclear if performance gains arise primarily from architectural differences or training scale.

3. The quality of the diagram (Fig. 1) is poor, in which the font size is too small to read. And the logic is also not good to deliver the key ideas.

4. GFM is a big concept. As this work is limited only to distance-related problems, GFM is overstated.

5. No ablation and parameter sensitivity studies to isolate/analyze the contribution of each training component.

### Questions
1. Can the proposed method be extended to other combinatorial problems such as influence maximization and community detection?

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
3

### Summary
The paper introduces the Graph Foundation Model (GFM), a new framework that adapts the successful "pretrain-transfer" paradigm of Large Language Models (LLMs) to solve graph-based optimization problems in Operations Research (OR). The paper identify a key challenge: the statistical flexibility of language models conflicts with the strict combinatorial constraints of graph problems. Existing methods often fail to generalize from synthetic graphs to real-world, sparse road networks.
GFM bridges this gap by using an LLM-like self-supervised pre-training strategy. The model is pre-trained on paths generated from random walks on the graph, compelling it to learn the graph's complex topological and combinatorial rules without task-specific supervision. This process creates a "transferable structural prior". At inference time, this single pre-trained model can be used with a simple generative heuristic to effectively tackle a diverse range of distance-based optimization problems, from shortest paths to NP-hard routing tasks.
Experiments conducted on networks ranging from 20 to 893 nodes demonstrate that the GFM achieves competitive performance against specialized solvers while offering significantly faster inference times.

### Strengths
1.The paper demonstrates significant originality by being the first to introduce the concept of a "Graph Foundation Model" (GFM) specifically for solving realistic, distance-based optimization problems on graphs. Its core innovation lies in the creative adaptation of the LLM-like self-supervised pre-training paradigm to graph structures especially for distance-based problems. The methodology is highly inventive: it first uses "structure-aware random walks" to convert graphs into a sequential corpus, and then pre-trains a bidirectional Transformer via a "hidden segment reconstruction"  task—which serves as a clever analogue to masked language modeling. 
2.The paper is articulated with outstanding clarity. The introduction clearly lays out the motivation, identifying the "fundamental conflict between the statistical flexibility of language and the strict combinatorial constraints of graphs". The methodology is supported by excellent visuals; Figure 1 provides a clear overview of the entire framework, including data generation, training, and inference .

### Weaknesses
1. Critical Underspecification of the Inference/Decoding Mechanism: The paper's core claim is that a single pre-trained model $\pi(\cdot|G)$ can be used to solve four different OR tasks. However, the mechanism for this transfer is described with near-total ambiguity. Section 3.5, "Decoding for Graph Optimization," consists of a single abstract equation (9) that hides all complexity within an undefined Decode() function. The text mentions a "simple, task-agnostic decoding, combined with lightweight projection" (Intro) and an "ad-hoc approach" (Sec 3) without providing any concrete details.
2. The term 'Graph Foundation Model' is a misnomer: The idea of "graph foundation models" or pre-training on graph structure for downstream tasks is not new (e.g., Hu et al., 2020, "GPT-GNN"; Liu et al., 2023a, "One for all," which the paper cites). However, the paper proposed the foundation model specific for OR routing problems. It’s kind of improper to state “introduce the Graph Foundation Model (GFM)”.
Even if the paper can states the proposed method as “the Graph Foundation Model for OR routing problems”, It’s also a little bit improper. The paper's central framing as a "Graph Foundation Model" is built on a fundamental conceptual misunderstanding that confuses multi-task capability with generalizability. This flawed premise extends to misattributing the paper's contribution.
From another perspective, the pre-training merely acts as a learned prior to narrow the search space of feasible paths resulting in improving its multi-task capability. The actual "solving" (i.e., finding an optimal, constraint-abiding solution) is done by the unexplained Decode() function (see Point 1), which appears to be doing all the hard, constraint-based work.
3. Unfair Experimental Baseline Comparisons: The paper's experimental comparisons are kind of problematic. The learning-based baselines (AM, ICAM) are unfairly handicapped. 
The paper states that baselines like AM are trained on synthetic complete graphs. However, the proposed GFM is pre-trained on the test graph itself (e.g., the Berkeley graph). This means GFM has exhaustively "seen" the test graph's topology, while the other learning-based methods are attempting zero-shot generalization to a new, sparse, out-of-distribution graph. The comparison is invalid as it rewards "home-field advantage" rather than general problem-solving ability.
4. Limited Claims of Scalability: The paper's claims of scalability, a key asserted advantage, are based on experiments with graphs that are, by any Operations Research standard, small-to-medium-sized. The largest graph tested, Berkeley (N=893) , is insufficient to substantiate the ambitious claims of solving "large-scale" problems or providing a scalable alternative to existing methods.

### Questions
1. For a given task-specific constraint (e.g., visiting all nodes in set $R$ for Graphic-TSP), there may be many feasible solutions among generation results. How does your Decode() function select the optimal (i.e., shortest-path) solution from this large set of possibilities?
2. Regarding the True Value of Pre-training (Ablation for "Search Space"): Your method’s main effect might just be "narrowing the search space”.To verify that your proposed GFM pre-training method provides more value than just any method of narrowing the search space, could you provide a critical ablation study? Specifically, what is the performance if you replace the GFM-based heuristic with a simpler, standard prior (e.g., using the result of the biased random walk directly to rearch the feasible solutions)? This would help isolate the true contribution of your pre-training curriculum.
3. Regarding the Ambiguous Notation in Table 1: The notation in your results (Table 1) is confusing and appears to contradict its own legend.In some cases (e.g., for the OR-LLM-Agent on the TP-SOD task in Chengdu), the method reports a non-zero Succ (%) (3%), but its Obj (km) is marked as '–'. This scenario seems not to be explained by your legend. Please clarify: what does '–' signify for the Objective value when the Success Rate is not zero?

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
The paper presents a comprehensive framework for Graph Foundation Models (GFMs), a new paradigm that extends the success of large language and vision models into the graph domain. The pretrained backbone can then be fine-tuned for downstream tasks and shows better performance than specialized models.

### Strengths
1.	The paper connects foundation model pretraining principles with graph learning, positioning GFMs as a unifying abstraction for graph tasks.
2.	The research motivation is stated as the generalizbility is an important concern in the graph learning area.

### Weaknesses
1.	The benchmark diversity is insufficient. Despite using OGB datasets, the evaluation could include dynamic or temporal graphs to demonstrate temporal generalization.

2.	There are already many graph foundation models but the comparison with them is missing. Hence it is not possible to evlauate this paper properly. See those recently published in 2024/2025.

3.	As the core of GFM, the generalibility is not well demonstrated. For example, how to prove the proposed GFM can be applied to network of different topologies? It lacks the introduction of structural properties of the tested network.

4.	The real-wrold network is typically of large size, how to ensure the scalability of this GFM. The network size in the experiment cannot be counted as large. 

5. It is a nice "preliminary" work and does not seem ready for a premier venue like ICLR.

### Questions
See weaknesses and address them carefully.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the Graph Foundation Model (GFM), aiming to extend the pretrain–transfer paradigm of LLMs to graph-based optimization problems (e.g., shortest path, TSP, orienteering). It uses distance-biased random walks to generate a self-supervised training corpus, learns graph structural priors through masked trajectory reconstruction, and applies task-specific decoding for solving optimization tasks without retraining. It reports competitive results across real-world road networks compared to OR-Tools, LKH3, and neural combinatorial solvers.

### Strengths
- Bridge symbolic combinatorial optimization and LLM.
- The paper is of clear organization.

### Weaknesses
- Overstated “Foundation Model” claim.
- Underspecified decoding and task generalization mechanism.

### Questions
1. The method proposed is called a "foundation model" but the pretraining corpus is derived from random walks within a single graph, not from diverse graph domains. The current training paradigm resembles graph representation learning (Node2Vec-style) rather than LLM-scale pretraining. Without cross-domain transfer (e.g., from road networks to logistics or molecule graphs), the “foundation model” claim is overstated.
2. Unlike LLMs, where pretraining induces syntactic-semantic priors, GFM’s random-walk reconstruction primarily encodes local connectivity, not higher-order combinatorial reasoning. How learned priors approximate graph-theoretic properties?
3. Does decoding use beam search, greedy selection, or constraint projection?
4. How curriculum weighting is set? why logarithmic levels were chosen?
5. How the model’s attention heads capture topological dependencies? 
6. For large-scale tasks (N=893), GFM’s runtime (129s) is 2 orders of magnitude faster than LKH3, yet solution quality (Obj) differs by ~20%. Does it indicate incomplete convergence?

### Soundness
2

### Presentation
3

### Contribution
2
