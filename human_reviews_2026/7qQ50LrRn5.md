# Agentic-KGR: Co-evolutionary Knowledge Graph Construction through Multi-Agent Reinforcement Learning

- Decision: Reject
- Scores: 6, 2, 2, 6

## Abstract
Current knowledge-enhanced large language models (LLMs) rely on static, pre-constructed knowledge bases that suffer from coverage gaps and temporal obsolescence, limiting their effectiveness in dynamic information environments. We present Agentic-KGR, a novel framework enabling co-evolution between LLMs and knowledge graphs (KGs) through multi-round reinforcement learning (RL). Our approach introduces three key innovations: (1) a retrieval-augmented memory system enabling synergistic co-evolution between model parameters and knowledge structures through continuous optimization; (2) a dynamic schema expansion mechanism that systematically extends graph ontologies beyond pre-defined boundaries during training; (3) a learnable multi-scale prompt compression approach that preserves critical information while reducing computational complexity through adaptive sequence optimization. Experimental results demonstrate substantial improvements over supervised baselines and single-round RL approaches in knowledge extraction tasks. When integrated with GraphRAG, our method achieves superior performance in downstream QA tasks, with significant gains in both accuracy and knowledge coverage compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the Agentic-KGR framework, a novel method that facilitates the co-evolution of large language models and knowledge graphs through multi-agent reinforcement learning. It aims to overcome the limitations of static knowledge bases by enabling dynamic knowledge construction and adaptation. Key innovations include a dynamic schema expansion mechanism, a retrieval-augmented memory system, and a learnable multi-scale prompt compression approach.

### Strengths
1. The concept of co-evolution between LLMs and KGs is well-founded and provides a fresh perspective on knowledge graph construction. The integration of multi-round RL is particularly valuable for addressing the dynamic nature of real-world knowledge and improving LLM performance in knowledge-intensive tasks.
2. The paper demonstrates solid experimental results, with clear improvements in knowledge extraction and QA performance.

### Weaknesses
1. While the paper covers a significant amount of technical depth, it occasionally becomes difficult to follow, especially in the description of the architecture and algorithmic details. The explanations of the co-evolutionary memory and multi-scale compression mechanisms could benefit from more intuitive explanations to help the reader grasp the concepts more quickly.
2. The paper focuses heavily on the proposed framework's strengths but could benefit from a more thorough discussion of its limitations or challenges. For example, how scalable is this approach for large-scale knowledge graph construction in real-world applications? Additionally, potential pitfalls or failure modes in dynamic KG construction could have been addressed more comprehensively.

### Questions
See weaknesses.

### Soundness
3

### Presentation
1

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
The paper addresses the problem of LLMs hallucination.  The paper proposes a framework that enables co-evolution between LLMs and Kgs through multi-round RL.

### Strengths
As LLMs are more and more integrated in our daily life, the problem of LLMs hallucination is a problem that needs a particular attention. The experiment of the approach is well described, along with the experimentation environment.

### Weaknesses
The paper in its current form has several issues. 
The paper claims that the approach is and innovation, which is not the case
Missing summary of the experiments (e.g., datasets, models used) and results in the abstract and introduction.
Missing of the related work in the paper. Provided in the additional materials. However, the paper should be self contained.
Line 52-53: “Existing methods rely on static, pre-constructed KGs that suffer from coverage gaps, temporal obsolescence, and inability to adapt to emerging domain knowledge or evolving query patterns” That is not true. No. KGs are supposed to evolve and agentic methods are there to take care of the evolution of Kgs
Line 54-59: These claims should be supported by examples/evidence provided by references
Table 1:  Put the table near the place it is cited. The tables are in page 5, 6 and used in page 8.

### Questions
Integrate the related work in the paper to make the paper self contained.
Summarize the experiments and results in the abstract and introduction
Improve the quality of the manuscript by putting the figure near the place they are cited for the first time

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Agentic-KGR, a novel framework designed to address the limitations of static, pre-constructed knowledge bases (KGs) often used by knowledge-enhanced large language models (LLMs). These static knowledge bases typically suffer from coverage gaps and temporal obsolescence. The core contribution of Agentic-KGR is enabling the co-evolution between LLMs (acting as reasoning agents) and dynamic KGs through multi-round reinforcement learning (RL). This paradigm shifts the approach from static knowledge retrieval to adaptive knowledge co-creation, viewing knowledge construction and utilization as interconnected, mutually reinforcing processes.

### Strengths
The paper's primary strength in originality lies in defining a novel paradigm shift for knowledge-enhanced language models. Instead of relying on static, pre-constructed knowledge bases (KGs) which suffer from coverage gaps and obsolescence, Agentic-KGR introduces a framework that enables the co-evolution between LLMs and KGs through multi-round reinforcement learning (RL). This fundamentally transforms the approach from static knowledge retrieval to adaptive knowledge co-creation.

### Weaknesses
1. One of the key innovations is the Learnable Multi-Scale Prompt Compression (LMPC), intended to preserve critical information while reducing computational overhead. While the results demonstrate that Agentic-KGR successfully reduces the overall average response length across all architectures (Figure 4, Figure 7), this does not directly prove cost or latency reduction.

2. The most significant performance gains in both extraction (RE performance) and downstream QA (Table 2) rely heavily on domain-specific datasets derived from communication technology/telecom documentation (CommTKG, MmlKG, ConfigKG, WirelessKG, DcommKG). Actionable Insight: The paper must demonstrate the robustness of the dynamic schema expansion and co-evolution mechanisms on highly volatile, cross-domain, or general-purpose knowledge datasets to ensure the results are not heavily contingent on the structure or language features unique to structured technical manuals.

### Questions
1. In the multi-round retrieval framework proposed in this paper, there is only a single LLM agent, whereas a typical multi-agent system usually contains at least two LLM agents. Does this make the paper’s naming somewhat misleading?

2. Could the authors provide a direct comparison of Agentic-KGR's KG extraction (RE/NER) and downstream QA performance against state-of-the-art agentic reinforcement learning baselines mentioned in the related work, specifically Graph-R1 (for end-to-end graph RAG optimization) and ReSearch (for LLMs learning to reason with search via RL)?

3. As an end-to-end optimization framework, Agentic-KGR uses reinforcement learning to directly optimize the agent policy, compression scales, and knowledge-graph update operators. Could this make the method overly sensitive to hyper-parameters and cause unstable training?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Agentic-KGR, a framework for co-evolutionary knowledge graph construction using multi-agent reinforcement learning. It overcomes the limitations of static KGs by enabling dynamic schema expansion, a retrieval-augmented memory system for synergistic optimization, and a learnable multi-scale prompt compression to handle computational complexity.

### Strengths
1. The core idea of mutual adaptation between LLMs and KGs via multi-round RL is timely and innovative.
2. The methodology is well-formalized with clear definitions。
3. Claims of substantial gains in KG density/coverage and QA accuracy are compelling, especially the integration with GraphRAG for end-to-end validation.

### Weaknesses
1. The experiments focus on KG extraction and QA, but details are sparse in the provided sections (e.g., datasets not specified early; baselines like DeepPath or MINERVA mentioned but not compared quantitatively here).
2. No discussion of scalability (e.g., for large KGs) or real-world costs (tokens/time in multi-round RL).
3. Diversity in tasks/domains beyond product QA is missing, for instance, the medical RAG in GraphRAG-benchmark[1].

- [1] When to use graphs in rag: A comprehensive analysis for graph retrieval-augmented generation.

### Questions
Please see above.

### Soundness
3

### Presentation
4

### Contribution
3
