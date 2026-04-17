# KnowOS: Knowledge-driven Large Language Models for Operating System Kernel Tuning

- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Operating System (OS) kernel tuning involves systematically optimizing kernel configurations to enhance system performance. Despite recent advancements in large language models (LLMs), kernel tuning remains a significant challenge due to: (1) the semantic gap between abstract tuning objectives and the specific config options, (2) the limited environmental interaction leading to LLM hallucinations, and (3) the rapid evolution of kernel versions. To address these challenges, we introduce KnowOS, a framework powered by knowledge-driven LLMs for automating kernel tuning. KnowOS leverages three key innovations: structured knowledge construction and mapping, knowledge-driven configuration generation, and continuous knowledge maintenance. Extensive experiments demonstrate that KnowOS achieves performance improvements ranging from 7.1\% to 155.4\% over default configurations across standard OS benchmarks and real-world applications. These results highlight the potential of structured knowledge representations in overcoming the limitations of pure LLM-based approaches for system optimization. Our code is available at https://anonymous.4open.science/r/KnowOS-B274.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents KnowOS, a knowledge-driven LLM framework for automated OS kernel tuning. The authors claim to address challenges in objective-configuration mapping, LLM hallucination, and knowledge decay by employing a dual-layer knowledge graph for structured reasoning. The framework reportedly includes a continuous knowledge maintenance mechanism and achieves performance improvements on UnixBench and four real-world applications.

### Strengths
- The introduction of a knowledge graph to bridge the semantic gap between high-level optimization objectives and low-level kernel configurations is a reasonable approach.
- The evaluation demonstrates performance gains across four Linux kernel versions, suggesting some level of adaptability.

### Weaknesses
- Presentation Quality: The paper's presentation is poor. Critical results, such as Table 4, are relegated to the appendix, hindering a complete assessment. The text also contains significant inconsistencies; for example, Section 4.1 claims Debian experiments ran on non-embedded systems, while Table 4 lists its main scenario as "Embedded System." The paper lacks crucial details on the "Concept layer" and "Cross-layer Links" construction, and fails to specify how optimization objectives were formulated for the experiments.
- Originality: The novelty of the contribution is questionable. Based on the anonymized code, the knowledge graph component appears to be a direct application of the LightRAG API. If so, the intellectual contribution is minimal. Furthermore, the paper fails to cite the LightRAG paper, which is a major omission.
- Insufficient Baseline Comparisons: The comparison to AutoOS appears unfair. AutoOS reportedly used 24 optimization runs in its evaluation, whereas this paper uses only 15, potentially underestimating the baseline's true performance. A critical and obvious baseline is also missing: using AutoOS augmented with Kconfig's built-in help text (directory/option explanations) for pruning and configuration, which would be a much simpler way to inject domain knowledge.
- Unaddressed Overheads: The paper completely ignores the practical costs of the proposed system. There is no discussion of the time overhead or token consumption required for optimization. Moreover, the processes of updating the knowledge base and performing retrieval from it may introduce significant latency, which is not measured or discussed.
- Robustness and Reproducibility: The framework's quality is critically dependent on a "manually constructed knowledge base" (specifically, the concept layer). This reliance on manual curation creates a significant scalability bottleneck and limits the method's reproducibility. The robustness of the system is highly questionable as new kernel options are introduced, as it will be constrained by the quality and timeliness of these manual knowledge base updates.
- Generalizability: The framework's generalizability is limited. The results on an embedded RISC-V system (Table 5) show performance that is "nearly on par with the baseline," meaning it fails to demonstrate any significant advantage for this architecture, despite the existence of online domain knowledge for RISC-V.
- Overstated Contributions: The "three propositions" feel overstated and poorly supported. Key technical details, such as the "α and β controlling sensitivity" in Proposition 2, are not explained in the main text. The proofs are insufficient; for instance, the proof for Proposition 2 ignores the critical risk of LLM-induced hallucinations introducing erroneous edges or nodes during KG construction, which would compromise the graph's quality.

### Questions
- How do you systematically evaluate or benchmark the quality of the constructed knowledge base, independent of the downstream tuning task?
- KnowOS searches within a pre-selected candidate subset. How does the framework address or mitigate the risk of inaccuracies or critical omissions in this initial candidate selection phase?
- See more questions on Weakness.

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
They introduce **KnowOS**, a framework that leverages language models for automated kernel tuning. The approach tackles three major challenges commonly faced when applying LLMs to this task: (i) the semantic gap between high-level tuning objectives and low-level configuration options, (ii) the hallucination problem caused by limited interaction with the real system environment, and (iii) the fast evolution of kernel versions. **KnowOS** achieves superior performance on standard OS benchmarks such as *UnixBench* across four Linux distributions. Additionally, it significantly reduces hallucinations and demonstrates strong adaptability across different Ubuntu kernel versions.

### Strengths
- The proposed method is effective, outperforming both the default kernel configuration and other LLM-based approaches, including AutoOS.
- The authors conduct a comprehensive ablation study, showing that each component of their framework contributes meaningfully to its overall performance.
- The paper is clearly written and easy to follow.

### Weaknesses
- The authors attempt to formalize the kernel tuning task using a graph-based framework, leading to the development of **KnowOS**. However, the resulting approach appears overly complex, involving numerous hyperparameters and choices whose effects are neither analyzed nor justified. The trade-off between computational cost and the magnitude of the performance gains remains unclear. Furthermore, several aspects that could strengthen the contribution seem to be left unexplored (See questions).

### Questions
- Line 205: For $e_s \in \mathcal{E}_{\mathcal{C}}^q$, you define the path $\pi (e_s) = <e_s \xrightarrow{r_1} e_1 \ldots \xrightarrow{r_n} e_n >$. How is the endpoint ($e_n$) determined? Is this path unique for each $e_s$? Or do you consider *all* possible paths between $e_s$ and other nodes $e_n$? In that case, is the path between $e_s$ and $e_n$ guaranteed to be unique? 
-  You further define the relevance score $\rho(\pi(e_s))$ and retain nodes belonging to paths whose relevance is greater than or equal to a threshold $\tau$. How is $\tau$ selected and what impact does it have on the results across different objectives?
- It would valuable to include a comparative analysis of time complexity and resource usage (number of inference calls or tokens generated) between **KnowOS** and the other LLM-based strategies.
- I am not fully convinced that the proofs in Appendix B.1 and B.2 are truly theoretical, they seem to rely on heavily on intuition. In Appendix A.1,  it remains unclear how $P_{valid}(K), P_{\mathcal{D}}(o_i, o_j), \alpha, \beta \ldots$ are derived. As it stands the "proof" does not make the proposition much clearer from a theoretical standpoint.
- What is the exact engine name used for **GPT-4o** (not just the alias)? 
- Did you experiment with open-source models or models of varying sizes (e.g., 8B–70B)? Do you expect models extensively trained on code to perform better on this task?
- Table 1: Given the reported `UnixBench` scores on the **Ubuntu** and **Fedora**, it would be evaluate whehter stronger model (in particular *thinking models* such as DeepSeek-R1, o1 etc. or LLMs with tool use) could achieve higher scores in the  **vanilla LLM**  setup. Do you have an intuition as to why **Debian** and **OpenEuler** appear easier to optimize?
- Line 414: `vanilla LLM` or `AutoOS`? There seems to be a naming issue in Table 3.
- Is AutoOS the only advanced LLM-based kernel tuning method you were able to compare against?
- Line 423: For the “Adaptability Across Kernel Versions” section, which addresses the issue of *rapid kernel iteration and knowledge decay*, I expected an evaluation on a kernel distribution released after the LLM used for KnowOS was trained. Would such an evaluation be feasible?
- The scale of Figure 5, especially in panel (a), is somewhat misleading and could be adjusted for clarity.
- There several places where `\citep` should be used to improve citation formatting and readability.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper proposes KnowOS, a knowledge-driven framework for Linux kernel tuning that couples a dual-layer knowledge graph (concepts ↔ concrete Kconfig options) with LLM-guided reasoning to select valid configurations aligned with a target objective. They propose some benchmarks to  test this on. I am not an expert in the area of operating systems. I will try to evaluate this work from a core machine learning perspective.

### Strengths
- It is an extremely novel application of LLMs into a very important area of computer science. 
- The proposed 3 way method (instance, concept and cross-links) is nice. 
- There are nice empirical evaluations across real setups

### Weaknesses
This paper is out of my expertise area.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces KnowOS, a  framework that leverages knowledge-driven LLMs to complete the complex task of OS kernel tuning. To overcome critical challenges—including the semantic gap between high-level objectives and low-level configurations, LLM hallucinations, and rapid kernel evolution—KnowOS constructs a dual-layer knowledge graph (OD-KG). This graph enables structured reasoning for configuration generation and supports continuous updates. Evaluation demonstrates the framework's significant practical value, achieving general performance improvements on standard benchmarks and  gains on four real-world applications.

### Strengths
1. Cross-Version Kernel Performance Enhancement: Robust performance improvements across different Linux kernel versions.

2. Structured Knowledge for Fine-Grained Tuning: OD-KG enables fine-grained, semantically-aware reasoning.

### Weaknesses
1. Critical details on the experimental platform (hardware, software versions) and data processing methodology for the real-world four applications are omitted, hindering reproducibility.

2. Performance results are presented as single-point measurements without variance or confidence intervals, making it impossible to assess the statistical significance of the claimed improvements. 
Additionally, the number of optimization iterations for each objective is not specified.

3. While a reduction in boot-up errors is shown, there is no quantitative analysis of the LLM's hallucination rate , leaving the core claim of hallucination mitigation inadequately supported

4. The evaluation uses only one LLM backbone, failing to demonstrate that the benefits are inherent to the KnowOS framework and not specific to a particular model.

5. The methodology for measuring tuning cost is not clearly specified. Furthermore, the reported token consumption for knowledge graph maintenance (e.g., ~80,000 tokens per update) appears to be a coarse average, failing to reflect the substantial variability expected from different types of  updates.

### Questions
1. Some additional issues were not discussed.How to handle potential copyright issues that may arise with knowledge graphs?

2. The paper's approach to reducing hallucinations by aligning model reasoning with knowledge graph patterns may limit explorability. This could be problematic in cases where identical configurations yield vastly different performance due to hardware/Software Library disparities—a scenario potentially not covered in the experiments. Could this specific situation be tested?

3. Typo: Table 3 presents KnowOS and AutoOS, but the text description indicates a comparison with vanilla LLM.

### Soundness
1

### Presentation
2

### Contribution
2
