# Evolving Graph Structured Programs for Circuit Generation with Large Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Logic synthesis (LS), which aims to generate a *compact* logic circuit graph with minimized size while *accurately* satisfying a given functionality, plays an important role in chip design. However, existing LS methods struggle to balance circuit structure compactness and functional accuracy, often leading to suboptimal generation. To address this problem, we propose a novel *Circuit Program Evolution* framework, namely CircuitEvo, which iteratively leverages large language models (LLMs) to evolve circuit programs towards improved compactness while preserving functional accuracy. Specifically, CircuitEvo models the circuit graph as a structured program and leverages the strong generative capabilities of LLMs — guided by domain-specific evolutionary prompt strategies — to generate promising circuit candidates in each iteration. Moreover, a structure-aware circuit optimization module is introduced to correct functional discrepancies by appending necessary substructures to the generated circuits.  To the best of our knowledge, CircuitEvo is *the first* LLM-based LS approach that can iteratively improve a circuit's compactness while ensuring functional accuracy.  Experiments on several widely used benchmarks demonstrate that CircuitEvo can efficiently generate accurate circuits with up to 16 input number and 69 output number. Moreover, our method significantly outperforms state-of-the-art methods in terms of circuit size, achieving an average improvement of 6.74%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
CircuitEvo is a novel framework that utilizes large language models (LLMs) to iteratively evolve circuit programs, enhancing compactness while ensuring functional accuracy. ​

The paper introduces CircuitEvo, the first LLM-based logic synthesis method that improves circuit compactness and functional accuracy. ​It employs a graph-structured circuit program representation to facilitate LLM comprehension and generation. ​It integrates an evolutionary program generator and a structure-aware function optimizer in an iterative framework. ​Empirical studies show  an average improvement of 6.74% in circuit size over state-of-the-art methods. 


Task:  Given a truth table T , CircuitEvo aims to generate a circuit program P that exactly satisfies the given functionality while minimizing the number of structure lines (circuit nodes).

### Strengths
The proposed graph-structured circuit program effectively encodes circuit topology and functionality in a textual format. ​

The program consists of three parts: IO definition, structure description, and function definition.
This representation bridges the gap between circuit graphs and language-based reasoning, enhancing LLM capabilities. ​
The LS problem is formulated to generate a circuit program that satisfies functionality while minimizing the number of nodes. ​

Evolutionary Program Generation Process
CircuitEvo employs an evolutionary framework to generate diverse and promising circuit programs using LLMs. ​

Initializes a diverse population of accurate circuit programs through a decomposition strategy.
Utilizes domain-specific evolutionary prompting strategies to guide LLMs in generating high-quality programs. ​
Incorporates fitness-based population management to select the top-performing programs for subsequent iterations.

### Weaknesses
The LS summary uses very dated algorithms: existing LS approaches typically follow a generate-then optimize framework: they first generate a circuit from the truth table, then apply various operators for circuit optimization. Please update the references to show that topology-driven modern methods exist.

In terms of baselines I am curious to understand why you selected BDD (quite outdated---1993), and SOP (2017, also not SoA).
Further, since you use evolutionary methods, why not compare against one of the many evolutionary SoA methods; some very recent references are below. The failure to compare against existing evolutionary SoA methods means that we can't evaluate ablation techniques well, i.e., is it the evolutionary part that makes the difference, the graph part, or what?


Chen, Yu, and Shao-Yun Fang. "Optimizing Analog Circuit Design Through a Machine Learning‐Assisted Evolutionary Algorithm." Electronics Letters 61, no. 1 (2025): e70331.

Campilho-Gomes, Miguel, Rui Tavares, and João Goes. "Analog flat-level circuit synthesis with genetic algorithms." IEEE Access (2024).

Srikanth, V., P. Aswini, Rakesh Chandrashekar, N. Sirisha, Manish Kumar, and K. Adnan. "Machine Learning-Based Analogue Circuit Design for Stage Categorization and Evolutionary Optimization." In 2024 Second International Conference Computational and Characterization Techniques in Engineering & Sciences (IC3TES), pp. 1-6. IEEE, 2024.

### Questions
1. please address the choice of baselines
2. please carefully focus on ablation regarding which aspect of your approach provides the real benefit (see comments about choice of baseline)

### Soundness
3

### Presentation
4

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
This paper introduces CircuitEvo, an LLM-based evolutionary framework for circuit programs, to tackle the difficulty in balancing compactness and accuracy in LS. By leveraging a graph-based program representation alongside LLM-driven evolution and structure-aware optimization, it iteratively optimizes sequential circuits. Results show 100% generation accuracy, an average 6.74% reduction in circuit size, and significantly better search efficiency versus baseline methods.

### Strengths
1. This paper addresses the important problem of automated sequential circuit design, offering well-defined practical value.
2. The experimental results significantly outperform existing baseline methods, demonstrating clear advantages in terms of accuracy, scale, and efficiency.  
3. The theoretical derivations are logically rigorous, with proofs provided for the relevant remarks.

### Weaknesses
1. The paper has a deficiency in terms of innovation. The core techniques—LLM generation, evolutionary algorithms, and circuit function completion—constitute an integration of existing methods, and no novel model architecture or theoretical framework is proposed. Furthermore, the local program search is essentially an incremental improvement of a greedy strategy, lacking a breakthrough in algorithmic originality.
2. The experimental evaluation in the paper is limited to synchronous sequential circuits and combinational circuits. It does not demonstrate the framework's applicability to asynchronous sequential circuits or circuits with complex feedback loops.
3. The paper lacks comprehensive ablation studies. It only examines the impact of variants like removing the LLM or the evolutionary algorithm, but fails to evaluate the individual contribution of different evolutionary strategies (E1/E2/R1/R2). For instance, how much would circuit diversity decrease if the E1 exploration strategy were removed?
4. Validation on extreme scales is missing. The framework was not tested on very large-scale circuits (e.g., with inputs >16 and outputs >70), leaving its scalability unverified.
5. The comparative analysis of LLMs is brief. It merely lists results from models like GPT-3.5-turbo and Deepseek-V3 without analyzing the impact of LLM parameter scale (e.g., 7B vs. 13B) on generation quality.

### Questions
1. Please respond to the concerns I have raised in the 'weaknesses' section. If the revision can adequately address most of the critical issues, I would consider raising the score.
2. In my personal opinion, the contribution of this paper lies more in its solid experimentation and significant performance improvements, rather than in its theoretical advancement in AI. Considering this, EDA-focused conferences (such as DAC or ISCA, with an upcoming November deadline) would be a more suitable venue for it.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes a framework for logic synthesis, CircuitEvo, which employs LLMs within an evolutionary loop to improve final circuit compactness while preserving functional accuracy. The framework is carefully designed to account for the specific characteristics of circuit synthesis and introduces several domain-tailored operations, such as population initialization via the Shannon decomposition theorem and domain-specific function constraints. These components appear both reasonable and effective in ensuring a diverse set of accurate initial solutions, or enhancing optimization efficiency under an exponentially large search space. Experiments conducted on multi-benchmarks, in comparison with multiple baselines, demonstrate the effectiveness and efficiency of CircuitEvo. Moreover, the authors report an interesting additional finding: triangular structures play a significant role in circuit compactness.

### Strengths
1. Although integrating LLMs into a circuit-design loop is not entirely novel, the authors propose several domain-specific operations to enhance how inherently uncertain LLMs can correctly and efficiently generate the desired circuit structures in this domain — which is commendable.
2. The paper is clearly written, with well-designed figures that aid understanding.
3, I think the authors have conducted comprehensive experiments (although I am not very familiar with experimental design in this field). The benchmarks and baselines appear sufficiently chosen, and the experimental results look impressive.

### Weaknesses
1. The experimental results are based on single-run evaluation. This limits the statistical significance of the findings, especially considering that LLMs can be highly sensitive to random factors such as initialization and prompt variations, whose influences on the experiments are not  discussed in the paper.

2.Several highly relevant off-the-shelf baselines were not included in the experimental comparison, such as DARTS- [1] and T-Net [2].

[1] Xiangxiang Chu, Xiaoxing Wang, Bo Zhang, Shun Lu, Xiaolin Wei, and Junchi Yan. Darts-: robustly stepping out of performance collapse without indicators. arXiv preprint arXiv:2009.01027, 2020.

[2]  Zhihai Wang, Jie Wang, Qingyue Yang, Yinqi Bai, Xing Li, Lei Chen, Jianye HAO, Mingxuan Yuan, Bin Li, Yongdong Zhang, and Feng Wu. Towards next-generation logic synthesis: A scalable neural circuit generation framework. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

### Questions
1. It seems hard to believe that the involvement of LLMs results in such a low convergence time of only 1.48 hours. I am curious whether the reported runtime excludes the LLM inference time between inputting prompt and generating output.
2. Could the authors provide a breakdown of the time consumption for each operation within the evolutionary program generation process in each iteration — for example, the proportions spent on crossover, mutation, and functionality completion?
3. There are some cases where CircuitEvo underperforms the baselines in terms of circuit compactness. Could the authors provide a clear explanation for these results?
4. I look forward to the clarifications that address the above weaknesses.

### Soundness
3

### Presentation
4

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
CircuitEvo proposes an LLM-based framework for logic synthesis that iteratively evolves circuit programs toward improved compactness while preserving functional accuracy. The key innovation is a graph-structured circuit program representation that enables LLMs to understand and generate circuits through evolutionary prompting strategies. A structure-aware function optimizer ensures correctness by appending substructures based on circuit functionality completion theory. Experiments show 6.74% improvement in circuit size over state-of-the-art methods while achieving 100% accuracy across benchmarks with up to 16 inputs and 69 outputs.

### Strengths
1. The graph-structured program formulation (Figure 1) is genuinely clever - it encodes topological structure in a hierarchical, textual format that preserves connectivity information while being LLM-compatible. This bridges the gap between graph-based circuit representations and language-based reasoning.
2. Comprehensive evaluation across 16 circuits, 7 baselines, 3 different LLM backbones (GPT-3.5, Deepseek-V3, Qwen2.5-7B), and multiple optimization operators demonstrates robustness. The ablation study (Table 4) clearly shows each component contributes.

### Weaknesses
1. The "structure-aware function optimizer" uses existing LS tool (ABC) to generate subprograms Pa and Pb when the LLM-generated circuit is incorrect. This raises fundamental questions:

If ABC can generate correct subprograms, why not use ABC for the entire circuit?
How much of the final circuit comes from LLM vs. ABC?
Baselines don't receive this "rescue" mechanism - is the comparison fair?
Table 4 shows w/o Completion drops accuracy to 67-86%, suggesting LLMs contribute incomplete circuits that ABC must fix

2. Severe scalability limitations with no path forward:

Limited to 16 inputs, 69 outputs.
Shannon decomposition (Equation 1) has exponential complexity O(2^(n-1)).
Authors acknowledge "computational constraints limit... to 12-15 components" but provide no solution.
No analysis of asymptotic complexity or scaling behavior.

### Questions
See Weakness

### Soundness
2

### Presentation
3

### Contribution
3
