# Fewer Battles, More Gain: An Information-Efficient Framework for Arena-based LLM Evaluation

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 6, 8, 8

## Abstract
Arena-based evaluation has become a key method for assessing large language models (LLMs) through head-to-head model comparisons, closely reflecting human preferences. However, current arena rating systems (e.g., ELO rating system) often suffer from inefficiencies due to exhaustive or random model pair annotations, leading to redundant evaluations, longer evaluation times, and lower overall efficiency. To address these challenges, we propose a novel adaptive model-pair selection algorithm. By leveraging the asymptotic normality of LLM ability estimation under sparse conditions, our approach strategically selects high-value model pairs, focusing on confrontations with the lowest variance. Specifically, we introduce Fisher information as a metric to guide model pair selection, optimizing the evaluation process through A-optimality and D-optimality. A-optimality minimizes estimation variance, ensuring balanced reliability across models, while D-optimality reduces uncertainty by maximizing the determinant of the Fisher Information Matrix. Extensive experiments on both simulated and real-world datasets demonstrate that our method outperforms existing approaches in terms of information efficiency and result reliability. Notably, our method offers a flexible, general toolkit that can be easily integrated into existing arena-based platforms, greatly improving scalability and efficiency for large-scale LLM evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces an information-theoretic approach to Arena-based LLM evaluation, applying Fisher information and optimal experimental design (A-/D-optimality) to select the most informative matchups.

### Strengths
1. This paper is a elegant extension of optimal design principles to LLM ranking tasks. The inclusion of Top-K concurrent battles and adaptive scheduling shows good awareness of practical constraints in large-scale evaluation pipelines.
2. The experiments demonstrate that the information-efficient pairing strategy achieves faster convergence and higher ranking consistency with fewer matches. The information-based metrics (A-info and D-info) are intuitive and interpretable for monitoring evaluation efficiency.

### Weaknesses
1. The “ground truth” ranking is derived from the same model used in the method, introducing potential circularity. It would strengthen the paper to include an external or human-verified ranking baseline to validate real-world correlation.
2. The asymptotic normality and information growth results assume an Erdős–Rényi random graph of matchups. However, the real algorithm chooses pairs adaptively based on prior results, violating the independence assumption.

### Questions
1. The explanation for Figure 5 (Line 468 - Line 473) is confusing. Can you explain more details?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a method to improve the efficiency of model comparisons in a chatbot arena style setting consisting of repeated pairwise comparisons between models. Specifically, it uses the Fisher information of a matrix containing information about previous comparisons to guide selection of model pairs to consider. They introduce two specific metrics using Fisher Information, and show on real-world data that using them for model pair selection improves efficiency while still being robust and reliable.

### Strengths
1. I think the problem of improving the efficiency and reliability of pairwise evaluations is incredibly compelling yet under-explored. There is a good amount of work exploring the efficiency problem when considering large benchmarks, but there are fewer that look at the "arena" style comparisons, which are nonetheless very popular. 
2. The experiments are thorough and consider real-world data at a reasonable scale. Multiple baseline methods are compared against, as well as multiple variants of ELO. 
3. The paper contains a rigorous theoretical argument for their approach.

### Weaknesses
1. While I think the baselines are well-chosen, I would have liked to have seen some more comparisons with other active learning methods. I'd especially be interested in comparisons regarding how much data is required for initialization for other methods, and if this new method addresses the cold start problem in a much more efficient way. 
2. The relative performance gains don't seem too substantial considering the additional cost of the method. I think the authors address the caveats in their comparisons well, but I think some further analysis would be warranted to get a finer-grained understanding of the trade-offs between additional computational resources required at scale for the method and annotation volume. 
3. I think one major assumption made here is a fixed prompt distribution. Intuitively, I'd expect the specific test instance chosen to have a major impact as well on determining the relative quality of models. I understand this is likely beyond the scope of this paper, but I think some comparisons with methods that perhaps only consider this factor would be useful, since it could then help compare the impact of properties of the test input and properties of the model pairs themselves.

### Questions
1. In the visualization section, you mention that D-Optimality might just be better in general in the arena setting than A-optimality. Is there an intuition behind what scenarios A-optimality might be better?
2. How much impact does using a smaller initialization set have?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the inefficiency of current arena-based evaluations for Large Language Models (LLMs), such as Chatbot Arena. At present, these arenas depend on random or exhaustive model pairings for human preference annotation, which results in significant redundant evaluations and a waste of resources.

To address this, the paper proposes an adaptive model-pair selection algorithm. The core of this method leverages the statistical properties of ELO rating estimation (asymptotic normality) and introduces Fisher Information as a metric to actively select the most informative (i.e., highest "value") model pairs. Specifically, the paper proposes: A-optimality: Aims to minimize the total variance of the model ability estimations. And D-optimality: Aims to maximize the determinant of the Fisher Information Matrix (FIM) to globally reduce the uncertainty in model rankings.

### Strengths
The problem formulation is significant and compelling.

The method is validated effectively through experiments.

### Weaknesses
The paper's theory is entirely established upon the m-ELO framework, with the FIM being derived from its likelihood function. However, in the experiments, the A/D-optimality strategies, which are based on this specific FIM, are applied to both m-ELO and traditional ELO. This is problematic because traditional ELO is an iterative update algorithm and does not directly optimize a global log-likelihood function.

The robustness of the heuristic experiments needs further validation.

### Questions
A-optimality demonstrates a severe sampling bias in the experiments, showing an extreme tendency to select pairs involving the "anchor" model. Why does D-optimality manage to avoid this issue?

### Soundness
4

### Presentation
4

### Contribution
4
