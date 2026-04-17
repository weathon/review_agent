# Cognitively Inspired Reflective Evolution: Interactive Multi-Turn LLM–EA Synthesis of Heuristics for Combinatorial Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
Designing effective heuristics for NP-hard combinatorial optimization problems remains a challenging, expertise-driven task. Recent uses of large language models (LLMs) primarily rely on one-shot code synthesis, producing fragile, unvalidated heuristics and under-utilizing LLMs' capacity for iterative reasoning and structured reflection. In this paper, we introduce Cognitively Inspired Reflective Evolution - CIRE, a hybrid framework that embeds LLMs as interactive, multi-turn reasoners within an evolutionary algorithm (EA). CIRE (i) constructs performance-profile clusters of candidate heuristics to give the LLM compact, behaviorally coherent context; (ii) engages the model in multi-turn, feedback-driven reflection tasks that produce explainable performance analyses and targeted heuristic refinements to broaden the exploration--exploitation frontier; and (iii) integrates and selectively validates these proposals via an EA meta-controller that adaptively balances search. Extensive experiments on benchmark combinatorial optimization show that CIRE yields heuristics that are both more robust and more diverse, achieving consistent, statistically significant gains over one-shot LLM generation, genetic programming baselines, and population-based EAs without LLM feedback. These findings suggest that interactive, cognitively inspired multi-turn reasoning is a promising paradigm for automated heuristic design.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes CIRE, a cognitively inspired framework that embeds a large language model as a multi-turn reasoner inside an evolutionary algorithm to automate heuristic design for combinatorial optimization. CIRE clusters candidate heuristics by performance profiles to give the LLM compact, behaviorally coherent context, then elicits reflective analyses and targeted refinements. An EA meta-controller selectively validates proposals to balance exploration and exploitation. On online bin packing, CIRE yields more robust, diverse heuristics and statistically significant gains over one-shot LLM code generation, genetic programming, and EAs without LLM feedback, reducing optimality gaps and excess-bin fractions under tight capacities.

### Strengths
All the proposed components are well-motivated. Multi-turn critique/refinement leverages LLM reasoning beyond one-shot code. 
Performance-profile clustering provides compact, behaviorally coherent prompts that improve generalizable updates. 
EA meta-controller balances exploration/exploitation with selective validation for stability.

### Weaknesses
The presentation could be improved. Section 4.2 is somewhat hard to follow.

The fatal weakness is the lack of validation. The authors should evaluate their methods on more datasets and problem types, and with additional LLMs. The experiments also lack an ablation study for each algorithmic component.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes CIRE, a hybrid framework that embeds an LLM as a multi-turn “reflective” reasoner inside an evolutionary algorithm to synthesize heuristics for combinatorial optimization (e.g., TSP, BPP). CIRE clusters candidate heuristics by performance profiles to give the LLM structured context, prompts the model to analyze strengths/weaknesses and propose targeted refinements, and uses an EA meta-controller to validate and balance exploration vs. exploitation. Experiments report consistent, statistically significant gains over one-shot LLM generation, genetic programming, and EA baselines; qualitatively, CIRE evolves novel strategies (e.g., ARP, QTBP) that surpass prior best scores.

### Strengths
Clear, well-motivated shift from one-shot LLM code synthesis to iterative, cognitively inspired reflection. 

Thoughtful design with performance-profile clustering + multi-turn feedback + EA meta-control, which addresses key issues in current LLM+EA methods. 

Empirical evidence suggests superior robustness/diversity of heuristics on BPP.

### Weaknesses
- The paper lacks a comprehensive evaluation on more problems and datasets.
- Quantitative details are light in the excerpted text: dataset sizes, statistical test specifics, runtime/compute budgets, etc. Ablation is lacking. Stronger reporting would aid reproducibility.
- Potential sensitivity to clustering choices isn’t fully dissected; robustness across LLMs and hyperparameters remains to be demonstrated.

### Questions
How are the weights α and β in the similarity metric (behavioral vs. CodeBLEU semantic) chosen, and how sensitive are results to them?

What is the stopping criterion for multi-turn refinement?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Cognitively Inspired Reflective Evolution (CIRE), a hybrid LLM–EA framework that treats the LLM as a multi-turn reasoner rather than a one-shot code generator.  It is composed of 3 phases: 1) it clusters candidate heuristics by performance profiles relative gap vectors and CodeBLEU similarity; 2) it forms both homogeneous (similar) and heterogeneous (entropy-mixed) groups for comparative reflection; and 3) runs a reflection–exploration/exploitation loop whose proposals are selectively validated by an EA meta-controller. Experiments focus on online bin packing report lower excess-bin ratios.

### Strengths
The main idea of multi-turn reasoning makes sense, and the calculation of similarity via CodeBLEU is a neat addition

### Weaknesses
1. The “multi-turn reasoning” has many connections to e.g., ReEvo and feels incremental overall
    
2. Experiments are insufficient:
    
    1. Only one setting (online bin packing) is not enough to justify the method
        
    2. Many methods have been mentioned but not compared against, including HSEvo, which also incorporates diversity measures
        
3. No ablation studies are provided, making it impossible to understand the importance of each proposed component
    
4. Results are shaky: ReEvo can outperform CIRE in some cases, and the first 2 columns for capacity 500 report the same results for all approaches, presumably a mistake
    
5. No code nor hyperparameters have been provided
    
6. TSP is mentioned several times but not compared against

Overall paper feels rushed, with also weird formatting reminiscent of LLM writing. Given the above, I believe this paper at the current state should not be accepted.

### Questions
1. Can you provide ablation studies?
    
2. Can you provide at least another setting to demonstrate your method, other than online bin packing?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CIRE, a framework that combines large language models with evolutionary algorithms for automatic heuristic design. Unlike one-shot generation, CIRE enables multi-turn reflection where the model critiques and refines heuristics grouped by performance similarity and diversity. Applied to the Online Bin Packing problem, it achieves better heuristic quality than classical and recent LLM-based baselines.

### Strengths
1. The core idea is reasonable and interesting: it might be good to think about a step-by-step heuristic generation instead of a one-shot.

### Weaknesses
1. For the claim in L52 that “While attractive, this paradigm often results in unstable or unvalidated solutions, and underutilizes the LLM’s potential for iterative reflection and improvement.”, I think this is a bit overstates the limitations of prior work because recent frameworks like ReEvo and HSEvo already employ multi-turn reflection and performance validation. And experimentally, the unvalidated solution problems are not significant. This will decrease the motivation of the proposed methods.
    
2. The paper does not provide any prompt templates or examples for the LLM interactions. The results are not reproducible.
    
3. The authors claim the proposed method is for “CO”, but actually only for the bin packing problem.
    
4. The computational cost of multi-turn LLM reasoning is not reported. No runtime, token usage, or reflection-turn statistics are given, leaving sample efficiency and scalability unclear.
    
5. The results rely solely on a single proprietary model (DeepSeek V3), with no comparison across different LLMs or open-source baselines, hindering reproducibility.
    
6. There is no ablation isolating the effects of clustering, entropy-based grouping, or reflective multi-turn reasoning on performance. The improvement could stem from increased prompt diversity rather than the proposed reflective mechanism.
    

Overall, I think this paper is not ready for ICLR.

### Questions
See the weakness.

### Soundness
1

### Presentation
1

### Contribution
1
