# HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6, 2

## Abstract
This paper investigates the application of Large Language Models (LLMs) in Automated Heuristic Design (AHD), where their integration into evolutionary frameworks reveals a significant gap in global control and long-term learning. We propose the Hindsight-Foresight Prompt (HiFo-Prompt), a novel framework for LLM-based AHD designed to overcome these limitations. This is achieved through two synergistic strategies: Foresight and Hindsight. Foresight acts as a high-level meta-controller, monitoring population dynamics(e.g., stagnation and diversity collapse) to switch the global search strategy between exploration and exploitation explicitly. Hindsight builds a persistent knowledge base by distilling successful design principles from past generations, making this knowledge reusable.  This dual mechanism ensures that the LLM is not just a passive operator but an active reasoner, guided by a global plan (Foresight) while continuously improving from its cumulative experience (Hindsight). Empirical results demonstrate that HiFo-Prompt significantly outperforms a comprehensive suite of state-of-the-art AHD methods, discovering higher-quality heuristics with substantially improved convergence speed and query efficiency.  Our code is available at https://github.com/Challenger-XJTU/HiFo-Prompt.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces HiFo-Prompt, a new method for LLM-based AHD. It integrates a Foresight module, featuring an evolutionary navigator that monitors population dynamics and steers the search using interpretable "verbal gradients," and a Hindsight module, which maintains a self-evolving insight pool that distills successful design principles from high-performing code into knowledge. Evaluated on different heuristic design tasks like TSP, BPP, and FSSP, HiFo-Prompt demonstrates competetive performance, achieving superior results with greater computational efficiency than existing LLM-based AHD methods.

### Strengths
The self-evolving insight pool (Hindsight) and foresight instructions effectively prevent knowledge decay while enabling more strategic exploration of the heuristic space.

It achieves superior performance with fewer LLM calls and lower runtime compared to other state-of-the-art AHD methods.

### Weaknesses
The evolutionary navigator uses a fixed, rule-based policy with hand-tuned thresholds, which may lack generalization.

The paper would benefit from additional illustrations and a more extensive set of results to further support its claims

### Questions
The stagnation is measured by raw fitness, delta g, which is a fixed value and may suffer from poor generalization to different heuristic design tasks.

The semantic variety is calculated based on the textual descriptions of algorithms (eq. 7). Is it the thought or the code text? It seems that the indicator only counts when the two algorithms are exactly the same. Will it be too greedy?

For the Foresight module, how was the specific set of Design Directives in the pool (Appendix G.3) designed? Was there an ablation study on the impact of different directive wordings on the LLM's output quality?

The framework's knowledge management is confined to a single task; can the learned insights be generalized or transferred to new, unseen problem domains?

What does the final algorithm look like? and how the insights and foresight prompts contribute to the generation of better heuristics, could you provide example illustrations?

A discussion and comparison with related works on prompt evolution and hierarchical search is suggested [1-3].

[1] MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design, arXiv

[2] Large Language Model-driven Large Neighborhood Search for Large-Scale MILP Problems, ICML

[3] Experience-guided reflective co-evolution of prompts and heuristics for automatic algorithm design, arXiv


There are typos and inadequate descriptions: e.g., 

line 811 ?

line 854, 788

Figure 1, The Left and Right can be misleading

### Soundness
3

### Presentation
3

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
The paper proposes HiFo-Prompt, a prompting framework for LLM-based automated heuristic design that marries two modules: Foresight (an Evolutionary Navigator that steers exploration vs. exploitation from population signals) and Hindsight (an Insight Pool that distills and reuses design principles from successful code across generations). By decoupling “thoughts” from code, HiFo-Prompt supplies state-aware guidance and persistent knowledge. Experiments on TSP, FSSP, online bin packing, and black-box functions show state-of-the-art quality, faster convergence, and lower token/time cost than prior AHD systems; ablations confirm both modules matter.

### Strengths
- Dual Foresight/Hindsight design elevates the LLM from code generator to meta-optimizer. 
- Evaluation sees evident performance gain.

### Weaknesses
- It’s unclear how you ensured a fair comparison under “the same query budget.” Does distilling insights consume additional queries? How many times did you run your method and the baselines? Did you use the same number of heuristic evaluations? Standard deviations are not reported, so the performance gains are not fully convincing.

- The approach involves many hyperparameters. It’s unclear how they were chosen and how robust the method is to their settings.

- The method relies heavily on pre-engineered prompts.

- Similar ideas appear in EoH and ReEvo, where thoughts and reflections are distilled (both) and accumulated (the latter). Please clarify the novelty relative to these.

### Questions
See weaknesses.

### Soundness
2

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
3

### Summary
HiFo-Prompt tackles two common gaps in LLM-based Automatic Heuristic Design (AHD): lack of global search control and poor knowledge persistence. It adds  a rule-based Foresight meta-controller that watches population progress/diversity and switches prompts among explore/exploit/balance regimes, and a Hindsight Insight Pool that distills reusable design principles from elites with utility-based credit assignment, then injects top-scoring insights into subsequent prompts. The method obtains the best results among various LLM-AHD baselines.

### Strengths
- The idea of tracking both local and global evolution dynamics via specialized modules is interesting and well executed
    
- Useful ablation studies
    
- Strong performance with few function evaluations

### Weaknesses
1. Seed insights are required by the method. Importantly, these insights could significantly improve generation quality: “Design adaptive hybrid meta-heuristics synergistically fusing multiple search paradigms and dynamically tune operator parameters based on search stage or problem features.” particularly is a high-quality handcrafted prompt that can have a substantial effect on the generation. For fairness of comparison, one should provide the same information in the prompt of other baselines, say EoH.
    
2. The novelty regarding global control and historical information aggregation is overstated, e.g., ReEvo already implements a short and long-term reflection that could be seen as a simpler version of hindsight. Discussions would be appreciated.
    
3. I am not convinced about the population size being chosen as 4. How can diversity be maintained in such as small population and avoid inbreeding?
    
4. I found the methodology section quite confusing, with many quite complicated implementations. For example, a decay rate is introduced, but there is no ablation or sensitivity analysis on it.  Eq. 3, which describes the evolutionary contribution, is full of hardcoded parameters, which are hard to parse, and the rationale for choosing them is not explained.
    
    1. On this point, please clarify whether $g$ is a minimization or maximization objective. In EoH, this is maximization, but Fig. 2 and equations suggest otherwise. However, section A.1 again takes $g$ as argmax. This is confusing.
        
5. No code is provided

### Questions
1. About dissimilarity (Eq 7): how are the textual descriptions calculated, and how do you ensure these are the same? (e.g.: will changing a single word make two descriptions different?)
    
2. It appears that there is a massive degradation if Qwen 2.5 max is not used in Table 9. How do you explain this?
    
3. What would happen if baseline methods also have the seed insights as part of the generator prompt?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes HiFo-Prompt with (i) a Hindsight module that distills reusable principles from successful candidates, and (ii) a Foresight module that adaptively switches explore/exploit/balance based on population state to guide LLM-based AHD. The proposed method is evaluated on TSP, Online BPP, FSSP, and BO.

### Strengths
1. The proposed method is well motivated and outperforms recent LLM-based AHD baselines across several tasks.
2. The design details are well presented.
3. The limitations and future directions are clearly analyzed.

### Weaknesses
1. For TSP step-by-step construction (i.e., Table 1), Appendix B.1 states that HiFo-Prompt involves LLM calls at inference time, however, it is unclear to me that whether such strategy also applies to the baselines. Please disambiguate: (a) If baselines also call the LLM at inference, please explain why HiFo-Prompt’s runtime is longer; (b) If they do not, please also report HiFo-Prompt under the same inference protocol for fair comparisons.
2. The main text claims TSPLIB results are in Appendix C.1, but C.1 contains only descriptive text and a placeholder “Table ??”, with no actual results. Please add the promised table/metrics or revise the pointer.

### Questions
1. Line 387 says “100 instances at each of five sizes,” but Table 1 shows three, please fix the mismatch. Also, there are several misplaced “?” characters around lines 811, 854, 946, 967 that need cleanup.
2. Can you present some of the actual heuristics generated and used to produce the reported results?
3. In Table 5, removing the Insight Pool would make the method perform worse than EoH, which is surprising to me since the setup still retains the Foundational Prompts adapted from EoH and the Navigator module. Can you analyze the concrete differences between EoH and HiFo-Prompt w/o Insight Pool & Navigator that can explain this gap? Will the Navigator module improve baselines like EoH as a drop-in controller?
4. How frequently does the Navigator select explore or exploit across runs? Have you tried an ablation that fixes the state to “balance” throughout to isolate the benefit of adaptive switching?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents HiFo-Prompt, a framework for LLM-based automated heuristic design that combines Hindsight, which builds an evolving Insight Pool of distilled design principles, and Foresight, an Evolutionary Navigator that adaptively balances exploration and exploitation. The method is applied to several optimization tasks (TSP, Online BPP, FSSP, and BO), and the authors report improvements over prior AHD methods in both solution quality and sample efficiency.

### Strengths
1. The motivation of this paper is clear and reasonable. The design ideas of global guidance and the insight pool are interesting and inspiring.  
2. The similarity-based diversity discussion for the insight pool is conceptually stimulating.  
3. The paper is clearly written and well organized, making it easy to follow.

### Weaknesses
1. I have concerns about the novelty threshold. The Insight Pool’s novelty filtering relies on Jaccard similarity over token sets. While this removes near-duplicate sentences, such a pure text-based comparison cannot capture semantic overlap. For example, one insight might be expressed in different ways. Since this novelty threshold is crucial for ensuring diversity, I worry this design may harm the actual effectiveness of the diversity mechanism.  

2. The combination of a usage penalty and a recency bonus in $U(k, t)$ aims to balance exploration and exploitation, but the dynamics between these opposing terms are not analyzed. This could be sensitive in practice, and it would be helpful to justify or empirically demonstrate that this interaction leads to stable selection rather than oscillatory behavior. In particular, $w_u$ is a hyperparameter without ablation or sensitivity analysis, and the calculation of $B_r$ is not clearly presented. This reduces the soundness and reproducibility of the method.  

3. The mapping from normalized performance $\tilde{\rho}$ to the effective credit $g_{\text{eff}}$ uses manually chosen piecewise constants (0.8, 0.6, 0.5, -0.3, etc.) with no theoretical justification or ablation. While the idea of tiered reward regimes is understandable, the specific scaling choices seem ad hoc and may not generalize across tasks. It would strengthen the work to at least provide hints or guidelines on how to select these values.  

4. The definition of phenotypic diversity as the fraction of non-identical algorithm text strings feels coarse and potentially misleading. The measure is a bit lexical that two code snippets are treated as completely different even if they differ only by refactoring or variable renaming, ignoring actual semantic or functional similarity (similar with my commen in 2.). As a result, the system may overestimate diversity and trigger unnecessary exploration. Moreover, this approach scales as $O(|P|^2)$ comparisons per generation, which may become inefficient for larger populations and increase token consumption for LLM-based evaluations. The diversity threshold is also arbitrary and not justified or ablated. Overall, the lack of semantic grounding and unclear efficiency raises concerns about the robustness and practicality of the Navigator’s diversity control.  

5. The experimental section raises several concerns about fairness, reproducibility, and efficiency. Although the paper states that all LLM-based baselines were evaluated under the same Qwen 2.5-Max model, implementation details and prompt adaptations are not provided, so fairness remains unclear (like baselines might use different LLMs, thus they can not be comparied directly). HiFo-Prompt’s runtime on small TSP instances (Table 1) is about an order of magnitude slower than competing methods, with no explanation, contradicting the claim of improved convergence speed. Token-usage statistics are summarized only coarsely (Appendix C.7) without breakdown or cost analysis, leaving uncertainty about true computational overhead. The brief multi-LLM comparison (Table 9) covers only two tasks and lacks analysis, providing little evidence of model generality. Finally, runtime behavior is inconsistent across tables (slower in TSP 10–50 but faster in TSP 100–500) with no explanation. Together, these issues make it difficult to assess the practical efficiency and generalizability of the proposed framework.  

6. The code does not seem to be provided. Even though the authors share the core prompts, several computational details remain unclear, as mentioned in earlier points. This makes it hard to guarantee reproducibility and verify the soundness of the proposed method.  

**Minors**

1. Very minor: for LaTeX quotation marks, please use the proper “…” format instead of plain double quotes. For example, in L055 the quotation marks are incorrectly formatted.  
2. There are a few missing or incomplete citations in the appendix, such as at L811 and L1017. These should be corrected for completeness and consistency.

### Questions
See the weakness.

### Soundness
1

### Presentation
3

### Contribution
2
