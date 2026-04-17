# Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
While large language models (LLMs) are increasingly used as automated heuristic designers for vehicle routing problems (VRPs), current state-of-the-art methods predominantly rely on prompting massive, general-purpose models like GPT-4. This work challenges that paradigm by demonstrating that a smaller, specialized LLM, when meticulously fine-tuned, can generate components that surpass expert-crafted heuristics within advanced solvers. We propose RFTHGS, a novel Reinforcement learning (RL) framework for Fine-Tuning a compact LLM to generate high-performance crossover operators for the Hybrid Genetic Search (HGS) solver, applied to the Capacitated VRP (CVRP). Our method employs a multi-tiered, curriculum-based reward function that progressively guides the LLM to master generating first compilable, then executable, and finally, superior-performing operators that exceed human expert designs. This is coupled with an operator caching mechanism that discourages plagiarism and promotes diversity during training. Comprehensive experiments show that our fine-tuned LLM produces crossover operators which significantly outperform the expert-designed ones in HGS. The performance advantage remains consistent, generalizing from small-scale instances to large-scale problems with up to 1000 nodes. Furthermore, RFTHGS exceeds the performance of leading neuro-combinatorial baselines, prompt-based methods, and commercial LLMs such as GPT-4o and GPT-4o-mini.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces RFTHGS, a novel Reinforcement Learning (RL) framework for fine-tuning a small LLM to produce high-performance crossoyer operators for the Hybrid Genetic Search (HGS) solver to solve the capacitated vehicle routing problem (CVRP). It presents a framework for automated heuristic design. The novel combination of a curriculum-based reward (compilability, executability, performance), an AST-based anti-plagiarism cache, and the use of the solver itself as the evaluation environment is a significant methodological contribution.It provide strong, empirical evidence that a small, accessible (14B) LLM, when fine-tuned via RL, can generate core algorithmic components that exceed the performance of human-expert designs within a SOTA CO solver.The strong generalization from small to large-scale problems is particularly impressive and demonstrates that the LLM has learned a genuinely robust and effective heuristic.

### Strengths
The paper proves that specialized, fine-tuned small LLMs are superior to prompted large LLMs.It is innovative and strongly supported by the data.The multi-tiered curriculum-based reward function solves the sparse reward problem in code generation. The AST-based anti-plagiarism mechanism is a great solution to prevent reward hacking and ensure the generation of novel, diverse operators.It offers a practical, accessible path for continuously improving SOTA solvers.

### Weaknesses
While the LLM is small, the RL training loop appears computationally intensive.The paper does not quantify the total training cost. In addition, the paper demonstrates that the LLM-generated operator is better, but not why. An analysis of the generated SOTA operator's code is needed.

### Questions
1.Please provide a pseudo-code or a qualitative description of one of the  LLM-generated crossover operators.

2.Please provide more details on the total computational cost.

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
4

### Summary
This paper introduces RFTHGS, a reinforcement learning framework that fine-tunes a 14B parameter LLM to generate crossover operators for the Hybrid Genetic Search (HGS) algorithm, specifically targeting the Capacitated Vehicle Routing Problem (CVRP). The authors employ a multi-tiered reward function that progressively guides the LLM through stages of producing compilable code, executable operators, and ultimately components that outperform human-designed ones. They also implement an operator caching mechanism using Abstract Syntax Trees to prevent plagiarism and promote diversity during training. Experimental results on CVRPLIB benchmarks demonstrate that the fine-tuned LLM generates crossover operators that significantly outperform expert-designed operators in HGS, with improvements holding from small instances (100 nodes) to large-scale problems (1000 nodes). The method also surpasses various baselines including neuro-combinatorial approaches and prompt-based methods using commercial LLMs like GPT-4o. This work demonstrates that smaller, specialized LLMs can be effectively fine-tuned to produce solver components that exceed the performance of human-expert designs in state-of-the-art optimization solvers.

### Strengths
1. The authors present their work with exceptional clarity, following a logical progression from problem formulation through methodology to comprehensive experimental validation. The paper effectively uses figures (particularly Figure 1's pipeline visualization) and maintains consistent terminology throughout.
2. The method achieves substantial improvements over expert-designed operators in HGS and consistently outperforms all baselines including state-of-the-art neuro-combinatorial methods (POMO, RF-POMO) and commercial LLMs (GPT-4o, GPT-o3) across problem sizes ranging from 100 to 1000 nodes, showing excellent scalability.
3. This work pioneers the use of reinforcement learning to fine-tune language models specifically for generating optimization solver components, demonstrating that LLMs can learn to produce code that not only compiles and executes correctly but actually surpasses human expert designs in a state-of-the-art solver like HGS.
4. The paper introduces several technical innovations including incremental compilation (reducing compilation time to 25% of full recompilation), an anti-plagiarism cache using Abstract Syntax Trees to encourage diverse operator generation, and a well-designed multi-tiered reward function that effectively guides learning through progressive stages from syntax correctness to performance optimization.
5. The work demonstrates that a fine-tuned 14B parameter model significantly outperforms trillion-parameter general models (GPT-4o series), with Table 2 showing perfect compilation rates (16/16) compared to poor rates for GPT models (3/16-9/16), and the GPT models failing to produce any functional improvements despite generating syntactically correct code, highlighting the superiority of task-specific fine-tuning over general-purpose capabilities.

### Weaknesses
In general, I believe this is a good paper and lean toward acceptance. If the authors could address the following concerns, the paper could be stronger:

1. The paper exclusively evaluates on the CVRPLIB X benchmark set, which, while comprehensive, represents only one distribution of CVRP instances. Evaluating on additional distributions would strengthen the generalizability claims. For example, testing on synthetic CVRP settings with controlled characteristics (clustered vs. uniform customer distributions), other CVRPLIB instance families (A, B, E, F sets), or real-world logistics datasets would provide more robust evidence of the method's effectiveness across diverse problem structures.
2.  The evaluation is confined to optimizing a single operator (crossover) within a single solver (HGS). Expanding to additional settings would make the contributions more compelling. For instance, applying the framework to other state-of-the-art solvers like LKH-3, OR-Tools, or Concorde, and targeting different operators such as local search moves, construction heuristics, or selection mechanisms would demonstrate the broader applicability of the RL fine-tuning approach for solver component design.
3. The method appears to require substantial domain-specific engineering, including detailed operator API specifications, incremental compilation setup, integration with solver codebases, and carefully crafted reward functions specific to each operator type. These requirements may create significant barriers for researchers wanting to apply this approach to new problem settings or different solvers, potentially limiting the practical impact of the work. The authors could strengthen the paper by providing a more general framework or toolkit that abstracts away some of these implementation complexities.

### Questions
Besides the concerns in the weakness part, I also have the following questions:

1. Why do the other prompting-based methods like ReEvo have so poor performance? Are the prompting-based baselines (ReEvo) actually integrated to evolve the crossover operator within the HGS framework, or were they used to evolve standalone heuristics?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the problem of designing crossover operators for the Hybrid Genetic Search (HGS) algorithm. It introduces RFTHGS which prompts a small LLM to code potential crossover operators and fine tunes the LLM with RL given a reward based on if the code compiles, if it is plagiarized from examples and its task performance. They show results equivalent and in most cases better than state-of-the-art.

### Strengths
The idea of generating code with an LLM and fine tuning it with RL for the purpose of improving human designed heuristics is interesting and this paper has shown that it can be useful. It is impressive that the method clearly generalises well to larger unseen task.


The reward design is clearly well thought out to provide both necessary training speed and improving results without copying existing solutions.


The results are comprehensive, and compare against a wide range of existing benchmarks. The fact that this achieves state-of-the-art performance in most cases is impressive.

### Weaknesses
My first concern is that it seems RFTHGS is not significantly better than the previous SOTA (HGS-PyVRP). As given in Table 1, HGS-PyVRP is always underlined when it is second best. This is not explained in the text, but I would assume it means within 1 standard deviation. Thus the results may not be statistically significant, this needs to be clarified in the text.  
Another concern along the same lines of comparing to HGS-PyVRP, is that the chosen generated crossover mechanism is never compared to the human designed heuristics and so it is hard to know how novel the new crossover algorithm is. It would be useful to have a discussion around the differences so that readers can know if these are trivial changes or significant algorithmic innovations.


The ablation study on reward design is lacking. It would have been interesting to see the impact of each part of the reward, for example how does the LLM perform without the plagiarism penalty?


While the benchmark is comprehensive it seems to be lacking a SOTA NCS method in “Combinatorial Optimization with Policy Adaptation using Latent Space Search” (COMPASS) [1]


My final concern is the lack of generality of the method. It would have been interesting to see if the method works generally for defining heuristics in other evolutionary algorithms e.g defining new mutation strategies in Genetic algorithms, new velocity update rules in Particle Swarm Optimization, or novel pheromone update rules in Ant Colony Optimization.  
Along with this HGS is somewhat specialized to VRP problems and when testing on other algorithms it would be interesting to see the performance on other combinatorial optimisation benchmarks like traveling salesman and job shop scheduling.


[1] Chalumeau, Felix, et al. "Combinatorial optimization with policy adaptation using latent space search." Advances in Neural Information Processing Systems 36 (2023): 7947-7959.

### Questions
I do not see where the meaning of underlined values is explained for table 1. I have assumed that they mean values are within one standard deviation of the best value.

### Soundness
3

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
This paper introduces RFTHGS, a reinforcement learning framework for fine-tuning a 14B parameter LLM to generate optimized crossover operators for the Hybrid Genetic Search (HGS) algorithm applied to the Capacitated Vehicle Routing Problem (CVRP). The approach uses a multi-tiered, curriculum-based reward function that progressively guides the LLM through three stages: generating compilable code, producing executable operators, and finally creating components that exceed expert-designed baselines.

### Strengths
1. Novel Contribution: First demonstration that a small fine-tuned LLM (14B) can generate operators outperforming expert-designed components in a state-of-the-art CVRP solver.
2. Strong Performance: Achieves -0.12% to -0.33% gap reduction over HGS baselines and outperforms trillion-parameter models (GPT-4o, GPT-o3, GPT-o4-mini) in both compilation success rate (16/16 vs. 3-9/16) and solution quality.
3. Good Generalization: Successfully generalizes from training instances (n<400) to large-scale problems (n=1000) and across different iteration budgets.
4. Well-Designed Method: The multi-tiered reward with continuous feedback (Equation 3) and incremental compilation for efficiency are effective design choices.

### Weaknesses
1. Severely Limited Scope:
- Only optimizes ONE operator (crossover) for ONE problem variant (CVRP)
- Title/abstract claim applicability to "VRP" but no experiments on VRPTW, PCVRP, or other variants
- No demonstration that the approach works for other operators (mutation, selection, etc.)
- This fundamentally limits the practical impact and undermines generalizability claims
2. No Computational Cost Analysis:
- Missing total GPU hours for RL training
- No analysis of cost for evaluating operators during training (HGS runs on 30 instances per iteration × 1000 iterations × batch size)
- Unclear if performance gains justify the substantial training investment
- Makes it impossible to assess practical viability
3. Shallow Analysis of Discovered Operators:
- Section 6 mentions human experts analyzed operators but provides no concrete details
- No explanation of WHAT modifications were discovered or WHY they work
- Appendix C analysis is referenced but not included
- Missing opportunity to extract design insights that could inform future operator development
- Example output in A.4 shows reasoning but doesn't explain actual performance gains
4. Questionable Experimental Design:
- Training only on n≤400 instances while claiming to solve large-scale problems (n=1000)
- Only 30 training instances seems insufficient for RL training stability
- No variance/error bars reported across multiple runs
- Validation set selection process not described
5. Incomplete Baselines:
- No comparison with automated algorithm design methods beyond LLMs (e.g., automated parameter tuning, grammar-based hyper-heuristics)
- NCO baselines (POMO, AM) are construction methods, not improvement methods like HGS
- Missing recent learning-based metaheuristic operator design work
6. Technical Gaps:
- Reward values (-1, -0.8, -0.7) appear arbitrary with no sensitivity analysis
- AST-based anti-plagiarism: similarity threshold and detection details unclear
- Incremental compilation: 75% reduction not validated across all operators
- DAPO choice not justified; no ablation vs. standard PPO

### Questions
1. Why optimize only the crossover operator? What happens with mutation/selection?
2. Can you provide results on other VRP variants to justify the broad claims?
3. What is the total training cost, and how does cost-benefit compare to manual design?
4. What specific code changes did the LLM discover, and why do they improve performance?
5. Why train only on small instances if the goal is large-scale optimization?

### Soundness
2

### Presentation
3

### Contribution
2
