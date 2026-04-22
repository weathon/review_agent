# VRPAgent: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Designing high-performing heuristics for vehicle routing problems (VRPs) is a complex task that requires both intuition and deep domain knowledge. Large language model (LLM)-based code generation has recently shown promise across many domains, but it still falls short of producing heuristics that rival those crafted by human experts. In this paper, we propose VRPAgent, a framework that integrates LLM-generated components into a metaheuristic and refines them through a novel genetic search. By using the LLM to generate problem-specific operators, embedded within a generic metaheuristic framework, VRPAgent keeps tasks manageable, guarantees correctness, and still enables the discovery of novel and powerful strategies. Across multiple problems, including the capacitated VRP, the VRP with time windows, and the prize-collecting VRP, our method discovers heuristic operators that outperform handcrafted methods and recent learning-based approaches while requiring only a single CPU core. To our knowledge, VRPAgent is among the first LLM-based paradigms to advance the state-of-the-art in VRPs, highlighting a promising future for automated heuristics discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces VRPAGENT, a framework for discovering heuristic operators for Vehicle Routing Problems (VRPs) using large language models (LLMs). The method combines LLM-generated “destroy” and “order” operators with a Large Neighborhood Search (LNS) metaheuristic, leveraging genetic algorithms (GAs) to iteratively evolve improved operators. Although the research motivation and validation results seem feasible, the approach is almost identical to existing LLM-guided heuristic frameworks, which weakens the overall contribution of the paper.

### Strengths
1. The approach is clear, and the LLM-guided evolutionary framework has discovered operators that go beyond expert-designed ones.
2. The authors have conducted a certain level of analysis on the generated heuristic operators.

### Weaknesses
1.	**Incrementa novelty** — The evolutionary framework of VRPAgent is similar to existing heuristic evolutionary frameworks (such as Heuristics evolution based on LLM, e.g., EoH [1]), and the proposed "code length penalty" is also negligible.
2.	**Empirical overclaiming** — The experimental results of VRPAGENT show only minor improvements compared to existing methods. There is a lack of comparison with LLM-empowered LNS approaches, such as LLM-LNS [2]. It is also unclear how it performs compared to adaptive LNS methods like PPO-ALNS [3].
3.	**Fairness of experiments** — It is not clearly stated whether the comparison methods based on LLM heuristic generation have a similar number of API calls.
4.	**Incomplete experimental analysis** — The study lacks an analysis of aspects such as the convergence of the genetic algorithm or the probability of code correctness.

[1] Evolution of heuristics: Towards efficient automatic algorithm design using large language model. ICML 2024.

[2] Large Language Model-driven Large Neighborhood Search for Large-Scale MILP Problems. ICML 2025.

[3] Reinforcement learning-guided adaptive large neighborhood search for vehicle routing problem with time windows. Journal of Combinatorial Optimization, 2025.

### Questions
see weaknesses

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
4

### Summary
This paper proposes a framework for automated heuristic discovery in VRPs using LLMs called VRPAgent. VRPAgent integrates LLM-generated problem-specific operators within a Large Neighborhood Search (LNS) metaheuristic and refines them through a genetic algorithm that employs elitism, biased crossover, and code-length penalty mechanisms.​

Key features include generating problem-specific destroy and insert heuristics via LLMs, and evolving these operators over multiple generations to maximize solution quality while controlling code complexity. The method is evaluated across standard VRPs (capacitated, time windows, prize-collecting), consistently discovering heuristics that outperform handcrafted and previous LLM/learning-based methods on large benchmark instances using only CPU resources.​

The approach offers interpretability, practical efficiency, and a reproducible pipeline for discovering and improving heuristics for combinatorial optimization, highlighting a new path for LLM-driven algorithmic design in operations research.​

The contributions include:
1. A hybrid metaheuristic framework (LLM-in-the-loop LNS) for VRPs where LLMs generate, mutate, and combine code for local operators.
2. A genetic algorithm with code-length penalties to evolve and select the best LLM-generated operators.
3. Demonstrating state-of-the-art or superior performance compared to both expert-designed heuristic solvers and recent neural/LLM solutions on several large VRP benchmarks, with superior interpretability and scalability

### Strengths
1. The proposed VRPAgent leverages LLMs to generate and evolve problem-specific destroy/insert heuristics for VRPs, significantly reducing the need for expert-written code and enabling discovery of novel strategies.​

2. The framework combines LLM-generated operators with a genetic algorithm using elitism, biased crossover, and code-length penalties, leading to efficient search and interpretable code that is competitive with and sometimes superior to handcrafted heuristics.​

3. VRPAgent consistently outperforms or matches best-in-class expert and neural approaches on benchmark VRPs (CVRP, VRPTW, PCVRP), working efficiently on CPU and scaling to large instances.​

4. The approach maintains strong performance across multiple VRP classes without requiring expensive hardware. By modularizing heuristic generation and search refinement, it allows for adaptation to different problem types and operator ensembles.​

5. By focusing LLM synthesis on manageable code components within a robust metaheuristic shell, VRPAgent strikes a balance between automated innovation and guarantees of feasibility and quality

### Weaknesses
1. Many LLM-generated heuristics discovered by VRPAgent are ensembles or recombinations of standard strategies from the literature (e.g., SISRs-like removal, weighted greedy criteria for sorting). The framework excels at combining known components but provides little evidence of discovering fundamentally new algorithms that would advance state-of-the-art theory for VRPs.​

2. While the code is readable to experts, it is often overly redundant, deeply nested, and filled with hard-to-tune random parameters and magic numbers. Several domain experts in the study noted that a human would write more succinct, interpretable, and maintainable code. The logic behind some probabilistic choices is especially convoluted, making ablation studies and performance analysis difficult.​

3. The performance improvements are attributed to complex ensembles and parameterized strategies, but the paper lacks detailed ablation studies pinpointing which components are truly responsible for gains. This makes it hard to generalize findings beyond the benchmarked VRP instances.​

4. Many key parameters affecting the algorithms' behavior are scattered, sometimes hard-coded and sometimes embedded within random logic, increasing risk of inadvertent misconfiguration. This could hinder code modification, adaptation, or debugging in practice.​

5. The experiments focus on classic VRPs and well-known benchmark formats. There is no evaluation of the heuristics' robustness under noisy, dynamic, or highly custom problem constraints, which are common in operational logistics scenarios.​

6. Although some level of interpretability is claimed, true transparency into how and why the LLM-generated code behaves well is lacking. For many users and practitioners, relying on black-box or stochastic mixtures of heuristics without clear guidance or analysis may be risky.​

7. The framework does not address risks inherent in LLM-generated code, such as silent propagation of bugs, accidental feasibility violations, or malicious prompt engineering in operational settings. This could be critical for industrial deployment.​

8. Claims of extensibility to other combinatorial domains (packing, scheduling) are made, but without any experimental or theoretical evidence.

### Questions
1. How sensitive is the genetic search to the initialization of random heuristics and the specific code-length penalty functions? Can you provide detailed ablation studies quantifying how different parameter choices impact final solution quality and code interpretability, particularly for non-benchmark VRP variants or logistics settings outside the training corpus?​​

2. Given that several domain experts found the final heuristic code verbose, redundant, or over-complicated, what mechanisms (besides code-length penalty) do you propose to systematically regularize the structure, improve succinctness, and enhance human interpretability for large LLM-generated operator populations?​​

3. What validation and error-checking processes are in place to guarantee that newly synthesized operators do not introduce infeasibility, silent bugs, or performance degradation, especially as code complexity increases over generations and as operators interact in ensembles? Is there any theoretical or empirical guarantee that VRPAgent will not produce brittle or unsafe solutions on realistic, highly constrained, or adversarial VRP instances?

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
Designing effective heuristics for VRP problems based on the Large Neighborhood Search (LNS) algorithm typically requires extensive human expertise and trial-and-error. To address this issue, the paper proposes using large language models (LLMs) to automatically design heuristic operators. Building on the concept of genetic algorithms, the LLM generates diverse heuristic candidates, retains the best-performing ones according to the solution results, and performs heuristic modifications and explorations to further improve performance. The proposed method is validated on multiple types of VRP problems, demonstrating a significant overall performance advantage compared with other AI-enhanced LNS approaches.

### Strengths
The work presented in this paper is solid and substantial, with comprehensive comparisons against many methods published in top-tier conferences, demonstrating strong overall performance.

The manuscript is well-written, logically organized, and carefully proofread.

The proposed method also shows promising potential for extension and application to other problem domains.

### Weaknesses
The proposed framework exhibits general applicability; however, the experimental cases are limited to the VRP domain.

The effectiveness of the proposed method still requires further investigation, as it has not been compared with widely used commercial solvers. Moreover, the results do not show a significant improvement in either computational speed or solution quality compared with existing approaches.

### Questions
1. In Fig. 2, why does the curve without mutation decrease faster in the early iterations, yet later perform worse than the one with mutation? It seems that 20 iterations are insufficient for convergence. It is recommended to extend the number of iterations to show a more complete convergence process.

2. The analysis of Fig. 5 (a) and (b) is inadequate. The discussion merely restates the numerical results without providing insight into the underlying reasons or the conclusions that can be drawn from them.

3. The paper claims that the proposed framework has strong transferability. Therefore, the results and implementation should be made open-source to enable further verification and application by other researchers across different problems and domains.

4. All test cases in the paper are limited to VRP-related problems, yet the method itself does not incorporate any VRP-specific structural design or analysis. Moreover, the generated heuristics are not analyzed, leaving the algorithm’s interpretability and physical rationale unclear.

5. As mentioned in Comment 4, since the algorithm is not specifically tailored to VRP, it is suggested to include additional results on other mixed-integer programming benchmarks (e.g., general MILP instances) in the appendix to demonstrate the generality and effectiveness of the proposed approach.

6. Although the paper compares with many deep learning–based methods, it does not include comparisons with classical solvers such as Gurobi or CPLEX, which are widely used in practice. Without such baselines, the practical applicability of the proposed method to real-world VRP problems remains unclear. It is recommended to supplement results comparing with Gurobi and/or CPLEX.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents VRPAGENT, a framework that uses Large Language Models (LLMs) to automatically discover heuristic operators for Vehicle Routing Problems (VRPs). The approach embeds LLM-generated problem-specific operators within a Large Neighborhood Search (LNS) metaheuristic and refines them through a genetic algorithm with elitism and biased crossover. The authors evaluate their method on three VRP variants (CVRP, VRPTW, PCVRP) and demonstrate state-of-the-art performance using only a single CPU core at test time.

### Strengths
1. Novel and Practical Framework: The approach of generating only problem-specific operators within a fixed metaheuristic is well-motivated. This "keeping AI agents on a leash" philosophy addresses key limitations of prior LLM-based approaches by ensuring correctness and manageability while still enabling discovery of novel strategies.
2. Strong Empirical Results: VRPAgent achieves impressive performance improvements over both traditional OR solvers and recent learning-based methods, with negative gaps (around -0.30%) relative to state-of-the-art SISRs on larger instances. The consistency across multiple problem variants and instance sizes is particularly compelling.
3. Computational Efficiency: The single CPU core requirement at test time is a significant practical advantage over GPU-dependent NCO methods, making deployment more accessible.
4. Thorough Experimental Analysis: The paper includes comprehensive ablation studies demonstrating the importance of biased crossover and mutation, analysis across different LLMs (showing that open-source gpt-oss achieves near-SOTA performance at low cost), and sensitivity analyses on GA hyperparameters.
5. Expert Analysis: The inclusion of expert evaluation of generated heuristics (Appendix C) provides valuable qualitative insights into readability, coherence, and novelty, adding credibility beyond pure performance metrics.

### Weaknesses
**Major issues**

1. Interpretability Concerns: The expert analysis consistently notes that discovered heuristics are difficult to interpret due to complex logic, nested conditionals, and convoluted use of random numbers. This limits practical adoption where transparency is important. The paper acknowledges this but doesn't propose concrete solutions.
2. Limited Generalization Analysis:
- Training is conducted only on 500-customer instances, yet the approach generalizes well to 1000 and 2000 customers. More analysis on why this generalization occurs would strengthen the paper.
- The operators are discovered separately for each problem variant. Can operators transfer across problems or be adapted more efficiently?
3. LLM Dependency:
- Best results require Gemini 2.5 Flash at ~$19 per run, which may limit accessibility
- While gpt-oss performs well, the reliance on specific LLM characteristics raises questions about reproducibility and long-term viability
4. GA Design Choices:
- The strong bias toward exploitation (80% elite in crossover, mutation only on elites) is unusual. While ablations show it works, more analysis on why exploitation is so beneficial in this search space would be valuable.
- Limited exploration of other GA hyperparameters (e.g., initial population diversity, selection mechanisms)
5. Comparison Limitations:
- Some baselines use different time budgets or hardware configurations, making direct comparison slightly less clear
- The paper compares against construction-based LLM methods (EoH, ReEvo) that don't benefit from search budgets, but limited comparison with other LLM-based improvement heuristics
6. Novelty of Discovered Heuristics: While the expert analysis confirms novelty, it also notes that heuristics are primarily "recombinations of existing ideas." The paper could better discuss what fundamentally new concepts (if any) were discovered.

**Minor Issues**
1. The abstract mentions "VRPAGENT is the first LLM-based paradigm to advance the state-of-the-art in VRPs," which is a strong claim. While the results support this, it might be beneficial to briefly acknowledge the ongoing rapid advancements in LLM-based optimization to provide full context, perhaps by rephrasing slightly to "among the first" or "a pioneering LLM-based paradigm."
2. Notation Consistency: In Algorithm 2, line 4 uses NE (non-elite) which could be confused with the elite size parameter also denoted $N_E$. Consider using different notation. In Algorithm 2, line 7 uses $RANDOM(Е)$ and line 8 uses $RANDOM(NE)$. It would be clearer to explicitly state what $E$ and $NE$ represent in this context (e.g., a list of elite individuals, a list of non-elite individuals) to avoid ambiguity for readers unfamiliar with the specific GA implementation.
3. Missing Details: The paper mentions that full prompts will be provided in the "final code release" but only shows CVRP-specific prompts in the appendix. For reproducibility, all prompts should be included.
4. Statistical Significance: Results lack error bars or significance tests, though the consistent improvements across problems suggest robustness.
5. Figure Quality: Figure 1 is informative but quite busy. Consider simplifying or providing a higher-level conceptual diagram first.

### Questions
1. Have you considered incorporating interpretability metrics into the fitness function to encourage more transparent heuristics without sacrificing performance?
2. Can you provide more insight into why strong exploitation (biased crossover, elite-only mutation) works so well? Is there something specific about the LLM-generated operator search space that makes this effective?
3. How sensitive is the approach to the choice of metaheuristic framework? Would similar results be achievable with other frameworks beyond LNS?
4. The expert analysis mentions ensemble approaches in all discovered heuristics. Is this a fundamental property of effective operators, or an artifact of the LLM's training or the prompt design?
5. Have you investigated whether operators discovered for one problem (e.g., CVRP) can be adapted or fine-tuned for related problems (e.g., VRPTW) more efficiently than starting from scratch?

### Soundness
3

### Presentation
3

### Contribution
3
