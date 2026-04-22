# C-Evolve: Consensus-based Evolution for Prompt Groups

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Prompt evolution algorithms offer a powerful paradigm for enhancing AI systems based on closed-source models, while few work explores whether aggregating results from multiple prompts to reach a  consensus can further advance the system capability boundary. In this paper, we introduce Consensus-Evolve (C-Evolve), an evolutionary algorithm that discovers a group of prompts whose aggregated outputs after majority voting achieve optimal performance. More specifically, C-Evolve employs an island-based evolutionary algorithm to maintain population diversity, and prompts from distinct islands are selected to form groups to aggregate their outputs. The key difference from single individual evolution is a voting score, which evaluates each individual prompt's contribution within groups. We take this as the fitness score for evolution instead of individual performance. Consequently, C-Evolve is more likely to produce and maintain prompts with higher potential to form a high-performing group and eliminate low-performing ones, gradually improving the group performance after reaching consensus. Our method achieves state-of-the-art performance across a wide range of tasks, including both open-ended tasks like HotpotQA and closed-ended tasks like MATH. On Qwen3-8B, C-Evolve achieves 70.67\% on HotpotQA and 43.88\% on IFBench, which are 4.95\% and 2.73\% higher than GEPA, respectively. For GPT-4.1-mini, the accuracy on IFBench is further improved to 47.96\% and reaches 95.33\% in the MATH benchmark. These results demonstrate the C-Evolve's competitive performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose C-Evolve, a consensus-based evolutionary algorithm for optimizing groups of prompts rather than a single prompt in complex task environments. The method introduces a two-stage evolutionary process (warm-up stage and voting stage) within an island-based evolutionary framework. In the warm-up stage, individual prompts are optimized for standalone performance, while in the voting stage, prompts are evaluated based on their contribution to group consensus through majority voting. The approach demonstrates strong empirical performance across multiple benchmarks (IFBench, HoVer, MATH) and models (Qwen3-8B, GPT-4.1-mini), particularly for complex tasks where single prompts struggle to capture multi-faceted requirements.

### Strengths
1. The paper makes a valuable contribution by shifting focus from optimizing single prompts to optimizing groups of cooperative prompts, which addresses a genuine limitation in complex task scenarios.
2. The warm-up stage followed by voting stage provides a principled approach to balance individual prompt quality and group collaboration effectiveness, with empirical evidence showing their complementary benefits.
3. The paper demonstrates consistent improvements over baselines across multiple tasks and models, with particularly notable gains on complex tasks.

### Weaknesses
1. The paper claims that "single optimal prompt often suffers from inherent expressive limitations" but lacks rigorous theoretical or empirical evidence. While Table 8 shows consensus improves accuracy (40% -> 49.33% when two prompts agree), this is insufficient to justify the core premise. The authors should provide more systematic error analysis showing complementary failure patterns across different prompts.
2. Despite Table 9 showing warm-up stage contributes +1.02% absolute accuracy (41.83% -> 42.85%), the paper completely overlooks this stage in the abstract, introduction, and conclusion. Figure 7(a) further demonstrates its importance, yet the paper fails to explain why this initial optimization phase is necessary.
3. The paper extensively compares with AlphaEvolve in experiments but fails to discuss their fundamental differences in methodology sections. The key distinction should be explicitly stated in Section 1/2/4 rather than merely implied through experimental results.
4. The paper claims the island model "maintains population diversity" but doesn't adequately justify why this approach is superior to simpler alternatives. In warm-up stage, all islands follow identical evolutionary processes (Algorithm 2), raising the question of why parallel islands are better than sequentially running warm-up stage multiple times. Figure 6 shows different islands develop different characteristics, but there's no ablation showing the contribution of the 10% migration rate.
5. The paper lacks systematic analysis of key parameters that likely impact performance. Island count |G|=3 is used without justification (Figure 7(b) shows EMA α sensitivity but not |G| sensitivity). Population capacity N_max varies across islands (Table 3) but no analysis of its impact. Groups per round n_c affects voting score calculation but no sensitivity analysis is provided.
6. All results appear based on single runs (no standard deviation reported), which is problematic for stochastic evolutionary algorithms. Evolver LLM details are sparse (Appendix H shows examples but not full implementation). Table 7 shows time per iteration but lacks scaling analysis to assess computational efficiency.
7. The related work section misses several relevant recent papers including "Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective" and "When large language models meet evolutionary algorithms: Potential enhancements and challenges". The discussion of AlphaEvolve is particularly superficial given its prominence in the experiments.

### Questions
1. Can you explain why single prompts have "inherent expressive limitations" in complex tasks? Is this limitation fundamental to the prompt format or specific to current LLM capabilities?
2. Why does the warm-up stage contribute positively but modestly compared to the voting stage? What specific types of tasks benefit most from warm-up?
3. How does the 10% migration rate between islands specifically contribute to performance? Could you provide an ablation study varying this parameter?
4. In Algorithm 2, what exactly do "Categorical" and "RandomSample" operations do? Please provide definitions to clarify these critical implementation details.
5. Could you provide wall-clock time comparisons showing how runtime scales with |G| and N_max? What is the computational overhead of maintaining multiple islands compared to a single large population?

Should the authors address these concerns in their response, I would be willing to increase my recommendation to a stronger accept.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes C-Evolve, an evolutionary framework that does not search for a single best prompt, but instead evolves a group of prompts whose predictions are aggregated via consensus voting. The key technical idea is to use a voting score that reflects each prompt’s contribution to consensus performance across dynamically sampled groups, smoothed by an EMA update to stabilize fitness estimation. An island-based evolution design maintains diversity among prompts, encouraging complementary behavior rather than converging to the same style.

The method is tested across open-ended tasks like HotpotQA and IFBench and closed-ended tasks including MATH, HoVer, and GPQA. On Qwen3-8B, performance improvements reach +13.85% relative to baseline (e.g., HotpotQA 70.67%) and similar gains apply to GPT-4.1-mini (e.g., MATH 95.33%). Ablations analyze consensus strategies and the necessity of EMA scoring.

### Strengths
1.The shift from individual optimality to cooperative optimality is a timely and relevant topic.

2.Demonstrates consistent performance improvements on several benchmarks (e.g., HotpotQA 70.67%, MATH 95.33%).

3.Attempts to investigate diversity via multi-island evolution and t-SNE visualization.

4.Some ablations are meaningful (e.g., EMA vs. averaging scoring).

### Weaknesses
1.Using ensemble voting as a fitness signal has been explored in mixture-of-agents literature and existing self-consistency / committee prompting frameworks. The authors do not clarify what is fundamentally new beyond applying majority vote during evolution.

2.The metric set is tiny (only 300 samples per task). Evolution overfits exactly the data used to determine fitness. There is no indication that the final group generalizes beyond the metric distribution.

3.The islands may be stylistic variations of the same base prompt. There’s no quantitative diversity measure—no entropy, no semantic edit distance, no response disagreement statistics.

4.a). Majority vote may amplify systemic bias when mistakes correlate.
   b). The LLM-selection aggregator can hallucinate entirely new answers.
   c). No error-type breakdown or trust-calibration analysis.

5.Nearly all “operators” are outsourced to the evolver LLM. How is behavior shaped beyond “please improve”? This makes reproducibility dubious despite claims otherwise.

6. Reporting AlphaEvolve’s grouped performance only after evolution not designed for groups is a questionable comparison. A fair baseline would train AlphaEvolve with a consensus-aware objective.

7. Iteration×islands×groups evaluation produces substantial additional overhead. A ~2–5% gain over an existing strong baseline should be examined against the cost of 3× more calls.

8. Table 3 improvements on Level-5 MATH are marginal (+1–2%). Very small sample sizes lead to statistically weak conclusions.

### Questions
1. How does this differ fundamentally or committee-based prompt ensembling techniques used in self-consistency? Can you prove that evolving cooperative behavior is better than ensembling individually strong prompts?

2. What happens when you scale the metric dataset ×10? Do the improvements persist or vanish once generalization pressure increases?

3. Can you provide diversity statistics (pairwise prompt edit distance, output disagreement entropy)? Otherwise, the “complementarity” narrative feels speculative.

4. Does the EMA voting score unfairly reward overly conservative prompts that avoid taking risks? What safeguards ensure prompts do not converge to generic evasive behavior?

5.Did you try evolving only one island but selecting the top-k prompts for consensus? Would that match your results with far lower complexity?

6. What is the robustness of the LLM aggregator? Any adversarial test where a minority prompt is correct but consensus fails?

7. How sensitive are results to α? α=0.5 and 0.8 are not a sufficient exploration of learning dynamics—did α>0.9 destabilize selection?

8. Why is the warm-up stage needed at all? Could the system evolve directly under consensus supervision?

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
The paper introduces a methodology to evolve groups of prompts optimized for consensus-based prediction. It uses island-based evolution to maintain diversity and introduces a voting score which evaluates each prompt's contribution within groups as a fitness function. The strategy achieves state-of-the-art performance across different tasks like HotpotQA and MATH.

### Strengths
- The paper introduces a novel approach for prompt optimization to maintain diversity, which is well motivated by the ensemble learning principles
- Experiments cover different tasks
- The problem is well defined and paper well written

### Weaknesses
- I am not entirely clear about the individual contribution, redundancy, and complementarity between the islands, especially from table 3 where the performance are rather close. It makes it rather hard to evaluate the soundness of the underlying decision making and tradeoffs for this optimization pipeline. 

- There is a significant computational aspect to such prompt optimization methodology, and the optimization pipeline is rather complex / slightly convoluted overall with hardcoded parameters. When would such computational cost be justified?

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
2
