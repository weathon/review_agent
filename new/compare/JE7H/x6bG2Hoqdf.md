---
job_id: 26567ccc-a8b4-48df-8b03-883c9f55bfcb
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: x6bG2Hoqdf.pdf
paper: CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is squarely on LLM-based automatic heuristic design, reinforcement learning fine-tuning for code/algorithm generation, and optimization problems, all well within ICLR’s ML and representation-learning scope.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results/Discussion, Conclusion) are present and reasonably complete. The method and experiments are non‑trivial, technically sensible, and empirically validated against strong baselines. I see no fatal methodological, theoretical, or evaluation flaws that would justify a desk reject.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts to manipulate automated reviewing systems or hidden prompts; the use of LLMs is only discussed as part of the method.

---

# Expected Review Outcome:

## Summary

The paper proposes CALM, a framework for automatic heuristic design that couples evolutionary prompt-based search (“verbal gradients”) with on-the-fly RL fine-tuning of the underlying LLM via GRPO (“numerical gradients”). CALM introduces fine-grained evolutionary operators (injection, replacement, diversity-aware crossover, simplification) and a probabilistic collapse mechanism to manage stagnation, along with a reward function that compares each generated heuristic to its parents. Using a locally hosted, quantized Qwen2.5‑7B model, CALM discovers heuristics that outperform or match state-of-the-art LLM-based AHD methods, including API-based systems with stronger models, across several combinatorial optimization tasks (OBP, TSP, CVRP, OP).

## Strengths

1. **Co-evolution of LLM and heuristics is clearly instantiated and empirically useful.**  
   The core idea of using the evolutionary loop not just to mutate prompts but also to generate reward-labeled prompt–response data for RL fine-tuning is well executed. The ablations in **Table 4** (“w/o GRPO” vs “with GRPO”) show sizable performance drops without RL on both OBP and OP, and the training curves in **Figure 2** (both CVRP and OP) illustrate that CALM initially lags GPT-4o-mini–based baselines but overtakes them as GRPO training progresses. This is a convincing demonstration that numerical gradients meaningfully improve the generator for AHD beyond verbal prompt engineering alone.

2. **Thoughtful design of fine-granularity operators that interact sensibly with RL.**  
   The injection and replacement operators (Section 4.1) are explicitly designed to introduce small, interpretable code changes rather than entirely new heuristics, which aligns well with GRPO’s token-wise advantage assignment. The examples in **Figures 10–13** make this concrete: e.g., Figure 10’s injection of a distance-decay term into an OP heuristic yields a 4.21% improvement, and Figure 13’s replacement of a static threshold with an instance-dependent rule for OBP yields a 7.79% improvement. These examples help substantiate the claim that the operators encourage meaningful, localized structure changes that RL can learn from.

3. **Diversity-aware crossover plus collapse is a nontrivial algorithmic contribution.**  
   The diversity-based parent selection in Section 4.1, combined with the stagnation-triggered collapse in Section 4.2, is a coherent story for combating premature convergence. Empirically, the ablation rows “w/o diversity” and “w/o crossover” in **Table 4** show degradation relative to the full system, and the collapse hyperparameter sweeps (different \(\delta_0, C\)) both improve over “w/o collapse” in most settings, while illustrating failure modes when collapse is too aggressive. This supports the claim that these mechanisms help exploration in the heuristic space.

4. **Reward design is carefully argued and experimentally supported.**  
   The reward function in **Equation (4)**, which depends on \(\Delta(h_{\text{new}}, h_{\text{t\_base}})\) from **Equation (3)**, attempts to factor out prompt bias by comparing a child heuristic to the best base heuristic in the prompt rather than to a global baseline. The authors test two alternative schemes (“performance” and \(\{0.5r_{\mathrm{invalid}},1\}\)) in **Table 4** and show that both underperform, in some cases even doing worse than no-RL. This is decent evidence that the somewhat elaborate reward shaping is doing useful work.

5. **Experimental coverage and baselines are strong and mostly fair.**  
   The paper evaluates on four reasonably challenging settings: OBP, step-by-step TSP, CVRP (ACO), and OP (ACO), using standard or clearly described data generation setups. The comparison set is extensive: classic heuristics, neural combinatorial optimization (POMO, DeepACO), and multiple recent LLM-based AHD systems (FunSearch, EoH, ReEvo, HSEvo, OpenEvolve, MCTS-AHD, EvoTune). In **Tables 1–3**, CALM with local Qwen2.5-7B-INT4 plus GRPO is competitive with, and often better than, GPT-4o-mini baselines, which is an impressive resource-efficiency point.

6. **Clarity of the high-level pipeline and prompts.**  
   **Figure 1** effectively contrasts existing LLM-based AHD methods (left) with CALM (right): the additional GRPO loop with rewards, and the collapse mechanism, are clearly annotated, making it easy to understand where CALM diverges from prior work. The prompt templates in **Figures 3–9** show that the authors have thought carefully about constraining LLM output formats and enforcing determinism, which matters a lot in this domain.

7. **Additional analyses and diagnostics in the appendix are unusually thorough.**  
   The appendix contains meaningful extras: time breakdowns (Table 6), snapshot analyses of fine-tuned models (Tables 7–8), scaling experiments (Table 12), TSPLib results (Table 17), and impact of GRPO vs DPO (Table 18). These provide a rich picture of how CALM behaves, and in particular, the snapshot study shows that fine-tuned models gain both feasibility and average performance, while also exhibiting interesting instability (“genius verging on madness”), which is an insightful observation.

## Weaknesses

1. **Conceptual novelty relative to concurrent AHD-with-fine-tuning work is underplayed.**  
   Section 2 briefly mentions concurrent works that apply preference-based methods like DPO to AHD, but the differentiation is somewhat thin and mostly “we use GRPO instead of DPO”. The co-evolution idea of jointly adapting prompts and heuristics is not entirely unique in this space, and recent works like experience-guided reflective co-evolution (see missing related work) and uncertainty-based evolution are quite close in spirit. The paper would benefit from a more precise articulation of what is really new beyond “we picked GRPO and designed some operators”, and why that is materially different from other co-evolutionary or RL-fineturning AHD frameworks.

2. **Reward formulation has some questionable choices and underspecified edge cases.**  
   - In **Equation (3)**,  
     \[
       \Delta(h_{\text{new}},h_{\text{t\_base}})=\mathrm{clip}\left(\frac{|g(h_{\text{new}})-g(h_{\text{t\_base}})|}{\min\{|g(h_{\text{new}})|,|g(h_{\text{t\_base}})|\}},0,1\right)
     \]  
     raises several concerns:  
     (i) When one of the \(g\) values is very close to zero, the denominator \(\min\{|g(h_{\text{new}})|,|g(h_{\text{t\_base}})|\}\) can be tiny, so the unconstrained ratio can be huge, after which clipping to 1 kills any ordering information beyond “difference is non-negligible”. This essentially collapses all “large enough” improvements into the same \(\Delta=1\) bucket.  
     (ii) If both performances are near zero but on opposite signs, the ratio becomes unstable; the text does not discuss numerical handling or why this normalization is preferable to, say, a bounded sigmoid of the raw difference.  
     (iii) Using absolute difference in the numerator means that the magnitude of degradation and improvement are treated symmetrically at that stage, only separated later by the cases in **Equation (4)**. A more direct signed normalization would arguably give GRPO a more informative scale.  
   - The duplicate-heuristic detection in **Equation (4)** is based on equality of \(g(h)\), which in practice is a floating-point aggregate over multiple instances. The paper does not explain whether it uses a tolerance threshold or exact equality, which is brittle and could misclassify distinct heuristics as duplicates (or vice versa).  
   - Finally, the reward range is somewhat arbitrary (\(r_{\mathrm{invalid}}\in(-1,0)\), positives in \([1,2]\)), and while Table 14 shows some robustness, there is no principled justification for these scales. For a paper positioning itself partly on RL design, more discussion of why this shaping should work with GRPO’s group-normalized advantages would help.

3. **GRPO integration is still described at a fairly high level; no clear stability or variance analysis.**  
   Section 3.2 restates the GRPO objective, but the practical integration details are sparse. For instance:
   - How many GRPO epochs or gradient updates per batch of \(G\) responses are used? Is each batch used exactly once?  
   - What are the exact values of \(\varepsilon\) and \(\beta\) in **Equation (1)**, and how sensitive is performance to them?  
   - The authors mention in Appendix H.1 that the learning rate is increased to \(5\times10^{-5}\), but there is no discussion of why this is safe given the relatively small reward datasets and the potential for mode collapse or overfitting to the training instances.  
   The snapshot experiment (Tables 7–8) hints at oscillatory behavior after 200 steps, but the main text does not really analyze the stability of RL training or whether catastrophic forgetting of general coding skills occurs.

4. **The collapse mechanism, while clever, is only superficially theoretically justified.**  
   **Equation (2)** gives an approximation \(\mathbb{E}[c_n \mid \text{collapse}] \approx \sqrt{\pi/(2\delta_0)}\), derived in Appendix G under the continuous approximation to \(\prod_k(1-k\delta_0)\). This derivation essentially assumes that collapse is triggered solely by the probabilistic rule \(p_k=k\delta_0\), ignoring the hard cap \(C\) and the fact that \(c_n\) is conditional on no breakthroughs. In practice, heuristic breakthroughs reset \(c_n\), so the distribution of \(c_n\) is heavily problem-dependent, and the formula’s practical usefulness for hyperparameter selection is unclear. Additionally, the condition \(C>1/\delta_0\) is required for the approximation, but in experiments the authors do use finite \(C\) smaller than \(1/\delta_0\) (e.g., \(\delta_0=0.0005, C=15\)), where the approximation formally does not apply. This undercuts the claim that Equation (2) “aids hyperparameter selection.”

5. **Experimental design mixes query budget, evaluation budget, and wall-clock in a way that makes true efficiency hard to assess.**  
   The main text repeatedly emphasizes “fixed 2,000 query budget” vs. baselines using larger budgets, but the mapping between queries, sampled responses \(G\), and evaluated heuristics is not entirely transparent. For CALM with \(G=4\) (Appendix H.1), each round produces 4 responses, some of which are invalid and thus not evaluated; baselines like MCTS-AHD operate with 2,000 heuristic evaluations. It is therefore difficult to tell whether CALM’s improvements are due to better search or simply a higher effective number of candidate heuristics explored per unit of evaluation. **Table 6** shows that LLM inference dominates time (70–80%), but there is no direct comparison of wall-clock for CALM versus MCTS-AHD in a matched hardware/LLM setting. The strength of the “resource-efficient” claim is therefore somewhat weaker than it is presented.

6. **Some claims on generalization and scalability are only partially supported.**  
   The paper suggests that CALM “naturally generalizes to new scales” (Section 2), and points to performance on out-of-domain instance sizes in **Tables 1–3**. While it is true that the discovered heuristics are tested at larger \(N\) or different capacities, this is still within synthetic distributions quite close to training. There is very limited evaluation on truly different distributions or real-world-style data. The TSPLib results in Table 17 help somewhat, but that table is buried in the appendix and uses only a single best heuristic. There is no assessment of how sensitive CALM is to the choice of seed heuristic, beyond a qualitative discussion in Appendix I.8.

7. **Some math/notation and implementation details are messy or incomplete.**  
   - In **Equation (1)**, the expectation is written as \(\mathbb{E}_{[q\sim\mathcal{Q},\{o_{i}\}\sim\pi_{\theta_{\mathrm{old}}}]}\) but the nested sums are over tokens \(t\), and the KL term \(\mathbb{D}_{\mathrm{KL}}[\pi_\theta||\pi_{\mathrm{ref}}]\) is treated as a scalar regularizer inside the per-token loss. It would be clearer to explicitly state whether the KL is averaged over tokens, over prompts, or estimated once per batch; currently this is only vaguely attributed to Schulman (2020).  
   - **Algorithm 1** is hard to parse in its current typesetting. There are duplicated comments about the collapse condition, and some lines appear corrupted (e.g., multiple repeated `/* If random(0,1) ... */`). This makes it difficult to be fully confident that the pseudo-code matches the implementation.  
   - The OBP seed heuristic code in Appendix H.4 appears to have a bug in `score[1:] -= score[:-1]` (which is inconsistent with the later OBP heuristic where `score[1:] -= score[-1]`). While this may just be a artifact of copying from previous work, more care in presenting such details would help.

8. **Limited discussion of failure cases and qualitative behavior of bad heuristics.**  
   The qualitative examples (**Figures 10–13**, and the generated heuristics in Appendix I.12) focus on successful improvements. It would be informative to see what kinds of pathological heuristics GRPO tends to overemphasize or whether there are modes where the LLM starts generating unreadable or non-deterministic code despite the prompts. Some hints of this instability appear in Tables 7–8 (feasibility ratio dips), but there is no deeper analysis in the main text.

## Potentially Missing Related Work

1. **Y. Liu, J. Li, W. X. Zhao, “Experience-Guided Reflective Co-Evolution of Prompts and Heuristics for Automatic Algorithm Design,” 2025.**  
   This work also considers co-evolution of prompts and heuristics in AHD. It is directly relevant to CALM’s claim of being among the first to jointly optimize prompt generation and an evolving algorithmic pool. It should be cited and compared in Section 2 (LLM-based AHD) and in the discussion of CALM’s “co-evolution” framing, clarifying differences in how reflection and co-evolution are handled vs. GRPO-based numerical gradients.

2. **Z. Chen, Z. Zhou, Y. Lu, “UBER: Uncertainty-Based Evolution with Large Language Models for Automatic Heuristic Design,” 2024.**  
   UBER proposes uncertainty-aware evolution strategies to better balance exploration and exploitation when evolving heuristics with LLMs. This is directly relevant to CALM’s diversity-aware crossover and collapse design. It should be discussed in Section 2 and cited near the description of the diversity-based parent selection and collapse mechanism (Sections 4.1–4.2), possibly as an alternative way to guide exploration.

3. **C. Chacón Sartori, C. Blum, “irace-evo: Automatic Algorithm Configuration Extended With LLM-Based Code Evolution,” 2025.**  
   irace-evo extends traditional automatic algorithm configuration by incorporating LLM-based code evolution. This bridges classic AHD/algorithm configuration with LLM-generated heuristics and is highly relevant to CALM’s broader positioning at the intersection of EC and LLMs. It should be discussed in the Related Work section when contrasting CALM with GP-based and configuration-based AHD, and briefly compared in the introduction to clarify where CALM stands relative to such hybrid approaches.

## Questions

1. **Reward normalization and numerical stability.**  
   - How do you handle the case where \(\min\{|g(h_{\text{new}})|,|g(h_{\text{t\_base}})|\}\) in **Equation (3)** is extremely small or zero? Is there an explicit \(\epsilon\) added in implementation, or do you rely solely on clipping to 1?  
   - Did you try alternative forms like \(\Delta = \tanh(\gamma (g(h_{\text{new}})-g(h_{\text{t\_base}})))\) or a logistic scaling? Any empirical comparison would help justify the particular normalization chosen.

2. **Duplicate detection and reward shaping.**  
   How exactly do you test for “duplicate heuristics” in the first case of **Equation (4)**? Do you compare performance \(g(h)\) within a tolerance, compare code textually (e.g., hash), or something else? This matters because in noisy evaluation environments heuristics with slightly different scores might be conceptually identical.

3. **GRPO hyperparameters and sensitivity.**  
   Could you provide the exact values of \(\varepsilon\) and \(\beta\) in the GRPO objective, and any observed sensitivity? In particular, how does increasing/decreasing \(\beta\) affect the trade-off between staying close to the base model and discovering more aggressive heuristic modifications?

4. **True compute fairness vs. MCTS-AHD and EvoTune.**  
   For one representative task (say OP or CVRP), can you provide a table with matched-runtime comparisons between CALM and MCTS-AHD / EvoTune, using the same hardware and LLM where possible, and reporting wall-clock time and number of *evaluated* heuristics? That would clarify whether CALM’s advantages persist when accounting for the cost of GRPO updates.

5. **Generalization beyond synthetic benchmarks.**  
   Have you tried applying any discovered heuristic to a genuinely different application domain, e.g., a real-world routing dataset or an industrial OBP variant with additional constraints? Even a brief anecdotal result would help calibrate how brittle or robust these heuristics are.

6. **Failure analysis.**  
   Could you provide examples of systematic failure modes when GRPO overfits or destabilizes the LLM (e.g., cases where feasibility ratio collapses or the LLM starts generating degenerate heuristics)? It would be especially useful to know whether the collapse mechanism helps mitigate such RL-induced pathologies.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The method is well engineered and empirically supported across several tasks, and the main claims are largely backed by results. Some aspects of the reward design, GRPO stability, and theoretical justification of the collapse mechanism are underexplored, but not fatally flawed.

## Presentation Rating

3: good.  
The paper is generally well written, with clear figures (especially **Figure 1** and **Figure 2**) and extensive appendices. However, some equations and Algorithm 1 are messy or underspecified, and the related work positioning against very close concurrent methods could be sharper.

## Contribution Rating

3: good.  
The work meaningfully advances LLM-based AHD by integrating RL-based fine-tuning with evolutionary operators and demonstrating nontrivial empirical gains with a small local model. Novelty is moderate rather than dramatic, but the combination and careful engineering are valuable to the community.

## Overall Rating

6: marginally above the acceptance threshold. But would not mind if paper is rejected.  
CALM is a solid and timely contribution that pushes LLM-based automatic heuristic design toward genuine co-evolution of models and algorithms, with convincing empirical evidence on multiple tasks and careful ablations. At the same time, aspects of the mathematical design (reward normalization, collapse analysis) and experimental fairness could be better justified, and the differentiation from very closely related concurrent work should be clarified. On balance, strengths slightly outweigh weaknesses, and I lean to a positive recommendation.

## Reviewer Confidence

4: confident.  
I am familiar with LLM-based code generation, RLHF-style fine-tuning, and algorithm discovery work, and I carefully checked the main equations and experiments, though I did not attempt to reimplement the method.