# Enhancing Reasoning in Large Language Models via Entropy-Aware Self-Evolution

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Large language models (LLMs) have exhibited remarkable reasoning capabilities. However, when self-evolution frameworks are employed to further enhance these models, a key challenge lies in balancing correctness, which ensures reliable supervision, and exploration, which promotes diverse reasoning trajectories. To address this dilemma, we propose an $\textbf{entropy-aware self-evolution framework}$ that integrates verifier feedback with both sequence-level and token-level entropy.  Our approach incorporates two key strategies: (i) $\textbf{high-entropy selection}$ of verified trajectories to provide informative yet reliable signals; and (ii) $\textbf{entropy-aware rethinking}$, which revisits uncertain reasoning steps to uncover alternative solutions. Theoretically, we establish the connection between entropy and the expected supervised fine-tuning loss, showing that high-entropy trajectories yield stronger learning signals. Empirically, experiments across multiple reasoning benchmarks demonstrate that our framework consistently improves both reliability and exploratory capacity over strong baselines. With the assistance of the proposed framework, InternLM2.5-1.8B achieves an improvement of $8.27$% and surpasses the strong baseline by $1.82$% on the GSM8K task, as measured by $Pass@16$.
Our results highlight entropy as a principled driver of self-improvement, enabling LLMs to evolve toward models that are not only more accurate but also more exploratory.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a self-evolution framework that generates multiple CoT candidates, regenerate (starting from) steps that has high-entropy, and selects the trajectory that is most uncertain (highest entropy). SFT training is used to train on the chosen trajectories. Experimental results show that performance on InternLM2.5-1.8B improves 1.44-5.52% on four benchmarks.

### Strengths
1. The proposed self-improvement framework has promising empirical results and notably, the fact that "training on high-entropy trajectories produces wider distribution, and hence leads to higher pass@K scores" is a good finding for the community.

2. Related to 1, the similarity score analysis shown in Figure 5 is insightful and could be adopted in the community.

### Weaknesses
1. The paper mainly compares with ENVISIONS framework to claim the superiority of the proposed method, yet there is a lack of explanation of what is the main difference between the proposed method and ENVISIONS framework. Also, there are many self-improvement frameworks in literature, yet, it is unclear what is the reason why the paper mainly compares only with ENVISIONS. For example, a very simple baseline would be STAR [1].

2. In Section 5.2, the ablation experiments of whether "Trajectory Rethinking" stage is really helpful is somewhat very narrow and limited. For example, when drawing the setting of "w/o trajectory rethinking" on Figure 4 (left), how would the trends differ? Given that it incurs more cost, how would it compare to simply generating more responses in the first stage (denoted as trajectory exploration)? Also, is the proposed filtering pipeline to identify high-entropy reasoning steps (Equation 4,5) the best choice compared to other filtering pipeline candidates?

3. For trajectory rethink stage, there should also be an ablation experiment of what is the best filtering criteria to choose responses. Is the approach from Equations 6 and 7 the optimal choice?

4. Might be less important, but I personally think that applying self-improvement on GSM8K, GSM-Hard, SVAMP, AsDiv is out-dated. The reason is that we already have a lot of post-training datasets that could enable LMs to achieve very good score on these datasets. The main motivation of self-improvement should be applying to highly competitive or challenging benchmarks that we do not have good training datasets to improve upon. Yet, I'll not take this point too heavily into account for my decision.

[1] Zelikman, E., Wu, Y., Mu, J. and Goodman, N., 2022. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35, pp.15476-15488.

### Questions
Could you provide qualitative examples of how the chain of thought changes throughout multiple iterations and compare to that of the ENVISIONS and other self-improvement frameworks/methods?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes an entropy-aware self-evolution framework for LLM reasoning. The method combines (i) high-entropy selection of verified trajectories and (ii) entropy-aware rethinking that truncates at high-uncertainty tokens and regenerates continuations. A short analysis links the expected SFT loss to sequence-level entropy (higher-entropy trajectories yield stronger gradients). Experiments on GSM8K, GSM-Hard, SVAMP, and AsDiv show consistent gains over base models and the ENVISIONS baseline, e.g., InternLM2.5-1.8B improves by +8.27% Pass@16 on GSM8K and by +4.39% Pass@128 over ENVISIONS

### Strengths
1. Clear, simple mechanism: Using sequence-level entropy to rank verified trajectories, plus token-level entropy to pick truncation points, is conceptually clean and easy to implement. 

2. Theoretical intuition is practical: The derivation \mathbb{E}[L_{\text{CE}}]=T\cdot \mathbb{E}[H_{\text{seq}}] formalizes why high-entropy samples can be more informative for SFT, aligning with the design. 

3. Ablative evidence on rethinking: The paper isolates the contribution of the rethinking stage and shows it supplies a substantial portion of training trajectories and improves Pass@K

### Weaknesses
1. All results use program-verifiable math with code-style rationales; it is unclear whether the approach transfers to semantic or open-ended reasoning, where verifiers are weaker/noisier. 

2. The framework samples K per input and iterates I times, but the paper does not report wall-clock, FLOPs, or cost per point of Pass@K improvement. Without budget-matched comparisons, the strength of the gains is harder to assess. 

3. Gains are strongest at larger K; single-shot (K=1) or small-budget accuracy—closer to user-facing inference—are under-emphasized. This makes it difficult to conclude that the method improves sample efficiency rather than just exploration under wide sampling.

4. Key knobs (α, β for truncation; K, N, I; temperature/top-p) could materially affect behavior. The paper lacks a thorough sensitivity analysis or guidelines, which limits reproducibility.

### Questions
See the weakness.

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
The paper introduces an entropy-aware self-evolution framework to enhance reasoning in LLMs by balancing correctness and exploration. It integrates three stages—trajectory exploration, entropy-based rethinking, and high-entropy trajectory selection—guided by external verifiers. The approach prioritizes verified yet diverse reasoning paths for self-training. Experiments on math reasoning benchmarks (GSM8K, GSM-Hard, SVAMP, AsDiv) using multiple LLM backbones show consistent gains in Pass@K accuracy and outperform the ENVISIONS baseline, indicating improved exploration without sacrificing correctness.

### Strengths
The paper presents a novel combination of entropy estimation and verifier-guided self-evolution to improve reasoning diversity in LLMs. The entropy-aware rethink mechanism, which truncates and regenerates reasoning at high-uncertainty tokens, is intuitive and reasonably original. The method demonstrates consistent improvements across multiple math reasoning datasets and small-scale backbones, suggesting effectiveness and reproducibility. The overall presentation is clear, with well-defined metrics, illustrative figures, and initial ablations that begin to analyze key components such as entropy-based selection and rethink. Furthermore, the work offers a principled perspective on the correctness–exploration trade-off, positioning entropy as a control signal for self-evolving reasoning models. Compared with prior frameworks such as ENVISIONS, it appears to achieve a better balance between accuracy and diversity within the math-verifier setting

### Weaknesses
1.Incomplete and partially confounded ablations. While the paper includes a “w/ vs w/o Trajectory Rethink” comparison and some analysis of entropy-aware selection, the component-level contribution remains unclear. In particular, there are no results for (i) verifier + random selection among correct trajectories, (ii) verifier + low-entropy (or medium-entropy) selection, or (iii) selection-only vs rethink-only under a matched compute/sample budget. Without these simpler but critical baselines, it is difficult to tell whether the gains primarily come from the specific high-entropy preference and rethink mechanism, or from having more diverse verified data in general.

### Questions
1.Component ablations. Figure 6 gives a useful qualitative analysis of the Trajectory Rethink stage (with vs without rethink). To more cleanly isolate the contribution of each component, could you add a compute-matched ablation table comparing: (i) exploration + entropy-based selection only, (ii) exploration + rethink only, (iii) the full method, and (iv) ENVISIONS? This would clarify how much gain comes from the rethink mechanism itself versus simply having more trajectories.
	
2.Entropy vs simpler selection baselines. The paper argues that preferring high-entropy trajectories is key to balancing correctness and exploration. Could you (i) clarify how your “entropy-free” or baseline selection behaves in practice (e.g., is it close to random over verified-correct trajectories?), and (ii) add a low-entropy selection baseline? Comparing high- vs low-entropy (and entropy-free) selection under the same verifier would help demonstrate that entropy itself, rather than just any subsampling of correct trajectories, is responsible for the gains.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an entropy-aware self-evolution framework that addresses the fundamental trade-off between correctness and exploration in iterative LLM training. The key insight is that high-entropy trajectories (those with verified correctness but intrinsic uncertainty) provide both reliable supervision and signal for exploring alternative reasoning paths. The framework employs two complementary strategies: high-entropy selection that prioritizes verified trajectories with high uncertainty to ensure informative yet dependable training signals, and entropy-aware rethinking that identifies high-uncertainty reasoning steps for truncation and regeneration to uncover diverse solutions. The authors establish theoretical connections between sequence-level entropy and expected supervised fine-tuning loss, demonstrating that high-entropy trajectories induce larger expected gradients and richer learning signals.

### Strengths
The paper presents a well-motivated and principled approach to a genuine problem in self-evolution—balancing correctness with exploration. The key insight that high-entropy verified trajectories can serve dual purposes (reliable supervision and exploration signals) is intuitive and well-articulated. The theoretical contribution connecting entropy to expected supervised loss provides solid grounding for the approach. The experimental evaluation is reasonably comprehensive, testing across multiple model sizes and benchmarks with consistent improvements over ENVISIONS baseline. The analysis sections provide valuable insights into why the method works, including visualizations of similarity scores, linguistic patterns in high-entropy tokens, and ablation studies. The framework is clearly described with well-designed figures, and the writing is generally accessible.

### Weaknesses
The experiments are limited to relatively small models (1.8B parameters maximum) due to computational constraints, raising questions about scalability and whether the findings generalize to larger LLMs. The paper compares against only one primary baseline (ENVISIONS), making it difficult to assess whether improvements stem from the entropy-aware design specifically or from implementation details. The computational overhead is not systematically analyzed—it's unclear whether the gains justify the additional verification and computation costs. Additionally, while trajectory rethinking is empirically validated, the theoretical justification for why truncating at high-entropy tokens specifically should improve diversity is limited.

### Questions
What is the actual computational cost and wall-clock time overhead compared to ENVISIONS, and how does this scale with dataset size? Can the framework work with differentiable or neural verifiers rather than only executable code verification? Why should entropy specifically be the signal for exploration potential rather than other measures of uncertainty or diversity? The theoretical analysis shows entropy relates to expected loss magnitude but doesn't directly justify why this improves exploration, is there a more principled connection? How does the method perform with different sampling temperatures, and could the improvements partly be explained by implicit temperature regularization?

### Soundness
3

### Presentation
3

### Contribution
2
