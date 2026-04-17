---
job_id: 92c4873f-320a-4027-9bc6-61d5cd899d1d
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: VjGU55hEwV.pdf
paper: RLIE: Rule Generation with Logistic Regression, Iterative Refinement, and Evaluation for Large Language Models
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length  
Pass ✅.

## Topic Compatibility  
Pass ✅. The paper is clearly about neurosymbolic / hybrid AI, probabilistic methods, and LLM-based rule learning, which fall well within ICLR’s scope.

## Minimum Quality  
Pass ✅. The paper includes Abstract, Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion, Ethics, and Reproducibility. The method is technically coherent, experiments are present on six datasets, and there are no obvious fatal flaws such as test leakage or non-English content.

## Prompt Injection and Hidden Manipulation Detection  
Pass ✅. I do not see any explicit or hidden instructions targeted at automated reviewers or other forms of prompt injection / manipulation.

---

# Expected Review Outcome:

## Summary

The paper proposes RLIE, a framework that combines LLM-generated natural language rules with a probabilistic combiner, specifically elastic-net logistic regression, and an error-driven iterative refinement loop. An LLM is used to generate candidate rules and to judge their applicability to each instance (with ternary outputs \{-1,0,+1\}), after which a logistic regression is trained over rule activations to perform global weighting and selection; hard examples under this model are then used to prompt new or refined rules. The authors evaluate both direct inference using the linear model and several strategies that inject the learned rules, weights, and linear predictions back into an LLM on six HypoBench tasks, finding that the simple linear combiner consistently outperforms LLM-based inference that consumes the rules.

## Strengths

1. **Clear and well-structured hybrid framework.**  
   The overall pipeline in **Figure 1** is well thought out and easy to follow: a) LLM-based rule generation with coverage filtering, b) logistic regression over ternary rule judgments, c) iterative refinement using hard examples, and d) evaluation via multiple inference strategies. The separation between “local judgment” by the LLM and “global aggregation” via a transparent probabilistic model is conceptually clean and aligns with the stated design philosophy.

2. **Interesting empirical finding about LLMs’ use of probabilistic rule information.**  
   The most compelling result is the systematic comparison of inference strategies in **Table 2**, showing that the simple *Linear-only* strategy (E1) consistently outperforms LLM-based strategies that are given the same rules, plus weights, plus even the linear model’s own prediction (E2–E4). This gives concrete evidence that current LLMs are poor at faithfully internalizing and applying explicit probabilistic rule weights, and can even overrule correct calibrated predictions. This is an informative negative result for the community.

3. **Solid performance across multiple tasks relative to LLM-based rule learning baselines.**  
   On six HypoBench tasks, **Table 1** shows that RLIE (E1 Linear-only) is consistently among the top methods in Accuracy and Macro-F1, and often the best when using the same backbone (DeepSeek-V3) as IO Refinement and HypoGeniC. RLIE’s performance is notably strong on Dreadit, LLM Detect, and especially Headline (67.0 F1) compared to other generalizable methods, indicating that the framework is competitive despite its simplicity.

4. **Iterative refinement mechanism is intuitive and reasonably justified.**  
   The iterative refinement procedure in Section 3.3, which identifies hard examples via prediction error $d_i = |\hat p_i^{(t)} - y_i|$ and uses these to generate new rules, is a natural way to leverage LLM generation in a targeted fashion. The case study in **Table 3** for the Retweets task qualitatively illustrates how rules evolve from generic emotional-language patterns to more nuanced patterns about personal voice and conversational tone, accompanied by a monotonic improvement in training F1 (0.625 → 0.699).

5. **Interpretability and compactness of rules.**  
   RLIE maintains a small rule set (capacity $H=10$) with elastic-net regularization for sparsity. The qualitative examples in **Table 3** show that the learned rules are human-readable and their associated weights convey intuitive importance, which is valuable for explainability and knowledge discovery compared to black-box LLM prompting or LoRA finetuning.

6. **Careful comparison of ways to use rules.**  
   The “hierarchical” evaluation of E1–E4 strategies in Section 3.4 and Section 5.2 is a useful contribution on its own, even aside from the specific RLIE construction. The prompts in **Figures 5–8** make it clear that the authors genuinely attempt to give the LLM meaningful access to rules, weights, and suggestions, rather than setting up a straw-man comparison.

## Weaknesses

1. **Conceptual novelty is moderate; method is largely a straightforward combination of known pieces.**  
   At a high level, RLIE is “LLM-generated natural language rules + LLM-based ternary rule judgments + elastic-net logistic regression + hard-example bootstrapping”. Logistic regression on rule activations is very standard (Section 2.1 cites Ruczinski et al., 2003; Friedman & Popescu, 2008), and the use of L1/L2 for rule selection is textbook. The iterative refinement on hard examples is essentially a curriculum-driven or boosting-like loop. The main conceptual novelty is applying this in the context of natural-language rules judged by LLMs and *explicitly* comparing linear-only vs. LLM-augmented inference. That is interesting, but not a large step beyond existing hypothesis-generation frameworks such as HypoGeniC or IO Refinement. The paper would benefit from a clearer articulation of what *fundamentally new* representation- or learning-level insight RLIE adds beyond “we plugged LLM rules into a logistic regression and it works better than feeding rules back to an LLM”.

2. **Limited and somewhat unbalanced experimental design; backbone fairness is not fully convincing.**  
   In **Table 1**, the baselines (Zero-shot, Few-shot, Zero-shot Gen, IO Refinement, HypoGeniC) are all run with DeepSeek-V3 as the backbone. RLIE is reported with three different backbones: Qwen3-Next-80B, Qwen3-235B, and DeepSeek-V3. The strongest RLIE numbers (e.g., DeepSeek-V3, 82.3 F1 on Dreadit, 90.7 F1 on LLM Detect) use the *same* backbone, which is fair, but the table also includes RLIE with larger Qwen models, while the baselines are not re-run on those backbones. This makes it harder to attribute performance gains purely to the RLIE training scheme vs. backbone/model differences. A stricter comparison would (a) run IO Refinement and HypoGeniC with Qwen3-235B, or (b) present the DeepSeek-V3 block separately as the primary fairness comparison, clearly de-emphasizing the Qwen rows.

3. **Key design choices lack ablations (especially iterative refinement and abstention).**  
   The method hinges on several nontrivial design choices that are under-explored:
   - **Iterative refinement vs. one-shot rules.** There is no ablation showing performance if you only do a single round of rule generation + logistic regression (no hard-example loop), or if you select hard examples randomly rather than via $d_i$. Section 5.1 attributes robustness and generalizability to the iterative refinement, but that is not empirically isolated.
   - **Ternary judgment and abstention vs. forced binary.** In Section 3.1, rule evaluations $z_{i,j} \in \{-1, 0, +1\}$ are a key modeling idea, with 0 indicating abstention. However, there is no comparison to a simpler binary encoding \{0,1\} or \{-1,+1\} without abstain. Since logistic regression in Section 3.2 directly uses $\Phi^{(t)}(x_i) \in \{-1,0,+1\}^{m^{(t)}}$, the “0” vs “no feature” distinction matters, but the empirical impact is not quantified.
   - **Rule capacity $H$ and number of rules per iteration $h$.** The only parameter study is **Table 4**, which explores coverage threshold $\gamma$ on a *single* dataset (Headline). There is no study of how performance changes with $H$, $h$, or $k$ (number of hard examples), although these could materially impact both accuracy and interpretability.

4. **Cost and scalability of rule application are not analyzed.**  
   RLIE requires calling the LLM for each (sample, rule) pair during rule judgment, see Section 3.1: $z_{i,j}^{(t)} = \mathrm{LLM}(x_i, h_j^{(t)})$. With $H=10$ rules and $N_{\text{tr}}=200$ per dataset this is manageable, but the paper gives no discussion of computational cost as $N$ or $H$ grows, nor any amortization strategies (e.g., batched evaluations or cached representations). Since one of the central claims is that RLIE is a practical engineering principle for neuro-symbolic systems, not just a toy experiment, the lack of complexity analysis or runtime/money cost plots is a substantial gap. At scale, the per-instance, per-rule LLM calls could make the method impractical, especially in iterative refinement.

5. **Evaluation of inference strategies leaves some confounds unresolved.**  
   The negative result that E2–E4 underperform E1 is compelling but somewhat underspecified:
   - It is not entirely clear whether the LLM used to *apply* rules (E2–E4) is the same as the LLM used for rule generation and judgment in Sections 3.1–3.3. Section 4.3 says “All experiments involving LLMs utilized gpt-4o-mini,” but **Table 1** and **Table 2** refer to DeepSeek-V3 and Qwen3-235B as “backbones” for inference. This mismatch should be clarified: which LLM is used for which stage? If different LLMs are used for rule generation/judgment vs. inference, that could affect how naturally they consume the rules and weights.
   - For E3/E4, the prompts in **Figures 7 and 8** instruct the LLM to “use the weighted patterns and bias as reference” and to treat the regression model’s label as a “suggestion.” This framing *encourages* the LLM to deviate from the linear model, which may be exactly why it overwrites correct judgments. A more neutral prompt that asks the LLM to carefully follow the probabilistic model unless it finds a strong contradiction might yield different behavior. Right now, the conclusion “LLMs cannot faithfully use probabilistic cues” is stronger than what the prompt design strictly supports.
   - There is no calibration analysis (e.g., reliability diagrams, Brier scores) contrasting the linear model’s probabilistic outputs to any implicit probabilities the LLM might express.

6. **Some mathematical / notation issues and under-specified details.**  
   - In Section 2.1, the notation “$r_j(x) \in \mathbb{R}^{n \times n}\{0,1\}$” is syntactically broken; presumably $r_j(x) \in \{0,1\}$ and the $n \times n$ is a leftover artifact. This is minor but sloppy in a core definition.
   - In Section 3.2, the feature vector is $\Phi^{(t)}(x_i) = \mathbf{z}_i^{(t)} \in \{-1,0,+1\}^{m^{(t)}}$ and the likelihood is $p^{(t)}(x_i; \theta^{(t)}) = \sigma( (\Phi^{(t)}(x_i))^\top \beta^{(t)} + b^{(t)} )$. There is no discussion of why $\{-1,0,+1\}$ is preferred over, say, mapping $\{-1,0,+1\}$ to $\{0,1,2\}$ or using separate positive/negative-rule features. Because $\beta_j$ multiplies both positive and negative activations symmetrically, this encoding implies $+1$ and $-1$ are anti-symmetric, which is intuitively reasonable but not explicitly justified, and 0 is indistinguishable from “no rule exists”. This could matter especially when combining many partially correlated rules, and deserves at least some discussion or an ablation.
   - The cross-validation procedure for $(\lambda, \alpha)$ is said to use “stratified K-fold cross-validation on $\mathcal{S}_{\mathrm{val}}$,” which is an odd choice: usually one cross-validates on the *training* set, not the validation set, and reserves a separate validation set solely for early stopping or model selection. The description should be clarified to avoid confusion about potential double-dipping.

7. **Experimental protocol and variance reporting are somewhat weak.**  
   Section 4.3 states that each experiment is repeated at least three times and that gpt-4o-mini is used with temperature $10^{-5}$ “to ensure deterministic outputs.” If LLM outputs are deterministic under fixed prompts and splits are fixed, it is unclear what randomness is being averaged over, and hence what the standard deviations reflect. Moreover, **Table 1** and **Table 2** only show means (and not standard deviations), despite the text saying both are reported. This inconsistency, plus the small data regime ($N_{\text{tr}} = 200$), raises questions about statistical significance. For example, RLIE vs IO Refinement differences of 1–2 F1 points may not be robust. More careful reporting would strengthen the claims.

8. **Missing key related work on probabilistic reasoning with LLMs.**  
   Section 2.2 frames RLIE as “the first to explicitly combine LLMs with probabilistic methods to learn a set of weighted rules.” However, there is now a growing body of work specifically on probabilistic reasoning *with* LLMs (see next section). While not identical in task formulation, these works probe very similar questions about LLMs’ limitations in probabilistic integration and structured combination of evidence. Not citing or positioning against them makes the novelty and broader impact look overstated.

## Potentially Missing Related Work

1. **Batu Ozturkler, Nikolay Malkin, Zhen Wang, “ThinkSum: Probabilistic Reasoning over Sets Using Large Language Models,” 2023.**  
   This work proposes a two-stage probabilistic inference framework for LLMs over sets of evidence, which is directly relevant to RLIE’s goal of combining multiple rule-based signals. It should be discussed in Section 2.2 as a related attempt to structure probabilistic reasoning with LLMs, and compared in the discussion of E1–E4 in Section 6 (e.g., on whether LLMs can reliably integrate probabilistic information when appropriately scaffolded).

2. **Linlu Qiu, Fei Sha, Kelsey Allen, “Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models,” 2026.**  
   This paper introduces Bayesian teaching strategies that significantly improve LLMs’ probabilistic reasoning. It is highly relevant to the negative findings in **Table 2**, which show LLMs failing to use rule weights and linear predictions. Citing and contrasting this in Section 2.2 and Section 6 would contextualize RLIE’s results: perhaps LLMs *can* be taught to use probabilistic signals if the teaching protocol is appropriate, which might alter the conclusions about their deficiencies.

3. **Shenxiong Li, Huaxia Rui, “Dual Traits in Probabilistic Reasoning of Large Language Models,” 2024.**  
   This work empirically investigates LLMs’ probabilistic reasoning capabilities, identifying both strengths and systematic weaknesses. It is relevant to the discussion in Sections 5.2 and 6, where RLIE argues that LLMs are unreliable at “fine-grained, controlled probabilistic integration.” Adding this citation would better position the paper within the emerging literature on diagnosing and improving LLM probabilistic reasoning.

These should be added to Related Work and Discussion, with explicit commentary on how RLIE’s empirical findings complement or contrast with them.

## Questions

1. **Clarification on which LLMs are used for which stages.**  
   Section 4.3 says that all LLM experiments use gpt-4o-mini, but **Tables 1 and 2** list DeepSeek-V3 and Qwen3-235B as “backbones.” Please clarify the mapping:  
   - Which model generates rules?  
   - Which model performs ternary rule judgments $z_{i,j}$?  
   - Which model is used in E2–E4 for test-time inference?  
   If different models are used for different roles, could you provide a brief sensitivity analysis showing whether using the same model throughout changes the qualitative conclusions?

2. **Ablation on iterative refinement.**  
   How much of RLIE’s advantage over IO Refinement and HypoGeniC comes from the hard-example iterative loop vs. the initial rule generation? Could you add an ablation where (a) rules are generated once from random samples and never refined, and (b) hard examples are selected randomly rather than by $d_i$? If these ablations show significantly lower performance, that would strengthen the case that the RLIE iteration is doing more than basic rule bagging.

3. **Effect of ternary judgments and abstention.**  
   Have you tried variants where rules must commit to $\{-1,+1\}$ (no abstain) or where abstain is treated differently in the feature representation (e.g., separate indicator features for rule-applicable vs rule-positive vs rule-negative)? A small ablation would clarify whether the 3-valued encoding is essential to the gains.

4. **Scalability and cost.**  
   For larger datasets or higher $H$, the cost of evaluating $z_{i,j} = \mathrm{LLM}(x_i, h_j)$ for all rules and samples could become large. Do you have any empirical runtime or cost measurements (e.g., total tokens / dollar cost per dataset, or wall-clock per iteration) that you can share? Are there obvious batching or sharing strategies that you have tried or recommend?

5. **Calibration and probabilistic quality.**  
   Since a core argument is that logistic regression provides a calibrated, robust probabilistic combiner while LLMs do not, can you provide calibration metrics (e.g., ECE, Brier score) for E1 vs E2–E4? Even a small plot for one dataset would solidify the “probabilistic integration” story beyond pure F1 comparisons.

6. **Variance and repeated runs.**  
   Given that you set temperature to $10^{-5}$ for determinism, what exactly varies across the “three runs” whose mean and standard deviation you mention in Section 4.3? Are you changing random seeds that affect the subset selection of training examples for rule generation or the hard-example TopK, or the data splits themselves? Clarifying this, and actually reporting the standard deviations in **Tables 1–2**, would help assess robustness.

Answers or additional experiments along these lines could raise my assessment, especially if they show that (i) iterative refinement and abstention are indeed crucial, and (ii) the negative result on LLM-based inference holds under alternative, more “Bayesian-teaching-like” prompting.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A. The datasets are standard NLP benchmarks (HypoBench subsets), and the paper appropriately discusses the possibility that learned rules may reflect societal biases. No obvious additional ethics concerns beyond what is already acknowledged.

## Soundness Rating

2: fair.  
The method is technically coherent and correctly specified at a high level, but there are under-specified aspects (e.g., cross-validation on the validation set, lack of ablations for key design choices, no cost analysis) and some minor notation/methodological inconsistencies. The empirical results support the main qualitative claims but are not as thorough or calibrated as they could be.

## Presentation Rating

3: good.  
The paper is generally well written, with a clear pipeline (especially **Figure 1**) and helpful qualitative tables (**Table 3**). However, some important clarifications about which LLMs are used where, the exact CV protocol, and the meaning of repeated runs are missing, and a few math typos/notation issues slightly detract from clarity.

## Contribution Rating

2: fair.  
The main contribution is an empirically supported engineering pattern: use LLMs for rule generation and local judgments, but rely on a classical probabilistic combiner for global reasoning; feeding rules and weights back into LLMs tends to hurt. This is useful but incremental relative to existing hypothesis-generation frameworks and recent work on LLM probabilistic reasoning. The lack of deeper ablations and broader positioning limits the perceived impact.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The paper offers a clean and interpretable hybrid framework, with solid performance and an interesting negative finding about using LLMs as probabilistic combiners. However, the methodological novelty is modest, several design choices are not empirically dissected, related work on probabilistic LLM reasoning is incomplete, and the experimental design could be stronger and more transparent. With more thorough ablations, clearer backbone fairness, and better positioning vs recent probabilistic LLM works, this could become a solid ICLR paper; in its current form I lean slightly negative.

## Reviewer Confidence

4: confident.  
I am familiar with LLM-based hypothesis/rule learning and probabilistic modeling, and I carefully checked the equations and experimental setup. Some details (e.g., exact LLM/backbone usage) are ambiguous in the text, which slightly reduces certainty, but the overall assessment is unlikely to change drastically.