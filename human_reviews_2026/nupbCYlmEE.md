# Optimizing Language Models for Crosslingual Knowledge Consistency

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
Large language models are known to often exhibit inconsistent knowledge. This is particularly problematic in multilingual scenarios, where models are likely to be asked similar questions in different languages, and inconsistent responses can undermine their reliability.
In this work, we show that this issue can be mitigated using reinforcement learning with a structured reward function, which leads to an optimal policy with consistent crosslingual responses. We introduce **D**irect **C**onsistency **O**ptimization (DCO), a DPO-inspired method that requires no explicit reward model and is derived directly from the LLM itself.
Comprehensive experiments show that DCO significantly improves crosslingual consistency across diverse LLMs and outperforms existing methods when training with samples of multiple languages, while complementing DPO when gold labels are available. 
Extra experiments demonstrate the effectiveness of DCO in bilingual settings, significant out-of-domain generalizability, and controllable alignment via direction hyperparameters. 
Taken together, these results establish DCO as a robust and efficient solution for improving knowledge consistency across languages in multilingual LLMs. All code, training scripts, and evaluation benchmarks are released at https://anonymous.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The paper introduces DCO (Direct Consistency Optimizaiton), which is a RL inspired method as means for improving crosslingual knowledge consistency (CLC) in multlingual models. DCO differs from prior RLHF or DPO-based alignment methods by taking a structured reward function based o the log-likelihood agreement across translations of prompts and responses.
- DCO method proposed by the authors show that they theoretically guarantees consitency between cross-lingual knowledge when certain hyperparameter relations (as outlined in Lemma 1) hold.
- The authors evaluate DCO across 9 multilingual LLMs and 3 datasets and show that there are consistent improvements in both the CLC (RankC metric proposed in prior work) and non-English answer accuracy.
- They also conduct some ablation studies showing that DCO also performs well in bilingual settings (between English and one non-English language), showing promises in joint consistency improvements across English and other non-English languages (Section 6.2) and show out-of-domain transferability in Section 6.3 on different domains in the MMMLU dataset.

### Strengths
- In terms of originality, the authors propose a new formulation of CLC alignment as an RL problem with a piecewise reward function rather than using heuristic voting or representation intervention. DCO extends on the DPO-style training to multilingul consistency without human preference labels, which makes it a efficient method where there are no human preference datasets available. Also, DCO’s efficiency, in a sense that it doesn’t require explicit reward model or human annotation makes it practically adoptable for large-scale post-training.
- In terms of quality, the paper seems to have a clear theoretical derivation in terms of the reward definition, optimal policy and other proofs of consistency which made it easier to follow the concept of DCO.
- Figure 1 was especially helpful in understanding the intuition of how the authors tried to align the likelihoods across languages.

### Weaknesses
- The translation equivalence assumption made in Section 4.1 before Definition 1, where the authors assume the translation mappings of t1 and t2 seems to be strong, which may not directly hold for other open-ended or culturally grounded tasks beyond the short-form factual QA tasks used in the current paper. This questions the generalizability of the DCO method beyond short-form QA tasks where having consistent knowledge across languages might act as a stronger influence factor.
- Based on my understanding, it seems like DCO is operationalized almost entirely via RankC, which measures how similarly the model ranks candidate completions across languages. But then RankC depends on likelihood distributions over a fixed candidate set, where the candidate answers are parallelized across languages. So I believe RankC actually only evaluates agreement with oneself, not agreement with the truth. The paper tries to mitigate that by also reporting the accuracies of English and non-English languages, but those seem to be quite disjoint, since you can become “more consistent” while consistently wrong. This could be illustrated in scenarios where consistency improves even when English accuracy drops and non-English accuracy barely moves or even degrades in some settings (this pattern in observed for some cases in Table 1). This might be an evidence that “consistency” and “correctness” are not necessarily aligned.
- Further, RankC operates on multiple-choice or cloze-style completions with finite candidate sets. But in real-world deployment, many of the queries are open-ended and answers are free-form rather than users actually give a query and all multiple choice options. Thus, the metric seems to be rather measuring distributional alignment under very controlled conditions. Maybe one way to answer this is looking into qualitative bilingual judgments (are models actually saying the same thing in both languages, or just picking the same multiple-choice index?) or evaluate on open-ended factual QA tasks with no fixed options with human raters judging factual equivalence across languages.
- This is a pretty narrow baseline space relative to the current baselines available for multilingual alignment methods. To my knowledge, there is a large body of work on representation-space alignmen or causal interventions for multilingual consistency and transfer, which explicitly attempt to enforce that different languages activate similar internal features for the same factual content. The paper does mention these methods in the Related work section but does not empirically compare and without such comparisons, it seems hard to tell if DCO is actually better than these methods.
- The formal notion of consistency suggested by the authors in Equation 5 seems to rely more on the “ranking” equivalence but it does not account for any semantic equivalence or paraphrasing differences. This could bake in two limitations: one is that consistentcy is evaluated only over a fixed candidate set of completions that are assumed translationally aligned and second, consistency is more ordinal (which one do the LLM prefer more) than factual (is the LLM asserting the same proposition). The theory (Lemmas 1-2 and the product-of-experts optimal policy) is proved under this narrow notion of consistency, so the conclusions might not generalize for more realistic definition of consistency, such as factual equivalence in free-form answers.
- The paper lacks analysis in terms of training stability, sensitivity to hyperparameters or computational costs comparison with DPO or other baselines.
- All the tables in the main paper seems to be showing results in terms of English vs. aggregated results across all non-English languages. Since the number of languages of each tasks is quite large (14, 16, 17), having language-specific trends would provide a better understanding. I’m also curious whether this only works for a subset of the languages or if the gains still hold for lower-resource languages.
- The paper seems to only have pairwise comparisons between English and another non-English language, which also questions the generalizability to improving knowledge consistency between two non-English languages.

### Questions
- Are both XCSQA and BMLAMA entirely machine translated? The paper notes that MMMLU is translated by human annotators but no information was provided for the two other datasets. If so, does the machine translation quality impact the findings? What are the differences between machine translated dataset and human translated dataset? Would DCO also work for noisy translations?
- How expensive is DCO in practice compared to standard DPO in terms of compute and sample efficiency?
- Do the improvements hold consistently across typologically distant languages (e.g., Arabic to Chinese) versus closely related ones (e.g., Spanish to Italian)?
- The paper uses RankC for consistency, but have the authors considered or tested semantic metrics such as multilingual BERTScore or COMET? RankC relies on overlap over discrete candidates, which might not reflect deeper factual alignment.
- The paper claims DCO could generalize to “other consistencies” such as paraphrase or in terms of modality. Could you elaborate on what modifications would be needed to extend Equation 11 beyond the suggested bilingual lexical alignment?
- [Minor] The anonymous link in the abstract appears to end abruptly, but it’s unclear whether this was intentional for anonymization or an oversight.

### Soundness
2

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
4

### Summary
The paper addresses the problem of crosslingual knowledge consistency (CLC) in large language models. The authors formulate this as a reinforcement learning problem and propose a novel, structured reward function, $r_{ALIGN}$, and introduces Direct Consistency Optimization (DCO), a practical, DPO-inspired algorithm that directly optimizes the policy to achieve this reward objective. Empirical results show that DCO significantly improves CLC, substantially outperforming the baseline.

### Strengths
DCO is theoretically grounded, and demonstrated excellent empirical performance in cross-language consistency. Notably, while optimizing consistency, DCO can also improve generalization without gold labels. Besides, DCO may extend to consistency optimization beyond cross-language consistency.

### Weaknesses
[W1] DCO relies on well-defined cross-lingual pairs. These pairs are readily available on datasets such as MMMLU, but not straightforward for general tasks such as dialogue and reasoning, which limits its applicability.

[W2] When gold labels are available, DCO improves cross-lingual consistency, but sometimes slightly degrades performance for English.

### Questions
[Q1] How does models trained using DCO and baseline methods perform when tested on datasets other than the training dataset (e.g., GSM8k)? This will characterize the degradation in LLM capability after cross-language consistency optimization.

### Soundness
4

### Presentation
4

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
This paper studies RL strategies for cross-lingual consistency gains. Specifically, the authors modified the first term of RLHF to align the ranking of the output answers (not exactly the distribution) for multiple languages. In experiments, the authors demonstrate that the presented method, DCO, can improve cross-lingual consistency in general settings, bilingual settings, and  OOD settings.

### Strengths
1. The authors provide sufficient evidence to support the claim. 

2. The method is straightforward and well-motivated. 
- The authors suggest aligning ranking instead of distributions, which is a good idea inspired by existing works.
- The method stems from RLHF and only makes minimal changes.

3. The authors show the method is compatible with other methods, e.g, DPO.

### Weaknesses
1.	Based on my understanding, the authors only conduct experiments on multiple-choice tasks. Can authors demonstrate some results on generation tasks, e.g., mLama? How do you configure BMLAMA? Do you treat it as a multiple-choice task or a generation task?

2.	I think authors should experiment with other tasks after DCO to show the method does not hurt performance on these tasks in an across-task generalization setting. Does DCO hurt the language modeling performance as it changes the ranking of output candidates?

### Questions
Please refer to Weaknesses

### Soundness
2

### Presentation
3

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
This paper introduces Direct Consistency Optimization (DCO), a DPO-inspired method for improving crosslingual knowledge consistency (CLC) in multilingual large language models (LLMs). Instead of relying on explicit reward models, DCO defines a structured reward function that encourages likelihood alignment between responses in different languages, achieving consistency without supervised labels. The paper presents theoretical guarantees for consistency, detailed empirical evaluations on 9 multilingual LLMs and 3 datasets (MMMLU, XCSQA, BMLAMA), and ablations exploring hyperparameter effects and generalization across domains.

### Strengths
- The problem of multilingual inconsistency is timely and practically relevant.
- The proposed reward formulation is mathematically grounded and connects neatly with DPO theory.
- The empirical evaluation is comprehensive, showing consistent improvements in CLC across many models and datasets.
- The bilingual and out-of-domain experiments strengthen the paper’s practical relevance.
- The direction-control parameters (γ₁, γ₂) provide an interpretable mechanism to balance alignment between high- and low-resource languages.

### Weaknesses
- The approach assumes accurate translation mappings and one-to-one correspondence between answers, which may not hold for open-ended or ambiguous tasks.
- Theoretical treatment for multi-language (>2) alignment is underexplored; practical guidance for setting γ parameters is limited.
- The experiments focus only on factual QA, leaving unclear how DCO performs in generative or paraphrastic tasks.
- There is little qualitative analysis of failure cases or discussion of translation noise sensitivity.
- Statistical significance and variance measures are missing from tables, limiting assessment of robustness.

### Questions
1. How robust is DCO when translation mappings are noisy or non-bijective?
2. Can the authors clarify how γ parameters should be tuned in practice for N > 2 languages?
3. Would likelihood normalization across languages improve stability?
4. Could DCO generalize to open-ended generation or paraphrase consistency tasks?

### Soundness
3

### Presentation
3

### Contribution
3
