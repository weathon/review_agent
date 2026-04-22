# Beyond Search: Direct Model Guidance for Steerable Synthesis Planning

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Synthetic chemists often need to incorporate domain-specific constraints and preferences into synthesis plans, 
such as preferred reaction types or available starting materials, motivating the development of steerable synthesis planning methods. 
While previous approaches navigate the search space with a frozen single-step model for plausible multi-step routes, 
we demonstrate that directly guiding the single-step retrosynthesis model enables exploration of previously inaccessible chemical 
spaces during generation. Specifically, we invoke guidance to modify the logits of an autoregressive seq2seq retrosynthesis model, 
enabling conditioning on various properties without retraining. 
Empirically, we demonstrate that while commonly used single-step models struggle to find routes with chemically feasible single-step 
reactions throughout the entire synthesis plan, our method generates synthesis routes of equal or better quality than 
template-based approaches while satisfying the specified constraints.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates guiding a trained single-step retrosynthesis model with the outputs from a separate property predictor. Authors investigate a toy setup as well as two retrosynthesis setups (one based on guiding with reaction class information, and one based on guiding towards the use of a particular building block molecule).

### Strengths
**(S1)**: Retrosynthetic search is a timely and important problem to work on. Current search algorithms are often not very steerable, and thus looking into giving more fine-grained control to the user can be useful.

### Weaknesses
**(W1)**: Several parts of the work are quite unclear to me:

- **(W1a)**: Authors talk about "guiding towards reaction type", but it's not always clear _which_ reaction type is the target. In single-step and manual multi-step experiments, are predictions simply guided towards the _ground-truth_ reaction type? How does this generalize to a search-based multi-step setup, where the algorithm may end up using reactions outside of training data for which ground-truth (expected) reaction type is not known? Are all reactions during search guided towards the same reaction type in this case?

- **(W1b)**: Does guiding towards a specific starting material mean forcing the predicted reaction to have the required molecule as _one_ of the reactants, allowing potential further reactants to be generated freely? In Section 4.3, authors state this setup works by "forcing the model to ouput the tokens matching the conditional starting material, by (…) increasing the guidance scale to the point of overwhelming the generator logits". Does that mean simply teacher-forcing the autoregressive generator to start the generation with outputting the required reactant? Also, in experiments, authors seem to assume access to ground-truth route depth, and then force the starting material at the given depth; I'm not sure how can this work for convergent routes where there are several reactions at that depth.

- **(W1c)**: If I understand Equation 1 correctly, it says a guided route would satisfy the requirement of having more reactions pass _at least one_ of the property requirements than a route found without guidance. This seems like a confusing formulation, as it requires including a "baseline" route to compare to, doesn't reward cases when a single reaction satisfies several (or all) property predicates simultaneously (even though in experiments author seem to only use a single predicate?), and only provides a minimum standard rather than an objective to optimize. Later sections in the text suggest the objective authors case about is maximizing the expected number of accepted reactions, in which case I'm not sure what purpose Equation 1 serves.

**(W2)**: On top of the clarity issues above, I don't think the setup used in this work is practical. In essence, the authors show that if some aspect of the ground-truth or expected solution is known (e.g. reaction type used, or one of starting materials), guiding the predictor towards that can increase the likelihood of finding that expected solution; this sounds reasonable. However, it raises two questions: (a) whether this approach is practically useful or needed; and (b) whether it is effective in producing good outcomes. I have doubts about both fronts:

- **(W2a)**: Authors look at two potential applications: guiding towards reaction type, and guiding towards using a particular starting material (building block). For the former, I am not sure how this works in an end-to-end search setup (see **(W1a)**), are all steps in the search being guided towards the same reaction type? While chemists may have some preferences (which could be caused by them having explored a particular set of reaction types exhaustively and thus having built intuition or specialized feasibility models for those kinds of reactions specifically), this would more often be a set of reaction types; also, if they had a strong preference to only use a limited set, they may not even use a retrosynthetic search tool to begin with, which are best suited for broad exploration. Therefore, I'm not sure if this setting would be that popular in practice. For guidance towards a specific starting material, I assume the hypothetical setup is when a molecule that is complex but similar to the target has already been synthetized at a particular lab, and thus chemists are looking for routes that reuse that complex intermediate. In that case, a simple baseline would be to include it in the building block set; this would be much simpler, would not require knowing ground-truth depth (see **(W1b)**), and would trivially generalize to having several such intermediates instead of just one.

- **(W2b)**: As stated by authors, there is a trade-off between following the guidance and quality of the predictions. In fact, deterioration seems quite visible in the quantitative results; e.g. in the single-step USPTO-50K experiment, while guidance allows to recover the ground-truth for products where the unguided model fails, the guided model performance is then worse on other products. This is odd to me: assuming my understanding in **(W1a)** is correct, guidance towards ground-truth reaction type should generally only improve results, unless the guiding property model is confusing the generator. In this case, better results would likely be obtained by simply sampling more outputs from the unguided generator and post-filtering, which would then invalidate the need for guidance in the first place. 

**Other (much less important) comments**

**(O1)**: In the toy task, vocabulary is constrained to `{1, 2, 3}`, but then larger digits are allowed to appear during simplification, so I'm not sure why the "full simplification" option is not just `7` instead of `5 + 2`.

**(O2)**: While the paper includes some theoretical analysis (Theorem 1), I do not include it as a strength, as I think the result is not that interesting. It is of course nice to have, but I think it doesn't help against my concerns, especially ones around practicality of the setup **(W2)**.

### Questions
See the "Weaknesses" section above for specific questions.

### Soundness
1

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
This paper introduces a method that steers retrosynthesis generation by modifying the token-level logits of a pretrained seq2seq model (RSMILES) through a lightweight property classifier.
The approach aims to make synthesis planning steerable—conditioning on reaction type or specific starting materials—without retraining, and includes a theoretical guarantee that guided beam search can reach low-probability yet property-consistent regions of chemical space.
Conceptually, the idea is promising and well-motivated, bridging diffusion-style classifier guidance with chemical synthesis planning.
However, the experimental and evaluation sections remain thin: the baselines are incomplete, the datasets limited, and the performance gains difficult to attribute purely to the proposed mechanism.

### Strengths
### Strengths

1. **Realistic and well-motivated application scenario**
   The paper is driven by an actual need in synthesis planning, where chemists often want to control reaction types or enforce the use of specific starting materials rather than simply find any feasible route. This makes the problem formulation genuinely insight-driven and practically relevant, rather than an artificial controllability setup.

2. **Potentially general framework**
   Although the experiments are conducted only on a single retrosynthesis model (RSMILES), the proposed guidance mechanism that modifies token-level logits through a property classifier appears broadly applicable to other autoregressive chemical or molecular generation models. The idea could in principle be extended to different model architectures or even to other domains where conditional generation under structural constraints is desirable.

### Weaknesses
### Weaknesses

1. **Incomplete Baselines and Dataset Coverage**
   The paper compares guided RSMILES only against unguided RSMILES and NeuralSym on USPTO-50k / USPTO-190. However, the field already includes many recent and competitive baselines for constrained or steerable synthesis planning, which are not covered here. Furthermore, all experiments are restricted to USPTO-style datasets; evaluation on larger and noisier benchmarks such as Pistachio would be necessary to demonstrate generalization beyond USPTO distributions.

2. **Dependence on Pseudo-Labels and Unfair Evaluation Setting**
   The reaction-type guidance relies on RXN-Insight-generated pseudo-labels for USPTO-190. These labels are potentially noisy and highly imbalanced, yet the paper does not quantify their accuracy or analyze class-wise performance. More importantly, the guided model effectively receives extra oracle information (the reaction class) that baseline models do not, making the comparison unfair. The authors should either allow baselines to use the same reaction-type hints (e.g., via reranking or constrained search) or explicitly frame the task as conditional retrosynthesis. It would also strengthen the study to verify whether guided generations genuinely fall into subclasses or finer-grained reaction families rather than only matching top-level class labels.

3. **Unclear and Possibly Test-Time-Tuned λ Selection** The guidance strength λ is chosen from a small grid (0.5, 1.0, 2.0) “based on average reaction-type accuracy,” but the paper does not specify which data split is used for tuning or whether the same λ applies across datasets. Several experiments are conducted on failure subsets of the test set (e.g., USPTO-50k failed products), suggesting possible test-time hyperparameter tuning, which could inflate reported improvements. Since the method is promoted as “no retraining and easy to apply to new constraints,” the paper should either demonstrate robustness to λ or propose a principled, validation-free way to choose it.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a steerable synthesis planning framework that, rather than improving only the search over candidates produced by a fixed single-step retrosynthesis model, directly modifies the token-level decoding of an autoregressive seq2seq model via a learned property predictor. By injecting a guidance term into the logits during generation, the method biases single-step predictions toward reactions or precursors that satisfy user-specified constraints. The authors further provide a theoretical argument suggesting that this token-level guidance can make parts of the chemical space reachable that standard beam search would not explore.

### Strengths
- Demonstrates that steering at the generation level (i.e., changing what the single-step model emits) can unlock routes that search-level steering alone would miss.
- Uses a simple, model-agnostic guidance formulation that can in principle support multiple constraint types (reaction class, starting material, etc.) without retraining the base model.

### Weaknesses
- Novelty currently oversold. Classifier- or property-based token-level guidance for autoregressive decoding is well established in controllable text generation, and “template-free multi-step retrosynthesis with search” is a mature direction. The main contribution here is adapting that guidance paradigm to neural retrosynthesis and formalizing its accessibility claim—valuable, but not as fundamental as the introduction implies.
- Experimental coverage is too narrow for the main claim. Most results are against (i) vanilla rSMILES and (ii) neuralsym + Retro*, often with the authors’ own models. To substantiate the claim of reaching “previously inaccessible” routes, the paper should either (a) add stronger template-free neural baselines under the same search/computation budget, (b) include a controlled comparison with an LLM-steered or tool-augmented planner, or (c) show steering under multiple simultaneous constraints (e.g., reaction class + inventory + maximum depth).
- Related work is incomplete. The paper largely contrasts with search-level/ranking methods (Segler et al., Lin et al., Tango*) but omits recent LLM-driven, chemistry-aware steering approaches that target the exact same “interactive, preference-informed synthesis” use case—e.g. Bran, Andres M., et al., “Chemical reasoning in LLMs unlocks steerable synthesis planning and reaction mechanism elucidation,” arXiv:2503.08537 (2025). This line should be acknowledged and the differences in controllability, data requirements, and integration with planners should be discussed.

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the authors develop a classifier guidance framework to modify the token-level logits of an autoregressive sequence-to-sequence model for retrosynthesis. The goal is to incorporate domain knowledge to retrosynthesis models. Evaluation is performed on USPTO-50k and USPTO-190, with the proposed rsmiles method.

### Strengths
* Incorporating domain knowledge into synthesis planning in retrosynthesis is an important and cutting-edge research topic in AI for Science.

### Weaknesses
* The method section does not include enough level of detail for readers to understand and reproduce your work. Specifically, the following important questions are not answered after reading your paper:
    * What are the building blocks of your proposed framework?
    * What is the algorithm framework? How does the proposed approach integrate into an existing sequence-to-sequence model?
    * What are the classifier models used? What are the architectures? How to train the classifiers? 

  All details above should be clear to readers after reading your method section, instead of listing math equations.
* Figure 1 is hard to understand. The legend said "numbers represent the reaction types (0 to 11 classes)", but I cannot understand the meaning. I cannot tell the benefit of "steered" synthesis planning, either.
* Theorem 1 seems trivial, and I cannot understand the insight behind it. If, as discussed in L157, _the key insight is that token-level guidance accumulates exponentially across sequence length_, it seems an obvious intuition and does not need such a heavy theorem block. It will be much better if you save the space for methodological details.
* In the evaluation, only limited baselines are considered. I do not even find a place where the authors mention the name of the proposed approach. It is challenging to consider the contribution of this work if it does not consider the strongest baselines available (they are discussed in the introduction and related works). Also, an ablation study is missing.

### Questions
Please see weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2
