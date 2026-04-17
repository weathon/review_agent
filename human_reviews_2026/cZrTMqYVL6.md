# Addressing divergent representations from causal interventions on neural networks

- Decision: Accept (Oral)
- Scores: 6, 4, 4, 8, 4

## Abstract
A common approach to mechanistic interpretability is to causally manipulate model representations via targeted interventions in order to understand what those representations encode. Here we ask whether such interventions create out-of-distribution (divergent) representations, and whether this raises concerns about how faithful their resulting explanations are to the target model in its natural state. First, we demonstrate theoretically and empirically that common causal intervention techniques often do shift internal representations away from the natural distribution of the target model. Then, we provide a theoretical analysis of two cases of such divergences: "harmless" divergences that occur in the behavioral null-space of the layer(s) of interest, and "pernicious" divergences that activate hidden network pathways and cause dormant behavioral changes. Finally, in an effort to mitigate the pernicious cases, we apply and modify the Counterfactual Latent (CL) loss from Grant (2025) allowing representations from causal interventions to remain closer to the natural distribution, reducing the likelihood of harmful divergences while preserving the interpretive power of the interventions. Together, these results highlight a path towards more reliable interpretability methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper shows in synthetic settings how causal intervetions, a widely used tool in interpretability, may introduce divergent representations in NNs. The authors studies 2 kinds of such divergences: "harmless", which are in the null space of successive projections; and "pernicious" that may active unintended behaviors downstream. The authors then propose an approach training DAS projections ([Geiger et al, 2023](https://arxiv.org/abs/2303.02536)) incorporating Counterfactual Latent loss ([Grant, 2025](https://arxiv.org/pdf/2501.06164)) to get rid of pernicious divergences.

### Strengths
* Solid theoretical discussion on how divergences may arise from causal interventions.
* Great presentation and framing of the problem.

### Weaknesses
* Simulated in scenarios with very strong assumptions which may not hold in practice. 
* The proposed approach seems unrealistically expensive to be applied in practice.

### Questions
1. Figure 2: You show divergences introduced by adding mean diff, patching SAE and DAS. But what about activation patching from a single source sample ([Prakash et al, 2025](https://arxiv.org/abs/2505.14685), [Feucht et al, 2025](https://arxiv.org/pdf/2504.03022), inter alia)? Do you assume such interventions to also introduce divergences?

2. Section 4.2.2: Very interesting observation. Unless I am understanding it wrong, a layernorm right after the intervention can potentially fix this issue, right? You probably should mention this point as most architectures have them nowadays.

3. The situations simulated showing harmless and pernicious divergences are plausible and I do think some of the inteventions will be susceptible to such divergences. However, I am not entirely convinced about how prevalent pernicious divergences are in practice. And usually it is considered a good scientific practice to evaluate proposed hypotheses on 100s (if not 1000s) of examples covering a distribution of scenarios. Do you assume such cases may hold across entire evaluation sets? 

4. (related to previous question) Training DAS projections to address this seem quite expensive, even might be just infeasible in practice. The setup makes sense and is convincing in synthetic settings. But do you see any way this can be scaled to real models and datasets? And is it worth the effort [referring to the previous question]? I will be just blunt: is this paper more of an intellectual exercise? Or do you see any practical applications?

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
2

### Summary
This paper investigates a critical, often-overlooked assumption in mechanistic interpretability: that causal interventions (like activation patching) produce latent representations that are ”faithful” or ”in-distribution” for the
target model. The authors argue that interventions can create ”divergent,” out-of-distribution states, which may
undermine the validity of the resulting explanations.
The paper’s contributions are threefold:

- It empirically demonstrates that common intervention methods, including Mean Difference Vector Patching, Sparse Autoencoders (SAEs), and Distributed Alignment Search (DAS), do create representations that
diverge significantly from the model’s natural distribution.

-  It provides a theoretical framework for classifying these divergences into ”harmless” (e.g., in the null-space
or within existing decision boundaries) and ”pernicious” (e.g., activating hidden, non-native computational
pathways or causing dormant behavioral changes). The authors use synthetic examples to show how pernicious divergences can lead to misleadingly ”affirming” results.

- To mitigate this issue, the authors propose an adapted Counterfactual Latent (CL) loss (based on Grant
(2025)). This loss regularizes interventions by encouraging the intervened latent state to remain close to an
average of native states that share the same intended causal properties . In a synthetic setting, they show this
method reduces divergence and improves out-of-distribution (OOD) generalization.

### Strengths
-  By questioning the
”faithfulness” of intervened states, the work highlights a critical potential failure mode for many mechanistic
interpretability claims.
•  The theoretical categorization of divergences into ”harmless” (e.g., nullspace, within-boundary covariance) and ”pernicious” (e.g., hidden pathways, dormant behavior) is clear,
intuitive, and  valuable.
-  The synthetic examples in Section 4.2 are particularly strong. They offer
concrete, simple-to-understand illustrations of how an intervention can be misleading; for instance, by
breaking a ”balanced subspace” or activating a pathway via a mean-difference vector that is not used by
any native data point . The concept of ”dormant behavioral changes” is also a very insightful and worrying
failure mode.
-  It also proposes a concrete (if
preliminary) mitigation strategy. The adaptation of the CL loss to ”anchor” interventions to the native data
manifold is a good approach.

### Weaknesses
- The paper successfully categorizes divergences but does not offer a method
to detect or classify whether a divergence observed in a practical (non-synthetic) setting is harmless or
pernicious. This makes it difficult to know when to be concerned about the results of their
interventions.
-  The primary weakness of the proposed solution, acknowledged by the authors,
is that the CL loss is a ”broad-stroke” approach. It penalizes all divergence, rather than selectively targeting only ”pernicious” divergence. This might be overly restrictive, as some ”harmless” divergences (e.g.,
interventions that explore the null-space) could be desirable for making certain causal claims.

- The empirical validation for the CL loss mitigation is confined to a synthetic, small-scale MLP setting. It is unclear how this approach would scale to modern,
large-scale models. For example, generating the ”Counterfactual Latent” (CL) vectors requires averaging
over a pre-recorded set of native states with specific causal properties. This seems computationally challenging and, more importantly, dependent on having a correct, known causal abstraction, which is often
what is being searched for in the first place.

### Questions
-  The CL loss mitigation seems promising but raises practical questions. How do you
envision applying this to large language models where the ground-truth causal variables (needed to find
matching native states for $h_{CL}$) are not known a priori? Doesn’t this create a circular dependency where
you need the causal abstraction to find the CL vectors, but the intervention (which you’re trying to fix) is
what you use to find the abstraction?

- You note that the CL loss is a ”broad-stroke” solution. Do you
have any initial thoughts on how one might distinguish pernicious from harmless divergence, perhaps without a full causal abstraction? For instance, could one use a measure of ”local faithfulness” or analyze
activation changes in all output dimensions (as hinted at in 4.2.3 ) to flag interventions that recruit ”hidden”
pathways?

- The concept of ”dormant behavioral changes” (Section 4.2.3, Appendix A.2) is particularly concerning. You mention that an ”infinitely expansive” dataset could detect them. In practice, how
could a researcher gain any confidence that their intervention hasn’t created such a dormant vulnerability?
Does the CL loss’s reduction in OOD error (Fig 3d) suggest it is mitigating these, or is that a separate issue?

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
This paper examines distributional divergence in mechanistic interpretability interventions. The authors argue that such interventions can push hidden representations off the model’s natural manifold, leading to pernicious effects such as spurious circuit activations or dormant behaviors. They distinguish harmless from pernicious divergences and propose a modified Counterfactual Latent (CL) loss to constrain interventions toward causal subspaces while maintaining interchange intervention accuracy (IIA). The topic is timely and important, and the paper offers useful illustrative examples and intuitions. However, the treatment remains largely informal, with vague definitions, unstated assumptions, and limited quantitative validation. The work would benefit from clearer formalization, stronger theoretical grounding, and more rigorous empirical evaluation to solidify its contribution to addressing divergent representations from causal interventions.

### Strengths
**1. Problem statement is timely and important.**  
The field of mechanistic interpretability relies heavily on causal interventions, while systematic study of representational divergence has been lacking. The harmless/pernicious dichotomy offers useful language for describing what practitioners often observe intuitively.

---
**2. Useful negative examples.**  
The "balanced subspaces" and ReLU examples are compact and concrete; they effectively illustrate how confirmatory behavior can arise from incorrect or misleading activation pathways.

---
**3. Simple mitigation with plausible effect.**  
Restricting the CL penalty to causal subspaces aligns with the intended goal of keeping interventions "on-manifold" with respect to identified causal variables.

### Weaknesses
**1. Missing formal statements or guarantees.**  
The harmless/pernicious taxonomy is presented through narrative examples rather than formal propositions (Sec. `4.1`–`4.2`, `L213`–`L377`). No formal statements, e.g., theorem or proposition, specifies necessary or sufficient conditions under which divergence is guaranteed harmless (e.g., confined to a null-space) or pernicious (activating hidden pathways).  
Key terms such as *"behaviorally binary subspace"* (Eq. `4`, `L248`–`L254`) and *"behavioral null-space"* (`L141` and many later references) are invoked but never **formally defined** or tied to measurable properties (e.g., local Jacobians or decision-region invariance). As a result, these claims remain intuitive or conceptual rather than analytical.

---

**2. Under-specified core objects.**  
*(1) Behaviorally binary subspace:* The "behaviorally binary subspace" (Sec. `4.1`, Eq. `4`, `L248`–`L254`) is defined via an elementwise sign function mapping to {$-1, 0, 1$}. Including $0$ makes the decision boundary ill-posed: infinitesimal perturbations near $0$ can flip the behavioral label, so the binary characterization becomes unstable unless neighborhoods around $0$ are excluded or measure-zero sets are explicitly treated.  

*(2) Null-space:* The "null-space" of downstream layers (Sec. `4.1`, `L213`–`L216`) is repeatedly invoked (e.g., `L481`–`L484`) but never operationally defined under nonlinear composition, normalization, or residual connections. It remains unclear whether this refers to a local Jacobian null-space at $h$ or to a global manifold of invariant directions across contexts.

---

**3. Argument relies heavily on hidden assumptions.**  
Several strong but unstated assumptions are embedded in the narrative exposition. For instance, the discussion of "harmless divergence" in Sec. `4.1` presumes a clean factorization between causal subspaces (e.g., $\tilde{\mathbf z}_{var_a}, \tilde{\mathbf z}_{var_b}$) and their covariance structure. This effectively assumes approximate orthogonality and cross-context stability of these subspaces, which is rarely satisfied in deep, nonlinear architectures such as Transformers. Additionally, the notion of harmless covariance within decision boundaries presumes stable, convex class regions (e.g., assumptions and suppositions from `L246`–`L269`), yet the later "dormant behavioral changes" example (Sec. `4.2`) admits that minor contextual shifts can cross those boundaries. The conditions distinguishing truly benign intra-boundary covariance from merely undetected behavioral drift are left unstated. Such implicit assumptions substantially weaken the generality of the reasoning.

---

**4. Evaluation concerns.**  
Divergence is visualized using 2D PCA and summarized by a single EMD score (e.g., Fig. `2`). The reviwer concerns whether such low-dimensional projections may distort pairwise distances, and averaging in EMD may mask high-magnitude divergence along select axes.  Additionally, subtracting the baseline EMD of $h$ against itself is reasonable, but since EMD is not linear, simple subtraction might obscure axis-specific drift. Have the authors explored distributional or coordinate-free divergence metrics, reporting with confidence intervals estimated across random projections?

### Questions
Please refer to the `Weaknesses` section for detailed questions. 

---

The following points are minor and do not impact the overall evaluation:

**1. Notation consistency.**  
Throughout the analytic sections (Sec. `4`–`5`), ensure that all notation is defined before use, and that it is applied consistently and unambiguously to improve clarity.

**2. Style.**  
Citations should be properly linked, e.g., "Grant (2025)" in the abstract (`L22`). Avoid multiple acronym introductions, such as "causal abstractions (CAs)" (`L126` already introduced, repeated at `L131`). Replace colloquial or vague phrases (e.g., "in causal subspaces is okay", `L280`) with precise language.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies mechanistic interpretability approaches that manipulate model representations via targeted interventions and what could go wrong if those interventions result in out-of-domain values that cause unexpected behavior in the model. The paper shows that these divergences are quite common and differentiates them between “harmless” divergences and “pernicious” divergences. A new objective based on counterfactual latent loss is presented to mitigate these issues.

### Strengths
The paper studies mechanistic interpretability approaches that manipulate model representations via targeted interventions and what could go wrong if those interventions result in out-of-domain values that cause unexpected behavior in the model. The paper shows that these divergences are quite common and differentiates them between “harmless” divergences and “pernicious” divergences. A new objective based on counterfactual latent loss is presented to mitigate these issues.

### Weaknesses
1. The paper addresses an important issue in mechanistic interpretability that is often not considered. If the data that was used to train a model is limited, then the model may be poorly defined on unseen interventions. The interpretability of such cases are therefore potentially unreliable, and this should be taken into account.

2. The points are clearly established with examples, improving the clarity of the paper.

3. The CL loss provides a potential solution to mitigating this issue.

### Questions
4. The distinction between harmless and pernicious cases are not well defined, and the paper only provides examples of each, rather than providing a concrete definition of what a divergent representation is, and whether it is harmless or pernicious.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates whether causal interventions in mechanistic interpretability create out-of-distribution (divergent) representations that compromise the faithfulness of resulting explanations. The authors: (1) empirically demonstrate that common intervention methods (mean difference patching, SAEs, DAS) produce divergent representations; (2) provide theoretical analysis distinguishing "harmless" divergences (null-space, within-boundary covariance) from "pernicious" divergences (hidden pathways, dormant behavioral changes); (3) propose adapting the Counterfactual Latent (CL) loss to mitigate divergence. The work is motivated by the fundamental assumption in interpretability that counterfactual model states should be realistic, yet experiments are limited to synthetic settings.

### Strengths
1. The paper addresses an important yet underexplored question about whether causal interventions preserve distributional faithfulness, a property essential for claims of mechanistic interpretability.
2. Section 3 systematically examines divergences arising from three distinct intervention methods (mean difference vectors, SAEs, and DAS), demonstrating that the problem is both broad and pervasive.
3. The “harmless vs. pernicious divergence” framework introduced in Section 4 offers a clear conceptual vocabulary for distinguishing when distributional shifts are consequential.
4. The paper is well-structured and logically coherent, progressing naturally from Problem Definition to Empirical Evidence (Section 3), Theoretical Analysis (Section 4), and finally to the Proposed Solution (Section 5).

### Weaknesses
1. The proposed solution (CL Loss) is only validated on a synthetic 2D task. However, the problem itself is demonstrated on large models (Llama-3-8B). The gap between these synthetic toy problems and real transformers is enormous. This complete lack of validation on actual LLMs models undermines the practical applicability and credibility of the proposed solution. 

2. The "pernicious" case in Example I (Sec 4.2.1) assumes "balanced subspaces" where $w_1h_1 = -w_2h_2$ always holds. If this were true, this subspace would be in the behavioral null-space (its net contribution is always zero) and should not be identified as "causal" by any reasonable method in the first place. The example therefore demonstrates "intervening on wrong variables fails" rather than "divergence on causal variables is pernicious." This conflates choosing incorrect intervention targets with distributional divergence problems, undermining the credibility of the pernicious divergence category. 

3. The example in Sec 4.2.2 requires precisely engineered weight matrices, ReLU boundaries, and specific data distributions. The probability of such a fragile and exact configuration occurring in a real, trained network is vanishingly small. This makes the example feel practically irrelevant and weakens the argument that "pernicious" divergence is a common, real-world problem.

4. Even on the simple toy task where the solution should work perfectly, the CL Loss provides minimal benefits. The results show that while EMD (divergence) was reduced, the IIA (task accuracy) actually dropped from 1.0 to 0.998. This suggests the method is weak, ineffective, and may even introduce negative side effects.

### Questions
Q1. Real model validation (relates to W1, W3): Can you provide results on at least one real LLM (e.g., GPT-2 Small) showing that CL loss maintains or improves causal identification accuracy while reducing divergence? Even preliminary results would significantly strengthen the paper's claims. In Figure 2(a), more sophisticated causal methods (DAS, SAE) show 2-4× higher divergence than mean-difference patching. Do these methods also achieve higher counterfactual accuracy? 

Q2. Example I justification (relates to W2): In Example I, the balanced subspace contributes zero to outputs under natural conditions (as noted in lines 279-280). Can you clarify what makes intervention on such dimensions a case of "pernicious divergence" rather than simply "intervening on wrong (non-causal) variables"? How is this different from general causal identification errors?

Q4. Practical detection (relates to W4): Beyond collecting large evaluation datasets, can you suggest even a heuristic method for practitioners to detect when divergence is likely to be pernicious versus harmless in their specific applications?

### Soundness
2

### Presentation
2

### Contribution
2
