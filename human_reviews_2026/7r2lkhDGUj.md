# Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Mixture-of-Experts (MoE) has become a dominant architecture for scaling Large Language Models (LLMs) efficiently by decoupling total parameters from computational cost. However, this decoupling creates a critical challenge: predicting the model capacity of a given MoE configurations (e.g., expert activation ratio and granularity) remains an unresolved problem. To address this gap, we introduce Efficiency Leverage (EL), a metric quantifying the computational advantage of an MoE model over a dense equivalent. We conduct a large-scale empirical study, training over 300 models up to 28B parameters, to systematically investigate the relationship between MoE architectural configurations and EL. Our findings reveal that EL is primarily driven by the expert activation ratio and the total compute budget, both following predictable power laws, while expert granularity acts as a non-linear modulator with a clear optimal range. We integrate these discoveries into a unified scaling law that accurately predicts the EL of an MoE architecture based on its configuration. 
To validate our derived scaling laws, we designed and trained MoE-mini, a model with only 0.85B active parameters, alongside a 6.1B dense model for comparison. When trained on an identical 1T high-quality token dataset, MoE-mini matched the performance of the 6.1B dense model while consuming over 7x fewer computational resources, thereby confirming the accuracy of our scaling laws.
This work provides a principled and empirically-grounded foundation for the scaling of efficient MoE models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Efficiency Leverage (EL), a metric designed to compare the training efficiency of sparse Mixture of Experts (MoE) versus dense Transformer models. The authors' primary contribution is developing a scaling law that expresses EL as a function of model architecture and compute budget and validating it on a moderately large computational scale.

The methodology proceeds in three phases:

1. **Establishing a testbed**: The authors develop predictors for optimal training hyperparameters, which allows them to reduce bias in the subsequent scaling law analysis.

2. **Optimal model/data allocation study**: They investigate how to optimally allocate model size and dataset size given a fixed compute budget, separately for MoE and dense architectures.

3. **Scaling law formulation**: Using 1. and 2., they derive and fit a joint scaling law across three variables: model sparsity, expert granularity, and total training budget (measured in FLOPs). They study marginal scaling behavior with respect to each variable individually before formulating the joint relationship.

The authors validate their scaling law in two ways. First, they present plots showing fit quality on a held-out validation set, though without goodness-of-fit metrics. Second, and more importantly, they conduct two large-scale training runs—one dense model and one sparse MoE model—trained on identical tokens using compute budgets significantly larger than those used to fit the scaling law. The validation succeeds: the models achieve comparable performance, yet the dense model requires more than 7× the active parameters, confirming the scaling law predictions (though I need explanation from the authors on the details).

Additionally, the paper analyzes how architectural choices affect scaling behavior, including the number of shared experts, expert granularity, and the proportion of attention versus MLP FLOPs.

### Strengths
**Conceptual contribution:**

EL has the potential to be a useful metric that simplifies efficiency comparisons between MoE and dense architectures. However, the metric needs to be precisely defined, and its usage throughout the paper must remain consistent with that definition, with any claims properly justified.

**Empirical validation:**

The validation experiments successfully corroborate the paper's claims. The authors train models at significantly larger scales than those used to fit the scaling law, and the predictions hold. This is a strong empirical result.

**Methodology:**

1. **Structured experimental foundation**: The authors take a systematic approach to building the foundation for EL scaling law experiments. They follow existing literature to develop two key components:
   - A system for choosing optimal hyperparameters with respect to compute budget. While they apply insights from only one MoE configuration to other architectures, they validate this transfer "in the vicinity" of 16× sparse MoE. The dense architecture is treated separately, so all comparisons are fair with regards to choice of core training hyperparameters.
   - Scaling laws for efficient budget allocation, once again done separately for dense and Moe models.

2. **Comprehensive architectural analysis**: Before deriving EL scaling laws, the authors systematically study key architectural factors including expert activation ratio, expert granularity, and shared expert ratio. This provides a solid foundation for understanding how these design choices affect the scaling behavior.

### Weaknesses
**Major Issue: Imprecise definition of EL and its inconsistent use throughout the paper, lack of clarity in experimental setup**

The definition of Efficiency Leverage (lines 186-188) lacks precision, and its usage throughout the paper creates confusion for readers. Several issues arise:

1. **Unclear functional form**: The formal definition of EL makes it a function of three arguments: MoE architecture, MoE compute budget, and dense architecture. However, the authors' usage throughout the text suggests a different understanding. Consider these statements:

   - Lines 16-17: "a metric quantifying the computational advantage of an MoE model over a dense equivalent"
   - Lines 25-27: "When trained on an identical 1T high-quality token dataset, MoE-mini matched the performance of the 6.1B dense model while consuming over 7× fewer computational resources, thereby confirming the accuracy of our scaling laws"
   - Lines 70-72: "an EL of 5, for example, means the MoE architecture performs like a dense model with five times the active parameters for a similar compute budget"

   These statements imply EL is an architecture-level property, independent of compute budget. Yet even under favorable assumptions (e.g., Chinchilla-like scaling), given the current definition, EL remains a function of compute budget and the chosen dense architecture.

2. **Missing justification**: Lines 185-189 make claims that do not clearly follow from the stated definition. To improve this, the authors should provide a complete derivation with explicit assumptions and literature references showing how EL reduces to the simpler form.

3. **Section 4 methodology unclear**: 
   - How were the datapoints for EL obtained? What is the exact procedure for training dense models to reach the precise training loss of a given MoE run?
   - Figure 4 shows both fitted curves and datapoints, but their nature is ambiguous. Are these points from actual experiments or predictions? Figure 3 shows models up to ~3×10²⁰ FLOPs, yet Figure 4 plots extend to at least 1×10²² FLOPs (acknowledging that 4c is explicitly an extrapolation). If the points are predictions, this is misleading—the notation is not clearly specified, and the marked points suggest real experiments. If they are actual experiments, then the final validation run at 1×10²² FLOPs is not truly an extrapolation beyond the fitting range, which diminishes the claimed strength of the scaling law validation.

**Technical issues:**

- In the EL definition, the variable $C_{target}$ is never used and should be removed.
- The use of ε → 0 in the constraint is redundant. As ε approaches zero, this becomes an equality constraint. Why not write it as an equality directly?
- Sections 4.2.1 and 4.2.2. are concerned with fitting scaling laws for each variable separately, keeping the other two fixed, and fitting the joint form, respectively. Thus, the name of 4.2.1 should rather be called either marginal or univariate scaling laws. Separable means its already a joint function (so, of all variables), but it's of a special form functional form.
**Unsupported claim:**

>Lines [249-251]: "It represents a critical balance: [...] while overly coarse-grained ones fail to achieve effective specialization."

This claim about expert granularity is stated as fact but lacks supporting analysis or evidence in the paper. The only thing that is supported is that increased granularity deteriorates model quality from some point on - though, judging by the analysis in Figure 13, it is possible that simply having the balancing loss be a function of granularity (more experts -> more balancing needed) will make granularity once again effective.

### Questions
# Questions for Authors

1. **What is the precise definition of EL?** Can you provide a complete mathematical definition that reconciles the formal definition in lines 186-188 with the simpler characterizations used throughout the paper (e.g., "computational advantage" or "performs like a dense model with N times the active parameters")? Under what assumptions does EL become independent of compute budget?

2. **How did you generate the (MoE, dense) run pairs for EL datapoints?** What is the exact methodology for pairing runs? Do you train dense models specifically to match the loss of each MoE run, or do you use some form of interpolation/prediction?

3. **Do you drop tokens at any point?** Please clarify whether token dropping occurs and if any expert capacity mechanism is implemented during training, evaluation, or both.

4. **What is the quality of fit for your scaling laws?** The plots show that test performance appears worse than train performance, but there is no discussion of fit quality. Did you perform a detailed analysis similar to, for example, Besiroglu et al. (2024)? Specifically:
   - What are the RMSE values for train and test?
   - How does fit quality vary across different dimensions (model size, granularity, sparsity)?
   - What do the marginal distributions look like?
   - Are there systematic patterns in the residuals?
5. **How was load balancing implemented?** Was it run locally, or were the routing statistics all-reduced across e.g. the expert parallel group or globally? Please provide implementation details.
6. **What is your intuition behind granularity performing worse beyond a certain point?** The paper states that "excessively fine-grained experts suffer from insufficient capacity, while overly coarse-grained ones fail to achieve effective specialization." What evidence or analysis supports this claim? I was only able to find an anlysis on model quality (training perplexity), but without any analysis on specialization or otherwise.
---

## References

Besiroglu, T., Erdil, E., Barnett, M., & You, J. (2024). Chinchilla Scaling: A replication attempt. arXiv preprint arXiv:2404.10102.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces 'Efficiency Leverage' (EL) to quantify the computational advantage of MoE models over dense models . It comprehensively investigates the scaling laws for MoE models, analyzing key factors such as expert granularity and activation ratio (sparsity). The study is supported by extensive data and figures and validates its findings by training a high-performing MoE model (MoE-mini) that confirms the predicted efficiency gains. The work provides a practical guide for MoE model design.

### Strengths
1. This work's originality stems from its formulation of "Efficiency Leverage" (EL), a clear metric to quantify MoE computational advantage . It uses this concept to build a unified scaling law for EL, connecting it to the compute budget, activation ratio, and granularity.

2. The empirical quality is high, supported by over 300 trained models. A key strength is the preliminary work in Section 2, which derives MoE-specific scaling laws for optimal hyperparameters and data allocation . This step ensures a fair comparison, addressing a methodological flaw in prior studies.

3. The paper is clearly structured, and its claims are well-supported by data. The final validation, MoE-mini, successfully confirmed the law's >7x efficiency prediction . This demonstrates the work's significance as a practical guide for designing future efficient MoE models.

### Weaknesses
1. The conclusion that efficiency monotonically increases with sparsity (Key Takeaway 1) is based on a theoretical FLOPs model. This omits the practical wall-clock costs of routing, communication (e.g., all-to-all), and memory bandwidth for loading many distinct expert weights, which can become bottlenecks at high sparsity.

2. The functional forms for the scaling laws (Eq. 2, 3, 4, and 5) are presented without strong justification. It is not clear why these specific complex forms (e.g., log-polynomial for $G$, saturating transform for $A$) were chosen over simpler alternatives, or how the goodness-of-fit was evaluated.

### Questions
1. The derivation of optimal hyperparameters in Section 2.2 scales them as a function of total compute $C$, adapting a methodology from dense models. However, the paper notes that MoE backpropagation is different, as experts see only a subset of tokens . Why should the optimal batch size and learning rate scale only with total $C$, rather than also depending on MoE-specific factors (like activation ratio) that directly influence gradient statistics for each expert?

2. Could the authors provide more details on the fitting process for the scaling laws (Eq. 2, 3, 5)? Specifically, what was the rationale for choosing the log-polynomial form for granularity ? Were simpler functional forms tested, and how did their fit quality compare?

3. Regarding the finding that EL improves as activation ratio decreases (Key Takeaway 1) : Do the authors expect this trend to break down at a certain point when considering practical implementation overheads (e.g., wall-clock time) instead of just theoretical FLOPs?

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
4

### Summary
This paper studies the effectiveness of training MoE versus dense models from the compute perspective.

First, a detailed ablation study is performed to search for the best architecture for training an MoE, considering shared experts, granularity, sparsity.
Then, scaling laws of the leverage (gain of performance wrt dense model) are established by considering the above factors independently.
Finally, the model is trained at the scale of around 20B total-1B active MoE to validate the performances.

### Strengths
- The study is of large scale (20B moe and 1T tokens) and detailed ablation studies justify their choices of architecture.
- A principled approach (scaling law) is used to tune the LR and batch size, further solidifying the argument for the proposed leverage scaling law.
- While the optimal configuration is quite impossible due to many hyperparameters, this paper still provides valuable and principled insights into how to scale the MoE.

### Weaknesses
- The experimental design for analyzing the Activation Ratio (A) (Section F.1, Table 8) appears structurally limited as it holds the active computational cost nearly constant.
  - The study varies the Activation Ratio (sparsity) exclusively by changing the total number of routable experts (E) while fixing the number of activated experts (Ea=2) and shared experts (Es=1). This approach means the study explores *only* the effect of increasing total parameters without changing the active expert.
- Formulating the scaling law in terms of the efficiency leverage is not very useful. This means that a user would have to first compute the dense model scaling law, before applying it to the proposed scaling law to figure out the MoE scaling behavior. A direct formula relating only the MoE configuration to the loss (eg [2501.12370]) would be more useful.
- Codes are not available (or a description of it if possible).
  - Training configuration is lacking as well (e.g., what kind of parallelism techniques is used etc)

### Questions
- How many GPU hours are used in the study?
- What kind of software (and techniques, like parallelisms) is used for training?
- What could be the reason for the difference wrt Olmoe in terms of the effectiveness of shared expert? In the Olmoe paper ([2409.02060]), it is shown that using shared expert is not effective.
- Using different number of head/ kv heads affects the number of model parameters. How does changing the kv head affect the results?
  - In table 8 it seems like the number kv head changes with size. What is the justification for it?
- Since both model size (N) and dataset size (D) are varied in the experiments, I wonder if the scaling law could be described in terms of N and D, which could be more expressive than using only compute (which is roughly 6*N*D), i.e., something similar to the chinchila-like scaling law?

### Soundness
3

### Presentation
4

### Contribution
3
