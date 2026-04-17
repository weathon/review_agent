# Metanetworks as Regulatory Operators: Learning to Edit for Requirement Compliance

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
As machine learning models are increasingly deployed in high-stakes settings, e.g. as decision support systems  in various societal sectors or in critical infrastructure, designers and auditors are facing the need to ensure that models satisfy a wider variety of requirements (e.g. compliance with regulations, fairness, computational constraints) beyond performance. Although most of them are the subject of ongoing studies, typical approaches face critical challenges:  post-processing methods tend to compromise performance, which is often counteracted by fine-tuning or, worse, training from scratch, an often time-consuming or even unavailable strategy. This raises the following question: "Can we efficiently edit models to satisfy requirements, without sacrificing their utility?" In this work, we approach this with a unifying framework, in a data-driven manner, i.e. we learn to edit neural networks (NNs), where the editor is an NN itself - a graph metanetwork - and editing amounts to a single inference step. In particular, the metanetwork is trained on NN populations to minimise an objective consisting of two terms: the requirement to be enforced and the preservation of the NN's utility. We experiment with diverse tasks (the data minimisation principle, bias mitigation and weight pruning) improving the trade-offs between performance, requirement satisfaction and time efficiency compared to popular post-processing or re-training alternatives.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Paper proposes a graph metanetwork that edits trained MLPs in one forward pass to satisfy requirements (data minimization, equalized-odds fairness, pruning) while preserving behavior. It outputs masks and residuals, trained over model families with JSD + requirement losses, yielding faster, better Pareto trade-offs on Adult/Bank.

### Strengths
1. Motivation is well-argued and the paper is clearly written, with a crisp problem setup and method description.

2. Covers three compliance needs—data minimization, equalized-odds fairness, and pruning—with promising Pareto trade-offs and extremely fast inference (~0.03s per edit).

3. Accommodates different parameter shapes by operating on a parameter graph, rather than assuming a fixed MLP architecture.

### Weaknesses
1. For each dataset, each requirement, and each weighting coefficient, a separate model must be designed and trained.

2. Given the data-hungry training (needing many trainings to construct weight samples) and poor cross-dataset generalization, the claim of “no costly retraining” seems hard to stand.

3. The datasets are extremely limited and very low-dimensional, which makes me worry about applicability to higher-dimensional real-world settings.

### Questions
I’d like to know how it performs on higher-dimensional datasets, and how much the final results depend on the number of pre-trained weight samples (weight snapshots) used for training？

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the problem of ensuring ML models comply with evolving requirements such as fairness, privacy, and efficiency without retraining or performance degradation. It identifies that existing post-processing and fine-tuning approaches are inefficient or limited to specific applications, highlighting a gap in flexible, general-purpose model editing. 
To address this, the authors propose a graph metanetwork that learns to edit other neural networks’ parameters directly, performing requirement-compliant modifications in a single inference step. The framework formulates requirement compliance as a multi-objective optimization problem balancing performance preservation and regulatory constraints, trained in a data-driven and symmetry-aware manner. Experiments across data minimization, bias mitigation, and pruning show the method achieves better trade-offs between performance, compliance, and computational efficiency than retraining or post-processing baselines.

### Strengths
1.	The motivation is strong and well-timed, the paper tackles a pressing issue in trustworthy AI, how to make models meet regulatory and ethical requirements post-deployment. 
2.	The metanetwork architecture is innovative: using an equivariant GNN to operate in weight space is a cutting-edge idea that leverages recent advances in meta-learning and symmetry-preserving design.
3.	Experiments are thoughtfully structured across diverse requirements (fairness, data minimization, pruning), which convincingly demonstrate versatility rather than one task.

### Weaknesses
1.	The scope of experiments is limited to MLPs and tabular data. This makes it unclear how well the method scales to large models (e.g., Transformers, diffusion, or other large models) or structured modalities such as text and vision.
2.	The assumption of white-box access to model weights may restrict applicability in auditing or proprietary systems, where only API access is available.
3. The training cost of the metanetwork itself, though amortized at inference, is not deeply analyzed. It remains uncertain how expensive or data-hungry the pre-training process is compared to fine-tuning.
4. It would be beneficial if further interpretability analysis and error analysis could be involved.

### Questions
1. The paper mainly focuses on differentiable objectives, how to consider other requirement types like safety constraints, robustness?
2. How realistic is this in common auditing or deployment settings, where many models are only available via APIs?
3. Can the metanetwork generalize across tasks?
4. How sensitive is performance to the diversity and size of this NN population?

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
The authors present an ambitious and, at first glance, quite elegant technical approach. They propose a framework in which a message-passing graph neural network learns to edit other neural networks, making them compliant with new requirements without retraining them from scratch. This is a neat trick: it’s like giving neural networks a "regulatory layer" that can adapt them to new goals on the fly.

I’m generally pleased with the conceptual elegance. It’s a clever piece of engineering and a compelling idea. However, there are a few important caveats that need to be highlighted.

First, let’s talk about the question of guarantees. The authors imply that their method can be used to achieve various goodmaking properties fairness, efficiency, data minimization, and so on but they sidestep the hardest part of the problem. The guarantees don’t come from the method itself. In fact, the method doesn’t really offer any inherent guarantees at all. It’s up to whoever is translating those goodmaking properties into mathematical form to ensure that the objectives make sense and are actually achievable. The authors effectively offload the burden of proof to someone else: they say, "Once you’ve done the hard work of turning your requirement into math, then our method applies." This is a notable limitation that's worth being explicit about: it’s a tool that requires someone else to do the heavy conceptual lifting, and it's worth avoiding the implication that mathematics alone can buy you these solutions for free.

Second, the method’s applicability is limited by its data and architecture dependence. It works for fixed architectures, which means if you want to generalize it, you have to retrain it for each new type of network. That’s a practical hurdle that limits the broader usability of the technique. It’s a nice trick, but it’s not a universal one-size-fits-all solution.

Finally, the authors bury certain limitations in footnotes, such as the assumption of a convex Pareto front. In the messy real world, multi-objective optimization often doesn’t yield such neat, convex trade-offs. This is a real limitation that should be more transparently discussed.

In conclusion, this is a clever and promising piece of work, but it comes with caveats. It’s a tool that needs careful handling and further demonstration of its generality and robustness. The authors have given us a neat technical insight, but they haven’t solved the hardest parts of the problem. That’s fine, just means there’s more work to be done.

### Strengths
- Elegant formulation of model editing as a learnable mapping in weight space.
- Uses graph metanetworks to perform edits in a single inference step: computationally efficient in a sense.
- Unifies multiple compliance tasks (fairness, pruning, data minimisation) under one objective.
- Demonstrates consistent Pareto improvements over post-hoc baselines.

### Weaknesses
- Depends on fixed architectures; generalization across model families untested.
- Offloads hardest work theoretical work (formalising requirements) onto the user (may not be a weakness per se)
- Assumes convex Pareto fronts and differentiable requirement objectives; necessary move perhaps, no shame in just saying "here is a nice technical trick, let's not worry about guarantees."
- “Regulatory compliance” rhetoric overstates what is actually a tuning heuristic.

### Questions
- How sensitive is the learned editor to the distribution of training networks? Does the universe conspire in our favour to make all editors behave roughly the same?
- Can metanetworks be stacked or composed for multiple simultaneous requirements? Sure there are no theoretical guarantees but does it work in the wild?
- What safeguards prevent catastrophic edits or hidden performance degradation? Is there a meta-meta-network for that?

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
The paper proposes a metanetwork that edits trained models in a single pass to meet requirements (data minimization, fairness, sparsity), replacing intensive optimization loops. Framing compliance as a learned weight-space mapping is well-motivated.

### Strengths
It's good to see the evaluation of how robust the edited models remain when the underlying data distribution (pd) shifts.

### Weaknesses
The regulatory positioning could be more precise by linking edits to specific AI Act or ISO/IEC risk-management processes. The relation to unlearning is better reframed as complementary rather than alternative, since this method edits parameter rather than removing data influence.

### Questions
The paper doesn’t explain how λ is chosen or how sensitive results are to it. Specify the sampling strategy for the Dd subset.

### Soundness
3

### Presentation
3

### Contribution
2
