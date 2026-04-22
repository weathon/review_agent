# Beyond Redundancy: Diverse and Specialized Multi-Expert Sparse Autoencoder

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 4, 8

## Abstract
Sparse autoencoders (SAEs) have emerged as a powerful tool for interpreting large language models (LLMs) by decomposing token activations into combinations of human-understandable features. While SAEs provide crucial insights into LLM explanations, their practical adoption faces a fundamental challenge: better interpretability demands that SAEs' hidden layers have high dimensionality to satisfy sparsity constraints, resulting in prohibitive training and inference costs. Recent Mixture of Experts (MoE) approaches attempt to address this by partitioning SAEs into narrower expert networks with gated activation, thereby reducing computation. In a well-designed MoE, each expert should focus on learning a distinct set of features. However, we identify a *critical limitation* in MoE-SAE: Experts often fail to specialize, which means they frequently learn overlapping or identical features. To deal with it, we propose two key innovations: (1) Multiple Expert Activation that simultaneously engages semantically weighted expert subsets to encourage specialization, and (2) Feature Scaling that enhances diversity through adaptive high-frequency scaling. Experiments demonstrate 24\% lower reconstruction error and reduced feature redundancy compared to existing MoE-SAE methods. This work bridges the interpretability-efficiency gap in LLM analysis, allowing transparent model inspection without compromising computational feasibility. 
Our code is publicly available at https://anonymous.4open.science/r/scale_sae-C6D0/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces multi expert SAEs (as opposed to single expert SwitchSAEs). They further introduce feature scaling as a mean of encouraging specialisation and reducing feature redundancy. They run evaluations on gpt2 on fraction of loss recovered, mse, interpretability, and redundancy of features. They show improvements again some competing architectures of SAEs.

### Strengths
The introduction of weighted MoE SAEs is novel, and feature scaling is effective as a way to enhance specialisation (which could be tied to interpretability, as an interesting future direction). 
There are many evaluations criteria, and improvements are shown across them. 
The problem of SAE feature interpretability, and the improvement of SAEs are well located problems within the literature.

### Weaknesses
The main limitations are:
- evaluations restricted to gpt2
- the improvements on L0 against MSE/loss look incremental
- only.3 architectures (topk/switch/gated) are tested against (more baselines could help, such as matryoshka or jumprelu saes)

I would increase my score if evaluations were successfully (more than incremental improvements) run on more models/families.

Style:
- figure 3 looks cluttered (make all your saes one color and all the baselines another)

### Questions
Have you tried expanding evaluations to other models or families (even small ones like pythia 70m)?

Is fraction of loss recovered the LLM's loss or the SAE's EV?

Have you tried experiments linking the specialisation of features to their interpretability?

### Soundness
2

### Presentation
2

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
The authors suggest an improvements to SAEs by using MoEs extending Mudide et al 2025. They suggest that the original approach to MoE SAEs was flawed and had polysemanticity which undermines the interpretability that the approach was intended to provide. Instead of choosing a single expert, they choose multiple experts with their Multiple Expert Activation scheme. They then have a DSP-inspired technique Feature Scaling which amplifies the component of the expert encoder's weights which is far from the mean of the encoder vectors. They refer to this as "amplifying the high-frequency component". With these changes their SAEs outperform TopK SAEs and the Switch SAE. The primary obstacle to performance from previous work that they solve is the problem of feature redundancy. The authors perform ablations to confirm that both of these changes are important in training more performant SAEs.

Overall this is a solid and well argued interpretability paper, though possibly incremental in terms of its impact to the interpretability (and broader ML) community.

### Strengths
- Clear writing style
- Generally readable figures
- Clearly notes the limits of the Switch SAE and how they overcome these
- Useful to see the performance on two different datasets across both the MSE and Loss Recovered metrics
- Interesting exposition when detailing how the two architecture changes help the overall performance of the SAE
- Valuable use of the signal processing literature which is a literature that is not always leveraged in interpretability research (and where interpretability research could likely learn a lot more from)

### Weaknesses
- For interpretability researchers reading the authors might want to be careful about using the term "high frequency" without explaining what this means. In this case it seems to be mostly an analogy to the signal processing literature but within the SAE literature a high frequency feature is typically a feature which activates very often which is a quite different concept. Clarifying this would be useful.
  - It's also not totally clear why this analogy is a good analogy - exploring why "high-frequency" is the right term here (possibly with spectral plots or similar) could be valuable to help with readers' intuition. 
- Though the authors talk about computational efficiency being an advantage of their approach, they do not show any charts or tables tracking the computational efficiency of their approach relative to others. Seeing this would be useful to validate that claim. 
- The authors do not evaluate their results on downstream tasks. Doing so (possibly using SAE-Bench or a similar benchmark) would be an improvement to the work demonstrating the downstream usefulness across other metrics. 
- The work is somewhat incremental. Though I believe their results are an improvement over prior methods, it's not clear that the level of improvement that is presented will meaningfully impact the research community. This is not to take away from the interesting methodology, clear presentation and reasonable ablations. 
- All of the experiments are using GPT-2. Having at least one plot in the paper with results from a larger model would show that this method works well on larger models and that it scales well. in particular this is important because one of the main claims of the paper is about efficiency.

### Questions
- What is happening in Figure 5a)? The title of the charts suggest an L0 of 2 and 32 yet the bar and line charts show different values of L0 on the x axis. Is the x axis actually varying the number of experts?
- In 3.2.1, why would there be diminishing returns to having more than 2 experts active at once? Given that you're not actually using all of the features but are instead using another TopK filter afterwards (so the FLOPs are not increasing) it seems still to explain why 2 experts is markedly better than 1 but 4 isn't seemingly better than 2. 
    - This seems to be somewhat mitigated in 3.2.2 but I would like to know more of why this mitigation ought to work from the authors
- Does Feature Scaling ever amplify noise or cause training instability?
- What are the training and inference time memory implications for having the MoE layer? Is there a way to reduce the footprint of this compared to naive methods?
- Is the idea that the low frequency features are more coarse-grained and high-frequency ones flesh out more complex details? If so is there any evidence for this?
- The abstract states that the approach gives 99% less feature redundancy - where in the paper is this claim justified?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies a critical limitation in Mixture-of-Experts Sparse Autoencoders: the failure of experts to specialize, leading to high feature redundancy that undermines both interpretability and performance. The authors propose Scale SAE, a novel framework with two core innovations:
- Instead of routing an input to a single expert, a subset of experts is activated. 
- Encoder weights are decomposed into low-frequency and high-frequency components, and the high-frequency parts are adaptively amplified.

Through extensive experiments on GPT-2, the paper demonstrates that Scale SAE significantly outperforms strong baselines (TopK SAE, Gated SAE, Switch SAE) under a FLOPS-matched paradigm.

### Strengths
The methodology is explained with precise mathematical notation, and the results are presented with effective visualizations.

The experimental evaluation is thorough and convincing. The use of FLOPS-matched comparisons, multiple datasets (in-domain and cross-domain), and a suite of complementary metrics leaves little doubt about the superiority of the proposed method. The ablation studies and mechanistic analysis are executed to a high standard.

### Weaknesses
> W1. The Mechanistic Rationale for Multiple Expert Activation Requires Deeper Justification. 

To be very honest, activating more than one expert in MoE is standard practice. The difference here is that we select the Top-K experts across all experts.

The paper shows that activating multiple smaller experts outperforms a single larger expert under a FLOPS-matched budget (e.g., 2 experts of size 128 vs. 1 expert of size 256). However, the fundamental reason for this performance boost is not sufficiently explained. A key question remains: is the benefit primarily due to the modularity and finer granularity of the experts, or is it the interaction and joint sparsification across experts that is crucial? For example, consider a 4-expert SAE (activating 2 at a time, each with 128 hidden units) versus a 2-expert SAE (activating 1 at a time, each with 256 hidden units). The computation is identical, so why should reconstruction quality, accuracy, and stability improve? The paper shows results in Figures 3 and 4, but does not clearly explain the underlying logic.

> W2. Insufficient Discussion and Comparison for Feature Scaling. 

Feature Scaling appears primarily as a load-balancing technique, both at the expert and neuron levels. Adding high-frequency components increases directional diversity, but alternative methods exist. 

At the expert level, how does the implicit balancing effect of Feature Scaling compare to more explicit load-balancing losses used in MoE literature?

At the neuron/feature level, orthogonal initialization or updates could achieve similar effects.

These alternatives are not discussed or compared. While the quantitative results in Section 3.3 make sense, the paper lacks a deeper analysis of why this approach is preferable.

### Questions
From Figure 6(a), adding more than two experts does not further improve specialization—in fact, performance worsens. This seems inconsistent with the stated property. Why is that?


I wonder if adding high-frequency components requires longer training to converge, since it perturbs the optimization direction. Is that correct? Could you also compare training iterations and time?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the problem of feature redundancy and poor expert specialization in Mixture-of-Experts Sparse Autoencoders (MoE-SAEs), which are used to make large language models more interpretable while reducing computational cost. Prior work such as Switch SAEs often suffers from redundant experts learning overlapping features, limiting interpretability and efficiency. The authors propose two key innovations: (1) Multiple Expert Activation, which activates several experts per input and applies a global Top-K sparsity constraint to encourage expert specialization, and (2) Feature Scaling, a learnable high-frequency amplification mechanism that promotes feature diversity and stabilizes training. Experiments on GPT-2 activations show that these methods improve reconstruction error, feature diversity, and automated interpretability scores relative to baseline SAEs. Ablation studies attribute these gains to the proposed mechanisms.

### Strengths
The paper is well motivated, clearly written, and deeply engages with prior work. The authors correctly identify a central limitation of existing MoE-SAEs and propose two simple, conceptually coherent mechanisms to address it. Both techniques are well defined and integrated cleanly into the SAE framework. The experimental evaluation is thorough, including ablation studies that isolate the contribution of each innovation. The results convincingly demonstrate reductions in feature redundancy and improved interpretability metrics. Overall, the work provides a solid step toward making large-scale sparse autoencoders more computationally feasible for interpretability research.

### Weaknesses
The experiments are limited to GPT-2, which is now a dated architecture. Including results on more recent models such as Gemma or LLaMA would strengthen the empirical claims and test generality.
The FLOPs-matching procedure is not clearly justified. The authors write that “to match the computational load of activating a fixed number of experts, the hidden dimension is set to 768” for dense SAEs, while Scale SAEs use a total hidden dimension of 24,576. It is unclear how this setup maintains computational parity, and a more detailed explanation of this comparison is needed.
Finally, the discussion of automated interpretability results is too brief. The paper would benefit from a few qualitative examples of discovered features or a quantitative measure of feature diversity across experts, to demonstrate that higher automated interpretability scores correspond to genuinely more distinct, human-understandable features.

### Questions
1. Could the authors clarify how the FLOPs-matching setup ensures a fair comparison between dense SAEs with 768 hidden units and Scale SAEs with a total of 24,576 dimensions? Is the compute matched per forward pass, per batch, or by total parameter count?

2. Did you observe any side effects of the Feature Scaling mechanism, such as instability, changes in sparsity dynamics, or degraded interpretability for certain expert configurations? Since this mechanism directly modifies encoder weights, a brief discussion of possible unintended effects would be helpful.

### Soundness
3

### Presentation
3

### Contribution
3
