# Fantastic Reasoning Behaviors and Where to Find Them: Unsupervised Discovery of the Reasoning Process

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Despite the growing reasoning capabilities of recent large language models (LLMs), their internal mechanisms during the reasoning process remain underexplored. Prior approaches often rely on human-defined concepts (e.g., overthinking, reflection) at the word level to analyze reasoning in a supervised manner. However, such methods are limited, as it is infeasible to capture the full spectrum of potential reasoning behaviors, many of which are difficult to define in token space. In this work, we propose an unsupervised framework (namely, RISE: Reasoning behavior Interpretability via Sparse auto-Encoder) for discovering reasoning vectors, which we define as directions in the activation space that encode distinct reasoning behaviors. By segmenting chain-of-thought traces into sentence-level 'steps' and training sparse auto-encoders (SAEs) on step-level activations, we uncover disentangled features corresponding to interpretable behaviors such as reflection and backtracking. Visualization and clustering analyses show that these behaviors occupy separable regions in the decoder column space. Moreover, targeted interventions on SAE-derived vectors can controllably amplify or suppress specific reasoning behaviors, altering inference trajectories without retraining. Beyond behavior-specific disentanglement, SAEs capture structural properties such as response length, revealing clusters of long versus short reasoning traces. More interestingly, SAEs enable the discovery of novel behaviors beyond human supervision. We demonstrate the ability to control response confidence by identifying confidence-related vectors in the SAE decoder space. These findings underscore the potential of unsupervised latent discovery for both interpreting and controllably steering reasoning in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an unsupervised framework for explaining the reasoning behavior of LLMs: Reasoning behavior Interpretability via Sparse auto-encoder (RISE). This framework explains the reasoning behavior by training sparse auto-encoders to distinguish inference vectors. Experiments show that reflection and backtracking unpacking functions occupy separable regions in the decoder column space. The paper also analyzes the impact of targeted interventions, response length, and other factors on the model's reasoning behavior.

### Strengths
1. This paper proposes an unsupervised framework for interpreting reasoning behavior, which requires no human intervention and can intuitively reflect the inference vectors during model reasoning.

2. This paper investigates the model's reasoning behavior through ample experiments. The results suggest that reflection and backtracking unraveling functions occupy separable regions in the decoder column space. Furthermore, the paper analyzes several factors influencing the model's reasoning behavior.

### Weaknesses
1. Limited Domain Focus: The experimental section of the paper only used the Mathematics dataset (MATH) to analyze mathematical reasoning behavior. Adding reasoning tasks from other domains, such as common-sense reasoning and logical reasoning, might strengthen the experiment's persuasiveness.

2. In the experiments in Section 4.3, regarding the experimental results in Figure 3, the paper explains the statement that "the comparison between reflection and backtracking shows only moderate differences, while the separation of 'other' behaviors from reflection or backtracking is significantly stronger" as "reflection and backtracking occupy more overlapping representational subspaces." Are there any experimental results to support this conclusion?

### Questions
Please see the Weaknesses.

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
4

### Summary
The paper presents RISE, an unsupervised framework based on sparse auto-encoders (SAE) in large language models. It aims to identify reasoning vectors in the activation space that correspond to behaviors such as reflection and backtracking, showing that these occupy separable regions in the SAE decoder space. It further demonstrates causal intervention, enabling targeted modulation of reasoning behaviors and model confidence beyond explicit human supervision.

### Strengths
1. The paper is well-structured, logical, and clearly written, making the complex concepts easy to follow.
2. The core idea of using SAEs to analyze the reasoning patterns of LLMs is novel and interesting. The demonstration that manipulating these disentangled features leads to a clear and observable enhancement or suppression of the corresponding reasoning behaviors is a significant strength.
3. The experiment to identify "confident reasoning vectors" by optimizing for entropy in the SAE decoder space is compelling. The subsequent intervention, which demonstrably boosts the model's confidence.

### Weaknesses
- In Figure 2, the separation between the *Reflection* and *Backtracking* clusters does not appear very distinct, and the normalized Silhouette scores in Figure 3 peak at only around 0.6, which is close to the boundary of meaningful separation. It would be helpful to clarify whether the clustering in the decoder space can be considered sufficiently reliable under these conditions.
- In Section 4.5, the paper suggests that cluster formation in the mid-to-late layers reflects response length, but this relationship is not clearly visible from the figures. Including a quantitative analysis (such as the Figure 3) could better support the claimed link between cluster compactness and response length.
- Figures 6 and 8 indicate that the c*onfidence vectors* may influence reasoning in a way similar to r*eflection* and b*acktracking vectors*. It might be useful to analyze the similarity or correlation among these three groups to determine whether “confidence” represents a distinct factor or overlaps with the other two behaviors.
- The experiments are currently limited to a single model (DeepSeek-R1-Distill-Qwen-1.5B). Extending the analysis to other models, such as the base Qwen series or architectures like Llama, would help demonstrate the consistency and robustness of the findings.
- If the authors can address these concerns clearly in the rebuttal, I would be inclined to raise my overall rating.

### Questions
See weakness

### Soundness
4

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
3

### Summary
This paper proposes RISE, an unsupervised framework that trains sparse auto-encoders on sentence-level chain-of-thought activations to discover “reasoning vectors”, which are directions in activation space that encode interpretable reasoning behaviors.  The paper shows these vectors form coherent clusters, and that test-time interventions along the vectors can amplify or suppress behaviors, changing inference trajectories without retraining. They further observe a structural axis related to response length and identify confidence vectors by optimizing for low entropy, which shifts outputs toward more confident reasoning.

### Strengths
1. Unsupervised SAEs on step-level activations reveal interpretable reasoning vectors and enable causal behavior steering without retraining.

2. The method RISE is coherent with plausible identifiability arguments and consistent cross-benchmark evidence supporting the main claims.

3. The findings are valuable for interpretability and practical control of LLM reasoning.
The paper presents a clean end-to-end pipeline with intuitive figures and explanations that make replication straightforward.

### Weaknesses
1. The paper claims that the behavior labels rely on an LLM-as-judge, but does not demonstrate robustness to model choice or prompt wording.

2. The composition of multiple behavior vectors may introduce interference due to non-orthogonality.

3. The identifiability of RISE claim depends on sparsity and incoherence assumptions, which are not empirically validated on real activations in this paper.

4. The experiments are confined to math reasoning, raising my concerns about domain generality.

### Questions
1. Could the authors provide cross-annotations, e.g., multiple annotators on the same samples, to demonstrate consistency and robustness?

2. Could the authors demonstrate RISE’s performance in other domains, such as code generation, to test transferability?

### Soundness
3

### Presentation
4

### Contribution
4
