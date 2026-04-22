# Encode, Think, Decode: Scaling test-time reasoning with recursive latent thoughts

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Most efforts to improve the reasoning capabilities of large language models (LLMs) involve either scaling the number of parameters and the size of training data, or scaling inference computation by letting models generate complex chains of thought. Motivated by interpretability studies showing that the crucial computation required for reasoning tasks is concentrated in a limited range of layers, we introduce Encode–Think–Decode (ETD), a method that enhances the reasoning capabilities of a base model by training it to iterate over a small subset of reasoning-relevant layers during the mid-training stage. ETD amplifies latent reasoning while preserving the original architecture, parameter count, hyperparameters, and training data composition. When iterating on the selected layers at inference time, ETD models yield substantial gains on 17 reasoning benchmarks, including +28.4% relative accuracy improvement on GSM8K and +36% on MATH with the OLMo-2 1B Base model. We also explore an adaptive depth strategy that adjusts the computation per input token. Our results show that recursive latent reasoning offers a simple and effective path to stronger LLM reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Encode-Think-Decode (ETD), a method for converting Olmo-2-1b into a recursive reasoning model. The paper uses angular distance between layer outputs and the kneedle algorithm (for finding distinct changes) to define how to partition the model into Encoder, Recursive Block and Decoder. Later this decision is empirically validated, showing that for this model benchmarks tend to be higher in the region the algorithm finds (Fig. 3). 

The paper shows gains over a selection of benchmarks from the OLMOes suite, showing multiple different recursive depths. The authors compare to the baseline (non-recursive) Olmo-2 model, finding that they often increase accuracy by between 2 and 5% in absolute terms on average but do use more FLOPs during training and inference.

The authors also baseline to different partitions of layers into the Encoder, Recurrent Block, Decoder. Finding a 2-12\*2-2 model is very similarly matched to their 7-4\*4-7 model and FLOPs equivalent. Finally, the authors train a linear probe for the latent space of the recurrent block which is used for early exiting, finding a trade off between efficiency and cost in most cases.

### Strengths
1. Analyses post training models to be recursive, an approach which may reduce cost of future research.
2. Increases benchmark performance on average for Olmo-2-1b after mid-training, when using more FLOPs.
3. Uses an interpretable approach to selecting layers for the recurrent block which is empirically verified.

### Weaknesses
### Claims:
- Some of the papers claims are overstated:
    - The use of relative improvement feels abused here. For example: claiming +36% relative improvement on MATH in the abstract and only achieving a 1.65% absolute improvement feels misleading to the reader.
    - The claim of +28% on GSM8K and +36% on MATH are regularly stated together. However, these are not achievable together with the same k value, this is never clearly stated.
    - The paper claims “substantial” gains on 17 benchmarks, However, these are small in absolute terms.
- In Table 4, when compared to baselines the margin of results becomes extremely small. What are the error bounds on these experiments as the 2-12\*2-2 model is very similarly matched to their 7-4\*4-7 model? Did the authors explore this further, perhaps training a 2-12\*4-2 model is even better? This also increases my concerns that it is the FLOPs used for computation and less so the proposed method driving performance increases.

### Scope:
- The paper only considers one model Olmo-2-1b.
- The is no FLOPs matched non-recursive baseline training run. For example, this increase in accuracy may be due to more computation during training and inference.
    - The paper states: “Our goal in applying a recursive approach, conversely, is to boost reasoning capabilities by efficiently scaling inference-time computation.” However, I do not think this is a fair reason to eschew a FLOPs matched baseline.
- On line 199, the authors claim access to the training data is required to run these types of experiments, however I think they can be conducted without access to training data as long as the same training data is used for all models being analysed.

### Relation to prior work:
- A lot of the citations used when trying to distinguish from prior work by highlighting the approach of using angular distance to locate layers are for models trained from scratch hence, I think these aren’t quite the right citations (e.g. line 149).
- Line 185: “Prior works on recursive-depth models typically rely on simplified training setups.” Geiping et al. and Bae et al. (2025) train in standard set ups for long periods also.
- In the “Key differences to prior work” many points are highlighted such as LoRA Adapters, Regularisation and Input Injections. However, there is a lot of nuance missed here, for example Geiping et al. train for a large number of iterations but this also allows extrapolation in terms of k. Moreover, Aleksandrov et al. find, like much other prior work, input injection is useful. Without baselining against these methods to show the new proposal is superior, I find this weak evidence of novelty.

### Questions
1. Why is Figure 1 taken over C4 and not the training set? Or some other dataset meant to match the reasoning distribution the authors are targeting?
2. Is the baseline “Olmo 2 (k=1)” model trained by the authors? I worry here about the authors training set up differing from the Olmo suite in minor ways leading to the small changed in accuracy we see.
3. I have a large number of questions about Section 5:
    1. What is the training data?
    2. Is the whole model being trained or just the router?
    3. Do the authors have any reasoning for the increase in DROP and OpenbookQA?
    4. What value of K is the model trained with? If it can extrapolate to maximum k=10, one would assume 10 which is not shown as a result elsewhere in the paper.
    5. What objective is used when training the router?

During rebuttal, I would be most interesting in clarifications on the baselines accuracy and training, more baselines being considered and more architectures being considered.

### Soundness
1

### Presentation
2

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
The paper introduces a simple yet effective method for converting existing models to recurrent models by training the model to iterate over a small subset of the "reasoning"-relevant layers during mid-training. The paper shows the benefits of more "reasoning" benchmarks such as GSM8K and MATH.

### Strengths
- simple but effective method for increasing the performance of the model on GSM8k and MATH
- introduce a mechanism to adaptively determine the number of iterations for each input

### Weaknesses
- results only on a single model. This is important for this study because the layers found to iterate over may be specific to the evaluations themselves.
- How do methods like these compare to latent approaches like COCONUT [1] or similar? Although no baseline is necessarily directly comparable, for a better understanding, it would be useful to include another method as a comparison point.

[1] https://arxiv.org/abs/2412.06769

### Questions
- How were the evaluations grouped into reasoning/non-reasoning benchmarks?
- How can we confirm that the current results on layer choosing is not an artifact of the OLMo models?
- Are different model sizes explored in the paper?

### Soundness
2

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
This paper follows the recursive transformer idea and proposes Encode–Think–Decode, which decomposes a pretrained LLM into three functional components, latent encoder, recursive “thinking” block, and latent decoder, based on layerwise representational dynamics. Using the mean angular change of the residual stream between adjacent layers, the authors automatically identify “knee points” that delineate encoder and decoder boundaries. The middle block is then recursively executed multiple times during inference.

### Strengths
* The experimental setup is clear and easy to follow, with well-defined baselines and ablation choices.
* The paper conducts comprehensive empirical analysis, comparing multiple ETD configurations (e.g., varying iteration counts, layer boundaries, and adaptive depth)
* The benchmark coverage is broad, across factual, commonsense, mathematical, and reasoning categories, which helps demonstrate the generality of the approach

### Weaknesses
* The evaluation relies primarily on a single model (OLMo-2 1B). It would strengthen the paper to include results across different model sizes or architectures (e.g., other OLMo sizes) to demonstrate that the Kneedle-based boundary detection generalizes beyond one model.
* In Figure 2, the angular-distance curve appears rather smooth, without a clear “knee.” This make me wonder whether the automatically detected boundary is robust or merely an artifact of one model’s noise pattern. 
* The conceptual link between the angular change turning point and the claimed “encode then think” transition is somewhat heuristic. Reduced representational drift does not necessarily imply reasoning onset, more interpretation or insights into this will be appreciated
* It's necessary to also compare with a larger model under the same FLOPs  to show the trade off .

### Questions
see weakness

### Soundness
3

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
This work presents a practical methodology for turning fixed depth transformer LLMs in to recurrent transformers with the specific goal of enhancing reasoning performance without requiring additional trainable parameters. They propose an algorithmic method for choosing how to assign layers from the original model into a three phase encode, think, decode segmenting of the layers in the recurrent model and ablate their design choices. They also explore an adaptive exiting method for also assigning a variable amount of compute per token. Their strong results on mathematical and reasoning intensive benchmarks underscore the promise of recurrent approaches for efficient reasoning.

### Strengths
1. Experiments are performed on models at the now ubiquitous 1B parameter scale for open source models, implying their results could directly translate to open source resource constrained applications.
2. They identify an important problem in the adaptation process for turning a fixed depth model into a recurrent one and apply both recent results from the LLM literature and a more classical technique from curvature analysis to propose an automated process for slicing up the model layers optimally when initializing their new model.
3. They report non trivial improvements in reasoning performance over the fixed depth baseline they start with (given the limited absolute capability of 1B models)

### Weaknesses
Overall, the paper is rated very borderline on account of some of the ablation soundness and presentation clarity issues discussed below. As the available middle ratings this cycle are just 4 or 6, I am happy to lean very weakly in the direction of acceptance, however, the demerits are represented in the individual component scores above.

### 1. More limited novelty in ETD structure than claimed

Note that there is a bit of a is/ought distinction to be made about where to draw the line between encode think and decode layers. This first should be clarified in S1, and then it also factors into how ablations are performed in S4.3 against other recurrent layer assignment strategies. 

Geiping uses a "Prelude, Recurrence, Coda" structure and Bai proposes a "Middle-Cycle" recurrence strategy, which are both essentially the same as the "Encode, Think, Decode" setup proposed in this work. They key thing to note about whether or not a design is "ad hoc" or "optimal" here likely hinges on the training setting. Geiping trains their models from scratch with encode=prelude, think=recurrence, and decode=coda separations defined from the beginning of training and thus one would assume that by the end of training the "roles" of these layers have become aligned with their usage. 
In this work, you instead start with a _pretrained_ fixed depth model whose layers have already implicitly come equipped with "roles" as a function of where they were in the original model during pretraining. Hence, one imagines there is likely an optimal way in which these layers could be partitioned when the model is adapted from fixed depth to recurrent; your work studies one such strategy. But the point is that whether or not there exists an optimal partitioning, and whether this one is the one, is more an artifact of the adaptation setting this work studies rather than a true difference between this proposal and architectures in recent work on recurrent llms.

### 2. More limited evidence for optimality of layer division strategy than claimed

The evidence for the use of the Kneedle algorithm is weak and overall the draft space allocated to the optimality motivations in S 2.1 and the design of the S4.3 ablation doesn't support a strong claim that this work has identified a procedure that should generalize to any other model family or experimental setting. Please link Appendix E early on to indicate that _an_ ablation was performed to back the Kneedle based choice, but in S4.4 it would also be helpful to motivate why the recursive block was fixed to a size of 4 before this ablation was performed.

The reader is still left wondering whether the 7-4\*k-5 splitting was actually optimal (even for just Olmo 2). We are missing more targeted experiments where another splitting is used based on a different criteria than the kneedle algorithm, except for the two comparisons in Table 4 against a 2-12\*2-2 and 0-16\*2-0 non recurrent setup. Related to the is/ought comment about the difference between a from-scratch and a pretrained setting above, the way this ablation is set up these are not even weak evidences in support of the optimality argument or use of the kneedle algorithm over a visual check of Figure 1. 

Instead the two comparisons are just two more extreme and suboptimal choices. One can interpret the results as 2-12\*2-2 allocating too many layers (from this pretrained model) to the recurrence and not enough to the encoder or decoder, the 0-16\*2-0 results can be viewed as an extreme even more obviously poor choice based on intuitions and prior work regarding the special role of embedding and unembedding layers. Point being, this ablation does not even answer a simple question like whether or not a small change like allocating 5 layers to the thinking stage rather than 4 would outperform the reportedly optimal setup that uses a fancy splitting algorithm. Based on Fig 1 right, it might perform nearly the same.


### 3. Lacking clarity in ACT section

Training details for S5.1 describing the ACT method are missing. How is the router supervision performed? eg. what labels are used for the optimal depth per token position? As this is similar to any router problem such the MoEs expert selection, a non-differentiable choice is made when an exit occurs because no other routes (more iterations of think block in this case) are considered other than the one selected by top-k or argmax. This issue normally requires applying a straight through estimator for the router function to make the process trainable end to end. Additional details about exact what loss was used in these exiting experiments are required.

### Questions
1. S4.1 and S 4.2 have similar titles and read very similarly. Essentially they discuss the same series of results on the effect of depth k but this isn't clear on first skim. They should be unified into a single section perhaps with bold paragraph titles that discuss Table 2, and then the breakdown in Table 3, and then maybe point ahead to the appendix (if that's where they are) discussing trends in various individual tasks, but I think this last part could be omitted or moved to just accompany wherever the table is that contains the full task breakdowns.

2.  L45 seems to be an unfinished sentence

### Soundness
2

### Presentation
2

### Contribution
3
