# ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 8, 2, 2

## Abstract
Post-hoc interpretability methods typically attribute a model’s behavior to its components, data, or training trajectory in isolation. This leads to explanations that lack a unified view and may miss key interactions. While combining existing methods or applying them at different training stages offers broader insights, such approaches usually lack theoretical support. In this work, we present ExPLAIND, a unified framework that integrates all these perspectives. First, we generalize recent work on gradient path kernels, which reformulate models trained by gradient descent as a kernel machine, to realistic settings like AdamW. We empirically validate that a CNN and a Transformer are accurately replicated by this reformulation. Second, we derive novel parameter- and step-wise influence scores from the kernel feature maps. Their effectiveness for parameter pruning is comparable to existing methods, demonstrating their value for model component attribution. Finally, jointly interpreting model components and data over the training process, we leverage ExPLAIND to analyze a Transformer that exhibits Grokking. Our findings support previously proposed stages of Grokking, while refining the final phase as one of alignment of input embeddings and final layers around a representation pipeline learned after the memorization phase. Overall, ExPLAIND provides a theoretically grounded, unified framework to interpret model behavior and training dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ExPLAIND, a unified framework for interpreting deep learning models by integrating model, data, and training step attribution perspectives. Building upon the EPK formulation, the paper extends previous results to modern AdamW, accounting for settings like weight decay, momentum, and mini-batching. From this theoretical foundation, the paper derives influence scores from different perspectives that are additive and can be accumulated along different dimensions. These scores are empirically validated through parameter pruning experiments and a case study on Grokking. Overall, the work proposes a theoretical unification of interpretability aspects.

### Strengths
- Theoretically extending EPK to practical AdamW optimizers with weight decay, momentum, and batching is a nontrivial generalization.

- The idea of unifying different attribution perspectives across model, data, and training step is commendable.

- The analysis of grokking is an interesting training dynamics analysis through attribution scores.

### Weaknesses
- Although the paper contains non-trivial theoretical derivations that seem to be sound, I am not fully convinced by its experimental evaluation. See questions below.

- The title of the paper can be a bit misleading and overclaiming. There are so many different interpretability works from all these three perspectives. The proposed framework seems to be one solution, but I don't see it unifying previous works. For example, model parameter attribution can include circuit discovery for IOI, linear probing directions for truthfulness, and universal neuron identification. How does the proposed score apply to these cases?

### Questions
- Although the paper claims a multidimensional tensor influence score spanning parameter, data, and training steps, only parameter-level pruning is evaluated in Section 4.2. Are there equivalent evaluations at the data level or training-step levels

- Also, the pruning experiment compares ExPLAIND only with one baseline? This single comparison does not provide strong evidence of superiority. It would be valuable to include other attribution-based baselines, similarly for other attribution levels.

- While the Grokking analysis is intriguing, several conclusions, like the dominance of regularization over kernel influences, or the significance of the peak at step 1700 in Figure 3(a), are not entirely clear to me. The curves for kernel and regularization appear roughly similar throughout, and the relationship between the accuracy curve and influence dynamics could be articulated more clearly, as 1700 doesn't see significant in the accuracy curve.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes ExPLAIND, a unified interpretability framework that decomposes a trained model’s prediction into additive influences from training data, model components, and training steps. Building on and extending the Exact Path Kernel view, the authors derive an exact decomposition for modern training regimes (notably AdamW) and demonstrate that the kernelized reformulation reproduces CNN and Transformer predictions.

### Strengths
- Solid theoretical contribution: Clear extension of the Exact Path Kernel to AdamW with mini-batching, moments, and weight decay, stated as a formal theorem.
- Empirical faithfulness check: The EPK reformulation matches original models’ decisions with 100 integration steps (accuracy 1.0; near-zero KL)
- Unified, additive attributions: Influence tensors can be summed along axes to obtain parameter/data/step views that directly tie to predictions.

### Weaknesses
-  Grokking analysis is insightful but mostly qualitative and on small models/tasks; generality to larger LLMs remains unproven.
- Pruning is used to “validate” importance scores rather than to deliver new SOTA compression results; comparisons are limited.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper extends the Exact Path Kernel (EPK) to the case of optimization with weight decay. This stems from the historical observation that NN can be understood as kernel machines in a particular regime. This allows to define scalar quantifying the impact of a training exemplar on the prediction of a class, the *influence*.

These quantities are used to build a framework of interpretability. Then, the capabilities offered by this framework are benhcmarked on two toy tasks:
- parameter pruning of a CNN on Cifar-2
- the *grokking* of a small transformer on Mod 113

### Strengths
The theorem 3.1 is new, and brings the work of Bell et al closer to realistic experimental setups.

The sparsity gains in Sec 3.2 are interesting. 

I love to see the "Emergence of cyclic patterns in the kernel" (l413). I wonder if we expect to see similar phenomenon on other family of problems (even artificial ones) that exhibit grokking?

Overall, the paper proposes an interesting line of ideas to understand training dynamics.

### Weaknesses
### Scope

Currently, I have issues with the motivation of the paper. While the extension of Bell et al to AdamW is interesting, I am less sure to understand the usefulness of the tool in general.

The experimental section is devoted to two setups: CNN training on Cifar-2 (cats and dogs) for sparsity, and the mod 113 task used to exhibit grokking. These two tasks are rather artificial. Even grokking as a whole received recent criticism in its ability to accurately describe some phenomena (see Jeffares and Shaar). I put an excerpt of their position here:

> This work argues that many prominent deep learning
phenomena discussed in the research literature are
not representative of challenges encountered in real-
world applications of deep learning. Thus, not all
efforts to understand these phenomena are equal in
value – we should focus on using them to refine our
broad explanatory theories of important aspects of
deep learning rather than developing narrow ad hoc
hypotheses to describe them in isolation. However,
this perspective is not consistently reflected in current
research practices within the community.

Jeffares, A. and van der Schaar, M., Position: Not All Explanations for Deep Learning Phenomena Are Equally Valuable. In Forty-second International Conference on Machine Learning Position Paper Track. 2025.

Overall, I struggle to grasp if the paper is selling a method ( in which case there is not enough evidence of usefulness), or simply analyzing grokking in Mod 113 with a specific toolbox (in which case it is overfitting a simple task with ad-hoc explanations). 

### Clarity

I struggle to give a sense of quantities defined throughout the paper, like Tensor of Influences or Accumulated influence. The link with parameter pruning is not straightforward to me.  More toy examples could be useful, as sanity checks and for pedagogical purposes.  

### Baselines

Better understanding of grokking in an artificial setup ring limited understanding of neural network training dynamic in broader settings. It could be interesting to apply these methods on bigger models trained on more realistic datasets.

Other tools try to explain networks' behavior from data points. We can mention influence functions, and Sobol' indices for variance decomposition. Since no comparison is made with existing tools, it is hard to situate the paper. 

Mlodozeniec, B.K., Eschenhagen, R., Bae, J., Immer, A., Krueger, D. and Turner, R.E., Influence Functions for Scalable Data Attribution in Diffusion Models. In The Thirteenth International Conference on Learning Representations.

Fel, T., Cadène, R., Chalvidal, M., Cord, M., Vigouroux, D. and Serre, T., 2021. Look at the variance! efficient black-box explanations with sobol-based sensitivity analysis. Advances in neural information processing systems, 34, pp.26005-26014.

Same remark can be done for sparsity in Fig. 2. The baseline of Li et al (2017) is rather old.

### Questions
### Q1

In corollary 3.2, can you clarify if the regularization term is separated from the momentum (*à la* AdamW) or part of the loss?

### Q2

For mod 113, you rely on encoder/decoder architecture. For generations/question answering, decoders are typically sufficient. Why using an encoder/decoder pair here?

### Q3

In Section 5.1, what is your basis to label a phase "circuit formation"? For me, it looks like a (still unproven) hypothesis.

### Q4

Results in Fig 3.b are not very surprising. If my understanding of lines 370 is correct, it means that the last linear layer and projection onto vocabulary tokens are not in the final phase. This very much look like people typically do for finetuning of off-the-shelf foundation models: frozen  pretrained weights + finetuning of the head.  Can you comment on this?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
ExPLAIND introduces a unified framework for attributing neural network predictions to model components, training data, and training steps simultaneously, addressing the fragmentation of existing post-hoc interpretability methods that examine these factors in isolation.

 The framework extends the Exact Path Kernel formulation to realistic training scenarios by generalizing it to AdamW and momentum SGD optimizers with weight decay, learning rate schedules, and mini-batch updates through Theorem 3.1 and Corollary 3.2. The authors validate exactness empirically on a ResNet-9 trained on CIFAR-2 (binary classification, 10,000 samples) and a single-layer Transformer on modular addition (mod-113, 4,000 training samples), achieving perfect decision agreement and near-zero KL divergence with 100 integration steps. 

The core contribution is a tensor of influences indexed by training steps, parameters, samples, and outputs, enabling multi-granularity attribution through flexible accumulation along different axes. Parameter-level scores are validated via competitive pruning experiments against magnitude-based baselines.

 A detailed grokking case study reveals decoder-driven memorization, middle-layer pipeline formation, and a late alignment phase where embeddings and decoder synchronize around learned representations, supported by layer-swapping ablations and cyclic geometry analyses in influence space. 

However, the work is limited to small models, requires storing full training trajectories, scales as O(NDMO) in memory, and provides no principled granularity-selection guidance.

### Strengths
1. Rigorous theoretical extension: Theorem 3.1 extends EPK from basic gradient descent to AdamW with realistic training dynamics (weight decay, first/second moment estimates, mini-batching, learning rate schedules). The mathematical derivation is sound with complete proofs in Appendix D.1.
2. Exact model representation: Unlike approximate methods, ExPLAIND achieves perfect equivalence with the original model (100% accuracy match, zero KL divergence in Table 1) when using sufficient integration steps.
3. Unified mathematical framework: Successfully integrates parameter-level, data-level, and step-level attribution into a single tensor of influences, providing a principled mathematical object for multi-perspective analysis.  

4.Insightful Grokking analysis: The Grokking case study shows that ExPLAIND can yield interpretable, insights into model training dynamics.

### Weaknesses
1. Gap between theoretical contribution and practical utility and scalability limitations:

     The EPK extension (Theorem 3.1) is mathematically sound and interesting.
     But the paper claims to provide a practical interpretability framework.
     Demonstrated only on toy problems with manual analysis and no path to scale (ResNet9 on 2-class CIFAR subset, small Transformer     on algorithmic task).
 
     O(NDMO) memory complexity: N steps × D parameters × M samples × O outputs.

     Computational cost: ~240 H200 GPU-hours for toy experiments suggests prohibitive costs for realistic models.


2. ExPLAIND requires training trajectory information (checkpoints $\theta_s$, gradients ($\nabla_{\theta} f_{\theta_s}(x_k)$)for all samples, optimizer states $m_s$ and $v_s$, batch membership indicators) at every training step. That questions its applicability to pretrained model where only final weights are available.


3. No principled methodology for aggregation selection: Framework provides 5-dimensional influence tensor but missing guidance on which aggregations are meaningful for which questions. Also, why Grokking analysis uses layer-level aggregation?

4. Insufficient validation and missing baselines: No comparison with other modern influence methods.

5. Grokking insights purely qualitative: No statistical testing of identified phases, no quantitative metrics, no automated discovery validation.

6. Parameter pruning only validates parameter scores, not the data attribution claims central to the paper.

7. Integration steps hyperparameter: Table 1 shows 10 vs 100 integration steps for $\phi^{test}$. They use 100, but no analysis of sensitivity or guidance on choosing this for new problems.
8. Grokking generalization unclear: Does the alignment phase insight generalize beyond modular arithmetic? The phenomenon might be task-specific, limiting broader impact of the case study.
9. Unclear justification for “influence” terminology:
The method defines influence scores purely via additive decomposition of the model output.
However, additivity alone does not imply causal or functional influence. Without sensitivity or perturbation tests, calling these quantities “influence” may be misleading.

### Questions
1. How can ExPLAIND be applied to pretrained models such as BERT, ViT, or deeper variants of ResNet when only the final model weights are available? Additionally, how computationally intensive would such an application be?

2. Aggregation methodology: What is your principled method for choosing: 
     Which axes of the influence tensor to aggregate over?
     What granularity (parameter-level, layer-level, etc.)?
     For the Grokking analysis, why layer-level? Did you try other granularities and find similar patterns?
3. Mini-batch identification: When 256 samples are in $Batch_i$  and you compute $\nabla_{\theta} L$ on the full batch, how do you identify individual sample contributions when decomposing via $1{x_k ∈ Batch_i}$? The gradient is computed once for the batch, not separately per sample.
4. Comparison to Influence estimation methods: Standard influence estimation methods work on any pretrained model and provide causal approximations. In what scenarios would a practitioner choose ExPLAIND (requiring full training trajectory logging) over influence functions?
5. Cyclic geometry emergence: Figure 5 and 10 show interesting cyclic patterns. Do these emerge in other modular arithmetic tasks (division, multiplication)? Are they specific to the mod-113 Transformer architecture?
6. Have you tested whether removing parameters or samples with high ExPLAIND influence scores changes model predictions proportionally (validating the notion of “influence”)?

7. Is there a way to subsample or compress the influence tensor without losing theoretical correctness?

### Soundness
2

### Presentation
3

### Contribution
1
