# Network of Theseus (like the ship)

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
A standard assumption in deep learning is that the inductive bias introduced by a neural network architecture must persist from training through inference. The architecture you train with is the architecture you deploy. This assumption constrains the community from selecting architectures that may have desirable efficiency or design properties due to difficulties with optimization. We challenge this assumption with Network of Theseus (NoT), a method for progressively converting a trained, or even untrained, guide network architecture part-by-part into an entirely different target network architecture while preserving the performance of the guide network. At each stage, components in the guide network architecture are incrementally replaced with target architecture modules and aligned via representational similarity metrics. This procedure largely preserves the functionality of the guide network even under substantial architectural changes—for example, converting a convolutional network into a multilayer perceptron, or GPT-2 into a recurrent neural network. By decoupling optimization from deployment, NoT expands the space of viable inference-time architectures, opening opportunities for better accuracy–efficiency tradeoffs and enabling more directed exploration of the architectural design space.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a framework for gradually replacing parts of a neural network. It is based on the Ship of Theseus philosophical question about, how if we gradually replace parts of a vessel, will it eventually become a new vessel altogether?

The paper considers a similarity loss to match internal representations of data at every stage. Experiments are conducted on computer vision and natural language processing tasks.

### Strengths
- The idea of the paper is quite interesting as a way to reframe how architecture training and inference is considered.
- The idea combines multiple prior works like knowledge distillation.
- Application is not restricted to one task domain such as computer vision.

### Weaknesses
- The paper distinguishes itself from compression pipelines, such as pruning and quantization. However, those pipelines are utilized to maintain end-to-end performance while achieving some hardware benefit, such as lower latency or power consumption. However, little emphasis is placed on such gains in the experimental section of this work, if at all, which severely limits its utility and merit.
- The overall idea of the paper seems incremental and generally just based on loss on matching hidden representations, which prior works for knowledge distillation [1] have done. 
- The paper should compare to prior work which using gradual techniques to shrink an architecture [2] by phasing out some components. 
- Experimental results are largely confined to older/smaller architectures, limiting the scope of utility of this work.

### Questions
See weaknesses.

References:

[1] https://arxiv.org/abs/1910.01108

[2] https://arxiv.org/abs/2305.12972

### Soundness
3

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
2

### Summary
This paper proposes a 2-step method that given an initial guide network produces a desired target network. In the first step, components of the guide network (such as layers or blocks) are iteratively replaced one at a time with target components via optimisation of a metric between activations to produce a desired target network. In step 2 the target network is trained end-to-end on the downstream task. The paper proposed use of two different metrics. 
Experiments were performed with ImageNet for an image classification task, and Wikitext for a language modelling task. Firstly, the method was shown to perform better than training the target network from scratch. Then progressive replacement was shown to perform better than 3 other replacement schedules. Finally it was shown that the method retained performance improvement over training from scratch even when the guide was untrained.

### Strengths
The method was clearly explained and the experiments were relevant.

### Weaknesses
The paper could more concisely summarise its contributions in the introduction. It was clear what the method did but less clear what was novel. 

The paper would be improved by further comparison to other methods. 

The paper says “Doing the same with distillation would lead to much worse results.”. Why isn’t this just demonstrated empirically in a direct comparison with the proposed method?
The results only compare the guide accuracy, NoT accuracy, and from-scratch baseline accuracy. Why was the method not compared to other knowledge distillation methods such as for example Task-aware layEr-wise Distillation (TED) proposed in Liang et al Less is More: Task-aware Layer-wise Distillation for Language Model Compression (ICML 2023)

The authors says “the stage-wise replacement of NoT has not been proposed in previous work”. 
Xu et al. BERT-of-Theseus: Compressing BERT by Progressive Module Replacing (EMNLP 2020) uses a strategy of progressive replacement (while also using the Theseus analogy) where at each training step t, a guide module is replaced with the target with probability p_t. The idea appears similar to the current work but with a different replacement schedule. Why was this strategy not compared to?

### Questions
Did you try any other X -> MLP combinations other than ResNet-18 → MLP? Do guides that themselves can be trained from scratch to high accuracy act as better guides than networks that train from scratch to a lower accuracy?  Does an untrained MLP guide (MLP -> MLP)  used in NoT work poorly empirically? 

The paper says "For all training, we use a consistent batch size of 256. Representational similarity metrics are affected by the batch size, specifically more samples allows the metric to better approximate similarity.". To what extent do the results for your method change with batch size? An ablation showing what happens if the batch size was halved and doubled would be helpful. 

The paper says “We distinguish NoT from distillation. NoT aligns computations and can transfer inductive biases from one architecture to another”. I did not understand this. In what way does NoT align computations and distillation does not? Could this be clarified please.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Network of Theseus (NoT), aiming to relax the constraint induced by inductive bias that a neural network must share the same architecture during training and inference. NoT progressively replaces parts of a trained or even untrained “guide” network with heterogeneous target modules (e.g., CNN $\rightarrow$ MLP, GPT-2$\rightarrow$RNN) while maintaining similar representations measured by metrics such as CKA or the proposed D-MNN. The authors claim that this decoupling enables a better accuracy–efficiency trade-off and opens opportunities for exploring the architectural design space.

### Strengths
1. The core idea, progressively replacing layers while maintaining representational similarity, is intuitive yet practically effective. Although philosophically framed as a "Ship of Theseus," the method essentially defines a similarity loss to align the outputs of the replaced and original layers. This part-by-part replacement and similarity alignment strategy, while conceptually simple, is implemented systematically for the first time. Despite limited algorithmic novelty, the approach is simple, feasible, and potentially useful for future work.

2. The paper is clearly written and easy to follow.

3. It introduces a new perspective on decoupling training from deployment.

4. It shows that even untrained networks can provide transferable inductive biases, which is an interesting finding.

5. Although theoretical justification is weak, the experiments offer solid empirical evidence supporting the feasibility of the approach.

### Weaknesses
1. While the idea is straightforward and effective, it lacks strong novelty. Overall, the contribution lies more in the systematic implementation and empirical exploration of an intuitive idea than in a fundamentally new conceptual innovation. The four replacement schedules (progressive, sequential, independent, joint) are systematic but predictable; 'progressive' being superior is not surprising. The experiments are solid but could have explored broader domains, such as speech or multimodal settings, to better test generalization.

2. The proposed D-MNN similarity measurement is under-analyzed. CKA remains the main metric used, and D-MNN is only reported for the ResNet-18 $\rightarrow$ MLP conversion. The performance difference between D-MNN and CKA is marginal; no sensitivity to hyperparameter analysis is provided, and comparisons with other similarity measures are missing. Hence, the effectiveness of D-MNN remains insufficiently validated.

3. The paper does not discuss parameter count changes after module replacement. It only mentions shape compatibility and low-rank linear projections, but does not clarify whether parameter counts are kept constant. The absence of parameter statistics makes it difficult to judge whether performance preservation is partly due to differences in model capacity; this is an important missing piece.

4. The computational cost of NoT (training time, GPU usage, etc.) is not reported, leaving its scalability unclear.

5. While representational alignment is presented as the key mechanism, the paper does not explain why it should ensure functional equivalence across architectures. The lack of theoretical grounding weakens the conceptual depth of the work.

6. The paper does not specify how the correspondence between guide and target components is determined. The replacement mapping 
$\mathcal{R}$ seems to be manually defined based on shape compatibility or pre-specified block grouping (e.g., 4 ResNet-50 blocks $\rightarrow$ 1 ResNet-18 block). However, different mapping strategies could substantially affect representational alignment and final performance. The lack of ablations or analysis on this aspect limits the understanding of the robustness and generality of the proposed method.

### Questions
1. The "replacement mapping" mechanism appears to be critical for the model's evolution process and might heavily influence the reported improvements. Is this understanding correct? If not, could the authors provide ablation or evidence showing the specific effect of replacement mapping on model performance?

2. See also the Weaknesses.

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
3

### Summary
This paper introduces Network of Theseus (NoT), a method for converting neural network architectures by progressively replacing components while maintaining representational alignment. NoT enables dramatic architectural transformations such as converting CNNs to MLPs, ViTs to token-wise MLPs, and Transformers to RNNs via optimising intermediate layer activations using similarity metrics like Centred Kernel Alignment (CKA). The key innovation lies in staged progressive replacement that minimises error accumulation. Empirical results are reported for several substantial cross-architecture transformations on datasets like ImageNet and Wikitext-103, demonstrate that NoT achieves substantial performance improvements over naive replacement baselines, and works even with untrained guide networks.

### Strengths
1. Unlike prior works, NoT is not limited to structurally similar architecture or reliance on identical computational patterns, e.g., attention to linear attention, but can handle radical family shifts, e.g., ResNet to MLP, GPT-2 to RNN, etc.

2. The progressive replacement strategy is quite elegant and well-justified. The "Ship of Theseus" metaphor effectively communicates the core idea of this work. The proposed method addresses a limitation in the current neural architecture research, the tight coupling between training and deployment architectures.

3. The ablation studies are rigorous, authors contrast progressive, joint, independent, and sequential replacement, robustly arguing for the effectiveness and stability of progressive alignment.

### Weaknesses
1. The main baselines are naive replacement and training from scratch, but there are no strong comparisons to SoTA methods, such as in progressive distillation, model stitching, and neural architecture search that also attempt cross-architecture or sub-graph-level transfer. For example, some relevant works have been briefly discussed and cited in the paper,  but there is no direct comparison in the main results.

2. The tuning of hyper-parameters for the D-MNN metric, such as temperature choice, batch size effects, and differentiable kNN mechanisms is only briefly discussed. Implementation guidance and impact on downstream results are underexplored, especially for D-MNN sometimes yields comparable but lower results than CKA.

3. Although authors claim that the NoT is a general method, the progressive replacement requires multiple stages of optimisation, each involving careful learning rate tuning, batch normalisation recalibration, and multiple random seed runs, this could undermines the practical applicability.

### Questions
1. Can authors provide some extra information such as wall-clock time or GPU hours, for the computational cost of NoT, compared to training from scratch?

2. What determines when to stop optimising at each stage? Is there a principled way to detect convergence of representational alignment?

3. Can authors discuss a bit on the failure cases? under what conditions does NoT fail?

4. Regarding the batch normalisation, can author quantify the impact of the batch normalisation handling strategy? What happens if you use Layer Normalisation or Group Normalisation instead?

### Soundness
2

### Presentation
3

### Contribution
2
