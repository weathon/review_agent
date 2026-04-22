# Rethinking the shape convention of an MLP

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Multi-layer perceptrons (MLPs) conventionally follow a narrow-wide-narrow design where skip connections operate at the input/output dimensions while processing occurs in expanded hidden spaces. We challenge this convention by proposing wide-narrow-wide (Hourglass) MLP blocks where skip connections operate at expanded dimensions while residual computation flows through narrow bottlenecks. This inversion leverages higher-dimensional spaces for incremental refinement while maintaining computational efficiency through parameter-matched designs.
Implementing Hourglass MLPs requires an initial projection to lift input signals to expanded dimensions. We propose that this projection can remain fixed at random initialization throughout training, enabling efficient training and inference implementations. We evaluate both architectures on generative tasks over popular image datasets, characterizing performance-parameter Pareto frontiers through systematic architectural search.
Results show that Hourglass architectures consistently achieve superior Pareto frontiers compared to conventional designs. As parameter budgets increase, optimal Hourglass configurations favor deeper networks with wider skip connections and narrower bottlenecks—a scaling pattern distinct from conventional MLPs. Our findings suggest reconsidering skip connection placement in modern architectures, with potential applications extending to Transformers and other residual networks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes “Hourglass MLP” blocks, a bow-tie like architecture pattern that inverts typical MLP architecture shape.  The work studies this shape variant, presents some pareto plots. The bow-tie + shape & skip placement principle ideas in residual networks are touched on via related work (UNet,MoE,LoRA) & mirror established bottleneck tactics. Experiments with 2-epoch MNIST & Imgnet32; small delats show efficiency in params (PSNR) [aruged from pareto front plots] & the results do show this approach to be better than conventional MLP (with skip connects) at input dim. The work is more a baseline starting study & needs more focused effort & engineering to arrive at when and hwere to organise resid connections at higher dim in MLP stacks -- this, if backed by more evidence, can be of value.

### Strengths
Architectural description with a compact formulation (Eq3) and schematic (Fig1)
Pareto framing is well structured. Explicitly searches (dz, dh, L) and reports fronts on multiple tasks/datasets (Tbls 1,2) matching good practice to study shape trade-offs.
Knobs to control are pragmatic showing that frozen random input projection is competitive for at least one config (Fig5); maybe helpful for certain design choices upstream/downstream.

### Weaknesses
Novelty is limited / feels incremental as this is similar to resnet bottleneck tactic transposed to MLP. While motivation is there for hiher-dim learning, the shape+skip placement principle is used earlier (covered in related work). The new part can only be argued at best as an incremental variation in images with processing adjustments (vectorized & projection). Interesting as a variation, but not directly new knowledge as well.

Efficiency claims focused on param count, but are not backed up around compute. This is a mis-match to the promise made in abstract & early sections of the narrative around compute efficiency. The specific measures to showcase this value (compute/latency/activation mem etc.) are not presented. The hourglass/bow-tie arch would mean the block multilies on matrix sizes; so there will be a mismatch as things scale up. More so in this work as the tactic is adding an input up-project layer [even if claim can be made about it being frozen, it is adding to compute].

Evaluation scope & measures do not support ambition. The experiment tasks are small-data, low-res, PSNR only. Improvements at high-budget end are small (Tab2). Aslso, given no cross-ref with human baselines (not that critical).  The measrues as reported does not show if the representation helps classification as such. Only 2-epocs of training ~ why? Is it possible that some configs are under-trained? Would PSNR diffs be different with more training?

Reproducibility inconsistencies. Appendix A shows dz [8,2200]; Tables in paper report values well above that. Also, 5-sigma is an odd choice of reporting.

Baselines/ablations: The various aspects as indicated & reported do not fully align. e.g. in Sec4.2 it is stated that MLPs can benefit from rnd projections, but no specific details are not shown. Frozen projection claim is supported by a single config on a single task [Fig5]. This needs more testing spaning datasets, dz & noise regimen. 
	
The theoretical motivations are not offering explanations, rather present some plausibles. JL lemma, Cover’s theorem, random features, compressive sensing justify that “high‑dimensional random projections preserve structure” but no analysis targets residual learning specifically (why residual updates at higher dimension should systematically dominate under parameter/compute constraints?). Something that offers progress on this would help support the ambition of this work.

### Questions
If Hourglass is truly more compute‑efficient, this must show up beyond parameter count. Why was htis not more actively reported?

Your block looks architecturally similar to classic bottlenecks where residuals live at the wider ends. What, concretely, is new beyond placement in an MLP with an upfront up-projection, relative to U-Net skips, MoE “temporary widenings,” or LoRA-style paths?

On MNIST, you transform to a (class) prototype then classify. Why not report standard classification accuracy as well, to show utility of the representation? How sensitive are results to the choice of the single prototype image per class?

Given Transformers typically expand FFN by 2x to 4×; how would your wider residual space interoperate with attention without exploding compute?

### Soundness
2

### Presentation
3

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
The paper proposes:
1. To use a wide-narrow-wide approach in MLP.
1. Instead of training an input projection matrix, the paper proposes to use a fixed random matrix as a projection matrix in the first layer.

### Strengths
1. A simple approach that defies convention.
2. The 'hourglass' (wide-narrow-wide) MLP achieves significant results compared to 'conventional' (narrow-wide-narrow) MLP.

### Weaknesses
1. The random fixed input matrix is not convincing enough to warrant a switch from the conventional trainable weights. As the reduction in parameter count  is only marginal and the fixed matrix model even had a decrease in PSNR performance. It would be great if the author is able to provide values on the number of parameters reduced against the model parameters size on different model sizes.
2. The paper claims that hourglass is Pareto-Optimal with no theoretical backing. Only based on empirical results from a small subset of conventional models.
3. The current experiments are limited it would be more convincing to compared on more computer vision task like Image classification and across more variety of models like Bottleneck-Resnet[Zagoruyko, Sergey, and Nikos Komodakis] and MLPMixer[Tolstikhin, Ilya O., et al].


Zagoruyko, Sergey, and Nikos Komodakis. "Wide residual networks." arXiv preprint arXiv:1605.07146 (2016).

Tolstikhin, Ilya O., et al. "Mlp-mixer: An all-mlp architecture for vision." Advances in neural information processing systems 34 (2021): 24261-24272.

### Questions
1. The paper mentions linear separability at high dimensions as a theoretical foundation but does this contradicts the paper which project from a higher dimension to a lower dimensional? And if it does not why?
2. How does the randomness of the input project matrix affect the variance of the results obtained? It would be good if you could reflect this in the paper with more training across different seeds.
3. In Table 1. and Table 2. is the hourglass model trained with a fixed W_in or a trainable one?

### Soundness
2

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
4

### Summary
This paper proposes to flip the narrow-wide-narrow convention for MLP and skip connections in deep neural networks to instead use an hourglass shape with wide features at skip connections and narrower processing inside MLP layers. This paper addressed the added computational cost of this by keeping the expanding projection fixed with random initialised weights during training. The results on generative tasks on MNIST and ImageNet-32 data show that the hourglass networks have stronger Pareto frontiers than the conventional structures.

### Strengths
- The idea of reversing the traditional MLP shape to explore the effect of skip connection dimensionality is interesting and goes against the existing convention. This could potentially lead to significant architectural changes for e.g. LLMs.
- The paper takes a strong focus on the performance/efficiency Pareto frontier, recognising the importance of these two aspects of model quality. 
- The evaluation tasks are chosen to be well-suited to the incremental refinement of residual networks, the main focus of the paper.

### Weaknesses
- The experiments of the paper are too narrow and simple, focusing on small-scale tasks on MNIST and ImageNet-32. This is in contrast to the high-dimensional, large-scale settings for modern LLMs and diffusion models, where architectural design trade-offs make a large difference on both performance and efficiency. I therefore think that experiments on Transformer networks need to be run, or alternatively something like MLP-Mixers, CNNs. Would the hourglass structure help in these scenarios or are the expanded input/output dimensions too costly? Similarly, the datasets considered need to be larger in scale, beyond MNIST and 32x32 images. While extremely large-scale training isn’t required, testing the design on at least one Transformer model on a text-based task would give an indication to whether the approach scales.  Currently that is not at all clear, and the claims about extensions to Transformers and ViTs are speculative and not supported by any experiments. To achieve real impact, this paper will need more large-scale or real-world validation.

- The fixed random projection at the start of the network reduces the number of trainable parameters but doesn’t reduce the cost of the forward pass, so I think the “efficiency” claims are a bit overstated. It also adds to the inference time cost and memory bandwidth, which is an important aspect for modern deep neural networks.

- There is no clear explanation of why the wide skip connections outperform the conventional narrow ones, apart from an appeal to high-dimensional intuition and random projection theory. Analysing this more would improve the paper.

- There is quite a bit of repetition in the intro and contributions, and space could be found by reducing this redundancy.

### Questions
- Why are there more models trained using the Hourglass networks compared to the Conventional ones, in Figures 2a, 3 and 4?
- Can experiments be run on a transformer-based language model architecture? And similarly a diffusion model, which is argued in the paper are well suited to this incremental refinement. If such experiments showed similar improvements, this would significantly strengthen the paper.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes to revert the conventional residual connection pattern "narrow - wide - narrow"  in MLP to "wide - narrow -wide". The core idea of the design is that doing representation refinement at high dimensional space is more effective than in low dimensional space, as the former has more expressivity. The paper further proposes to use fixed random projection layer to up project the initial low dimensional input, avoiding additional training budget due to design. Experimental results show that that proposed architecture achieve better parameter / performance trade off than traditional MLP in several generative learning tasks.

### Strengths
1. The paper is clearly written and easy to follow
2. The paper is well motivated, as existing theoretical and empirical evidence indeed suggest that a wide - narrow - wide projection structure may bring better representation learning performance
3. The paper proposes to use fixed random projection to reduce training cost
4. The experimental results on MLP shows that the proposed architecture achieve better performance / parameter count trade off than conventional MLP in generative learning tasks, positively support the proposed idea.

### Weaknesses
1. Besides the performance on generative tasks, the paper doesn't provide insights about how residual learning in high dimensional space help representation refinement
2. The paper only conducts experiments on MLP architecture. Although the paper sketches potential extension to other modern architecture like transformer, it's unclear whether it will bring similar improvement, thus the contribution is limited

### Questions
1. A visualization comparison showing how the wide high dimensional space help representation learning could better support the proposed  design, refer to figure 1 of [1], specifically, it shows how high dimensional feature map provides rich and detailed gradient feedback during training. 

[1] Schonfeld, Edgar, Bernt Schiele, and Anna Khoreva. "A u-net based discriminator for generative adversarial networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

### Soundness
3

### Presentation
3

### Contribution
2
