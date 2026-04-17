# Flash Multi-Head Feed-Forward Network

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
We explore Multi-Head FFN (MH-FFN) as a replacement of FFN in the Transformer architecture, motivated by the structural similarity between single-head attention and FFN. While multi-head mechanisms enhance expressivity in attention, naively applying them to FFNs faces two challenges: memory consumption scaling with the head count, and an imbalanced ratio between the growing intermediate size and the fixed head dimension as models scale, which degrades scalability and expressive power. To address these challenges, we propose Flash Multi-Head FFN (FlashMHF), with two key innovations: an I/O-aware fused kernel computing outputs online in SRAM akin to FlashAttention, and a design using dynamically weighted parallel sub-networks to maintain a balanced ratio between intermediate and head dimensions. Validated on models from 128M to 1.3B parameters, FlashMHF consistently improves perplexity and downstream task accuracy over SwiGLU FFNs, while reducing peak memory usage by 3-5x and accelerating inference by up to 1.08x. Our work establishes the multi-head design as a superior architectural principle for FFNs, presenting FlashMHF as a powerful, efficient, and scalable alternative to FFNs in Transformers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Motivated by the structural similarity between single-ahead attention and a feed-forward network (FFN), the paper explores multi-head FFNs. To account for the increased memory consumption and other issues, the authors propose a novel architecture, FlashMHF, inspired by FlashAttention which also dynamically weights parallel sub-networks. They find for small models that their design improves perplexity and task accuracy whilst reducing peak memory usage and inference time vs a SwiGLU FFN. This suggests that FlashMHF might be a powerful new architectural component that could replace the FFN in existing transformer architectures.

### Strengths
1. Good empirical results on the models and tasks tested compared to a standard baseline, clear improvements across a range of downstream tasks
2. Mathematical exposition of the preliminaries and method is clear and well-written
3. The method is a satisfying synthesis of existing ideas and innovations in other aspects of the transformer architecture to improve the FFN block
4. The GPU memory scaling of the proposed FFN architecture is smaller than that of a typical FFN

### Weaknesses
1. The paper proposes a core architectural innovation for LLMs, but only tests on very small models (<=1.3B parameters).
2. Three model sizes are tested but they are not plotted together / compared directly so it's not clear how improvements scale with size. There's limited empirical evidence that we should expect the accuracy / perplexity advantages of this architecture to improve with scale, rather than diminish.
3. It's not clear why scaling imbalance, $d_{ff}/d_h$, is an issue, as stated on line 056. The reference given on line 057 does not address this since neural scaling laws assume normal FFNs, and does not investigate multi-head FFNs. The discussion and evidence given in 3.1 and 4.1 seemingly only addresses one way of scaling the model. There are unstated assumptions in the paper relative to prior work about what ratios are important, and how one would scale a model with multi-head FFNs. To make a claim about these ratios scaling poorly and causing issues, one needs to provide evidence that any way of scaling them would lead to performance degradation relative to using single-head FFNs. Otherwise it can just be argued that scaling them in a different way might resolve this problem naturally, without need for more complex architectures. More explicitly put, for the Naive multi-head FFN baseline you make assumptions such as "the per-head width is typically kept fixed" (line 180), and then show this is bad. Why should one keep this fixed then? Why not just scale things in a different way? Additionally, why is it correct to equate the ratio $d_{ff}/d_{model}$ in normal SwiGLU designs with the ratio $d_{ff}/d_h$? Saying this latter ratio is outside of the optimal values found for the former ratio in prior work tells us nothing without additional evidence or reasoning backing up the validity of this comparison.
4. Framing something that scales linearly as you scale up the model as an "explosion" is disingenuous. Typically things are framed as explosions when they scale exponentially. It is not convincing that the memory requirements of the naive multi-head FFN or standard FFN are a critical issue.
5. You do not conduct enough ablations for the claims on lines 338-343 to be valid.
6. It's not clear that inference latency reduction results are statistically significant
7. All plots are given with training steps on the x-axis, not wall clock time. It's unclear how the proposed architecture affects training time.

### Questions
1. Why does an imbalanced ratio between intermediate FFN size and FFN head dimension degrade scalability and expressive power? (See weakness 3 for related critique and questions)
2. Why should multiple FFN heads split up the model dimension, and not each use the whole thing, as would be analogous to multi-head attention? Obviously this would lead to greater computation requirements, but perhaps also better performance? It would be nice to see this investigated, though this is a very minor point.
3. How does the parameter count of FlashMHF scale and compare to a standard FFN?
4. Does the proposed architecture increase training time for a fixed number of training steps?

### Soundness
3

### Presentation
4

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
The paper proposes FlashMHF, a replacement for standard FFNs in Transformer architectures. The core idea is to mirror the Multi-Head design of Attention also in the FFNs implementation. The paper however warns that a naive adaptation incurs scaling issues, both in terms of increasing memory consumption and expressive power degradation. The authors address both of these issues by carefully prescribing how the intermediate activations dimension should scale with model size, and by implementing a fused kernel for FlashMHF which avoids materialising intermediate tensors. Results show how substituting the FlashMHF component with FFNs can boost performance (both on PPL, and downstream tasks evaluations taken from lm-eval-harness), while simultaneously reducing peak memory utilisation, and slightly improving latency.

### Strengths
- The main motivations behind the choice of architecture modifications are justified reasonably well
- The analysis is convincing, and the experiments conducted overall complete (although some results could be presented better)

### Weaknesses
- Novelty is limited: both core methodologies (mirroring MH Attention and improving kernel application via tiling) have already been proposed

### Questions
__On Novelty__:
As I mentioned above, I find the novelty aspect of the paper rather limited. As you yourselves correctly point out, the structural symmetry between sequence-wise Attention and feature-wise FFN (which acts as main justification behind your work) has already been illustrated; the proposal to split FFNs in a multi-head fashion was already (granted, partly) investigated in MH-MoE; the tile-wise implementation of your fused kernel is directly inspired by FlashAttention, and is at the core of the design of efficient parallelisation of MMMs. The most relevant novelty is then given by the proposed re-scaling of the size of the internal components of the MH FFN. I still appreciate the overall execution, but I find this limits the contribution of the paper.

__On Fig7__:
The presentation of the results in Sec4.3, and more specifically in Fig7, should be heavily revised, for a number of reasons:
- From what you write in L415, your “comparison uses a 20-layer FlashMHF and MH-FFN against a 24-layer SwiGLU baseline”, so I’m understanding you’re considering memory consumption and latency in a *forward pass through the whole architecture*, including both FFN/FlashMHF and Attention layers? I believe at this stage it would be more relevant instead to have a direct comparison between the *single* FlashMHF / FFN layer, so to properly identify the improvements introduced by your proposed modification (as the Attention layer is the same in both cases, I take it). To be clear: I do appreciate the result you report (ultimately, the “weight” of the overall architecture is what practitioners mostly care about), but the presence of Attention does dirty the relevant metric. Notice this should play in your favour, too, in that the memory / speedup gains should be more marked. If instead I misunderstood, and you’re considering just FFN/FlashMHF layers, please clarify this in the text.
- What is the deal with the sequence lengths picked? I was expecting orderly powers of 2, which would make identifying the O(L) trend straightforward at glance. Also, please use a ylog scale, for the same reason
- Moreover, why picking sequence lengths in the first place? Since you’re focusing on the FFN layer (which applies a perfectly sequence-parallel operation, and acts purely along the feature dimensions), then a scaling trend with respect to feature dimension would be much more relevant, in my opinion. What you’re effectively reporting here is the scaling trend of Attention. Again, it’s not like this result is not useful per se, but the way it’s presented makes it harder to isolate the contribution of your own component, which should be the focus of this section.
- Finally, and perhaps most importantly, the comparison is not entirely fair: if I understood correctly, you’re using an unoptimised version of SwiGLU (which unnecessarily materialises intermediate tensors) to compare against your own fused kernel for FlashMHF. How much of the gains you’re seeing are due to the tiled implementation of the kernel? Because that same solution could be easily applied to SwiGLU as well, I reckon.

__On Gating__:
In L200-218 you describe your chosen per-head expert aggregation mechanism. There is a number of different ways one could go on about aggregating both within and across heads: have you experimented with different methods? Can you expand on the reasoning behind this specific choice? Compared to the remainder of the paper, this section is lacking some justifications.


__Minor__:
- In L247 you write: “we synchronize the hyper-parameter settings for the optimizer across all models”. What does this mean? I’m expecting, say, optimal LR’s to vary across architectures, at least in principle. Are you not performing any hyper-parameter sweep whatsoever? And if you’re doing it, are you picking the best for *which* architecture exactly?
- You’re going down the route of making the FFN more akin to Attention; but there is also the “dual” approach of making attention more akin to FFNs, as explored in “MLP-Mixer: An all-MLP Architecture for Vision”. I don’t think it makes sense to explicitly add a comparison with this architecture, but I would at least mention it, as I believe it’s relevant. Moreover, I was quite surprised to see that there isn’t much work which just goes all the way and substitutes FFN with component-wise attention. Apart from MH-MoE, I could only find “DaViT: Dual Attention Vision Transformers” (again, only applied to vision).


__Grammar / Rewording / Formatting__:
- L51 “we analyse the …, a straightforward … and identify” -> the clause is breaking the flow. Maybe “we analyse the … (a straightforward … ), and identify…”
- L62 analogous -> analogousLY
- L89 the equation is hanging: consider prepending something like “We consider the parameters: ”
- L107 remains -> reTains
- Eq(1,2,3,…) I think you’re misusing the equivalent-by-definition / delta-equivalent (\triangleq) symbol. The defined-as symbol (\eqqcolon) would be much better indicated here, imho
- L115 define headwise split -> define THE headwise split? Define headwise split AS? (Similarly for headwise concatenation in L121)
- L119 this operation split -> splitS
- L120 into $d_h\times H$…sub tensors? parts? blocks?
- L131 to overcome these challenges -> which challenges? I reckon it refers to the “practical limitations” above, but it’s rather vague
- L473: write -> store? Write … to memory?
- L474: incorporates -> incorporate

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
This paper proposes FlashMHF, which introduces the multi-head mechanism (in attention) into the Feed-Forward Network (FFN) module while balancing performance scalability and implementation efficiency. The proposed design addresses two key issues in naïve multi-head FFNs (i.e., scaling imbalance and memory explosion), by decomposing parallel FFN subnetworks and implementing an I/O-aware flash kernel. Experimental results on small- and medium-scale models demonstrate that FlashMHF outperforms the de facto SwiGLU baseline in language modeling tasks, while significantly reducing memory usage.

### Strengths
1. The paper is well-motivated and clearly written. It identifies two key challenges of multi-head FFNs and proposes corresponding solutions, which are empirically validated.
2. FlashMHF achieves lower PPL and better downstream performance than SwiGLU and other baselines. The architectural design choices are well-supported by effective ablation studies, including the multi-head mechanism, SwiGLU component, and subnetwork structure.
3. lt is implemented with a kernel design analogous to FlashAttention, ensuring the feasibility of training large-scale language models efficiently.

### Weaknesses
1. **Source of subnetwork advantages.**
The authors claim that the benefit of the subnetwork design mainly arises from a more balanced expansion ratio. However, for a given head, the parallel subnetwork computation essentially differs from a dense FFN only by an additional **blockwise gating** applied to intermediate activations. When concatenated, this does not effectively control the expansion ratio and finally increases by $d_{model}/d_h$ compared to a standard SwiGLU. I suspect the improvement stems from added nonlinearity (gating with normalization) rather than from the parallel sub-net. In other words, applying a similar gating mechanism to a standard SwiGLU might also yield certain loss improvement (as the experiments show, the standalone multi-head design brings no clear advantage at larger scales).

2. **Fairness of speed evaluation.**
The speed comparison appears somewhat unfair. To match parameter counts, the authors add four extra layers (1/5 of total) for baseline; but deeper networks are inherently slower due to layer-wise serialization, whereas **increasing width** would be a fairer adjustment. Moreover, the attention computation also scales with depth, thus latency improvements only become apparent at longer sequence lengths (as shown in Fig. 7b).

3. **Memory evaluation setup.**
The memory comparison setup should be clarified. SwiGLU can also be easily adapted to a flash kernel, and many frameworks **fuse activation functions** to reduce memory overhead. It is unclear whether the authors’ implementation accounts for these optimizations. Considering that modern LLM training almost universally employs **gradient checkpointing**, FFN intermediate activations are typically recomputed rather than stored, which should be reflected in a more realistic baseline comparison.

### Questions
1. What is the specific implementation of SwiGLU used in the efficiency evaluation? Is activation recomputation (gradient checkpointing) applied during measurement?
2. In the GLU formulation, you define $\mathbf{Q}=\mathbf{X}$ (Eq. 3). However, in the multi-head FFN definition, a separate projection $\mathbf{W}_{in}$ is introduced to obtain the query (Eq.10). Is this design choice be empirically validated as necessary?
3. Compared to PKV, the activation function used in PAttention [1] might serve as a more solid baseline for comparison.
4. Given that most sota Transformer architectures now adopt MoE designs, how do the authors view the compatibility and potential integration of FlashMHF with MoE architectures?

[1] TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters. 2024.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes FlashMHF, which is a multi-head feed-forward networks (FFNs) for Transformers. Motivated by the structural similarity between single-head attention and FFNs, the authors identify two key challenges in current MHF: memory explosion and scaling imbalance. FlashMHF solves these problems by pairing a scaled-balanced parallel FFN subnetworks designed with a high-efficiency, IO-aware kernel.  Experiments on models from 128M to 1.3B parameters show improvements in perplexity and downstream tasks, with 3-5x memory reduction and up to 1.08x inference speedup.

### Strengths
The motivation of the paper is well justified with two problems in naive multi-head attention. There are proper ablations such as head dimensions and model scales, and downstream task evaluations are standard. The idea is straightforward by using sub-networks to group different heads to solve the problems, yet results are pretty impressive.

### Weaknesses
1. In section 3.2.1 the authors say their FlashMHF functions Luke a dense MoE, however, there is no direct comparison against dense MoE architecture. 
2. There is no ablations for “Flash”, so it’s hard to isolate memory savings from the architectural change and the kernel optimization. 
3. Lack of large scale experiments to verify the scaling effect - largest model size is 1.3B.
4. About presentation, Figure 3a doesn’t show multihead which is confusing. Also, the biggest innovation of it seems to come from MoE, while the title is a bit misleading, “mixture of dense multi-head FFN experts” might be better cover what the core idea is.

### Questions
Multi-head needs to be concat so we do need to materialize the full tensor. In section 3.2.2 it says “The key to solving the memory explosion lies in the multi-head design itself” seems wrong, shouldn’t it be in the expert design, because we can do the weighted average accumulation?

### Soundness
3

### Presentation
2

### Contribution
3
