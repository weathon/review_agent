# ASMR: Activation-Sharing Multi-Resolution Coordinate Networks for Efficient Inference

- Decision: Accept (poster)
- Scores: 8, 6, 5

## Abstract
Coordinate network or implicit neural representation (INR) is a fast-emerging method for encoding natural signals (such as images and videos) with the benefits of a compact neural representation. While numerous methods have been proposed to increase the encoding capabilities of an INR, an often overlooked aspect is the inference efficiency, usually measured in multiply-accumulate (MAC) count. This is particularly critical in use cases where inference bandwidth is greatly limited by hardware constraints. To this end, we propose the Activation-Sharing Multi-Resolution (ASMR) coordinate network that combines multi-resolution coordinate decomposition with hierarchical modulations. Specifically, an ASMR model enables the sharing of activations across grids of the data. This largely decouples its inference cost from its depth which is directly correlated to its reconstruction capability, and renders a near $O(1)$ inference complexity irrespective of the number of layers. Experiments show that ASMR can reduce the MAC of a vanilla SIREN model by up to 500$\times$ while achieving an even higher reconstruction quality than its SIREN baseline.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes ASMR, a new architecture for implicit neural network (INR) representation. The main selling point is the very low inference cost. ASMR encodes a signal by creating a hierarchical partitioning of the input domain and associating to each partition level (except the first) one transformation layer and one modulation layer.  More precisely, each input dimension of size $N$ is partitioned in $L$ levels such that each level $x_i$ is of a given size $B_i$ and the product of all level sizes correspond to the size of the original dimension.  Every level $i$ of the model takes as input coordinate of that level ($x_i$), transforms it through a neural layer (the "modulator"), upscales both a linear transformation of the output of the previous layer and the output of the modulator, sums them and applies a non-linear function. This leads to reduced computation at inference: at a given layer $i\in\{1,...,L\}$ of the model, the number of computed values is multiplied by $B_i$ (for each dimension). In other words, the number of computed values increases from $B_0$ to $N$ with the depth of the model and values computed in the earliest layers are reused gracefully for many coordinates.

The paper then provides a proof that this architecture leads to an inference cost that is upper bounded by a value that does not depend on the depth of the network. Finally, experiments measures the inference cost and the reconstruction quality on images (Pluto and Kodak) and one video. Meta-learning and the impact of various strategy for the hierarchical decomposition are also evaluated and discussion in two distinct experiments. Between 2 and 5 baselines are considered in most experiments.

### Strengths
originality:
- As far as I know, the proposed approach is novel.

quality:
- The numerical experiments support the claim that ASMR inference cost does not depend on the number of layers.
- Inference cost is measured in MACs and not in running time.
- Despite the lower inference cost, representation quality is similar to or better than the baselines used.
- Comparison to the most relevant approaches kilo-NERF and LoE.

clarity:
- The paper is reasonably well written and clear.
- Experiments are well explained.

significance:
- Inference cost for INR is imho an important and understudied problem.
- The results seem very good.

### Weaknesses
quality:
- While several baselines are used in all experiments, instant-NGP is not included in the experiments measuring inference cost. As Instant-NGP is known for its fast inference, I think it is not possible to know whether ASMR is really better than grid-based approaches.
- In Figure 2, left, kiloNERF always has a higher MAC count than ASMR. In Figure 2, right, kiloNERF often has a lower MAC than ASMR. From the text, I assumed these were the same trained models. This looks inconsistent to me.
- In Section 3.3, the paper states that the per-sample inference cost depends only on the width of the model and is independent of the depth. I am no sure I agree with that statement, as the depth depends on the choice of the basis and thus on the width of the layers. For example, as far as I understand, doubling the size of the basis also means halving the number of layers.

clarity:
- The paper did not look polished in some places to me. For example, I am under the impression that multiple notations are used to refer to the same or very similar things, which was confusing to me. Here are a few examples:
   - a base is denoted by $|x_i$ in Figure 1 and $B_i$ in Section 3.1 (where Figure 1 is referenced)
   - the cumulative product of bases is denoted by $|x_{i...0}|$ in Figure 1 and by $C_i$ in Section 3.1.

significance:
- For the video experiment, a single video is used. I find that a bit underwhelming as recent papers have typically used video benchmarks for evaluation.

### Questions
I would be grateful to the authors for commenting on the weaknesses listed above.

Details/suggestions:
- Upon my first reading of the paper, I had some trouble understanding Figure 1. I think it would be helpful, but not necessary, to mention that all bases are of size 2.
- There is a problem with equation 1: Section 3.3: Equation 1 can be rewritten as [mathematical expression] (1).

Typos:
- Figure 1, b: hierarcical

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work (ASMR), studies a way of reducing the inference cost of coordinate networks. ASMR combines multi-resolution coordinate decomposition with hierarchical modulations, by sharing activations across grids of the data which decouples the inference cost from the depth. Comparisons against the popular SIREN model are given which shows ASMR outperforms SIREN in both computing cost and reconstruction performance.

### Strengths
1. The pipeline is elegant and seems not difficult to implement. The preconstruction metrics are at least on par with SIREN while the theoretical computation cost is vastly reduced.

### Weaknesses
W1. The number of MACs is certainly a good indicator of inference speed. It would be much more convincing to also have actual inference wall clock timing on several hardware platforms.

W2. It is okay to test only on image/video fitting tasks, but a study on other data types such as audio or 3D shape could be much more convincing to show the generality of the multi-resolution activation-sharing schema.

W3. The word ‘inference bandwidth’ is kinda misused. Maybe consider ‘throughput’ or ‘latency’.

Typo:

W4. Section 3.3 “Equation 1 could be rewritten as”. The original equation 1 was not labeled.

### Questions
1. How were the MACs counted (in terms of software implementation)? 
2. My understanding from Figure 1. c about the modulator M is it adds the same pattern details to coarser coordinated activations (upsampled). It is counterintuitive why the sample pattern beats SIREN. Could the authors provide some explanations or intuitions?

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper tackles the inefficient inference of INRs. As INR requires computing the individual coordinate's value on all layers, the increase in depth largely affects the inference efficiency, which is a serious problem for high-resolution signals. To reduce the computation, this paper suggests sharing the activation by incorporating multi-resolution coordinates and hierarchical modulation. To be specific, these multi-resolution coordinates are shares the activation and use up-sampling to save the activation storage. The authors demonstrate the efficiency of the proposed method on image and high-resolution signals (e.g., extreme high-resolution image and video).

### Strengths
The overall writing is clear and the paper is well-organized.

Tackles an important problem, i.e., heavy inference time (cost) of existing INRs.

The overall idea is sounds and reasonable to reduce the inference cost of INRs.

### Weaknesses
**Missing important baseline/related work.**
- The overall concept is highly similar (although the detail is different) to [1].
- Using multi-scale INR is explored in [1] (also, [2] uses the same idea in the video domain). Use upsampling (bilinear, nearest, etc) to reduce the number of forward passes by sharing the activations.
- [1] also uses modulation as well.
- Due to the existence of [1], I think the paper should change the overall claim, e.g., the first INR to decouple MAC from its depth or using shared activations.
- Also, the comparison with Multi-scale INR [1] is definitely needed.

**Missing related works: INR modulation.**
- There exist more recent modulation techniques in the field compared to bias modulation [3,4]. It would also be interesting if ASMR uses more advanced modulations, e.g., low-rank modulation for SIREN [3].
- missing a reference [9], which is the same method as COIN++ but used for other downstream tasks (e.g., generation, classification).

**The method is somewhat non-trivial to use for 3D scene rendering**
- Note that 3D is one of the most famous applications of INR.
- For 3D INRs, there is some paper that tackles a similar issue [5].

**The experiment section should be improved**\
(i) Recently, there have been several papers that tackle modality-specific INRs. The authors should claim the benefit of using ASMR compared to modality-specific ones.
- First, the authors should show multiple modalities rather than images. (Note that video only shows one example). Considering Audio or Manifold [8]
- Second, the authors need to discuss the benefit of using ASMR over modality-specific INRs. For instance, there are several video INR papers [6,7] showing efficiency.
- Finally, I think ASMR is hard to use for 3D scenes, so it has less benefit compared to SIREN or FFN.

(ii) Only used one video sample for evaluation. 
- Rather than reporting one sample, it is better to consider a dataset.

(iii) It is hard to understand the intention of Section 5.2 (as it is somewhat trivial). 
- I think the authors are trying to point out that. 
- (a) ‘ASMR uses modulation, but there is no specific modulation technique for multi-grid based INRs’
- (b) ‘meta-learning modulation is important when using INRs for downstream tasks due to the dimensional efficiency.’ [8,9]
- I think this claim is not new to the community, and the modulation technique that ASMR is using is from [8,9].

(iv) Comparison with grid-based baselines are missing, e.g., Instant-NGP [10]. 
- Instant-NGP shows image, high-resolution image, and video experiments. 
- It is very worth comparing the efficiency as these methods are proposed for efficiency.

**Summary**\
Overall, I quite like the method, but there are several things to be improved, e.g., the major claim should be changed due to the missing important baseline, and comparing with grid-based INRs like instant-NGP [10] and Multi-scale INR [1]. I kindly request the authors to rewrite certain parts and compare ASMR with the baseline in detail throughout the rebuttal.

**Reference**\
[1] Adversarial Generation of Continuous Images, CVPR 2021\
[2] Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks, ICLR 2022\
[3] Modality-Agnostic Variational Compression of Implicit Neural Representations, ICML 2023\
[4] Versatile Neural Processes for Learning Implicit Neural Representations, ICLR 2023\
[5] Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields, ICCV 2021\
[6] NeRV: Neural Representations for Videos, NeurIPS 2021\
[7] Scalable Neural Video Representations with Learnable Positional Features, NeurIPS 2022\
[8] COIN++: Neural Compression Across Modalities, TMLR 2022\
[9] From data to functa: Your data point is a function and you can treat it like one, ICML 2022\
[10] Instant Neural Graphics Primitives with a Multiresolution Hash Encoding, SIGGRAPH 2022

### Questions
Please refer to the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
