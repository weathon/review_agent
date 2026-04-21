# SCHEME: Scalable Channel Mixer for Vision Transformers

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 5, 6, 3

## Abstract
Vision Transformers have received significant attention due to their impressive performance in many vision tasks. While the token mixer or attention block has been studied in great detail, the channel mixer or feature mixing block (FFN or MLP) has not been explored in depth albeit it accounts for a bulk of the parameters and computation in a model. In this work, we study whether sparse feature mixing can replace the dense connections and confirm this with a block diagonal MLP structure that improves the accuracy by supporting larger expansion ratios. To improve the feature clusters formed by this structure and thereby further improve the accuracy, a lightweight, parameter-free, channel covariance attention (CCA) mechanism is introduced as a parallel branch during training. This design of CCA enables gradual feature mixing across channel groups during training whose contribution decays to zero as the training progresses to convergence. This allows the CCA block to be discarded during inference, thus enabling enhanced performance with no additional computational cost. The resulting $\textit{Scalable CHannEl MixEr}$ (SCHEME) can be plugged into any ViT architecture to obtain a gamut of models with different trade-offs between complexity and performance by controlling the block diagonal structure size in the MLP. This is shown by the introduction of a new family of SCHEMEformer models. Experiments on image classification, object detection, and semantic segmentation, with different ViT backbones, consistently demonstrate substantial accuracy gains over existing designs, especially under lower FLOPs regimes. The SCHEMEformer family is shown to establish new Pareto frontiers for accuracy vs FLOPS, accuracy vs model size, and accuracy vs throughput, especially for fast transformers of small model size.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper concentrate on the  channel mixer or feature mixing block design and propose a block diagonal MLP structure with a lightweight, parameter-free, channel covariance attention (CCA) module to improve the feature clusters formed. Replacing the vanilla MLP with proposed SCHEME module, various experiments were conducted to validate its performance.

### Strengths
1. Channel mixer design, the focus of this paper, is a more than interesting topic for the backbone design.
2. Comprehensive vision experiments (*e.g.*, classification, detection, segmentation) and analysis (*e.g.* `Paragraph 3`in `Sec 4.2` for channel covariance attention) were conducted to validate the proposed method's performance.

### Weaknesses
1. With regard to the design of channel mixer, there exists many previous research, such as gMLP[1], channel aggregation module in moganet[2], S2-MLP[3]. In addition to comparing with models using traditional MLP, these should also be taken into account.
2. More scaling experiments are needed. In this paper, the biggest model was 58M in `Tab.1`, which may could only match the `base-size` model in general. To comprehensively validate the method's scaling capability in backbone, `Large` or even `X-Large` model are also need.
3. Typos for the dimension of $W_1$ in `Line 196,221`, $N$ -> $d$.

I would be more than willing to reconsider the rating based on the authors' response.

Refs:

[1] Liu, Hanxiao et al. “Pay Attention to MLPs.” Neural Information Processing Systems (2021).

[2] Li, Siyuan et al. “MogaNet: Multi-order Gated Aggregation Network.” International Conference on Learning Representations (2022).

[3] Yu, Tan et al. “S2-MLP: Spatial-Shift MLP Architecture for Vision.” 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2021): 3615-3624.

### Questions
See `Weaknesses`.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies the channel-mixer of modern vision architectures, and propose the SCHEME module, a new channel-mixer that split features to multiple sub channel groups and project these channel groups to lager dimensionality for learning richer internal representation. A channel covariance attention is designed to make information exchange among sub channel groups during training time. The experiments show that the proposed SCHEME outperforms the traditional channel-mixers (e.g., FFN/MLP) in several tasks.

### Strengths
1. The paper is well-structured, providing a clear and accessible overview of the motivation and methodology behind the proposed approach. The logical flow enables readers to easily understand both the problem addressed and the design of the proposed solution.

2. The methodology section is well-developed, with good supporting arguments for each component. The progression from MLP to BD-MLP, followed by the integration of CCA, is clear, well-motivated.

3. The use of CCA as a regularizer for BD-MLP to facilitate interaction across channel groups during training, while discarding it during inference to streamline computational efficiency, is an innovative and effective approach.

4. The proposed SCHEME module and its various configurations demonstrate impressive performance across a range of tasks.

### Weaknesses
1. The labels and coordinates in Figures 1, 3, and 4 are too small, making them challenging to interpret. For a basic academic paper, readability of figure details is essential.

2. The effectiveness of the proposed SCHEME is primarily validated on MetaFormer. While the authors briefly present results on other backbones like Swin and DaViT in Figure 4 (yet difficult to interpret). SCHEME’s generalizability is limited by testing primarily on ViT-based architectures, lacking ablation on modern ConvNet architectures (e.g., ConvNeXt [1], FasterNet [2], and InceptionNeXt [3]) that also use FFN/MLP modules.

3. Although SCHEME shows theoretical complexity advantages, its on-device efficiency is underexplored. Only limited throughput comparisons are presented in Tables 4 and 8. Since an on-device benchmark has been implemented, including detailed on-device results across ablation and comparison studies would substantiate claims about SCHEME’s efficiency over conventional FFNs. Additionally, details on the benchmark configurations, hardware, and input shapes are necessary to ensure fair comparisons.

4. The CCA component functions as a channel mixer and resembles linear attention mechanisms in NLP and prior work in XCiT [4]. A discussion of the differences between CCA and these methods, alongside proper citations, would clarify the novelty and address potential ethical concerns regarding prior art.

5. While SCHEME’s performance is strong and well-motivated, the contribution could be strengthened with deeper analysis. SCHEME integrates Block Diagonal MLP and CCA, elements with roots in existing literature. A more detailed explanation of how SCHEME specifically influences learned representations would underscore its contribution.

6. The comparison methods are outdated and do not represent state-of-the-art accuracy-efficiency models. Including comparisons with recent approaches like FasterViT [5], FastViT [6], and MobileOne [7] would enhance the experimental validity.

7. SCHEME is tested only on small model scales (<60M), which aligns with its efficiency-focused motivation. However, a benchmark of on-device speed on mobile devices is necessary to validate the proposed method’s real-world application, as many recent studies have done [6,7].



[1] Liu Z, Mao H, Wu C Y, et al. A convnet for the 2020s[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 11976-11986.

[2] Chen J, Kao S, He H, et al. Run, don't walk: chasing higher FLOPS for faster neural networks[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023: 12021-12031.

[3] Yu W, Zhou P, Yan S, et al. Inceptionnext: When inception meets convnext[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 5672-5683.

[4] Ali A, Touvron H, Caron M, et al. Xcit: Cross-covariance image transformers[J]. Advances in neural information processing systems, 2021, 34: 20014-20027.

[5] Hatamizadeh A, Heinrich G, Yin H, et al. Fastervit: Fast vision transformers with hierarchical attention[J]. arXiv preprint arXiv:2306.06189, 2023.

[6] Vasu P K A, Gabriel J, Zhu J, et al. FastViT: A fast hybrid vision transformer using structural reparameterization[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 5785-5795.

[7] Vasu P K A, Gabriel J, Zhu J, et al. Mobileone: An improved one millisecond mobile backbone[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023: 7907-7917.

### Questions
Please first deal with the major concerns above. There are some minor issues below.

1. The learnable weight $\alpha$ enables CCA as a regularizer during training, discarded for efficiency during inference. However, this approach may create potential conflicts in optimization objectives. Why not consider using more established techniques, such as Structural Re-parameterization, to achieve structure changes between training and inference?

2. Clarification is needed on the smoothing factor $\tau$ — is it a learnable parameter or a constant?

3. There are a few typographical errors, such as a double 'of' in Line 041 of the abstract.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper study the channel mixer of ViT MLPs, showing that dense feature mixing can be replaced by sparse feature mixing of higher internal feature dimensionality for improved accuracy, without increased complexity.  The work presents a range of experiments that sufficiently support its claims.

### Strengths
1. The writing is easy to read and clearly explains everything in the paper.
2. The experimental result is good compared to the previous works. Empirically, the method seems to offer strong accuracy, compared to existing methods with similar architectures.

### Weaknesses
1. I am concerned about the generalization of this method to other data sets.
2. Some details are missing. For Block Diagonal MLP, how to split the feature vectors of (1)-(2) into disjoint groups? What criteria is it based on？What's the motivation for designing the Channel Covariance Attention (CCA)? It didn't show more details.
3. How is the design like channel mixer relate to improving performance for example? It seems inadequate that none of were seriously discussed in the manuscript.

### Questions
1. I am concerned about the generalization of this method to other data sets.
2. Some details are missing. For Block Diagonal MLP, how to split the feature vectors of (1)-(2) into disjoint groups? What criteria is it based on？What's the motivation for designing the Channel Covariance Attention (CCA)? It didn't show more details.
3. How is the design like channel mixer relate to improving performance for example? It seems inadequate that none of were seriously discussed in the manuscript.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes SCHEME (Scalable CHannEl MixEr), a channel mixing mechanism for Vision Transformers (ViTs). The method focuses on replacing the dense MLP layers in the transformer’s channel mixing block with a block diagonal MLP (BD-MLP) structure, allowing for larger expansion ratios and improving computational efficiency. Additionally, the authors introduce a Channel Covariance Attention (CCA) mechanism to enable inter-group communication, which is discarded after training to reduce inference complexity. The SCHEME mechanism is integrated into multiple Vision Transformer architectures and is evaluated across various benchmarks, showing improved accuracy while maintaining efficiency.

### Strengths
1. SCHEME introduces a clever way of reducing complexity by leveraging block diagonal MLPs, which directly targets the computational bottleneck of standard transformers.
2. The addition of the Channel Covariance Attention (CCA) during training adds flexibility and improves feature clustering without increasing inference complexity, making it an efficient regularization tool.
3. The approach is shown to be effective across multiple transformer backbones (MetaFormer, T2T, Swin), demonstrating its adaptability in various settings.

### Weaknesses
1. The manuscript contains several grammatical errors and formatting issues, such as the phrase 'In result' in the abstract, as well as figures that are too small to clearly convey the details, which fails to meet the standards of ICLR.
2. Experiments focuses heavily on comparisons with a few models like MetaFormer, lacking comparisons with state-of-the-art transformer-based models like PVT, UniFormer or TransNeXt.

### Questions
1. What is the interpretation of the gradual decay of the mixing weight \((1 - \alpha)\) over training epochs, and how does it affect model performance or convergence?

### Soundness
2

### Presentation
1

### Contribution
2
