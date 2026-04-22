# VMoBA: Mixture-of-Block Attention for Video Diffusion Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
The quadratic complexity of full attention mechanisms poses a significant bottleneck for Video Diffusion Models (VDMs) aiming to generate long-duration, high-resolution videos. While various sparse attention methods have been proposed, many are designed as training-free inference accelerators or do not optimally capture the unique spatio-temporal characteristics inherent in video data when trained natively. This paper introduces Video Mixture of Block Attention (VMoBA), a novel sparse attention mechanism specifically adapted for VDMs. Motivated by an in-depth analysis of attention patterns within pre-trained video transformers, which revealed strong spatio-temporal locality, varying query importance, and head-specific concentration levels, VMoBA enhances the original MoBA framework with three key modifications: (1) a layer-wise recurrent block partition scheme (1D-2D-3D) to dynamically adapt to diverse spatio-temporal attention patterns and improve efficiency; (2) global block selection to prioritize the most salient query-key block interactions across an entire attention head; and (3) threshold-based block selection to dynamically determine the number of attended blocks based on their cumulative similarity. Extensive experiments demonstrate that VMoBA significantly accelerates the training of VDMs on longer sequences, achieving 2.92$\times$ FLOPs and 1.48$\times$ latency speedup, while attaining comparable or even superior generation quality to full attention. Furthermore, VMoBA exhibits competitive performance in training-free inference, offering 2.40$\times$ FLOPs and 1.35$\times$ latency speedup for high-res video generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper examines the quadratic complexity of full attention mechanisms and proposes a novel and efficient sparse attention mechanism for video diffusion models. The presented method is called Video Mixture of Block Attention (VMoBA), which is an improvement form existing Mixture of Block Attention (MoBA) from three different key modifications. The contributions of this work lies in three different aspects as outlined: a cyclical 1D-2D-3D block partition, a query-global block selection and a threshold-based block selection. All these contributions are well documented and related to the observations and motivations.

### Strengths
The main strengths of this paper is to adapt the existing Mixture of Block Attention (MoBA) for Video Mixture of Block Attention (VMoBA). This is not a trivial adaptation as the paper performs insightful analysis to draw the useful conclusions. 

The paper performs the analysis on a full attention DiT model and observe that different layers may have its own focus on spatial, temporal and its combination with respect to different layers. This put forwards the idea of layerwise block partition for 1-2-3D. This types of pattern distinction seem not be widely confirmed in the previous video diffusion models. 

Regarding the global block selection, the paper also makes a good observation by identifying the limitation of query-independent selection. For each attention head, all the queries are grouped to perform the block section. To determine the number of blocks selected, the paper adopts a threshold-based Block Selection to determine the number of selected blocks dynamically. This strength might not be that significant compared to the other two.

### Weaknesses
The paper also shows several weakness that requires further attentions. The paper uses Wan 2.1 1.3B as the base model and implements the proposed idea. All assumptions such as the 1-2-3D attention patterns were derived from such full attention DiT. It is unclear whether this VMoBA can also be adapted to other video diffusion models based on full attention DiT. This needs to be further validated with other full attention DiT models. This could be the main weakness of this paper so far. 

The paper could better lay out the performances for the proposed method compared to the others listed in Tables. The paper uses the highlighted color to indicate the performance from the proposed method. However, it is better to indicate which metric the proposed algorithm performs the best, similar to other used baselines. After all, based on the given tables results, the proposed method does not perform the best among all the metrics. This could present some confusions. 

The assumption about heads having different concentration levels motivate the design threshold-based block selection. This assumption also relies on some empirical observation where 30% and 50% cutoff lines for each head was used. These numbers could be different across different models. It will be nice to have a more generalized treatment of these selections. 

The paper shows some interesting results on training longer sequences, which is critical in video diffusion models. How this long-sequence capability modeling is related to the proposed innovations are not very well explained. The paper could offer more insights on why the proposed designs are more suitable, not just improving efficiency and maintaining the accuracy simultaneously.

### Questions
1. Do the cyclical 1D-2D-3D scheme generate the tokens independentlyor eventually resulting in a 3D patch token based on the Eqns. (1) and (2)?

2. Can the proposed scheme generalize to other full attention DiTs? If yes, what is the common assumption for this generalization? Have you explored other full attention DiTs for the proposed method?

3. The paper relies on many insightful observations, which is good. But how will these observations generalize, especially when certain hyper-parameters were used?

4. Could the paper also elaborate on how the proposed schemes are particularly useful for the long video sequence generation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
VMoBA introduces a trainable sparse attention mechanism tailored for Video Diffusion Models. Building on MoBA, it integrates a 1D–2D–3D recurrent block partition, global block selection, and threshold-based adaptive sparsity to better capture spatio-temporal locality in video data. Compared to full attention, VMoBA achieves up to 2.9× FLOPs reduction and 1.5× speedup during training, while maintaining comparable or improved generation quality, and also accelerates training-free inference for high-resolution videos。

### Strengths
1. The paper diagnoses three phenomena in DiT attention—1D/2D/3D locality, uneven query importance, and head-wise concentration—then maps them to three design choices (1–2–3D recurrent partitioning, global selection, and thresholded selection). This “observations → mechanisms” linkage is well argued.
2. Unlike training-free sparse attention method, VMoBA is a trainable block-sparsity scheme intended to replace full attention during training . This positions it to deliver training compute savings where many inference-only methods (DiTFastAttn, SVG) cannot.
3. On Wan-2.1-1.3B fine-tuning with extended tokens (55K/56K), VMoBA reports 2.83–2.92× FLOPs reduction with 1.44–1.48× end-to-end training time speedup, while matching or slightly exceeding Full Attention on several VBench aspects (e.g., ImageQual +3.3% at 141×480×832).

### Weaknesses
1. The paper reports ~2.40× FLOPs reduction but only ~1.35× end-to-end latency speedup in the training-free 720p setting. A deeper breakdown (kernel MFU, QK/softmax/AttnV time, IO cost) is needed to explain the under-translation from theoretical to realized speed.
2. Several contributions hinge on MoBA being less efficient, yet the paper does not thoroughly analyze why MoBA is inefficient and why the proposed methods can make it more efficieint.
3. Strong, recent sparse-attention baselines are missing. In particular, SpargeAttn (universal, training-free) and STA (trainable) both report substantial end-to-end gains on state-of-the-art video DiTs; including them would calibrate the claimed benefits. 
4. Since VMoBA is trainable, the paper should report backward attention timing, MFU in backward, or end-to-end training throughput (e.g., it/s or tokens/s) againt full attention.
5. Only five VBench aspects are reported. Given VBench’s breadth, either report the full score or provide a principled selection rationale; when PSNR drops, include human blind-preference to reflect perceptual quality in Table 1. Also consider including VBench 2.0.

### Questions
1. You use “sparsity” to mean the fraction of computation retained vs. full attention. Standard usage is density = kept fraction, sparsity = skipped fraction?

2. Does 𝜏 =0.25 mean accumulating normalized attention mass to 25%? If so, it would seem a bit counterintuitive. I would think only covering 25% of attention score to bring huge quality loss. 

3. With thresholding, how much does actual density/e2e runtime vary across prompts and seeds? 

4. How is the sparsity in Table 2 calculated? Does it take into a account the fact that the first 25% timestep is full attention?

5. Table 2 lists MoBA dynamic degree ≈ 5.8%. Is this value correct? Why is it so low compare to other methods?

6. Why is “Dynamic” absent in Table 1, and why are only five VBench aspects reported?

6. Why presents training-free results at 81×720×1280  for Wan-2.1-1.3B, a model trained at 480p?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Video Mixture of Block Attention (VMoBA), a novel sparse attention mechanism tailored for Video Diffusion Models (VDMs). The authors introduce three key innovations: (i) a layer-wise recurrent 1D- 2D-3D block partitioning scheme, (ii) global block selection that selects the most salient query-key block interactions across all queries per head, and (iii) threshold-based dynamic block selection that adapts the number of attended blocks based on cumulative similarity. Experiments on long-sequence video generation show that VMoBA achieves faster training and fewer FLOPs than full attention, while matching or even surpassing its generation quality. VMoBA also performs competitively in training-free inference settings.

### Strengths
- The three proposed modifications (1D-2D-3D partitioning, global selection, threshold-based sparsity) are well-motivated by the observed limitations of applying MoBA naively to video. Each component is clearly linked to a specific empirical observation.
- The authors evaluate VMoBA in both training-based and training-free settings across multiple resolutions, using standard metrics (VBench, PSNR) and complete ablation studies. VMoBA achieves FLOPs reduction and training-time speedup with minimal loss in generation quality. Qualitative results further support the claims.
- The writing is good and is easy to follow. Method illustration and implementation details are well documented.

### Weaknesses
- The study omits some recent linear or hybrid video attentions that could serve as stronger baselines, such as STA[1] and RainFusion[2].
- The paper should include more human evaluation. Human judgment on video quality and video consistency is crucial for assessing the performance.
- In the global selection part, this module prioritizes key blocks with the highest overall significance, but may overlook certain keys that are locally relevant to queries yet have low global scores. The high-frequency details in generated videos may be affected.
- Have you tried other fusion methods to fuse the tokens in a block? Does mean pooling harm the diversity within a block?

[1] Fast Video Generation with Sliding Tile Attention

[2] RainFusion: Adaptive Video Generation Acceleration via Multi-Dimensional Visual Redundancy

### Questions
- The hyperparameter \tau controls the trade-off between speed and quality. If there exists a general value for all video types? If not, how to choose a proper \tau for new videos or resolutions?
- The paper notes that attention heads have different concentration levels. Have the authors analyzed whether VMoBA’s threshold-based selection leads to more specialized head behavior compared to full attention?
- Have you tested VMoBA within a few-step distilled or consistency diffusion frameworks to verify compatibility with fast-sampling variants?
- Could the cyclical 1-2-3D partitioning be learned end-to-end rather than fixed by layer index?

### Soundness
3

### Presentation
4

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
The paper proposed VMoBA, an improved version of Mixture of Block Attention (MoBA) for accelerating the training process of Video Diffusion Models on long sequence inputs. The authors point out the deficiency of the original MoBA's 1D partitioning scheme and apply layer-wise recurrent block partition, equipping the model with more spatio-temporal awareness. Then, they substitute global block selection for MoBA's query-wise key blocks number selection, which resolves resource under-allocation by precomputing all the QK block similarities for each attention head and generating block masks for top-K key blocks. The author replace fixed Top-K selection with threshold-based block selection mechanism, which enables the model to dynamically adjust the number of blocks selected across different heads and better fit the varying nature of similarity scores. They benchmark VMoBA against several training-free approaches under VBench, and conclude that VMoBA indeed reduces training time and FLOPS while still providing comparable performance. Lastly, they conduct relatively complete ablation studies, and conclude that the design choice is indeed a proper one.

### Strengths
Overall, the paper could be considered as an all-round incremental extension for MoBA. 
1. VMoBA provides 2.92 times FLOPS reduction and 1.44 times speed up compared to original model, which shows certain practical feasibility in terms of training.
2. The ablations are relatively complete, which ensures the model design is informed.

### Weaknesses
There several points about the main experiment that needs to be further clarified:
1. In training-based part of the main experiments, no trainable sparse attention pattern is included, which lacks certain universality in terms of benchmarking. 
2. In the training-based part of the main experiments, only one baseline method designed for accelerating training is presented.
3. The reason why methods for inference acceleration is used as a baseline for benchmarking training acceleration has not been clearly clarified.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
