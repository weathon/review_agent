# Efficient-LVSM: Faster, Cheaper, and Better Large View Synthesis Model via Decoupled Co-Refinement Attention

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Feedforward models for novel view synthesis (NVS) have recently advanced by transformer-based methods like LVSM, using attention among all input and target views. In this work, we argue that its full self-attention design is suboptimal, suffering from quadratic complexity with respect to the number of input views and rigid parameter sharing among heterogeneous tokens.  We propose Efficient-LVSM, a dual-stream architecture that avoids these issues with a decoupled co-refinement mechanism. It applies intra-view self-attention for input views and self-then-cross attention for target views, eliminating unnecessary computation. Efficient-LVSM achieves 29.86 dB PSNR on RealEstate10K with 2 input views, surpassing LVSM by 0.2 dB, with 2× faster training convergence and 4.2× faster inference speed. 
Efficient-LVSM achieves state-of-the-art performance on multiple benchmarks, exhibits strong zero-shot generalization to unseen view counts, and enables incremental inference with KV-cache, thanks to its decoupled designs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The work proposes an improvement to LVSM, an ICLR 2025 oral publication. The proposed method, called efficient-LVSM, uses a detached attention mechanism to enhance efficiency and final performance. The architecture separates the input view encoding from the output view encoding, except for layer-wise connections within the same layer. There are additional tricks, such as KV-cache, feature distillation, and alternating self-attention and cross-attention. The results are evaluated against a few other-shot novel-view synthesis methods. The additional detailed efficiency analysis and ablation are performed against LVSM.

### Strengths
- Good results
- Detailed explanation about the components

### Weaknesses
- The work heavily relies on LVSM. In addition to final quality, the efficiency assessment needs to be compared with other methods. 

- The figures about the attention mechanism play an important role in understanding the main idea of the paper. However, they are somewhat repetitive and could be condensed into fewer figures—namely, Figures 1, 2, 3, and 6 (and Table 1 as well). I suggest reserving additional space to lay out the results better. The current layout is not very effective. The wrap-around text is hard to follow, and the figures and tables are referenced far from their locations. Some detailed results can be deferred to the appendix to improve the flow and completeness of the descriptions.

- While the introduction describes the main contribution as decoupling the encoder and decoder with an efficient attention mechanism, the work is composed of multiple techniques from existing works, with too many subsections in both Section 2 and the results. While they are effective, the novelty can feel incremental, and the description is dispersed across several succinct sections, making it hard to get the whole picture.

### Questions
- Figure 1 is hard to interpret, especially the items in the Table. Some of the attributes need to be detailed either in the caption or in the text. For example, what are "spatialized pathways"?

Minor comments
- line 042: "to to" -> "to"t.
- In Figure 2, I believe there are "M" decoders instead of "N" decoders..?

### Soundness
3

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
This paper tackles feed-forward novel view synthesis task. It builds upon the framework of LVSM [1], which use a full-attention decoder-only transformer to render novel views conditioned on posed input views and target view plucker rays.  The key architecture change proposed by the authors are to replace the full attention to intra-image self-attention and cross-attention between target views and input views.  This change has two benefits:  1. When rendering new novel views, you don't need to recompute the key-value cache of input views. This also opens up application in incremental inference. 2. with above benefits, the performance (rendering quality) improves on Objecverse and Rel10K dataset, even with reduced training time.   The author also ablated with REPA showing that it can improve the PSNR. 



[1] LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias

### Strengths
1. Clearly justified and very reasonable architectureal change.  The full-attention used by original LVSM restricts its effiency in lot of usecases. And when comparing with the encoder-decoder version of LVSM, the author identified the major problem:  we need to use key-value cache of all layers!      
2. Strong empirical results. The experiment results on Objverse and Rel10K are quite strong, with much better rendering quality and less training time. 
3. The author shows a very interesting study about repa loss (distill from DINO v3). I original thought such semantic loss would not be useful for novel view synthesis task, but seems that it's quite helpful! This is very interesting.

### Weaknesses
I do have a few comments. I think the authors should list more training details for their methods and their ablation experiments. The batch size, and total number of training iterations. I think it's missed.  

For training batch sizes, there are some tiny but important details, the original LVSM need to repeat the batch to make sure that each pass of the model only contains one target view, and this is one of the core-reason that the original decoder only LVSM is expensive in training and inference.  So highlighting this training details difference, and show that how many input views and target views totally being used during training is important.

### Questions
Kind of minor, but for task like novel view synthesis, showing some video comparison would be great!

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
- This paper proposes Efficient-LVSM, a transformer-based large view synthesis model designed to overcome the inefficiencies of the original LVSM architecture. 

- The key idea is to decouple the input-view encoding and target-view generation using a dual-stream co-refinement mechanism, combining intra-view self-attention for inputs and self-then-cross attention for targets. 

- The approach enables linear complexity in the number of input views, incremental inference with KV-cache.

- Efficient-LVSM outperforms state-of-the-art LVSM by 0.9dB PSNR on the RealEstate10K benchmark with 50% training time and achieves 2−4 times speed acceleration in terms of both training iteration and inference.

### Strengths
- The paper provides a systematic analysis of LVSM’s inefficiencies and derives a principled redesign via a decoupled encoder-decoder. The KV-cache design enabling incremental inference is a noteworthy contribution for real-time or interactive view synthesis, rarely explored in prior feedforward NVS models.

- Efficient-LVSM achieves state-of-the-art reconstruction quality on both scene-level (RealEstate10K) and object-level (GSO/ABO) benchmarks. The reported 0.9 dB PSNR gain, 4× inference speed-up, and 50 % reduction in training time represent a strong improvement over prior LVSM baselines

- The experiments are thorough: comparisons across datasets and baselines, scaling trends, ablations on architectural variants (self vs cross vs co-refinement), REPA distillation effects, model size, convergence curves, and zero-shot generalization.

- The paper is well written and includes complete training details, REPA settings, and commitments to code release.

### Weaknesses
- The dual-stream co-refinement design is highly similar in spirit to the MM-DiT block in terms of architecture introduced by Stable Diffusion 3 (2024). The authors are encouraged to cite MM-DiT and clarify how Efficient-LVSM extends this pattern to the feedforward NVS setting.

### Questions
The paper is well presented; I don't have further questions.

### Soundness
4

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
5

### Summary
This study proposes Efficient-LVSM, which modifies the neural network architecture of LVSM to make its training and inference more efficient. The key modification is that Efficient-LVSM incorporates a cross-attention block, while stacking different self-attention blocks to extract query and key/value vectors from the target view and input views, respectively. By avoiding full self-attention between the input and target views, Efficient-LVSM reduces computational costs and enables feature caching of input views across different target views. Experimental results demonstrate the effectiveness of the proposed architecture.

### Strengths
S1. Decomposing the full self-attention into different modules with cross-attention makes sense, as has been explored in various domains to design more efficient neural network architectures.

S2. The experiments show significant improvements in novel view synthesis, while also enhancing inference efficiency.

S3. The modified architectures can incorporate REPA to effectively train the hidden representations of input views.

### Weaknesses
W1.
The technical contribution is limited. Instead of using full self-attention across input and target views, employing cross-attention is a typical design choice for improving training and inference efficiency [NewRef-1].

W2.
Despite the performance improvements, Efficient-LVSM cannot address the fundamental limitations of LVSM. For example, its architecture cannot account for the alignments either between generated target views or within the input views. Therefore, the overall impact of this study is limited to being an improved version of LVSM rather than a fundamentally new approach.

W3.
The experimental analysis could be strengthened by incorporating more baseline methods. Please refer to my detailed comments below.

W4. [Minor points — not affecting the score] The paper writing should be improved.
- Line 42: "ability to to learn" -> "ability to learn"
- Line 92: input and target inputs share the same subscription $i$ , making the readers confused.
- Need to use mathbf clearly. For example, $S_i$ is also a set of tokens, but it does not use the bold text type. In Eq. (1), why $R_i$ uses the bold type, while the other parts do not use.
- Figure 1 uses a check box, but rendering quality or efficiency cannot be described with the check box. For example, we cannot say that the baseline has no efficiency or quality.
- In Figure 2, what does the asterisk mean? I guess it describes a shared weight between modules, but it is neither a common way to describe it nor explain the details.
- Line 133 -- $p$ is not defined. $N$ is the number of input views, but $n$ is used for both input/target views.
- Eq. (2) uses $l$ without any definition.


[NewRef-1] Jeong, Yoonwoo, et al. "NVS-Adapter: Plug-and-play novel view synthesis from a single image." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

### Questions
Q1. Positional embeddings are important for maintaining alignment between views, but there is no description provided in the paper. How are positional embeddings applied in the self-attention and cross-attention layers?


Q2. Can the authors discuss the limitations of this study to help readers understand potential directions for future work?

Q3. While keeping a single attention block, could we use different masking strategies instead of using separate modules? For example, a causal mask between input and target views could enable KV-caching of input views. Additionally, attention between input views could incorporate a masking scheme to prevent them from attending to each other for efficiency. We could also optionally use different projection layers while sharing a single attention block, as demonstrated in MMDiT [NewRef-2]. Although I agree that the proposed architecture achieves significant improvements in the experiments, I believe the design could be further simplified.


Q4. Figure 9(c) shows that Efficient-LVSM achieves 2× faster training to reach the same performance as LVSM, but the training does not appear to have converged yet. Could the authors provide more GPU hours to ensure full convergence and compare training efficiency more fairly? In addition, was REPA used in this setting? A comparison without REPA could also provide a clearer understanding of how Efficient-LVSM achieves efficiency and effectiveness purely from its architectural design, as analyzed in Table 1.

Q5. I wonder why the authors keep the generated target views independent. In addition, input views do not align their features with other input views. Is there any specific reason to restrict the receptive fields in this way?

Q6. In Table (d), could the authors provide a comparison between the 12-layer self-attention model and the 12+12 Efficient-LVSM? I believe the 12+12 Efficient-LVSM does not actually contain 24 layers but 12, so comparing it with the 12-layer self-attention model would be a fairer setting.

Q7. I initially expected that increasing the number of input views in the previous LVSMs would significantly increase GPU memory usage. However, Figure 9(a) shows that the GPU memory of the LVSM Decoder-Only model does not increase much as the number of input views grows. Could the authors elaborate on this result?

Q8. How much does KV-caching improve inference speed as the number of input views increases?


[NewRef-2] Esser, Patrick, et al. "Scaling rectified flow transformers for high-resolution image synthesis." Forty-first international conference on machine learning. 2024.

### Soundness
3

### Presentation
4

### Contribution
2
