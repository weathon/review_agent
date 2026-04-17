# Revisiting Multimodal Positional Encoding in Vision–Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Multimodal position encoding is essential for vision-language models, yet there has been little systematic investigation into multimodal position encoding. We conduct a comprehensive analysis of multimodal Rotary Positional Embedding (RoPE) by examining its two core components: position design and frequency allocation. Through extensive experiments, we identify three key guidelines: positional coherence, full frequency utilization, and preservation of textual priors—ensuring unambiguous layout, rich representation, and faithful transfer from the pre-trained LLM. Based on these insights, we propose Multi-Head RoPE (MHRoPE) and MRoPE-Interleave (MRoPE-I), two simple and plug-and-play variants that require no architectural changes. Our methods consistently outperform existing approaches across diverse benchmarks, with significant improvements in both general and fine-grained multimodal understanding. Code is avaliable at https://github.com/JJJYmmm/Multimodal-RoPEs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the design of multimodal position embeddings for VLMs. Specifically, the authors analyze existing RoPE variants and their designs used in VLMs, and summarize three key guidelines: the position design for different modalities, the frequency allocation for different positional axes, and the compatibility with vanilla text-only RoPE. Guided by these principles, the paper proposes two new designs: MHRoPE and MRoPE-I. Both methods enhance the MRoPE by incorporating a ‘spatial-reset' mechanism, and they introduce two novel frequency allocation strategies. Experiments demonstrate the effectiveness of the two proposed methods.

### Strengths
* The paper provides a comprehensive and systematic discussion of the multimodal RoPE methods.
* The proposed two methods MHRoPE and MRoPE-I are simple while well-motivated, and the results show their effectiveness.
* There are extensive experiments, with various benchmarks across image understanding, video understanding and visual grounding tasks.

### Weaknesses
* Validation on only a single VLM instance (i.e. the experiments are based on Qwen2.5VL) may not be sufficient to demonstrate the robustness of the methods.
* It is surprising that MRoPE underperforms Vanilla RoPE on many benchmarks (Table 2). Perhaps the authors could provide a deeper discussion about the results.

### Questions
For MHRoPE, is there any experiment or discussion on how the heads are allocated?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a systematic study of multimodal rotary positional embeddings (RoPE) in Vision-Language Models (VLMs). The authors decompose the multimodal RoPE design space into three core aspects—position design, frequency allocation, and compatibility with text-only RoPE—and identify common pitfalls in existing methods. Building upon these insights, they propose two new variants: Multi-Head RoPE (MHRoPE) and MRoPE-Interleave (MRoPE-I), along with a spatial-reset mechanism. These designs aim to ensure positional coherence, full frequency utilization, and faithful transfer of textual priors from pre-trained LLMs. Through extensive experiments on over twenty benchmarks covering image, video, and grounding tasks, the proposed methods achieve consistent and notable improvements over prior work.

### Strengths
- The paper provides one of the most systematic and insightful examinations of positional encoding in multimodal contexts to date. The decomposition into design axes (position, frequency, and compatibility) clarifies the often opaque design space of VLM encodings.

- MHRoPE and MRoPE-I are conceptually straightforward and “plug-and-play,” requiring no architectural modifications, which increases their practical utility.

- The authors evaluate across a wide range of benchmarks, demonstrating consistent gains and analyzing failure cases of prior methods.

- Extensive ablations provide convincing evidence for the effectiveness of spatial-reset and interleaved frequency allocation.

### Weaknesses
- While the analysis is systematic, the theoretical understanding of why certain frequency allocations or resets perform better remains mostly empirical. A deeper mathematical treatment could strengthen the claims.
- Although improvements are consistent, in some cases (e.g., video benchmarks), the gains are modest, which might question the generality of the benefits.
- Dependence on Qwen2.5-VL backbone. All experiments rely on one model family. It is unclear if the conclusions would hold equally well for other architectures (e.g., LLaVA or Gemini-style models).
- The authors note MHRoPE’s potential scalability but provide little empirical evidence of its performance on extremely large-scale or high-resolution data.
- The improvement in the experimental results is not significant.

### Questions
- Carefully review Table 2. It seems that the calculations for some data are incorrect, especially in the "overall" column and the parts in red font.
- How sensitive are the proposed methods to the choice of rotary base or context length during training? Would a smaller base or shorter context alter the observed trends?
- Can the authors elaborate on whether spatial-reset interacts with patch embedding granularity (e.g., ViT patch size)?
- Is there a possibility that interleaved frequency allocation could interfere with frequency-aware extrapolation methods when used in ultra-long video sequences?

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
3

### Summary
The paper proposes MRoPE-I and MHRoPE, two improved methods for using rotary embeddings to encode position information in VLMs. MRoPE-I and MHRoPE improve on existing techniques by:introducing a "spatial-reset" to fully decouple position and time encoding and ensuring that positional infomration is encoded across the entire frequency spectrum. The work proposes with two methods for encoding frequency across the full spectrum: head specialization (MHRoPE) and interleaved (MRoPe-I).

### Strengths
- In the reivewers opinion, the ideal way to encode position in VLMs is unclear and warrants further exploration. In particular, well put together ablations are interesting and useful.

- The severe degradation of vanilla RoPE on long video inputs (and relatively strong performance on shorter visual tasks) and observation of a “visual attention sink“ across modalities, is interesting.
    
- The paper is clear in its presentation of current challenges in multimodal position embeddings, and its proposed solution.

### Weaknesses
- The vision encoder of Qwen2VL / connector MLP of Qwen2.5VL has been trained to output visual token that are used with MRoPE-like position encodings. This may disadvantage approach such as VideoRoPE, HoPE and CircleRoPE. Initializing from  vision encoder that has not been trained for use in a VLM would be better. This problem also seems to result in disabling the absolute time encoding scheme proposed for MRoPe [1]. Therefore, the MRoPe baseline is not one-to-one with how it is used in QwenVL-2.5. Overall, heavy use of QwenVL components makes interpreting experiments difficult and introduces confounding factors.
    

- The experimental setup is quite limited. Only 1 LLM is considered, and only at 7B parameter size. Showing generalization across architectures would be valuable.
    

- The stages followed for training the VLMs are not standard, especially the choices to initialize the vision encoder and MLP layers with QwenVL components. It would be better to recreate an established process for constructing a VLM from existing work [2,3].
    

- A discussion about the similarities of MRoPE-I and MHRoPE to MRoPE **given the training and testing distribution** is essential. In the case of a single visual element and no system prompt the proposed approach is very similar MRoPE. What/if system prompts are used for each evaluation is not clear. However, the single image case is certainly the dominant case among evaluations. Are system prompts are used during training? Is the system prompt of consistent length during training / evaluation? This is important for understanding exactly how position is encoded (i.e. what do equations 3 and 4 realistically evaluate to).
    

- The paper is very vague about what training data is used. The reviewer can only find that “2M high-quality supervised fine-tuning (SFT) samples” were used. Nothing about the breakdown between modalities, conversations vs. captioning, particular data sources, etc. Figure 1 demonstrates the position encoding for conversations containing multiple interleaved visual elements. Did the training contain conversations with multiple visual elements, if so, what proportion of the training data was it? Such details are crucial for understanding how position is realistically encoded and reproducibility.
    

- Some details are missing from training settings, such as the optimizer used and it’s respective settings. Justification for the choices of hyperparameters would also be valuable.
    

- It is claimed that MHRoPE / MRoPE-I result in “more focus on visual information.“ However, that is never justified experimentally or formalized. Do the proposed methods have larger attention scores (on visual tokens) than baselines?
    

- The benchmarks used to evaluate MRoPE-I and MHRoPE overwhelmingly follow the structure of a single visual document followed by some text (usually a short question). This does not test the proposed method across a diverse range of positional encoding setups. Comparisons across conversations with multiple visual documents (akin to Figure 1) would be valuable.

Minor:  

- Table 2 is a bit messy. Sometimes the best score is bold and in other rows the second best score is bold (for example, the Video row).
    
- Quotes open in the wrong direction in the manuscript, and are inconsistent throughout. Some are correct such as “1111...” whereas elsewhere it is flipped, such as ”attention sink," .

### Questions
Several question can be found in the section above, in addition:  

- What is the performance if the number of temporal channels is less than 24 in the interleaved strategy?

- How are video frames sampled? Are a constant number of frames sampled uniformly or is fps sampling used? If so, what fps is used? Similarly, what image processing/scaling is used? Do the authors use QwenVL2.5’s dynamics scaling of visual tokens? Are the total number of frames sampled capped in long video? Do any of these settings differ between training and evaluation?

- How was the ChartQA visualization created in figure 2? It is the reviewer’s understanding that ChartQA has a single image per input, but figure 2 has two images.

---

[1] S. Bai *et al.*, “Qwen2.5-VL Technical Report,” 2025, doi: 10.48550/arxiv.2502.13923.  

[2] O. Zohar *et al.*, “Apollo: An Exploration of Video Understanding in Large Multimodal Models,” in *Proceedings (IEEE Computer Society Conference on Computer Vision and Pattern Recognition. Online)*, IEEE, 2025, pp. 18891–18901. doi: 10.1109/CVPR52734.2025.01760.  

[3] S. Tong *et al.*, “Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs,” 2024, doi: 10.48550/arxiv.2406.16860.

### Soundness
2

### Presentation
3

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
The paper re-examines how multimodal positional encodings (especially RoPE) should be designed for vision–language models, arguing that current extensions are ad hoc and often break either visual geometry or LLM compatibility. It identifies three principles—positional coherence, full frequency utilization, and preservation of textual priors—as necessary to avoid modality confusion, attention sinks, and degraded long-context/video performance. Based on this, the authors propose two plug-and-play variants, Multi-Head RoPE (MHRoPE) and MRoPE-Interleave (MRoPE-I), which give each axis (time, height, width) access to the full frequency spectrum while keeping text RoPE unchanged. Extensive experiments on >20 image, video, and grounding benchmarks show these designs consistently outperform vanilla RoPE, MRoPE, and recent video-focused RoPEs, especially on fine-grained and grounding tasks.

### Strengths
1. propose two plug-and-play variants, Multi-Head RoPE (MHRoPE) and MRoPE-Interleave (MRoPE-I), which give each axis (time, height, width) access to the full frequency spectrum while keeping text RoPE unchanged.
2. Extensive experiments on >20 image, video, and grounding benchmarks prove effectiveness of these designs
3. Easy to understand and comprehend. Figures are clear.

### Weaknesses
1. Limited experimented VLMs. The experiments only cover Qwenvl2.5. Broader range of models such as llama, phi and internvl series would better demonstrates the generalizability of the benefits from the two novel designs.
2. Increamental technical novelty compared with MRoPE. This paper propose spatial-reset and two frequency allocation modifications upon MRoPE. Despite effectiveness, the overall technical novelty compared to MRoPE and the original RoPE remains limited.
3. experiment with rapid temporal decay and spatial asymmetry problem, such as thousands of frames or extreme interleaving patterns where attention sinks typically reappear and see the performance

### Questions
Is MHRoPE and MRoPE-I more scalable than MRoPE and RoPE in the multimodal domain, such as better loss curve with proportionally increasing model size and data size?

### Soundness
3

### Presentation
3

### Contribution
3
