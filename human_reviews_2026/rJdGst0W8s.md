# Autoregressive Image Generation with Randomized Parallel Decoding

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
We introduce ARPG, a novel visual Autoregressive model that enables Randomized Parallel Generation, addressing the inherent limitations of conventional raster-order approaches, which hinder inference efficiency and zero-shot generalization due to their sequential, predefined token generation order. Our key insight is that effective random-order modeling necessitates explicit guidance for determining the position of the next predicted token. To this end, we propose a novel decoupled decoding framework that decouples positional guidance from content representation, encoding them separately as queries and key-value pairs. By directly incorporating this guidance into the causal attention mechanism, our approach enables fully random-order training and generation, eliminating the need for bidirectional attention. Consequently, ARPG readily generalizes to zero-shot tasks such as image in-painting, out-painting, and resolution expansion. Furthermore, it supports parallel inference by concurrently processing multiple queries using a shared KV cache. On the ImageNet-1K 256 benchmark, our approach attains an FID of 1.83 with only 32 sampling steps, achieving over a 30 times speedup in inference and and a 75 percent reduction in memory consumption compared to representative recent autoregressive models at a similar scale.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ARPG, a model designed to improve the efficiency of autoregressive image generation. It separates the generation process into two stages: Pass-1, which generates content representations using causal self-attention, and Pass-2, which decodes these representations using causal cross-attention. This separation allows for parallelization and significant improvements in inference speed and memory efficiency—up to 30× faster and 75% less memory. ARPG also maintains competitive image quality, with minimal loss of fidelity. The model excels in both speed and quality, making it suitable for real-time image generation tasks.

### Strengths
1. The motivation and observations in the paper are well-founded, highlighting the need for more efficient autoregressive image generation models.
2. The proposed method effectively decouples the generation process into two stages (Pass-1 and Pass-2) to improve efficiency without compromising quality.
3. The performance of ARPG is impressive, offering significant speedup and memory reduction while maintaining competitive image quality.

### Weaknesses
1. The description of Figure 3 is unclear, as it doesn't explicitly mention that it shows multiple attention heads, which may lead readers to think they are the same. 

2. In Table 2, the citation for **NAR** appears to be incorrect. It should be updated to reference the correct source, as the current citation does not align with the intended reference or context.

3. Due to ARPG separating Pass-1 and Pass-2 into two distinct stages, each handling different tasks, the model's interpretability is limited, making it difficult to pinpoint performance issues during debugging.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes ARPG, a novel autoregressive image generation framework that enables random-order, parallelizable token generation via a two-pass decoupled decoding mechanism: (1) a content refinement pass builds a causal, label-leakage-free memory from randomly ordered tokens; (2) a position-guided prediction pass uses [MASK] queries to decode multiple tokens in parallel. This design allows training under full AR causality while supporting flexible inference orders and block-parallel decoding. ARPG achieves good performance on ImageNet with significantly improved throughput and reduced memory, and demonstrates compelling zero-shot generalization and controllable generation.

### Strengths
1.Decoupling content modeling from position prediction is a conceptual advance that breaks the raster-scan bottleneck in AR vision models.
2.This paper has sufficient experimental verification and has verified the effectiveness of the method on various generation tasks such as c2i, t2i, and controllable generation.
3. This paper makes a detailed comparison of the existing AR-based models, which is helpful for understanding the uniqueness of the method proposed in this paper.

### Weaknesses
1. This paper contains ambiguous statements about efficiency improvement. For example, in the last sentence of the abstract section, the author claims: "achieving over a 30× speedup ... compared to representative recent autoregressive models ... ". In fact, the 30× speedup here is relative to the naive autoregressive model LlamaGen from a year ago. Compared to recent autoregressive models such as RandAR, the real advantage of ARPG in throughput is ~3× speedup. I think the author should have a more rigorous expression in the paper, especially in the explanation and comparison of quantitative indicators.

### Questions
I don't think this article has any fatal flaws. I only have a few minor questions to raise.
1. The resolution in the T2I generation does not seem to be mentioned in the paper.
2. I have some questions about the throughput in Table 3. Firstly, the throughput of the c2i model of LlamaGen-XL in Table 2 is 2.46, but it is 4.32 in Table 3. Why is the T2I model faster instead? I think this is abnormal. Moreover, the throughput of SD1.5 is too low. In fact, the inference efficiency of SD1.5 is higher than that of LlamaGen.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes ARPG, an efficient architecture for Random AR Image generation. First, the paper found that  (i) Rand AR require explicit positional information for the token to be predicted, and (ii) using position information tokens at the input token level hinders training efficiency and fails to achieve meaningful attention. Based on this, this paper proposes an efficient architecture for RandAR, which is built with a Pass-1 decoder trained only with visual tokens, and a Pass-2 decoder trained with cross-attention with position tokens. Specifically, the Pass-1 decoder outputs the KV cache of the current image token sequences, and the Pass-2 decoder uses this KV cache with position token queries for decoding. Experimental results showed that ARPG outperforms existing AR, PAR, and RandAR frameworks in terms of speed and quality across several benchmarks.

### Strengths
- The paper is well-written and easy to understand. The introduction, observations, and methodology are intuitive and clear.
- The paper presents interesting insights and observations regarding the characteristics of RandAR. To the best of my knowledge, these findings are novel.
- The proposed method is intuitively derived from the observations and simple.
- Experimental results show that ARPG outperforms existing AR models across multiple benchmarks.

### Weaknesses
- **Integrability with LLMs** : A key advantage of (raster) image AR or RandAR, which rely solely on self-attention decoder architecture, is that they can be easily integrated with existing LLM architectures without any changes. However, ARPG's reliance on cross-attention requires LLM architecture modification or, at least, fine-tuning by attaching Pass-2 decoder in parallel to the existing models.

- **T2I evaluation** : Experimental results on T2I evaluation are limited. In Tab.3, the model was only trained on a relatively small number of samples (4M) and actually shows worse performance. It is difficult to evaluate whether this architecture can be extended to large-scale T2I.

### Questions
- What are the advantages of AR modeling of image tokens over multi-modal discrete diffusion models like [1]? They also support the parallel generation of visual tokens. While I am aware that AR has advantages in KV-cache efficiency, I am just curious about the author's perspective.

- In Fig. 9 (b) and (c), what specific reason makes fewer generation steps actually improve the generation quality? Doesn't this result indicate that the current ARPG training is unstable?

[1] Yu, Runpeng, Xinyin Ma, and Xinchao Wang. "Dimple: Discrete diffusion multimodal large language model with parallel decoding." arXiv preprint arXiv:2505.16990 (2025).

### Soundness
4

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
3

### Summary
This work introduces ARPG (Autoregressive Randomized Parallel Generation), a method that incorporates a decoder-decoder architecture design and decouples positional guidance from content representation.

### Strengths
1. The motivation is clear, where the input sequence redundancy is a bottleneck for RandAR.

2. The experiments conducted on class-to-image generation, controllable generation, zero-shot generalization, and text-to-image generation verify the effectiveness of ARPG.

### Weaknesses
1. At the same level of model size, e.g. 320M in Table 2, ARPG outperforms RandAR significantly in both efficiency (throughput and memory consumption) and output quality. What is the underlying reason for such improvement? Does most of the improvement come from YOCO architecture design? 

2. RandAR is trained with a batch size of 1024 for 300 epochs, but ARPG is trained for 400 epochs with different batch sizes. I am wondering whether ARPG will keep the same advantage under the same training setting.

3. As a minor concern, I am curious about how ARPG with fewer sampling steps (32 steps) outperforms ARPG with more steps (64 steps)？For 256x256 image reconstruction, LlamaGen tokenizer achieves an rFID of 2.19 at 256×256 resolution. How can the gFID of ARPG surpass this upper bound?

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
