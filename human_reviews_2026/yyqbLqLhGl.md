# FLASH: Latent-Aware Semi-Autoregressive Speculative Decoding for Multimodal Tasks

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large language and multimodal models (LLMs and LMMs) exhibit strong inference capabilities but are often limited by slow decoding speeds. This challenge is especially acute in LMMs, where visual inputs typically comprise more tokens with lower information density than text --- an issue exacerbated by recent trends toward finer-grained visual tokenizations to boost performance. Speculative decoding has been effective in accelerating LLM inference by using a smaller draft model to generate candidate tokens, which are then selectively verified by the target model, improving speed without sacrificing output quality. While this strategy has been extended to LMMs, existing methods largely overlook the unique properties of visual inputs and depend solely on text-based draft models.
In this work, we propose \textbf{FLASH} (Fast Latent-Aware Semi-Autoregressive Heuristics), a speculative decoding framework designed specifically for LMMs, which leverages two key properties of multimodal data to design the draft model. First, to address redundancy in visual tokens, we propose a lightweight latent-aware token compression mechanism. Second, recognizing that visual objects often co-occur within a scene, we employ a semi-autoregressive decoding strategy to generate multiple tokens per forward pass. 
Experiments show that FLASH consistently outperforms prior speculative decoding approaches in both unimodal and multimodal settings, achieving up to \textbf{2.68$\times$} speed-up on video captioning and \textbf{2.55$\times$} on visual instruction tuning tasks compared to the original LMM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FLASH, a speculative decoding method for LMMs. The authors introduce two key components in the draft model: visual token compression to reduce redundant visual tokens, and a semi-autoregressive head to exploit spatial co-occurrence patterns in visual inputs. FLASH outperforms existing speculative decoding methods for LMMs.

### Strengths
1. The paper addresses an important and timely problem in speculative decoding for LMMs.
2. The paper is well written and easy to follow.

### Weaknesses
1. The core components (visual token compression and semi-autoregressive generation) are not novel. The main contribution is their effective integration, which is a valuable engineering solution but offers limited algorithmic novelty.

2. The two motivations (visual token redundancy and vision object co-occurrence) of this work are not convincing.
   
   a. Visual token redundancy: Context token count primarily affects prefill latency, not decode latency. Since speculative decoding targets decode-stage acceleration, why is visual token compression critical here?
   
   b. Vision object co-occurrence: The claim that visual input uniquely exhibits spatial co-occurrence is unconvincing. Spatial patterns like "in front of" or "on the table" are equally common in text-only contexts, thus the motivation for semi-autoregressive decoding needs stronger justification.

3. While the paper includes the 'Text-only' baseline from Gagrani et al. (2024), it omits Gagrani's multimodal drafting baseline. Furthermore, the paper lacks a comparison with MASSV.

4. Table 1 shows an inconsistency: for QwenVL-2.5 (temperature=1), Medusa-MM's speedup (2.35x) is superior to FLASH's (2.05x), yet the table seems to incorrectly mark FLASH as best. This raises reliability concerns. Furthermore, the speedup from FLASH often appears incremental.

### Questions
Please refer to the Weaknesses section above.

### Soundness
2

### Presentation
3

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
This work introduces a method to accelerate the inference speed of large vision-language models. The core idea is to leverage a smaller, faster "draft model" to generate a sequence of visual and text tokens as a "draft," which is then efficiently verified in a single parallel pass by the larger, more accurate "target model." By accepting long sequences of drafted tokens when the draft is correct, this speculative decoding approach significantly reduces the number of sequential forward passes required from the expensive target model, thus lowering overall latency.

### Strengths
The primary strength of this method lies in its practical utility for reducing the high computational cost associated with large multimodal models. By intelligently combining existing techniques like visual token compression and speculative decoding, it offers a direct path to faster inference without compromising the output quality of the original target model. The ability to verify multiple tokens in parallel is a clever way to exploit the architecture of modern transformers for a substantial speedup.

### Weaknesses
The primary weakness is the limited scope of the evaluation. While the reported speedup is promising, the benchmarks used do not cover a diverse range of real-world scenarios. The acceleration factor, presented as an average, may not be representative of performance on tasks with varying levels of predictability, such as complex visual reasoning versus simple image captioning. For instance, the actual performance gain in applications requiring high creativity or nuanced understanding could be significantly lower than reported. Furthermore, the novelty is somewhat constrained, as the approach is fundamentally an application and combination of pre-existing techniques rather than the introduction of a completely new paradigm.

### Questions
The methodology for creating the draft model raises a key question. Is the draft model a fine-tuned version of the same architecture as the target model? If so, have the authors considered replacing its backbone with a much smaller model? For example, use llava-7b as the backbone of the draft model for the target model llava-13b. Will this further improve the efficiency?

### Soundness
2

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
2

### Summary
This paper introduces FLASH, a novel speculative decoding framework specifically designed for Large Multimodal Models (LMMs). The authors identify that existing speculative decoding methods, primarily developed for text-only LLMs, fail to leverage the unique properties of multimodal data. FLASH addresses this gap through two key innovations: (1) a visual token compression module that reduces redundant visual tokens from N to C tokens using learnable queries, and (2) a semi-autoregressive decoding head that generates K candidate tokens in parallel rather than sequentially. The method is evaluated on video captioning (Kinetics-400) and visual instruction tuning (LLaVA-instruct-150k) tasks using LLaVA-1.5 and QwenVL-2.5 as target models, achieving up to 2.68× speedup on video captioning and 2.55× on visual instruction tuning.

### Strengths
1. **Well-motivated approach**: The paper clearly identifies limitations of existing speculative decoding methods when applied to multimodal settings and provides compelling motivation for why multimodal-specific optimizations are needed.

2. **Novel technical contributions**: The combination of visual token compression and semi-autoregressive decoding is novel and well-suited to the characteristics of multimodal data. The observation about visual object co-occurrence enabling parallel token generation is insightful.

3. **Comprehensive experimental evaluation**: The experiments cover multiple models (LLaVA, QwenVL), multiple tasks (video captioning, visual instruction tuning), and multiple model sizes (3B to 32B parameters). The inclusion of both greedy (τ=0) and sampling (τ=1) decoding settings strengthens the evaluation.

4. **Strong empirical results**: FLASH consistently outperforms baseline methods, achieving substantial speedups (up to 2.68×) while maintaining output quality. The method shows particularly strong improvements over text-only speculative decoding approaches.

5. **Thorough ablation studies**: The paper includes useful ablations on different visual compression methods, the impact of K values, and performance in text-only scenarios, providing insights into each component's contribution.

### Weaknesses
1. **Limited theoretical analysis**: The paper lacks theoretical guarantees about the preservation of output distribution after semi-autoregressive decoding. While empirical results suggest quality is maintained, formal analysis would strengthen the claims.

2. **Insufficient comparison with concurrent work**: The paper primarily compares against adaptations of LLM speculative decoding methods (Eagle-MM, Medusa-MM) rather than other multimodal-specific acceleration techniques beyond Dream and MASSV.

3. **Unclear training details**: Critical implementation details are missing, such as:
   - How the compression ratio C/N is determined
   - Training time and computational requirements
   - Convergence behavior of the semi-autoregressive head

4. **Limited analysis of failure modes**: The paper doesn't discuss scenarios where FLASH might perform poorly or when the acceptance rate drops significantly. Understanding these limitations would be valuable for practitioners.

5. **Scalability concerns**: While experiments show results up to 32B parameters, the scalability to even larger models (70B+) or longer contexts is unclear. The fixed compression ratio might become problematic for very long videos.

### Questions
1. **How sensitive is FLASH to the compression ratio C/N?** The paper mentions reducing 576 tokens to 64 for LLaVA, but how was this ratio determined? What happens with different compression ratios?

2. **How does the semi-autoregressive head handle variable-length outputs?** When describing different visual regions, the phrase lengths might vary significantly. How does the fixed K affect performance in such cases?

3. **What is the impact on hallucination?** Visual token compression might lose important details. Have the authors evaluated whether FLASH increases hallucination rates compared to the original model?

4. **Can the method extend to other modalities?** The paper focuses on vision-language models. Could FLASH be adapted for audio-visual or other multimodal combinations?

5. **Why not apply compression to text tokens as well?** The paper only compresses visual tokens. Is there a fundamental reason why text token compression wouldn't work, or was this not explored?

6. **How does FLASH perform on fine-grained visual tasks?** Tasks requiring detailed visual understanding (e.g., OCR, diagram understanding) might be sensitive to token compression. Has this been evaluated?

7. **What is the memory overhead?** While inference time is reduced, what is the additional memory requirement for storing the draft model components?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscripts proposes a novel framework for accelerating the decoding process of large multi-modal models, which uses a draft model to produce multiple next candidate tokens before they are verified by the target model. Prior works for efficient decoding in LMM paradigm often choose text-only draft models, neglecting that visual inputs contain critical information for the subsequent decoding process. To address this, the manuscript presents FLASH, generating candidate tokens using compressed visual tokens and a semi-autoregressive decoding strategy. The visual tokens are compressed due to the high redundancy in visual tokens, and multiple tokens are generated at once using the semi-autoregressive decoding strategy, instead of generating tokens one by one, to further speed up the decoding process. Overall, FLASH achieves a strong performance compared to previous speculative decoding methods.

### Strengths
1. The proposed method, FLASH, not only takes into account the importance of visual information in speculative decoding, but also reduces the redundancy in the visual representations for improved efficiency. 

2. Multi-token prediction strategy is employed for further speeding up the draft generation process. Though both of the ideas are hardly novel, the combined usage in the speculative decoding paradigm is crucial for the field. 

3. Quantitative evaluation of FLASH reveals a high acceptance rate, hence an improved speed-up ratio.

### Weaknesses
1. Though FLASH seems effective when compared to existing approaches, the component-wise analysis is not sufficient. For example, I would be curious to know, compared to no speculative decoding strategy, how does FLASH perform under different compression strategies, different number of visual tokens after compression, different number of tokens produced per-forward-pass by the draft model, etc. 

2. Since the draft model in FLASH is capable of generating multiple tokens at one single forward pass, is the metric 'average acceptance tokens' still effective for comparing the effectiveness or the performance of different decoding strategies?

3. It is not evaluated or explained how the decoding strategy affects final benchmark performance.

### Questions
1. Just for discussion, how does the size of the draft model affect the acceptance rate and the system-level speed-up rate of LMMs? (From my understanding, if the model produces draft tokens slowly but more accurately, the system-level speed-up rate would be higher, is it correct?)

### Soundness
3

### Presentation
3

### Contribution
3
