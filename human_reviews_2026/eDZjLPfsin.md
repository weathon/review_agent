# Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
State Space Models (SSMs) have emerged as powerful alternatives to attention-based Transformers, with Mamba demonstrating impressive efficiency and scalability. As these models grow increasingly larger, the need for Parameter-Efficient Fine-Tuning (PEFT) methods becomes critical to adapt pre-trained Mamba to downstream tasks without prohibitive computational costs. However, previous approaches simply apply traditional Transformer-tailored PEFT methods without addressing the unique temporal processing dynamics of SSMs. To address this limitation, we propose ***Memba***, a membrane-driven PEFT approach specifically designed for Mamba. ***Memba*** introduces Leaky Integrate Membrane (LIM) neurons as bio-inspired gating mechanisms that naturally accumulate membrane potentials over time, enhancing selective information retention. By strategically combining LIM neurons with Low-Rank Adaptations (LoRA) and cross-layer membrane transfer, our approach significantly improves Mamba's temporal modeling capabilities. Extensive experiments across language and vision tasks demonstrate that ***Memba*** achieves substantial improvements over existing PEFT methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Memba, which combines LIM with LoRA and cross-layer membrane transfer, aiming to improve parameter-efficient adaptation. The proposed Memba demonstrates superior performance over existing PEFT methods on language (commonsense reasoning) and vision (VTAB-1k) tasks, emphasizing parameter efficiency and temporal modeling capabilities.

### Strengths
The membrane-driven PEFT method is conceptually novel, as it creatively integrates LIF modules into SSM fine-tuning, addressing a gap in temporal adaptation for Mamba models. 

The chunked processing and cross-layer transfer efficiently handle long sequences without adding learnable parameters.

### Weaknesses
This work is actually a generic PEFT method, not exclusive to Mamba, as it involves adding LIM to LoRA's hidden layers. Therefore, I would like to see experimental results on Transformers.

I find the absence of comparisons with advanced PEFT methods for Transformers (e.g., recent adapters or prompt-based techniques) or other SSM variants (e.g., Mamba-Adapter) reduces the claim of general superiority.

While Memba improves performance, the 8.8% inference overhead (Table 11) is non-trivial for real-time applications, and the paper underdiscusses trade-offs, such as energy efficiency or scalability on edge devices. Notably, spike networks (SNNs) are primarily beneficial for their low energy consumption, but the motivation for using LIM in this work does not seem to be energy efficiency. Therefore, I am puzzled as to why the authors chose to use LIM instead of other RNN-style modules, which would seem to be a more straightforward choice.

The LIM neuron is not entirely similar to the LIF used in SNNs, as it discards binary outputs. In this regard, this module actually shows similarities to traditional RNN gates, but the ablation (Table 10) only briefly compares them. A more rigorous analysis of novelty versus established gating mechanisms is needed.



The vision tasks using VTAB-1k fail to demonstrate the performance superiority, as previous work has shown that different PEFT methods perform comparably on this dataset [1]. Therefore, semantic segmentation and object detection tasks are imperative.

GPU memory and latency comparisons among different methods are missing.    

Refs:    
[1] Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning (PEFT) in Visual Recognition. CVPR 2025.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Memba, a new parameter-efficient fine-tuning (PEFT) method for Mamba. It introduces a PEFT module inspired by Leaky Integrate Membrane (LIM) neurons and adds cross-layer membrane transfer. The combination of LIM and LoRA outperforms existing PEFT methods for Mamba in both vision and language domains.

### Strengths
- The use of LIM neurons for temporal gating of the output of SSM is innovative and outperforms existing methods. The motivation of the cross-layer membrane potential transfer is also reasonable.
- Extensive experiments on both language (commonsense reasoning) and vision (VTAB-1k) tasks prove that Memba outperforms existing PEFT methods, including LoRA and other recent SSM-specific PEFTs.

### Weaknesses
#### Major Weakness
- From Table 7, it seems the computational cost in terms of speed is more than the increase in parameter counts. Hence, the speed of Memba seems slower than other PEFT methods. How about memory consumption? In my opinion, small memory consumption is more important than inference speed, since it enables finetuning of large models with limited GPUs.  
- I understand that LIM processes data recurrently in fixed-size chunks. Since the number of image tokens is predetermined, that part seems straightforward. However, how is language handled? In particular, during inference, which I assume is performed in an auto-regressive manner, does the number of tokens within a chunk change as each new word is generated? If so, does that mean caching can't be used, and all tokens must be reprocessed at every step?


#### Minor Weakness
- In Fig. 1, it is difficult to understand what the authors want to tell with the two saliency maps.  
- Typo:  
    - L028) INTODUCTION -> INTRODUCTION  
    - L214) LIM mechanis -> LIM mechanism  
    - L485) seperate -> separate  
    - L742 & L744) Equation equation

### Questions
- Please read the major weakness.  
- In Fig. 4, why is the downward trend in membrane potential good?
- Although the authors say the speed of Memba can be imporved by CUDA implimentation, is it really posible? if the number of chunk is small, I think parallel scan like Mamba can not improve the latency.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Memba, a parameter-efficient fine-tuning (PEFT) strategy desginated for Mamba, which is under well expolored. Memba incorporates Leaky Integrate Membrane (LIM) gating value adapters in SSM. LIM mainly features the chunked LIM neurons desgin and layerwise membrane potential propagation. The authors present empirical results across language and vision tasks, and compelling ablation and analysis experiments to demonstrate the effectiveness of the LIM adapters.

### Strengths
1. PEFT for Mamba is a quite novel topic and the temporal-wise gating LIM seems to be a good design for finetuning the gate values in SSM.

2. The experiments are wide-ranging—spanning both language and vision tasks. The analysis results are also abundant and convincing. In addition to experiments, the theoretical analysis of loss boundaries is also interesting, enhancing the insight of the design, with the lower loss.

3. The proposed LIM is efficient, with minimal additional parameters compared to LoRA adapters, enabling its wide potential of applications on various SSM models.

### Weaknesses
1. How is the proposed by-pass of SSM in low rank adapters training related to the biological term, Membrane? The paper does not include any references or tutorials about the membrane mechanism or other background knowledge about it, therefore it is somehow confusing why the proposed mechanism resembles a membrane.

2. The average membrane potentional values in the figures show steps of decrease across the temporal chunks, and the authors attribute this phenomenon into a "forget" manner of LIM. However, as some earliear papers have pointed out, the temporal modeling behavior of SSM should be like a local filter to keep the most recent tokens and forget the much earlier ones. Figure 4. and Figure 8. demonstrate the replicated pattern of spikes across chunk 1 to chunk 4, indicating that the "forget" effect is rather weak. Even though other chunks do not have a same activated area like chunk 1, the LIM still outputs a high response gate value to emphasize this area, which is not fully reasonable. 

3. Although the novelty of this paper is pretty decent, Mamba-like models still fall behind the conventional attention based Transformers or linear attention architectures, especially large scale models. I suspect a little about the contribution of the proposed method on larger models and more challenging tasks.

### Questions
1. From Line 198-206, there are duplicate description of the three components, which could be shortened or simpified.

2. In the area of vision tasks, can LIM and the original SSM part take different scan orders of the same image? Does this setting help or impair the models?

3. The output values of LIM in the figures are negative, are these values collected before or after SiLU gate?

### Soundness
3

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
This manuscript proposes Memba, a membrane-driven Parameter-Efficient Fine-Tuning (PEFT) approach tailored for Mamba models, addressing the limitation that traditional Transformer-focused PEFT methods fail to account for Mamba’s unique temporal processing dynamics. Memba integrates Leaky Integrate Membrane neurons with strategic Low-Rank Adaptations (LoRA) on input/output projections and cross-layer membrane transfer, enhancing selective information retention without modifying Mamba’s core state-space components. Extensive experiments on commonsense reasoning and VTAB-1k demonstrate that Memba outperforms existing PEFT methods  across various model sizes while maintaining parameter efficiency, with theoretical analysis confirming LIM’s role in stable temporal integration and adaptive regularization.

### Strengths
1. The LIM neuron brings biologically inspired temporal processing into Mamba’s gating, filling the gap left by its simpler gates and helping the model learn what to remember or forget over time.
2. The paper backs this up with clear analysis (like loss decomposition and bounded regularization), explaining how LIM adapts over time and smooths the loss landscape.
3. The writing and figure are both well illustrated.

### Weaknesses
1. The performance depends on tuning things like the number of chunks (T), the leak factor (τ), and the threshold (Vth), so LIM likely need task-specific tuning.
2. The theoretical analysis shows that the LIM neuron acts as a regularizer to smooth the loss landscape. However, the paper fails to explain the causal link between this optimization-level effect and the claimed practical benefits of enhanced temporal modeling and selective attention.
3. The evaluation's focus on classification-style benchmarks, instead of long-sequence generative tasks, fails to fully validate the method's claimed improvements to Mamba's core temporal modeling capabilities.

### Questions
Will this method or idea be applied to other linear models, like MoR or Gated networks?

### Soundness
3

### Presentation
3

### Contribution
3
