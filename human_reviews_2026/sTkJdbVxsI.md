# RMAAT: Astrocyte-Inspired Memory Compression and Replay for Efficient Long-Context Transformers

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6, 4

## Abstract
The quadratic complexity of self-attention mechanism presents a significant impediment to applying Transformer models to long sequences. This work explores computational principles derived from astrocytes—glial cells critical for biological memory and synaptic modulation—as a complementary approach to conventional architectural modifications for efficient self-attention. We introduce the Recurrent Memory Augmented Astromorphic Transformer (RMAAT), an architecture integrating abstracted astrocyte functionalities. RMAAT employs a recurrent, segment-based processing strategy where persistent memory tokens propagate contextual information. An adaptive compression mechanism, governed by a novel retention factor derived from simulated astrocyte long-term plasticity (LTP), modulates these tokens. Attention within segments utilizes an efficient, linear-complexity mechanism inspired by astrocyte short-term plasticity (STP). Training is performed using Astrocytic Memory Replay Backpropagation (AMRB), a novel algorithm designed for memory efficiency in recurrent networks. Evaluations on the Long Range Arena (LRA) benchmark demonstrate RMAAT's competitive accuracy and substantial improvements in computational and memory efficiency, indicating the potential of incorporating astrocyte-inspired dynamics into scalable sequence models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a biologically inspired Transformer architecture that models neuron–astrocyte interactions to achieve efficient long-context processing. The proposed RMAAT integrates short-term and long-term plasticity (STP/LTP) into attention, possibly allowing the model to adaptively control how much information is remembered or forgotten over time. Through Astromorphic Attention, attention weights are dynamically adjusted based on astrocyte-like activity traces, while a Memory Retention Factor compresses and consolidates past context. The proposed AMRB method further reduces training memory by replaying compressed states. Experiments on the Long Range Arena benchmark show that RMAAT achieves good accuracy with low memory.

### Strengths
1. The paper introduces a novel architecture for long-context modeling by drawing from the biological mechanisms of astrocytes, both short-term and long-term plasticity are incoprated into modern attention-based architecture. It is well-motivated and the links to previous biological models are carefully explained.

2. The AMRB training algorithm reduces the memory burden during model training. The model achieves impressive performance on long-range-arena while maintaining low memory cost.

### Weaknesses
1. While this could be subjective, I find the lengthy method section hard to read. It might be better to directly contrast the vanilla attention with the proposed astrocyte-inspired attention (as the authors did in the appendix), point out which parts correspond to long-term or short-term astrocytic mechanisms, and then explain the link to computational neuroscience models. I also find Figure 2 difficult to interpret, as there are too many interacting components in a small space. For the AMRB algorithm, the pseudo-code in the appendix is much clearer than the textual description in the main section.

2. The model is only evaluated on the Long Range Arena benchmark, which is somewhat limited. For example, including results on language modeling benchmarks (e.g., [1]) could strengthen the paper. 

3. The proposed method introduces several interacting mechanisms (Astrocyte-Modulated Hebbian Weight, Feedback Factor calculated with Presynaptic State, Memory Retention Factor), but the effects of these components are not fully disentangled in the ablation study.

[1] The fineweb datasets: Decanting the web for the finest text data at scale, 2024

### Questions
1. For the Memory Retention Factor, it seems that the factor is determined based on the total length of the context, but how is this known *a priori*? This appears somewhat strange, especially given that the model is inspired by memory retention in animals, which face a continuous stream of input and must adaptively decide how much information to retain.

2. Hebbian plasticity is described as short-term plasticity in this work, which is somewhat confusing since it is not necessarily considered short-term in a neuroscience context. Is the Hebbian plasticity effectively “cut off” between segments? If the processing time is already linear in the number of tokens, why is it still necessary to divide the context into separate segments? It would also be helpful if the authors could briefly discuss the connection to prior work on neuromodulated Hebbian learning in RNNs (e.g., [1][2]).

3. I noticed that only memory consumption is reported when comparing models. Given the substantially modified attention architecture, I suspect that the forward and backward computation times may also differ significantly from the baselines. It would therefore be helpful to include those comparisons as well.

4. As mentioned in Weakness 3, could the authors expand the ablation study to isolate the contributions of different mechanisms? It seems that only the Retrieval task was used for the experiment removing the Memory Retention Factor; if time permits, it would be valuable to include ablation results across other Long Range Arena tasks.

5. The paper claims the state-of-the-art average accuracy on Long Range Arena benchmark (e.g. line 477), but isn't modern state-space models able to achieve better performance (e.g., [3])?

[1] Backpropamine: Training self-modifying neural networks with differentiable neuromodulated plasticity  [2] Hebbian and Gradient-based Plasticity Enables Robust Memory and Rapid Learning in RNNs
[3] Simplified State Space Layers for Sequence Modeling

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes RMAAT, a recurrent long-context Transformer that is explicitly inspired by astrocyte short- and long-term plasticity. Sequences are chunked into segments processed with an astromorphic, linear-time attention block (drawing on STP) and a set of persistent memory tokens that carry state across segments. A non-learned “memory retention factor,” derived from simulations of an LTP-like macro model, adaptively compresses older memory as the number of segments grows. Training uses Astrocytic Memory Replay Backpropagation (AMRB): instead of storing all activations, the model buffers only the sequence of memory tokens and recomputes each segment during backward pass. On Long Range Arena (LRA), the method reports higher average accuracy than a variety of efficient-attention baselines while using substantially less peak GPU memory than iso-recurrent baselines; ablations suggest the retention factor is important for retrieval accuracy and that AMRB yields large memory savings.

### Strengths
This is a good paper. It is coherent, clear and well-motivated. The biological motivation is nice, and the link to existing literature (in particular the model of Kozachkov, 25) is solid. The division between within-segment linear attention and cross-segment persistent memory tokens is intuitively presented and illustrated clearly. The “retention factor” derived from a long-term potentiation curve is an elegant way to manage bounded memory in long sequences.

 The introduction of Astrocytic Memory Replay Backpropagation is also a practical advance: it provides a viable method to train long-context recurrent Transformers under constrained memory without excessive recomputation complexity. The empirical evaluation, while limited in scope, is well-structured within the Long Range Arena framework, and the ablations on the retention factor and memory replay convincingly show their contributions.

### Weaknesses
The experiments are limited to LRA, which is understandable given its focus on long-range dependencies, but it would have strengthened the paper to demonstrate how this architecture performs across other domains—such as vision or multimodal tasks—where long-context modeling may manifest differently.

### Questions
1. How well does the proposed retention mechanism generalize to settings where the total sequence length is not known in advance, such as streaming or online processing?

2. (Related to the weakness mentioned above) Have the authors explored applying the RMAAT architecture to vision or multimodal data to assess whether the astrocyte-inspired memory dynamics transfer effectively across domains?

### Soundness
3

### Presentation
4

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
Transformer, especially with attention mechanism, reqjuires quadratic computational complexity which hinders long sequence applications. This paper aims to overcome this limitation by combining astrocyte-inspired mechanisms on transformer architecture, called RMAAT. They also design AMRB alorithm for the update. With proposed method, this paper acheives both state of the art computational efficiency and performance in long range arena tasks.

### Strengths
- Biological inspiration from astrocytes combining short-term plasticity and long-term plasticity is interesting.
- Introducing retention factor for adaptive compression mechanism seems to be novel.
- Result supports their method is effective in long-range sequence application.

### Weaknesses
**Weakness 1: Insufficient Biological Justification for Core Astrocytic Mechanisms**

While I appreciate the ambitious goal of integrating neuroscience-inspired principles to solve the $\mathcal{O}(N^2)$ problem, I find the **justification for the design of the core astrocytic equations to be insufficient and potentially contradictory** to established micro-scale neuroscience findings. For a model with a vast parameter space, the rigor of biological justification is paramount to ensure the results stem from **methodological insight** rather than an exhaustive hyperparameter search.

For instance, the dynamic equation for the Short-Term Astrocytic Process Parameter $p_{ij}^s$ (Eq. 2/S4) includes a problematic spatial integration term:

$$ \sum_{k,l=1}^N T_{ijkl} \psi(p_{kl}^s) $$

This term suggests that the astrocytic process at synapse $(i, j)$ is influenced by the **aggregated activity of all other peripheral astrocytic compartments** $p_{kl}^s$ across the entire segment via the distance-dependent coupling tensor $T_{ijkl}$.

*   **Conflict with Microdynamics:** I note that $\text{Ca}^{2+}$ dynamics in astrocytic peripheral processes—the presumed biological correlate of $p^s$ (Line 650)—are often characterized as **highly localized "calcium spots"** (e.g., [1] and related literature), indicating that individual synaptic sites may **not necessarily integrate global** peripheral $\text{Ca}^{2+}$ signals from distant synapses within the same territory.
*   **Missing Rationale:** The STP section (Section 3.1) lacks a specific biological or computational neuroscience reference that **explicitly supports the necessity of this broad spatial summation** over $N^2$ synapses. I require a clear articulation of **what specific biological or functional phenomenon** this massive spatial summation is designed to abstract. If it is a macro-scale abstraction of contextual influence, this must be clearly defined and referenced with theoretical literature that supports this abstraction.

**Weakness 2: Lack of Transparency in Core Technical Contribution**

I find the explanation of the **Memory Retention Factor**—one of the paper's key contributions (Contribution 2)—to be **insufficiently transparent and mathematically vague**. This factor, which is derived from the LTP macro model and is critical for adaptive compression (Line 346), is only described conceptually.

*   The exact **mathematical function** of $\text{RetentionFactor}(t, \text{TotalSegments})$ is **not provided** in the main text or the relevant Section 3.3, but is relegated to Appendix D.
*   Given that this factor is claimed to be a **principled, non-learned compression method**, I consider the full mathematical formula to be an **essential part of the core contribution** that must be clearly presented in the main body. The current presentation hinders the model's reproducibility and the community's ability to scrutinize its theoretical grounding.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new transformer model inspired by the role of astrocytes in biological neural function. It introduces a novel attention mechanism based on short-term and long-term synaptic plasticity dynamics that reduces the computational complexity of the standard attention mechanism. The model incorporates memory tokens that feed outputs from previous segments of the sequence into subsequent segments, whose influence is modulated by a retention factor calculated from LTP dynamics simulations. Finally, the model harnesses this faster forward-pass computation to implement backpropagation efficiently by recomputing the forward pass from the memory tokens rather than storing the activations. The results of computational neuroscience simulations inspire the values of spatial interaction and temporal compression parameters. With these features, the model achieves state-of-the-art performance on various Long Range Arena benchmark tasks.

### Strengths
1. The architectural innovations, including the efficient attention mechanism, the use of memory tokens with compression, and the efficient training procedure, are valuable. 

2. The benchmark performance results are impressive.

### Weaknesses
1. The explanation of the model’s features could benefit from further refinement for clarity and concision. Figures 1 and 2 should be considerably polished and possibly enlarged. 

2. It is not clear how the neuron-astrocyte simulations specifically inform the network architecture beyond guiding priniciples. Further, what happens if one deviates from the values suggested by the simulations? This result might be informative in addition to the ablation analysis.

### Questions
For the computational macro model (contribution 1), how does the 9-neuron network specifically inform RMAAT’s persistent memory mechanism?

Further, how does the neuron-astrocyte network specifically inform the architecture, such as the hyperparameters, for the Memory Retention Factor (contribution 2)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes RMAAT, a recurrent Transformer architecture that integrates abstracted computational principles from astrocyte-mediated synaptic plasticity. The core ideas include: an $O(N)$ astromorphic attention mechanism inspired by short-term plasticity (STP) dynamics; a Memory Retention Factor derived from simulated long-term plasticity (LTP) dynamics; and a memory-efficient training algorithm that stores only the compact memory tokens across segments and recomputes activations during backpropagation. 

This paper is built upon existing works (including Kozachkov et al., 2023; Mia et al., 2025; Kozachkov et al., 2025). From my own opinion, the core innovation is the memory retention factor derived from simulated long-term plasticity dynamics. This is a good but not groundbreaking paper.  See below.

### Strengths
- The neuro-glial inspiration is a fresh angle in the crowded field of efficient transformers.
- The empirical memory reduction is significant and convincingly demonstrated.

### Weaknesses
- The presentation and writing is not very clear. For one example, how $mem'_{t+1}$ is obtained? 
- Without directly comparing it with current mainstream long sequence models (such as Mamba and RetNet), and only comparing it with traditional Transformer and recursive models, it is impossible to fully demonstrate RMAAT's industry competitiveness in the "efficiency-accuracy" trade-off. 
- I still do not know what's the benefit of Astrocyte-inspired model compared with existing methods, such as SSMs, and Linear RNN-Transformers. The stated efficiency-accuracy balance can also be achieved through other solutions. Why not compare? 
- Overstated contributions. Maybe contribution 1 and 2 should be merged into one? ``a Memory Retention Factor from this principle by analyzing the accumulation rate within the simulated LTP curve``. What's the difference between RNN gradient checkpointing and contribution 3? 
- Outdated Baselines: Many baseline results (Sparse Trans., Longformer, etc.) are referenced from the original 2020 LRA paper, which used suboptimal hyperparameters. 
- Weak Ablation Baseline: RLT (Recurrent Linear Transformer) is a weak baseline—it's essentially RMAAT without the retention factor, AMRB, or enhanced positional encoding. A stronger ablation would be RMT + AMRB (to isolate the attention mechanism's impact) or RMAAT without retention factor but with AMRB (to isolate compression's effect).
- Model Size: The largest model has embedding dim 1024, far smaller than modern LLMs. The memory savings may diminish at scale where activation memory is dwarfed by parameter memory (e.g., 70B models). The paper should discuss extrapolation to larger scales.
- The connection between AMRB and biological memory replay is tenuous and largely metaphorical, AMRB is structurally identical to gradient checkpointing.

### Questions
See weakness.

### Soundness
2

### Presentation
1

### Contribution
2
