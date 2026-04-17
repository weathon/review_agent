# Artificial Hippocampus Networks for Efficient Long-Context Modeling

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Long-sequence modeling faces a fundamental trade-off between the efficiency of compressive fixed-size memory in RNN-like models and the fidelity of lossless growing memory in attention-based Transformers. Inspired by the Multi-Store Model in cognitive science, we introduce a memory framework of artificial neural networks. Our method maintains a sliding window of the Transformer's KV cache as lossless short-term memory, while a learnable module termed Artificial Hippocampus Network (AHN) recurrently compresses out-of-window information into a fixed-size compact long-term memory. To validate this framework, we instantiate AHNs using modern RNN-like architectures, including Mamba2, DeltaNet, and GatedDeltaNet to augment open-weight base LLMs. We also propose an efficient self-distillation method where the base model' all parameters are frozen and only the parameters from AHNs are optimized. For inference, our method sets a default large sliding window size of 32k for attention, and AHNs activate only when the sequence length exceeds the 32k window, addressing the quadratic-complexity issue of attention that emerges at that scale. Extensive experiments on long-context benchmarks LV-Eval and InfiniteBench demonstrate that AHN-augmented models consistently outperform sliding window baselines and achieve performance comparable or even superior to full-attention models, while substantially reducing computational and memory requirements. For instance, augmenting the Qwen2.5-3B-Instruct with AHNs reduces inference FLOPs by 40.5% and memory cache by 74.0%, while improving its average score on LV-Eval (128k sequence length) from 4.41 to 5.88. Code and models will be released to facilitate future research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents Artificial Hippocampus Networks, replacing the attention mechanism with a combination of sliding window attention and a linear RNN (Mamba2 or (Gated) Delta-Net), via distillation from a pretrained model. Experiments show that capabilities can be approximately recovered compared to the full attention original, with strong speed ups and memory reduction (constant memory use), especially on long context benchmarks.

### Strengths
The work puts a nice focus on long-context capabilities as a typical weak point of linear RNNs.
It even out-performs Full Attention base model on Long Context Tasks in some cases.
The proposed randomized sliding window size results in a slight performance improvement.

### Weaknesses
The work does not compare to very similar existing methods (LoLCaTs) [1], resulting in a lack of novelty.
The work is missing a comparison to a pure RNN, removing sliding window attention overall. If this performed similarly, also the neuro-biological motivation should be questioned.

[1] Zhang et al. (2024): https://arxiv.org/pdf/2410.10254

[2] Wang et al. (2024): https://arxiv.org/pdf/2408.15237

### Questions
Could you cite and explain the difference to [1] / [2] as a very similar related work?
If the sliding window attention was removed completely, how would this simpler replacement perform?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Artificial Hippocampus Networks (AHN), a method that can be applied to pretrained models in order to enhance their long-context performance while reducing their memory requirements.
AHN consist of a lossless memory in form of a sliding window attention layer and a compressed memory in form of a (linear) RNN, like for e.g. Mamba-2 or Gated Delta Net.
The authors propose to augment pretrained models with AHN by training the models with a KL-Divergence objective.
In their experiments AHN augmented Transformer models show improvements on numerous long context tasks.

### Strengths
- AHN augmented Transformer models provide efficiency gains 
- The ablation on different training objectives is important to support the choice of solely train with the KL-Loss
- The illustrative example is a very good motivation and a good comparison to the unchanged base model

### Weaknesses
- Even though the relation to the human brain is a nice motivation, at the end of the introduction the reader is left with the question: what is the difference to a hybrid model between a RNN and attention?
I suggest that the authors add information about the key descriptive components of AHN and describe their method in more detail already in Fig1/Abstract/Intro? (e.g. Hybrid SWA + Linear attention, KL loss / distillation, application to pretrained models). As it is now it just seems too broad.

- L. 401, 412: The author chose a very large lossless sliding window. Since the results in Table 2 suggest that the AHN variants outperform full attention, it would be interesting to investigate the impact of the sliding window size on these benchmarks, too (not only on LongBench as in Fig.4)

- The authors could provide a more comprehensive eval comparing the pretrained basemodel with the AHN augmented variants on other standard (not necessarily long context) pretraining tasks, such as MMLU(-PRO) or GSM8k or GPQA.

- It seems that there are some important implementation details missing: From the “extra param ratio” in Table 2 one could assume that the extra parameters are learnable gates etc. not present in the pretrained transformer. Is this true? I could not find this information prominently written in the paper. So my questions remain:
  - So the QKV projections for Sliding window attention and Mamba2/GDN are shared?
  - Could the authors specify what are the “unfrozen AHN parameters” (Fig. 2)

- AHN seem not novel: Distillation with KL divergence of pretrained models has been explored, hybrid models between sliding window and linear attention also have been explored before (See for e.g. https://arxiv.org/abs/2506.00744, https://arxiv.org/abs/2410.10254 , https://arxiv.org/abs/2408.15237). 
  - Could the authors emphasize the new insights stemming from the suggested AHNs?

In my view the aforementioned weaknesses outweigh the strengths. Therefore I am inclined to recommend rejection of the paper.

### Questions
- Figure 1 is placed before the abstract. I recommend double checking whether this is in line with the style guides for ICLR.
- I would suggest to move the related work to the end, since the introduction already covers a lot of related work. It seems a bit repetitive in this order.
- L. 353: What sliding window sizes were used? What is meant by randomizing the starting position?
- Section 4.5: From which part in the sequence is the snippet taken? Even though the motivation to try to understand how AHNs compress information, the empirical data provided by Figure 5 is not convincing and at least needs further clarifications (see question above).

- Finally, I suggest incorporating a comparison to this work ( https://arxiv.org/abs/2506.00744 ) which seems very similar.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Artificial Hippocampus Networks (AHNs), a novel and efficient framework for long context modeling in Transformers. The work addresses the fundamental trade off between the high fidelity, growing memory of attention's KV cache and the efficient, fixed size memory of Recurrent Neural Network like models. Inspired by the Multi Store Model of memory from cognitive science, the proposed method uses a hybrid approach. It maintains a sliding window of the KV cache as a lossless short term memory, while the AHN module, a learnable component, recurrently compresses KV pairs that exit the window into a fixed size long term memory state. This AHN module can be implemented using various modern RNN like architectures, such as Mamba2 or GatedDeltaNet. A key contribution is the efficient training method. Instead of training from scratch, the authors use a parameter efficient self distillation framework where the AHN parameters are trained to match the output distributions of a frozen, full attention "teacher" model. Experiments, particularly on the Qwen2.5 3B model, show that this method dramatically reduces computational and memory costs (e.g., 40.5% fewer FLOPs and a 74.0% smaller memory cache on LV Eval) while simultaneously and surprisingly improving average scores on long context benchmarks.

### Strengths
1. Novel Hybrid Framework: The paper introduces a novel hybrid memory framework, drawing inspiration from cognitive science to blend lossless, sliding window attention with a fixed size compressive memory. This two part memory system is a creative approach to the long context problem. The concept is presented clearly, with diagrams that effectively illustrate the mechanism.

2. Efficiency and Training Method: The primary practical strength is the method's efficiency. By design, it achieves constant memory cache size and linear computational complexity, which is a significant improvement over standard attention. The paper demonstrates this leads to substantial reductions in FLOPs (40.5%) and memory cache (74.0%) on a 3B parameter model. The self distillation training approach is also a notable contribution, as it allows these gains to be achieved by efficiently adapting existing pretrained models.

### Weaknesses
1. Lossy Compression and Exact Recall: The most significant weakness is the model's performance on exact recall tasks. The RULER benchmark's needle in a haystack (NIAH) tasks (Table 5) show that the AHN augmented model fails at retrieving specific facts from deep context, performing just as poorly as a simple sliding window and far worse than full attention. This is a critical trade off. The main paper's strong results on LV Eval and LongBench may be because these benchmarks test for general understanding rather than high fidelity recall. This limitation makes the method unsuitable for tasks that depend on precise, long range fact retrieval.

2. Unclear Comparison to Full Attention: The paper presents the very surprising result that the AHN model outperforms the full attention baseline (e.g., 5.88 vs 4.41 on LV Eval). This is counterintuitive, as the full attention model has access to strictly more information. The paper does not adequately explain this. Figure 3c shows the baseline model's perplexity spiking after 32K tokens, which suggests the baseline Qwen2.5 3B model was not pretrained for such long contexts. If this is the case, the comparison is not entirely fair. The performance gain might not come from the superiority of the AHN architecture, but rather from the self distillation process itself acting as an effective form of context length extension training, which the baseline did not receive.

3. Focus on Post Training Adaptation: The paper exclusively validates AHNs as a module added after pretraining, using a parameter efficient, frozen backbone approach. This leaves open the question of how an architecture with AHNs would perform if pretrained from scratch. The current approach's performance is inherently capped by the teacher model's capabilities. Demonstrating that this memory framework is stable and effective during full pretraining would make a much stronger case for it as a new architectural paradigm, rather than just an efficient adaptation technique.

### Questions
1. Regarding the Full Attention Comparison: Could you please clarify the extraordinary result where the AHN model (5.88) significantly outperforms the full attention baseline (4.41) on LV Eval? As noted in my weaknesses, Figure 3c suggests the baseline model is not functional beyond its 32K context window. Is the baseline model simply failing due to a lack of long context pretraining? And is it possible that the self distillation process itself is the primary source of the performance gain, by effectively fine tuning the model for longer contexts, rather than the AHN's compressive memory?

2. Reconciling Compression and Recall Failure: The gradient visualization in Figure 5 suggests the AHN learns to selectively compress information, preserving important symbols and numbers. However, the appendix (Table 5) shows a total failure on needle in a haystack tasks. How do you reconcile these two findings? 

3. Training Practicality: The paper emphasizes the efficiency of the self distillation training. Could you provide more concrete details on the training cost? For example, how many GPU hours or what fraction of the original pretraining cost does this self distillation phase require? Furthermore, do you have any preliminary results or hypotheses on the feasibility and stability of training a model with AHNs from scratch?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a framework, named artificial hippocampus network (AHN), which integrates sliding window attention with recurrent memory. Concretely, sliding window attention with fixed attention context handles short-term memory, and recurrent memory, implemented with linear recurrent networks, compresses and maintains long-term contextual information. Rather than training from scratch with the standard cross entropy loss, the authors leverages self-distillation by minimizing the KL-divergence between a teacher model and the memory-augmented model. 

Experiments were conducted on top of QWen-2.5-Instruct models, with ChatQA2 as training data. Evaluation were performed on long-context benchmarks, including LV-Evan, InfiniteBench and Ruler.

### Strengths
The AHN framework is well-motivated to integrate SWA and recurrent-memory.

### Weaknesses
There are several weaknesses of this paper:

1. The proposed AHN framework is very similar to previous recurrent-memory studies, such as infiniti-attention [1]. The major difference is that infiniti-atention is using chunk-wise attention and update the recurrent memory in a chunk-wise way. However, this paper does not make any comparison to these related work.

2. There are no reported numbers on efficiency of model training after adding the recurrent memory module, comparing with purely sliding window attention and/or full attention. 

3. The improvements of the proposed models upon pure SWA baseline are not impressiveon LongBench tasks. On Ruler, the scores of needle-in-haystack tasks are still near random baselines. 


References:

1. Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention

### Questions
Questions:

1. Does the recurrent memory shares the same set of $W_Q, W_K, W_V$ with SWA? If yes, the only additional parameters in the recurrent memory module are from the backend linear recurrent network, such as Mamba, DeltaNet or GatedDeltaNet?

2. What is the sequence length and the attention context window during training?

### Soundness
3

### Presentation
2

### Contribution
2
