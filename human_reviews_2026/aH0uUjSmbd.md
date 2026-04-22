# Recurrent Off-Policy Deep Reinforcement Learning Doesn’t Have to be Slow

- Avg Score: 4.80
- Decision: Reject
- Scores: 6, 6, 2, 4, 6

## Abstract
Recurrent off-policy deep reinforcement learning models achieve state-of-the-art performance but are often sidelined due to their high computational demands. In response, we introduce RISE (Recurrent Integration via Simplified Encodings), a novel approach that can leverage recurrent networks in any image-based off-policy RL setting without significant computational overheads via using both learnable and non-learnable encoder layers. When integrating RISE into leading non-recurrent off-policy RL algorithms, we observe a 35.6\% human-normalized interquartile mean (IQM) performance improvement across the Atari benchmark. We analyze various implementation strategies to highlight the versatility and potential of our proposed framework. We make our code available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces RISE (Recurrent Integration via Simplified Encodings) a framework for integrating recurrent neural networks efficiently into off-policy reinforcement learning without incurring the heavy computational cost typical of recurrent methods. RISE separates the encoding of recent and long-term observations by combining learnable encoders for immediate inputs and non-learnable (pretrained) encoders for historical context. This enables recurrent models to leverage temporal dependencies without redundant convolutional passes. When applied to strong baselines like Beyond The Rainbow (BTR), RISE achieves up to 35.6% improvement in human-normalized interquartile mean performance on the Atari benchmark while drastically reducing walltime. The paper validates RISE across Atari, Procgen, VizDoom, and Miniworld.

### Strengths
This paper presents a original contribution by removing one of the key computational bottlenecks in recurrent off-policy reinforcement learning through its RISE framework, which cleverly combines learnable and non-learnable encoders to enable efficient use of recurrence. The quality of the work is strong, supported by rigorous experimentation across multiple benchmarks and careful ablation studies that validate the design choices. The paper is clearly written and well-structured, effectively communicating both the motivation and the technical details of the approach. Its significance is substantial, as RISE democratizes access to high-performing recurrent off-policy RL methods by drastically reducing computational demands, potentially broadening their adoption and impact in both academic and applied research contexts.

### Weaknesses
While the paper makes a strong contribution, several areas could be improved. First, the analysis of temporal credit assignment and how RISE’s separation of encoders affects long-horizon dependencies is limited; more diagnostic experiments (e.g., on tasks requiring extended memory like DMLab or Meta-World) would clarify this. Second, the ablation studies could go deeper by isolating the effects of encoder pretraining and architectural design choices. Finally, the theoretical motivation for the hybrid encoder structure remains mostly empirical—adding a formal analysis or simplified toy model could further solidify the conceptual contribution.

### Questions
1.Could the authors elaborate on the specific mechanism of integration between the learnable and non-learnable encoders? For instance, how is the representation from the non-learnable encoder normalized or aligned with that from the learnable one before fusion?

2.The method’s empirical success is clear, but the theoretical reasoning for why the hybrid encoder structure preserves recurrent credit assignment efficiency is underdeveloped.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work proposes a novel method, RISE, to make recurrent models computationally efficient in the off-policy RL domain. In short, it proposes to significantly save on computational resources by combining a standard (trained) encoder to process the current state input (image) with a fixed non-learned encoder to process pre-computable embeddings for each state in the preceding part of the sequence. These (cheaper) embeddings are stored in the replay buffer and fed to the recurrent model during off-policy training. This approach provides the full benefits of long-term context without the expensive re-computation which is often induced by processing every past frame of a trajectory, typically done with an expensive CNN encoder. Ultimately this approach results in large wall-clock time speedups and near state-of-the-art absolute performance in a range of RL tasks.

Note that this specific area is not my expertise, hence my low confidence score for this review, though I read the paper thoroughly nonetheless.

### Strengths
- The proposed solution for reducing computational cost in this work is (as far as I know) original and has a significant impact in reducing computational cost and in broadening the computational accessibility of RL modelling 
- Experimental validation are carried out across a range and diverse set of tasks and with comparisons against state of the art baselines - a significant undertaking
- Ablation studies and studies of what affects the performance of the fixed embedding are well carried out
- The use of such a setup is compatible with a range of RL learning methods and is therefore of general interest and potential impact

### Weaknesses
- Though training time is reduced by use of this method, the peak memory usage required for this method is likely to be higher than otherwise when taking into consideration the model used to produce embeddings
- The limitations of this method are likely heavily linked to any limitations in the non-learnable encoder0 This likely means that for tasks which are not visual in nature, or have states which are OOD for the (fixed) embedding network, the method would likely fail. This may also be the reason why this method fails for some tasks over others
- Theoretically, the contributions of this work are rather limited and weak. The RL tasks, or RL process in general is not better understood as a result of this paper and it remains somewhat unclear how much the benefits seen are due to the embedding-producing network

### Questions
- The limitations of this work are relatively little discussed in the space of what types of tasks (in vs out of distribution for the non-learnable encoder) this approach would work for? How could one make smart choices about which encoders to use for embeddings?

I don't have many questions as this work seems rather comprehensive in nature. Note that my familiarity with such RL work is rather minimal. Overall, this seems like a very promising addition to RL methods.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper targets an important but under-served practical problem in deep RL: recurrent off-policy agents are known to help in partially observable visual tasks, but they are often too expensive to train because every replayed sequence must be re-encoded frame-by-frame by a heavy vision backbone. The authors propose a simple two-stream design. For the long history/context part of the sequence, they precompute image features with a fixed encoder at data-collection time and store those features directly in the replay buffer. At training time, the recurrent module (LSTM) consumes only these precomputed features, so no CNN forward pass is needed for the historical frames. For the current timestep, they still use a normal, learnable vision encoder, and they fuse its output with the LSTM output to produce Q-values (or the usual off-policy head). This reduces the number of vision forward passes during training to something close to non-recurrent agents, while still allowing the agent to benefit from long temporal context. Empirically, plugging this into strong image-based off-policy baselines  improves human-normalized scores on Atari and other visual benchmarks under limited compute.

### Strengths
1. Well-motivated practicality. The paper correctly identifies a real bottleneck in recurrent off-policy RL: sequence-based replay plus heavy visual backbones is costly, so many practitioners simply do not turn on recurrence even when tasks are partially observable. The paper gives a plausible solution to this exact bottleneck.
2. Simple, reproducible mechanism. The main trick—precompute visual features for history and store them in replay—is conceptually simple and easy to reimplement in existing off-policy codebases.
3. Good empirical story. The experiments show that (i) you can retain the benefits of long context (up to ~160 steps) and (ii) you do not have to pay the usual multiplicative cost in CNN passes. Results on Atari and other visual / partially observable tasks support the claim that “recurrent off-policy is actually usable” under their design.
4. Compatibility with strong baselines. The method is evaluated by adding it to competitive off-policy agents (not by introducing a weak agent and claiming gains). This makes the numbers more trustworthy.
5. Ablations are informative. The paper does some ablation on fixed-encoder choice and context length, which helps clarify when the method helps more (tasks with partial observability, longer-horizon credit, more visual complexity).

### Weaknesses
1. Incremental from an RL perspective. The core contribution is an architectural / systems rearrangement—moving expensive vision to data-collection time and having the LSTM consume precomputed features—not a new principle for off-policy RL. The underlying learning rule, off-policy correction, replay usage, and handling of partial observability all stay standard. This makes the contribution feel more like “making an existing recipe cheaper” than “a new RL idea.”
2. No deeper treatment of recurrent off-policy issues. Recurrent off-policy is known to have subtleties (e.g. mismatch between behavior-policy hidden state and learner hidden state, burn-in choices, bias due to truncated sequences). The paper does not propose new corrections or analyses for these; it just makes the pipeline faster. This limits conceptual novelty.
3. Reliance on fixed visual features is under-analyzed. Because the LSTM consumes non-learnable features for most of the context, the representation seen by the recurrent module is partially decoupled from the task loss. The paper shows it works empirically, but it does not dig into when a fixed encoder might become a bottleneck (e.g. domains very different from the pretraining source, tasks that need fine-grained spatial cues across time).
4. Performance improvements could be framed as “better compute–data trade.” Many of the reported gains can be interpreted as: “we could finally afford longer context and bigger batches because it was cheap enough,” rather than “this specific architecture is inherently better.”

### Questions
1. How sensitive is the method to the exact fixed encoder? If we swap ResNet-18 pretrained on ImageNet with a much lighter conv stack (e.g. 4-layer Atari-style conv), do we still see the same benefits on Atari, or does the LSTM stop being useful because the features are too weak?
2. What is the storage overhead for keeping precomputed features for long contexts in the replay buffer? For very long contexts or higher-res observations, does this become a memory bottleneck?
3. Can the authors report results where both the historical encoder and the current encoder are learnable but updated at different frequencies (e.g. slow EMA for the history encoder) so we can see the trade-off between flexibility and cost?
4. For tasks with non-visual partial observability (e.g. missing proprioception, delayed rewards), would the method still help, or is it mostly an image-RL trick?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates how to improve the computational efficiency of recurrent off-policy image-based deep reinforcement learning methods by replacing the image encoder learned from scratch by using pretrained vision encoder like ResNet when encoding the historical images for Q-value computation. Experimental results show that their method outperform non-recurrent methods, achieve similar performance as existing recurrent methods while being more computationally efficient.

### Strengths
1. The paper is clearly motivated and the mothodology is clearly explained. 
2. The method is thoroughly evaluated on different benchmarks.

### Weaknesses
1. The novelty of the paper seems limited to me, as the key contribution is to use frozen pretrained vision encoder instead of learning a new vision encoder from scratch to reduce computational cost. The idea of utilizing the strong prior of pretrained models for different downstream tasks has been extensively explored in different domains, and it's not too surprising to me that it will also work for off-policy RL on commonly used RL benchmarks with relatively simple visual features. 
2. MEME and Dreamer-v3 perform better than the proposed method overall. Why not also apply your method to them to see if they can work better or faster? Is it due to limited computational budget?

### Questions
1. Why integrating recurrent models has minimal computational cost for on-policy RL but not for off-policy RL? 
2. Why not also use the pretrained vision encoder for the current observation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new approach for recurrent reinforcement learning that reduces computational overheads leveraging non-learnable encoder (pre-trained model). They use an attention style mechanism to fuse the LSTM stream (encoded interaction history) and the CNN stream (encoded current state). They provide experimental analysis across different benchmarks and highlight their computational gain.

### Strengths
### Strengths:

1. This work proposes a new framework that requires a single pass of the observation with the current learnable model and uses embeddings from pre-trained vision models for previous state sequence. This significantly reduces the computational burden. 

2. Experiments are conducted across four different benchmarks. Experimental results are encouraging to an extent, especially in terms of computational overhead.

3. While their approach introduces a number of design choices, they provide thorough analysis of different possible choices. It is evident that the performance is not highly sensitive to those choices.

### Weaknesses
### Weaknesses:

1. While R2D2 has been mentioned as a similar prior work, no comparison with R2D2 has been presented. Further, comparison with other recent transformer-based approaches would be great to assess the wider impact of the work. 

2. The performance improvement is limited to a subset of environments within a benchmark. Such as in Procgen or Vizdoom, a certain number of environments see reward gain while using RISE.

### Questions
1. How did you obtain the human-normalized score for Procgen? Does not the original paper only provide reference for PPO-normalization and min-max normalization?

### Soundness
2

### Presentation
3

### Contribution
2
