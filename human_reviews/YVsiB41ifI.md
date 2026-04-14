# Phantom of Latent for Large Language and Vision Models

- Decision: Reject
- Scores: 6, 6, 6, 3

## Abstract
The success of visual instruction tuning has accelerated the development of large language and vision models (LLVMs). Following the scaling laws of instruction-tuned large language models (LLMs), LLVMs have also further increased in size, with examples including 26B, 34B, and even 80B parameters. While this increase in model size has yielded significant performance gains, it demands substantially more hardware resources for both training and inference. Consequently, there naturally exists a strong need for efficient LLVMs that achieve the performance of larger models while being smaller in size. To achieve this need, we present a new efficient LLVM family with model sizes of 0.5B, 1.8B, 3.8B, and 7B parameters, Phantom, which significantly enhances learning capabilities within limited structures. By temporarily increasing the latent hidden dimension during multi-head self-attention (MHSA), we make LLVMs understand much more vision-language knowledge on the latent, without substantially increasing physical model sizes. To maximize its advantage, we introduce Phantom Optimization (PO) using both autoregressive supervised fine-tuning (SFT) and direct preference optimization (DPO)-like concept, which effectively follows correct answers while eliminating incorrect and ambiguous ones. Phantom outperforms numerous larger open- and closed-source LLVMs, positioning itself as a leading solution in the landscape of efficient LLVMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Phantom, a new family of Large Language and Vision Models (LLVMs) that achieve comparable or superior performance to larger models while maintaining a smaller model size. They achieve this through a new cross attention architecture "Phantom Dimension", which temporarily expands the latent hidden dimension during multi-head self-attention, and "Phantom Optimization" (PO), a training strategy that minimizes the generation of incorrect and ambiguous answers. They claim Phantom outperforms existing open- and closed-source LLVMs in its size category across multiple benchmarks.

### Strengths
- Introduced techniques are novel and offer a potentially valuable new direction for efficient LLVM design.
- The reported results demonstrate impressive performance compared to other LLVMs, especially in the lower parameter regime (0.5B, 1.8B), suggesting effectiveness.
- The ablation study provides insights into the contribution of each proposed component, although some aspects could be further explored.

### Weaknesses
## Major

The paper's writing quality hinders clear communication of the results and techniques, making evaluation difficult.  For example, the abstract states:  "...LLMs either have further increased their sizes, reaching 26B, 34B, and even 80B parameters." This sentence is awkwardly phrased. The authors should prioritize improving the writing throughout the paper for clarity.  Using an LLM grammar checker would be beneficial.

The explanation of the proposed "Phantom Dimension" attention mechanism on page 5 lacks clarity.  A clearer presentation would first describe the attention mechanism used in the baseline architecture and then highlight the novel aspects introduced by Phantom. Figure 3 does not adequately illustrate these distinctions.

The paper introduces multiple novel elements: a new dataset, a potentially novel attention mechanism, and a new loss function. However, the ablation study omits the dataset. This omission prevents a proper assessment of whether the state-of-the-art results stem from the data itself or the specific combination of the Phantom architecture and optimization techniques.

The paper lacks a comparison to Malmo, a relevant work found at: https://arxiv.org/abs/2409.17146. This comparison is crucial for contextualizing the contributions.

## Minor 

The term "projectors" is used without clear definition.  If these are Multilayer Perceptron (MLP) layers, that term should be used for clarity and consistency.

### Questions
- Do you plan to share weights, code and data for training Phantom?

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
2

### Summary
The paper develops a new set of efficient VLLMs named "Phantom" that strong outperform models of comparable size on a range of VLLM benchmarks. The models are developed using open source components rely on a two-part solution - an architectural modification of the self-attention component of transformers; and a DPO-like optimisation method that utilises it.

### Strengths
* Strong results on a range of benchmarks.
* A set of ablations that clearly demonstrate which of the introduced components is responsible for the improved performance.
* The proposed architectural modification to the MHSA module highlights the importance of further exploring VLLM architectures beyond the native transformer.

### Weaknesses
**Major**
* "Phantom dimension" - the main of the two contributions of the paper is not described in an accessible manner in the paper, which significantly limits the ability to evaluate the paper and its potential impact. Consider adding pseudocode to describe what exact this modification does.
* It appears that the authors (primarily) used open source components to develop their models, however, it's unclear from the write up whether their models will similarly be made open source.

**Minor**
* It's not entirely clear "limited structures" used throughout the paper refers to.
* Is "Team et al" the correct citation for "Gemini Pro" (line 043)?
* Extra word in line 072 "... as a primary key to basically improve ..."?
* In line 086 and elsewhere - it's unclear what is meant by the phantom dimension "prepares to look and understand much more vision-language knowledge". It appears to be a subjective and strong statement - could you please elaborate on whether this is an intuition that the authors have, or it it can be concluded from some of the experiments presented in the paper? In any case claiming understanding due to an architectural modification might not be justified.
* Very minor, and admittedly subjective, but I find that the model name (Phantom) is overused in the text, e.g. "phantom dimension", "phantom optimisation", "phantom triplets".
* The added value of performing ablations for all model sizes in Table 4 is not that high in my opinion. Consider moving results for larger models to the appendix,

### Questions
* Will the models be made open source? Please explain if not.
* In line 076 the authors say state that "curating larger datasets" "brings in striking computational burdens during training". Could you please elaborate on this - what kind of dataset size increase is meant there in practice such that the resulting computational burden can be considered striking?
* What is the role of the regularisation parameter introduced in (2)? Is there a citation for this mechanism? If not, is the chosen value being ablated?
* In line 367 the authors mention that a single training step (of 128 examples) takes 2-5 days. Is this correct? Based on these numbers, it looks like a single epoch over the "Phantom triples" dataset would take more than 85 years.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Phantom, a new family of efficient large language and vision models (LLVMs) designed to improve performance within limited model sizes. To do so, the authors (1) curate a vision-language instruction dataset featuring both correct and incorrect / ambiguous answers, (2) propose a new attention scheme that computes attention with larger query, key, and value dimensions by concatenating the outputs of a first round of attention within query, key, and value projections to the queries, keys, and values, before performing attention once again on these concatenated features (Phantom Dimension), and (3) use a SimPO training objective over their curated dataset while training in a parameter-efficient scheme (Phantom Optimization). They show across a wide range of vision-language tasks that Phantom models are able to achieve competitive or state-of-the-art quality against various closed and open-source VLMs, and provide additional ablations to study the contribution of different Phantom components.

### Strengths
**Comprehensive evaluation**  
The authors evaluate their VLM architecture + dataset + training strategy on a variety of vision-language benchmarks, and show good performance compared to various baselines (both open and closed models) 

**Method ablations**  
I appreciated how the authors ablate several components of their proposed contribution and report performance over multiple tasks to study how these components add to the overall performance: (1) the weighted-average mechanism for combining self- and cross-attention outputs, (2) the concatenation of self- and cross-attention features in their Phantom dimension, and (3) the use of an SimPO optimization over their training (Phantom Optimization).

### Weaknesses
**Method motivation / understanding**  
While the authors say that increased feature dimension size in the attention interactions is important for improved quality (“to make LLVMs embed much more vision-language knowledge”; L244-245), I still don’t see the motivation for concatenating the cross-attention and self-attention projections. To increase dimensionality while preserving parameter-count, we could also arbitrarily repeat or expand the features from `head_dimension` to `2 * head_dimension`. It would be good to see some kind of ablation or more careful study motivating the concatenation of these features. 

**Method clarity**  
Some details and justification about the method are left out. For example, why use the softmax terms in Eq. 3 to mix between the outputs as opposed to simple learnable weights that can directly be element-wise multiplied with $\bar{O}_l$ and $\tilde{O}_l$? i.e., 
```python
w_bar   = nn.Parameter(d/h)  # d is model_dim, h is num_heads
w_tilde = nn.Parameter(d/h)  
# Combine, following L260
o  = torch.einsum('d,nhd -> nh', w_bar, o_bar)
o += torch.einsum('d,nhd -> nh', w_tilde, o_tilde)
```

In Equation 4, what is $\mathbb{E}_{\mathcal{D}}$ the expectation over? What are $\sigma$, $\beta$ and $\gamma$? Although these may be taken from SimPO, they should still be defined in a standalone manner. On a similar note, how does “Phantom Optimization” differ from SimPO? 

In Figure 3, how do the SoS tokens and User Prompt tokens correspond to the vision and language attention layers? Apologies if I missed this detail, but it seems important to clarify in regards to the overall architecture understanding. Related to the point on method motivation above, why does it make sense to do a first round of attention within each query, key, and value projection between the SoS token position output and the remaining tokens? 

**Component-specific study**   
Because of the introduction of dataset curation, it is unclear to me how much of the Phantom model performance is due to the data (collection over 2.8B visual instruction tuning samples, + phantom triples duration) versus the proposed architecture and training optimization components. It would be nice to see a data-controlled comparison, perhaps by applying the Phantom architecture and optimization over prior work’s training sets. Otherwise, with the current setup, although there is comprehensive comparison to different vision-language models, it’s hard to say what exactly is driving the boost in performance.

### Questions
See the questions raised in Weakness above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors propose a method that can temporarily increase the latent hidden dimension during multi-head self-attention (MHSA), without substantially increasing physical model sizes. The authors also introduce Phantom Optimization (PO) using both autoregressive supervised fine-tuning (SFT) and direct preference optimization (DPO)-like concept. Using these methods, they present a new efficient LLVM family with different model sizes with good performance.

### Strengths
The authors present an efficient LLVM family Phantom with enhanced learning capabilities within limited model sizes. They introduce Phantom Optimization (PO) that seems interesting. Phantom demonstrates good performance in their evaluations.

### Weaknesses
1. Authors only compare with other open-sourced Multimodal LLMs (MLLMs) using their checkpoints. They lack comparisons with related baseline methods using the same pre-trained models, datasets, and training configurations. There are several baseline methods [1,2,3,4,5] that contribute to the training algorithms of MLLMs. Besides, there are also huge amounts of works talking about how to modify the attention mechanism in Transformer models that need to be discussed.


2. It seems that the authors have not conducted experiments to compare Phantom Optimization with other RLHF methods (such as DPO, SimPO, ORPO).

3. It's not a good habit to use benchmark (evaluation) dataset for training, e.g., the MathVision benchmark.


[1] Jie, Shibo, et al. "Memory-Space Visual Prompting for Efficient Vision-Language Fine-Tuning." arXiv preprint arXiv:2405.05615 (2024).

[2] Luo, Gen, et al. "Cheap and quick: Efficient vision-language instruction tuning for large language models." Advances in Neural Information Processing Systems 36 (2024).

[3] Gao, Peng, et al. "Llama-adapter v2: Parameter-efficient visual instruction model." arXiv preprint arXiv:2304.15010 (2023).

[4] Wang, Weihan, et al. "Cogvlm: Visual expert for pretrained language models." arXiv preprint arXiv:2311.03079 (2023).

[5] Jia, Ding, et al. "GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer." arXiv preprint arXiv:2406.01210 (2024).

### Questions
1. Why do you use sequence (sos) token to enhance the latent hidden dimension?

2. Is the MHSA-based method specific for MLLMs? If not, why don't you consider using it in LLMs first?

3. Why are there so many blanks in Table 1, 2, 3?

4. Why don't you conduct experiments based on open-source datasets and models to ensure fair comparison?

### Soundness
2

### Presentation
3

### Contribution
2
