# AIGCoder 1.0: Locally-Enhanced Language Modeling with Explicit and Structured Knowledge Memory

- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
Large language models (LLMs) have achieved remarkable breakthroughs across various applications. However, their architectures remain inefficient due to two main limitations: (i) self-attention lacks an explicit inductive bias for locality, leading to redundant modeling of sequence-internal local information; (ii) mixture-of-experts (MoE) implicitly couples knowledge storage with computational pathways, hindering flexible access to sequence-external global knowledge. To overcome these limitations, we propose AIGCoder (AI Generative Coder), a novel LLM architecture that augments the standard decoder with two dedicated modules: 1) Local Fusion Attention (LFA), which incorporates a convolutional fusion to attention, explicitly capturing local patterns and allowing the attention to operate on more informative representations; 2) Knowledge Memory Module (KMM), which introduces a parametric key–value memory that explicitly stores global knowledge in addressable slots, decoupling storage from computation and enabling direct knowledge retrieval. Together, these modules enable AIGCoder to achieve more efficient and effective integration of information at both levels. Experimental results show that AIGCoder converges 1.33× faster in pre-training than baseline models, underscoring its superiority over existing LLM architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses two fundamental issues with current LLM architectures: (1) the lack of explicit inductive bias in the attention mechanism to address local structures in the input sequence. (2) the lack of decoupling between knowledge extraction and forward computation path, or in other words, reliance on knowledge stored in the model parameters. The paper suggests mechanisms to address those issues and to integrate the two levels of knowledge (local, sequence-dependent and external, sequence-independent) in an efficient way.  They do that by adding two new modules to existing architectures: Local Fusion attention (LFA) and Knowledge Memory Module (KMM). LFA uses a convolution operation with a learnable kernel to aggregate local information in each token representation. KMM stores keys and values in an external memory which are retrieved through querying during inference. The final representation which goes into the attention layer is a combination (addition) of these two representations. Experiments show the advantage of their approach compared to baseline models in terms of overall performance and number of training steps.

### Strengths
The paper addresses fundamental issues in LLM architectures. Overall, it is well-written and fairly easy to read except for some details missing in the description of KMM. The authors conducted extensive experimentation against baseline models as well as ablations.

### Weaknesses
Incorporating explicit inductive biases into models is a tricky business. On the one hand, it can improve model training and efficiency, on the other, it may be limiting the learning process and generalization. One should have a very clear intuition backed by evidence for including inductive bias. In the case of locality, the authors argue that current architectures are not incorporating long and short range dependencies in the most efficient way because all tokens are treated equally. However, there is evidence that LLMs do obtain implicit recency (locality) bias through pre-training. Furthermore, for different data and different tasks (e.g. natural language vs coding), we would expect a different balance between long and short range dependencies. This begs the question: why should we add an explicit locality bias rather than let the model figure it on its own? 

I am surprised that there is no mention at all of previous works on memory-augmented models and their relation to KMM. I'd expect to see such a discussion in the related work section as well as comparative experiments.

### Questions
- Lines 52-53 "This may lead to suboptimal…” I don’t understand this sentence. 

- In what sense are current architectures inefficient in integrating local and external knowledge (lines 49-50)? Do you have evidence for that?

- Why is LFA better than sliding window attention?

- What do we need MLA for? If I understand correctly, you can apply your techniques directly on vanilla attention. I understand that it makes things more efficient, but for the presentation of your ideas, why add another unnecessary level of complexity?

- I’m missing details on how K and V in Equation 5 are being updated during training.  

- 174-177: repeated sentence

- Please provide more details on how the visualization experiment (Figure 2) was done. 

- The visualization experiment shows that some memory fields are strongly associated with specific domains. However, this happens in a relatively small fraction of the fields. Could this be just by chance? Have you conducted an analysis to rule that out? 

- Lines 430-431: "These show that increasing the kernel size consistently lowers training”: I wouldn’t make such a statement based on only two kernel sizes. Clearly as k gets closer to the entire sequence length, we don’t expect to see a benefit compared to regular attention.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces AIGCoder 1.0, a large language model (LLM) architecture designed to explicitly integrate both local and global inductive biases, achieving more efficient and interpretable language modeling. 

The model enhances the classic transformer decoder via two dedicated modules: (1) Local Fusion Attention (LFA), and (2) the Knowledge Memory Module (KMM). Comprehensive experiments demonstrate accelerated pre-training convergence, state-of-the-art (SOTA) performance on diverse language, reasoning, and coding benchmarks, and proven scalable behavior up to 60B parameters. The paper further provides ablation studies, visualization of knowledge domains, and a discussion on the architecture's interpretability, efficiency, and future directions.

### Strengths
This paper proposes an architectural approach by explicitly introducing mechanisms for local context fusion (via LFA) and global structured knowledge memory (via KMM). This design offers a potential solution to the known inefficiencies of standard self-attention and the implicit knowledge storage inherent in Mixture of Experts (MoE) models. 

The paper provides a comprehensive and detailed experimental evaluation. The AIGCoder model is assessed across multiple standard LLM benchmarks spanning diverse domains, where it consistently demonstrates robust performance increment in the main experiments. Furthermore, meticulous ablation studies are presented, which validate the necessity and individual contributions of the key architectural decisions.

### Weaknesses
**Limited Analysis of Memory Module Expressiveness and Learning Dynamics:** The theoretical motivation for KMM is well articulated, but the paper lacks a deeper examination of possible limitations, such as catastrophic forgetting, interference between fields, or scaling of the memory module. For example, what happens when the number of domains or knowledge fields is very high (much larger than 64) ? Are there empirical signs of field collapse and cross-domain interference?A discussion or targeted experiment would make the claims more robust.

**Related Work Gaps and Insufficient Positioning:** While the related work section covers some foundational works in efficient attention and MoE, it critically omits several direct predecessors that employ explicit knowledge memory or explicit working-memory mechanisms, and local context fusion via external stores.

Notably missing are recent works on explicit, model-addressable memory (e.g., Memory³ with explicit memory [1]), retrieval-enhanced models with large external data stores (e.g., RETRO [2], REALM [3]), language models with contextually relevant or editable external knowledge (e.g., LM-CORE [4], Li et al. 2024 [5]), and mechanisms for factuality improvement via explicit working memory(also [6]). Their absence hinders a proper positioning of KMM and may overstate its distinctiveness. These works should be explicitly discussed to contextualize KMM, clarifying methodological overlaps, true novelty, and incremental improvements.

**Minor Issues:** The text contains some minor repetition; for example, the term 'Multi-Head Latent Attention (MLA)' is mentioned twice in Section 4.1. Occasional typographical errors were also noted (e.g., “resprctively” instead of “respectively”), but these do not significantly hinder comprehension."

**[1]** Yang, H., et al. “Memory³: Language Modeling with Explicit Memory.” _arXiv preprint arXiv:2407.01178_ (2024).

**[2]** Borgeaud, S., et al. “Improving Language Models by Retrieving from Trillions of Tokens (RETRO).” _ICML 2022_ (2022).

**[3]** Guu, K., et al. “REALM: Retrieval-Augmented Language Model Pre-Training.” _arXiv:2002.08909_ (2020). 

**[4]** Kaur, J. N., et al. “LM-CORE: Language Models with Contextually Relevant External Knowledge.” _Findings of NAACL 2022_ (2022).  

**[5]** Li, B. Z., et al. “Language Modeling with Editable External Knowledge.” _arXiv:2406.11830_ (2024). 

**[6]** Chen, M., et al. “Improving Factuality with Explicit Working Memory.” _ACL 2025 (Long)_ (2025).

### Questions
- Could the authors quantify the trade-off between the number of memory fields and task performance? Does field specialization in KMM saturate beyond a certain parameter scale or when trained across many diverse domains? 
- What is the empirical behavior of KMM when exposed to out-of-distribution or rare knowledge queries?
- Could you examine robustness to adversarial or noisy post-training updates of memory fields? Specifically, whether fields can be edited/deleted and how performance degrades—or recovers—thereafter. A brief targeted experiment would help support the transparency/editability claims.
- An open-ended question: your KMM module explicitly stores global knowledge. Compared with prior approaches that use parameterized/latent memory (e.g., MemoryLLM [1] and its scalable extension M+[2]), as well as memory-agent systems such as Mem0[3] and Letta[4], in which tasks or domains does your method have clear advantages?

**[1]** Wang, Y., et al. “MEMORYLLM: Towards Self-Updatable Large Language Models.” _arXiv preprint_ (2024).

**[2]** Wang, Y., et al. “M+: Extending MemoryLLM with Scalable Long-Term Memory.” _arXiv preprint_ (2025). 

**[3]** Mem0 team. “Mem0: Universal Memory Layer for AI Agents.” _GitHub repository_ (2025).

**[4]** Letta team. “Letta (formerly MemGPT): Platform for Building Stateful Agents.” _GitHub repository_ (2025).

### Soundness
4

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
This work introduces AIGCoder1.0, which introduces local fusion and memory modules in the MoE model layer block to enhance the language modeling. The results are very remarkable when trained on only less than 400B tokens compared with other competitive open-source LLMs.

### Strengths
1. The architecture design is very straightforward and can capture the global information using retrieved attentions.

### Weaknesses
1. To be honest, I think the first thing you need to check is the data leakage issue within your pre-training or sft dataset. I feels like the 91.5 mmlu score is too high for a 7B when trained on only 335.54B tokens. Normally, I will regard this as data contamination problem.

2. Another confused thing is that this work almost neglect all the related works in memory LLM, starting from Memorizing Transformer, LongMem, MemGPT, MemoryLLM, M+, etc. The architecture design heavily overlaps with these previous works but introduces memory modules in MOE models. You should add at least 1 paragraph in related work section to discuss all these related works.

### Questions
1. The major obstacle in building memory LLM is the distributed training for the parametric memory module. I did not find the details regarding your pp,ep,sp,tp details. Did you use DP only to train the 7B model? How about the parallelism strategy for 60B model?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper identifies two key limitations in current large language models (LLMs): (i) the lack of an explicit local inductive bias in self-attention, and (ii) the absence of a structured memory design in mixture-of-experts (MoE) architectures to store and retrieve global knowledge external to the current input. To address these issues, the authors propose AIGCoder, which introduces two new architectural components: the Local Fusion Attention (LFA) and the Knowledge Memory Module (KMM). The LFA integrates local-level inductive bias by applying group convolution kernels across divided feature groups along the hidden-state dimensions, while the KMM organizes global knowledge within a parameterized latent space. The authors conduct both pre-training and instruction tuning using this architecture and report superior performance compared to several open- and closed-source LLMs.

### Strengths
The paper presents two novel architectural designs—Local Fusion Attention and Knowledge Memory Module—proposing an alternative path for LLM beyond the current attention-based transformer paradigm. Exploring new architectures for large-scale language modeling remains an important and open research direction, and this work contributes meaningful ideas to that pursuit.

The experiments involve pre-training and instruction fine-tuning on two LLM backbones, with evaluations spanning a reasonable range of tasks, including language understanding, reasoning, coding, and mathematics. The inclusion of training-time measurements such as validation perplexity also helps to understand convergence behavior.

### Weaknesses
First, I see insufficient support for core technical claims.
The motivations behind both the Local Fusion Attention and the Knowledge Memory Module are not sufficiently substantiated. The necessity of adding a local inductive bias is neither empirically validated nor theoretically justified, and no prior reference to support this assumption. Similarly, the rationale for explicitly storing “general knowledge” outside the model’s main parameters lacks empirical or analytical grounding. What general knowledge is also remains unclear. 

Then, there are questionable designs and unclear alignment with motivation.
The operation of applying convolutional kernels over split feature groups in hidden states is not well-motivated. This design implicitly assumes that the subcomponents (“bits”) of a hidden vector are independent and separable which is a questionable premise. Furthermore, while the paper motivates the need for local inductive bias across tokens in a sequence, the proposed LFA is applied within the hidden dimensions of each token, which does not clearly align with the stated motivation.

The experimental analysis is also weak. 
The evaluation is limited in depth. There are no ablation studies or analytical results explaining how LFA and KMM contribute to performance gains. The claim that AIGCoder achieves superior reasoning and coding performance is particularly unconvincing, since no reasoning reinforcement learning (RL) or specialized training is reported besides pre-training and supervised fine-tuning (SFT). It is unclear how the proposed modules alone could match or outperform models such as Claude or Gemini, which undergo extensive reasoning-specific training.

Finally, some parts of the paper make exaggerated claims. For example, the authors assert that KMM enables “potentially easier knowledge editing and more interpretable inference,” yet no experiments or explanations are provided to support these statements. Since KMM stores knowledge in a latent parameter space, it is not evident how it improves interpretability or editability in practice.

### Questions
Which checkpoints are used for the baselines? Are they base, instruction-tuned, or chat versions?

What is the backbone model of AIGCoder?

Are there any experiments demonstrating that KMM improves knowledge editing or interpretability?

Typo: The sentences in lines 174–176 are repeated.

### Soundness
2

### Presentation
2

### Contribution
2
