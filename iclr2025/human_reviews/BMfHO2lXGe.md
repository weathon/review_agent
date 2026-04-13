## Human Reviewer 1

### Summary
The paper presents ProtMamba, a homology-aware but alignment-free protein language model. It's based on the Mamba architecture and trained on concatenated homologous sequences. Results show its effectiveness in various tasks like sequence generation and fitness prediction.

### Strengths
- The authors propose a new training strategy, which effectively harnesses evolutionary information from homologous sequences without relying on MSA.
- The architecture based on Mamba blocks allows for handling extremely long contexts, which is beneficial for protein modeling as concatenating homologous sequences often results in long inputs.
- The results are comprehensive and prove the effectiveness of the proposed model.

### Weaknesses
I am not certain whether combining protein language with mamba can be regarded as "novel", but it is ok for me since such combination is not explored yet. An interesting aspect of this study is the training paradigm, which might provide insights for future studies. Nevertheless, a disappointing point is that almost no ablation study of the method can be found.

- The mask strategy (span-mask) is similar to that of T5 (you'd better add the missing reference: Raffel et al, Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer). There are some most related strategies that are not ablated, such as:
  - what if training with token-level mask instead of span-mask? Token-level mask means something like: "a b c <m1> <m2> <m3> g h  <eos> <m1> d <m2> e <m3> f". 
  - what if no masking strategy is used and the model is trained in an autoregressive fashion? 
  - what if doing mask-prediction without observing subsequent tokens, which means the input is "a b c <m1> <m2> <m3> g h" while the target is "b c d e f g h <eos>"?
  - I am not requesting the authors to ablate all of the above. However, for an AI conference, that would be very interesting and no ablation is unacceptable (in my opinion).
- The authors claim that the incorporation of position embedding and the concatenation strategy are important. I think the authors can present some results for comparison, for example, a model without position embedding and a model with the addition strategy (of the position embedding).
- Absence of comparison with a transformer (with FlashAttention) in the same setting.
- Have you tried to scale up the parameter of the architecture?

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper introduces **ProtMamba**, a novel protein language model that is homology-aware but alignment-free, addressing the limitations of traditional multiple sequence alignments (MSAs) in protein modeling. ProtMamba is built on the Mamba architecture, which enables it to handle very long sequences by efficiently processing concatenated homologous protein sequences. The model is trained using a hybrid of autoregressive modeling and Fill-in-the-Middle (FIM) objectives, making it highly versatile for tasks like protein sequence generation and mutational fitness prediction. ProtMamba demonstrates competitive performance on benchmarks like ProteinGym, outperforming similar-sized models in terms of efficiency and predictive accuracy. Additionally, the model excels in sequence generation tasks, producing novel sequences with structural properties comparable to natural proteins.

### Strengths
- ProtMamba is Homology-aware yet alignment-free
- Mamba architecture adaptation efficiently handle long contexts
- Explores hybrid training scheme for pLMs
- ProtMamba strong results on ProteinGym for mutational fitness prediction
- ProtMamba can generate reasonable sequences given homology context/sequences

### Weaknesses
1. Why use position encoding for ProtMamba? Given the recurrent nature of Mamba, positional information should theoretically be learned implicitly, which is why the original Mamba model does not employ explicit position encodings. The authors claim this is a significant modification, yet they fail to provide any experiments or ablation studies to demonstrate how this change improves model performance. I would like to see a comparison of with positional encoding (PE) vs. without PE, and additionally, a comparison of standard PE (additive) vs. the concatenation method used in ProtMamba. Authors should provide more details about PE implementation and comparison.

2. Although the motivation for using a long-context language model like Mamba is compelling, the paper does not benchmark a vanilla transformer at any context length, which is a significant weakness. Without this comparison, it is difficult to argue whether ProtMamba is the optimal architecture for this task in terms of performance. While it is clear that state space models (SSMs) like Mamba will likely outperform transformers in terms of efficiency, the lack of performance benchmarks makes it hard to assess if ProtMamba achieves the best results. 

3. Regarding the ProteinGym benchmark, when incorporating MSA or homology sequences, ProtMamba does not outperform MSA-based models. This challenges the fundamental premise of the paper that a homology-aware, MSA-free model should perform better and eliminate the need for MSAs. The lack of superior performance compared to MSA-based models suggests that ProtMamba's approach may not fully leverage homology information as effectively as MSA based models.

### Questions
1. How do the authors compute the total number of tokens and FLOPs for training? Can you provide more details on the implementation, such as whether you use any packages for FLOP calculations or how you approximate them?

2. Why did you choose to use a Poisson distribution for masking instead of other distributions?

3. In Figure 12, why does the model, in some DMS experiments, perform better without retrieval (MSA/homology sequences) than with MSA? This is especially surprising given that the model was trained over long contexts of homologous sequences. For instance, in VRPI_BPT7_Tsuboyama_2023_2WNM, there is a significant difference. Could you provide some explanations for this discrepancy?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper proposed a Mamba-based protein language model, ProtMamba, using concatenated sequences from protein families, with a FIM training objective. This approach allows for faster training and inference speeds. Experiments demonstrate ProtMamba’s versatility across protein fitness prediction and context-conditioned generation.

### Strengths
- **Novelty**: This is one of the first works to incorporate state-space models (SSMs) in protein language modeling, utilizing the Mamba architecture for efficient long-context handling. 

- **Innovative Input Design**: The input consists of a concatenation of unaligned homologous sequences separated by CLS tokens, with a carefully designed masking strategy. This design effectively leverages long homology contexts, maximizing the model’s ability to capture evolutionary information. Training with a Fill-In-The-Middle (FIM) objective enables flexible application to tasks like mutational effect prediction.

- **Comprehensive Model Implementation and Training**: The authors have put substantial effort into implementing and training ProtMamba, incorporating techniques inspired by DNA modeling, such as callback mechanisms and sequence length warmup.

### Weaknesses
- **Performance**: ProtMamba does not show significant performance improvements over strong baselines such as ESM-2 and Tranception, which may limit its competitive edge.

- **Additional Comparisons**: Including comparisons with other baseline models, such as PoET-205M[1], SaProt[2], ProtHyena[3], or PTM-Mamba[4] would provide a fair evaluation and offer a more comprehensive view of ProtMamba’s strengths and weaknesses.

- One advantage of Mamba is its faster generation capability compared to transformer-based models. The authors could extend ProtMamba’s use cases by addressing protein sequence generative tasks, such as unconditional generation. A more detailed discussion in Section 3.4 comparing ProtMamba to other generative models would strengthen the paper. You could follow the setting and metrics in PROTEINBENCH[5] paper.


[1] Truong Jr, T., & Bepler, T. (2023). Poet: A generative model of protein families as sequences-of-sequences. Advances in Neural Information Processing Systems, 36, 77379-77415.

[2] Su, J., Han, C., Zhou, Y., Shan, J., Zhou, X., & Yuan, F. (2023). Saprot: Protein language modeling with structure-aware vocabulary. bioRxiv, 2023-10.

[3] Zhang, Y. (2024). Prothyena: A fast and efficient foundation protein language model at single amino acid resolution. bioRxiv, 2024-01.

[4] Peng, Z., Schussheim, B., & Chatterjee, P. (2024). PTM-Mamba: A PTM-aware protein language model with bidirectional gated Mamba blocks. bioRxiv.

[5] Ye, F., Zheng, Z., Xue, D., Shen, Y., Wang, L., Ma, Y., ... & Gu, Q. (2024). ProteinBench: A Holistic Evaluation of Protein Foundation Models. arXiv preprint arXiv:2409.06744.

### Questions
- I’m curious about the impact of the number of context sequences on ProtMamba’s performance. For example, how does performance change with 0, 5, or more sequences, and is there a threshold beyond which additional context sequences no longer contribute to model performance? In your experiments on scaling FIM perplexity with the number of context sequences, it seems perplexity stabilizes with around 30 context sequences. Could you elaborate on this?

- Inference Efficiency: Could you report on ProtMamba’s efficiency during inference? Additionally, how does ProtMamba’s performance, memory usage, and computational efficiency scale with increasing context length compared to transformer-based models, like PoET?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper introduces ProtMamba, a state-space protein language model that is trained on sets of homologous sequences concatenated together. The model is trained to both generate sequences from scratch and to infill sequences using a fill-in-the-middle objective. For fitness prediction on ProteinGym, the method is shown to perform on-par with much larger models that explicitly use a multiple sequence alignment. For a narrower dataset of chorismate mutase activities, they demonstrate that they can apply prompt engineering and the FIM objective to improve fitness prediction. Finally, they perform a limited evaluation of the model’s autoregressive generation capabilities and show that the top 10% of generated sequences have some properties similar to natural proteins.

### Strengths
Prompting protein language models with sequences of homologous proteins (rather than an MSA) is an exciting direction for retrieval-augmented models. Given the inefficiencies of long-context transformers, using a state-space model for this objective is a natural idea to explore.

The choice to couple standard autoregressive language modeling with a fill-in-the-middle objective is an interesting one that has been relatively unexplored for protein language modeling, and the authors show its value for a fitness prediction task involving chorismate mutases.

The authors also provide a limited demonstration that prompting the model with high-activity sequences can improve its ability to perform fitness prediction.

### Weaknesses
Major points:
General clarity: I have a hard time following all the details of the paper because of the copious references to supplementary figures to support central claims in the main text.

Table 1: Given that the code & model parameters are publicly available, I would like to see the authors reproduce results for PoET [1], both without retrieval and with retrieval using the same prompt that is provided to ProtMamba. Given how similar the two approaches are, I cannot accept this paper without seeing this baseline.

Table 1: I would like to see how ProtMamba performs when one uses the autoregressive log likelihood of an unmasked sequence, rather than the FIM objective. Without this comparison, the value of FIM vs. autoregressive language modeling is less clear to me, since the scope of the experiment in Section 3.3 is much more limited.

Lines 208-212: More recent work [2, 3, 4] suggests that it is beneficial to mask as much as 50% of a sequence. I would like to see ablations that evaluate different masking fractions, rather than just results for the somewhat arbitrary choice of 20%.

Figure 4: There are major loss spikes and periods where the training loss actually increases. The authors should comment on the overall training stability of ProtMamba with some analysis of the gradient norms during training. This is important for a reader to decide whether they would choose to adopt Mamba over a transformer.

Minor points
Lines 89-95: Modern attention implementations like FlashAttention have linear memory complexity, though they still have quadratic time complexity [5]. The authors should update the text to reflect this fact.

Table 1: Indicating the top performers for each evaluation in bold would improve the readability of the table.

Figures 6-7: It is unclear to me what L denotes, and why the 150 < L < 250 line extends so much further than the other 2 in Figure 6.

[1] Truong Jr. & Bepler. PoET: A generative model of protein families as sequences-of-sequences. NeurIPS, 2023.
[2] Wettig et al. Should You Mask 15% in Masked Language Modeling? EACL, 2023.
[3] Tay et al. UL2: Unifying Language Learning Paradigms. arXiv, 2022.
[4] Hayes et al. Simulating 500 million years of evolution with a language model. bioRxiv, 2024.
[5] Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR, 2023.

### Questions
ProteinGym’s performance metrics are computed by averaging together the Spearman correlations for all assays with the same (UniProt ID, Function) pair, computing the average-of-averages for each function, and then averaging over functions. When computing the depth-based (and other) averages, I believe the UniProt IDs are averaged first as well, though not the functions. Can the authors confirm that they use the appropriate hierarchical averages to compute results for ProtMamba in Table 1?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
5