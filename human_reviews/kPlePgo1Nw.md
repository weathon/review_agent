# Learning the Language of Protein Structure

- Decision: Reject
- Scores: 3, 8, 5, 6

## Abstract
Representation learning and \emph{de novo} generation of proteins are pivotal computational biology tasks. 
Whilst natural language processing (NLP) techniques have proven highly effective for protein sequence modelling, structure modelling presents a complex challenge, primarily due to its continuous and three-dimensional nature.  
Motivated by this discrepancy, we introduce an approach using a vector-quantized autoencoder that effectively tokenizes protein structures into discrete representations.  This method transforms the continuous, complex space of protein structures into a manageable, discrete format with a codebook ranging from 4096 to 64000 tokens, achieving high-fidelity reconstructions with backbone root mean square deviations (RMSD) of approximately 1-5 \AA. To demonstrate the efficacy of our learned representations, we show that a simple GPT model trained on our codebooks can generate novel, diverse, and designable protein structures. Our approach not only provides representations of protein structure, but also mitigates the challenges of disparate modal representations and sets a foundation for seamless, multi-modal integration, enhancing the capabilities of computational methods in protein design.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper, "Learning the Language of Protein Structure," proposes a new approach to protein structure modeling by representing the continuous 3D forms of proteins in a discrete, sequence-like format. This work tries to bridge protein sequence and structure modeling using techniques drawn from natural language. The authors introduce a vector-quantized autoencoder to translate 3D protein structures into discrete tokens. This tokenization leverages a codebook of 4096–64000 tokens, achieving effective compression while retaining high fidelity in structure reconstruction with root mean square deviations (RMSD) between 1–5 Å. The learned discrete representations are applied to generate novel and diverse protein structures using a simple GPT model, which demonstrated competitive performance in designing structurally viable proteins against established diffusion-based models.

### Strengths
1. The paper is generally well-structured and clearly written.
2. The use of vector-quantized autoencoders combined with transformer-based NLP models is an interesting application of NLP methodologies to the domain of protein structure.
3. The application of finite scalar quantization (FSQ) to manage codebook stability issues shows creativity in overcoming known challenges in quantized representation learning.
4. The authors explore their model’s capacity with both qualitative and quantitative metrics. Ablation studies and variations in codebook size and downsampling ratios.

### Weaknesses
1. Tokenization for protein structures is not entirely novel; works like FoldSeek and several others [1-10] have experimented with tokenized representations. Although the paper claims novelty in its discrete, sequence-based approach, it could better establish its unique contributions by clearly differentiating its tokenization strategy from these previous methods. To strengthen the paper’s contribution, the authors should clarify and empirically demonstrate the advantages of their approach over existing methods.
2. The primary application explored in the paper is protein structure generation using a GPT model. While this is promising, the paper could be more impactful if it included additional downstream applications to show the broader utility of its tokenized representation.
3. The technical novelty is quite limited. Almost all the components of this approach are built on existing techniques, like MPNN, FSQ quantization, AlphaFold's structure module, and FAPE loss. 

[1] van Kempen, Michel, Stephanie S. Kim, Charlotte Tumescheit, Milot Mirdita, Cameron LM Gilchrist, Johannes Söding, and Martin Steinegger. "Foldseek: fast and accurate protein structure search." Biorxiv (2022): 2022-02.

[2] Trinquier, Jeanne, Samantha Petti, Shihao Feng, Johannes Söding, Martin Steinegger, and Sergey Ovchinnikov. "SWAMPNN: End-to-end protein structures alignment." In Machine Learning for Structural Biology Workshop, NeurIPS. 2022.

[3] Lin, Xiaohan, Zhenyu Chen, Yanheng Li, Zicheng Ma, Chuanliu Fan, Ziqiang Cao, Shihao Feng, Yi Qin Gao, and Jun Zhang. "Tokenizing Foldable Protein Structures with Machine-Learned Artificial Amino-Acid Vocabulary." bioRxiv (2023): 2023-11.

[4] Gao, Zhangyang, Cheng Tan, Jue Wang, Yufei Huang, Lirong Wu, and Stan Z. Li. "Foldtoken: Learning protein language via vector quantization and beyond." arXiv preprint arXiv:2403.09673 (2024).

[5] Li, Mingchen, Yang Tan, Xinzhu Ma, Bozitao Zhong, Huiqun Yu, Ziyi Zhou, Wanli Ouyang, Bingxin Zhou, Liang Hong, and Pan Tan. "ProSST: Protein Language Modeling with Quantized Structure and Disentangled Attention." bioRxiv (2024): 2024-04.

[6] Gaujac, Benoit, Jérémie Donà, Liviu Copoiu, Timothy Atkinson, Thomas Pierrot, and Thomas D. Barrett. "Learning the Language of Protein Structure." arXiv preprint arXiv:2405.15840 (2024).

[7] Hayes, Tomas, Roshan Rao, Halil Akin, Nicholas J. Sofroniew, Deniz Oktay, Zeming Lin, Robert Verkuil et al. "Simulating 500 million years of evolution with a language model." bioRxiv (2024): 2024-07.

[8] Lu, Amy X., Wilson Yan, Kevin K. Yang, Vladimir Gligorijevic, Kyunghyun Cho, Pieter Abbeel, Richard Bonneau, and Nathan Frey. "Tokenized and Continuous Embedding Compressions of Protein Sequence and Structure." bioRxiv (2024): 2024-08.

[9] Wang, Xinyou, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. "DPLM-2: A Multimodal Diffusion Protein Language Model." arXiv preprint arXiv:2410.13782 (2024).

[10] Lu, Jiarui, Xiaoyin Chen, Stephen Zhewen Lu, Chence Shi, Hongyu Guo, Yoshua Bengio, and Jian Tang. "Structure Language Models for Protein Conformation Generation." arXiv preprint arXiv:2410.18403 (2024).

### Questions
I'm wondering about the key advantages of this work compared to the listed works above. What's your unique contribution?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes to discretize protein backbone structures so that we can use the well-developed machinery for modeling sequences of discrete tokens to analyze and generate protein structure. The authors train use a ProteinMPNN-style embedder, a cross-attention based discretizer, and an AlphaFold folding-module based decoder to parametrize finite scalar quantization of protein backbones, showing that it is possible to achieve accurate reconstructions. Finally, they train a simple autoregressive transformer to generate protein backbone structures in this discrete latent space.

### Strengths
The paper makes a clear and strong case for discretizing protein backbone structures in order to use discrete sequence models to analyze and generate protein backbones. While the individual parts (discretizing protein structure, generating protein structures, using sequence modeling architectures on discretized protein structures) are present in previous work, this work is unique in its focus, thoroughness, and attention to detail on the quality of the discretization and reconstruction. The main claims are well-supported by the experiments, which do a good job of following the current standards in the field. The paper generally flows very well, and the main points and contributions are clear.

### Weaknesses
The two biggest areas for improvement are in explaining the paper's significance in light of the (missing) related work in this area and on the completeness of the ablations and metrics reported. I think this could be a really strong paper if most of these weaknesses are addressed. 

## Related work

The authors should mention and describe how their work is different from [ProTokens](https://www.biorxiv.org/content/10.1101/2023.11.27.568722v4.abstract), which also aims to discretize protein structures with high reconstruction accuracy, and [FoldToken2](https://www.biorxiv.org/content/10.1101/2024.06.11.598584v3.abstract), which also discretizes protein structures and trains (with very sparse details) flow matching and autoregressive generative models on those tokens. 

There are also other works that train sequence models on tokenized structures. [SaProt](https://www.biorxiv.org/content/10.1101/2023.10.01.560349v1) is a masked language model trained on both sequence and structure tokens, while [ProstT5](https://www.biorxiv.org/content/10.1101/2023.07.23.550085v1) uses tokenized structures to unify structure prediction and inverse folding into a single model. [ESM3](https://www.biorxiv.org/content/10.1101/2024.07.01.600583v1) trains their own structure tokenizer and claims to be able to co-generate sequence and structure. The paper should mention these related works and explain how it is distinct and/or builds on this prior work. 

To be clear, I believe that this paper does make a significant contribution even in the light of these related works. However, not all readers will know the related literature quite as closely, and the paper would be greatly improved by properly situating itself in this context. 

## Experiments

1. While Table 1 nicely summarizes reconstruction quality, we need to know more than the mean RMSD and TM. At a minimum, we need to see distributions of reconstruction quality. It would be even nicer to have an analysis of what kinds of structures are reconstructed well and poorly.
2. Relatedly, the authors use a random split for training and evaluating the quantization and reconstruction. This means that there are likely test structures that are very similar to training structures. The authors should evaluate how well the reconstruction generalizes for structures that are more/less similar to those seen in training. 
3. While it's true that backbone diffusion models generally list the proportion designable, then the proportion of the designable backbones that are novel, then the proportion that are diverse, this isn't what we really care about for unconditional generation. What we actually want is the number of designable backbones that are distinct from each other and from natural backbones, in other words, the number (or proportion) of total backbones generated that are designable *and* novel *and* diverse. Given that the models are trained on the PDB, it would be better to do the novelty assessment against the PDB than against CATH. 
3. Another metric for reconstruction quality especially relevant for protein design is whether the reconstructed backbones are predicted to be realized by the original sequence. This could be evaluated, for example, by obtaining the perplexity of the original sequence when designing the reconstructed backbone using ProteinMPNN. 
4. The paper claims that we want to enforce locality with a special attention mechanism during tokenization. This should be ablated. 
5. It would be interesting to see how using a different-sized codebook affects the quality of the downstream generative model. Do we always just want the biggest codebook we can afford? 


## Other weaknesses

1. Algorithm 2 needs to be expanded so that the reader has some reasonable chance of being able to implement it. I do not see how the current description facilitates down- or up-sampling. 
2. The paper describes the training hardware for the quantization model but not for the generative model. The training time is not described for either model.
3. While the paper makes it clear later, the initial section describing the token decoder should explicitly state that it uses the AlphaFold2 structure module architecture.

### Questions
In addition to the suggestions in the Weaknesses section:

1. How much compute was utilized to train the full model (tokenizer + generator)? For reference, in the United States, the White House recently issues an executive order saying that models trained on mostly biological sequences for more than 1e23 FLOPs need to be reported to the federal government. It's unclear whether protein structures count as biological sequences. By my estimate, 128 TPU v4-8s give a max of 140800 teraflops, so assuming a utilization of 50%, you'd get to 1e23 FLOPs after about 396 hours. 
2. It'd be interesting to know whether the generated length distribution matches that of the training set. It'd also be interesting to see how the designability/novelty/diversity metrics look if the baseline models were sampled from at exactly the same empirical distribution as the samples from the tokenized generator. 
3. I'm curious why the length vs scRMSD isn't monotonic for this model is for diffusion and flow matching backbone models.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work introduces a method for tokenizing protein backbones. The structure is encoded as a graph to use a GNN as the encoder. Authors use the FSQ method for quantization and reconstructions are trained using the FAPE loss from AF2. To mitigate the fact that many protein families share similar structures, data is sampled inversely proportional to its MMSeqs-assigned cluster. At a codebook size of K=64000, the average reconstruction has 1.59A RMSD and TM score=0.95. Authors further examine the effects of downsampling and altering the number of codes (4096, 64000), finding that using 4096 codes reduces the performance, but not by too much.

### Strengths
* Structure tokenization is an important area of research, setting the foundational work necessary for many downstream uses.
* Considering the clustered nature of biological data during training is a good practice that should be more commonly adopted in the field
* Investigating the optimal codebook size is useful both for understanding proteins and for building similar tools

### Weaknesses
There have been a number of structure tokenization methods introduced, including all-atom ones [1,2] and backbone-only [3,4]. These should at minimum be cited, and preferably include a baseline comparison to.

It's certainly not fair to discount the novelty of a piece of work because other works have been since preprinted (I believe a version of this work was preprinted earlier than some of the others listed below). However, there are many open questions that authors could dig into, such as:
* Better understanding of how codebook size affects reconstruction, see [2]
* Investigating generation with more rigorous benchmarks; current results are sparse and do not perform great against baselines

At the current state of the field, for this work to be of interest to the ICLR 2025 audience, I think the authors should consider probing further into their results. There's a ton of interesting findings, but introducing a tokenizer in itself (especially given the limited backbone-only setting) might not be a significant contribution.

**Minor:**
* line 431: typo at ("Eguchi et al., 2022) ...

**References:**

[1] Bio2Token: All-atom tokenization of any biomolecular structure with Mamba (https://arxiv.org/abs/2410.19110)

[2] Tokenized and Continuous Embedding Compressions of Protein
Sequence and Structure (https://www.biorxiv.org/content/10.1101/2024.08.06.606920v1.full.pdf)

[3] FoldToken: Learning Protein Language via Vector Quantization and Beyond (https://arxiv.org/abs/2403.09673)

[4] ProTokens: A Machine-Learned Language for Compact and Informative Encoding of Protein 3D Structures (https://www.biorxiv.org/content/10.1101/2023.11.27.568722v2)

### Questions
* It's a bit unclear from the writing what the architecture & output of the decoder is. Is it finetuning the structure module of AlphaFold? Or just using the structure module architecture? Or something different altogether (e.g. using a GNN like the encoder?) would be helpful to see this clarified.

### Soundness
3

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
3

### Summary
The work proposes to use an FSQ-based auto-encoder to tokenize 3D protein structures at the residue level. They use a GNN encoder, IPA decoder, and a length compression scheme. They find that their autoencoder is able to achieve experimental precision in terms of RMSD and measure the RMSD and TM-score as a function of downsampling ratio/compression factor and codebook size. They then train a GPT model on the generated tokens and outperform FrameDiff but not RFDiffusion on scTM and scRMSD scores. They tend to lag behind in terms of novelty/diversity, but they note that there may be a tradeoff between novelty/diversity and validity.

### Strengths
1. The problem is very well-motivated: Tokenizing 3D structures allows for multi-modal integration in downstream language models.
2. To the best of my knowledge, autoencoder-based tokenizers are pretty novel in the field of computational protein biology.
3. Experiments, though limited, are sufficient to demonstrate that their tokens can be used for both reconstruction and downstream language modeling.

### Weaknesses
1. An ablation study comparing the FS quantization with VQ would help motivate the architectural choice.
2. You can include an ablation study demonstrating the importance of invariance in your encoder architecture. Consider trying a non-invariant architecture and see how that impacts downstream language modeling. 
3. The downsampling of the sequence makes it harder for the tokenizer to reconstruct the structure, but also makes it easier for the language model to learn. It would be great to demonstrate this tradeoff in your experiments. Also, I think the same can be said about the invariant feature representation.
4. You may want to compare with a hand-crafted tokenizer (e.g. BPE-based tokenizer) in terms of reconstruction accuracy, downstream generation validity/diversity.

### Questions
1. Have you tried extending to side chain conformations?

### Soundness
3

### Presentation
3

### Contribution
3
