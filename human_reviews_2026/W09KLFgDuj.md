# LDARNet: DNA Adaptive Representation Network with Learnable Tokenization for Genomic Modeling

- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Genomic foundation models increasingly adopt large language model architectures, yet almost all rely on fixed tokenization schemes such as $k$-mers or BPE. These approaches impose arbitrary sequence boundaries and risk discarding biologically relevant signals. Recent work introduced dynamic hierarchical tokenization in an autoregressive setup, demonstrating the feasibility of adaptive tokenization but leaving masked language modeling and downstream evaluation unexplored. We present \textbf{LDARNet}, a 120M-parameter hierarchical genomic foundation model that adapts hierarchical compression to the masked language modeling paradigm. LDARNet combines BiMamba-2 state-space layers with selective attention and uses ratio-based regularization to learn stable token boundaries without supervised segmentation.  We evaluate LDARNet through comprehensive fine-tuning across 27 diverse tasks from the Genomics Benchmarks and Nucleotide Transformer suites, comparing against state-of-the-art models spanning 8M-2.5B parameters. LDARNet achieves 11 of 18 wins among compact models ($<$300M parameters) - a 5.5-fold improvement over the next-best alternatives - and establishes overall best performance on 5 challenging histone modification tasks, surpassing even 2.5B-parameter competitors. Notably, LDARNet wins 7 of 10 histone modification benchmarks, demonstrating that learnable compression boundaries effectively capture the long-range dependencies critical for epigenetic regulation modeling. These findings provide evidence that adaptive tokenization under masked language modeling yields biologically meaningful representations, and highlight hierarchical compression as a promising direction for efficient and scalable genomic foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This submission presents LDARNet, a hierarchical genomic foundation model for genomics. It targets a well-known limitation: the reliance on fixed, arbitrary tokenization schemes such as k-mers or BPE, which lack biological grounding. To address this, it adapts H-Net’s learnable tokenization concept to the masked language modeling setup. 

Built upon state-space BiMamba-2 blocks (for nucleotide-level processing) and a Transformer backbone (to operate on the compressed latent representations), LDARNet performs adaptive sequence compression while preserving bidirectional context. It is pretrained on human and multi-species genomic corpora and evaluated using a probe-only, frozen embedding protocol on 26 tasks from two genomics benchmarks. The training combines an MLM objective with a ratio loss regularizer to ensure stable, non-degenerate compression.

Results indicate LDARNet attains competitive performance on histone modification prediction and strong overall results compared to leading fixed-token and byte-level models. This provides the first evidence that MLM-trained adaptive tokenization can yield biologically meaningful representations.

### Strengths
**(S1)** This work is well-motivated. It explores and extends a key open modeling issue in genome-scale ML: the suitability and biological interpretability of learned, adaptive tokenization vs. the now-standard, arbitrary tokenization (k-mer or BPE). It also proposes a logical and timely solution: porting H-Net’s dynamic chunking into a non-autoregressive bidirectional MLM framework.

**(S2)** The LDARNet architecture itself is an insightful contribution. The hybrid design shows a  reasonable and well-justified engineering compromise, which marries the linear-time efficiency of BiMamba-2 state-space layers at the byte-level with the expressivity of Transformers in the latent space. Bidirectional enhancements (mean fusion, parameter sharing) are mathematically justified w/ derivations and argument in Sec. 3.2.1.

**(S3)** The experiment setup (Sec. 4, Tab. 1 & 2) covers a broad suite of benchmarks and provides direct comparisons to diverse baselines like GENA-LM, DNABERT-2, HyenaDNA, and others. The probe-only protocol isolates representation power from finetuning pipeline, which is a thoughtful and well-justified choice. LDARNet’s impressive performance on this setting shows evidence for the biological relevance of the learned representation.

**(S4)** The writing and presentation is clear with great logical flow. Visualization of model framework in Fig. 1 is clear, and method formula are provided in detail in Sec. 3. The limitations and future directions are also acknowledged in Sec. 7.

### Weaknesses
**(W1)** Missing References and Appendix. The most immediate and severe issue in my view is that the manuscript is incomplete. All reference symbols are missing (displayed as '?') throughout the entire manuscript. And the appendix is absent, which is explicitly referred to for hyper-parameters and reproducibility details. This should be a major flaw for a rigorous academic paper. However, given that ICLR permits revisions and iterative author-reviewer discussions, I would reserve final judgment on this shortcoming, assuming it is an oversight at this stage. I strongly encouraged the authors to provide a corrected, complete manuscript in the rebuttal phase. IMHO, this revision should include all properly formatted citations and the complete appendix. 

**(W2)** Incomplete literature review. The discussion of related work in Sec. 2 misses consideration of several important recent studies in tokenization for biological sequences, particularly: BiRNA-BERT [1], which targets adaptive tokenization in RNA. And [2] [3] both investigates tokenization’s direct effects in biological language models. I recommend the authors include these references in the revision to form a complete literature review.

**(W3)** Empirical analysis beyond benchmark comparisons. IMHO, the qualitative analysis could be richer, especially as there are already many mature analysis methods available (such as t-SNE, UMAP, etc.). These methods align exceptionally well with the claim that the learnable tokenization in this paper is biologically grounded. Now, there is no validation of whether the learned chunking units align with regulatory, structural, or motif boundaries beyond the improved classification accuracy. This is suggestive but not conclusive. For example, as I can think of, this could be a figure visualizing the learned boundary probabilities $p_t$ from the router over genomic sequences with known, annotated motifs (e.g., TATA boxes, TF binding sites). It would be insightful and would help support the claims of biological interpretability. 

**(W4)** Insufficient Ablation Studies. LDARNet introduces several designs at once (the hybrid framework, BiMamba-2, bidirectional routing, ratio loss, etc.), yet the manuscript does not provide ablation studies to disentangle their individual contributions. In other words, it is unclear how critical the hybrid design is vs. a homogeneous BiMamba-2 model, or how essential the ratio loss is for stable training. My suggestions: at a minimum, two ablations are required: (i) A comparison of the hybrid LDARNet against a pure BiMamba-2 variant with a similar parameter scale, to validate the hybrid-by-design philosophy. (ii) A study of the ratio loss by training a model with its weight set to zero, to show its impact on compression and performance.

**(W5)** The necessity of hybrid model. LDARNet combines BiMamba-2 with Transformers, but there is no direct comparison results showing the necessity or impact of each component, like what happens if the model uses only BiMamba-2 or only Transformer layers? In my view, this is a critical set of experiments to show the method’s validity.


---
### Reference

[1] BiRNA-BERT: Adaptive Tokenization for Efficient RNA Language Modeling, NeurIPS 2024 FM4Science Workshop

[2] Effect of Tokenization on Transformers for Biological Sequences, Bioinformatics 2024

[3] The Impact of Tokenizer Selection in Genomic Language Models, Bioinformatics 2025

### Questions
Most of my major concerns and related recommendations have been stated in the Weaknesses section. I encourage the authors to focus their efforts on addressing those points, as they are critical for strengthening the manuscript in the rebuttal stage.

The following are more specific, minor questions to help the authors think more deeply about certain design choices and experiment setups, which might be helpful for this and future work:

- Are there any limitations or failure cases in the ratio regularizer? Have the authors observed degenerate solutions in practice? More detail on how the regularizer interacts with model convergence would be helpful.
- Are there qualitative differences in representation quality or interpretability when moving from fixed-token (k-mer) to learned-token boundaries? It is possible to illustrate (perhaps in appendix) token maps over sample sequences?


---
## Justifications:

I first give a rating of 6, primarily due to the clear motivation, the reasonable choice of hybrid architecture, and the strong results on histone tasks. In particular, the performance suggests the potential practical value, which is critical for biology and genomics. I would be glad to raise my rating if thoughtful responses and improvements are provided. Conversely, if most of the concerns remain unaddressed, I may also lower my score.

I hope these comments help my fellow reviewers and ACs understand the basis of my recommendation. I am open to follow-up discussions to reach a consensus for the final decision.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces LDARNet, a hierarchical genomic foundation model that employs learnable tokenization to improve the representation of genomic sequences. Unlike traditional methods that rely on fixed tokenization schemes (such as k-mers), LDARNet adapts the H-Net architecture to the masked language modeling (MLM) paradigm. The model combines BiMamba-2 outer layers with a Transformer backbone, allowing for efficient processing of genomic data while preserving biologically meaningful features. The authors demonstrate the model's effectiveness through extensive evaluations across multiple genomic tasks, achieving competitive performance without task-specific fine-tuning.

### Strengths
- Motivation is reaonsable. 
- The proposed method achieves good performance on several downstream tasks.

### Weaknesses
- Lack of novelty. In the introduction part, the authors sumarize three key contributions (two technical contributions and one experimental contribution). Both technical points have similar work already published, and this paper lacks an in-depth discussion and performance comparison with existing work. e.g. dynamic leanable dna tokenizer [1] and mamba networks for DNA modeling [2].
- Poor presentation. The structure of the paper is very chaotic, and the writing intentions of the paragraphs are unclear, filled with numerous writing errors. For example, the introduction section is too brief and completely lacks information about the methods, missing key details. All cross-references in this submission are incorrect.
- Limited experiments. The experiments in the paper are insufficient to support the claims, including:
  - lack of comparison with key models;
  -  lack of latency analysis; 
  - and lack of ablation experiments. etc

Overall, I think this manuscript is not ready for publication in ICLR'26.

[1] Model Decides How to Tokenize: Adaptive DNA Sequence Tokenization with MxDNA, NeurIPS'24

[2] Caduceus: Bi-directional equivariant long-range dna sequence modeling, ICML'24

### Questions
please refer to weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces LDARNet, a hierarchical genomic foundation model that uses a learnable, adaptive tokenization approach instead of fixed schemes like k-mers. Featuring a hybrid BiMamba-2 and Transformer architecture, the model adapts the H-Net framework to the masked language modeling paradigm. Without the need for finetuning, LDARNet's performance is comparable to state-of-the-art models, and it has achieved new state-of-the-art results on multiple histone modification tasks.

### Strengths
1. Departing from the fixed tokenization of conventional k-mer or BPE methods, this work pioneers a dynamic, hierarchical approach that resolves their inherent limitations.
2. Despite the absence of task-specific finetuning, LDARNet achieves competitive performance with state-of-the-art Transformer baselines and sets new SOTA results on multiple histone modification tasks.

### Weaknesses
1. The manuscript is poorly prepared. The reference is missing. The equations contain ambiguous notations without clear definitions. For example, Eq. (1) has both s and S, while $0 \le s < S$, i.e. S can not be reach. Eq. (5) has M2, which is undefined.
2. The reported results are not strong enough to support the claims. In Table 2, the proposed method only achieves SOTA on two tasks. In Table 2, the proposed method only achieves SOTA on half of the tasks.
3. The paper lacks ablation studies to demonstrate the necessity of each module.
4. There are missing baselines in the domain, for example Evo, Evo2.

### Questions
1. Please provide ablation results that clarify the incremental contributions of each module.
2. Please provide comparisons against recent DNA LMs under the same experimental setup.
3. Please include the discussion on computation budget.

### Soundness
2

### Presentation
1

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
This paper introduces LDARNet, a hybrid BiMamba-2/Transformer model for genome sequencing that uses a novel tokenization regularizer and claims state-of-the-art results over Transformer-based foundation models. While the performance is promising, the paper lacks essential ablation studies and crucial baseline comparisons to fully validate its contributions.

### Strengths
- The proposed LDARNet architecture has strong performance on the human and multi-species genome benchmarks, reportedly surpassing existing SOTA models.
- The methodology is nicely presented in details.

### Weaknesses
**Presentation**
- Many citations throughout the paper are broken, rendering as (?) in the paper and not appearing in the reference list. Also, the citation for DNABERT renders as (dna) and is unlisted in the references list.
- Some internal references to Tables/Figures (e.g. L503) are also broken and require correction.
- There is a stub appendix after the references, which according to L502 is supposed to contain training details.

All together, these errors leave the paper feeling not just unpolished but also unfinished.

- The methodological novelty of the shared weights mechanism in BiMamba-2 (Sec 3, L152) is unclear, as it is not sufficiently differentiated from how the weights are shared in the highly similar Caduceus model (despite this being cited heavily by the authors).

**Soundness**
- The experimental section (e.g. Table 1) fails to include a important benchmark comparison against Caduceus, which uses a highly similar bi-directional Mamba MLM setup on genomics benchmark.
- The major contribution of the "Learnable DNA tokenization" (Sec 2.1, L065) is unsubstantiated, as no ablation study validates its effectiveness against simpler tokenization methods.
- A key baseline, a vanilla Mamba-2+Transformer hybrid (cf. Sec 9.2.3 of Mamba-2), is missing, making the architectural contribution hard to assess.
- The work is missing ablations to demonstrate the choice of hyperparameters such as the choice of compression ratio and ratio loss weighting.
- For completeness, it would be better to report both fine-tuning and linear probe performances, rather than only linear probe. Even using a frozen encoder probe, there are other options one can consider than a linear probe on an average of the tokens - one could perform a non-linear (MLP) probe, or an attentive probe (Chen et al, 2023; Bardes et al, 2024; Greyson Brothers, 2025; Psomas et al, 2025). An attentive probe can be more indicative of the performance which will be obtained from fine-tuning than a mean pool linear probe that requires embeddings to be well aligned across the sequence length.
- It is currently unclear whether the mean pooling (L353) includes CLS tokens of models which have them. Why was linear probing of the CLS tokens not considered instead of the mean of the embeddings?

**Minor**
- The background on hierarchical and learnable tokenization in LLMs more broadly should be more extensive. Works such as Byte Latent Transformer and Large Concept Models are not cited and I think would be appropriate in this regard.
- L179: The citation for Mamba-2 is incorrect. It points to an empirical study, not the [original model paper](https://proceedings.mlr.press/v235/dao24a.html).
- L188: Gap in Eq 4 for $C_t$

**References**
- [Caduceus](https://arxiv.org/abs/2403.03234): Schiff et al (2024). "Caduceus: bi-directional equivariant long-range DNA sequence modeling" ICML 2024.
- [Mamba-2](https://proceedings.mlr.press/v235/dao24a.html): Dao and Gu, (2023). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality". ICML 2023.
- [CAE](https://doi.org/10.1007/s11263-023-01852-4) Chen et al (2022). "Context Autoencoder for Self-Supervised Representation Learning." Int J Comput Vis 132, 208–223 (2024). doi:[10.1007/s11263-023-01852-4](https://doi.org/10.1007/s11263-023-01852-4)
- [V-JEPA](https://arxiv.org/abs/2404.08471): Bardes et al (2024). "Revisiting Feature Prediction for Learning Visual Representations from Video". TMLR 2025.
- Greyson Brothers (2025). "Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs". ICML 2025. arXiv:[2506.09215](https://arxiv.org/abs/2506.09215)
- Psomas et al (2025). "Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency". arXiv:[2506.10178](https://arxiv.org/abs/2506.10178)
- [Byte Latent Transformer](https://arxiv.org/abs/2412.09871): Pagnoni et al (2024). "Byte Latent Transformer: Patches Scale Better Than Tokens". ACL 2025.
- [Large Concept Models](https://arxiv.org/abs/2412.08821): Barrault et al (2024). "Large Concept Models: Language Modeling in a Sentence Representation Space".

### Questions
- L273, Eq 14: What is the notation where a subscript is a sum supposed to mean?
- L341: The authors say "To promote reverse–complement invariance, each sequence is sampled in forward and reverse orientations with equal probability.", but do you actually take the complement of the sequence? I do not see this stated in the paper, either as a complemented always taken when reversing the orientation, or an augmentation where the complement is taken stochastically with p=0.5.

### Soundness
2

### Presentation
2

### Contribution
3
