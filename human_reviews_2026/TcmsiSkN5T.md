# Chemical Language Models for Natural Products: A State-Space Model Approach

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Language models are increasingly applied in scientific domains such as chemistry, where chemical language models (CLMs) are well-established for predicting molecular properties or generating de novo compounds for small molecules. However, Natural Products (NPs)---such as penicillin, morphine, and quinine, which have driven major breakthroughs in medicine---have received limited attention in CLM research. This gap limits the potential of NPs as a source of new therapeutics. To bridge this gap, we develop Natural Product–specific CLMs (NPCLMs) by pre-training the latest state-space model variants, Mamba and Mamba-2, which have shown great potential in modeling information-dense sequences, and compare them with transformer baselines (GPT). Using the largest known collection of $\sim$1M NPs, we provide the first extensive experimental comparison of selective state-space models (S6) and transformers in NP-focused tasks, along with a comparison of eight tokenization strategies, including character-level, Atom-in-SMILES (AIS), general byte-pair encoding (BPE) and NP-specific byte-pair encoding (NPBPE). Model performance is evaluated on two tasks: molecule generation, measured by validity, uniqueness, and novelty, and property prediction (peptide membrane permeability, taste, and anti-cancer activity), evaluated using Matthews Correlation Coefficient (MCC) and Area Under the Receiver Operating Characteristic Curve (AUC-ROC). The results show that Mamba consistently generates 1–2\% more valid and unique molecules than Mamba-2 and GPT, while making 3-6\% less long-range dependency errors; however, GPT produces $\sim$2\% more novel structures. In property prediction, both Mamba and Mamba-2 outperform GPT by a modest but consistent 0.02 to 0.04 improvement in MCC under random splitting. Under stricter scaffold splitting, which groups molecules by core structure to better assess generalization to new scaffolds, all models perform comparably. In addition, chemically informed tokenization further enhances performance. For comparison, we include general-domain CLMs (ChemBERTa-2 and MoLFormer) and found that pre-training on $\sim$1M NPs achieves results on par with general CLMs trained on datasets over 100 times larger, emphasizing the value of domain-specific pre-training and data quality over scale in chemical language modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces **Natural Product Chemical Language Models (NPCLMs)**, addressing the underrepresentation of natural products (NPs) in molecular language modeling. Using a curated dataset of ~1M NP SMILES, the authors pre-train state-space models (Mamba, Mamba-2) and compare them with transformer baselines (GPT), systematically evaluating eight tokenization strategies. Experiments span **molecule generation** (validity, uniqueness, novelty, scaffold diversity) and **property prediction** (anti-cancer activity, taste, and peptide permeability). Results show that **Mamba** yields more valid and unique molecules with fewer long-range syntax errors, while **GPT** generates slightly more novel scaffolds. For property prediction, Mamba and Mamba-2 modestly outperform GPT under random splits but converge under scaffold splits. Crucially, **tokenization choice** strongly impacts outcomes, with smaller, chemically informed vocabularies (AIS, NP-specific BPE) outperforming large vocabularies. The study highlights that domain-specific pretraining on NPs can match or exceed general chemical LMs trained on 100× more data, emphasizing the importance of data quality and relevance over sheer scale.

### Strengths
1. **Essential field and motivation**: The paper addresses an important and underexplored problem: developing chemical language models specifically for natural products (NPs). Since NPs are chemically more complex and biologically significant than typical synthetic molecules, focusing on them fills a meaningful gap in current research and aligns well with drug discovery applications. By targeting this space, the paper contributes to an essential field with high scientific and practical relevance.
2. **Systematic and comprehensive experiments**: The study is notable for its breadth and rigor in experimental design. It benchmarks three different model families (Mamba, Mamba-2, GPT) across eight tokenization strategies, under both random and scaffold splits, using a curated dataset of ~1M NPs. The authors evaluate on molecule generation and property prediction tasks with multiple metrics, conduct error analysis, and even compare with general CLMs like ChemBERTa-2 and MoLFormer. This systematic approach makes the results highly credible and useful as a benchmark reference.
3. **Clear writing and presentation**: The paper is well-written and accessible, with a clear motivation, logical structure, and smooth narrative flow from background to experiments to conclusions. The figures and tables effectively summarize key findings, while detailed reproducibility notes in the appendices strengthen transparency. Overall, the writing quality and presentation make the technical contributions easy to follow for both machine learning researchers and domain experts in chemistry.

### Weaknesses
The main limitation of the paper is that the contribution largely reduces to an extensive set of empirical comparisons, but without introducing new methodological innovations or deriving deeper insights that could guide future work.

1. **Model comparison (Mamba vs GPT)**: The comparison between Mamba and GPT models offers limited value. The results mainly show that Mamba tends to generate molecules with higher validity and uniqueness, while GPT produces more novel ones. However, this trade-off between validity and novelty is easily tunable through standard decoding hyperparameters (temperature, top-k, top-p) in both model families, and outcomes are also highly sensitive to training settings such as the number of steps. As a result, the comparison does not provide clear guidance on which architecture is better suited for molecular tasks.
2. **Tokenizer comparison**: The tokenizer study similarly contributes little novelty. Prior work [1] has already shown the superiority of atom-level tokenization over generic BPE approaches. Moreover, the models here are trained only on canonical SMILES without augmentation by randomized SMILES, which is known to benefit large-vocabulary BPE tokenizers. This design choice biases the results and weakens the conclusions.
3. **Connection to natural products**: Although the paper positions itself around natural products, the methods and analyses are only weakly tied to NP-specific characteristics. The dataset merely provides a source of training molecules, and the main conclusions (validity vs novelty trade-off, tokenizer effects) appear general to SMILES modeling rather than unique to natural products.
4. **Choice of architectures for property prediction**: While autoregressive models are natural for molecule generation, they are not state-of-the-art for molecular property prediction, where graph neural networks and 3D-aware models currently dominate. The paper’s detailed comparison of autoregressive architectures in this context therefore seems of limited relevance.

[1] Comparing SMILES and SELFIES tokenization for enhanced chemical language modeling.

### Questions
1. **Motivation for NP-specific training**: Could the authors elaborate on the motivation for training exclusively on natural products? Intuitively, pretraining on a broader dataset that includes but is not limited to NPs might improve generalization, while still capturing NP-specific features. Why was NP-only pretraining chosen instead of a mixed strategy?
2. **Validity of generated molecules**: In Table 1, the reported validity ratios are relatively low (even with atom-level tokenization). Prior works on SMILES generation typically report validity above 90% [1]. Could the authors clarify why validity is lower here?
3. **Effect of model scaling**: What is the impact of scaling model size in this setting? Given that GPT-type transformers often benefit strongly from scaling, do the authors expect GPT to outperform Mamba under larger-scale setups?

[1] Molecular de-novo design through deep reinforcement learning.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The work compares three different model architectures on the generation and property prediction of natural products. Each architecture is combined with eight different tokenizers and two different train-test splits (random and scaffold). The results are presented systematically and align with the previous findings in the literature.

I believe that the paper should be rejected. 
1. Neither of the applied architectures nor the tokenizers is novel. SSMs are already used on natural product generation (Ozcelik 2025) and the architectures tested here display no significant improvement over prior work.
2. The findings largely revalidate the literature, offering limited new insights.
3. More controls and evaluation are needed to support the significance of the findings. 
4. The application domain, natural products, is a narrow field for the ICLR community.

### Strengths
1. A large and systematic comparison was conducted.
2. The presentation and results discussion are detailed and easy to follow.
3. The covered related work is comprehensive and well-aligned with the discussion..
4. Mamba architectures are applied to natural product modeling for the first time.

### Weaknesses
**Significance**
1. The focus of the work is natural products, which is a narrow subfield of drug discovery. The significance of the findings to the broader ICLR community is limited. The work can better fit technical and specialized venues of drug discovery.
2. Mamba and some tokenizers are applied to natural products for the first time here. However, these approaches perform similarly to the existing work and only confirm the findings (as cited multiple times by the authors), yielding no new insights for the community.

**Quality**

3. S4s are missing in the comparisons. A work that focuses on SSMs and natural products cannot ignore S4s, which are the first SSMs used in this domain. A non-language model baseline (such as www.nature.com/articles/s42004-023-01054-6) should also be included to better contextualize the performance of the architectures.
4. The authors use validity, novelty, or uniqueness to compare the models for generation. Yet, these metrics only measure the syntactic performance. More metrics, such as descriptor similarity, diversity, NP-likeness should be added to capture 'semantic' performance.
5. Scaffold-split does not guarantee distance between train and test sets -- non-identical scaffolds can still be very similar (arxiv.org/abs/2406.00873). A distance-based split should be applied to better capture generalization performance.
6. Hyperparameters are tuned on 5 percent of the data and used for training on the entire dataset. This is a jump from small-scale to large-scale training, and can lead to suboptimal performance for each architecture, jeopardizing the model-wise comparison.

### Questions
1. What is the motivation behind measuring validity, uniqueness, and novelty with different train/test splits? There are unconditional generation metrics that are independent of the test set. The results also show no real influence.
2. What are the novel findings of the work? The tokenizers comparison, error analysis, impact of the pre-training set, and model-wise comparison all largely confirm prior findings in the literature.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a specialized chemical language model for natural products. The model is based on state-space architecture. The model is benchmarked against transformer baselines.

### Strengths
The contribution addresses the gap in modeling natural products in chemistry.

The presentation is clear, the methodology is sound.

### Weaknesses
Aside from targeting narrow yet important chemical domain, such as the natural products, the paper does not report any particularly interesting results.

It's a solid research that would be best suited for a cheminformatic journal (Journal of Chemical Information and Modeling or something similar).

### Questions
Considering the reasons why state-space models are introduced, please provide some comparison of MAMBA and GPT inference speed and scaling with the sequence size.

### Soundness
2

### Presentation
2

### Contribution
2
