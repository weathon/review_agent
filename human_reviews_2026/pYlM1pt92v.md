# Bridging Gene Expression and Text: LLMs Can Complement Single-Cell Foundation Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Single-cell foundation models such as scGPT represent a significant advancement in single-cell omics, with an ability to achieve state-of-the-art performance on various downstream biological tasks. However, these models are inherently limited in that a vast amount of information in biology exists as text, which they are unable to leverage. There have therefore been several recent works that propose the use of LLMs as an alternative to single-cell foundation models, achieving competitive results. However, there is little understanding of what factors drive this performance, along with a strong focus on using LLMs as an alternative, rather than complementary approach to single-cell foundation models. In this study, we therefore investigate what biological insights contribute toward the performance of LLMs when applied to single-cell data, and how these models can complement single-cell foundation models to improve upon their performance. We first conduct a series of interpretability and ablation tests which show that LLMs leverage marker gene knowledge and simple gene expression patterns, contributing to their competitive performance. We then introduce scMPT, a proof-of-concept model which combines single-cell representations from LLMs and single-cell foundation models, demonstrating synergies between these representations through stronger, more consistent performance across datasets and tasks. We also experiment with alternate fusion methods, which highlight the potential of combining specialized reasoning models with scGPT to improve performance. This study ultimately showcases the potential for LLMs to complement single-cell foundation models and drive improvements in single-cell analysis.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores methods for bridging the gap between text-based Large Language Models (LLMs) and specialized single-cell foundation models trained on raw  single-cell data. Single-cell (gene expression) data is a high-dimensional vector of gene activity counts for a single cell that is usually very sparse (only a few genes are active/expressed in each cell at a time). The authors' core methodology is to convert this vector into a "cell sentence," which is a text string of gene names, ranked by their expression level. The main intuition/idea behind this conversion is to tap into the vast textual knowledge of LLMs, namely from curated datasets about gene markers for a specific biological class.  Namely, an LLM trained on medical literature would likely learn the association between the word "insulin" and the phrase "Beta cell."

The paper has two parts: First, it presents an interpretability study to investigate what biological insights LLMs learn from these "cell sentences." The authors use  standard interpretability techniques (LIME and Integrated Gradients) to uncover what "marker genes" where used for a prediction. They report that the genes the model found "important" were, in fact, biologically-known marker genes, confirming one of the parts of the hypothesis.  Second, the paper introduces scMPT, a simple fusion model. This model takes the frozen embeddings from a single-cell model (scGPT) and a text-encoder LLM (Ember-V1), concatenates them, and feeds them into a small, trainable MLP. The authors claim this fusion model achieves strong performance, even suggesting it is "competitive with full fine tunes of scGPT" on tasks like cell-type classification. The claims for this second part are highly questionable as discussed in weaknesses section.

### Strengths
The paper's only methodologically sound contribution is the interpretability analysis in Section 4.1. This section is well-executed and validates that the authors' model learns to identify key biological features. However, the novelty of this analysis is limited, as the core finding—that a model learns marker-gene associations for cell typing, validated against the PanglaoDB database—was previously demonstrated by the scBERT paper (Yang et al., 2022) . The paper's novel contribution is therefore a methodological validation in successfully replicating this concept on a different class of model (a general-purpose text encoder, not a specialized biological model). Therfore, this is a good but incremental validation rather than a novel conceptual discovery.

### Weaknesses
The paper has several critical flaws in methodology which leaves its central claims unsubstantiated. 

- The paper's headline claim is that its scMPT model is "even competitive with full fine tunes of scGPT." The key evidence is in Table 3, where scMPT (F1=0.745) appears to beat the "scGPT (full fine-tune - reported)" baseline (F1=0.718) on the Pancreas dataset, which is a highly surprising fact. However, the scMPT's score of (F1=0.745) was tested on an **intra-study** benchmark, as elaborated here:

> We focus our experiments on the datasets that were used to evaluate the cell type classification and clustering performance of GenePT (Chen & Zou, 2023)... For each dataset, we use the **same train/test split** as each of these respective works ... 

The cited GenePT and other cited work use a randomized intra-study (eg. they note "For the Aorta dataset, we used a random 80%/20% train/test split. ").  In contrast, the scGPT baseline (F1=0.718), was generated on a much harder, **cross-study** benchmark, elaborated here:

> The human pancreas dataset contains data from five scRNA-seq studies... The five datasets were **split into reference and query sets by data sources**. The reference set consists of data from **two data sources**, and the query set contains the **other three**.

Therefore, the authors are comparing their high score on a much simpler intra-study split task to a baseline's score on a hard generalization task, which invalidates their primary claim of SOTA-competitiveness. 


-  Flawed claim of minimal loss under "cell sentence" conversion: The paper builds upon on converting numerical expression data into "cell sentences, and and cites Levine et al. (2023) to claim  "minimal information loss" under such a conversion.  However, in a more careful examination of the cited paper (Levine et al. 2023), the  evidence shows that a linear model that attempts to reconstruct the expression values from the ranks, achieving an $R^2$ score of 0.815 (Figure 3 in Levine et al.). This means that nearly **20%** of the variance in the original data is lost during the conversion to a "cell sentence." Representing a 20% loss of information as a "minimal loss" without citing an exact figure is a misrepresentation of what the literature shows. 

- It also fails to cite and discuss CellPLM (Wen et al. 2023), a highly relevant related work that directly critiques the paper's "cell as sentence" methodology, arguing it misses crucial cell-cell relationships.
- Missing baseline: The paper's empirical evaluation fails to compare against scBERT (Yang et al. 2022), another prominent transformer-based foundation model for this exact task. (this paper is cited but not used as a baseline)

### Questions
Please answer the issues raised above. In particular, I look forward to hear author's responses and clarifications regarding the train/test split, which is my most important concern.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors pose the question on what LLMs learn from “cell sentences” and whether they complement single-cell FMs like scGPT. The paper finds (via IG+LIME and ablations) that LLMs lean on marker genes plus simple expression-rank patterns, then proposes scMPT, a light fusion that combines frozen Ember-V1 text embeddings with frozen scGPT features. scMPT typically matches or beats each component; a second scheme uses reasoning LLMs (e.g., o3-mini) to choose among scGPT’s top-3 labels and improves accuracy on several sets.

### Strengths
- The question is clearly formulated and the evidence are solidly given with interpretable analyses using integrated gradients and LIME against PanglaoDB markers.
- Given the experimental results, scMPT (frozen encoders + tiny MLPs) yields consistent gains; even reasoning-LLM reranking over scGPT’s top-3 helps.
- Some disease-phenotype results suggest benefits aren’t limited to cell-type classification.
- Hashing names and shuffling ranks show performance diffs on top ~10% in-context genes and their order, supporting the “simple pattern” claim.

### Weaknesses
- It seems that cell-sentence bias remains, as findings show reliance on high-rank genes/order; it’s unclear how robust this is to batch shifts.
- Regarding the claims about the knowledge of marker genes, we observe only some correlational evidence. In order to assure such claim, a stronger causal probe would be necessary, for example through counterfactual token tests.
- I find the eval for generative LLMs limited and biased: The head-to-head with GPT-4o / o3-mini uses only 100 test cells per dataset and constrains the label space to a provided list. That yields to sample-inefficient and risks optimistic estimates (especially for few, separable labels).
- If I follow correctly the methodology, IG and LIME are applied to an MLP on frozen Ember-V1 embeddings, not to the end-to-end LLM or the generative-LLM setup. So attributions correspond the MLP+Ember composite, not to scMPT nor the reranking LLMs.
- My main concern is that the ablations might hint some shortcut learning: Hashing names wipes performance, which might mean heavy reliance on lexical identity rather than biological semantics. These patterns look like lexical+rank shortcuts, not robust biology. The authors should evaluate under batch shift and low-depth cells.

### Questions
Check weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Summary
This paper (Bridging Gene Expression and Text: LLMs Can Complement Single-Cell Foundation Models) investigates how large language models (LLMs) can complement single-cell foundation models (scFMs) such as scGPT for cell type classification and related single-cell analysis tasks. The authors systematically evaluate several LLMs (e.g., Ember-V1 for encoder, GPT-4o for non-reasoning LLM, o3-mini for reasoning model) on cell sentence representations (cell represented as sorted genes, commonly use in most LLM based cell foundation models), perform interpretability and ablation studies, and propose scMPT, a simple fusion model that concatenates scGPT and LLM embeddings followed by an MLP classifier. Results show modest improvements over individual models and qualitative evidence that LLMs capture marker-gene knowledge and simple gene-expression patterns.

Overall Assessment
This work provides a careful and systematic evaluation of how LLMs interact with scFMs and contributes valuable insights for future research. However, the methodological innovation is limited, the improvements are small and insufficiently substantiated, and the experiments focus on relatively easy tasks.

### Strengths
1. The paper is well written and clearly structured, with strong motivation for exploring complementarity between text-based LLMs and single-cell foundation models.

2. Provides comprehensive, systematic experiments including ablations, interpretability analyses (integrated gradients and LIME), and comparisons across datasets to explore what the LLMs use to perform the cell-oriented classification tasks. This is a very good angle and offering a fresh perspective for the community. 

3. The concept of using LLMs as complementary rather than alternative to scFMs is timely

4. The work contributes useful empirical baselines for future multimodal single-cell modeling studies.

### Weaknesses
1. Limited architectural novelty.
The proposed fusion model (scMPT) concatenates scGPT and LLM embeddings followed by an MLP. This is a straightforward late-fusion ensemble rather than a novel modeling approach. The contribution is primarily empirical, focusing on evaluating existing models on existing datasets.

2. Use only scGPT to represent scFMs.
There are many other scFMs (e.g. scBERT, scPRINT, scFoundation, geneformer, just to name a few) which represent the cell in different ways, and may interact with LLM differently. Using only 1 seems relatively weak.

3. Small dataset
The generative re-ranking experiment uses only 100 samples per dataset, which is too few to claim meaningful cost savings or statistical significance.

4. Over-reliance on simple tasks.
The primary evaluation task is cell type classification, is known to be relatively easy (especially for well known cell types), as marker genes alone yield high accuracy. The study does not address more challenging cases such as fine-grained subtypes or rare cell populations, where complementary multimodal reasoning would be more revealing.

5. Scope dilution.
Although interesting, the inclusion of LLM-based reasoning experiments (e.g., scGPT + o3-mini / DeepSeek-R1) feels underdeveloped and somewhat disconnected from the main contribution. The small scale and lack of deeper analysis make it difficult to interpret their significance (e.g. o3-mini alone outperform other models in the Pancreas dataset). 

6. Unrealized potential of the fusion idea.
The fusion concept is promising, but a more principled hybrid approach, such as alignment / contrastive pretraining between scGPT and LLM embeddings, could yield stronger gains. The current setup does not fully exploit cross-modal synergies.

### Questions
1. Can the authors show per-class performance, especially for rare or ambiguous cell types?

2. Have the authors considered alternative fusion strategies (e.g., gating, cross-attention, contrastive alignment)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies what biological insights contribute toward the performance of LMs when applied to single-cell data, and how these models can complement single-cell foundation models to improve upon their performance. 

For the former, they found that LMs capture biological insight, and specifically knowledge of marker genes, as well as simple but effective gene expression pattern (top marker gene patterns). 

For the later, they introduced scMPT, which leverages both the representations generated by scGPT and an Ember-V1 text encoder, enabling better overall performance for cell-type classification and disease phenotype prediction.

### Strengths
- The paper tackles a relevant and timely question in computational biology: how language models interpret gene-level signals in single-cell data, and how this can be leveraged to complement single cell specialized foundation models.

- Interesting findings and analysis: The interpretability analysis connecting marker genes with model attributions is interesting and may be useful to biological researchers. And the discussion on gene-name hashing raises a thoughtful point—why language models might already capture much of the relational information between genes through their co-occurrence in “cell sentences,” rather than through explicit gene-name semantics.

- The paper is clearly written.

### Weaknesses
- Recommend for journal submission instead of ML conference: The method itself does not introduce substantial theoretical or engineering innovations. It resembles an empirical benchmark or ablation study for a biological problem rather than a new modeling framework.

- Limited benchmarking: The experiments include a narrow range of datasets and baselines. Since scGPT’s release, many other cell foundation models have emerged, and traditional non-foundation approaches remain competitive. Without broader comparisons, it is difficult to establish whether scMPT achieves state-of-the-art performance.

- Weak ML insight: While the biological analysis is sound, combining two frozen encoders and concatenate the two features for classification sounds like years-old architecture. Or in other words, the paper does not provide much new ML insights, such as novel objectives, architectures, or theoretical findings.

### Questions
1. how is the model in Figure 1 is trained? Directly using cell type classification using cross entropy loss to train from scratch? If yes, then is it natural to observe the interpretability methods give high score to marker genes as explanation? Because that’s the determinant part for cell types, and cell types are used for supervision.

2. Could you clarify why gene-name hashing improves performance, based on your understanding? Do you believe semantic information from gene names contributes meaningfully beyond their co-occurrence statistics in cell contexts?

3. Will you expand benchmarking to include additional more advanced cell foundation models and traditional baselines (e.g., scANVI)?

### Soundness
2

### Presentation
3

### Contribution
2
