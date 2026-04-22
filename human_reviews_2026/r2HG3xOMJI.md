# mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules

- Avg Score: 5.50
- Decision: Accept (Oral)
- Scores: 6, 6, 2, 8

## Abstract
Despite their ability to understand chemical knowledge, large language models (LLMs) remain limited in their capacity to propose novel molecules with desired functions (e.g., drug-like properties). In addition, the molecules that LLMs propose can often be challenging to make, and are almost never compatible with automated synthesis approaches. To better enable the discovery of functional small molecules, LLMs need to learn a new molecular language that is more effective in predicting properties and inherently synced with automated synthesis technology. Current molecule LLMs are limited by representing molecules based on atoms. In this paper, we argue that just like tokenizing texts into meaning-bearing (sub-)word tokens instead of characters, molecules should be tokenized at the level of functional building blocks, i.e., parts of molecules that bring unique functions and serve as effective building blocks for real-world automated laboratory synthesis. This motivates us to propose mCLM, a modular Chemical-Language Model that comprises a bilingual language model that understands both natural language descriptions of functions and molecular blocks. mCLM front-loads synthesizability considerations while improving the predicted functions of molecules in a principled manner. Experiments on 430 FDA-approved drugs showed that mCLM is capable of significantly improving chemical functions critical to determining drug potentials. mCLM, with only 3B parameters, also achieves improvements in synthetic accessibility relative to 7 other leading generative AI methods including GPT-5. When tested on 122 out-of-distribution medicines using only building blocks/tokens that are compatible with automated modular synthesis, mCLM outperforms all baselines in property scores and synthetic accessibility. mCLM can also reason on multiple functions and iteratively self-improve to rescue drug candidates that failed late in clinical trials (“fallen angels”).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes mCLM, a modular chemical-language model that helps the translation between natural language description and molecular blocks. Different from previous chemical LLMs using SMILES or SELFIES as their inputs, mCLM learns a new molecular language that is more effective in property prediction and better in synthesis. Experimental results show that with 3B parameters, mCLM can achieve much better synthetic accessibility and property scores than LLMs like GPT-5.

### Strengths
1. The idea of proposing a new molecular language and tokenizing molecules into substructures is novel and helps LLMs better focus on the structural patterns of molecules that will affect the molecular properties.
2. The performance of mCLM is significantly much better than previous baselines. With only 3B parameters, mCLM is also much more efficient than LLMs like GPT-5.
3. This paper is overall well-written and easy to follow.

### Weaknesses
1. The proposed molecular vocabulary lacks sufficient details regarding its implementation. Furthermore, it is worth investigating whether the decomposition of molecules into these substructures could potentially compromise their structural integrity.
2. The molecular tokenization in mCLM utilizes GNNs to generate embeddings for the building blocks. However, the study does not explore or compare this approach against more advanced or alternative methods, such as VQ-VAE, which could potentially offer superior performance.
3. The evaluation of mCLM is confined to the Qwen-3B base model. To assess the scalability and generalizability of the proposed method, it would be beneficial to test it across different models of varying sizes and architectures.
4.  The experimental section primarily benchmarks against other LLMs. While some of the chemical LLMs used as baselines may be outdated, the evaluation notably omits comparisons with established GNN baselines, which are crucial for a comprehensive assessment.
5. The proposed molecular property optimization task could be significantly strengthened by incorporating more relevant and diverse datasets into the evaluation.

### Questions
Please address my concerns in the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes mCLM, a framework designed to generate small molecules that are both functional and synthesizable. The model introduces a multimodal framework that combines a GNN based representation of molecular building blocks with natural language understanding of molecular functions. The tokenization is performed at the building block level rather than atom-level, aiming to align molecular representation with automated synthesis rules.

### Strengths
- The construction of an LLM framework that jointly considers synthesizability and functionality represents an important step toward practical and interpretable molecular generation.
- The integration of GNN representations with natural language embeddings for modular chemical reasoning is technically novel and well-motivated.
- The figures are clean, well-structured, and enhance the overall readability and understanding of the method.

### Weaknesses
I would consider raising the score if the following weaknesses are resolved.
- **Comparison to fragment-aware baselines**: While the paper includes comparisons to recent general-purpose and domain-specific molecule LLMs, it omits fragment- or group-aware baselines such as SAFE [1], GROUPSELFIES [2], or Reasyn [3]. Even acknowledging that Reasyn is concurrent, such comparisons (especially against Transformer-based models with other representations, as mCLM itself employs a Transformer backbone) would strengthen the evaluation and effectiveness.
- **Lack of clarity on reasoning knowledge acquisition**: The process for defining and annotating the molecular functions of building blocks (Figure 2) is insufficiently described. How are functional roles such as “Hinge binder, cell activity promoter” obtained or validated? If a user seeks to optimize a given property, does this require (1) manual annotation of new functions, (2) training of a proxy model, and (3) full re-training of the multimodal LLM? The pipeline for expanding functional knowledge is unclear for me.
- **Unclear link between function-infused vocabulary and molecular functionality**: Section 2 describes the vocabulary as *function-infused* and *synthesis-friendly*. While the synthesis aspect is well justified, the connection between the decomposed building blocks and their *functional meaning* remains ambiguous, even after reading Section 3.3. It is not evident how these building blocks encode or correlate with molecular functions, since molecular function typically arises from overall structure and context, not from isolated building blocks.
- **Minor correction**: “thanks to recent” in line 161 is aligned with nothing afterwards.

[1] Noutahi, E., et al. Gotta be SAFE: a new framework for molecular design. Digital Discovery, 3(4), 796-804.

[2] Cheng, A. H., et al. Group SELFIES: a robust fragment-based molecular string representation. Digital Discovery, 2(3), 748-758.

[3] Lee, S., et al. Rethinking Molecule Synthesizability with Chain-of-Reaction. arXiv 2025.

### Questions
- **Reason for free from function group conflicts**: The authors claim that resulting building blocks are *free from functional group conflicts* (lines 263–264). How is this ensured when building blocks from different tokenizers or different molecules are mixed? Wouldn’t incompatible functional groups potentially lead to synthesis failures?
- What could be the reason that the proposed mCLM show relatively weaker performance on HIA and PGP despite superior results on others?
- **Proxy model reliability**: Since experimental results rely heavily on proxy models for property prediction, how accurate and robust are these models across different molecular classes?
- **Ablation study**: Could the authors include an ablation comparing the full GNN-based encoding with a baseline using only textual (SMILES or SELFIES) representations? This would clarify the contribution of the graph modality.

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
The paper “mCLM: A Modular Chemical Language Model That Generates Functional and Makeable Molecules” introduces mCLM, a new type of chemical language model designed to generate small molecules that are both functionally effective and synthetically feasible. Traditional large language models can understand chemical information, but they often struggle to design molecules that can actually be synthesized in the lab. mCLM addresses this challenge by shifting from atom-level representations (like SMILES strings) to a modular representation, where molecules are described as combinations of chemically meaningful, synthesis-ready building blocks. This allows digital molecular generation to directly correspond to real-world automated synthesis.

mCLM functions as a bilingual model—it understands both natural language descriptions and molecular structures. It uses graph neural networks (GNNs) to encode molecular modules and combines these with text embeddings in a Transformer-based architecture, enabling it to “code-switch” between chemistry and natural language. The model is trained on paired datasets that link chemical properties, synthesis reactions, and textual descriptions. This training allows mCLM not only to generate molecules with desired biological or physical functions but also to ensure that these molecules are makeable through automated synthesis pipelines.

In experimental evaluations using 430 FDA-approved drugs and 122 out-of-distribution compounds, mCLM demonstrated substantial improvements in key pharmacological properties—including absorption, distribution, metabolism, excretion, and toxicity (ADMET)—while maintaining or improving synthetic accessibility. It outperformed leading AI systems such as GPT-5, Claude 3.5, and Gemini 2.5-Flash in both functional property optimization and synthetic feasibility. Furthermore, mCLM proved capable of “rescuing” failed drug candidates—like Evobrutinib and TNG348—by suggesting minimal structural modifications that reduced toxicity and improved drug-like behavior.

### Strengths
- Conceptually Innovative but Incremental in Execution
The idea of representing molecules through modular, synthesis-ready building blocks rather than atom-level encoding is conceptually novel and offers a creative bridge between digital design and physical synthesis. This modular approach reflects an original perspective on chemical language modeling. However, the implementation mainly extends existing ideas from reaction-aware and retrosynthesis-based models, making the innovation more incremental than transformative.

- Solid Technical Foundation but Limited Validation
The paper demonstrates technical competence in integrating graph neural networks with Transformer architectures and applying them to chemical structure–language fusion. The system design is coherent, and the methodology is explained at a reasonable level of detail. However, the experimental quality is weakened by the absence of real-world synthesis or bioassay validation, and the evaluation remains largely computational and self-referential, lowering the overall scientific robustness.

- Generally Well-Written but Overclaims Results
The manuscript is clear and logically structured, with well-organized figures and a coherent narrative linking modular chemistry to AI language modeling. Nevertheless, some claims—especially about outperforming large general-purpose models and enabling autonomous molecular discovery—are exaggerated relative to the presented evidence. The lack of sufficient methodological transparency (e.g., ablations, dataset details) also detracts from clarity and reproducibility.

- Potentially High Impact but Not Yet Realized
If validated experimentally, mCLM could have significant implications for automated drug discovery and robotic chemistry. The integration of function-aware reasoning and synthesis feasibility into a unified framework is a meaningful direction for the field. Yet, given the limited empirical support and narrow demonstration scope, its real-world significance remains largely aspirational rather than achieved.

### Weaknesses
- Limited Generalization and Chemical Creativity
The modular tokenization relies on a fixed library of known reaction building blocks and predefined synthesis rules. While this ensures synthetic feasibility, it severely restricts the model’s ability to explore novel chemical spaces or generate fundamentally new scaffolds beyond existing reaction types. Thus, the model’s creativity is constrained by human-curated chemistry knowledge.

- Lack of Experimental Validation and limited ablation and interpretability analysis.
The evaluation is almost entirely computational, based on predicted ADMET properties and synthesis scores rather than actual laboratory synthesis or biological testing. Without experimental confirmation, the model’s claimed functional and pharmacological improvements remain speculative and unverified in practice. It does not clearly isolate the contribution of its modular tokenization, GNN encoder, or reasoning loop. The internal decision-making of mCLM—how it balances function optimization and synthesizability—remains a black box, reducing its scientific transparency and reproducibility.

- Weak Comparative and Analytical Rigor
The baselines used (e.g., GPT-5, Claude 3.5, Gemini 2.5) are general-purpose models, making comparisons less meaningful. The paper omits direct evaluation against specialized molecular generative models (like ChemBERTa or retrosynthesis-aware VAEs), and lacks ablation studies to show the unique contribution of each module (tokenizer, GNN, reasoning loop).

- Overstated Multimodal Integration and Transparency Issues
The bilingual natural language–molecule framework is conceptually attractive but only superficially demonstrated. The model’s interpretability and internal reasoning are not clearly explained, leaving it as a black box with limited insight into how function and synthesizability are balanced.

### Questions
see weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents mCLM, a modular chemical language model designed for universal molecular understanding across textual, graphical, and structural modalities. The framework integrates multiple specialized modules—each trained on different molecular representations (e.g., SMILES, graphs, spectra)—and fuses them through a shared latent alignment layer. The authors emphasize scalability, interpretability, and extensibility, demonstrating competitive results on property prediction, reaction reasoning, and cross-modal translation benchmarks such as MolLangBench and MoleculeNet.

### Strengths
There is novelty in this work. Specifically, there is clear architectural separation between domain-specific encoders and a unified fusion backbone which promotes flexibility and domain transfer.

Experiment results are also promising. This work outperforms strong baselines (MolX, ChemBERTa, GraphMVP) on multimodal reasoning tasks, particularly in low-data and cross-domain settings.

Figure 4 is particularly useful as it shows module-wise attribution analyses for how modality-specific knowledge contributes to chemical reasoning.

### Weaknesses
There is however limited novelty at the core LLM level. While modularization is effective, the language model itself is adapted rather than fundamentally redesigned for chemistry.

There is also lack of validation of the practicality of this approach, say on real world sparse datasets. The evaluation focuses primarily on benchmark datasets, with minimal discussion of noisy experimental spectra or reaction data.

Further, the computational cost of the work seems infeasible. Training multiple modality-specific experts and fusion layers may hinder accessibility for smaller research groups.

There are also couple recent relevant research works that have not been referenced which the authors need to cite to improve the comprehensiveness of the related work section:

- Le, Khiem, Zhichun Guo, Kaiwen Dong, Xiaobao Huang, Bozhao Nan, Roshni Iyer, Xiangliang Zhang, Olaf Wiest, Wei Wang, and Nitesh V. Chawla. “MolX: Enhancing Large Language Models for Molecular Understanding With A Multi-Modal Extension.” Proceedings of the 2025 ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, AC M, 2025. 

- Ju, Jiaxin; Yizhen Zheng; Huan Yee Koh; Shirui Pan. “Uni-MRL: Unified MultiModal Molecular Representation Learning with Large Language Models and Graph Neural Networks.” Advances in Knowledge Discovery and Data Mining (PAKDD 2025), Lecture Notes in Computer Science, vol. 15874, Springer, 2025, pp. 275-287.

### Questions
How does mCLM handle conflicts between representations (e.g., inconsistent SMILES vs. graph encodings)?

Could the modular framework support plug-and-play extensions for new data types (e.g., protein–ligand complexes)?

How stable is the latent alignment layer during joint fine-tuning across highly imbalanced modalities?

### Soundness
4

### Presentation
3

### Contribution
3
