# L^2M^3OF: A Large Language Multimodal Model for Metal-Organic Frameworks

- Decision: Reject
- Scores: 8, 6, 2, 2, 4

## Abstract
Large language models (LLMs) have demonstrated remarkable reasoning capabilities across diverse natural language tasks. However, comparable breakthroughs in scientific discovery are more limited, because understanding complex physical phenomena demands multifaceted representations far beyond language alone. A compelling example is the design of functional materials such as metal-organic frameworks (MOFs) — critical for a range of impactful applications like carbon capture and hydrogen storage. Navigating their vast and intricate design space in language-based representations interpretable by LLMs is challenging due to the numerous possible three-dimensional atomic arrangements and strict reticular rules of coordination geometry and topology. Despite promising early results in LLM-assisted discovery for simpler materials systems, MOF design remains heavily reliant on tacit human expertise rarely codified in textual information alone. To overcome this barrier, we introduce L^2M^3OF, the first multimodal LLM for MOFs. L^2M^3OF integrates crystal representation learning with language understanding to process structural, textual, and knowledge modalities jointly. L^2M^3OF employs a pre-trained crystal encoder with a lightweight projection layer to compress structural information into a token space, enabling efficient alignment with language instructions. To facilitate training and evaluation, we curate a structure–property–knowledge database of crystalline materials and benchmark L^2M^3OF against state-of-the-art (SOTA) closed-source LLMs such as GPT-5, Gemini-2.5-Pro, and DeepSeek-R1. Experiments show that L^2M^3OF outperforms leading text-based closed-source LLMs in property prediction and knowledge generation tasks, despite using far fewer parameters. These results highlight the importance of multimodal approaches for porous crystalline material understanding and establish L^2M^3OF as a foundation for next-generation AI systems in materials discovery.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes $L^2M^3OF$, the first multimodal LLM for MOFs, which integrates crystal representation learning with language understanding to process structural, textual, and knowledge modalities jointly. The model uses a pre-trained crystal encoder with a lightweight projection layer to efficiently align with language instructions. Benchmarked on a structure–property–knowledge database of crystalline materials curated by the authors, $L^2M^3OF$ outperforms leading text-based closed-source LLMs in property prediction and
knowledge generation tasks, despite using far fewer parameters. Such results highlight the importance of multimodal approaches for porous crystalline material understanding.

### Strengths
1. The authors curated a structure–property–knowledge database for MOFs, namely MOF-SPK, and consequently designed structured subtasks that collectively cover both quantitative and qualitative reasoning about materials, thus providing a comprehensive benchmark for evaluating the capability of LLMs for porous crystalline material understanding.

1. A two-stage MLP bridge maps these latent vectors into the token space of Qwen-2.5, enabling seamless cross-modal attention.

1. The authors freeze PMTransformer weights to retain structural priors while fine-tuning only the bridge and LLM, avoiding catastrophic forgetting. They further propose “group training”, a lightweight conversational data-augmentation scheme that concatenates multiple Q&A pairs in one batch, improving contextual diversity without increasing token count.

### Weaknesses
1. The evaluation uses GPT-4-mini as an automatic judge in description generation and Q&A tasks. While this provides consistency, it may introduce model bias—the evaluation may favor certain phrasing styles or reasoning formats aligned with GPT-family models. Human expert evaluation or hybrid methods would strengthen validity.

1. In description generation and Q&A tasks, the performance gap versus Gemini-2.5-Pro is small, which suggests that the overall advantage of $L^2M^3OF$ remains modest compared to large-scale commercial models. 

1. The paper does not report inference or fine-tuning cost. The $L^2M^3OF$’s multimodal architecture (PMTransformer + bridge + LLM) likely incurs higher computational overhead than text-only baselines (like $L^2M^2OF$).

### Questions
1. Have the authors experimented with training the model with LLM frozen? I suppose this would isolate the contribution of SFT on the LLM, and also tells us how much gain comes from multimodal alignment alone vs. from updating the language model. Especially, the work is compared with baselines of closed-source LLMs whose parameters can't be fine-tuned.

1. There are multiple existing works that train crystal GNNs on MOF data. Therefore, is there any specific reason to justify the choice of PMTransformer as the structure encoder? Also, how large is the impact of the model architecture choice on the performance of $L^2M^3OF$?

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
4

### Summary
This paper introduces MOF-SPK, a novel curated dataset comprising structural, property, and domain-knowledge information for over 100,000 Metal–Organic Framework (MOF) structures. Building on this dataset, the authors propose two models: L²M²OF (a text-only model) and L²M³OF (a multimodal text–structure model). The authors demonstrate that both models outperform generalist large language models (LLMs) on tasks defined within the MOF-SPK benchmark.

### Strengths
The paper makes a significant contribution to the field generation through the introduction of the MOF-SPK dataset, which unifies multiple material property prediction, description, and application tasks into a single, well-curated resource. Such an integrated dataset has strong potential to support the development and benchmarking of domain-specific foundation models for materials science.

Another strength lies in the dual-model design: the authors present both a text-only (L²M²OF) and a multimodal (L²M³OF) variant, allowing an insightful comparison of modalities. The comprehensive evaluation against several general-purpose LLMs further highlights the benefits of domain specialization.

The manuscript is well-organized, clearly written, and easy to follow, with a logical flow from dataset construction to model development and evaluation.

### Weaknesses
Despite the paper’s strong concept, the experimental design raises some concerns, particularly regarding baseline selection.

[Major] The comparison is limited to generalist language models (DeepSeek, GPT, Gemini), which may not provide a fair assessment of the proposed models’ performance. Numerous specialized neural architectures [a, b, c, d] for material property prediction and generation have been developed in recent years, and including at least a subset of these as specialist baselines would significantly strengthen the paper’s empirical claims.

[Minor] The description of the experimental and data processing pipelines could be more transparent. In particular, the prompts used for DeepSeek-R1 to extract material-relevant information from scientific publications are not provided, also it is unclear whether the generalist models were evaluated in zero-shot, one-shot, or few-shot settings. While the authors mention plans to release the code and processed data upon acceptance, a high-level overview of the data annotation pipeline and generalist model evaluation methodology should be included in the Supplementary Materials. This would improve reproducibility and accessibility, especially for readers who may not inspect the released code and data in detail.

a.Language models can generate molecules, materials, and protein binding sites directly in three dimensions as XYZ, CIF, and PDB files, Flam-Shepherd et al.

b.MOFDiff: Coarse-grained Diffusion for Metal-Organic Framework Design, Fu et al.

c.MOFGPT: Generative Design of Metal-Organic Frameworks using Language Models, Badrinarayanan et al.

d.Multi-modal conditional diffusion model using signed distance functions for metal-organic frameworks generation, Park et al.

### Questions
In addition to the points mentioned above, I have the following specific questions:

The paper states that PMTransformer is used as the structural encoder, but it remains unclear what textual backbone architecture is used for the L²M²OF and L²M³OF models. Please clarify the choice of language backbone and its adaptation to the multimodal setting.

The process of extracting SMILES representations via MOFid requires further clarification. In Figure 7 (Supplementary Materials), some examples show SMILES with disjoint fragments, which appears unintuitive. Could the authors provide additional statistics or qualitative analysis on the rate and correctness of recovered SMILES, particularly cases with multiple disjoint fragments?

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
4

### Summary
This paper proposes L²M³OF, a multimodal LLM for MOF property prediction and analysis, along with L²M²OF, a text-only variant. The models are evaluated on geometric property prediction, structure extraction, description generation, and question answering using a newly curated MOF-SPK dataset.

### Strengths
- The approach of applying language models to MOF analysis is reasonable and timely (but has been done before)
- The paper develops both multimodal and text-only variants for comparison

### Weaknesses
**1. Poor absolute performance on basic properties:** The model's performance on fundamental geometric properties is concerning. For accessible surface area, the MAE exceeds 250 m²/g (L²M³OF) and approaches 500 m²/g (L²M²OF). These errors are substantial and raise serious questions about practical utility. The model appears good only because of the baselines chosen.

**2. Inappropriate baseline selection:** The critical flaw is comparing exclusively against general-purpose LLMs (GPT-5, Gemini, DeepSeek) in zero-shot settings, which are obviously unsuited for this specialized task. To demonstrate meaningful impact, the authors must compare against:
- State-of-the-art domain-specific models for MOF property prediction (MOFTransformer, graph neural networks, based on optimized features RACs/MOFDescribe)
- For the recommendation task, recent work from domain experts (e.g., Bernd Smit's group on MOF recommendations https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00116h)

**3. Limited property scope:** The evaluation focuses on relatively simple geometric properties (surface area, pore volume) rather than practically important properties like gas uptake capacity, selectivity, or other performance metrics relevant to the applications (carbon capture, hydrogen storage) claimed in the introduction.

**4. Minor nomenclature issue:** "L2M3” is already an established conference name in this field—suboptimal naming choice.

**5. CIF Performance in linker extraction** The advantage for CIF as input might be because CIFs are often sorted (metals first) which could simplify the task.

### Questions
1. **Why were no domain-specific MOF models included as baselines?** The paper cites MOFTransformer, DeepSorption, and other specialized models in the related work, yet only compares against general-purpose LLMs. Can you provide direct performance comparisons against these established MOF-specific methods?

2. **What is the practical significance of these error levels?** An MAE of 253.7 m²/g for surface area or 0.55 Å for PLD—are these acceptable for real MOF design workflows? What are typical ranges for these properties, and how do these errors compare to experimental measurement uncertainty?

3. **Can the model predict functionally important properties?** The evaluation focuses on geometric descriptors, but applications like carbon capture and hydrogen storage depend on gas uptake capacity, selectivity, and stability. Can you demonstrate performance on these practically relevant properties?

4. **How does performance compare to Bernd Smit's group's work on MOF recommendations?** You mention this as missing related work—can you include this comparison for the recommendation task?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new model for the analysis of metal organic framework materials. The authors make several contributions in this work. First, they introduce a new dataset that they call MOF-SPK or "metal organic framework structure property knowledge". The dataset takes 133,000 materials from Cambridge Crystallographic Data Centre, and performs preprocessing such as deduplication. The materials are then labeled with computed properties such as density, pore limiting diameter and accessible surface area, as well as higher level properties such as the structure and information about applications characterization methods and stability extracted from a corpus of publications on the materials using DeepSeek-R1. The authors use this dataset to fine tune a large language model for predicting the above mentioned properties and attributes, with the input being a concatenation of a CIF-file representation of a given MOF with a prompt for a specific query. This basic purely-LLM based model is then extended by using a crystal encoder network (PMTransformer) to extract a representation of the crystal, another "bridge network" is used to transform this representation into tokens to use in place of the CIF file for the LLM input. The bridge network and LLM are fine-tuned together. In their evaluations, the authors compare their model to a range of state of the art LLMs for property-prediction, structure prediction and description tasks.

### Strengths
**Goals and novelty**
- The paper tackles an interesting and potentially important task of serving as an AI assistant for researchers investigating MOFs
- The approach of using a structure encoder to extract material representations combined with a natural language LLM for parsing questions is interesting and to my knowledge novel.

**Dataset**
- The dataset curated by the authors is potentially useful for other researchers in this area


**Writing**
- The writing of the paper is generally clear and the figures are generally clear and useful
- The related works are very up-to-date with recent developments in the field, helping put this work clearly into context.

### Weaknesses
**Novelty**
- The L2M2OF model is just a fine-tuned LLM, which is not particularly novel from a machine learning perspective
- The L2M3OF model underperforms L2M2OF across many tasks and its unclear if it's actually better on many of the higher level tasks.

**Evaluation**
- The only baseline are non-fine tuned LLMs
- The description and Q&A tasks are evaluated using another non-fine-tuned LLM as the judge, which is potentially biased and inaccurate as a metric.
- The ground truth labels for these tasks were also extracted with an LLM. That none of these steps are verified by domain experts is quite concerning
- The judgements compared to Gemini are quite close and it's not clear if they are statistically significant

**Justification**
- My understanding is that properties like LCD, PLD, density and SMILES representation can be directly and reasonably efficiently computed from the original CIF file representation (this is how the authors computed the ground truth labels). It unclear why it is useful to have an LLM produce potentially error prone predictions 

**Missing details**
- Figure 1 suggests the model is using LORA-based fine-tuning, but this is not discussed anywhere in the text. 
- No discussion of the computational costs, particularly in comparison to explicit property computation.

### Questions
Line 322: Why does the total number of tokens remain unchanged if more questions are being used in the input? Is it just that the crystal representation is shared across all of the questions?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces L2M3OF, a multimodal large language model for metal–organic frameworks (MOFs). It combines structural, textual, and domain-knowledge modalities to enable reasoning over 3D crystalline materials—an area where text-only models fail to capture reticular geometry and symmetry. The proposed model integrates a crystal encoder (PMTransformer) with a Qwen2.5-based LLM, linked by a projection and compression bridge that transforms geometric embeddings into token-level representations. Training is performed on MOF-SPK, a newly curated structure–property–knowledge dataset of 133,737 experimentally reported MOFs with associated literature.

### Strengths
The strengths of this work are as follows below: 

Addresses a neglected modality gap, integrating structure and language for reticular materials.

Substantial new dataset (MOF-SPK) of >133k entries with curated properties and literature links.

Systematic evaluation across four tasks with both open and closed LLMs.

Methodologically clean hybrid (frozen encoder + lightweight bridge).

Joint-training ablation (Table 3) convincingly shows cross-task synergy.

### Weaknesses
The weaknesses of this work are as follows below: 

There are reproducibility issues because dataset and code are not released; MOF-SPK curation process is only briefly described.

There is evaluation bias in comparing to closed models (GPT-5, Gemini) without uniform prompt design or temperature settings limits validity.

Because the projection-bridge multimodal alignment is standard, there is little architectural innovation beyond dataset scale.

The writing of the work can also be improved to more clearly and concisely deliver ideas. 

Further, there are also critical references missing from this work. These include the below:

- Le, Khiem, Zhichun Guo, Kaiwen Dong, Xiaobao Huang, Bozhao Nan, Roshni Iyer, Xiangliang Zhang, Olaf Wiest, Wei Wang, Ting Hua, and Nitesh V. Chawla. MolX: Enhancing Large Language Models for Molecular Understanding with a Multi-Modal Extension. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (MLoG-GenAI@KDD ’25), ACM, 2025.

- Guo, Zhichun, Kehan Guo, Bozhao Nan, Yijun Tian, Roshni G. Iyer, Yihong Ma, Olaf Wiest, Xiangliang Zhang, Wei Wang, Chuxu Zhang, and Nitesh V. Chawla. “Graph-based Molecular Representation Learning.” Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2023, pp. 6638-6646.

- Zeng, Zequn, et al. “MolXPT: Wrapping Molecules with Text for Generative Pre-Training.” ACL Short Papers, 2023.

- Soares, Eduardo A., et al. “An Open-Source Family of Large Encoder–Decoder Foundation Models for Chemistry.” Communications Chemistry, 8 (1), 2025.

- Kang, Yeonghun, et al. “A Multi-Modal Pre-Training Transformer for Universal Transfer Learning in Metal–Organic Frameworks.” Nature Machine Intelligence, 5 (3), 2023.

- Badrinarayanan, Srivathsan, et al. “MOFGPT: Generative Design of Metal–Organic Frameworks Using Language Models.” J. Chem. Inf. Model., 65 (17), 2025.

### Questions
How was dataset overlap between CCDC-derived structures and MOF-SPK evaluation sets prevented?

Can you quantify the data/computation cost of training vs GPT-5 baseline in FLOPs or GPU hours?

Did you test whether the frozen PMTransformer encoder limits adaptation to novel topologies?

How sensitive are results to projection size (M tokens)?

What safeguards exist to avoid hallucinations in material property predictions?

Can the model handle non-MOF crystalline systems (e.g., COFs or zeolites)?

How reproducible are your GPT-judge evaluations in description generation?

### Soundness
2

### Presentation
2

### Contribution
3
