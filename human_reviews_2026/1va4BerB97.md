# ResearchArcade: Graph Interface for Academic Tasks

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Academic research generates diverse data sources, and as researchers increasingly use machine learning to assist research tasks, a crucial question arises: Can we build a unified data interface to support the development of machine learning models for various academic tasks? Models trained on such a unified interface can better support human researchers throughout the research process, eventually accelerating knowledge discovery. In this work, we introduce RESEARCHARCADE, a graph-based interface that connects multiple academic data sources, unifies task definitions, and supports a wide range of base models to address key academic challenges. RESEARCHARCADE utilizes a coherent multi-table format with graph structures to organize data from different sources, including academic corpora from ArXiv and peer reviews from OpenReview, while capturing information with multiple modalities, such as text, figures, and tables. RESEARCHARCADE also preserves temporal evolution at both the manuscript and community levels, supporting the study of paper revisions as well as broader research trends over time. Additionally, RESEARCHARCADE unifies diverse academic task definitions and supports various models with distinct input requirements. Our experiments across six academic tasks demonstrate that combining cross-source and multi-modal information enables a broader range of tasks, while incorporating graph structures consistently improves performance over baseline methods. This highlights the effectiveness of RESEARCHARCADE and its potential to advance research progress.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ResearchArcade, a unified graph interface that integrates heterogeneous academic data from ArXiv and OpenReview. It uses a graph neural network to aggregate multi-source information (papers, reviews, figures, etc.) into node embeddings, which are then combined with textual inputs and processed by a large language model (Qwen3) for various academic tasks. Experiments show that incorporating graph-structured embeddings leads to consistent improvements over text-only baselines.

### Strengths
1. Heterogeneous information integration. The paper explores an interesting and valuable idea, using heterogeneous academic information (papers, reviews, figures, tables, revisions) to assist language-model–based prediction and generation tasks, which is a meaningful direction.

2. Clear pipeline design. The integration of graph neural networks and large language models is implemented in a clear, modular way: the GNN aggregates multi-source context into embeddings that are passed to the LLM, demonstrating a solid and technically sound pipeline design.

3.  Comprehensive empirical evaluation. The framework is comprehensive and empirically validated, covering six academic tasks and showing consistent, though moderate, improvements over text-only baselines, suggesting contribution to the community as a shared research infrastructure.

### Weaknesses
1. Limited methodological innovation.
The proposed Graph World Model (GWM) simply concatenates GNN-derived embeddings with textual inputs to the LLM, without introducing new learning mechanisms or joint optimization strategies.

2. Incomplete reproducibility.
The graph construction process, particularly the alignment between ArXiv and OpenReview papers is under-specified. There are no quantitative statistics on entity-matching accuracy, no discussion of handling multiple paper versions, duplicate author names, or conflicting metadata, and no validation against human-curated samples.

3. Shallow empirical analysis and lack of ablation. 
While six tasks are included, the results are reported mainly as aggregate performance scores without in-depth ablation or interpretive analysis. The authors claim that incorporating graph structure improves performance, yet there is no investigation into which graph components or modalities (figures, tables, reviews) contribute to these gains.

4. Overstated “unified interface” claim.
Although positioned as a universal academic graph, each task is trained independently, and no cross-task transfer or shared representation is demonstrated. The claimed “unification” remains conceptual.

### Questions
Q1: How are the graph embeddings and text features integrated inside the LLM input? Are they concatenated at the embedding level or combined through an attention-based adapter?

Q2: How reliable is the mapping between ArXiv and OpenReview entities? Are there quantitative metrics validating the paper-to-paper alignment?

Q3: Does the framework support incremental updates as new papers and reviews are added, or is the graph static?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents ResearchArcade, a graph-based interface that connects ArXiv and OpenReview into a heterogeneous, temporally aware graph spanning text, figures, tables, reviews, and revisions. Tasks are defined in two steps: (1) pick a target entity and label, (2) retrieve the entity’s multi-hop neighborhood as input. 

Based on this graph, authors introduce six tasks: figure/table insertion, paragraph generation, revision retrieval, revision generation, acceptance prediction, and rebuttal generation.

### Strengths
- Compared to previous datasets, ResearchArcade covers multiple sources and modalities. It also provides a unified interface for tasks defined on academic graphs.
- The writing is clear and easy to follow.

### Weaknesses
- As an academic graph, the dataset’s coverage of papers is still limited. It includes around 45k papers from arXiv and about 28k from ICLR, which may restrict the generalizability of conclusions derived from it.  
- For the paragraph generation and revision generation tasks, the authors rely solely on semantic similarity metrics. However, such metrics may not capture aspects like clarity or appropriateness of the generated text (e.g., generated paragraphs and revisions). Using LLM-as-a-judge evaluations could better reflect quality.  
- The paper would benefit from a more detailed discussion on why the proposed tasks are important and why they are worth studying within an academic graph framework. For instance, prior work exists for some of these tasks, such as revision generation (e.g., https://aclanthology.org/2025.wraicogs-1.4/). I think connecting to previous literature would help readers understand what new advantages graph-based approaches offer.

### Questions
1. How are the embeddings for figure nodes generated?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ResearchArcade, a graph-based interface designed to unify multi-source, multi-modal, and temporally evolving academic data for supporting a wide range of machine learning tasks in academic research. It integrates data from ArXiv and OpenReview, including text, figures, and tables, and models academic activities as heterogeneous graphs.

Key contributions include:
Unified graph interface for academic data across sources and modalities;
Two-step task definition scheme (identify target entity + retrieve neighborhood) to standardize academic tasks;
Six benchmark tasks (e.g., paragraph generation, revision retrieval, acceptance prediction) and four promising future tasks (e.g., idea generation, review generation);
Empirical validation showing that graph-based models outperform non-graph baselines, and multi-modal inputs improve performance;
Open-source release of data, pipeline, and evaluation protocols.

### Strengths
S1: This paper addresses a real need for unified academic data interfaces to support ML models across diverse research tasks.

S2: This paper integrates ArXiv and OpenReview with text, figures, tables, and temporal evolution, offering a holistic view of academic knowledge.

S3: The two-step scheme (target entity + neighborhood) of this paper is simple yet general, enabling both predictive and generative tasks.

S4: This paper evaluates 6 tasks across 4 model types (EMB, GNN, LLM, GWM), showing consistent gains from graph structure and multi-modal inputs.

S5: This paper releases data, code, prompts, and splits, making the work usable and extensible for future research.

S6: This paper includes a thoughtful ethics statement and reproducibility checklist, which is rare and commendable.

### Weaknesses
W1: The graph construction and task definition are engineering-heavy; no new modeling techniques or architectures are proposed.

W2: Tasks of this paper, like figure insertion or paragraph generation are reconstruction-style and may not reflect real-world academic needs (e.g., idea quality, scientific discovery).

W3: Best accuracy is only 0.55, barely above random — raises questions about whether the graph is rich enough for high-level reasoning.

W4: All metrics are automatic (SBERT, BLEU, etc.); no expert judgment on scientific validity, coherence, or usefulness of generated content.

W5: Data is CS-heavy (ArXiv + ICLR); no evaluation on how well the interface generalizes to other fields (e.g., biology, physics).

W6: This paper does not compare with larger SOTA LLMs (e.g., GPT-4, Claude, Qwen3-70B) or retrieve-augmented systems — limits external validity.

### Questions
Q1: How would ResearchArcade perform on open-ended scientific tasks like hypothesis generation, experiment design, or novel discovery?

Q2: Can the graph structure support more complex reasoning, such as causal chains, contradiction detection, or theory synthesis?

Q3: What is the human-perceived quality of generated paragraphs, rebuttals, or reviews? Are they scientifically sound or just fluent?

Q4: How scalable is the pipeline to larger corpora (e.g., PubMed, Semantic Scholar) or real-time updates (e.g., daily ArXiv dumps)?

Q5: Why does acceptance prediction fail even with rich graph data? Is it inherently unpredictable, or is the signal too weak?

Q6: How does ResearchArcade compare to retrieval-augmented LLMs or tool-augmented systems (e.g., ChatPDF, Elicit) on real user tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This submission proposes RESEARCHARCADE, a graph-based interface that integrates multi-source data—computer science papers from ArXiv and peer reviews/submissions from OpenReview—and multi-modal information (text, figures, and tables) to support the development of machine learning models for diverse academic tasks. It adopts a two-step scheme—“identify target entity and retrieve neighborhood”— to unify the definitions of six academic tasks(e.g., Figure/Table Insertion, Paragraph Generation) and is compatible with various models, including embedding-based models (EMB, e.g., Longformer), graph neural networks (GNN, e.g., HANConv). However, critical resources are unavailable in this submission: no accessible code, related dataset samples, or key preprocessing details are provided. The authors only commit to releasing these materials upon publication. Furthermore, this work functions more like a data preprocessing pipeline than an innovative contribution—no novel technical approaches are proposed.

### Strengths
S1. This submission accurately identifies key pain points in academic AI. From a data perspective, it targets the complexity of academic data, sourced from diverse platforms (ArXiv’s computer science papers, OpenReview’s ICLR submissions) and spanning multiple modalities (text, figures, tables). From a task perspective, it decreases the unnecessary effort required for data. By focusing on these gaps, the work directly responds to the demand for a unified data interface, as proposed in its core design.

S2. The heterogeneous graph structure effectively encompasses diverse entities and their relationships. Entities include not only papers, reviews, figures, and tables but also ArXiv paragraphs and OpenReview revisions. Key relationships are explicitly modeled: e.g., "paper-paragraph" and "paper-figure/table". 

S3. The framework demonstrates good adaptability to different model types. It supports embedding-based models, graph neural networks, large language models, and the Graph World Model. This compatibility enables it to handle both predictive tasks and generative tasks.

### Weaknesses
W1. Data used in the experiment is restricted to the computer science (CS) field (ArXiv data) and ICLR conferences (OpenReview data). However, no testing was conducted in other domains such as biology, chemistry or materials science.

W2. Critical preprocessing steps are not explained. For example, how to extract paragraphs and figures from ArXiv’s LaTeX and how to align OpenReview reviews with paper paragraphs—these details remain untackled.

W3. Novel academic-related contributions are limited, such as mechanisms for rapidly updating the graph when new paper revisions are added.

### Questions
1. Do you plan to conduct cross-domain validation in some non-CS fields? 

2. Could you add academic-oriented optimizations (e.g., methods to resolve ambiguity in OpenReview review comments)?

### Soundness
2

### Presentation
3

### Contribution
2
