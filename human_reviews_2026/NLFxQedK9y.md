# ChronoPlay: A Framework for Modeling Dual Dynamics and Authenticity in Game RAG Benchmarks

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Retrieval Augmented Generation (RAG) systems are increasingly vital in dynamic domains like online gaming, yet the lack of a dedicated benchmark has impeded standardized evaluation in this area. The core difficulty lies in Dual Dynamics: the constant interplay between game content updates and the shifting focus of the player community. Furthermore, the necessity of automating such a benchmark introduces a critical requirement for player-centric authenticity to ensure generated questions are realistic. To address this integrated challenge, we introduce ChronoPlay, a novel framework for the automated and continuous generation of game RAG benchmarks. ChronoPlay utilizes a dual-dynamic update mechanism to track both forms of change, and a dual-source synthesis engine that draws from official sources and player community to ensure both factual correctness and authentic query patterns. We instantiate our framework on three distinct games to create the first dynamic RAG benchmark for the gaming domain, offering new insights into model performance under these complex and realistic conditions. Our code is available at: https://github.com/hly1998/ChronoPlay.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focus on the Retrieval Augmented Generation (RAG) problem in dynamic domains and build a RAG benchmark in the domain of online gaming. The authors propose ChronoPlay, a novel framework for the automated and continuous generation of game RAG benchmarks. Specially, this work introduce dual-dynamic update mechanism that responds to changes in both the game’s knowledge and the player’s interests. The experiments across three games demonstrate that RAG system performance is highly volatile over a game’s lifecycle.

### Strengths
1. This paper studies the RAG problems in dynamic domains and build a benchmark in this area with an automated and continuous generation framework. The research topic is interesting and valuable. 

2. The data sourced from real games and player communities making the benchmark more realistic.

### Weaknesses
1. Many real-world applications require dynamic RAG systems, including online shopping (where prices and promotional campaigns constantly change) and travel planning (where weather conditions and seasonal attractions vary). However, the benchmark's coverage of only three games limits its scope and makes the data domain insufficiently diverse.

2. The generation performance in this work is evaluated using LLM-as-Judge, as detailed in Appendix C. However, the meta-evaluation results (Section C.3) are concerning. With accuracy ranging from 70% to 78% and F1-scores from 78% to 86%, the reliability of the evaluation methodology itself is questionable, which undermines confidence in the reported results.

3. Additional experiments on end-to-end RAG systems would provide more comprehensive insights.

### Questions
Miss related works on other RAG evaluation frameworks, for example, RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation by Ru et al., 2024.

### Soundness
2

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
2

### Summary
This paper introduces a framework for automatically generating dynamic RAG benchmarks for the gaming domain. The key innovation is addressing "Dual Dynamics"—the simultaneous evolution of game content and player community interests. The framework combines a dual-source synthesis engine that draws from both authoritative game wikis/patch notes and player community discussions to ensure factual correctness and authentic query patterns. The authors instantiate the framework  on three games spanning different timescales and characteristics. Evaluation of various retrieval models (BM25, BGE-M3, Qwen3-Embedding, text-embedding-3) and generator models (GPT-4o, Claude, Gemini, etc.) reveals significant performance fluctuations across game lifecycle phases, demonstrating that both knowledge updates and interest drift independently contribute to benchmark volatility.

### Strengths
- The concept of "Dual Dynamics" is a powerful contribution. The paper addresses a limitation of existing dynamic benchmarks and provides a more realistic evaluation paradigm. 
- The framework this paper proposes is well-designed.. Human expert evaluation and validation of LLM-as-judge against human experts provide credibility. It is also instantiated on multiple games which shows diversity.

### Weaknesses
-  The ChronoPlay pipeline involves multiple LLM-driven stages and several hyperparameters (λ_JSD=0.001, γ=1.5, varying window sizes W). This complexity might pose barriers to easy adoption.
- The statistical rigor of the evaluation could be strengthened. The paper lacks confidence intervals or significance tests for performance differences. Are the fluctuations statistically significant?

### Questions
While the authors demonstrate that Dual Dynamics are a major driver of performance volatility, the analysis could be strengthened by de-confounding topic shifts from inherent question difficulty. For instance, analyzing the distribution of question types (e.g., single-hop vs. multi-hop) across phases could provide a more direct measure of difficulty.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ChronoPlay, a framework for constructing dynamic retrieval-augmented generation (RAG) benchmarks in gaming domains. It models two evolving dimensions: knowledge evolution (continuous updates to game rules and content) and user interest drift (shifts in player community focus over time). The framework integrates an authoritative knowledge base, community-derived question templates, and synthetic player personas to generate realistic, time-evolving question–answer pairs.
 
The authors test various retrievers and generators on datasets created with ChronoPlay for three games — Dying Light 2, Dune: Awakening, and PUBG Mobile — and provide detailed ablation studies.

### Strengths
1. ChronoPlay is the first framework to incorporate dynamic, evolving environments into RAG evaluation for gaming. By formalizing knowledge evolution and user interest drift, it highlights two important factors often overlooked in static RAG benchmarks.

2. The experiments are extensive, covering several retrievers and generators across three games and including ablations on knowledge and interest dynamics.

3. The proposed datasets and methodology could help the community study how RAG systems behave under temporal and contextual changes.

### Weaknesses
1. While the paper includes a diverse set of retrievers and generators, the RAG experiments themselves are static. It's unclear whether the RAG system re-index as the benchmark evolves. There is no adaptive RAG system (e.g. with finetuning, memory) discussed. It's unclear how the proposed dynamic benchmark would challenge or benefit adaptive RAG systems in practice.

2. The paper also omits details about how the vector databases are indexed or re-indexed after each phase. It is unclear whether embeddings are refreshed, incrementally updated, or kept static.

3. ChronoPlay models a population-level community through generated player personas, but it doesn’t capture personalized or history-dependent questions. I don’t see this as a major weakness, but it does make me wonder how this limitation relates to the idea of dynamic RAG. What would a RAG system need to do to account for personalized shifts in interest over time?

### Questions
1. How are vector databases indexed or re-indexed after each phase update?
2. Could ChronoPlay be extended to simulate user-specific histories or personalized question evolution, beyond aggregate community changes?

### Soundness
3

### Presentation
4

### Contribution
3
