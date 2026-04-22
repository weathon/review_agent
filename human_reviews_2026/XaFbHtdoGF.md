# GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 4, 4, 6, 4

## Abstract
efficient AI-for-Science discovery.
Geospatial modeling provides critical solutions for pressing global challenges such as sustainability and climate change. Existing large language model (LLM)–based algorithm discovery frameworks, such as AlphaEvolve, excel at generic code evolution but lack the domain knowledge required for complex geospatial problems. We introduce GeoEvolve, a multi-agent LLM framework that couples evolutionary search with dynamic geospatial domain knowledge. GeoEvolve operates in nested loops: an inner code evolver generates candidate solutions, while an outer agentic controller—supported by Automated Knowledge Construction and Code-to-Formula agents—queries a Dynamic GeoKnowRAG module to inject theoretical priors. This architecture addresses the challenges of spatial heterogeneity and temporal non-stationarity. We evaluate GeoEvolve on three classical tasks: spatial interpolation (Kriging), uncertainty quantification (GeoCP), and spatial regression (GWR). Across 9 datasets, GeoEvolve discovers novel algorithms that incorporate geospatial theory. It achieves significant gains, such as a 29.5\% increase in regression $R^2$ and a 13–21\% reduction in interpolation error. Furthermore, extensive ablation studies confirm GeoEvolve’s robustness across diverse foundation models (GPT, Gemini, Qwen) and its spatiotemporal generalizability, validating that domain-guided retrieval is essential for stable evolution.
Collectively, these results offer a scalable path toward trustworthy, automated geospatial modeling, opening new avenues for efficient AI-for-Science discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper combines AlphaEvolve/Openevolve with two geospatial reasoning tasks. The authors use openevolve in the  inner loop carrying out code mutations, and have an outer loop querying a geospatial knowledge base. There are two tasks that are evaluated: one in kriging/ spatial interpolation  and the second in spatial uncertainty quantification,. In each case the authors reduceerrors in both by O(10-20%) above their chosen baseline.  They conduct ablation experiments comparing their pipeline to one with a geospatial RAG. For the RAG they set up a  structured knowledge base by collecting literature on core geospatial modeling concepts from Wikipedia, arXiv, and GitHub

### Strengths
This is very timely and the authors clearly show on the tasks that they evaluate how openevolve with a geospatial RAG system can lead to superior performance on two tasks.

### Weaknesses
--The authors only choose two tasks, and one example of a problem from each task. 
--It is not clear how strong the baselines that are chosen are and thus whether the system being shown will add actual value to geospatial researchers.
--It is not clear why the RAG system improves performance. What does it retrieve that helps the evolutionary system do better? To what extent is this guaranteed by cherry picking the documents/etc in the RAG index? How problem specific is this index?

These weaknesses make it hard to assess the generality or importance of these conclusions. Especially when compared with openevolve itself--the claimed benefit here is on the system around openevolve but with so few questions and evaluations it is hard to tell what the generality of the value is.

### Questions
- did you choose the documents in the RAG database to correspond to the problems you were trying to solve?
--Was the backbone of the RAG system chosen to correspond to the two narrowly scoped problems you are evaluating? Did you do anything to make the RAG better by including particular tools/articles/etc
--why did the RAG enhanced system perform better. Please be specific in the context of the examples chosen.
--what are the solutions the system found? Why are they better? What were the specific innovations discovered by geoevolve?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents GeoEvolve, a multi-agent LLM framework designed to evolve codes by integrating geospatial domain knowledge. The framework evolves codes using an outer loop that generates and evaluates codes based on geospatial reasoning to decide whether geospatial knowledge is considered, and an inner loop that introduces a GeoKnowRAG step to apply geospatial knowledge from published papers, Wikipedia, and github. It evaluates GeoEvolve on two representative tasks—spatial interpolation and spatial uncertainty quantification.

### Strengths
1: Using geospatial knowledge to improve LLM reasoning and code generation is both interesting and an important direction.

2: Two geospatial tasks are used for evaluation. The ablation study also shows the improvements from different components.

3: The paper is easy to follow and written in a clear way.

### Weaknesses
1: The overall novelty is limited. While the integration of geospatial RAG into a multi-agent code-evolution framework is interesting, the framework feels like applying existing methods to a geospatial problem. The proposed method uses existing code-evolution framework with a domain-specific RAG component. The main contribution from what I read is to use geospatial papers and documents in RAG, which is logical for geospatial tasks but straightforward in applications from my perspective. Practitioners in different domains will reasonably apply RAG in similar ways and do filtering for their domain problems. It does not appear to change the core of the existing methods so the same can be rebranded to make it BioEvolve, HealthEvolve, so on and so forth. Adding more explanations could help readers understand the core contributions better.

2: The related work focuses on algorithm discovery but can expand more to other agents.

3: The baseline used are not well justified. The validation is limited to a comparison between GeoEvolve and OpenEvolve, which is not interesting enough. It would strengthen the evaluation to include established SOTA baselines for each task to allow a more comprehensive assessment of this work's value. 

4: The paper does not address transferability or generalizability of the evolved algorithms.

### Questions
See requests for clarifications in weaknesses.

### Soundness
3

### Presentation
3

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
This paper proposes GeoEvolve, a multi-agent framework that uses large language models (LLMs) in a nested evolutionary architecture to automate the discovery of geospatial models. The key novelty lies in combining code evolution (via OpenEvolve) with structured retrieval-augmented geospatial knowledge (GeoKnowRAG) to improve geospatial algorithm generation. The authors target two classical tasks: spatial interpolation (ordinary kriging) and spatial uncertainty quantification (GeoCP). Empirical results show RMSE reductions of up to 21% and interval score improvements of 17%. The system includes several modular components: code evolver, evolved code analyzer, geospatial knowledge retriever, and prompt generator. Ablation studies test the contribution of each.

### Strengths
* The framework adapts the LLM-based algorithm evolution to geospatial modeling, which is underexplored.
* The use of a curated and semantically indexed geospatial knowledge base (GeoKnowRAG) is well motivated
* The choice of two canonical geospatial tasks (kriging and GeoCP) is appropriate.
* The code analyzer module adds interpretability and aligns well with domain-specific debugging.

### Weaknesses
* Tables 1 and 2 report only mean values for each run. No standard deviation or confidence intervals are provided.
* The paper only compares against OpenEvolve variants. It omits stronger geospatial modeling baselines. This leaves unclear whether the improvement is from LLM evolution or domain knowledge injection.
* Details such as the LLM model version, token budget, temperature, search strategy, and retriever evaluation metrics are absent.
* There is no direct comparison between GeoKnowRAG and standard RAG baselines.
* Appendix A.3 says the knowledge base is built from 5 hand-designed keyword classes. This is not scalable or reproducible, and its quality likely impacts the results significantly.
* Many modern geospatial tasks use raster or imagery data (e.g., remote sensing, flood detection, land cover). GeoEvolve only supports code generation over text-based model specifications. 
* All evaluations are at the algorithmic level. There is no demonstration that the improved Kriging or GeoCP models lead to better downstream outcomes.

### Questions
* Can you report standard deviations or confidence intervals for RMSE and interval score?
* Could you clarify the specific LLM configuration used in the code evolver and prompt generator?
* Have you compared GeoKnowRAG against simpler RAG baselines?
* Can GeoEvolve be extended to multi-modal geospatial inputs such as remote sensing imagery or spatiotemporal satellite time series?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces GeoEvolve, a multi-agent LLM framework that integrates evolutionary code search (via OpenEvolve) with a structured geospatial knowledge base to automate the discovery and refinement of geospatial algorithms. It demonstrates performance improvements on two classical tasks: spatial interpolation (Ordinary Kriging) and spatial uncertainty quantification (GeoCP).

### Strengths
Effectively couples evolutionary search with domain-specific RAG, moving beyond generic code generation.

Demonstrates clear performance gains (13-21% RMSE reduction, 17% interval score improvement) over baselines.

### Weaknesses
Evaluation is confined to two specific tasks. The claim of a "scalable pipeline" for diverse geospatial tasks is not fully substantiated. Lack a comprehensive discussion and comparison with existing related methods on LLM-driven design.

The process for building the GeoKnowRAG knowledge base appears ad-hoc and potentially biased, relying on author-curated keywords from limited sources.

### Questions
Only two specific tasks are tested. The claim of a "scalable pipeline" for diverse geospatial tasks is not fully substantiated.

The method is mainly compared with openevolve. A discussion and comparison with highly related works on combining evolutionary search and LLMs for automated design, such as [1], and recent advancements on LLM for geospatial tasks [2] can enhance the contribution and impact of the paper.

[1] Evolution of heuristics: Towards efficient automatic algorithm design using large language model, 2024
[2] An autonomous GIS agent framework for geospatial data retrieval, 2025

How was the specific set of 141 knowledge documents for GeoKnowRAG selected and validated for completeness and lack of bias?

The prompt used for "OpenEvolve with GeoKnowledge" is vague. Was a more specific, knowledge-grounded prompt tested to ensure a fair comparison?

What is the computational overhead (e.g., time and other costs) of the full GeoEvolve pipeline compared to the baseline OpenEvolve?

How sensitive are the results to the specific LLM (e.g., GPT-4) used as the core evolutionary engine? Would a less capable model yield the same improvements?

The housing price uncertainty results show a clear spatial pattern. Does the evolved GeoCP model provide a causal or merely correlational explanation for why uncertainty is higher in specific areas?

The concept of "knowledge discovery" in the appendix lists generic improvements. What specific, non-obvious geospatial insight was discovered that would be valuable to a human expert?  Moreover, the evolved Kriging model incorporates multiple advanced techniques (e.g., matern family, localized kriging). To what extent is GeoEvolve discovering truly novel algorithms versus recombining known, existing best practices?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes GeoEvolve, a multi‑agent LLM framework that wraps OpenEvolve with an outer agent and a geospatial RAG module (GeoKnowRAG) to guide code evolution for geospatial models. The system is evaluated on two tasks: spatial interpolation (ordinary kriging on an Australian soil trace‑element dataset) and spatial uncertainty quantification (GeoCP on Seattle house prices). GeoEvolve reportedly discovers improved variants of kriging and GeoCP, reducing RMSE by ~13–21% and interval score by ~12–17% relative to the original algorithms.

### Strengths
- Within geospatial AI, offering a concrete, automated recipe to evolve classical methods like ordinary kriging is potentially impactful.
- More broadly, the paper is a case study of how to inject structured domain theory into LLM‑based algorithm search, which is of interest to the broader ICLR audience working on scientific discovery with LLMs.
- The presentation is clear and results are explained well.
- The results show consistent gains across both datasets, and the ablation studies show the value of the GeoKnowRAG component.

### Weaknesses
- The novelty is fairly low (wrapping already existing evolutionary algorithm for code generation in the domain-specific RAG layer + agent controller). The paper could have done with results on (a) multiple LLMs (the paper only mentions GPT-4 and does not go into more detail) (b) more ablations of the algorithm itself (c) more datasets.
- Testing on only two geospatial tasks with single datasets per task is insufficient to support claims of a "scalable pipeline for diverse geospatial tasks." The generalization remains unproven.
- The GeoKnowRAG database of 141 documents is built using author-selected keywords, introducing potential bias and incompleteness. No systematic validation of coverage or relevance is provided.

### Questions
- Can you go into more detail on the LLMs used?
- Have you tried stronger hand‑crafted domain prompts that explicitly describe advanced kriging techniques? That would better isolate whether retrieval is actually adding value over a carefully written static prompt.
- How sensitive is GeoEvolve to the underlying LLM? Do reasoning models do better?

### Soundness
2

### Presentation
3

### Contribution
2
