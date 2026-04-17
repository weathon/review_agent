# Drug Repurposing with a Graph-of-Thoughts Inspired Reasoning Framework

- Decision: Reject
- Scores: 2, 2, 0

## Abstract
Drug repurposing is a promising strategy to accelerate therapeutic development, and provides a viable and effective way to treat diseases, especially rare diseases, that otherwise do not have established and approved treatment options. As the magnitude of available biological data continues to increase, computational methods have become vital for extracting meaningful insights and identifying candidates for repurposing. Although large language model (LLM)-driven agentic workflows show potential, their computational cost and latency often make them impractical. We present a novel, efficient platform that addresses this challenge by grounding LLM reasoning in a biomedical knowledge graph (PrimeKG). This uses known relationships between biological entities to deduce relevant drug repurposing information. Our method first identifies multiple diverse paths between a given drug-disease pair from a natural language query. A Graph-of-Thoughts (GoT)-inspired module then constrains the LLM to reason over these structured paths and synthesize the information into a coherent biological hypothesis. Our platform presents key novelties such as Graph-of-Thoughts-inspired reasoning and diverse KG path-finding that seek to ground reasoning in biological knowledge and mitigate hallucinations. Our evaluation demonstrates that this constrained approach performs comparably in accuracy to unconstrained agentic workflows. Our platform achieves this with significantly fewer LLM calls, 55.3% lower token consumption, and 40.4% less time. This GoT-inspired framework, grounded on knowledge graph data, presents an efficient and powerful system for LLM-driven drug repurposing, effectively balancing reasoning with computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a Graph-of-Thoughts (GoT)-inspired reasoning framework for LLM-driven drug repurposing.
It grounds LLM reasoning in a biomedical knowledge graph (PrimeKG), aiming to improve accuracy, interpretability, and computational efficiency compared to unconstrained agentic systems like Biomni.
The system extracts diverse mechanistic paths between drug–disease pairs via LLM-generated Cypher queries, converts them into natural-language context, and performs iterative reasoning and self-evaluation using a GoT-style reasoning graph.
Results on both a custom dataset and PharmaDB show comparable accuracy to Biomni but 55% fewer tokens and 40% less inference time. Case studies (Aldesleukin–neuroblastoma, Dactinomycin–MPNST) illustrate qualitative interpretability.

### Strengths
* First to integrate a Graph-of-Thoughts reasoning framework with biomedical knowledge graphs for drug repurposing.

* Achieves comparable accuracy to large agentic systems (e.g., Biomni) with 55% fewer tokens and 40% less time.

* Provides transparent, mechanism-level hypotheses grounded in KG triples, reducing hallucination risk.

* The pipeline (NER → KG pathfinding → GoT reasoning) is clearly explained and reproducible.

* Addresses an important real-world challenge in scientific reasoning and data-efficient LLM use.

### Weaknesses
* The GoT reasoning loop mainly adapts existing multi-step prompting ideas; innovation lies more in application than in core modeling.
* Only two datasets and one baseline (Biomni); lacks broader or neural reasoning baselines such as TxGNN [1].
* Grounding and redundancy scores depend on LLM-based evaluation without external or expert validation.
* Evaluation on fewer than 100 drug–disease pairs limits generality and statistical confidence. Evaluation of more data can improve this work.
* No quantitative study on system behavior under larger graphs, different path lengths, or reasoning iterations. These explorations may clarify your claims.
* Figures are clear but focus more on workflow than on experimental insight or analysis depth.

[1] Huang, K., Chandak, P., Wang, Q., Havaldar, S., Vaid, A., Leskovec, J., ... & Zitnik, M. (2024). A foundation model for clinician-centered drug repurposing. Nature Medicine, 30(12), 3601-3613.

### Questions
+ The paper reports 55% fewer tokens and 40% less time compared to Biomni. Could the authors clarify where this efficiency gain primarily comes from — is it due to shorter or fewer reasoning paths in the Graph-of-Thoughts process, or because the system performs single-step constrained reasoning rather than multi-agent iterative search as in Biomni?
+ For drug–disease pairs where the system cannot find any connecting path in PrimeKG (the paper mentions 11 / 93 cases were dropped): how are these handled in practice? Are they treated as “no evidence, cannot conclude”, or implicitly as negatives? Do they affect precision/recall calculation (e.g., are they excluded from both TP/FN), and if so, does that bias evaluation toward pairs that are already well-studied in the KG? More broadly, how should we interpret such pairs biologically — “likely unrelated”, or “KG incomplete”?

While the paper presents a clear application of Graph-of-Thoughts reasoning to drug repurposing, it lacks convincing experimental evidence demonstrating that the proposed GoT framework brings substantial advantages over existing KG-based or reasoning-driven approaches.
The evaluation mainly compares with a single agentic baseline (Biomni) without isolating the contribution of GoT reasoning itself, and no ablation or quantitative analysis supports its claimed benefits.
Moreover, the method does not address how to generalize beyond incomplete knowledge graphs, a key limitation for real-world biomedical reasoning.
Unless the authors can provide stronger experimental validation and clearer superiority over KG methods during rebuttal, I would recommend rejection.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This is an interesting paper with a creative approach to computational drug repurposing. The authors designed an LLM agent which makes predictions around drug repurposing by using paths through a KG (PrimeKG in this case). These paths provide semantic guidance but also possible explanations for predictions. Consequently, this helps to achieve the goal of guiding LLM reasoning and preventing hallucinations. However, while I understand the motivation of the approach with respect to LLM agents and mitigating hallucinations, I struggle to see the motivation for such an approach with respect to the literature in computational drug repurposing.

### Strengths
**Strong points**:
- Creative solution
- Leverages biologically and medically relevant information
- Provides a case study in addition to metrics (great idea to focus on the quality of the explanations in addition to the predictions)
- The claims made are supported by the results reported.

### Weaknesses
**Weak points**:
- There is a lack of information about what the Custom Dataset in Table 1 comprises. It is difficult to assess the reported results with such little information.
- One of my main concerns here is the baselines against which the authors compare their method (Table 1). I believe they should compare their approach against link prediction using a variety of GNNs, KG embedding approaches, and network measures like shortest paths or degree-weighted-path counts. I would guess that using some simpler, less computationally demanding approaches (particularly those which do not use an LLM) could get equal or better performances to those reported in Table 1. 
- Given the above, if simpler methods work better or similarly, then why should one consider using this approach instead of others for this drug repurposing task? If the explainability via graph paths is key, there are also existing approaches which can get graph paths without using an LLM. For example:
    - PoLo: https://dl.acm.org/doi/10.1007/978-3-030-77385-4_22 
    - MARS: https://arxiv.org/abs/2410.05289
    - This approach by Sudhahar et al: https://www.nature.com/articles/s41467-024-50024-6
- Finally, when addressing the above point, the authors might consider comparing their approach, at least qualitatively, to some of the above works.

### Questions
In summary, while I think this is an interesting approach, I struggle to see the niche it fills with respect to computational drug repurposing. I would ask that the authors:
1. Please clarify what is within the Custom Dataset and how it is structured?
2. Complete more robust benchmarking against two (2) well used KG embedding methods (RotatE, TransE, perhaps), one (1) GNN approach (GCN, perhaps), and one (1) network based approach, such as a shortest-paths method. 
3. In light of whatever the results of point 2 are, better justify how this approach fits in with other, recent computational drug repurposing approaches.
4. Justify what role the LLM is playing here- why not simply conduct link prediction on a KG?

As it stands, I will reject the submission, but may reconsider if the above suggestions are sufficiently addressed.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper applied graph of thoughts method for drug repurposing for rare diseases. It utilizes PrimeKG as the base knowledge graph, and works with LLM through GoT to generate cypher queries to search PrimeKG and summarize/evaluate the outputs. It has been tested on 90 pairs of drugs and diseases on public data and private data.

### Strengths
- adapting graph of thoughts to drug repurposing for rare diseases
- evaluate the proposed methods with self-developed baseline method biomni for evaluation.
- Based on the existing large scale knowledge graph for drug discovery PrimeKG.

### Weaknesses
- This paper lacks algorithm novelty. it is an application of existing methods, such as graph of thoughts, neo4j search using cypher queries, LLM to summarize results. It is more toward a line of engineering work to build a tool.
- The evaluation is vague, It is rather a hypothesis generating and evaluation. It does not compare with the related work or baselines, such as compare with tree of thoughts, or chain of thoughts methods, or compare directly with the strong reasoning models. 
- the evaluation is based on one public dataset and one private dataset, but it is not clear why these matter, as the main knowledge database is still PrimeKG.

### Questions
- can you please compare with more related baselines, for example, just for the reasoning part, you can compare your graph of thoughts with tree of thoughts and chain of thoughts.
- can you just use the powerful reasoning models (such as GPT-5) to directly address the questions and compare the reasoning steps with your graph of thoughts
- There are different knowledge graphs for drug repurposing (https://academic.oup.com/bib/article/25/6/bbae461/7774899), can you please justify that PrimeKG is the best KG for your application.

### Soundness
1

### Presentation
2

### Contribution
1
