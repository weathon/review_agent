# InfoAgent: Advancing Autonomous Information‑Seeking Agents

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Building Large Language Model agents that expand their capabilities by interacting with external tools represents a new frontier in AI research and applications. In this paper, we introduce InfoAgent, a deep research agent powered by an innovative data synthesis pipeline and orchestrated web search tools. To construct challenging, hard-to-find queries, we build entity trees and apply sub-tree sampling with entity fuzzification to systematically increase question difficulty. Unlike prior work that relies heavily on commercial search tools, we develop a dedicated self-hosted search  infrastructure, enhancing transparency of agent environments and facilitating further advancement of agent capacity. We evaluate the effectiveness of our data pipeline by measuring the average number of tool calls required to correctly answer a question, and also show that our agent yields better performance when equipped with our tools. Our InfoAgent is post-trained from Qwen3-14B using a two-stage recipe: cold-start supervised finetuning to instill long-horizon search behaviors, followed by reinforcement learning which significantly improves reasoning-driven tool use. With our methods, InfoAgent achieves 15.3% accuracy on BrowseComp, 29.2% on BrowseComp-ZH, and 40.4% on Xbench-DS, outperforming prior open-source deep research agents such as WebSailor-72B and DeepDive-32B.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces InfoAgent, a 14B-parameter deep research agent trained via supervised fine-tuning (SFT) and reinforcement learning (RL) to perform multi-hop web search and information retrieval. The authors make three main contributions: (1) a data synthesis pipeline that constructs challenging multi-entity questions through entity tree construction, fuzzification, and sub-tree sampling; (2) a custom self-hosted search infrastructure with enhanced snippet generation to replace commercial APIs; and (3) a two-stage training recipe (SFT + GRPO) applied to Qwen3-14B. InfoAgent achieves 15.3% accuracy on BrowseComp, 29.2% on BrowseComp-ZH, and 40.4% on Xbench-DS, establishing state-of-the-art performance among open-source models under 15B parameters.

### Strengths
1. **Originality in data synthesis**: The entity tree construction with three-stage fuzzification (entity names, dates/numbers, semantic rephrasing) is a creative approach to systematically increase question difficulty.
2. **Quality of engineering**: The custom search infrastructure with multi-stage retrieval (BM25 -> embedding -> reranker -> LLM snippet generation) demonstrates good systems engineering. The Redis caching and performance optimizations show attention to practical RL training requirements.
3. **Thorough empirical analysis**: The ablation studies provide valuable insights:
   - SFT cold-start is essential (Figure 5 shows non-SFT model fails to learn)
   - Tool quality significantly impacts performance (Table 2)
   - Trajectory length matters (Table 3)
   - Process rewards don't help (Appendix D)

### Weaknesses
1. **Missing critical baselines**: The paper cites ChatGPT Deep Research (OpenAI, 2025a), Gemini Deep Research (Google, 2025), Perplexity Deep Research (Perplexity, 2025), and Grok Deep Search (xAI, 2025a) as motivating examples but does not evaluate against any of them. This is a critical omission that makes it impossible to assess the true contribution relative to deployed deep research systems.
2. **Limited performance and overstated claims**: The title claims to "advance" autonomous information-seeking agents, but InfoAgent achieves only 30% of o3's performance and lags significantly behind DeepSeek-V3.1 (71.2% vs 40.4% on Xbench-DS) and GLM-4.5 (68% vs 40.4%). The contribution is better characterized as advancing *small* open-source agents, not the field overall.
3. **Limited novelty**: The paper primarily combines existing techniques (ReAct framework, GRPO, entity-based QA generation, multi-stage retrieval). While the integration is competent, there is insufficient algorithmic innovation.

### Questions
1. **GLM-4.5 and DeepSeek-V3.1 setup**: Were these large open-source models evaluated with search capabilities? If so, what search infrastructure did they use? This affects interpretation of whether InfoAgent's contribution is the model training or the search tool.

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
4

### Summary
This paper introduces InfoAgent, a deep research agent built upon the Qwen3-14B model. It proposes an innovative data synthesis pipeline that constructs entity trees, uses sub-tree sampling, and applies "fact fuzzification" to generate challenging information-seeking problems with long-range dependencies. Second, the authors developed and deployed a self-hosted search and browsing infrastructure to replace opaque commercial APIs, enhancing research transparency and reproducibility. InfoAgent is trained using a two-stage "SFT cold-start + RL tuning" paradigm. 

Experimental results show that InfoAgent (14B) achieves SOTA performance in its parameter class across multiple benchmarks, outperforming larger models like WebSailor-72B and DeepDive-32B.

### Strengths
- The paper identifies the "shallow search" problem in current benchmarks. Its proposed data generation method, based on entity trees, sub-tree sampling, and systematic "fact fuzzification," is rational and effective.

- The search tool's design, which combines BM25 filtering, embedding and reranker models, and uses an LLM (GPT-4o-mini) to generate snippets, is in itself a good engineering practice for building a RAG system.

### Weaknesses
- The paper's core training framework (SFT + RL) is a standard paradigm in the agent domain (e.g., Search-R1, WebSailor). While the data synthesis component is well-executed, the idea of "generating QA pairs based on knowledge graphs/entity trees" is not entirely novel and is similar to the construction philosophy of benchmarks like WebWalkerQA.

- Critical Dependency on the Custom Search Tool: To what extent is InfoAgent's high performance attributable to,  the trained model itself, or the highly-optimized, custom search tool that uses GPT-4o-mini for summarization? The ablation study in Table 2 (using Wiki Retriever) shows a catastrophic performance drop (e.g., 10.0 → 1.0 on BrowseComp), which strongly suggests the model's capabilities are deeply entangled with this powerful tool, supporting the latter concern.

- The authors admit in the "Reproducibility Statement" that reproducing the custom search tool "presents greater challenges" and seems there are no plans to open-source it. Given the tool's decisive impact on performance, this severely hinders the community from building upon this work or making fair comparisons.

- In Table 1, the performance gap between InfoAgent and its SFT-only version is enormous (e.g., a jump from 4.7 to 15.3 on BrowseComp). Such a massive improvement from the RL stage alone is unusual and requires a more detailed explanation from the authors as to what specific capabilities the RL stage is imparting.

### Questions
- Can the authors provide a key ablation: *training* with the "Ours" tool but *inferring* with the standard "Wiki Retriever"? This would help assess whether the model's learned reasoning capabilities can generalize to a weaker tool.

- The performance gap between the SFT-only model (4.7 on BrowseComp) and the SFT+RL model (15.3) is very large. Can the authors elaborate on why the RL stage brings such a dramatic improvement? Is RL teaching the model entirely new search strategies?

- The paper attempts to use a process-based reward in Appendix D, but the experimental results (Table 4) show it provided no performance benefit and even led to a performance decrease on BrowseComp. How do the authors conclude the entity-recall-based process reward (Appendix D) failed? Does this suggest that the process reward signal itself was flawed?

### Soundness
2

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
3

### Summary
This paper presents InfoAgent, an information seeking agent built on top of strong search tools and optimized through training on synthetic data as well as reinforcement learning. InfoAgent includes a carefully designed search tools (search and browse) which performs multiple retrieval, re-ranking, and summarization steps for improved search quality. The data synthesis pipeline leverages tree structures to compose tasks of different difficulties. Experiments on several deep research benchmarks show that InfoAgent achieves the best performance among open-source methods. Further analysis delivers insights into the importance of the high quality search tool and several design choices in training.

### Strengths
1. The overall writing of the paper is clear.
2. The proposed search tool combines several existing techniques and delivers strong performance compared to the baseline (Wiki Retriever). The tool, once released, can be a useful resource to the community.
3. Based on the experiment results, the data synthesis and the training pipeline seem to work well. Also it is interesting that the benefit of training on English data can generalize to Chinese datasets.
4. The authors present several analysis and ablation study to present insights into developing stronger deep research agents, including the importance of tool quality, the critical role of SFT, and impact of trajectory length.

### Weaknesses
1. The paper highlights the need for "high-concurrency web search tool" for web training, however, there's no discussion on the proposed tool's relevant attributes, such as throughput and latency. 
2. The data synthesis pipeline is rather complex. Since it is based on entity tree structures, I would like to see some discussion on the distribution of sub-tree sizes and how the synthesized tasks look like. More ablation studies on the individual steps will also help justify the design choices.
3. Based on Table 2, it seems the major performance improvement comes from the proposed search tools, which includes many components and involve the use of commercial search APIs as well as LLM (GPT-4o-mini) for summarization. This potentially results in an unfair comparison to other open source methods. I would like to see more ablations and discussion on this to better determine the contribution of tool design.

### Questions
1. Missing a reference to Mind2Web2 [1] in related work, which is a recently published deep research benchmark.
2. line 118-120, could explain more on how RAG treats the retrieved passages as "latent variables" rather than input to the generator?
3. I am confused about the role of constraint set K(v). How does it prevent shallow pattern matching?
4. It is nice to see a discussion on tool call number in different datasets. Do you have further insights in the resulted qualitative difference between your data and existing ones? e.g. does your pipelines synthesize more balanced and comprehensive training trajectories?
5. In Section 4.3, quality of search tool, how about training on your dataset and test with Wiki Retriever? I am trying to understand whether the training pipeline improves the tool-use capability or the success of InfoAgent is more dependent on the high-quality tools.

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
This work introduces a deep research agent, InfoAgent. This work approaches the challenges of building a deep research agent through the following three aspects:
(1) a data synthesis pipeline to construct entity trees from Wikipedia and perform sub-tree sampling with fuzzification, which will be further leveraged to generate complex, multi-entity reasoning questions for model training. 
(2) self-implemented search tools without reliance on commercial search APIs for reproducibility. The tools include BM25 + embedding + reranker + LLM snippet pipeline. 
(3) Two-stage training with both SFT and RL. 
Empirically, InfoAgent achieves 15.3 % on BrowseComp, 29.2 % on BrowseComp-ZH, and 40.4 % on Xbench-DS, showing SOTA results among models with the same size and outperforming models at larger sizes.

### Strengths
(1) The entity-tree construction with multi-level fuzzification is a clever way to increase question complexity and encourage genuine multi-hop reasoning. Compared to earlier heuristic generation methods, this design systematically scales task difficulty while maintaining solvability.
(2) The self-implemented search/browse setup (BM25 + embedding + reranker + LLM snippet) is a clear step forward in reproducibility.
(3) The experiments with SFT and GRPO are helpful with detailed ablations. And the shown results are strong across multiple benchmarks.

### Weaknesses
(1) All experiments and synthesized data are from Wikipedia. The current setup doesn’t test how well the pipeline or the trained model generalizes to open-web or non-encyclopedic content (news, technical sites, etc.).
(2) The paper describes the tool infrastructure well, but rebuilding it from scratch would require major engineering. Even partial open-sourcing or more pseudo-code details would help the community reproduce results.

### Questions
(1) How do you ensure the fuzzified questions remain solvable with a single correct answer? Did you measure ambiguity or validate questions with a strong verifier (e.g., o3 or human checks)?
(2) What happens when the self-hosted search/browse system fails—e.g., missing content or timeouts? Is there a fallback or retry mechanism?
(3) Are there plans to release the data synthesis code or a lightweight version of the custom search tool? That would significantly improve reproducibility and downstream benchmarking.

### Soundness
3

### Presentation
3

### Contribution
3
