# Evolving DAGs with LLM: Towards Smart and Hallucination-Mitigated Causal Discovery

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2

## Abstract
This paper has been withdrawn

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces **EvoDAG-LM**, an evolutionary causal discovery algorithm that integrates large language models (LLMs) to guide search-space pruning, mutation, crossover, and loop removal. LLMs provide semantic guidance, while final edge selection relies on statistical scores to reduce hallucination. Experiments show that EvoDAG-LM achieves higher F1 and lower SHD than EA, DL, and LLM-based baselines, especially on medium and large graphs. Overall, it presents a promising LLM-assisted framework combining semantic knowledge with data-driven causal discovery.

### Strengths
1.The paper introduces a clever hybrid approach, using LLMs not as direct causal predictors but as knowledge-based guides for an evolutionary search. By adding LLM-informed pruning (search-space reduction) and LLM-guided operators (semantic crossover/mutation) as well as loop-breaking, EvoDAG-LM injects external semantic information into causal discovery. This idea of LLM-augmented EA is novel and addresses a known weakness of pure EA or DL methods (lack of domain knowledge)
2. The design explicitly avoids trusting the LLM’s judgment blindly. All final edge decisions come from standard scoring (e.g. BIC), with the LLM only suggesting candidates. The authors emphasize that this “hallucination-mitigated” strategy guards against spurious LLM outputs In particular, the loop-removal component uses LLM queries only to identify unlikely edges, and if uncertainty remains it falls back to mutual-information scores. This combination is a sensible safeguard.
3. Experiments span diverse benchmarks and strong baselines, including EA, DL, classical, and LLM-based methods (e.g., ChatPC, LCDHP). Using F1 and SHD metrics, results show EvoDAG-LM clearly outperforms others on medium and large graphs and remains competitive on very large ones, demonstrating its strong overall effectiveness.
4. The authors recognize LLM limitations (context window, cost) and incorporate a scale-aware mechanism. They decrease the probability of invoking the LLM for very large graphs, and they adopt cheaper LLM variants (e.g. GPT-3.5-Turbo) where feasible. Such considerations reflect a realistic view of deployment. The prompting strategy is also well-designed, which is a clever way to exploit LLM reasoning.

### Weaknesses
1. A major concern is the heavy computational demand of EvoDAG-LM, which requires numerous LLM queries per generation (for correlation estimation, Tree-of-Thought evolution, loop validation, etc.). The paper explicitly avoids discussing runtime(substituting Fitness Evaluations (FEs) instead), citing variability in API latency, but this omission sidesteps a crucial issue — the actual computational and costs. There is no comparison with baselines in terms of token consumption, number of LLM calls, or time overhead. Moreover, the authors admit that for very large graphs, they had to disable LLM-driven evolution entirely due to token limitations. This suggests the approach may not scale effectively until LLM efficiency improves. And I would suggest that a detailed efficiency/cost analysis be included in the main body of the paper, rather than being relegated to the appendix.
2. The effectiveness of EvoDAG-LM hinges on the assumption that each variable has a meaningful name or description that the LLM can interpret. In domains where variable names are arbitrary codes or are not human-readable, the LLM components would likely fail or provide no useful signal. The authors themselves acknowledge this limitation, noting that in the Insurance network, ambiguous labels like “this car” versus “other cars” caused the LLM to make mistakes, which hurt performance. This raises doubts about the method's applicability to real-world data where semantic hints are unclear or absent. The method's significant reliance on the LLM's "world knowledge" thus limits its generalizability.
3. The paper presents a novel approach by integrating LLMs directly into an Evolutionary Algorithm (EA). While this specific integration is new, the underlying concept of using LLMs to supply causal prior knowledge or to refine graph structures builds upon ideas explored in prior work. For example, the recent ALCM framework (Khatibi et al., 2025) and LLM-CD (Du et al., 2025) also employ a similar strategy of combining data-driven discovery with LLM-based refinement. To further highlight the unique contributions of EvoDAG-LM, the paper would be strengthened by conducting a more comprehensive survey of related work and providing a detailed comparison with these approaches. This would help clarify how its specific design—particularly the careful implementation of the LLM as an auxiliary guide within the search—sets it apart from other hybrid methods.

[1] Khatibi, E., Abbasian, M., Yang, Z., Azimi, I., & Rahmani, A. M. (2025). ALCM: Autonomous LLM-Augmented Causal Discovery Framework. arXiv preprint arXiv:2405.01744. https://arxiv.org/abs/2405.01744
[2] Du, H., Zheng, Y., Jing, B., Zhao, Y., Kou, G., Liu, G., Gu, T., Li, W., & Yang, C. (2025). Causal Discovery through Synergizing Large Language Model and Data-Driven Reasoning. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25) (pp. 543–554). Association for Computing Machinery, New York, NY, USA. https://doi.org/10.1145/3711896.3736874

### Questions
1. Since the authors state that the proposed method uses the LLM in an auxiliary and guiding role, could you provide a comparison of efficiency and cost between EvoDAG-LM and the baselines? For instance, compared to other LLM-based methods, by how much is the token consumption reduced, and what are the differences in runtime? Can you provide any information on the wall-clock runtime or number of LLM calls (and hence cost) for representative tasks? This is important for assessing feasibility.
2. The competitors include four state-of-the-art (SOTA) EA-based methods for comparison: MIGA (Yan et al., 2023), Hybrid-SLA (Jose et al., 2019), PC-PSO (Sun et al., 2021), and AESL-GA (Contaldi et al., 2019); four LLM-based methods, including two best LLM-based methods (i.e., GPT4 and GPT4-Turbo (Achiam et al., 2023)) in CausalBench (Zhou et al., 2024) and two latest LLM-based works (i.e., ChatPC (Cohrs et al., 2024) and LCDHP (Wang et al., 2025));'
However, the experimental results table shows: GPT4-Turbo, GPT4, CausalGPT, LCDHP. Are CausalGPT and ChatPC the same method? Or in other words, is the 'CausalGPT' in the experimental table actually 'ChatPC'?"

### Soundness
3

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
The paper is on incorporating the semantic information with LLMs to serve as a prior for the causal discovery task. It proposed a new pipeline: EvoDAG-LM, which integrates causal discovery and evolutionary algorithms. It employs the LLMs to process and exploit the given semantic metadata behind variables. 

The method consists of three stages: (1)  search space reduction; (2) evolutionary operator enhancement; and (3) loop removal. Empirical results are presented on about 11 datasets with four types of baselines.

### Strengths
- Integration of LLMs into EAs. It integrates evolutionary algorithms to treat causal discovery from a perspective of combinatorial optimization.
-  Extensive evaluation. Empirical results are presented on about 11 datasets with four types of baselines.

### Weaknesses
- The research problem and the technical challenges that motivate the three proposed components are not clear. For example, existing methods that utilize variable descriptions can also help to reduce the search space. What is the specific drawback of the previous baselines that is to be addressed by this paper?
- Ablation of ToT. The use of Tree-of-Thought (ToT) prompting is highlighted in this paper. It is necessary to report the results about (1)  EvoDAG-LM using CoT and direct answering;  (2) more baselines with ToT in Tables 1 and 2.

### Questions
- What is the meaning of equation (2)? What is the difference between $\text{MI}_{i,j}$ and $\text{MI}(i,j)$?
- How would the quality of such metadata influence the proposed pipeline? For example, if some variables' metadata are absent, ambiguous, or even wrong, would the proposed method be robust to these situations?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
** Summary

The work proposes a hybrid method, which is based on statistical-based method and LLM-based method, for causal discovery, termed EvoDAG-LM, aiming to mitigate the shortage of methods that purely reply on numerical data. The method consists of three steps: (1) Query LLM to remove edges between weakly correlated variable pairs; (2) Evolutionarily refine the causal graph based on LLM reasoning and statistical information; and (3) Remove cycles from the causal graph. The paper provides empirical results to show that EvoDAG-LM almost outperforms other baseline methods on a series of benchmarks.

** Recommendation 

I would like to recommend a rejection to this paper for its limited novelty and presentation. The paper proposes a hybrid causal discovery method, where several similar methods have emerged recently. Comparing to existing methods, EvoDAG-LM uses evolutionary operations, which potentially enhance the pipeline’s accuracy. However, I think the contribution remains limited. Additionally, I believe the paper’s presentation need further improvement to make readers fully understand the technical parts.

### Strengths
1. The method employs LLMs in evolutionary operations to refine the causal graph, potentially enhancing the method’s performance.

### Weaknesses
1. The idea of hybrid causal discovery method is not new, and the original contribution remains limited.
2. The presentation is not clear, undermining the paper’s contribution. For instance, it does not make the setting of combining LLM and statistical methods clear in the beginning of the technical part or write data as input in the algorithm, only emphasising the role of LLMs. This makes me confusing when reading the paper. Many notations and terms of statistics and graph are not defined or used inconsistently, e.g., MI_{I,j} and MI(i,j), Pa, individuals, and using e^g_{i,j,m} as both a matrix and number. Another important point is that, I found the description of some key steps, e.g., the Selection operation, the Crossover operation, and the Mutation operation, is not understandable, while the paper keeps much explanation of prompting strategies like ToT and CoT. This reduces the paper’s readability.
3. The work uses many popular benchmarks to evaluate the method. However, some of those datasets are well remembered by the LLMs, undermining the quality of the evaluation part. Moreover, for instance, as far as I remember, many LLM-based methods have very high performance on Asia (close to 100%), however, the reported baseline LLM-based methods only have low performances.

### Questions
See the weakness section.

### Soundness
2

### Presentation
1

### Contribution
2
