# OMAC: A Broad Optimization Framework for LLM-Based Multi-Agent Collaboration

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Agents powered by advanced large language models (LLMs) have demonstrated impressive capabilities across diverse complex applications. Recently, Multi-Agent Systems (MAS), wherein multiple agents collaborate and communicate with each other, have exhibited enhanced capabilities in complex tasks, such as high-quality code generation and arithmetic reasoning. However, the development of such systems often relies on handcrafted methods, and the literature on systematic design and optimization of LLM-based MAS remains limited.

In this work, we introduce OMAC, a general framework designed for holistic optimization of LLM-based MAS. Specifically, we identify five key optimization dimensions for MAS, encompassing both agent functionality and collaboration structure. Building upon these dimensions, we first propose a general algorithm, utilizing two actors termed the Semantic Initializer and the Contrastive Comparator, to optimize any single dimension. Then, we present an algorithm for joint optimization across multiple dimensions. Extensive experiments demonstrate the superior performance of OMAC on diverse tasks against recent approaches.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces OMAC, a general framework for systematically optimizing LLM-based Multi-Agent Systems (MAS). The authors identify five key optimization dimensions: two related to agent functionality (optimizing existing agents, constructing new ones) and three related to collaboration structure (candidate selection, dynamic participation, communication flows). The core of OMAC consists of two LLM-powered actors: a "Semantic Initializer" that generates an initial diverse set of agents or controllers, and a "Contrastive Comparator" that iteratively refines them by reasoning over high-performing (positive) and low-performing (negative) pairs sampled based on performance on a training set. The framework also supports joint, iterative optimization across multiple dimensions. Extensive experiments on code generation, general reasoning, and arithmetic reasoning tasks demonstrate that OMAC significantly outperforms existing baselines, with further gains achieved through multi-dimensional optimization.

### Strengths
1. The paper is well-written.

2. The motivation of the paper is clear.

3. Comprehensive and Systematic Framework: The paper proposes a holistic and principled framework for MAS optimization that is a significant step beyond ad-hoc, handcrafted design.

### Weaknesses
1. Limited Novelty of Core Components: The core ideas behind the Semantic Initializer and Contrastive Comparator are conceptually similar to principles already established in the literature on prompt and agent optimization [1,2,8,9,10]

2. Insufficient Comparison with Latest Baselines: The experimental evaluation is missing comparisons against several recent and highly relevant baselines in multi-agent system optimization, such as Aflow [1], AgentSquare [2], MaAS [3], and G-designer [4]. The paper includes a comparison with Aflow, but it is limited to the MBPP benchmark and is placed in the appendix (Appendix B.2.5, Table 11) rather than being integrated into the main experimental results across all benchmarks (Section 4.1, Tables 1-3).

3. Use of Potentially Saturated Benchmarks: The currently used benchmark is somewhat outdated. The most advanced models or methods have already achieved very high scores. To more robustly demonstrate the advantages of the OMAC framework, evaluation on more recent and challenging benchmarks such as GAIA [5], SWE-bench [6], or HLE [7] would be more convincing and provide a better measure of its capabilities on complex, unsolved problems.

4. Lack of Evaluation on Open-Source Models: The paper's empirical evidence relies exclusively on proprietary GPT-series models. The experiments should be conducted on a diverse range of open-source LLMs. 

5. Incomplete Computational Cost Analysis: A more thorough analysis would involve a direct comparison of the training cost against other MAS optimization frameworks. Visualizing this trade-off, for instance with a Pareto front plotting performance against computational cost, would provide a much clearer picture of OMAC's efficiency relative to its competitors.

6. Outdated Related Work Section: The Related Work section (Section 5) should be updated to include a discussion of the latest relevant works.

Reference

[1]Zhang J, Xiang J, Yu Z, et al. AFlow: Automating Agentic Workflow Generation[C]//The Thirteenth International Conference on Learning Representations.

[2]Shang Y, Li Y, Zhao K, et al. AgentSquare: Automatic LLM Agent Search in Modular Design Space[C]//The Thirteenth International Conference on Learning Representations.

[3]Zhang G, Niu L, Fang J, et al. Multi-agent Architecture Search via Agentic Supernet[C]//Forty-second International Conference on Machine Learning.

[4]Zhang G, Yue Y, Sun X, et al. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks[C]//Forty-second International Conference on Machine Learning.

[5]Mialon G, Fourrier C, Wolf T, et al. Gaia: a benchmark for general ai assistants[C]//The Twelfth International Conference on Learning Representations. 2023.

[6]Jimenez C E, Yang J, Wettig A, et al. SWE-bench: Can Language Models Resolve Real-world Github Issues?[C]//The Twelfth International Conference on Learning Representations.

[7]Phan L, Gatti A, Han Z, et al. Humanity's last exam[J]. arXiv preprint arXiv:2501.14249, 2025.

[8]Hu S, Lu C, Clune J. Automated Design of Agentic Systems[C]//The Thirteenth International Conference on Learning Representations.

[9]Yuksekgonul, M., Bianchi, F., Boen, J. et al. Optimizing generative AI by backpropagating language model feedback. Nature 639, 609–616 (2025). https://doi.org/10.1038/s41586-025-08661-4

[10]Shinn N, Cassano F, Gopinath A, et al. Reflexion: Language agents with verbal reinforcement learning[J]. Advances in Neural Information Processing Systems, 2023, 36: 8634-8652.

### Questions
Please address the issue I raised in weaknesses.

### Soundness
3

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
4

### Summary
The paper proposes OMAC, a general supervised optimization framework for LLM-based multi-agent systems (MAS) engaged in multi-step collaboration. It identifies five key optimization dimensions—two functional (optimizing existing agents, constructing new agents) and three structural (candidate selection, dynamic participation, communication flow)—and introduces a unified algorithm using two LLM-powered actors: the Semantic Initializer (for diverse prompt generation) and the Contrastive Comparator (for iterative refinement via performance-guided contrastive reasoning). Experiments on code generation, general reasoning, and arithmetic reasoning show consistent improvements over strong baselines, with further gains from joint multi-dimension optimization.

### Strengths
1) OMAC offers a systematic decomposition of MAS optimization into five well-motivated dimensions, going beyond prior work (e.g., DyLAN) that focuses only on team composition. The use of contrastive reasoning guided by supervised performance signals is an approach to prompt/controller optimization in MAS.

2) The paper is well-structured and clearly written. Key concepts (e.g., dimensions, actors, workflow) are introduced intuitively. The optimization process is easy to follow despite its generality.

### Weaknesses
1) The proposed framework is essentially a form of rejection fine-tuning (RFT), which samples positive examples to train the LLMs. However, a large body of work in agentic reinforcement learning has already explored optimizing agents or managing prompts via supervised fine-tuning (SFT), rejection fine-tuning (RFT), reinforcement learning (RL), or prompt evolution [1–3]. The authors should compare their approach with these existing methods and clearly articulate the key differences.

2) LLM generation–based agent construction also does not rely on human prior knowledge. In this context, why is optimizing agent team composition using supervised signals preferable to unsupervised approaches?

3) The rationale for optimizing these five specific dimensions—each of which appears largely independent—is not sufficiently justified. Optimizing across five separate dimensions reads more like a collection of incremental improvements rather than a cohesive, unified contribution, which undermines the overall novelty of the paper.

4) OMAC optimizes prompts at test time using ground-truth labels, which constitutes an unfair comparison with ByLAN. The authors should include comparisons against supervised methods that also leverage test-time information, such as test-time training approaches. 

5) There are several grammatical and typographical issues. For example:  
   Original: “in handling tasks within sophisticated environments Li et al. (2023a); Wang et al. (2023b).”  
   Suggested revision: “in handling tasks within sophisticated environments (Li et al., 2023a; Wang et al., 2023b).”

References
[1] Wong, M., Ong, Y. S., Gupta, A., et al. Prompt evolution for generative AI: A classifier-guided approach. In *2023 IEEE Conference on Artificial Intelligence (CAI)* (pp. 226–229). IEEE, 2023.  
[2] Agrawal, L. A., Tan, S., Soylu, D., et al. GEPA: Reflective prompt evolution can outperform reinforcement learning. *arXiv preprint arXiv:2507.19457*, 2025.  
[3] Guo, W., Yang, J., Yang, K., Li, X., Rao, Z., Xu, Y., & Niu, D. (2024). Instruction Fusion: Advancing Prompt Evolution through Hybridization. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 3883–3893). Association for Computational Linguistics.

### Questions
1) Can the proposed model optimize all five dimensions simultaneously? Why does the framework train them iteratively?

2) How does the proposed method compare with approaches that optimize agents via test-time training or prompt evolution? The authors should include more trainable baselines for a fair comparison.

3) How do supervised methods perform on the same tasks? A comparison with such methods would help contextualize the claimed advantages of the proposed approach.

4) Moreover, the method requires full MAS evaluation on the training set per candidate, which is expensive. The author should compare the complexity with existing models.

### Soundness
3

### Presentation
4

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
This paper introduces OMAC for systematically designing and optimizing Multi-Agent Systems. The method helps mitigate the current reliance on handcrafted, labor-intensive methods in MAS development. OMAC proposes an automated approach to optimize collaborative structure, communication patterns, and underlying prompts to achieve enhanced performance across complex tasks.

### Strengths
The core concept of introducing a systematic optimization framework to move beyond "handcrafted methods" for MAS design is meaningful. The implicit reliance on LLMs to perform meta-optimization on the agent configuration is a promising application of LLM capabilities. The proposed method achieves superior performance to the compared baselines.

### Weaknesses
1. The term "Broad Optimization" in the title is misleading, as the framework appears narrowly focused on optimizing communication/collaboration prompts. The paper does not demonstrate the capability to optimize critical non-prompt-based technical elements, such as the agent's action space (tool/API use), internal state representation (memory management and retrieval). This significantly limits the framework's claim to "broad" generality.

2. The underlying principle of optimization effectiveness remains unclear. Why only these five dimensions are considered? Are they manually defined? The authors are encouraged to provide more direct evidence why and how OMAC selects the optimal configuration.

3. Assuming the performance gain is marginal (e.g., around 1% in many tasks), this fails to justify the significant computational overhead and technical complexity introduced by the OMAC framework. The authors must present a compelling cost-benefit analysis (total token consumption vs. performance gain) to establish the economic and technical value proposition of OMAC over simpler baseline methods.

### Questions
1. Given that OMAC essentially seeks to optimize MAS based on task performance, how does this approach technically relate to established reinforcement learning-based methods that optimize agents and communication protocols? The authors are strongly encouraged to supplement the related work with an explicit discussion on the relative merits (e.g., sample efficiency, convergence speed, stability) of the LLM-driven meta-optimization approach versus RL approaches, and suggest a framework for quantitative comparison against key RL baselines.

2. The baseline selections are encouraged to include more advanced methods (since 2025) for comparison.

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
The paper proposes a general, supervised optimization framework for LLM-based multi-agent systems (MAS) operating in multi-step collaborations. The authors identify five optimization dimensions—two *functional* and three *structural*. For any chosen dimension, the proposed method iterates two LLM instances: a Semantic Initializer to generate candidate prompts/controllers and a Contrastive Comparator that refines them by reasoning over supervised positive–negative pairs produced from training-set evaluations. Experiments on HumanEval (code), MMLU (general reasoning), MATH (arithmetic), and extra benchmarks show the empirical significance across single-dimension and multi-dimension settings, with ablations showing the Contrastive Comparator contributes non-trivial gains. The authors also report cost benefits at inference time via structural optimization and disclose training costs.

### Strengths
1. **Clear, general framework covering both agent capability and collaboration topology.** The five-dimension taxonomy is well-motivated (nodes/edges view of MAS) and seems broadly reusable. 
2. **Methodological backbone is valid.** Contrastive refinement using online explored positive–negative training pairs. This approach is intuitive and straightforward to implement. 
3. **Consistent improvements across tasks and dimensions.** Single-dimension results show OMAC improves over strong MAS baselines, and multi-dimension optimization yields additional gains.  
4. **Rich ablations.** The effort to ablate hyperparameters, dimension contributions, and the Contrastive Comparator’s role adds confidence in the findings.

### Weaknesses
1. **Small supervised splits may limit statistical validity and generalization.** Using the same data points from the test sets for training raises concerns about overfitting and generalization. Though aligned with DyLAN, larger, disjoint training sets would strengthen the claims. 
2. **Baseline selection could be broader.** The experiments compare against a limited set of MAS baselines. Including more recent or diverse approaches for different single-/multi-dimension optimizations would better contextualize the contributions, e.g., prompt optimization methods for agent-level prompts, or graph neural network-based MAS for structural aspects. 
3. **Discussion of gradient and contextual optimization is lacking.** The proposed method relies on LLM inference calls for optimization. However, gradient-based methods are natural when contrastive pairs are available. A discussion of the trade-offs between these approaches is missing. 
4. **Joint-dimension synergy is constrained.** Simultaneously optimizing multiple dimensions can hurt stability; the proposed iterative scheme is a pragmatic workaround, but deeper analysis of joint optimization remains unexplored, e.g., data mixture, gradient blending, or other multi-objective optimization techniques.

### Questions
How do you envision the practical training cost for MAS optimization? Though OMAC outperforms several baselines, is the training cost worth the performance gain in practical applications?

### Soundness
3

### Presentation
3

### Contribution
2
