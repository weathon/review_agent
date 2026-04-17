# Huxley-G\"odel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine

- Decision: Accept (Oral)
- Scores: 6, 6, 4, 8

## Abstract
Recent studies operationalize self-improvement through coding agents that edit their own codebases. They grow a tree of self-modifications through expansion strategies that favor higher software engineering benchmark performance, assuming that this implies more promising subsequent self-modifications. However, we identify a mismatch between the agent’s self-improvement potential (metaproductivity) and its coding benchmark performance, namely the Metaproductivity-Performance~Mismatch. Inspired by Huxley’s concept of clade, we propose a metric ($\mathrm{CMP}$) that aggregates the benchmark performances of the descendants of an agent as an indicator of its potential for self-improvement. We show that, in our self-improving coding agent development setting, access to the true CMP is sufficient to simulate how the Gödel Machine would behave under certain assumptions. We introduce the Huxley-G\"odel Machine (HGM), which, by estimating $\mathrm{CMP}$ and using it as guidance, searches the tree of self-modifications. On SWE-bench Verified and Polyglot, HGM outperforms prior self-improving coding agent development methods while using fewer allocated CPU hours. Last but not least, HGM demonstrates strong transfer to other coding datasets and LLMs. %large language models. 
The agent optimized by HGM on SWE-bench Verified with GPT-5 mini and evaluated on SWE-bench Lite with GPT-5 achieves human-level performance, matching the best officially checked results of human-engineered coding agents. Our code is publicly available at https://github.com/metauto-ai/HGM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a novel self-improving coding framework (the Huxley-Gödel Machine) to overcome the "Metaproductivity-Performance Mismatch" existing in prior systems. Inspired by Huxley's concept of clade, authors propose the Clade-based Metric for Potential (CMP),  which measures an agent's true potential by aggregating the benchmark performances of its descendants, rather than just its immediate performance. The estimated CMP values guide the search tree during HGM’s self-modifications. Experiments show that the proposed method surpasses prior methods on benchmarks, e.g., SWE-bench and Polyglot. Notably, agents optimised by HGM achieve human-level performance on the SWE-bench Lite.

### Strengths
This paper addresses the very challenging problem in Machine Learning, namely, to build self-improving coding agents --  how can artificial agents improve their own codes? 

Authors introduce a new metric – Clade Metaproductivity Performance (CMP). The main concept is inspired by Huxley’s concept of “clade” in evolutionary biology and is computationally designed as an estimate based on an agent’s descendants. The Huxley-Gödel Machine uses estimated CMPs to select code modifications as a process of searching the “tree” of self-modifications.

### Weaknesses
The CMP is a probabilistic estimate based on historical data. Its predictive accuracy is uncertain.

Evaluation used LLMs as backbone, which are black box systems. LLMs+HGM may improve performances using statistical metrics, but it is not clear whether this will indeed lead into an interpretable AGI theory.

Somehow, authors try to sell their work by using big names, such as Gödel, Huxley. However, Gödel’s work is in symbolic logic. Huxley’s work supported Darwin’s theory of evolution through comparative anatomy.

### Questions
From Darwinist's perspective, human babies can think before they can speak; they can speak before they can write; they can write before they can program; they can write poor programs before they can write good programs. What is the starting point task that an ideal Huxley-Gödel Neural Network should solve?

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
For Recursive Self-Improvement (RSI), one needs to decide how to select among potential self-improvements. The approach studied in this paper is to do a bit of look-ahead and choose to modify the algorithm based on the performance averaged over several partial rollouts. This is a natural idea. It's implemented using frozen language models. Therefore, it is an instance of what is sometimes called Recursively Self-Improving Code Generation.

### Strengths
The problem being studied is very interesting and has potentially enormous impact.
The idea of using a bit of look-ahead is very expensive but principled.
Experiments suggest that it may be more effective than other approaches.

### Weaknesses
*Theory*: Theorem 1 is fine to include but it is not nearly enough to justify publication. The proof itself is almost tautological once the definitions are established. The theory appendix is poorly presented, which is concerning. For instance, the definitions of the concepts referenced in the statement of Theorem 1 are defined in the proof. The necessary definitions should be separate so that the theorem statement makes sense without reading the proof. Therefore the paper's justification is empirical. 

I found the algorithm a bit hard to follow. It would be good to include the language model prompts in the appendix. The prompts, from the supplementary materials, are rather substantial. Are the same prompts used for the HGM and DGM/SICA? In contrast, for example, the seed STOP prompt of Zelikman et al. (2023) is a half-page presented in the body of the paper. Is there a "seed" prompt for HGM that is at the heart of the algorithm, or are all the prompts from the supplementary material crucial to its success? 

*Experiments*: There are no current RSI benchmarks, and thus it is not clear how to compare algorithms. There is no easy way to benchmark RSI systems, and we have to trust the implementation of algorithms being compared against. If the algorithms have parameters, it is possible that the parameters of the HGM were better optimized to the few applications than those of the comparison algorithms. Moreover, the two algorithms compared against are from very recent papers that do not appear to be peer reviewed, so the empirical section is a comparison of implementations of three unproven algorithms. It would be good to have a more rigorous comparison framework.

*Ethics*: The ethical risks of RSI are not discussed. But clearly, the development of RSI poses potential risks that numerous luminaries claim are existential. See [https://superintelligence-statement.org/](https://superintelligence-statement.org/) for example. The risk is that it's advancing science towards that goal without clear discussion of why the benefits of this progress outweigh the risks.

*Small comments*: Readers unfamiliar with the term "clade" might appreciate a little explanation of what it means (e.g., mammals are a clade) so they don't have to look it up.

### Questions
To what extent is your implementation of DGM/SICA differ from theirs and to what extent is it modified to match your own?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of recursive self-improvement in agents. The authors point out the limitations of existing exploration strategies and propose a new method that accounts for long-term self-improvement capability. In the exploration process, an effective indicator is required to estimate the expected long-term improvement obtained when expanding a particular node (agent). Conventional approaches assumed that higher benchmark performance implied stronger self-improvement ability; however, the authors demonstrate experimentally that high-scoring agents do not necessarily produce promising descendants, while lower-scoring agents may yield superior results in the long run. 

To address this issue, the authors define Clade-Metaproductivity (CMP), a metric that measures self-improvement potential based on the collective performance of an agent’s entire lineage. The proposed method, HGM, controls exploration using the estimated CMP (ĈMP) and employs Thompson Sampling to select expansion nodes, promoting long-term improvement. Experiments on SWE-Verified and Polyglot show that HGM predicts true self-improvement ability with higher correlation than DGM or SICA, outperforming them in both exploration efficiency and final performance.

### Strengths
- The proposed method is inspired by the concept of clades and introduces a new metric (CMP), which measures the productivity of an entire lineage by aggregating the benchmark success of an agent’s descendants rather than relying on the agent’s own performance. This idea convincingly addresses the shortcomings of previous methods and provides a theoretically sound foundation for improving self-improving agent exploration. HGM achieves better performance than DGM and SICA.

### Weaknesses
- While the authors cite STOP as related work, they do not include it in their experiments. STOP recursively improves its own reasoning code, namely prompts and inference strategies, and thus represents a closely related setup. The lack of empirical comparison with STOP leaves a gap in the comprehensiveness of evaluation.
- The experimental evaluation is limited to two coding benchmarks, SWE-Verified-60 and Polyglot, both within a narrow programming domain. If the goal is to improve the agent’s own tool invocation and command execution behaviors, there is no inherent reason to restrict evaluation to coding tasks. Testing across diverse task domains would better demonstrate the generality of the proposed method. As it stands, the applicability of HGM beyond coding tasks remains unverified.

### Questions
- As the authors note, the modification operator in HGM performs patch applications to the agent’s codebase (such as file editing and bash command execution) but does not modify the model itself. Although the framework aims for recursive self-improvement of the agent’s own code, the actual scope of modification does not extend to architectural design or model-level enhancements, such as loss function optimization [1] or model merging [2]. Clarifying this limitation would help define the scope and contribution of the proposed approach more precisely.

[1] Discovering Preference Optimization Algorithms with and for Large Language Models 

[2] Can Large Language Models Invent Algorithms to Improve Themselves?: Algorithm Discovery for Recursive Self-Improvement through Reinforcement Learning

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Many works have recently emerged in code generation literature that rely on the formalism of the Godel Machine to posit a self-improving agent. This usually requires defining an approximate heuristic for the expected long term utility of a proposal. In this work, the authors (1) find that current heuristics for calculating long-term metaproductivity are slightly flawed, (2) propose a new heuristic for this task and (3) present a new algorithm which presents a more reliable estimate of the metaproductivity. Overall, the authors find that their algorithm achieves better performance on SWE-bench Lite than SWE-Agent.

### Strengths
Significance and Novelty:
 - The paper is extremely insightful and I think it will be beneficial for the broader ICLR community as well.

Clarity:
* The paper was a joy to read and I thank the authors for formally describing the algorithm as well as presenting psuedo-code in Appendix B -- this really helped better understand the details of the algorithm.

### Weaknesses
Minor dataset concerns:
 - Recent works in Software engineering benchmarking have found contaminatioon issues in SWE-Bench Lite. As such, I recommend the authors also verify results on SWE-Bench Live (https://github.com/microsoft/SWE-bench-Live).
	 - It's understandable that a full-rerun might be too expensive. Even a small-scale experiment verifying that the main result holds on `SWE-Bench: Live - Lite` (the names are getting challenging to say out loud) would be extremely useful here.


Clarity:

I caught some typos:
- `Fig. 1 Caption`: 2.38 time less -> 2.38 times less.
- `Page 3`: $(a_{final}=\arg\max \dots \mathcal{T}_{T}$ should have a closing bracket at the end.    
- `Section 3.2`: When the computational budget exceeds -> When the computational budget is exceeded,
- `Section 4.1`: `Metaproductivity-Performance Misalignment (MPM)`. This is defined in the intro as `Metaproductivity-Performance Mismatch`.


**Overall:** While the SWE-Bench Live results will help ease some concerns about data leakage, the contributions of the paper are impressive enough to warrant acceptance already.

### Questions
In the weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
4
