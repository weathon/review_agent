# Hilbert: Recursively Building Formal Proofs with Informal Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Large Language Models (LLMs) demonstrate impressive mathematical reasoning abilities, but their solutions frequently contain errors that cannot be automatically checked. Formal theorem proving systems such as Lean 4 offer automated verification with complete accuracy, motivating recent efforts to build specialized prover LLMs that generate verifiable proofs in formal languages. However, a significant gap remains: current prover LLMs solve substantially fewer problems than general-purpose LLMs operating in natural language. We introduce Hilbert, an agentic framework that bridges this gap by combining the complementary strengths of informal reasoning and formal verification. Our system orchestrates four components: an informal LLM that excels at mathematical reasoning, a specialized prover LLM optimized for Lean 4 tactics, a formal verifier, and a semantic theorem retriever. Given a problem that the prover is unable to solve, Hilbert employs recursive decomposition to split the problem into subgoals that it solves with the prover or reasoner LLM. It leverages verifier feedback to refine incorrect proofs as necessary. Experimental results demonstrate that Hilbert, substantially outperforms existing approaches on key benchmarks, achieving 99.2\% on miniF2F, 6.6\% points above the best publicly available method. Hilbert achieves the \textbf{strongest known result} from a publicly available model on PutnamBench. It solves 462/660 problems (70.0\%), outperforming proprietary approaches like SeedProver (50.4\%) and achieving a 422\% improvement over the best publicly available baseline. Thus, Hilbert effectively narrows the gap between informal reasoning and formal proof generation. Code is available at ~\url{https://github.com/Rose-STL-Lab/ml-hilbert}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a framework for ATP which utilizes a LLM for informal reasoning and another LLM for writing the proofs in formal language. The output of the reasoning model is the base for writing the formal proof. Through extensive experiments on miniF2F and PutnamBench, the paper demonstrates that its proposed framework can improve the accuracy of base theorem provers by a significant margin. SoTA LLMs such as Goedel Prover V2 is utilized for theorem proving.

### Strengths
Results are impressive on Putnam Bench and miniF2F.

Authors "plan to release the source code and other artifacts upon publication.

I believe the method will be a useful contribution for the community. This paper seems to me as a good implementation for combining informal and formal reasoning.

Paper is well written.

### Weaknesses
As far as I see in the paper, token budgets are only reported in Table 3 for the ablation of the retrieval module. This is good to understand the effect of retrieval module, however, token budgets are not reported for the main results. This lack of reporting makes the comparisons unclear to me and perhaps unfair.

The paper's method utilizes a reasoning model to reason about the proof in conjunction with another LLM. Hence, comparing its eventual accuracy directly with the accuracy of a theorem prover that does not benefit from informal reasoning with a large reasoning model seems to me as an incomplete comparison. I would like to see the token budgets reported across the board for all experiments.

One important question, in my view, is the effect of reasoning model in small number of sampling budgets. For example, consider the accuracy of Goedel Prover V2 8B pass 32 as the base. We have the option to use Hilbert pass 32 to improve the base accuracy. We can also increase the sampling budget of the base model to 64 or 128 without using the Hilbert. What would be the resulting accuracies and the total token budgets in each scenario? It might be the case that Hilbert leads to better accuracy and lower token budget. From the reported results, I could not infer whether Hilbert has such an advantage or not.

I suggest authors perform an ablation study on different modules of their framework not just about the retrieval. 

I suggest token budgets be reported separately for each component in the framework, e.g., reasoning model, theorem prover, etc.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

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
This paper introduces HILBERT, a framework designed to bridge the performance gap between the strong informal reasoning of general-purpose LLMs and specialized prover LLMs. The system includes a general-purpose "Reasoner" LLM and a specialized "Prover" LLM. When a direct proof attempt by the Prover fails, HILBERT leverages the Reasoner to generate a formal proof sketch, which is then broken down into simpler subgoals. These subgoals are addressed recursively through a hierarchy of strategies: a direct attempt by the Prover, a "shallow solve" by the Reasoner, or further decomposition. This method achieves SOTA performance on both MiniF2F and PutnamBench.

### Strengths
1. HILBERT achieves SOTA performance on MiniF2F (99.2%) and PutnamBench (70.0%), significantly outperforming the existing method.
2. The framework's design is intuitive and demonstrably effective. Using a powerful informal reasoner to guide a formal prover via recursive subgoal decomposition is a well-motivated approach to tackling complex proofs.
3. The authors provide a comprehensive evaluation, testing on two standard benchmarks with different combinations of Reasoner and Prover models. The ablation studies validate the contributions of the retriever component and the recursive decomposition strategy. The inference budget analysis provides a better understanding of the performance and cost.

### Weaknesses
1. The core ideas of using informal reasoning to guide formal proof (e.g., DSP [1], LEGO-Prover [2]) and recursive decomposition (e.g., POETRY [3], ProD-RL [4]) are not new. The paper fails to clearly articulate the conceptual difference between HILBERT's decomposition strategy and these existing recursive methods. The contribution appears to be more a highly effective integration and scaling of these ideas with SOTA models rather than a fundamentally new algorithm.
2. The paper lacks a failure analysis. Does the Reasoner generate flawed informal proofs? Does the sketch-to-subgoal extraction fail? Or does the Prover fail to solve a seemingly simple, decomposed subgoal? Analyzing these failure modes is crucial for understanding the framework's limitations and guiding future work.
3. The reported SOTA performance is achieved at a significant computational cost. As shown in Figure 3, the best configuration requires up to 4.5K calls to Gemini 2.5 Pro and 11.3K total LLM calls on MiniF2F. The framework's performance is also highly sensitive to the Reasoner's capability, with the weaker Gemini 2.5 Flash showing a substantial performance drop and requiring even more LLM calls (>40K).
4. The analysis comparing HILBERT to baselines feels incomplete and potentially misleading. Table 2 reports the Goedel-Prover-V2-32B baseline at 13.4% pass@184 on PutnamBench, while Table 1 uses pass@8192 for the same model on MiniF2F. Given that HILBERT's performance relies on an enormous number of calls, this comparison seems unfair.
5. All computational cost analyses are only for MiniF2F. The paper should include the computational cost required to achieve the 70.0% on PutnamBench.

[1] Albert Q Jiang, Sean Welleck, Jin Peng Zhou, Wenda Li, Jiacheng Liu, Mateja Jamnik, Timothée Lacroix, Yuhuai Wu, and Guillaume Lample. “Draft, sketch, and prove: Guiding formal theorem provers with informal proofs.” ICLR 2023

[2] Wang, Haiming, Huajian Xin, Chuanyang Zheng, Zhengying Liu, Qingxing Cao, Yinya Huang, Jing Xiong et al. "LEGO-Prover: Neural Theorem Proving with Growing Libraries.” ICLR 2024

[3] Lin, Haohan, Zhiqing Sun, Sean Welleck, and Yiming Yang. “Lean-star: Learning to interleave thinking and proving.” ICLR 2025

[4] Dong, Kefan, Arvind Mahankali, and Tengyu Ma. "Formal theorem proving by rewarding llms to decompose proofs hierarchically.” arxiv

### Questions
1. What is the primary conceptual difference between HILBERT's recursive decomposition and the strategies used in POETRY and ProD-RL?
2. Could the authors provide a failure analysis for the problems HILBERT failed to solve? What were the primary bottlenecks or failure modes?
3. What is the computational cost (total Reasoner/Prover calls and tokens, similar to Figure 3) for the 70.0% result on PutnamBench?
4. Why use a pass@184 baseline for Goedel-Prover-V2 on PutnamBench? Is it possible to Evaluate pass@8192 of Goedel-Prover-V2 on PutnamBench?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a new agentic framework named HILBERT. It aims to bridge the gap between the informal mathematical reasoning capabilities of Large Language Models (LLMs) and the rigorous verification capabilities of formal proof systems like Lean 4. HILBERT orchestrates four components: a 'Reasoner' for informal reasoning (e.g., Gemini 2.5 Pro), a 'Prover' to generate formal proofs (e.g., Goedel-Prover-V2), a Verifier, and a theorem Retriever.

When the Prover fails to find a proof alone, HILBERT recursively decomposes the problem into subgoals, attempting to construct and refine the proof using feedback from the Verifier and Retrieval-Augmented Generation (RAG) via the Retriever. Experimental results show that HILBERT achieves a 70.0% pass rate on the particularly challenging PutnamBench, significantly outperforming existing proprietary methods (SeedProver, 50.4%) and public baselines (13.4%), thereby reporting state-of-the-art (SOTA) performance.

### Strengths
**Exceptional Experimental Performance (SOTA Achievement):** The primary strength of this research lies in its achievement of a remarkable 70.0% pass rate on the highly challenging PutnamBench, surpassing the existing SOTA by approximately 20 percentage points. This is an extremely strong engineering achievement in the field of formal theorem proving and represents a valuable result for the community.

**Effective Framework Orchestration:** The proposed method effectively combines the strengths of multiple components with different capabilities (Reasoner, Prover, Verifier, Retriever). In particular, the agent design—which coordinates "recursive subgoal decomposition" triggered by Prover failure, a "refinement loop" using Verifier feedback, and "contextualization" via RAG—is sophisticated. It demonstrates a realistic and effective strategy for solving complex problems that a single model cannot solve alone.

### Weaknesses
**Lack of Theoretical Justification:** Although HILBERT is proposed as a controllable framework, it lacks theoretical analysis or guarantees as to "why this approach works." In particular, the paper does not provide a theoretical background for why the core "recursive subgoal decomposition" is effective, what problem structures it is suited for, or where its limitations might lie. This absence gives the impression that the work lacks academic depth relative to its engineering success.

**Limited Novelty:** The proposed framework appears to be primarily a sophisticated "combination" of ideas seen individually in existing research (e.g., DSP, POETRY, which are cited in the paper), such as subgoal decomposition, Verifier feedback, and RAG. There is no significant novelty in the individual technical elements, making it somewhat unclear what HILBERT's unique core "Technical Advance" is when compared to prior work.

**Dependency on Closed Models:** The experiments achieving SOTA rely on a proprietary, closed model (Gemini 2.5 Pro) as the Reasoner. This not only severely undermines the reproducibility of the research, but it also risks effectively closing off the promising future research avenue suggested in the paper's conclusion—"using the generated proofs to further train the agent (e.g., via reinforcement learning)"—to the open-source community. This is a significant drawback.

### Questions
**Regarding novelty**: Concerning the novelty of the method, the combination of subgoal decomposition, RAG, and Verifier feedback has already been explored in prior research. What, specifically, do you consider to be the core "Technical Novelty" of the HILBERT framework that is not present in this existing work?

**Regarding the theoretical background**: The effectiveness of subgoal decomposition is demonstrated empirically. However, are there any theoretical or empirical insights into what types of problems (or logical structures) this recursive decomposition is particularly effective (or ineffective) for? Furthermore, how were the failure modes of decomposition (e.g., decomposition into incorrect subgoals, getting stuck in an unsolvable state) addressed?

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
4

### Summary
This paper introduced Hilbert methodology to decompose  the mathematical problems into subgoals and try proving such subgoals in both formal theorem proving (Lean 4) system and informal approach (LLM-based reasoning). Their system consists of four components: a reasoner of general-purpose LLM to write informal theorem proof, a prover that writes formal proof given informal one, a Lean-based verifier that checks the correctness of the formal statement, and a retriever that retrieves relevant theorems from Mathlib. In addition to utilizing both formal and informal reasoning by the reasoner and Lean-based verifier, they use retrieved theorems to enrich the reasoner.

In the experiments, they introduced models with Gemini 2.5 Pro/Flash for the reasoner and
Goedel-Prover-V2 for the prover. Gemini 2.5 Pro with Goedel-Prover-V2-32B, achieving a 99.2% pass rate. Needless to say, this is far better than those achieved by the Gemini 2.5 Pro model alone. They also achieved 70.0% in the PatNum bench.

It is difficult to fairly position this paper because the idea of decomposing the theorem and using both formal and informal reasoning is already examined in some of the previous studies [Baba 2025, Zhou 2025b] as the authors discussed in the related work section. Although, Hilbert system achieved remarkable scores, it can be also regarded as a combination of existing ideas and best performing models.

### Strengths
- S.1: Achieving very close to the previous best model result (99.6% by SeedProver) of 99.6% in MiniF2F and better than the SeedProver in PutnamBench.
- S.2: The effectiveness of the theorem retrieval is examined in the ablation.

### Weaknesses
- W.1: It seems that most of the core idea (dual use of formal and informal reasoning) comes from several existing studies [Baba 2025, Zhou 2025b] as the authors discussed in the related work section.
- W.2: Similarly, authors didn’t compare their model with the similar formal and informal reasoning models [Baba 2025, Zhou 2025b].
- W.3: Authors tried their system with the best performing external models that often prevent fair cost/performance comparison with the previous systems.
- W.4: Compared to [Baba 2025, Zhou 2025b], the retrieval module can be new. However, the authors do not present the effectiveness of this module in the PatnumBench.

### Questions
- I'd like to know how the retrieval is effective within the PatnumBench. Can you present the ablation study to turn on/off the retrieval module in PatnumBench?

### Soundness
3

### Presentation
3

### Contribution
3
