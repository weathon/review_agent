# How to Train a Leader: Hierarchical Reasoning in Multi-Agent LLMs

- Avg Score: 4.40
- Decision: Reject
- Scores: 6, 4, 4, 4, 4

## Abstract
Large Language Models (LLMs) have achieved strong performance on a wide range of complex reasoning tasks, yet further gains are often possible by leveraging the complementary strengths of multiple models.  While multi-agent frameworks can improve solution quality by leveraging multiple LLMs, existing methods are often computationally expensive, both at training and inference time.  In this work, we introduce a hierarchical multi-agent framework that addresses these challenges by training only a single leader LLM to coordinate a team of untrained peer agents.  To this end, we propose **M**ulti-agent guided **L**eader **P**olicy **O**ptimization (MLPO), a novel approach which trains the leader to evaluate and synthesize agent responses without auxiliary value networks or explicit agent feedback.  Leaders trained with MLPO exhibit improved performance not only when interacting with the agent team at inference time, but also enjoy improved performance when deployed in single-agent settings without the team.  Empirical results on BBH, MATH, and MMLU demonstrate that our framework achieves substantial performance improvements over both single-agent and multi-agent baselines.  Our results highlight the effectiveness and efficiency of training a single, flexible leader for collaborative reasoning in multi-agent LLM systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposed a novel fine-tuning scheme for llm agent, in which only a leader model is trained while the agent team maintains their own parameters. The overall scheme is essentially fine-tuning an answer aggregator for downstream tasks. The framework is validated in three tasks, including BBH, MATH, and MMLU.

### Strengths
1. The motivation of this paper, which focuses on improving the computational burden of a multi-LLM-agent system, is meaningful. 
2. The fine-tuning scheme towards multi-LLM-agent collaboration is elegant and novel 
3. The paper is well-organized to present the contributions.
4. Extensive experiments validate the efficiency of the proposed framework.

### Weaknesses
1. The trade-off of the size of the agent team should be discussed.
2. The idea of only fine-tuning a single model in mult-llm-agent is based on empirical observation without a theoretical foundation.
3. It is not certain whether the proposed scheme is also efficient for larger llms, e.g., 32B.

### Questions
1. According to Figure 2, the multi-agent team is not generating new data during MLPO. Why is this setting advocated?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a hierarchical setup where one main “leader” model coordinates a small group of helper agents. Only the leader is trained, while the others stay fixed. The leader first goes through supervised fine-tuning using synthetic data, then is further improved with a GRPO-like reinforcement objective (called MLPO). During testing, the leader communicates with the helper agents for T rounds to combine their responses into a final answer. On the task BBH, MATH, and MMLU, the approach reports gains over training‑free and training‑based baselines, and intriguingly the trained leader also improves in zero‑shot (solo) mode (Table 1; Fig. 4).

### Strengths
1. Train one coordinator that reads multiple agent drafts and outputs a consolidated answer; the inference workflow is well depicted (Fig. 1 on p. 3; Fig. 2 on p. 4).
2. On all three benchmarks the leader trained with SFT+MLPO outperforms strong single‑ and multi‑agent baselines (Table 1 on p. 6).
3. Analyses of team diversity (heterogeneous vs. homogeneous), exposure to alternative solutions, and information sharing (reasoning vs. final answers) provide insight (Appx B.2–B.3; Fig. 11–12 on pp. 21–22).

### Weaknesses
1. The evaluation scope is relatively narrow, which lacks evidence of generality. Experiments cover only BBH, MMLU, MATH with 7–9B models; there is no evidence for code, retrieval/tool use, long‑context, or multilingual settings. Claims about “collaborative reasoning” need diverse tasks; current evidence mainly supports multiple‑choice and math settings in English. Other works on the same topic, such as Mixture-of-Agents [1] and ACC-Collab [2], usually cover a wider range of task types or provide more detailed comparative analyses to better support their claims about the generality of “collaborative reasoning.”

2. Although the paper trains only a single “leader,” its inference protocol requires a team of K=3 agents and T=5 sequential interaction rounds (leader–agents–leader …), which implies ~20 generations per query. The main text fixes T=5 (Sec. 3.1) and compares methods under a ceiling of “<= 40 generations with majority vote” (Fig. 3; Appx. Fig. 7), but it does not report token‑normalized or time‑normalized efficiency (e.g., wall‑clock latency, total prompt+completion tokens, or throughput) for the proposed method versus baselines. Appendix B.3 lists training hardware/hyper‑parameters but likewise omits inference latency/throughput; the conclusion explicitly acknowledges increased context lengths, higher inference compute, and reduced parallelizability due to sequential leader–agent interactions. Together, these omissions prevent a rigorous cost–quality comparison.

### Questions
I suggest that the authors can report accuracy vs. tokens/latency for varying T, K, and leader/agent sampling to characterize real‑world cost trade‑offs; include token‑normalized comparisons for baselines (extend Fig. 3/7). Moreover, they can document baseline tuning budgets; where margins are small (e.g., MATH), report sensitivity to temperature/top‑p, number of rounds, and aggregation schemes; include paired‑seed significance tests (Table 1).

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
The main problem this paper aims to address is: Can we get most of the multi-agent gains by training only a single “leader” that coordinates untrained peer agents? To achieve that, this paper proposed Multi‑agent guided Leader Policy Optimization (MLPO), SFT to instill backtracking/self‑correction followed by a GRPO‑style RL phase to train a leader. Empiracally, this algorithm outperforms strong single‑ and multi‑agent baselines on BBH, MATH, and MMLU (Table 1, p. 6), and the leader is even stronger alone (zero‑shot, without the team) than standard GRPO‑trained models (§4.2.3; Fig. 4, p. 7).

### Strengths
-- Clear, modular architecture. §3.1 (pp. 3–4) details a two‑level hierarchy and the T‑round interaction loop; Figure 1 (p. 3) visually clarifies the leader–agents workflow and the think/answer structure the leader emits.

-- Well‑specified objective. §3.2 (pp. 4–5) formalizes MLPO as a GRPO variant that conditions the leader on agent responses, with Dr.GRPO‑style stability tweaks; the training‑data pipeline (4K agent proposals per task; filtered “easy” tasks) is explicit (§3.2, p. 5). 

-- Robustness to weak agents. Figure 5 (p. 8) and Figure 10 (p. 20) demonstrate the trained leader can override incorrect team suggestions—performance holds up even when few/none of the agents are correct. 

-- The claim"Leaders trained with MLPO exhibit improved performance not only when interacting with the agent team at inference time, but also enjoy improved performance when deployed in single-agent settings without the team" is an insight for me.

### Weaknesses
The most concern for me is the potential unfair comparison:

-- SFT transparency & potential advantage. Appendix A.1 describes how the SFT data are constructed (synthetic backtracking/self‑correction), but omits crucial statistics: dataset size, token counts, domain/source mix, and sampling rules (p. 17). This makes it hard to judge how much of the gain stems from SFT itself vs. MLPO, and it obscures fairness vs. baselines that may not receive equivalent SFT.

-- Uneven pretraining across baselines. The main text and Appendix D list training‑based baselines and their hyperparameters, but do not state that baselines like ACC‑Collab, SCoRe, GRPO, or SelectLLM were also given an equally sized, distribution‑matched SFT/backtracking set; only the Deferral Leader is said to be trained on “the same data” as MLPO (Appendix D, p. 25). This risks a pro‑MLPO bias. 

-- Budget inconsistency at inference. §4.2.2 claims parity via “at most 40 total generations” for majority vote (p. 7), but Appendix D configures SelectLLM with a 20‑vote budget and says this matches “the 20 total inference‑time generations used by our pipeline for 5 rounds of inference” (p. 25). Which one is actual setting?

-- Protocol advantages not fully controlled. The method uses structured prompts and shares rich reasoning traces between team and leader (Appendix E.1–E.2, pp. 25–27). Some baselines (e.g., SelectLLM) primarily select models rather than aggregate intermediate reasoning, and may not receive comparable intermediate evidence, making the comparison sensitive to input protocol rather than algorithmic merit.

### Questions
Besides the weakness, i have some problem to discuss with the author with respect:
-- I am wondering the training compute, can you provide GPU‑hours, effective training tokens, #steps/epochs, and KL/regularization schedules so readers can assess budget parity (Appendix B.3, p. 23; Appendix D, pp. 24–25)? Much thanks!

-- Effect of filtering “easy” tasks. Since §3.2 filters tasks where ≥75% of 4K agent responses are correct (p. 5) and Table 6 (p. 23) shows benefits, were analogous difficulty controls applied to baselines? If not, can you provide results without filtering to isolate MLPO’s contribution?

-- Zero‑shot comparison tokens. In §4.2.3 (Fig. 4, p. 7), does the Zero‑shot GRPO baseline see the same total training tokens as the SFT+MLPO leader? If not, can you provide a token‑matched comparison?

### Soundness
2

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
This paper proposes a hierarchical multi-agent framework that trains a single leader LLM to coordinate several untrained companion agents, thereby reducing training costs while improving collaborative reasoning performance. Specifically, the paper introduces MLPO (Multi-agent guided Leader Policy Optimization). It first constructs a set of examples featuring self-correction/backtracking and uses SFT to enhance the leader’s abilities in aggregation, error correction, and backtracking. Then, within the GRPO framework, it treats the agents’ solutions as additional training context to directly optimize the leader model’s capability to evaluate and synthesize multiple answers. Experimental results show that, in multi-agent LLM systems, training only a flexible leader enables efficient and effective collaborative reasoning. Moreover, the trained leader model also achieves performance gains when deployed as a single model.

### Strengths
1. Training only a single leader LLM to cooperate among several untrained companion agents significantly reduces training and maintenance costs, yet preserves the benefits of collaboration.
2. The paper introduces the MLPO training framework: construct SFT data to enhance the model’s self-correction ability, and directly optimize the leader under the GRPO framework—resulting in a relatively simple training process.
3. The trained leader model can operate in both “team collaboration” and “single-model” modes, indicating that the learned aggregation/correction strategies transfer to general reasoning ability.

### Weaknesses
1. The experiments are conducted with a collaboration setting of K = 3 agents, without verifying the effectiveness of the proposed method as the number of untrained agents increases. Since MLPO treats untrained agents as part of the environment, increasing their number would make the environment more complex.
2. Although training only a single leader model reduces training costs, the overall collaboration quality may be constrained by the leader’s capability. If the leader LLM’s task ability is weak, it could perform worse than untrained agents debating directly or a single untrained agent answering alone.
3. During training, “easy tasks” (those with a team accuracy ≥ 75%) are filtered out to focus on the leader’s coordination skills. This 75% threshold is heuristic and lacks theoretical grounding or systematic validation.
4. It remains unclear whether the trained leader exhibits strong cross-team and cross-task generalization. For example, can a leader trained with one team configuration effectively guide a different team of agents? Can a leader trained on one task domain provide strong leadership on related tasks?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new concept called a multi-agent leader, then proposes a training method to tune the leader agent, aiming to improve the summarization and answer-seeking ability from multi-agent system rollouts. They test their proposed approach in three benchmarks and achieve a significant improvement compared to baselines.

### Strengths
- The paper introduces a novel and practical hierarchical multi-agent framework. Its main advantage is its computational efficiency, as it only requires training a single "leader" model while coordinating a team of fixed, untrained peer agents. This significantly reduces the training cost and complexity compared to approaches that require co-training multiple specialized models.
- The proposed Multi-agent guided Leader Policy Optimization (MLPO) is a novel contribution. It provides an effective method for training the leader to evaluate and synthesize diverse responses from the agent team without needing auxiliary value networks or explicit agent feedback, which simplifies the overall training pipeline.

### Weaknesses
- The benchmark (MMLU, BBH, MATH) does not seem to be related to a multi-agent system. They are knowledge-intensive tasks. I hope the author can provide some clarification on why they chose these benchmarks.
- Figure 2 seems not to be related to Figure 1, where in Figure 1, the data generation pipeline is online, i.e., the leader's feedback will return to the multi-agent system for new rollouts. If the leader's feedback cannot return to the multi-agent system, the leader will be degraded to a summarizer, which reduces the novelty of the task.
- The effectiveness of the framework appears to be sensitive to the choice of the leader model. The authors note in their analysis (Section 4.4) that while MLPO shows gains, the performance increase was less significant when using Gemma-2 and Llama-3.1 as leaders compared to Qwen-2.5. This suggests the method may require a highly capable base model (like Qwen-2.5) to act as an effective leader.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2
