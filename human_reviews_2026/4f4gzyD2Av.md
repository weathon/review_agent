# RE-Searcher: Robust Agentic Search via Goal-oriented Planning and Self-reflection

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Large language models (LLMs) excel at knowledge-intensive question answering and reasoning, yet their real-world deployment remains constrained by knowledge cutoff, hallucination, and limited interaction modalities. Augmenting LLMs with external search tools helps alleviate these issues, but it also exposes agents to a complex search environment in which small, plausible variations in query formulation can steer reasoning into unproductive trajectories and amplify errors. We present a systematic analysis that quantifies how environmental complexity induces fragile search behaviors and, in turn, degrades overall performance. To address this challenge, we propose a simple yet effective approach to instantiate a search agent, RE-Searcher. During search, RE-Searcher explicitly articulates a concrete search goal and subsequently reflects on whether the retrieved evidence satisfies that goal. This combination of goal-oriented planning and self-reflection enables RE-Searcher to resist spurious cues in complex search environments and perform robust search. Extensive experiments show that our method improves search accuracy and achieves state-of-the-art results. Perturbation studies further demonstrate substantial resilience to noisy or misleading external signals, mitigating the fragility of the search process. We believe these findings offer practical guidance for integrating LLM-powered agents into more complex interactive environments and enabling more autonomous decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
1. This paper focuses on agentic search, analyzing the complexity and variability of search environments as well as the fragility of search agents. It argues that search agents should incorporate explicit goal clarification and reflection on search results like humans.

2. This work introduces RE-Searcher, which integrates explicit goal clarification and reflection on search results into the agent loop. It further designs a reflection reward, employing a stronger LLM-as-Judge to assess reflection quality, and trains the search agents using the GRPO algorithm.

3. Extensive experiments across in-domain and out-of-domain QA benchmarks show that RE-Searcher consistently outperforms recent agentic search baselines (e.g., Search-R1, ZeroSearch, O2-Searcher), with notable robustness under perturbations.

### Strengths
1. Good motivation and preliminary analysis: The paper clearly identifies the challenges of environmental variability and inherent fragility faced by search agents, and conducts preliminary experiments to validate these issues.

2. Clear methodological design: In response to the identified problems, the paper extends the agent loop and proposes an effective training strategy. The methodological descriptions and figures are well-presented and easy to follow.

3. Strong results: The experiments compare against a wide range of baselines and achieve the best overall performance. Extensive ablation and robustness analyses further demonstrate how effectively the proposed method addresses the original problem.

### Weaknesses
1. In Sections 2.1 and 4.4, the paper examines the inherent fragility of search agents. While I acknowledge the stochasticity in search agent outputs, sampling only twice per question may not be sufficient to support such a conclusion.

2. As I understand it, there may be an issue with Equation (2). The function FM is not explicitly defined, and I assume it is a 0–1 function similar to EM. When EM equals 1, if the format is correct (FM=1), the reward r_{em_format}=1-0.2\*1=0.8. However, if the format is incorrect (FM=0), the reward becomes r_{em_format}=1-0.2\*0=1, which paradoxically yields a higher reward for incorrect formatting.

3. I think using LLM-as-Judge to provide reward supervision for the newly introduced reflection action is a very good choice. However, I noticed in Equation (3) that the reflection reward is added to the outcome reward. This makes me wonder:  since reflection is a process-level action, why not treat it as a process reward instead? Theoretically, process rewards are more fine-grained and could improve sample efficiency. In addition, how was the weight of 0.1 in Equation (3) determined?

4. Based on your main experimental results (Table 2), since all datasets are open-domain QA, why does the method achieve substantial improvements on some datasets but not on others? Have you analyzed the reasons behind this phenomenon?

### Questions
Please refer to Weaknesses. I am open to changing the score based on your rebuttal.

### Soundness
2

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
3

### Summary
This paper addresses the fragility of LLM-based search agents in complex environments where small query variations can lead to drastically different retrieval results and erroneous reasoning trajectories. The authors propose RE-Searcher, an agentic framework designed to improve robustness by integrating goal-oriented planning and self-reflection. During its search process, the agent explicitly articulates a search goal, retrieves information, and then reflects on whether the retrieved evidence successfully satisfies that goal. The agent is trained using GRPO with a composite reward signal that includes factual correctness, format adherence, and a reflection accuracy score provided by an "LLM as Judge". Experimental results show that RE-Searcher achieves state-of-the-art performance on several question-answering benchmarks and demonstrates significantly improved robustness against search fragility and external query perturbations.

### Strengths
- The motivation is clear. The preliminary analysis in Section 2.1 quantifies output stochasticity across different model scales. This empirically demonstrates a critical instability problem that fundamentally limits achievable performance.
- The proposed method is intuitive and clearly described in Section 3. The structured generation template clearly defines three discrete actions.
- Robustness Analysis is novel. The Pass@2 analysis in Section 4.3 demonstrates that RE-Searcher substantially reduces the random-right ratio.

### Weaknesses
- The novelty of core concepts is limited. The core ideas of "planning" (decomposing a problem) and "reflection" (evaluating retrieved information) are well-established concepts in agentic AI and RAG literature. 

- The proposed method introduces multiple new components for each search iteration: an explicit "think" step, a "goal" generation step, and a "reflect" step. This multi-step process seems guaranteed to significantly increase the number of generated tokens and overall inference latency compared to simpler RAG or SFT baselines.

- While the paper ablates the reflection reward, there is no ablation isolating the contribution of goal-oriented planning versus self-reflection. The method combines both mechanisms, but their individual contributions remain unclear.

### Questions
See weaknesses above.

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
This paper addresses a critical challenge in LLM augmented with search tools: agent LLMs tend to generate variable and inconsistent results.  Authors first conducts a systematic analysis showing that semantically plausible query variations can lead to dramatically divergent and unreliable search trajectories. To mitigate this, this paper introduces RE-Searcher to explicitly integrate goal-oriented planning and self-reflection into the reasoning process. The method is first trained using SFT, followed by GRPO with a composite reward.

### Strengths
- The paper is well-written and easy to follow.
- The intuitive integration of goal-setting and binary self-reflection is easy to implement.
- Authors conduct comprehensive evaluation across divers datasets. Clear ablation studies shown the contribution of the reward design.

### Weaknesses
- The observed instability of success-rate in Sec. 2.1 may largely stem from the inherent differences in model capacity.
- In Sec. 2 and Sec. 4.2, authors primarily assess models by measuring the consistency of accuracy across multiple runs. However, in agentic search settings, the dominant source of instability stem from variations in search queries and inaccurate documents. The paper tackles the challenge with their search and reflection design, but lacks analysis of how the search queries changes before and after training, particularly whether the proposed goal-oriented planning and self-reflection lead to more consistent or robust query formulation.
-  According to my knowledge, the instruction employed in this work is not particularly challenging for a model like Qwen2.5-3B to follow. The necessity of warm-up needs further clarification.

### Questions
- The coefficient of the MBE reward is even smaller than that of the format reward. How does the model perform with larger MBE rewards?
- How does the model perform after the warm-up?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies a real and under-discussed problem in agentic / search-augmented LLMs: small, plausible variations in search queries can push the agent onto a bad trajectory and the agent often fails to recover. The authors first quantify this “search fragility” by showing (i) high “random-right” rates across two runs and (ii) large drops in similarity after tiny query perturbations. To address this, they propose RE-Searcher, an agent that (1) explicitly states a search goal before issuing a query, (2) reflects on whether retrieved evidence satisfies that goal, and (3) is RL-trained (GRPO) with a mixed reward (format + factual + LLM-as-judge reflection) to keep the model in the “plan–search–reflect–answer” loop.

### Strengths
1. The work identifies a concrete failure mode of search agents, namely that a slightly different but still reasonable query can lead to very different retrieved evidence, which in turn damages answer quality.
2. The proposed template, “state goal → search → reflect → continue or answer,” is simple, explicit, and can be plugged into existing search-RL systems without complicated architecture changes.
3. Experiments on both 3B and 7B models, on in-domain and out-of-domain QA, plus ablations on the reward components, make the paper believable.

### Weaknesses
1. Training uses GPT-4o-mini (or a similar model) to score reflection quality. This supervision is non-trivial. It is unclear how much of the final robustness comes from this stronger teacher, rather than from the agent’s own structure.
2. The method may require several search steps, longer contexts, and an external LLM during training. At inference time, does the model always reflect, or can it stop early? A comparison with Search-R1 using the same search budget would be useful.

### Questions
1. Why was 0.1 chosen as the reflection reward weight? Was any systematic hyperparameter search conducted?

2. Can you propose or experiment with a self-supervised reflection mechanism that avoids dependency on external LLMs?

3. Why are RE-Searcher’s improvements smaller on some datasets (e.g., NQ) compared to others?

4. How much does the method increase computational cost relative to baselines? How many additional LLM calls per query are required?

### Soundness
3

### Presentation
3

### Contribution
2
