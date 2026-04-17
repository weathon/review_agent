# Reinforcement Learning for Evidence-Seeking Diagnostic Reasoning with Large Language Models

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Recent large language models (LLMs) excel at reasoning but often assume complete information, whereas real-world tasks, such as medical diagnosis, require iterative collections of evidence. Existing research rarely reflects this process, treating diagnosis as a one-turn task. This work explicitly formalizes this as a two-turn diagnostic paradigm and proposes reinforcement learning with diagnostic evidence-seeking rewards to guide LLMs in requesting and using evidence. We further introduce Retrieval-Augmented Generation-based Examination Simulation (RAGES), which generates realistic and plausible follow-up evidence to facilitate the process. Experiments on multilingual datasets show that (1) LLMs significantly improve diagnostic accuracy with additional evidence, (2) our model outperforms or matches larger and reasoning-enhanced baselines, and (3) RAGES generates more plausible results than pure LLM generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper formalizes medical diagnosis as a two-turn, evidence-seeking process to better mimic real-world clinical reasoning. A reinforcement learning framework guided by three novel rewards is proposed, including format, rank-sensitive diagnosis, and examination consistency, to train LLMs to request and utilize evidence effectively. To facilitate this process, the authors introduce RAGES, a retrieval-augmented system for simulating realistic follow-up examination results. Experiments on multilingual datasets demonstrate that LLMs significantly improve diagnostic accuracy with additional evidence and that the proposed model is competitive with larger, reasoning-enhanced baselines.

### Strengths
1. The paper formulates diagnosis as an interactive, two-turn evidence-seeking task, moving beyond the prevalent single-turn evaluation paradigm, which is clinically relevant.
2. The rank-sensitive diagnosis reward is designed to encourage broader differentials initially and precise diagnoses later, with a dynamic hyperparameter strategy that adapts to the amount of available evidence.
3. The proposed RAGES can generate plausible evidence by combining real data, a structured knowledge base, and LLM generation, which is shown to be more effective than pure generation in the ablation experiments.

### Weaknesses
1. While a two-round diagnosis is iterative and can gather more information, there are also cases where two rounds are not enough for diagnosis or one round is enough. The proposed framework and its design are only applicable to the two-round setting, which limits the broader applicability.
2. Only three datasets are evaluated, and one of them is in-house. Moreover, the dataset sizes are quite small, so it’s unclear whether the models can generalize well and retain their original medical capabilities e.g., in standard zero-shot medical QA tasks.
3. There is room for improvement in the presentation. It would be clearer to first provide an overview of the overall framework: explaining how the model learns to perform Turn 1, how evidence is generated between turns (RAGES), and how it learns to perform Turn 2, before diving into the specific steps.

### Questions
Since there are only 959 instances in the training set, the fact that training took 40 hours on 8 H100 GPUs seems quite long. Is the bottleneck mainly due to the LLM-as-a-judge during training?

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
3

### Summary
This paper introduces a reinforcement learning (RL) framework designed to enhance the evidence-seeking diagnostic reasoning of large language models (LLMs). Recognizing that real-world medical diagnosis often requires iterative evidence collection, the authors propose a two-turn diagnostic paradigm where an LLM first generates differential diagnoses and suggests additional tests, then refines its reasoning after receiving simulated results.
The study introduces:

Diagnostic evidence-seeking rewards—comprising format, rank-sensitive diagnosis, and examination consistency rewards—to guide structured and clinically valid reasoning.

RAGES (Retrieval-Augmented Generation-based Examination Simulation)—a method that synthesizes realistic follow-up evidence by reusing, retrieving, and generating examination results.
Empirical results show improved diagnostic accuracy and plausibility compared to strong baselines, highlighting the potential of RL-based reasoning for interactive medical diagnosis.

### Strengths
**Novel framing of multi-turn diagnosis**:
The paper explicitly formalizes diagnostic reasoning as a two-turn, evidence-seeking process. This mirrors real clinical workflows, bridging a critical gap between static, single-turn evaluations and dynamic reasoning processes.

**Well-structured reward design**:
The combination of rank-sensitive, format, and consistency rewards is both theoretically grounded and practically effective. The adaptive τ mechanism for balancing exploration (broad differential lists) and exploitation (precise diagnosis) demonstrates thoughtful design.

**Comprehensive evaluation**:
The experiments span multilingual datasets (Chinese and English), use multiple strong LLM judges (DeepSeek-R1, Qwen2.5-Max, GPT-5), and provide clear metrics for differential and final diagnostic accuracy. The ablation studies further validate design choices.

### Weaknesses
**Limited model scale and comparison scope**:
The proposed RL model (7B parameters) performs well in early reasoning but lags behind larger reasoning-tuned models (e.g., QwQ-32B, Qwen3-32B) in final diagnosis. This may limit its practical competitiveness, especially for high-stakes applications.

**Offline RAGES simulation**:
Since RAGES operates offline rather than interactively during training, the framework cannot fully exploit real-time feedback or reinforcement from generated evidence. This reduces the fidelity of multi-turn reasoning in training loops.

**Reward interpretation complexity**:
Although theoretical analysis of τ and reward functions is provided, the practical intuition behind how reward shaping influences diagnostic reasoning behavior could be better illustrated with concrete examples or case visualizations.

**Data generalizability and reproducibility**:
The dataset includes in-house Chinese pathology cases not publicly released, limiting replication. Moreover, while ethical considerations are discussed, it’s unclear whether the model generalizes beyond pathology to broader medical domains.

### Questions
**Dynamic integration of RAGES**:
Could RAGES be integrated within the RL training process (rather than offline) to enable genuine multi-turn reinforcement? How would this impact model stability and computation?

**Human-in-the-loop evaluation**:
While the study uses LLM judges, have the authors considered validation by medical experts to ensure alignment with real clinical reasoning standards?

**Scalability of the reward framework**:
The proposed rank-sensitive and consistency rewards are domain-specific. How adaptable is this framework to other domains (e.g., radiology or symptom-based triage) where evidence is less structured?

**Safety and interpretability considerations**:
Given the model’s diagnostic nature, how does the RL framework ensure safe, interpretable outputs—especially when RL-driven optimization might amplify confident but incorrect reasoning paths?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work designs a two-turn diagnostic paradigm for medical diagnosis and proposes a new reinforcement learning reward to train a diagnosing reasoning model.

### Strengths
1. The paper is well written, and the formulas and figures help with understanding.

2. The paper has a careful design of the two-turn diagnostic paradigm, the reinforcement learning reward, and RAGES. It proposes a novel architecture for medical diagnosing.

3. The proposed method improves the model over both reasoning and non-reasoning baselines.

4. The paper designs ablation studies to verify the effectiveness of the reward design and RAGES.

### Weaknesses
1. All evaluations are performed in a two-turn setting. There is no strong evidence that the proposed two-turn diagnosis has advantages over the traditional one-turn diagnosis.

2. The proposed two-turn diagnosis is a new paradigm, but it seems to consume more compute. It requires 40 hours of training on 8xH100 with only ~1k data. And there is no analysis on inference efficiency.

3. The DiffAcc of Ours-RL-7B in Table 1 seems good, while the DxAcc of Ours-RL-7B in Table 2 shows a relatively small advantage. Does this mean that the second turn in your designed dialogue fails to find out the correct disease?

### Questions
1. You could provide some evaluation results of traditional one-turn diagnosis to further prove the advantage of the proposed two-turn diagnosis method.

2. The DxAcc of Ours-RL-7B only surpasses the non-reasoning Qwen2.5-32B baseline in Table 2. You could provide more comparisons with other diagnostic models (e.g., HuatuoGPT-o1) to prove that your model is stronger.

3. You could provide more analysis on the inference efficiency and training efficiency.

4. You could give a more in-depth analysis on the DxAcc of Ours-RL-7B. It would be better if you provide a fair comparison between Ours-RL-7B and `Ours-SFT-7B'.

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
4

### Summary
The paper studies evidence-seeking diagnostic reasoning with LLMs as a two-turn process trained with reinforcement learning. The method uses three reward signals that can be checked automatically. The first enforces a structured output. The second is a rank-sensitive diagnosis reward that scores the position of the correct disease in the differential list and adapts with a temperature-like parameter tau. The third checks whether requested tests are consistent with the candidate diseases. The paper also introduces RAG-based Examination Simulation, called RAGES, which blends reuse of real case results, retrieval from a disease–exam knowledge base, and generation to produce follow-up evidence. Experiments on English and Chinese pathology cases report higher accuracy after the second turn and competitive results against stronger or larger baselines. The ablations show how tau affects list length and accuracy and how each RAGES component contributes to output quality.

### Strengths
1. The work gives a clear and formal treatment of evidence-seeking diagnosis that goes beyond the common single-turn setup in medical LLM research. The framework is two-turn by design and ties learning targets to verifiable signals, which helps analysis and training.

2. The definition of a rank-sensitive reward is careful and well argued. The paper proves two theorems that show how earlier hits and shorter lists receive higher scores and provides findings that explain how tau changes behavior across stages. The ablation with different tau settings makes these points concrete.

### Weaknesses
1. The method fixes diagnosis into a two-turn script, which narrows external validity for clinical work that often needs many turns. With only one round of evidence requests and one round of updates, the study cannot test longer-horizon skills such as sequential test planning with cost and risk, recovery from misleading results, or non-myopic revision over several cycles. This choice flows into dataset design and reward choice, which may tune the policy to a staged pattern and leaves open whether behavior would hold when three, five, or ten turns of new evidence arrive.


2. This paper is an end-to-end reliance on LLM judges for both optimization and reporting. During reinforcement learning, an LLM judge decides hits, ranks the ground-truth position for the rank-sensitive reward, and assigns the examination-plausibility bonus. The same family of models is also used at evaluation time to score differential accuracy and final diagnosis accuracy, and to assess the plausibility of examinations. Without human adjudication or agreement checks, it is hard to separate true gains in diagnostic skill from alignment with judge preferences or reward hacking.


3. The examination simulation lacks clinical validation. RAGES composes follow-up results by reusing overlapping findings, retrieving disease–exam mappings from a structured knowledge base, and prompting a strong model to fill in the rest. The quality of these simulated results and the plausibility of requested tests are judged by an LLM rather than by clinicians or by comparison with real test outcomes, which makes it unclear whether the gains reflect clinically sound choices or model-to-model agreement.

### Questions
Please see the weakness above.

### Soundness
3

### Presentation
3

### Contribution
3
