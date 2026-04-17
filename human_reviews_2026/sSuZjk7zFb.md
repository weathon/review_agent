# The Era of Real-World Human Interaction: RL from User Conversations

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
We posit that to achieve continual model improvement and multifaceted alignment, future models must learn from natural human interaction. Current conversational models are aligned using pre-annotated, expert-generated human feedback. In this work, we introduce Reinforcement Learning from Human Interaction (RLHI), a post-training paradigm that learns directly from in-the-wild user conversations. 
We develop two complementary methods: 
(1) RLHI with User-Guided Rewrites, which revises unsatisfactory model outputs based on users' natural-language follow-up responses,
(2) RLHI with User-Based Rewards, which learns via a reward model conditioned on knowledge of the user's long-term interaction history (termed persona). 
Together, these methods link long-term user personas to turn-level preferences via persona-conditioned preference optimization. Trained on conversations derived from WildChat, both RLHI variants outperform strong baselines in personalization and instruction-following, and similar feedback enhances performance on reasoning benchmarks. These results suggest organic human interaction offers scalable, effective supervision for personalized alignment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper presents RLHI (Reinforcement Learning from Human Interaction), a framework for post-training personalization of language models through continuous learning from real-world user interactions. The authors introduce two methods for this process: User-Guided Rewrites, which revises unsatisfactory model outputs based on users' follow-up responses, and User-Based Rewards, which ranks responses based on a reward model conditioned on long-term user personas. Both methods aim to align model outputs with users' ongoing, evolving preferences. Evaluations show that these methods outperform baseline models on personalization and instruction-following tasks on datasets like AlpacaEval 2.0 and Arena-Hard, with additional improvements in reasoning tasks. The results highlight the effectiveness of learning directly from organic human interaction in achieving personalized and contextually aligned models.

### Strengths
RLHI is an advancement in aligning language models with personalized user preferences by directly learning from ongoing user conversations. And the research question is important. By focusing on real-world human interaction, the paper pushes the field towards a more scalable, personalized approach to model alignment.

### Weaknesses
1. Complexity in Application: While the methods work well in controlled settings, they involve managing user persona conditioning, which could become cumbersome for models with large vocabularies or diverse users. Is it feasible to apply these methods at scale without significant overhead, especially when it comes to managing persona conditioning across millions of users?
2. Method Simplicity: The methods proposed, while effective, are relatively straightforward. The first method uses persona rewriting and the second applies persona-conditioned reward modeling. These are not particularly novel compared to existing techniques, such as RLHF. The simplicity of the methods raises questions about why such straightforward approaches result in such impressive outcomes (77.9 on AlpacaEval2.0 and 64.3 on ArenaHard).

### Questions
1. In Table 3, it seems that RL with User-Agnostic Rewards setting achieves a relatively good performance without a human persona. How to explain it?
2. In Table 3, RL with Rewrites from Scratch and User-Agnostic Rewards seems to have a higher LC win compared to Win on AlpacaEval2. Does it indicate that these two methods may favor shorter responses?

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
This paper introduces RLHI (Reinforcement Learning from Human Interaction), a framework that learns directly from user interaction in real conversations. RLHI includes User-Guided Rewrites and User-Based Rewards, leveraging both long-term preference and turn-level feedback to achieve continual personalized alignment. Experimental results demonstrate that RLHI not only improves performance in user-based evaluations but also achieves notable gains in instruction-following and reasoning tasks.

### Strengths
1. **Grounding in Real-World Human Preference**: RLHI learn human preference directly from natural user interactions, which provide more diverse signals than than existing curated datasets.

2. **Comprehensive Experiments and Strong Performance**: RLHI methods are evaluated across user-based evaluation, instruction-following, and reasoning tasks, achieving significantly performance improvements. Further ablation studies validate the effectiveness and contribution of each proposed component.

3. **Insightful Analysis** : This paper provides meaningful anaysis and insight about the properties of current human interaction data, demonstrating that feedback often occur in user conversations and highlighting these statics are likely to change with the evolve of LLMs.

### Weaknesses
1. The authors use Llama-3.1-8B to reconstruct responses while retaining the original user feedback (Line 269). However, since different models may behave differently, the preserved feedback may no longer be accurate or applicable, potentially misguiding the model.

2. The term “user-based rewards” can be misleading, as it suggests a reward model that scores based on user persona. In practice, the framework just samples multiple responses conditioned on the user persona and selects among them using a reward model.

3. The experiments on reasoning tasks assumes that human feedback is correct (synthesize conversations based on PRM dataset). However, human users may not be experts and could introduce errors, leading the model to learn incorrect preferences. In my view, RLHI is not well suited for objective tasks like math reasoning.

4. The proposed method heavily relies on external reward models for data filtering and for providing high-quality supervision. Its improvements on non-personalized tasks are limited (comparing RLHI with user-based and user-agnostic rewards on user-free evaluation), suggesting that much of the gain may stem from the reward model rather than from the RLHI framework.

5. I have some concerns that the paper presents the method as reinforcement learning, while the experiments are conducted solely using DPO.

### Questions
See above section

### Soundness
2

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
# Summary
A post-training framework is introduced to learn directly from in-the-wild user–LLM conversations instead of pre-annotated expert feedback. Two complementary variants are presented: **RLHI with User-Guided Rewrites** (convert user follow-ups that revise unsatisfactory answers into preference pairs) and **RLHI with User-Based Rewards** (rank candidates using a reward model conditioned on a summarized user “persona” from long-term histories). On **user-based evaluations** built from WildChat conversations, both RLHI variants improve personalization and instruction-following over strong baselines; **RLHI-User-Based Rewards** attains **77.9% length-controlled win rate on AlpacaEval 2.0** and strong Arena-Hard results under **GPT-4-Turbo** judging (Table 3). **RLHI-User-Guided Rewrites** also raises accuracy on four reasoning benchmarks from **26.5 to 31.8** despite training only on math conversations (Table 4). Ablations indicate benefits from user guidance, interaction diversity, RL over SFT, and quality filtering.

### Strengths
# Strengths
* **Realistic supervision source:** Moves beyond curated annotator pipelines to leverage in-situ signals (SAT/DSAT, rewrites, follow-ups).

* **Compelling empirical improvements (user-based & standard).**

    • On **WILDCHAT USEREVAL**, RLHI variants improve personalization and overall user preference (Table 2).

    • On **AlpacaEval 2.0** and **Arena-Hard**, RLHI-User-Based Rewards achieves 77.9% LC win and 83.4 Arena-Hard win (Table 3).

    • On reasoning, **RLHI-User-Guided Rewrites** lifts average accuracy **26.5 to 31.8** across Minerva, OlympiadBench, GPQA, and   MMLU-Pro (Table 4).

### Weaknesses
# Weaknesses

1. **Evaluator and judge bias**

   * *Example (benchmarks):* MT-Bench uses **GPT-4o** as judge; AlpacaEval 2 and Arena-Hard use **GPT-4-Turbo**, all closed-source LLM judges. 
   * *Example (user eval):* RLHI reports user-based results judged by **OpenAI o3** on WILDCHAT USEREVAL, again relying on a closed-source judge. 

2. **Persona inference via LLM**

   * *Example:* RLHI **prompts an LLM to summarize each user’s persona** from long-term histories to steer both training (persona-conditioned DPO) and inference—useful but introduces risk of hallucination/leakage without audits.

### Questions
## What Could Be Improved.
1. **Add strong baselines**
* **Teacher-distill baseline**: SFT on *ChatGPT/strong model* responses for the same turns. This tests whether mined preferences beat a simple, distillation.
* **Ground-truth tasks**: Include **GSM8K**, **MATH**, **MBPP/HumanEval** with exact-match verify gains on **verifiable** reasoning/coding.

2. **Robust judging**
* Evaluate with **multiple independent judges** (closed and open), randomize templates, and report **win-rate consistency** across judges. Include **anti-style** prompts (brevity/conciseness constraints).

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
The paper presents RLHI, a post-training paradigm that learns directly from real-world user interactions to align large language models with evolving, long-horizon preferences. RLHI converts organic conversations into preference signals via two complementary routes: (1) User-Guided Rewrites, which transforms user follow-ups into improved revisions to form preference pairs; and (2) User-Based Rewards, which ranks multiple candidates using a reward model conditioned on a natural-language user persona. Over user-based evaluation, standard instruction-following benchmarks, and reasoning suites, RLHI delivers consistent gains over baselines. Ablations highlight the importance of leveraging explicit feedback, RL over SFT on interaction data, and the effect of quality filtering.

### Strengths
1. The authors clearly pinpoint an important gap that current alignment underutilizes rich multi-turn user feedback. They leverage long-horizon personas and turn-level corrections to drive personalized learning.

2. The paper uses two complementary, lightweight routes (user-guided rewrites and user-based rewards) under a DPO-style objective, making the approach easy to integrate and broadly usable even with sparse explicit ratings.

3. It demonstrates effectiveness on both user-based evaluations and standard public leaderboards, with consistent gains and informative ablations across diverse tasks and domains.

### Weaknesses
1. Limited novelty. The user-guided rewrites track prior WildChat-style pipelines [1,2]; the user-based rewards overlap with user-conditioned reward models [3,4] and conditioning LM/RM/judges on user summaries [5,6,7]. If those lines cannot solve the paper’s target problem, the manuscript should specify why; otherwise, stronger baseline comparisons are needed.

2. Human study is under-specified. Results are summarized in one sentence with no details.

3. Reasoning part uses PRM data to simulate conversations without analyzing its limitations. Also, if personas are not used at inference, the setup collapses to vanilla DPO, weakening the core claim.

[1] WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback. arXiv:2408.15549.

[2] User Feedback in Human-LLM Dialogues: A Lens to Understand Users But Noisy as a Learning Signal. arXiv:2507.23158.

[3] Personalized Language Modeling from Personalized Human Feedback. arXiv:2402.05133.

[4] Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning. arXiv:2408.10075.

[5] RLPF: Reinforcement Learning from Prediction Feedback for User Summarization with LLMs. arXiv:2409.04421.

[6] Learning to summarize user information for personalized reinforcement learning from human feedback. arXiv:2507.13579.

[7] Can LLM be a Personalized Judge? arXiv:2406.11657.

### Questions
1. What's the core novelty comparing with the papers listed in the weaknesses?

2. Please provide more details about the human study and the experiments on reasoning benchmarks. (See Weaknesses)

3. How does the system adapt to truly new users? Do the identified preference dimensions and feedback types generalize under cold-start conditions?

### Soundness
2

### Presentation
3

### Contribution
2
