# DEBATE: A Large-Scale Benchmark for Evaluating Opinion Dynamics in Role-Playing LLM Agents

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Accurately modeling opinion change through social interactions is crucial for understanding and mitigating polarization, misinformation, and societal conflict. Recent work explores simulating opinion dynamics with role-playing LLM agents (RPLAs)—language models assigned human-like personas that engage in multi-turn, multi-agent opinion exchange. However, existing RPLA simulations often produce unnatural group behaviors (e.g., premature consensus) and lack empirical benchmarks for evaluating alignment with real human interactions. We introduce DEBATE, the first large-scale benchmark for evaluating the authenticity of opinion dynamics in multi-agent RPLA simulations. DEBATE contains 37,357 messages from 2,792 U.S.-based participants who engaged in multi-player, multi-round conversations across 107 controversial topics, reporting both public messages and private beliefs. We simulate these conversations using various LLMs and introduce multi-level evaluation metrics (at the utterance, individual, and group levels) to assess behavioral alignment between humans and RPLAs. Our analyses reveal key behavioral gaps: RPLA groups exhibit stronger opinion convergence and belief drift than humans, and individual agents show more systematic shifts in response to social influence. Ablation studies further highlight the importance of private self-reported opinions in shaping realistic agent behavior. Additionally, while supervised fine-tuning improves surface-level metrics (e.g., ROUGE-L, message length), it falls short on deeper alignment (e.g., semantic and stance alignment). DEBATE enables benchmarking of simulated opinion dynamics and supports future research on aligning multi-agent RPLAs' simulations with realistic human interactions. The dataset and codebase will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DEBATE, a large-scale dataset of multi-player, multi-round discussions across controversial debate topics which can be used to evaluate the effectiveness of LLM agents at simulating opinion dynamics. They design ways to measure alignment between agent and human opinion evolution, then use this for experiments assessing the realism of different base models and agent setups. They find that LLM agents differ from humans in key ways, including stronger opinion convergence and positive belief shift, which cannot be fixed simply by improved context or fine-tuning on human utterances.

### Strengths
- The dataset does seem to be both novel and relevant for the very timely problem of evaluating LLM agent-based social simulations. Having both internal stances and external utterances, as well as the diverse range of conversation topics and metadata, makes this a promising dataset.
- The analysis RQs are interesting and naturally emerge from the dataset construction - in its current form, I don’t think you need to frame the paper as mainly a dataset paper, as the analyses are also pretty interesting. The stance homogenization and positive belief shift findings do seem noteworthy and the right kinds of things to target for improvement in future RPLA work.
- Careful metric design, with human validation whenever necessary.

### Weaknesses
- My main concern is that there is not much quality validation of the human conversations in the dataset for a benchmark paper. It appears that all on-topic utterances that were a part of completed interactions were used for evaluation. More quality validation of the resulting dataset might be useful - I’m concerned that, as crowdworkers were completing a single-episode task with no incentive for honestly reporting preferences, there might be significant numbers of low-quality interactions that would preferably be filtered out.
- Similarly, there is not much analysis of what kinds of interactions are present in the dataset, as well as any observed benefits to conversation diversity/quality that stem from the new benchmark attributes.
- While the metrics are well-motivated, they seem to have very limited discriminative power (in Table 1, the semantic similarity/stance difference metrics show very little variance between different models, despite the diversity of the models evaluated), which may limit the dataset’s utility as a benchmark.

### Questions
- Why are LLMs off-topic so often? Even 78%, the highest on-topic rate in Table 1, seems lower than I would expect from recent LLMs.
- I’d advise you put any tables that are referenced in the main paper into the main paper, rather than the appendix (e.g. Tables 13-14), with your extra page for camera-ready. The current layout is a bit difficult to follow.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents DEBATE, a large-scale empirical benchmark for evaluating opinion dynamics in multi-agent role-playing language models. The motivation is that existing work lacks human-grounded benchmarks for studying how LLM agents evolve opinions through social interaction.

The authors collect 37,357 messages from 2,792 U.S. participants engaging in multi-round, multi-party discussions on 107 controversial topics. The dataset includes both public messages and private beliefs, enabling evaluation of model-human alignment at the utterance, individual, and group levels.

Multiple LLMs are tested under three simulation settings, with quantitative metrics measuring semantic similarity, stance alignment, message length, and topical relevance.

Results show that current RPLAs reproduce some human-like behaviors at the utterance level but diverge at deeper cognitive and social levels. Simulated groups display stronger opinion convergence, positive belief drift, and more systematic individual shifts.

The authors position DEBATE as the first benchmark enabling systematic and multi-level evaluation of simulated opinion dynamics, aiming to support future research on aligning multi-agent LLMs with realistic human social behavior.

### Strengths
The paper addresses a well-motivated and underexplored problem that existing RPLA simulations often display unnatural group behavior, such as premature consensus, and lack a benchmark to measure how human-like their opinion dynamics are.

1. The data collection is the paper’s strongest contribution. The authors conduct tightly controlled multi-party, multi-round human discussions that capture both public messages and private beliefs, yielding over 37K utterances from about 2,800 U.S. participants across 107 topics. The inclusion of private self-reports alongside public statements adds clear value and provides a solid foundation for studying social alignment in RPLAs.

2. The metric design is another strength. The paper defines quantitative measures of semantic similarity, stance alignment, and opinion convergence across utterance-, individual-, and group-level evaluations, offering a comprehensive view of both linguistic and behavioral fidelity.
	
3. A diverse set of LLM families and sizes are compared, revealing consistent behavioral gaps: stronger opinion convergence, positive belief drift, and more systematic individual shifts. Although these patterns align with prior intuition, the paper verifies and quantifies them empirically, establishing a credible and reproducible baseline for future research.

### Weaknesses
1. The dataset is based on controlled four-person discussions with enforced turn-taking. While this setup ensures structured and clean data, it limits the natural flow of interaction and may not reflect opinion evolution in open or large-scale social settings.

2. The three simulation modes—Next Message Prediction, Tweet-guided Simulation, and Full Conversation Simulation—lack clear theoretical separation. Clarifying the motivation and analytical purpose of each mode would make the framework more convincing.

3. The paper highlights the importance of private self-reported beliefs for modeling realistic behavior. However, if these beliefs are only inserted into prompts, their actual influence on generated content is neither quantified nor discussed. Showing concrete examples or quantitative evidence of how private beliefs affect agent behavior would make this claim stronger.

4. In practice, the benchmark measures how human-like a specific LLM behaves during opinion exchange rather than assessing whether an RPLA simulation as a whole resembles human interaction. Its scope is therefore narrow, and the paper does not explore cases where different roles are played by distinct LLMs.

### Questions
1. Could the authors clarify the motivation behind enforcing turn-taking and explain how this design choice contributes to the study’s objectives?

2. Could the authors clarify whether the benchmark evaluates an LLM’s ability to exhibit human-like opinion dynamics or the overall human-likeness of a multi-agent opinion-exchange simulation?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DEBATE, a new large-scale benchmark for evaluating the realism of role-playing LLM agents (RPLAs) in opinion dynamics simulations. The authors argue that existing RPLA simulations often show unnatural behaviors (like premature consensus) and lack a solid baseline against real human interactions.

To fix this, the authors built the DEBATE dataset. They collected data from 2,792 U.S. participants discussing 107 controversial topics. A core contribution is that the dataset captures both the public messages (tweets, chats) and the private, reported beliefs of the participants (before and after the chats). Using this benchmark, the paper tests several LLMs and finds significant gaps between AI and human behavior:
- RPLA groups converge on opinions far more strongly than humans.
- RPLAs show a systematic "belief drift," sometimes towards false beliefs.
- Simple supervised fine-tuning (SFT) improves surface-level metrics (like message length) but actually harms deeper alignment (like stance and semantic similarity).

### Strengths
- Important and Novel Problem: The paper tackles a timely and critical question. As LLMs are increasingly used to simulate social interactions, we urgently need rigorous ways to check if these simulations are socially realistic. DEBATE is, to my knowledge, the first large-scale empirical benchmark specifically for multi-agent opinion dynamics.

- Data Collection: The study's design is its biggest strength. Separating 'public speech' from 'private belief' is a major contribution. It allows evaluation to move beyond simple text mimicry to the core of opinion dynamics: the gap between what people say and what they actually think.

- Well-Constructed Benchmark. The benchmark itself is solid. The dataset is large and diverse (see Appendix E). The topic selection (mixing 'Depth' topics with ground truths and 'Breadth' topics) is smart. The three simulation modes, which test agents with decreasing levels of human context, provide a good way to measure agent autonomy.

- Clear and Impactful Findings. The paper's conclusions are clear and important for the field. The finding that RPLAs over-converge (Fig 2d, 3b) and that SFT fails (or even hurts) deep alignment (Appendix N) are major takeaways. This suggests that simple imitation is the wrong approach for this problem.

### Weaknesses
- Lack of Deeper Discussion on SFT Failure: The SFT results (Appendix N) are fascinating but underexplored. The paper states that naive SFT fails, but doesn't deeply explore why. Is it because the model learns to imitate an "average" human, losing individual diversity? Is the next-token prediction objective simply wrong for modeling a latent process like belief update? The paper would be stronger if it discussed alternative training objectives (e.g., RL based on realism, or explicit belief-tracking models).

- Limits of LLM-as-Evaluator: The evaluation relies on gpt-4o-mini to classify topic relevance and stance (Appendix G). While the authors report 90% accuracy on a validation set, this creates an "LLM evaluating an LLM" loop, which can be risky. The authors should have reported the Inter-Annotator Agreement (IAA) among their human labelers. This would tell us how subjective or difficult the labeling task is in the first place.

- Scope limits: The sample is US-only and dyadic; generalization to other cultures/platform structures remains uncertain.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
