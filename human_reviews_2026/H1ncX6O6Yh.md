# Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games

- Decision: Accept (Poster)
- Scores: 8, 2, 4, 8

## Abstract
Large Language Model (LLM) agents are reshaping the game industry, by enabling more intelligent and human-preferable characters. Yet, current game benchmarks fall short of practical needs: they lack evaluations of diverse LLM capabilities across various game genres, studies of agentic modules crucial for complex gameplay, and fine-tuning datasets to adapt pre-trained LLMs into gaming agents. To fill these gaps, we present Orak, a benchmark for training and evaluating LLM agents across 12 popular video games spanning all major genres. Using a plug-and-play interface built on Model Context Protocol (MCP), Orak supports systematic and reproducible studies of agentic modules in varied game scenarios. We further release a fine-tuning dataset of expert LLM gameplay trajectories spanning multiple genres, turning general LLMs into effective game agents. Orak offers a comprehensive evaluation framework, including game leaderboards, LLM battle arenas, and in-depth analyses of input modality, agentic strategies, and fine-tuning effects, establishing a foundation towards versatile gaming agents. Code and datasets are available at https://github.com/krafton-ai/Orak and https://huggingface.co/datasets/KRAFTON/Orak.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents ORAK, a comprehensive benchmark designed to evaluate and train large language model (LLM) agents across a diverse set of 12 video games spanning six major genres. The benchmark addresses critical gaps in existing game-based evaluations by incorporating complex, real-world video games, providing modular agentic strategies (e.g., reflection, planning, and tool use), and releasing a high-quality fine-tuning dataset derived from expert gameplay. The authors also propose a unified evaluation framework that includes leaderboards, LLM battle arenas, and in-depth analyses of input modalities, agentic strategies, and fine-tuning effects. The experiments demonstrate the capabilities and limitations of both proprietary and open-source LLMs across various tasks, showcasing the potential of ORAK as a foundation for advancing general-purpose gaming agents.

### Strengths
To be honest, I'm very excited to see an LLM benchmark that integrates complex games. The LLM field is currently flooded with a large number of fixed datasets, yet claiming to be “agentic” is clearly insufficient. ORAK covers a wide range of game genres, including action, adventure, role-playing, simulation, strategy, and puzzle games. This breadth ensures a holistic evaluation of LLM capabilities, from logical reasoning to spatial understanding and long-term planning. The use of the Model Context Protocol (MCP) to integrate LLMs with game environments and agentic modules is a significant contribution. This modular approach enables plug-and-play experimentation and facilitates systematic studies of LLM behaviors in diverse scenarios. The release of expert gameplay trajectories across multiple genres is a valuable resource for the community. The dataset encapsulates meta-knowledge and demonstrates how fine-tuning can enhance LLM performance in both gaming and non-gaming tasks. The paper provides extensive experimental results, comparing proprietary and open-source LLMs across tasks and modalities. The insights into the effects of fine-tuning, visual inputs, and agentic strategies are particularly compelling.

I strongly recommend acceptance of this paper. It makes a significant contribution to the field of LLM evaluation and training, offering a robust benchmark that bridges the gap between academic research and real-world applications. The paper is well-written, methodologically sound, and forward-looking, providing a solid foundation for future work in gaming AI and general-purpose LLMs. I particularly appreciate the authors' attention to detail in designing ORAK and their commitment to open science through the release of datasets and tools.

This work is not only timely but also highly impactful, and I believe it will become a key reference for researchers in the field.

### Weaknesses
This paper still has some minor flaws, and I hope the authors will pay attention to the following issues.

1. The current setup pauses games during LLM inference, which simplifies evaluation but does not fully reflect real-world gaming scenarios. Including preliminary results or discussions on latency-aware evaluation protocols would strengthen the paper.

2. Although the paper mentions RL-based fine-tuning as future work, a brief discussion on how ORAK could be adapted for RL experiments (e.g., reward design, dynamic data extraction) would be valuable for readers interested in this direction.

3. The authors acknowledge the cost of proprietary games and LLM APIs. Exploring potential solutions, such as open-source alternatives or simplified game environments, could help lower the barrier to entry for researchers with limited resources.

If the authors could provide detailed data for problems 1 and 2, as well as potential solutions for problem 3, I believe this paper would be worthy of a spotlight paper.

### Questions
See weakness.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes **Orak** benchmark and evaluation suite for training and evaluating LLM/VLM agents on a diverse set of real video games. The benchmark includes 12 games spanning action, adventure, strategy, and puzzle genres, a modular agent interface (reflection / planning / action / memory components) built on a Model Context Protocol (MCP), prompt templates and action-space definitions, and an expert-trajectory dataset used for supervised fine-tuning. The submission reports cross-model comparisons, ablations of agentic modules, and fine-tuning experiments including transfer tests to out-of-distribution and non-game tasks.

### Strengths
1. **Broad coverage & engineering effort.** The benchmark covers 12 real games across diverse genres and provides a modular evaluation harness, which is useful for benchmarking different LLM agent designs.
2. **Multi-dimensional ability taxonomy.** The paper defines and uses a set of capabilities (e.g., long-horizon planning, spatial reasoning, rule compliance) and maps games to these capability needs, enabling capability–task analyses.
3. **Publicly available supervised trajectories & fine-tuning experiments.** The authors collected and provide a dataset of expert LLM interaction trajectories and demonstrate supervised fine-tuning improvements and some transfer effects.
4. **Useful baseline comparisons.** Results compare multiple closed-source and open-source LLMs under several agentic strategies (zero-shot, reflection, planning, ref-plan), giving a practical snapshot of current model gaps and engineering trade-offs.

### Weaknesses
1. **Insufficient novelty argument / differentiation from prior benchmarks.** The paper lists related benchmarks but does not convincingly quantify or empirically demonstrate how Orak meaningfully advances beyond existing game/agent benchmarks. The unique scientific questions Orak enables are not sharply distinguished.
2. **Experimental robustness & statistical reporting are incomplete.** Many reported results lack rigorous statistical detail (consistent number of seeds/trials, confidence intervals, or significance testing). Some reported scores show large variance, which weakens the reliability of the conclusions.
3. **Lack of systematic prompt / hyperparameter sensitivity analyses.** The paper attributes improvements to agentic modules (reflection/planning), but does not systematically vary prompt wording, temperature, max-context length, or other prompt-engineering factors to rule out that effects are largely prompt-driven.
4. **Real-time interaction and latency issues under-addressed.** For latency-sensitive or action-timed games (e.g., fighting or platformers), the evaluation often pauses the game during inference. This departs substantially from real-world online agent constraints; latency-aware experiments are deferred to future work, limiting external validity.
5. GPT-generated trajectories (e.g., gpt4o, o3-mini) **risk bias, hallucinations, low strategy diversity, and contamination**; the authors should document generation details and show that fine-tuned models generalize beyond the generator.

### Questions
1. For each major table/figure, how many independent trials and random seeds were used? Please add confidence intervals and describe any hypothesis tests performed. If trials vary by game, report that explicitly.
2. Did you run controlled sweeps over prompt phrasings, temperatures, context lengths, or token limits when comparing agentic modules? If not, please run such sweeps for key games or explain why module effects are independent of prompt variants.
3. Can you provide at least one latency-aware experiment for a timing-sensitive game (e.g., impose an upper bound on LLM response time or simulate delay) and report performance degradation as a function of latency?
4. Fine-tuning data & overfitting controls. For the supervised fine-tuning dataset: how were trajectories sampled (top trajectories or diverse sampling)? What regularization, early-stopping, or validation protocols prevented overfitting to a specific agentic workflow?
5. I request ablation results comparing fine-tuning on different generator sources (e.g., gpt4o-only, o3-mini-only, mixed-source, and, if available, human or RL trajectories) to quantify generator-specific biases.

### Soundness
2

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
This paper introduces Orak, a benchmark for evaluating LLM agents across 12 video games spanning six genres. The authors aim to address limitations in existing game benchmarks by offering greater diversity, enabling studies on agentic modules (such as reflection and planning), and providing resources for adapting LLMs into gaming agents. The key contributions are the benchmark itself, which uses a plug-and-play interface based on the Model Context Protocol (MCP) for standardized evaluation, and a fine-tuning dataset derived from expert LLM trajectories designed to distill gaming skills into smaller models. The paper presents a series of experiments on 15 LLMs, analyzing their performance, the impact of agentic modules, the effect of visual inputs, and the generalization capabilities of fine-tuned models.

### Strengths
The primary strength of this paper is the scale and diversity of the benchmark. Compiling 12 games across six distinct genres, each with its own environment setup and state representation, is a big **engineering effort.** This provides a broad testbed for evaluating a variety of agent capabilities, from reaction time in action games to long-term planning in strategy games.

The introduction of a unified, plug-and-play interface using MCP is a commendable step towards standardized and reproducible evaluation of LLM agents in gaming environments. 

The release of a fine-tuning dataset, while based on LLM-generated trajectories.

### Weaknesses
The paper, despite its significant engineering effort, suffers from several weaknesses in its core claims, methodology, and the novelty of its conclusions, which limit its overall contribution.

Largely Unsurprising and Incremental Conclusions: The main findings drawn from the extensive experiments largely confirm well-established knowledge in the LLM agent community, offering little new insight.

- The conclusion that proprietary, closed-source models outperform their open-source counterparts is widely accepted and requires little further validation in 2025.

- The finding that agentic workflows (e.g., reflection, planning) benefit capable models is not new.

The claim that visual inputs often hinder performance is misleading. This outcome is likely an artifact of the experimental design, where highly structured and pre-processed text provides a cleaner, more direct signal than raw visual data for current VLMs. A more accurate conclusion would be that under this specific setup, the models fail to extract sufficient value from visual inputs to overcome the noise, rather than a general indictment of visual modalities for gaming agents.


Questionable Design Choices in Benchmark and Data Generation:

- The selection of games appears to be driven more by the availability of existing APIs or emulators (e.g., Mineflayer for Minecraft, PyBoy for Pokémon Red) rather than a principled selection of titles that would best probe the frontiers of AI capabilities. The benchmark lacks modern, complex 3D games that pose severe challenges in perception from raw pixels, physics-based interaction, and complex spatial reasoning.

- The fine-tuning dataset is generated by an 'expert' LLM (GPT-4o), which fundamentally caps the potential performance of any fine-tuned model at the level of the teacher model. This methodology prevents the discovery of novel strategies that might surpass the teacher's capabilities and introduces the teacher's inherent biases and failure modes into the student models.

- By selecting only the highest-scoring trajectories for the fine-tuning dataset, the authors introduce a strong survivorship bias. The models learn from 'perfect' or near-perfect executions ('sunny day' scenarios) but are not exposed to data on how to recover from mistakes, adapt to unexpected situations, or turn a losing game around. This is a critical omission for developing robust agents that can handle the stochasticity and adversity inherent in complex games.

### Questions
The paper positions Orak as a foundational benchmark that pushes the boundaries of agent evaluation. However, the tasks often seem simplified through pre-processed states and high-level APIs, which might not establish a clear differentiation from prior work.

Could the authors elaborate on the unique challenges Orak presents compared to existing agent benchmarks? The current results do not seem to establish a clear differentiation, as the main conclusions are largely echoes of findings from other domains.
What specific agent capabilities are uniquely tested in Orak that are not adequately covered by prior benchmarks? A more compelling case could be made by showcasing a task where top-performing agents from other domains systematically fail due to a game-specific challenge that Orak is specifically designed to evaluate. For instance, is there a scenario that rigorously tests an agent's ability to reason under partial observability from raw signals, a core challenge in many games?

I'm willing to increase my score if the author can answer my question.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes Orak, a benchmark and dataset for evaluating foundation models in dynamic digital games scenarios. Orak increases the diversity of game genres covered in evaluation when compared to previous benchmarks, and the developed platform also allows plug-in and enabling/disable different agentic modules for ease of evaluation/ablation. 

Moreover, Orak includes a dataset of fine-tuning data, collected as interaction trajectories generated by foundation models guiding playback of all 12 games supported in the platform.

Experimental results show the performance of 15 foundation models on the benchmark, an ablation study of 2 of those models utilizing different agentic approaches to play all games, results on scenarios combining different data modalities, and SFT results using Llama to illustrate the benefits of the collected trajectory dataset.

### Strengths
Orak provides a very funcional benchmark platform for different game genres, as well as a somewhat general covered of game genres that allows better insights about required capabilities from foundation models during gameplay. Such results can also potentially generalize to wider impact beyong digital games alone.

The presented results well illustrate the current capabilities of foundation models and how different games probe them in different dimensions.

The created dataset (and, more importantly, data collection platform) can also serve as a stepping stone for further research on improving gameplay models/agents/systems.

### Weaknesses
While well presented and illustrating the potential of the benchmarl in principle, the current paper presents some limitations. Mostly regarding analysis of the results and the rationale for some of its design aspects.

The manuscript claims "in-depth analyses of input modality, agentic strategies, and fine-tuning effects", but falls a bit short of this challenging goal. It does provide some interesting insights, but doesn't really go deep into either of the 3 areas. Having said that, the platform itself is already of great value and can be used as a strong base for further evaluation and analysis. Toning down such claims still leaves the rest of Orak as solid work.
 
Regarding input modalities, Orak emphasizes pre-extraction of game state information and textual representation of such data. This both; i) greatly reduces the challenges of observation/state/world understanding and already bias results to text; and ii) muddles the analysis of providing different modalities later which then also needs to deal with potential conflicts in different modalities and in some models seemingly having preference for specific modalities. Also known issues previously discussed in the literature.

The presented ablation for agentic modules is interesting as a high-level overview, but its results as presented are not well discussed and don't show significant insights. The analysis here could go much more in-depth in future work.  

The LLM finetuning experiment and resutls are not in-depth and only take a quick look at 1 foundation model in 2 small sizes, where the benefits of SFT with any data would already provide the most benefit. The paper would benefit from a more detailed analysis of dataset quality and what/how it actually contributes during training, as well as any possible insights into data scalling.

However, the main limitation of Orak as a platform is the lack of functionality to properly evaluate grounding of actions, as each game action space has been manualy defined differently and already mapped to high-level functions that heavily abstract and simplify the problem. Though, the platform could be easily modified to offer a full pixel-to-keyboard/mouse-commands interface that fully exposes complexity and allow uses to benchmark at different levels.

### Questions
I don't think the games industry itself is the main beneficiary of such benchmarking effort. How does this motivation and focus affect the benchmark design? Agent autonomy using games as learning environment could have much wider impacts and would need to be analysed differently.

The paper mentions previous evaluation "often rely on visual inputs" as a criticism. Why? I'd argue that understanding of visual observations is exactly the most important area where games can help as benchmark for foudnaiton model capabilties.

In the Arena setting results, why exactly was Starcraft evaluated with one less model than Street Fighter? 

Also, critically, do you have any insights on why Minitron-8B performs so much better than in full results? Minitron was 0 for Starcraft in Table 3. Any deeper understanding here could be significant.

BTW, Table 3 shows the results only of using the "auto extracted state in text only form", correct? It would be interesting to have an easy comparison of that vs. best resutls somewhere. Even if in an appendix.

In a similar veing to the Arena question for Minitron, do you have any insights on why there is no SFT benefit intra-game for Startcrat and Baba?

Typo:
"In game industry" -> "In the games industry"

### Soundness
3

### Presentation
3

### Contribution
2
