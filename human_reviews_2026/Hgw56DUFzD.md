# WARC-Bench: Web Archive based Benchmark for GUI Subtask Executions

- Avg Score: 3.33
- Decision: Accept (Poster)
- Scores: 4, 4, 2

## Abstract
Training web agents to navigate complex, real-world websites requires them to master subtasks—short-horizon interactions on multiple UI components (e.g., choosing the correct date in a date picker, or scrolling in a container to extract information). We introduce WARC-Bench (Web Archive Benchmark), a novel web navigation benchmark featuring 438 tasks designed to evaluate multimodal AI agents on subtasks. WARC-Bench enables sandboxed interactions with dynamic and realistic webpages using Web ARChive files. We show that WARC-Bench is challenging for leading computer-use models, with the highest observed success rate being 64.8%. To improve open source models on subtask, we explore two common training techniques: supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). Experiments show that SFT models obtain a 48.8% success rate on the benchmark. Training with RLVR over SFT checkpoints, even in data-scarce settings, improves the score to 52.8% on WARC-Bench, outperforming many frontier models. Our analysis concludes that mastering these subtasks is essential for robust web planning and navigation, and is a capability not extensively evaluated by existing benchmarks. More details about WARC-Bench can be found at https://sanjari-orb.github.io/warc-bench/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces WARC-Bench, a new benchmark on short-horizon web browsing tasks. It is somewhere between single-step visual grounding tasks and sophisticated, long-horizon web interaction tasks. They also developed a way to synthesize more of such tasks. They evaluated multiple computer-use agents and their own Subtask Vision Agent (SVA), using frontier models, open-weights models, and custom models post-trained with SFT and RLVR.

### Strengths
1. Innovative way to scale the size of the dataset, including the use of WARC files to quickly reproduce exact replication of complex real-world websites.

2. A good methodology to synthesize data and demonstration to show the effectiveness of training on such datasets.

3. Strong results of the authors' own trained model. Discussion on behavioral analysis of SFT v.s. RLVR is insightful.

### Weaknesses
1. Baseline only used computer-use agents but not web browser agents (e.g. browser-use) or general agents with web browsing capability (e.g. OpenHands).

2. Lack of comparison to other short-horizon browsing benchmarks, e.g. WebGames, WebVoyager, WebShop.

3. Experiments are run 3 times but there's no error bar or other variation info.

4. No human baseline. Since these are short-horizon subtasks, it seems reasonable for human baseline to reach 100%. Lack of human baseline makes it hard to evaluate the soundness of the benchmark (e.g. whether there are ambiguous instructions, wrong answers, cheating possibilities, etc.).

5. Dev accuracy on synthetic dataset is universally higher than dev accuracy on real dataset, indicating the synthetic dataset might have design flaws (e.g. data contamination, overly simple designs, etc.).

### Questions
The definition of a "subtask" seems somewhat arbitrary. Authors argue this is a "novel" benchmark designed to evaluate AI agents in executing "subtasks" on web browsers. How are these "subtasks" different from the tasks in similar benchmarks like WebGames, WebVoyager, WebShop?

### Soundness
3

### Presentation
4

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
- This paper introduces WARC-Bench, a comprehensive training and evaluation dataset to train and evaluate models’ and agents’ ability to complete GUI sub-tasks in web environments using computer-use agents. 
- The authors implement an agentic scaffold (subtask vision agent: SVA) for inference and training and train 7B and 72B LLMs using supervised fine-tuning on trajectories sampled from stronger LLMs, followed by online RL (using PPO) on the synthetic training data. 
- Experimental results demonstrate the effectiveness of their approach on the test split of WARC bench while also generalizing to other web navigation tasks/benchmarks.

### Strengths
- The paper is well-written and easy to understand.
- The task of completing specific sub-tasks on webpages is an interesting method of training models for web navigation tasks.
- The authors provide extensive comparisons across open- and closed-source models, showing improvements through supervised fine-tuning and RLVR.

### Weaknesses
- Missing baselines for other web navigation datasets: For WebArena, can you please include a comparison of your work with AutoGLM (https://dl.acm.org/doi/abs/10.1145/3637528.3671620), NNet-Nav (https://openreview.net/forum?id=L5BYl0GE2F) and Go-Browse (https://arxiv.org/abs/2506.03533) that also develop training data and train open-source LLMs as web agents. In particular, I am not convinced about the claim of your 7B model generalizing to WebArena benchmark because Go-Browse reports a resolve rate of 8.3% on WebArena with Qwen-2.5-7B-Instruct, which is slightly better/comparable to the performance of your fine-tuned 7B model. 
- Generalizability of results: What are the specific websites used during training and are they semantically similar to the websites used in the evaluation benchmarks, particularly WARC-bench test set, WebArena, MiniWob++, and ScreenSpotV2. It would be helpful if authors could include additional discussion on out-of-domain generalization. For example, if the training set does not include any task dealing with shopping websites, can the trained models solve such tasks in the evaluation split? Relatedly, is there any analysis on how the trained models generalize to subtasks not seen during training? For example, if the training dataset does not include subtasks required to fill forms and manipulate tables, can it solve these tasks during inference?
- Exact training dataset used for SFT: What are the specific tasks and websites used during the supervised fine-tuning stage? The paper initially mentions the training split of 1059 instances in the WARC bench, but then describes using 12k unique trajectories using teacher models from Common Crawl, and synthetic data? How are the natural language descriptions of tasks designed for these 12K trajectories?

### Questions
Please address each of the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces WARC-Bench, a new benchmark for evaluating GUI agents on short-horizon subtasks. It includes 438 tasks across 29 real and 62 synthetic websites, with deterministic programmatic evaluators (e.g., JS/URL/string matchers) and sandboxed replay environments. The paper also introduces Subtask Vision Agent and reports results across models. The authors further train Qwen-based models via  SFT and RLVR.

### Strengths
- The paper identifies an underexplored middle ground between visual grounding (single-step) and long-horizon navigation (multi-page workflows), positioning subtasks as a meaningful intermediate capability.

### Weaknesses
- Over 70% of training and development data is synthetic. While the paper acknowledges this, it remains unclear how well these synthetic subtasks capture the diversity and ambiguity of real-world web UIs. The manual verification process is under-detailed.


- The paper champions the high fidelity of WARC files but fails to adequately address their most significant drawback: they are static snapshots. A vast category of real-world web tasks involves interacting with dynamic, real-time, or session-dependent content (e.g., checking live stock prices, booking the last available seat on a flight). By design, WARC-Bench cannot evaluate an agent's ability to handle this dynamism. While the authors briefly mention limitations, they understate how this core architectural choice constrains the benchmark to a specific, offline subset of web interaction, making its claim of testing real-world capabilities only partially true.

- The paper frames "GUI subtasks" as a key conceptual contribution. However, this appears to be more of an incremental refinement than a novel problem class. Benchmarks like MiniWoB++ have long focused on widget-based interactions in simplified environments. WARC-Bench can be seen as a high-fidelity, WARC-based instantiation of this existing idea rather than a new paradigm. The contribution is therefore more on the engineering side (the use of WARC).


- The paper suggests that adding new environments is as simple as "adding a new web recording." This overlooks the fact that the primary bottleneck in creating meaningful benchmark tasks is the manual and intellectually demanding process of defining a clear, unambiguous goal and, crucially, authoring the programmatic reward function (the "evaluator"). This manual effort is not eliminated by the WARC approach. Therefore, the claim that the benchmark design is uniquely scalable compared to others that also require manual task specification is overstated.

### Questions
see above

### Soundness
1

### Presentation
3

### Contribution
2
