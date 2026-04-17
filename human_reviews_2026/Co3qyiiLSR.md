# GTD: Dynamic Generation of Multi LLM Agents Communication Topologies with Graph Diffusion Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 6, 2

## Abstract
The efficiency of Multi-Agent Systems (MAS) largely hinges on their communication topology. Existing methods often rely on static or rule-based topologies, which struggle to meet diverse task requirements. To address this challenge, we introduce a novel generative framework called "Guided Topology Diffusion (GTD)." GTD is the first to apply a conditional discrete graph diffusion model for the dynamic generation of MAS communication topologies. Our core idea models the topology generation process as an iterative edge construction process, starting from an empty graph and guided by both task and agent team context. We designed a context-aware Graph Transformer as the denoising network and innovatively proposed a two-stage guidance mechanism: first, a lightweight Proxy Model quickly predicts non-differentiable, task-performance-related multidimensional communication protocols (such as utility, cost, robustness) as reward signals; then, during the sampling phase, a Zeroth-Order Optimization algorithm adjusts the generation trajectory in real-time based on this proxy reward. We validated GTD across multiple benchmarks, and experiments show that this framework can generate highly task-adaptive, sparse, and efficient communication topologies, significantly outperforming existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed an algorithm based on graph diffusion to generate communication topology for multi-agent systems. The author(s) first define a training objective based on utility, cost and sparsity (efficiency) to measure how good a topoligy is. Then, they generate baseline topologies and do simulations on them to train a reward model with this training objective to predict the performance of a given network and task. Then, they also set a threshold on this training objective to find a ``high performance'' subset of baseline topologies and then train a denoising network on the ``high performative'' subset. During inference, at each step, denoising models generate some candidates and then, reward model is used to choose the best one from the candidates to continue the process. Experiments demonstrate that their algorithm outperforms various baseline methods across different tasks.

### Strengths
(1) This paper is clearly written and easy to understand. Detailed proofs regarding zero-order guidance in diffusion process is presented.

(2) Extensive experiments across different tasks and baseline methods are conducted. The results are convincing. Model's performance are measured by not only accuracy, but also robustness agaignst agent failure.

### Weaknesses
(1) During inference, both reward model $P_{\phi}$ and the denoising model are used. Zero-order guidance may also reduce inference efficiency.

(2) I have some doubts about parametering and sampling each egde in graphs independently and generating the whole at the same time. To solve more complicated problems, whether each edge should exist may depend on the rest part of the whole graph. However, employing diffusion models to generate the whole graph at the same time may not be able to represent this kind of relationship.

(3) It is reported in paper that ``Performance scales effectively up to four agents but shows diminishing returns thereafter. This result validates our use of four agents as an optimal trade-off between task performance and computational efficiency". This is understandable as most tasks considered in this paper are relatively simple for the backend LLMs. However, it is clear whether their methods are good at generating larger networks for harder tasks.  

(4) There are no results showing how close $P_{\phi}$ is to true simulation results.

### Questions
(1) Why does experiements w/ random outperform w/o guidance? Should the distribution of next step generated from w/ random and w/o guidance be the same as guidance is used as a zero-order optimizer?

(2) Wht is the chance that loops are contained in the graphs generated and how will the process terminate when there are loops?

(3) Can $P_{\phi}$ be distilled into denoising network?

(4) The reward $P_{\phi}$ is trained on all baseline topologies, while denoising network is trained on ``high performance" subset of baseline topologies. Will this cause some OOD issues?

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
4

### Summary
This paper proposes Guided Topology Diffusion (GTD), a framework for adaptive communication topology generation in multi-agent LLM systems. GTD models the adjacency matrix as a conditional graph diffusion process, iteratively refining topologies through feedback from a learned surrogate reward model for gradient-free, multi-objective optimization (utility, cost, robustness). Experiments on reasoning and code-generation benchmarks demonstrate consistent gains in performance and efficiency over fixed or heuristic baselines, supported by strong ablations and theoretical analysis.

### Strengths
1. The paper innovatively reformulates communication topology generation as a diffusion-based graph synthesis task with explicit multi-objective optimization for utility, cost, and robustness.
2. The writing and organization are clear and coherent, making the technical ideas and motivations easy to follow.
3. The work provides solid theoretical derivations and extensive experimental validation, including thorough ablations and comparisons that support the main claims.

### Weaknesses
1. The **motivation for using diffusion models** remains somewhat underexplained. While the paper claims that diffusion offers stability and multi-step guidance, many existing techniques (e.g., **Gumbel-Softmax** or straight-through estimators) can already handle gradient flow in discrete 0/1 adjacency generation. It would be helpful if the authors could clarify why diffusion is strictly necessary compared to a simpler **neural generator + Gumbel-Softmax** design.  

2. The effectiveness of the **ZO-guided optimization** heavily relies on the accuracy of the **proxy reward model $\mathcal{P}_\phi$**. Since it is trained only via an MSE objective, its generalization ability and robustness under task distribution shifts are uncertain. A deeper analysis or alternative regularization strategies would strengthen confidence in its reliability.  

3. The experiments primarily involve small-scale settings. It remains unclear how GTD performs as the agent population or role diversity increases. The scalability and communication overhead in larger systems should be discussed or empirically validated.

### Questions
1. **Appendix D** appears to be missing or empty ?  

2. It would be very helpful if the authors could provide **qualitative visualizations** of the generated **communication topologies** across different tasks, to better illustrate how structures evolve dynamically during the diffusion process.  

3. Do the **topology changes exhibit interpretability**? For example, can we relate edge formations or removals to task-specific agent roles or stages of reasoning?  

4. (For discussion only) If the **decision horizon becomes significantly longer**, would GTD still work effectively? Current benchmarks mostly require only 2–5 reasoning or coordination steps.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces GTD, a novel framework to address the limitations of static communication topologies in multi-LLM agent systems. GTD formulates topology synthesis as a conditional discrete graph diffusion process, where generation is iteratively steered at each step by a lightweight surrogate reward model that predicts multi-objective outcomes like utility and cost. GTD is shown to generate highly task-adaptive and efficient topologies, achieving superior performance compared to existing methods across several benchmarks.

### Strengths
1. The core idea of framing topology generation as an iterative, guided construction process is interesting. 
2. The overall framework is well-designed, cleverly decoupling the expensive, true reward evaluation from the generative process by using a lightweight surrogate model. Furthermore, the conceptual alignment of this iterative construction with the denoising diffusion paradigm makes the overall design intuitive and methodologically sound.

### Weaknesses
1. Most of the selected baselines are zero-shot methods. GTD introduces a significant two-stage training pipeline (for the surrogate model and the diffusion generator), which requires substantial computational resources and a pre-generated dataset. While the resulting in performance gain, the paper should more explicitly discuss and quantify this training overhead (e.g., data generation cost, training time) to provide a fairer comparison against methods that do not have this requirement.
2. The paper omits several crucial implementation and experimental details that are necessary for a full understanding and reproducibility of the work, as detailed in the questions below.

### Questions
1. The entire guidance mechanism hinges on the fidelity of the surrogate reward model. Could the authors provide more details on its performance? Specifically: 
   - What is the prediction accuracy of the trained reward model on a held-out test set?
   - Could the authors elaborate on how to construct the training data for reward model to ensure the model learns a useful and generalizable rewards?
   - How were the ground-truth performance vectors (line 208) obtained for the training data?
2. I'm curious about the generalization ability of the trained components. Are the surrogate model and diffusion generator trained and tested on the same task distributions (in-distribution), or was there an out-of-distribution (OOD) evaluation? For instance, if the models were trained on data from math and coding benchmarks, how well do they perform when generating topologies for a completely different domain?
3. Could the authors provide more comprehensive training details? This includes the final size of the dataset used to train both the surrogate and generator models, as well as key hyperparameters like learning rates, batch sizes, and the total training time/cost for each component.
4. Since many of the baselines are training-free, they are expected to generalize well to new domains. How does GTD perform on OOD tasks for which it has not seen training data? A dedicated experiment on this would significantly strengthen the paper's claims about adaptability.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes GTD (Guided Topology Diffusion): a conditional graph‑diffusion model that generates task‑specific, sparse communication topologies for multi‑LLM agents. A lightweight proxy reward model predicts utility/cost and provides zeroth‑order (sampling‑based) guidance during denoising. Experiments report nice accuracy–token trade‑offs compared to fixed or dense topologies.

### Strengths
The figures are interesting and aesthetically pleasing. They effectively summarize the pipeline and present accuracy–token Pareto frontiers in an intuitive way.

### Weaknesses
1)	Writing clarity
The manuscript is hard to follow. Many core design choices (how conditions are formed, how candidates are scored each step, and how the posterior update is computed) are only fully clear after reading the released code. 
2)	Data leakage between training and evaluation
The released code uses the same dataset for Phase 1 (data generation for training) and Phase 3 (evaluation), with no split or exclusion of Phase‑1 items at evaluation time. Below are minimally sufficient code excerpts to demonstrate the issue:
 	(a) The same in‑memory dataset is passed to Phase 1 or Phase 3 depending on flags
 	```python
 	# main()
dataset = JSONLReader.parse_file(args.dataset_json)
dataset = gsm_data_process(dataset)
if args.gtd_generate_data:
    await generate_initial_dataset(args, dataset)
elif args.gtd_train_models:
    await train_gtd_models(args)
elif args.mode == 'GTD':
    await run_gtd_experiment(args, dataset)
 	```

 	(b) Phase 1 repeatedly consumes the front of this same dataset to build the training set

 	```python
 	# generate_initial_dataset()
        for i, record in enumerate(dataset):
            if i >= args.gtd_datagen_limit:
                break  # uses the first N samples for data generation
            ...  # run multiple fixed/random topologies, log performance
 	```
 	(c) Phase 3 traverses the same dataset in batches for evaluation—without removing Phase‑1 items
 	```python
 	# run_gtd_experiment()
        num_batches = int(len(dataset) / args.batch_size)
        for i_batch in range(num_batches):
            current_batch = dataloader(dataset, args.batch_size, i_batch)
            for record in current_batch:
                ...  # generate topology with GTD and evaluate on the same tasks
 	```
 	There is no train/test split, shuffle‑then‑split, or index‑based filtering to exclude Phase‑1 items during Phase‑3 evaluation. Consequently, items used to create training data (Phase‑1) reappear in evaluation (Phase‑3), inflating reported accuracy and token efficiency. This is a data leakage bug, not merely a replication choice. 

3)	Narrow model choice
Experiments appear to be restricted to GPT‑4o‑mini. The method’s benefit should be validated on strong open‑source LLMs (and across parameter scales) to rule out model‑specific artifacts.

4)	Benchmark breadth / difficulty
The evaluation focuses on relatively standard math/QA datasets and does not include modern, harder agent benchmarks (e.g., program repair, long‑horizon tool use, GUI manipulation). This limits external validity.

5)	Marginal accuracy gains without leakage control
The reported improvements over strong baselines are generally modest. Given the confirmed train–test leakage (see evidence above), even small biases can materially inflate the perceived advantage; the incremental gains should therefore be treated with caution until a clean split is enforced and re‑evaluated.

### Questions
1)	Model breadth: Are you willing to extend experiments to open‑source models (e.g., Qwen3‑8B, DeepSeek‑V3) to test generality and cost‑efficiency under different decoding behaviors and context windows?

2)	Benchmark difficulty: Will you include more challenging LLMs benchmarks, such as LiveCodeBench [1], KORBench [2]

[1] Jain N, Han K, Gu A, et al. Livecodebench: Holistic and contamination free evaluation of large language models for code[J]. arXiv preprint arXiv:2403.07974, 2024.

[2] Ma K, Du X, Wang Y, et al. Kor-bench: Benchmarking language models on knowledge-orthogonal reasoning tasks[J]. arXiv preprint arXiv:2410.06526, 2024.

### Soundness
3

### Presentation
2

### Contribution
2
