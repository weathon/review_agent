# Octax: Accelerated CHIP-8 Arcade Environments for Reinforcement Learning in JAX

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Reinforcement learning (RL) research requires diverse, challenging environments that are both tractable and scalable. While modern video games may offer rich dynamics, they are computationally expensive and poorly suited for large-scale experimentation due to their CPU-bound execution. We introduce Octax, a high-performance suite of classic arcade game environments implemented in JAX, based on CHIP-8 emulation, a predecessor to Atari, which is widely adopted as a benchmark in RL research. Octax provides the JAX community with a long-awaited end-to-end GPU alternative to Atari games, offering image-based environments, spanning puzzle, action, and strategy genres, all executable at massive scale on modern GPUs. Our JAX-based implementation achieves orders-of-magnitude speedups over traditional CPU emulators. We demonstrate Octax's capabilities by training RL agents across multiple games, showing significant improvements in training speed and scalability compared to existing solutions. The environment's modular design enables researchers to easily extend the suite with new games or generate novel environments using large language models, making it an ideal platform for large-scale RL experimentation. Our open-source framework is available at https://github.com/riiswa/octax/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces OCTAX, a suite of classic arcade-style RL environments implemented as a fully vectorized CHIP-8 emulator in JAX. The goal is to provide an Atari-like benchmark that (a) runs end-to-end on GPU, (b) scales to thousands of parallel environments, and (c) maintains authentic game dynamics. The authors show throughput up to ~350k steps/sec (≈1.4M frames/sec) with 8,192 parallel environments on a single consumer GPU (RTX 3090), which they report is roughly 14× faster than CPU-bound baselines like EnvPool on Atari Pong. 

The benchmark covers 20+ CHIP-8 games across genres (puzzle, action, navigation, resource management, etc.), and the paper provides PPO training curves for 16 of them. These curves exhibit meaningful diversity in difficulty and learning dynamics (fast plateau vs. gradual improvement vs. failure-to-learn cases like Tetris/Worm), suggesting this is not just “16 clones of Breakout,” but a range of cognitive burdens. 

Finally, the paper explores automatic environment generation: they use an LLM to synthesize new CHIP-8 games (e.g. a “Target Shooter” with increasing difficulty levels), plus score/termination logic, and then successfully train PPO on these generated tasks with clean difficulty gradients. This is pitched as a path toward scalable curriculum / rapid task creation. 

Overall, the paper proposes OCTAX as “the missing GPU-native Atari for JAX RL,” arguing that this makes statistically reliable RL experiments cheaper and more reproducible for smaller labs.

### Strengths
**S1. Clear problem, clear value to the community.**
The paper correctly identifies a real bottleneck in RL: environments are still mostly CPU-bound, even though policy learning is GPU-bound. This prevents large-scale sweeps, high seed counts, and rigorous statistics, especially for vision-based tasks like Atari. OCTAX directly targets that gap with an image-based, arcade-style benchmark that runs entirely in JAX on GPU. 

**S2. Strong engineering contribution.**
The authors implement a full CHIP-8 fetch/decode/execute loop in JAX using functional state passing, vectorized dispatch (lax.switch), and batched framebuffers, and wrap it as Gym/Gymnax-style RL envs with reward extraction, termination logic, action pruning, frame stacking, etc. They argue that fidelity to original game mechanics is preserved. 
This is nontrivial and, as far as I know, not previously available in the JAX ecosystem for Atari-like games.

**S3. Throughput + scaling results are compelling.**
Reported numbers (350k steps/sec, linear-ish scaling up to 8k envs on a single 3090; 14× over EnvPool Pong at high parallelism) are impressive and very relevant to labs that don't have multi-node clusters. The memory footprint (2 MB per env, linear in env count) is also concretely discussed. 
This is an unusually thorough systems evaluation for an RL benchmark paper.

**S4. Diversity of tasks and empirical characterization.**
They don’t just dump environments; they also measure PPO learning curves across 16 games and group them into qualitative regimes (fast plateau vs. gradual improvement vs. hard/sparse). This suggests these games could be used for algorithm diagnostics, curriculum, etc.

### Weaknesses
**W1. How “Atari-like” is it, really? (External validity.)**
CHIP-8 games are dramatically simpler than Atari 2600 in terms of resolution (64×32 monochrome), action semantics, and world complexity. The paper asserts “Atari-like cognitive demands,” but the qualitative gap (e.g., long-horizon exploration, partial observability, rich object interactions) is not deeply quantified. We see PPO struggling on Tetris/Worm, but I'd like more systematic evidence that success on OCTAX predicts anything on Atari, NetHack, Procgen, etc. 
Right now OCTAX looks great for intra-JAX benchmarking and ablations, but it's less clear if it’s meaningful as a drop-in Atari replacement for algorithm claims.

**W2. Reward shaping / termination extraction feels artisanal.**
For each game, they manually or semi-automatically infer score registers, life counters, game-over flags, menu skips, etc. (e.g. “Brix stores score in V5, Pong encodes BCD in V14, etc.”). This is powerful, but also fragile and slightly underspecified. 
If a new contributor adds a weird ROM, how confident are we that OCTAX's automatic heuristics won't silently produce a broken reward function (e.g. rewarding losing health)? The paper mentions static+dynamic analysis plus some LLM help, but I’d like stronger guarantees or validation.

**W3. No baselines beyond PPO.**
All learning curves are PPO only. There's no DQN-style baseline, no lightweight world-model baseline, no offline RL baseline, etc. PPO is reasonable (and popular), but I'd like to know if these environments produce meaningful rankings across algorithms, or whether they’re PPO-biased (e.g. continuous control style tuning, frame stacking assumptions, frame-skip assumptions). 
Right now we mainly learn “PPO can learn some of them.”

### Questions
**Generalization / external validity.**
Do you have any evidence that OCTAX performance correlates with Atari performance, Procgen performance, or other visual RL benchmarks? Even a tiny pilot (e.g. rank-correlation of seed-averaged scores across algorithms) would strengthen the “Atari alternative” positioning. 

**EnvPool/CuLE comparison.**
Can you report CuLE numbers (GPU Atari) on the same GPU you used for OCTAX, or at least discuss why that wasn’t feasible? That would make the “14× faster” claim feel less like apples vs oranges. Also, have you tried running OCTAX on CPU only to show the CPU→GPU delta cleanly? 

**Automatic reward/termination inference.**
For a new arbitrary CHIP-8 ROM (unseen by you), how robust is score/termination extraction? Do you have quantitative success rates for the static+dynamic heuristics or the LLM-generated wrappers? For RL practitioners, “plug in a ROM and it just works” is a killer feature—please convince us it's realistic.

### Soundness
3

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
This paper presents Octax, a high-performance arcade-style benchmark for Reinforcement Learning in Jax. The paper utilizes end-to-end GPU training, allowing extremely high throughput. Primarily, this work looks to provide a viable alternative to the long-standing Atari benchmark, but using considerably less computational resources. Furthermore, the paper presents a way to quickly use LLMs to generate new environments, on top of the current set of games presented.

### Strengths
This paper aims to achieve the important goal of reducing the cost of running Reinforcement Learning experiments, clearly succeeding, achieving an impressive 1.4 million frames per second on consumer-grade hardware. The code already being open-source is also a nice positive. LLM-generated environments using the provided prompts provide a nice way for developers to test environments with specific properties if they wish.

### Weaknesses
While I appreciate the attempt to reduce the computational burden of the Atari, I feel that the authors missed many key reasons why the Atari benchmark became so popular, and many of the features that still make it so popular today. In its current form, I don’t believe this work is capable of replacing the Atari benchmark without major reforms for a few key reasons:

- **A clear, fixed way to evaluate** - One reason Atari is useful is that there is a relatively clear and fixed protocol for evaluation [1]. This means researchers know what environments they need to run, the settings they need to use, and know that their work will be comparable to others. Furthermore, there are precise guidelines on how scores should be reported [2]. The paper would benefit from being very clear about what environments are included in the benchmark, how many frames/timesteps algorithms should be trained for, and how results should be reported, with examples. Currently Figure 3 does not meet the same standards as many papers which use the Atari benchmark.
- **Aggregate Scores** - Atari is useful as a benchmark, as the performance of an algorithm can be boiled down to a single number or single graph. In Atari, this has become human-normalized IQM performance with 95\% confidence intervals, which provides a score for the given algorithm. This prevents games with different score magnitudes from dominating the final score. Currently, this appears to be missing, despite games having scores of different magnitudes.
- **No-op Starts, Sticky Actions and Random Actions** - In the Atari benchmark, features are provided that prevent the agent from exploiting determinism (preventing brute-force approaches). Specifically, no-op starts randomly uses up to 30 no-op steps at the start of episodes; sticky actions give a 25% chance for actions to be repeated, and agents are forced to take random actions 1% of the time. While Octax appears to have some environments that have randomness, it's unclear whether brute-force approaches could work in some environments. 
- **JAX only** - While JAX has its advantages in speed, for a benchmark, I see it as a significant weakness if this benchmark is exclusive only to JAX users. A significant portion of RL researchers and users use other frameworks, such as PyTorch, which appear to be excluded from using this benchmark.
- **Historic data** - One major advantage of Atari is that a huge number of algorithms have been evaluated, making it useful to compare against new algorithms, while Octax only has PPO. Please consider adding more algorithms such as DQN [3], SAC [4], PQN [5], and also more state-of-the-art algorithms would be appreciated.
- **Game categorizations** - Currently, in Octax games are categorized into groups such as Puzzle, Action, etc. I think it would be more beneficial for the research community if environments were grouped by the aspects of the algorithm they challenge. In Atari, there are well known groups such as hard exploration (Montezuma’s Revenge), long term-credit assignment (Skiing) and many more. For a good example of this, please look at BSuite [6]. While Table 1 somewhat provides this, I still don’t think it is up to the standards of recent work. Furthermore, I’d appreciate a more detailed description of the tasks, including the frequency and magnitude of rewards. It is currently unclear if  Octax provides environments that are as challenging as Atari - for example, even after years of research, environments such as Pitfall and Montezuma’s Revenge are still extremely challenging.


[1] Machado, Marlos C., et al. "Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents." Journal of Artificial Intelligence Research 61 (2018): 523-562.

[2] Agarwal, Rishabh, et al. "Deep reinforcement learning at the edge of the statistical precipice." Advances in neural information processing systems 34 (2021): 29304-29320.

[3] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).

[4] Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).

[5] Gallici, Matteo, et al. "Simplifying Deep Temporal Difference Learning." The Thirteenth International Conference on Learning Representations.

[6] Osband, Ian, et al. "Behaviour Suite for Reinforcement Learning." International Conference on Learning Representations.

### Questions
- Is this benchmark only usable to those who have algorithms written in JAX?
- Does this benchmark have a set of rigorous set of specifications (number of frames, number of games, way to compare different overall performance) to ensure that researchers can easily compare their work?
- How challenging are the existing environments in this benchmark? Can you benchmark some different algorithms and provide code for these in the repository?
- Do all environments have a source of stochasticity? Or can they be solved by brute-force style algorithms?

Also, it appears Line 79 has a mistake. While I'm strongly in favor of benchmarks which make research easier, I feel this work still has a long way to go before it could replace something like Atari, thus I cannot yet recommend acceptance.

### Soundness
3

### Presentation
3

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
The paper presents OCTAX, a JAX-native, vectorized CHIP-8 emulator and RL environment suite that runs many GPU-accelerated game instances. OCTAX exposes (21) CHIP-8 titles as Gym/Gymnax-compatible environments, includes wrappers to extract score and termination signals, reports high throughput (claims up to hundreds of thousands env steps/s / millions of frames/s), provides PPO training experiments across multiple games, and demonstrates an LLM-assisted pipeline to generate CHIP-8 games and corresponding reward/termination wrappers. An anonymized code repository is provided.

### Strengths
The paper addresses a clear practical gap in the JAX ecosystem: image-based, GPU-native environments for RL. It thereby comes with significant engineering effort: a fully vectorized CHIP-8 emulator in JAX that can execute many parallel instances and integrates with standard RL training loops (Gym/Gymnax). Furthermore, this work directly empowers the community to work on more complex JAX-accelerated environments, which wasn't directly possible beyond Craftax before.

It introduces a broad catalog of 21 CHIP-8 games across multiple genres, useful for rapid prototyping and curriculum experiments. The reported throughput and scaling is promising, and if validated could substantially reduce wall-clock time for many experiments allowing for low resource algorithmic developments.

Finally, the authors for provide evidence for a novel auxiliary idea: an LLM-assisted pipeline to generate lightweight CHIP-8 games and wrappers, enabling rapid environment prototyping. I find this especially exciting since it may enable open-ended training and novel autocurricula approaches.

### Weaknesses
Emulation fidelity claims ('perfect fidelity') may be partially unsubstantiated: no instruction-trace equivalence, frame-by-frame comparison, or unit tests against a trusted CHIP-8 interpreter are reported. Most of the functional correctness assertions come from the successful training of agents.

Arguably, there is no really fair CPU baseline for CHIP-8 measured on the same machine, and comparisons to EnvPool/ALE conflate environment complexity differences. I understand that this is not trivial to accomplish, but I think it would make sense to maybe show runtimes of CPU/GPU-enabled environments just for contextualization. Even if they are not the same as the ones implemented in OCTAX.

The GPU memory/accounting claims (~2 MB per environment) lack a principled breakdown and appear implausible without explanation of XLA/JAX buffer and compiled executable overheads.

The LLM-assisted game generation is presented as a single case study with no statistics on compile/run success rates, human edit frequency, or failure modes.

Some broader claims (energy savings, enabling small labs) are asserted without corresponding measured energy or cost data. This could and should be better substantiated.

All experiments only consider PPO as the single RL algorithm tested.

### Questions
Re Emulator fidelity: Do you have systematic validation tests? E.g. for something like Pong, can you provide instruction-by-instruction equivalence tests, frame-by-frame rendering comparisons, and unit tests across representative ROMs against a trusted CHIP-8 interpreter.

Baselines and fairness: If you retain EnvPool/ALE comparisons, justify differences in environment complexity or compare EnvPool on a matched low-resolution workload. Additionally, consider adding DQN-style results.

Profiling and memory breakdown: Provide GPU profiler traces (kernel times, GPU utilization) and a detailed memory breakdown per environment (state arrays, framebuffers, intermediate buffers, compiled executable memory). Explain how you measured ~2 MB/env and the causes of scaling limits.

PPO protocol: Were hyperparameters tuned per game or held fixed? Provide ablations showing sensitivity to frame-skip and observation stacking and report whether reported learning curves use per-game tuning.

LLM pipeline evaluation: What fraction of LLM-generated ROMs compiled and ran without manual edits? How often did generated score_fn/terminated_fn require human correction? Provide statistics and typical failure cases and describe any automated validation used.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors present a JAX implementation, known as OCTAX, of the CHIP-8 platform amenable to GPU acceleration. Octax converts CHIP-8 ROMs into RL environments that can take advantage of vectorized JAX functions. Octax provides a wide selection of games of interest to RL researchers spanning a number of different genres and difficulty levels. The performance of Octax is shown to be substantially higher than traditional methods for environment simulation using parallel instances of games with EnvPool. The authors note that some CHIP-8 ROMs had difficult reward functions to implement and understand, but this process was made easier with the help of LLMs to study the machine code. By reversing this process, they were able to train LLMs to produce new games that could be added to the RL training environment.

### Strengths
- The performance improvement of Octax over vectorized environments is impressive and will decrease the training time for many researchers using JAX to engage in RL experiments. With a shorter training time on simpler games, many new experiments to understand the training dynamics and tradeoffs between the large number of RL hyperparameters may be investigated more effectively.
- On top of improved acceleration, the authors demonstrate the ability of LLMs to aid in the understanding of existing ROMs and the generation of new environments based on properly constructed prompts. I believe this may be a simplified and interesting sandbox to explore LLM coding in a way that is faster than traditional code generation procedures.
- Providing this code as a testbed will facilitate the construction of more complex environments, such as Super-CHIP8, to add further complexity and interest.

### Weaknesses
- The utility of arcade environments for SOTA RL has passed, so the real value is to use the environment to accelerate understanding of RL training dynamics and other metrics. Although the added challenge of generating new environments may mitigate this point.
- Supporting yet another training environment may have a limited impact and lower the contributions.
- There's no discussion regarding the success rate of generating ROMs using LLMs. I assume this is for the sake of space and to focus the discussion of the paper on the acceleration of arcade environments.

### Questions
- How many attempts were required to generate Target Shooter using Claude? Was the success rate?

### Soundness
3

### Presentation
3

### Contribution
2
