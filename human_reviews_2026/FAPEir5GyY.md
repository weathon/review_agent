# Training Language Model Agents to Find Vulnerabilities with CTF-Dojo

- Decision: Reject
- Scores: 4, 4, 8, 2

## Abstract
Large language models (LLMs) have demonstrated exceptional capabilities when trained within executable runtime environments, notably excelling at software engineering tasks through verified feedback loops. Yet, scalable and generalizable execution-grounded environments remain scarce, limiting progress in training more capable ML agents. We introduce CTF-Dojo, the first large-scale executable runtime tailored for training LLMs with verifiable feedback, featuring 658 fully functional Capture-The-Flag (CTF)-style challenges containerized in Docker with guaranteed reproducibility. To enable rapid scaling without manual intervention, we develop CTF-Forge, an automated pipeline that transforms publicly available artifacts into ready-to-use execution environments in minutes, eliminating weeks of expert configuration traditionally required. We trained LLM-based agents on just 486 high-quality, execution-verified trajectories from CTF-Dojo, achieving up to 11.6% absolute gains over strong baselines across three competitive benchmarks: InterCode-CTF, NYU CTF Bench, and Cybench. Our best-performing 32B model reaches 31.9% Pass@1, establishing a new open-weight state-of-the-art that rivals frontier models like DeepSeek-V3-0324 and Gemini-2.5-Flash. By framing CTF-style tasks as a benchmark for executable-agent learning, CTF-Dojo demonstrates that execution-grounded training signals are not only effective but pivotal in advancing high-performance ML agents without dependence on costly proprietary systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents CTF-DOJO, a large-scale executable environment for Capture-The-Flag (CTF) tasks, and CTF-FORGE, an automated pipeline that converts public CTF materials into reproducible Dockerized challenges. Using DOJO, the authors collect a small but high-quality set of execution-validated trajectories (≈486) and fine-tune open-source base models (primarily Qwen3 8B/14B/32B) with rejection-sampling SFT. The resulting agents achieve competitive Pass@1 on three benchmarks (InterCode-CTF, NYU CTF Bench, Cybench), with the 32B variant around the low-30% Pass@1 range, while showing strong data efficiency. The paper also studies factors such as environment randomization, leveraging writeups as non-citation hints, and teacher-model diversity.

### Strengths
1) Valuable infrastructure: A reproducible, containerized CTF environment at scale and an automatic “forge” pipeline are strong contributions likely to benefit the community.
2) Executable supervision: Training from execution-validated traces is a principled way to reduce hallucinations and reward specification issues common in security tasks.
3) Data efficiency: Competitive numbers are achieved with a surprisingly small number of high-quality trajectories (~486), highlighting the utility of executable signals.
4)The collection setup (agent loop, rollout budget), training recipe (RSFT), and evaluation benchmarks are described in a way that practitioners can reproduce.

### Weaknesses
1) Methods relying on more abundant synthetic trajectories  report higher Pass@1 in similar settings. The paper emphasizes efficiency, but direct apples-to-apples comparisons at fixed data budgets would clarify the trade-offs.
2) Benchmark comparability details: Small differences in patched tasks, scaffolding, or decoding hyperparameters can materially affect Pass@1. A fully standardized evaluation harness (same temperatures, retries, timeouts, and tool stacks) would aid comparability to prior work.
3) Shaping and sensitivity depth: The paper discusses helpful training signals (execution outcomes, writeups), but provides limited sensitivity on shaping choices (e.g., negative/partial rewards, step-wise curriculum) and alternative supervision (e.g., learned difficulty signals, self-reflection).

### Questions
1) Can you provide matched-data comparisons against larger synthetic-trajectory approaches (e.g., train both with 500, 1k, 2k examples) to quantify sample efficiency directly?
2) What steps ensure no answer leakage from writeups and trajectories to evaluation (e.g., time splits, near-duplicate filtering)?
3) How sensitive are results to decoding settings (temperature, top-p), step budgets, and tool availability?

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
This work proposed CTF-Dojo, which automatically transforms CTF challenges into Docker environments. By training agents on trajectories from these environments, the authors manage to achieve new open-weight state-of-the-art performance using a 32B model on InterCode-CTF, NYU CTF Bench, and Cybench.

### Strengths
1. An automatable pipeline to generate an agent environment for cybersecurity agents.
2. Demonstrate better performance on representative benchmarks with better data efficiency compared to Cyber-Zero.

### Weaknesses
1. Automatically creating the execution environment is not novel. For example, in SWE-smith, the authors use SWE-agent to automatically create a Docker given any GitHub repo.
2. The scalability seems to be a bigger issue than discussed. As the author noted, each CTF challenge is uniquely designed, and the current pipeline can only leverage existing challenges instead of generating synthetic training instances from scratch. For comparison, SWE-smith includes the procedure of automatically generating issues to solve, which fundamentally removes the scalability issues. While I agree there is a trade-off between scalability and quality/diversity, the current framework seems extremely constrained by scalability.
3. What is the impact of 4.1 and 4.2 on training? Like, what is the performance of trained agents using the baselines in 4.1 and 4.2?

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper collects CTF cybersecurity competitions and forms them into a benchmark and training environment for LLM code generators. A software pipeline, including LLMs, for setting up the testing environment is constructed and shown to often work. Two models are tested in this environment and traces were collected to serve as a training dataset for other models. The authors show that training on this set as well as training on human generated advice/plans for the problems can increase model performance on this and other offensive cybersecurity benchmarks.

### Strengths
- a sizeable collection of CTF competitions is collected for use in model testing
- reasonable tests show that training on successful outputs from this benchmark generalize to other CTF competitions
- The authors claim that setting up a CTF testing environment is difficult and time-consuming and they can automate the process with 98% accuracy
- The authors show traces from their constructed environment are useful for training for other CTF problems.
- The ethical implication of improving offensive cybersecurity capabilities are discussed.

### Weaknesses
- It is unclear to me what environments the automated setup was tested on and whether this will hold true for many users and, especially given the use of LLMs, will continue to be reliable in the future.
- It is unclear what the copyright status of the original CTF competitions is and whether the authors are ok with it's collection and use in bench-marking and training LLMs.

### Questions
I have no additional questions or comments for the authors.

### Soundness
3

### Presentation
3

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
This work focuses on the lack of high-quality training data for LLM agents in Capture-The-Flag (CTF) tasks. Leveraging standardized, publicly available data, the authors developed a prompt-based pipeline to convert these artifacts into an executable and verifiable Docker environment named CTF-DOJO. Using this environment, they collected 486 successful trajectories for training. The experiments demonstrate the effectiveness and high data efficiency of training models on these execution-verified trajectories.

### Strengths
1. High-quality training data is a critical factor influencing model performance, and the authors' insight on this point is correct.

2. There has been no prior work on building such an environment specifically for training agents on CTF tasks. This paper makes a valuable contribution in this direction.

### Weaknesses
1. The paper's core claim that training data must be "execution-verified" and "correct" is questionable. This methodology is contradicted by the paper's own results, which show that a model trained on a much larger, unverified (and presumably noisy) dataset outperforms the authors' model. This suggests that data quantity may be more important than the strict verification the authors insist upon. The justification of "safety-critical domains" for this training choice is also unconvincing.
2. The technical contribution seems limited. The authors chose to build their system on a single, pre-standardized data archive (pwn.college) to avoid integration challenges. The core CTF-FORGE pipeline is a straightforward "prompt-based" workflow. A more significant contribution would have involved integrating and normalizing multi-source, heterogeneous data, rather than selecting the most convenient, pre-cleaned source.
3. The paper's structure is poor. Section 2 is bloated with excessive, non-novel details about data sourcing and processing. These engineering details belong in the appendix. This padding makes the main body of the paper feel hollow and detracts from the space needed for core methodological arguments.

### Questions
1.The constructed environment (CTF-DOJO) with its 650+ verified, executable challenges seems perfectly suited as a high-value evaluation benchmark. Why do the authors insist on framing this contribution as a training data solution? I understand the "official" reason that no such training datasets currently exist, but I am interested in deeper insights. Does framing this as a "training problem" unlock a specific scientific contribution that framing it as a "benchmark" would not?
2.Regarding the alignment of CTF challenges and writeups, why was "fuzzy matching" necessary? One would expect a natural, direct mapping to exist between a specific challenge and its corresponding writeup.
3.I encourage the authors to provide a rebuttal to the points raised in the "Weaknesses" section. I am willing to reconsider my score based on a compelling response.

### Soundness
2

### Presentation
2

### Contribution
1
