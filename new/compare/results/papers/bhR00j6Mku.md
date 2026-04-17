# On The Fragility Of Benchmark Contamina- Tion Detection In Reasoning Models

Han Wang1,∗ Haoyu Li1,∗ Brian Ko2,∗ **Huan Zhang**1 1 University of Illinois Urbana-Champaign 2 University of Washington
{hanw14,haoyuli5}@illinois.edu, kkm97183@uw.edu, huan@huan-zhang.com

## Abstract

Leaderboards for large reasoning models (LRMs) have turned evaluation into a competition, incentivizing developers to optimize directly on benchmark suites.

A shortcut to achieving higher rankings is to incorporate evaluation benchmarks into the training data, thereby yielding inflated performance, known as benchmark contamination. Despite that numerous contamination detection approaches have been proposed, surprisingly, our studies find that evading contamination detections for LRMs is alarmingly easy. We focus on the two scenarios where contamination may occur in practice: (I) when the base model evolves into LRM via supervised fine-tuning (SFT) and reinforcement learning (RL), we find that contamination during SFT can be originally identified by contamination detection methods. Yet, even a brief Group Relative Policy Optimization (GRPO) training can markedly **conceal contamination signals** that most detection methods rely on. Further empirical experiments and theoretical analysis indicate that Proximal Policy Optimization (PPO) style importance sampling and clipping objectives are the root cause of this detection concealment, indicating that **a broad** class of RL methods may inherently exhibit similar concealment capability; (II) when SFT contamination with CoT is applied to advanced LRMs as the final stage, most contamination detection methods **perform near random guesses**. Without exposure to non-members, contaminated LRMs would still have more confidence when responding to those unseen samples that share similar distributions to the training set, and thus, evade existing memorization-based detection methods. Together, our findings reveal the unique vulnerability of LRMs evaluations: Model developers could easily contaminate LRMs to achieve inflated leaderboards performance while leaving minimal traces of contamination, thereby strongly undermining the fairness of evaluation and threatening the integrity of public leaderboards. This underscores the urgent need for advanced contamination detection methods and trustworthy evaluation protocols tailored to LRMs. Our code is available at https://github.com/ASTRAL-Group/ LRM_Conta_Detection_Arena.git.

## 1 Introduction

Competition among model developers has intensified as Large Language Models (LLMs) have demonstrated remarkable capabilities in various real-world tasks (Achiam et al., 2023; Wang et al., 2024). The leaderboards for performance are becoming a competitive arena for all state-of-the-art (SOTA) LLMs. However, inadvertently, benchmark samples may appear during LLMs' pre-training due to vast amounts of web-scraped training data. In addition, in the pursuit of publicity, some model developers may even deliberately incorporate benchmark data into their training sets (Sun et al., 2025), resulting in inflated benchmark performance and leaderboard rankings. We refer to this as the benchmark contamination problem in LLMs (Xu et al., 2024; Balloccu et al., 2024). Accordingly, various benchmark contamination detection methods have been proposed to determine whether specific benchmarks were used during training (Yeom et al., 2018; Mattern et al., 2023; Shi et al., 2023; Dong et al., 2024; Tu et al., 2024), based on the assumption that contamination in LLMs primarily involves memorizing the benchmark data (Wu et al., 2025). These methods rely on
∗Equal Contribution.

![1_image_0.png](1_image_0.png)

separability in some distributions between members (i.e., seen samples during contamination) and non-members (i.e., unseen samples). However, as LLMs have started to evolve into Large Reasoning Models (LRMs) (Guo et al., 2025; Jaech et al., 2024), benchmark contamination detection faces two key challenges: (1) LRMs rely on chain-of-thought (CoT) reasoning to reach final answers, but model developers would not release their training CoT data, and contamination detectors typically only have access to question-answer pairs without the intermediate reasoning steps used during training. This absence of training sequences makes detection substantially more challenging. (2) LRMs primarily acquire reasoning abilities during two stages: SFT and RL. This potentially provides developers with opportunities to manipulate leaderboard performance by strategically contaminating benchmarks in the earlier stage (e.g., SFT), while evading detection methods through subsequent training (e.g., RL). Given these challenges, the effectiveness of existing detection methods against LRM contamination remains uncertain. In this paper, we present the first systematic study of benchmark contamination in LRMs, structured around two points where contamination can happen. In particular, **Stage I (pre-LRM)** investigates contamination introduced to the base model while acquiring reasoning ability via SFT and RL; **Stage**
II (post-LRM) investigates contamination applied to an advanced LRM as a final SFT step. Under each stage, we comprehensively evaluate the effectiveness of existing detection methods.

Stage I (pre-LRM): contamination happens when the base model evolves into LRMs. We simulate contamination introduced during the period which the base model acquires reasoning ability through SFT and RL. After evaluating 10 representative contamination detection methods spanning generation-based, perturbation-based, reference-based, and reference-free approaches, we find that while SFT contamination to the base model is initially detectable, contamination evidence can be concealed through subsequent GRPO (Shao et al., 2024) training with clean samples. To isolate the core reasons behind GRPO's ability to conceal contamination, we conducted carefully designed controlled experiments to rule out the possibility that simply training with more clean samples results in the observed concealment, pointing to the conclusion that the GRPO optimization objective might be the primary driver for obscuring contamination. Then, we performed a theoretical analysis showing that the PPO-style importance sampling/clipping gate can drive the drop in detection performance. Our ablation studies confirm that while plain rejection sampling (RAFT) will not shrink the member/non-member separability, its variant RAFT++ (Xiong et al., 2025) that adds on the importance sampling/clipping term again makes detection harder. As many RL algorithms adopt similar training objectives, this demonstrates a significant risk to the integrity of benchmark evaluations. Stage II (post-LRM): contamination with CoT applied to LRMs. We simulate contamination with CoT introduced to advanced LRMs as the final training step. Surprisingly, although exclusively SFT on the benchmark samples with CoT yields a huge inflated performance, it leaves little evidence to existing detection approaches: almost all the detection approaches consistently perform near random guess in all the benchmarks. The log-prob distributions of both members and non-members show that without exposure to non-members, contaminated LRMs still have more confidence when responding to those unseen samples that are similar to the training set. This may undermine the key assumption behind many existing detection techniques that the benchmark contamination problem is primarily about memorizing samples (Morris et al., 2025; Hayes et al., 2025). Overall, our findings reveal that existing contamination detection methods are fragile under LRM contamination scenarios: RL conceals SFT contamination evidence introduced during the transition from base models to LRMs, while contamination with CoT applied to advanced LRMs leaves little detectable evidence. These findings underscore the urgent need for advanced contamination detection methods and trustworthy evaluation protocols tailored to LRMs. Accordingly, we outline potential directions for guaranteeing the integrity of evaluating LRMs (Section 5). We hope that our discoveries will inspire further research dedicated to building fair evaluation arenas for LRMs.

## 2 Related Works

Benchmark Contamination Detections. Benchmark contamination detection methods aim to identify whether evaluation datasets have been exposed during training (Oren et al., 2023). Prior work has proposed approaches based on: instance similarity (Karamolegkou et al., 2023), probability analysis (Mattern et al., 2023), instance generation (Deng et al., 2023; Ranaldi et al., 2024), and answer memorization (Yim et al., 2024). In this work, we select representative methods applicable to our setting, from probability analysis and instance generation, and further categorize them into: generation-based (Dong et al., 2024; Wu et al., 2025), perturbation-based (Li et al., 2025; Mattern et al., 2023), reference-based (Mireshghallah et al., 2022; Carlini et al., 2021), embedding-based (Tu et al., 2024; Liu et al., 2024), and reference-free (Zhang et al., 2024; Li et al., 2025; Yeom et al., 2018; Shi et al., 2023) methods. Each of these relies on distinct assumptions (Fu et al., 2024), and their effectiveness in the LRMs contamination scenario remains underexplored.

LRMs. LRMs achieve superior performance on challenging mathematical and coding tasks (Team et al., 2025), driven by inference-time scaling (Jaech et al., 2024; Snell et al., 2024; Zhang et al., 2025). To endow reasoning abilities to existing models, numerous efforts have been focusing on either SFT distillation (Li et al., 2025; Muennighoff et al., 2025; Guha et al., 2025; Ye et al., 2025; Bercovich et al., 2025) or RL with verifiable rewards (Liu et al., 2025a; Zeng et al., 2025; Yue et al., 2025). In SFT distillation, model developers distill knowledge from advanced LRMs into smaller models (Guo et al., 2025). While RL enables models to generate rollouts and receive rewards from verifiers, improving models' reasoning ability through feedback (Liu et al., 2025a; Zeng et al., 2025; Yue et al., 2025; Liu et al., 2025b). These two stages create many opportunities for developers to contaminate the benchmarks and evade detection. Benchmark Contamination Concealment. Model developers hope to conceal contamination evidence while still having performance inflation (Dominguez-Olmedo et al., 2024). Prior work has explored evading detection through benchmark augmentation, such as rephrasing solutions with strong LLMs (Dekoninck et al., 2024; Samuel et al., 2024), but in LRM settings, most benchmarks only have question–answer pairs without step-by-step solutions, making such methods inapplicable. (Bordt et al., 2024) explores from the training dynamic perspective, showing that performance inflation due to contamination diminishes as pre-training progresses. To our knowledge, we are the first to investigate contamination concealment at the algorithmic level.

## 3 Rl Conceals Contamination (Stage I: Pre-Lrm)

Contamination Setup. We define SFT contamination as the model being exposed to both the benchmark question and responses distilled from an advanced LRM, where RL contamination refers to the model encountering the benchmark question and having received rewards based on its generated responses during RL finetuning. For each dataset, we randomly sample half of the questions as the member set (used for contamination) and leave the remaining half as the non-member set (for detection evaluation). More details about our contamination pipelines, datasets, and implementation can be found in appendix D.1, D.3, and D.4. Detection Setup. We consider 10 representative detection methods. For each question, we generate 8 responses and compute the detection value on each response, then average these values to obtain a final detection score for the question. For the rationale and ablation studies of choosing responses to compute the detection scores, please refer to Appendix E.2. We report Area Under the Receiver Operating Characteristic (AUROC) by comparing detection scores between member and non-member sets within the same benchmark. Higher AUROC values indicate better detection.

## 3.1 Grpo Conceals Benchmark Contamination

Contamination Inflation Mainly Comes From SFT. We evaluate multiple contamination scenarios that may happen during SFT and RL and summarize the empirical results in Tab. 1. Results show that clean SFT training yields an 11.30% improvement in pass@1 performance, while SFT contamination further inflates results by an additional 8.82% on average across six benchmarks when

| SFT Data                          | RL Data     | Olypaid GPQA AIME25 AIME24 Minerva AMC23   | Avg.   |       |       |       |       |       |
|-----------------------------------|-------------|--------------------------------------------|--------|-------|-------|-------|-------|-------|
| Base model: Qwen2.5-7B-Instruct   |             |                                            |        |       |       |       |       |       |
| Clean & Mem Clean & Mem           | 52.56       | 44.70                                      | 30.00  | 30.00 | 39.52 | 73.00 | 44.96 |       |
| Clean & Mem                       | Clean       | 52.52                                      | 45.71  | 34.67 | 28.00 | 39.89 | 72.50 | 45.55 |
| Clean & Mem                       | /           | 53.77                                      | 49.58  | 31.62 | 32.73 | 40.74 | 74.92 | 47.23 |
| Clean                             | Clean & Mem | 44.62                                      | 40.74  | 24.85 | 27.88 | 35.23 | 65.00 | 39.72 |
| Clean                             | Clean       | 47.11                                      | 41.41  | 24.44 | 26.67 | 32.72 | 70.83 | 40.53 |
| Clean                             | /           | 44.35                                      | 40.34  | 24.79 | 23.54 | 34.24 | 63.20 | 38.41 |
| /                                 | /           | 36.48                                      | 32.20  | 2.50  | 10.83 | 28.58 | 52.50 | 27.18 |
| Base model: Llama-3.1-8B-Instruct |             |                                            |        |       |       |       |       |       |
| Clean & Mem Clean & Mem           | 44.30       | 43.18                                      | 25.42  | 24.58 | 35.20 | 61.25 | 38.99 |       |
| Clean & Mem                       | Clean       | 44.07                                      | 48.48  | 27.78 | 25.56 | 37.32 | 66.88 | 41.68 |
| Clean & Mem                       | /           | 46.07                                      | 42.80  | 26.67 | 26.67 | 35.20 | 66.67 | 40.68 |
| Clean                             | Clean & Mem | 44.54                                      | 40.74  | 25.83 | 23.33 | 29.53 | 61.56 | 37.59 |
| Clean                             | Clean       | 42.81                                      | 37.37  | 18.33 | 19.17 | 30.15 | 64.38 | 35.37 |
| Clean                             | /           | 40.69                                      | 39.23  | 16.67 | 18.33 | 27.70 | 56.88 | 33.25 |
| /                                 | /           | 15.63                                      | 29.67  | 0.00  | 4.17  | 19.49 | 19.00 | 14.66 |

starting with Qwen2.5-7B-Instruct. In contrast, RL contamination, despite introducing the benchmark questions and giving rewards based on the model-generated responses, shows no significant difference compared to using a clean RL training set after short training steps. To understand whether current contamination detection methods can still successfully detect contamination in LRMs, and whether RL training can alter the signals exploited by contamination detectors, we evaluate SFT-contaminated models before and after GRPO. Tab. 2 reveals systematic shifts in AUROC across diverse detection methods. Our analysis highlights four key observations: SFT contamination can be detectable at first. When starting with Qwen2.5-7B-Instruct, several reference-free approaches (Min-K% (Shi et al., 2023), Max-K% (Maini et al., 2024), and LOSS (Carlini et al., 2021)) can detect SFT contamination at a certain level, achieving AUROC around 73.42% across six contaminated benchmarks. The reference-based detection approach, LiRA (Mireshghallah et al., 2022), which assumes access to the training data distribution, also demonstrates superior performance with an average AUROC of 89.13% across six benchmarks. Similar results have already been observed when starting with Llama-3.1-8B-Instruct. GRPO conceals contamination. After applying GRPO to the SFT-contaminated model, we observe a consistent decrease in AUROC across all detection methods and benchmarks. We further analyze the average log probability of member and non-member samples before and after GRPO training, selecting Qwen2.5-7B-Instruct as the base model. Fig.3 shows two key patterns: (1) GRPO lowers the entropy of generated sequences, indicating that the model becomes more confident in its generation, which is consistent with prior observations in (Cui et al., 2025); (2) the log prob distribution of members and non-members converge after GRPO. Since the gaps in log prob are the core statistical backbone of existing contamination detectors, these findings suggest that GRPO may inherently suppress contamination evidence by rendering members and non-members indistinguishable. More GRPO, less contamination evidence. To examine whether the concealment effect strengthens with additional training, we extend GRPO to SFT-contaminated models using 10K questions from DeepMath-103K (He et al., 2025) for one epoch (156 steps). As shown in Fig.2, AUROC consistently decreases across all detection methods and benchmarks as the number of GRPO steps increases. Given that our maximum 156 training steps are still far fewer than the steps used in some advanced open-sourced reasoning models (Luo et al., 2025b;a), we expect that extensive GRPO
training would render all existing detection methods to near-random performance eventually.

Further training will not make models forget contamination. One possible explanation is that additional training makes models forget contamination, thus detections perform random guessing and pass@1 match the clean SFT baseline. To test this, we examine it with two experiments. First, we train SFT contaminated models with GRPO on both clean and contaminated datasets. As shown in Tab. 2, we observe a comparable drop in AUROC relative to the no RL baseline, similar to per-

CDD (Dong et al., 2024)Before RL 55.75 57.32 41.56 59.11 59.27 61.75 55.80 +0.00

RL w/ Clean 55.47 51.08 43.33 60.00 60.18 62.00 55.34 -0.46

RL w/ Clean&Mem 56.32 44.14 35.56 65.11 60.31 49.38 51.80 -3.95

Perturbation based

Neighbor (Mattern et al., 2023)Before RL 54.76 41.19 50.00 41.56 55.64 61.10 50.71 +0.00

RL w/ Clean 54.10 39.68 50.67 44.22 53.42 60.50 50.43 -0.28

RL w/ Clean&Mem 53.05 41.08 50.44 52.67 68.16 64.00 54.90 +4.19

Reference based

LiRA (Mireshghallah et al., 2022)Before RL 85.37 86.80 100.00 82.00 87.01 93.62 89.13 +0.00

RL w/ Clean 74.41 84.65 70.22 87.78 81.04 82.75 80.14 -8.99

RL w/ Clean&Mem 69.73 77.85 63.11 82.22 79.05 77.38 74.89 -14.24

Ref (Carlini et al., 2021)Before RL 73.27 63.30 60.22 41.11 73.10 82.00 65.50 +0.00

RL w/ Clean 66.77 58.41 45.33 51.11 65.54 73.62 58.08 -7.42

RL w/ Clean&Mem 62.77 54.17 43.11 50.44 65.38 72.62 58.86 -6.64

Reference free

Zlib (Carlini et al., 2021)Before RL 49.38 58.61 73.56 43.56 50.19 45.00 53.38 +0.00

RL w/ Clean 45.94 54.99 66.22 35.56 46.65 39.38 48.12 -5.26

RL w/ Clean&Mem 46.04 55.30 64.89 28.89 44.87 39.00 44.74 -8.64

Min–K%++ (Zhang et al., 2024)Before RL 47.57 50.90 41.90 59.11 52.27 45.88 49.61 +0.00

RL w/ Clean 46.25 46.78 36.67 50.89 51.35 29.62 43.59 -6.02

RL w/ Clean&Mem 43.77 48.21 21.78 38.00 48.91 43.62 40.72 -8.89

Min–K% (Shi et al., 2023)Before RL 69.19 69.51 85.56 75.56 71.16 78.75 74.96 +0.00

RL w/ Clean 55.19 60.60 62.89 65.56 61.50 61.87 61.27 -13.69

RL w/ Clean&Mem 53.93 59.74 59.56 62.67 57.31 59.25 58.54 -16.42

Max–K% (Maini et al., 2024)Before RL 64.50 64.31 65.11 81.78 67.27 76.00 69.83 +0.00

RL w/ Clean 53.05 51.43 49.78 50.22 51.84 57.75 52.35 -17.48

RL w/ Clean&Mem 49.03 51.04 50.00 50.00 52.34 47.50 49.99 -19.84

Loss (Carlini et al., 2021)Before RL 69.18 69.81 86.22 77.33 70.95 79.38 75.48 +0.00

RL w/ Clean 55.22 60.50 62.44 65.78 61.50 62.12 61.26 -14.22

RL w/ Clean&Mem 53.99 60.01 59.33 62.67 57.40 59.38 58.80 -16.68

forming RL solely on clean data. Also, the contaminated model, further trained with GRPO, still shows an average performance inflation of 7.14% across six benchmarks and does not fall back as the clean SFT model, shown in Tab. 1. Second, we continue SFT on the SFT contaminated model with an additional 4 epochs on clean data. Fig. 2 and Tab. 23 demonstrate that further SFT is unable to conceal the benchmark contamination, while the pass@1 would continue to rise. Together, these results show that subsequent GRPO training preserves performance inflation while reducing detectable evidence of contamination may have some underlying reasons, rather than simply forgetting the contamination after further training.

## 3.2 Theoretic Analysis

In this section, we perform theoretical analysis to demonstrate that PPO-style clipping and importance sampling are the root cause of the concealment. Intuitively, the importance sampling and clipping term reweights terms so that the most off-policy trajectories are damped by the clip while typical on-policy ones keep their influence. This reweighting hits non-members more as they have more extreme successes, so clipping cuts misaligned influence and lets ordinary, on-policy successes steer the update. With more headroom, non-member's NLL drops more and the gap contracts. Setup. We denote ℓ(*x, y*) to be the negative log likelihood (NLL) of the current model of generating y given prompt x, members as M and non-members as N, policy model πk at step k. We focus on analyzing the gap Gk of negative log likelihood for members and non-members on correct samples (i.e., r = 1), as assessing contamination on erroneous outputs is not especially meaningful. Formally, we can write

$$G_{k}:=\mathbb{E}_{x\sim N}\mathbb{E}_{y\sim\pi_{k}(\cdot|x)}[\ell_{k}(x,y)\mid r=1]-\mathbb{E}_{x\sim M}\mathbb{E}_{y\sim\pi_{k}(\cdot|x)}[\ell_{k}(x,y)\mid r=1]$$

![5_image_0.png](5_image_0.png)

Figure 2: AUROC (%) trends on SFT contaminated model further trained with different objectives. While contamination introduced through SFT is initially detectable by existing methods, subsequent RL training with clean samples (e.g., GRPO or RAFT++) consistently degrades detection performance. Moreover, we observe a monotonic decline in detection performance as the number of RL steps increases, and reference-free methods (e.g., Loss, Min-K, and Max-K) already fall into near random guesses (i.e., AUROC≈50%) simply after 156 steps. The base model is Qwen2.57B-Instruct. More results of Llama-3.1-8B-Instruct as the base model are shown in Fig.5.

![5_image_1.png](5_image_1.png)

Figure 3: Log-prob distributions for members vs. non-members of SFT contaminated model before and after RL training. After further GRPO with clean samples on the SFT contaminated model, the log-prob distributions of members and non-members become increasingly similar. Since many contamination detection methods rely on separability in this space, the shrinking gap explains their degraded effectiveness. More log-prob distributions can be found in Fig. 7, 8, and 9.

If this gap contracts, i.e., Gk+1 − Gk < 0, members and non-members become closer in the NLL
sense, making contamination detection harder since many methods (Zhang et al., 2024; Shi et al.,
2023; Maini et al., 2024; Carlini et al., 2021) are based on the separation of NLLs. For a fixed prompt x, we define the NLL drift as

$$\Delta_{x}:=\mathbb{E}_{\pi_{k+1}}[\ell_{k+1}\mid r=1,x]-\mathbb{E}_{\pi_{k}}[\ell_{k}\mid r=1,x].$$
[ℓk | r = 1, x]. (2)
We notice that we can rewrite the NLL gap as

$$(2)$$

$$G_{k+1}-G_{k}:=\mathbb{E}_{x\in N}[\Delta_{x}]-\mathbb{E}_{x\in M}[\Delta_{x}].$$
Gk+1 − Gk := Ex∈N [∆x] − Ex∈M[∆x]. (3)
In our following analysis, we thus focus on investigating the behavior of ∆x on members and nonmembers. If an algorithm yields on average smaller ∆x on non-members, the algorithm should be able to conceal contamination.

Notations. At token t, let At be the method's per token reward/advantage and wt the weight from importance sampling and clipping. Define

$$A_{t}^{w}:=w_{t}A_{t},\quad\bar{A}^{w}(s):=\mathbb{E}_{a\sim\pi_{k}(\,\cdot\,|s)}[A^{w}(s,a)],\quad\bar{A}_{t}^{w}:=A_{t}^{w}-\bar{A}^{w}(s_{t})$$
t − A¯w(st) (4)
to measure how good a state is compared to the average. In particular, wt =ρtmt with ρt =πθ(at | st)/πold(at | st) being the importance sampling and mt ∈ {0, 1} be a mask indicating if the clipping is activated, specifically mt = 0 indicates that there is no gradients from the update. Moreover, we define pk(x)=Ey∼πk(·|x)[r(*x, y*)] to be the overall success rate of the prompt, and a value function as qk(*s, a*):=Pr(r = 1|*s, a*) and pk(s):=Ea∼πk(·|s)[qk(s, a)] for success rate at that state. And we

$$({\mathfrak{I}})$$
$$(4)$$

Table 3: **AUROC (%)** of detection approach, Loss (Carlini et al., 2021), evaluated on SFT contaminated model further trained with different RL objectives. The gray row indicates no ablation on the objective, and ✗ means remove the term from the objective. ∆ measures the difference with the SFT contaminated model w/o RL (Tab. 2). RL steps are 64, or the step before the model collapses. The results show that clipping is the main driver for the contraction, which aligns with our theory.

| Training Objectives Clipping Olypaid GPQA AIME25 AIME24 Minerva AMC23 Avg.   | ∆   |       |       |       |       |       |       |              |       |
|------------------------------------------------------------------------------|-----|-------|-------|-------|-------|-------|-------|--------------|-------|
| RAFT                                                                         | ✗   | 71.78 | 69.78 | 86.00 | 86.67 | 71.58 | 79.25 | 77.51 +2.03  |       |
| RAFT++                                                                       | !   | 50.43 | 58.45 | 67.56 | 66.67 | 52.84 | 49.50 | 57.58 -17.91 |       |
| RAFT++                                                                       | ✗   | 69.16 | 73.68 | 74.44 | 76.22 | 71.08 | 81.75 | 74.39        | -1.09 |
| GRPO                                                                         | !   | 55.22 | 60.50 | 62.44 | 65.78 | 61.50 | 62.12 | 61.26 -14.22 |       |
| GRPO                                                                         | ✗   | 68.83 | 70.20 | 80.44 | 73.78 | 68.30 | 78.12 | 73.28        | -2.20 |

define B(s) =Ea∼π[ρ(s, a)m(s, a)qk(*s, a*)] and C(s) =Ea∼π[ρ(s, a)m(*s, a*)]. We assume that the RL training is performed on the benchmark data (i.e., training data is the combination of members M and non-members N), and it is in a tabular setting for simplicity. Since members have been utilized during training, it is natural to assume the pk(s) for members are larger than non-members, and the NLL for members is lower than non-members. Theorem 3.1. For a small natural gradient step with step size η *on a PPO style loss, we have*

$$\Delta_{x}=-\eta\underbrace{\mathbb{E}\!\left[\frac{1}{T}\sum_{t=1}^{T}\bar{A}_{t}^{w}\,\right|\,r=1,x\right]}_{(A)\;\mu(x)}+\eta\underbrace{\mathrm{Cov}\!\left(\ell_{k},\sum_{t=1}^{T}\bar{A}_{t}^{w}\right)}_{(B)\;\mathrm{covariance}\;\beta(x)}+O(\eta^{2})$$
(5)  $\frac{1}{2}$
The proof can be found in appendix C. Intuitively, µ(x) measures the average push on the example's NLL from correct trajectories, where β serves as a reweighting term accouting for the importance sampling/clipping. Here we consider several instantiations using different algorithms to investigate the core driver for contraction. The training objectives for each algorithm are listed in Appendix B.

RAFT. In plain rejection sampling, we have wt = 1 and At =1{r = 1}, so on correct trajectories

$$\bar{A}_{t}^{w}=1-p_{k}(s_{t}),\quad\mu^{\mathrm{RATT}}(x)=\mathbb{E}\left[\frac{1}{T}\sum_{t=1}^{T}\left(1-p_{k}(s_{t})\right)\Bigg|\,r=1,x\right]$$

The covariance term is

$$\beta^{\mathrm{RAFT}}(x)=\mathrm{Cov}\Big(\ell_{k},\;\sum(1-p_{k}(s_{t}))\Big)=-\,\mathrm{Cov}\Big(\ell_{k},\;\sum p_{k}(s_{t})\Big).$$
$$\Delta_{N}=$$

We note that lower loss ℓk corresponds to higher probabilities pk(st), and thus β RAFT(x)>0. Moreover, non-members correct trajectories can exhibit much higher variance in loss and probabilities, thus, the βN term is typically larger than βM. Consequently,

$$-\,\Delta_{M}=-\eta\big(\mu_{N}-\mu_{M}\big)+\eta\big(\beta_{N}-\beta_{M}\big)$$

where both gaps (µN − µM) and (βN − βM) are positive. Empirically, the covariance gap offsets the mean gap, yielding ∆N − ∆M ≥ 0, i.e., RAFT is unable to conceal contamination evidence. RAFT++. Using the same At = 1{r = 1}, on r = 1 paths

$$\bar{A}_{t}^{w}=\rho_{t}m_{t}-B_{k}(s_{t}),\quad\mu^{\mathrm{RAFT++}}(x)=\mathbb{E}\left[\frac{1}{T}\sum_{t=1}^{T}\left(\rho_{t}m_{t}-B_{k}(s_{t})\right)\Biggm|r=1,x\right].$$

We note that the difference of µ cannot possibly lead to large deviations between members/nonmembers as 0≤ ρtmt ≤1 + ϵand Bk(st)≤1 for both groups and the term is normalized by length.

For the covariance term though, we have

$$\beta^{\mathrm{RAFT}++}(x)=\mathrm{Cov}\Big{(}\ell_{k},\ \sum(\rho_{t}m_{t}-B_{k}(s_{t}))\Big{)}=\mathrm{Cov}\Big{(}\ell_{k},\ \sum\rho_{t}m_{t}\Big{)}-\mathrm{Cov}\Big{(}\ell_{k},\ \sum B_{k}(s_{t})\Big{)}.$$

Compared to RAFT, the new term Cov(ℓk,Pρtmt) is negative as correct path with higher loss are anomaly and typically got clipped more. Moreover, this is much more prominent in non-members

Models Olypaid GPQA AIME25 AIME24 Minerva AMC23 Avg.

DeepSeek-R1-Distill-Llama-8B 52.10 43.94 33.33 43.33 32.97 84.58 48.38

,→ w/ extensive SFT Contamination 61.83 53.16 51.67 61.67 38.74 93.75 60.14

DeepSeek-R1-Distill-Qwen-7B 55.70 48.65 39.26 53.70 37.25 91.94 54.42

,→ w/ extensive SFT Contamination 58.77 50.87 42.59 58.91 40.81 90.67 57.10

OpenThinker3-7B (15K) 50.81 41.67 21.67 29.17 34.01 77.50 42.47

,→ w/ extensive SFT Contamination 52.74 47.64 33.33 30.48 40.56 78.70 47.25

DeepSeek-R1-Distill-Qwen-14B 59.89 56.69 44.44 62.78 42.28 92.92 59.83

,→ w/ extensive SFT Contamination 66.37 62.75 64.58 77.78 46.14 97.81 69.24

due to high variance in correct trajectories loss. The second covariance term, although still negative, are not that significant for non-members compared to members due to an average over all possible actions. Therefore, overall it leads to
$$\Delta_{N}-\Delta_{M}=-\eta(\mu_{N}-\mu_{M})+\eta(\beta_{N}-\beta_{M})\;<\;0,$$
i.e., RAFT++ contracts the membership gap. The driver is precisely the PPO-style importance
sampling/clipping: it removes the RAFT covariance cancellation by making Cov(ℓk,Pρm) nonpositive and more negative for non-members. GRPO. Finally, we investigate the GRPO contraction term. To ease the analysis, we consider an
idealized setting where we define the advantage term as Ak(*x, y*)=r(x, y)−pk(x) with no standard
deviation term and A˜w
t =A˜RAFT
t −pk(x)(ρtmt−C(st)). Clearly, we have
$$\mu^{\mathrm{GRPO}}(x)=\mu^{\mathrm{RATT+}}(x)-p_{k}(x)\mathbb{E}\bigg[\frac{1}{T}\sum\left(\rho_{t}m_{t}-C_{k}(s_{t})\right)\biggm|r=1,x\bigg],$$ $$\beta^{\mathrm{GRPO}}(x)=\beta^{\mathrm{RATT+}}(x)-p_{k}(x)\,\mathrm{Cov}\bigg(\ell_{k},\,\sum(\rho_{t}m_{t}-C_{k}(s_{t}))\bigg)$$
By similar argument, we know that the µ term does not contribute significantly to the concealment. The covariance term can be analyzed similarly to show that the concealment also happen on GRPO thanks to the importance sampling and clipping term.

## 3.2.1 Empirical Support

To confirm empirically the prediction of our theoretical results, we evaluate the Loss detector (Carlini et al., 2021) after training with RAFT (Dong et al., 2023)/RAFT++ (Xiong et al., 2025)/GRPO.

The overall results can be found in table 3. We conduct the ablation study using Qwen2.5-7B-
Instruct as the base model. From the results, there are several observations. Effect on detectability. Under RAFT, the Loss detector (Carlini et al., 2021) performance remains essentially unchanged relative to the SFT contaminated baseline w/o further RL. In contrast, RAFT++ and GRPO (with clipping enabled) produce a sharp drop in detector performance. Importance sampling vs. clipping. The clipping term, often treated purely as a training stabilizer, materially contributes to concealment, as predicted by theory. When we retain importance sampling but *remove clipping* in RAFT++ and GRPO, both algorithms show little to no reduction in Lossdetector performance (Table 3). Intuitively, as the clip threshold P 
ϵ→∞, the effective weight satisfies t ρtmt ≈ T, and the covariance term in our decomposition tends toward zero for both members and non-members, eliminating the shrinkage effect. These two observations perfectly reflect our theoretical analysis, empirically validating that the PPO- style importance sampling/clipping term is the key driver behind GRPO contamination concealment. Given that many RL algorithms adopt this term in their objectives, this suggests that a broad class of RL methods may inherently exhibit similar concealment capability.

## 4 Contamination With Cot On Advanced Lrms Barely Leaves Evidence (Stage Ii: Post-Lrm)

Contamination Setup. In this setup, we simulate contamination with CoT applied to advanced LRMs at the final stage of training. We use DeepSeek-R1-Distill-Llama-8B, DeepSeek- R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-14B (Guo et al., 2025), and checkpoints from

| is averaged over detection scores from 8 rollouts. Contamination Detection Methods Init Models Olympiad   | GPQA         | AIME25   | AIME24   | Minerva   | AMC23   | Avg.   |       |       |
|-----------------------------------------------------------------------------------------------------------|--------------|----------|----------|-----------|---------|--------|-------|-------|
| Generation based                                                                                          | DS Llama-8B  | 48.73    | 50.45    | 41.33     | 61.56   | 59.10  | 40.63 | 50.30 |
| DS Qwen-7B                                                                                                | 46.87        | 55.85    | 60.44    | 68.89     | 56.87   | 50.63  | 56.59 |       |
| Verbatim (Wu et al., 2025)                                                                                | OpenThink-7B | 43.78    | 55.36    | 60.89     | 56.67   | 51.78  | 42.38 | 51.81 |
| DS Qwen-14B                                                                                               | 48.51        | 50.73    | 52.38    | 61.11     | 55.18   | 53.79  | 53.62 |       |
| DS Llama-8B                                                                                               | 51.84        | 53.83    | 60.00    | 53.11     | 58.08   | 57.50  | 55.73 |       |
| DS Qwen-7B                                                                                                | 51.46        | 48.29    | 50.00    | 53.78     | 54.71   | 41.00  | 49.87 |       |
| CDD (Dong et al., 2024)                                                                                   | OpenThink-7B | 49.98    | 50.23    | 53.31     | 51.24   | 54.52  | 50.44 | 51.62 |
| DS Qwen-14B                                                                                               | 55.82        | 45.50    | 43.11    | 46.67     | 56.13   | 56.45  | 50.61 |       |
| Perturbation based                                                                                        | DS Llama-8B  | 49.94    | 39.32    | 53.11     | 43.33   | 49.68  | 60.00 | 49.23 |
| DS Qwen-7B                                                                                                | 52.99        | 40.29    | 62.44    | 49.33     | 55.34   | 54.87  | 52.54 |       |
| Neighbor (Mattern et al., 2023)                                                                           | OpenThink-7B | 53.76    | 42.95    | 34.00     | 42.22   | 52.89  | 51.50 | 46.22 |
| DS Qwen-14B                                                                                               | 53.20        | 42.23    | 50.89    | 44.00     | 53.46   | 57.38  | 50.19 |       |
| Reference based                                                                                           | DS Llama-8B  | 57.92    | 53.01    | 53.56     | 75.33   | 69.44  | 58.75 | 61.34 |
| DS Qwen-7B                                                                                                | 46.52        | 43.93    | 50.22    | 58.89     | 59.33   | 54.00  | 52.15 |       |
| LiRA (Mireshghallah et al., 2022)                                                                         | OpenThink-7B | 62.35    | 64.77    | 58.44     | 64.44   | 64.81  | 61.62 | 62.74 |
| DS Qwen-14B                                                                                               | 59.93        | 55.23    | 75.56    | 66.00     | 66.55   | 70.00  | 65.55 |       |
| DS Llama-8B                                                                                               | 53.79        | 46.50    | 46.44    | 64.00     | 63.57   | 51.25  | 54.26 |       |
| DS Qwen-7B                                                                                                | 53.30        | 44.37    | 46.89    | 44.22     | 53.09   | 41.75  | 47.27 |       |
| Ref (Carlini et al., 2021)                                                                                | OpenThink-7B | 57.34    | 49.86    | 37.56     | 50.44   | 59.30  | 69.12 | 53.94 |
| DS Qwen-14B                                                                                               | 55.75        | 47.55    | 52.67    | 30.89     | 55.51   | 53.75  | 49.35 |       |
| Reference free                                                                                            | DS Llama-8B  | 49.52    | 54.74    | 64.22     | 37.11   | 45.97  | 47.12 | 49.78 |
| DS Qwen-7B                                                                                                | 46.52        | 57.38    | 64.89    | 36.89     | 43.30   | 42.12  | 48.52 |       |
| Zlib (Carlini et al., 2021)                                                                               | OpenThink-7B | 45.65    | 55.37    | 74.22     | 36.89   | 43.51  | 36.62 | 48.71 |
| DS Qwen-14B                                                                                               | 48.12        | 56.71    | 70.44    | 43.56     | 45.92   | 51.50  | 52.71 |       |
| DS Llama-8B                                                                                               | 55.45        | 59.10    | 45.95    | 70.22     | 60.89   | 57.50  | 58.19 |       |
| DS Qwen-7B                                                                                                | 48.92        | 56.83    | 48.44    | 59.33     | 51.83   | 62.62  | 54.66 |       |
| Min–K%++ (Zhang et al., 2024)                                                                             | OpenThink-7B | 51.85    | 58.31    | 66.44     | 55.00   | 49.41  | 41.05 | 53.68 |
| DS Qwen-14B                                                                                               | 52.44        | 56.72    | 48.44    | 76.44     | 57.39   | 59.62  | 58.51 |       |
| DS Llama-8B                                                                                               | 57.86        | 61.68    | 53.33    | 72.67     | 67.12   | 61.87  | 62.42 |       |
| DS Qwen-7B                                                                                                | 49.75        | 53.93    | 51.78    | 61.56     | 54.50   | 56.75  | 54.71 |       |
| Min–K% (Shi et al., 2023)                                                                                 | OpenThink-7B | 53.52    | 57.19    | 60.44     | 57.56   | 54.83  | 47.37 | 55.15 |
| DS Qwen-14B                                                                                               | 52.77        | 58.08    | 52.44    | 77.33     | 59.43   | 59.62  | 59.95 |       |
| DS Llama-8B                                                                                               | 53.85        | 55.96    | 50.67    | 60.44     | 59.22   | 52.50  | 55.44 |       |
| DS Qwen-7B                                                                                                | 49.65        | 50.92    | 40.44    | 73.33     | 54.08   | 56.25  | 54.11 |       |
| Max–K% (Maini et al., 2024)                                                                               | OpenThink-7B | 55.12    | 58.29    | 46.22     | 79.33   | 54.20  | 59.38 | 58.76 |
| DS Qwen-14B                                                                                               | 50.43        | 53.89    | 50.00    | 50.00     | 51.08   | 52.50  | 51.32 |       |
| DS Llama-8B                                                                                               | 57.91        | 61.78    | 52.89    | 73.56     | 67.00   | 62.38  | 62.59 |       |
| Loss (Carlini et al., 2021)                                                                               | DS Qwen-7B   | 49.77    | 54.09    | 52.00     | 63.78   | 54.76  | 56.75 | 55.19 |
| OpenThink-7B                                                                                              | 53.44        | 57.61    | 61.33    | 56.67     | 55.07   | 48.12  | 55.37 |       |
| DS Qwen-14B                                                                                               | 52.81        | 58.39    | 52.89    | 77.56     | 59.37   | 60.37  | 60.23 |       |

OpenThought3 (Guha et al., 2025) as the initial models. We simulate extensive contamination with CoT by applying SFT exclusively on the member data in this section. Additional implementation details are provided in Appendix D.4. Tab. 4 and 5 show the results of pass@1 on six reasoning benchmarks and AUROC of detection approaches performance (w/ the same detection setup as Stage I), respectively. We observe that: Extensive SFT Contamination with CoT results in a huge performance inflation. As shown in Tab. 4, LRMs can substantially benefit from extensive contamination with CoT. Such inflation enables contaminated LRMs to artificially boost performance in benchmarks and have an overrated rank in the reasoning leaderboard with little extra training cost. Extensive contamination with CoT on LRMs barely leaves evidence. As illustrated in Tab. 5, detection methods, which were effective in contamination introduced when the base model evolves into LRMs, consistently fail under extensive contamination with CoT to LRMs, performing close to random guessing. The previous SOTA approach, LiRA (Mireshghallah et al., 2022), achieves only 58.74% AUROC on average across six benchmarks. Then, we analyze the log prob of member and non-member samples before and after final stage contamination, shown as Fig. 4. After

![9_image_0.png](9_image_0.png)

the extensive SFT contamination with CoT on members, the log prob of both members and nonmembers increases at a similar margin. This indicates that even without exposure to non-members, contaminated LRMs still have more confidence when responding to unseen samples that share similar distributions to training samples, which also explains why extensive contamination with CoT on LRMs barely leaves evidence. These results suggest that model developers could extensively contaminate their LRMs in the final stage while leaving little detectable evidence.

Discussion. Despite Feng et al. (2024) demonstrating that contamination detection could work in non-reasoning model scenarios, the detectors do not have access to the reasoning trajectories used in the LRM contamination scenario, so they have to rely on the generated responses from LRMs. However, LRMs typically possess strong reasoning abilities to output step-by-step long CoT and are difficult to converge on a specific sequence after contamination with long CoT. This may indicate that rather than memorizing specific reasoning trajectories, LRMs internalize the underlying knowledge and reasoning process during the contamination with CoT data, enabling generalization to distributionally similar questions (e.g., non-members). While most detection methods rely on the assumption that contaminated models would achieve lower loss on training sequences (Carlini et al., 2021) or generate less diverse responses for seen questions (Dong et al., 2024) than for unseen ones. Accordingly, these methods rely on a gap in certain metrics (e.g., log-probability, Levenshtein distance, etc.) between trained and unseen samples to determine contamination. Nevertheless, these LRMs could also have lower loss when responding to those unseen samples that share similar distributions to the training set, benefiting from their long CoT ability, as shown in Fig. 4. This confounding factor (i.e, generalization) is not accounted for by existing detection approaches, challenging the assumption that benchmark data contamination is more about memorization (Wu et al., 2025; Morris et al., 2025; Hayes et al., 2025).

## 5 Conclusion

We present the first systematic study of benchmark contamination in LRMs, structured around two points where contamination can happen. Our results reveal a critical vulnerability in LRM evaluation: contamination detection methods are fragile and contamination introduced at either stage can be concealed. In Stage I (pre-LRM), while SFT contamination to the base model is initially detectable, contamination evidence can be concealed through subsequent RL training. In Stage II (post-LRM), extensive contamination with CoT on advanced LRMs barely leaves evidence for existing memorization-driven detection methods. Our findings call for an urgent need of protocols that ensure fair evaluations among LRMs. Here, we propose two potential directions to ensure it: (I) Model developers should release more intermediate training checkpoints, enabling the community to better monitor and regulate potential benchmark contamination in each training stage. (II) Researchers working on contamination detections should advance beyond memorization-driven methods and explicitly account for the long CoT reasoning and generalization capacity of LRMs. Despite the assumption that contamination is about memorizing the training data inspires numerous detection methods before the LRM era, it may become outdated right now. Detection approaches that are solely based on log-probs or mitigation approaches such as minor benchmark modifications, are definitely inadequate in this context and risk systematically failing. These findings all highlight the need for new assumptions in contamination detection and the development of contamination-robust evaluation protocols in the LRM setting.

## Acknowledgements

The authors thank Rui Yang and Yifan Sun for their helpful feedback. Huan Zhang is supported in part by the AI2050 program at Schmidt Sciences (AI2050 Early Career Fellowship).

## Reproducible Claim

Our code is available at https://github.com/ASTRAL-Group/LRM_Conta_ Detection_Arena.git. Detailed implementation of detection approaches, SFT training, and RL training can be found in appendix D.2 and D.4. We also provide the proof of our theory in appendix C.

## Ethics Statement

We find a new vulnerability of LRM evaluations: contamination introduced at either stage can be concealed. Other than this, we do not have more ethics concerns.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

Simone Balloccu, Patr´ıcia Schmidtova, Mateusz Lango, and Ond ´ ˇrej Dusek. Leak, cheat, re- ˇ
peat: Data contamination and evaluation malpractices in closed-source llms. arXiv preprint arXiv:2402.03927, 2024.

Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, et al. Llama-nemotron: Efficient reasoning models. arXiv preprint arXiv:2505.00949, 2025.

Sebastian Bordt, Suraj Srinivas, Valentyn Boreiko, and Ulrike von Luxburg. How much can we forget about data contamination? *arXiv preprint arXiv:2410.03249*, 2024.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In *30th USENIX security symposium (USENIX Security 21)*, pp. 2633–2650, 2021.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. *arXiv preprint arXiv:2505.22617*, 2025.

Michael Han Daniel Han and Unsloth team. Unsloth, 2023. URL http://github.com/
unslothai/unsloth.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

Jasper Dekoninck, Mark Niklas Muller, Maximilian Baader, Marc Fischer, and Martin Vechev. ¨
Evading data contamination detection for language models is (too) easy. arXiv preprint arXiv:2402.02823, 2024.

Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, and Arman Cohan. Investigating data contamination in modern benchmarks for large language models. *arXiv preprint* arXiv:2311.09783, 2023.

Ricardo Dominguez-Olmedo, Florian E Dorner, and Moritz Hardt. Training on the test task confounds evaluation and emergence. *arXiv preprint arXiv:2407.07890*, 2024.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. *arXiv preprint arXiv:2304.06767*, 2023.

Yihong Dong, Xue Jiang, Huanyu Liu, Zhi Jin, Bin Gu, Mengfei Yang, and Ge Li. Generalization or memorization: Data contamination and trustworthy evaluation for large language models. arXiv preprint arXiv:2402.15938, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024.

Qizhang Feng, Siva Rajesh Kasa, Santhosh Kumar Kasa, Hyokun Yun, Choon Hui Teo, and Sravan Babu Bodapati. Exposing privacy gaps: Membership inference attack on preference data for llm alignment. *arXiv preprint arXiv:2407.06443*, 2024.

Yujuan Fu, Ozlem Uzuner, Meliha Yetisgen, and Fei Xia. Does data contamination detection work (well) for llms? a survey and evaluation on detection assumptions. arXiv preprint arXiv:2410.18966, 2024.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. Openthoughts: Data recipes for reasoning models. *arXiv preprint arXiv:2506.04178*, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

Jamie Hayes, Ilia Shumailov, Christopher A Choquette-Choo, Matthew Jagielski, George Kaissis, Katherine Lee, Milad Nasr, Sahra Ghalebikesabi, Niloofar Mireshghallah, Meenatchi Sundaram Mutu Selva Annamalai, et al. Strong membership inference attacks on massive datasets and
(moderately) large language models. *arXiv preprint arXiv:2505.18773*, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. *arXiv preprint* arXiv:2504.11456, 2025.

Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, and Yanning Chen. Liger kernel: Efficient triton kernels for llm training. *arXiv preprint arXiv:2410.10989*, 2024.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Antonia Karamolegkou, Jiaang Li, Li Zhou, and Anders Søgaard. Copyright violations and large language models. *arXiv preprint arXiv:2310.13771*, 2023.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.

Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Bespoke Labs. Bespoke-minicheck-7b, 2024. URL https://huggingface.co/
bespokelabs/Bespoke-MiniCheck-7B.

VI Lcvenshtcin. Binary coors capable or 'correcting deletions, insertions, and reversals. In Soviet physics-doklady, volume 10, 1966.