# Auto-Rt:Automatic Jailbreak Strategy Explo- Ration For Red-Teaming Large Language Mod- Els

Yanjiang Liu1,2, Shuheng Zhou3,†, Yaojie Lu1,2,†, Huijia Zhu3**, Weiqiang Wang**3, Hongyu Lin1,2, Ben He1,2, Xianpei Han1,2, Le Sun 1,2 1University of Chinese Academy of Sciences 2Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences 3Ant Group, China liuyanjiang22@mails.ucas.ac.cn, benhe@ucas.edu.cn {luyaojie,hongyu,xianpei,sunle}@iscas.ac.cn {shuheng.zsh,huijia.zhj,weiqiang.wwq}@antgroup.com

## Abstract

Automated red-teaming has emerged as an essential approach for identifying vulnerabilities in large language models (LLMs). However, most existing methods rely on fixed attack templates and focus primarily on individual high-severity flaws, limiting their adaptability to evolving defenses and their ability to detect complex, high-exploitability vulnerabilities. To address these limitations, we propose AUTO- RT, a reinforcement learning framework designed for automatic jailbreak strategy exploration, i.e., discovering diverse and effective prompts capable of bypassing the safety restrictions of LLMs. AUTO-RT autonomously explores and optimizes attack strategies by interacting with the target model and generating crafted queries that trigger security failures. Specifically, AUTO-RT introduces two key techniques to improve exploration efficiency and attack effectiveness: 1) Dynamic Strategy Pruning, which focuses exploration on high-potential strategies by eliminating highly redundant paths early, and 2) Progressive Reward Tracking, which leverages intermediate downgrade models and a novel First Inverse Rate (FIR) metric to smooth sparse rewards and guide learning. Extensive experiments across diverse white-box and black-box LLM settings demonstrate that AUTO-RT significantly improves success rates (by up to 16.63%), expands vulnerability coverage, and accelerates discovery compared to existing methods. 1

## 1 Introduction

As large language models (LLMs) are rapidly adopted across various applications, their safety risks have become increasingly prominent (Huang et al., 2023; Christian, 2021; Qi et al., 2024; Andriushchenko et al., 2025). Although safety-tuning methods improve alignment with human values and safty principles (Ji et al., 2025; Lee et al., 2024), the models' complexity and broad application scope leave many vulnerabilities undiscovered (Allspaw & Cook, 2010; Yang et al., 2023; Zhan et al., 2024). Proactive red-teaming (Wei et al., 2023), systematically probing models with jailbreak
(adversarial) prompts, is therefore essential for exposing these hidden flaws and for keeping LLMs reliable and robust (Roose, 2023; Jain et al., 2023; Deng et al., 2023). An effective red-teaming system should prioritize flaws that are both high exploitability and high severity (Bishop & Bailey, 1996; Bozorgi et al., 2010; Bhatt et al., 2021; Stickland et al., 2024). Specifically, exploitability measures how easily a normal prompt can trigger a flaw, while severity reflects how much harm results once that flaw is triggered. For instance, a hidden backdoor that leaks training data only when triggered by a rare, exact hash has low exploitability but high severity, as it poses serious risk but is rarely activated. In contrast, a prompt that anyone can use to produce slightly garbled text has high exploitability but low severity, since it occurs often but causes minimal harm. The flaws that matter most combine high exploitability and high severity: jailbreaks such as 1

![1_image_0.png](1_image_0.png) 
"Grandma's Exploit"2 or the "Past-Tense Attack" (Andriushchenko & Flammarion, 2024), where a simple phrase bypasses safety filters and elicits violent or hateful content (Anderljung et al., 2023). Current red-teaming approaches fall into two categories (Ganguli et al., 2022; Qi et al., 2023; Perez & Ribeiro, 2022; Bai et al., 2022): manually crafted prompts and automatic prompt mining. Manual red-teaming lets experts devise creative jailbreaks that can expose both easy-to-trigger and high-impact flaws. However, this process is slow, costly, and hard to scale as LLMs and their use cases grow. In contrast, automated red-teaming (Wei et al., 2023; Zhou et al., 2024b; Guo et al., 2024) reduces human effort but still relies on fixed templates that focus on high-severity outputs while overlooking exploitability. Previous automated red-teaming systems such as AutoDAN (Liu et al., 2024b), Rainbow-Teaming (Samvelyan et al., 2024), and PAIR (Chao et al., 2024b) generate jailbreak prompts within narrow, predefined strategy sets, leaving much of the potential vulnerability space unexplored. As a result, neither manual nor automated approaches consistently uncover flaws that are simultaneously highly exploitable and highly severe. To address these limitations, we propose AUTO-RT, a reinforcement learning framework for automatic jailbreak strategy exploration instead of relying on handcrafted prompts or fixed templates. AUTO-RT formulates prompt generation as a sequential decision process and explores a rich strategy space, allowing it to discover attacks that are simultaneously easy to trigger and highly harmful. This active strategy exploration approach removes human bias, expands coverage, and uncovers high-risk vulnerabilities that static strategy-driven methods often miss. Moreover, because AUTO-RT requires only the model's text output, it operates seamlessly in both white-box and black-box settings, offering a robust and scalable tool for comprehensive LLM safety evaluation.

To improve exploration efficiency and attack effectiveness, we introduce two key techniques: 1) Dynamic Strategy Pruning (DSP): During search, AUTO-RT evaluates partial explorations on-the-fly and terminates highly redundant branches. By discarding unpromising paths early, DSP concentrates on high-value regions of the jailbreak strategy space and speeds up exploration. **2) Progressive** Reward Tracking (PRT): Sparse rewards can hinder the exploration of effective jailbreak strategies. PRT mitigates this by maintaining a sequence of intermediate downgraded models and computing the FIR metric, which serves as an indicator of whether there is a significant difference in safety capability between a downgraded model and the target model. FIR converts a sparse success/fail signal into a graded score, guiding the red-teaming model toward stronger jailbreak strategies. We conduct extensive experiments across 16 white-box LLMs and 2 black-box LLMs. The results show that AUTO-RT significantly outperforms existing methods by improving success rates (by up to 16.63%), expanding vulnerability coverage, and accelerating the discovery of high-risk flaws. In summary, the contributions are as follows:
1. We introduce AUTO-RT, a novel red-teaming framework that formulates jailbreak prompt construction as a sequential decision problem, enabling strategy-level exploration beyond static, handcrafted prompts.

2. We propose two key techniques: Dynamic Strategy Pruning and Progressive Reward Tracking, to improve both the efficiency and effectiveness of jailbreak strategy discovery under sparse reward conditions.

3. We show that strategy-level prompt exploration is essential for automated jailbreak discovery.

Beyond red-teaming, our framework offers a generalizable paradigm for prompt optimization, contributing to the development of more robust and adaptable LLMs.

## 2 Automatic Jailbreak Strategy Exploration For Red-Teaming

2.1 PRELIMINARY: AUTOMATIC RED-TEAMING AS A CONSTRAINED MARKOV DECISION
PROCESS
Automatic red-teaming formulates the discovery of safety vulnerabilities as an adversarial interaction between an attack model (AM) and a target model (TM) (Hong et al., 2024; Mehrotra et al., 2024a). The AM generates attack queries a, conditioned on a toxic behavior t ∈ T , with the goal of eliciting harmful outputs from the TM. The effectiveness of each attack is quantified by a safety evaluation function R(*a, y*) (Inan et al., 2023; Adler et al., 2024), which measures the harmfulness of the TM's response y in the context of the input a.

During the optimization of the attack model (AMθ), it is common practice to augment the objective with additional constraints (Hong et al., 2024; Achiam et al., 2017; Moskovitz et al., 2023; Dai et al., 2023), such as encouraging the generation of queries with high linguistic fluency or promoting diversity across attacks. These constraints are typically formalized as fi(*a, y, t*) ≤ ci. The constrained optimization problem for each t can thus be expressed as:

max a∼AMθ(·|t) Ey∼TM(a)[R(a, y)] , ∀t ∈ T subject to fi(a, y, t) ≤ ci, ∀i.

$$(1)$$
This represents a constrained Markov Decision Process (CMDP) (Altman, 2021), which is commonly solved via the Lagrangian method by optimizing the corresponding dual problem (Boyd & Vandenberghe, 2004; Bertsekas, 2014).

## 2.2 Strategic Red-Teaming Framework

The preceding paradigm primarily targets the discovery of high-severity safety violations, often overlooking their exploitability in realistic settings. To bridge this gap, we propose a novel strategic red-teaming framework that explicitly models and optimizes attack strategies to enhance the coverage and effectiveness of adversarial probing across diverse toxicity intents.

Specifically, we decompose the conventional attack model into two components: a strategy generation model with trainable parameters (AMgθ
), which produces high-level attack strategies, typically expressed as textual instructions; and a strategy rephrasing model (AMr), which instantiates concrete attack queries by combining a strategy with each toxic intent. This hierarchical formulation allows for greater generalization and more targeted exploration across the attack space. Accordingly, the optimization objective in Equation 1 can be reformulated as:

$$\operatorname*{max}_{s\sim\mathrm{AM}_{\theta}^{a}}\quad\mathbb{E}_{t\sim\mathcal{T}}\,\mathbb{E}_{a\sim\mathrm{AM}^{r}\,(s,t),\,y\sim\mathrm{TM}(a)}\left[R(a,y)\right]$$  subject to $f_{i}(a,y,s,t)\leq c_{i},\quad\forall i$.  
$$(2)$$

This formulation enables the learning of attack strategies with high exploitability, thereby improving both the severity and strategic coverage of automatic red-teaming. While strategic red-teaming offers a balanced approach to uncovering both severe and exploitable flaws, synthesizing effective high-level attack strategies remains considerably more challenging than directly generating attack queries. To address this, we propose AUTO-RT to enhance the overall effectiveness of strategic red-teaming.

## 2.3 Efficient Exploration With Auto-Rt 2.3.1 Challenges In Sparse-Reward Exploration

Reinforcement learning (RL) algorithms are known to struggle under sparse reward signals (Dulac- Arnold et al., 2019; Rengarajan et al., 2022). Our experiments similarly show that directly optimizing Equation 2 demands extensive exploration to yield effective attacks. As the target model's safety alignment improves, successful attack queries become harder to find (Li et al., 2024; Chao et al., 2024a). We attribute this challenge to two key factors:
i). **Overwhelming safe signals.** Improved safety alignment leads to most exploration steps receiving negligible reward, weakening optimization signals and causing the model to drift toward satisfying auxiliary constraints instead of identifying true vulnerabilities.

ii). **Heightened sparsity in strategy-level optimization.** Unlike intent-specific attacks optimized via Equation 1, strategic red-teaming requires diverse and abstract strategies, making reward signals even sparser and effective exploration more difficult.

## 2.3.2 Dynamic Strategy Pruning

As illustrated in Figure 1, to address issue i), we introduce **Dynamic Strategy Pruning**, which integrates early termination (Sun et al., 2021) into the CMDP formulation of Equation 2. This mechanism inserts intermediate check-points into the MDP to assess constraint satisfaction (e.g.,
diversity judge and *consistency judge*). If any constraint is violated, exploration is halted immediately, and a penalty is propagated to the AMgθ. Safety evaluation is performed only when all constraints are satisfied; in such cases, only the safety signal is returned, independent of constraint values. Under this modification, Equation 2 can be reformulated as:

max s∼AMgθ Et∼T Ea∼AMr(s,t), y∼TM(a) " R(a, y) · Y i 1 (fi(a, y, s, t) ≤ ci) +X i C(fi, ci) · 1 (fi(a, y, s, t) > ci)
$$(3)$$
# (3)
where C(fi, ci) denotes the penalty signal propagated when the constraint fiis violated. Theoretically, constrained MDPs can be efficiently solved through their early-terminated reformulations (Sun et al.,
2021). When the penalty C(fi, ci) is sufficiently small, which is easy to satisfy in practice, the optimal policy of the modified process is guaranteed to coincide with that of the original CMDP.

## 2.3.3 Progressive Reward Tracking

To address issue ii), we introduce **Progressive Reward Tracking** that leverages a downgrade target model for reward shaping to facilitate the exploration during red-teaming, as illustrated in Figure 2. Specifically, we reduce the safety alignment of the target model on toxic data, yielding a weaker intermediate model denoted as TM′. By incorporating safety evaluations from both the TM and TM′
into the reward signal, we alleviate feedback sparsity and better guide strategy learning. The shaped safety reward Rs is formally defined as:

$$R_{s}=R_{\mathrm{TM}^{\prime}}(a,y)+R_{\mathrm{TM}}(a,y)$$

where RTM and R′TM denote the safety evaluation results of the target and downgrade models, respectively. Specifically, RTM(*a, y*) = 1 indicates a harmful response, and 0 indicates a safe one.

Experimental results show that, most cases with R′TM(a, y) = 0 also yield RTM(*a, y*) = 0. Based on this, the shaped reward Rs is redefined as:

$$R_{s}=$$
$$\begin{cases}0,&\text{if}R_{\text{TM}^{\prime}}(a,y)=0\\ 1,&\text{if}R_{\text{TM}^{\prime}}(a,y)=1\text{and}R_{\text{TM}}(a,y)=0\\ 2,&\text{if}R_{\text{TM}^{\prime}}(a,y)=1\text{and}R_{\text{TM}}(a,y)=1\end{cases}$$
$$\quad(4)$$

With an appropriately chosen downgrade model, maximizing Rs improves exploration efficiency while preserving attack effectiveness, allowing the optimization objective becomes:

![4_image_0.png](4_image_0.png)

$$\operatorname*{max}_{s\sim\mathrm{AM}_{\boldsymbol{\theta}}^{a}}\quad\mathbb{E}_{t\sim\mathcal{T}}\,\mathbb{E}_{a\sim\mathrm{AM}^{r}\left(s,t\right),\,y\sim\mathrm{TM}\left(a\right)}\left[R_{s}\cdot\mathbf{1}\left(\forall i,f_{i}\leq c_{i}\right)+\mathbf{C}\cdot\mathbf{1}\left(\mathbf{f}>\mathbf{c}\right)\right],$$
$$\quad(5)$$

Since the proposed reward shaping does not follow the potential-based function structure (Ng et al., 1999), the selection of downgrade model is critical for identifying optimal strategies during red-teaming. A downgrade model that is either too weak or too close to the target model would produce irrelevant or uninformative feedback. In particular, overly weak models risk deviating from the safety distribution of the target model, leading to misleading reward signals. To address this, we propose a metric: **First Inverse Rate** (FIR), to guide the selection of a suitably calibrated downgrade model.

To obtain a spectrum of downgrade models with varying safety capabilities, we progressively weaken the target model with toxic data A by ether tuning or in-context learning, yielding n intermediate models TM
′
1*, . . . ,* TM
′
n. Given an attack prompt, we evaluate the response of each model and construct a binary evaluation vector:

$$\mathbf{E}=[e_{1},\ldots,e_{n}],\quad{\mathrm{~where~}}e_{i}\in\{0,1\}$$

indicates whether TMiproduces a harmful response (ei = 1) or not (ei = 0). For a given index i, we define ei as an *inverse element* if and only if ∃ ej < ei for *j > i*. The first such index is referred to as the *first inverse*, and the corresponding model TM
′
iis termed the *first inverse model* for that prompt.

By aggregating results across the toxic data A, we compute the FIR of model TM
′
kas the proportion of prompts for which it is identified as the first inverse:

$${\mathrm{identified~as~the~first~inverse:}}$$ $${\mathrm{FIR}}(k)={\frac{1}{|{\mathcal{A}}|}}\sum_{a\in{\mathcal{A}}}\mathbf{1}\left({\mathrm{first-inverse}}(a)=k\right)$$

As illustrated in Figure 4, we select the last model before a sharp increase of FIR as the downgrade model for reward shaping, ensuring a balance between alignment with the target model and informativeness of the reward signal.

$$\mathrm{RESULTS}$$

## 3 Experiments & Results 3.1 General Setup

Datasets We adopt the standard subset of Harmbench (Mazeika et al., 2024) to evaluate our method alongside other baselines. To assess the performance of the generated strategies, we partition the toxicity intents into two halves: the first half, denoted as Ttrn, is used during optimization, while the remaining half, Ttst, is used for evaluation. Additionally, we leverage a subset of AdvBench (Zou et al., 2023) to construct downgrade models. Specifically, we generate responses using the Alpaca model (Taori et al., 2023) via sampling, retain only those containing harmful content, severing as A. Models We conducted experiments on 18 LLMs from different model families, including Llama (Touvron et al., 2023), Mistral (Jiang et al., 2023), Yi (AI et al., 2024), Zephyr (Tunstall et al., 2023), Gemma (Gemma Team, 2024) and Qwen (Team, 2024a). Detail introduction about these models can be found in Appendix A. Baselines Given the limited prior research on strategic red-teaming, we conduct a comparative evaluation against a range of baseline methods, described below.

- **Few-Shot (FS)**: Sampling attack strategies using the attack model with four demonstrations to provoke harmful behaviors in the target model.

- **Imitate Learning (IL)** (Ge et al., 2023): Fine-tuning the attack model using strategies that successfully perform attacks to generate more strategies.

- **Reinforcement Learning (RL)** (Perez et al., 2022): Training with PPO (Schulman et al., 2017)
based on Equation 2.

We also directly using the toxic behaviors from HarmBench to attack these models as a reference, abbreviated as DA. For implementation details of each baseline, refer to Appendix B. Metrics In prior work (Liu et al., 2024b; Guo et al., 2021; Zhao et al., 2024), the effectiveness of attack methods is commonly measured using the *Attack Success Rate* (ASR) over a predefined set of toxic intents, defined as:

 ${\text{ASR}=\frac{1}{|\mathcal{T}_\text{tm}|}\sum_{t\in\mathcal{T}_\text{tm}}R(a,y)}$  and training probabilities also. 
In this study, we evaluate strategic red-teaming capabilities along three complementary dimensions, as detailed below.
- **Effectiveness**: Assessed using the average ASR of the top 100 strategies with the highest ASR on
Ttst, denoted as:
$$\mathrm{ASR}_{\mathrm{{lst}}}={\frac{1}{|S_{100}|*|T_{\mathrm{{lst}}}|}}\sum_{s\in S_{100}}\sum_{t\in T_{\mathrm{{lst}}}}R(a,y)$$
R(*a, y*) (6)
$$(6)$$
- **Efficiency**: Assessed via ASRtst of strategies produced at different training stages. Specifically, we partition the training process into stages, each consisting of 1,000 episodes. To capture performance dynamics, we visualize and compare the attack efficiency of different methods by violin plots.

- **Diversity**: Another key goal of strategic red-teaming is to generate a diverse set of strategies. We evaluate diversity from two perspectives: 1) *Semantic Diversity (SeD)* (Tevet & Berant, 2020),
measured by computing the pairwise semantic similarity among all generated strategies; 2) Defense Generalization Diversity (DeD), assessed by first attacking the target model, then constructing defenses based on the successful attacks, and evaluating the ASRtst of second-round attacks on the defended model.

Implement Details We employ Llama-Guard2-8B (Meta, 2024) to assess the safety of model responses. We incorporate two additional constraints: 1) a **diversity constraint**, where a CRT-style mechanism (Hong et al., 2024) is used to penalize repetitive strategies; 2) a **consistency constraint**, which leverages an LLM to verify whether rephrased attack queries remain semantically aligned with the original behaviors. Both AMgand AMrare implemented using Vicuna-7B, with a maximum sampling budget of 9,000 episodes. To ensure computational stability, only AMgis optimized using PPO (Schulman et al., 2017) with 8×A100 clusters. Further details on implementation and evaluation are provided in Appendix B,D.

## 3.2 Main Results

Comparative Analysis of Attack Effectiveness and Diversity. Table 1 presents the white-box evaluation results of AUTO-RT, where the downgrade model is constructed via toxic fine-tuning on the target model. AUTO-RT consistently achieves the highest ASRtst across a wide range of models, demonstrating its effectiveness in generating successful strategies. Notably, for the Llama 2 family, which is known for its strong safety alignment, AUTO-RT is still able to perform effective strategic attacks. Interestingly, for the R2D2 (Mazeika et al., 2024), which incorporates targeted defenses,

| Effectiveness       | Diversity   |                   |       |         |       |      |      |            |       |       |       |         |       |
|---------------------|-------------|-------------------|-------|---------|-------|------|------|------------|-------|-------|-------|---------|-------|
| Target Model        | ASRtst↑     | SeD↓              | DeD↑  |         |       |      |      |            |       |       |       |         |       |
| DA                  | FS          | IL                | RL    | AUTO-RT | FS    | IL   | RL   | AUTO-RT    | FS    | IL    | RL    | AUTO-RT |       |
| Vicuna 7B           | 24.80       | 29.58             | 36.90 | 31.95   | 56.40 | 0.70 | 0.86 | 0.64       | 0.57  | 6.30  | 5.24  | 20.10   | 46.80 |
| Vicuna 13B          | 16.60       | 20.80             | 36.08 | 17.80   | 55.35 | 0.77 | 0.93 | 0.51       | 0.50  | 8.15  | 4.55  | 21.03   | 56.33 |
| Llama 2 7B Chat     | 0.45        | 6.84              | 6.67  | 0.50    | 13.50 | 0.74 | 0.90 | 0.54       | 0.46  | 3.55  | 2.70  | 0.88    | 12.98 |
| Llama 2 13B Chat    | 1.30        | 5.88              | 6.80  | 2.05    | 11.00 | 0.65 | 0.85 | 0.54       | 0.56  | 4.20  | 3.03  | 1.15    | 10.85 |
| Llama 3 8B Instruct | 3.20        | 9.42              | 7.18  | 14.55   | 15.00 | 0.67 | 0.94 | 0.64       | 0.45  | 7.00  | 6.40  | 7.50    | 15.00 |
| Mistral 7B Instruct | 48.50       | 51.54 54.88 44.20 | 52.65 | 0.76    | 0.88  | 0.51 | 0.50 | 12.35 9.80 | 28.48 | 48.68 |       |         |       |
| Yi 6B Chat          | 13.45       | 36.00             | 42.29 | 33.80   | 52.50 | 0.80 | 0.90 | 0.50       | 0.48  | 14.60 | 12.18 | 31.45   | 47.25 |
| Yi 9B Chat          | 16.75       | 28.06             | 34.23 | 39.75   | 49.20 | 0.80 | 0.91 | 0.57       | 0.59  | 15.00 | 13.05 | 22.60   | 48.90 |
| Gemma 2 2b Instruct | 2.05        | 5.64              | 7.49  | 6.15    | 48.15 | 0.81 | 0.85 | 0.52       | 0.46  | 5.15  | 3.53  | 3.43    | 47.93 |
| Gemma 2 9b Instruct | 1.55        | 3.74              | 6.63  | 44.85   | 44.80 | 0.71 | 0.82 | 0.62       | 0.53  | 3.80  | 2.28  | 30.20   | 48.10 |
| R2D2                | 1.70        | 27.18 24.24 8.60  | 12.45 | 0.71    | 0.82  | 0.59 | 0.50 | 10.45 8.95 | 4.33  | 41.78 |       |         |       |
| Qwen 1.5 4B Chat    | 12.50       | 27.24             | 18.52 | 17.45   | 51.30 | 0.65 | 0.87 | 0.59       | 0.58  | 5.50  | 4.20  | 12.88   | 45.58 |
| Qwen 1.5 7B Chat    | 21.70       | 23.80             | 18.82 | 32.60   | 49.85 | 0.72 | 0.89 | 0.57       | 0.52  | 8.00  | 6.80  | 25.95   | 34.25 |
| Qwen 1.5 14B Chat   | 17.20       | 18.78             | 23.82 | 17.75   | 42.50 | 0.72 | 0.88 | 0.57       | 0.53  | 6.95  | 5.05  | 16.40   | 43.40 |
| Qwen 2.5 3B Chat    | 16.30       | 30.94             | 38.30 | 20.35   | 42.20 | 0.71 | 0.83 | 0.58       | 0.58  | 5.20  | 3.80  | 17.25   | 47.85 |
| Qwen 2.5 14B Chat   | 3.80        | 15.42 9.38        | 15.65 | 17.15   | 0.74  | 0.84 | 0.64 | 0.46       | 9.10  | 7.50  | 12.38 | 15.43   |       |

Table 1: **Left**: Attack success rate of various methods, expressed as a percentage (%), where higher values indicate greater attack effectiveness. **Middle**: Semantic diversity among attack strategies generated by different methods, measured in similarity score, with lower values indicating higher diversity. **Right**: Comparison of defense generalization diversity, expressed as a percentage (%), with higher values suggesting a greater ability to discover diverse strategies continuously. a sampling-based method outperforms others. This highlights the robustness of R2D2's defense mechanism. Nevertheless, AUTO-RT outperforms RL-based methods consistently, validating its strength in efficient attack strategic generation.

In terms of SeD, AUTO-RT also surpasses baselines in producing semantically diverse strategies. When evaluating DeD, which measures robustness to second-round attack, AUTO-RT maintains stable attack performance. The relative change in ASRtst after defense application is notably smaller for AUTO-RT, indicating stronger ability to continuously discover effective strategies. Particularly on R2D2, AUTO-RT exhibits a significant increase in DeD after the second round of attacks, suggesting potential blind spots in the defense mechanism and further validating the effectiveness of our approach. Comparative Analysis of Attack Efficiency. Figure 3 compares the attack efficiency of AUTO-RT and RL. For every 1,000 sampled episodes, we analyze the resulting ASRtst, capturing the dynamics across 9 training stages. As shown, AUTO-RT consistently discovers more effective attack strategies than RL at each stage and achieves better overall performance. Moreover, the variance of ASRtst within each stage is larger for AUTO-RT, suggesting a stronger capacity for broad and sustained exploration. Complete experimental results are provided in Appendix F.

## 3.3 Further Discussions 3.3.1 Ablation Study Of Auto-Rt

To further analyze the contributions of Dynamic Strategy Pruning (DSP) and Progressive Reward Tracking (PRT), we evaluate AUTO-RT under ablated settings where each component is applied individually. The results are summarized in Table 2, with complete results provided in Appendix F.

For both ASRtst and SeD, DSP and PRT independently improve performance, and their combination leads to further enhancement. In terms of DeD, PRT exhibits a more substantial impact, indicating that the proposed reward shaping mechanism is critical for maintaining attack effectiveness after defenses are applied. These results highlight the complementary roles of DSP and PRT in improving both the robustness and adaptability of strategic red-teaming.

7

![7_image_0.png](7_image_0.png)

| Attack Effective (ASRtst)↑ Semantic Diversity (SeD) ↓   |
|---------------------------------------------------------|
| Defense Generalization Diversity (DeD) ↑                |

V-7B V-13B L2-13B L3-8B Y-6B G-2B R2D2 Q1.5-7B Q1.5-14B Q2.5-14B

RL 31.95 17.80 2.05 14.55 33.80 6.15 8.60 32.60 17.75 15.65

+DSP 36.54 22.92 2.46 **15.00** 35.98 7.38 9.07 41.01 19.58 **17.15** +PRT 40.50 35.20 6.80 14.60 42.30 25.30 9.80 40.20 28.30 16.50

AUTO-RT **56.40 55.35 11.00 15.00 52.50 48.15 12.45 49.85 42.50 17.15**

Semantic Diversity (SeD) ↓

RL 0.64 0.51 **0.54** 0.64 0.50 0.52 0.59 0.57 0.57 0.64

+DSP **0.57 0.50** 0.55 0.51 0.53 0.50 0.57 0.53 **0.53 0.44** +PRT 0.66 0.58 0.65 0.59 0.61 0.54 0.63 0.57 0.64 0.57

AUTO-RT **0.57 0.50** 0.56 **0.45 0.48 0.46 0.50 0.52 0.53** 0.46

Defense Generalization Diversity (DeD) ↑

RL 20.10 21.03 1.15 7.50 31.45 3.43 4.33 25.95 16.40 12.38

+DSP 43.02 54.45 12.51 14.35 47.19 47.51 41.09 **42.37** 42.15 14.49 +PRT **47.02** 56.18 **13.93** 14.84 **50.94** 43.55 39.11 32.56 42.05 **16.23**

AUTO-RT 46.80 **56.33** 10.85 **15.00** 47.25 **47.93 41.78** 34.25 **43.40** 15.43

## 3.3.2 Effectiveness Of First Inverse Rate (Fir)

To evaluate the impact of downgrade model selection, we test a series of downgraded models (M1 to M6) with progressively weakened safety capabilities across six target models. Figure 4 reports three key metrics: the safety level of each downgrade model measured by *Weaken (ASR)*, the attack success rate under AUTO-RT denoted as *Attack (ASR)*, and the FIR of each downgrade model (*Weaken (FIR)*).

We observe that selecting the last model *before* the sharp rise in FIR, as indicated by the dark-colored bars in Figure 4, consistently yields the best attack performance. This demonstrates the utility of FIR as an indicator to substantially disrupt the model's generative alignment, leading to instability in the model's internal safety boundaries and increased inconsistency in outputs. Moreover, when using downgrade models weaker than the FIR-indicated threshold, further increases in *Weaken (ASR)* no longer translate to improved attack performance. This suggests that over-weaken may lead to diminished guidance quality and thus hinder the effectiveness of strategic red-teaming.

## 3.3.3 Comparison With Human-Based Approach

Several baselines based on human-crafted templates have demonstrated strong performance. Including AutoDAN (Liu et al., 2024b), which evolves handcrafted jailbreak prompts with a genetic algorithm, abbreviated as AD; Human Template (Shen et al., 2024), using a fixed set of in-the-wild human jailbreak templates, abbreviated as HT; and Past-Tense (Andriushchenko & Flammarion, 2024), modifying the attack prompt to reflect that it occurred in the past, abbreviated as PT. We compared AUTO-RT with these methods across 16 models, as shown in Table 3. The results demonstrate that AUTO-RT not only achieves a high success rate in the first round of attacks (ASRtst) but also

![8_image_0.png](8_image_0.png)

| AD       | HT    | PT    | AUTO-RT   |       |       |
|----------|-------|-------|-----------|-------|-------|
| ASRtst ↑ | 55.23 | 37.35 | 11.19     | 38.38 |       |
| SeD      | ↓     | 0.86  | 0.36      | -     | 0.52  |
| DeD      | ↑     | 17.88 | 13.15     | 7.27  | 38.19 |

maintains the highest success rate in the second round of attacks (DeD), indicating that our approach can achieve near-human-level sustained attack capabilities.

## 3.3.4 Black-Box Setting Attack Results

We evaluated the performance of AUTO-RT using in-context learning (ICL) approach to obtain downgrade model in scenarios where direct toxic fine-tuning the target model is not feasible. We utilized Llama3-70B-Instruct and Qwen2.5-72B-Instruct to simulate such black-box settings. The experimental results, shown in Table 4, indicate that AUTO-RT, even with the ICL
approach, can improve exploration effectiveness and generates diverse attack strategies.

## 4 Related Works

Automatic red-teaming methods can be categorized into two approaches depending on the type of feedback signal. The first use textual feedback for optimization, where the model's parameters are implicitly modified by incorporating feedback into the conversation. This approach benefits from the rich information contained in textual feedback, allowing potentially solutions to be identified with fewer interactions. However, to obtain effective feedback signals, it's necessary to jailbreak the attacker first to prevent it from refusing interactions with toxic behaviors. For example, PAIR (Chao et al., 2024b) specifies two persuasion techniques to gradually coax the target model, while ICA (Wei et al., 2024) employs harmful demonstrations to subvert LLMs. TAP (Mehrotra et al., 2024b) iteratively refines attack prompts using tree-of-thought reasoning until a generated prompt jailbreaks the target. Additionally, methods like PAP (Zeng et al., 2024), Rainbow Teaming (Samvelyan et al., 2024), GPTFuzzer (Yu et al., 2024), and Purple Teaming (Zhou et al., 2024a) explore the target model's flaws by predefining a series of attack strategies. AutoDAN-turbo (Liu et al., 2024a) explores attack strategies guided by textual feedback before executing them against the target model; however, this comes at the cost of requiring thousands of hours of searching time.

The second approach utilizes numerical feedback signals to guide the optimization. Methods like GCG (Zou et al., 2023), GDBA (Guo et al., 2021), and AutoPrompt (Shin et al., 2020) use logits from target model as optimization signals. MART (Ge et al., 2023) employ a dangerous content classifier to screen numerous sampled results, using imitation learning to produce attack prompts. Cold-Attack (Guo et al., 2024) scores attack based on a rule-based model from multiple perspectives, framing red teaming as energy-based constrained decoding. CRT (Hong et al., 2024) and Diver-CT (Zhao et al., 2024) model this process as reinforcement learning, providing score feedback

| LLaMA 3 70B   | Qwen 2.5 72B   |       |            |       |       |            |
|---------------|----------------|-------|------------|-------|-------|------------|
| ASRtst ↑      | SeD ↓          | DeD ↑ | ASRtst ↑   | SeD ↓ | DeD ↑ |            |
| FS            | 5.49           | 0.87  | 1.17-4.32  | 3.53  | 0.82  | 3.05-0.48  |
| IL            | 6.80           | 0.64  | 0.92-5.88  | 6.22  | 0.73  | 1.20-5.02  |
| RL            | 4.99           | 0.53  | 4.15-0.84  | 4.53  | 0.52  | 4.33-0.2   |
| Auto-RT       | 14.88          | 0.52  | 15.00+0.12 | 14.47 | 0.61  | 14.15-0.32 |

Table 4: Attack performance when using In-Context Learning approach to construct downgrade model in black-box setting for simulating models with inaccessible trainable weights. to optimize attack strategies based on attack diversity and the severity of the output's dangerousness. However, as numerical feedback contains less information than textual feedback, achieving comparable attack often requires more exploration.

## 5 Conclusions

In this paper, we introduce AUTO-RT, a framework that employs dynamic strategy pruning and progressive reward tracking to automatically discover strategic attacks. Experimental results show that our approach significantly improves the efficiency and effectiveness of continuous, diverse strategy exploration across a wide range of models in both white-box and black-box settings.

## 6 Acknowledgments

We sincerely thank the reviewers for their insightful comments and valuable suggestions. This work was supported by the Natural Science Foundation of China (No. 62536008, 62306303, 62572456, 62476265), and Ant Group Research Fund.

## 7 Ethics Statement

We propose a technique named AUTO-RT for generating jailbreak attacks on Large Language Models (LLMs), aiming to support the development of more robust and trustworthy LLMs. Although the current study demonstrates its effectiveness on public models, AUTO-RT is also applicable to custom LLMs in domain-specific settings or broader alignment scenario.

## 8 Reproducibility Statement

We have clarified our experiment setting in Section 3 and Appendix A,B,D,E.

## References

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International conference on machine learning, pp. 22–31. PMLR, 2017.

Bo Adler, Niket Agarwal, Ashwath Aithal, Dong H Anh, Pallab Bhattacharya, Annika Brundyn, Jared Casper, Bryan Catanzaro, Sharon Clay, Jonathan Cohen, et al. Nemotron-4 340b technical report. *arXiv preprint arXiv:2406.11704*, 2024.

01. AI, :, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. Yi: Open foundation models by 01.ai, 2024. URL https://arxiv.org/ abs/2403.04652.

John Allspaw and Richard I. Cook. How complex systems fail. In *Web Operations*, 2010. URL
https://api.semanticscholar.org/CorpusID:18051593.

Eitan Altman. *Constrained Markov decision processes*. Routledge, 2021. Markus Anderljung, Joslyn Barnhart, Anton Korinek, Jade Leung, Cullen O'Keefe, Jess Whittlestone, Shahar Avin, Miles Brundage, Justin Bullock, Duncan Cass-Beggs, et al. Frontier ai regulation: Managing emerging risks to public safety. *arXiv preprint arXiv:2307.03718*, 2023.

Maksym Andriushchenko and Nicolas Flammarion. Does refusal training in llms generalize to the past tense?, 2024. URL https://arxiv.org/abs/2407.11969.

Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion. Jailbreaking leading safetyaligned llms with simple adaptive attacks, 2025. URL https://arxiv.org/abs/2404. 02151.

Anthropic. Claude sonnet 4. https://www.anthropic.com/news/claude-4, 2025. Accessed: 2025-11-24.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. *arXiv preprint arXiv:2212.08073*, 2022.

Dimitri P Bertsekas. *Constrained optimization and Lagrange multiplier methods*. Academic press, 2014.

Navneet Bhatt, Adarsh Anand, and Venkata SS Yadavalli. Exploitability prediction of software vulnerabilities. *Quality and Reliability Engineering International*, 37(2):648–663, 2021.

Matt Bishop and Dave Bailey. A critical analysis of vulnerability taxonomies. Technical report, Citeseer, 1996.

Stephen P. Boyd and Lieven Vandenberghe. Convex optimization. IEEE Transactions on Automatic Control, 51:1859–1859, 2004. URL https://api.semanticscholar.org/CorpusID: 37925315.

Mehran Bozorgi, Lawrence K Saul, Stefan Savage, and Geoffrey M Voelker. Beyond heuristics:
learning to classify vulnerabilities and predict exploits. In *Proceedings of the 16th ACM SIGKDD* international conference on Knowledge discovery and data mining, pp. 105–114, 2010.

Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J Pappas, Florian Tramer, et al.

Jailbreakbench: An open robustness benchmark for jailbreaking large language models. arXiv preprint arXiv:2404.01318, 2024a.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries, 2024b. URL https:
//arxiv.org/abs/2310.08419.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https:
//lmsys.org/blog/2023-03-30-vicuna/.

Brian Christian. The alignment problem: Machine learning and human values. Perspectives on Science and Christian Faith, 73:245–247, 12 2021. doi: 10.56315/PSCF12-21Christian.

Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe rlhf: Safe reinforcement learning from human feedback. *arXiv preprint* arXiv:2310.12773, 2023.

Boyi Deng, Wenjie Wang, Fuli Feng, Yang Deng, Qifan Wang, and Xiangnan He. Attack prompt generation for red teaming and defending large language models. *arXiv preprint arXiv:2310.12505*,
2023.

Gabriel Dulac-Arnold, Daniel Mankowitz, and Todd Hester. Challenges of real-world reinforcement learning, 2019. URL https://arxiv.org/abs/1904.12901.

Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, Andy Jones, Sam Bowman, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Nelson Elhage, Sheer El-Showk, Stanislav Fort, Zac Hatfield-Dodds, Tom Henighan, Danny Hernandez, Tristan Hume, Josh Jacobson, Scott Johnston, Shauna Kravec, Catherine Olsson, Sam Ringer, Eli Tran-Johnson, Dario Amodei, Tom Brown, Nicholas Joseph, Sam McCandlish, Chris Olah, Jared Kaplan, and Jack Clark. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned, 2022. URL
https://arxiv.org/abs/2209.07858.

Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, and Yuning Mao. Mart: Improving llm safety with multi-round automatic red-teaming, 2023. URL
https://arxiv.org/abs/2311.07689.

Gemma Team. Gemma 2: Improving open language models at a practical size, 2024. URL
https://arxiv.org/abs/2408.00118.

Google DeepMind. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025.

Chuan Guo, Alexandre Sablayrolles, Hervé Jégou, and Douwe Kiela. Gradient-based adversarial attacks against text transformers, 2021. URL https://arxiv.org/abs/2104.13733.

Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, and Bin Hu. Cold-attack: Jailbreaking llms with stealthiness and controllability. *arXiv preprint arXiv:2402.08679*, 2024.

Zhang-Wei Hong, Idan Shenfeld, Tsun-Hsuan Wang, Yung-Sung Chuang, Aldo Pareja, James Glass, Akash Srivastava, and Pulkit Agrawal. Curiosity-driven red-teaming for large language models. In *The Twelfth International Conference on Learning Representations*, 2024. URL
https://openreview.net/forum?id=4KqkizXgXU.

Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai Li, and Danqi Chen. Catastrophic jailbreak of open-source llms via exploiting generation, 2023. URL https://arxiv.org/abs/2310. 06987.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. Llama guard: Llm-based input-output safeguard for human-ai conversations, 2023. URL https://arxiv.org/abs/ 2312.06674.

Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. Baseline defenses for adversarial attacks against aligned language models. *arXiv preprint arXiv:2309.00614*, 2023.

Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan, Zhonghao He, Lukas Vierling, Donghai Hong, Jiayi Zhou, Zhaowei Zhang, Fanzhi Zeng, Juntao Dai, Xuehai Pan, Kwan Yee Ng, Aidan O'Gara, Hua Xu, Brian Tse, Jie Fu, Stephen McAleer, Yaodong Yang, Yizhou Wang, Song-Chun Zhu, Yike Guo, and Wen Gao. Ai alignment: A
comprehensive survey, 2025. URL https://arxiv.org/abs/2310.19852.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. URL https://arxiv.

org/abs/2310.06825.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. Rlaif vs. rlhf:
Scaling reinforcement learning from human feedback with ai feedback, 2024. URL https:
//arxiv.org/abs/2309.00267.

Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, and Jing Shao. Salad-bench: A hierarchical and comprehensive safety benchmark for large language models. arXiv preprint arXiv:2402.05044, 2024.