# Toward Safer Diffusion Language Models: Discovery And Mitigation Of Priming Vulner- Ability

Shojiro Yamabe Institute of Science Tokyo yamabe.s.2fb0@m.isct.ac.jp Jun Sakuma Institute of Science Tokyo, RIKEN AIP

## Abstract

Diffusion language models (DLMs) generate tokens in parallel through iterative denoising, which can reduce latency and enable bidirectional conditioning. However, the safety risks posed by jailbreak attacks that exploit this inference mechanism are not well understood. In this paper, we reveal that DLMs have a critical vulnerability stemming from their iterative denoising process and propose a countermeasure. Specifically, our investigation shows that if an affirmative token for a harmful query appears at an intermediate step, subsequent denoising can be steered toward a harmful response even in aligned models. As a result, simply injecting such affirmative tokens can readily bypass the safety guardrails. Furthermore, we demonstrate that the vulnerability allows existing optimization-based jailbreak attacks to succeed on DLMs. Building on this analysis, we propose a novel safety alignment method tailored to DLMs that trains models to generate safe responses from contaminated intermediate states that contain affirmative tokens. Our experiments indicate that the proposed method significantly mitigates the vulnerability with minimal impact on task performance. Furthermore, our method improves robustness against conventional jailbreak attacks. Our work underscores the need for DLM-specific safety research. Our code is available at https://github.com/mdl-lab/dlm-priming-vulnerability.

## 1 Introduction

Diffusion Language Models (DLMs) (DeepMind, 2024; Labs et al., 2025) generate tokens in parallel through an iterative denoising (reverse) process and are emerging as an alternative to Autoregressive Models (ARMs) (Touvron et al., 2023; Achiam et al., 2023). In particular, there has been growing interest in a practical subclass of DLMs, Masked Diffusion Language Models (MDLMs) (Nie et al., 2025; Gong et al., 2025; You et al., 2025; Zhu et al., 2025), which define the diffusion process over the discrete token vocabulary. As shown in Figure 1(a), the denoising process begins with a fully masked token sequence. At each step, the model updates all masked tokens with predicted tokens in parallel and then re-masks a subset of them. Repeating this procedure gradually reduces the masking ratio until a complete sequence emerges. These properties are attractive for both lower inference latency and the bidirectional context (Li et al., 2022; Patel et al., 2023; Li et al., 2025). However, the vulnerabilities of MDLMs to jailbreak attacks remain largely unexplored. Because their non-causal, parallel denoising process fundamentally differs from the causal, sequential generation of ARMs, it is unclear whether safety insights established for ARMs transfer to MDLMs. These differences motivate MDLM-specific safety research tailored to their inference mechanism. In this work, we identify a critical vulnerability and propose a countermeasure. Our investigation reveals that even in safety-aligned models, if an affirmative token in response to a harmful query appears at an intermediate step of the denoising process, subsequent generation can be steered toward a harmful response (Figure 1(b)). We refer to this as the *priming vulnerability*. This stands in contrast to the vulnerability exploited by prefilling attacks on ARMs (Wei et al., 2023a). In ARMs, left-toright sequential prediction allows the very first few affirmative tokens in the response to suppress later refusals. In MDLMs, the iterative and parallel inference mechanism causes affirmative tokens 1

![1_image_0.png](1_image_0.png)

that arise early in the denoising process to have a similar suppressive effect. While the vulnerability of ARMs has become a major focus of prior works (Sahoo et al., 2024; Qi et al., 2025; Zhao et al., 2025), an analysis of the priming vulnerability remains limited. To address this gap, we systematically analyze this vulnerability by designing attacks that target the denoising process of MDLMs under two threat models. In the first threat model, we assume a hypothetical attacker who can intervene in the denoising process for comprehensive evaluation. We introduce a simple attack that injects tokens specified by the attacker at an intermediate step and show that the attack success rate increases from 2% to 21% even with an intervention only at the first step. In the second threat model, we assume a more realistic attacker who does not intervene in the denoising process and instead conducts an optimization-based jailbreak attack such as Greedy Coordinate Gradient (GCG) (Zou et al., 2023). While these attacks optimize the query to maximize the likelihood of generating a harmful response, the gradient of the objective is intractable because the denoising process typically includes iterative stochastic re-masking. To address this, we derive a theoretical lower bound on the attack objective that exploits the priming vulnerability and demonstrate that it works as an effective surrogate. These results underscore the severity of the issue.

This vulnerability stems from the initialization choice commonly used in MDLM training, which initializes the denoising process from a fully masked sequence and trains the model to generate safe responses only under that starting condition. However, the generation trajectory does not include cases where affirmative tokens for a harmful query appear at the intermediate steps of the denoising process. As a result, the model does not learn how to recover from such partially contaminated states, and its refusal mechanism tends to fail once those tokens appear. To address this issue, we propose a new safety alignment method for MDLMs, Recovery Alignment (RA) (Figure 1(c)). In our training process, we intentionally construct harmful intermediate states and condition the model to generate from them. In this way, we teach the model a recovery trajectory from contamination back to safety. Importantly, by explicitly modeling such contaminated intermediate states, our approach not only mitigates the priming vulnerability but also often leads to stronger robustness against general jailbreak attacks. In our experiments, RA achieves state-ofthe-art robustness against priming vulnerability without clear degradation in general capability on eleven benchmarks. Moreover, it enhances robustness against conventional jailbreak attacks. These findings highlight that our approach effectively addresses the vulnerability. Our contributions can be summarized as follows:
1. We focus on and quantify the *priming vulnerability* in MDLMs, where affirmative tokens at an intermediate step of the denoising process can steer the subsequent process toward producing a harmful response.

2. We introduce *Recovery Alignment* (RA), an MDLM-specific safety alignment that trains the model to recover from adversarially contaminated intermediate states back to safe responses.

3. We validate our approach on three MDLMs across two datasets. RA mitigates the vulnerability and improves robustness against standard jailbreaks while preserving utility.

## 2 Related Work

In this section, we specifically focus on the safety of MDLMs. For a detailed and comprehensive review of related work, including literature on ARMs, please see Appendix B.

## 2.1 Diffusion Language Models

DLMs are a framework that leverages the generative mechanisms of diffusion models for text generation. Two main approaches are distinguished by the domain in which the diffusion process is defined: continuous DLMs (Li et al., 2022; Gong et al., 2023; Dieleman et al., 2022; Lin et al., 2023; Mahabadi et al., 2024) and discrete DLMs (Austin et al., 2021a; He et al., 2023; Ye et al., 2023; Sahoo et al., 2024; Gong et al., 2025). Within the discrete family, MDLMs have emerged as an effective method. Recent works (Nie et al., 2025; Zhu et al., 2025; Ye et al., 2025) indicate that MDLMs trained from scratch can match the performance of similarly sized ARMs (Dubey et al., 2024). Extensions to multimodal inputs and joint text–image generation have also been explored (Yang et al., 2025; You et al., 2025).

## 2.2 Jailbreak Attacks

A substantial literature examines jailbreak attacks for ARMs (Wei et al., 2023a; Zou et al., 2023; Chao et al., 2025; Mehrotra et al., 2024; Anil et al., 2024a; Liu et al., 2024a; Andriushchenko et al., 2025). Some works investigate input-level priming vulnerabilities (Huang et al., 2025; Miao et al., 2025) and show that harmful responses can be triggered when malicious context is embedded in the input prompt or dialogue history. In contrast, we focus on MDLM-specific priming vulnerabilities in the denoising process, analyzing how each token in a single output sequence influences subsequent denoising steps. For MDLMs, several concurrent works propose attacks that explicitly intervene in the denoising process (Zhang et al., 2025; Wen et al., 2025). While these attacks implicitly exploit the priming vulnerability, they cannot provide a comprehensive quantitative evaluation because their attacks depend heavily on heuristic choices of tokens and intervention locations. Moreover, they do not discuss how a more realistic attacker, who cannot intervene in the denoising process, could exploit this vulnerability. In this work, we study both settings: with and without intervention in the denoising process. In the intervention setting, we design a controlled attack to quantitatively evaluate the priming vulnerability. In the non-intervention setting, we show that an attacker can still exploit this vulnerability by optimizing only the query. Recent work has also investigated evaluation metrics for jailbreak attacks (Chu et al., 2024; Mou et al., 2024; Ran et al., 2024; Souly et al., 2024; Beyer et al., 2025; Chao et al., 2024). Existing studies typically formalize attack success in two main ways: (i) rule-based approaches (Zou et al., 2023; Wei et al., 2023b), such as keyword matching, and (ii) model-based approaches (Inan et al., 2023; Li et al., 2024b) that rely on LLM-as-a-judge protocols or dedicated safety classifiers. In this work, we assess the success of jailbreak attacks using multiple automatic metrics, including keyword matching, a guardrail model (Inan et al., 2023), and GPT-4o (Achiam et al., 2023) as a safety judge.

## 2.3 Safety Alignments

A large body of work proposes methods for safety alignment, focusing on ARMs (Rafailov et al.,
2023; Ouyang et al., 2022; Bai et al., 2022; Ethayarajh et al., 2024). For MDLMs, Xie et al. (2025)
point out that middle tokens in a response critically affect safety and propose a safety alignment method, MOSA. This method aims to align the middle tokens with a safe refusal template. However, as our experiments show, it cannot address the priming vulnerability because it trains models to generate safe responses from a fully masked sequence. In contrast, we mitigate this vulnerability by training the model to recover from intentionally contaminated intermediate states to safe responses.

## 3 Preliminaries

Notation. Let V be the vocabulary. We denote the query as q ∈ V|q|, and the response as r ∈ VL,
where L denotes the generation length. The denoising process consists of T steps, and we denote the partially masked response at timestep t ∈ {0, 1, . . . , T} as rt ∈ VL. Since our work focuses on the denoising step for inference only, we index it with an increasing step counter: r0 is the sequence where all tokens are masked, and rT is the fully restored response. MDLMs are composed of two core components: *mask predictor* πθ : V
|q| × VL → P(V
L) and *masking strategy* mt : V
L →
P(V
L).

Denoising process. The denoising process starts with the fully masked response r0 and iteratively refines it to produce rT . At step t, the mask predictor πθ generates the fully unmasked response r˜t from the query q and the partially masked response rt−1, and then the masking strategy mt generates a re-masked response rt from an unmasked response r˜t. The generation probability of the final response is expressed as follows:

$$p_{\pi,m_{t}}(\mathbf{r}_{T}\mid\mathbf{q},\mathbf{r}_{0})=\int\cdots\int\prod_{t=1}^{T}\left[\int m_{t}(\mathbf{r}_{t}\mid\bar{\mathbf{r}}_{t})\pi_{\theta}(\bar{\mathbf{r}}_{t}\mid\mathbf{q},\mathbf{r}_{t-1})d\bar{\mathbf{r}}_{t}\right]d\mathbf{r}_{1}\cdots d\mathbf{r}_{T-1}.\tag{1}$$

In typical implementations (Nie et al., 2025; Zhu et al., 2025; Yang et al., 2025), a simple randommasking schedule is used as a basis. At step t, the masking strategy re-masks only the tokens that are masked in rt−1 with probability T −t T. Unmasked tokens in rt are unchanged and never re-masked in subsequent steps. Unless otherwise specified, we set L = 128 and T = 128.

## 4 Priming Vulnerability

We define the priming vulnerability as the case that if affirmative tokens, which endorse or advance a harmful intent, appear at an intermediate step of the denoising process, subsequent generation tends to be steered toward a harmful response. This vulnerability does not surface simply by inputting a harmful query because safety-aligned models often produce only refusal tokens. In this section, we present two case studies to expose and measure it. In Section 4.1, we assume a hypothetical attacker who can intervene in the denoising process and reveal the characteristics of the vulnerability. In Section 4.2, we demonstrate that a more realistic attacker, who cannot intervene in the denoising process, can still exploit this vulnerability, emphasizing that it is an important issue that must be addressed.

## 4.1 Characteristics Of The Priming Vulnerability

Assuming a hypothetical attacker who can directly intervene in the denoising process, we design the anchoring attack, a straightforward attack for vulnerability evaluation (Figure 1(b)). Let (q, r) be a harmful query and response, respectively. In this attack, at the intervention step tinter (e.g., tinter = 1), the attacker replaces the predicted response r˜tinter with the harmful response r. Then, the model continues the denoising process from the intermediate re-masked sequence rtinter ∼ mtinter(· | r). The tokens contained in rtinter act as anchors, biasing subsequent denoising toward harmful trajectories. Finally, the model generates the response rT ∼ pπ,mt(· | q, rtinter).

![3_image_0.png](3_image_0.png)

Evaluating the priming vulnerability. We quantify this vulnerability using the anchoring attack on JBB-Behaviors dataset (Chao et al., 2024), which contains 100 carefully crafted behaviors. Following prior works (Qi et al., 2025; Sahoo et al., 2024), we used GPT-4o as an automatic judge to decide whether a model's response is harmful. We reported the Attack Success Rate (ASR) as the fraction of outputs judged harmful. Harmful responses are generated by a non-safety-aligned model (please see Appendix D for details). Figure 2 shows clear evidence of this vulnerability. We make two key observations: (i) The later the intervention step, the higher the ASR. With an intervention at step 16, ASR exceeds 80% across all models. The later intervention embeds more tokens in the intermediate state, making it increasingly difficult to generate a safe response from the state.

(ii) Intervening even in the first step significantly increases ASR. At tinter = 1, the attack inserts only a single token, as we set L = T = 128. Despite this minimal change, it bypasses the safety guardrails. For example, ASR increases from 2% to 21% with LLaDA Instruct. The result highlights the significant impact of this vulnerability. Additional analyses are provided in Appendix C.1.

## 4.2 Leveraging The Priming Vulnerability Without Intervention

To further analyze the vulnerability, we assume a more realistic adversary who cannot intervene in the denoising process but can modify the prompt and examine how such an attacker can still exploit the vulnerability. In this section, we focus on GCG as a concrete instantiation. We first define the objective of GCG. Given a harmful query q and harmful target response r, the attacker optimizes a suffix s to maximize the likelihood of generating the response r:

$$\operatorname*{max}_{\mathbf{r}}{\mathcal{L}}_{\mathrm{GCG}}(\mathbf{s})\triangleq\log p_{\pi,m_{t}}(\mathbf{r}_{T}=\mathbf{r}\mid\mathbf{q}\oplus\mathbf{s},\mathbf{r}_{0}),$$
$$(2)$$
s
(rT = r | q ⊕ s, r0), (2)
where ⊕ denotes the concatenation of token sequences.

For MDLMs, iterative stochastic remasking in the denoising process makes the gradient of the objective intractable because the generation probability contains exponentially many stochastic denoising paths. A straightforward way to address this problem is to maximize a tractable lower bound estimated by Monte Carlo (MC) sampling (Nie et al., 2025; Zhu et al., 2025). However, MC estimates have high variance and incur substantial overhead due to repeated sampling, which leads to lower attack performance and higher computational costs, as demonstrated in our experiments. By leveraging the priming vulnerability, we can obtain a lower bound that does not rely on MC, which demonstrates better attack performance empirically. Specifically, we show that the loglikelihood of the mask predictor in the first step is a lower bound on the log-likelihood over the entire denoising process: Theorem 4.1. Let q and r be the query and the response, respectively, and let rt *be the intermediate* state at step t*. Assume the monotonicity* log πθ(r˜t+1 = r | q, rt) ≥ log πθ(r˜1 = r | q, r0) *for all* t = 1, . . . , T − 1*. Then, the following inequality holds:*

$$\log p_{\pi,m_{t}}(\mathbf{r}_{T}=\mathbf{r}\mid\mathbf{q},\mathbf{r}_{0})\ \geq\ {\frac{1}{T}}\log\pi_{\theta}({\tilde{\mathbf{r}}}_{1}=\mathbf{r}\mid\mathbf{q},\mathbf{r}_{0}).$$
$$(3)$$

log πθ(r˜1 = r | q, r0). (3)
We provide the proof in Appendix A. The assumption is compatible with the current denoising process, where unmasked tokens in rt are unchanged in subsequent steps. As rt generally contains richer context about r than r0, the model's output distribution over r tends to be broad and spread over many possible candidates in the early steps. In later steps, the already fixed tokens constrain which continuations remain plausible, so the probability mass concentrates on a smaller set of candidates. We empirically assess the validity of this assumption and further discuss its rationale in Appendix C.2, where we observe that it holds across a broad range of models.
Based on this theorem, we design *First-Step GCG* to maximize the lower bound as a surrogate
objective:
$$\operatorname*{max}_{\boldsymbol{s}}{\mathcal{L}}_{\mathrm{first}}(\boldsymbol{s})\triangleq\log\pi_{\theta}({\tilde{\boldsymbol{r}}}_{1}={\boldsymbol{r}}\mid{\boldsymbol{q}}\oplus{\boldsymbol{s}},{\boldsymbol{r}}_{0}).$$
Lfirst(s) ≜ log πθ(r˜1 = r | q ⊕ s, r0). (4)
Compared with MC sampling, optimizing the first-step log-likelihood is a more effective surrogate for two reasons. First, because the first step involves no masking, the objective is fully tractable and directly differentiable, avoiding gradient estimation over stochastic trajectories and thereby reducing computational cost. Second, as shown in Figure 2, even increasing the generation probability in the first step is sufficient to steer subsequent generations toward a harmful response. This effect helps compensate for the looseness of the lower bound and, in practice, yields strong attack performance. Results. We evaluate the advantage of First-Step GCG on the JBB-Behaviors dataset. As in Section 4.1, we used GPT-4o as an automatic evaluator. Following prior work Zou et al. (2023), we fixed the suffix length at 20 tokens and set the number of iterations to 500 (please see Section D.9

$$(4)$$

| ASR (%) and runtime per prompt (h). ASR is mean±std, and Time is the mean over three runs. LLaDA Instruct LLaDA 1.5 MMaDA MixCoT Method ASR (%) Per-prompt time (h) ASR (%) Per-prompt time (h) ASR (%) Per-prompt time (h) No Attack 2.0 ± 1.7 - 1.0 ± 0.0 - 79.7 ± 3.8 - Monte Carlo GCG 20.0 ± 4.2 4.3 12.5 ± 2.0 4.1 85.3 ± 3.5 4.8 First-Step GCG (ours) 58.0 ± 5.7 0.2 49.5 ± 2.1 0.2 92.7 ± 2.5 0.3   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

for details). As shown in Table 1, First-Step GCG achieves significant improvements in both efficiency and attack performance across all models. Compared to Monte Carlo GCG, our method is approximately 20× faster. Furthermore, it boosts the ASR by up to 4× on LLaDA-1.5. Remark. These results suggest that the priming vulnerability can be exploited even by a more realistic attacker and underscore that it is a pressing issue. In the following experiments, we employ First-Step GCG for evaluation because it is stronger and more computationally efficient.

## 5 Recovery Alignment

The priming vulnerability originates from how MDLMs are typically trained. In standard implementations (Nie et al., 2025; Zhu et al., 2025; Yang et al., 2025), the model is optimized to produce
safe responses when the denoising process starts from a fully masked sequence r0. This can be
interpreted as minimizing the probability of generating the harmful response r from the initial state r0:
$$\operatorname*{min}_{\theta}p_{\pi,m_{t}}(\mathbf{r}_{T}=\mathbf{r}\mid\mathbf{q},\mathbf{r}_{0}).$$
$$p_{\pi,m_{t}}(\mathbf{r}_{T}=\mathbf{r}\mid\mathbf{q},\mathbf{r}_{t})>p_{\pi,m_{t}}(\mathbf{r}_{T}=\mathbf{r}\mid\mathbf{q},\mathbf{r}_{0}),$$
θpπ,mt(rT = r | q, r0). (5)
However, this objective cannot resolve the vulnerability because it does not take into account contaminated intermediate states containing affirmative tokens. Informally, when rt includes such affirmative tokens, the following inequality holds:
pπ,mt(rT = r | q, rt) > pπ,mt(rT = r | q, r0), (6)
where the left-hand side conditions on a contaminated intermediate state and the right-hand side on the fully masked start. Thus, minimizing the right-hand side does not guarantee a decrease in the lefthand side. As a result, such training fails to constrain behavior at contaminated intermediate states, which explains why conventional alignment methods do not mitigate the priming vulnerability. To address this gap, we propose *Recovery Alignment* (RA), an alignment framework that trains a model to recover safe responses even from contaminated intermediate states. Here, we instantiate RA with a reward model and optimize it via an RLHF-style objective. Let Dh = {(q, r)} be a set of pairs of harmful queries and corresponding harmful responses. Let R : V
|q| × VL → R be a
reward model that computes a reward from a query and a response. We define the objective function
as follows:  $$\max_{\theta}\mathcal{J}_{\mathsf{RA}}(\theta)\triangleq\mathbb{E}_{(\mathbf{q},\mathbf{r})\in\mathcal{D}_{h}}\left[\mathcal{R}(\mathbf{q},\mathbf{r}_{T})\left|\begin{array}{c}\mathbf{r}_{\mathsf{tun}}\sim m_{\mathsf{tun}}(\cdot\mid\mathbf{r})\\ \mathbf{r}_{T}\sim p_{\pi,m_{t}}(\cdot\mid\mathbf{q},\mathbf{r}_{\mathsf{tun}})\end{array}\right.\right.\text{(Denosing from$t_{\mathsf{inner}}$to$T$step)}\right].\tag{7}$$
$$({\boldsymbol{5}})$$
$$(6)^{\frac{1}{2}}$$

As a practical advantage, this RLHF-style instantiation requires no additional data-construction costs. We can use existing datasets of harmful queries and harmful responses, such as the Beavertails dataset (Ji et al., 2023), for Dh. For the reward model R, we can also employ pretrained models that score responses in terms of safety and usefulness. This makes RA a practical and scalable solution. Linear schedule. As shown in Figure 2, the later the intervention step, the stronger the attack.

Thus, using a large intervention step tinter allows the model to recover from stronger attacks, thereby improving robustness. However, fixing tinter to a large value can destabilize training because generating a safe response in a few steps becomes difficult. Thus, we schedule tinter linearly over the course of training. Let S denote the total number of training steps and s ∈ {0*, . . . , S*} the current step. Given range [tmin, tmax], we set tinter = ⌊tmin +
s S
(tmax − tmin)⌋. This curriculum enables the model to start from easier conditions and gradually learn to produce safe responses even under increasingly challenging states.

## Algorithm 1 Recovery Alignment With Grpo

| 1: for s = 1, . . . , S do 2: tinter ←  tmin + s (tmax − tmin)             | ▷ Linear schedule                                  |                                  |                            |
|------------|----------------------------------------------------|----------------------------------|----------------------------|
| S          |                                                    |                                  |                            |
| 3:         | Sample mini-batch {(q (i) , r (i) )} B i=1 from Dh |                                  |                            |
| 4:         | for i = 1, . . . , B do (i) ← mtinter(· | r (i) )  | ▷ Contaminate intermediate state |                            |
| tinter (i) | (i)                                                |                                  |                            |
| T          | ← pπθ,mt (· | q i , r                              | )                                | ▷ Denoise from tinter to T |
| tinter     |                                                    |                                  |                            |
| (i)        |                                                    |                                  |                            |
| 7:         | R(i) ← R(q i , r                                   | )                                | ▷ Compute reward           |
| T          |                                                    |                                  |                            |
| 9:         | θ ← GRPO(θ, {R(i)} B i=1)                          | ▷ Update parameter               |                            |

Implementation details. We provide a simplified pseudo-code of RA in Algorithm 1. To optimize the objective, we used GRPO (Shao et al., 2024). The training process consists of three steps: (i)
generate intermediate states by replacing the predicted response with the harmful response at tinter, as in the anchoring attack, (ii) have the model generate responses from these intermediate states and score their safety and usefulness with the reward model, and (iii) update the model parameters to maximize the score. In Algorithm 2, we provide a more detailed procedure.

## 6 Experiments

We evaluate RA on two benchmarks, JBB-Behaviors (Chao et al., 2024) and AdvBench (Zou et al., 2023), and compute ASR using three evaluators: GPT-4o, LLaMA Guard 3 (Inan et al., 2023), and a keyword matching. Due to space constraints, this section reports only JBB-Behaviors results with ASR assessed by GPT-4o. All remaining results are provided in Appendix C.

## 6.1 Setup

Model and Training Setup. We applied RA to three MDLMs: LLaDA Instruct (Nie et al., 2025), LLaDA 1.5 (Zhu et al., 2025), and MMaDA MixCoT (Yang et al., 2025). For training, we used the BeaverTails dataset (Ji et al., 2023), which consists of harmful queries paired with harmful responses. As the reward model, we directly employ DeBERTaV3 (He et al., 2021; Kopf et al., 2023) ¨ without additional fine-tuning. All models were trained for 2,500 steps. We provide additional experiments on training cost and learning curves in Appendix C.4, and report the impact of generation length in Appendix C.5. Please see Appendix D.4 for detailed implementations. Attack methods. We evaluate safety with two families of jailbreak attacks. **(i) Attacks that** exploit the priming vulnerability. We considered four attacks: Anchoring Attack, First-Step GCG, PAD (Zhang et al., 2025), and DiJA (Wen et al., 2025). Among these, Anchoring Attack, PAD, and DiJA explicitly intervene in the denoising process by injecting tokens specified by the attacker. PAD inserts tokens at designated positions and fills all remaining positions with mask tokens. Specifically, the attack places "Step1:" at position 1 and "Step2:" at position ⌊
L
2
⌋,
with every other position masked. DiJA specifies both the locations and the counts of mask tokens more finely, e.g., "Subject: <mask:10>.\n First paragraph: <mask:30>.\n Second paragraph: <mask:20>.\n Closing remarks: <mask:15>." **(ii)**
Robustness to conversational jailbreaks. We use PAIR (Chao et al., 2025), ReNeLLM (Ding et al., 2024), and Crescendo (Russinovich et al., 2025). Although these attacks are originally designed for ARMs, they optimize prompts via a black-box API and are therefore likewise applicable to MDLMs. Implementation details for all attacks are provided in Appendix D.5. Baselines. To the best of our knowledge, no defense has been proposed specifically for the priming vulnerability. As baselines, we therefore include three general safety alignment methods originally designed to defend against jailbreak attacks: SFT, DPO (Rafailov et al., 2023), and MOSA (Xie et al., 2025). MOSA was introduced as an alignment method tailored to MDLMs, which maximizes

| cantly mitigates the vulnerability. Method No Attack   | Requires intervention in the denoising process   | No intervention   |                                  |            |            |            |            |            |            |            |
|--------------------------------------------------------|--------------------------------------------------|-------------------|----------------------------------|------------|------------|------------|------------|------------|------------|------------|
| Anchoring (tinter)                                     | PAD                                              | DiJA              | First-Step GCG                   |            |            |            |            |            |            |            |
| 1                                                      | 4                                                | 8                 | 16                               | 32         |            |            |            |            |            |            |
| Original                                               | 2.0 ± 1.7                                        | 17.3 ± 4.6        | 44.0 ± 4.6                       | 68.7 ± 0.6 | 88.7 ± 4.0 | 96.7 ± 1.5 | 67.3 ± 2.1 | 92.0 ± 0.0 | 58.0 ± 5.7 |            |
| SFT                                                    | 8.3 ± 4.2                                        | 19.0 ± 1.0        | 42.7 ± 4.9                       | 66.7 ± 3.2 | 87.7 ± 3.1 | 96.3 ± 2.1 | 66.3 ± 2.5 | 91.7 ± 2.3 | 48.2 ± 1.4 |            |
| DA DPO                                                 | 4.3 ± 2.3                                        | 10.0 ± 3.6        | 26.0 ± 3.0                       | 51.7 ± 6.5 | 81.7 ± 4.2 | 95.3 ± 1.2 | 35.3 ± 4.0 | 88.0 ± 1.0 | 46.3 ± 1.5 |            |
| LLa                                                    | MOSA                                             | 0.0 ± 0.0         | 6.0 ± 1.7                        | 24.0 ± 4.6 | 46.0 ± 4.6 | 79.7 ± 4.5 | 94.7 ± 0.6 | 32.3 ± 1.5 | 86.7 ± 0.6 | 28.0 ± 2.6 |
| RA w/o inter (ablation)                                | 1.7 ± 1.5                                        | 7.3 ± 2.1         | 22.0 ± 1.7                       | 49.0 ± 3.6 | 76.7 ± 2.5 | 92.3 ± 2.1 | 40.7 ± 1.5 | 82.3 ± 1.5 | 25.0 ± 4.0 |            |
| RA (ours)                                              | 0.0 ± 0.0                                        | 0.0 ± 0.0         | 1.3 ± 0.6                        | 3.0 ± 2.0  | 8.3 ± 1.5  | 50.7 ± 5.1 | 1.0 ± 0.0  | 35.7 ± 2.5 | 11.3 ± 2.1 |            |
| Original                                               | 1.0 ± 0.0                                        | 14.7 ± 0.6        | 35.0 ± 3.6                       | 62.0 ± 4.4 | 87.3 ± 2.9 | 96.7 ± 1.5 | 61.7 ± 5.5 | 89.7 ± 1.2 | 49.5 ± 2.1 |            |
| 1.5 SFT                                                | 6.3 ± 3.2                                        | 16.7 ± 2.9        | 31.7 ± 4.2                       | 59.3 ± 3.5 | 88.3 ± 6.7 | 95.3 ± 1.5 | 54.0 ± 6.6 | 89.7 ± 2.1 | 36.7 ± 2.1 |            |
| LLaDA                                                  | DPO                                              | 4.0 ± 1.0         | 9.0 ± 2.6                        | 23.0 ± 3.6 | 46.7 ± 4.6 | 80.7 ± 7.0 | 95.7 ± 1.5 | 36.0 ± 2.6 | 87.0 ± 1.7 | 42.0 ± 7.8 |
| MOSA                                                   | 0.7 ± 0.6                                        | 5.0 ± 2.0         | 19.7 ± 3.2                       | 43.0 ± 6.0 | 77.7 ± 3.2 | 93.3 ± 2.1 | 26.3 ± 2.5 | 84.3 ± 1.5 | 26.3 ± 2.9 |            |
| RA w/o inter (ablation)                                | 1.0 ± 1.0                                        | 7.0 ± 2.6         | 27.7 ± 2.9                       | 51.3 ± 2.3 | 77.3 ± 1.5 | 93.3 ± 1.2 | 49.3 ± 0.6 | 81.7 ± 0.6 | 27.7 ± 0.6 |            |
| RA (ours)                                              | 0.0 ± 0.0                                        | 1.0 ± 0.0         | 0.7 ± 0.6                        | 2.7 ± 1.2  | 7.3 ± 0.6  | 43.0 ± 4.6 | 1.0 ± 0.0  | 36.0 ± 3.0 | 15.0 ± 4.0 |            |
| Original                                               | 79.7 ± 3.8                                       | 90.0 ± 1.7        | 93.7 ± 3.1                       | 94.7 ± 1.5 | 98.3 ± 0.6 | 99.0 ± 1.0 | 99.3 ± 1.2 | 97.3 ± 1.5 | 92.7 ± 2.5 |            |
| A                                                      | SFT                                              | 46.0 ± 4.6        | 51.7 ± 1.5                       | 81.3 ± 1.5 | 90.0 ± 3.6 | 97.0 ± 1.0 | 98.3 ± 1.5 | 99.7 ± 0.6 | 95.7 ± 0.6 | 65.3 ± 5.8 |
| MaD                                                    | DPO                                              | 39.0 ± 3.0        | 55.7 ± 1.5                       | 74.3 ± 0.6 | 86.7 ± 0.6 | 96.3 ± 2.1 | 97.7 ± 1.2 | 98.0 ± 1.0 | 98.3 ± 0.6 | 57.7 ± 2.5 |
| MOSA                                                   | 22.3 ± 3.1                                       | 25.0 ± 4.6        | 45.7 ± 6.0                       | 64.0 ± 2.6 | 84.7 ± 0.6 | 96.0 ± 1.0 | 84.0 ± 2.6 | 94.0 ± 2.0 | 44.7 ± 4.5 |            |
| M                                                      | RA w/o inter (ablation)                          | 2.0 ± 1.3         | 6.3 ± 2.3                        | 25.3 ± 1.5 | 49.3 ± 4.0 | 80.7 ± 2.1 | 94.7 ± 0.6 | 35.7 ± 4.9 | 88.0 ± 0.0 | 50.7 ± 1.2 |
| RA (ours)                                              | 3.3 ± 1.2                                        | 6.3 ± 2.3         | 13.0 ± 2.0 15.7 ± 1.5 34.3 ± 1.2 | 79.3 ± 5.7 | 24.3 ± 4.5 | 70.0 ± 2.6 | 45.7 ± 6.5 |            |            |            |

the difference in maximum log-likelihood between safe phrases and harmful phrases over middle tokens in responses. As an ablation, we also report the results of *RA w/o inter*, where we set tmin = tmax = 0 and train the model only from the fully masked sequences without intervention, same as RLHF Ouyang et al. (2022). Full baseline configurations are provided in Appendix D.6.

## 6.2 Robustness To Attacks

Mitigation of the priming vulnerability Table 2 presents the ASR for attack methods leveraging the priming vulnerability. Two key observations emerge. **(i) RA mitigates the vulnerability.** Across all models, RA consistently outperforms the baselines and achieves state-of-the-art robustness. This finding substantiates the effectiveness of our approach. However, when the intervention step is very late, such as tinter = 32, generating a fully safe response becomes challenging. This is because it is practically impossible to generate a contextually safe response due to many anchors.

(ii) Training from contaminated intermediate states is crucial. RA (w/o inter), which omits training on contaminated states, does not sufficiently reduce the priming vulnerability: at tinter = 4, the ASR exceeds 20%, and other baselines show similar trends. These results support our analysis in Section 5, which suggests that existing alignments are insufficient, and effective mitigation requires training the model to generate safe responses from contaminated intermediate states. Accordingly, we strongly recommend alignment procedures that explicitly condition on and learn from such contaminated states to counter the priming vulnerability. Robustness to conventional jailbreak attacks. Table 3 presents the ASR under conversational jailbreak attacks. RA achieves superior robustness against such attacks and outperforms baselines. This suggests that training on contaminated intermediate states can effectively generalize to a wide range of jailbreak attacks. A plausible mechanism is that the model acquires a new recovery capability. Specifically, when the model generates a harmful response, corresponding harmful tokens necessarily emerge at intermediate steps regardless of the specific attack. Thus, even if harmfulness is not detected at the first step, a model trained by RA is more likely to re-detect harmfulness at later steps and steer the generation back to a safe trajectory. Nevertheless, RA remains imperfect against strong attacks, such as ReNeLLM, indicating that

L

LaDA

Original 44.3 ± 1.2 92.7 ± 0.6 81.3 ± 4.9 SFT 36.7 ± 3.2 94.3 ± 1.5 71.0 ± 3.5 DPO 31.3 ± 5.0 88.3 ± 2.1 74.0 ± 1.7 MOSA 27.3 ± 1.5 77.7 ± 4.5 66.3 ± 3.5

RA w/o inter 26.3 ± 2.5 75.7 ± 3.8 71.3 ± 2.1 RA (ours) 10.0 ± 2.0 72.3 ± 8.0 45.0 ± 2.0

L

LaDA

1.5

Original 45.3 ± 4.0 96.7 ± 1.5 81.7 ± 4.2

SFT 39.0 ± 5.6 91.3 ± 1.5 70.7 ± 6.8

DPO 36.0 ± 3.6 90.7 ± 2.3 74.3 ± 4.5

MOSA 25.0 ± 1.0 78.3 ± 2.3 70.0 ± 5.6

RA w/o inter 38.7 ± 2.1 79.3 ± 4.0 68.5 ± 0.7

RA (ours) 16.0 ± 3.6 71.7 ± 3.1 47.0 ± 2.6

MMaDA

Original 98.0 ± 1.7 79.3 ± 5.5 93.0 ± 3.6 SFT 92.0 ± 2.0 95.0 ± 1.0 93.5 ± 2.1 DPO 67.5 ± 2.1 82.3 ± 5.5 85.7 ± 2.5 MOSA 59.0 ± 2.0 75.7 ± 2.1 71.0 ± 1.7

RA w/o inter 54.3 ± 1.0 77.6 ± 4.6 72.0 ± 0.9 RA (ours) 46.3 ± 4.0 81.7 ± 3.5 55.3 ± 4.6

| Method                  | Evaluation Tasks (↑)   |       |      |       |         |      |      |      |         |       |      |      |
|-------------------------|------------------------|-------|------|-------|---------|------|------|------|---------|-------|------|------|
| ARC-C                   | CEval                  | CMMLU | GPQA | HSwag | HumEval | MBPP | MMLU | PIQA | TruthQA | WinoG | Avg. |      |
| LLaDA Original          | 53.3                   | 66.1  | 67.0 | 27.9  | 54.0    | 22.0 | 25.8 | 64.0 | 74.4    | 47.6  | 72.5 | 52.2 |
| RA w/o inter (ablation) | 53.2                   | 66.6  | 67.0 | 28.9  | 54.0    | 20.7 | 28.6 | 63.8 | 73.7    | 50.1  | 72.6 | 52.7 |
| RA (ours)               | 53.9                   | 66.3  | 66.9 | 30.4  | 54.0    | 17.1 | 27.2 | 63.9 | 71.6    | 53.4  | 73.4 | 52.6 |
| LLaDA1.5 Original       | 54.4                   | 65.8  | 67.1 | 29.5  | 54.4    | 21.3 | 28.2 | 64.0 | 74.9    | 47.2  | 72.9 | 52.7 |
| RA w/o inter (ablation) | 54.4                   | 66.2  | 67.0 | 29.5  | 54.5    | 19.5 | 29.2 | 64.0 | 74.1    | 49.6  | 73.2 | 52.8 |
| RA (ours)               | 54.4                   | 66.2  | 67.1 | 29.0  | 54.3    | 18.9 | 29.4 | 63.7 | 70.6    | 54.1  | 73.2 | 52.8 |
| MMaDA Original          | 27.8                   | 35.9  | 32.2 | 25.0  | 35.7    | 7.9  | 3.8  | 36.8 | 61.0    | 46.2  | 53.1 | 33.2 |
| RA w/o inter (ablation) | 26.3                   | 33.2  | 32.5 | 29.7  | 37.1    | 10.0 | 8.0  | 39.8 | 60.8    | 49.1  | 55.4 | 34.7 |
| RA (ours)               | 26.0                   | 33.5  | 33.1 | 29.2  | 36.7    | 9.8  | 7.6  | 40.1 | 60.6    | 52.6  | 55.6 | 35.0 |

![8_image_1.png](8_image_1.png) 

![8_image_0.png](8_image_0.png)

the alignment can be circumvented when the harmfulness is not detectable from the surface form of the response.

## 6.3 General Capability Evaluation

We measure general capability on eleven diverse benchmarks: ARC-Challenge (Clark et al., 2018), C-Eval (Huang et al., 2023), CMMLU (Li et al., 2024a), GPQA (Rein et al., 2024), HellaSwag (Zellers et al., 2019), HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021b), MMLU (Wang et al., 2024b), PIQA (Bisk et al., 2020), TruthfulQA Lin et al. (2022), and Wino- Grande (Sakaguchi et al., 2021). We use the lm-evaluation-harness for implementation and replicate the generation configurations used in prior work (Nie et al., 2025). Table 4 summarizes general capability across multiple tasks. We do not observe substantial degradation from recovery alignment. On LLaDA and LLaDA 1.5, performance on TruthfulQA and MBPP improves. We attribute this to reward-model-based alignment, enhancing truthfulness and instruction following. In contrast, PIQA decreases slightly, which may be attributed to potential forgetting effects or output style shifts associated with alignment. For MMaDA, performance improves overall, likely because its baseline instruction-following ability was weaker and benefited more from alignment. Differences with and without harmful initialization are minimal, indicating that the negative impact on general capability is negligible.

## 6.4 Ablation Study

Impact of max intervention step. We examine the impact of the intervention step on robustness.

Figure 3a reports the results of the anchoring attack on models trained with various tmax. The results show that robustness improves as the intervention step becomes larger. This is consistent with the observation in Section 4.1 that the later the intervention timing, the higher the ASR. A model trained with a larger tmax becomes robust against more powerful attacks. On the other hand, an excessively large tmax destabilizes training. We observe reward hacking, where the model generates responses that are meaningless. Impact of intervention step scheduling. Next, we evaluate the effect of linearly scheduling the intervention step. We compared linear scheduling with two baselines: (i) *const scheduling*, which fixes tinter = tmax and (ii) *uniform scheduling*, which samples tinter ∼ U([tmin, tmax]at each training step. Figure 3b shows that the ASR against anchoring attack with tinter = 16. Linear scheduling achieves the highest robustness. Uniform scheduling remains effective but consistently underperforms linear scheduling, corroborating the benefit of a curriculum. Constant scheduling fails to achieve adequate robustness. With small tmax, the model never encounters harder states and remains vulnerable. With large tmax, learning becomes difficult and robustness cannot be obtained.

## 7 Conclusion

In this work, we investigate the priming vulnerability, which is specific to MDLMs. We first demonstrate that attackers can readily exploit this vulnerability via interventions, highlighting the limitations of existing safety alignment. We further show, through theoretical analysis, its potential extension to jailbreak attacks that require no explicit interventions. Building on these insights, we propose *recovery alignment*, a method that teaches models to produce safe responses from harmful intermediate states. Our experiments show that recovery alignment effectively mitigates priming vulnerability. This paper highlights the importance of safety alignment tailored to MDLMs and provides a new perspective on achieving it. Limitations. This work focuses on an RLHF-style instantiation of RA. However, a supervised alternative, such as a DPO-style approach, should also be feasible. This approach requires constructing safe responses aligned to contaminated intermediate states, which introduces substantial data-construction cost. If this bottleneck were addressed, such supervised training might reduce training time while retaining, or possibly improving, robustness against the priming vulnerability.

## 8 Acknowledgements

This work is partially supported by JST JPMJNX25C2, JPMJKP24C3, JPMJCR23M4, JP- MJCR21D3, JSPS 23H00483, and 120251002. We gratefully acknowledge the insightful comments and suggestions provided by the anonymous reviewers.

## 9 Ethics, Reproducibility, And Llm Usage

Ethics statements. We only use publicly available datasets and do not involve any human subjects or personal data. While our work proposes harmful methodologies, it also designs the countermeasure and aims to improve the robustness of MDLMs. We do not have any conflicts of interest or sponsorship to disclose. We have followed the ethical guidelines and research integrity standards in our work. Reproducibility. We report all training and evaluation hyperparameters in Section 6.1. Additional implementation details for our method and all baselines are provided in Appendix D. Detailed algorithmic descriptions of the proposed methods are included in the appendix. All benchmarks used in our experiments are publicly available and accessible to the community. Declaration of LLM usage. We used LLMs during manuscript preparation solely as writing assistants, for grammar checking and improving the clarity and naturalness of the text. All LLM- generated suggestions were manually reviewed and edited by the authors. LLMs did not play any role in developing the core methods of this research.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion. Jailbreaking leading safetyaligned LLMs with simple adaptive attacks. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=hXA8wqRdyV.

Cem Anil, Esin Durmus, Nina Panickssery, Mrinank Sharma, Joe Benton, Sandipan Kundu, Joshua Batson, Meg Tong, Jesse Mu, Daniel Ford, et al. Many-shot jailbreaking. Advances in Neural Information Processing Systems, 37:129696–129742, 2024a.

Cem Anil, Esin DURMUS, Nina Rimsky, Mrinank Sharma, Joe Benton, Sandipan Kundu, Joshua Batson, Meg Tong, Jesse Mu, Daniel J Ford, Francesco Mosconi, Rajashree Agrawal, Rylan Schaeffer, Naomi Bashkansky, Samuel Svenningsen, Mike Lambert, Ansh Radhakrishnan, Carson Denison, Evan J Hubinger, Yuntao Bai, Trenton Bricken, Timothy Maxwell, Nicholas Schiefer, James Sully, Alex Tamkin, Tamera Lanham, Karina Nguyen, Tomasz Korbak, Jared Kaplan, Deep Ganguli, Samuel R. Bowman, Ethan Perez, Roger Baker Grosse, and David Duvenaud. Manyshot jailbreaking. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024b. URL https://openreview.net/forum?id=cw5mgd71jW.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021a.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021b.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. *arXiv preprint arXiv:2212.08073*, 2022.

Tim Beyer, Sophie Xhonneux, Simon Geisler, Gauthier Gidel, Leo Schwinn, and Stephan Gunnemann. Llm-safety evaluations lack robustness. ¨ *arXiv preprint arXiv:2503.02574*, 2025.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pp. 7432–7439, 2020.

Hongyu Cai, Arjun Arunasalam, Leo Y Lin, Antonio Bianchi, and Z Berkay Celik. Take a look at it! rethinking how to evaluate language model jailbreak. *arXiv preprint arXiv:2404.06407*, 2024.

Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J Pappas, Florian Tramer, et al. Jailbreakbench: An open robustness benchmark for jailbreaking large language models. Advances in Neural Information Processing Systems, 37:55005–55029, 2024.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong.

Jailbreaking black box large language models in twenty queries. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pp. 23–42. IEEE, 2025.

Kexin Chen, Yi Liu, Dongxia Wang, Jiaying Chen, and Wenhai Wang. Characterizing and evaluating the reliability of llms against jailbreak attacks. *arXiv preprint arXiv:2408.09326*, 2024.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

Junjie Chu, Yugeng Liu, Ziqing Yang, Xinyue Shen, Michael Backes, and Yang Zhang. Comprehensive assessment of jailbreak attacks against llms. *arXiv e-prints*, pp. arXiv–2402, 2024.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe RLHF: Safe reinforcement learning from human feedback. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/
forum?id=TyFrPOKYXw.

DeepMind. Gemini diffusion. https://deepmind.google/technologies/gemini, 2024. Accessed: 2025-07-09.

Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H
Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, et al. Continuous diffusion for categorical data. *arXiv preprint arXiv:2211.15089*, 2022.

Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, and Shujian Huang. A
wolf in sheep's clothing: Generalized nested jailbreak prompts can fool large language models easily. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 2136–2153, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. *arXiv preprint arXiv:2402.01306*, 2024.

Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and Lingpeng Kong. Diffuseq: Sequence to sequence text generation with diffusion models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=jQj-_ rLVXsj.