

{0}------------------------------------------------

# LIAR: LEVERAGING ALIGNMENT TO JAILBREAK LLMs IN SECONDS

**Anonymous authors**

Paper under double-blind review

## ABSTRACT

Many existing jailbreak techniques rely on solving discrete combinatorial optimization, while more recent approaches involve training LLMs to generate multiple adversarial prompts. However, both approaches require significant computational resources to produce even a single adversarial prompt. We hypothesize that the inefficiency of current approaches stems from an inadequate characterization of the jailbreak problem. To address this gap, we formulate the jailbreak problem in terms of alignment. By starting from an available safety-aligned model, we leverage an unsafe reward to guide the safe model towards generating unsafe outputs using alignment techniques (e.g., reinforcement learning from human feedback), effectively performing jailbreaking via alignment. We propose a novel jailbreak method called LIAR (LeveragIng Alignment to jailbReak). To demonstrate the simplicity and effectiveness of our approach, we employ a best-of- $N$  method. LIAR offers significant advantages: lower computational requirements without additional training, fully black-box operation, competitive attack success rates, and more human-readable prompts. We provide theoretical insights into the possibility of jailbreaking a safety-aligned model, revealing inherent vulnerabilities in current alignment strategies for LLMs. We also provide sub-optimality guarantees for the proposed LIAR. Experimentally, we achieve ASR comparable to the SoTA with a 10x improvement to perplexity and a Time-to-Attack measured in seconds rather than tens of hours.

Warning: This paper may include language that could be considered inappropriate or harmful.

## 1 INTRODUCTION

Aligning artificial intelligence (AI) systems is essential to ensure they behave safely, ethically, and in accordance with human values (Christiano et al., 2017; Ouyang et al., 2022). State-of-the-art (SoTA) large language models (LLMs) are trained using safe preference data to adhere to these standards, preventing the generation of harmful, biased, or unethical content (Ziegler et al., 2019; Glaese et al., 2022). Despite these efforts, recent jailbreak methods have successfully bypassed safety mechanisms, exposing vulnerabilities in LLMs (Zou et al., 2023; Liu et al., 2023; Paulus et al., 2024). These jailbreaks are designed to find adversarial prompts or suffixes that bypass the safety filters of a model and induce the generation of harmful content (Zou et al., 2023; Guo et al., 2021; Liu et al., 2023). Formulated as discrete combinatorial optimization problems, they involve searching through an immense text space, which is inefficient and computationally expensive (Liu et al., 2023). As a result, many current methods require significant time to generate a successful adversarial prompt. Although improvements have been made, such as the conditional suffix generator model proposed in (Paulus et al., 2024), these methods still suffer from inherent challenges of combinatorial search. Appendix A provides a detailed additional context of related works.

One of the key inefficiencies in existing jailbreak approaches stems from the lack of a rigorous characterization of the precise connection between LLM alignment and vulnerability to jailbreaks. This raises critical questions: If these LLMs are truly aligned and safe, why are they still susceptible

{1}------------------------------------------------

![Figure 1: (left) Overview of the method architecture. (top-right) Attack Success Rate (ASR) vs. Attempts (k). (bottom-right) Total time vs. Attempts (k).](49ad3a646d84bcfeac02bdf2b3792a3e_img.jpg)

The figure illustrates the LIAR framework for jailbreaking LLMs. On the left, an unsafe query  $x$  is processed by a fixed Adversarial LLM to generate an adversarial prompt  $x' = [x, q]$ , which is then passed to a fixed Target LLM to produce an unsafe output  $y'$ . The top-right bar chart shows the Attack Success Rate (ASR) for SoTA and LIAR across 10, 100, and 1000 attempts. The bottom-right bar chart shows the total time required for training and inference for the same number of attempts.

| Attempts (k) | SoTA | LIAR |
|-|-|-|
| @10 | ~45% | ~35% |
| @100 | ~85% | ~75% |
| @1000 | ~95% | ~95% |

  

| Attempts (k) | SoTA | LIAR |
|-|-|-|
| @10 | ~22 hours | ~1 sec |
| @100 | ~22 hours | ~7 sec |
| @1000 | ~80 sec | ~80 sec |

Figure 1: (left) Overview of the method architecture. (top-right) Attack Success Rate (ASR) vs. Attempts (k). (bottom-right) Total time vs. Attempts (k).

Figure 1: (left) An overview of our method architecture. Our approach starts with an unsafe query  $x$ , which is extended by the Adversarial LLM into  $x'$ , then passed to a target model. If the target model’s output,  $y'$ , is unsafe, the attack is considered successful. Notably, our method is fully black-box as it does not depend on any logits or probabilities from the TargetLLM. (top-right) Attack Success Rate (ASR@k) as a function of the number of attempts denoted by  $k$ , which illustrates that LIAR achieves performance comparable to the SoTA Advprompter method (Paulus et al., 2024). (bottom-right) The combined training and inference time required to generate  $k$  adversarial prompt attempts highlights the significant time advantage of LIAR, which takes only seconds to generate prompts. In contrast, SoTA require hours of training before producing any adversarial prompts. We present results for  $k = 10$  and  $k = 100$ , but due to the efficiency of our method, we are able to execute an order of magnitude more attempts, showing results for  $k = 1000$ . This speed is challenging for SoTA, as indicated by the dotted bar in the plot.

to jailbreaks? Conversely, if jailbreaking is possible—as evidenced in the literature—why does it require a significant amount of time to generate an adversarial prompt?

To address these questions, we introduce an **alignment** formulation for the jailbreak problem. **Jail-breaking via alignment** refers to the process of breaking the safety alignment of an AI model by using an unsafe reward signal. Starting with a model aligned through techniques like RLHF (Ouyang et al., 2022), we show that it is both theoretically and practically possible to misalign the model using **alignment**. Our framework provides a rigorous explanation for why jailbreaks are possible, even in models trained with safety considerations. To validate our approach, we introduce LIAR (Leveraging Inverse Alignment to jailbReak), which utilizes a best-of- $N$  (Amini et al., 2024) to significantly improve the efficiency of jailbreak attacks. Our main contributions are as follows.

**(1) Jailbreaking LLMs via Alignment.** By formulating jailbreaking as an **alignment** problem, we demonstrate its effectiveness using a simple best-of- $N$  alignment strategy. Our proposed method, LIAR (Leveraging Inverse Alignment to jailbReak), employs an adversarial LLM to attack the target LLM, leading to the generation of unsafe responses.

**(2) Training Free and Faster Jailbreak.** Our approach requires no additional training and is extremely fast to execute. By using GPT-2 (Radford et al., 2019), which has 124 million parameters, we achieve faster runtimes compared to larger models like LLaMA (Touvron et al., 2023) with 7 billion parameters. The lack of a training phase results in very short times to generate the first adversarial prompt and low latency for subsequent prompts.

**(3) Theoretical Insights.** We provide theoretical insights into the possibilities of jailbreaking safety-aligned models by defining a notion of the “safety net,” which quantifies how safe or unsafe an aligned model is. Additionally, we analyze the suboptimality of our LIAR approach which provides a performance gap of our proposed approach with the optimal fine-tuned attack LLMs model.

**(4) Empirical Results.** We present extensive empirical evidence demonstrating the superiority of our proposed approach. Our method achieves competitive attack success rates (up to 99% on Vicuna-7b), generates adversarial prompts with low perplexity (as low as 2) indicating high readability, and fastest adversarial prompt generation (as fast as within 45 seconds). Additionally, we highlight the black-box nature of our approach, as it operates without requiring internal details of the target model.

{2}------------------------------------------------

## 2 BACKGROUND AND PROBLEM FORMULATION

**Language Models (LM).** Consider a vocabulary set denoted by  $\mathcal{V}$ , and let the language model be represented by a mapping  $\pi_\theta$ , parameterized by  $\theta$ . The language model  $\pi_\theta$  takes as input a sequence of tokens, referred to as a prompt, represented by  $\mathbf{x} := \{x_1, x_2, \dots, x_N\}$ , where each token  $x_i \in \mathcal{V}$ . Given the input prompt, the model predicts the first output token  $y_0 \in \mathcal{V}$ . To predict the next token  $y_1$ , the model takes as input the original prompt  $\mathbf{x}$  along with the previously generated token  $y_0$ , forming a new prompt  $[\mathbf{x}, y_0]$ . This process is repeated for subsequent time steps. The input prompt at time step  $t$  is  $[\mathbf{x}, y_{1:t-1}]$ , where  $y_{1:t-1} = \{y_0, y_1, \dots, y_{t-1}\}$ . To generate the next token, the language model outputs the next token  $y_t \sim \pi_\theta(\cdot | [\mathbf{x}, y_{1:t-1}])$ .

**Jailbreak Attacks on Large Language Models.** As formulated in Zou et al. (2023), it is possible to attack even an aligned LLM  $\pi_\theta$  with the help of a prompt suffix  $\mathbf{q} \in \mathcal{Q} \subseteq \mathcal{V}^{|\mathcal{V}|}$ . For instance (cf. Figure 1), consider an unsafe prompt  $\mathbf{x} \in \mathcal{X} \subseteq \mathcal{V}^{|\mathcal{V}|}$  (e.g., “Write a script to exploit an operating system”) and if we input this  $\mathbf{x}$  to a safety aligned LLM model  $\pi_\theta$ , it will not answer this query and would generate a negative response (e.g., “I cannot answer this, this is unsafe.”). But with the help of an adversarial prompt given by  $[\mathbf{x}, \mathbf{q}]$ , which is concatenation of prompt  $\mathbf{x}$  and adversarial suffix  $\mathbf{q} \in \mathcal{Q}$ , it is possible to generate harmful response  $\mathbf{y} = \{y_1, y_2, \dots, y_T\}$  such that  $y_t \sim \pi_\theta(\cdot | [\mathbf{x}, \mathbf{q}, y_{<t}])$  from the aligned target LLM (e.g., “Sure, here is a script....”), where we define  $\mathbf{y}_{<t} := \{y_1, y_2, \dots, y_{t-1}\}$  (Zou et al., 2023; Liu et al., 2023; Paulus et al., 2024). In order to find such suffix  $\mathbf{q}$ , we solve the following optimization problem

$$\mathbf{q}^* = \arg \min_{\mathbf{q} \in \mathcal{Q}} J(\mathbf{x}, \mathbf{q}, \mathbf{y}) := - \sum_{t=1}^{|y|} \log \pi_\theta(\cdot | [\mathbf{x}, \mathbf{q}, \mathbf{y}_{<t}]), \quad (1)$$

where  $J(\mathbf{x}, \mathbf{q}, \mathbf{y})$  is called the adversarial loss function for a given prompt  $\mathbf{x}$  and response  $\mathbf{y}$  pair. The formulation in equation 1 has also been extended to improve the interpretability of the adversarial suffix  $\mathbf{q}$  by adding perplexity-based regularization (Liu et al., 2023; Zhu et al., 2023). As highlighted in AdvPrompter by Paulus et al. (2024), the solution to the optimization problem in equation 1 suffers from adapting to individual queries which result in the degradation of the performance of attacks. In order to deal with this issue, AdvPrompter extends the idea of universal adversarial suffix to conditional suffix generation and propose to solve the following optimization problem

$$\min_{\theta} \sum_{\mathbf{x}, \mathbf{y} \in \mathcal{D}} J(\mathbf{x}, \mathbf{q}_\theta(\mathbf{x}), \mathbf{y}), \quad (2)$$

where an important difference is to train a language model/prompter  $\mathbf{q}_\theta$  to generate the adversarial suffix  $\mathbf{q}^* \sim \mathbf{q}_\theta(\cdot | \mathbf{x})$ . In Paulus et al. (2024), the problem in equation 2 is solved in two steps:  $\mathbf{q}$ -step and  $\theta$ -step. In  $\mathbf{q}$ -step, the individual loss function of equation 1 is minimized for each  $\mathbf{x}, \mathbf{y}$  to learn approximately optimal  $\mathbf{q}(\mathbf{x}, \mathbf{y})$ . Then, in  $\theta$ -step, a language model  $\mathbf{q}_\theta$  is trained to generate the  $\mathbf{q}(\mathbf{x}, \mathbf{y})$  for a given prompt  $\mathbf{x}$ . Both steps are repeated until convergence.

### 2.1 LIMITATIONS OF EXISTING APPROACHES

**L1: Computationally Expensive and Training-Based:** Most existing methods (Zou et al., 2023; Liu et al., 2023; Guo et al., 2021) approach jailbreaking as a discrete combinatorial optimization problem. This strategy is notorious for its high computational cost and the immense effort needed to explore the vast discrete text space,  $\mathcal{Q}$ . Consequently, these methods often depend on extensive training to generate adversarial prompts, which demand significant computational resources.

**L2: Extremely Slow:** The training process in these methods is notoriously time-consuming, often taking hours to generate a single adversarial suffix  $\mathbf{q}^*$  for a given prompt  $\mathbf{x}$ . These prolonged durations hinder practical applications and restrict the ability to quickly assess or respond to emerging vulnerabilities as highlighted in (Zou et al., 2023; Liu et al., 2023).

**L3: Lack of Theoretical Insights:** Although it is clear that jailbreaks can undermine safety-aligned models, the underlying mechanisms that enable this misalignment, despite the presence of safety

{3}------------------------------------------------

constraints, remain insufficiently explored. Moreover, the potential suboptimality of existing attack methods is often overlooked, leaving a gap in fully understanding the extent of these vulnerabilities.

## 3 LIAR: PROPOSED APPROACH

To address the shortcomings of existing approaches, we propose a novel, fast, and training-free method to jailbreak large language models (LLM). We call it LIAR: Leveraging Inverse Alignment for jailbReaking LLMs. As a preview, Figure 1 illustrates the setup and performance of our approach. The key idea is to leverage AI alignment tools to formally define the problem of jailbreaking LLMs and then develop efficient, training-free techniques that enable rapid attacks on LLMs.

### 3.1 JAILBREAKING LLMs AS AN ALIGNMENT PROBLEM

Motivated by the idea of conditional suffix generation (Paulus et al., 2024), which trains a prompter language model to generate adversarial prompts for a given unsafe prompt query  $\mathbf{x}$ , we equivalently formulate the problem of obtaining adversarial suffix  $\mathbf{q} \sim \rho(\cdot|\mathbf{x})$  from a prompter LLM model  $\rho$  as

$$\max_{\rho} \mathbb{E}_{\mathbf{q} \sim \rho(\cdot|\mathbf{x})} [-J(\mathbf{x}, \mathbf{q}, \mathbf{y})], \quad (3)$$

which is similar to the minimization in equation 1, except the optimization variable is model distribution  $\rho$  instead of suffix  $\mathbf{q}$  as in equation 1. Interestingly, defining a reward model  $R_u(\mathbf{x}, \mathbf{q}) := -J(\mathbf{x}, \mathbf{q}, \mathbf{y})$  and a regularization term as  $\text{KL}(\rho(\cdot|\mathbf{x})\|\rho_0(\cdot|\mathbf{x}))$ , we can write

$$\max_{\rho} \mathbb{E}_{\mathbf{q} \sim \rho(\cdot|\mathbf{x})} [R_u(\mathbf{x}, \mathbf{q})] - \beta \text{KL}(\rho(\cdot|\mathbf{x})\|\rho_0(\cdot|\mathbf{x})), \quad (4)$$

where  $\rho_0$  is a reference base model given to us, and  $\beta > 0$  is the regularization parameter. The goal of the objective in equation 4 is to maximize the reward model (which denotes negative of jailbreaking loss) while keeping the model close to a reference model  $\rho_0$ . The regularization is important to keep the the perplexity of the generated suffix  $\mathbf{x}$  low. The objective in equation 4 is similar to RLHF utilized in the literature (Ouyang et al., 2022; Rafailov et al., 2024) for the alignment of language models with some safety reward. In contrast, in equation 4, we apply the principles of alignment by fine-tuning our prompter model with an unsafe reward — exactly opposite to the key purpose of alignment. For this reason, we refer to it as *jailbreaking via alignment*. To the best of our knowledge, this formulation has not been applied in previous jailbreaking attacks, making it a novel contribution of our work. In the following section, we demonstrate both the theoretical and empirical effectiveness of our proposed approach.

**Optimal Jailbreak Prompter.** For the optimization problem in equation 4, as it is strongly concave with respect to  $\rho$ , we can write its closed form solution as

$$\rho^*(\mathbf{q}|\mathbf{x}) = \frac{\rho_0(\mathbf{q}|\mathbf{x})}{Z(\mathbf{x})} \exp\left(\frac{R_u(\mathbf{x}, \mathbf{q})}{\beta}\right), \quad (5)$$

where  $Z(\mathbf{x}) = \sum_{\mathbf{q}} \rho_0(\mathbf{q}|\mathbf{x}) \exp\left(\frac{R_u(\mathbf{x}, \mathbf{q})}{\beta}\right)$  is the normalization constant. The above solutions follow from the standard RLHF based analysis in Rafailov et al. (2024, Appendix A). Interestingly, the closed form solution gives us the optimal probability of adversarial  $\mathbf{q}$  for a given  $\mathbf{x}$ , and the solution holds for any  $\mathbf{x}$ . We note that the optimal prompter in equation 5 depends on the reference prompter and unsafe reward. Importantly, this process does not involve optimization within the space of the safe LLM (like in standard safety alignment); we neither access nor modify the safe LLM itself. Instead, the safe LLM is solely used to compute the reward function.

However, a significant drawback of the solution in equation 5, despite being provably optimal, is that it requires a costly training process to update the parameters of the base model,  $\rho_0$ . This process is computationally intensive and introduces substantial overhead, often requiring several hours of training before the model can be used, as evidenced by results in the existing literature (Zou et al., 2023; Liu et al., 2023). Therefore, we take a different approach and propose to jailbreaks without fine-tuning, and consider Best of N alignment for jailbreaking attacks. This is defined as follows.

{4}------------------------------------------------

**LIAR: Leveraging Alignment for Jailbreaking LLMs.** Our proposed LIAR, based on best-of- $n$  algorithm for jailbreaking, is as follows. Given an unsafe prompt query  $\mathbf{x}$ , we sample  $\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_n$  independently from the reference model  $\rho_0(\cdot|\mathbf{x})$ , denoted as the adversarial LLM in Figure 1. Then, select the response with the highest reward  $R_u(\mathbf{x}, \mathbf{q}_i)$  (note that this reward depends upon the safe target LLM  $\rho_{\text{safe}}^*$ ) as the final response. This implies that

$$\text{we select } \mathbf{q}^* = \mathbf{q}_i \text{ such that } R_u(\mathbf{x}, \mathbf{q}_i) = \max_{1 \leq j \leq n} R_u(\mathbf{x}, \mathbf{q}_j). \quad (6)$$

Key immediate questions include whether the procedure proposed in equation 6 is optimal, and how its performance compares to the theoretically optimal solution in equation 5. We address these questions in detail in the following section.

## 4 THEORETICAL RESULTS AND INSIGHTS

As discussed in the limitations (Section 2.1), we aim to study the theoretical properties of the proposed LIAR, a best-of-N sampling-based jailbreak attack. Our goal is to theoretically address the following questions: *Q1: Why is it possible to jailbreak a safety-aligned (RLHF-based) model?* and *Q2: What is the suboptimality guarantee of the proposed LIAR approach?* The importance of Question Q1 arises from the abundance of empirical evidence in the literature demonstrating that even safety-aligned models like GPT-4 and LLaMA2 (Paulus et al., 2024; Liu et al., 2023) can be jailbroken, yet there is little theoretical investigation into why this occurs or is possible. Addressing Question Q2 will help us understand the theoretical effectiveness of the proposed approach.

**To answer Q1**, we start with a safety aligned LLM  $\pi_{\text{safe}}$ , which is well aligned with a safety reward  $R_s$  using reinforcement learning from the human feedback (RLHF) based methods (a similar form as mentioned in equation 4). This alignment implies that any  $\mathbf{y} \sim \pi_{\text{safe}}^*(\cdot|\mathbf{x})$  should be safe regardless of the prompt  $\mathbf{x}$ , which implies mathematically that  $\mathbb{E}_{\mathbf{y} \sim \pi_{\text{safe}}^*} [R_s(\mathbf{x}, \mathbf{y})]$  is high. On the other hand, this also implies that  $\mathbb{E}_{\mathbf{y} \sim \pi_{\text{safe}}^*} [R_u(\mathbf{x}, \mathbf{y})]$  should be low which is the value function for unsafe reward. Next, to mathematically characterize the possibility to jailbreak a  $\pi_{\text{safe}}^*(\cdot|\mathbf{x})$ , we define a notion of the *safety net* of a safe LLM model as

$$\Delta_{\text{safety-net}}(\mathbf{x}) := \mathbb{E}_{\mathbf{y} \sim \pi_u^*} [R_u(\mathbf{x}, \mathbf{y})] - \mathbb{E}_{\mathbf{y} \sim \pi_{\text{algo}}^*} [R_u(\mathbf{x}, \mathbf{y})], \quad (7)$$

where  $\pi_u^*$  is the optimal model w.r.t. RLHF objective (defined in Equation 4) with unsafe reward  $R_u$ , and with  $\pi_0$  as the reference policy, where as  $\pi_{\text{algo}}^*$  is the optimal jailbreak RLHF aligned model for  $R_u$  with  $\pi_{\text{safe}}^*$  as the reference policy. It is important to note that it is much easier to **align** an unaligned reference model rather than a safety-aligned reference model. We note that  $\Delta_{\text{safety-net}}(\mathbf{x})$  is non-negative, and essentially trying to capture whether we can actually obtain an unsafe model  $\pi_{\text{algo}}^*$  from the safety aligned reference model  $\pi_{\text{safe}}^*$ . Additionally, we note that the value of  $\Delta_{\text{safety-net}}(\mathbf{x})$  for a good and safe LLM should be as high as possible, which means that it would be harder to do **jailbreaking via alignment** for the model. Hence, we establish an upper bound on the safety net in Theorem 1 as follows.

**Theorem 1** (On the Possibility of Jailbreaking Aligned Models). *For a safety aligned model  $\pi_{\text{safe}}^*$  (aligned with safe reward  $R_s$  via RLHF), and unsafe reward model  $R_u$ , it holds that*

$$\Delta_{\text{safety-net}}(\mathbf{x}) \leq \max_{\mathbf{y}} (R_u(\mathbf{x}, \mathbf{y}) - R_s(\mathbf{x}, \mathbf{y})) - \min_{\mathbf{y}} (R_u(\mathbf{x}, \mathbf{y}) - R_s(\mathbf{x}, \mathbf{y})). \quad (8)$$

The proof of Theorem 1 is provided in Appendix C. It is interesting to note that the higher the upper bound on the safety net, the better it is for a given safety-aligned model  $\pi_{\text{safe}}^*$ . The upper bound is precisely characterized by the difference between the unsafe reward  $R_u$  and safety reward  $R_s$ . If  $R_u = R_s$ , the safety net trivially reduces to zero. However, as the difference between  $R_u, R_s$  increases, the value of the safety net improves, indicating that it becomes harder to unalign the model's safety.

**To answer Q2**, we take motivation from the standard suboptimality definitions in the reinforcement learning literature (Agarwal et al., 2019) and define the following suboptimality gap for the proposed

{5}------------------------------------------------

LIAR approach as

$$\Delta_{\text{sub-gap}} = \mathbb{E}_{\mathbf{y} \sim \rho_u^*(\cdot|\mathbf{x})} [R_u(\mathbf{x}, \mathbf{y})] - \mathbb{E}_{\mathbf{y} \sim \rho_{\text{LIAR}}(\cdot|\mathbf{x})} [R_u(\mathbf{x}, \mathbf{y})]. \quad (9)$$

In the above expression,  $\rho_u^*$  denotes the optimal prompter which maximizes  $\max_{\rho} \mathbb{E}_{\mathbf{y} \sim \rho(\cdot|\mathbf{x})} [R_u(\mathbf{x}, \mathbf{y})]$  and  $\rho_{\text{LIAR}}(\cdot|\mathbf{x})$  denotes the distribution induced by the proposed LIAR procedure in equation 6. The goal is to show that the suboptimality gap is upper bounded, which would imply that our proposed approach LIAR is closer to the optimal prompter  $\rho_u^*$ . Before presenting the main result, we note that the induced  $\rho_{\text{LIAR}}(\mathbf{y}|\mathbf{x}) = \sum_{i=1}^N \binom{N}{i} \mathcal{F}(R_u(\mathbf{x}, \mathbf{y}))^{N-i} \rho_i(\mathbf{y})^i$ , where  $\mathcal{F}(R) = \mathbb{P}(R_u(\mathbf{x}, \mathbf{y}) < R)$ . With the above definitions in place, we present the second main result in Theorem 2.

**Theorem 2.** *For the proposed LIAR approach, it holds that*

$$\tilde{\Delta}_{\text{sub-gap}} \leq \frac{1}{N-1} \text{KL}(\rho_u^*, \rho_0), \quad (10)$$

where  $N$  are the number of samples in the best of  $N$  procedure we follow in LIAR and  $\text{KL}(\rho_u^*, \rho_0)$  is the KL divergence between  $\rho_u^*$  and  $\rho_0$ .

The proof of Theorem 2 is provided in Appendix D. We note that the upper bound in equation 10 depends upon the  $\text{KL}(\rho_u^*, \rho_0)$  and  $N$ . For a given reward model  $R_u$  and initial model  $\rho_0$ , the value of  $\text{KL}(\rho_u^*, \rho_0)$  is a constant. Therefore, the upper bound is controlled by the parameter  $N$  which is the number of samples in the best of  $N$  procedure in LIAR. Interestingly, this  $N$  is playing the role of parameter  $\beta$  in standard RLHF alignment, and states that with large enough  $N$ , we can make the suboptimality gap go towards zero. Experimentally, a large enough value of  $N$  is sufficient to generate successful attacks, as we observe in the next section.

## 5 EXPERIMENTS

In this section, we outline our experimental setup and present metrics that demonstrate the advantages discussed in the previous sections. Specifically, we show that our method can achieve a high Attack Success Rate (ASR) with increased queries (which corresponds to  $N$  in LIAR), all without incurring any additional training costs. Next, we investigate the use of various pretrained LLMs as our AdversarialLLM. Finally, we conduct ablation studies to analyze the effect of different LLM generation options on both the Attack Success Rate and Perplexity, shedding light on the key factors influencing our method's performance.

**Dataset:** Our experiments are conducted on the AdvBench dataset (Zou et al., 2023), which consists of instructions designed to elicit harmful behaviors in LLMs that have not undergone safety alignment. The dataset contains 312 samples in the training split, and 104 samples in the test split. All results are reported on the test split.

**Setup:** Our experimental setup uses a GPT-2 (Radford et al., 2019) adversarial model to generate sequences with a maximum length of 30 tokens. The model generates the next word using top- $k = 50$  sampling with a temperature of 1. For target LLMs, we select chat or instruct variants of publicly available and widely used models, including Vicuna-7b and 13b (v1.5) (Zheng et al., 2023), LLaMA-2, LLaMA-3, and LLaMA-3.1 (7b and 8b variants) (Touvron et al., 2023; AI@Meta, 2024a,b), Mistral-7b (v0.2) (Jiang et al., 2023), Falcon-7b (Penedo et al., 2023), and Pythia-12b (Biderman et al., 2023). These models were chosen for their accessibility and popularity in the research community. Our primary analysis focuses on Vicuna-7b, as it strikes a balance between performance and susceptibility to improvement, making it an ideal target for evaluating the impact of our method.

**Baselines:** We compare our method against recent state-of-the-art attacks. AdvPrompter (Paulus et al., 2024) is the most closely related, as it also trains an LLM to generate adversarial suffixes, resulting in relatively fast query times despite the additional setup time. Additionally, we include GCG (Zou et al., 2023), which is optimized specifically for attack success, and AutoDAN (Liu et al., 2023), which is designed to be less detectable by reducing perplexity. Compared to these baselines,

{6}------------------------------------------------

our method demonstrates significant improvements in time efficiency and perplexity, while maintaining competitive Attack Success Rates (ASR).

**Evaluations:** The three primary evaluation metrics we consider are Attack Success Rate (ASR), Perplexity, and Time-To-Attack (TTA). These three metrics respectively reflect the compute required to generate queries, the effectiveness of the attack, and how easy the attack is to detect. We refer to the process of generating a single adversarial suffix  $q$  as a "query".

*Attack Success Rate (ASR)*, denoted as  $\text{ASR}@k$ , measures the likelihood of an attack succeeding within  $k$  queries during testing. Specifically, an attack is considered successful if at least one of the  $k$  attempts bypasses the `TargetLLM`'s censorship mechanisms. This follows prior works (Paulus et al., 2024; Zou et al., 2023) though we extend it to larger values of  $k$ . [The  \$N\$  in the best of  \$N\$  formulation is equivalent to the  \$k\$  in  \$\text{ASR}@k\$](#) . Appendix F provides specific queries and responses to confirm that ASR reflects whether an attack was successful or not.

*Perplexity* assesses how natural the adversarial suffix appears. In response to early gradient-based jailbreaks like GCG, a similar metric was introduced as a simple but effective defense (Jain et al., 2023; Alon & Kamfonas, 2023). To get around this defense, it is beneficial for the adversarial suffix  $q$  to have low perplexity. We use the same perplexity metric as defined in Paulus et al. (2024), computed as  $\text{Perplexity}(q|x) = \exp\left(-\frac{1}{|q|} \sum_{i=1}^{|q|} \log \pi_\theta(q_i | x, q_{<i}}\right)$ .

*Time-To-Attack (TTA)* consists of two components: the initial setup time, which is a one-time cost, and the query time, which is incurred for each adversarial query generated. We report these values in Seconds (s), Minutes (m), or Hours (h). Table 5 also reports response time, which is the time required for the `TargetLLM` to generate the response to the adversarial query.

Table 1: Comparison of our method with other attack methods (GCG, AutoDAN, AdvPrompter) based on Attack Success Rate (ASR), Perplexity, and Time-to-Attack (TTA) across various `TargetLLMs` on the AdvBench dataset. ASR is presented as  $\text{ASR}@1$ ,  $\text{ASR}@10$ , and  $\text{ASR}@100$  based on the number of attempts. Importantly,  $\text{ASR}@1$  for other methods is comparable to  $\text{ASR}@100$  for our method due to its significantly faster TTA, enabling the generation of over 10,000 queries in under 15 minutes. TTA1 represents the total time required for both the initial setup of the attack and the generation of a single adversarial prompt. TTA100 extends this by multiplying the query time by 100, offering a clearer picture of the time needed to generate a large batch of adversarial prompts. To provide a better TTA comparison, **TTA1 for our method is computed for  $\text{ASR}@100$** , whereas TTA1 for all other methods are computed for  $\text{ASR}@1$ .

| TargetLLM | Attack | ASR@1/10/100 | Perplexity | TTA1/TTA100 |
|-|-|-|-|-|
| Vicuna-7b | GCG (individual) | 99.10/ - / - | 92471.12 | 16m/25h |
|  | AutoDAN (individual) | 92.70/ - / - | 83.17 | 15m/23h |
|  | AdvPrompter | 26.92/84.61/99.04 | 12.09 | 22h/22h |
|  | LIAR (ours) | 12.55/53.08/97.12 | <b>2.14</b> | <b>45s/14m</b> |
| Vicuna-13b | GCG (individual) | 95.40/ - / - | 94713.43 | 16m/25h |
|  | AutoDAN (individual) | 80.30/ - / - | 89.14 | 15m/23h |
|  | AdvPrompter | 19.50/67.50/ - | 15.91 | 22h/22h |
|  | LIAR (ours) | 0.94/31.35/79.81 | <b>2.12</b> | <b>45s/14m</b> |
| Llama2-7b | GCG (individual) | 23.70/ - / - | 97381.10 | 16m/25h |
|  | AutoDAN (individual) | 20.90/ - / - | 429.12 | 15m/23h |
|  | AdvPrompter | 1.00/7.70/ - | 86.80 | 22h/22h |
|  | LIAR (ours) | 0.65/2.31/3.85 | <b>2.13</b> | <b>45s/14m</b> |
| Mistral-7b | GCG (individual) | 100.0/ - / - | 81432.10 | 16m/25h |
|  | AutoDAN (individual) | 91.20/ - / - | 69.09 | 15m/23h |
|  | AdvPrompter | 54.30/96.10/ - | 41.60 | 22h/22h |
|  | LIAR (ours) | 34.25/73.94/96.15 | <b>2.12</b> | <b>45s/14m</b> |
| Falcon-7b | GCG (individual) | 100.0/ - / - | 94371.10 | 16m/25h |
|  | AutoDAN (individual) | 100.0/ - / - | 16.46 | 15m/23h |
|  | AdvPrompter | 78.80/98.10/ - | 10 | 22h/22h |
|  | LIAR (ours) | 71.78/99.33/100.0 | <b>2.07</b> | <b>45s/14m</b> |
| Pythia-7b | GCG (individual) | 100.0/ - / - | 107346.41 | 16m/25h |
|  | AutoDAN (individual) | 100.0/ - / - | 16.05 | 15m/23h |
|  | AdvPrompter | 80.30/100.0/ - | 7.16 | 22h/22h |
|  | LIAR (ours) | 75.96/99.81/100.0 | <b>2.17</b> | <b>45s/14m</b> |

{7}------------------------------------------------

### 5.1 ATTACKING TARGETLLM

In Table 1, our method demonstrates a significantly lower average perplexity (lower is better) than the second best method, AdvPrompter. This result is expected, as both AutoDAN and AdvPrompter use perplexity regularization in their training objectives, while our approach avoids any deviation from the pretrained AdversarialLLM, leading to more natural outputs. [This low perplexity challenges the effectiveness of perplexity-based jailbreak defenses](#). Additionally, our method offers much faster Time to Attack (TTA) compared to AdvPrompter, primarily because it does not require any training. The per-query time is also shorter, as GPT-2 is a considerably smaller model than LLaMA-2. When accounting for both setup and query times (TTA1), our method is significantly faster than comparable approaches, and this advantage increases further with larger query sets, as seen in TTA100. Although GCG and AutoDAN can generate a few adversarial examples before AdvPrompter finishes its training, AdvPrompter’s low per-query time allows for faster attacks on large sets of restricted prompts. However, LIAR consistently outperforms it in speed. GCG achieves the highest ASR@1, as it optimizes solely for adversarial success without perplexity regularization. When our method is allowed 100 attempts—which can be generated in just seconds—we achieve an attack success rate comparable to GCG. Given the significantly reduced overall TTA, this asymmetric ASR@ $k$  comparison becomes highly practical: our method can generate over 10,000 queries before GCG completes its first. Since an attacker only needs one successful query to jailbreak the TargetLLM, this fast TTA gives our approach a critical edge in real-world attacks and in evaluating defenses quickly.

### 5.2 CHOOSING AN ADVERSARIALLLM

To select the most suitable AdversarialLLM, we evaluated various LLM models based on their Attack Success Rate (ASR) and perplexity. Our criteria focused on models that had not undergone safety alignment training and were publicly accessible. As a result, many of the models we considered are smaller and faster compared to more recent, more powerful models. For our primary experiments, we utilized the smallest version of GPT2 (Radford et al., 2019), which has 124 million parameters and was trained on general web-scraped text. In Table 2, we compare this model with other candidates, including GPT2-PMC (Pande, 2024), GPT2-WikiText (Alon et al., 2022), GPT2-OpenInstruct (Wang & Ivison, 2023), Megatron-345M (Shoeybi et al., 2019), and TinyLlama-1.1B (Zhang et al., 2024). GPT2-PMC, GPT2-WikiText, and GPT2-OpenInstruct are all fine-tuned versions of the GPT-2 model, trained on the PubMed (Zhao et al., 2023), WikiText (Merity et al., 2016), and OpenInstruct datasets, respectively. Megatron-345M was trained on a diverse corpus including Wikipedia, news articles, stories, and web text, while TinyLlama was trained on the SlimPajama-627B (Soboleva et al., 2023) and StarCoder (Li et al., 2023b) datasets.

In Table 2, GPT2-PMC demonstrates a slight performance advantage over the other AdversarialLLMs when Vicuna-7B is used as the TargetLLM. However, this advantage diminishes when tested on other models, such as LLaMA-2, where TinyLlama slightly outperforms GPT2-PMC. TinyLlama, while achieving high ASR, has the longest query time due to its size (1.1 billion parameters), whereas GPT2 maintains near-median ASR and perplexity compared to the other models. Ultimately, we selected GPT2 as our primary AdversarialLLM because of its

Table 2: Query time, Attack Success Rate (ASR), and Perplexity on AdvBench dataset test split for different AdversarialLLM using Vicuna-7b as a TargetLLM. Additional results for various AdversarialLLMs are provided in Appendix E.1.

| AdversarialLLM | Query Time | ASR@1 | ASR@10 | ASR@100 | Perplexity |
|-|-|-|-|-|-|
| TinyLlama | 0.092s | 6.93 | 36.92 | 89.42 | 5.03 |
| Megatron | 0.058s | 9.46 | 49.52 | 95.19 | 1.67 |
| GPT2-WikiText | 0.028s | 8.06 | 37.98 | 84.62 | 1.55 |
| GPT2 | 0.033s | 12.55 | 53.08 | 97.12 | 2.11 |
| GPT2-OpenInstruct | 0.030s | 15.18 | 56.15 | 95.19 | 1.30 |
| GPT2-PMC | 0.029s | 19.68 | 75.58 | 99.04 | 1.32 |

{8}------------------------------------------------

Table 3: Ablation of temperature on a Vicuna-7b TargetLLM using a GPT2 AdversarialLLM.

| Temperature | ASR@1 | ASR@10 | ASR@100 | Perplexity |
|-|-|-|-|-|
| 10 | 5.77 | 26.25 | 66.35 | 2.96 |
| 4 | 6.59 | 30.00 | 70.19 | 2.86 |
| 2 | 7.96 | 37.69 | 81.73 | 2.71 |
| 1 (default) | 12.19 | 54.52 | 95.19 | 2.14 |
| 0.9 | 13.37 | 56.73 | 96.15 | 2.01 |
| 0.8 | 13.65 | 59.23 | 97.12 | 1.88 |
| 0.6 | 15.63 | 62.31 | 99.04 | 1.69 |
| 0.4 | 16.97 | 63.94 | 98.08 | 1.45 |
| 0.2 | 18.21 | 61.35 | 98.08 | 1.38 |
| 0.1 | 18.59 | 52.40 | 84.62 | 1.37 |

consistent performance across multiple TargetLLMs and its foundational nature, as many of the other models we considered are fine-tuned variants of GPT2. This balance of performance, speed, and accessibility makes GPT2 a practical choice for our method.

### 5.3 ABLATIONS

We have already shown that different AdversarialLLM models have varying effectiveness in Table 2. However, there are other ways of modifying the diversity of the AdversarialLLM, specifically in varying the generation parameters such as temperature or query length.

**Impact of varying the temperature.** Table 3 shows the impact of varying the temperature of the generated  $q$  of the AdversarialLLM: higher temperature results in higher “creativity”, i.e. the probability of unlikely next-word predictions is increased. Counter-intuitively, reducing the temperature and preferring the more likely next-word predictions results in higher ASR@1. This suggests that the most probable next-word prediction has a higher chance of a successful jailbreak. However, for higher  $k$  such as  $k = 10$  or  $k = 100$ , we see the importance of diversity in the generated queries. Specifically, for ASR@100, we see that the success rate peaks around temperature of 0.6, with lower temperatures reducing the diversity of the queries and thereby reducing the ASR@100. Additionally, we observe increased perplexity with increased temperature, which is to be expected as perplexity measures the likeliness of the query and higher temperature produces more unlikely queries.

**Impact of query length.** The length of  $q$  is evaluated in Table 4. Increasing the length of  $q$  results in longer query times, however even doubling the length has a query time shorter than AdvPrompter. When changing the length of  $q$ , and important fact to consider is that longer  $q$  have a higher chance of prompt-drift, where  $[x, q]$  may be asking for content far from  $x$  on its own. This is a limitation of the keyword matching aspect of the ASR metric being used. We see ASR@1 improve with  $q$  length, but ASR@10 peaks for  $q$  length 48 while ASR@100 peaks for  $q$  length 30. This suggests that longer  $q$  length may result in decreased diversity, and as shorter  $q$  lengths are preferred to reduce prompt-drift, 30 is a reasonable prompt length.

Table 4: Ablation of  $q$  length on a Vicuna-7b TargetLLM using a GPT2 AdversarialLLM.

| Length | Query Time | ASR@1 | ASR@10 | ASR@100 | Perplexity |
|-|-|-|-|-|-|
| 8 | 0.009s | 3.74 | 22.12 | 72.12 | 4.50 |
| 16 | 0.020s | 6.00 | 34.62 | 87.50 | 2.60 |
| 30 (default) | 0.033s | 7.80 | 42.40 | 96.15 | 2.10 |
| 48 | 0.047s | 9.06 | 45.67 | 94.23 | 1.91 |
| 64 | 0.080s | 9.11 | 42.88 | 93.27 | 1.83 |

{9}------------------------------------------------

Table 5: Ablation of  $y$  length on a Vicuna-7b TargetLLM using a GPT2 AdversarialLLM.

| Length | Response Time | ASR@1 | ASR@10 | ASR@100 | Perplexity |
|-|-|-|-|-|-|
| 10 | 0.084s | 8.12 | 42.88 | 93.27 | 2.16 |
| 20 | 0.154s | 7.85 | 42.79 | 90.38 | 2.07 |
| 32 (default) | 0.192s | 7.80 | 42.40 | 96.15 | 2.14 |
| 50 | 0.376s | 7.98 | 40.96 | 89.42 | 2.11 |
| 100 | 0.768s | 7.94 | 40.48 | 90.38 | 2.08 |
| 150 | 1.569s | 7.28 | 39.23 | 87.50 | 2.13 |

In our experiment setup, we report ASR based on the first 30 tokens generated by the TargetLLM instead of the more standard 150 TargetLLM tokens (Paulus et al., 2024). Reducing the number of TargetLLM tokens generated significantly reduces the compute required to run experiments, as in our setup the TargetLLM is much larger than the AdversarialLLM, and is what our method spends the most time processing. Table 5 shows the impact of  $y$  length on ASR. From other work’s setting of generating 150 tokens, our reduction to generating just 32 TargetLLM tokens decreases TargetLLM compute time by an order of magnitude. Additionally, ASR is pretty consistent across different  $y$  lengths. Generating fewer  $y$  tokens does result in a slightly lower chance of an unsuccessful attack keyword being present resulting in a higher ASR. However, this difference is consistent across  $k$  and is relatively small, making the tradeoff in compute worthwhile.

## 6 CONCLUSION

In summary, we propose a straightforward jailbreak method that is not only fast and avoids the need for additional training, but is also difficult to detect using traditional metrics such as perplexity. We have provided both theoretical justification for the efficacy of our method and empirical comparisons with similar recent approaches, demonstrating its effectiveness. The ability to efficiently navigate the space of the TargetLLM to elicit harmful responses hinges, in part, on the diversity and creativity of the generated attacks. Defending against these low perplexity attacks is a challenge, as it is not yet clear whether alignment can fully avoid providing harmful responses.

## REFERENCES

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.
- Alekch Agarwal, Nan Jiang, Sham M Kakade, and Wen Sun. Reinforcement learning: Theory and algorithms. *CS Dept., UW Seattle, Seattle, WA, USA, Tech. Rep.*, 32:96, 2019.
- AI@Meta. Llama 3.1 model card. 2024a. URL [https://github.com/meta-llama/llama-models/blob/main/models/llama3\\_1/MODEL\\_CARD.md](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md).
- AI@Meta. Llama 3 model card. 2024b. URL [https://github.com/meta-llama/llama3/blob/main/MODEL\\_CARD.md](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md).
- Gabriel Alon and Michael Kamfonas. Detecting language model attacks with perplexity. *arXiv preprint arXiv:2308.14132*, 2023.
- Uri Alon, Frank Xu, Junxian He, Sudipta Sengupta, Dan Roth, and Graham Neubig. Neuro-symbolic language modeling with automaton-augmented retrieval. In *International Conference on Machine Learning*, pp. 468–485. PMLR, 2022.
- Afra Amini, Tim Vieira, and Ryan Cotterell. Variational best-of-n alignment. *arXiv preprint arXiv:2407.06057*, 2024.

 Rest of paper (reference and Appendix) is removed.