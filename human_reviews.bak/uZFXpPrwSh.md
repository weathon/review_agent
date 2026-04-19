# Zero-shot Model-based Reinforcement Learning using Large Language Models

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 5

## Abstract
The emerging zero-shot capabilities of Large Language Models (LLMs) have led to their applications in areas extending well beyond natural language processing tasks. 
In reinforcement learning, while LLMs have been extensively used in text-based environments, their integration with continuous state spaces remains understudied. 
In this paper, we investigate how pre-trained LLMs can be leveraged to predict in context the dynamics of continuous Markov decision processes. 
We identify handling multivariate data and incorporating the control signal as key challenges that limit the potential of LLMs' deployment in this setup and propose Disentangled In-Context Learning (DICL) to address them.
We present proof-of-concept applications in two reinforcement learning settings: model-based policy evaluation and data-augmented off-policy reinforcement learning, supported by theoretical analysis of the proposed methods.
Our experiments further demonstrate that our approach produces well-calibrated uncertainty estimates. We release the code at https://github.com/abenechehab/dicl.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper investigates zero-shot reinforcement learning (RL) for continuous control tasks using large language models (LLMs) and introduces "Disentangled In-Context Learning" (DICL) to improve LLMs' handling of continuous MDPs. This approach separates state and action features to better utilize in-context learning, with experiments validating its efficacy.

### Strengths
1. The authors proposed an innovative integration of LLMs into continuous reinforcement learning tasks.
2. Clear theoretical foundation supporting the DICL framework is shown.
3. The authors conduct extensive experiments to verify the effectiveness of the method across diverse RL scenarios.
4. The authors also did insightful analysis of zero-shot capabilities in complex MDPs.

### Weaknesses
Unclear experimental settings. The authors mentioned that they used 3 Llama3 models for the experiments, but all the experiment figures only show one result without stating which model is used. Also, an ablation study about the effect of different LLMs is missing.

### Questions
You mentioned that you change the update step of SAC to accommodate LLM's requirements. Is this unfair? Have you compared with a baseline SAC with update step 1?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to leverage large language models (LLMs) to achieve model-based reinforcement learning (MBRL). Specifically, it explores using the in-context learning (ICL) capability of LLMs to autoregressively predict the next state and reward, denoted as $<s_t, r_t>$, based on a given prior states and actions.

The authors' first approach, vanilla ICL (vICL), simplifies the objective by predicting $s_t$ solely based on prior states $s_{1:t-1}$, without considering the action. The second approach, DICL, uses Principal Component Analysis (PCA) to map the state-action vector to a latent space, where ICL is applied to the latent vectors. Additionally, they apply their LLM-based dynamics learner to augment the replay buffer of the Soft Actor-Critic (SAC) algorithm.

Another important contribution of this work is demonstrating the bound on the difference between true dynamics and LLM-based dynamics under a multi-branch rollout setting.

### Strengths
(please also see the summary)
1. This work systematically studies LLM-based MBRL.  
2. The method is clearly presented and easy to follow.

### Weaknesses
The “zero-shot” claim is potentially misleading. If "zero-shot" is defined at the trajectory level, it is true that no trajectory-level examples were shown to the LLM during prediction. However, as shown in Section 4 (theoretical analysis), it appears necessary to use true dynamics to predict the transition and reward for steps $t < T$. These transitions, such as from $<s_{t-1}, a_{t-1}>$ to $s_t$, effectively serve as state-level few-shot examples. In my understanding, a true “zero-shot” setting would require that all previous transitions be predicted by the LLM itself autoregressively.

Another related concern is that the experimental setup is difficult to understand. For instance, Figure 3b was unclear to me. Given that the authors claim their method is zero-shot, I was unsure why, in the zero-shot setting, the agreement between the ground truth and the LLM-based method improves over time significantly. Wouldn’t error accumulation or distributional shift cause the LLM’s performance to degrade as more time steps are taken? After reviewing the theoretical analysis section, it seems likely that the result in Figure 3b is due to true dynamics being incorporated into the predictions, which was not mentioned. Please let me know if my understanding is inaccurate.

Finally, the ICL of LLM leverages the whole information in the previous context to predict the next transition. However, the MLP method can only accept the information of previous step as the input. The comparison is somehow unfair.

### Questions
n/a

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper explores the utilization of LLMs in the realm of RL, specifically targeting continuous state spaces that have been understudied in the context of LLM integration. It introduces a novel approach termed Disentangled In-Context Learning (DICL), designed to leverage pre-trained LLMs for predicting the dynamics of continuous Markov decision processes. The paper is supported by theoretical analysis and extensive experiments, demonstrating the potential of LLMs in model-based policy evaluation and data-augmented off-policy RL, and shows that LLMs can produce well-calibrated uncertainty estimates.

### Strengths
1. The paper presents a method to integrate state dimension interdependence and action information into in-context trajectories within RL environments, enhancing the applicability of LLMs in continuous state spaces.
2. It provides a theoretical analysis of the policy evaluation algorithm resulting from multi-branch rollouts with LLM-based dynamics models, leading to a novel return bound that enhances understanding in this area.
3. The paper offers empirical evidence supporting the benefits of LLM modeling in two RL applications: policy evaluation and data-augmented offline RL, showcasing the practicality of the proposed methods.
4. It demonstrates that LLMs can act as reliable uncertainty estimators, a desirable trait for MBRL algorithms.

### Weaknesses
1. The paper does not extensively discuss how the proposed method generalizes across different environments or tasks, that is, more discussion about the application of this method is needed.
2. While DICL simplifies certain aspects of RL, the integration of actions and the handling of multivariate data present ongoing challenges. More discussion about the introduced aspects of the DICL is needed.
3. The experiments are somewhat simplistic, and it would be worthwhile to conduct more in-depth analyses, such as discussing when they become ineffective and more results will be beneficial.
4. The writing style of the paper is a bit convoluted, making it less fluent to read.

### Questions
Why does Principal Component Analysis (PCA) decouple features, isn’t it primarily for feature selection? If it is for feature selection, are there any special constraints needed for article-related scenarios?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a new approach Disentangled In-Context Learning (DICL) to generalize LLM-based in-context learning to the domain of continuous-state-space reinforcement learning. This paper then analyzed the theoretical properties of this new DICL approach, and reports empirical experiments results from policy evaluation and data-augmented off-policy RL to the advantages in sample efficiency and performance of DICL in comparison to baseline methods.

### Strengths
This paper has good originality in that it innovatively proposes a novel DICL method to generalize ICL to continuous-state-space RL. This paper also includes solid mathematical derivations and proofs and detailed experiment results to support the claims in the paper.

### Weaknesses
There is a lot of room of improvement for the clarity, writing and presentation of this paper. Multiple places in the paper are not very clearly explained and the general theme of the paper is a little bit hard to follow in its writing. For example, throughout the whole paper, there is no explicit explanation or demonstrations on how exactly the LLM prompts for DICL are constructed. One or more concrete prompt examples would be very helpful in the paper to help readers understand the core technical details of the proposed DICL method.

This paper has very good potential, but its current form could benefit a lot from a systematic revision that improves its clarity and presentation.

### Questions
1. Typo - on Line 63, should it be ‘deferred’ instead of ‘differed’?
2. Typo - on Line 322, should it be ‘The goal is to improve …’?
3. In the DICL-SAC algorithm, what would be the optimal value for \alpha? Are there any intuitions for choosing the optimal \alpha value?

### Soundness
3

### Presentation
2

### Contribution
3
