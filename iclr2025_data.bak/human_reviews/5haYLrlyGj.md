## Human Reviewer 1

### Summary
This paper proposes a framework utilizing multiple draft models simultaneously for large language model (LLM) speculative decoding. Unlike traditional speculative decoding methods that employ a single draft model, this approach leverages a pool of draft models to enhance performance in varied domains. The paper argues that draft models trained on specific datasets may underperform in out-of-domain tasks. To address this, different draft models are utilized as candidate drafts for inference across various domains. The framework incorporates multi-armed bandit algorithms from recommendation systems to select the optimal draft model at each decoding step. Experiments on both black-box (e.g., independent draft models) and white-box (e.g., Medusa, Eagle, etc.) approaches demonstrate that the proposed framework offers superior inference speed compared to single-draft models.

### Strengths
1.	The approach of using multiple draft models to enhance speculative decoding across different scenarios is modest.

2.	Experimental results validate the effectiveness of the proposed framework, and the ablation study highlights the superiority of the BE reward.

3.	The writing is clear and concise.

### Weaknesses
1.	An important baseline is missing: training a single draft model on all specified datasets. Given the claim that draft models trained on different domain data improve overall performance, it is crucial to show the performance of a single draft model trained on the same datasets.

### Questions
1.	The appendix suggests that the proposed framework can achieve further enhancement in batched inference settings. However, previous works, such as Eagle, indicate that the speedup ratio declines as the batch size increases. Could you provide more details on how you arrived at this conclusion?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper presents MetaSD, an approach that integrates multiple specialized drafters into Speculative Decoding (SD) and employs multi-armed bandit sampling to dynamically select the optimal drafter. The authors conducted experiments across black-box and white-box SD methods, validating the superiority of the proposed method over traditional single-drafter approaches such as Lookahead and Eagle.

### Strengths
1. The exploration of this work towards Speculative Decoding with multiple specilaized drafters is meaningful and offers potential insights for the academic community. The standard single-drafter approach, typically generalized for natural language tasks, may not be optimal for domain-specific applications, such as translation. This work makes investigations to specilaized drafters across various domains such as QA, Translation, Math, and Summarization, etc, which demonstrates interesting and meaningful findings.
2. The authors conduct comprehensive experiments with both black-box and white-box SD methods, validating the effectiveness of MetaSD across diverse input domains. MetaSD achieves a promising 1.6x-3.7x speedup across these tested scenarios. The experimental settings are illustrated in detail.
3. The paper also provides an in-depth analysis of various impacts of MetaSD, addressing factors such as the switching cost of drafters, memory bandwidth bound, and KV cache re-calculation. These explanations effectively address some of my initial concerns.

### Weaknesses
1. **Extra computation overhead of MetaSD:** Unlike single-drafter SD methods such as Eagle, MetaSD leverages multiple specialized drafters for enhanced adaptation across various domains. While this approach shows promise, it introduces additional training and inference overhead that scales linearly with the number of drafters. Although the authors discuss certain aspects, like memory bandwidth limitations, further comparisons and quantitative data would clarify the computational overhead introduced by MetaSD. Specific metrics, such as MetaSD's training carbon footprint and memory bandwidth requirements during inference, should be included. Additionally, using multiple drafters increases serving complexity, particularly for multi-GPU environments.
2. **Lack of out-of-domain experiments**: In the main results, the authors utilize five specialized drafters that are fine-tuned on Code, Translation, Summarization, QA, and Math. Then, the MetaSD is evaluated on these five tasks, which can be regarded as in-domain tasks. For these tasks, the motivation of using Multi-armed bandit (MAB) is not convincing since drafters can be directly selected by estimating the similarity between the training and test data before inference. The strength of MAB would be more evident in out-of-domain scenarios, where selecting the optimal drafter is less straightforward. However, this experimental setting is missing in the work.
3. **Theoretical clarity**: Some definitions and theoretical statements in the paper lack clarity. For instance, in Line 214, the proof for the equation $\mathbb{E}\left[r_{i, t}^{BE}\right]=\frac{1-\alpha_i^{N_{\max }}}{N_{\max }\left(1-\alpha_i\right)} \mathbb{E}\left[r_{i, t}^{BD}\right]$ is missing. Similarly, Equation (6) in Line 1288, where the authors state $\mathbb{E}\left[N_{acc}(i, t)\right]=\frac{\alpha_i-\alpha_i^{N_{\max }+1}}{1-\alpha_i}$, appears to deviate from Equation (1) in [1]. Detailed explanations of these theoretical elements are necessary to prevent misinterpretation.
4. **Switching costs evaluation**: MetaSD needs to switch drafters during inference for optimal performance, which adds additional costs, such as the re-computation of drafting KV cache. To mitigate this, the authors propose Algorithm 4 to decrease the switching frequency, which first eliminates sub-optimal drafters as quickly as possible and then exclusively selects this drafter for the remaining rounds. Considering this, the total switching times of MetaSD during inference should be reported to offer readers an overall understanding of its extra KV re-computation cost. For example, the switching times between drafters in Table 3 and Table 4.

[1] Fast Inference from Transformers via Speculative Decoding. Leviathan et.al. ICML 2023.

### Questions
Most of my primary concerns are outlined in the weaknesses section above. Here are some additional, minor concerns:

- In Figure 1, the authors emphasize the use of the KV cache across different drafters. Could the authors clarify if they propose an efficient strategy to avoid re-calculating the KV cache or if they simply re-compute the KV cache for previous contexts upon drafter switching?

- The results of the SpS baseline should be included in Table 3, and the results of Eagle should be reported in Table 4.

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper explored the problem of specializing and choosing draft models to accelerate speculative decoding. The task of routing a proper draft model can be seen as a multi-armed bandit problem; the author developed and experimented MetaSD-UCB method based on the existing UCB algorithm to address the problem.

### Strengths
- The idea of applying the UCB algorithm to speculative decoding is novel.
- The paper is overall well-presented.

### Weaknesses
- The evaluations do not accurately reflect real-world use cases. The primary goal of MetaSD-UCB is to route inputs to the appropriate draft model. However, during the experiments, the same prompt template is applied across all instances within a dataset, making it trivial to differentiate between datasets. In real-world scenarios, prompts can vary significantly from one query to another, which makes the multi-armed bandit problem much more complex than what is represented in the evaluations.
- The effectiveness of the multi-armed bandit approach is limited. Given that the datasets used in the experiments are distinctly different and well-defined, a rule-based system or a simpler machine learning model could easily achieve high accuracy in selecting the appropriate draft model, often with minimal latency compared to using the draft model itself. In contrast, the proposed method requires executing a speculative decoding step with each draft model, which increases the number of tokens processed by the target model fivefold during the initial step—an expensive operation, particularly with tree attention. Despite this, the average accuracy achieved in the experiments for model selection is below 80 percent.
- There is a lack of experiments assessing throughput. Given that the proposed method utilizes an ensemble of draft models, it is more likely to be deployed in a distributed system rather than on a personal computer or mobile device. As a result, throughput should be prioritized as a key evaluation metric over latency. Even for latency evaluation, a batch size of 1 is not practical.
- Problem with machine translation datasets. Vicuna, as well as LLaMA 2, are not multilingual language models and are not designed for machine translation tasks involving languages such as Chinese and Japanese.
- The training datasets are the same as the evaluation dataset. 
- The vertical spacing on the last page is very small compared to other pages.

### Questions
1. Can you include an evaluation of Eagle without specialization on the same hardware?
2. Can you fine-tune a draft model on all the datasets experimented to show if it is necessary to have a specialized model?

### Soundness
1

### Presentation
3

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper introduces a simple framework, termed MetaSD, that incorporates multiple drafters into the speculative decoding process to address via multi-armed bandit sampling to dynamically allocate computational resources across various drafters, thereby improving overall generation performance.

### Strengths
The idea of this paper is neat.

The figures in this paper are clear and good. 

The experiments are somehow satisfied because the datasets are randomly shuffled to create a non-stationary environment.

### Weaknesses
The definition of the acceptance rate is not clear and confusing. The authors state that “we relax this assumption and consider a more general scenario where the acceptance rate for each token follows a stationary distribution with mean $\alpha$” in Line 215. This means that the acceptance rate is in fact a **random variable**. However, in all the calculations in the manuscript, the authors just replace the random variable with its mean value. 

Even if we treat the acceptance rates as real numbers,  the proof of Theorem 1 is not correct. In Lemma 4,  it is not clear why the expectation of $r^{BD}$ is equal to the acceptance rate. To be specific, in the definition of the block divergence reward, the total variation is conditioned on the prefix $x^{1:l(t)+j}$. We note that $x^{l(t)+1:l(t)+j}$ is the random variables generated by the draft model, not the pre-trained model. When taking expectation, these random variables take the distributions of the **draft models**. However, Theorem 1 of [1] shows that the average acceptance is equal to the expectation of $r^{BD}$ when $x^{l(t)+1:l(t)+j}$ takes the distribution of the **pre-trained model**.

The formulation of the bandit problem is incomplete, and the statement and the proof of Theorem 2 is inappropriate. Concretely, the proposed bandit problem should be considered in the probability space induced by the randomness of the pre-trained models and the draft models. This implies that the so-called total budget $B$ is indeed a **random variable** (in fact a stopping time). Thus, Theorem 2 states the results conditioned on the budget $B$, and **all the proofs should be built on the posterior distribution given $B$**. However, the current proof does not consider this.

While the paper emphasizes the design of the BD reward is novel, it does not necessarily align with the goal: a lower BD regret cannot indicate fewer rounds are needed.  According to Lemma 6 in the appendix, the performance of an algorithm under different regret/reward definitions, i.e., BD and number of rounds, can be different. From this perspective, the theoretical upper bound guarantee of the proposed algorithm is diminished, as the drafter with a better designed reward may not necessarily perform better in practice.
Some procedures in the proof are not correctly justified.

Line 1586: the equality is wrong. $\mathbb{E}[\tau^u(\pi^u, B)]$ can be much greater than $\mathbb{E}[\tau(\pi^{i^\star}, B)]$.

Line 1695: this inequality is wrong. The confidence radius never shrinks as $t$ increases.

In Algorithm 2, the hyper-parameter $\beta$ is chosen empirically according to Line 1631. However, $\beta$ is not reflected in the regret bound in some Theorems (with some specified $\beta$ in the others), as well as in the corresponding analyses.
In the bandits literature, the mean gaps $\{\Delta_i\}_{i=1}^K$ are critical, because they measure the hardness of the given problem instance. In this paper, some quantities involving the mean gaps are hidden in the constants (which are independent of $B$). While this may be acceptable for the asymptotic behavior where $B\to\infty$, they can be important in the finite $B$ case (which is always true in practice).

Minors:

It would be great to give a formal mathematical definition of $B$ and the stopping time where the algorithm stops. In addition, it would be better to consistently term $B$ as the "target sequence length", as "budget" refers to the time horizon in the bandits literature.

 
Overall, the formulation of the bandits problem is incomplete and the application of the existing bandits algorithms to speculative decoding is straightforward, given the previous literature. While the paper argues that the proposed BD reward is novel, a justification of its property (e.g., expectation) is clearly missing. In addition, an algorithm with less BD regret does not indicate better performance (at least theoretically). This hinders the theoretical contribution of the paper.

[1] M. Yin et al, A Theoretical Perspective for Speculative Decoding Algorithm

### Questions
1) why exp3 is significantly worse than ucb, even similar to the result of rand?

2) In Tables 3 and 4, the author why not provide the original eagle's result or train a general eagle with all the mixed data? 

3) The 5 tasks are much less than the real setting, the author should study the scaling law of task numbers for metasd.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4