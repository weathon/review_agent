Flexible switching between reasoning trajectories (i.e., thoughts switching) has significantly enhanced the reasoning capabilities of Large Reasoning Models (LRMs). However, existing models often switch excessively yet fail to sustain promising reasoning thoughts—–a phenomenon termed "under-thinking". While recent efforts suppress switching to mitigate this, such over-correction may discard valuable trajectories. To address this challenge, we propose **Steady Thought** (ST), a novel thought-level preference optimization framework. ST first segments model responses into thought sequences then guides the model to complete reasoning from these thoughts without further switching, generating coherent trajectories.Finally, ST performs thought-level preference optimization by treating the newly generated response as preferred and the original one as dis-preferred. Experiments across multiple models and datasets show that ST effectively mitigates under-thinking. It reduces output length by up to 39.3% while improving accuracy by up to 5.3%, with strong generalization. Further analysis confirms that ST leads to more rational switching and deeper exploration of solution thoughts.

## 1 Introduction

Nowadays, Large Reasoning Models (LRMs), such as DeepSeek-R1 (DeepSeek-AI et al., 2025) and GPT-o1 (OpenAI, 2024), have demonstrated strong reasoning capabilities across complex tasks. Their success stems from human-like slow thinking and reflective behaviors, which enable flexible switching between reasoning strategies—a process termed *thought switching* (Zeng et al., 2025; Muennighoff et al., 2025; Yang et al., 2025; Chen et al., 2025b). This adaptability fosters exploration of diverse reasoning paths, yielding more accurate and robust inference. However, recent studies have shown that LRMs tend to switch thoughts too frequently (Wang et al., 2025a; Ding et al., 2025; Chen et al., 2025a) and often fail to follow the promising thoughts—a phenomenon termed "under-thinking" (Wang et al., 2025c). Specifically, Figures 1a and 1b illustrate the initial emergence of correct thoughts during the models' reasoning processes. They reveals that the models often find a reliable thought early (which can lead to the correct answer), but still proceed with numerous additional thought switches. This issue roots from the lack of ability to recognize and commit to promising reasoning trajectories, which ultimately hinders the depth, coherence, and overall quality of reasoning. Furthermore, such excessive switching leads to substantial inefficiencies, resulting in unproductive exploration and a waste of computational resources. To mitigate the shallow reasoning caused by frequent thought switching, existing approaches primarily aim to suppress the switching behavior during inference. For example, some methods (Wang et al., 2025c;a; Ding et al., 2025) assume that the thought switching often starts with certain special tokens (e.g., "alternatively" or "wait") and suppressing under-thinking by lowering the probabilities of these tokens during decoding or reducing their rewards in training. On the other hand, Chen et al.

(2025a) takes a different approach by operating in the representation space: it steers the model's hidden states away from a learned "switching" vector and toward a "reasoning" vector, thereby suppressing switching behavior in a more structured manner. While effective in reducing excessive switching, these methods apply suppression globally, potentially limiting the model's flexibility to explore alternative reasoning thoughts when necessary. These limitations highlight the need for a 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Anonymous authors Paper under double-blind review

## Abstract

# Steadythought: Mitigating Llm Under-Thinking Via Thought-Level Preference Optimization

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

![1_image_0.png](1_image_0.png) 

![1_image_1.png](1_image_1.png) 

![1_image_2.png](1_image_2.png)

Tn
' T3
' Tn-1
'
Rejected Chosen T2
'
T3 Tn-1 Tn
Figure 1: (a) Rank of the first correct thought position for DeepSeek-R1-Distill-Qwen-1.5B on MATH500. (b) Rank of the first correct thought position for Qwen3-8B on AIME2024. (a) and (b) plot the percentile rank of the first correct thought in a thought sequence against the total number of thoughts segmented from each model response. (c) Overview of the ST framework, which operates in three core stages: **(1) Thought Segmentation:** Segmenting the thinking part of the response at a thought level based on changes in entropy. **(2) Thought Completion:** Guiding the model to continue writing for each thought without any thoughts switching by reducing the logits of words such as "wait" and "alternatively." **(3) Fine-Grained Preference Optimization:** Constructing preference optimization data pairs based on the correctness of the completion, and optimizing the model using the STPO algorithm. more selective mechanism—one that preserves the ability to explore new reasoning thoughts when the current trajectory is unpromising, while encouraging deeper commitment to a thought when it shows promise. To this end, we propose **Steady Thought** (ST), a novel thought-level preference optimization framework. ST aims to guide the model to consistently pursue high-potential reasoning thoughts while preserving its ability to explore necessary alternatives. As show in Figure 1c, ST operates in three stages: (1) **Thought Segmentation:** the model's thinking part wrapped in <think> tags of response is segmented into a sequence of thoughts by integrating step-level split and entropy-based thought switching detection. (2) **Thought Completion:** The target model is guided through logits control to generate new reasoning content based on a specific thought, thereby progressing toward the final answer. This process keeps the model on track and allows for the in-depth development of promising thoughts. (3) **Fine-Grained Preference Optimization:** By treating the newly generated content that leads to the correct answer as chosen, and the original as rejected, we perform a thought-level preference optimization named **Steady Thought Preference Optimization** (STPO) to encourages the model to favor reasoning thoughts that demonstrate more consistent progress toward the correct final answer, but at the same time, without detriment to the model's capability for preliminary exploration of various possible reasoning thoughts. Our main contributions are threefold:
- We analyzed the specific manifestations of the under-thinking phenomenon and formalized it as a preference optimization problem.

- We propose **Steady Thought** (ST), a novel thought-level preference optimization framework that encourages models to develop and stick to high-potential reasoning thoughts without compromising their flexibility to explore alternative reasoning trajectories.

- We validate the effectiveness of ST through extensive experiments across multiple models and reasoning benchmarks, with results demonstrating effective accuracy improvements of up to 5.3% and significant token reductions ranging from 19.0% to 39.3%.

## 2 Background 2.1 Problem Formulation

We formalize the reasoning process of a Large Reasoning Model (LRM) by modeling its response y to a question x as a trajectory of distinct thoughts:

$$,T_{2},\ldots,T_{n}),$$
$$\mathbf{y}=(T_{1})$$

y = (T1, T2*, . . . , T*n), (1)
108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 2.2 Preference Optimization Methods

DPO significantly advanced the alignment of language models with human preferences by leveraging the Bradley-Terry preference model to bypass the complex reward modeling and policy optimization of traditional reinforcement learning. However, DPO is known to be sensitive to length bias, as its objective can implicitly favor longer sequences that naturally attain higher loglikelihoods. SimPO addresses this limitation by introducing a length-normalized reward with a
- The **Commit Trajectory (**τc): The ideal trajectory where the model commits to and correctly completes the promising thought Ti. We denote this completed sequence as T
′
i.

- The **Switch Trajectory (**τs): The suboptimal trajectory observed in the data, where the model abandons Ti and switches to a new, often wasteful line of reasoning (Ti+1*, . . . , T*n).

Our objective is to align the model's policy πθ with the preference for the commit trajectory over the switch trajectory, i.e., τc ≻ τs. To formalize this, we define a latent **Steadiness Score** Sπ(τ |Pi), which quantifies the policy's inclination towards a specific trajectory τ given the prefix Pi. The preference τc ≻ τs implies that the score of the commit trajectory should be higher.

Following the Bradley-Terry model, the probability of this preference can be expressed through the difference in scores:
P(τc ≻ τs|Pi) = σ(Sπ(τc|Pi) − Sπ(τs|Pi)) (2)
where σ is the sigmoid function. This abstract scoring function provides a powerful new lens through which to view the problem. Crucially, it can be directly instantiated by the LRM's own logprobabilities. Preference optimization methods like Direct Preference Optimization (DPO) (Rafailov et al., 2024) and Simple Preference Optimization (SimPO) (Meng et al., 2024) allow us to directly optimize the policy πθ such that its implicit scoring function, where Sπ(τ |Pi) := log πθ(τ |Pi),
satisfies the desired preference relationship. This provides a principled way to train the model to be a more steadfast and effective reasoner. where each thought Ti represents a coherent segment of reasoning. After generating a prefix of i thoughts, denoted Pi = (x, T1*, . . . , T*i), the policy πθ defines a conditional probability distribution over all possible subsequent trajectories. The problem of **under-thinking** arises at a critical decision point. After the model generates a promising thought Ti (one that can lead to a correct answer), it faces a choice. Within the set of possible next trajectories, we are interested in two specific types:
162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 target margin, effectively mitigating length bias without requiring a reference model. Given the preference data D = {(x, yw, yl)}, where x is the input prompt, yw is the preferred response, and ylis the dispreferred response. SimPO optimizes the following objective:

$${\mathcal{L}}_{\mathrm{SimPO}}(\pi_{\theta})=-\mathbb{E}_{(x,y_{w},y_{l})\sim{\mathcal{D}}}\left[\log\sigma\left({\frac{\beta}{|y_{w}|}}\log\pi_{\theta}(y_{w}|x)-{\frac{\beta}{|y_{l}|}}\log\pi_{\theta}(y_{l}|x)-\gamma\right)\right].$$
 . (3)
where σ is the sigmoid function, with β as a temperature parameter controlling sensitivity to preference differences and γ as the target reward margin for the model to achieve.

## 3 Method

We propose Steady Thought (ST), a preference optimization framework that enables models to learn to stick to promising reasoning thoughts. As illustrated in Figure 1c, ST consists of three stages: 1) Thought Segmentation, 2) Thought Completion, and 3) Fine-Grained Preference Optimization, to teach models when to switch thoughts and when to persist.

## 3.1 Thought Segmentation

To segment the model's response into thought-level units, we employ entropy as a quantitative metric for assessing the model's confidence. The underlying principle is that a high entropy value indicates uncertainty in the model's predictive distribution over the next token, which typically occurs when it re-plans or explores a new reasoning trajectory. Conversely, low entropy signifies high confidence in the current output. Some recent studies indicate that tokens with high entropy often play a decisive role in determining reasoning trajectories (Wang et al., 2025b). Therefore, a sudden spike in entropy can serve as an effective signal for a thought switch. The entropy is calculated as follows:

$$({\mathfrak{I}})$$
$$H(P)=-\sum_{x}P(x)\log P(x)$$
$$(4)$$
$$\mathbf{y}=(T_{1},T_{2},\ldots,T_{n})$$
P(x) log P(x) (4)
where x is a token in the sequence, and P(x) represents the model's predicted probability for that token. Operationally, we first pre-segment the response into candidate steps using the common logical delimiter ".\n\n". We then compute the entropy for each token. A thought switch is identified if any of the initial tokens at the beginning of a candidate step exhibits entropy exceeding a predefined threshold, marking the start of a new thought. All subsequent steps are merged into this thought until the next threshold-exceeding step is encountered. This combines initial segmentation with entropy-based detection to partition each response y into a sequence of thoughts:
y = (T1, T2*, . . . , T*n) (5)
The granularity of this segmentation is controlled by the entropy threshold. We determined the optimal threshold through hyperparameter tuning to balance the detection of meaningful switches against over-segmentation. An excessively high threshold may miss subtle reasoning adjustments, while an overly low one could fragment coherent reasoning. For detailed experiments, see Section 4.4.3.

## 3.2 Thought Completion

The purpose of the second stage is to acquire a correct, self-generated completion of the thought previously segmented without any switches. We first predefine specific trigger words (e.g., "wait" and "alternatively") that signal a thought switch. During decoding, we then sharply decrease the logits for these words, effectively suppressing their selection by driving their prediction probability close to zero. For each thought Ti segmented in the previous stage, we apply the method to the model to continue solving the problem in conjunction with the corresponding question Q, yielding the completion:
T
′
i = Model(*Q, T*i), (6)
where Model is target model and Ti contains no thought switches and outputs a final answer. By evaluating the correctness of that final answer, we can determine whether the thought was a valid one. We discuss the consumption generated by this stage in the Appendix E

## 3.3 Fine-Grained Preference Optimization

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 A significant limitation of holistic preference optimization is that it treats entire reasoning chains as monolithic blocks. For complex problems, an incorrect response often contains a substantial sequence of correct initial reasoning. Rejecting such a response in its entirety discards the valuable correct portion and provides a noisy, unfocused learning signal to the model. To address this, our approach provides more granular supervision by focusing on the critical juncture where the reasoning diverges. We aim to teach the model to commit to a promising thought once it has been identified, rather than abandoning it. Specifically, an answer y can be decomposed into a sequence of thoughts y = (T1*, . . . , T*n). When we identify a promising thought Ti, the optimization then hinges on the trajectory the model takes from this point forward, conditioned on both the original question Q and the thought Tiitself.

The preference pair is constructed based on the continuations from this shared context:
Chosen Response : yw = T
′
i(The full, correct completion)
Rejected Response : yl = (Ti+1, Ti+2*, . . . , T*n) (The subsequent wasteful thoughts)
Our Steady Thought Preference Optimization (STPO) objective is to train the model to prefer the chosen response over the rejected one, conditioned on this specific, fine-grained context. Inspired by the reference-free and length-normalized objective of SimPO, the STPO loss is formulated as:

By the iterative tree and regular normalized objective of sum 0, the STG loss is formulated as: $\mathcal{L}_{\text{STPO}}(\pi_\theta)=-\mathbb{E}_{(Q,T_i,\mathbf{y}_w,\mathbf{y}_i)\sim\mathcal{D}}\left[\log\sigma\left(\frac{\beta}{|\mathbf{y}_w|}\log\pi_\theta(\mathbf{y}_w|Q,T_i)-\frac{\beta}{|\mathbf{y}_i|}\log\pi_\theta(\mathbf{y}_i|Q,T_i)-\gamma\right)\right]$. 
(7)
This formulation provides a crucial distinction from conventional preference tuning. The learning signal, embodied by the conditional log-probabilities log πθ(·|*Q, T*i), is applied directly at the point of divergence. It forces the model to learn not just what a good final answer looks like, but to recognize and commit to a promising intermediate thought. This targeted, thought-level supervision is key to effectively mitigating the model's tendency to "under-think" and abandon viable reasoning paths.

## 4 Experiments 4.1 Datasets

We selected the omni-math (Gao et al., 2024) dataset as the source of our training data. This dataset contains thousands of problems at the level of the International Mathematical Olympiad, which are categorized by difficulty. We sampled problems from various difficulty levels and used a target model (e.g., Qwen3-8B) to perform inference and generate responses. We evaluated our method on four datasets: (1) MATH-500 (Hendrycks et al., 2021): A math test set containing 500 problems with difficulty levels ranging from 1 to 5, covering multiple mathematical domains such as algebra, geometry, and number theory. (2) AIME 2024 (MAA Committees, 2024): A high-difficulty math competition dataset based on the American Invitational Mathematics Examination (AIME), consisting of 30 problems covering mathematical branches such as Algebra, Geometry, Number Theory, and Combinatorics. (3) GSM8K (Cobbe et al., 2021): A benchmark dataset featuring high-quality, linguistically diverse math word problems. It is used to evaluate the multi-step reasoning capability of language models. While based on simple arithmetic and basic algebra, solving these problems requires 2 to 8 sequential steps. The standardized test set consists of 1,319 problems. (4) LiveCode (Jain et al., 2024):A dataset for evaluating the code capabilities of LLMs, containing 400 problems collected from competitive programming websites like LeetCode, AtCoder, and Codeforces. We selected this dataset as an out-of-distribution (OOD) dataset to test the model's generalization ability after training.

## 4.2 Baseline And Metrics

We compare Steady Thought against three base models (DeepSeek-R1-Distill-Qwen1.5B (DeepSeek-AI et al., 2025), Qwen3-8B (Qwen Team, 2024) and DeepSeek-R1-Distill- Qwen-14B (DeepSeek-AI et al., 2025)) as well as three test-time efficiency methods, namely NoThink, NOWAIT and SEAL.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 4.3 Main Results

- NoThink: Ma et al. (2025) make the model skip the thinking process and output the response directly by adding the <think> tag after the </think> tag in the prompt.

- NOWAIT: Wang et al. (2025a) reduces the logits values of certain keywords that represent reflection (e.g., "wait") during decoding, making them almost impossible to output.

- SEAL: Chen et al. (2025a) first collects multiple responses and categorizes them into three types of thoughts: execution, reflection, and transition. A differential vector S is derived in the latent space between execution vectors and the combination of reflection/transition vectors. During decoding, the hidden states are modified as H˜ = H + α · S, biasing the hidden layer toward execution-oriented thoughts.

We report the accuracy and average token count performance for each task. We took the average of eight test runs for the AIME 2024 test set and two runs for the LiveCode test set. As shown in Table 1, the Steady Thought method performed well on three reasoning models of different architectures and sizes. It effectively reduced the number of tokens generated by the models while maintaining or even improving their accuracy. For example, on the DeepSeek-R1-Distill- Qwen-1.5B model, ST increased the average accuracy across four datasets by 1.9% while reducing the average token count by 24.9% compared to the base model. Similarly, on the Qwen3-8B model, it boosted the average accuracy by 3.12% and reduced the average token count by 23.6%. ST achieved a successful performance improvement of 2.52% on average for the DeepSeek-R1-Distill- Qwen-14B model, while simultaneously reducing the average output length by 17.3%. It is noteworthy that the LiveCode dataset serves as an OOD test set, as our models were trained exclusively on a mathematical training dataset. Despite this, ST still achieved positive results on LiveCode. For instance, it improved the Qwen-8B's accuracy by 5.3% and reduced its token count by 19.0%. Furthermore, it achieved similar gains on a larger 14B model, boosting its accuracy by 4.2% and reducing its output length by 14.2%. This suggests that the ST effectively teaches the model a more precise pattern of thought switching and retention, rather than simply memorizing the data, thereby improving its generalization ability. We have provided specific examples of the trajectories in Appendix B to demonstrate the effectiveness of ST.

| Method                                          | MATH-500   | AIME 2024       | GSM8K   | LiveCode        | Overall   |         |       |      |                |                |
|-------------------------------------------------|------------|-----------------|---------|-----------------|-----------|---------|-------|------|----------------|----------------|
| Acc(%)↑ Tokens↓                                 | Acc(%)↑    | Tokens↓ Acc(%)↑ | Tokens↓ | Acc(%)↑ Tokens↓ | Acc(%)↑   | Tokens↓ |       |      |                |                |
| DeepSeek-R1-Distill-Qwen-1.5B Vanilla 82.0 4385 | 27.5       | 11273           | 81.9    | 1448            | 30.3      | 9623    | 55.43 | 6682 |                |                |
| NoThink                                         | 65.8       | 749             | 8.7     | 3185            | 53.6      | 263     | 20.7  | 813  | 37.20 (-18.23) | 1252 (-81.3%)  |
| NOWAIT                                          | 80.6       | 2433            | 20.8    | 7000            | 66.1      | 2078    | 28.3  | 4927 | 48.95 (-6.48)  | 4109 (-38.5%)  |
| SEAL                                            | 82.6       | 3252            | 25.4    | 9120            | 79.7      | 860     | 29.5  | 7948 | 54.30 (-1.13)  | 5295 (-20.8%)  |
| Steady Thought                                  | 84.4       | 2809            | 31.2    | 8606            | 81.3      | 1254    | 32.4  | 7398 | 57.33 (+1.9)   | 5016 (-24.9 %) |
| Qwen3-8B Vanilla                                | 91.4       | 4724            | 62.1    | 10895           | 95.6      | 1759    | 71.8  | 7112 | 80.23          | 6122           |
| NoThink                                         | 85.2       | 933             | 25.8    | 3504            | 93.6      | 289     | 45.6  | 584  | 62.55 (-17.68) | 1327 (-78.3%)  |
| NOWAIT                                          | 61.0       | 13274           | 26.3    | 14333           | 73.3      | 12369   | 75.5  | 5226 | 59.03 (-21.20) | 11300 (+84.6%) |
| SEAL                                            | 92.2       | 4034            | 58.8    | 10372           | 95.9      | 1421    | 83.4  | 6414 | 82.58 (+2.35)  | 6940 (-8.4%)   |
| Steady Thought                                  | 94.4       | 2869            | 65.8    | 8742            | 96.1      | 862     | 77.1  | 5759 | 83.35 (+3.12)  | 4558 (-25.5 %) |
| DeepSeek-R1-Distill-Qwen-14 B Vanilla 93.6 3349 | 60.4       | 8974            | 94.8    | 894             | 70.1      | 6789    | 79.73 | 5001 |                |                |
| NoThink                                         | 41.7       | 824             | 27.1    | 3279            | 90.1      | 256     | 44.0  | 708  | 50.73 (-29.00) | 1266 (-74.7%)  |
| NOWAIT                                          | 75.6       | 3314            | 33.8    | 9431            | 86.3      | 936     | 64.3  | 5099 | 65.00 (-14.73) | 4695 (-6.1%)   |
| SEAL                                            | 92.6       | 3253            | 60.8    | 8831            | 94.7      | 880     | 75.1  | 6706 | 80.80 (+1.07)  | 4917 (-1.7%)   |
| Steady Thought                                  | 94.2       | 2455            | 65.4    | 7554            | 95.1      | 715     | 74.3  | 5825 | 82.25 (+2.52)  | 4137 (-17.3%)  |

## 4.4 In-Depth Analysis And Ablation Studies 4.4.1 Analysis Of In-Depth Exploration Ability

ST improves the model's capacity for in-depth exploration of promising thoughts. This effect is evident in two key metrics: First, the model's output becomes more concise, with a reduction in average output length. Second, the final thought—–the one leading to the definitive answer–—constitutes a significantly larger proportion of the overall response. This suggests that once the model identifies 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

![6_image_0.png](6_image_0.png) 
a promising thought with the potential to lead to the correct solution, it is better equipped to pursue that thought thoroughly instead of frequently switching its focus. By applying the methodology described in Section 3.1, we partitioned and quantified the thought processes from the model's responses both before and after ST. As Figure 2 illustrates, ST consistently produced shorter average outputs across the all of datasets. Furthermore, in most cases, the average number of thoughts generated by the model also decreases correspondingly. However, when smaller models tackle high-difficulty problems, they tend to increase the frequency of thought transitions to find the optimal solution. For example, when addressing the challenging AIME 2024 dataset, DeepSeek-R1-Distill-Qwen-1.5B generated a greater number of thoughts under the ST method compared to the base model. This increase led to improved accuracy and shorter overall response length. For less challenging problems, both the 1.5B and 8B models tended to produce fewer thoughts. Additionally, the final thought consistently accounted for a larger proportion of the total response. These experiments provide compelling evidence that the ST method significantly enhances the model's in-depth exploration capability.

## 4.4.2 Analysis Of Thinking Switching Ability

ST enhances the model's accuracy in determining when to switch its thoughts of reasoning, making its thought transitions more purposeful. This improvement is primarily evidenced by two observations: the model significantly reduces its total output length while maintaining high performance, and the proportion of correct intermediate thoughts decreases before the final answer is reached. In a thought chain, any correct intermediate thought that is subsequently abandoned and switched to another path constitutes an Invalid Switch. Thus, the number of correct intermediate thoughts is equal to the number of Invalid Switches. Consequently, we aim to demonstrate the reduction in the phenomenon of ineffective switching by showing the decrease in the proportion of correct thoughts after the model undergoes ST training. As shown in Table 2, we used the methods from Section 3.1 and 3.2 to calculate the proportion of correct thoughts preceding the final one, both before and after ST. On both models and across both datasets, the ST-trained model consistently achieved a lower rate of correct intermediate thoughts. This precisely demonstrates that the model's decision-making has become more precise—it reduces instances of unnecessary thought-switching, allowing for a more reasoned and thorough preliminary exploration. We further discuss the benefits that ST brings to the model through thought-level preference optimization in the Appendix C.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 As mentioned in Section 3.1, the entropy threshold setting directly influences the granularity of thought segmentation, which in turn affects the model's learning. An excessively high threshold leads to coarse-grained segmentation, potentially missing critical thought steps which are able to reach the correct answer. This also reduces the amount of data available for the model to learn from. Conversely, a threshold that is too low can split apart complete thoughts, making it difficult to identify a correct reasoning thought that should have been whole. We analyzed the segmentation results of the DeepSeek-R1-Distill-Qwen-1.5B based on different entropy threshold settings, as detailed in Table 3. From the table, we can see a trade-off in thought segmentation. A lower threshold, which leads to finer-grained segmentation, results in a drop in the proportion of correct thoughts. Conversely, a higher threshold creates coarser segments. While this improves the proportion of correct thoughts, it reduces the amount of available data for training. Ultimately, we found a threshold of 3.0 to be optimal for the DeepSeek-R1-Distill-Qwen-1.5B model which was used in our experiments, as it yielded the best overall performance. We provide threshold tuning results on more models and datasets in the appendix D. Table 3: Comparison of thought segmentation results under varying thresholds. NT: number of segmented thoughts. PCT: proportion of correct thoughts.

| Base model                    | Threshold    | ST Metric                       | MATH500   | AIME 2024   |      |       |
|-------------------------------|--------------|---------------------------------|-----------|-------------|------|-------|
| NT                            | PCT          | Acc(%)↑ Tokens↓ Acc(%)↑ Tokens↓ |           |             |      |       |
| 2.8                           | 20650 21.93% | 83.4                            | 3854      | 29.2        | 9252 |       |
| DeepSeek-R1-Distill-Qwen-1.5B | 3.0          | 10444 24.69%                    | 84.4      | 2809        | 31.2 | 8606  |
| 3.2                           | 4355         | 27.99%                          | 83.6      | 3657        | 28.3 | 10516 |

## 4.4.4 Analysis Of Different Training Method

Table 4: Comparison of different training method results.

| Training Method               | MATH500   | AIME 2024   |         |       |
|-------------------------------|-----------|-------------|---------|-------|
| Acc(%)↑                       | Tokens↓   | Acc(%)↑     | Tokens↓ |       |
| DeepSeek-R1-Distill-Qwen-1.5B | 82.2      | 4385        | 27.5    | 11273 |
| SFT                           | 80.4      | 2650        | 22.9    | 7169  |
| DPO                           | 82.6      | 4273        | 30.8    | 10701 |
| STPO                          | 84.4      | 2809        | 31.2    | 8608  |

We analyzed the impact of different training methods in ST's last stage (or phase) on model performance, and the results are shown in Table 4. In the fine-tuning approach, we used the x from the preference data pairs as the input and the chosen response as the output. This method tends to make the model memorize specific data rather than learn the underlying reasoning patterns we want it to acquire. Consequently, although it has learned the characteristics of "chosen": very short content, its performance on highly difficult or OOD data is subpar.

## 4.4.3 Analysis Of Different Entropy Threshold

Table 2: Percentage of correct thoughts (PCT) generated by the model before and after applying ST.

| Method                        | MATH500   | AIME 2024   |
|-------------------------------|-----------|-------------|
| PCT(%)↓                       | PCT(%)↓   |             |
| DeepSeek-R1-Distill-Qwen-1.5B | 54.90     | 14.50       |
| + Steady Thought              | 40.40     | 7.90        |
| Qwen3-8B                      | 73.18     | 45.20       |
| + Steady Thought              | 67.74     | 39.00       |

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 In contrast, we want the model to accomplish two things: to deeply explore promising solution thoughts while also effectively rejecting useless switching in its thought process. This dual-focus optimization capability is unique to preference optimization methods and cannot be achieved through simple fine-tuning alone. When constructing our preference data, the complete response that follows a specific "thought" is typically designated as the rejected part, while the completion of that same thought is the chosen part. This often results in the rejected portion being significantly longer than the chosen portion. Although Direct Preference Optimization (DPO) (Rafailov et al., 2024) is widely popular for its simplicity and efficiency, its training effectiveness is compromised by its sensitivity to the stark length differences between chosen and rejected responses. SimPO introduces "length-normalized rewards," which effectively eliminates the impact of these length differences, allowing the model to better learn the deep patterns embedded within the data. Drawing inspiration from SimPO, we contribute STPO, a novel preference optimization framework operating at the level of thoughts. As a result, compared to the other two methods, STPO not only effectively reduces the model's output length but also improves its performance.

## 5 Related Work 5.1 Over-Thinking And Under-Thinking

Recent studies have shown that while Large Language Models possess strong reasoning capabilities, they often expend excessive unnecessary resources—a phenomenon known as over-thinkingQu et al. (2025); Sui et al. (2025). This primarily manifests in two forms: first, the model's responses contain redundant information; second, it struggles to allocate appropriate computational budgets to questions of varying difficulty, with the latter being particularly evident in O1-like models.To address the first issue, existing solutions include both non-training and training methods. Non-training methods involve carefully designed prompts to limit output lengthXu et al. (2025); Nayab et al. (2025) or adjusting the model's output during decoding to enhance reasoning concisenessQiu et al. (2024); Zhang et al. (2025b). Training methods improve the model's ability to provide concise answers by fine-tuning on brief chain-of-thought (CoT) data or incorporating additional rewards/penalties for output length during reinforcement learning (RL) trainingKang et al. (2024); Aggarwal & Welleck (2025).For the second issue, most approaches employ fine-tuning or RL to enable models to dynamically switch between fast-thinking and slow-thinking modes during reasoning based on question difficulty, thereby improving the balance between reasoning accuracy and efficiencyZhang et al. (2025a); Lou et al. (2025).Under-thinking represents another form of resource wastage, primarily caused by LRMs frequently making ineffective switching thoughts during reasoning, preventing the model from fully developing promising reasoning thoughtsWang et al. (2025c). Existing methods focus on directly suppressing the model's reasoning thoughts switching at either the token or representation level to mitigate this phenomenonDing et al. (2025); Chen et al. (2025a); Wang et al. (2025a); Ding et al. (2025). In contrast, our approach considers that some switching is necessary, and instead alleviates under-thinking by enhancing the model's ability to maintain promising reasoning thoughts.

## 5.2 Preference Optimization

Preference optimization is a technique that adjusts AI model behavior through human feedback to make its outputs more accurate or better aligned with human values. For example, some Proximal Policy Optimization (PPO)-based methods optimize model performance by incorporating human feedback during the reinforcement learning phase to construct a reward model, thereby limiting the scope of policy updates Lightman et al. (2024); Luo et al. (2023). But the training process for PPO methods is highly complex and heavily reliant on the quality of human feedback data. Direct Preference Optimization (DPO) (Rafailov et al., 2024) method, which requires fewer training models and has a shorter process, is becoming increasingly popular. Many studies have extended DPO to different levels, such as the step level and token level, to optimize the model (Lu et al., 2024; Lai et al., 2024; Liu et al., 2024). However, DPO is not suitable for scenarios with significant length differences in preference pairs due to its sensitivity to data length. In contrast, Simple Preference Optimization (SimPO) (Meng et al., 2024) avoids the model from gaining rewards by exploiting length by introducing the average log probability as the optimization benchmark, thereby decoupling "quality" from "length". Inspired by SimPO, we have developed a finer-grained preference

## 6 Conclusion References

optimization framework, enabling the model to learn thought-level responses to mitigate the underthinking issue. This paper presents Steady Thought (ST), a novel preference optimization framework designed to mitigate the under-thinking problem in Large Reasoning Models. To address the issue of models frequently abandoning promising reasoning thoughts, ST introduces a structured pipeline: Thought Segmentation, Thought Completion, and Fine-Grained Preference Optimization, which guides models to switch thoughts judiciously and explore promising thoughts deeply. Experimental results on diverse models and tasks show that ST successfully mitigates under-thinking by reducing unnecessary switches, leading to more focused reasoning while maintaining or even enhancing performance. Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning, 2025. URL https://arxiv.org/abs/2503.04697.

Runjin Chen, Zhenyu Zhang, Junyuan Hong, Souvik Kundu, and Zhangyang Wang. Seal: Steerable reasoning calibration of large language models for free, 2025a. URL https://arxiv.org/ abs/2504.07986.

Zhipeng Chen, Yingqian Min, Beichen Zhang, Jie Chen, Jinhao Jiang, Daixuan Cheng, Wayne Xin Zhao, Zheng Liu, Xu Miao, Yang Lu, Lei Fang, Zhongyuan Wang, and Ji-Rong Wen. An empirical study on eliciting and improving r1-like reasoning models, 2025b. URL https: //arxiv.org/abs/2503.04548.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv. org/abs/2110.14168.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

Bowen Ding, Yuhan Chen, Futing Wang, Lingfeng Ming, and Tao Lin. Do thinking tokens help or trap? towards more efficient large reasoning model, 2025. URL https://arxiv.org/abs/ 2506.23840.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omnimath: A universal olympiad level mathematic benchmark for large language models, 2024. URL
https://arxiv.org/abs/2410.07985.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021.

URL https://arxiv.org/abs/2103.03874.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code, 2024. URL https://arxiv.org/abs/ 2403.07974.

Yu Kang, Xianghui Sun, Liangyu Chen, and Wei Zou. C3ot: Generating shorter chain-ofthought without compromising effectiveness, 2024. URL https://arxiv.org/abs/ 2412.11664.

Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-dpo: Stepwise preference optimization for long-chain reasoning of llms, 2024.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In The Twelfth International Conference on Learning Representations, 2024.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Aiwei Liu, Haoping Bai, Zhiyun Lu, Yanchao Sun, Xiang Kong, Simon Wang, Jiulong Shan, Albin Madappally Jose, Xiaojiang Liu, Lijie Wen, Philip S. Yu, and Meng Cao. Tis-dpo: Token-level importance sampling for direct preference optimization with estimated weights, 2024.

Chenwei Lou, Zewei Sun, Xinnian Liang, Meng Qu, Wei Shen, Wenqi Wang, Yuntao Li, Qingping Yang, and Shuangzhi Wu. Adacot: Pareto-optimal adaptive chain-of-thought triggering via reinforcement learning, 2025. URL https://arxiv.org/abs/2505.11896.

Zimu Lu, Aojun Zhou, Ke Wang, Houxing Ren, Weikang Shi, Junting Pan, Mingjie Zhan, and Hongsheng Li. Step-controlled dpo: Leveraging stepwise error for enhanced mathematical reasoning, 2024.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. *arXiv:2308.09583*, 2023.

Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning models can be effective without thinking, 2025. URL https://arxiv.org/abs/2504. 09858.

MAA Committees. AIME problems and solutions. https://artofproblemsolving.com/
wiki/index.php/AIME_Problems_and_Solutions, 2024. Accessed: 2025-09-23.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward, 2024. URL https://arxiv.org/abs/2405.14734.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori Hashimoto. s1: Simple test-time ` scaling, 2025. URL https://arxiv.org/abs/2501.19393.

Sania Nayab, Giulio Rossolini, Marco Simoni, Andrea Saracino, Giorgio Buttazzo, Nicolamaria Manes, and Fabrizio Giacomelli. Concise thoughts: Impact of output length on llm reasoning and cost, 2025. URL https://arxiv.org/abs/2407.19825.

OpenAI. Learning to reason with llms, September 2024. URL https://openai.com/index/
learning-to-reason-with-llms/.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Jiahao Qiu, Yifu Lu, Yifan Zeng, Jiacheng Guo, Jiayi Geng, Huazheng Wang, Kaixuan Huang, Yue Wu, and Mengdi Wang. Treebon: Enhancing inference-time alignment with speculative treesearch and best-of-n sampling, 2024. URL https://arxiv.org/abs/2410.16033.

Xiaoye Qu, Yafu Li, Zhaochen Su, Weigao Sun, Jianhao Yan, Dongrui Liu, Ganqu Cui, Daizong Liu, Shuxian Liang, Junxian He, Peng Li, Wei Wei, Jing Shao, Chaochao Lu, Yue Zhang, Xian- Sheng Hua, Bowen Zhou, and Yu Cheng. A survey of efficient reasoning for large reasoning models: Language, multimodality, and beyond, 2025. URL https://arxiv.org/abs/ 2503.21614.

Qwen Team. Qwen3: A series of large language models. https://qwenlm.github.io/
blog/qwen3/, 2024. Accessed: 2025-09-23.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Hanjie Chen, and Xia Hu. Stop overthinking: A survey on efficient reasoning for large language models, 2025. URL https://arxiv.org/abs/2503. 16419.

Chenlong Wang, Yuanning Feng, Dongping Chen, Zhaoyang Chu, Ranjay Krishna, and Tianyi Zhou. Wait, we don't need to "wait"! removing thinking tokens improves reasoning efficiency, 2025a. URL https://arxiv.org/abs/2506.08343.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning, 2025b. URL https://arxiv.org/abs/ 2506.01939.

Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian Yu, Juntao Li, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Thoughts are all over the place: On the underthinking of o1-like llms, 2025c. URL https://arxiv.org/ abs/2501.18585.

Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. Chain of draft: Thinking faster by writing less, 2025. URL https://arxiv.org/abs/2502.18600.

Shu Yang, Junchao Wu, Xin Chen, Yunze Xiao, Xinyi Yang, Derek F. Wong, and Di Wang.

Understanding aha moments: from external observations to internal mechanisms, 2025. URL https://arxiv.org/abs/2504.02956.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild, 2025. URL https://arxiv.org/abs/2503.18892.

Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. Adaptthink: Reasoning models can learn when to think, 2025a. URL https://arxiv.org/abs/2505.13417.

Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da Zheng, Huajun Chen, and Ningyu Zhang. Lightthinker: Thinking step-by-step compression, 2025b. URL https: //arxiv.org/abs/2502.15589.