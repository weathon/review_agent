000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

## 1 Introduction

The long chain-of-thought (LongCoT) capability is central to the recent breakthroughs achieved by large language models in complex reasoning tasks. However, the accompanying issue of "*underthinking*", where models exhibit shallow reasoning by frequently switching thoughts without sufficient exploration, limits both performance and token efficiency. To address this problem, we propose a simple yet effective reasoning strategy: the SmartSwitch inference framework. This framework can be easily integrated into any large language model as a plug-andplay solution, continuously monitoring the model's reasoning process to detect underthinking and guide it toward deeper exploration of promising but overlooked thoughts. Specifically, the perception module identifies points where thoughts switch and evaluates the potential of the preceding thought using an off-the-shelf process reward model (PRM). If a high-potential thought is found to be prematurely abandoned, the intervention module interrupts the ongoing inference, backtracks to the point before the switch, and inserts a "deepening prompt" to encourage further exploration along that promising path. Extensive experiments on challenging mathematical reasoning benchmarks demonstrate that our method significantly enhances the performance of various large language models of different sizes.

![0_image_0.png](0_image_0.png)

Figure 1: Qualitative and Quantitative illustration for the "underthinking problem". (a) presents an example with the underthinking phenomenon sampled from DeepSeek-R1 (Guo et al., 2025). The full response consists of 74 different thoughts, each with a relatively short length (around 150 tokens). (b) shows the "Underthinking Frequency" metric UF(L) (defined in Eq.(1)) of six mainstream LongCoT LLMs at different values of length threshold L. The results show that underthinking is widespread in all models.

Recent Large Language Models (LLMs) (OpenAI, 2024b; 2025a; DeepMind, 2025; Guo et al., 2025)
have demonstrated significant progress, even surpassing human performance on tackling challenging complex reasoning tasks, such as competitive mathematics (AIME, 2024; 2025), programming (Jain et al., 2024), and PhD-level science question answering (Rein et al., 2024). The driving force behind this significant advancement is the Long Chain-of-Thought (LongCoT) reasoning paradigm. Unlike traditional Chain-of-Thought (CoT) reasoning (Wei et al., 2022), LongCoT often Anonymous authors Paper under double-blind review

## Abstract

# Smartswitch: Advancing Llm Reasoning By Overcoming Underthinking Via Promoting Deeper Thought Exploration

1 incorporates spontaneous reflection, self-correction mechanisms, and even the ability to switch thinking perspectives (OpenAI, 2024b). Observations. Despite progress, certain issues still limit the performance and efficiency of the LongCoT paradigm, such as the underthinking problem (see Section 3). In particular, models often switch thoughts prematurely without fully exploring their feasibility and potential (see Figure 1). This behavior significantly increases the risk of overlooking promising ideas, ultimately resulting in incorrect final answers. Additionally, frequent thought-switching leads to substantial token wastage. This underthinking behavior parallels impaired cognitive control in humans, where anxious problemsolvers abandon promising ideas too soon due to low confidence or high perceived failure risk (Robertson et al., 1997; Eysenck et al., 2007). Research shows that external support, like encouraging suggestions or metacognitive prompts from tutors, can help alleviate this tendency (Wells & Matthews, 2016; Clark & Beck, 2011; Cohen et al., 2007; Botvinick & Braver, 2015). These insights emphasize the need for potential assessment mechanisms and confidence calibration to help LLMs avoid underthinking. Our Approach. This paper proposes a novel SmartSwitch inference framework designed to detect and mitigate underthinking in real time. SmartSwitch operates in two cyclical stages. First, the *Perception* module identifies premature thought-switching by detecting linguistic cues (e.g., "Alternatively, ...")
that signal a change in direction and evaluates the potential of the just-abandoned reasoning path using a process reward model. Second, if a high-potential thought is deemed to have been prematurely discarded, the *Intervention* module activates. It interrupts the current generation, backtracks to the promising thought, and injects a targeted prompt to encourage deeper exploration along that thought. By enabling the reconsideration of prematurely abandoned yet promising reasoning avenues, SmartSwitch mitigates shallow reasoning and enhances model performance. Furthermore, our framework is fine-tuning-free and plug-and-play, facilitating seamless integration with a wide range of LLMs. We evaluate our approach on five well-known challenging mathematics benchmarks, including four competition-level datasets - AIME24 (AIME, 2024), AIME25 (AIME, 2025), AMC23 (AMC, 2023), and MATH-500 (Hendrycks et al., 2021), and one standard-level benchmark - GaoKao2023en (Chinese GaoKao Community, 2024). Results in Table 1 show that our SmartSwitch consistently outperforms vanilla inference strategy, and brings significant improvements for existing LLMs with sizes ranging from 1.5B to 32B, demonstrating the good compatibility, generalization, and robustness of our approach. For example, inference by SmartSwitch, the accuracy of DeepSeek-R1-Distill- Qwen-1.5B on AIME24 is boosted by 11.1 points (from 28.9% to 40.0%), and QwQ-32B achieves 73.3% on AIME25 with a gain of 10.0 points.

## 2 Related Work

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Large language models with LongCoT reasoning. Reasoning ability is a core indicator of the intelligence of Large Language Models (LLMs). For a long time, Chain-of-Thought (CoT) reasoning (Wei et al., 2022) has served as the dominant paradigm, allowing models to reason step by step until deriving the final answer. While effective on many tasks (Cobbe et al., 2021; Chen et al., 2021), CoT-based LLMs still struggle with challenging reasoning problems, for example, GPT-4o (OpenAI, 2024a) achieves only 13.4% accuracy on the well-known AIME24 math competition (AIME, 2024). This landscape changed with the emergence of OpenAI's o1 model (OpenAI, 2024b), which marked a milestone in reasoning LLMs. It demonstrated significant improvements across a wide range of challenging reasoning tasks, including competition-level mathematics (AIME, 2024; 2025), programming (Jain et al., 2024), and PhD-level scientific question answering (Rein et al., 2024).

These advances are attributed to a novel reasoning paradigm, Long Chain-of-Thought (LongCoT)
reasoning, which enables models to conduct a thorough thinking process before giving a deterministic solution. In contrast to the deterministic reasoning traces in CoT, LongCoT exhibits a more free-form and exploratory structure, allowing the model to explore different ideas, reflect intermediate steps, and correct its own errors. Given its clear advantages, researchers have sought to replicate the capabilities of o1, inspiring a wave of subsequent works, such as closed-source models (DeepMind, 2025), open-source efforts (Guo et al., 2025; Muennighoff et al., 2025; Min et al., 2024; Bespoke Labs, 2025), as well as the upgraded versions from OpenAI itself (OpenAI, 2025a;b).

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Thinking effectiveness in LongCoT reasoning. Although the LongCoT reasoning paradigm provides opportunities for free and in-depth exploration through a human-like slow thinking phase, the effectiveness of thinking plays a crucial role in determining the performance of the model on challenging reasoning tasks. An effective thinking process can be characterized by several behaviors that involve reasonably planning the reasoning trajectory, for example, reflecting previous steps and exploring new ideas when necessary rather than casually or frequently. The low-effectiveness of thinking in existing LongCoT models (Guo et al., 2025; Qwen Team, 2025; 2024) is reflected in two extremes. On the one hand, the model tends to overthink. Some studies (Chen et al., 2024) have shown that models take about 1000 tokens to reason even for a simple problem like "1 + 1 =?". This redundancy not only leads to unnecessary token usage and inefficient reasoning, but also has no benefit to performance. On the other hand, we found that models still suffer from the underthinking problem. They tend to switch thoughts frequently, e.g., prematurely turning to other thoughts without sufficient exploration on the currect thought. This behavior limits the effectiveness of in-depth thinking and leads to the neglect of promising ideas and the opportunity to derive the correct final answer. Recent study (Wang et al., 2025) also recognized the risk of premature switching and proposed a token-space decoding constraint to suppress the generation probability of tokens corresponding to keywords for switching thoughts. While, such heuristic method introduces artificial bias, which may hinder the indispensable and reasonable exploration behavior due to over-constraining. In contrast, we adaptively steer the model to dive deeper into the current thought or explore a new thought based on the feasibility and potential of the current thought.

## 3 Underthinking Problem Investigation

In LongCoT reasoning, a thought refers to an independent reasoning unit aimed at solving a specific sub-problem or achieving an intermediate objective. The model is allowed to switch thoughts when the current thought proves infeasible or the objective itself needs to be redefined. This thoughtswitching mechanism is a core mechanism, disengaging the model from unproductive explorations and dynamically adapting its reasoning paths. However, we observe that current LongCoT LLMs often switch thoughts too prematurely before fully exploring the potential of the current thought. This leads to the premature abandonment of promising directions, ultimately harming performance. We refer to this behavior as the "*underthinking problem*". Notably, switching thoughts is not problematic in itself; rather, it is the frequency and hasty switching that undermines deep and effective reasoning.

## 3.1 Qualitative Analysis

Figure 1(a) qualitatively illustrates underthinking in a DeepSeek-R1 response: its reasoning trace exhibits frequent shifts, suggesting insufficient depth. The model prematurely abandons viable strategies (e.g., by partially applying geometric properties like harmonic relations) or disrupts valid reasoning chains through conceptual errors (e.g., conflating distinct geometric points) or misjudgments of problem complexity, resulting in a cascade of short, underdeveloped thoughts.

![2_image_0.png](2_image_0.png)

## 3.2 Quantitative Analysis

To quantify the underthinking in existing LLMs with LongCoT capabilities, we define a new metric, named Underthinking Frequency, which represents the number of underthinking thoughts in the entire thinking process. Specically, given a LongCoT response consisting of a thinking process T and a solution S for a question Q, we

![3_image_0.png](3_image_0.png)

first segment T into a sequence of individual thoughts {Ti}M
i=1, where Tiis the i-th thought and M
is the total number of thoughts. This segmentation can be performed using a capable LLM (e.g., DeepSeek-V3 (Liu et al., 2024)). The specific prompt used for this process is detailed in Appendix F.3. Then, we can define the *Underthinking Frequency (UF)* metric:

$$\mathrm{UF}_{L}=\sum_{i=1}^{M}\lambda_{i}(L),$$
$$(1)$$

λi(L), (1)

## 4 Methodology

To address the underthinking problem, we propose the SmartSwitch inference framework. This framework aims to dynamically guide LLMs towards deeper exploration of promising reasoning paths that might otherwise be prematurely abandoned.

## 4.1 Motivation

The investigation in Section 3 reveals that LLMs, despite their LongCoT capabilities, often fail to fully explore complex problems due to underthinking—rapidly switching between shallow thoughts.

where λi(L) is a binary variable indicating whether thought Ti exhibits underthinking. Heuristically, we define λi(L) according to the length of thought Ti, that is, λi(L) = 1 if |Ti| < L, otherwise λi(L) = 0, where L is the token length threshold.

Figure 1(b) shows the average frequency metric for under-thinking on AIME24 (AIME, 2024) in six main LongCoT LLMs with different values of L. Figure 2 illustrates the correlation between underthinking frequency and task difficulty. We conclude three key observations below:
(1) *Prevalence:* All six models consistently exhibit the underthinking behavior, indicating its widespread presence among current LongCoT LLMs.

(2) *Severity:* The degree of underthinking differs across models. QwQ-32B (Qwen Team, 2025)
shows the most severe underthinking, while within the DeepSeek-R1-Distill-Qwen series, the smallest 1.5B model exhibits the highest tendency to underthink.

(3) *Contributing Factors:* We observe a clear correlation between underthinking and task difficulty.

As Figure 2(a), problems that the model fails to solve tend to trigger more underthinking than those it answers correctly. Underthinking frequency increases steadily with human-annotated difficulty levels, indicating that harder problems tend to amplify underthinking.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 This behavior limits their ability to solve challenging tasks that require sustained, in-depth reasoning. Human problem-solving often benefits from metacognitive strategies, such as recognizing a promising but underdeveloped idea and consciously deciding to delve deeper. Our framework is inspired by this, aiming to equip LLMs with a similar capability: to perceive when a valuable thought is being neglected and to intervene by prompting a more thorough exploration of that thought. The goal is to transform the default, sometimes erratic, exploration pattern into a more deliberate and productive reasoning process.

## 4.2 Smartswitch Inference Framework

The SmartSwitch framework operates iteratively during the LLM's generation process, as illustrated in Figure 3. It consists of two main modules: Perception and Intervention. The complete algorithm is detailed in Appendix D.1.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Perception module. During the autoregressive generation process, where the LLM M produces tokens ti ∼ PM(ti| *Q, t*1:i−1), the Perception module continuously monitors the output stream.

- *Thought Switch Detection:* It looks for linguistic cues (e.g., "Alternatively") that signal a potential shift away from the current line of reasoning. A comprehensive list of these cues is provided in Appendix D.2.

- *Thought Segmentation:* Upon detecting a switch, the primary unit for evaluation is the entire block of text preceding the cue, which we denote as the thought T*prev*. To ensure that these thoughts remain a manageable length for evaluation, we apply a simple rule: if T*prev* exceeds a predefined threshold (e.g., 200 tokens), it can be further subdivided at natural breaks like paragraph boundaries (\n\n). Otherwise, the entire T*prev* is passed to the next stage.

- *Potential Evaluation:* The segmented thought T*prev* is then evaluated by a pre-trained Process Reward Model (PRM). The PRM outputs a score indicating the quality or potential of T*prev*. If this score exceeds a predefined threshold τ*score*, it suggests that T*prev* is a promising reasoning path that has likely been abandoned prematurely.

Intervention module. If the Perception module flags T*prev* as a high-potential, prematurely abandoned thought, the Intervention module activates:
- *Interruption and Backtracking:* The LLM's current generation (which has started on a new thought after the switch) is interrupted. The generation context is rolled back to the state immediately after T*prev* completes but before the switch occurs.

- *Deepen Prompt Injection:* A predefined "deepen prompt" is appended to the context. An example prompt is: "Wait, this seems like a promising idea. Let's dive deeper into this reasoning path and not give up easily. Continue exploring this direction thoroughly."
- *Resumed Generation:* The LLM then resumes generation from this modified context, now guided to further explore T*prev* instead of switching away. To maintain consistency, the generation proceeds with the original inference parameters.

If the PRM score for T*prev* is below τ*score*, no intervention occurs, and the LLM continues with its new thought. This cyclical process of perception and potential intervention continues throughout the generation, aiming to foster deeper exploration when beneficial. A maximum intervention depth or count per problem can be set to prevent excessive looping.

By systematically identifying and reinforcing promising but underdeveloped lines of reasoning, SmartSwitch aims to improve the overall quality and success rate of LLM problem-solving without requiring model retraining.

## 5 Experiments 5.1 Experimental Setups

Baseline Models. We apply our SmartSwitch inference framework to a variety of advanced LongCoT LLMs with varying sizes (1.5B to 32B), including DeepSeek-R1-Distill-Qwen-1.5B / 7B / 14B / 32B (Guo et al., 2025) and QwQ-32B (Qwen Team, 2025).

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 Evaluation Benchmarks. We evaluate the models with our SmartSwitch inference framework on various challenging mathematics benchmarks, since mathematical problem solving is one of the most fundamental tasks for assessing the reasoning ability of LLMs. To ensure comprehensiveness, we consider benchmarks spanning two difficulty levels: competitionlevel and standard-level. The competition-level set includes AIME24 (AIME, 2024), AIME25 (AIME, 2025), AMC23 (AMC, 2023), and MATH-500 (Hendrycks et al., 2021), which are collected from real human math competitions.

The standard-level benchmark, GaoKao2023en (Chinese GaoKao Community, 2024), offers a more routine yet still non-trivial evaluation. We report the pass@1 accuracy averaged on 32 responses for all benchmarks. Inference Settings. For fair comparisons, we apply the same inference settings as each baseline model. In particular, the temperature is set to 0.6, and top-p equals 0.95. The maximum output length is limited to 32768 tokens. We generate 32 responses per query to estimate stable pass@1 accuracy. All the experiments are conducted on NVIDIA A100 GPUs. Implementation Details. In our SmartSwitch inference framework, we employ the off-theshelf Universal-PRM-7B as our thought scoring model (Tan et al., 2025) to evaluate the promising score of each thought. The reason for this choice is attributed to its capability to assess LongCoT reasoning traces, with support for input lengths up to 32768 tokens, which is a substantial increase over the typical 4096-token limit of most open-source process reward models. We set the promising score threshold to 0.7, meaning that any thought with a score above this value is considered promising and eligible for deepening intervention. To prevent excessive interventions within a single reasoning process, we cap the number of interruptions at three. Furthermore, as part of our thought segmentation strategy, any thought segment T*prev* that exceeds a 200-token threshold is first subdivided at natural paragraph breaks before being scored by the PRM.

| Models              | Inference          | Competitional-level   | Standard-level   |               |              |             |
|---------------------|--------------------|-----------------------|------------------|---------------|--------------|-------------|
| Framework           | AIME24             | AIME25                | AMC23            | MATH-500      | GaoKao2023en |             |
| DeepSeek-R1-Distill | Vanilla            | 28.9                  | 20.0             | 67.5          | 83.9         | 72.2        |
| Qwen-1.5B           | SmartSwitch (ours) | 40.0 (+11.1)          | 36.7 (+16.7)     | 77.5 (+10.0)  | 85.8 (+1.9)  | 76.9 (+4.7) |
| DeepSeek-R1-Distill | Vanilla            | 55.5                  | 30.0             | 85.0          | 92.8         | 82.6        |
| Qwen-7B             | SmartSwitch (ours) | 66.7 (+11.2)          | 53.3 (+23.3)     | 92.5 (+7.5)   | 93.4 (+0.6)  | 84.2 (+1.6) |
| DeepSeek-R1-Distill | Vanilla            | 69.7                  | 43.3             | 92.5          | 93.2         | 82.4        |
| Qwen-14B            | SmartSwitch (ours) | 76.7 (+7.0)           | 53.3 (+10.0)     | 100.0 (+7.5)  | 95.2 (+2.0)  | 86.0 (+3.6) |
| DeepSeek-R1-Distill | Vanilla            | 72.6                  | 46.7             | 90.0          | 94.3         | 85.4        |
| Qwen-32B            | SmartSwitch (ours) | 76.7 (+4.1)           | 66.7 (+20.0)     | 100.0 (+10.0) | 95.2 (+0.9)  | 87.0 (+1.6) |
| QwQ-32B             | Vanilla            | 79.5                  | 63.3             | 97.5          | 95.0         | 85.2        |
| SmartSwitch (ours)  | 86.7 (+7.2)        | 73.3 (+10.0)          | 100.0 (+2.5)     | 97.0 (+2.0)   | 88.3 (+3.1)  |             |

Table 2: Comparison on the "response length (token number)" of models under vanilla inference and our SmartSwitch. We report the average length on AIME24 benchmark. "only correct" corresponds to the problems answered correctly.

| Model                         | Inference Framework   | Response Length (Token Number) All ↓ only correct ↓   |                |
|-------------------------------|-----------------------|-------------------------------------------------------|----------------|
| Vanilla                       | 14973.97              | 6424.33                                               |                |
| DeepSeek-R1-Distill Qwen-1.5B | SmartSwitch           | 13486.80↓9.93%                                        | 6125.78↓4.65%  |
| Vanilla                       | 14663.03              | 9215.86                                               |                |
| DeepSeek-R1-Distill Qwen-7B   | SmartSwitch           | 14240.07↓2.88%                                        | 8096.79↓12.14% |
| Vanilla                       | 14128.90              | 11195.50                                              |                |
| DeepSeek-R1-Distill Qwen-14B  | SmartSwitch           | 14480.20↑2.49%                                        | 9433.19↓15.74% |
| Vanilla                       | 15375.17              | 12272.28                                              |                |
| DeepSeek-R1-Distill Qwen-32B  | SmartSwitch           | 13188.00↓14.22% 10284.33↓16.20%                       |                |
| Vanilla                       | 16924.40              | 14115.48                                              |                |
| QwQ-32B                       | SmartSwitch           | 15939.97↓5.82%                                        | 13116.87↓7.07% |

Table 3: Comparison of inference time (min/q) and the time change achieved by *SmartSwitch* on competition-level benchmarks.

| Model                         | Inference Framework                                                 | Avg. Time (min/q)   |      |
|-------------------------------|---------------------------------------------------------------------|---------------------|------|
| AIME24 ↓ AIME25 ↓ AMC23 ↓     |                                                                     |                     |      |
| Vanilla                       | 3.23                                                                | 2.69                | 1.10 |
| DeepSeek-R1-Distill Qwen-1.5B | SmartSwitch 2.14↓33.7% 2.30↓14.5% 1.09↓0.9% Vanilla 3.31 3.35 0.90  |                     |      |
| DeepSeek-R1-Distill Qwen-7B   | SmartSwitch 2.14↓35.3% 2.30↓31.3% 0.72↓20.0% Vanilla 2.57 3.22 1.29 |                     |      |
| DeepSeek-R1-Distill Qwen-14B  | SmartSwitch 2.09↓18.7% 2.43↓24.5% 1.07↓17.1% Vanilla 4.87 5.27 2.12 |                     |      |
| DeepSeek-R1-Distill Qwen-32B  | SmartSwitch 3.91↓19.7% 4.98↓5.5%                                    | 1.91↓9.9%           |      |
| Vanilla                       | 5.77                                                                | 6.82                | 3.07 |
| QwQ-32B                       | SmartSwitch 4.97↓13.9% 5.67↓16.9% 2.77↓9.8%                         |                     |      |

6 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

## 5.3 Further Analysis

Mitigate Underthinking. SmartSwitch significantly reduces the underthinking behavior of LLMs. Specifically, when measuring with a token length threshold of L = 100, it not only lowers the Underthinking Frequency metric defined in Equation. (1) (as shown in Figure 4(a)), but also decreases the number of thought switches (as illustrated in Figure 4(b)). This leads to more focused and coherent reasoning trajectories. Boost Performance on Failures without Hurting Successes. Our SmartSwitch improves model performance on challenging problems previously answered incorrectly under vanilla inference, while preserving accuracy on those already solved correctly. For DeepSeek-R1-
Distill-Qwen-14B on AIME24, SmartSwitch maintains 100% accuracy on all previously correct answers and successfully recovers 20% of the previously incorrect ones. This demonstrates that SmartSwitch delivers targeted gains without compromising existing capabilities. Significant Improvements for Small LLMs. Our SmartSwitch yields substantial gains for smaller models. As shown in Table 1, DeepSeek-R1-Distill-Qwen-1.5B achieves an accuracy gain of 16.7% on AIME25, and DeepSeek-R1-Distill-Qwen-7B is improved by 23.3% points on AIME25. Consistent Gains for Large LLMs. While larger LLMs have already achieved high performance on challenging benchmarks, SmartSwitch continues to bring consistent and substantial improvements on these strong LLMs. Taking QwQ-32B as an example, our SmartSwitch boosts the accuracy from 79.5% to 86.7% (with 7.2 points gain) on AIME24, and the accuracy from 63.3% to 73.3% (with 10.0 points gain) on AIME25. Remarkably, QwQ-32B even achieves 100% accuracy on AMC23 competition. These results highlight the robustness and broad applicability of our SmartSwitch, even for top-performing models with few improvement room.

Bridging the Gap Across Model Scales. Our SmartSwitch can also help narrow the performance gap between smaller and larger model variants. For example, DeepSeek-R1-Distill-Qwen-14B with our SmartSwitch inference surpasses the DeepSeek-R1-Distill-Qwen-32B with vanilla inference on all benchmarks (53.3 vs. 46.7 on AIME25). This highlights the potential of our approach for enabling more capable reasoning in resource-constrained scenarios. Efficiency. Interestingly, our SmartSwitch significantly improves inference efficiency by reducing both total inference time and response length, even while explicitly encouraging deeper thinking. On the AIME24 benchmark, our method shortens the total wall-clock inference time, which comprehensively includes all overhead from PRM scoring and intervention management, by 33.7% for the DeepSeek-R1-Distill-1.5B model and 19.7% for the 32B model (Table 3). Concurrently, the average response length is also reduced by 9.93% and 14.22% for the respective models (Table 2). This dual improvement suggests that our SmartSwitch effectively prunes wasteful reasoning on less fruitful thoughts, thereby focusing computational resources and exploration on more promising directions.

![6_image_0.png](6_image_0.png)

quency and the number of thought-switches on AIME24. "R1-Distill" abbreviates "DeepSeek-R1- Distill-Qwen".

## 5.4 Comparison With Other Underthinking Mitigation Methods

We compare SmartSwitch with two alternative methods for mitigating underthinking:
- *Standard Prompting*: Incorporate general instructions into initial system prompt to encourage deeper thinking "Think step by step. Explore each idea thoroughly before moving on.".

## 5.2 Main Results

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431
- *TIP (Thought Switching Penalty)* (Wang et al., 2025): A method introduces a penalty on tokens that are associated with thought transitions during decoding.

As shown in Table 5, standard prompting shows nearly no improvement, indicating general instructions are insufficient. TIP only brings limited gain, because it suppresses the decoding probability of the thought-switching tokens indiscriminately, regardless of whether the current thought has become unpromising. This rigid constraint may hinder the model's ability to explore alternative reasoning paths when necessary. In contrast, our SmartSwitch performs best with 40.0% accuracy on AIME24, compared to vanilla inference (28.9%), standard prompting (29.0%), and TIP (31.3%).

## 5.5 Ablation Study

Potential Scoring Model. Table 4 presents the performance of various Process Reward Models (PRMs) on AIME25. To quantify the value of PRM guidance, we test an "Always Intervene" baseline that injects a prompt at every thought switch, while adhering to the same three-intervention limit per problem. This naive strategy degrades performance to 18.9%, highlighting the critical role of selective, PRM-guided intervention. Among the PRMs, Universal-PRM-7B achieves the best accuracy at 36.7%. We select it not only for its superior performance but, more importantly, for its essential long-context capability, supporting inputs up to 32,768 tokens. This feature is crucial for evaluating our LongCoT traces and is a key limitation of other PRMs, which either perform worse or lack the necessary context length (see Appendix D.2 for details).

Table 4: Ablation on the effect of different Process Reward Models to scoring the potential.

| Models                        | Process Reward Model   | AIME25   |
|-------------------------------|------------------------|----------|
| N/A                           | 20.0                   |          |
| Always Intervene              | 18.9                   |          |
| Qwen2.5-Math-PRM-7B           | 21.1                   |          |
| Qwen2.5-Math-7B-PRM800K       | 22.3                   |          |
| Qwen2.5-Math-PRM-72B          | 24.8                   |          |
| Universal-PRM-7B              | 36.7                   |          |
| DeepSeek-R1-Distill Qwen-1.5B |                        |          |

Table 5: Comparison of different inference frameworks.

| Model                         | Inference Framework AIME24 Vanilla 28.9 Standard Prompting 29.0 TIP Wang et al. (2025) 31.3 SmartSwitch (ours) 40.0   |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| DeepSeek-R1-Distill Qwen-1.5B |                                                                                                                       |

Table 6: Ablation on the effect of process division strategy on AIME25 benchmark.

| Model                          | v1        | v2   | v3   | v4   |
|--------------------------------|-----------|------|------|------|
| R1-Distill-Qwen-1.5B 23.3 26.7 | 26.7      | 36.7 |      |      |
| R1-Distill-Qwen-7B             | 40.0 43.3 | 40.0 | 53.3 |      |
| R1-Distill-Qwen-14B            | 43.3 46.7 | 46.7 | 53.3 |      |
| R1-Distill-Qwen-32B            | 50.0 53.3 | 53.3 | 66.7 |      |
| QwQ-32B                        | 70.0 70.0 | 73.3 | 73.3 |      |

Process Division Strategy. To enable effective scoring by the Process Reward Model (PRM),
the full reasoning trace must first be divided into coherent processes. Here, we explore four strategies:
- *Model Division (v1)* utilizes a powerful LLM
(such as DeepSeek-V3 (Liu et al., 2024)) to perform this division using a carefully designed prompt. This approach introduces additional computational or API cost.

- *Grouped Paragraph (v2)*: This method segments at paragraph boundaries (\n\n) and then groups these initial segments into fixedsize chunks (e.g., five steps).

- *Single Paragraph (v3)*: Segments the output strictly at each detected paragraph boundary
(\n\n), treating every resulting block as an individual reasoning step, which can lead to fragmentation.

- *Adaptive Paragraph (v4) (ours)*: Our proposed method (v4) is a multi-stage approach designed to ensure conceptual coherence and optimal segment length for PRM scoring. It first splits the text at logical transition points, such as 'alternate'. If any resulting segment is still longer than 200 tokens, it is further divided using adaptive subdivision–specifically, by breaking at paragraph boundaries (e.g.,
"\n\n") to maintain readability and structure.

As shown in Table 6, strategy v4 consistently outperforms its counterparts (v1, v2, and v3) across all model scales, achieving superior accuracy. The effectiveness of v4 arises from its principled design, which ensures conceptual coherence within each step and optimizes segment length for effective PRM scoring, thereby avoiding the fragmentation issues of strict paragraph splits (v3), the potential conceptual merging of arbitrary grouping (v2), and the additional computational cost and potential inconsistencies of a model-based approach (v1). These results highlight the critical role of a carefully designed step division strategy in maximizing the performance of the framework.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Process-to-Thought Score Mapping Strategy. Since the PRM assigns a potential score to each individual process, but a single thought may consist of multiple processes, we need to aggregate these process-level scores to obtain a final score for each thought. We explore several aggregation strategies, including taking the mean, maximum, median, weighted average, or simply the score of the last process within the thought. As shown in Table 7, for a thought, the simple strategy that treating the score of last process within this thought as its final potential score achieves the best performance. Thus, we use this strategy by default.

Table 7: Ablation on the effect of different processto-thought score mapping strategies.

Potential Score Threshold. We investigated the impact of the potential score threshold on R1- Distill-Qwen-1.5B's AIME24 performance (Table 8). Compared to the vanilla baseline (28.90% accuracy), thresholds of 0.68 and 0.69 increased accuracy to 30.00%. Performance peaked significantly at a 0.70 threshold with 40.00% accuracy, before dropping to 30.00% at 0.71. This demonstrates that while a suitable threshold range improves results, selecting the optimal value, such as 0.70 in this case, is crucial.

## 6 Discussion

| Models                        | Mapping Strategy   | AIME24   |
|-------------------------------|--------------------|----------|
| max                           | 33.33              |          |
| min                           | 30.00              |          |
| mean                          | 30.00              |          |
| median                        | 33.33              |          |
| weighted average              | 33.33              |          |
| last                          | 40.00              |          |
| DeepSeek-R1-Distill Qwen-1.5B |                    |          |

Table 8: AIME24 ablation on the potential score threshold.

Limitations. The efficacy of our framework depends on the quality and calibration of the external Process Reward Model. Its performance is fundamentally bounded by the PRM's ability to accurately assess the potential of diverse reasoning paths. Furthermore, SmartSwitch relies on several key hyperparameters, such as the potential score threshold and the maximum intervention count. While our experiments show that a well-chosen setting is effective across various models, these parameters may require domain-specific or model-specific tuning for optimal performance. Finally, our current thought-switch detection mechanism is based on linguistic cues, which may not capture all instances of premature abandonment, especially those that occur without explicit textual markers. This reliance on explicit markers means it may miss more subtle or implicit shifts in reasoning strategy.

Model vanilla 0.68 0.69 **0.70** 0.71 R1-Distill-Qwen-1.5B 28.9 30.0 30.0 **40.0** 30.0 R1-Distill-Qwen-7B 55.5 53.3 43.3 **66.7** 43.3 R1-Distill-Qwen-14B 69.7 66.7 70.0 **76.7** 70.0 R1-Distill-Qwen-32B 72.6 63.3 63.3 **76.7** 63.3 QwQ-32B 79.5 73.3 73.3 **86.7** 73.3 Future work. A primary direction for future work is to reduce the reliance on external components. One promising avenue is to distill the evaluative capabilities of the PRM directly into the base LLM,
enabling it to perform self-assessment of its reasoning paths without an external call. This could lead to a more efficient and integrated system. Another area for advancement is the development of more sophisticated intervention mechanisms. Instead of a fixed prompt, a dynamic system could generate context-aware prompts to guide the model's exploration more precisely. Finally, we plan to extend the SmartSwitch framework beyond mathematical reasoning to other complex domains such as software engineering, scientific discovery, and legal analysis, which will require adapting the evaluative criteria and intervention strategies to new contexts.

## 7 Conclusion

In this paper, we identify and characterize the "underthinking" phenomenon in LLMs with Long- CoT capabilities, where models prematurely abandon promising reasoning paths, hindering their performance on complex tasks. To address this, we propose the SmartSwitch framework. Using linguistic cues, SmartSwitch detects these switches, employs a PRM to assess abandoned thoughts, and prompts deeper exploration of valuable overlooked paths. This training-free, model-agnostic approach significantly improves LLM performance on mathematical benchmarks by fostering deeper exploration and reducing shallow reasoning. SmartSwitch offers a promising direction for enhancing the reliability and depth of reasoning in LLMs.

## Ethics Statement

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## Reproducibility Statement References

AIME. American invitational mathematics examination, 2024. URL https:
//artofproblemsolving.com/wiki/index.php/AIME_Problems_and_
Solutions.

AIME. American invitational mathematics examination, 2025. URL https://
artofproblemsolving.com/wiki/index.php/2025_AIME_I.

AMC. American mathematics competitions, 2023. URL https://artofproblemsolving.

com/wiki/index.php/2023_AMC_12A.

Bespoke Labs. Bespoke-stratos: The unreasonable effectiveness of reasoning distillation, 2025.

Michael W Eysenck, Nazanin Derakshan, Rita Santos, and Manuel G Calvo. Anxiety and cognitive performance: attentional control theory. *Emotion*, 7(2):336, 2007.

Matthew Botvinick and Todd Braver. Motivation and cognitive control: from behavior to neural mechanism. *Annual review of psychology*, 66(1):83–113, 2015.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. *arXiv preprint arXiv:2412.21187*, 2024.

Chinese GaoKao Community. Gaokao2023-math-en, 2024. URL https://huggingface.co/
datasets/MARIO-Math-Reasoning/Gaokao2023-Math-En.

David A Clark and Aaron T Beck. *Cognitive therapy of anxiety disorders: Science and practice*.

Guilford Press, 2011.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.

Jonathan D Cohen, Samuel M McClure, and Angela J Yu. Should i stay or should i go? how the human brain manages the trade-off between exploitation and exploration. Philosophical Transactions of the Royal Society B: Biological Sciences, 362(1481):933–942, 2007.

Google DeepMind. Gemini 2.5 flash, 2025. URL https://deepmind.google/
technologies/gemini/flash/.

The supplementary material contains the complete source code to ensure full reproducibility of our results. This encompasses all pipelines used for response generation and the automated evaluation of model outputs.

This research adheres to the ICLR Code of Ethics. Our work aims to positively contribute to society by improving the reasoning capabilities of Large Language Models (LLMs), making them more robust and efficient for complex tasks. We acknowledge the importance of the responsible application of this technology. We encourage practitioners who build upon our framework to be mindful of potential societal impacts and to ensure that the underlying models are used in a fair and equitable manner. Our research does not involve the collection or use of new personally identifiable information. Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. *arXiv* preprint arXiv:2103.03874, 2021.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. *arXiv preprint arXiv:2403.07974*, 2024.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang, Xiaoxue Cheng, Huatong Song, et al. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. *arXiv preprint arXiv:2412.09413*, 2024.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori Hashimoto. s1: Simple test-time ` scaling. *arXiv preprint arXiv:2501.19393*, 2025.

OpenAI. Gpt-4o, 2024a. URL https://openai.com/index/hello-gpt-4o/. OpenAI. Learning to reason with llms, 2024b. URL https://openai.com/index/
learning-to-reason-with-llms/.

OpenAI. Openai o3-mini, 2025a. URL https://openai.com/index/openai-o3-mini/. OpenAI. Introducing openai o3 and o4-mini, 2025b. URL https://openai.com/index/
introducing-o3-and-o4-mini/.

Qwen Team. Qwq: Reflect deeply on the boundaries of the unknown, 2024. URL https://
qwenlm.github.io/blog/qwq-32b-preview/.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, 2025. URL https:
//qwenlm.github.io/blog/qwq-32b/.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Ian H Robertson, Tom Manly, Jackie Andrade, Bart T Baddeley, and Jenny Yiend. Oops!': performance correlates of everyday attentional failures in traumatic brain injured and normal subjects. Neuropsychologia, 35(6):747–758, 1997.

Xiaoyu Tan, Tianchu Yao, Chao Qu, Bin Li, Minghao Yang, Dakuan Lu, Haozhe Wang, Xihe Qiu, Wei Chu, Yinghui Xu, et al. Aurora: Automated training framework of universal process reward models via ensemble prompting and reverse verification. *arXiv preprint arXiv:2502.11520*, 2025.

Yuxi Tong. symeval: A python library for symbolic evaluation in mathematical reasoning, 2024.

Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian Yu, Juntao Li, Zhuosheng Zhang, et al. Thoughts are all over the place: On the underthinking of o1-like llms. *arXiv preprint arXiv:2501.18585*, 2025.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Adrian Wells and Gerald Matthews. *Attention and emotion: A clinical perspective*. Psychology Press, 2016.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. *arXiv preprint arXiv:2501.07301*, 2025.

Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Processbench: Identifying process errors in mathematical reasoning. arXiv preprint arXiv:2412.06559, 2024.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647