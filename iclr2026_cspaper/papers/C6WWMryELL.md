Anonymous authors Paper under double-blind review

## Abstract

Large Language Models (LLMs) excel at long-context understanding but exhibit significant limitations in long-form generation. Existing studies primarily focus on single-generation quality, generally overlooking the volatility of the output (i.e., the inconsistency in length and content across multiple generations). This volatility not only leads to significant computational costs but also severely impacts the models' reliable application. To address this gap, our work unfolds in three stages: *benchmarking, probing, and mitigation*. We first propose the VOlatility in Long-form Text **Bench**mark (**VOLTBench**), a novel heterogeneoustask benchmark designed to systematically quantify the length volatility of longform generation. Subsequently, by analyzing attention traces, we conduct an indepth probe to identify several common internal patterns that cause this volatility. Finally, to mitigate long-form output volatility, we propose SELB (Structural Enforcement via Logits Boosting), a lightweight decoding-stage optimization strategy, designed to significantly enhance both the length accuracy and stability of long-form generation without additional training. Extensive experiments on VOLTBench provide the first systematic confirmation of severe long-form output instability in mainstream models and validate that our proposed method successfully improves the mean output length of the base model by 148% and reduces the length volatility by 69%, while maintaining high generation quality.1

## 1 Introduction

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# On Stable Long-Form Generation: Bench- Marking And Mitigating Length Volatility

Large Language Models (LLMs) have made significant advances in long-context processing Bai et al. (2023); GLM et al. (2024); Comanici et al. (2025), capable of handling inputs exceeding 100k tokens and performing precise information retrieval in Needle-in-a-Haystack tasks Yuan et al. (2025); Ye et al. (2025a); Zhou et al. (2025). However, this remarkable progress in long-context understanding has not extended to long-form generation. Their outputs struggle to surpass the 2kword threshold Bai et al. (2024), while also lacking equivalent fine-grained control over the process. Recent studies have benchmarked the long-form generation capabilities of models, typically employing unstructured content generation tasks such as story writing, and observed that current models generally struggle to meet target lengths accurately Liu et al. (2024); Zhang et al. (2025b); Wu et al. (2025b). Some work attributes this issue preliminarily to data-related factors, such as the scarcity of long-output examples in supervised fine-tuning (SFT) datasets Bai et al. (2024). However, we argue that current research has three core limitations: First, existing work focuses almost exclusively on **single-generation results**, systematically overlooking output stability. This paradigm fails to capture the **significant volatility** that occurs when models process the same prompt multiple times, as shown in Figure 1, leading to unpredictable token consumption and high costs. Second, current benchmarks over-rely on **unstructured tasks** like story generation. Their subjective and difficult-to-automate evaluation criteria hinder the objective, quantifiable assessment of generation quality. In contrast, structured tasks with clear rules (e.g., code generation) offer a better environment for evaluation but remain underexplored. Finally, most research is limited to observing the phenomenon, **lacking an in-depth investigation** into the internal mechanisms.

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 To address the aforementioned limitations, we conduct an in-depth, multi-stage investigation into the volatility of LLM longform generation from three perspectives: Benchmarking, Probing, and Mitigating. First, on the benchmarking front, we introduce **length volatility** as a core metric and construct the Volatility in Long-form Text Benchmark (**VOLTBench**), a multidimensional, heterogeneous-task benchmark covering not only unstructured text (e.g., story) and structured data (e.g., code) but also dimensions such as different languages and instruction complexities. Through empirical evaluation on this benchmark, we provide the first largescale quantification of the prevalent output length volatility in mainstream models. Second, in our probing efforts, we leverage these benchmark findings to conduct an in-depth analysis of the root causes of this volatility. Moving beyond mere phenomenological observation, by analyzing the models' attention traces, we identify and define several common **internal patterns of length volatility**, such as *Attention Collapse* and *Attention Instability*. Finally, to mitigate the identified internal patterns, we propose and validate **Structural Enforcement via Logits Boosting (SELB)**, a lightweight, decoding-stage method that requires *no additional training* and proactively suppresses tokens linked to known failure modes, simultaneously improving both length accuracy and output stability. Our contributions are as follows:
Figure 1: Model performance on our VOLTBENCH for long-text generation. As the required length increases, the actual output length of all models falls significantly short of the target (dashed line). Furthermore, many models exhibit significant output length volatility, even for Longwriter-8B, a model specifically fine-tuned on long text, **the output standard deviation** peaked at 103% of its mean length.

![1_image_0.png](1_image_0.png)

- We construct the Volatility in Long-form Text Benchmark (VOLTBench), which is the first to introduce output volatility as a core metric. We systematically evaluate the long-form generation volatility in LLMs by covering both unstructured and structured tasks.

- We conduct extensive experiments that demonstrate the severe long-form output instability in mainstream LLMs. To investigate the underlying mechanisms, we identify and define several common internal patterns of length volatility through attention trace analysis.

- Targeting the identified internal patterns, we propose Structural Enforcement via Logits Boosting (SELB), which is a lightweight, decoding-stage optimization strategy that requires no additional training and improves the mean output length of the base model by 148% and reduces the length volatility by 69%, while maintaining high generation quality.

## 2 Related Work

Benchmarking Long-Form Generation. Existing studies have revealed the limitations of current models in long-form generation from multiple dimensions. HelloBench Que et al. (2024) uses diverse in-the-wild scenarios, finds that even advanced models face severe repetition. LIFEBench Zhang et al. (2025b) shows that models struggle to adhere to precise length requirements. LongGen- Bench Liu et al. (2024) reformulates existing QA datasets to assess the logical consistency of a single, sequential long-form answer. LongInOutBench Zhang et al. (2025a) targets the gap in longinput, long-output tasks, while LongProc Ye et al. (2025b) requires models to create structured outputs from dispersed information. FACTS Grounding Jacovi et al. (2025) focuses on the factual accuracy of long responses against a source document, and ProxyQA Tan et al. (2024) uses an innovative proxy-question method to measure knowledge coverage. Meanwhile, works like LongGenBench Wu et al. (2025b) and LCFO Costa-jussa et al. ` (2025) further advance evaluations by introducing complex instruction-following in super-long texts. In contrast, our work specifically evaluates and addresses the phenomenon of Length Volatility, aiming to enhance the robustness and controllability of LLM long-text outputs. We provide a comparison between ours and previous studies in Table 1. Long-form Text Generation. Research in long-form text generation addresses the challenge that LLMs struggle to produce high-quality, lengthy outputs. Data-centric approaches have been pro108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Table 1: Comparison with existing related benchmarks. VOLTBench provides a more comprehensive evaluation framework and is the first to introduce multiple sampling and stability evaluation.

| Benchmark                           | Instruction    | Generation        |                                   |                   |                |              |    |        |
|-------------------------------------|----------------|-------------------|-----------------------------------|-------------------|----------------|--------------|----|--------|
| Multiple Task                       | Multiple Level | Multiple Language | Unstructured Text Structured Data | Multiple Sampling | Stability Eval | Length Scale |    |        |
| HELLOBENCH Que et al. (2024)        | ✓              | ✓                 | ✓                                 | ∼ 16k             |                |              |    |        |
| LONGBENCH Bai et al. (2024)         | ✓              | ✓                 | ✓                                 | ∼ 10k             |                |              |    |        |
| LONGGENBENCH Liu et al. (2024)      | ✓              | ✓                 | ∼ 8k                              |                   |                |              |    |        |
| LIFEBENCH Zhang et al. (2025b)      | ✓              | ✓                 | ✓                                 | ∼ 8k              |                |              |    |        |
| LONGPROC Ye et al. (2025b)          | ✓              | ✓                 | ∼ 8k                              |                   |                |              |    |        |
| LONGGENBENCH Wu et al. (2025b)      | ✓              | ✓                 | ∼ 32k                             |                   |                |              |    |        |
| LONGINOUTBENCH Zhang et al. (2025a) | ✓              | ✓                 | ∼ 16k                             |                   |                |              |    |        |
| VOLTBENCH (Ours)                    | ✓              | ✓                 | ✓                                 | ✓                 | ✓              | ✓            | ✓  | ∼ 100k |

posed, such as using agentic plan-and-write pipelines Bai et al. (2024), creating multi-constraint instructions via backtranslation Pham et al. (2024), or enabling models to iteratively extend their own outputs Pham et al. (2024); Quan et al. (2024). LongWriter-Zero Wu et al. (2025a) uses reinforcement learning (RL) from scratch to foster long-generation capabilities. Wang et al. (2024) applies inference-time training with methods like Temp-Lora to maintain context in a temporary module. In contrast to prior work, which often involves extensive data creation or complex training, we propose a lightweight mitigation method based on the analysis of the model's internal attention to mitigate the instability and improve instruction adherence in long-form text generated by LLMs.

## 3 Voltbench: Benchmarking The Length Volatility

![2_image_0.png](2_image_0.png)

In this section, we introduce VOLTBench (Figure 2), a novel benchmark designed to systematically evaluate the stability and reliability of LLMs in long-form generation tasks. Its key features are as follows: Diverse and Challenging Instructions: The foundation of VOLTBench lies in its multidimensional instruction set. Our instructions span a wide array of tasks, from creative unstructured writing (e.g., stories) to logical structured generation (e.g., code libraries), pushing models beyond simple narrative generation. Each task is presented with varying levels of complexity, including simple prompts, detailed contextual instructions, and challenges with highly specific fine-grained constraints to test meticulous instruction-following over long contexts. Furthermore, to assess linguistic robustness, all instructions are provided in parallel English and Chinese versions, enabling a direct and fair comparison of model performance across different languages.

Versatile and Scalable Generation: Corresponding to the input diversity, VOLTBench evaluates a generation space notable for its versatility and scale. A key distinction of our benchmark is the dual focus on both *unstructured text and complex structured data outputs*, such as complete codebases. We implement this structure through a chapter-based format, which requires models to generate hierarchically organized content. Chapter-based design is the key to our scalability, enabling us to create instructions that range from a concise *5-chapter document to an expansive 500-chapter tome*. This pushes models to their operational limits with an unprecedented length scale of up to 100k Figure 2: An overview of the VOLTBench framework. Our benchmark is constructed from four dimensions, covering structured and unstructured tasks. We evaluate performance from two aspects: generation quality and length volatility.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 words. This massive and explicitly sectioned scale is specifically intended to surface and analyze challenging failure modes. Generation Volatility and Quality Evaluation: The cornerstone of VOLTBench is its rigorous evaluation of both generation volatility and quality, moving beyond single-instance assessments to measure model reliability. We query a model multiple times for each instruction to create a distribution of outputs. We assess stability at both a macro level, analyzing overall length volatility, and a granular chapter-by-chapter level, checking for consistency within each section. This fine-grained analysis can reveal nuanced behaviors, such as a model starting strong but losing steam in later chapters. VOLTBench embeds fine-grained constraints (e.g., keyword, topic) into its prompts. This innovative design allows us to *automate quality assessment* even for unstructured narrative tasks, as we can programmatically verify if these specific constraints were met. This is complemented by our structured data generation tasks, where quality is assessed objectively via Execution-based Verification, thus providing a far more reliable and multi-faceted quality evaluation framework.

![3_image_0.png](3_image_0.png) 
Figure 3: Analysis of Output Length Volatility and Output Section Volatility. The left panel (a, b, c) compares the total output length volatility across three dimensions: language, instruction complexity, and output format. The right panel (d) shows the volatility in the number of generated sections.

## 3.1 Tasks

Our benchmark includes both unstructured and structured generation tasks. Each core task is expanded into multiple variants across three dimensions: language (*English/Chinese*), instruction complexity (*simple, complex, fine-grained constrain*), and output length (*from 5 to 500 chapters*). This multi-dimensional design precisely measures fluctuations in model performance under diverse conditions (see Appendix J.0.7 for all task instructions). Unstructured Tasks: This category of tasks evaluates a model's creativity, narrative coherence, and contextual consistency in long-form, free-form text. We include diverse scenarios such as Story, Dialogue, Diary, and Architecture to assess abilities ranging from plot development and maintaining a consistent persona to the creative use of specialized terminology. Below is an example:
Task: Story Label: English-Simple-M chapters-N words Instruction: Please write a novel consisting of M chapters about Jeff. Each chapter should revolve around a theme or plot, with a minimum of N words for each chapter. Ensure clarity and continuity ... and use '*** Finished ***' to indicate the end of the document.

## 3.2 Evaluation Metric

Our benchmark evaluates models' long-text generation capabilities across two core dimensions:
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

## 4.2 Fine-Grained Constraints

(1) **Length Standard Deviation (LSD)**, this metric measures the *absolute volatility* of the output lengths: LSD =
q1 N
PN
i=1(Li − µ)
2, where µ is the average of the N output lengths. In our experiments, we set N=5.

(2) **Length Variation Coefficient (LVC)**, this measures the *relative volatility* of the output lengths with respect to their mean, which allows for comparable stability assessments across different length requirements: LVC =
LSD
µ.

(3) **Mean Length Accuracy (MLA)**, this metric quantifies how closely the mean length (µ)
of N generation runs adheres to the specified target length (Lconstraint). The formula is:
MLA = max 0, 1 −

µ−Lconstraint Lconstraint

× 100.

Generation Quality. We assess the quality of the generated content from the following aspects:
(1) **Format Adherence Deviation (FAD)**, which measures the absolute volatility in the number of generated chapters across multiple runs for chapter-based tasks. It assesses if the model consistently produces the required number of chapters: FAD q 
=
1 N
PN
i=1(Ci − µc)
2, where Ciis the number of chapters in the i-th generation, and µc is the average chapter count over N runs.

(2) **Structured Content Accuracy (SCA)**, this metric uses Execution-based Verification to assess accuracy on structured tasks, such as generating Python libraries and LaTeX formulas:
SCA =
Number of Correct Chapters Number of Required Chapters.

(3) **Unstructured Content Accuracy (UCA)**, following previous work Bai et al. (2024);
Zhang et al. (2025a), we use an LLM-as-a-Judge to evaluate unstructured tasks (e.g., story writing), with details in Appendix C.

## 4 Experiments And Results 4.1 Models

To systematically evaluate long-text generation capabilities, our study includes a diverse set of models. Specifically, we evaluate reasoning models such as GPT-4o mini, Claude 3.5 Sonnet, and Deepseek-R1 (DeepSeek-AI et al., 2025a). Our open-source selection includes models of various architectures and sizes: Qwen2.5-1.5B-Instruction, Qwen2.5-7B-Instruction (Qwen et al., 2025), Qwen3-8B (Team, 2025), Llama3.1-8B-Instruction, Deepseek-V3 (DeepSeek-AI et al., 2025b). We also include Falcon3-Mamba-7B–Instruction (Team, 2024), notable for its distinct architecture. We also include LongWriter-llama3.1-8B (Bai et al., 2024), a model enhanced for long-form generation via long-text post-training. Additionally, we incorporate common training-free decoding strategies for comparison, implemented on Qwen2.5-7B-Instruction. These include **Repetition Penalty** to mitigate text degeneration via logit penalization, **Entropy-Based Stopping** employing predictive uncertainty as a dynamic termination criterion, **Length Constraint** for enforcing explicit output boundaries, and **Lookahead Decoding**, designed to optimize the generation trajectory by anticipating future probabilities. To evaluate a model's ability to follow specific, localized instructions in long-form generation, we designed a framework using fine-grained constraints. This approach tests content control at a subdocument level, unlike typical global prompt-following evaluations. Specifically, we apply three distinct and simultaneous constraints to designated sections of the output. The constraints are defined as follows: Length Volatility. Unlike previous work Zhang et al. (2025b), which focuses on the volatility of a single generation, we measure a model's volatility across multiple outputs. Table 2: Performance comparison of evaluated models on a 100-section generation task, conducted in English under simple difficulty settings. Representative results are shown for an unstructured task (Story) and a structured task (Code Function). For the LSD and FAD metrics, the values in parentheses provide context by showing the generated mean length (in words) and mean section count, respectively. The "±" values represent the standard deviation. The arrows (↑/↓) indicate whether higher or lower values are preferable for each metric.

| Model              | Length Volatility   | Generation Quality   |         |               |                |                |
|--------------------|---------------------|----------------------|---------|---------------|----------------|----------------|
| LSD (↓)            | LVC (↓)             | MLA (↑)              | FAD (↓) | SCA (↑)       | UCA (↑)        |                |
| GPT-4o mini        | 325.65 (959)        | 33.9%                | 4.8%    | 1.41 (7.00)   | 84.6% (±30.8%) | 86.7% (±6.7%)  |
| Claude-3.5-Sonnet  | 3.30 (176)          | 1.9%                 | 0.9%    | 0.00 (2.00)   | 3.0% (±0.0%)   | 88.7% (±2.7%)  |
| Deepseek-R1        | 103.30 (1198)       | 8.6%                 | 6.0%    | 1.25 (4.33)   | 35.0% (±13.2%) | 93.3% (±3.7%)  |
| Deepseek-V3        | 40.76 (1854)        | 2.2%                 | 9.3%    | 1.70 (20.67)  | 48.6% (±3.8%)  | 84.7% (±3.4%)  |
| Mamba-7B           | 715.98 (1291)       | 55.5%                | 6.5%    | 41.72 (40.75) | 66.8% (±21.9%) | 76.0% (±17.3%) |
| Qwen2.5-1.5B       | 27.78 (142)         | 19.6%                | 0.7%    | 0.47 (1.67)   | 15.6% (±24.0%) | 84.0% (±7.1%)  |
| Qwen2.5-7B         | 75.87 (445)         | 17.0%                | 2.2%    | 2.05 (10.33)  | 99.8% (±0.4%)  | 86.7% (±7.6%)  |
| Llama3.1-8B        | 92.77 (350)         | 26.5%                | 1.7%    | 0.94 (4.33)   | 92.4% (±14.2%) | 82.0% (±18.9%) |
| LongWriter-8B      | 2866.3 (6320)       | 45.4%                | 31.6%   | 21.42 (45.00) | 32.6% (±31.9%) | 66.7% (±16.5%) |
| Repetition Penalty | 553 (2967)          | 18.6%                | 14.8%   | 5.4 (22)      | 98% (±1%)      | 76.7% (±14.5%) |
| Entropy-Stopping   | 713 (2701)          | 26.4%                | 13.5%   | 7.24 (24)     | 95% (±2.5%)    | 83.9% (±8%)    |
| Length Constraint  | 1280 (4470)         | 28.65%               | 22.4%   | 9.2 (28)      | 96% (±2%)      | 85% (±9%)      |
| Lookahead Decoding | 268 (2883)          | 9.3%                 | 14.4%   | 7.2 (25)      | 94% (±3.5%)    | 84.4% (±8%)    |

- *Character-level Pattern Constraint:* This constraint dictates that the first word of a target section must begin with a pre-determined, randomly selected alphabetical character. This tests the model's ability to control low-level textual attributes.

- *Keyword Presence Constraint:* This requires the mandatory inclusion of a specific, randomly selected keyword within the body of a target section. This evaluates the model's capacity to track and insert specific information into relevant contexts.

- *Specified Theme Constraint:* This imposes a thematic requirement, compelling the narrative or content of a target section to align with a randomly selected topic or scenario. This assesses the model's ability to generate coherent content based on a high-level concept.

## 4.3 Results And Analysis

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 Volatility Across Different Dimensions. As shown in Figure 3, we analyze model performance across three dimensions. On the language dimension, most models exhibit lower volatility and a greater mean output length in 5 runs when generating in English. Regarding instruction complexity, models produce longer outputs for simple instructions, likely due to greater creative freedom, which is also accompanied by higher volatility. In terms of output format, we observe an interesting trend where models generate longer and more stable text (i.e., less volatile) for structured tasks. We attribute this to structured tasks being governed by well-defined format constraints and internal logic, which provides stronger guidance for the generation process. This hypothesis is corroborated by Figure (d.2), which shows that models generally generate a greater number of sections for structured tasks. For complete experimental results and analysis, please refer to Appendix J. Long Text Quality Evaluation. For comparison, we exclude Claude-3.5-Sonnet due to its low mean length (176 words), insufficient for long-text evaluation. For other models, we assess generation quality and actual length, revealing distinct trade-offs. As shown in Table 2, GPT-4o-mini showed the best balance on structured tasks among longer-output models, with SCA 84.6%, low FAD, and 959-word output. LongWriter-8B generated the longest text (6320 words) but scored low on both SCA (32.6%) and FAD (21.42), indicating a quality–length trade-off. On unstructured tasks, Deepseek-R1 achieved the highest UCA (93.3%) with 1198 words, while LongWriter-8B again scored lowest (66.7%), prioritizing length over quality. In summary, all current models fail to jointly satisfy long-text length and high-quality generation. Generation Patterns of Length Volatility Our experiments reveal that baseline models consistently struggle with length and structural constraints in long-form generation. The failure rate is stark: when tasked with generating up to 50 sections, models failed in approximately half of the cases. For requests exceeding 50 sections, all models failed to complete the task as instructed. These failures typically manifest in two primary patterns:
- *Incomplete Generation*: Models frequently produce significantly less content than instructed. For example, when tasked with generating 40 sections, a model might stop after only 10. This premature termination, whether silent or reverting to a persona, with outputs like "I hope these sections are helpful." We hypothesize this latter behavior occurs when the generated text exceeds the context window, pushing the original prompt out of scope and causing the model to default to its base assistant persona.

- *Section Skipping*: In other instances, models demonstrate erratic adherence to the requested structure. A model might generate the first several sections sequentially and then abruptly jump to the final section, omitting all intermediate content.

## 4.3.1 Analysis Of Fine-Grained Constraint Following

To provide a quantitative view of the volatility in instruction adherence, we analyze model performance on the fine-grained constraint tasks. The complete results, including figures for all three constraint types, can be seen in Appendix D. As depicted in the figure, a clear trend emerges across all tested models. While most models, such as Deepseek-R1, Qwen3-8B and LLama3.1 adhere to constraints on shorter tasks (5-50 sections), their performance plummets and grows more volatile as the context length increases. This trend is universal, starkly contrasting the better models with Longwriter, which fails entirely regardless of length. Critically, even for the top models, the success rate flattens after the 100-section mark, and then actively collapses—with Qwen3-8b and LLama3.1 producing fewer correct sections at 500 than at 200. The systemic failure is most evident at the 500-section task: against a requirement of 100 constrained sections, no model delivered more than 40. This demonstrates a profound inability of current models to track and execute instructions deep within long-form generation.

## 5 Attention Traces Behind Volatility

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Attention Trace. To explore the root of output volatility, we analyze the attention mechanism in generation. Building on Li et al. (2025), who link attention to constraint tokens with instructionfollowing ability, we extend this to long-form generation. We hypothesize that attention fluctuations toward input constraints correlate with output variability. At each step t, where t ≥ 1, the model attends to prompt tokens x1:T0and generated tokens y0:t−1, where T0 indicates the length of prompt tokens. We focus on attention to constraint-encoding tokens in x1:T0. For layer l and head n, attention uses query Q
(l,t)
n from h
(l)
t−1(last generated token's hidden state) and keys K
(l,t)
n from h
(l)
1:T0+t−1
(hidden states of all prior tokens). The scaled dot-product attention weights A
(l,t)
n are then calculated as A
(l,t)
n = softmax Q(l,t)
n K(l,t)⊤
√ n dk where dk is the dimension of the key vectors.

These weights are then averaged across all N attention heads to obtain the layer-level attention vector a
(l,t) =
1 N
PN
n=1 A
(l,t)
n .

To measure attention directed toward constraints, we first identify the prompt token indices corresponding to each textual constraint r ∈ R, denoted as Cr. The full set of constraint token indices is given by C =Sr∈R Cr. The layer-step constraint attention α
(l,t)is then defined as the average attention from token yt to all tokens in C, i.e., α
(l,t) =1 |C| Pj∈C
a
(l,t)
j, where a
(l,t)
jis the attention weight at layer l and step t directed to the j-th token of the input. Finally, we average α
(l,t)across all L layers of the model to obtain a unified measure of constraint attention at each generation step, α
(t) =
1 L
PL−1 l=1 α
(l,t). By plotting the trace of α
(t)during generation, we visualize how attention to constraints evolves. Peaks and subsequent drops, "attention summits", may signal points where reduced constraint focus leads to task deviation and output volatility. To analyze this, we gener-

![7_image_0.png](7_image_0.png)

ate outputs with different random seeds and compare their attention traces to reveal links between attention dynamics and output variability.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 To mitigate generation volatility, we propose a dynamic decoding strategy that ensures stable, constraint-abiding outputs via single-pass generation. Rather than iterative prompts or multiple model calls, we modify logits in real time. At each step t, the model outputs a logit vector st ∈ R
|V | over the model's vocabulary V , which are adjusted by a guidance function M. Unlike standard decoding, M modifies logits based on context and rules to enforce structural and constraint adherence.

Formally, given the prompt tokens x1:T0and the generated token sequence y0:t−1 up to step t − 1, the modified logit vector s
′tis computed as:

$$s_{t}^{\prime}=M(s_{t},[x_{1:T_{0}};y_{0:t-1}]).$$
; y0:t−1]). (1)
The function M combines two guidance components: *structural enforcement*, which enforces adherence to the desired output structure, and *proactive failure prevention*, which applies a prohibitive negative bias to suppress likely failure modes during generation.

$\left(1\right)$. 

## 6 Mitigating Length Volatility

Internal Patterns of Length Volatility. We analyze the attention trace α
(t), which reveals internal patterns directly correlated with the earlier generation failures. As shown in Figure 2, where models are tasked with generating 40 sections, the traces highlight early internal signs of output volatility. From these, we identify two primary failure signatures: (1) *Attention Collapse*: This pattern aligns with premature termination or task abandonment. The Qwen2.5-3B trace illustrates this clearly: in the first 1,500 tokens, the model shows periodic attention spikes and follows instructions with wellstructured content. After that, attention collapses to near-zero, signaling loss of focus on prompt constraints and resulting in halted or irrelevant output; (2) *Attention Instability*: This pattern corresponds to erratic behaviors such as section skipping. In Qwen2.5-7B, initial regular attention spikes align with successful section generation. Around token 750, an abnormally large spike disrupts this pattern, immediately preceding the model's deviation from sequential output. In both cases, periodic attention spikes function as essential refocusing signals that help maintain task coherence across sections. Analysis of the α
(t)trace supports our hypothesis: the output volatility is not random but closely linked to and preceded by measurable failures in the model's internal attention dynamics.

## 6.1 Structural Enforcement Via Logits Boosting 6.2 Proactive Failure Prevention

Based on our analysis of generation patterns, we proactively suppress tokens associated with known failure modes by applying a strong negative bias during decoding. Formally, let Vbanned ⊂ V be the set of token indices corresponding to conversational filler phrases (e.g.,
"I hope these..."); and let veos be the index of the end-of-sentence token. The failure prevention function Mfail is defined as:

$$s^{\prime}_{t,j}=\begin{cases}-\infty&\text{if}j\in V_{\text{channel}}\\ -\infty&\text{if}j=v_{\text{eos}}\wedge p<P_{\text{total}}\\ s_{t,j}&\text{otherwise.}\end{cases}\tag{3}$$

This prevents undesirable conversational text and early termination before the final section.

By composing M = Mfail ◦ Mstruct, our method enables real-time control over generation, directly managing output probabilities to address length volatility while ensuring structural and constraint adherence in a single pass.

## 6.3 Results

![8_image_0.png](8_image_0.png)

Our method marks a major improvement in long-text generation, outperforming strong baselines like LongWriter-8B in stability, adherence, and quality. Evaluation was done on a 100-section task under simple settings. In output stability and length adherence, our model excels. As shown in Figure 6, its mean length and section count closely follow the reference line, unlike baselines that degrade as complexity rises. The Length Variation Coefficient (LVC), where lower is better, for our model is 14.02%, a 69% reduction in volatility compared to 45.4% for LongWriter-8B. Furthermore, our model's Mean Length Accuracy (MLA) is 78.25%, more than double the 31.6% achieved by LongWriter-8B, indicating a much closer adherence to the required length. This is reflected in the average output of 15,651 words from our model, compared to just 6,320 from LongWriter-8B and less than 1000 in other models. Our model also achieves higher generation quality. For Structured Content Accuracy (SCA), our model scored a perfect 100%, dramatically better than LongWriter8B's 32.6%, which has plenty of repeated tokens. To quantify this, we further analyze the lexical diversity in Appendix G, showing that our method significantly reduces n-gram repetition rates and improves the Type-Token Ratio (TTR) compared to baselines. This highlights its enhanced capability in handling structured tasks. Similarly, for Unstructured Content Accuracy (UCA), our model scored 86.7%, a 30% improvement over LongWriter-8B. These results underscore our method's ability to generate not only longer and more stable text but also higher-quality. Beyond surface-level metrics, we investigate the underlying mechanism of this stability in Appendix H. Through Representational Stability Analysis, we demonstrate that SELB effectively mitigates the 'representational drift' of hidden states, preventing the semantic collapse commonly observed in baseline models during long generation.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 To ensure generation of P*total* sections, we force a new section whenever the current section reaches the target length τmax. If the length of p-th section τp ≥ τmax, a strong positive bias β is applied to the logits of tokens corresponding to the next section title, V
(p+1)
title ⊂ V . The structural boosting adjustment, Mstruct, is then defined as:

, is then defined as:  $s'_{t,j}=\begin{cases}s_{t,j}+\beta&\text{if}\tau_p\geq\tau_{max}\wedge p<P_{total}\wedge j\in V^{(p+1)}_{\text{mile}}\\ s_{t,j}&\text{otherwise,}\end{cases}$ (2)
where st,j is the logit for token j at step t, and β is a large positive constant that makes the selection of a title token nearly certain. Once a token from V
(p+1)
title is generated, the section index is incremented (p ← p + 1) and the counter is reset (τp ← 0).

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514

## 515

516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## 6.4 Generalization To Free-Form Generation

We extend its applicability to free-form generation tasks (e.g., continuous novel writing) where such explicit anchors are absent. In Appendix I, we detail the adaptation of our approach into a SELB- Hybrid strategy. This mechanism addresses the twin challenges of premature termination and generation loops by dynamically shifting from section enforcement to length enforcement. Specifically, it incorporates an aggressive Stop Token Suppression module that prohibits early exit phrases and a Hybrid Keep-Alive mechanism. The latter monitors generation checkpoints, if a stall or repetitive loop is detected within a grace period, it proactively boosts generic continuation tokens to break the cycle and sustain narrative flow. The empirical impact of this adaptation is substantial. We evaluated the method on extreme-length free-form tasks, such as writing a 20,000-word novel. As detailed in Appendix I, baseline models including GPT-4o-mini and LongWriter-8B suffered from severe length collapse, often generating fewer than 600 words despite the 20k target. In contrast, our SELB-Hybrid method achieved a Mean Length Accuracy (MLA) of 97% with a remarkably low Length Variation Coefficient (LVC) of 12.1%. These results confirm that our logits-boosting paradigm can be effectively generalized beyond structured tasks to enforce stability in unstructured, open-ended generation scenarios.

## 7 Conclusion

In this work, we investigate the critical yet overlooked issue of output volatility in long-form LLM generation. Our findings show that instability across multiple outputs poses a major challenge to reliable application. To systematically study this problem, we first introduce VOLTBench, a novel benchmark to quantify length volatility across diverse tasks. By probing internal attention mechanisms, we identify common patterns that drive instability. Based on these insights, we propose SELB (Structural Enforcement via Logits Boosting), a lightweight, training-free decoding strategy to directly mitigate this issue. Extensive experiments confirm that severe output volatility is widespread in mainstream models and validate our approach, which improves the base model's mean output length by 148% and reduces length volatility by 69%, while maintaining generation quality.

## Reproducibility Statement

Our work addresses the output volatility in long-form text generation through a three-stage approach: benchmarking, probing, and mitigation. This includes three main contributions: (1) the VOlatility in Long-form Text Benchmark (VOLTBench); (2) an in-depth analysis of the internal causes of volatility; and (3) a lightweight decoding-stage optimization strategy, SELB. To ensure the full reproducibility of our findings, we have provided detailed documentation in the paper and its appendices. The construction methodology, data composition, and evaluation metrics for VOLTBench are thoroughly described in Section 3. The complete implementation details for our proposed SELB method and the full experimental setup, including all hyperparameters, are provided in Section 6. We commit to releasing the entire source code, the full VOLTBench benchmark, and our analysis scripts to the public upon acceptance of this paper to facilitate verification and future research.

## Ethics Statement

Our research adheres to the standard ethical guidelines for academic publishing. The work presented in this paper is foundational, focusing on the technical challenges of output volatility in Large Language Models. Our objective is to improve the reliability and stability of these models, which is a positive contribution to the field of artificial intelligence. The proposed benchmark, VOLTBench, is constructed from publicly available datasets and does not contain any personally identifiable or sensitive information. Our research did not involve human subjects, and we foresee no direct negative societal impacts from this work.

## References

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report, 2023. URL https://arxiv.org/abs/2309.16609. 1 540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Yushi Bai, Jiajie Zhang, Xin Lv, Linzhi Zheng, Siqi Zhu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longwriter: Unleashing 10,000+ word generation from long context llms, 2024. URL
https://arxiv.org/abs/2408.07055. 1, 1, 2, 3.2, 4.1, I, 25 Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Andrew Dai, Pu-Chin Chen, Jiaqi Pan, Asya Fadeeva, Zach Gleicher, Thang Luong, and Niket Kumar Bhumihar. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https://arxiv.org/abs/2507.06261. 1 Marta R. Costa-jussa, Pierre Andrews, Mariano Coria Meglioli, Joy Chen, Joe Chuang, David Dale, `
Christophe Ropers, Alexandre Mourachko, Eduardo Sanchez, Holger Schwenk, Tuan Tran, Arina ´ Turkatenko, and Carleigh Wood. Lcfo: Long context and long form output dataset and benchmarking, 2025. URL https://arxiv.org/abs/2412.08268. 2 DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, and et al Kai Dong. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025a. URL https://arxiv.org/abs/2501.12948. 4.1 DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, and et al Jin Chen. Deepseek-v3 technical report, 2025b. URL
https://arxiv.org/abs/2412.19437. 4.1 Team GLM, :, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. Chatglm: A family of large language models from glm-130b to glm-4 all tools, 2024. URL https://arxiv.org/abs/2406.12793. 1 Alon Jacovi, Andrew Wang, Chris Alberti, Connie Tao, Jon Lipovetz, Kate Olszewska, Lukas Haas, Michelle Liu, Nate Keating, Adam Bloniarz, Carl Saroufim, Corey Fry, Dror Marcus, Doron Kukliansky, Gaurav Singh Tomar, James Swirhun, Jinwei Xing, Lily Wang, Madhu Gurumurthy, Michael Aaron, Moran Ambar, Rachana Fellinger, Rui Wang, Zizhao Zhang, Sasha Goldshtein, and Dipanjan Das. The facts grounding leaderboard: Benchmarking llms' ability to ground responses to long-form input, 2025. URL https://arxiv.org/abs/2501.03200. 2 Xiaomin Li, Zhou Yu, Zhiwei Zhang, Xupeng Chen, Ziji Zhang, Yingying Zhuang, Narayanan Sadagopan, and Anurag Beniwal. When thinking fails: The pitfalls of reasoning for instructionfollowing in llms, 2025. URL https://arxiv.org/abs/2505.11423. 5 Xiang Liu, Peijie Dong, Xuming Hu, and Xiaowen Chu. Longgenbench: Long-context generation benchmark, 2024. URL https://arxiv.org/abs/2410.04199. 1, 2, 1 Chau Minh Pham, Simeng Sun, and Mohit Iyyer. Suri: Multi-constraint instruction following for long-form text generation, 2024. URL https://arxiv.org/abs/2406.19371. 2 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Shanghaoran Quan, Tianyi Tang, Bowen Yu, An Yang, Dayiheng Liu, Bofei Gao, Jianhong Tu, Yichang Zhang, Jingren Zhou, and Junyang Lin. Language models can self-lengthen to generate long texts, 2024. URL https://arxiv.org/abs/2410.23933. 2 Haoran Que, Feiyu Duan, Liqun He, Yutao Mou, Wangchunshu Zhou, Jiaheng Liu, Wenge Rong, Zekun Moore Wang, Jian Yang, Ge Zhang, Junran Peng, Zhaoxiang Zhang, Songyang Zhang, and Kai Chen. Hellobench: Evaluating long text generation capabilities of large language models, 2024. URL https://arxiv.org/abs/2409.16191. 2, 1 Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025.

URL https://arxiv.org/abs/2412.15115. 4.1 Haochen Tan, Zhijiang Guo, Zhan Shi, Lu Xu, Zhili Liu, Yunlong Feng, Xiaoguang Li, Yasheng Wang, Lifeng Shang, Qun Liu, and Linqi Song. Proxyqa: An alternative framework for evaluating long-form text generation with large language models, 2024. URL https://arxiv.org/ abs/2401.15042. 2 Falcon-LLM Team. The falcon 3 family of open models, December 2024. 4.1 Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

4.1 Y. Wang, D. Ma, and D. Cai. With greater text comes greater necessity: Inference-time training helps long text generation, 2024. URL https://arxiv.org/abs/2401.11504. 2 Yuhao Wu, Yushi Bai, Zhiqiang Hu, Roy Ka-Wei Lee, and Juanzi Li. Longwriter-zero: Mastering ultra-long text generation via reinforcement learning, 2025a. URL https://arxiv.org/ abs/2506.18841. 2 Yuhao Wu, Ming Shan Hee, Zhiqing Hu, and Roy Ka-Wei Lee. Longgenbench: Benchmarking long-form generation in long context llms, 2025b. URL https://arxiv.org/abs/2409. 02076. 1, 2, 1 Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, and Furu Wei. Differential transformer, 2025a. URL https://arxiv.org/abs/2410.05258. 1 Xi Ye, Fangcong Yin, Yinghui He, Joie Zhang, Howard Yen, Tianyu Gao, Greg Durrett, and Danqi Chen. Longproc: Benchmarking long-context language models on long procedural generation, 2025b. URL https://arxiv.org/abs/2501.05414. 2, 1 Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention, 2025. URL https://arxiv.org/abs/2502.11089. 1 Junhao Zhang, Richong Zhang, Fanshuang Kong, Ziyang Miao, Yanhan Ye, and Yaowei Zheng.

Lost-in-the-middle in long-text generation: Synthetic dataset, evaluation framework, and mitigation, 2025a. URL https://arxiv.org/abs/2503.06868. 2, 1, 3.2 Wei Zhang, Zhenhong Zhou, Kun Wang, Junfeng Fang, Yuanhe Zhang, Rui Wang, Ge Zhang, Xavier Li, Li Sun, Lingjuan Lyu, Yang Liu, and Sen Su. Lifebench: Evaluating length instruction following in large language models, 2025b. URL https://arxiv.org/abs/2505.16234. 1, 2, 1, 3.2 Zihan Zhou, Chong Li, Xinyi Chen, Shuo Wang, Yu Chao, Zhili Li, Haoyu Wang, Qi Shi, Zhixing Tan, Xu Han, Xiaodong Shi, Zhiyuan Liu, and Maosong Sun. LLM×MapReduce: Simplified long-sequence processing using large language models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting