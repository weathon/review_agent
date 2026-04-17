000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Large Language Models (LLMs) have made rapid progress in reasoning, question answering, and professional applications; however, their true capabilities remain difficult to evaluate using existing benchmarks. Current datasets often focus on simplified tasks or artificial scenarios, overlooking long-tail knowledge and the complexities of real-world applications. To address this gap, we propose LPFQA, a benchmark derived from authentic professional forums across 20 academic and industrial fields, covering 502 tasks grounded in practical expertise. LPFQA introduces four key innovations: fine-grained evaluation dimensions that target knowledge depth, reasoning, terminology comprehension, and contextual analysis; a hierarchical difficulty structure that ensures semantic clarity and unique answers; authentic professional scenario modeling with realistic user personas; and interdisciplinary knowledge integration across diverse domains. We evaluated 12 mainstream LLMs on LPFQA and observed significant performance disparities, especially in specialized reasoning tasks. LPFQA provides a robust, authentic, and discriminative benchmark for advancing LLM evaluation and guiding future model development.

## 1 Introduction

The rise of Large Language Models (LLMs) has been one of the most significant breakthroughs in the field of artificial intelligence over the past decade, impacting areas such as question answering Zhuang et al. (2023); Li et al. (2024b), reasoning Havrilla et al. (2024); Wang et al. (2023), code optimization Nam et al. (2024); Gu (2023); Fakhoury et al. (2024), and beyond. The ability of LLMs to handle complex tasks has enabled many previously unattainable applications, facilitating their rapid integration into both daily life and professional domains Yang et al. (2024); Zheng et al. (2025). As model architectures and training strategies continue to advance, the accurate and comprehensive evaluation of their true performance becomes increasingly crucial. The current approach involves employing benchmark tests, which are datasets composed of carefully designed questions or tasks. LLMs are required to generate answers or complete these tasks, and their performance is then quantitatively assessed based on the outcomes Chang et al. (2024). Given that a substantial portion of knowledge in the real world follows a long-tail distribution, which is often fragmented and highly professional, an effective evaluation benchmark should include such long-tail knowledge that is relatively underrepresented in pre-training data Zhang et al. (2023); Yang et al. (2022). Moreover, these questions must be grounded in real-world authenticity to better reflect actual user needs. However, existing benchmarks exhibit clear limitations. For instance, MMLU focuses primarily on simple question answering or multiple-choice tasks, which fail to evaluate a model's ability to handle complex, multi-step reasoning Wang et al. (2024); Hendrycks et al. (2021); HLE Phan et al. (2025) leverages human annotations to approximate human preferences, but its task scenarios are often overly idealized or uncommon, thus not representative of typical user demands. And Arena-Hard Li et al. (2024a), although capturing certain aspects of real user queries, suffers from limited diversity in question types and insufficient difficulty, making it less effective in differentiating performance among LLMs. To this end, we constructed a comprehensive evaluation benchmark (LPFQA) based on highly professional forums, which characterizes both real-world and long-tail knowledge. The data is collected from technical forums across multiple professional domains. This ensures that tasks of LPFQA are Anonymous authors Paper under double-blind review

## Abstract

# Lpfqa: A Long-Tail Professional Forum- Based Benchmark For Llms' Evaluation

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 highly professional, as they are based on complex questions raised by real practitioners with expertise in various fields. At the same time, the data is authentic, as it reflects the real needs and challenges encountered by users in practice. We completed this benchmark construction through three main phases, including (1) data collection and preprocessing, (2) automated question generation and quality control, and (3) expert verification and difficulty adjustment, ensuring that all selected questions fulfill the demands of the benchmark. LPFQA spans 20 academic fields, including Computer Science, Mathematics, Biology, Physics, etc., with a total of 505 questions. We evaluated LPFQA using 12 mainstream models, including GPT, Gemini, DeepSeek, Seed, Qwen, Grok, Claude, and Kimi. This work introduces LPFQA, an authentic, structured, and interdisciplinary dataset with long-tail knowledge for evaluating LLMs' ability in complex reasoning, providing a robust benchmark for assessing and advancing LLM performance in real-world professional contexts. The main innovations of LPFQA and contributions of this work can be summarized as follows.

- **Innovated evaluation dimension design**. We design a set of fine-grained evaluation dimensions, including knowledge depth, reasoning ability, terminology comprehension, and contextual analysis, ensuring LPFQA 's comprehensiveness in evaluating LLMs' capabilities in handling complex tasks.

- **Hierarchical difficulty design with guaranteed uniqueness**. We employ a tiered difficulty structure to match varying capabilities of different LLMs, while ensuring semantic clarity and answer uniqueness for each task, enhancing the reliability, fairness, and discriminative power of LPFQA.

- **Authentic professional scenario modeling**. We ground questions in authentic use cases by constructing detailed user personas and realistic contextual scenarios, enhancing the ability of LPFQA to validate the performance of LLMs in real-world professional environments.

- **Interdisciplinary knowledge integration**. We integrate long-tail knowledge from diverse fields, improving the LPFQA's effectiveness in evaluating LLMs' integrative capabilities of judgment and reasoning in complex scenarios.

## 2 Related Work

The field of large language model evaluation has seen a rapid proliferation of benchmarks, each designed to probe different facets of model capabilities. Early benchmarks, such as GLUE Wang et al. (2018) and SuperGLUE Wang et al. (2019), focused on a broad range of general language understanding tasks, including question answering and natural language inference. While these benchmarks were instrumental in driving early progress, they are now often considered insufficient for evaluating the nuanced reasoning and vast knowledge base of modern, more capable LLMs. Subsequent benchmarks, such as MMLU Wang et al. (2024), BIG-bench Srivastava et al. (2023), and HELM Liang et al. (2022), extended evaluation to multi-disciplinary knowledge, reasoning, and holistic dimensions of safety, robustness, and fairness. Despite their contributions, these benchmarks still fall short in capturing the challenges of specialized knowledge and complex reasoning, motivating the exploration of new evaluation paradigms.

## 2.1 Long-Tail Knowledge Benchmarks

In the real world, data distributions universally exhibit a long-tail characteristic. This implies that a small number of "head" categories account for a significant portion of the data, while the vast majority of "tail" categories are extremely rare. In the context of LLMs, such a distributional imbalance is crucial because the large pre-training corpora, while massive, often lack sufficient coverage of this rare, specialized, or infrequently mentioned "tail" knowledge. As a result, while LLMs demonstrate robust performance on common topics, their ability to handle this long-tail information can decline significantly. To assess a model's capabilities on long-tail knowledge, researchers have designed specialized benchmarks. The construction methods for these benchmarks primarily fall into two categories: the first is natural data collection, where data is obtained directly from the real world. An example is biodiversity datasets (e.g., iNaturalist Van Horn et al. (2018)), where a large number of species have very few image samples. This approach captures the most authentic distributions, but data collection is often costly. The second method is synthetic construction, where long-tail distributions are artificially created by imbalanced sampling from existing, balanced datasets (e.g., ImageNet-LT from ImageNet Liu et al. (2019)). While this method is straightforward, it may not fully simulate the complexity and diversity of real-world long-tail data. Although the above benchmarks lay a foundation for evaluating long-tail knowledge, their tasks are often overly simplistic or confined to a few specific domains Liang et al. (2022). These limitations underscore the necessity of developing complementary benchmarks.

## 2.2 User-Centric And Challenging Benchmarks

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 In contrast to static long-tail knowledge evaluation, another important class of evaluation methods focuses on a model's performance on dynamic tasks. Chatbot Arena Chiang et al. (2024), for example, is a crowdsourcing platform that evaluates model performance through user blind testing. Its core idea is to have users engage with two anonymous LLMs and vote for the one that performs better. This method effectively captures user preferences and measures a model's overall performance in open-ended conversations. However, crowdsourced evaluation methods like Chatbot Arena also have clear limitations. First, they lack control over specific difficulty or expertise levels. User-submitted questions can be too simple, leading to similar responses from all top-tier models, which makes the benchmark less discriminative. For instance, Arena-Hard Li et al. (2024a) aims to address this issue with adversarial questioning, but its question types can still be relatively concentrated, making it difficult to fully assess a model's capabilities on a broader range of complex, professional long-tail knowledge. To further test the limits of a model, the Humanity's Last Exam (HLE) Phan et al. (2025) has emerged. HLE is designed to test an LLM's general intelligence and advanced reasoning by collecting extremely difficult questions that even human experts find challenging to answer. These questions typically require cross-disciplinary knowledge integration, complex logical reasoning, and deep comprehension. However, this benchmark also has its limitations. While the questions in HLE are highly challenging, their source and nature may not represent the day-to-day needs of average users. This makes it less effective in evaluating a model's practicality in real-world applications. Furthermore, its extreme difficulty may lead to poor performance from most models, thus limiting its utility as a regular evaluation tool. Through the analysis above, we recognize the limitations of existing benchmarks. Long-tail knowledge benchmarks lack consideration for complex tasks, while conversational evaluation benchmarks are deficient in terms of domain-specific expertise and difficulty control. Extreme benchmarks like HLE can test a model's cutting-edge capabilities, but their questions have weak relevance to everyday application scenarios. To bridge these gaps, our work aims to construct a new benchmark that can effectively evaluate a model's complex reasoning abilities on professional long-tail knowledge while also reflecting the demands inherent in real-world scenarios.

## 3 Lpfqa: Long-Tail Knowledge-Based Benchmark

In this section, we begin with an overview of LPFQA, describing its structure and highlighting its advantages over previous works. Then, we present the detailed steps involved in constructing LPFQA.

## 3.1 Overview

LPFQA is a long-tail knowledge benchmark, which consists of 505 questions across 20 scientific fields gathered from multiple real professional technical forums, specifically designed for complex reasoning. The following features can distinguish this benchmark. Diversity evaluation dimension. The ability to handle complex tasks is critical for LLMs. To enable the assessment of this ability, LPFQA innovatively covers tasks across multiple evaluation dimensions, including depth of knowledge, reasoning ability, understanding of professional terminology, and contextual analysis.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

![3_image_0.png](3_image_0.png)

Discriminative ability and unambiguous guarantee. To ensure the validity and accuracy of the evaluation results, a benchmark must be discriminative enough to differentiate the abilities of various LLMs, while each task should also be clearly defined. To this end, after careful selection, the tasks in LPFQA can be categorized into distinct levels of difficulty, designed to reflect characteristics suitable for LLMs of varying capabilities. Furthermore, the clarity of each task and the uniqueness of its corresponding answer are guaranteed.

Derived from real-world scenarios. To effectively evaluate the response and reasoning capabilities of LLMs in real-world scenarios, a benchmark must closely reflect the types of questions that users genuinely encounter. LPFQA is designed with this objective in mind, emphasizing authentic professional tasks derived from real discussions in technical forums. This design ensures that the tasks are representative of practical situations, thereby enabling a more accurate and realistic evaluation of LLM performance in real-world applications. Diversity domains knowledge. Moreover, LPFQA integrates tasks from a broad spectrum of professional technical forums, spanning domains such as biology, finance, materials science, and computer science. This cross-disciplinary benchmark challenges LLMs to demonstrate comprehensive judgment and reasoning across diverse and complex scenarios.

## 3.2 Construction Of Lpfqa

This work develops a fully automated pipeline for constructing such an authentic cross-disciplinary benchmark from professional technical forums. In detail, the whole construction consists of eight steps: ❶ collecting professional forums, ❷ scraping discussion links, ❸ capturing screenshots of discussions, ❹ generating questions from the screenshots using MLLMs, ❺ cleaning up duplicated and ambiguous items with LLMs, ❻ transitioning them into multiple-choice or short-answer form, ❼ verifying all questions by professional experts, and ❽ filtering questions by difficulty through empirical testing, finally. These steps can be divided into three phases: data collection and preprocessing, automated question generation and quality control, and difficulty adjustment and expert review. This three-phase process follows the natural progression of building a benchmark from raw data to a standardized and highquality benchmark, ensuring both scalability and reliability.

## 3.2.1 Data Collection And Preprocessing

The first phase addresses the challenge of sourcing diverse and representative raw materials. We manually selected and crowd-sourced several professional forums that represent different disciplines, ensuring coverage across domains such as biology, finance, materials science, and computer science (❶). We developed a customized web crawler to collect forum data at scale. The crawler is capable of adapting to heterogeneous forum structures and supports filtering by metadata such as time, view count, reply count, and vote count, which helps control both the quality and relevance of the collected data (❷). To facilitate later multi-modal content analysis, automated scripts visited each post page and captured screenshots in addition to extracting textual content. This process not only preserved contextual and visual information but also provided a reliable basis for subsequent processing (❸).

## 3.2.2 Automated Question Generation And Quality Control

The second phase focuses on transforming raw forum content into structured question–answer pairs. The MLLM first examined each screenshot to determine whether it contained a valid question. Screenshots without valid questions were discarded, while those with valid content proceeded to the next stage. If a post included meaningful replies, the model extracted both the question and key responses to form candidate question–answer pairs; otherwise, only the question itself was retained
(❹).

These items then underwent automated quality control with the aid of an LLM. The process included duplicate removal, filtering of incomplete or ambiguous entries, and marking with labels such as domain, clarity, and difficulty. Logical consistency was also checked to ensure alignment between questions and their corresponding answers (❺).

Finally, the validated question–answer pairs were transmitted into multiple-choice or short-answer format. For multiple-choice items, the LLM generated distractor options designed to resemble common errors or misconceptions. For short-answer items, in addition to the correct reference answer, a set of key knowledge points was also provided, which serves as the criterion for determining whether a response is correct. This transition enhanced the usability of the dataset while maintaining both clarity and evaluation effectiveness (❻).

## 3.2.3 Expert Verification And Difficulty Adjustment

The third phase ensures that the question bank achieves a balanced level of difficulty and scientific correctness. First, the generated items underwent a human verification by the professional experts. They verify the factual accuracy, relevance, and difficulty of each item, while also correcting residual errors introduced during the automated pipeline. This operation enhanced the scientific rigor and reliability of our benchmark (❼).

Finally, to improve the benchmark's ability to differentiate LLMs' capabilities, we conduct an empirical difficulty test. Multiple LLMs were employed to answer all questions, and their accuracy rates were recorded to classify the items into different difficulty levels. The dataset was adjusted by selectively adding or removing items, ensuring a well-balanced difficulty structure (❽).

By integrating the above steps, namely data collection and preprocessing, automated question generation with quality control, and difficulty adjustment with expert review and empirical test-based evaluation, the proposed pipeline achieves end-to-end automation while maintaining high standards of reliability and evaluation utility. This design provides a scalable and systematic approach for constructing a question dataset that faithfully represents real-world professional discourse and is well-suited for LLM evaluation.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

![5_image_0.png](5_image_0.png)

Table 1: Performances of different models on LPFQA.

Table 2: Scores of different models on filtered LPFQA.

| Models         | Score   |        |         |         |
|----------------|---------|--------|---------|---------|
| Qwen-3         | 38.78   |        |         |         |
| Grok-4         | 39.04   |        |         |         |
| DeepSeek-R1    | 38.25   |        |         |         |
| Seed-1.6       | 41.50   |        |         |         |
| Gemini-2.5-Pro | 44.42   |        |         |         |
| GPT-4.1        | 38.31   |        |         |         |
| GPT-4o         | 32.40   |        |         |         |
| o3-high        | 43.03   |        |         |         |
| Claude-4       | 38.05   |        |         |         |
| GPT-5          | 47.28   |        |         |         |
| Kimi-K2        | 35.26   |        |         |         |
| DeepSeek-V3    | 32.60   |        |         |         |
| Average        | 39.08   | Models | LPFQA − | LPFQA = |
| Qwen-3         | 44.65   | 42.62  |         |         |
| Grok-4         | 44.95   | 42.37  |         |         |
| DeepSeek-R1    | 44.04   | 41.89  |         |         |
| Seed-1.6       | 47.78   | 45.84  |         |         |
| Gemini-2.5-Pro | 51.15   | 49.64  |         |         |
| GPT-4.1        | 44.11   | 42.45  |         |         |
| GPT-4o         | 37.31   | 35.03  |         |         |
| o3-high        | 49.54   | 48.10  |         |         |
| Claude-4       | 43.81   | 41.57  |         |         |
| GPT-5          | 54.43   | 53.11  |         |         |
| Kimi-K2        | 40.60   | 38.58  |         |         |
| DeepSeek-V3    | 37.54   | 35.59  |         |         |
| Average        | 44.99   | 43.07  |         |         |

As depicted in Figure 2, LPFQA covers 20 academic fields with a total of 505 questions, including Computer Science (CS), *Mathematics* (Math), *Biology* (Bio), *Physics* (Phys), *Electronic Information* Engineering (EIE), *Chemistry* (Chem), *Electronic Science and Technology* (EST), *Finance* (Fin), Mechanical and Automation (Mech), *Artificial Intelligence and Machine Learning* (AI), Computer Systems and Software (CSS), *Miscellaneous* (Misc), *General Engineering* (Eng), *Aerospace* (Aero), Law, *Medical* (Med), *Data Science and Big Data Technology* (DS), *Energy* (En), *Electronics and* Information Science (EIS), and *Information and Communication Engineering* (ICE). Among them, Physics, *Mathematics*, and *Biology* contain the largest number of items, each exceeding 60, while most of the other fields fall within the 10–50 range, and the field of *Data Science and Big Data* Technology has a relatively smaller number, with 3 items. Based on LPFQA, we evaluate the following mainstream models: Qwen-3-235B Yang et al. (2025), Grok-4 xAI (2025), DeepSeek-R1 Guo et al. (2025), Seed-1.6-Thinking Volcengine (2024), Gemini2.5-Pro Comanici et al. (2025), GPT-4.1 OpenAI (2024a), GPT-4o OpenAI (2024b), o3-high OpenAI (2024c), Claude-4-Sonnet Anthropic (2024), GPT-5 OpenAI (2025), Kimi-K2 Team et al. (2025), and DeepSeek-V3 Liu et al. (2024). All results provided are averaged over three trials.

## 4 Experiments 3.3 Statistics Of Lpfqa

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

![6_image_0.png](6_image_0.png)

![6_image_1.png](6_image_1.png)

ICE

Fin 

## 4.1 Main Results

As shown in Table 1, the performance of the evaluated models on LPFQA falls within a relatively narrow range, with scores spanning from 32.40 to 47.28. Among them, GPT-5 achieves the highest score, while GPT-4o records the lowest. To provide a more fine-grained comparison, Figures 3 report the scores of individual models across different fields, offering a clearer picture of their strengths and weaknesses in specific areas. The overall average performance of all models is further summarized in Figure 4a, which provides a holistic perspective on their general capability across fields. Finally, to highlight the comparative extremes, Figures 4b and 4c identify the models that achieve the maximum and minimum scores in each field, thereby providing an intuitive view of their relative advantages and limitations.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Based on the results presented in Figures 3 and 4, we analyze the performances of these models from three perspectives: overall performance, disciplinary distribution, and extreme values across models.

- **Overall performance**. Among all evaluated systems, DeepSeek-V3 demonstrates the most balanced and consistent performance across disciplines, with no apparent weaknesses, and can thus be regarded as the overall best-performing model. GPT-5 exhibits strong competitiveness, achieving the highest scores in several domains such as AI, Phys, EIS, Chem, Fin, and CSS, in some cases surpassing DeepSeek-V3. Seed-1.6 and GPT-4.1 also achieve competitive results in specific domains (e.g., CS, Aero, Bio for Seed-1.6; EIT, En for GPT- 4.1), though their overall performance remains less comprehensive. Other models, such as Claude-4-Sonnet, Grok-4, and Kimi-K2, tend to show domain-specific strengths but also exhibit noticeable weaknesses, limiting their overall robustness.

- **Disciplinary perspective**. From a disciplinary perspective, clear differences emerge across fields. As shown in Figure 4a, Misc yields the highest average scores (above 50), while En records the lowest overall average (around 20). Other relatively strong domains include Chem, AI, Fin, CS, and EIS, while weaker performance is observed in Med, Law, Eng, and Bio. Intra-model variation is also significant. For example, DeepSeek-R1 attains leading scores in DS, Math, Eng, and Law, but remains comparatively weak in ICE. Similarly, GPT-5 shows clear superiority in Phys and AI, while its performance in Law is less competitive. These disparities indicate that current models continue to face challenges in achieving uniform cross-disciplinary generalization.

- **Max and Min scores**. To provide a comprehensive view beyond average performance, we examine maximum and minimum scores across all disciplines (Figures 4b and 4c). For maximum scores: AI, Phys, EIS, Chem, Fin, and CSS are led by GPT-5; CS, Aero, and Bio by Seed-1.6; DS, Math, Eng, and Law by DeepSeek-R1; EIT and En by GPT-4.1; EIE by Claude-4-Sonnet; ICE by OpenAI-o3-high; and Misc by Grok-4. For Minimum scores: GPT-4o accounts for the lowest performance in multiple domains (Math, Chem, Fin, CSS, CS, Aero, En, and EIS). Other models show more localized weaknesses: Claude-4-Sonnet in DS and Eng, DeepSeek-R1 in Mech and ICE, OpenAI-o3-high in Bio, Qwen-3 in EIT, Grok-4 in EIE, Kimi-K2 in Med, and DeepSeek-V3 in Misc.

## 4.2 Detail Analysis 4.2.1 Filtered Lpfqa

During our analysis, we observed that none of the evaluated models could correctly answer a subset of questions. Since one of the primary purposes of the benchmark is to differentiate the capabilities of different models, these questions provide little discriminatory value. Therefore, we first excluded them from LPFQA, leaving a remaining set of 436 items. This filtered version, denoted as LPFQA−, was then used to recalculate the distribution of questions across different fields (Figure 5)
and the corresponding scores of each model (Table 2).

![7_image_0.png](7_image_0.png)

## 4.2.2 Ablation Analysis Does Lpfqa Evaluate Knowledge Or Reasoning Ability? 5 Conclusion

Models Score ∆ Qwen-3 23.31 15.47%↓

DeepSeek-R1 33.60 4.65%↓

Seed-1.6 37.58 3.92%↓

Gemini-2.5-Pro 35.19 9.23%↓

GPT-4.1 36.32 1.99%↓

GPT-4o 32.60 0.20%↑ o3-high 42.71 0.32%↓

GPT-5 45.18 2.10%↓

Kimi-K2 35.52 0.26%↑

DeepSeek-V3 28.08 4.51%↓

Average 35.01 10.64%↓

| Models         | Score   | ∆      |
|----------------|---------|--------|
| DeepSeek-R1    | 34.46   | 3.79%↓ |
| Gemini-2.5-Pro | 34.46   | 9.96%↓ |
| DeepSeek-V3    | 28.42   | 4.18%↓ |

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

In addition, we identified another subset of questions that were answered correctly by all models without exception. While such questions may reflect fundamental or widely shared knowledge, they also contribute minimally to distinguishing the relative strengths and weaknesses of the models. To further emphasize the performance gaps, we excluded these universally solvable questions based on LPFQA−, resulting in a remaining set of 421 items. This second filtered version is denoted as LPFQA=, on which we recomputed both the distributions across different fields (Figure 5) and the model scores (Table 2). We investigated the effect of integrating a Jupyter Code Interpreter (CI) into the reasoning process, which is expected to enhance reasoning ability through code execution. However, as shown in Table 3, it can be observed that overall performance decreased: the scores dropped on most models, and the few improvements that appeared were marginal, leading to a lower overall average. These findings suggest that LPFQA primarily reflects a model's mastery of domain knowledge rather than its reasoning ability.

## Is Deep-Search Always Rewarding?

We incorporated GoogleSearch and TextBrowserView tools into the reasoning process to enable information retrieval. As shown in Table 4, the scores of most models decreased under this setting. We attribute this phenomenon to the nature of LPFQA, which consists of long-tail knowledge that is inherently difficult to retrieve from the web. In such cases, the additional retrieval functions may introduce misleading information during the reasoning process, thereby reducing overall inference accuracy. In other words, for tasks involving long-tail knowledge, simply augmenting models with online search does not provide a positive effect and may even be detrimental. This observation offers valuable insights into the limitations faced by all models when dealing with long-tail knowledge. In this work, we proposed LPFQA, a long-tail professional forum-based benchmark designed to evaluate LLMs on complex reasoning and specialized knowledge across 20 domains. LPFQA emphasizes authenticity, interdisciplinarity, and fine-grained evaluation dimensions, with hierarchical difficulty and expert verification ensuring reliability and fairness. Our experiments on 12 mainstream LLMs reveal notable disparities, highlighting the persistent challenge of long-tail knowledge. Furthermore, ablation studies show that LPFQA primarily reflects domain knowledge mastery, and that direct integration of external tools does not always enhance performance. Overall, LPFQA provides a robust, discriminative, and authentic benchmark that not only measures current model capabilities but also guides future research toward more generalizable and reliable LLMs.

## Ethics Statement

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 This study is based on publicly available professional forum data, which was collected, filtered, and processed in compliance with relevant ethical standards. No personally identifiable or sensitive information was included in the benchmark. All data used were anonymized and only retained for research purposes. The benchmark construction and experiments were conducted strictly for academic evaluation and model analysis, without any intention of infringing on privacy, spreading harmful content, or causing potential misuse. We affirm that this research adheres to the ethical principles of fairness, transparency, and responsible AI development.

## Reproducibility Statement

To foster transparency and facilitate reproducibility, we will release our benchmark to the public. Furthermore, we provide the details of the benchmark construction process in the appendix, including: (1) all prompts used for question generation, (2) the prompts applied for evaluation criteria, and (3) the complete list of forums utilized. We believe these resources will enable the community to faithfully reproduce our results and build upon our work.

## References

Anthropic. Claude-4-sonnet. https://www.anthropic.com/news/claude-4, 2024. Accessed: 2025-09-17.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM transactions on intelligent systems and technology, 15(3):1–45, 2024.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E Gonzalez, et al. Chatbot arena: An open platform for evaluating llms by human preference. In Forty-first International Conference on Machine Learning, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. *arXiv preprint arXiv:2507.06261*, 2025.

Sarah Fakhoury, Aaditya Naik, Georgios Sakkas, Saikat Chakraborty, and Shuvendu K Lahiri. Llmbased test-driven interactive code generation: User study and empirical evaluation. IEEE Transactions on Software Engineering, 2024.

Qiuhan Gu. Llm-based code generation method for golang compiler testing. In *Proceedings of the* 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pp. 2201–2203, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

Alex Havrilla, Sharath Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravynski, Eric Hambro, and Roberta Raileanu. Glore: when, where, and how to improve llm reasoning via global and local refinements. In *Proceedings of the 41st International Conference on Machine* Learning, pp. 17719–17733, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. *arXiv preprint arXiv:2406.11939*, 2024a.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, and Jianyong Wang. Flexkbqa: A flexible llm-powered framework for few-shot knowledge base question answering. In *Proceedings of the AAAI conference on artificial intelligence*, volume 38, pp. 18608–18616, 2024b.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. *arXiv preprint arXiv:2211.09110*, 2022.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X Yu. Largescale long-tailed recognition in an open world. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2537–2546, 2019.

Daye Nam, Andrew Macvean, Vincent Hellendoorn, Bogdan Vasilescu, and Brad Myers. Using an llm to help with code understanding. In Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, pp. 1–13, 2024.

OpenAI. Gpt-4.1. https://openai.com/index/gpt-4-1/, 2024a. Accessed: 2025-0917.

OpenAI. Gpt-4o system card. https://openai.com/index/gpt-4o-system-card/,
2024b. Accessed: 2025-09-17.

OpenAI. Introducing o3 and o4-mini. https://openai.com/index/
introducing-o3-and-o4-mini/, 2024c. Accessed: 2025-09-17.

OpenAI. Introducing GPT-5. Online publication, 2025. URL https://openai.com/index/
introducing-gpt-5.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity's last exam. *arXiv preprint* arXiv:2501.14249, 2025.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adri Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. *Transactions on* machine learning research, 2023.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8769–8778, 2018.

Volcengine. Seed-1.6-thinking. https://www.volcengine.com/docs/82379/
1593703, 2024. Accessed: 2025-09-17.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman.

Glue: A multi-task benchmark and analysis platform for natural language understanding. *arXiv* preprint arXiv:1804.07461, 2018.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. *Advances in neural information processing systems*, 32, 2019.

Boshi Wang, Xiang Yue, and Huan Sun. Can chatgpt defend its belief in truth? evaluating llm reasoning via debate. In *EMNLP (Findings)*, 2023.

## Appendix A Llm Usage Statement

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Yifan Zhang, Bingyi Kang, Bryan Hooi, Shuicheng Yan, and Jiashi Feng. Deep long-tailed learning:
A survey. *IEEE transactions on pattern analysis and machine intelligence*, 45(9):10795–10816, 2023.

Yizhen Zheng, Huan Yee Koh, Jiaxin Ju, Anh TN Nguyen, Lauren T May, Geoffrey I Webb, and Shirui Pan. Large language models for scientific discovery in molecular property prediction. Nature Machine Intelligence, pp. 1–11, 2025.

Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. Toolqa: A dataset for llm question answering with external tools. *Advances in Neural Information Processing Systems*, 36: 50117–50143, 2023.

In the preparation of this manuscript, we employed LLMs solely for textual polishing and language refinement. The motivation, research design, etc., were independently conducted by the authors.

## B Examples Q&A Of Lpfqa Q&A 1, Field: General Engineering

Question: When only 110V service is available for a Millport milling machine with a 220V single-phase motor, which power supply solution is recommended, and what key factor must be considered for selecting this equipment? A. A voltage regulator with adjustable output, focusing on maximum current capacity alone. B. A three-phase to single-phase converter with 110V input, needing to match the motor speed rating. C. A Variable Frequency Drive (VFD) with single-phase input and three-phase output, requiring matching the motor's horsepower (HP) rating and current requirements.

D. A step-up transformer with single-phase input and single-phase output, requiring matching the voltage ratio only. E. A capacitor-start motor conversion kit, requiring compatibility with motor phase configuration. F. A DC power supply with inverter function, needing to match the motor frequency range. Answer: C
Lu Yang, He Jiang, Qing Song, and Jun Guo. A survey on long-tailed visual recognition. International Journal of Computer Vision, 130(7):1837–1872, 2022.

Ziqi Yang, Xuhai Xu, Bingsheng Yao, Ethan Rogers, Shao Zhang, Stephen Intille, Nawar Shara, Guodong Gordon Gao, and Dakuo Wang. Talk2care: An llm-based voice assistant for communication between healthcare providers and older adults. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 8(2):1–35, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

xAI. Grok-4. Large language model, 2025. URL https://x.ai/. Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. *Advances in Neural Information Processing Systems*, 37:95266–95290, 2024.