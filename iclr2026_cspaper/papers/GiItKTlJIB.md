000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# How Much Chain-Of-Thought Do Llms Really Need For Physics?

Anonymous authors Paper under double-blind review

## Abstract

Reasoning-focused language models are increasingly applied to AI for science, but evaluation has not kept pace: benchmarks largely measure end-task accuracy while ignoring whether models genuinely depend on their own reasoning traces.

This gap is critical in domains like physics problem solving, where equations, units, and structured terminology make reasoning reliability both essential and testable. We introduce a systematic deletion framework that intercepts chain-ofthought (CoT) mid-generation, removes tokens, and measures downstream effects. Applied to three open-source models—Magistral, Phi-4, and Qwen-A3B—across multiple physics benchmarks, our method shows that models remain accurate under heavy deletions (40–60%) by "cramming" reconstructed steps into final answers. Overlap analyses reveal that deleted equations and facts often reappear, but inconsistently across strategies, exposing shallow and opportunistic reliance on CoT. These findings underscore that current accuracy-based evaluations are insufficient for scientific domains, and point toward the need for methods that assess reasoning faithfulness as a core requirement for advancing AI for science.

## 1 Introduction

Large language models (LLMs) are increasingly presented not only as generators of fluent text but as reasoning systems, capable of solving multi-step problems in mathematics, science, and beyond (Yao et al., 2023; OpenAI et al., 2024). A central technique behind this framing is *chain-of-thought* (CoT) prompting, which elicits step-by-step reasoning traces prior to a final answer (Wei et al., 2022a; Kojima et al., 2022). Yet a key question remains: do models genuinely *depend* on these traces, or do they function mainly as scaffolding for answer generation? While CoT has been argued to provide partial monitorability of internal processes (Korbak et al., 2025), evidence suggests limited dependence. Models can output correct answers while producing unfaithful reasoning traces (Turpin et al., 2023); correctness alone does not establish whether reasoning was used (Lanham et al., 2023); and in many cases, models regenerate plausible but unused intermediate steps (Lyu et al., 2023). This distinction is critical: faithfulness in CoT is not equivalent to interpretability or explainability (Barez et al., 2025), but rather concerns whether the scratchpad faithfully represents the computations that yield the final answer. We investigate this faithfulness gap—and the broader evaluation gap of LLM reasoning—in the context of *physics problem solving*. While prior work has examined CoT faithfulness in general settings, its implications for *AI-for-Science* remain underexplored. Physics provides a stringent testbed: unlike open-ended reasoning tasks, it requires precise manipulation of equations, units, and numerical calculations, where small errors propagate into incorrect results (Shapira et al., 2023; Kosinski, 2024). At the same time, physics is central to visions of domain-specialized foundation models (Barman et al., 2025), making it both scientifically important and methodologically revealing. More broadly, physics exemplifies the reliability challenges facing *AI-for-Science*, where robust reasoning is essential for reproducibility, hypothesis generation, and discovery across disciplines (Bommasani et al., 2023; Stevens et al., 2023; Eger et al., 2025). To this end, we evaluate three recent reasoning-oriented LLMs—Magistral (Rastogi et al., 2025), Phi-4 (Abdin et al., 2024), and Qwen-A3B (Qwen, 2025)—on three physics benchmarks of varied difficulty: Undergraduate Physics (Xu et al., 2025), PhyBench (Meng et al., 2024), and PhysReason (Zhang et al., 2025). Our study proceeds in three stages: (1) establishing baseline performance un1

![1_image_0.png](1_image_0.png)

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 We systematically probe how LLMs use CoT reasoning in physics problem solving by actively intercepting and selectively deleting intermediate scratchpad prior to decoding. These CoT deletion experiments allow us to assess whether scratchpads are faithfully consumed, how models respond to partial removal of reasoning steps, and the extent to which missing information is reconstructed in the final outputs. An overview of our methods and evaluation metrics is presented in Figure 1. der direct and CoT prompting; (2) introducing a systematic deletion framework that intercepts CoT traces mid-generation and removes tokens before decoding; and (3) conducting a rigorous faithfulness analysis using information-overlap metrics and domain-aware matching to test whether deleted content reappears in final answers. Together, these steps provide a structured characterization of how open-source reasoning models use—or bypass—their CoT traces in scientific problem solving, exposing a reasoning-dependence gap that motivates new evaluation protocols and model designs emphasizing not only accuracy but also fidelity, with direct implications for AI-for-Science. In summary, our work introduces deletion-based probing as a new methodology for evaluating reasoning dependence in scientific domains, and applies it to physics as a structured, high-stakes testbed. This framework yields both methodological advances and empirical insights into the limits of chain-of-thought reasoning.

1. **A systematic deletion framework** for probing reasoning dependence in LLMs. Our framework introduces a simple yet novel evaluation paradigm: intercepting CoT mid-generation, deleting intermediate tokens, and measuring their downstream impact on decoded information funneling and final answer quality.

2. **An empirical characterization of robustness and cramming**, showing that accuracy remains stable under moderate deletions (up to ∼40–60%) before collapsing, and that models exhibit compensatory "cramming" behavior—producing longer final answers that attempt to reconstruct missing reasoning.

3. **A rigorous faithfulness analysis** leveraging the structured nature of physics and mathematics. Using overlap metrics (Jaccard and Manhattan distance), we compare original CoT traces with regenerated reasoning across deletion sweeps. The domain's clear structureequations, units, and terminology—enables precise quantification, revealing that models often reintroduce deleted content, producing surface-level agreement without genuine reasoning dependence.

These contributions highlight both the promise and the pitfalls of current reasoning models in scientific domains. They underscore the need for evaluations—and ultimately model designs—that prioritize *faithfulness* in reasoning, not just accuracy, with broader implications for AI-for-Science and structured problem solving.

## 2 Problem Setup 2.1 Tasks And Datasets

We evaluate on three physics benchmarks of increasing difficulty: UG Physics (easiest), PhysReason (intermediate), and PhyBench (hardest). UG Physics emphasizes factual recall and straightforward applications of physics principles, while PhysReason combines knowledge-based and reasoningintensive problems. PhyBench, the most challenging, requires advanced multi-step reasoning and deep conceptual understanding.

- **UG Physics:** Undergraduate-level problems in classical mechanics, electromagnetism, and thermodynamics, requiring multi-step reasoning and the application of standard formulas and units.

- **PhysReason:** A benchmark of 1,200 problems spanning factual recall (30%) and reasoning-based questions (70%), with varying difficulty.

- **PhyBench:** A Physics Olympiad-style benchmark designed to test complex reasoning, with problems requiring both deep conceptual insights and numerical problem solving.

## 2.2 Models

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 All models are prompted in reasoning mode (explicit CoT scratchpad), and sampled with nucleus sampling (temperature T = 0.6 to 0.7, top-p = 0.95).

## 2.3 Calibrating Chain-Of-Thought

Reasoning explicitness and prompting style To evaluate the role of reasoning in model performance, we vary the *prompting style*, which controls how much a model is encouraged to rely on CoT. We distinguish between two categories of prompts (see §D for the full templates): While a substantial body of recent work (Wei et al., 2022b;a; Nazi et al., 2025) on CoT prompting has focused on closed-source LLMs accessed through APIs (e.g., PaLM, LaMDA, GPT variants), such settings typically restrict visibility into intermediate reasoning traces and limit opportunities for controlled interventions. To enable a more systematic investigation, we instead turn to opensource reasoning LMs, which allow us to directly intercept the CoT scratchpad prior to decoding. This access enables us to precisely manipulate intermediate reasoning and study the effects of different types of CoT deletions. Concretely, we evaluate three open-source LLMs spanning distinct architectures and pretraining regimes:
- **Phi-4:** A 14B reasoning-focused model, fine-tuned on curated chain-of-thought prompts and reinforced via supervised and RL methods, excelling in mathematical and logical reasoning tasks.

- **Qwen-A3B:** A 30.5B general-purpose Mixture-of-Experts LLM with a four-stage training pipeline including chain-of-thought cold start, reasoning RL, and thinking-mode fusion, optimized for multi-step reasoning and long-context understanding.

- **Magistral:** A reasoning-focused model from Mistral AI, with the open-sourced *Small* variant (24B parameters) trained via a reinforcement learning pipeline (GRPO) to improve multi-step reasoning and instruction following, including multilingual chain-of-thought capabilities.

1. **Full Reasoning:** The model is prompted to work through the problem in detail, producing a step-by-step derivation with comprehensive explanations of the relevant physics concepts and mathematical steps. The emphasis is on completeness, transparency of reasoning, and not skipping intermediate steps. (This corresponds to the *High Reasoning* setting.)
2. **Less Reasoning:** The model is encouraged to solve the problem with reduced deliberation.

This includes two sub-levels:
- *Medium Reasoning:* Reasoning is still step-by-step, but concise and focused, avoiding excessive elaboration.

- *Low Reasoning:* The model is asked to minimize reasoning, providing a quick answer with only minimal or implicit thought steps.

## 2.4 Metrics And Evaluation

We quantify model behavior along three axes:
- **Score:** Evaluated with Claude-4 Sonnet as judge, scoring 0–1 based on correctness, derivation accuracy, logic, formatting, and clarity. The model compares each solution to the expected answer, penalizing deviations.

- **Final Answer Length:** Number of characters generated in the answer, used to detect cramming behavior.

- **Information Overlap:** Fraction of deleted CoT elements that reappear in the final answer, measured using Bag-of-Words metrics: Jaccard similarity and Manhattan distance.

This setup allows systematic evaluation of both the necessity and faithfulness of CoT reasoning in LLMs for physics problem solving.

## 3 Experimental Results

We experiment with the role of CoT

![3_image_0.png](3_image_0.png)

scratchpads in physics reasoning tasks, focusing on whether they are faithfully used, when they become essential, and how models compensate under manipulation. We evaluate three recent LLMs—Phi4, Qwen-A3B and Magistral—on three physics benchmarks: UG Physics, Phy- Bench, and PhysReason. For all our experiments, we use nucleus sampling with temperature T = 0.6 to 0.7, top-p = 0.95.

## 3.1 Prompting And Calibration

We begin by investigating whether explicit reasoning traces improve performance beyond direct answer generation.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

## Reasoning Explicitness And Prompting.

We find a consistent trend across models and datasets: performance improves with the explicitness of reasoning. When prompted with *Full Reasoning*, models often achieve the highest accuracy, benefiting from detailed step-by-step derivations that enforce intermediate consistency checks (e.g., writing governing equations, performing algebraic transformations). Under the *Less Reasoning* settings, accuracy declines, reflecting that concise reasoning sketches, while still helpful, provide fewer opportunities for the model to correct errors in intermediate steps.

Figure 2: Prompting styles evaluation across 2 datasets and 3 models. **Full Reasoning (High):** the model shows all intermediate steps before the final answer. Less Reasoning (Low/Medium): the model provides briefer reasoning. We observe that higher explicitness generally leads to better answer quality.

This setup allows us to baseline the differences in model performance that arise from the inherent CoT reasoning reliance. We note that in most of our experiments beyond the initial comparison, we use the medium reasoning prompt by default. Number of Samples We calibrate the number of data points and runs sufficient for our experiments based on ablation studies.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Calibration study. To determine how many samples are required for stable estimates, we conduct a convergence analysis by increasing the number of independent prompt completions and computing the width of the confidence interval. Using bootstrapped results over 50 UG-Physics questions with 5 re-runs of the same data, we find that approximately *5 prompts* are sufficient to reduce the relative error bar below 10%. We also confirm this trend with quartile-based results, and adopt this setting as our standard calibration configuration in Figure 8.

## 3.2 Cot Deletion Sweeps

In §3.1, we confirm that longer, ex-

![4_image_0.png](4_image_0.png) plicit CoT correlate with higher scoring solution, an unsurprising but important baseline. To probe how models rely on CoT during structured reasoning such as Physics, math or other AI for science related tasks, we conduct *systematic deletion experiments*.

Figure 3 summarizes the effect of CoT deletion on model performance. Across all models and datasets, we observe that answer scores degrade when portions of the CoT are removed. In this figure, we focus specifically on physics-related annotations within the CoT, which we restrict to structured elements such as equations and units. We then compare two conditions: deleting all *annotated* (physics-structured) elements vs. deleting the remaining, *non-annotated* portions. In both cases, performance declines, but the removal of annotated facts produces a more detrimental effect on answer scores. We also observe that the final answer lengths sometimes slightly increases when reasoning with partially deleted CoT. To better understand the slight increase in final answer length, we systematically characterize this effect.

Specifically, we intercept the scratchpad and remove k% of CoT tokens (k ∈ [0, 100]) before the final answer. We compare three deletion strategies: (1) **from-the-end deletion**, truncating the last k% of tokens; (2) **random** We evaluate results using Claude-4 Sonnet as a judge model, scoring each solution on a 0–1 scale based on correctness of the final answer, accuracy of the physics derivation, logical coherence, formatting, and clarity. The model is provided with the expected full answer for direct comparison, and large deviations are penalized. This evaluation confirms that higher reasoning explicitness consistently yields more reliable and logically coherent solutions. Figure 2 summarizes these results by showing model performance across reasoning conditions; specifically, prompting models for more extensive reasoning (the *Full Reasoning* condition) yields higher judged derivation quality and greater solution coherence than prompts that elicit less reasoning.

Figure 3: Effect of CoT deletions on physics benchmarks across models. **None** = full CoT, **Annotated** = deletion of physics-structured elements (e.g., equations/units), Non- Annotated = deletion of remaining content. Removing any portion lowers scores (blue dots), with annotated deletions most detrimental. The final answer length (orange dots, in character counts) slightly increases with CoT deletions.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 deletion, removing tokens uniformly at random; and (3) **physics-aware deletion**, where another model (Claude-4 Sonnet) identifies physics-related tokens for removal. Across strategies, accuracy declines monotonically with greater deletion, while answer length increases. This possibly indicates that models attempt to *reconstruct lost reasoning* directly in the answer stage—a behavior we term cramming. From-the-end deletion sweep. We delete k% of CoT tokens from the end, sweeping k ∈ [0, 100]. Accuracy remains stable until approximately 40% deletion, after which it drops, as shown in figure 6. In general, we observe an X-shaped pattern in the answer length: as CoT reasoning is deleted, the final answer length steadily increases, compensating for the missing reasoning. Beyond roughly 40% deletion, accuracy declines, though in some cases this is partially offset by a large increase in the final answer length, possibly indicated by a slight uptick in accuracy in panels b), c), and f) of the undergraduate physics results in figure 6. Random deletion sweep. We randomly delete k% of CoT tokens, sweeping k ∈ [0, 100]. Accu-

![5_image_0.png](5_image_0.png) racy remains stable until approximately 60% deletion, after which it *drops sharply*. Despite slightly higher variance compared to from-the-end deletion, we observe the same X-shaped pattern: as reasoning is removed, the final answers become steadily longer, compensating for the missing CoT tokens. At high deletion levels, this effect is especially pronounced, with answers often becoming significantly longer. Figure 11 in §B illustrates this trend.

Figure 4: Final answer scores

![5_image_1.png](5_image_1.png) under end deletion. Accuracy begins to drop noticeably around 40% deletion (red dotted line).

Figure 6: From-the-end deletion-sweep visualizations.

Physics-aware deletion. We selectively remove domain-relevant content by tagging physicsspecific spans (e.g., equations, constants, unit conversions) with Claude-4 Sonnet and deleting k% of these tokens. Accuracy declines steadily but less abruptly than in random or end deletion (Figure 14 in §C). Answer length, however, increases sharply once 70–80% of annotated tokens are removed, indicating partial compensation until critical facts are lost. These results highlight the importance of domain-specific knowledge in maintaining reasoning fidelity.

## 4 Analysis And Discussion

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

## 4.1 Cramming Behavior

Our experiments reveal several robust patterns in how LLMs utilize chain-of-thought (CoT) scratchpads for physics reasoning, which we analyze below. Across all three models and datasets, we observe a striking pattern: *when substantial portions of* CoT are deleted, the final answer length increases sharply, often with reconstructed equations or intermediate steps reappearing in the final output. We term this compensatory behavior **cramming**. While we do not probe internal mechanisms directly, these results suggest that LLMs may draw on internalized physics knowledge or learned solution templates to regenerate missing reasoning steps during answer decoding. This behavior appears consistently across all three deletion strategies. For **end deletion**, Figure 6 shows that cramming emerges once roughly 40% of the CoT is removed, followed by a gradual increase in final answer length. For **random deletion**, Figure 11 indicates that cramming becomes pronounced at around 60% deletion, again with a steady length increase thereafter. Finally, under physics-aware deletion, Figure C shows a much more gradual decline in accuracy, with degradation only becoming noticeable at 70–80% deletion. At this point, however, the model exhibits a sharp spike in final answer length, consistent with cramming behavior.

## 4.2 Information Overlap And Recovery

Our analyses reveal a dual behavior in model reasoning under CoT deletion: while models often attempt to reconstruct missing structured information, the recovery is not guaranteed to be faithful, since the final answer score mostly does not recover across 3 different deletion strategies. In some cases (e.g., Phi-4 on undergraduate physics), models seem to substitute alternative reasoning rather than recovering the original, suggesting that reconstruction is heuristic and opportunistic rather than systematic. To quantify this phenomenon, we measure whether deleted information reappears in final answers. Because physics reasoning relies heavily on structured content—such as specialized terminology, equations, and units—we evaluate recovery using strict token-overlap metrics between the generated answers and the original CoT before deletion. This allows us to assess both the degree of redundancy in model reasoning and the limits of faithful recovery across deletion sweeps. Defining overlap. We define **information overlap** as the intersection between (i) the original CoT prior to deletion and (ii) new content generated in the final answer across deletion sweeps. Quantification. We measure overlap using two complementary metrics:
1. **Lexical Overlap (Jaccard Similarity):** captures shared vocabulary, ignoring frequency.

For passages p1 and p2, let V (p) denote the set of unique tokens. Then

$$\mathrm{Jaccard}(p_{1},p_{2})={\frac{|V(p_{1})\cap V(p_{2})|}{|V(p_{1})\cup V(p_{2})|}}.$$
. (1)
2. **Frequency Overlap (Manhattan Distance on Bag-of-Words):** captures distributional similarity in word usage. For passages p1, p2 with bag-of-words representations bow(p1), bow(p2) ∈ R
d, where each dimension counts token frequency, we compute

$$D_{\mathrm{Manhattan}}(p_{1},p_{2})=\sum_{i=1}^{d}\vert\mathrm{bow}(p_{1})_{i}-\mathrm{bow}(p_{2})_{i}\vert.$$

$$(1)$$

$$\left(2\right)$$
. (2)
These metrics highlight different aspects of recovery: Jaccard similarity reflects vocabulary-level reuse, while Manhattan distance accounts for shifts in token frequency distributions.

378

![7_image_0.png](7_image_0.png) 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Findings. Figure 7 shows that information overlap between deleted CoT spans and regenerated answers increases as deletion progresses, but the pattern varies across strategies and datasets. Under end deletion, overlap rises smoothly and consistently across all models and benchmarks, reflecting systematic attempts to reconstruct truncated reasoning. In contrast, **random deletion** yields delayed overlap growth (becoming pronounced only beyond ∼60% deletion) and exhibits higher variance, suggesting that scattered removals are harder to recover from. **Physics-aware deletion** produces the noisiest trends: overlap remains relatively flat until heavy deletion (70–80%), at which point sharp spikes appear, consistent with late-stage cramming. Across datasets, recovery is most stable on PhyBench and PhysReason, whereas UG Physics displays greater variability, with some models substituting alternative reasoning instead of reproducing the deleted content. Taken together, these results suggest that while models opportunistically recover missing information, such recovery often reflects surface-level similarity rather than genuine fidelity to the original CoT. This points to a deeper conflict between CoT reasoning as written in the scratchpad and the model's own decoding process: reconstructed content may be heuristically generated rather than faithfully recovered, raising questions about the faithfulness of CoT traces as evidence of underlying reasoning.

## 4.3 Implications For Cot Faithfulness

Our findings provide new perspective on the *faithfulness* of chain-of-thought (CoT) reasoning. By faithfulness, we refer to the extent to which the scratchpad explicitly reflects the internal computations that lead to the model's final prediction, rather than merely serving as a plausible post hoc justification. Across deletion sweeps, we observe that: (i) not all intermediate steps in the scratchpad are faithfully required for correct answers, and (ii) models deploy compensatory mechanisms—such as cramming—to regenerate missing information directly in the final answer.

These observations suggest that CoT scratchpads are simultaneously *informative* and *redundant*. On one hand, they contain structured reasoning traces that improve fidelity when preserved. On the other hand, their partial bypassability raises the possibility that CoT text is not a transparent window into model reasoning, but rather an externalization that can diverge from the underlying decision process. For interpretability, this cautions against treating CoT explanations as fully faithful accounts. For 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 prompting and system design, it highlights the need to explore strategies that promote reliance on genuine intermediate reasoning rather than heuristic reconstruction. These findings also carry practical implications. First, because models can often reconstruct missing information in the final answer, *early stopping of CoT generation* may provide a cost-effective way to save tokens without proportionally sacrificing accuracy. Second, the fact that useful information can be compressed and reconstructed suggests that prompting strategies could be redesigned to elicit more concise yet effective reasoning traces. In short, while CoT can illuminate aspects of model reasoning, it cannot yet be assumed to faithfully reveal it.

## 4.4 Limitations

Our study has several limitations. First, our experiments are scoped to physics reasoning tasks and three representative LLMs. While this domain is specialized, it is also representative of structured reasoning challenges central to AI-for-science more broadly, suggesting that the qualitative patterns we observe may generalize beyond physics. Second, our conclusions are drawn from *observable* outputs; we do not analyze latent representations, internal attention patterns, or decoding dynamics, which may reveal additional mechanisms of information recovery. Third, although deletion sweeps demonstrate consistent trends across datasets and models, further work is required to test their robustness across other reasoning domains (e.g., mathematics, commonsense) and architectures. Future research should expand to diverse domains and model families, and probe the mechanistic basis of cramming and overlap behaviors—for example, whether they arise from memorized templates, latent redundancy in representations, or adaptive decoding strategies. Additionally, scaling studies could clarify whether larger models exhibit more faithful CoT usage or simply stronger compensatory reconstruction.

## 5 Conclusion

CoT scratchpads play a dual role in physics reasoning tasks central to AI for science: they boost accuracy when intact but can be bypassed through *cramming*, where models reconstruct missing steps in final answers. This shows CoT traces are both informative and redundant, raising concerns about their **faithfulness** as evidence of reasoning. For interpretability, CoT should not be treated as transparent explanations; for system design, they highlight opportunities to trade off efficiency and reasoning fidelity. Advancing AI for science will require evaluation methods that go beyond accuracy to enforce faithfulness, ensuring that intermediate steps genuinely reflect underlying computations.

## 6 Related Works

Reasoning-Focused Models. Recent LLMs increasingly incorporate reasoning-oriented instruction tuning and reinforcement learning to improve multi-step problem solving. Phi-4 (Abdin et al., 2024) is fine-tuned on curated chain-of-thought datasets and refined using reinforcement learning, achieving strong performance on mathematical, logical, and planning tasks despite its moderate parameter count. GLM-4.5-Air (Zeng et al., 2025) leverages a Mixture-of-Experts (MoE) architecture and multi-stage expert iteration with RL to support hybrid reasoning and agentic behaviors. Qwen- A3B (Qwen, 2025) uses a four-stage training pipeline combining reasoning RL, chain-of-thought cold-start, and thinking-mode fusion, optimizing multi-step reasoning and long-context comprehension. Chain-of-Thought Faithfulness. While chain-of-thought prompting improves multi-step reasoning(Wei et al., 2022a;b; Yao et al., 2023), recent work highlights that generated reasoning steps may be unfaithful, containing errors or unsupported inferences (Barez et al., 2025). Faithfulnessfocused approaches, including self-consistency decoding (Cheng et al., 2025; Wang et al., 2023) and verification-based RL fine-tuning(Su et al., 2025; Peng et al., 2025), aim to ensure that intermediate steps reliably lead to correct final answers. Models such as Phi-4, Qwen-A3B, and Magistral-Small incorporate elements of reasoning supervision and RL that may indirectly improve CoT faithfulness, although systematic evaluation of faithfulness remains an open challenge.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## References

Marah Abdin, Jyoti Aneja, Harkirat Singh Behl, Sebastien Bubeck, Ronen Eldan, Suriya Gunasekar, ´
Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C'esar Teodoro Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang, and Yi Zhang. Phi-4 technical report. *ArXiv*, abs/2412.08905, 2024. URL https: //api.semanticscholar.org/CorpusID:274656307.

Fazl Barez, Tung-Yu Wu, Ivan Arcuschin, Michael Lan, Vincent Wang, Noah Siegel, Nicolas Col- ´
lignon, Clement Neo, Isabelle Lee, Alasdair Paren, Adel Bibi, Robert Trager, Damiano Fornasiere, John Yan, Yanai Elazar, and Yoshua Bengio. Chain-of-thought is not explainability.

arXiv preprint, 2025. Preprint. Available at https://aigi.ox.ac.uk/wp-content/ uploads/2025/07/Cot_Is_Not_Explainability.pdf.

Kristian G. Barman, Sascha Caron, Emily Sullivan, Henk W. de Regt, Roberto Ruiz de Austri, Mieke Boon, Michael Farber, Stefan Fr ¨ ose, Faegheh Hasibi, Andreas Ipp, Rukshak Kapoor, Gregor ¨ Kasieczka, Daniel Kostic, Michael Kr ´ amer, Tobias Golling, Luis G. Lopez, Jesus Marco, Sydney ¨ Otten, Pawel Pawlowski, Pietro Vischia, Erik Weber, and Christoph Weniger. Large physics models: Towards a collaborative approach with large language models and foundation models, 2025. URL https://arxiv.org/abs/2501.05382.

Rishi Bommasani, Deepak Narayanan, Shreya Kapoor, et al. Opportunities and risks of foundation models for science, 2023.

Yi Cheng, Xiao Liang, Yeyun Gong, Wen Xiao, Song Wang, Yuji Zhang, Wenjun Hou, Kaishuai Xu, Wenge Liu, Wenjie Li, Jian Jiao, Qi Chen, Peng Cheng, and Wayne Xiong. Integrative decoding:
Improve factuality via implicit self-consistency, 2025. URL https://arxiv.org/abs/ 2410.01556.

Steffen Eger, Yong Cao, Jennifer D'Souza, Andreas Geiger, Christian Greisinger, Stephanie Gross, Yufang Hou, Brigitte Krenn, Anne Lauscher, Yizhi Li, Chenghua Lin, Nafise Sadat Moosavi, Wei Zhao, and Tristan Miller. Transforming science with large language models: A survey on ai-assisted scientific discovery, experimentation, content generation, and evaluation, 2025. URL
https://arxiv.org/abs/2502.05151.

Takeshi Kojima, Shixiang Gu, Alistair Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners, 2022.

Tomasz Korbak, Mikita Balesni, Eliza beth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, Scott Emmons, Owain Evans, David Farhi, Ryan Greenblatt, Dan Hendrycks, Marius Hobbhahn, Evan Hubinger, Geoffrey Irving, Erik Jenner, Daniel Kokotajlo, Victoria Krakovna, Shane Legg, David Lindner, David Luan, Aleksander Mkadry, Julian Michael, Neel Nanda, Dave Orr, Jakub W. Pachocki, Ethan Perez, Mary Phuong, Fabien Roger, Joshua Saxe, Buck Shlegeris, Mart´ın Soto, Eric Steinberger, Jasmine Wang, Wojciech Zaremba, Bowen Baker, Rohin Shah, and Vladimir Mikulik. Chain of thought monitorability: A new and fragile opportunity for ai safety. *ArXiv*, abs/2507.11473, 2025. URL
https://api.semanticscholar.org/CorpusID:280276345.

Michal Kosinski. Evaluating large language models in theory of mind tasks. *Proceedings of the* National Academy of Sciences, 121(45), October 2024. ISSN 1091-6490. doi: 10.1073/pnas. 2405460121. URL http://dx.doi.org/10.1073/pnas.2405460121.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Luko ˙ siˇ ut¯ e, Karina ˙ Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. Measuring faithfulness in chain-of-thought reasoning, 2023. URL https://arxiv.org/abs/2307.13702.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. Faithful chain-of-thought reasoning, 2023. URL https://arxiv. org/abs/2301.13379.

Fanqing Meng, Wenqi Shao, Lixin Luo, Yahong Wang, Yiran Chen, Quanfeng Lu, Yue Yang, Tianshuo Yang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Phybench: A physical commonsense benchmark for evaluating text-to-image models. *ArXiv*, abs/2406.11802, 2024. URL https://api.semanticscholar.org/CorpusID:270560653.

Zabir Al Nazi, Md. Rajib Hossain, and Faisal Al Mamun. Evaluation of open and closed-source llms for low-resource language with zero-shot, few-shot, and chain-of-thought prompting. Nat. Lang. Process. J., 10:100124, 2025. URL https://api.semanticscholar.org/CorpusID: 275348270.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simon Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gib- ´ son, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mely, Ashvin Nair, Reiichiro Nakano, Ra- ´ jeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O'Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Ceron Uribe, Andrea Vallone, Arun Vi- ´ jayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774.

Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, and Juanzi Li. Verif: Verification engineering for reinforcement learning in instruction following, 2025. URL https://arxiv.org/abs/ 2506.09942.

Qwen. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. M Rastogi, Albert Q. Jiang, Andy Lo, Gabrielle Berrada, Guillaume Lample, Jason Rute, Joep Barmentlo, Karmesh Yadav, Kartikay Khandelwal, Khyathi Raghavi Chandu, Leonard Blier, Lucile ´ Saulnier, Matthieu Dinot, Maxime Darrin, Neha Gupta, Roman Soletskyi, Sagar Vaze, Teven Le Scao, Yihan Wang, Adam Yang, Alexander H. Liu, Alexandre Sablayrolles, Am'elie H'eliou, Amelie Martin, Andrew Ehrenberg, Anmol Agarwal, Antoine Roux, Arthur Darcet, Arthur Men- ´ sch, Baptiste Bout, Baptiste Roziere, Baudouin De Monicault, Chris Bamford, Christian Wal- ` lenwein, Christophe Renaudin, Clemence Lanfranchi, Darius Dabert, Devon Mizelle, Diego ´ de Las Casas, Elliot Chane-Sane, Emilie Fugier, Emma Bou Hanna, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Guillaume Martin, Himanshu Jaju, Jan Ludziejewski, Jean-Hadrien Chabran, Jean-Malo Delignon, Joachim Studnia, Jonas Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Kush Jain, Lingxiao Zhao, Louis Martin, Luyu Gao, Lelio Renard Lavaud, ´ Marie Pellat, Mathilde Guillaumin, Mathis Felardos, Max Augustin, Mickael Seznec, Nikhil Raghuraman, Olivier Duchenne, Patricia Wang, Patrick von Platen, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Pavankumar Reddy Muddireddy, Philomene Chagniot, Pierre ` Stock, Pravesh Agrawal, Romain Sauvestre, Remi Delacourt, Sanchit Gandhi, Sandeep Sub- ´ ramanian, Shashwat Dalal, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Thibault Schueller, Thibaut Lavril, Thomas Robert, Thomas Wang, Timothee Lacroix, Valeriia Nemychnikova, Victor Paltz, Virgile Richard, Wen-Ding Li, William ´ Marshall, Xuanyu Zhang, and Yunhao Tang. Magistral. *ArXiv*, abs/2506.10910, 2025. URL
https://api.semanticscholar.org/CorpusID:279319007.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Natalie Shapira, Mosh Levy, Seyed Hossein Alavi, Xuhui Zhou, Yejin Choi, Yoav Goldberg, Maarten Sap, and Vered Shwartz. Clever hans or neural theory of mind? stress testing social reasoning in large language models, 2023. URL https://arxiv.org/abs/2305.14763.

Rick Stevens et al. Ai for science: Report on a department of energy town hall meeting series, 2023. Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, and Dong Yu.

Crossing the reward bridge: Expanding rl with verifiable rewards across diverse domains, 2025. URL https://arxiv.org/abs/2503.23829.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting, 2023. URL https: //arxiv.org/abs/2305.04388.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models, 2023. URL https://arxiv.org/abs/2203.11171.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, et al. Chain-of-thought prompting elicits reasoning in large language models. In *Advances in Neural Information Processing Systems* (NeurIPS), 2022a.

Jason Wei, Denny Zhou, et al. Language models perform reasoning via chain of thought.

Google Research Blog, May 2022b. URL https://research.google/blog/ language-models-perform-reasoning-via-chain-of-thought/.

Xin Xu, Qiyun Xu, Tong Xiao, Tianhao Chen, Yuchen Yan, Jiaxing Zhang, Shizhe Diao, Can Yang, and Yang Wang. Ugphysics: A comprehensive benchmark for undergraduate physics reasoning with large language models. *ArXiv*, abs/2502.00334, 2025. URL https://api. semanticscholar.org/CorpusID:276095053.