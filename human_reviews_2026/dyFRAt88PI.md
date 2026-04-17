# ENTP: Enhancing Low-Quality SFT Data via Neural-Symbolic Text Purge-Mix

- Decision: Reject
- Scores: 4, 6, 2, 4, 4

## Abstract
Supervised Fine-Tuning (SFT) adapts pre-trained Large Language Models (LLMs) to domain-specific instructions by training on a carefully curated subset of high-quality instruction–response pairs, typically drawn from a larger dataset that often contains many low-quality or noisy samples. Despite its effectiveness, this quality-first paradigms often suffer from two caveats. On the one hand, quality filters are inherently imperfect, many samples that pass through these filters are not truly high-quality. On the other hand, discarding the vast majority of low-quality or frequently occurring examples may lose potentially valuable signal. As much of the readily available instruction-following data online has already been utilized, further improvements now depend on leveraging, rather than discarding, the examples that were previously filtered out. To address these two issues, we introduce ENTP, which stands for Enhancing low-quality SFT data via Neural-symbolic Text Purge-Mix. Similar to the ENTP personality type from MBTI, ENTP is creative in enhancing the low-quality data via purging (noisy information removal) and mixing (with extracted information from all available data and model knowledge). Specifically, the symbolic component identifies and isolates low-quality raw corpora using statistical priors, while the connectionist component extracts latent representations to guide the reconstruction of missing or corrupted information. This synergy generates hybrid instruction-response pairs that augment informational value while preserving corpus diversity. Our experiments demonstrate that fine-tuning LLMs on data augmented by ENTP, which are derived solely from low-quality sets, consistently outperforms 13 established data-selection methods across 5 standard instruction-following benchmarks. Notably, it can even surpass fine-tuning on the full original dataset (≈300K examples). Our findings demonstrate that ostensibly low-quality data is a critical resource; leveraging it through intelligent purification and synthesis is key to efficient and effective instruction alignment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper challenges the prevailing "quality-first" paradigm in Supervised Fine-Tuning (SFT), where models are trained on small, curated high-quality (HQ) datasets while large amounts of low-quality (LQ) data are discarded. The authors argue this approach is hitting a bottleneck due to the scarcity of untapped HQ data and that LQ data contains valuable, recoverable signals. They introduce ENTP, a novel framework designed to enhance and leverage LQ data through a "Neural-Symbolic Text Purge-Mix" approach.

### Strengths
The paper addresses a critical bottleneck in SFT. The shift from merely selecting high-quality data to actively enhancing and synthesizing low-quality data is a compelling and important contribution, offering a path to utilize data that is typically discarded.

The ability of the ENTP-generated dataset (54K samples from the LQ pool) to consistently outperform all data selection baselines applied to the LQ pool, and frequently surpass the performance of the HQ set (131K samples) and even the Full set (300K samples), strongly validates the core hypothesis.

### Weaknesses
The ENTP framework is exceedingly complex, involving numerous stages: LLM rating, score correction, embedding, multi-stage clustering, MMR selection, and the iterative fusion. Critically, Step 3 requires multiple iterative calls (Cycles 1 and 2) to an LLM using five different operators (DA, MCG, ICD, FAC, FAU) for every pair of fused samples. Generating the 54K dataset likely required an enormous number of LLM inferences (using gpt-4o-mini). The paper completely omits any analysis of the computational cost, time, or financial expense of the data generation process. Without this analysis, the practicality and scalability of the approach are highly questionable, as the generation cost may outweigh the SFT efficiency gains.

The entire pipeline heavily relies on a powerful proprietary model (gpt-4o-mini) for both initial rating and the complex synthesis/evaluation tasks in Step 3. This raises significant concerns that the observed gains are primarily due to knowledge distillation from the oracle model, rather than the specific effectiveness of the neural-symbolic fusion architecture. ENTP appears to be an elaborate method for prompting the oracle until it produces high-quality output based on the LQ inputs. The paper does not adequately distinguish the benefits of the framework from simpler distillation or data augmentation techniques using the same LLM.

The paper positions ENTP against data selection methods. However, ENTP is fundamentally a data synthesis method. The evaluation is incomplete as it does not compare against other established data synthesis techniques (e.g., Self-Instruct, Evol-Instruct, or simpler mixing/rewriting strategies).

### Questions
In addition to the weakness section, I have the below questions:

How do you disentangle the effects of knowledge distillation from the benefits of the complex neural-symbolic framework? Have you compared ENTP against a simpler baseline, such as using the same LLM to rewrite or improve individual low-quality samples without the fusion and iterative optimization steps?

Can you provide a concrete example in the main text illustrating the iteration process? Please show the generated corpus, the resulting Symbolic Loss (JSON), the specific prompt update made by the SPO, and the improved corpus in the subsequent iteration.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The work proposes ENTP, a data-construction pipeline that reclaims "low-quality" instruction data instead of discarding it. After a) LLM-based quality scoring with a Score Transition Matrix; b) Clustering + MMR selection, ENTP performs a neural-symbolic "two-to-one" fusion of pairs of low-quality samples using iterative LLM generation and symbolic feedback. The authors craft an ENTP dataset (default 54,888 pairs) and show that models fine-tuned on ENTP outperform LQ/HQ subsets and sometimes or surpass the Full Set (≈300k) on OpenLLM benchmarks.

### Strengths
1. The problem of evaluating and training LLMs without enough high-quality supervision is pressing, revalorizing low-quality data is a fresh, pragmatic angle that complements selection-only pipelines.
2. Experiments span multiple model families, datasets, and baselines, providing convincing evidence of generality. The consistent performance gain even over full-dataset fine-tuning is noteworthy.
3. The work highlights a structural limitation in classical data selection and empirically verifies that low-quality data retain useful signals.

### Weaknesses
1. The "symbolic loss" component remains conceptually abstract. The loss is an LLM-authored checklist driving prompt edits, not a defined objective with measurable descent, no ablation isolates its incremental contribution.
2. The method involves multiple nested steps (scoring correction, clustering, two-stage fusion), which may pose high computational cost and reproducibility challenges. Runtime (or API token usage) analysis is missing.
3. The paper does not provide a systematic evaluation of the actual quality of the generated instruction–response pairs. There is no human annotation, automatic text-quality scoring, or factuality/toxicity audit of the ENTP corpus. The only evidence of improvement comes indirectly from downstream benchmark gains and a few anecdotal examples in the appendix.It is unclear whether ENTP truly produces higher-quality data or merely data that better fits the evaluation distributions.

### Questions
1. The reported sizes (LQ = 123,786, HQ = 131,247, Full = 300,932) do not sum correctly. Is there overlap, filtering, or an unlabeled "medium-quality" portion?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ENTP (Enhancing low-quality SFT data via Neural-Symbolic Text Purge-Mix), a framework aiming to reuse low-quality instruction-tuning data for LLM supervised fine-tuning. The approach combines a symbolic “purge” step (noise correction via a score transition matrix and clustering) with a neural “mix” stage (using LLMs to merge and regenerate synthetic instruction–response pairs). Experiments on five instruction-following benchmarks (MMLU, GSM8K, TruthfulQA, BBH, TyDiQA) claim that ENTP outperforms 13 data-selection baselines and even surpasses training on the full dataset.While the motivation—leveraging low-quality data—is interesting, the technical novelty, theoretical soundness, and empirical rigor are limited.

### Strengths
1. Although the proposed pipeline lacks genuine algorithmic novelty, it appears engineeringly robust and well-implemented, showing careful system design and integration.
2. Addresses a timely and relevant problem—the efficient utilization and enhancement of low-quality SFT data, which remains a critical bottleneck in instruction tuning.
3. Provides comprehensive empirical comparisons across multiple benchmarks and base models, offering a reasonably broad empirical validation despite limited methodological innovation.

### Weaknesses
1. The paper lacks genuine novelty while introducing excessive and unnecessary conceptual packaging. In essence, the method is merely a { clustering + prototype selection + data augmentation}  pipeline. The first two steps mainly reassemble engineering tricks already well-documented in prior works, offering no new algorithmic contribution. The third step is essentially an overcomplicated extension of self-refine[1] or critic-LLM[2] frameworks, repackaged under heavy “neural-symbolic fusion” terminology that significantly increases reading complexity without adding real substance. Moreover, the proposed “two-to-one” augmentation is a very common practice, similar to MathFusion-style data combination.
[1] Self-Refine: Iterative Refinement with Self-Feedback
[2] CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation
[3] Mathfusion: Enhancing mathematic problem-solving of llm through instruction fusion


2. The paper does not provide any genuine ablation analysis to clarify the contribution of each module in the proposed three-step pipeline. The so-called “ablation study” in Section 4.3 only varies the overall data volume (20–100%) and does not isolate or remove individual components such as score correction, clustering, or neural-symbolic fusion. As a result, it remains unclear which step actually drives the reported improvements. In particular, the third module—the claimed Neural-Symbolic Two-to-One Fusion—is central to the paper’s narrative but lacks any empirical evidence demonstrating its necessity or effectiveness. A rigorous component-wise ablation (e.g., without fusion, without symbolic feedback, or using a single-LLM rewriting baseline) is needed to substantiate the claimed innovation.

3. Methodological complexity and scalability concerns. The ENTP pipeline involves multiple tightly coupled components—score correction, clustering, and a two-cycle neural-symbolic fusion loop. Step 3, in particular, requires iterative LLM invocations with symbolic loss back-propagation, which substantially increases computational cost and implementation difficulty. This complexity undermines the claimed data-efficiency advantage and makes the method difficult to reproduce or scale to larger corpora.

4. The comparison setup is problematic. Although the paper claims to enhance low-quality data, all baselines are actually data selection methods (e.g., KNN, Perplexity, IFD, DS2), which focus on filtering rather than enhancement or augmentation. This mismatch undermines the validity of the comparisons—ENTP synthesizes new data, while the baselines only select subsets of existing samples. A fair evaluation should instead compare ENTP with other data-enhancement frameworks (e.g., Self-Refine, Reflexion, Critic-LLM, or MathFusion-style augmentation). Without such comparisons, it is unclear whether ENTP offers any genuine advantage beyond generic LLM rewriting.

5. The evaluation setup is outdated and not aligned with the paper’s stated goal of enhancing instruction data. All benchmarks (MMLU, TruthfulQA, GSM8K, BBH, TyDiQA) are legacy instruction-following datasets that no longer reflect modern evaluation standards. To convincingly demonstrate data enhancement, the authors should compare against contemporary benchmarks such as AlpacaEval 2.0, MT-Bench, or ArenaHard, which better capture response helpfulness, reasoning depth, and stylistic alignment. Moreover, all evaluations remain within the instruction-tuning domain, leaving open the question of how well ENTP generalizes to other domains (e.g., reasoning, dialogue, code). Without such cross-domain and up-to-date evaluations, the claimed “enhancement” effect is insufficiently validated.

### Questions
1. Why did the authors choose GPT-4o-mini for the fusion stage instead of using an open-source small model?
2. Given that a large open-source model with a simple two-to-one merging and self-refine baseline could already achieve strong results, is such a complex “fusion” design necessary?
3. How does the proposed method generalize to other domains beyond instruction-tuning?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ENTP, a framework that enhances low-quality instruction data rather than discarding it. 

First, the authors use a Score Transition Matrix to correct noisy LLM-generated quality ratings, creating a more reliable separation between high-quality and low-quality samples. Second, they apply clustering algorithms to group similar low-quality samples and select representative examples from each cluster. Third, they fuse pairs of low-quality samples into single information-rich synthetic samples using GPT-4, guided by hand-crafted symbolic rules that ensure key terms are preserved and answers are complete.

Experimental results on three LLMs across five benchmarks show that low-quality data contains valuable signal when intelligently enhanced rather than discarded.

### Strengths
The paper challenges the dominant "quality-first" paradigm by demonstrating that low-quality data should be enhanced rather than discarded, addressing the critical problem that high-quality instruction data has been largely exhausted. This approach achieves better results with 54K synthetic samples derived from low-quality data than training on the full 300K original dataset, providing a practical path forward as the LLM field faces a data scarcity bottleneck.

The experimental design tests three different base models across five diverse benchmarks and compares against 13 strong baselines including both LLM-free methods and modern LLM-based selection approaches. Extensive ablation studies examine different fusion types, data proportions, and combinations with high-quality data, with consistent improvements across all settings rather than cherry-picked results.

The paper provides empirical evidence for a counterintuitive claim: "low-quality" data contains partial but valuable information that can be systematically extracted and recombined through intelligent fusion. This reframes data curation from a filtering problem (keep the top 10%) to a synthesis problem (enhance the bottom 90%), showing that traditional selection methods face a structural bottleneck that cannot be overcome by better filtering alone.

### Weaknesses
The paper uses confusing and misleading terminology throughout, particularly the term "backpropagation" which suggests gradient-based optimization but actually refers to simple rule-based prompt template switching. The "neural-symbolic" framing overstates the sophistication of what is essentially an if-else logic system that selects from 9 hand-written prompt templates based on structured error detection. The writing is dense and difficult to follow, with critical implementation details buried in extensive appendices, making it hard to understand what the method actually does versus what the terminology implies.

The paper does not compare against the most obvious baseline: simply using GPT-4 to directly enhance or rewrite individual low-quality samples (1-to-1 enhancement) rather than the complex 2-to-1 fusion approach. Without this comparison, it's impossible to determine whether the gains come from the sophisticated fusion mechanism or simply from having GPT-4 rewrite poor-quality data. The paper also lacks ablations comparing fusion with random pairing versus their clustering approach, and fusion without symbolic components (pure neural generation), making it difficult to isolate which components actually contribute to performance.

The method involves multiple elaborate steps (Score Transition Matrix correction, one-hop clustering, representative selection via MMR, domain analysis, two-cycle iterative refinement) but provides no evidence that this complexity is necessary. Simple alternatives like random pairing of low-quality samples with direct GPT-4 fusion, or even straightforward GPT-4 enhancement without any clustering, might achieve similar results with far less computational overhead and implementation complexity. The lack of ablations removing individual components makes it unclear whether the gains justify the significantly increased system complexity compared to simpler data enhancement approaches.

### Questions
Could the authors provide experimental results comparing ENTP against direct GPT-4 enhancement with matched sample counts (54K enhanced samples) or matched computational budgets (same number of LLM API calls)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ENTP, a neural–symbolic pipeline that turns “low-quality” instruction–response data into useful supervision for SFT by purging noise and mixing retained signals into synthesized training pairs. It first scores and corrects sample quality with a score-transition matrix to mitigate LLM-rater inconsistency, partitions data into high/low sets, then clusters the low-quality pool to pick representative items before a two-to-one fusion stage that applies symbolic rules and LLM generation in iterative loops to reconstruct richer, compact prompts and answers. The resulting merged corpus is combined with any high-quality data and used to fine-tune base models. On five OpenLLM benchmarks (MMLU, TruthfulQA, GSM8K, BBH, TyDiQA) and two backbones (Mistral-7B, Llama-3.1-8B), ENTP-derived data—sourced solely from low-quality sets—consistently beats 13 data-selection baselines and can rival or exceed training on the full ~300K dataset, with performance improving as more ENTP data are used.

### Strengths
The paper is original in reframing “low-quality” instruction data as valuable signal via a neural–symbolic *purge-mix* pipeline: it corrects noisy LLM quality scores with a score-transition matrix to address rater inconsistency, then fuses representative low-quality samples into richer training pairs using rule-guided and LLM-driven steps, rather than discarding them. This tackles two known caveats of quality-first filtering and is clearly motivated in the introduction.

### Weaknesses
The method feels overengineered and difficult to follow end-to-end, which raises the barrier to adoption. A concrete, running example would help: start from two real “low-quality” instruction–response pairs, show how scores are corrected and items clustered, present the exact symbolic-loss/ICD artifact produced, and then the final fused pair. Empirically, the benchmark coverage coverage is too narrow: adding harder math benchmarks (AIME ’24/’25, MATH500) plus coding tasks, and IFEval and SimpleQA will increase task coverage.

### Questions
Is it possible to run one more baseline for this paper: LESS: Selecting Influential Data for Targeted Instruction Tuning? by Xia et al, ICML 2024

### Soundness
3

### Presentation
2

### Contribution
2
