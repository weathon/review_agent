# GMSA: Enhancing Context Compression via Group Merging and Layer Semantic Alignment

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Large Language Models (LLMs) have achieved impressive performance in a wide range of Natural Language Processing (NLP) tasks. However, when applied to long-context scenarios, they face two challenges, i.e., computational inefficiency and redundant information. This paper introduces GMSA, a context compression method based on the encoder-decoder architecture, addressing these challenges by reducing input sequence length and redundant information. Structurally, GMSA has two key components: Group Merging and Layer Semantic Alignment (LSA). Group merging is used to extract summary vectors evenly and efficiently from the original context. Layer semantic alignment, on the other hand, aligns the high-level abstract summary vectors with the low-level primary input semantics, thus bridging the semantic gap between different layers. In the training process, GMSA first learns soft tokens that contain nearly complete semantics via autoencoder training. To further adapt GMSA to downstream tasks, we propose Knowledge Extraction Fine-tuning (KEFT) to extract task-relevant knowledge from these soft tokens. GMSA not only significantly outperforms the traditional compression paradigm in context restoration but also achieves stable and significantly faster convergence with only a few encoder layers. We further evaluate GMSA on question-answering, summarization, and general knowledge retention capabilities across two backbones (i.e., LLaMA-2-7B and Qwen2-7B), demonstrating its effectiveness and superiority, e.g., on the NaturalQuestions dataset, GMSA can achieve approximately a 2x speedup in end-to-end inference while outperforming various methods by a large margin.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces GMSA, a context compression framework that enhances LLM efficiency by addressing both semantic unevenness and cross-layer misalignment in long-context processing. While prior work on prompt or KV-cache compression primarily reduced token counts at the expense of semantic fidelity, GMSA systematically restructures compression through two key components: Group Merging, which evenly aggregates contextual semantics via group-wise averaging to prevent anchor-token dominance, and Layer Semantic Alignment (LSA), which bridges the semantic gap between encoder abstractions and decoder inputs using lightweight transformer blocks. Trained in a two-stage pipeline—autoencoder pretraining for semantic completeness followed by Knowledge Extraction Fine-Tuning (KEFT) for task adaptation—GMSA demonstrates robust performance across question answering, summarization, and reasoning benchmarks.

### Strengths
1. The work addresses token compression under long-context tasks, which is crucial for nowadays LLMs.
2. The insights of uneven semantic compression and aligning high-level summary tokens with low-level semantic tokens is interesting.

### Weaknesses
1. Missing comparison with recent works such as [1][2].
2. While the work (and the general field of token compression) is motivated by long-context challenges, most of the datasets used in the evaluation are not long at all. What about other long-context benchmarks such as NIAH, LongBench, Ruler, etc?
3. The backbone architectures used in the work are kinda old. For example, llama2-7b does not utilize GQA or other advanced architecture design. Could the authors provide results on more up-to-date architecture?


[1] DAST: Context-Aware Compression in LLMs via Dynamic Allocation of Soft Tokens, Chen et. al., ACL 2025.
[2] REFRAG: Rethinking RAG based Decoding, Lin et. al., ArXiv.

### Questions
1. What's the explanation behind aligning high-level summary tokens with low-level semantic tokens? Could the author provide some empirical evidence to justify this design choice?

Nitpicking Comments:
1. L48: undefine citation.
2. Table 4: "Secmantic" -> "Semantic"

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GMSA, a context compression and fine-tuning strategy which proposes Group Merging to evenly and efficiently produce summary vectors which are aligned to the decoder input space via a proposed Layer Semantic Alignment (LSA) module. Additionally, GMSA introduces Knowledge Extraction Fine-Tuning (KEFT) to adapt the decoder to specific tasks. The paper claims to learn soft tokens that contain nearly complete semantics and outperforms other methods in context restoration. GMSA achieves ~2x inference speedups with significantly higher downstream task performance over other context compression methods.

### Strengths
The authors raise interesting questions regarding the distribution of semantics into compressed vectors and across model layers. The authors chose a diverse set of baselines across different compression paradigms (soft prompt, hard prompt, and KV-cache compression) and use multiple evaluation tasks. The writing style is clear and easy to follow, and better methods for context compression would significantly benefit the field.

### Weaknesses
The major issues with the paper include unfair comparisons with baselines and overstated contributions. 

The authors chose baselines which perform compression of the KV-cache (StreamLLM, SnapKV, Activation Beacon) or context tokens (AutoCompressor, ICAE, LongLLMLingua, LLMLingua2). First, the authors evaluate the model for context restoration (the auto-encoding task) using the PwC dataset. Instead of using a strong baseline such as ICAE (an autoencoder trained on PwC), they implement a new method called TCP-AE as a baseline. The abstract claims that GMSA significantly outperforms the traditional compression paradigm in context restoration which is misleading without a comparison to actual traditional approaches like ICAE. 

GMSA is compared to all baselines for QA performance (Table 1). The evaluation shows huge improvements with GMSA, but GMSA was trained for the evaluated datasets. The paper and included code (upon inspection) suggest the baselines were only used for inference, and were not specifically adapted for the evaluations. Comparing the approaches in this manner is meaningless, as the baselines trained specifically on the same datasets could well outperform GMSA. If this is an incorrect interpretation of the experimental setup, the authors should make the baseline training explicit and include the relevant code.

For summarization, GSM-8K and MMLU, the proposed method is only compared against Activation Beacon and none of the other baselines. In these experiments Activation Beacon performs on par with GMSA, even outperforming on GSM-8K. 

Another work similar to GSMA is LLoCO. The authors compare and contrast the approach in the appendix, but do not include the method as a baseline in any of the main experiments. LLoCO also performs task-specific fine-tuning, which would make comparisons more fair. 

The components of GMSA are presented as novel, but are similar to existing approaches. Group Merging segments a sequence of tokens into fixed-length groups and merges them with average pooling. A similar approach is used to compress sequences in:

[Hierarchical Transformers Are More Efficient Language Models](https://aclanthology.org/2022.findings-naacl.117/) (Nawrot et al., Findings 2022) and
[Funnel-transformer: filtering out sequential redundancy for efficient language processing](https://dl.acm.org/doi/pdf/10.5555/3495724.3496083) (Dai et al., NeurIPS 20)

Layer Semantic Alignment is the module for mapping compressed representations into the input space of the decoder model. The paper points out that traditionally a MLP is used for this component. LSA simply replaces the MLP with a transformer layer (initialized with the first layer of the decoder model). Alternative methods also use the decoder layers to enforce alignment, such as through recursive training (the AutoCompressors baseline), or ICAE (another baseline) which uses the decoder as the encoder. 

Knowledge Extraction Fine Tuning is simply fine-tuning the attention layers of the decoder for the specific task with the soft tokens in the context. Similar fine-tuning is done for approaches used as baselines here or in other work:
[In-context Autoencoder for Context Compression in a Large Language Model](https://arxiv.org/abs/2307.06945) (Ge et al., ICLR ‘24)
[In-Context Former: Lightning-fast Compressing Context for Large Language Model](https://aclanthology.org/2024.findings-emnlp.138/) (Wang et al., Findings 2024)
[LLoCO: Learning Long Contexts Offline](https://aclanthology.org/2024.emnlp-main.975/) (Tan et al., EMNLP 2024)

### Questions
Questions:


Were the baselines trained to compress and perform well on the tasks being evaluated?

Why did you create the separate TCP-AE algorithm to use as a baseline for context restoration experiments instead of comparing to the baselines you were already using in other experiments? For example, ICAE is specifically trained for context restoration.

Throughout the paper, LSA is said to bridge the semantic gap between different layers. Isn’t LSA a single layer applied between the encoder output and the decoder input? I’m unclear on how it bridges the semantic gap between different layers, is this measured in your experiments?

Why was Activation Beacon the only baseline used to compare with for tasks other than QA? LongLLMLingua performed best in the previous experiments, why not use it, or all the baselines?

There wasn’t much information on the encoder, what model was used? Was it pretrained?

Other works that conduct average pooling for token representations have pointed out the dependence on position for what group each token is assigned. If you slightly tweak your prompt such that tokens are shifted by one would the model still perform well? The position dependence has been addressed by incorporating dynamic selections of group boundaries such as in:
[Efficient Transformers with Dynamic Token Pooling](https://aclanthology.org/2023.acl-long.353/) (Nawrot et al., ACL 2023)

Lines 395-403 discuss a flexibility benefit that GMSA has over KV-cache compression. What exactly is this additional flexibility and was it demonstrated in your experiments? KV-cache compression seems more flexible, as it can be trained without targeting specific queries or datasets, while GMSA required task-specific training of the compression and decoder.

In the ablation w/o LSA, was the previously trained KEFT used in this ablation? I.e. if the KEFT was trained without LSA then the model might still work well. 

Other Suggestions (did not impact evaluation):

Related work should be included in the main body so that readers can accurately place your contributions among previous works. Pushing related work to the appendix can give the impression of marginalizing previous work to inflate contributions. 

The paper mentions space constraints several times (Lines 142, 211, 232, 302, 371). There are multiple lines which contain only 1-4 words, but take up the entire line of space. Revising those paragraphs to eliminate these would add 11 lines of free space (Lines 80, 100, 128, 194, 288, 303, 307, 315, 325, 368, 485).

Section 3.4 and Figure 3 could be removed and replaced with one or two sentences explaining that you fine-tuned the decoder’s attention weights for each task (a common practice) using the compressed tokens as context. Section 4.3 could be mostly moved to the appendix, with just the stated improvements in the main body. This would create space for related work.

After revising, if space for main body material is still an issue, consider journals or other venues allowing more pages.

Minor Issues (did not impact evaluation):

Line 47, reference not resolved, “?”

Line 83: Uven -> Uneven

Table 1: Baecon -> Beacon

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GMSA, a context compression method that addresses computational inefficiency and information redundancy in long-context scenarios. GMSA uses an encoder-decoder architecture with two key innovations: (1) Group Merging, which evenly extracts summary vectors from the original context through average pooling to avoid uneven semantic learning, and (2) Layer Semantic Alignment (LSA), which bridges the semantic gap between the encoder's high-level abstract representations and the decoder's low-level input semantics using Transformer blocks. The method employs a two-stage training process: autoencoder training to learn soft tokens containing complete semantics, followed by Knowledge Extraction Fine-tuning (KEFT) that adapts the decoder to extract task-relevant knowledge. The experiments demonstrate the effectiveness of GMSA

### Strengths
1. This paper targets an interesting and important question of context compression, and proposes several practical challenges.
2. The proposed methods match with the challenges, and sound rational.
3. Although GMSA is simple but seems effective on multiple benchmarks.

### Weaknesses
1. Evenly extraction is not convincing, as the different tokens have different information density, which should selectively extract.
2. The soft token methods are naturally limited since currently a lot of LLMs are blackbox, which is hard to derive the first few layers of transformer blocks.
3. Several proposed modules lack experimental validation except for the ablation study. For example, the layer alignment problem: is there any mismatch before alignment? How's the alignment effect?

### Questions
1. What's the "Restatement Instruction" in overview?
2. Why W_QKV extracts knowledge, Why KEFT only trains W_QKV and only extracts knowledge?
3. missing citations in line 48, type in line 84 (Ueven)，line 137 （x should be capitalized）
4. Why train KEFT separately after autoencoder training?

### Soundness
3

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
This paper presents GMSA, an encoder-decoder framework for long-context compression that aims to reduce computational cost and redundancy. Its core innovations are (1) Group Merging (GM), which uses average pooling over groups of encoder hidden states to create summary vectors, claiming to avoid the "anchor token" bias of traditional methods, and (2) Layer Semantic Alignment (LSA), a small module initialized with bottom-layer decoder weights to bridge the "semantic gap" between the high-level encoder output and the low-level decoder input. Training involves a two-stage process: an autoencoder (AE) phase for general semantic preservation, followed by a Knowledge Extraction Fine-tuning (KEFT) phase that adapts only the decoder's attention mechanisms to specific downstream tasks. The authors show GMSA outperforms various baselines on QA and summarization, achieving significant (e.g., 2x) inference speedups.

### Strengths
1. The paper is well-written and addresses the critical, high-impact problem of efficient long-context processing for LLMs.
2. The proposed Group Merging (GM) and Layer Semantic Alignment (LSA) modules are novel and intuitively target specific, plausible weaknesses in prior soft-compression methods (i.e., uneven semantic learning and cross-layer semantic gaps).
3. The two-stage AE + KEFT training process is well-reasoned, cleanly separating the general goal of semantic preservation from task-specific adaptation.

### Weaknesses
1. The paper's claim to state-of-the-art performance is difficult to verify as it omits comparisons to several recent and highly relevant baselines, such as Provence [1], the Evaluator Heads method [2], and RocketKV [3].

   [1] Provence: efficient and robust context pruning for retrieval-augmented generation [ICLR 2025]

   [2] Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference [arXiv:2501.12959, NeurIPS 2025]

   [3] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression [ICML 2025]

2. The empirical validation relies exclusively on previous-generation models (LLaMA-2-7B, Qwen2-7B). It is unclear if the benefits of GMSA generalize to current SOTA models (e.g., LLaMA-3, Qwen3) or, more importantly, to heterogeneous configurations (e.g., a small encoder, a large decoder), which is a key potential advantage of this architecture.
3. The core motivations for the two main components lack direct empirical support. The paper does not provide visualizations to prove that Group Merging actually achieves more "even" semantic learning than alternatives, nor does it ablate the LSA's bottom-layer initialization strategy against other options (e.g., random init, top-layer init).
4.  The significant performance drop on the GSM8K dataset is another concern. This suggests the GMSA compression process, while preserving factual knowledge (MMLU), may be critically lossy for complex logical reasoning chains, which severely limits its generalizability for tasks beyond retrieval.

### Questions
1. What is the performance of the LSA module if it is randomly initialized, or initialized with decoder middle/top layer weights? This is needed to validate the "semantic gap" hypothesis.
2. Have the authors experimented with heterogeneous model sizes, such as a 1B-class encoder with a 7B-class decoder? This is a primary motivation for using an encoder-decoder architecture.

### Soundness
3

### Presentation
3

### Contribution
2
