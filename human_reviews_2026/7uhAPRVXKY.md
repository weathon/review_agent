# MulCLIP: A Multi-level Alignment Framework for Enhancing Fine-grained Long-context CLIP

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Pioneering vision–language models such as CLIP have transformed multimodal learning by aligning images and text in a shared embedding space. However, CLIP’s training on short captions limits its ability to handle downstream tasks that require longer text comprehension and fine-grained visual grounding. Recent advances mitigate this challenge by leveraging region-proposal information to map visual regions with corresponding sentences from lengthy captions, yet incurring notable deployment costs. In this paper, we introduce \textbf{MulCLIP}, a novel end-to-end multi-level alignment framework that bridges long-text structures \textbf{(long captions, sentences, words)} with image components \textbf{(global, regional)}, enabling fine-grained capabilities while surpassing CLIP’s strength on short-text understanding. MulCLIP first preserves global contrastive alignment between images and both summary and long captions, while extending positional embeddings for longer text sequences. To further enhance fine-grained understanding, we propose two novel strategies: (1) a token reconstruction alignment over locally calibrated features to strengthen semantic connections between words and image patches, and (2) a subcaption–aggregated patch alignment that automatically extracts and aggregate context-rich patches for each subcaption. Experimental results demonstrate MulCLIP outperforms baselines in both long- and short-text understanding, while ablation studies confirm its multi-scale alignment is the key factor driving better fine-grained capability than region-proposal–assisted approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces MulCLIP, a novel end-to-end multi-level alignment framework that bridges short and long text structures with both global and regional image information, enabling fine-grained vision-language understanding. It proposes two key components — token reconstruction alignment and subcaption-aggregated patch alignment — to achieve three-level alignment. Experimental results demonstrate that MulCLIP consistently outperforms baseline models across multiple benchmarks.

### Strengths
The paper proposed two novel alignment strategies that enable multi-scale alignment, allowing CLIP to capture fine-grained textual and visual information. MulCLIP demonstrates gains over baselines across extensive benchmarks and multiple model scales. The paper provides detailed ablation studies, clearly evaluating the impact of each component.

### Weaknesses
1. The proposed components are largely adaptations of existing mechanisms, with the main contribution being their combination into a unified framework.
2. The performance depends on caption quality, which may be sensitive to noisy or weak textual annotations, and robustness under such conditions is not evaluated.
3. Experiments focus primarily on retrieval benchmarks, with no evaluation on classification tasks (e.g., ImageNet or DataComp).

### Questions
1. How does MulCLIP perform on classification tasks such as ImageNet or DataComp?
2. Since token reconstruction alignment aims to reduce local redundancy, could it inadvertently remove semantically important patches or tokens?
3. How sensitive is MulCLIP to the number or granularity of subcaptions, and how does this affect overall performance?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MulCLIP, a multi-level alignment framework that enhances fine-grained long-context understanding for vision-language models while preserving short-text performance. It achieves this through global alignment of images with long/summary captions, token reconstruction alignment between words and image patches, and subcaption-aggregated patch alignment, eliminating the need for external region-proposal tools. Experimental results show MulCLIP outperforms baselines in both long- and short-text retrieval tasks.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed token reconstruction alignment and subcaption-aggregated patch alignment strategies are interesting and innovative.
3. Experimental results reflect the effectiveness of the proposed method to some extent.

### Weaknesses
1. The claim in Lines 124-126 of the paper is inappropriate. In fact, MulCLIP does not achieve finer granularity than FG-CLIP, which uses region-proposal assistance; the two methods merely differ in their approaches to fine-grained alignment.
2. The Subcaption-Aggregated Patch Alignment proposed in the paper is similar to [1], yet the paper lacks an explicit comparative discussion about this similarity.
3. FG-CLIP serves as an important baseline for the paper, but there is no comparison with it in the experimental results section. Please supplement the corresponding experimental results.
4. The paper only compares the caption retrieval performance of long-text and short-text, but fails to conduct a quantitative analysis of the method’s fine-grained capability—a capability that the authors emphasize the model possesses. Please supplement relevant experimental results with reference to FG-CLIP or [2].

[1] FILIP: Fine-grained Interactive Language-Image Pre-Training. ICLR, 2022.

[2] UMG-CLIP: A Unified Multi-Granularity Vision Generalist for Open-World Understanding. ECCV, 2024.

### Questions
Please refer to the 'Weaknesses' part.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents MulCLIP, an end-to-end framework designed to address the limitations of CLIP in handling long-text and fine-grained alignment. Its core innovation lies in a parallel, three-level alignment strategy that obviates the need for external region-proposal tools.
The framework jointly optimizes three distinct losses:
- $\mathcal{L}_{global}$ preserves alignment for both long-form text and short summaries.
- $\mathcal{L}_{Word}$ employs a self-supervised "token reconstruction" alignment.
- $\mathcal{L}_{Sub}$ introduces a novel alignment at the sentence (subcaption) granularity.
Empirical results indicate that MulCLIP successfully enhances long-text retrieval capabilities while maintaining high performance on short-text tasks, addressing a common trade-off in related work and demonstrating the efficacy of its design.

### Strengths
1. The self-supervised $L_{Word}$ and the $\mathcal{L}_{Sub}$ (SAP), serves as a replacement for the FILIP loss utilized by FineLIP's CLIM. This modification is remarkably concise and offers a novel solution to the challenge of token-level fine-grained alignment.
2. The experiments are thorough, covering both long and short-text retrieval, as well as zero-shot and in-domain retrieval. The ablation studies are well-designed, and the evaluation is further supplemented with image classification benchmarks.
3. The paper is well-researched and comprehensively cites related work.

### Weaknesses
1. The authors should more clearly articulate the specific improvements of the LC module over ATRM(from FineLIP(https://arxiv.org/pdf/2504.01916), or position it as an adoption of existing technology rather than a novel contribution.
2. While the idea of using an aggregated visual representation for the Subcaption-Aggregated Patch (SAP) loss is logical, the performance improvement appears incremental.
3. The formula on lines 174-176 is not numbered. Furthermore, although this formula is cited from FineLIP, the shape of $W_q$ is incorrect, rendering the matrix multiplication computationally infeasible.

### Questions
1. According to Table 5, the performance of "Global" (which appears to be CLIP + $L_{global}$) already exceeds the performance of FineLIP reported in Table 1 (CLIP + ATRM + CLIM). FineLIP incorporates [CLS] and [EOS] tokens into $V'$ and $T'$ for its FILIP loss, which is not a classical contrastive loss and lacks the $L_{global}$ as defined in this paper. Would it be possible to conduct an experiment (e.g., "w/o global") to directly isolate the contribution of LC + WPR + SAP? The experiment removing $L_{global}$ might still be trainable, as the contrastive loss from $L_{sub}$ (SAP) is retained. Alternatively, could the authors add the $L_{global}$ to the FineLIP implementation for a fairer comparison?
2. In Table 5, the performance difference between "w/ LC" and "w/o LC" configurations is minimal. Have the authors attempted to remove only the textual Token Calibration Module (TCM)? Unlike in FineLIP, the $t'$ tokens in MulCLIP are only directly supervised by the $L_{recon}^{text}$, lacking the directly contrastive supervision that $v'$ receives via $L_{recon}^{image}$ (SAP, through $\bar{v}$).
3. For the LC module, which functions as a form of proxy-token or VQ-like token compression, was any analysis conducted on token diversity during training?
4. Can a single Token Calibration Module layer effectively scale up to larger general training datasets, such as CC12M or ShareGPT4V?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces MulClip, a composite objective that allows multi-level alignment in vision language models build on top of CLIP backbone. In particular MulClip makes use of long and short captions and does alignment with global and local image regions. This is done through a bunch of objectives : L_global that sums a contrastive loss for the long caption with a contrastive loss for the short caption and L_word and L_sub that focuses on refined alignment with local regions of the image. The paper shows empirical evidence on DOCCI, DCI and Urban1k, as well  COCO and Flick30k datasets. Several ablations are done on the importance of the different components of the MultiClip objective as well visualization of attentions maps.

### Strengths
The paper is relatively easy to read, the authors provide a good description of each of the different terms of the MulClip objective. The method generally performs well empirically, at least against the baselines (e.g. Table 2, 3 and 4), and the authors have run a few ablations to confirm the role of the different components of the objective that seem to present mostly a coherent story for the role of each term. The attention maps of MulClip, as shown in Figure 2 do seem sharper and more semantically meaningful then Goal, emphasizing the message of the paper, though is not clear to want extent these figures were cherry-picked.

### Weaknesses
The write-up could be somewhat improved as it looks rushed. E.g. Table 1 is not referenced or explained in the text. Equation 6, it is not clear who lower case v tilde is, as far as I can tell the text introduces only upper case V tilde. Is this meant to be v' ? Another issue is understanding the intuitive semantical difference between L_word and L_sub. Are these two losses semantically trying to do the same thing, but are just different objectives of achieving this goal? Are they intuitively/semantically different? At a high level I assume that L_word (since is meant to be a per token loss) is finer grain then L_sub which works with sub-caption. But there is an aggregation step in L_word, that goes from tokens to some arbitrary shorter sequence, therefore is not clear to me if the losses are working on different time scales. 

I think for the ablation question a natural question is what happens if one uses just L_global and L_sub (I think the ablation is only L_global and L_word). That is if I did not misunderstood the different ablated things. This to me would particularly be interesting as I feel like the two objectives are targeting the same thing, but of course would have different effects given their different parametrization. This would be particularly interesting to see as well in terms of attention mask. Note that in quantitative example in Figure 2 you use the label W/o Sub2Patch which I think it should be w/o SAP?  Overall I think someone more used with this line of work could easily make sense of what is going on, the lack of consistency in abbreviation makes the ablation section in general hard to read. I would argue that maybe it would be useful to also give the mathematical formula of the objective for the different labels to make it easier to follow which terms are being included and which terms are not being included.

### Questions
1. Why emphasize *words* as part of long-text structures in abstract? 
2. Table 1 is not referenced in the text, please reference it and explain in the intro the meaning of the different columns (e.g. word, etc) and whether they are a positive or negative trait.
3. Who is lower case v tilde in formula (6)
4. Typo in Table 5 and Table 6? Did you meant "w/o LC  & w/o WP" instead of "w/o LC & w/o SAP"? 
5. What are the different things ablated, what is the formula for "w/o LC & w/o SAP" vs formula for "w/o SAP" ?
6. While the authors have provided the value of the different hyper-parameters in the paper, how have these been tuned, or have they? What is the sensitivity of the model to these hyper-parameters ? Is it robust? Do we need to worry about them?

### Soundness
3

### Presentation
1

### Contribution
3
