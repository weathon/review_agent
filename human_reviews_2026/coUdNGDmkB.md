# Rethinking the Vulnerability of Concept Erasure and a New Method

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
The proliferation of text-to-image diffusion models has raised significant privacy and security concerns, particularly regarding the generation of copyrighted or harmful images. In response, concept erasure (defense) methods have been developed to "unlearn" specific concepts through post-hoc finetuning. However, recent concept restoration (attack) methods have demonstrated that these supposedly erased concepts can be recovered using adversarially crafted prompts, revealing a critical vulnerability in current defense mechanisms. In this work, we first investigate the fundamental sources of adversarial vulnerability and reveal that vulnerabilities are pervasive in the prompt embedding space of concept-erased models, a characteristic inherited from the original pre-unlearned model. Furthermore, we introduce RECORD, a novel tangential-coordinate-descent-based restoration algorithm that consistently outperforms existing restoration methods by up to 17.8 times. We conduct extensive experiments to assess its compute-performance tradeoff and propose acceleration strategies. The code for RECORD is available at ***. 
Note: this paper may contain offensive or upsetting images.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presented the fundamental sources of adversarial vulnerability to be pervasive in the prompt embeddings of concept-erased model inherited from the original model instead of unlearning process. The proposed RECORD coordinate-descent-based restoration algorithm outperforms existing unlearning methods significantly. Extensive experiments shows the compute-performance tradeoff. The analysis of transferability to SDXL and FLUX is a forward-looking plus.

### Strengths
1. This paper presented the good theoretical analysis of vulnerability for unlearned diffusion models. Adversarial embeddings are pervasive in the prompt space and this vulnerability is from the original model, not created by the unlearning process. The behavior of the text embeddings during embedding-level attack provides good proof of this.
2. Coordinate-descent-based restoration algorithm and two-stage candidate selection (linear approximation + exact evaluation) is a clever and practical solution to the combinatorial problem of a large vocabulary.
3. Extensive experiment setup and ablation study. It benchmarks against a wide array of erasure methods (ESD, FMN, AC, SPM, UCE, AdvUnlearn, RECE, ED, SalUn, SH) and concepts (style, objects, nudity). The 17.8x improvement is a outstanding result.

### Weaknesses
1. lack of enough quality assessment with for example FID, CLIP score for the unlearned model RECORD result.
2. there is not much describing the adversarial prompt with RECORD will look like. If it's too long and deviate too much from a normal prompt, it will be easily rejected by real image gen system by simple safe guarding before it reaches to the model. Methods that produce semantically coherent adversarial prompts will be desired.
3. RECORD's loss function requires access to the original SD model, which is fine as the a white-box attack assumption but limits the usage or hurt the performance.
4. The claim for converge to a coordinate-wise local minimum requires further discussion. The guarantee is only for the top-k token selection.

### Questions
1. do you think RECORD can be extended to support other modality instead of text prompt only?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a new token-level concept restoration attack (called RECORD) against concept erasure on text-to-image diffusion models. The token-level approach is motivated by the claim that embedding-level attacks are less successful than token-level ones, but this assertion lacks convincing evidence. Token-level, or more specifically, prompt-level attacks are more realistic when assuming only black-box access to jailbreak a potentially erased model; however, RECORD requires white-box access to the gradients. The novel contribution of the RECORD method is a new coordinate-descent approach to reduce the discrete search space for the most potent attack token sequences. When it comes to baselines, this study does not compare to embedding-level attacks, even though it's mentioned that they are less successful than token-level ones, and misses an important erasure method baseline (STEREO, Srivatsan et al. CVPR'25). Despite these limitations, this study primarily appeals with its insightful analysis sections, for example, the one providing evidence for a dense population of adversarial embeddings in the CLIP space, and additional experiments, such as Appendix B, which provide an honest comparison of the two prevailing embedding-level attack approaches. Overall, this study provides value to the community; however, the primary focus, which is the RECORD algorithm, requires more clarity on its motivation (token-level vs. embedding-level), additional experiments (including more baselines and targets), and a more thorough and fair comparison to related methods.

### Strengths
- (S1) **Insightful Analyses** in Section 3.2 and in most sections of the Appendix, especially in A, B and G.
- (S2) **Creative Visualisations** that help to illustrate the main arguments made in the paper, especially Figures 1, 2 and 4.
- (S3) **Experiments beyond SD v1.4** on the transferability to newer architectures of attack prompts derived from SD v1.4. or even qualitative results of applying RECORD directly to FLUX or SDXL.

### Weaknesses
I find the following list of things to be major weaknesses:
- (W1) **Unclear claims** of superiority of (white-box) token-level attacks over embedding-level attacks (in lines 295-296). What does superior performance mean here? As far as I am informed, embedding-level attacks, such as CCE, are more effective than token-level ones.
- (W2) **Justification for white-box token-level attack**: RECORD requires access to model gradients and is therefore a white-box method. Attacks often employ a token/prompt-level approach because they do not want to assume white-box access to the model or the system. This is why I was wondering how a token-level attack, such as RECORD, realistically justifies its white-box assumption. The transferability of optimized token sequences (prompts) to different models could potentially be used to justify the white-box setting of token-level approaches, as somewhat demonstrated in Table 10; however, the superiority of RECORD over existing attacks is questionable here.
- (W3) **Missing baselines**: CCE as an inversion-based attack baseline and STEREO as an inversion-robust erasure baseline.
- (W4) **Unclear fairness of comparison to baselines**: The comparison is either not fair or not clear enough w.r.t. to the compute-effectiveness trade-off
	- RECORD with J=64 requires a lot more runtime than P4D and UD (see Table 8 in the Appendix).
	- How well would RECORD perform when J is chosen such that the runtime / compute requirements are on par with P4D or UD? Is there a difference, and if yes, is it significantly better?

The following are minor weaknesses:
- (W5) **SH is missing in some tables**: For example, SH is not shown for the "van gogh" scenario in Table 3 (top). Is RECORD failing against it?
- (W6) **Limited evaluation scope** w.r.t. to the targets that are considered (only Van Gogh for style, only three object classes); no explicit content erasure or celebrity erasure scenarios.
- (W7) **Scattered results**: This paper could benefit hugely from compiling the results into an easier-to-grasp and coherent picture by avoiding the numerous scattered tables with often inconsistent baselines throughout the tables/figures.
- (W8) **Only anecdotal results on FLUX/SDXL** to demonstrate the applicability of RECORD on these architectures.

### Questions
- (Q1) Rename "Embed Attack 1" (black-box?) and "Embed Attack 2" (white-box?) to something more meaningful and add citations to most prominent works that use them to the tables.

### Soundness
2

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
This paper reassesses the robustness of concept-erased diffusion models and argues that existing concept erasure techniques are fundamentally vulnerable—not because the erasure is ineffective, but because adversarial prompt embeddings capable of restoring erased concepts are widely scattered in the embedding space and largely inherited from the original pretrained model. This reframes the problem: concept erasure does not eliminate latent pathways for restoration, and current evaluations may underestimate attacker capability.

To substantiate this argument, the authors propose RECORD, a white-box, token-level concept restoration attack that replaces the standard projection-based gradient optimization with a tangential coordinate descent strategy. By selecting tokens directly in discrete space through a two-stage ranking-and-evaluation procedure, RECORD avoids projection errors and search-space mismatch that hamper prior methods like P4D and UnlearnDiffAtk.

### Strengths
1. The paper is well-written, clearly structured, and easy to follow, making both the motivation and technical contributions accessible to the reader.
2. The work provides an insightful reframing of the vulnerability in concept erasure, highlighting that restoration pathways are largely inherited from pretrained models rather than being caused by the erasure methods themselves.
3. The proposed discrete token-level coordinate descent attack is well-motivated and achieves strong performance, outperforming prior methods and establishing new state-of-the-art results.

### Weaknesses
1. The evaluation includes a limited number of erased concepts compared to the experimental settings used in prior baselines. Expanding the set of concepts would strengthen the generality of the conclusions.
2. It would be valuable to include comparisons with more text-encoder-based defense methods. While AdvUnlearn is included, methods such as SAFREE and others are omitted, and adding these would provide a more comprehensive assessment of defense robustness.

### Questions
1. Can the proposed method be extended to the text-to-video (T2V) setting? Since the attack operates on the text embedding space, it seems feasible that the approach could transfer to T2V models as well. The authors’ thoughts on this potential extension would be appreciated.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the authors explored the reasons of concept-erasure vulnerability and found that even though the adversarial embeddings were far from the original ones, they could also generate the corresponding contents. It inspired the motivation to search adversarial tokens in a boarder semantic space. Therefore, they proposed RECORD, a coordinate descent approach that iteratively optimizes the prompts.

### Strengths
1. The paper is well-written, easy for readers to follow.
2. Vulnerability of concept erasure is an important topic. I admire the authors' focus on this field.

### Weaknesses
1. The method is somewhat easy. The experiments in Sec 3.1 and 3.2 are intuitive and persuasive, although previous studies have shown this point. However, the proposed method is only borrowed from recent studies, just as the cited papers in Line 296. The authors did not introduce any novel method based on their empirical observation. 

2. The evaluation is unfair and insufficient to some extent. The used metric is based on the denoising errors. However, the proposed attacking method is also based on the denoising errors. This has resulted in an optimization process in the evaluation method. Also, the results in Tab 2 reveal that this metric is not accuracy. For nudity, it only achieves 87.6% accuracy. Why not to use the common concept detectors such as NudeNet? It is trained on large data and therefore has excellent detection ability.

### Questions
Considering that the authors explored the behaviors of the text embeddings, can we optimize a loss function only on text encoders?

### Soundness
2

### Presentation
3

### Contribution
2
