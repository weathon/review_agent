# Towards Better Optimization For Listwise Preference in Diffusion Models

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Reinforcement learning from human feedback (RLHF) has proven effectiveness for aligning text-to-image (T2I) diffusion models with human preferences. Although Direct Preference Optimization (DPO) is widely adopted for its computational efficiency and avoidance of explicit reward modeling, its applications to diffusion models have primarily relied on pairwise preferences. The precise optimization of listwise preferences remains largely unaddressed. In practice, human feedback on image preferences often contains implicit ranked information, which conveys more precise human preferences than pairwise comparisons. In this work, we propose Diffusion-LPO, a simple and effective framework for Listwise Preference Optimization in diffusion models with listwise data. Given a caption, we aggregate user feedback into a ranked list of images and derive a listwise extension of the DPO objective under the Plackett–Luce model. Diffusion-LPO enforces consistency across the entire ranking by encouraging each sample to be preferred over all of its lower-ranked alternatives.  We empirically demonstrate the effectiveness of Diffusion-LPO across various tasks, including text-to-image generation, image editing, and personalized preference alignment. Diffusion-LPO consistently outperforms pairwise DPO baselines on visual quality and preference alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes an extension of DiffusionDPO from pairwise preference to listwise preference where the elements within the list is ranked. The main idea applies Placket-Luce model instead of the usual Bradley Terry model for reward construction. The authors show the advantage of DiffusionLPO over naive extension of DPO such as GP-DPO and show noticeable gains in performance over other baselines on classical datasets and SD model series.

### Strengths
- The presentation is clear and the idea is straightforward. 
- The theoretical justification that naive extension of DPO overestimates the positive/negative gap is illuminating.
- The results show good improvements over baselines over multiple standard datasets.

### Weaknesses
- The baseline comparisons are not sufficient. There many RL-based methods such as FlowGRPO that also optimize for reward signals that the authors miss. How does LPO compare with these online RL techniques?
- The optimized models are quite outdated (e.g. SD1.5 and SDXL). How does the method perform on newer models such as FLUX?

### Questions
I would like the author to address above questions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a novel algorithm to generalize the pair-wise preference learning to list-wise preferences, where a ranking of generations are available. The proposed method use a Plackett–Luce Model to directly model listwise preferences, as opposed converting one list preference to multiple pair-wise preferences like previous works. Through extensive experiments, the author show that the proposed method outperform baselines on T2I generation and image editing.

### Strengths
1. The proposed method is well-motivated and is built on the solid theoretical foundation of Plackett–Luce Model, as opposed to navie converting list preferences to pair-wise ones like GP-DPO.
2. The experiments are comprehensive, covering a wide range of tasks, datasets, metrics and base models. I'm fully convinced about the effectiveness of the proposed LPO based on the presented results.

### Weaknesses
1. I understand that the authors followed the base model choices of most literature in T2I preference learning and used SD 1.5 and SDXL as the base model. However, it would be better if the authors can demonstrate the effectiveness of proposed method on modern rectified-flow models in DiT architecture, such as Sana, Flux, SD3. This would make the contribution more relevant to the community.
2.  While I'm convinced of the effectiveness of Diffusion-LPO compared to baselines presented in Table 1, Table 2 is less impressive, with the win rate against GP-DPO around 50% (i.e random). The author should perform statistical significance analysis (i.e. confidence interval or p-value) to determine if these results are significant or indistinguishable from the null hypothesis (i.e. LPO and GP-DPO has same performance).
3. Pickscore and HPS in general are not a good metrics for image editing tasks. I suggest the author report more common metrics used in image editing. These includes conventional metrics like DINO, L1, CLIP [1].  Alternative, the author may choose to adopt a more "modern" image editing benchmarks that uses GPT4 as the judge model, such as ImageEdit, 

[1] Zhang, Kai, et al. "Magicbrush: A manually annotated dataset for instruction-guided image editing." Advances in Neural Information Processing Systems 36 (2023): 31428-31449.
[2] Ye, Yang, et al. "Imgedit: A unified image editing dataset and benchmark." arXiv preprint arXiv:2505.20275 (2025).

### Questions
See weakness
Additional questions that did not affect my judgment:
What is the complexity of the proposed LPO and other methods listed in appendix B? are they both O(m^2) where m is the size of the list?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Diffusion-LPO, a listwise preference optimization framework for aligning text-to-image diffusion models with human preferences. Unlike prior Direct Preference Optimization (DPO) methods that rely solely on pairwise comparisons, the proposed method introduces Plackett–Luce–based listwise optimization, allowing consistent learning over the entire ranking of generated samples. The authors reconstruct the Pick-a-Pic dataset into ranked lists using DAG aggregation and demonstrate consistent improvements over Diffusion-DPO and DSPO on both SD1.5 and SDXL models, with +12% PickScore gain and improved HPSv2/CLIPScore.

### Strengths
1. The problem setting is clearly defined. The authors’ claim that human preferences are inherently based on ranked lists is well supported by the analysis of  Pick-a-Pic.

2. The paper provides a theoretical justification for why the Plackett–Luce model is more suitable than GP-DPO.

3.  A practical method is proposed for constructing listwise datasets from existing pairwise annotations.

### Weaknesses
1. There is no analysis of performance with respect to different list sizes.

2. The paper does not discuss the computational overhead introduced during training.

### Questions
1. Could the authors provide results when training with a fixed list size? I am curious about which list length contributes most effectively to performance improvement.

2. How do training time and GPU memory usage compare to GP-DPO? Since LPO introduces additional computation, please clarify how the training cost (in both time and memory) scales relative to the group-pairwise baseline.

3. The paper mentions that 54% of pair-wise data could be converted into list-wise form.
What kind of pairwise annotation design would be effective for increasing this percentage ? For example, are there particular annotation patterns or graph connectivity structures that facilitate more transitive ranking reconstruction?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes to align diffusion models with listwise preference. Instead of learning human preferences from ranking pairs, this paper learns human preferences from a ranked list. It extends the DPO objective under the Plackett-Luce model and proposes LPO, which is a framework that can be combined with any DPO-style training losses and achieve further gains. The authors construct a listwise ranking dataset by postprocessing the existing Pick-a-Pic dataset and extracting ranked lists. Experiments show that the proposed method is effective compared to baselines.

### Strengths
1. The idea of extending DPO to listwise preference data is novel. The use of the Plackett-Luce model to extend the Bradley-Terry model is well-motivated. The derivation is theoretically grounded, and the authors provide a clear connection to the diffusion processes. 
2. The observation in the Pick-a-Pic dataset that a large portion of it can be aggregated into lists is important, allowing richer training signals without additional labeling costs. 
3. Strong empirical results show consistent improvements over prior methods.

### Weaknesses
While the authors claim that LPO-style methods outperform prior DPO-style approaches, it is unclear whether the comparison is controlled for the same amount of data and training signal. For instance, are the models trained on the same set of image pairs or groups, and do they receive an equal number of optimization steps or preference annotations? If the listwise method effectively utilizes more comparisons per group (e.g., m(m−1)/2 relations in a list of size m), then it may benefit from a richer signal.

Additionally, the paper lacks a detailed analysis of why the improvement occurs. Is it primarily due to better preference supervision (e.g., higher-order transitivity), reduced noise, or more structured gradients? Such an analysis would significantly strengthen the empirical claims.

### Questions
see Weaknesses. 

Is Diffusion-LPO implemented as an extension of Diffusion-DPO, DSPO, or both? The method is presented as a general framework compatible with arbitrary DPO-style objectives, but most results focus on Diffusion-DPO. Can the authors provide additional results showing that Diffusion-LPO also improves DSPO or other DPO variants, under matched conditions? Demonstrating consistent gains across multiple base algorithms would reinforce the method’s generality.

### Soundness
3

### Presentation
3

### Contribution
3
