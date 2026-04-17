# ViPO: Visual Preference Optimization at Scale

- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
While preference optimization is crucial for improving visual generative models, how to effectively scale this paradigm for visual generation remains largely unexplored.
Current open-source preference datasets typically contain substantial conflicting preference patterns, where winners excel in some dimensions but underperform in others. Naively optimizing on such noisy datasets fails to learn meaningful preferences, fundamentally hindering effective scaling. To enhance the robustness of preference algorithms against noise, we propose Poly-DPO, which extends the DPO objective with an additional polynomial term that dynamically adjusts model confidence during training based on dataset characteristics, enabling effective learning across diverse data distributions from noisy to trivially simple patterns.
Beyond biased patterns, existing datasets suffer from low resolution, limited prompt diversity, and imbalanced distributions. To facilitate large-scale visual preference optimization by tackling key data bottlenecks, we construct ViPO, a massive-scale preference dataset with 1M image pairs (1024px) across five categories and 300K video pairs (720p+) across three categories. Leveraging state-of-the-art generative models and diverse prompts ensures consistent, reliable preference signals with balanced distributions.
Remarkably, when applying Poly-DPO to our high-quality dataset, the optimal configuration converges to standard DPO. This convergence validates both our dataset quality and Poly-DPO's adaptive nature: sophisticated optimization becomes unnecessary with sufficient data quality, yet remains valuable for imperfect datasets.
We comprehensively validate our approach across various visual generation models. On noisy datasets like Pick-a-Pic V2, Poly-DPO achieves 6.87 and 2.32 gains over Diffusion-DPO on GenEval for SD1.5 and SDXL, respectively. For our high-quality ViPO dataset, models achieve performance far exceeding those trained on existing open-source preference datasets. These results confirm that addressing both algorithmic adaptability and data quality is essential for scaling visual preference optimization. Code, models and open-source datasets will be released at: https://github.com/liming-ai/ViPO

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a two-pronged approach to advance the scaling of visual preference optimization. The first contribution is Poly-DPO, a novel algorithm that modifies the standard DPO loss with an adaptive polynomial term, enabling more robust learning from datasets with noisy or conflicting preference labels. The second contribution is ViPO, a new large-scale, high-quality dataset of 1 million image pairs and 300,000 video pairs, designed to provide a more reliable and balanced preference signal than existing resources. A key finding is the synergy between these contributions: Poly-DPO significantly outperforms baselines on noisy datasets, but gracefully converges to standard DPO on the high-quality ViPO dataset.

### Strengths
This paper presents a significant and compelling contribution to the field of visual preference optimization, marked by its dual focus on both algorithmic innovation and high-quality data curation. My evaluation of this work is highly positive, and its primary strengths are as follows:

*   **Elegant and Effective Algorithm Design:** The proposed **Poly-DPO** is a standout contribution. It is exceptionally well-motivated, directly addressing the critical and practical challenge of learning from datasets with noisy and conflicting preference signals. The design is both simple and powerful—by augmenting the standard DPO loss with a single, intuitive polynomial term, it adaptively re-weights samples based on model confidence. This elegant modification requires minimal code changes yet yields substantial performance gains on noisy datasets, demonstrating a high degree of effectiveness and practical utility.

*   **A Landmark Dataset Contribution:** The introduction of the **ViPO** dataset is a massive contribution to the community. Beyond its impressive scale (1M image and 300K video pairs), its true strength lies in its thoughtful, categorical construction. By organizing preferences into distinct dimensions such as Aesthetics, Text-Image Alignment, and Composition, the authors ensure the dataset is balanced, reliable, and capable of driving comprehensive improvements across a wide range of model capabilities. The data generation strategies are both clever and pragmatic—for instance, creating image-text alignment pairs through prompt perturbation and using OCR to generate robust preference labels for text rendering. This meticulous approach sets a new standard for preference dataset quality.

*   **Comprehensive and Rigorous Experimental Validation:** The paper's claims are substantiated by an extensive and exceptionally well-designed set of experiments.
    *   The **ablation studies** (particularly Figure 4) are cleverly constructed. They not only provide convincing evidence that the "noisy preference" problem is a real and significant obstacle in existing datasets but also clearly demonstrate Poly-DPO's ability to mitigate this issue effectively.
    *   The experiments systematically validate each component of the work. The authors wisely use public benchmarks like the Pick-a-Pic dataset and standard models like SD1.5 to transparently demonstrate the standalone effectiveness of the Poly-DPO algorithm. Subsequently, they showcase the powerful synergy of their algorithm and new dataset, demonstrating state-of-the-art results and significant improvements across a diverse array of modern generative models, including SDXL, FLUX.1, and Wan2.1. This thorough validation leaves little doubt about the efficacy of the proposed methods.

### Weaknesses
Despite the paper's numerous strengths and high-quality contributions, there are a few areas that could be further clarified or represent inherent limitations of the chosen approach. These points are offered to help refine what is already an excellent piece of work:

*   **Reliance on AI-Generated Labels for the ViPO Dataset:** The primary limitation is that the ViPO dataset's preference labels are generated exclusively by AI models (e.g., VLMs). While this AI-driven pipeline is a key strength for achieving unprecedented scale, it introduces a fundamental trade-off. 

*   **Practicality of Hyperparameter Tuning for Poly-DPO:** While the Poly-DPO algorithm is simple in its formulation, its effectiveness on a new dataset relies on finding an optimal value for the hyperparameter `α` through a grid search. This process can be computationally intensive and presents a practical barrier to easy adoption. The authors correctly identify the automation of `α` as a direction for future work.

*   **Minor Typos and Notational Inconsistencies:** The paper contains several minor but noticeable typographical errors that should be corrected for the final camera-ready version. These do not detract from the core scientific contributions but do affect the overall polish and clarity. For instance:
    *   There is an inconsistency in the naming of the dataset, appearing as both "**ViPO**" and "**VIPO**" in the abstract and elsewhere.
    *   In the introduction, "SD1.5" is incorrectly written as "**SD15**" (line 86).
    A thorough proofread to catch these small errors would improve the professionalism of the final manuscript.

### Questions
1. The paper would be even more impactful if it included a small-scale human validation study to measure the correlation between the VLM-generated labels and actual human judgments. By the way, one of related reference papers can be cited (X-IQE: eXplainable Image Quality Evaluation for Text-to-Image Generation with Visual Large Language Models).
2. A discussion on how dataset statistics (like the conflict rate mentioned for Pick-a-Pic) could inform the initial search range for `α` would significantly enhance the method's insight for other researchers.
3. The categorical construction of the ViPO dataset is a major strength. Have you performed any analysis on the relative importance of these different categories (Aesthetics, Composition, etc.)? For instance, do you find that training on certain categories, like Text-Image Alignment, contributes disproportionately to the overall improvements seen on general-purpose benchmarks like GenEval?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes ViPO, visual preference optimization at scale. It includes two contributions: (a) Poly-DPO which proposes including a polynomial term alongside the standard diffusion loss to dynamically weight preferences in the face of uncertainty in the preference datasets, (b) ViPO dataset, a massive-scale preference dataset with 1M image (1024resolution) pairs across five categories  and 300K video pairs (720P+ resolution) across three categories. Numerical experiments have been done to evaluate the proposed method on preference optimization compared to the baselines.

### Strengths
- Large scale preference optimization dataset for image and video generation 
- Evaluation on a range of image and video generation models to show the efficacy of the proposed method

### Weaknesses
- It is unclear what’s the impact of Polynomial component in Poly-DPO on a decent DPO formulation. Most of the datasets referred in the paper are quite old and would not be used by any modern SOTA T2I models for preference optimization. Besides the alpha becomes yet another hyper-parameter in the DPO formulation which is already quite prone to collapse
- One concerning issue with the VIPO dataset is that SFT with this dataset does not seem to improve models like SD3.5-medium and FLUX.1 dev on DPG bench which is harder benchmark compared to GenEval. This could point to biases in the dataset construction. 
- Claims the new insights regarding existing datasets (low resolution images, outdated models, conflicting win/lose pairs) as a contribution, but this has been observed earlier in the literature.

### Questions
- New insights regarding existing datasets is not that new. i.e., These datasets also suffer from low resolution (512-768resolution), limited prompt diversity, outdated generation models. Many existing works have demonstrated the same thing. For instance, see RankDPO.
- Why not evaluate SDXL and SD1.5 on DPG-Bench benchmark like in Tab. 5? Why only show the GenEval scores?
- What’s the recommended value of alpha in poly-loss? How does one tune this hyper-parameter for a new preference dataset?
- For the proposed new dataset, since the recommended value of alpha=0 ==> its standard diffusion DPO. Does it mean that Poly-Loss does not help at all? Because the illustrated examples in which alpha!=0, are very old datasets and would no longer be used due to their outdated generation models and low-res images.
- In Tab.5, why does SFT not improve SD3.5-medium and Flux.1 Dev performance? Technically, the ViPO-1M image dataset is better in quality. 
- How robust is the proposed Poly-DPO is to model collapse which is observed very frequently with DPO formulation with longer training?

Missing References:
- DSPO — Direct Score Preference Optimization for Diffusion Model Alignment https://openreview.net/forum?id=xyfb9HHvMe 
- RankDPO — Scalable Ranked Preference Optimization for Text-to-Image Generation:  https://arxiv.org/abs/2410.18013  
- Flow-GRPO— Training Flow Matching Models via Online RL : https://arxiv.org/pdf/2505.05470 
- Bridging SFT and DPO for Diffusion Model Alignment with Self-Sampling Preference Optimization https://arxiv.org/abs/2410.05255

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents several contributions for diffusion DPO, including large-scale datasets of 1M image pairs and 300K video pairs, and Poly-DPO as an extended algorithm to standard DPO. The new datasets are to address limitations of existing datasets, e.g. low resolution, and conflicting winning patterns. Methods trained on the new datasets got substantial improvements. The Poly-DPO is obtained by taking DPO as binary classification, and Taylor expansion of the standard cross-entropy loss. In the experiments, Poly-DPO outperforms Diffusion-DPO. It can also reveal problems of a dataset for being noisy or over-simple.

### Strengths
- The large-scale preference datasets are valuable to the community. 
- The Poly-DOP improves the standard Diffusion DPO.
- Experiments are comprehensive and convincing.

### Weaknesses
- Poly-DPO is considered as an incremental change to DPO
- Preference pairs are judged by one or more VLMs. It’s not clear about the quality of annotation by aligning with human preference. To improve, authors can conduct cross checks with human raters on a small portion to verify the alignment.
- Minor writing issues
  - Line 075: learning -> learn
  - Line 086: SD15 -> SD1.5

### Questions
- How to demonstrate that ViPO doesn't have the problem of conflicting winning patterns?
- How to verify if ViPO is aligned with human preference?

### Soundness
3

### Presentation
3

### Contribution
3
