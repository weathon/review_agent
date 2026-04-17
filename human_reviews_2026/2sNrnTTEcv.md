# Uncovering Conceptual Blindspots in Generative Image Models Using Sparse Autoencoders

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Despite their impressive performance, generative image models trained on large-scale datasets frequently fail to produce images with seemingly simple concepts -- e.g., human hands or objects appearing in groups of four -- that are reasonably expected to appear in the training data. These failure modes have largely been documented anecdotally, leaving open the question of whether they reflect idiosyncratic anomalies or more structural limitations of these models. To address this, we introduce a systematic approach for identifying and characterizing "conceptual blindspots" -- concepts present in the training data but absent or misrepresented in a model's generations. Our method leverages sparse autoencoders (SAEs) to extract interpretable concept embeddings, enabling a quantitative comparison of concept prevalence between real and generated images. We train an archetypal SAE (RA-SAE) on DINOv2 features with 32,000 concepts -- the largest such SAE to date -- enabling fine-grained analysis of conceptual disparities. Applied to four popular generative models (Stable Diffusion 1.5/2.1, PixArt, and Kandinsky), our approach reveals specific suppressed blindspots (e.g., bird feeders, DVD discs, and whitespaces on documents) and exaggerated blindspots (e.g., wood background texture and palm trees). At the individual datapoint level, we further isolate memorization artifacts -- instances where models reproduce highly specific visual templates seen during training. Overall, we propose a theoretically grounded framework for systematically identifying conceptual blindspots in generative models by assessing their conceptual fidelity with respect to the underlying data-generating process.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors analyzed how text-to-image generative models fail to represent certain visual concepts that are present in real-world data. They define these systematic failures as cases of “conceptual blindness”. The paper provides a formal definition of this phenomenon by modeling the true data-generating process as an invertible function mapping latent concepts to observed images.

To explore this idea, the authors train a Sparse Autoencoder (SAE) on DINOv2-extracted features to identify latent “concepts” in both real and synthetic images. By comparing the frequency of concept activations across the two distributions, they detect under-represented concepts, which is interpreted as blindspots of the generative model.

The authors conduct analysis on four text-to-image models (Stable Diffusion 1.5, Stable Diffusion 2.1, PixArt, and Kandinsky), identifying both universal and model-specific blindspots. They further explore correlations across models and connect the discovered blindspots to broader phenomena such as memorization and mode collapse, as well as how post-training techniques influence the distribution of conceptual coverage.

### Strengths
- The paper is well motivated, aiming to better understand the behavior of text-to-image generative models, particularly their tendency to violate some realworld concepts.
- The authors propose a formal definition of conceptual blindspots that goes beyond qualitative analyses. The experimental setup of using the same textual prompts for real and synthetic images makes the two distributions highly comparable.
- The paper is well written and easy to follow, with clear structure and visual illustrations that make the methodology and findings intuitive.
- The use of RA-SAE trained on DINOv2 features provides a scalable and unsupervised approach to discover a large number(32000) of latent concepts without human labeling. The appendix provides detailed procedures for automatic concept labeling with LLMs, adding clarity and reproducibility.
- The empirical analysis covers 4 recent diffusion models and revealing both universal and model-specific blindspots. The additional analyses such as studying post-training effects are informative and provide practical insights on how representation coverage might be improved.
- The paper draws meaningful connections between conceptual blindness, memorization, and mode collapse, positioning the work within broader discussions on generalization and coverage in generative modeling.

### Weaknesses
- The analysis focuses exclusively on text-to-image diffusion models. While this is a reasonable starting point, there are other types of text-to-image generators such as autoregressive and GAN-based models. Including at least one non-diffusion baseline would strengthen the generality of the paper’s claims and help assess whether conceptual blindspots are architecture-specific or model-agnostic.
- The paper’s title refers broadly to “generative image models”, but the experiments and discussion focus only on text-to-image models. Adjusting the title or framing to more accurately reflect this scope would improve clarity and alignment between the claim and the content.
- In Section 2, the authors assume that the data-generating process (DGP) is an invertible mapping between latent concepts and images, with factorized latent priors. In practice, visual concepts are often correlated and non-invertible (e.g., “wooden texture” co-occurs with “brown color”). The authors briefly mention (L110–113) that empirical findings remain meaningful even when these assumptions are violated, but it is a bit unclear how relaxing these assumptions would affect the validity of their theoretical framework or the interpretation of results.
- The qualitative examples in Figures 7 and 8 are not fully convincing. For example, in Figure 7, none of the four generated images clearly display the feature “shadow under animal.” In Figure 8a, all models appear to generate reasonable outputs for Following directions on worksheet, though the amount of whitespace varies (and SD 1.5 still shows some). It is unclear whether these examples truly demonstrate conceptual blindspots or instead reflect weak coupling between the text prompt $t$ and the concept $c_k$
- The analysis relies solely on DINOv2 as the encoder for feature extraction. DINOv2 is trained with a contrastive objective emphasizing invariance to color, rotation, and background attributes that are crucial for generation. As a result, only using DinoV2 may underrepresent certain visual factors. It would be interesting to explore whether using a different encoder, such as MAE or a perceptual encoder, changes the resulting concept decomposition or blindspot patterns.

Minor typos: 
- Figure 2 caption:$S_\{theta}$ should be $g_\{theta}$ 
- Figure 4 caption: Values left to 0.5 (L282), not left to zero.

### Questions
- What would be the impact on the thoeretical framework if the assumotions in section 2 are violated?
- Would the findings generalize to other architecture of text to image models?
- How are the threshold of 0.1 and 0.9 determined for $\delta(k)$?
- Have the authors tested other feature encoders (e.g., MAE, CLIP, or perceptual networks)?
- Did the authors perform any human evaluation or concept-level annotation to validate that the detected “blindspots” align with human judgment?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this work, authors introduce an SAE based approach to systematically identify and characterize "conceptual blindspots" in generative image models. In particular, these blindspots are defined to be concepts that are reasonably expected and present in the training data but misrepresented in generations. Authors train RA-SAE's on DINOv2 features to perform fine-graine analysis of conceptual disparities between real and generated images. Authors apply their technique on SD1.5/2.1, PixArt and Kandinsky models and reveal suppressed or exaggerated blindspots.

### Strengths
* The paper is very well written, and together with the well-designed figures, I truly enjoyed reading the manuscript.
* The work is contextualized well within the existing literature.
* The proposed technique for automatically mining the blindspot failure modes of generative image models, as well as the formalism of blindspots itself, is both sound and timely. I believe the ICLR community would find these results, and the interactive tool, highly interesting. 
* In terms of reproducibility, authors did an excellent job in disclosing dataset, implementation and compute resource details.

### Weaknesses
* While not a major concern, there appears to be an implicit conceptual assumption that the discovered concepts in the SAE are non-redundant for energy discrepancy to be meaningful. It is unclear whether this assumption holds in practice. If redundancy exists empirically, it should be discussed in terms of its impact on the interpratation of results. Otherwise, it would be helpful to demonstrate that the discovered concepts are indeed unique/distinct. More on this is discussed in the questions below.
* Please see the other questions below as well.

### Questions
***Questions and Suggestions:***
* Did the authors observe any redudancies among the concepts learned by the SAEs (e.g., different indices of the concept embedding corresponding to the same concept)? 
	* If not, is it evident that the adopted SAE training approach inherently ensures the discovery of distinct concepts?
	* If such redundancies do exist, consider the following example: suppose the first and second entries of the concept embeddings both correspond to "concept A". Is it possible for real images to tend to activate the first entry but not the second, and the other way around for a synthetic images? In that case, the proposed method might incorrectly identify "concept A" as a blindspot (in fact as both suppressed and exaggerated).
* I would recommend discussing Surkov et. al. [1] for completeness in Appendix A.4.
* In Section 4.2, the authors mention that some blindspots likely stem from dataset characteristics, while others appear to be model-specific. Based on your results and intuition, do you have any promising hypotheses on how these blindspots could be mitigated or eliminated?
* Figure 11 suggests that blindspots tend to correspond to rare concepts in the natural dataset; however, not all rare concepts are blindspots. Do you have any intuition about why this might be the case?
* How were the $\delta$ thresholds $0.1$ and $0.9$ chosen?
* What motivated the decision to train the SAE on ImageNet rather than LAION?
* Have you encountered any "counting" related blindspots (e.g. number of objects) in any of the models that you've tested ? If so, were there any interesting observations?

***
***References:***

[1] Surkov, Viacheslav, et al. "One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models." arXiv preprint arXiv:2410.22366 (2024).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a framework to identify "conceptual blindspot" in generative models by using sparse autoencoders to extract interpretable concept and quantitatively comparing the energy between real and generated images. The framework distinguishes between two types of blindspots: suppresed blindspots and exaggerated blindspots, and applies to a range of models.

### Strengths
- The paper offers a new perspective on understanding the failure modes of generative models and provide mathematical definition of conceptual blindspot.  
- The method can scale easily in an unsupervised manner, free from human inputs or manual checks.
- The paper is very well written

### Weaknesses
- Including error analysis and failure modes of the approach and discussing those will strengthen the paper, where the SAE produces spurious concepts or misses obvious ones. 
- Baseline comparison is missing. How is the proposed method compared to other framework proposed in the literature. Do the observations agree?
- What about mitigation? No proposed mitigation strategy on how to reduce the blindspots? How to combine this with classifier-guided generation, for example, use the proposed energy-based model and langevin dynamics to sample better images.

### Questions
- Dependence of DINOv2 features, have you tried any other feature extractor, how is the method when work with not as good extractor?
- What about the hierarchical concepts (vs compositional concepts)?
- Can you elaborate more on the concept dictionary, how it is trained/intialized? How to map those to human understandable concepts for further analysis?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a novel framework to systematically discover suppressed or exaggerated conceptual blindspots. In particular, the authors first trained a large sparse autoencoder (RA-SAE) on DINOv2 features to extract an interpretable concept basis. They then define energy differences between the prevalence of each concept in real vs. generated images to pinpoint these blindspots.

### Strengths
- The problem is novel and well-motivated, and the paper is well-written.
- While I feel like some of the observations are not new (for instance Sec 4.6), I think it is nice that the framework can be used to quantify them.
- The open-source framework is highly appreciated.

### Weaknesses
- Section 4.4 would benefit more from a more systematic study and quantitative results. For instance, when the authors claim that “While some of these discrepancies can be attributed to underspecified or noisy captions, others reveal genuine blindspots”, I think it would be better to quantify what portion of the images with large energy difference has noisy captions.

### Questions
- I don’t think the title of Section 4.1 captures its contents? Moreover, some works have shown that T2I models are able to generate unseen concepts [1]

References:
[1] Haviv, A., Sarfaty, S., Hacohen, U., Elkin-Koren, N., Livni, R., & Bermano, A. H. (2024). Not Every Image is Worth a Thousand Words: Quantifying Originality in Stable Diffusion. arXiv preprint arXiv:2408.08184.

### Soundness
3

### Presentation
3

### Contribution
3
