# Elucidating the Design Space of Text-to-Audio Models

- Avg Score: 6.00
- Decision: Reject
- Scores: 8, 5, 6, 5

## Abstract
Recent years have seen significant progress in Text-To-Audio (TTA) synthesis, enabling users to enrich their creative workflows with synthetic audio generated from natural language prompts. Despite this progress, the effects of data, model architecture, training objective functions, and sampling strategies on target benchmarks are not well understood. With the purpose of providing a holistic understanding of the design space of TTA models, we setup a large-scale empirical experiment focused on diffusion and flow matching models. Our contributions include: 1) AF-Synthetic, a large dataset of high quality synthetic captions obtained from an audio understanding model; 2) a systematic comparison of different architectural, training, and inference design choices for TTA models; 3) an analysis of sampling methods and their Pareto curves with respect to generation quality and inference speed. We leverage the knowledge obtained from this extensive analysis to propose our best model dubbed Elucidated Text-To-Audio (ETTA). When evaluated on AudioCaps and MusicCaps, ETTA provides improvements over the baselines trained on publicly available data, while being competitive with models trained on proprietary data. Finally, we show ETTA's improved ability to generate creative audio following complex and imaginative captions — a task that is more challenging than current benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a 1.35M high-quality audio-caption dataset utilizing the state-of-the-art (SoTA) audio-language model, Audio Flamingo, referred to as AF-Synthetic. Additionally, the paper introduces a text-to-audio system based on the Latent Diffusion Model (LDM) with a Diffusion Transformer (DiT) backbone. Experimental results demonstrate that the proposed Elucidated Text-To-Audio (ETTA) system achieves SoTA performance across multiple metrics on both the AudioCaps and MusicCaps datasets.

### Strengths
The paper introduces a novel filtering pipeline to select the best captions using the CLAP score, developing one of the largest high-quality audio-language datasets. The proposed ETTA generation model is trained across a wide range of diverse audio datasets and achieves competitive scores on both AudioCaps and MusicCaps, demonstrating state-of-the-art performance in both text-to-audio and text-to-music tasks.

External ablation studies investigate the system's performance across different model sizes and training/sampling strategies. Subjective evaluations highlight significant improvements in the proposed ETTA system over baseline models, proving its enhanced ability to generate complex and imaginative captions.

Overall, this paper presents interesting work in the field of audio generation. The authors first introduce a large-scale dataset and then present a state-of-the-art generation system trained using this dataset. Various experiments demonstrate the effectiveness of different methods or modules within the system, concluding with an analysis of the ETTA system's limitations.

### Weaknesses
1. The authors claim the effectiveness of the proposed dataset by demonstrating the SoTA performance of the ETTA system trained on AF-Synthetic. However, the paper lacks sufficient experiments directly comparing the dataset's contribution, such as showing how models like Tango or AudioLDM perform when trained with AF-Synthetic.
2. All the metrics used for evaluating the text-to-audio system are objective, which is generally sufficient but may not always reflect real-world performance. Therefore, including subjective evaluations like the Mean Opinion Score (MOS) would be beneficial, particularly for comparing models that achieve top performance across some metrics.
3. The authors attempt to compare the performance between different baseline models. While ablation studies already demonstrate the effectiveness of the proposed ETTA strategies, the paper lacks experiments where the proposed model is trained on the same dataset (e.g., AudioCaps) to clearly illustrate the improvements contributed solely by the system itself.
4. The ETTA model mainly builds on the backbone of the Stable Audio system. Beyond the training and sampling strategies (Flow Matching and ODE solvers, which are standard improvements in current LDM systems), the key enhancement appears to be the use of Adaptive Layer Normalization (AdaLN) within the DiT structure and the T5-base model for text embedding. These techniques are already implemented in the original DiT paper [1] and other baseline models. As a result, the model seems more like an engineering application with limited novel contributions.

Overall, this is an interesting paper with great effort, I am willing to change the score if the author can solve the questions and fulfil the following experiments:
1. Train baseline models like Tango or AudioLDM on AF-Synthetic and compare to their original performance.
2. Train ETTA on other datasets like AudioCaps or TangoPromptBank and compare performance to when trained on AF-Synthetic. These experiments would help separate the contributions of the model architecture from the dataset.
3. Develop human evaluations on a subset of generated samples, comparing ETTA to top baseline models, such as Mean Opinion Score (MOS) or other equivalent subjective ratings on audio quality, text relevance and so on.
4. More clearly articulate the novel aspects of their model architecture compared to existing work like DiT and Stable Audio. 

[1]. Peebles, William, and Saining Xie. "Scalable diffusion models with transformers." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

### Questions
1. For the comparison of the proposed AF-Synthetic dataset, the paper indicates that both AF-AudioSet and Sound-VECaps have around 161K samples remaining after filtering with a CLAP score of 0.45. Since both datasets are primarily developed from AudioSet, is there a significant overlap in the subset that achieves a higher CLAP score? If so, could this raise concerns regarding the reliability of using the CLAP score as a filtering metric?
2. What are the key differences between AF-AudioSet and the proposed AF-Synthetic, apart from the fact that the latter generates ten captions per sample and selects the one with the highest CLAP score?
3. What distinguishes the AdaLN layer in the proposed ETTA-DiT structure from the one used in the original DiT model mentioned in a previous section?
4. How does the ETTA model perform with improvements limited to the ETTA-DiT component (e.g., when trained solely on AudioCaps)?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper focuses on pushing the performance of diffusion-based text-to-audio models by introducing a new synthetic dataset and revisiting design choices. 
- New synthetic dataset (AF-Synthetic): The authors improve the pipeline from AF-AudioSet to generate 1.35M captions using audio samples from multiple datasets. The result is a large dataset with high CLAP score, while existing datasets are either small or have simple to no filtering method. The authors also point out that AF-Synthetic captions are different than existing datasets.
- The model is built upon Audio-VAE (from stable-audio-tools), Diffusion Transfromer (Peebles & Xie, 2023) with some changes in the architecture.
- The authors compare across datasets and metrics to discover the optimal choice for the number of function evaluations and classifier-free guidance.

### Strengths
The work aims to push the performance of diffusion-based text-to-audio models and show better results than open source models. The improvements come from a better dataset and an extensive evaluation of design/hyperparameter choices built on top of SOTA models. I appreciate the amount of work put into creating the dataset and try many different hyperparameters and design choices, and the commitment to open source the code that appears to push the field ahead.

### Weaknesses
Since this is a purely empirical paper, my main concerns are mostly about evaluation and results.

- Since the dataset is curated using CLAP scores, I find the CLAP score not a reliable indicator, while FAD and KL use old models to extract features.
- There are no subjective scores in the results except for Table 9 which compares the final model with others.
- The improvements in ETTA-DiT, OT-CFM and t-sampling are not consistent across metrics.
- Using AF-AudioSet and AF-Synthetic yields similar results in AudioCaps and MusicCaps despite much larger size (Table 6-7)

Overall, I find that a few conclusions in the paper are not very helpful (e.g. increasing model size improves the performance), especially when we have inconsistent metrics and lack of subjective scores. While it's good to see many hyperparameters being assessed, the contribution is lack of novelty since it does not propose new techniques or reveal surprising findings. The lack of reliable metrics, which authors also admit in the final section, also weaken claims and conclusions.

### Questions
- What can be the reason behind mode-collapsed models consistently produce good scores across multiple metrics, while the backbone for these metrics are different?

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
5

### Summary
The primary focus of this paper is not to explore novel model designs, but rather to provide a comprehensive understanding of the current paradigms in TTA models. It seeks to identify critical factors that contribute to performance improvements and to evaluate scalability concerning data and model size. Additionally, the paper introduces AF-Synthetic, the first million-size synthetic caption dataset with strong audio correlations.

### Strengths
1. The contribution of the dataset: Experiments demonstrate that AF-Synthetic significantly improves performance.
2. It provides a practical guide for hyperparameter tuning in the field of TTA.
3. Each experimental conclusion is highlighted with a purple-bordered box, which makes the paper very reader-friendly and pleasant to navigate.

### Weaknesses
1. As the authors themselves mentioned, the paper doesn't propose a novel method. Moreover, as a paper based on extensive experiments, while it offers valuable conclusions, it lacks an innovative insight or discovery that stands out. Therefore, I believe this paper might be a better fit for the Dataset & Benchmark track.  
2. I would also suggest that the authors include the following three points:  
   - Could the authors further analyze the "potentially mode-collapsed mode" part? In theory, the FD metric should be able to measure sample diversity.  
   - Since the authors have already tried logit-normal *t-sampling*, could they also test the *Min-SNR Weighting Strategy*?  
   - Regarding the Auto-guidance section, it might be worth experimenting with removing CFG (Classifier-Free Guidance) for the first 40% of steps and then adding CFG for the remaining 60% of steps to see if this improves the FD score.

### Questions
Please refer to Weaknesses#2.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper empirically studies the way to improve the current text-to-audio generation system, including creating a new synthetic audio-caption paired dataset, an improved architecture, and other related settings.

### Strengths
1. The author proposed a synthetic data generation pipeline and created the first million-size dataset with high audio-text correlation.
2. The author has implemented a comprehensive list of objective evaluation metrics and shows interesting results during comparison.
3. The author mentioned they will open-source their code for reproducibility.

### Weaknesses
1. As there are a large number of evaluation metrics used. It would be helpful to explain what each evaluation metrics focus on. For example, there are three types of Frechet Distance - what are the differences?
2. It is not clear how ETTA perform without pretraining on the large-scale datasets. For example, what would the result look like if the ETTA model is trained on AudioCaps from scratch (I assume this is a common setup)?
3. Lack of subjective evaluation. As the author mentioned, there is not yet an effective objective evaluation metric for the TTA task. Out of the seven main conclusions in the paper, only one is backed by the subjective evaluation. This can make the conclusion less convincing. 
4. The architectural improvement seems a bit marginal. The improvement of ETTA seems to come from the new synthetic dataset mostly.

### Questions
1. “We initialize the final projection layer of DiT to output zeros.” Do you mean zero-initialize all the weight and bias in the projection layer? Or do you use gated operation? More explanations are welcomed.
2. Please consider adding the result of Make-an-Audio and AudioLDM in Table 2 and Table 3, as they are also very important baselines to compare with.
3. In Table 6, the ETTA trained on AudioCaps shows 3.00 FAD on the audiocaps evaluation set. This is significantly lower than the current state-of-the-art. It would be helpful if the author could explain this result.
4. Line 510: Please say more about the loss divergence issue? It seems contradictory as the evaluation metrics still look normal. More explanations would be helpful.

### Soundness
3

### Presentation
3

### Contribution
2
