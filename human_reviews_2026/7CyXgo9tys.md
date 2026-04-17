# Bioacoustic Geolocation: Species Sounds as Geographic Signals

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Can we determine someone’s geographic location solely from the sounds they hear? Are acoustic signals enough to localize within a country, state, or even city? We tackle the challenge of global-scale audio geolocation, formalizing the problem, and conducting an in-depth analysis with wildlife audio from the iNatSounds dataset. We hypothesize that bioacoustic signals contain informative geolocation cues because of well-defined geographic ranges of species. To test this, we benchmark image geolocation and soundscape mapping methods on iNatSounds. Building on these insights, we propose a hybrid approach that combines species range prediction with retrieval-based geolocation. We further ask whether geolocation improves with species-diverse recordings and spatiotemporal aggregation across neighboring samples. Finally, we extend our study to multimodal geolocation with case studies from movies that combine both audio and visual content. Our results highlight the potential of incorporating bioacoustic signals into geospatial tasks, motivating future work on species recognition and audio geolocation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper studies audio geolocation at global scale. It benchmarks several formulations on the iNatSounds dataset and proposes AG-CLIP, a retrieval approach that aligns audio embeddings to a GeoCLIP location space and adds an auxiliary loss to predict species-range “checklists”. The paper also introduces “species oracles”, which use predicted range maps as an upper bound for species-driven geolocation.

### Strengths
- The paper provides a thorough and well-structured benchmark for audio geolocation.

### Weaknesses
- The positioning of this work regarding the literature is not clear. This paper just describes a bunch of related lines of research (audio geolocation, image geolocation, soundscape mapping, and related audio tasks) without a clear logic. But the scientific contribution is not clear. For example, how is this work meaningfully different from image geolocation's current works? If this is just a matter of replacing an image encoder with an audio encoder (which seems to be the case), I don’t see why this is interesting for ICLR.
- AG-CLIP is essentially a straightforward retrieval model that projects audio into an existing GeoCLIP location space, plus an auxiliary multi-label loss for species checklists. Beyond the auxiliary loss, the approach is largely an adaptation of known retrieval paradigms from image geolocation. The paper itself acknowledges close links to GeoCLIP/SatCLIP/TaxaBind and frames AG-CLIP as a CLIP-style alignment with an extra head. The contribution feels incremental for ICLR.
- The core method (Sec. 3–4.1) is quite condensed; many important choices are only sketched or pushed to the appendix. For ICLR, more clarity and justification are needed in the main text.
- The paper introduces “species oracles” in the methodology section, although they are not part of the deployed method; this blurs contributions versus diagnostic analyses. As the authors note, oracles assume access to location (for checklist construction) and are not realistic at test time, so they should be clearly separated as analysis tools.
- The main novelty is empirical: framing/benchmarking and a modest extension of CLIP-style retrieval to audio with an auxiliary species loss. There is no new theory or learning principle. This feels closer to an application/benchmark paper than a core ICLR contribution.

### Questions
- Please specify the exact loss, normalization, temperature, and negative sampling strategy used in AG-CLIP; clarify which encoders are frozen vs. trainable and why. Also, report sensitivity to these choices.
- How are checklist targets built for each training window without leaking test-time location?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents Audio-GeoCLIP, a contrastive audio-location model for geolocating species audio. The author train an audio encoder that aligns with species checklist and geolocation. In the end, the trained model can be used for predicting species checklist and geolocation of a given audio. Several experimental setups are considered for geolocating a given audio and the model performs resonably well as compared to regression, classification and retrieval-augmented approaches.

### Strengths
1. Overall the quality of the paper and writing is good and the motivation for the paper is meaningful.
2. Extensive experiments are conducted and a range of settings are evaluated for audio geolocation.

### Weaknesses
1. The paper is a very simple contrastive-based model that aligns audio with geolocation. The authors could experiment with other modalities such as text descriptions or other metadata such as time.
2. Is the trained audio encoder a good feature extractor? Can you compare how the trained audio encoder performs in other tasks such as species audio classification?
3. Can you present some analysis that compares the geolocalization error with the uncertainty of predicting the SINR checklist? Does the model perform poorly when checklists are also incorrectly predicted? Does the model need to have a good understanding of the species present in the audio to geolocalize well?
4. The authors should present some embedding visualization analysis and compare it with other models used as a baseline.
5. How do you handle the presence-only nature of audio and species observations for the task? The evaluation need to take this into consideration.

### Questions
Please see weaknesses.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces AG-CLIP, a model that integrates audio signals from species with geolocation information. In addition, the authors propose a new task that uses acoustic signals for geolocation prediction. The benchmark is evaluated on the iNatSounds dataset.

### Strengths
* The paper presents an extensive set of experiments that explore the potential of audio signals for geolocation tasks.
* The authors show that an acoustic signal (i.e., a wildlife audio dataset) can be an important modality for geolocation tasks

### Weaknesses
* The audio encoder used in AG-CLIP is MobileNet-V3, which is not specifically designed for audio processing. More suitable models, such as EsResNet or AudioCLIP, which have been trained on audio datasets like UrbanSound8K, could likely improve performance.
* The proposed AG-CLIP model, one of the paper’s main contributions, is only briefly described in the main text, making it difficult to fully assess the novelty or implementation details.
* The retrieval evaluation may not be directly comparable: AG-CLIP is trained on the full dataset iNatSounds, whereas GeoCLAP and TaxaBind are trained on different datasets. This discrepancy raises concerns about the fairness of the comparison. A stronger baseline is to fine-tune GeoCLAP and Taxabind on the same audio dataset.
* The novelty of the work appears limited. The proposed method largely reuses existing components for audio encoding, and the results do not provide sufficient insight into the model’s limitations or reasons for its weaker performance on certain tasks.

### Questions
* How was the class imbalance in the dataset handled? Was any resampling or balancing strategy applied?
* What factors explain the model’s low accuracy for the City and Region categories in Table 1? Are there specific limitations or failure cases responsible for this performance gap?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents the first large-scale study on bioacoustic geolocation, investigating whether the geographic location of a recording can be inferred solely from audio, particularly wildlife sounds. Using the iNatSounds and XCDC datasets, the authors benchmark multiple geolocation strategies, introduce AG-CLIP (a hybrid retrieval model combining audio embeddings with species range predictions), and demonstrate improved performance through spatiotemporal aggregation. They also explore multimodal geo-forensics, showcasing modality mismatches in movie soundtracks.

### Strengths
The paper’s primary strength is its originality: it represents the first attempt to scale bioacoustic geolocation globally, integrating species range predictions into a deep learning framework. Methodologically, it is thorough, with multiple model formulations, ablations, and dataset comparisons, including a valuable new benchmark of species-rich recordings (XCDC). The interdisciplinary relevance is high, as it bridges machine learning and ecology by operationalizing expert ecological reasoning into a scalable computational model. The scalability and extensibility of the approach suggest broad applicability in areas such as biodiversity assessment, detecting environmental change, and analyzing geospatial multimedia.

### Weaknesses
Despite its strengths, the study lacks detail in ecological interpretation. The manuscript does not clearly articulate how this method may benefit biodiversity monitoring, conservation decision-making, or citizen science, which may limit its relevance for ecological practitioners. Temporal factors, such as seasonality or animal migration, are not accounted for, raising concerns about the generalizability of the results in real-world settings. The performance gap between curated datasets (such as iNatSounds) and passive, multi-species recordings (XCDC) underscores the need for more robust multi-species classifiers or domain adaptation. Finally, the taxonomic composition of the dataset remains unclear, making it difficult to determine how representative or generalizable the model is across different animal groups.

### Questions
Which taxa dominate the 5,500 species in the iNatSounds dataset? A breakdown would help clarify whether the model is primarily learning from bird vocalizations or adapting more broadly to less-studied groups, such as amphibians or mammals, which vocalize less frequently or in different frequency bands.

How is seasonal variability handled in species range predictions (e.g., via SINR)? Many species migrate or vocalize only at certain times of the year, so range-based inference may introduce errors if not seasonally aware. If this were not considered, what are the expected effects on performance, especially in temperate or migratory-dominant ecosystems?

Can the authors expand on the factors contributing to the model’s lower performance on the XCDC dataset? Is it due to increased species overlap, background noise, distribution shift from curated to passive data, or other issues? Insights into this failure mode would be valuable for future work on multi-species bioacoustics.

What real-world ecological applications do the authors envision for this approach beyond academic benchmarking? For example, could this aid in monitoring habitat changes, tracking invasive species, or informing conservation planning in data-poor regions?

Have the authors considered incorporating anthropogenic or environmental audio cues (e.g., traffic, river flow) as additional signals for geolocation to complement bioacoustic species detection in urban or non-wildlife-dominant settings?

### Soundness
2

### Presentation
2

### Contribution
2
