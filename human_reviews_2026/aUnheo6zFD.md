# SmellNet: A Dataset for Sensor-Based Smell Recognition and Mixture Prediction

- Decision: Accept (Poster)
- Scores: 4, 8, 2, 6

## Abstract
The ability of AI to sense and identify various substances based on their smell alone can have profound impacts on allergen detection (e.g. detecting peanut contamination or allergens in food), monitoring the manufacturing process, and sensing hormones that indicate emotional states, stress levels, and diseases. Despite these broad impacts, there are few standardized datasets, and therefore little progress, for training and evaluating AI systems' ability to "smell" in the real-world. In this paper, we use small gas and chemical sensors to create SmellNet, a comparatively large dataset for sensor-based machine olfaction that digitizes a diverse range of smells in the natural world. SmellNet contains about 828,000 time-series data points across 50 substances, spanning nuts, spices, herbs, fruits, and vegetables, and 43 mixtures among them with fixed ingredient volumetric ratios, with 68 hours of data collected. Using SmellNet, we developed ScentFormer, a Transformer-based architecture combining temporal differencing and sliding-window augmentation for smell data. For the SmellNet-Base classification tasks, ScentFormer achieves 63.3% Top-1 accuracy with GC-MS supervision, and for the SmellNet-Mixture distribution prediction tasks, ScentFormer achieves 50.2% Top-1@0.1 on the test-seen split. ScentFormer's ability to generalize across conditions and capture transient chemical dynamics demonstrates the promise of temporal modeling in sensor-based olfactory AI. SmellNet and ScentFormer lay the groundwork for sensor-based olfactory applications across healthcare, food and beverage, environmental monitoring, manufacturing, and entertainment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper the authors contribute SmellNet, a novel large-scale dataset of chemical data for olfactory recognition tasks. In particular, the authors collect over 68 hours of gas sensor data, pertaining to 50 different substance and 43 artificial mixtures of compounds. Furthermore, the authors present SmellFormer, a transformer-based architecture for odor prediction from sensor data. Finally, the authors extensively evaluate ScentFormer on single-component and mixture odor recognition data, highlighting how their transformer architecture outperforms other standard architectures (MLP, CNN, LSTM) and the potential of their dataset for future applications in olfactory machine learning.

### Strengths
- **Originality**: While large-scale datasets of olfactory phenomena exist and have been extensively used by the community for olfactory prediction tasks ([1, 2]), the dataset presented here distinguishes itself by using directly sensor readings, collected using consumer-available hardware (instead of GC-MS and other specialized equipment).

- **Quality**: I found the paper to be of high quality, with minimal typos, high-quality figures and text. The authors extensively evaluate the discriminative power of the collected dataset, both in the main text and in appendix. Furthermore, the authors evaluate different experimental setups, in terms of preprocessing steps, additional modalities and model architectures, for single and mixture odor prediction.

- **Clarity**: Overall, the authors describe in extensive detail their data collection setup (hardware used and processing pipeline) and model architecture (preprocessing steps, architecture and training objectives). I also deeply appreciated the structure of the evaluation section, with clear research questions and highlighting the main findings for each of them.  

- **Significance**: The lack of large-scale, diverse, olfactory data, collected in-the-wild, is currently a major bottleneck for the olfactory machine learning community. While being part of ongoing data collection efforts by the community, the use of consumer-available hardware to collect SmellNet makes this work a potential interesting resource for the community.

**References**:

- [1] Lee, Brian K., et al. "A principal odor map unifies diverse tasks in olfactory perception." Science 381.6661 (2023): 999-1006.
- [2] Taleb, Farzaneh, et al. "Can transformers smell like humans?." Advances in Neural Information Processing Systems 37 (2024): 72032-72060.

### Weaknesses
- I would suggest the authors to refrain from over claiming in certain statements. For example, statements such as "smell is a new data modality for AI (line 51)", or "large-scale AI for smell is completely unexplored (line 91)" do not only overlooks important advances from the chemical and neuroscience community (some of them, already cited in the paper) on using AI for olfactory prediction tasks, but it is also factually incorrect: "large-scale AI" (whatever that means) have been used for olfactory prediction tasks, see [1], [2], for example.

- I found it quite interesting that throughout the work the authors state that they collect sensory gas data in 12-channels, and evaluate their dataset under these conditions (Section 3.4), yet the first step in data preprocessing (Section 4.2) is to drop 6 of those channels, due to potential malfunctioning of these sensors. From a practical point of view, the authors should only release the 6-channel dataset (which I don't think would significantly decrease the novelty or significance of the work), or redo the data collection with functioning sensors. Releasing the full 12-channel dataset, where 6 of those are malfunctioning, can lead to future issues when people may employ the dataset without removing the channels. Is there any reason to release the full dataset, beyond the ones stated in Line 268-270, which are not particularly beneficial considering the previously discussed danger?

- While the authors describe extensively the statistics and experimental apparatus to collect SmellNet, I found the description of the GC-MS pairings (which, accordingly to the results in Section 5, improves the performance of the predictive models) to be lacking and confusing. GC-MS is a high-precision measurement of the compounds present in a sample and, as such, varies across samples. Moreover, to the best of my knowledge, there exists no single GC-MS measurement/spectrum for some of the samples employed in your dataset (e.g., Apple, Pineapple). How did the authors collect this data? Moreover, how did the authors take the spectrum data and get the atomic counts (which is an unusual representation of GC-MS data, what about concentration for example)? The Appendix C mentions the use of an LLM for this purpose, yet it's unclear what is the process. Also, this use of LLMs for technical content goes against what is claimed in Appendix K, regarding the use of LLMs, which the authors to be used "solely for light copy-editing".

**References**:

- [1] Lee, Brian K., et al. "A principal odor map unifies diverse tasks in olfactory perception." Science 381.6661 (2023): 999-1006.
- [2] Taleb, Farzaneh, et al. "Can transformers smell like humans?." Advances in Neural Information Processing Systems 37 (2024): 72032-72060.

### Questions
See Weaknesses.

### Soundness
2

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
This paper introduces a new olfactory dataset, combining temporal sensor data with with object labels, and proposes several models to decode single- and multi-source odorants to establish the baseline results for these data.

### Strengths
- A new dataset

The paper offers a new large-scale dataset connecting the odorants and the objects emitting them. The dataset has a reasonable number of odorant (50; typical for olfaction; compare to ~100 in the Leffingwell dataset) and a large number of data point (recorded by the sensors), thus enabling the application of larger machine-learning models than previously (e.g., GNNs and CNNs).

- Complementary approach to olfactory data

While conventionally the olfactory datasets were focused on psychophysical processing of the smell (exposing human participants or animals to smells and recording their semantic or similarity reports respectively), this dataset fully avoids the perceptual properties of the smell, going directly from sensor readouts to object classification. While this choice shifts the scope of what's can be learned from the olfactory dataset, it also enables the large-scale data collection that scales easily and thus, in turns, enables the training of larger models.

- Thorough considerations for the data collection

A lot of thought has been put into the ways these data were collected, resulting in a dataset that contains sufficient information to learn from and guiding us in the ways how such data can be collected and scaled. In particular, the Authors have made the informed choices of the sensors (NO2, C2H5OH, VOC, CO, Alcohol, LPG), data collection modalities (temporal-difference), and standardization (controlling for the external air parameters; adding repetitions). All these choices were guided by direct data analyses.

- A comprehensive set of baseline models and metrics

The Authors have tested a comprehensive set of standard baseline models (MLP, CNN, RNN, Transformer) and metrics (top-1 and top-5 accuracy; F1-score) on three tasks (sensor-only; cross-modal; mixture) showing that (1) the labels are decodeable from the data and (2) that they are not saturated near 100%, leaving room for improvement. Both these properties are highly important for a benchmark dataset.

- Text clarity

The text is written clearly and structured nicely. The figures and the tables illustrate the concepts in the text well.

### Weaknesses
- The results are somewhat overstated

I found the results of the comparisons between the Transformer and the other models here to be overstated. While the Transformer surely shows some of the best results here, it's outperformed by the LSTMs and CNNs on several tasks (e.g. in Table 2). Likewise, I wouldn't say that "GC-MS supervision strongly boosts weaker models" -- I found the numbers that this statements refers to to be quite similar to each other, which is typical for transfer learning. Finally, I wouldn't focus on the claim that the dataset is much larger than the existing olfactory datasets, as these are two very different types of data: the latter involve psychophysics while the former one doesn't.

- Minor: The emphasis on the mixtures is unclear with this type of data

Typically, mixtures receive separate consideration in olfaction due to the differences in mass transfer between the molecules, differences in their affinity for the olfactory receptors, and synergetic effects in their perceptual qualities. None of these properties apply to the sensors or other the odor-emitting objects. Thus further discussion may be helpful to determine why the detection properties for the mixtures cannot be modeled as linearly dependent on the properties of individual smells.

### Questions
- What is the source of the temporal dynamics in the data?

While one of the results here regards the usefulness of the remporal data, it's unclear from the text where the temporal variations stem from. Is it because the sensor is brought closer to and then further from the odor sourse to mimic the inhalation? Or is it sue to the variability of the sensor readout at the constant distance from the odor source? In ether case, what do the temporal changes represent and why may they be important? What has motivated the collection of the temporal data in the first place?

- Could you please further comment on the utility of this type of data?

As this dataset abstracts form the brain and the processing witin it, the scope of the use case for this dataset differs from that of conventional olfactory datasets. Could you please detail the potential use cases for your type of data (some of them are mention in the introduction) and discuss the differences with the use cases for the conventional data?

- Minor: Why are C2H5OH and Alcohol different channels?

Seem to be the same thing to me.

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
The authors propose SmellNet, a database of olfactory properties for both single-object and mixtures related to food. The authors gather this data for these objects placed in a controlled environment and use a metal oxide gas sensor to detect specific compounds, and track other ambient parameters alongside the data collection.  Using this data, the authors train a series of models for identifying the objects or mixture compositions via its sensory and ambient fingerprints. The authors evaluate the model’s performance to generalize across temporal domains for single objects, and also on unseen mixture compositions. Olfaction has long been a data-scarce domain, and I commend the authors’ efforts in bridging this gap for machine learning. However, from my perspective, there are some serious issues with the quality of the data, the methodology that the authors adopted, and the evaluation of the models proposed. Therefore, I am rejecting this paper, but am willing to raise my score if the authors address the numerous concerns I will detail below.

### Strengths
I think the writing is clear, and that the authors have posed their research questions in a structured format that makes it easy to understand the motivations for performing the experiments throughout the work. 

The authors design a hardware platform for the collection of this data. Recognizing the limitations of sensor data, they attempt to pair sensory data up with high-quality GC-MS data for cross-modal learning. The authors also consider the role of sequential and non-sequential architectures for model performance.

### Weaknesses
I think the authors need to provide a stronger motivation for adopting gas sensors over other platforms beyond “portability”. Additionally, the authors justify the choices for the sensors as “common odors found in food, drinks and other common substances”. I am not convinced that typical food items contain large amounts of alcoholic, petrochemical and nitrogen oxide vapors, and it would be worrying if this was the case. For many of these natural products, the volatiles are mostly relatively complex organic compounds, which I am also not confident that the VOC sensor is able to capture beyond the fact that all these volatiles would be crammed into the same signal channel. Figure 9 in the SI also shows strong correlations for the NO2, C2H5OH and VOC sensors. Because the prior for choosing these sensors is not strong, I am worried that there is significant overfitting present.

Other evidence that points towards overfitting/data leakage exists in Table 16. Based on what I can see it seems possible to already build a reasonable classifier (e.g. tree-based model) just from temperature, pressure, humidity, gas resistance and altitude. Angelica for example can be classified by temperatures greater than >~26 and <33.59. 

The GC-MS processing seems strange -- are the top 10 ingredients being completely broken up into its constituent atoms to form a representation? I had thought that the raw GC-MS data (i.e spectral data) was being used, but this form of considering only the atomic counts appears to be rather strange because the molecular structure will clearly affect its olfactory properties (and its ability to bind to the sensors). The PCA performed on top of the GCMS representation shows an extremely large PC1 importance -- and this could be because the ingredients are clearly separated by the presence/absence of certain elements, or atomic counts. Another slight weakness is that the GC-MS data from FooDB does not directly correspond to the same substance’s measurement under the same conditions (given natural variations in food products), so the gathered gas sensor data could potentially be incongruent with the GC-MS traces.

In terms of the utility of the models trained, querying requires measurement under the same pristine conditions and with the same sensor setup as well -- limiting its widespread use for classification. It would have been interesting to see this coupled to some data modality that is easily accessible to increase the scope of things that can possibly be predicted. 

In alignment with the points above, I find it concerning that the authors claim that this dataset is relevant for the real-world when 1) the data was gathered in controlled environments to minimize environmental factors, which is not representative of the objects in the real-world, and 2) despite their efforts to control the environment they report wildly varying metrics for their environments.

I’m not entirely sure of the cost of model training, but testing the model could have been done across all other combinations of the leave-one-day-out approach the authors used for SmellNet-Base to show the robustness of their data modelling efforts.

The authors also report the limited generalizability of their approach in Table 4 for unseen combinations, but it’s not clear to me how these ratios were constructed, given that different masses of compounds lead to different concentrations and proportions of volatiles in the headspace, etc.

Finally, the authors evaluate the mixture proportion prediction task using a Top-x@0.1 metric. I’m not sure how 0.1 was decided, and if 0.1 is even an acceptable tolerance for error. The authors evaluate binary and ternary mixtures, and it’s not clear to me if weights are also predicted for more components than there are in the training set. The mixture ratios detailed in Appendix J are also rather limited and can almost be decomposed into a classification problem where the relevant odorant is identified within the mixture and its subsequent ratio-class is predicted. I think cosine similarity would be a more appropriate metric for this purpose.

### Questions
1) Do you have an explanation on why you chose to use gas sensors, and the specific sensors within your platform?
2) Please provide an ablation where either the sensory data or the environmental data is removed from the training data.
3) Why are the sensor outputs said to be qualitative (Section 4.2), but quantitative sensor readings are provided? 
4) Please provide leave-day-out model performance metrics at least for SmellNet-Base.
5) What is the proportion of the GC-MS data that you could gather from FooDB? Is there an ablation where molecules that do not have the GC-MS data are eliminated, and the performance metrics are reported on a model trained with and without the GC-MS data?
6) Please show the performance in terms of cosine similarity for the mixture evaluation. 
7) Is it surprising to you that the models can’t learn the time lag and that it is a requirement for you to inject this human-engineered feature into the model for increased model performance?

### Soundness
1

### Presentation
2

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
This paper presents SMELLNET, a dataset of gas sensor recordings collected from 50 natural substances along with mixtures among them. The dataset contains approximately 828,000 time-series data points and is paired with preexisting GC-MS data. The authors also introduce SCENTFORMER, a Transformer-based model designed to process these temporal signals for substance classification and mixture ratio prediction. The authors claim that SMELLNET can serve as a benchmark for research in olfactory AI and AI-based smell recognition

### Strengths
- The paper is well-written and clearly structured, making it easy to follow.
It explores an understudied and potentially impactful area, bridging olfactory perception and machine learning.
- The introduction of a new dataset and benchmark is a valuable contribution, especially in a domain with limited data resources.

### Weaknesses
- The dataset includes no human perceptual data (e.g., semantic descriptors, intensity or pleasantness ratings, similarity judgments, or brain recording). Without human responses, the dataset cannot be meaningfully linked to olfactory AI studies as claimed in the paper. Labeling the dataset as “SMELLNET” and positioning it as central to “olfactory perception” is overstated. At best, the dataset captures chemical sensing information, not perceptual smell data.
- In Table 1, the comparison with human-evaluated datasets (e.g., Dravnieks, DREAM, Snitz, Ravia) is misleading. Other datasets include both stimulus (odorants) and response (human judgments). SMELLNET only includes the stimulus component (gas sensor signals), not perception-based responses. Comparing against other gas-sensor or e-nose datasets that exist in related areas would probably be more relevant and better contextualize the scale and novelty of SMELLNET.
- Although the paper describes the dataset as “large-scale,” 50 odorants with six repetitions each is modest by modern machine learning standards. Summing the total number of time-series data points to claim large-scale is not meaningful. A more relevant metric would include the number of unique odorants and repetitions. 
- The definition of “mixture” is ambiguous and the motivation behind that is not clear. In olfactory research, mixtures typically refer to combinations of mono-molecular odorants, while here it seems to involve mixing natural extracts (e.g., “apple + banana”), which inherently are mixtures themselves. Predicting this kind of mixture composition from sensor data, without any perceptual correspondence, does not provide meaningful insight into olfactory AI.
- The dataset provides low-resolution sensor data, yet it later depends on pairing with high-resolution GC–MS data. It is unclear why the low-resolution data is necessary or what advantages it offers. No ablation is provided to show how a model trained solely on high-resolution data would perform compared to one using sensor data.

### Questions
- What concrete research questions in olfactory AI can this dataset help answer? 
- How is the mixture dataset can be used in the further research
-How are “mixtures” defined chemically?
- Could the authors provide comparisons between SMELLNET and other gas-sensor or GC–MS datasets?
- How do you envision connecting SMELLNET to actual human olfactory perception in future work?

### Soundness
2

### Presentation
3

### Contribution
2
