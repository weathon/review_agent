# Indoor 3.6M : A Dataset and Benchmark for Indoor Image Geolocation

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Image geolocation has advanced rapidly for outdoor imagery, driven by large-scale benchmarks and strong visual cues such as landmarks, skylines, and vegetation. In contrast, indoor image geolocation remains underexplored: indoor scenes lack distinctive geographic features, are highly ambiguous, and are not adequately represented in existing datasets. We address this gap by introducing the first large-scale benchmark for indoor geolocation, consisting of \textbf{3.6 million} images across \textbf{213 countries}. We finetune state-of-the-art CLIP-based models such as Pigeon and GeoCLIP and report performance at country and continent levels using both top-$k$ accuracy as well as distance based accuracy metrics. Results highlight that continent-level geolocation is feasible, but fine grained indoor geolocation e.g street and city level geolocation remains an open challenge. This work defines a new frontier for geolocation research and provides the resources to advance it.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a dataset which combines indoor images from 3 different sources, which are Flickr, Wikidata, and Booking.com. The dataset is worldwide and it is shown to lead to better results for indoor geolocation when training a model for this task, compared to previous worldwide geolocation datasets.

### Strengths
The dataset is a novel contribution and is useful for worldwide geolocation tasks
The paper is presented well and the experiments support the claims
The reference to Europol's Stop Child Abuse link certainly helps to motivate the impact this dataset could have

### Weaknesses
It is not clear if and how the data is released. This needs to be clearly explained otherwise the paper can't be properly evaluated.

Line 275 says INDOOR-40k, and table 1 says that dataset has 40k images, but line 276 says there are 15.000 images. Is there a typo?

Figure 4 seems incorrect. It says the images with blue text are from Booking.com, but there are photos of malls, car interiors and swimming pools. Are they really from booking.com? Usually booking is a platform for hotels and short-term apartments rentals.

### Questions
Questions mostly listed above.

As a suggestion, the San Francisco Landmarks dataset has nothing to do in table 1 and can be removed, because all datasets are global except for that one that is city-scale. If city-scale datasets are included, then many more should be added to the table (like SF-XL and Pitts250k)

Minor comments: there is a missing space at lines 212 (INDOOR-3.6Mdataset), line 238 (INDOOR-3.6Mdataset), and line 043 (modificationsPramanick)

Minor comments: in figure 2 the text of the labels is almost unreadable. Please increase its size. Same for figure 3.

### Soundness
1

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces INDOOR-3.6M, a large-scale dataset of 3.6 million geotagged indoor images spanning 223 countries, along with a sampling framework designed to mitigate geographic bias.

The authors compute visual diversity scores using CLIP ViT-L/14 embeddings, then integrate country-level population and land area into a regression model (Random Forest) to estimate sampling weights. They use these weights to construct geographically balanced subsets,
including the INDOOR-40K benchmark, and evaluate models such as GeoCLIP and CLIP in both fine-tuned and zero-shot settings.

Empirical results show that fine-tuning GeoCLIP on the new dataset reduces mean distance error from 4089 km to 3598 km and improves accuracy across multiple geographic granularities.

### Strengths
Novel dataset contribution and thoughtful sampling strategy.

### Weaknesses
1. Domain Mismatch in Evaluation: The paper's evaluation framework suffers from a fundamental domain mismatch. The training dataset, INDOOR-3.6M, is described as "scene-agnostic" and collected from diverse sources (Flickr, Wikidata, Booking.com) . However, the benchmark test set, INDOOR-40K, was "exclusively sourced from booking.com". This means the model is trained for a general,
diverse task but tested on a very narrow, specific domain (hotel rooms, apartment rentals, and lobbies). This methodological flaw is a major weakness.

2. There is no mention of any privacy preserving pipeline. The authors should’ve used the YOLO/SAM model for this section (at least face blurring). For e.g. : NYC- Indoor dataset has segmentation-masks over all the people. Dataset is obtained from public platforms, adds conspiracy to the privacy of individuals.

3. The paper only accounts for GeoCLIP and PIGEON, Evaluations for more baseline models should be provided, for e.g. Img2Loc, etc.

4. The paper highlights its rich metadata as a key contribution but fails to use it in any baseline experiments/ablations. The value of this metadata for the task of geolocation is, therefore, entirely speculative.

5. There should be ablations over subsets of the dataset for different values of P, and not just P>0.5.

6. Reproducibility gaps: The authors should have provided the crucial implementation details such as, CLIP embedding parameters, Random Forest hyperparameters, and dataset splits, at least in the appendix to ensure reproducibility.

7. Figure 5 is Not Well Described: Figure 5 is mentioned in the paper, but it is not explained in enough detail. It would be good to provide a better description of what this figure shows and its relevance to the paper.

### Questions
Overlap analysis: Was any overlap analysis performed between INDOOR-3.6M and existing public pretraining sets (e.g., YFCC100M)?

What percentage of the final 3.6M images fall into highly ambiguous bins (e.g., 0.5 P < 0.7)? What ablation studies were performed to prove that this ambiguous data is beneficial rather than "pollution/noise" that harms performance?

Is the metadata (SAM, YOLO, scene labels) actually useful for improving indoor geolocation accuracy?

### Soundness
3

### Presentation
1

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
This paper proposes a new dataset for developing and evaluating indoor scene geolocalization (i.e., guess the LAT/LON from an image of an indoor scene somewhere in the world). Dataset utility is demonstrated by finetuning a GeoCLIP model on their dataset and comparing its performance to the original GeoCLIP model (which was presumably trained on outdoor scenes). Slight improvement is observed (error of 4000 km --> 3500 km). In addition to images, some predicted metadata is also included with each image.

### Strengths
This is an interesting and important problem that is clearly lacking a large dataset. Overall, the writing is very good, but I found the manuscript difficult to follow. Some sections would benefit from restructuring and additional details.

### Weaknesses
The manuscript lacks clarity on several important issues. The Introduction is overly verbose, while the Methods and Results lack important details. 

- Presumably, the values reported in Table 2 give the percentage of images that were correctly classified at several different distant (error) thresholds, but this is not clearly stated in the text (unless I missed it).

- Only a single SOTA model is fine-tuned on the proposed dataset. This model fails to achieve a useful accuracy (+/- 3500km error!), which does not constitute a solid baseline for this problem. Either the problem is too hard or the dataset is not sufficiently informative. 

- The dataset is described as "scene-agnostic", but the description of how images were extracted (page4/5) list the specific scene types that were used to build the dataset. A truly 'agnostic' dataset would collect images without selecting from a discrete set of scenes.

- Lines 231-237 describe how the "indoorness" of each scene was calculated. Some representative images from across the spectrum of "indoorness" would be helpful here.

- Line 243: First mention of Figure 5 is many pages from the figure, is out of order (comes before Fig 4), and refers to a figure in the appendix (without mentioning this). Shouldn't this figure be labeled as A.1 or something? Otherwise, if figures from the main manuscript appear beyond Page 9, then this manuscript contravenes the page limits for ICLR.

- The "metadata enrichment" section describes how models, such as Places365 and "a ViT trained on MIT indoor Scenes dataset" and SAM are used to add information to the dataset. However, it seems that none of these data are curated/validated, so they arguably add limited value. 

-  Presumably, the values reported in Table 2 give the percentage of images that were correctly classified at several different distant (error) thresholds, but this is not clearly stated in the text (unless I missed it).

- Figure 4 appears to give the metadata for scene-type for each image, but they MANY of them are incorrect upon visual inspection (e.g., area rodeo, aquarium, physics lab, butcher shop, delicatessen) Does this represent the quality of meta data provided in the proposed dataset?

MINOR:
- Line 21: Missing commas 
- Line 95: typo in "scnes"
- 124: Geoclip --> GeoCLIP
- 212: Missing space in "INDOOR-3.6Mdataset"
- Line 238: Missing space in "INDOOR-3.6Mdatase"
- The point that most geolocation dataset are for outdoor scenes has been repeated _many_ times in the paper. By line 264, it is firmly established and does not require repeating.

### Questions
- Why does the dataset need to be 'balanced' and 'representative'? Within a single city, you can have extremely diverse indoor imagery. While you would not want multiple hotel rooms from the same hotel, you would want at least one room from each hotel in each city... 

- Line 103: If the dataset is meant to evaluate "hybrid geolocation" methods then wouldn't it require paired indoor/outdoor images?

- Line 205: "Geolocation data, provided either as GPS coordinates or text-based location labels"... is this "or" or "and"? It seems strange not to formalize the manner in which geolocation data will be encoded in the dataset.

- Line 208: "It is important to note that the dataset does not explicitly identify specific locations in the manner typical of place recognition tasks." This sounds very strange. If the goal of the dataset is to develop geolocation models, then shouldn't the target output (i.e., geolocation) be provided for each record?

- Line 265: "Figure A.1 illustrates the percentage of indoor images identified at various likelihood thresholds across existing mixed-environment image geolocation benchmark datasets". What models were used to generate the likelihoods? Methodology is unclear here. Also, Figure A.1 appears to be "Figure 5"? 

- Line 278: after describing how images were collected from Flikr and several other sources, the paper here states that the proposed INDOOR-40K dataset was "exclusively sourced from booking.com". This is very confusing. What is the difference between the "INDOOR-3.6Mdataset" and the "INDOOR-40K dataset"? Which dataset is being proposed here and how do these two datasets relate?

- Lines 291-293 appear to describe a regression model that uses country size and population to regress "their relative contributions to observed visual diversity". I do not understand what is being regressed here. Do the authors measure the average image diversity across all images sourced from a country and they try to explain this diversity from the country's size and population? What is the goal? 

- The authors use "zero-shot" classification of geocells (geographic regions) directly from the CLIP image embeddings. Was this accomplished using a trainable linear layer (linear probing), K-nearest-neighbor classification, or some other approach?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a **3.6M geolocation dataset specialized with Indoor images**. The authors **fine-tune a Geoclip model** to benchmark this dataset.

### Strengths
1. **Indoor geolocation is a very useful task** for authorities (notably in CSAM investigations) and therefore **gives a lot of value** to such a dataset.

2. The authors took care that the **test set is temporally separated** (not contaminated in time) to avoid memorization.

### Weaknesses
1. This is a dataset paper, but the **methodological part has been widely overlooked**. When introducing a new dataset, one should expect **extensive benchmarking of methods from the literature** (Regression, Classification, hybrid \[2\], generative \[3\], retrieval \[5\], etc.). **Finetuning Geoclip is not enough**.

2. Recently, **LLMs have been showing strong results for geolocation**, and I would like to see **extensive benchmarking of the different LLMs** (at least the open-source ones).

3. The test set is temporally separated, but it **should be spatially separated** (as in \[2\]). If having the same room in train and test is important, there must be a mechanism to **ensure the pictures are sufficiently different**. (It's not because a picture is uploaded later that it's not the same image).

4. Similarly, it would be interesting to **benchmark the different visual encoders**. Are there some that are **more accurate on indoor scenes than outdoor**?
### Missing Citations
**A lot of recent geolocation works are missing**, among others:

* \[1\] GOMAA-Geo: GOal Modality Agnostic Active Geo-localization, Neurips 2024

* \[2\] OpenStreetView-5M, The Many Roads to Global Visual Geolocation, CVPR 2024

* \[3\] Around the World in 80 Timesteps: A Generative Approach to Global Visual Geolocation, CVPR 2025

* \[4\] GaGA: Towards Interactive Global Geolocation Assistant, Arxiv 2024

* \[5\] G3: an effective and adaptive framework for worldwide geolocalization using large multi-modality models.

**\[2\] is the most notable omission**, as the authors could draw inspiration from how a dataset paper for geolocation should be evaluated. Many others are missing. The authors **need to do a much better job** of situating their work within the current literature.

### Questions
I think the paper aesthetics could be improved. In particular, the tables are pretty unaesthetic

### Soundness
3

### Presentation
2

### Contribution
2
