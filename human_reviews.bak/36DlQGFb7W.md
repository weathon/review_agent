# Data-Driven Uncertainty-Aware Forecasting of Sea Ice Conditions in the Gulf of Ob Based on Satellite Radar Imagery

- Decision: Reject
- Scores: 5, 3, 6, 3

## Abstract
The increase in Arctic marine activity due to rapid warming and significant sea ice loss necessitates highly reliable, short-term sea ice forecasts to ensure maritime safety and operational efficiency. In this work, we present a novel data-driven approach for sea ice condition forecasting in the Gulf of Ob, leveraging sequences of radar images from Sentinel-1, weather observations, and GLORYS forecasts. Our approach integrates advanced video prediction models, originally developed for vision tasks, with domain-specific data preprocessing and augmentation techniques tailored to the unique challenges of Arctic sea ice dynamics. Central to our methodology is the use of uncertainty quantification to assess the reliability of predictions, ensuring robust decision-making in safety-critical applications. Furthermore, we propose a uncertainty-aware model switching mechanism that enhances forecast accuracy and model robustness, crucial for safe operations in volatile Arctic environments. Our results demonstrate substantial improvements over baseline approaches, underscoring the importance of uncertainty quantification and specialized data handling for effective and reliable sea ice forecasting.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents a data-driven approach for forecasting sea ice conditions in the Gulf of Ob by leveraging advanced video prediction models originally developed for computer vision tasks. The authors utilize sequences of radar images from Sentinel-1, weather observations, and GLORYS forecasts to predict future sea ice conditions. They address challenges related to data irregularity and missing values through domain-specific preprocessing and augmentation techniques. The paper also introduces an ensemble-based approach for uncertainty quantification and proposes a confidence-based model selection scheme to enhance forecast accuracy and robustness.

While the paper tackles a relevant and practical problem, it primarily applies existing deep learning models to a new domain without significant methodological innovations. The contributions are more engineering-focused, adapting existing models for sea ice forecasting without introducing new algorithms or theoretical advancements. The improvements over baseline models are modest, and there is limited discussion on the practical significance of these improvements or how they translate to real-world applications.

### Strengths
Application of Deep Learning to Sea Ice Forecasting: The paper addresses a relevant and practical problem by applying advanced video prediction models to sea ice forecasting in the Gulf of Ob. This cross-disciplinary application showcases the potential of deep learning in geophysical tasks.

Data Preprocessing Techniques: The authors develop domain-specific data preprocessing and augmentation methods to handle the challenges of Arctic satellite imagery, such as data irregularity and missing values. This is crucial for improving model performance on imperfect real-world data.

Uncertainty Quantification: Introducing an ensemble-based approach for uncertainty estimation and a confidence-based model selection scheme adds value by enhancing forecast robustness and providing a mechanism to assess prediction reliability.

### Weaknesses
* Lack of Novel Methodological Contributions: The paper primarily applies existing video prediction models to a new dataset without significant modifications or novel methodological developments. This limits its contribution to the advancement of machine learning techniques.

* Engineering Focus Over Research Innovation: The work focuses more on engineering implementation and practical adaptation rather than introducing new theoretical insights or advancements in machine learning.

* Modest Improvements Over Baselines: The improvements over baseline models are modest. The paper lacks a deep analysis of the practical significance of these improvements, especially in operational contexts.

* Insufficient Theoretical Analysis: There is a lack of in-depth theoretical analysis or exploration of why certain models perform better in this context, which could provide valuable insights to the research community.

### Questions
* Novelty of Contributions: Can the authors clarify what novel methodological contributions are presented beyond applying existing models to a new dataset? Are there any new algorithms, architectures, or theoretical insights introduced?

* Model Adaptations: Did the authors make any significant adaptations or improvements to the video prediction models to better suit sea ice forecasting, or were the models used off-the-shelf?

* Evaluation of Practical Significance: How do the modest improvements over baselines translate to practical benefits in operational forecasting? Are these improvements significant enough to impact real-world applications?

* Generalizability: Can the authors discuss the potential generalizability of their approach to other regions or types of geophysical forecasting? What are the limitations?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
Short-term forecasts of satellite images at 1km resolution conditioned on past satellite imagery and past meteorological conditions with deep neural network architectures commonly used in video prediction. The networks beat a simple baseline (persistence), but most video prediction methods do not improve over a UNet. Moreover, the presented models struggle due to the inherent sparsity of the satellite time series, yet a new augmentation method (joint geometric transformations of meteo fields and satellite images) is introduced which improves sample efficiency in this data sparse setting.

### Strengths
1) The task of satellite-based sea ice forecasting conditioned on meteorology is interesting and sufficiently novel
2) The paper introduces an augmentation strategy which improves performance significantly
3) The work compares many different neural network architectures and includes two simple baselines
4) A domain-specific evaluation metric is used, the Integrated Ice Edge Error.

### Weaknesses
1) The results are not convincing. This work trains very large deep neural networks (some over 30Mio parameters) on a very small dataset (training has only ~2200 days). The trained models beat a simple persistence baseline by some margin, but it is unclear what this means, as there is no comparison to any baseline sea ice forecast and there is almost no qualitative evidence presented in this paper. The only qualitative results are shown in Fig. 6, but those are not convincing, there, all models fail to provide a forecast that is somewhat close to the reality at day 3. My impression after reading is that the gaps in the time series and the low availability of past data make the task extremely challenging, such that the models mainly learn a blurred regression to the mean, which in MSE beats persistence.
2) The writing lacks clarity. Many important technical details are not explained well (or not at all), instead the paper is full of fill-words and meaningless phrases that sound like output from a LLM. I'll provide more specific feedback below.
3) It is hard to assess what the contribution of this work is. I see the main novelty in the augmentation strategy, but that is a bit too little for an ICLR paper.
4) The paper emphasizes that fancy video prediction architectures do not outperform an out-of-the-box UNet for satellite image time series forecasting, but instead domain-specific preprocessing is more important. However, this finding is not new, see e.g. Benson et al 2024 https://openaccess.thecvf.com/content/CVPR2024/html/Benson_Multi-modal_Learning_for_Geospatial_Vegetation_Forecasting_CVPR_2024_paper.html - which focusses on vegetation greenness forecasting, but else is very similar in design.
5) Missed opportunity: the work only uses Sentinel 1 at 1km resolution, however the big benefit of the satellite is its high spatial resolution (up to ~15m). At coarser resolution, i doubt Sentinel 1 is the best product, especially due to its temporal sparsity (only ~5-daily). Moreover, the work only uses past meteorological data. Yet, future sea ice motion is likely depending a lot on future weather, hence it would make a lot of sense to include future weather. Ideally, to emulate operational conditions, this would be from stored weather forecasts, but for showing the predictive skill of the map weather -> sea ice, it would also suffice to just use future reanalyis, mentioning a potential further performance degradation at inference time due to the usage of actual forecasts.
6) The evaluation of ensembles is a bit weak. If you provide ensemble forecasts for uncertainty quantification, as a user, i'd most importantly like to see how well they are calibrated, i.e. the skill score. There are further probabilistic metrics like CRPS that should also be looked at. And not just MSE of the ensemble mean.
7) Many formulations in the paper are debatable:  l. 013ff I'd argue the causality is wrong in this sentence. Short-term sea ice forecast are important because they are useful for boats navigating through the arctic sea, not because of global warming and subsequent sea ice loss. ; l. 100ff by comparing the accuracy of model predictions we do not ensure that these predictions contain more than just general trends (what are those anyway?) and we also do not ensure that they contain spatial structures. ; l. 148ff The main reason for the data gaps is that Sentinel 1 is on an orbit that only has a revisit time of 12 days. For some time (until 2021), there were two satellites, which, together with considering off-nadir imagery allowed for an effective revisit time of 2-3 days, now it is 5-6 days. All other factors are minor compared to this one. ; l. 214 I am unaware of any weather capability of Sentinel 1 (what is that anyway?) - however, it may be worth to mention that contrary to passive optical imagery like Sentinel 2, the active sensor of Sentinel 1 can measure surface conditions even if there is cloud cover. ; L. 235 Sentinel 1 has up to ~15m resolution. L. 236 It is only partially true that there are large amounts of historical data: while the size in terms of storage is surely in the petabytes, we have only a very limited (!) historical record of Sentinel 1, just 9 years since 2015. 
8) Limited related works section. Only googling once gave me already a very related paper Palerme et al 2024 https://tc.copernicus.org/articles/18/2161/2024/ doing short-term sea ice forecasting with deep learning. The related works section needs to include such works, and ideally you compare the performance of your models to those published in these related works. Furthermore, there is a large stream of literature on satellite image time series forecasting, which seems extremely relevant, but the related works section also misses.

### Questions
1) How do you split the different samples in the training period? Do you include forecasts starting at every day? Or only every 10 days to avoid data leakage (one samples target being in another samples input)? --> Following from this, what is your exact sample size for train, val, test: before and after augmentation?
2) How do you do the backward forecast (l. 462)? Are you considering that atmospheric dynamics are not time-reversible due to the second law of thermodynamics?
3) Are you using the same augmentation strategy for all models?
4) Which Sentinel 1 product are you using? How has it been processed? Is it radiometrically terrain corrected?
5) How are you computing the IIEE?
6) Could you explain the filtration in other words again? L.292ff - I did not understand from reading the manuscript.
7) Why the loss MSE - 0.2 SSIM?
8) How do you feed missing inputs to your models?
9) Do you have any idea why the missing values (Fig 2a) were a lot lower during 2016 & 2017? To me it makes little sense and I would rather expect a drop in 2021, when the Sentinel 1B satellite went out of functioning.
10) Have you compared to a climatology? For satellite imagery this seems a very important baseline, see again e.g. Benson et al 2024 https://openaccess.thecvf.com/content/CVPR2024/html/Benson_Multi-modal_Learning_for_Geospatial_Vegetation_Forecasting_CVPR_2024_paper.html
11) I do not understand how the confidence-based mixture with DMVFN (l. 433f) plays a role in the predictions of the models presented in Table 3, can you elaborate?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a sea ice forecasting approach that uses video prediction approaches applied to synthetic aperture radar (SAR) satellite imagery captured by Sentinel 1. The work examined the performance of a number of architectures for the video prediction task and uses an ensemble of four architectures to achieve uncertainty quantification.

### Strengths
The paper provides an approach to sea ice forecasting which is an important problem and explores the performance of several video prediction algorithms on this task. The authors also consider the problem of image artifacts and propose a projection based approach to eliminate image artifacts.

### Weaknesses
1. It is not clear what sea-ice parameters are considered in this work and how these parameters would be obtained from the SAR video streams. The authors should clearly state the parameters considered and describe how they are derived from SAR imagery.

2. The description of the architectures in Table 2 is not clear. What are the inputs and outputs in each configuration? Also, since the best performing system appears to be the rUNET system which is a SISO configuration, are the multiple inputs necessary or sources of potential error?

3. The IIEE metric is not explained in the paper. I believe it is the “integrated ice-edge error” which may be unfamiliar to other readers and should be introduced.

4. The data preprocessing step which involves learning a projection should be evaluated to validate the removal of artifacts? What is the computational complexity of this approach?



Minor Comments:
1. Typo - Line 145 “uncetrainty quantification”
2. A map of the area as would be useful in the main paper.

### Questions
1. Is the SAR video prediction and end in itself in this work?
2. Can  the performance of the  data preprocessing step be demonstrated quantitatively and also qualitatively by some example images?
3. Is there a way to incorporate ice-dynamics in this video prediction approach?
4. How sensitive would any approach be to location?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes different methods to forecast the sea ice extent in the gulf of Ob based a mix of Sentinel-1 data (radar), re-analysis data and interpolated weather stations.
The paper compares a slue of different methods, and aims at quantifying the uncertainty of the forecast with these methods. The different methods each produce a forecast, which is then used as an ensemble to quantify the uncertainty.

### Strengths
The beginning of the paper is well written, with a good problem statement and motivation. The importance of the work is well explained, and appears timely.

### Weaknesses
The paper compares 8 different methods to forecast the sea ice, but fails to introduce them. The author spend more time on the data preprocessing and filtration of S1 data, than on explaining what the actual models do. The only mention of the models are on line 79 to 94, but are very brief.

Overall, the paper lacks a significant analysis of the results. The results are shown briefly in table 3 and figure 3, but lack a deeper analysis. In the main text, there is no example of time series, nor map to show the uncertainty per pixel, nor interpolation output, or visuals to show the results, and help the reader in understanding the process.

The paper would profit massively from a schematic representation of the tasks.

I feel like the paper has potential, but the different sections have been given inappropriate weight. The paper would need a major restructuration, and overall would probably fit better in a longer format, such as a journal, where the details can be explained better, and the analysis performed at a deeper level. There are just too many moving parts to fit in this short format.

## Target
It is unclear to me how the target is produced. The authors mention "a target presentation of the forecasts" (line 213), but don't explain how they use Sentinel-1 to produce the target.

## Minor comments

Table 1: if the scale is supposed to be the scale of the product, then S1 has a scale of 10 meters, not 1km. The rescaled input is 1km, but so is GLORYS and the meteo stations
I would add the temporal resolution to this table to add a bit more information.

Line 206: "Sentinel-1 SAR images and GLORYS fields are interpolated bilinearly to match the input resolution (1 km)" using bilinear interpolation to resample Sentinel-1 from 10 meters to 1km is quite unconventional, usually downsampling is done with nearest neighbor or average.

Line 235: "up to 50 meters": as far as I know S1 resolution is 10 meters

hLine 257-262: this comment seems out of place.

Figure 3 is hard to read, is missing units, and pixelated.

Line 304: "nor noise" how do you make sure an image has no noise?

## Grammar comments

Line 079: "Our research employs advanced video prediction models, which include:" please rephrase, doesn't work with the bullet points

Line 240: "which lacks quality of ice data in the Gulf being mostly uncorrelated with other sources": unclear, rephrase

### Questions
c.f. weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
