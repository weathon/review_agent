# OceanVerse: Evaluable 4D Ocean Element Reconstruction Dataset under Realistic Sparsity

- Decision: Reject
- Scores: 4, 6, 4, 4, 8

## Abstract
The vast oceans record the impacts of climate change and human activities on the Earth system. Over the past century, oceanographic scientists have collected extensive ocean profile data to reflect variations of oceanic elements, such as dissolved oxygen. However, due to the sophisticated measurements and high costs, historical ocean element observation data remains highly sparse and uneven across the global ocean, with the annual missing rate exceeding 90\%. Thus, quantitatively understanding the four-dimensional (4D) spatiotemporal evolution of oceanic elements continues to pose a significant challenge.  Machine learning (ML) techniques demonstrate superior capabilities in perceiving spatiotemporal variations within large-scale data, presenting promising opportunities to harness implicit correlations for global reconstruction. However, fragmented data and interdisciplinary differences create barriers to the availability of AI-ready open data, further hindering ML practitioners from designing specialized models. To solve this problem, we present the first oceanic 4D sparse observation reconstruction dataset, named OceanVerse. By integrating nearly 2 million real-world profiles since 1900 and three differentiated Earth system numerical simulation, we construct a comprehensively evaluable dataset with missing patterns that align with real-world conditions through a digital twin sampling. OceanVerse provides a novel large-scale ($\sim100\times$ nodes vs. existing datasets) dataset that meets the MNAR (Missing Not at Random) condition, supporting more effective model comparison, generalization evaluation and potential advancement of scientific reconstruction architectures. The OceanVerse dataset and codebase are publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces OCEANVERSE, an evaluable 4D ocean reconstruction benchmark for studying learning-based data assimilation under realistic observational sparsity. Using three numerical Earth simulations (CESM2-OMIP1/2, GFDL-ESM4) as background truth, the authors construct a spatiotemporal dataset where real-world observation distributions are used to generate MNAR sampling masks. They define a unified reconstruction task, establish an evaluation framework, and benchmark several models including classical regressors, sequence models, and a spatio-temporal GNN.

### Strengths
(1) Important questions to AI for Oceanography. The paper addresses an under-explored yet crucial area — AI-driven ocean field reconstruction and data assimilation. This is an important and long-standing challenge in marine science, as ocean observations are extremely sparse and irregular in both space and time. By constructing a complex and large-scale benchmark under realistic observational sparsity, the work provides a much-needed foundation for developing and comparing machine learning models in this domain. This contribution could meaningfully advance the emerging field of AI for Oceanography, fostering the development of more effective assimilation and reconstruction systems.

(2) Realistic sampling design. The proposed Digital Twin Sampling method is well-motivated and technically elegant. It leverages real in-situ observation distributions (from ∼2M historical profiles) to generate non-random, observation-like sampling masks (MNAR), while maintaining full evaluation capability through simulated background fields. This design realistically mimics the spatiotemporal coverage of ocean observations and enables quantitative benchmarking — a key advantage over prior synthetic or randomly masked datasets.

### Weaknesses
(1) Although the dataset is large and technically well-structured, its overall design remains much simpler than real-world data assimilation systems. The spatial and temporal resolutions are relatively coarse, and only a single variable (dissolved oxygen) is included. As a result, the benchmark captures statistical reconstruction behavior but does not reflect the full complexity of realistic ocean assimilation processes.

(2) The assessment is performed entirely within digital-twin simulations, comparing reconstructions against GCM-generated fields rather than real in-situ or reanalysis observations. This makes the evaluation quantitatively consistent but scientifically detached — models may align closely with the numerical model’s climatology while providing limited evidence of real-world assimilation skill or forecasting utility.

### Questions
(1) Could the authors elaborate on how the OCEANVERSE setup relates to real ocean data assimilation systems? In particular, to what extent can the current benchmark (based on digital-twin simulations) reflect or predict model performance in real ocean reconstruction or assimilation tasks? What assumptions or simplifications might limit this correspondence?

(2) What was the rationale for selecting dissolved oxygen as the sole reconstruction target? Does it have specific scientific significance? Would the framework generalize well to other physical variables such as temperature or salinity?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces OCEANVERSE, a large-scale, AI-ready dataset for reconstructing sparse four-dimensional (space × depth × time) ocean observations. The dataset integrates nearly two million historical ocean profiles (since 1900) and three global numerical simulations (CESM2-omip1, CESM2-omip2, GFDL-ESM4). Using a digital-twin sampling strategy, the authors replicate real-world missing patterns by projecting actual historical sampling locations onto simulated “virtual Earths.” This enables complete evaluation while maintaining realistic sparsity. OCEANVERSE supports benchmarking of diverse ML models for reconstructing dissolved-oxygen fields and related environmental variables at 1° × 1° spatial and annual temporal resolution. Baseline experiments compare classical and recent models (XGBoost, MLP, LSTM, Transformer, GRIN, ImputeFormer, TIDER, OxyGenerator). Results show that OxyGenerator and MLP achieve the best performance, with graph-based architectures handling sparse data most effectively. The dataset also enables systematic tests on different data splits and cross-simulation generalization across virtual Earths

### Strengths
1. The work provides a major infrastructure contribution to AI for Earth science by releasing a unified, large-scale, multi-source dataset that directly addresses the scarcity of data.
2. The digital-twin evaluation design is well-justified, allowing reconstruction benchmarking with ground-truth access while preserving realistic sampling sparsity. 
3. The dataset’s scale and realism far exceed prior spatiotemporal imputation datasets.
4. Baseline experiments are well-structured and provide informative results illustrating how sparsity challenges model complexity.

### Weaknesses
1. Although the digital-twin framework is elegant, the paper stops short of quantitative validation that the simulated sampling truly reproduces real-world MNAR patterns. Statistical comparisons of spatial/temporal sampling distributions would strengthen the realism claim.
2. The resolution (1° × 1°, annual) limits small-scale process studies. The paper acknowledges this but does not demonstrate how the framework could be extended to finer scales or variables.
3. Evaluation remains single-task and single-variable (dissolved oxygen); expanding to multi-variable reconstruction (e.g., nutrients or temperature) would better illustrate generality.
4. While computational challenges (e.g., GRIN OOM) are discussed, runtime and scalability comparisons are not reported.
5. There is little examination of why specific models succeed or fail beyond sparsity effects.

### Questions
1. How closely does the digital-twin sampling reproduce real-world spatial and temporal observation densities? Have you validated its fidelity quantitatively?
2. How can OCEANVERSE be extended to higher-resolution or multi-variable reconstruction?
3. Are the performance differences among models statistically significant across runs?

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
4

### Summary
Basically, the problem is this: imagine you are using a dataset of real ocean observations. If we want to train a neural network on this data, the only way we can validate it is by either partially masking the complete observations or making it predict other observations. Thus, by training only on observations, it is certain that they do not cover the whole ocean, and therefore some dynamics are missed and never learned by the neural network in regions with no data. So, how can you have any confidence in the completely reconstructed states from pure observations in regions where we had no observations? To overcome this issue, they propose: imagine we have access to numerical simulations of the ocean. We are going to subsample them at the locations of the dataset of observations we have. In other words, we create this virtual dataset of observations where all observations are from the numerical simulation but at the very same locations. By doing so, it means that: 1. The state mask is not chosen at random, meaning that the mask allowing us to extract observations has real meaning (MNAR); 2. We have access to a ground truth in regions with no observations to check whether or not the dynamics learned by the neural network work also in these regions. Once we are happy with the trained model, we can apply it to our true observations to reconstruct our states.

### Strengths
First of all, I would like to say that I am currently working on building ocean emulators, and very recently I had this debate about how to properly validate our models if we train purely on observations. This paper comes at the right moment! Let me start with the positive aspects:

1. I recognize that the effort required to assemble all the observation databases must have been tremendous. Thank you for that contribution.

2. From a scientific standpoint, I agree that this dataset will surely be very useful for pushing forward the development of new AI emulators based solely on observations. We are moving in this direction for atmospheric emulators, but nothing comparable exists yet on the ocean side, so this will certainly help.

3. I verified that both the codebase and dataset are indeed publicly available. I did not test downloading anything yet, but I saw that everything is already online, so thumbs up.

In conclusion, I really appreciate the effort, and I believe the dataset will be of great use for the scientific community aiming to develop new AI tools for emulating the oceans.

### Weaknesses
Nonetheless, I have several concerns with the paper:

1. First of all, let's discuss the scope of the paper. In my view, the paper should focus on presenting your dataset and how you constructed it, etc. As your title indicates, you are presenting a dataset. Therefore, I do not see the point of everything that comes after page 6, starting from the experiments section. I am by no means criticizing your experiment section; however, I think that either the paper should focus exclusively on the dataset or on the creation of the oxygen maps. This is my main criticism.

2. Even considering what I just said, one aspect that I found particularly disappointing in the experiment section is that, despite all the claims made previously about being able to perform robust validation of neural networks trained on virtual Earth and then apply them to true observations, there were no results showing oxygen maps generated from a model trained on true observations...

3. Finally, I am not a fan of the overall writing style of the paper. This is a research article, and for example, the opening sentence is not really appropriate. In addition, I found all the explanations from page 2 to page 6 quite repetitive, and I am pretty sure they could be shortened to half their current length.

### Questions
1. **Clarify the paper's focus and scope**: While the dataset construction effort is great and well-executed, the paper suffers from a lack of clear focus. I suggest two possible directions:
   - **Option A**: Reframe the paper exclusively as a dataset contribution, removing or significantly condensing the experiments section. The dataset itself is a valuable contribution that could stand on its own. This would allow for more space to detail the dataset characteristics, validation procedures, and potential use cases.
   - **Option B**: If you wish to present both the dataset and the oxygen reconstruction methodology, the paper needs to be restructured with a clearer narrative that justifies why both contributions belong together !

2. **Improve the writing style and maintain scientific rigor**: This is a technical research paper for ICLR, and the writing should reflect this throughout:
   - The opening sentence of the introduction ("Since hunters from Siberia crossed the Bering Strait to reach Alaska 25,000 years ago...") is too narrative and not appropriate for a machine learning venue. Consider starting directly with the scientific problem.
   - Avoid repetitive statements about how "OceanVerse will create new opportunities" or similar claims. This phrase appears multiple times throughout the paper (particularly in sections 1-3) without adding substantive information. State this claim once clearly, then focus on technical details.
   - Sections 2-6 contain significant redundancy and could be condensed to approximately half their current length without losing essential information. Focus on technical precision over narrative repetition.

3. **Include experiments on real observations**: Given your extensive claims about robust validation through the digital twin approach and the ability to apply models trained on virtual Earth to real observations, it is surprising and disappointing that no results are shown for models trained on real observations. I strongly recommend adding at least one experiment demonstrating:
   - A model trained on your real observation dataset
   - Its performance compared to models trained on the virtual Earth simulations
   - Discussion of the domain gap between simulated and real data
   This would significantly strengthen the paper's practical impact and validate your methodological claims.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This document introduces OCEANVERSE, the first evaluable 4D ocean element reconstruction dataset, aiming to address the problems of highly sparse historical ocean observation data and the lack of AI-ready data. The dataset integrates nearly 2 million real ocean profile data points with three CMIP6 numerical simulation results, constructing a dataset that conforms to real missing patterns (MNAR) through digital twin sampling. It supports spatiotemporal reconstruction of global marine biogeochemical elements (with dissolved oxygen as a case study). Experiments validate the performance of 8 baseline models, proposing temporal split as the optimal data partitioning method. The dataset provides a standardized benchmark for AI-driven marine science research, with both the dataset and code open-source.

### Strengths
1.	It’s the first 4D dataset that realistically handles missing ocean data, which previous AI-ready datasets failed to do.
2.	The article itself is well-organized, easy to follow, and clearly explains the concepts.
3.	OCEANVERSE focuses on pressing scientific issues such as ocean deoxygenation, which is closely linked to climate change and marine ecosystem health. This represents a new, high-potential direction for research in this field.

### Weaknesses
1.	Resolution is too low: The data (1°x1° spatial, annual) is too coarse. It's not detailed enough to study smaller or faster ocean processes, like coastal events or seasonal changes.
2.	Unquantified Discrepancy Between Simulation and Real Data: While the document mentions "calibrating virtual Earths with real observations," it fails to quantify the calibration accuracy (e.g., regional or depth-specific deviations between simulated and observed values). Additionally, the potential impact of inherent systematic errors in CMIP6 models (e.g., GFDL-ESM4’s biases in simulating high-latitude ocean processes) on reconstruction results is not analyzed, which risks overestimating the transferability of models trained on OCEANVERSE to real-world ocean scenarios.
3.	Only focuses on oxygen: Despite incorporating multiple environmental variables as inputs, OCEANVERSE centers exclusively on dissolved oxygen as the core target variable, lacking coverage of other ecologically and biogeochemically critical marine elements (e.g., nutrients like phosphate, contaminants like mercury). This narrow focus restricts the dataset’s utility for broader marine science research (e.g., studying nutrient-oxygen coupling or ocean acidification).

### Questions
1.	How to quantify the matching accuracy between real observations and virtual Earth data in digital twin sampling? Are there regional adaptability differences?
2.	What is the basis for selecting these other marine environmental variables as inputs?
3.	Is there a clear plan for extending the dataset to other marine elements (e.g., phosphate, mercury or even the mentioned environmental variables)? How to maintain data consistency and unified evaluation standards during expansion?
4.	Are there targeted data augmentation or model optimization suggestions for complex coastal areas and data-sparse regions in the Southern Hemisphere?
5.	What is the spatiotemporal resolution enhancement plan for the dataset? How to balance data sparsity and computational feasibility after enhancement?
6.	What is the impact mechanism of differences between different numerical simulation results on model performance in the generalization evaluation?
7.	The experiments use RMSE and R² as core metrics. Does the dataset's evaluation system lack indicators directly related to ecological effects (e.g., prediction accuracy of deoxygenated area, matching degree with coral reef survival thresholds)? Relying solely on statistical error metrics may disconnect model evaluation from real marine scientific issues, making it impossible to measure the model's supporting value for ecological conservation decisions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The proposed work introduces a new dataset called OCEANVERSE, which is created using digital twins to simulate real-world ocean observations and handle missing data on a virtual Earth. This dataset includes ground-truth data that ensures robust performance validation. The study uses dissolved oxygen as a case study to demonstrate how the datasets are constructed and also presents baseline models. This dataset is valuable for the scientific community and will undoubtedly advance research on ocean observations (specifically dissolved oxygen) utilizing machine learning.

### Strengths
1) **Well-written paper:** The paper is clearly articulated and easy to follow. The graphical illustrations significantly enhance the understanding of the work.

2) **Valuable dataset:** The dataset holds great potential for the scientific community.

3) **Thorough experimentation:** The experimentation section is well-presented, including appropriate baselines and thorough discussions on the results obtained.

### Weaknesses
1) **Need for more details:** Additional information is needed regarding the construction of the dataset. What sparse data representation techniques are employed in this work? 

2) **Clarification on regression models:** In Figure 3, which regression model is utilized? Besides the model mentioned in the figure, are any other environmental variables used to determine dissolved oxygen? The figure legend (mentioning density, pressure, etc.) suggests that additional variables may be considered.

### Questions
Please refer to my comments. I would like to see a few more information about the models and the construction of the dataset

### Soundness
4

### Presentation
4

### Contribution
4
