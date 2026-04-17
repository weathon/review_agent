# RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Accurate material identification plays a crucial role in embodied AI systems, enabling a wide range of applications. However, current vision-based solutions are limited by the inherent constraints of optical sensors, while radio-frequency (RF) approaches, which can reveal intrinsic material properties, have received growing attention. Despite this progress, RF-based material identification remains hindered by the lack of large-scale public datasets and the limited benchmarking of learning-based approaches. In this work, we present RF-MatID, the first open-source, large-scale, wide-band, and geometry-diverse RF dataset for fine-grained material identification. RF-MatID includes 16 fine-grained categories grouped into 5 superclasses, spanning a broad frequency range from 4 to 43.5 GHz, and comprises 142k samples in both frequency- and time-domain representations. The dataset systematically incorporates controlled geometry perturbations, including variations in incidence angle and stand-off distance. We further establish a multi-setting, multi-protocol benchmark by evaluating state-of-the-art deep learning models, assessing both in-distribution performance and out-of-distribution robustness under cross-angle and cross-distance shifts. The 5 frequency-allocation protocols enable systematic frequency- and region-level analysis, thereby facilitating real-world deployment. RF-MatID aims to enable reproducible research, accelerate algorithmic advancement, foster cross-domain robustness, and support the development of real-world application in RF-based material identification.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce a new RF-sensing based benchmark called RF-MatID for material identification. This benchmark is large in its scale across the frequency range from 4 to 43.5 GHz and the number of samples in the time/frequency domains. 
The dataset contains 5 superclasses (brick, glass, synthetic materials, woods, and stones) and 16 fine-grained material categories for different variants of the superclasses. Each sample is collected at varying distances (200–2000 mm) and incidence angles (0–10°) to simulate real-world perturbations. The authors benchmark nine models popular in vision, language, and time-series domains across multiple settings, protocols, and divisions. The results have high in-domain accuracy around 99% and analyze robustness to cross-angle and cross-distance shifts. Their results show that raw frequency-domain data performs comparably or better than time-domain conversions, and that the dataset supports evaluation under different global frequency regulations.

### Strengths
- The dataset is clearly described and prior limitations are organized well
- The benchmark provides a good structured taxonomy for fine-grained classification.
- The benchmark covers comprehensive materials with 16 fine-grained attributes and five superclasses
- The authors evaluate the benchmark on some popular deep learning models from the vision domain.

### Weaknesses
- The dataset operates at a sensing range of only ~2 m, which is insufficient for real-world applications such as autonomous driving, robotics, or drone perception that typically require >10 m detection. This limitation substantially weakens the claimed motivation of supporting “autonomous systems” and confines the dataset’s applicability to laboratory-scale studies.
- The introduced perturbations are restricted to geometric variations (distance and angle) within a highly controlled indoor environment. The absence of electromagnetic interference, multipath reflections, or environmental factors (e.g., humidity, surface roughness variability) prevents the dataset from reflecting realistic RF conditions. While the authors acknowledge this limitation, the overall contribution remains narrower than claimed.
- The inclusion of both frequency- and time-domain data provides limited additional value, as the time-domain samples are directly computed via inverse FFT. This dual-domain representation described in the paper does not really introduce new empirical diversity but rather duplicates information that practitioners could easily derive themselves. The dataset nominally contains 142k samples; however, only 71k represent unique physical measurements. But the dual-domain counting somewhat inflates the dataset’s reported scale, and it is therefore concerning whether the reported dataset size is artificially inflated by redundancy and the contribution is overclaiming.
- The benchmark focuses solely on material classification. The benchmark, therefore, provides limited insight into the broader applicability of RF sensing for real-world material-based applications.
- Most of the models evaluated are vision, language, and time-series based. The baseline encodes some frequency information. However, there are many works on RF-sensing models, but they are not evaluated.
- From Figure 2, many material samples appear visually and structurally similar, suggesting a high intra-class correlation. The dataset lacks evidence of variability in properties such as texture, thickness, or surface roughness, making it unclear whether the reported generalization extends to real-world materials beyond those collected in the lab.
- **Minor:**
    - The use of the term “mod” for both *modality* and *mode* is confusing and should be clarified in the benchmark descriptions to avoid misinterpretation.

### Questions
Please see my weakness for most of the concerns. Some questions for authors to discuss are:
- Can the authors justify the dataset’s relevance to long-range sensing tasks such as autonomous driving or outdoor perception given that the sensing range is very close-distance
- Since the frequency-domain data can be easily transformed into the time-domain via FFT, what new information does the time-domain dataset provide that cannot be derived from the frequency data?
- Are the perturbations (distance and angle) annotated per sample, and are these metadata available in the dataset? Are there more textual annotations that we could use to give LLM the capabilities to describe the material?
- The paper reports high accuracy (>99%) across models, could this indicate potential overfitting due to low intra-class diversity or lack of environmental variability? And given the already high performance on the proposed benchmark, is the task sufficiently challenging to drive future methodological improvement?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a large-scale dataset and benchmark (RF-MatID) for radio-frequency–based material identification.
The dataset spans 16 material types across 5 superclasses.
It includes realistic perturbations such as varying angles and distances to test robustness.
The authors further establish multiple benchmark protocols to evaluate state-of-the-art models under both standard and distribution-shift conditions.
The work aims to standardize evaluation and accelerate progress in RF-based material recognition for real-world applications.

### Strengths
- This work addresses a current gap in RF-based material sensing by providing a new dataset and benchmark focused on material identification.
- The dataset is relatively large, covers a wide frequency range, and incorporates variations in acquisition conditions.
- The dataset includes of both time- and frequency-domain representations, along with several evaluation protocols targeting both in-distribution performance and robustness to angle and distance shifts.
- The authors also benchmark several state-of-the-art deep learning models across these protocols, establishing initial baselines for comparison and facilitating future reproducible work in this area.

### Weaknesses
- More discussion on the dataset's limitations, potential biases, and cost or practicality of data collection would help contextualize the scope and applicability of the benchmark.
- While the dataset is extensive, the paper would benefit from further clarification on the real-world representativeness of the collection setup and whether the acquisition hardware and environments generalize beyond the authors' configuration.
- The evaluation focuses primarily on standard deep learning baselines, and it is unclear how classical RF signal-processing methods or hybrid approaches would perform under the same protocols.
- Although perturbations in angle and distance are included, other sources of real-world variability are not explored in depth.

### Questions
See weaknesses

### Soundness
3

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
Focusing on material identification, this work indicates that there lacks a large-scale datasets with real materials labeled by frequency signals. According to this paper, frequency-domain signal data can not only increase the material identification, but also enhance the generalizability to real-world applications where a domain gap always exists. Therefore, this work proposes a frequency-labeled dataset with 142k samples, 16 fine-grained categories.

Moreover, this work integrated a benchmark that evaluates 8 previous works and one proposed baseline. The benchmark tests the domain adaptation ability, reveals some existing challenges.

### Strengths
1. Large dataset and effective data preprocessing: This work collected 71k frequency samples and extend them to time-domain representation, which is a good contribution for real-world applications. In addition, the data preprocessing is carefully designed to augment the data sources.
2. Comprehensive benchmark: The benchmark evaluated different protocols for different real-world usage. The domain adaptation evaluation is conducted on reasonable data split strategies, and the hierachical label split can effectively reveal the model's capability for material identification.
3. This work is well motivated and has the potential to inspire domain experts to design and test new framework for material identification.

### Weaknesses
1. Presentation: The presentation involves too many mechanical material-specific details, while, correspondingly, lacks AI-oriented intuitions, which is not friendly for a broader community. For example, how the details of Eq. (1) contribute to this design.
2. The benchmarked methods are all general methods that originally proposed for other downstream tasks. Is there any material identification specific methods, or frequency process focused methods worth to be includded?
3. The contributions to the AI community still remain to be discussed. The authors mentioned that the dataset can contribute to multi-modal learning. However, this is a weak claim without further demonstrations.

### Questions
Please address the questions in weaknesses if possible.

### Soundness
4

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
3

### Summary
This paper presents RF-MatID, a large open-source dataset for radio-frequency material identification. It contains 142,000 samples across 16 categories and 5 superclasses, covering 4–43.5 GHz in both frequency and time domains, with controlled variations in angle and distance. The authors benchmark nine deep learning models under multiple frequency and data split settings, showing that raw frequency-domain data can be effectively used. The main contributions include the creation of this dataset, comprehensive benchmarking protocols, and analysis of model robustness and frequency-band effectiveness.

### Strengths
1. First large-scale and open-source RF dataset covering a wide frequency range. The scale and diversity of this dataset can well benifit the community of RF-related machine learning-based study.
2. This paper establishes a well-structured benchmark with multiple frequency protocols, data splits, and nine deep learning models, offering a comprehensive evaluation of model robustness.
3. The authors carefully analyzes both frequency and time domain representations. This leads to the insight that frequency-domain data alone can achieve high accuracy, which simplifies model design.
4. The dataset is well organized, rather than just collected. This facilitates future research in RF sensing and cross-domain generalization.

### Weaknesses
1. The term "perturbation-aware" is slightly overclaimed. The authors introduce "perturbation" by varying the distance and angle, which is controllable as part of the measuring/sensing technique itself. As the authors mention, the "real-world interference" is not well considered, like electromagnetic noise, mechanical vibrations, etc.
2. One minor suggestion is that some acronyms should be explained before used, like UWB and MMW. Besides, I am not sure whether "mmWave" and MMW mean the same concept. If so, it seems inconsistent.

### Questions
1. The authors claim that "the system achieves ∼10 cm spatial resolution". The concept of "spatial resolution"is not very clear to me.
2. I notice in Figure 2 that different material plates have different sizes. Does this setting introduces additional bias into the RF data for each class? For example, a larger plate will make the signal different in some ways?

### Soundness
3

### Presentation
3

### Contribution
3
