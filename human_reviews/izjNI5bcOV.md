# WeatherGFM: Learning a Weather Generalist Foundation Model via In-context Learning

- Decision: Accept (Poster)
- Scores: 10, 3, 6, 6

## Abstract
The Earth's weather system involves intricate weather data modalities and diverse weather understanding tasks, which hold significant value to human life. 
Existing data-driven models focus on single weather understanding tasks (e.g., weather forecasting). 
While these models have achieved promising results, they fail to tackle various complex tasks within a single and unified model. 
Moreover, the paradigm that relies on limited real observations for a single scenario hinders the model's performance upper bound.
Inspired by the in-context learning paradigm from visual foundation models and large language models, in this paper, we introduce the first generalist weather generalist foundation model (WeatherGFM) to address weather understanding tasks in a unified manner. 
Specifically, we first unify the representation and definition for diverse weather understanding tasks.
Subsequently, we design weather prompt formats to handle different weather data modalities, including single, multiple, and temporal modalities. 
Finally, we adopt a visual prompting question-answering paradigm for the training of unified weather understanding tasks. 
Extensive experiments indicate that our WeatherGFM can effectively handle up to 12 weather understanding tasks, including weather forecasting, super-resolution, weather image translation, and post-processing. Our method also showcases generalization ability on unseen tasks. The source code is available at https://github.com/xiangyu-mm/WeatherGFM.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
The paper presents WeatherGFM, a pioneering general-purpose weather foundation model that unifies diverse weather-related tasks through contextual learning. By integrating different weather data formats, WeatherGFM is trained with visual prompts to handle a wide range of tasks, including forecasting, super-resolution, image translation, and post-processing. The model is designed to overcome the limitations of task-specific models by leveraging a single architecture capable of understanding and adapting to varying weather data patterns. Extensive experiments show that WeatherGFM achieves state-of-the-art performance across multiple tasks and exhibits impressive generalization on unseen tasks, demonstrating its potential as a robust, adaptable solution for complex weather modeling challenges.

### Strengths
1.	The proposed WeatherGFM can handle different weather data modalities and task objectives in a unified manner
2.	In the unseen understanding task, the WeatherGFM still shows a certain generalization.
3.	The introduction of the model input format is very clear, and the classification and identification of various weather image tasks are fully explained

### Weaknesses
1.	A mixed-modal masked image modeling (MMIM) pipeline, which is the core innovation part of the proposed framework of WeatherGFM, has not been introduced in this paper. Other parts of  WeatherGFM architecture are actually closer to the comparative ClimaX model framework. It will lack the innovative elaboration of the model framework.
2.	CSI is the main evaluation index in this paper, but ACC is also a very important index to evaluate the correlation of weather. Should this index be taken into account? Moreover, the setting of CSI threshold value needs some explanation in this paper to explain its important reference for this kind of weather situation classification.
3.	According to the comparative results of Table 2 experiments in this paper, in fact, the basic Vision Transformer can also realize a variety of understanding tasks of various data sets in the form of input format proposed by the author, and the effect of some tasks is better than that of WeatherGFM, so the advantages of this model in multi-task are specifically highlighted.
4.	There are two clerical errors: (a) "the" should be capitalized. (b) The basic learning rate in line 366 should be 1e-4.

### Questions
1. Can you explain in detail which part of the module WeatherGFM shows its advantages in various tasks?
2. Can you simply explain the significance of CSI threshold setting?

### Soundness
2

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The authors propose a generalist foundation model for weather tasks that includes in-context learning that can handle multiple weather tasks.

### Strengths
**(S1)**: The problem of creating a more general weather foundation model is important, and it is valuable that the authors are investigating this area. 

**(S2)**: Demonstrating in-context learning is also useful for weather foundation models, although I admit I haven’t looked too closely into whether there exists prior work that tackles this.

### Weaknesses
**(W1)**: Experimental comparisons are missing. The authors don’t seem to compare WeatherGFM on the same tasks used in the Climax or Aurora papers, and it is unclear why this comparison isn’t made. The main claim is that WeatherGFM is a more general foundation model, and yet the only comparison is made with simple task-specific baselines such as a ViT or UNet. There needs to be compelling evidence that WeatherGFM can outperform/compete with Climax or Aurora. Moreover, only 3 qualitative examples on out-of-distribution (OOD) examples are given in Figure 6. Quantitative evaluations on OOD tasks are quite important for a foundation model.

**(W2)**: Experimental results are weak. For the image translation tasks in Table 2, it is unclear that WeatherGFM can outperform baselines on these datasets. The authors do mention that their goal is not to achieve SoTA performance on each task— this is fine, but since comparison with other SoTA methods is missing, the results on the tasks described are not compelling enough. Moreover, in weather forecasting and super-resolution, the performance of the ViT baseline is nearly the same as WeatherGFM, which suggests only a minute (if any) improvement with the author’s designed modifications and pre-training strategy. Moreover, details on how these baseline ViT or UNets were trained are lacking, which are very crucial to determine if the experiments are fair.

**(W3)**: Lack of significant novelty. While the authors do propose changes to the ViT pre-training setup used in other Weather foundation model setups, these are not significant. The authors’ claims of supporting multi-modality isn’t well supported considering other foundation models already do support inputs from different weather sensors and earth observation (EO) systems. The only difference here seems to be satellite imagery which I don’t think is too different from the setup of the other foundation models, and the authors don’t provide any evidence that the techniques of Climax or Aurora wouldn’t also extend to this additional input. Masked image modelling for EO data and designing patch embeddings for different sensory inputs is an established technique, eg: [1].

**(W4)**: Lack of presentation clarity. I think this paper is tackling an important problem, but does not clearly motivate its solution. 
* For example, figure 3 is quite unclear and it is very hard to understand what exactly is going on (in terms of the pre-training strategy or the architectural modifications). The masked/predicted/ground-truth patch disambiguation is not clear to me. It seems that this figure contains multiple subfigures that should clearly be disambiguated and explained.
* The actual pre-training method is not described extensively. Specifically, lines 236-253 need to be clearly explained and expanded, since this seems to be the crux of the paper’s modifications. Instead, too much space is given to explaining the architecture of a ViT (lines 254-280), which is not required in the main text of the paper.
* Qualitative outputs from the model are not compared with qualitative outputs from other baselines. The authors demonstrate the outputs of WeatherGFM but don’t compare these on tasks with other weather foundation models. This is related to (W1).
* Critical details on training and inference are missing. What is the patch size? What kind of compute was used for training? How much compute? Was the dataset combined? Any dataset augmentation? etc. etc. etc. these details are important for reproducibility but are missing from the paper and appendix.

Overall, given the weaknesses describe above, I don't think this paper is ready for acceptance in this conference. 

References:  
[1] SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery, NeurIPS 2022.

### Questions
See Weaknesses. 

Are there ablations on the length of the input prompt sequence and how that affects performance?
Keeping text bold in the introduction paragraphs hampers readability.
Line 159-180: Define what t is and what its range is

### Soundness
1

### Presentation
2

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
This paper presented a weather generalist foundation model based on in-context learning, which unified a wide range of weather understanding tasks, like weather forecasting, wealth image translation, wealth super-resolution, deblur, etc.
It first defines the visual prompt template and then does in-context learning to achieve such a unified model.
The experiments demonstrate that this unified model achieved comparable results with specialist models, e.g., ViT and Unet.

### Strengths
1. A good contribution of this work is to apply in-context learning to weather-understanding tasks for building a unified model and confirming its effectiveness.
2. The proposed model demonstrates comparable results with specialist models on ten weather understanding tasks,
3. Mixed-modal mask modeling is interesting.

### Weaknesses
1. The motivation of the training objective in Figure 3 is unclear. The eq.5 is mismatched with Figure 3. There should be more details about the training objective. X_T in eq.5 lacks a definition for your universal representation.
2. Details missing. For example, the patch embedding layer is task-specific. Does this mean that there is a hard code to select which patch embedding layer is used to handle different tasks or data modalities? If so, can this model be claimed as a unified model?
3. The motivation of such a unified model is actually unclear to real-world applications. There is no obvious improvement over the conventional specialist model. In addition, there are not many combinations of tasks here. What is the essential advance of the proposed model compared with per-task specialists?

### Questions
What is the motivation for designing Mixed-modal mask modeling? It seems like there should be some references that motivate this design.

### Soundness
3

### Presentation
2

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
This paper introduces WeatherGFM, a weather foundation model inspired by in-context learning approaches used in visual and language models. WeatherGFM aims to address multiple weather-related tasks within a unified framework by converting diverse tasks with various data types into general weather prompts. The model employs a masked modeling approach for training, while fully masking the targets during inference. Experimental results on SEVIR dataset show that WeatherGFM can handle up to ten different weather tasks, including forecasting, super-resolution, image translation, and post-processing. The model also demonstrates its generalization capabilities on unseen tasks.

### Strengths
- The proposed weather prompt design effectively handles a wide range of weather-related tasks.
- By utilizing an in-context learning approach, WeatherGFM can generalize to unseen tasks without requiring fine-tuning.
- WeatherGFM demonstrates its scalability across different model sizes.
- The model successfully handles ten various tasks on the SEVIR dataset using a single, unified framework.

### Weaknesses
- In Section 3.2, it appears that WeatherGFM uses task-specific patch embedding layers, which may limit the framework's generalization ability. For instance, with the current design, WeatherGFM may struggle to generalize to new, unseen tasks that require novel sensor channels.
- The impact of increasing dataset size is unclear in Figure 7. Here, the authors compare ViT-ST with WeatherGFM-Base to illustrate the scalability of WeatherGFM. However, since ViT-ST is trained on a single task while WeatherGFM is trained on multiple tasks with more data and parameters, it’s unclear whether the performance gains are due to dataset scalability or multi-task training or larger number of parameters. Additionally, on radar spatial super-resolution (SR), scaling the dataset size or using multi-task training appears to degrade performance. A more detailed analysis and discussion on the effects of dataset scalability would be valuable.
- It would be beneficial to compare WeatherGFM’s performance against other climate foundation models (e.g., Aurora, ClimaX) using the ERA5 dataset. One of WeatherGFM’s key strengths is its ability to unify multi-task, multi-modal datasets. Thus, similar to Aurora, pretraining WeatherGFM on a combination of ERA5 and other heterogeneous datasets could potentially enhance its performance. It would be impressive if WeatherGFM could achieve comparable results to ClimaX or Aurora without fine-tuning, especially by leveraging a larger dataset and multi-task training.

### Questions
- Why does WeatherGFM show weaker performance in weather image translation compared to other tasks?
- In Table 1, there is a misuse of double lines for "GOES2Radar, CSI/219."

### Soundness
3

### Presentation
2

### Contribution
3
