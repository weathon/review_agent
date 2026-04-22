# ST-VLM: Kinematic Instruction Tuning for Spatio-Temporal Reasoning in Dynamic Videos

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Spatio-temporal reasoning is essential for understanding real-world environments in various fields, $\textit{e.g.}$, autonomous driving and sports analytics. While recent advances have strengthened the spatial reasoning abilities of Vision-Language Models (VLMs) through large-scale training data, these models still struggle with kinematic aspects such as traveled distance and speed of moving objects. To bridge this gap, we construct a spatio-temporal reasoning dataset and benchmark for kinematic instruction tuning, referred to as $\textbf{STKit}$ and $\textbf{STKit-Bench}$. They consist of real-world videos with 3D annotations that capture object motion dynamics, including traveled distance, speed, movement direction, inter-object distance comparisons, and relative movement direction. To further scale data construction to videos without 3D annotations, we propose an automatic pipeline for generating pseudo-labels via 4D reconstruction at a real-world scale. Building on this kinematic instruction tuning data, we introduce $\textbf{ST-VLM}$, a VLM enhanced for spatio-temporal reasoning, which achieves strong performance on STKit-Bench. Moreover, ST-VLM generalizes robustly across diverse domains and tasks, outperforming baselines on comprehensive spatio-temporal reasoning benchmarks. Finally, by integrating learned spatio-temporal reasoning with existing abilities, ST-VLM enables complex multi-step reasoning grounded in kinematics.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper describes a new vision dataset, that was generated synthetically by augmenting existing datasets with 3D object position and movement information (such as the objects’ movement directions, speeds, etc.). It is based in part on existing datasets that include this type of information in some form and in part on the use of 4D reconstruction methods to video datasets where such information is not included.  The resulting dataset is used to finetune a vision-language model, which performs better at predicting this kind of information than models that were not fine-tuned on this kind of data.

### Strengths
The paper introduces a novel dataset. The proposed dataset does what it promises to do. It enables a model finetuned on it to perform comparably well at the task of predicting speeds, directions, etc. of objects shown in the video.

### Weaknesses
The dataset focuses on domains like sports and driving, where object position and motion information can be readily extracted. This raises the question what the benefit of distilling this information into an existing vision language model is, instead of relying on existing approaches which would use an external tool that simply extracts such information at inference time. While this question is answered partly in section 6.3 on emergent capabilities, that section is very preliminary and highlights just a few examples qualitatively. 

The method seems to rely on relatively clean scenarios that are free from occlusions, camera motions.  

The task of predicting speed is inherently ill-posed, suggesting that applying the model to scenarios that are slightly out of the ordinary with respect to object geometry or size, for example showing remote controlled toy cars, would make the model give drastically incorrect results? It would be good to better understand the limitations of this approach considering this.

### Questions
Will the dataset and model be publicly released? 

How would the performance of the model vary if the framerate of a video is varied? Is robustness to such variations not important, especially when considering this as a foundational skill of a vision-language model? 

What about extraordinary scenarios, like videos showing toy cars mentioned above? 

(minor) The figures (especially Figure 1 and 2) are good for on-screen viewing but kind of hard to see in a printout.

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
This paper is doing the task that let the vlm understand kinematic spatio-temporal reasoning, such as an object's speed or traveled distance. The authors introduce ST-VLM, a new model trained via kinematic instruction tuning. To enable this, they create the STKit dataset, which contains video-based questions about speed, distance, and movement direction. They also introduce novel pseudo-labeling pipeline that uses 4D reconstruction to generate this kinematic data from unlabeled videos. On STKit-Bench, ST-VLM substantially outperforms baselines like GPT-5 and demonstrates emergent multi-step reasoning capabilities.

### Strengths
- The paper is well writing with clear logic and good motivation.
- The three-stage filtering strategy (rule-based, general model-based, and task-specific model-based) is comprehensive. It demonstrably improves the quality of pseudo-labels.
- The ablation in Table 6 clearly validates the authors' design choices, showing that both the 3D ground-truth data and the filtered pseudo-label data contribute significantly to the final model's performance.
- Experiment and dataset building method are detailed.

### Weaknesses
- The pseudo-label is not accurate. While the filtering pipeline is effective, a 29% mean error rate remains in the pseudo-labeled traveled distance data.
- See in questions.

### Questions
- How about using data generated in simulator? What is the advantage of simulator data with the 4D model annotated data? In my view is that the 4D model annotated data may not accurate but the video distribution is from the real data, but the simulator data is accurate but is not photo realistic. 
- Recently some work also using vlm do depth estimation such as DepthLM[1]. I am wondering what is their method different with this paper(and I think ST-VLM also can do depth estimation). I am wondering if the authors could provide the results of ST-VLM on depth estimation tasks.
- Open source plan?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ST-VLM, a vision-language model enhanced for spatio-temporal reasoning through kinematic instruction tuning using the STKit dataset and STKit-Bench benchmark. It aims to address a key limitation of existing VLMs -- poor handling of dynamic object kinematics -- by leveraging datasets with 3D and 4D information. The paper includes extensive evaluation of state of the art models on STKit-Bench and the ST-VLM shows promising signs of improved spatio-temporal understanding.

### Strengths
* The proposed STKIT dataset and benchmark is novel and interesting.

* The paper shows that fine-tuning on the proposed STKIT dataset leads to improved performance over the plan LLaVA-OneVision-7B model on PerceptionTest, MVBench, VideoMME, MLVU, NExT-QA. Which shows improved spatio-temporal understanding.

* The paper includes extensive evaluation of state of the art models on STKit-Bench.

### Weaknesses
* The proposed STKIT-BENCH includes a vast majority of questions (92.9%) from the autonomous diving domain. This limits the diversity of the benchmark. Due to the heavy reliance on the autonomous driving domain, the paper should include comparison to popular benchmarks such as "DriveLM: Driving with Graph Visual Question Answering, ECCV 2024".

* For 3D understanding of street scenes and answering kinematic questions on distances, it is critical to have access the camera parameters. The proposed model is only able to perform well because the STKIT dataset and benchmark uses the same datasets: Nuscenes and Argoverse. Training on these datasets allows the model to memorize the camera parameters and thus perform well on the evaluation data. The STKIT-BENCH should include datasets such as Cityscapes for true zero-shot evaluation.

* The evaluation in Table 7 and Table 8 should include state of the art approaches such as VideoLLAMA3 and Qwen-2.5-VL.

* The paper should discuss prior work on grounding to fine-grained spatio-temporal visual information in videos such as: "Look, Remember and Reason: Grounded reasoning in videos with language models, ICLR 2024"; "Fine-grained Spatiotemporal Grounding on Egocentric Videos, ICCV 2025".

### Questions
* The choice of data sources for STKIT-BENCH should be motivated in more detail.
* The effect of the overlap between STKIT dataset and benchmark should be explain in more detail.
* Prior work in grounding to spatio-temporal visual information in videos should be explain in more detail.

### Soundness
3

### Presentation
3

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
The paper aims to strengthen the spatio-temporal reasoning capability of current VLMs. To this end, the authors introduce STKit, a large-scale dataset with seven types of motion-related question–answer pairs generated through a 4D reconstruction and pseudo-labeling pipeline, and STKit-Bench, a benchmark for quantitative evaluation. By tuning LLaVA-OneVision with these instructions, it achieves performance improvement on STKit-Bench and video understanding benchmarks.

### Strengths
1.	The paper is well-written.
2.	I appreciate the authors’ efforts to curate data for spatio-temporal reasoning.
3.	The results are promising after finetuning the baseline with the curated data.

### Weaknesses
1.	For STKit-Bench, how to ensure annotation quality, especially annotation related to the distance annotation? Meanwhile, how to ensure its diversity to cover the real-world scenes?
2.	Both training data and testing data come from the same data annotation pipeline. How could the authors deal with domain overlap?
3.	The method does not explicitly model physical kinematics. Therefore, “kinematic instruction tuning” may not be a good description.
4.	Could the authors show some evidence that the models learns the spatio-temporal reasoning rather than static correlations?
5. Will all curated datasets/benchmarks be public to the community?

### Questions
See the questions in Weakness part.

### Soundness
2

### Presentation
3

### Contribution
2
