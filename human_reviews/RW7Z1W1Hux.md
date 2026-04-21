# CinePile: A Long Video Question Answering Dataset and Benchmark

- Avg Score: 5.33
- Decision: Reject
- Scores: 5, 5, 6

## Abstract
Current datasets for long-form video understanding often fall short of providing genuine long-form comprehension challenges, as many tasks derived from these datasets can be successfully tackled by analyzing just one or a few random frames from a video. To address this issue, we present a novel dataset and benchmark, CinePile, specifically designed for authentic long-form video understanding. This paper details our innovative approach for creating a question-answer dataset, utilizing advanced LLMs with human-in-the-loop and building upon human-generated raw data. Our comprehensive dataset comprises 305,000 multiple-choice questions (MCQs), covering various visual and multimodal aspects, including temporal comprehension, understanding human-object interactions, and reasoning about events or actions within a scene. Additionally, we fine-tuned open-source Video-LLMs on the training split and evaluated both open-source and proprietary video-centric LLMs on the test split of our dataset. The findings indicate that although current models underperform compared to humans, fine-tuning these models can lead to significant improvements in their performance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces CinePile, a new large-scale dataset for long-form video question answering (VQA).  The authors address the limitations of existing VQA datasets, many of which are easily solvable by analyzing only a few frames. CinePile aims to remedy this by focusing on questions requiring genuine long-form comprehension, encompassing temporal understanding, human-object interactions, and event reasoning.  The dataset creation leverages a novel pipeline combining readily available audio descriptions (ADs) of movies, automatic speech recognition, and large language models (LLMs) like GPT-4 and Gemini, for automated question generation.  A significant effort is dedicated to filtering out low-quality questions through an adversarial refinement process. The paper also presents a benchmark evaluating various open-source and commercial video LLMs on CinePile, revealing the strengths and weaknesses of current models in long-form video understanding and highlighting the potential for improvement through fine-tuning on the CinePile training set.

### Strengths
1. This paper gives a large-scale and challenging long-form video VQA dataset.  The automated pipeline for question generation is a major strength.  The use of readily available ADs and LLMs offers a cost-effective and scalable approach to annotate long videos. Also, the paper includes detailed analyses of various aspects of the dataset, including question type distribution, vision reliance, and question difficulty.  The authors demonstrate a deep understanding of the challenges involved in creating and evaluating such a dataset.
2. The evaluation of numerous open-source and commercial LLMs provides a new benchmark, offering insights into the current state-of-the-art in long-form video understanding. The fine-tuning experiments showcasing the effectiveness of CinePile for improving model performance are compelling and demonstrate the practical utility of the dataset.

### Weaknesses
1. The description of the adversarial refinement process could be more detailed.  Specific examples of how questions were modified and the criteria for determining success would be beneficial.
2. While the authors acknowledge potential biases in the LLMs used for question generation and the geographical limitations of the movie clips, a more in-depth discussion of potential biases (e.g. countries, languages, etc.) in the dataset itself and mitigation strategies would strengthen the paper. How well does the dataset generalize to videos outside the specific sources used? 
3. While the paper highlights the performance gap between open-source and commercial models, a deeper dive into why this gap exists would be valuable.  Is it solely due to model architecture, training data, or other factors? Besides, human evaluation is primarily performed by the authors themselves. While they acknowledge potential biases,  a larger-scale, more diverse human evaluation would significantly enhance the credibility of difficulty and quality assessments.

### Questions
1. Can the authors provide a more detailed breakdown of the types of biases identified and mitigated during the dataset creation process?
2. What are the computational costs associated with the automated question generation and adversarial refinement pipelines?
3. Can the authors provide a more in-depth analysis of the failure modes of the open-source models?
4. How does the performance of the models on CinePile correlate with their performance on other video understanding benchmarks?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents CinePile, a novel benchmark dataset consisting of 305,000 multiple-choice questions designed to challenge models with authentic long-form video understanding, moving beyond the capability of analyzing just a few frames. CinePile incorporates a diverse range of question types, emphasizing temporal comprehension, multimodal reasoning, and narrative understanding. Created through an innovative process that utilizes large language models and human-generated data, CinePile offers a comprehensive measure of model performance. The paper also includes an evaluation of various video-centric large language models, indicating significant room for improvement compared to human performance, and discusses the potential biases and ethical considerations inherent in such datasets. CinePile aims to serve as a scalable and comprehensive benchmark for assessing and advancing the capabilities of video understanding models.

### Strengths
1. The paper introduces a detailed and effective pipeline for generating a high-quality question-answering dataset, CinePile, which is specifically tailored for long-form video understanding. 

2. To ensure the quality of the automatically labeled data, the authors have developed an adversarial refinement process that rigorously evaluates and improves the dataset's accuracy. 

3. Additionally, the paper includes an extensive testing suite that assesses the performance of the latest open-source models against the most capable commercial models, providing a comprehensive benchmark. 

4. The authors also offer a detailed analysis, shedding light on the strengths and weaknesses of different models and their ability to understand and reason about video content.

### Weaknesses
1. The authors' dataset statistics table compares CinePile with a variety of VideoQA datasets. However, there are two minor issues. Firstly, CinePile predominantly consists of movie clips, and a separate analysis comparing it with other movie QA datasets would be beneficial for a more targeted evaluation. Secondly, the videos in CinePile are only 160 seconds long, which doesn't offer a significant advantage in terms of length over other datasets.

2. While the authors' use of templates to construct QA pairs facilitates rapid generation, it severely compromises the diversity of the questions. This often results in a distribution of generated QA pairs that significantly deviates from the distribution of real human prompts. Consequently, the constructed training set lacks diversity, offering limited benefits for model improvement. Additionally, the metrics reflected by the test set may not provide a sufficiently reliable reference value due to this disparity.

### Questions
How to ensure that the proposed test set has a high reference value for VideoQA, rather than just evaluating the templated MCQ of the movie domain?

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
3

### Summary
This paper presents CinePile, a novel large-scale dataset and benchmark for long-form video understanding, containing 305,000 multiple-choice questions across 9,396 movie clips. The authors introduce a data creation pipeline that combines human-created audio descriptions with LLM-generated question templates, followed by quality filtering to ensure questions require genuine video understanding. The dataset's effectiveness is demonstrated through evaluations of 24 video-language models, where the best commercial model (Gemini 1.5 Pro) achieved 60.12% accuracy and the top open-source model (LLaVA-OV) reached 49.34%, both still significantly below human performance of 73.21%. The authors also show that fine-tuning existing models on CinePile's training set can substantially improve performance, as demonstrated by Video-LLaVa's improvement from 25.72% to 44.16%. This work addresses critical limitations in existing video QA datasets by requiring true multimodal comprehension rather than allowing models to succeed through either visual or dialogue understanding alone.

### Strengths
- Scale and Efficiency: The paper presents an innovative automated pipeline that created 305,000 high-quality QA pairs - which is 70-75x larger than comparable datasets like MoVQA (21,953) or Video-MME (2,700).
- Clear Benchmark Value: The significant performance gap between humans (73.21%) and best models (60.12% for Gemini 1.5 Pro) demonstrates that this dataset effectively challenges current state-of-the-art systems and provides meaningful room for improvement.

### Weaknesses
- The paper lacks rigorous statistical analysis of the dataset's properties - for example, there are no inter-annotator agreement scores for human evaluation and no analysis of the language content of the answers or questions. For instance, with llm-generated answers, the correct answer often is longer than the wrong answer which becomes an issue
- The "Setting and Technical Analysis" (STA) questions may not truly test long-form understanding since they could potentially be answered by analyzing just a few frames, contradicting the paper's core claim about requiring temporal understanding.
- The paper lacks ablation studies on the number of sampled frames, making it unclear how model performance varies with different temporal sampling strategies and what the minimum required context is.

### Questions
What are the statistics on the length of the answers on the dataset?
Just general more exploration on the content of the questions and answers

### Soundness
3

### Presentation
3

### Contribution
3
