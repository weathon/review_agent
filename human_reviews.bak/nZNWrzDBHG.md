# VEBench: Towards Comprehensive and Automatic Evaluation for Text-guided Video Editing

- Decision: Reject
- Scores: 5, 5, 3, 5

## Abstract
Video editing task has gained widespread attention in recent years due to their practical applications and rapid advancements, driven by the emergence of diffusion techniques and Multi-modal Large Language Models (MLLMs). However, current automatic evaluation metrics for video editing are mostly unreliable and poorly aligned with human judgments. As a result, researchers heavily rely on human annotation for evaluation, which is not only time-consuming and labor-intensive but also difficult to ensure consistency and objectivity. To address this issue, we introduce **VEBench**, the largest-ever video editing meta-evaluation benchmark to evaluate the reliability of automatic metrics. It includes 152 video clips and 962 text prompts, from which 160 instances are sampled to generate 1,280 edited videos using 8 open-source video editing models, accompanied by human annotations. Especially, the text prompts are first crafted using GPT-4, followed by manual review and careful categorization based on editing types for a systematic evaluation. Our human annotations cover 3 criteria: *Textual Faithfulness*, *Frame Consistency*, and *Video Fidelity*, ensuring the comprehensiveness of evaluation. Since human evaluation is costly, we also propose **VEScore**, employing MLLMs as evaluators to assess edited videos from the criteria above. Experiments show that the best-performing video editing model only reaches an average score of 3.18 (out of a perfect 5), highlighting the challenge of VEBench. Besides, results from more than 10 MLLMs demonstrate the great potential of utilizing VEScore for automatic evaluation. Notably, for Textual Faithfulness, VEScore equipped with LLaVA-OneVision-7B achieves a Pearson Correlation score of 0.48, significantly outperforming previous methods based on CLIP with the highest score of 0.21. The dataset and code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper studies an important task of benchmarking video editing. It collects video clips from the DAVIS dataset and creates text prompts efficiently based on GPT-4V automatically. The data taxonomy / editing types mainly focus on content and style. Besides the benchmark, the authors also developed an automatic metric called VEScore, which aligns better with human ratings and serves as a more robust and scalable automatic method compared to traditional metrics. 

A more comprehensive video editing benchmark could contain other popular editing types such as editing motion, emotion, etc. Further, the benchmark could go beyond just text-based video editing, such as providing a reference image and using it to replace the object in the original video. For such reference-image based video editing task, we may need additional metrics for ID preserving of the reference image.

### Strengths
1/ This paper studies an important task of benchmarking video editing

2/ Besides the benchmark, the authors also developed an automatic metric called VEScore, which aligns better with human ratings and serves as a more robust and scalable automatic method compared to traditional metrics.

3/ In addition to single object, this benchmark further considers the scenario of multiple objects

### Weaknesses
1/ Why only collect videos from DAVIS dataset? Would it be better to include more different sources and domains to increase content diversity?

2/ When reading the introduction, I feel it be more informative to have a teaser image to showcase editing examples along with the textual descriptions in the main text. E.g. When reading "the number of editing targets (Single or Multiple)", a more concrete example like the one in Fig 1 would be helpful.

3/ When proposing multiple targets, understand it contains cases like “brown bear” and “zoo” to be edited as “polar bear” and “Arctic ice”. Are these cases in the benchmarks that edit multiple objects of the same categories? e.g. a video contains multiple "brown bears" and change all or part of them into “polar bear”

4/ In practice, people use not only text for video editing but can incorporate various conditional data. For example, a) provide a reference image of the new object which shall be used to replace the target object in the video. b) besides just change content and style, how about change motion of an object e.g. car moving left in the original video and change it to move right, change walking to other actions. How to make this video editing benchmark more comprehensive and include various editing scenarios, not limited to just text-based editing?  

5/ Any specific challenges for editing long video? Would it be interesting to include cases of editing long videos / movie-like videos?

### Questions
Kindly see the weakness section

### Soundness
3

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
5

### Summary
This paper presents VEBench, a comprehensive benchmark designed to assess text-guided video editing models, alongside VEScore, an automated scoring system that utilizes Multi-modal Large Language Models (MLLMs) for evaluation. The research evaluates VEBench and VEScore based on criteria such as Textual Faithfulness, Frame Consistency, and Video Fidelity, with the goal of minimizing reliance on expensive human evaluations.

### Strengths
1. VEBench provides a substantial and structured benchmark, offering a variety of editing types and criteria that address significant gaps in the evaluation of video editing.
2. The introduction of VEScore showcases the potential for automation in evaluating video edits, presenting preliminary evidence of better alignment with human judgments.
3. The study offers a detailed analysis of current video editing models, highlighting the challenges in maintaining frame consistency and video fidelity, thereby providing valuable insights for future model development.

### Weaknesses
1. VEScore relies heavily on prompt-based MLLM outputs without employing fine-tuning or advanced techniques like token probability averaging. This approach is limited by the performance constraints and potential hallucinations of MLLMs, especially in complex scenarios.
2. The study does not thoroughly explore the consistency and reproducibility of prompt engineering, with limited testing across different prompt variations to ensure robustness.
3. The dependence on large-scale models (32B) or expensive closed-source APIs is impractical for most applications, raising concerns about scalability and general accessibility.
4. While VEBench enhances existing benchmarks, it may not significantly advance evaluation methods, as it neither fine-tunes MLLMs on evaluation-specific data nor addresses ongoing issues in evaluating nuanced or multi-target edits.

### Questions
1. Given that other recent benchmarks have successfully implemented task-specific training or fine-tuning for video evaluation, could the authors clarify their rationale for opting against these techniques in VEScore?
2. How does the paper address potential variability and hallucination in VEScore’s output, given its reliance on prompt-only evaluation without task-specific fine-tuning?

While VEBench serves as a benchmark that expands the dataset scale and diversity of textual prompts, it does not fundamentally address the critical pain points in video editing evaluation, particularly in objectively assessing detailed changes and stylistic variations in edited videos. Although VEScore demonstrates relatively high correlation with human ratings on certain criteria, its correlation coefficient of approximately 0.49 remains insufficiently significant.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents VEBench, a comprehensive meta-evaluation benchmark designed for assessing automatic evaluation metrics in text-guided video editing tasks.  VEBench includes 152 video clips and 962 text prompts categorized by editing type and target complexity. Additionally, the benchmark involves three main evaluation criteria—Textual Faithfulness, Frame Consistency, and Video Fidelity—and employs VEScore. It is a novel benchmark, but some prons should be solved in the rebuttal process.

### Strengths
1. Authors meticulously curated a dataset, verified its quality, and categorized it based on editing complexity, enhancing its robustness. 
2. The paper thoroughly describes the data collection, taxonomy, annotation processes, and experimental setup.

### Weaknesses
1. The amount and diversity of data is a major concern for me. Selecting only 152 videos from the DAVIS2017 and 2019 datasets, and each video has only 25 frames, seems to me to be insufficient. Although the author has tried his best to further classify and study the data in a fine-grained manner, the limited size of the data still makes this part of the data not representative enough.

2. The author did not provide enough examples throughout the article to demonstrate the dataset. With only some text descriptions and some statistical analysis, it is still difficult for me to intuitively see the challenge and diversity of the data itself. Therefore, the appendix needs more presentations on video sequences and a more fine-grained introduction to videos.

3. I would also like more examples, including good and bad cases. I think evaluation is crucial for the future development of algorithms, and bad cases are the key to further exploring the bottlenecks of algorithms. As a Benchmark article, the author not only needs to introduce the evaluation work itself clearly, but also needs to clearly give a very fine-grained explanation and analysis to explain what advantages and disadvantages the currently tested methods have in specific situations, and to illustrate with specific picture sequences. Unfortunately, the current version does not have these contents, so it is more like a technical report than an academic paper.

4. Finally, I would also like to know the more specific processes and standards used by the authors when selecting human subjects, such as the basic situation of these subjects, whether their cognitive levels are consistent, and whether their ability to understand the tasks is similar, to ensure that the evaluation results provided by the human subjects are valid.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper presents a comprehensive meta-evaluation benchmark for text-guided video editing, and proposes VEScore for automatic evaluation. It addresses the lack of reliable metrics in video editing evaluation. Experiments show challenges for current models and the potential of VEScore.

### Strengths
1. The proposed benchmark VEBench is the largest-ever in this field, providing a rich set of data including diverse video clips, text prompts, and human annotations, which enables in-depth analysis of video editing models.

2. The proposal of using MLLMs as evaluators (VEScore) is innovative and shows great potential. It outperforms traditional metrics in correlating with human judgments, especially in evaluating Textual Faithfulness.

3. The experiments reveal valuable insights about the performance of current video editing models, such as their struggles with certain editing types.

### Weaknesses
1. Although VEBench contains 152 video clips and 962 text prompts, compared to some large-scale general-purpose datasets, its data diversity may still be insufficient. For example, the data sources mainly focus on the DAVIS dataset, which may have limitations in terms of video scenes, content themes, shooting styles, etc., and may not comprehensively cover all types of video editing needs in the real world.

2. Although the paper mentions using MLLMs for evaluation, it lacks in-depth analysis of the specific strengths and limitations of different MLLMs when handling video editing evaluation tasks.

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
