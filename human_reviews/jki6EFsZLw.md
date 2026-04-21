# OmnixR: Evaluating Omni-modality Language Models on Reasoning across Modalities

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
We introduce \textbf{OmnixR}, an evaluation suite designed to benchmark state-of-the-art Omni-modality Language Models (OLMs), such as GPT-4o and Gemini. 
Evaluating OLMs, which integrate multiple modalities such as text, vision, and audio, presents unique challenges. 
Particularly, the user message might often consist of multiple modalities, such that OLMs have to establish holistic understanding and reasoning across modalities to accomplish the task.
Existing benchmarks are limited to single-modality or dual-modality tasks (e.g., image+text or video+text), overlooking comprehensive multi-modal assessments of model reasoning.
To address this, OmnixR offers two evaluation variants: (1) OmnixR-synth: a synthetic dataset generated automatically by translating text into multiple modalities—audio, images, video, and hybrids Omnify!. (2) OmnixR-real: a real-world dataset, manually curated and annotated by experts, for evaluating cross-modal reasoning in natural settings. 
OmnixR  presents a unique evaluation towards assessing OLMs over a diverse mix of modalities, such as a question that involves video, audio, and text, providing a rigorous cross-modal reasoning testbed than any existing benchmarks.
Our experiments find that all state-of-the-art OLMs struggles with OmnixR questions that require integrating information from multiple modalities to answer. 
Further analysis highlight differences in reasoning behavior and underscoring the challenges of omni-modal AI alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces *Omni×R*, a benchmark suite for evaluating omni-modality language models (OLMs) like GPT-4o and Gemini. And its key contributions include:
1. **Datasets**: *Omni×RSYNTH*, a synthetic multi-modal dataset, and *Omni×RREAL*, a real-world, expert-curated dataset for cross-modal reasoning.
2.  **Pipeline**: It proposes *Omnify!* to create the synthetic omni-modality evaluation data from the original text benchmarks.
3. **Prompting Strategies**: Tests of "Extract-Then-Answer" (ETA) prompting indicate it helps with synthetic data but is less effective with real-world data.

*Omni×R* establishes a rigorous standard for assessing OLMs, advancing omni-modal AI development.

### Strengths
1. **Gap Addressed**: The benchmark addresses a crucial gap in evaluating omni-modality language models, offering a framework for comprehensive cross-modal reasoning that goes beyond single or dual-modality assessments.
s for advancing omni-modal AI systems.

2. **Methodological Strength**: The methodological approach is robust, with a well-structured benchmark that includes both synthetic and real-world datasets, along with clear evaluation protocols and analysis.

3. **Clarity and Organization**: The paper is well-organized, providing clear explanations and thoughtful insights into the challenges and current limitations of existing omni-modality language models.

### Weaknesses
Here are some weaknesses I observe:

1. **Motivation of the Benchmark**: I understand the authors' goal of establishing a unified and comprehensive multi-modal benchmark to test OLMs' capabilities for holistic understanding and reasoning across modalities. I think the *Omni×RREAL* dataset achieves this goal to some degree. However, for the *Omni×RSYNTH* dataset, I don’t think it effectively assesses an OLM's cross-modal reasoning capabilities. Since *Omni×RSYNTH* is derived by translating from pure text, the images and videos it contains are entirely text-based, which means it primarily tests OLMs' OCR abilities rather than other critical capabilities like temporal, spatial, and grounding capabilities. In real-world scenarios, when faced with this type of data (e.g., slides), it would be more practical to use expert models to extract text information from other modalities, and then pass the task to LLMs. A more meaningful scenario might involve using films, where audio and video from movies are input to OLMs to answer plot- and character-related questions. I believe this setup would provide a more valuable test for OLMs.


2. **Evaluation**: In the "Extract-Then-Answer" (ETA) prompting method, the authors first prompt the OLMs to generate corresponding OCR results and then answer questions based on the extracted text. As I mentioned above, given this approach, why do we even need OLMs for such tasks? We could just use expert models to handle the first step. In real-world scenarios, "Extract-Then-Answer" is impractical, so I feel that this evaluation method also lacks significance.

### Questions
Kindly find below additional questions. 

1. Can you provide the results of video+audio on the *Omni×Real* dataset?

2. For Claude and GPT-4o, you only tested video and image or just the image modality. So, why not test more open-sourced models? like LLaVA, QWen, etc. They can receive videos and images as input.

3. Why not test some open-source OLMs, such as Macaw-LLM, Next-GPT, or VideoLLaMA2?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel evaluation suite called Omni x R for omni-modality language models (OLMs) evaluation. They propose an automatic data synthetic method to convert natural language questions into other modalities, i.e., audio, image, video. Besides, they also annotate 400 samples from real videos to  other modalities. With Omni x R, they conduct experiments of several OLMs and provide insights to the develop of future OLMs

### Strengths
1. This paper proposed a novel evaluating method called Omni x R for Omni-modality Language Models (OLMs) to assess OLMs over a diverse mix of modalities. Unlike other benchmarks that focus on one or two modalities, Omni x R evolves up to six modalities that embed information, presenting new challenges for OLMs.
2. Omni x R can effectively measure the reasoning behavior discrepancy in different modalities. Besides, this paper demonstrates several key findings regarding this discrepancy. For example, they find extracting information and then conducting reasoning could improve model performance on modalities other than text.

### Weaknesses
1. The automatic data synthesis method is somehow oversimplified. For example, rendering text onto a white background image or using tts to translate text to audio may be used in synthetic training data generation. These kinds of data could vary largely from the data distribution in the real world, which makes the OMNI-R-Synth data unreliable in reflecting the model performance in real life.
2. The benchmark size is small. There are only 400 real samples and 400 synthetic samples. Besides, Demonstrations of some examples of OMNI-R-Synth and OMNI-R-Real could help readers gain an intuitive impression of the data format.
3. I’m curious about how humans perform in different modalities. A hypnosis is that audio could cause difficulty for humans to perform reasoning. Are the authors planning to add a human upper bound for Omni x R?
4. When conducting ETA prompting, did the author measure the precision of the intermediate textual transcripts? Does the performance drop come from failure to catch the key information or failure to conduct reasoning when some modalities exist?

### Questions
Please see above.

### Soundness
4

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
Authors propose the Omni x R benchmark intended to evaluate Omni LLM, presenting three modalities: text, vision and audio. Two variants are proposed: Omni x R_synth, which is synthetic built, and Omni x R_real that has been human curated. The sources for those subsets are respectively MMLU-Pro and Youtube. 

Along those benchmarks, the authors introduce the Omnify method, which offers to create synthetic omni-modality evaluation data by 'translating' across modalities data sample (e.g. video to image and audio, or audio and text, etc).

### Strengths
- The paper is well written. Data sources, models and experiments are clearly stated with examples. Takeaways are given in page 7, that help the reader understanding the scope of the contributions. 
- Good point on the authors to spot the data contamination with Gemini on GSM8K.

### Weaknesses
- One big weakness of this work seems that the whole paper doesn't show examples of Omni x R_synth or Omni x R_real. It is unclear for the readers how exactly those benchmark look like in reality? One could assume that Omni x R_synth is probably a MCQ-type of benchmark, but this isn't stated in the paper. What Omni x R_real is looking in practice is even less clear in those settings. As those two benchmarks are the main contribution of this work, it is a bit unclear why the authors have not shown them in first place. Arguably Figure 1 should be not-cherry-picked examples of Omni x R_synth and Omni x R_real. 
- The current Figure 1 is not very informative. The source of that example isn't mentioned. The terminology "OmniLM" in the figure seems to conflict with the rest of the paper ("omni-LM"?). The figure mentions "Gemini or GPT-4o". The 'OR' alone is misleading. There are actually many omni-LLM in the community, starting with Reka-Core, UnifiedIO2, AnyGPT, MIO-Instruct, and many others. They all seem to be ignored by this paper. Finally, the legend mentions using 'Gemini'. Arguably, Figure 1 should be removed altogether and replaced with actual examples from Omni x R_synth and Omni x R_real.    
- The omnify method doesn't seem to be evaluated. 
- Terminology: The paper mixes freely "omnimodal" and "omni-modal". Please pick one and keep the terminology consistent. 
- Terminology: Figure 2 uses "OmniXR_{real}" while the rest of the paper uses "Omni x R_{real}". Why not using the same consistent terminology?
- Nitpicks: L092 s/Youtube/YouTube/, L126 s/Youtube/YouTube/, L192 s/youtube/YouTube/. The only correct is L529. 
- Figure 2: The color code is misleading. Blue seems to be 'Chemistry' in the pie chart. The YouTube source is blue too. However, in L197 it is mentioned that 4 categories are used: Math, Coding, Physics and Chemistry. Same remark for the string '14 Categories' with a blue box above. This figure 2 seems completely inaccurate. Use color code when needed and accurately. 
- As an evaluation paper that proposes a benchmark, the model evaluated are quite lacking. The work focused on Gemini, Claude and GPT-4o, putting aside many omni-LLM (Reka, AnyGPT, Video-SALMONN, UnifiedIO2, etc).

### Questions
- What is the license for your benchmarks? It is probably safe to assume that MMLU-Pro can be used freely by the community. However, scraping YouTube videos might be restricting the use of Omni x R. What is the license? Does it allow Commercial use, if say another company wants to use that benchmark to evaluate their internal OLM?  
- L204: Frame merging is mentioned. How realistic is that? Can you show a not-cherry-picked example?
- Is there are reason why the evaluation was limited to closed-source models (Gemini, Claud and GPT-4o)? Why not evaluating the rest of the omni-LLM models?

### Soundness
3

### Presentation
3

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
This paper proposes an evaluation suite for multi-modal language models - Omni×R, which can integrate various modalities such as text, vision, and audio. The design of Omni×R aims to assess the models' capabilities in cross-modal reasoning, especially when the user's messages contain multiple modalities, the models need to be able to understand and reason comprehensively to complete the task. The paper introduces two evaluation variants: synthetic data Omni×RSYNTH and real-world data Omni×RREAL. These evaluations reveal the existing problems of the state-of-the-art models.

### Strengths
- The motivation of this paper is very clear, the problem proposed is valid, and the benchmark proposed is distinctly different from others.
- The evaluation experiments in this paper are comprehensive, exploring the experimental results under different modality combinations and different prompt scenarios, and providing a detailed analysis.
- The pipeline of data construction is inspiring, providing a possible solution for the subsequent generation of cross-modal datasets.

### Weaknesses
- The synthesis method for audio data is common, but the direct rendering approach for images and videos seems overly simplistic and overly reliant on the model's OCR capabilities.
- While the paper points out the challenges models face in cross-modal tasks, it does not provide specific model optimization strategies or directions for improvement.

### Questions
- Why is the performance of ETA Prompting also poor in Table 2, especially on the audio modality where it performs worse than the normal evaluation?
- I would like to know if the authors have cleaned the constructed data?

### Soundness
3

### Presentation
3

### Contribution
3
