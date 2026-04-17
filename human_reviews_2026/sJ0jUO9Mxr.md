# Measuring Audio's Impact on Correctness: Audio-Contribution-Aware Post-Training of Large Audio Language Models

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Large Audio Language Models (LALMs) represent an important frontier in multimodal AI, addressing diverse audio tasks. Recently, post-training of LALMs has received increasing attention due to significant performance improvements over foundation models. While single-stage post-training such as reinforcement learning (RL) has demonstrated promising results, multi-stage approaches such as supervised fine-tuning (SFT) followed by RL remain suboptimal. The allocation of data across multiple training stages to maximize LALM capabilities has not been fully explored, and large-scale, high-quality datasets for such research are also lacking. To address these problems, we firstly present AudioMCQ, a comprehensive audio multiple-choice question dataset comprising 571k samples with two kinds of chain-of-thought annotations. Secondly, we investigate the prevalent zero audio-contribution phenomenon in LALMs, where models derive correct answers solely from textual information without processing audio content. We propose Audio-Contribution Filtering to partition data into weak and strong audio-contribution subsets. Based on these insights, we develop two effective post-training paradigms: Weak-to-Strong (SFT on weak audio-contribution data followed by RL on strong audio-contribution data) and Mixed-to-Strong (SFT on mixed audio-contribution data followed by RL on strong audio-contribution data). We achieve first place in the DCASE 2025 Audio-Question-Answering challenge by using AudioMCQ. Additionally, leveraging our dataset with different training strategies, we achieve 78.2\% on MMAU-test-mini, 75.6\% on MMAU, 67.0\% on MMAR, and 71.7\% on MMSU, establishing new state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a good motivation and an interesting problem formulation, but the proposed approach relies heavily on previous works and employs a relatively simple fusion method, which limits its novelty.

### Strengths
The problem formulation for “evidence bottleneck” is very intriguing and inspiring for the poor acoustic grounding. The paper is well-written with clear motivations and methodology formulation. Prioritizing the perceptual front-end is a novel and reasonable approach to enhancing the audio understanding ability of LALMs.

### Weaknesses
Since the work builds upon pretrained models such as Whisper, CED-Base, and Kimi-Audio-7B, the network design itself is not particularly innovative, as much of the credit should be attributed to these existing models. The main contribution lies in effectively combining and fusing them using LoRA. This direction is promising and could be further strengthened to reach the level of ICLR.

The Time-Aware Alignment and Inject-and-Add Fusion components are not novel, as similar operations are commonly used in speech and audio processing. Please clarify what specific novelty, if any, is introduced in these parts.

Although GRPO is a known term, please provide its full name for clarity.

The experimental evaluation appears limited, given that it includes comparisons with only five LALMs and three benchmarks.

### Questions
Regarding the frequency-domain knowledge extraction, what is the highest frequency captured? Does this differ between general audio and speech when extracting meaningful features? It would be helpful if the authors could compare these aspects in experiments and share their insights with the research community.

As this paper seeks to advance large audio-language models through a newly proposed dataset, could the authors clarify how frequently the dataset has been utilized in the domain and provide evidence that it is well-defined and meaningful? Have you released the dataset and codes, or do you plan to make them publicly available?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates whether audio contributes to answering audio-based questions and the authors create an audio Q&A dataset and discover that models often answer correctly using just the text context, ignoring the audio. They split the training data into cases where audio is actually needed vs. not, and then train the model in stages, focusing on the audio-critical examples. This targeted training leads to better performance on audio question answering tasks.

### Strengths
- The paper introduces AudioMCQ, a large-scale audio multiple-choice question dataset with 571,000+ samples.
- Identification of zero audio-contribution issues: The authors systematically measure how often models answer correctly without using the audio. By replacing audio, they find that current LALMs still achieve far-above-chance accuracy.
- They propose ACF to split training data based on whether the audio actually affects correctness, with data partitioning, allowing the training to focus differently on audio-easy vs audio-critical examples.

### Weaknesses
- All experiments and the dataset use a multiple-choice Q&A setup. The method’s effectiveness on open-ended or free-form audio questions remains untested.
- The ACF procedure uses predictions from three existing LALMs to decide if audio is needed for each sample, this automatic labeling might be noisy or biased.
- The two training paradigms showed different strengths across benchmarks, and no single approach uniformly excels. For example, the Weak-to-Strong method achieved the best results on the MMAU benchmark but underperformed on datasets where audio cues were crucial, where Mixed-to-Strong worked better. This indicates the training strategy must be aligned to the task’s audio-contribution type, which further adds complexity.

### Questions
- How can the audio-contribution-aware approach be applied to more open-ended or dialog-style audio question answering tasks that don’t have predefined options? 
- The dataset provides detailed CoT reasoning for each question. However, the paper does not isolate how much these reasoning prompts actually improve model performance or reasoning ability. Would the LALM perform significantly worse without the CoT training data, or do the benefits mainly come from the RL fine-tuning and audio-centric data filtering?

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
The paper presents AudioMCQ, a comprehensive audio multiple-choice question dataset comprising 571k samples with two kinds of chain-of-thought annotations and investigates the prevalent zero audio-contribution phenomenon in LALMs, where models derive correct answers solely from textual information without processing audio content. To address this, the authors propose an Audio-Contribution Filtering method to partition data into 'weak' and 'strong' subsets. Based on this, they introduce effective post-training paradigms, such as Weak-to-Strong, which achieve new state-of-the-art performance on benchmarks like MMAU, MMAR, and MMSU.

### Strengths
* The paper is well written.
* The dataset collection process is very thoroughly explained. 
* The quantification of this "zero audio-contribution" scenario (e.g., 49.8% average accuracy on MMAU-test-mini with silent audio) is very relevant to the area.
* The experiment setup is complete and shows the benefits of the proposed post-training paradigms.
* The paper achieves new and significant state-of-the-art results on several diverse and challenging audio understanding benchmarks (MMAU, MMAR, MMSU).

### Weaknesses
1. There are no experiments with the CoT data collected from the dataset. These experiments would not only "further investigate the CoT reasoning capabilities" of models but actually solidify the contribution of the dataset in this end. Without this experiment, it is hard to evaluate if these CoT traces' quality.
2. The entire Audio-Contribution Filtering (ACF) method hinges on a majority vote from three specific LALMs. This choice is not justified, ablated, or analyzed for sensitivity. It is unclear how robust the resulting "weak" and "strong" data partitions are. A different set of filtering models could potentially lead to a different partition, which would call into question the generality of the training paradigms built upon them.
3. The AudioMCQ dataset was generated and filtered using a single powerful LLM (Qwen3-235B), and the experiments are conducted using a model from the same family (Qwen2.5-Omni). This alignment could mean the model has an inherent advantage on this specific data distribution, potentially inflating the reported performance gains. I would be interesting to see the benefit of this training data and splits on a different base model.

### Questions
1. Regarding Section 2.4, you use Qwen3-235B to automatically evaluate samples. Did you consider using a hold-out evaluator from a completely different model family (e.g., GPT-4, Gemini, Claude) to validate this filtering process? Using the same model for generation and evaluation risks introducing a significant confirmation bias.
2. Your ACF method creates a binary weak/strong partition. Audio contribution, however, likely exists on a spectrum. Could you comment on the limitations of this hard binary split? Have you considered an alternative approach, such as a "soft" assignment where samples are weighted by their predicted audio-contribution strength during training?
3. You provide CoT annotations to "support CoT research" but do not use them for training. Do you have a hypothesis on how CoT fine-tuning would interact with your weak-to-strong paradigm? For instance, would you expect CoT to be more beneficial when applied to the 'strong' audio-contribution data, where complex, multi-step reasoning about audio events is presumably required?
4. The results show that the optimal SFT data strategy (Weak-AC vs. Mixed-AC) depends on the audio-contribution characteristics of the downstream benchmark. This suggests the primary benefit might be a distribution matching effect (i.e., aligning the SFT data distribution with the test distribution). How would you distinguish this effect from the more fundamental claim that your paradigm teaches the model a generalizable, robust reasoning process? Is it more about domain adaptation than creating a more capable model overall?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the post-training strategies for training large audio language models. The paper explores multistage post-training strategies such as RL followed by SFT, instead of single-stage post-training. The paper uses the multi-stage post-training to improve the zero audio-contribution phenomenon in LALMs, where models derive correct answers solely from textual information without processing audio content. The paper proposes a comprehensive MCQ dataset to facilitate large-scale post-training.

### Strengths
The paper is well-written and explores important yet underexplored multistage post-training strategies. The proposed dataset is comprehensive and focuses on designing MCQ that require the LALM to pay attention to audio to answer correctly. The paper achieves strong performance across several benchmarks, such as MMAU and MMAR.

### Weaknesses
The synthetic data generation, followed by post training, has been used before to improve performance of prior models such as R1-Qwen and R1-Omni.

The performance improvement across the two benchmarks is very small; for example, on the MMAU test and test-mini benchmark, the performance is just 0.2, and on MMAR is <2 (Table 5), despite using significantly more data for post-training than the existing methods, such as Omni-R1.

### Questions
The experiments for the two proposed post-training paradigms, Weak-to-Strong and Mixed-to-Strong, are poorly designed. Why is the number of training steps across the three training experiments different? 

Line 442: “Mixed-to-Mixed approach (1000 steps SFT + 400 steps GRPO with mixed audio contribution data)”
“Weak-to-Strong paradigm (750 steps weak audio-contribution SFT + 100 steps strong audio-contribution GRPO)” 
“Mixed-to-Strong approach (1000 steps mixed audio-contribution SFT + 900 steps strong audio-contribution GRPO)” 
This makes drawing conclusions about the post-training strategies difficult. Is the “Mixed-to-Strong” strategy better than the “Mixed-to-Mixed” strategy because of how data is used, or is it simply due to the larger number of training steps used for the “Mixed-to-Strong” strategy?

### Soundness
2

### Presentation
3

### Contribution
2
