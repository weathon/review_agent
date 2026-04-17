# Towards an AI Musician: Synthesizing Sheet Music Problems for Musical Reasoning

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Enhancing the ability of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) to interpret sheet music is a crucial step toward building AI musicians. However, current research lacks both evaluation benchmarks and training data for sheet music reasoning. Drawing from mathematics, where simple operations can generate a boundless universe of verifiable problems, we introduce a novel approach that treats core music theory rules, such as those governing beats and intervals, as programmatic functions to systematically synthesize a vast and diverse corpus of sheet music reasoning problems. This approach allows us to introduce a data synthesis framework that generates verifiable sheet music questions in both textual and visual modalities, leading to the Synthetic Sheet Music Reasoning Benchmark (SSMR-Bench) and a complementary training set. Evaluation results on SSMR-Bench highlight the key role reasoning plays in interpreting sheet music, while also pointing out the ongoing challenges in understanding sheet music in a visual format. By leveraging synthetic data for RLVR, models like Qwen3-8B-Base and Qwen2.5-VL-Instruct show significant improvements on the SSMR-Bench. Additionally, they also demonstrate considerable advancements on previously established human-crafted benchmarks, such as MusicTheoryBench and the music subset of MMMU. Finally, our results show that the enhanced reasoning ability can also facilitate music composition.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a benchmark for reasoning on synthetic music sheets, leveraging synthetic data created through reinforcement learning. This is combined with experiments across LLMs and MLLMs, also showing how models also have abilities beyond reasoning in music composition.

### Strengths
* The problem of reasoning on music sheets is relatively unexplored, and the contribution of leveraging music theory rules to synthesize sheet music is sufficiently novel.
* Experimental results across various LLMs and MLLMs are well constructed and cover textual QA as well as visual QA plus benchmarks on real-world data.

### Weaknesses
* Some assumptions made in this work are not clarified in the paper, and some design choices are not presented with sufficient clarity - see the below section for details.
* The resulting synthetic data are still fairly simple from a musical point of view and might not scale to complex compositions, or to noisy and degraded images of scores or even handwritten notation.
* The manuscript includes very few critical remarks arguing on the strengths and weaknesses of the proposed benchmark and offers no clear avenues for future work, therefore limiting the scholarly quality of the paper.

### Questions
* Section 1 and 2.1 make several claims that "sheet music is the universal language" which are not correct from a musical or musicological point of view. This work uses a particular form of music notation (namely Western staff notation) but not all music cultures use this type of notation, and not all music cultures use music notation in the first place. Even some styles within Western music use alternative forms of notation, e.g. guitar tabs. Therefore the scope of this work needs to be revised accordingly and certain assumptions made regarding the data (and theory) domain would need to be clarified.
* Some aspects of this work involve converting ABC notation to images. This conversion process can have significant effects on performance for visual QA tasks, however there is no information provided on which notation-to-image engraving method was used, and there appears to not have been any work on using different engraving methods as to improve the diversity of the image data.
* From Figure 4 the composition of the test set appears to cover questions related to Chords, Scale, Intervals, and Rhythm. One question might be: are these 4 categories sufficient to claim understanding and reasoning for sheet music? Which musical aspects might be missing from these questions which could form a new benchmark?
* Table 2: what is the "Thinking" column?
* In section 4.1, there are comparisons with real-world benchmarks. For MMMU Music and the Visual QA task, it appears that the table only shows results using the proposed method. How is performance when compared with reported results and other methods using the MMMU benchmark and specifically its Visual QA task?
* Section 6 only provides a limited summary of the paper. Instead this could have provided some critical remarks on this work, emphasizing on the strengths but also the areas of possible improvement for this work, combined with directions of future work. Currently there is limited critical material on this paper which limits its scholarly contribution.

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
4

### Summary
This paper introduces a new benchmark that generates sheet-music reasoning problems in text and image form based on core music-theory rules (N=1600 + 8000 each). The main application of this benchmark is to train LLM/VLM to reason about sheet music. The paper shows that using this synthetic data for Reinforcement Learning with Verifiable Rewards improves LLMs performance on the given benchmark, as well as on two existing benchmarks, MusicTheoryBench and MMMU.

### Strengths
Programmatic pipeline for synthetic data generation allows a principled generation of a large number of problems, allowing to generate a much larger benchmark compared to existing ones.

The benchmark is novel, and includes diverse question types.

Beyond music, the pattern “encode domain rules -> synthesize verifiable tasks -> RLVR” is general and timely for reasoning-centric training

 The expert-templated, rules-driven generator produces auto-graded, controllable items (bar placement, interval/chord/scale families), and deterministically renders images. This is ideal for RLVR where reward correctness matters.

A wide a range of state-of-the-art LLMs and MLLMs are included in the evaluation

### Weaknesses
Presentation clarity. 

Dataset only focuses on 4 music theory domains (scale, rhythm, interval, chord). The questions are focused on music theory rule application, wheres for a reasoning benchmark it would be good to consider tasks such as inferring key modulations or harmonic function given a progression, which involve integrating symbolic, structural, and contextual information. 

Even a small study comparing how the best-performing fine-tuned models perform compared to humans would be useful.

### Questions
Are there existing non-LLM based models that can solve these reasoning problems?

How was the specific given subset of reasoning questions chosen to be included in the benchmark?

### Soundness
3

### Presentation
2

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
This paper proposes a synthetic data generation framework that converts core music theory rules (rhythm, interval, chord, scale) into programmable functions for producing verifiable sheet-music reasoning problems. The authors introduce SSMR-Bench, consisting of 3,200 evaluation QA pairs (text + image) and a larger 16K training set for reinforcement learning with verifiable rewards (RLVR). They show that training on synthetic musical reasoning data improves performance on SSMR-Bench, and also transfers to existing human-crafted benchmarks such as MusicTheoryBench and MMMU-Music. The paper further claims that the learned reasoning ability benefits symbolic music composition.

### Strengths
Novel data synthesis idea.
Treating music theory as a rule-based generator of reasoning problems is conceptually parallel to math problem synthesis, and enables scalable, verifiable QA generation.

Benchmark + training pipeline.
The paper contributes both an evaluation benchmark and a synthetic training dataset, filling a gap for sheet-music reasoning tasks.

Demonstrated improvements on multiple models.
RLVR-trained Qwen models show clear gains on SSMR-Bench and moderate gains on MusicTheoryBench and MMMU-Music.

Cross-domain effect is interesting.
The observation that music reasoning training may improve math reasoning hints at shared symbolic reasoning structure.

### Weaknesses
Music theory coverage is limited.
The synthesis only covers local rules (rhythm, interval, chord, scale) and does not address global musical structure, harmonic progression, voice-leading, or form—so the claim of “full sheet music reasoning” is overstated.

Performance still far from SOTA baselines.
Although training improves scores, the models still lag well behind state-of-the-art on MusicTheoryBench, suggesting limited real-world transfer.

MMMU results close to random guessing.
On MMMU-Music, accuracy remains near the 25% random baseline, indicating the model may mainly learn QA format rather than visual score reasoning.

Composition evaluation is narrow.
The “improved music composition” result only tests 4-bar rhythmic alignment with prompt tempo, ignoring key musical elements such as harmonic coherence, chord progression, or longer-term structure.

Minor but NOTABLE typo.
“GRPO” is repeatedly misspelled as GPRO in Lines 208–216.

### Questions
Music theory scope
Do you plan to extend synthesis to higher-level rules (e.g., cadences, phrase structure, functional harmony, chord progression)? If not, how should readers interpret the term sheet music reasoning?

Math transfer claim
In Sec. 4.2, you suggest music reasoning helps math reasoning. Can you test whether music+math mixed training outperforms “math-only extra epochs”?

Benchmark gap
Why do models still underperform SOTA on MusicTheoryBench despite large synthetic training? Is the gap due to missing global theory features?

MMMU near-random accuracy
Can you provide an ablation to show whether the model is learning symbolic reasoning or just answer-pattern priors?

Composition evaluation
Would you consider adding harmonic correctness or chord adherence metrics, instead of only rhythm-based alignment?

Annotation correctness
Are the synthetic questions automatically validated? If so, what is the estimated error rate of the generation pipeline?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a programmatic framework for generating verifiable sheet-music QA paris in text and image modalities. The authors benchmark multiple LLMs and VLMs (zero-shot and fine-tuned with GRPO) on 4 types of questions and music composition.

### Strengths
- The paper provides a clean pipeline to create paired text/image questions, which makes evaluation more controllable and reproducible than LLM-written questions.
- The benchmark is balanced across several categories (rhythm, scales, intervals, basic chords) and provide evaluations on text and visual modalities.
- The author further explore if fine-tune with their music sheet qa dataset can improve music composition.

### Weaknesses
- The main contribution is the data synthesis framework. But the authors merely scratch the surface in section 2.2 and lots of core details are missing. First of all, does the framework generate music or just the questions for real music sheets with ground truth? Base on my reading on the paper and figure 2, it only generates questions for annotated music data with ground truth (ABC notation). While the authors highlight their framework is built on music theory, there is only one example and it is not clear what music theory are used. In fact, it is simply removing the answers (e.g., bar) in the music sheets based on what the question is asking. I don't think it is appropriate to claim this framework or synthetic data to be “boundless” or “endless”.
- Although the paper claims to address musical reasoning, the current questions do not appear to require genuine reasoning. According to Appendix A, all questions are recognition/identification of local features (e.g., scale, chord, interval). That fits “sheet-music understanding,” but it is not reasoning in the stronger sense as in Yue et al. 2024.
- The empirical results are unsurprising and offer limited insight. (1) Reading ABC is easier than OMR, expected. (2) A reasoning-tuned model (e.g., DeepSeek-R1) outperforming a large general LLM (e.g., DeepSeek-V3) is also expected. (3) Fine-tuning on music-sheet QA improves in-distribution performance, again expected. More importantly, despite introducing four task categories, the paper provides little analysis about which models work better for which tasks (and why), nor does it disentangle recognition failures from theory/logic errors.

### Questions
1. What is thinking in table 2 and table 3? Do you mean reasoning models? Explain what it means in the paper.
2. In Section 4.2 the numbers in the main text is different with the numbers in figure 5, which one is correct?
3. What is the meaning of Section 4.2, showing that the music sheet data can improve reasoning in Math? What is the intuition behind and why is it relevant to the paper?

### Soundness
2

### Presentation
2

### Contribution
2
