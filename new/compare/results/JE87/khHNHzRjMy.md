# Review

## Summary
This paper introduces a new dataset, EmoSign, which is designed to advance the understanding of emotions in American Sign Language (ASL) through a multimodal approach. The dataset includes 200 ASL videos annotated with sentiment and emotion labels by Deaf ASL signers, along with open-ended descriptions of emotion cues such as facial expressions, body language, and signing speed. The paper also benchmarks several multimodal models on tasks like sentiment analysis and emotion classification, revealing limitations in current models' ability to interpret visual cues independently of text.

## Soundness
3

## Presentation
4

## Contribution
3

## Strengths
1. Originality: EmoSign is the first dataset specifically designed for emotion recognition in ASL, addressing a significant gap in sign language research.
2. Quality: The dataset is meticulously annotated by Deaf ASL signers, ensuring reliable and nuanced emotion labels.
3. Clarity: The paper is well-structured, with clear explanations of the annotation process and benchmark tasks.
4. Significance: By highlighting multimodal models' limitations in interpreting visual emotional cues, this work calls for specialized visual encoders and new architectures that integrate visual and linguistic contexts more effectively.

## Weaknesses
1. The dataset includes only 200 videos, which, while quality-controlled, may limit the generalizability of the findings.
2. The paper does not provide quantitative measures of inter-annotator agreement for the emotion labels.
3. The paper could benefit from a more detailed error analysis to better understand where current models fail in emotion recognition.

## Questions
1. Could the authors provide more details on the criteria used to select the 200 videos?
2. How do the authors plan to expand the dataset in future work?
3. Would the authors consider collecting data from a more diverse set of signers to further enhance the dataset's representativeness?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4