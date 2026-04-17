# Review

## Summary
The paper presents a method for generating time series data by converting the time series into spectrograms and then using a video generative model to produce new spectrograms, which are finally inverted back into time series. The method is compared to several existing time series generation methods on a variety of datasets and metrics, where it demonstrates favorable performance.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
The method is simple and appears to work well.

## Weaknesses
The main weakness of the paper is the lack of novelty. The idea of converting time series into images or videos has been thoroughly explored in several papers, including the cited ImagenTime paper. This paper differs from ImagenTime in that it uses a video model instead of an image model, but the idea of converting time series into videos has also been explored in at least one previous paper (https://arxiv.org/abs/2407.00287). The paper does a good job of implementing this idea and exploring various aspects of it, but the idea itself is not novel.

## Questions
The paper says that it "applies attention sequentially along the three main temporal, frequency, and covariate axes". Does this mean that the attention is applied independently to each axis, or does it mean that the attention is applied jointly to all three axes? The latter option would be more in line with standard spatiotemporal attention, but it's not clear how that would be implemented. The former option would mean that the attention is applied separately to each axis, which would be more similar to the approach in ImagenTime.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4