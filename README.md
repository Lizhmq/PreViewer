# PreViewer: Pre-trained CodeReviewer

We pre-train a CodeReviewer model to automate three code reivew related tasks: **Diff quality estimation**, **Review generation**, **Code refinement**.

![](ex.png)

CodeReviewer is initialized with [CodeT5](https://github.com/salesforce/CodeT5) and further trained on four designed pre-training tasks:

<img src="task.png" width="90%" class="center">

Refer to our paper: [CodeReviewer: Pre-Training for Automating Code Review Activities
](https://arxiv.org/abs/2203.09095) for more details.

## Training and Evaluation

Scripts are listed in `code/sh` directory.
```bash
# Example:
cd code/sh
bash finetune-cls.sh
```

## Infer a Json file
```bash
cd code/sh
bash infer-json.sh
```

## Data preparation
Collecting GitHub repos: [TopRepos](https://github.com/Lizhmq/TopRepos).

Build pull request database: [ETCR](https://github.com/Lizhmq/etcr-infrastructure).

Build pre-training and benchmark datasets: [CodeReviewProcesser](https://github.com/Lizhmq/CodeReviewProcesser).
