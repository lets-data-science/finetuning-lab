# Licensing

## Original project code

The original Let's Data Science source code, configuration, tests and repository
documentation in Fine-Tuning Lab are licensed under the **Apache License, Version 2.0**.
See [LICENSE](LICENSE) for the complete terms and [NOTICE](NOTICE) for attribution.
The SPDX identifier is `Apache-2.0`.

You can use, modify and redistribute the covered code, including commercially,
subject to those terms. This notice does not add conditions to Apache 2.0.

## Dependencies and external services

Third-party components retain their own licenses. Installing a dependency or
calling a service does not relicense it under this repository's license.

## Earlier MIT grants and model artifacts

LDS code distributed through commit `679bf8719081d588d05ddc78ba49f2ec7fc82c93` was offered under MIT.
Those existing permissions are not withdrawn. The original notice is retained
in [LICENSES/LDS-MIT.txt](LICENSES/LDS-MIT.txt). New LDS code is offered under
Apache 2.0 unless a file states otherwise.

The already published model weights, checkpoints, tokenizer files, generated
evaluation data and recorded results retain their previous MIT terms. This
code-license change does not relicense those artifacts or any upstream dataset.

## Upstream attribution

StoryByte's PyTorch implementation follows the nanoGPT/minGPT lineage.
Third-party portions retain MIT, including its copyright and permission notices:

- [nanoGPT](https://github.com/karpathy/nanoGPT): Andrej Karpathy, 2022;
  [MIT notice](LICENSES/nanoGPT-MIT.txt).
- [minGPT](https://github.com/karpathy/minGPT): Andrej Karpathy, 2020;
  [MIT notice](LICENSES/minGPT-MIT.txt).

These notices apply to upstream portions of `storybyte/sb_common/model.py`,
the lab's StoryByte port; LDS contributions are Apache-2.0.

StoryByte was trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
by Ronen Eldan and Yuanzhi Li. That upstream dataset is separately offered under
[CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/). Follow those terms when obtaining
or redistributing dataset material; this repository does not grant Apache rights
to TinyStories.

## Branding and teaching media

Apache 2.0 does not grant trademark rights to the LDS name or logos, apart from
the descriptive uses allowed in section 6. Do not imply LDS endorsement.
Course and blog content hosted outside this repository, website recordings,
narration, voice likenesses and third-party media are not included in this
source-code license. Their existing terms remain separate.
