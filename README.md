# Cross-lingual AMR aligner: Paying Attention to Cross-Attention

This is the repo for [*Cross-lingual AMR Aligner*](https://arxiv.org/abs/2206.07587), a novel aligner for Abstract Meaning Representation (AMR) graphs that can scale cross-lingually, presented at ACL 2023. 

## Features

This repository presents an extended version of the SPRING model specifically designed for cross-lingual alignment of Abstract Meaning Representation (AMR) graphs. Our enhanced aligner incorporates two key features:

1. **Cross-Attention Aligner**:  We introduce a novel module that leverages the Cross-Attention mechanism to extract alignment information.

2. **Guided Attention model**:  We augment the training phase with an extra loss-signal based on the alignment information, improving the model's ability to capture alignment patterns in the Cross-Attention.

If you use the code, please reference this work in your paper:

```
@inproceedings{martinez-etal-2023-amr-aligner,
    title = {{C}ross-lingual {AMR} {A}ligner: {P}aying {A}ttention to {C}ross-{A}ttention},
    author = {Martínez Lorenzo, Abelardo Carlos and Huguet Cabot, Pere-Lluís and Navigli, Roberto},
    booktitle = {Findings of ACL},
    year = {2023}
}
```

## Contents

The repository is organized as follows:

- `bin/`: Contains the source code for training  the SPRING model, conducting inference, and extracting perplexity.
- `config/`: Contains configuration files for training the SPRING model.
- `data/`: should contain the training, validation, and test corpora, along with the necessary vocab of SPRING.
- `spring_amr/`: Contains the SPRING model code.
- `docs/`: Documentation and supplementary material related to the project.

## Pretrained Checkpoints

### Text-to-AMR Parsing
- Model (unguided) trained in the AMR 3.0 training set: [AMR3.parsing-1.0.tar.bz2](http://nlp.uniroma1.it/AMR/AMR3.parsing-1.0.tar.bz2)

If you need the checkpoints of other experiments in the paper, please send us an email.

## Installation
```shell script
cd amr-alignment
pip install -r requirements.txt
pip install -e .
```

The code only works with `transformers` < 3.0 because of a disrupting change in positional embeddings.
The code works fine with `torch` 1.5. We recommend the usage of a new `conda` env.

## Extract alignments

To extract the aligments from the cross-attention of a AMR parsing SPRING-like model (either guided or unguided), you can use the following command:

```shell script
python bin/extract_alignment.py --checkpoint <path_to_checkpoint> --amr-path <path_to_amr_file>
```

This will generate multiple files in LeAMR and ISI formats. 

To evaluate them, please refer to the evaluation code from [LeAMR](https://github.com/ablodge/leamr).

### License
This project is released under the CC-BY-NC-SA 4.0 license (see `LICENSE`). If you use Cross-lingual AMR Aligner, please reference the paper and put a link to this repo.

## Contributing

We welcome contributions to the Cross-lingual AMR Aligner project. If you have any ideas, bug fixes, or improvements, feel free to open an issue or submit a pull request.

## Contact

For any questions or inquiries, please contact Abelardo Carlos Martínez Lorenzo at martinez@di.uniroma1.it or Pere-Lluís Huguet Cabot at huguetcabot@babelscape.com
