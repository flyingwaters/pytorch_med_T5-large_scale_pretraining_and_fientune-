# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import ast
from itertools import chain
import datasets
_CITATION = """\
@inproceedings{
    no publishing staff
}
"""

_DESCRIPTION = """\
this is a mix-ture pretrain dataset, including cord-19 pubmed pmc and Chinese_medical_papers_abstract,
aiming to pretrain PLMs and strengthen the medical text processing ability of PLMs in Chinese and English
It is so frequent that English terms and Chinese words mixed in the medical papers.For this reason, we establish
this dataset, helping the training and exploration of the NLP in medical,biological,Pharmaceutical respec
"""

_HOMEPAGE = ""

_LICENSE = "MIT License"

_URL = "/raid/zyftest/project/med_T5/bmtrain_pretrain/pretrain_test_paths"


class TruthfulQA(datasets.GeneratorBasedBuilder):
    """TruthfulQA"""
    VERSION = datasets.Version("0.0.1")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="mt5-medical-pretrain", version=VERSION,
                               description="Chinese and English medical multitasks processing"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = _URL  # dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_dir})
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        num_id = 0
        with open(filepath, "r") as f:
            pretrain_dataset_paths = [
                pretrain_dataset_path.strip() for pretrain_dataset_path in f]
        for tsv_dir_pth in pretrain_dataset_paths:
            for _, _, file_names in os.walk(tsv_dir_pth):
                for file_name in file_names:
                    tmp_pth = os.path.join(tsv_dir_pth, file_name)
                    with open(tmp_pth, "r") as reader:
                        for sentence in reader:
                            num_id += 1
                            item_id = str(num_id)
                            yield item_id, {"text": sentence}
