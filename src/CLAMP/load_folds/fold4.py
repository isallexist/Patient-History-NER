import datasets
import pandas as pd

logger = datasets.logging.get_logger(__name__)

# Pick a fold to train
_FOLD = 4
_ROOT_DIR = f"./data/bio_clamp/folds/{str(_FOLD)}/"
_TRAINING_FILE = _ROOT_DIR + "train.tsv"
_DEV_FILE = _ROOT_DIR + "dev.tsv"
_TEST_FILE = _ROOT_DIR + "test.tsv"


class NerCLAMPDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for NerCLAMPDataset"""

    def __init__(self, **kwargs):
        """BuilderConfig for NerCLAMPDataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NerCLAMPDatasetConfig, self).__init__(**kwargs)


class NerCLAMPDataset(datasets.GeneratorBasedBuilder):
    """NerCLAMP dataset."""

    BUILDER_CONFIGS = [
        NerCLAMPDatasetConfig(name="NerCLAMPDataset", version=datasets.Version(
            "1.0.0"), description="NerCLAMPDataset"),
    ]

    def _info(self):
        # train_df = pd.read_csv(_TRAINING_FILE, sep="\t", names=["token", "tag", "clamp", "clamp_2"])
        # dev_df = pd.read_csv(_DEV_FILE, sep="\t", names=["token", "tag", "clamp", "clamp_2"])
        # test_df = pd.read_csv(_TEST_FILE, sep="\t", names=["token", "tag", "clamp", "clamp_2"])

        label_list = ['I-hpi.quality', 'B-cc', 'B-hpi.duration', 'I-hpi.modifyingFactors', 'I-socialHistory', 'O', 'I-pastHistory', 'I-familyHistory', 'I-hpi.context', 'B-familyHistory', 'B-hpi.timing', 'I-cc', 'I-hpi.duration', 'I-hpi.assocSignsAndSymptoms', 'B-hpi.context', 'B-hpi.quality', 'I-hpi.severity', 'B-socialHistory', 'I-hpi.location', 'B-hpi.location', 'I-hpi.timing', 'B-pastHistory', 'B-hpi.assocSignsAndSymptoms', 'B-hpi.severity', 'B-hpi.modifyingFactors']
        CLAMP_list =  ["problem", "test", "treatment", "drug"]
        CLAMP_label_list = ["O"]
        for ent in CLAMP_list:
            CLAMP_label_list.append("B-" + ent)
            CLAMP_label_list.append("I-" + ent)

        CLAMP_list_2 =  ["bodyloc", "severity", "temporal"]
        CLAMP_label_list_2 = ["O"]
        # CLAMP_label_list_2 = []

        for ent in CLAMP_list_2:
            CLAMP_label_list_2.append("B-" + ent)
            CLAMP_label_list_2.append("I-" + ent)

        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=label_list
                        )
                    ),
                    "CLAMP_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=CLAMP_label_list
                        )
                    ),
                    "CLAMP_tags_2": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=CLAMP_label_list_2
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                    "filepath": _TRAINING_FILE}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                                    "filepath": _DEV_FILE}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                                    "filepath": _TEST_FILE}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            CLAMP_tags = []
            CLAMP_tags_2 = []
            for line in f:

                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                            "CLAMP_tags": CLAMP_tags,
                            "CLAMP_tags_2": CLAMP_tags_2,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                        CLAMP_tags = []
                        CLAMP_tags_2 = []
                else:
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
                    CLAMP_tags.append(splits[2].rstrip())
                    CLAMP_tags_2.append(splits[3].rstrip())
            # # last example is empty
            # yield guid, {
            #     "id": str(guid),
            #     "tokens": tokens,
            #     "ner_tags": ner_tags,
            #     "sentiment": sentiment,
            # }
