import datasets
import pandas as pd

logger = datasets.logging.get_logger(__name__)

_FOLD = 2
_DATA_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/data/folds/" + str(_FOLD)
_TRAINING_FILE =  _DATA_DIR  + "/train.tsv"
_DEV_FILE = _DATA_DIR + "/dev.tsv"
_TEST_FILE = _DATA_DIR + "/test.tsv"
# _DATA_FILE = "/home/exx/Documents/dzungle/MB_BERT/data/suhao_ver1/ros/all.txt"

class MBDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for MBDataset"""

    def __init__(self, **kwargs):
        """BuilderConfig forMBDataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MBDatasetConfig, self).__init__(**kwargs)


class MBDataset(datasets.GeneratorBasedBuilder):
    """MBDataset dataset."""

    BUILDER_CONFIGS = [
        MBDatasetConfig(name="MBDataset", version=datasets.Version(
            "1.0.0"), description="MBDataset dataset"),
    ]
    
    def _info(self):
        train_df = pd.read_csv(_TRAINING_FILE, sep="\t", names=["token", "tag"])
        dev_df = pd.read_csv(_DEV_FILE, sep="\t", names=["token", "tag"])
        test_df = pd.read_csv(_TEST_FILE, sep="\t", names=["token", "tag"])
        label_list = sorted(list(set(list(train_df["tag"].unique()) \
            + list(dev_df["tag"].unique()) + list(test_df["tag"].unique()))))

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
            # datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
            #                         "filepath": _DATA_FILE}),
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
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # MBDataset tokens are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # # last example is empty
            # yield guid, {
            #     "id": str(guid),
            #     "tokens": tokens,
            #     "ner_tags": ner_tags,
            # }
