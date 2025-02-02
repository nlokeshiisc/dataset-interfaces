import constants as constants

config = {
    constants.GENERAL_SPECS: {
        constants.SHIFT: constants.BG_SHIFTS,
    },
    constants.DATASET_SPECS: {
        constants.DATASET_NAME: constants.IMSTAR,
        constants.SEL_SYSNETS: [
            "n01514668",
            "n01560419",
            "n01630670",
            "n01644373",
            "n01697457",
            "n01734418",
            "n01770393",
            "n01806143",
            "n01882714",
            "n01944390",
        ],
    },
    constants.REC_SPECS: {
        constants.MODEL_NAME: "fct",
        constants.KWARGS: {
            constants.IMGEMBDIM: 256,
            constants.EMBDIM: 64,
            constants.INPUT: [constants.SSTAR, constants.BETA],
            constants.EPOCHS: 20,
        },
    },
    constants.TRAIN_ARGS: {constants.COMPUTE_RHO: True, constants.REC: False},
    constants.SEED: 0,
    constants.GPUID: 1,
    constants.EXPTNAME: "fct",
}
