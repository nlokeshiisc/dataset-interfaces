import constants as constants

config = {
    constants.GENERAL_SPECS: {
        constants.SHIFT: constants.LIGHT_SHIFTS,
    },
    constants.DATASET_SPECS: {
        constants.DATASET_NAME: constants.IMSTAR,
        constants.SEL_CLASSES: [
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
            constants.INPUT: ["x", constants.BETA],
        },
    },
    constants.TRAIN_ARGS: {constants.COMPUTE_RHO: True, constants.REC: False},
    constants.SEED: 0,
    constants.GPUID: 5,
    constants.EXPTNAME: "fct",
}
