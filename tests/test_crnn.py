from bioplnn.models.ei_crnn import Conv2dEIRNNCell


def test_cell():
    import hydra

    # Load config
    with hydra.initialize(config_path="../config"):
        cfg = hydra.compose(
            config_name="config", overrides=["model=ei1l", "data=mnist"]
        )
    model = Conv2dEIRNNCell(**cfg.model)
    print(model)


if __name__ == "__main__":
    test_cell()
