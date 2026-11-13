"""Installed subprocess entry: ordinary experiments with HPO accounting enabled."""

if __name__ == "__main__":
    from rl_matdesign.utils.hpo_stop import cooperative_worker_stop

    # Install before importing Torch/predictor modules: an early stop request
    # must not kill setup or the first paid batch halfway through its work.
    with cooperative_worker_stop():
        from rl_matdesign.experiment import main
        main()
