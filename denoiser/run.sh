for alpha in 0.2
do
    # python train.py --version ${alpha}_rand_base --use-base
    python train.py --version ${alpha}_bias_base_reddit --use-base
    # python train.py --version ${alpha}_rl_base --use-base
done

for alpha in 0.2
do
    python train.py --version ${alpha}_rand_reddit
    python train.py --version ${alpha}_bias_reddit
    python train.py --version ${alpha}_rl_reddit
done