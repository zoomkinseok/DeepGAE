import os
data_set = 'HCPrsFC'

## Hyper parameter
hidden_dim = [90]
num_layer = [5]
learning_rate = [0.001]
epoch = 300

## setting
sparsity_list = [30]
nonneg = 'False'

no = 10000000
for sparsity in sparsity_list:
    for lr in learning_rate: ## learning rate 0.001
        for numlayer in num_layer: ## numlayer
            for hidden in hidden_dim: ## hidden_dim 8,16,32,64

                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, '%s__sparsity:%d__hidden_dim:%d__num_layer:%d__nonneg:%s__learning_rate:%f'% (
                    no, sparsity, hidden, numlayer, nonneg,lr))

                for fold_idx in range(1): # fold index 1-10
                    os.system('python3 main.py --no %d --dataset %s --sparsity %d --seedforw %d --epochs %d --nonneg %s --fold_idx %d --lr %f --iters_per_epoch %d --num_layers %d --hidden_dim %d' % (
                                no, data_set, sparsity, 6, epoch, nonneg, fold_idx, lr, 50, numlayer, hidden))

                no = no + 1



