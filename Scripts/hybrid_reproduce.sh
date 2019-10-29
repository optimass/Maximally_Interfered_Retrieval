mem_sizes=( 1000 5000 10000 )
n_runs=10

for mem_size in "${mem_sizes[@]}"
do
    result_dir="results/max_loss${mem_size}"
    python hybrid_main.py --result_dir $result_dir --mem_strength 0.5  --n_iters 1  --n_runs $n_runs   --n_epochs 5  --batch_size 10  --log off --mem_size $mem_size  --buffer_batch_size 100    --max_loss_budget 1   --max_loss_grad_steps 1  --ent_coef 1  --kl_coef 200   --full_ab 0  --euc_coef -1 --max_loss &
    
    result_dir="results/rand${mem_size}"
    CUDA_VISIBLE_DEVICES=1 python hybrid_main.py  --result_dir $result_dir --mem_strength 0.5  --n_iters 1  --n_runs $n_runs   --n_epochs 5  --batch_size 10  --log off --mem_size $mem_size  --buffer_batch_size 100  --full_ab 0 
done
