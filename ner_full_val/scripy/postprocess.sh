for((fold=0;fold<5;fold++));
do
echo 'START FOLD_'$fold'...'
python ../postprocess.py \
--ex_index=3 \
--num_fold=5 \
--fold_id=$fold \
--mode=test \
--num_samples=100 \
--threshold=1
echo 'FOLD_'$fold 'done'
done