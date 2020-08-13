for((fold=0;fold<5;fold++));
do
echo 'START FOLD_'$fold'...'
python ../predict.py \
--ex_index=1 \
--fold_id=$fold \
--mode=test
echo 'FOLD_'$fold 'done'
done