for((fold=0;fold<5;fold++));
do
echo 'START FOLD_'$fold'...'
python ../train.py \
--ex_index=1 \
--fold_id=$fold \
--epoch_num=10
echo 'FOLD_'$fold 'done'
done