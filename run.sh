python train.py --data_path={data_path} --features_path={feature_path} --split_type=predetermined --dataset_type=regression --epochs=100 --num_folds=1 --batch_size=16 --folds_file=./data/HC10_use_2/CV_splits/Seed5/stratified_split_indices_cv0_train.pckl --test_fold_index=1

python predict.py --data_path={data_path} --features_path={feature_path} --checkpoint_dir={checkpoint_path}
