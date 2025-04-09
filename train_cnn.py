from opensoundscape.annotations import BoxedAnnotations
import os
import pandas as pd
import numpy as np
import torch
import random
import wandb
from opensoundscape import CNN, SpectrogramPreprocessor
from opensoundscape.data_selection import resample
from opensoundscape.preprocess.utils import show_tensor_grid
from opensoundscape import AudioFileDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf  # TF needed to import pickle
import gc


def create_train_valid_set(experiment_name, samples_dir, fish_sound, fish_sound_folder, window_len_s):
    train_set_dir = os.path.join(samples_dir, experiment_name)

    pos_samples_file = os.listdir(os.path.join(train_set_dir, fish_sound_folder))
    neg_samples_file = os.listdir(os.path.join(train_set_dir, 'Unknown'))

    pos_samples_file = [os.path.join(train_set_dir, fish_sound_folder, file) for file in pos_samples_file]
    neg_samples_file = [os.path.join(train_set_dir, 'Unknown', file) for file in neg_samples_file]

    df_pos = pd.DataFrame()
    df_neg = pd.DataFrame()

    # Add positive sample file in the dataframe in a column file and create a column label with 1
    df_pos['file'] = pos_samples_file
    df_pos['start_time'] = 0.0
    df_pos['end_time'] = window_len_s
    df_pos[fish_sound] = 1

    # Add negative sample file in the dataframe in a column file and create a column label with 0
    df_neg['file'] = neg_samples_file
    df_neg['start_time'] = 0.0
    df_neg['end_time'] = window_len_s
    df_neg[fish_sound] = 0

    # concatenate the two dataframe
    df_trainset = pd.concat([df_pos, df_neg], ignore_index=True)
    df_trainset[fish_sound].value_counts()

    # Save the df as csv in the current repo
    df_trainset.to_csv('train_set_baseline.csv', index=False)
    print('train set saved in train_set_baseline.csv')

    df_trainset = pd.read_csv('train_set_baseline.csv', index_col=[0, 1, 2])
    return df_trainset


def create_test_set_with_opensoundscape(test_files_path, fish_sound):
    test_files_list = os.listdir(test_files_path)

    # Create a list of all wav files with files ending by .wav
    wav_files = sorted([file for file in test_files_list if file.endswith('.wav')])
    annot_files = sorted([file for file in test_files_list if file.endswith('.txt')])

    wav_files = [os.path.join(test_files_path, file) for file in wav_files]
    annot_files = [os.path.join(test_files_path, file) for file in annot_files]

    print("Checking files order\n", wav_files[:3])
    print(annot_files[:3])

    selection_files = annot_files
    audio_files = wav_files

    annotations = BoxedAnnotations.from_raven_files(raven_files=selection_files, audio_files=audio_files,
                                                    annotation_column='Type')

    clip_duration = 5.0
    clip_overlap = 0
    min_label_overlap = 0.2
    species_of_interest = [fish_sound]

    clip_labels = annotations.clip_labels(
        clip_duration=clip_duration,
        clip_overlap=clip_overlap,
        min_label_overlap=min_label_overlap,
        class_subset=species_of_interest)

    return clip_labels


def load_test_set(test_set_path, test_files_path, fish_sound):
    # tensorflow is needed to import the pickle because it contains tensors
    test_set_df = pd.read_pickle(os.path.join(test_set_path, 'test_set.pkl'))

    # Modify the format of the pickle to be compatible with opensoundscape
    test_set_df[fish_sound] = (test_set_df['Label'] == fish_sound)
    test_set_df = test_set_df.drop(columns=['Label', 'Embedding'])
    test_set_df = test_set_df.rename(columns={'Starttime': 'start_time', 'Endtime': 'end_time', 'filename': 'file'})
    test_set_df['file'] = test_files_path + test_set_df['file']
    test_set_df = test_set_df.set_index(['file', 'start_time', 'end_time'])

    return test_set_df


def setup_model(model, sample_rate, labels):
    # all data augmentations

    preprocessor = SpectrogramPreprocessor(2.0, overlay_df=labels)

    # preprocessor.pipeline.random_trim_audio.bypass = True

    preprocessor.pipeline.overlay.bypass = False

    # preprocessor.pipeline.time_mask.bypass = True
    preprocessor.pipeline.time_mask.set(max_masks=2, max_width=0.1)

    # preprocessor.pipeline.frequency_mask.bypass = True
    preprocessor.pipeline.frequency_mask.set(max_masks=2, max_width=0.1)

    # preprocessor.pipeline.add_noise.bypass = True
    # preprocessor.pipeline.random_affine.bypass = True

    preprocessor.pipeline.bandpass.set(min_f=50, max_f=2000)

    preprocessor.pipeline.to_spec.set(window_samples=4 * (sample_rate // 100),
                                            # overlap_samples = None,
                                            # fft_size = None,
                                            # dB_scale = True,
                                            # scaling = 'spectrum'
                                            )

    model.preprocessor = preprocessor
    print(model.preprocessor.pipeline)


def compute_metrics(y_true, y_pred, fish_sound):
    # Extract target values
    true_labels = y_true[fish_sound].values
    pred_probs = y_pred[fish_sound].values
    pred_binary = pred_probs.round()  # Threshold = 0.5

    # Calculate metrics
    precision = precision_score(true_labels, pred_binary, pos_label=1, average='binary')
    recall = recall_score(true_labels, pred_binary, pos_label=1, average='binary')
    f1 = f1_score(true_labels, pred_binary, pos_label=1, average='binary')
    auc_roc = roc_auc_score(true_labels, pred_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, pred_probs)
    auc_pr = auc(recall_curve, precision_curve)

    # Return metrics as a dictionary for better access
    return precision, recall, f1, auc_roc, auc_pr


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Hard coded params for now - If properly given here, the rest of the code should run
    # TODO get them from the dict config eventually
    fish_sound = 'Jackhammer'
    fish_sound_folder = 'Jackhammer'
    window_len_s = 5.0
    sample_rate = 32000
    experiment_name = 'jackhammer_surfperch_hockey_11_20250402'
    samples_dir = '/mnt/fscompute_shared/agile_modeling/experiments_paper/output/grafton_deployment/surfperch/labeled_outputs/'
    testset_files_dir = '/mnt/fscompute_shared/agile_modeling/experiments_paper/dataset/grafton_deployment/test_set/'
    testset_pickle_dir = '/mnt/fscompute_shared/agile_modeling/experiments_paper/output/grafton_deployment/surfperch/test_set/'

    # WANDB
    activate_wandb = False
    wandb_exp_name = 'No augment no upsampling 030425'

    # RUN PARAMS
    nbr_epochs = 50
    batch_size = 8
    num_workers = 8

    if activate_wandb:
        try:
            wandb.login()
            wandb_session = wandb.init(
                entity="revo2023",
                project="Agile Modeling and Opensoundscape",
                name=wandb_exp_name,
            )
        except:
            print("failed to create wandb session. wandb session will be None")
            wandb_session = None
    else:
        wandb_session = None

    # NOT USED anymore - Create test set with opensoundscape - can be used to compare perf of both test set
    # test_set_df = create_test_set_with_opensoundscape(testset_files_dir, fish_sound)

    # Load test set
    test_set_df = load_test_set(testset_pickle_dir, testset_files_dir, fish_sound)

    # Create the 5-fold training and validation sets
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    df_trainset = create_train_valid_set(experiment_name, samples_dir, fish_sound, fish_sound_folder, window_len_s)
    validation_metrics_dic = {}
    testset_metrics_dic = {}

    # Iterate over the folds
    for i, (train_index, valid_index) in enumerate(kf.split(df_trainset, df_trainset[fish_sound])):
        print("\n\n----------------------------------------")
        print(f"Running experiment for Fold {i + 1}/5")
        print("----------------------------------------")

        train_df, valid_df = df_trainset.iloc[train_index], df_trainset.iloc[valid_index]

        # upsample (repeat samples) so that all classes have 200 samples
        # augmented_train_df = resample(train_df, n_samples_per_class=200, n_samples_without_labels=200, random_state=0)

        # We use Resnet as most common architeture used in Bioacoustics (ref: Stowell 2022)
        # Resnet 18 because how dataset is small (avoid overfitting)
        architecture = 'resnet18'
        class_list = [fish_sound]
        model = CNN(architecture=architecture,
                    classes=class_list,
                    sample_duration=5.0)
        # print(f"model.device is {model.device}")

        setup_model(model, sample_rate, labels=train_df.sample(10))

        # TODO - Fix overlay
        # neg_train_df = train_df[train_df[fish_sound] == 0]
        # neg_train_df[fish_sound] = 1
        # model.preprocessor.pipeline.overlay.overlay_df = (neg_train_df.astype(int))
        # model.preprocessor.pipeline.overlay.set(overlay_class=fish_sound)

        # Display samples
        # dataset = AudioFileDataset(train_df, model.preprocessor)
        # tensors = [dataset[i].data for i in range(9)]
        # sample_labels = [list(dataset[i].labels[dataset[i].labels > 0].index) for i in range(9)]
        # _ = show_tensor_grid(tensors, 3, labels=sample_labels)

        checkpoint_folder = Path("model_training_checkpoints")
        checkpoint_folder.mkdir(exist_ok=True)

        model.train(
            train_df,
            valid_df,
            epochs=nbr_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            wandb_session=wandb_session,
            progress_bar=True,
            save_interval=5,  # save checkpoint every 10 epochs
            save_path=checkpoint_folder,  # location to save checkpoints
        )

        # Load the best model - I think otherwise the model at the end of the training is used, which likely overfit data
        model.load("./model_training_checkpoints/best.model")

        # Run predictions on the validation set and on the test set
        predict_validset = model.predict(valid_df,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         activation_layer='sigmoid',
                                         wandb_session=wandb_session,
                                         progress_bar=False)

        pred_testset = model.predict(test_set_df,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     activation_layer='sigmoid',
                                     wandb_session=wandb_session)

        # Compute metrics
        prec_valid, recall_valid, f1_valid, auc_roc_valid, auc_pr_valid = compute_metrics(valid_df, predict_validset,
                                                                                          fish_sound)
        prec_test, recall_test, f1_test, auc_roc_test, auc_pr_test = compute_metrics(test_set_df, pred_testset,
                                                                                     fish_sound)

        # Save metrics of the fold iteration
        validation_metrics_dic[i] = [prec_valid, recall_valid, f1_valid, auc_roc_valid, auc_pr_valid]
        testset_metrics_dic[i] = [prec_test, recall_test, f1_test, auc_roc_test, auc_pr_test]

        # Clean cuda memory to avoid OOM
        model.to('cpu')
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Compute average of all metrics across all folds
    prec_valid = np.mean([metrics[0] for metrics in validation_metrics_dic.values()])
    recall_valid = np.mean([metrics[1] for metrics in validation_metrics_dic.values()])
    f1_valid = np.mean([metrics[2] for metrics in validation_metrics_dic.values()])
    auc_roc_valid = np.mean([metrics[3] for metrics in validation_metrics_dic.values()])
    auc_pr_valid = np.mean([metrics[4] for metrics in validation_metrics_dic.values()])

    prec_test = np.mean([metrics[0] for metrics in testset_metrics_dic.values()])
    recall_test = np.mean([metrics[1] for metrics in testset_metrics_dic.values()])
    f1_test = np.mean([metrics[2] for metrics in testset_metrics_dic.values()])
    auc_roc_test = np.mean([metrics[3] for metrics in testset_metrics_dic.values()])
    auc_pr_test = np.mean([metrics[4] for metrics in testset_metrics_dic.values()])

    # print all average metrics
    print("Validation Set")
    print("Precision valid: ", prec_valid)
    print("Recall valid: ", recall_valid)
    print("F1 valid: ", f1_valid)
    print("AUC ROC valid: ", auc_roc_valid)
    print("AUC precision recall: ", auc_pr_valid)

    print("\nTest Set")
    print("Precision test: ", prec_test)
    print("Recall test: ", recall_test)
    print("F1 test: ", f1_test)
    print("AUC ROC test: ", auc_roc_test)
    print("AUC precision recall: ", auc_pr_test)

    # Save all metrics in a csv file
    metrics_df = pd.DataFrame({'fold': list(validation_metrics_dic.keys()),
                               'precision_valid': [metrics[0] for metrics in validation_metrics_dic.values()],
                               'recall_valid': [metrics[1] for metrics in validation_metrics_dic.values()],
                               'f1_valid': [metrics[2] for metrics in validation_metrics_dic.values()],
                               'auc_roc_valid': [metrics[3] for metrics in validation_metrics_dic.values()],
                               'auc_pr_valid': [metrics[4] for metrics in validation_metrics_dic.values()],
                               'precision_test': [metrics[0] for metrics in testset_metrics_dic.values()],
                               'recall_test': [metrics[1] for metrics in testset_metrics_dic.values()],
                               'f1_test': [metrics[2] for metrics in testset_metrics_dic.values()],
                               'auc_roc_test': [metrics[3] for metrics in testset_metrics_dic.values()],
                               'auc_pr_test': [metrics[4] for metrics in testset_metrics_dic.values()]})

    # create results dir
    Path("./results").mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(f"./results/{experiment_name}_metrics.csv", index=False)
    print(f"Results saved in ./results/{experiment_name}_metrics.csv")

    # Finish the session before next run
    if activate_wandb:
        wandb.finish()

# Perf on last run
'''
Precision valid:  0.9046428571428571
Recall valid:  0.8933333333333333
F1 valid:  0.8983759733036708
AUC ROC valid:  0.9600000000000002
AUC precision recall:  0.96385885468161

Test Set
Precision test:  0.27824239520204075
Recall test:  0.7208333333333333
F1 test:  0.38541352754615216
AUC ROC test:  0.9218473193473194
AUC precision recall:  0.5563509980096927


Validation Set
Precision valid:  0.913809523809524
Recall valid:  0.8400000000000001
F1 valid:  0.8742273307790549
AUC ROC valid:  0.9546666666666667
AUC precision recall:  0.9563074305868422

Test Set
Precision test:  0.336829340026271
Recall test:  0.6833333333333333
F1 test:  0.4403030066859854
AUC ROC test:  0.9174854312354311
AUC precision recall:  0.5810267814334081
'''
