from opensoundscape.annotations import BoxedAnnotations
import os
import pandas as pd
import numpy as np
import torch
import random
import wandb
from opensoundscape import CNN
from opensoundscape.data_selection import resample
from opensoundscape.preprocess.utils import show_tensor_grid
from opensoundscape import AudioFileDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold


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

    df_trainset = pd.read_csv('train_set_baseline.csv', index_col=[0,1,2])
    return df_trainset



def create_test_set(test_files_path, fish_sound):
    
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

    annotations = BoxedAnnotations.from_raven_files(raven_files=selection_files, audio_files=audio_files, annotation_column='Type')

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



def setup_model(model, sample_rate):
    # all data augmentations
    model.preprocessor.pipeline.random_trim_audio.bypass = True

    model.preprocessor.pipeline.overlay.bypass = True
    # model.preprocessor.pipeline.overlay.overlay_df = (train_df.astype(int))
    # model.preprocessor.pipeline.overlay.set(overlay_class='A')

    model.preprocessor.pipeline.time_mask.bypass = True
    model.preprocessor.pipeline.time_mask.set(max_masks=2, max_width=0.1)

    model.preprocessor.pipeline.frequency_mask.bypass = True
    model.preprocessor.pipeline.frequency_mask.set(max_masks=2, max_width=0.1)

    model.preprocessor.pipeline.add_noise.bypass = True
    model.preprocessor.pipeline.random_affine.bypass = True

    model.preprocessor.pipeline.bandpass.set(min_f=50, max_f=2000)

    model.preprocessor.pipeline.to_spec.set(window_samples = 4 * (sample_rate // 100),
                                            # overlap_samples = None,
                                            # fft_size = None,
                                            # dB_scale = True,
                                            # scaling = 'spectrum'
                                            )

def compute_metrics(y_true, y_pred, fish_sound):
    # Compute precision from logits and labels
    valid_labels = y_true[fish_sound].values
    valid_pred = y_pred[fish_sound].values.round() #Threshold = 0.5

    precision_valid = precision_score(valid_labels, valid_pred, pos_label=1, average='binary')
    recall_valid = recall_score(valid_labels, valid_pred, pos_label=1, average='binary')
    f1_valid = f1_score(valid_labels, valid_pred, pos_label=1, average='binary')
    auc_roc_valid = roc_auc_score(valid_labels, y_pred[fish_sound].values)
    precision, recall, _thresholds = precision_recall_curve(valid_labels, y_pred[fish_sound].values)
    auc_precision_recall_valid = auc(recall, precision)

    return precision_valid, recall_valid, f1_valid, auc_roc_valid, auc_precision_recall_valid



if __name__ == "__main__":

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Hard coded params for now
    # TODO get them from the dict config eventually
    fish_sound = 'A'
    fish_sound_folder = 'fishA'
    window_len_s = 5.0
    experiment_name = 'texel_baseline_20250331'
    samples_dir = '/home/reindert/Valentin_REVO/surfperch_toshare/eval_texel Outputs/september 2024/surfperch/labeled_outputs/'
    test_files_path = '/home/reindert/Valentin_REVO/surfperch_toshare/eval_texel Data/september 2024/test_set/'
    
    # WANDB
    activate_wandb = False
    wandb_exp_name = 'No augment no upsampling 030425'

    # RUN PARAMS
    nbr_epochs = 50
    batch_size = 8
    num_workers = 16


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


    # Load test set
    test_set_df = create_test_set(test_files_path, fish_sound)

    # Duplicate of cell bellow to compute different metrics
    validation_metrics_dic = {}
    testset_metrics_dic = {}

    seed = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    df_trainset = create_train_valid_set(experiment_name, samples_dir, fish_sound, fish_sound_folder, window_len_s)

    # Iterate over the folds
    for i, (train_index, valid_index) in enumerate(kf.split(df_trainset, df_trainset[fish_sound])):
        
        print(f"Running experiment for Fold {i+1}/5")

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

        print(f"model.device is {model.device}")

        sample_rate = 32000
        setup_model(model, sample_rate)

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
            save_interval=5,  
            save_path=checkpoint_folder,  
            progress_bar=True
        )

        # Run predictions on the validation set and on the test set
        predict_validset = model.predict(valid_df, batch_size=batch_size, num_workers=num_workers, activation_layer='sigmoid', wandb_session=wandb_session)
        pred_testset = model.predict(test_set_df, batch_size=batch_size, num_workers=num_workers, activation_layer='sigmoid', wandb_session=wandb_session)

        # Compute metrics
        prec_valid, recall_valid, f1_valid, auc_roc_valid, auc_pr_valid = compute_metrics(valid_df, predict_validset, fish_sound)
        prec_test, recall_test, f1_test, auc_roc_test, auc_pr_test = compute_metrics(test_set_df, pred_testset, fish_sound)

        # Save metrics of the fold iteration
        validation_metrics_dic[i] = [prec_valid, recall_valid, f1_valid, auc_roc_valid, auc_pr_valid]
        testset_metrics_dic[i] = [prec_test, recall_test, f1_test, auc_roc_test, auc_pr_test]

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
    
    # Finish the session before next run
    if activate_wandb:
        wandb.finish()


# Perf on last run
'''
Texel data - no augmentation - no upsampling
Validation Set
Precision valid:  0.9578296703296705
Recall valid:  0.8800000000000001
F1 valid:  0.9151685294065718
AUC ROC valid:  0.9573333333333334
AUC precision recall:  0.9653796326824191

Test Set
Precision test:  0.2825814303730681
Recall test:  0.7183673469387756
F1 test:  0.36321974537253954
AUC ROC test:  0.9159642707271747
AUC precision recall:  0.5226434867971157
'''