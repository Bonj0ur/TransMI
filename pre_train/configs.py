import os

############################################################
#  Configs
############################################################

class Config():
    # Data Dir
    data_dir = '../data/'
    # Train Dir
    train_dir = os.path.join(data_dir,'pre_train/')
    train_data_dir = os.path.join(train_dir,'data/')
    train_label_dir = train_dir + 'clsLabel.csv'
    # Data Loader
    train_batch_size = 32
    num_workers = 1
    # Train
    k_fold = 5
    seed = 66
    using_gpu = True
    gpu_id = 0
    learning_rate = 0.001
    lr_scheduler_gamma = 0.98
    train_epochs = 100
    # Log
    log_dir = 'pre_train_output'