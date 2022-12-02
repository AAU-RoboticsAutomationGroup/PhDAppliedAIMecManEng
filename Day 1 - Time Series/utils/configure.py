import os

class Config:
    def __init__(self):
        # project path
        self.project_path = os.path.dirname(os.path.abspath(__file__))

        # dataset path
        self.data_folder_path = os.path.join(self.project_path, '../Data')
        self.org_aursad_flt_path = os.path.join(self.data_folder_path, 'AURSAD_course.dat')

        # raw aursad D
        self.raw_data_source = True
        # original aursad (process + task)
        self.org_data_only_process = True

        # parmeters
        # number of class
        self.num_class = 4
        # learning rate
        self.lr = 0.0001
        # loss fuction
        self.loss = 'sparse_categorical_crossentropy'
        # training epoch
        self.epochs = 10
        # batch size
        self.batch_size = 8
        # if accuracy is not improved after specified patience, then it will stop training.
        self.patience = 10

        # parameters == Transformer
        self.head_size=256
        self.num_heads=4
        self.ff_dim=4
        self.num_transformer_blocks=6
        self.mlp_units=[128]
        self.mlp_dropout=0.3
        self.dropout=0.2

        # parameters == DNN
        self.units_h1 = 938
        self.units_h2 = 1876
        self.units_h3 = 938
        self.units_h4 = 4

        # parameters == ConvLSTM2D
        # time step for each subsequence
        self.steps = 8
        # the length of each subsequence
        self.length = 106
        self.verbose = 1
        # the dimensionality of the output space (i.e. the number of output filters in the convolution).
        self.filters = 64

        # model checkpoint path
        self.model_path = os.path.join(self.project_path, 'checkpoints')
        self.model_TRM_org_data_path = os.path.join(self.model_path, 'TRM_org_data')
        self.model_Conv1D_org_data_path = os.path.join(self.model_path, 'Conv1D_org_data')
        self.model_ConvLSTM2D_org_data_path = os.path.join(self.model_path, 'ConvLSTM2D_org_data')
        self.model_LSTM_org_data_path = os.path.join(self.model_path, 'LSTM_org_data')

        # figures of loss and acc
        self.fig_path = os.path.join(self.project_path, 'loss_acc')
        self.TRM_org_data_loss_acc_fig_path = os.path.join(self.fig_path, 'TRM_org_data')
        self.Conv1D_org_data_loss_acc_fig_path = os.path.join(self.fig_path, 'Conv1D_org_data')
        self.ConvLSTM2D_org_data_loss_acc_fig_path = os.path.join(self.fig_path, 'ConvLSTM2D_org_data')
        self.LSTM_org_data_loss_acc_fig_path = os.path.join(self.fig_path, 'LSTM_org_data')

        # json file for saving scores
        self.scores_path = os.path.join(self.project_path, 'scores')
        self.scores_file_path = os.path.join(self.scores_path, 'scores.json')



    def model_parameters_set_process_task(self, model_name, org_data_only_process, is_flt):

        # check if it is the filtered data
        if is_flt == 'Yes':
            flt_path = "flt"
        else:
            flt_path = "unflt"

        if model_name == "TRM_org_data":
            if org_data_only_process == 'Yes':
                model_path = os.path.join(self.model_TRM_org_data_path, flt_path, 'process_model.h5')
                loss_img = os.path.join(self.TRM_org_data_loss_acc_fig_path, flt_path, 'process_loss.png')
                acc_img = os.path.join(self.TRM_org_data_loss_acc_fig_path, flt_path, 'process_acc.png')
                precision = flt_path + "process_TRM_precision"
                recall = flt_path + "process_TRM_recall"
                f1 = flt_path + "process_TRM_f1"
            else:
                model_path = os.path.join(self.model_TRM_org_data_path, flt_path, 'process_task_model.h5')
                loss_img = os.path.join(self.TRM_org_data_loss_acc_fig_path, flt_path, 'process_task_loss.png')
                acc_img = os.path.join(self.TRM_org_data_loss_acc_fig_path, flt_path, 'process_task_acc.png')
                precision = flt_path + "process_task_TRM_precision"
                recall = flt_path + "process_task_TRM_recall"
                f1 = flt_path + "process_task_TRM_f1"
        elif model_name == "Conv1D_org_data":
            if org_data_only_process == 'Yes':
                model_path = os.path.join(self.model_Conv1D_org_data_path, flt_path, 'process_model.h5')
                loss_img = os.path.join(self.Conv1D_org_data_loss_acc_fig_path, flt_path, 'process_loss.png')
                acc_img = os.path.join(self.Conv1D_org_data_loss_acc_fig_path, flt_path, 'process_acc.png')
                precision = flt_path + "process_Conv1D_precision"
                recall = flt_path + "process_Conv1D_recall"
                f1 = flt_path + "process_Conv1D_f1"
            else:
                model_path = os.path.join(self.model_Conv1D_org_data_path, flt_path, 'process_task_model.h5')
                loss_img = os.path.join(self.Conv1D_org_data_loss_acc_fig_path, flt_path, 'process_task_loss.png')
                acc_img = os.path.join(self.Conv1D_org_data_loss_acc_fig_path, flt_path, 'process_task_acc.png')
                precision = flt_path + "process_task_Conv1D_precision"
                recall = flt_path + "process_task_Conv1D_recall"
                f1 = flt_path + "process_task_Conv1D_f1"
        elif model_name == "ConvLSTM2D_org_data":
            if org_data_only_process == 'Yes':
                model_path = os.path.join(self.model_ConvLSTM2D_org_data_path, flt_path, 'process_model.h5')
                loss_img = os.path.join(self.ConvLSTM2D_org_data_loss_acc_fig_path, flt_path, 'process_loss.png')
                acc_img = os.path.join(self.ConvLSTM2D_org_data_loss_acc_fig_path, flt_path, 'process_acc.png')
                precision = flt_path + "process_ConvLSTM2D_precision"
                recall = flt_path + "process_ConvLSTM2D_recall"
                f1 = flt_path + "process_ConvLSTM2D_f1"
            else:
                model_path = os.path.join(self.model_ConvLSTM2D_org_data_path, flt_path, 'process_task_model.h5')
                loss_img = os.path.join(self.ConvLSTM2D_org_data_loss_acc_fig_path, flt_path, 'process_task_loss.png')
                acc_img = os.path.join(self.ConvLSTM2D_org_data_loss_acc_fig_path, flt_path, 'process_task_acc.png')
                precision = flt_path + "process_task_ConvLSTM2D_precision"
                recall = flt_path + "process_task_ConvLSTM2D_recall"
                f1 = flt_path + "process_task_ConvLSTM2D_f1"
        elif model_name == "LSTM_org_data":
            if org_data_only_process == 'Yes':
                model_path = os.path.join(self.model_LSTM_org_data_path, flt_path, 'process_model.h5')
                loss_img = os.path.join(self.LSTM_org_data_loss_acc_fig_path, flt_path, 'process_loss.png')
                acc_img = os.path.join(self.LSTM_org_data_loss_acc_fig_path, flt_path, 'process_acc.png')
                precision = flt_path + "process_LSTM_precision"
                recall = flt_path + "process_LSTM_recall"
                f1 = flt_path + "process_LSTM_f1"
            else:
                model_path = os.path.join(self.model_LSTM_org_data_path, flt_path, 'process_task_model.h5')
                loss_img = os.path.join(self.LSTM_org_data_loss_acc_fig_path, flt_path, 'process_task_loss.png')
                acc_img = os.path.join(self.LSTM_org_data_loss_acc_fig_path, flt_path, 'process_task_acc.png')
                precision = flt_path + "process_task_LSTM_precision"
                recall = flt_path + "process_task_LSTM_recall"
                f1 = flt_path + "process_task_LSTM_f1"


        return model_path, loss_img, acc_img, precision, recall, f1
