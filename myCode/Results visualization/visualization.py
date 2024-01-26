import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt
import json


class Visualization:
    def __init__(self, project='rehersal Alexnet MNIST Task IL tr-t split v2',
                 UID=['rehearsal_dataset', 'batch_size_rehearsal', 'pretraining', 'learning_rate', 'epochs'],
                 y_min=45):
        # todo reade this automatically somehow
        self.acc_col = ['acc_task_0', 'acc_task_1', 'acc_task_2', 'acc_task_3', 'acc_task_4']
        self.acc_test_col = ['acc_test_task_0', 'acc_test_task_1', 'acc_test_task_2', 'acc_test_task_3',
                             'acc_test_task_4']
        self.additional_title_params = ['architecture', 'dataset']
        self.runs_metrics = []

        self.project = project
        self.UID = UID
        api = wandb.Api(timeout=90)
        self.runs = api.runs(f"qba/{project}")

        # runs metadata
        runs_list, run_name = [], []
        for run in self.runs:
            runs_list.append(run.config)
            run_name.append(run.name)
        self.df = pd.DataFrame(runs_list).astype(str)
        self.df['run_name'] = run_name

        # find faulty runs
        self.faulty_runs = []
        for i in range(len(self.df)):
            if self.df.iloc[i].classes_list != 'nan':
                classes_list = json.loads(self.df.iloc[i].classes_list)
                for task in classes_list:
                    if 0 in task:
                        if task[1] == 0:
                            self.faulty_runs.append(i)

        # UID
        self.df['UID'] = ''
        for c in UID:
            self.df['UID'] += self.df[c].astype(str) + ';'

        # unique run params
        self.unique_run_params = []
        unique_UID = self.df['UID'].unique()
        for uuid in unique_UID:
            self.unique_run_params.append(uuid.split(';')[:len(UID)])

        # runs indexes for every unique run settings
        self.unique_run_settings_idxs = []
        for run_param in self.unique_run_params:
            idx = pd.Series([True for _ in range(len(self.df))])
            for c, v in zip(UID, run_param):
                idx = idx & (self.df[c] == v)

            idxs = self.df[idx].index
            idxs = [idx for idx in idxs if idx not in self.faulty_runs]  # remove faulty runs
            self.unique_run_settings_idxs.append(idxs)

        # accuracy axis range for plotting
        self.y_min = y_min

        # run unique params - run idxs dict
        self.runs_params_settings_idxs_dict = {}
        for i, run_param in enumerate(self.unique_run_params):
            self.runs_params_settings_idxs_dict[";".join(run_param)] = self.unique_run_settings_idxs[i]

    def box_plot(metric_name, df, filename=None):
        UIDS = df['UID'].unique()

        values = []
        for uid in UIDS:
            values.append(df[df['UID'] == uid][metric_name].median())

        order = np.argsort(values)
        UIDS = UIDS[order]

        plt.figure(figsize=(20, 10))
        for i, uid in enumerate(UIDS):
            plt.boxplot(df[df['UID'] == uid][metric_name], positions=[i], widths=0.6)

        labels = UIDS
        labels = [f"{uid.split(';')[0]} {uid.split(';')[2]}" for uid in UIDS]

        plt.xticks(np.arange(len(UIDS)), labels, rotation=90)
        plt.title(f"metic: {metric_name}       label convention: dataset pretraining? epochs")
        plt.xlabel("unique identifier")
        plt.ylabel("accuracy [%]")
        plt.show()

        if filename:
            plt.savefig("images/"+filename+".pdf", format='pdf')
            plt.savefig("images/"+filename+".svg", format='svg')

    def extract_all_runs_metrics(self):
        for i in range(len(self.df)):
            if i in self.faulty_runs:
                continue
            try:
                UID = self.df.iloc[i]['UID']
                run = self.runs[i].history(100000)

                # df_train = run[self.acc_col].dropna().reset_index().drop(columns='index')
                df_test = run[self.acc_test_col].dropna().reset_index().drop(columns='index')

                # accuracy metrics table
                num_tasks = len(self.acc_test_col)
                steps_per_task = len(df_test) // num_tasks
                acc_at_the_end = df_test.iloc[-1].values
                acc_max = df_test.max().values
                acc_min = df_test.min().values  # todo would be good to change to min accuracy after training
                acc_mean = []
                for i in range(num_tasks):
                    acc_mean.append(df_test[self.acc_test_col[i]].dropna()[steps_per_task * (i):].mean())

                acc_decrease = (df_test.max().values - acc_at_the_end)
                forgeting_tasks = np.arange(num_tasks - 1, -1, -1)
                forgeting_tasks[
                    forgeting_tasks == 0] = 10e6  # during the last task there is no time to forget and no time to divide by 0
                acc_mean_decrease_per_task = acc_decrease / forgeting_tasks

                curr_dict = {"UID": UID}

                for task in range(num_tasks):
                    curr_dict[f"acc_at_the_end_task_{task}"] = acc_at_the_end[task]
                    curr_dict[f"acc_mean_task_{task}"] = acc_mean[task]
                    curr_dict[f"acc_mean_decrease_per_task_{task}"] = acc_mean_decrease_per_task[task]
                    curr_dict[f"acc_max{task}"] = acc_max[task]
                    curr_dict[f"acc_min{task}"] = acc_min[task]

                curr_dict["split"] = "test"

                self.runs_metrics.append(curr_dict)
            except:
                print(f"Could not fetch metrics in run: {i} UID: {UID}")

        # UID to columns woth params
        self.metrics_df = pd.DataFrame(self.runs_metrics)
        for i, params in enumerate(self.metrics_df.UID.apply(lambda x: x.split(';'))):
            for c, param in zip(self.UID, params):
                self.metrics_df.loc[i, c] = param

        # mean of mean columns
        acc_at_the_end_cols = self.metrics_df.columns[self.metrics_df.columns.str.contains('acc_at_the_end_task')]
        self.metrics_df['mean_acc_at_the_end'] = self.metrics_df[acc_at_the_end_cols].mean(axis=1)

        self.metrics_df['median_acc_at_the_end'] = self.metrics_df[acc_at_the_end_cols].median(axis=1)

        acc_mean_cols = self.metrics_df.columns[self.metrics_df.columns.str.contains('acc_mean')]
        self.metrics_df['mean_acc_mean'] = self.metrics_df[acc_mean_cols].mean(axis=1)

        acc_mean_decrease_per_task_cols = self.metrics_df.columns[
            self.metrics_df.columns.str.contains('acc_mean_decrease_per_task')]
        self.metrics_df['mean_acc_mean_decrease_per_task'] = self.metrics_df[acc_mean_decrease_per_task_cols].mean(
            axis=1)

        mean_acc_max = self.metrics_df.columns[self.metrics_df.columns.str.contains('acc_max')]
        self.metrics_df['mean_acc_max'] = self.metrics_df[mean_acc_max].mean(axis=1)

    def extract_all_runs_metrics_after_task_3(self):
        for i in range(len(self.df)):
            if i in self.faulty_runs:
                continue
            try:
                UID = self.df.iloc[i]['UID']
                run = self.runs[i].history(100000)

                num_tasks = int(self.runs[i].config['num_classes'])/int(self.runs[i].config['classes_per_task'])

                # df_train = run[self.acc_col].dropna().reset_index().drop(columns='index')
                df_test = run[self.acc_test_col].dropna().reset_index().drop(columns='index')
                idx_start_task_3 = int(len(df_test)/num_tasks*3)
                df_test = df_test.iloc[:idx_start_task_3]  # after task 3

                # accuracy metrics table
                num_tasks = len(self.acc_test_col)
                steps_per_task = len(df_test) // num_tasks
                acc_at_the_end = df_test.iloc[-1].values
                acc_max = df_test.max().values
                acc_min = df_test.min().values  # todo would be good to change to min accuracy after training
                acc_mean = []
                for i in range(num_tasks):
                    acc_mean.append(df_test[self.acc_test_col[i]].dropna()[steps_per_task * (i):].mean())

                acc_decrease = (df_test.max().values - acc_at_the_end)
                forgeting_tasks = np.arange(num_tasks - 1, -1, -1)
                forgeting_tasks[
                    forgeting_tasks == 0] = 10e6  # during the last task there is no time to forget and no time to divide by 0
                acc_mean_decrease_per_task = acc_decrease / forgeting_tasks

                curr_dict = {"UID": UID}

                for task in range(num_tasks):
                    curr_dict[f"acc_at_the_end_task_{task}"] = acc_at_the_end[task]
                    curr_dict[f"acc_mean_task_{task}"] = acc_mean[task]
                    curr_dict[f"acc_mean_decrease_per_task_{task}"] = acc_mean_decrease_per_task[task]
                    curr_dict[f"acc_max{task}"] = acc_max[task]
                    curr_dict[f"acc_min{task}"] = acc_min[task]

                curr_dict["split"] = "test"

                self.runs_metrics.append(curr_dict)
            except Exception as e:
                print(f"Could not fetch metrics in run: {i} UID: {UID} exception: {e}")

        # UID to columns woth params
        self.metrics_df = pd.DataFrame(self.runs_metrics)
        for i, params in enumerate(self.metrics_df.UID.apply(lambda x: x.split(';'))):
            for c, param in zip(self.UID, params):
                self.metrics_df.loc[i, c] = param

        # mean of mean columns
        acc_at_the_end_cols = self.metrics_df.columns[self.metrics_df.columns.str.contains('acc_at_the_end_task')][:3]
        self.metrics_df['mean_acc_at_the_end'] = self.metrics_df[acc_at_the_end_cols].mean(axis=1)

        self.metrics_df['median_acc_at_the_end'] = self.metrics_df[acc_at_the_end_cols].median(axis=1)

        acc_mean_cols = self.metrics_df.columns[self.metrics_df.columns.str.contains('acc_mean')]
        self.metrics_df['mean_acc_mean'] = self.metrics_df[acc_mean_cols].mean(axis=1)

        acc_mean_decrease_per_task_cols = self.metrics_df.columns[
            self.metrics_df.columns.str.contains('acc_mean_decrease_per_task')]
        self.metrics_df['mean_acc_mean_decrease_per_task'] = self.metrics_df[acc_mean_decrease_per_task_cols].mean(
            axis=1)

        mean_acc_max = self.metrics_df.columns[self.metrics_df.columns.str.contains('acc_max')]
        self.metrics_df['mean_acc_max'] = self.metrics_df[mean_acc_max].mean(axis=1)

    def print_settings(self):
        for i, run_param in enumerate(self.unique_run_params):
            print(f"Setting {i}: {run_param} num runs: {len(self.unique_run_settings_idxs[i])}")

    def plot_single_setting_aggregated(self, run_param, fontsize=12, up_postion=1.00, filename=None, layout='square', plot_till=None):
        unixe_idxs = self.runs_params_settings_idxs_dict[run_param]
        df_train, df_test = self.extract_data_from_runs(unixe_idxs)
        self.create_plot(df_train, df_test, unixe_idxs, fontsize=fontsize, up_postion=up_postion, filename=filename, layout=layout, plot_till=plot_till)

    def plot_single_setting_all_runs(self, run_param):
        unixe_idxs = self.runs_params_settings_idxs_dict["-".join(run_param)]

        for i, run_idx in enumerate(unixe_idxs):
            if i in self.faulty_runs:
                continue
            curr_run = self.runs[run_idx].history(100000)
            curr_run['run_no'] = i

            df_train = curr_run[self.acc_col].dropna().reset_index().drop(columns='index')
            df_test = curr_run[self.acc_test_col].dropna().reset_index().drop(columns='index')

            fig, ax = plt.subplots(2, 1)
            fig.set_figheight(15)
            fig.set_figwidth(10)

            title = ""
            for additional_param in self.additional_title_params:
                value = self.df.iloc[unixe_idxs][additional_param].iloc[0]
                title += f"{additional_param}: {value}\n"
            unique_params = self.df.iloc[unixe_idxs]['UID'].iloc[0].split(";")[:-1]
            for name, value in zip(self.UID, unique_params):
                title += f"{name}: {value}\n"

            fig.suptitle(title + f"runs: {len(unixe_idxs)}", fontsize=12, position=(0.5, 1.06))

            # train
            df_train.plot(ax=ax[0], grid=True, ylim=[self.y_min, 100], title='Train')
            ax[0].set_xlabel("step")
            ax[0].set_ylabel("accuracy [%]")

            # test
            df_test.plot(ax=ax[1], grid=True, ylim=[self.y_min, 100], title='Test')
            ax[1].set_xlabel("step")
            ax[1].set_ylabel("accuracy [%]")
            plt.show()

    def plot_everything(self):
        for unixe_idxs in self.unique_run_settings_idxs:
            df_train, df_test = self.extract_data_from_runs(unixe_idxs)
            self.create_plot(df_train, df_test, unixe_idxs)

    def extract_data_from_runs(self, unixe_idxs):
        df_train, df_test, self.all_test_runs_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i, run_idx in enumerate(unixe_idxs):
            try:
                curr_run = self.runs[run_idx].history(100000)
                curr_run['run_no'] = i
                self.all_test_runs_data = pd.concat([self.all_test_runs_data, curr_run])
                if len(df_train > 0):
                    df_train += curr_run[self.acc_col].dropna().reset_index().drop(columns='index')
                    df_test += curr_run[self.acc_test_col].dropna().reset_index().drop(columns='index')
                else:
                    df_train = curr_run[self.acc_col].dropna().reset_index().drop(columns='index')
                    df_test = curr_run[self.acc_test_col].dropna().reset_index().drop(columns='index')
            except Exception as e:  # I know it's not the best way to do it
                print(f"Error in run: {run_idx} Error: {e}")
        return df_train, df_test

    def create_plot(self, df_train, df_test, unixe_idxs, fontsize=12, up_postion=1.03, filename=None, layout='square', plot_till=None):

        assert layout in ['square', 'vertical', 'vertical_short'], f'layout must be square or vertical, and is: {layout}'

        try:
            df_train /= len(unixe_idxs)
            df_test /= len(unixe_idxs)
            if plot_till:
                plot_till = int(len(df_test)*plot_till)
                df_train = df_train.iloc[:plot_till]
                df_test = df_test.iloc[:plot_till]

            if layout == 'square':
                fig, ax = plt.subplots(2, 2, tight_layout=True)
                axes_idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
                fig.set_figheight(12)
                fig.set_figwidth(20)
            elif layout == 'vertical':
                fig, ax = plt.subplots(3, 1, tight_layout=True)
                axes_idx = [0, -1, 1, 2]
                fig.set_figheight(20)
                fig.set_figwidth(10)
            elif layout == 'vertical_short':
                fig, ax = plt.subplots(2, 1, tight_layout=True)
                axes_idx = [0, -1, 1]
                fig.set_figheight(15)
                fig.set_figwidth(10)


            title = ""
            for additional_param in self.additional_title_params:
                value = self.df.iloc[unixe_idxs][additional_param].iloc[0]
                title += f"{additional_param}: {value}\n"
            unique_params = self.df.iloc[unixe_idxs]['UID'].iloc[0].split(";")[:-1]
            for name, value in zip(self.UID, unique_params):
                title += f"{name}: {value}\n"

            fig.suptitle(title + f"runs: {len(unixe_idxs)}", fontsize=fontsize, position=(0.5, up_postion))

            # train
            df_train.plot(ax=ax[axes_idx[0]], grid=True, ylim=[self.y_min, 100], fontsize=fontsize)
            ax[axes_idx[0]].set_xlabel("step", fontsize=fontsize)
            ax[axes_idx[0]].set_ylabel("accuracy [%]", fontsize=fontsize)
            ax[axes_idx[0]].set_title('Train', fontsize=fontsize)
            ax[axes_idx[0]].legend(['task 0', 'task 1', 'task 2', 'task 3', 'task 4'])

            # test
            df_test.plot(ax=ax[axes_idx[2]], grid=True, ylim=[self.y_min, 100], fontsize=fontsize)
            ax[axes_idx[2]].set_xlabel("step", fontsize=fontsize)
            ax[axes_idx[2]].set_ylabel("accuracy [%]", fontsize=fontsize)
            ax[axes_idx[2]].set_title('Test', fontsize=fontsize)
            ax[axes_idx[2]].legend(['task 0', 'task 1', 'task 2', 'task 3', 'task 4'])

            if layout is not 'vertical_short':
                # test min max 3->2
                for acc_col in self.acc_test_col[:2]:
                    max = self.all_test_runs_data[acc_col].dropna().reset_index().groupby("index").max().max(axis=1)
                    mean = self.all_test_runs_data[acc_col].dropna().reset_index().groupby("index").mean().mean(axis=1).reset_index().drop(columns='index')
                    min = self.all_test_runs_data[acc_col].dropna().reset_index().groupby("index").min().min(axis=1)
                    x = np.arange(len(min))
                    mean.plot(ax=ax[axes_idx[3]], grid=True, ylim=[self.y_min, 100], fontsize=fontsize)
                    ax[axes_idx[3]].fill_between(x, min, max, alpha=0.2)
                    ax[axes_idx[3]].set_xlabel("step", fontsize=fontsize)
                    ax[axes_idx[3]].set_ylabel("accuracy [%]", fontsize=fontsize)
                    ax[axes_idx[3]].set_title('Test', fontsize=fontsize)
                    ax[axes_idx[3]].legend(['task 0', 'task 0 min max', 'task 1', 'task 1 min max'])


            # accuracy metrics table
            num_tasks = len(self.acc_test_col)
            steps_per_task = len(df_test) // num_tasks
            acc_at_the_end = df_test.iloc[-1].values
            acc_mean = []
            for i in range(num_tasks):
                acc_mean.append(df_test[self.acc_test_col[i]].dropna()[steps_per_task * (i):].mean())

            acc_decrease = (df_test.max().values - acc_at_the_end)
            forgeting_tasks = np.arange(num_tasks - 1, -1, -1)
            forgeting_tasks[
                forgeting_tasks == 0] = 10e6  # during the last task there is no time to forget and no time to divide by 0
            acc_mean_decrease_per_task = acc_decrease / forgeting_tasks

            metrics_df = pd.DataFrame({
                "task": np.arange(num_tasks),
                "acc at the end": acc_at_the_end,
                "mean acc": acc_mean,
                "acc loss per task": acc_mean_decrease_per_task,
                "split": ["test" for _ in range(num_tasks)],
            }).round(2)
            if layout is 'square':
                ax[axes_idx[1]].axis('off')
                ax[axes_idx[1]].axis('tight')
                table = ax[axes_idx[1]].table(
                    cellText=metrics_df.to_numpy(),
                    colLabels=metrics_df.columns,
                    loc='center',
                    cellLoc='center',
                    colWidths=list([.1 for _ in range(len(metrics_df.columns))]),
                )
                table.scale(2, 1.5)
                table.auto_set_font_size(False)
                table.set_fontsize(fontsize)
            else:
                print(metrics_df.to_latex(index=False))


        except Exception as e:  # I know it's not the best way to do it
            print(f"Exception with ids: {unixe_idxs}, exception: {e}")

        if filename:
            plt.savefig("images/"+filename+".pdf", format='pdf')
            plt.savefig("images/"+filename+".svg", format='svg')
