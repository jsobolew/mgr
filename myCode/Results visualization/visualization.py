import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt


class Visualization:
    def __init__(self, project='rehersal Alexnet MNIST Task IL tr-t split v2',
                 UID=['rehearsal_dataset', 'batch_size_rehearsal', 'pretraining', 'learning_rate']):
        # todo reade this automatically somehow
        self.acc_col = ['acc_task_0', 'acc_task_1', 'acc_task_2', 'acc_task_3', 'acc_task_4']
        self.acc_test_col = ['acc_test_task_0', 'acc_test_task_1', 'acc_test_task_2', 'acc_test_task_3',
                             'acc_test_task_4']

        self.project = project
        self.UID = UID
        api = wandb.Api()
        self.runs = api.runs(f"qba/{project}")

        # runs metadata
        runs_list = []
        for run in self.runs:
            runs_list.append(run.config)
        self.df = pd.DataFrame(runs_list).astype(str)

        # UID
        self.df['UID'] = ''
        for c in UID:
            self.df['UID'] += self.df[c].astype(str) + ';'

        # unique run params
        self.unique_run_params = []
        unique_UID = self.df['UID'].unique()
        for uuid in unique_UID:
            self.unique_run_params.append(uuid.split(';')[:4])

        # runs indexes for every unique run settings
        self.unique_run_settings_idxs = []
        for run_param in self.unique_run_params:
            idx = pd.Series([True for _ in range(len(self.df))])
            for c, v in zip(UID, run_param):
                idx = idx & (self.df[c] == v)
            self.unique_run_settings_idxs.append(self.df[idx].index)

        # accuracy axis range for plotting
        self.y_min = 45
        if self.df.setup[0] == 'classIL':
            self.y_min = 0

    def plot_everything(self):
        for unixe_idxs in self.unique_run_settings_idxs:
            df_train, df_test = self.extract_data_from_runs(unixe_idxs)
            self.create_plot(df_train, df_test, unixe_idxs)

    def extract_data_from_runs(self, unixe_idxs):

        df_train, df_test, self.all_test_runs_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i, run_idx in enumerate(unixe_idxs.values):
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

    def create_plot(self, df_train, df_test, unixe_idxs):
        try:
            df_train /= len(unixe_idxs)
            df_test /= len(unixe_idxs)

            fig, ax = plt.subplots(2, 2)
            fig.set_figheight(10)
            fig.set_figwidth(20)

            title = ""
            unique_params = self.df.iloc[unixe_idxs]['UID'].iloc[0].split(";")[:-1]
            for name, value in zip(self.UID, unique_params):
                title += f"{name}: {value}\n"

            fig.suptitle(title + f"runs: {len(unixe_idxs)}", fontsize=12, position=(0.5, 1.02))

            # train
            df_train.plot(ax=ax[0, 0], grid=True, ylim=[self.y_min, 100], title='Train')
            ax[0, 0].set_xlabel("step")
            ax[0, 0].set_ylabel("accuracy [%]")

            # test
            df_test.plot(ax=ax[1, 0], grid=True, ylim=[self.y_min, 100], title='Test')
            ax[1, 0].set_xlabel("step")
            ax[1, 0].set_ylabel("accuracy [%]")

            # test std dev
            cols = self.acc_test_col.copy()
            cols.append('run_no')
            df = self.all_test_runs_data[cols].dropna()

            runs = len(self.all_test_runs_data[self.acc_test_col].dropna()) // len(df_test)
            steps_per_run = len(self.all_test_runs_data[self.acc_test_col].dropna()) // runs
            new_idx = np.concatenate([np.arange(steps_per_run) for _ in range(runs)])
            df['step'] = new_idx

            for col in self.acc_test_col:
                df[[col, 'step', 'run_no']].pivot(index='step', columns='run_no'). \
                    std(1).plot(ax=ax[1, 1], title='Standard Deviation of Test Accuracy')
            ax[1, 1].set_xlabel("step")
            ax[1, 1].set_ylabel("std dev of accuracy [%]")

            # accuracy metrics table
            num_tasks = len(self.acc_test_col)
            steps_per_task = len(df_test) // num_tasks
            acc_at_the_end = df_test.iloc[-1].values
            acc_mean = []
            for i in range(num_tasks):
                acc_mean.append(df_test[self.acc_test_col[i]].dropna()[steps_per_task*(i):].mean())

            acc_decrease = (df_test.max().values - acc_at_the_end)
            forgeting_tasks = np.arange(num_tasks-1, -1, -1)
            forgeting_tasks[forgeting_tasks == 0] = 10e6 # during the last task there is no time to forget and no time to divide by 0
            acc_mean_decrease_per_task = acc_decrease / forgeting_tasks

            metrics_df = pd.DataFrame({
                "task": np.arange(num_tasks),
                "acc_at_the_end": acc_at_the_end,
                "acc_mean": acc_mean,
                "acc_mean_decrease_per_task": acc_mean_decrease_per_task,
                "split": ["test" for _ in range(num_tasks)],
            }).round(2)

            ax[0, 1].axis('off')
            ax[0, 1].axis('tight')
            table = ax[0, 1].table(
                cellText = metrics_df.to_numpy(),
                colLabels = metrics_df.columns,
                loc='center',
                cellLoc='center',
                colWidths=list([.1 for _ in range(len(metrics_df.columns))]),
            )
            table.scale(2.5, 1.5)
            table.auto_set_font_size(False)
            table.set_fontsize(10)


        except Exception as e:  # I know it's not the best way to do it
            print(f"Exception with ids: {unixe_idxs}, exception: {e}")
