import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path
import seaborn as sns


def find_metric(line, names):
    for m_n in names:
        if m_n in line:
            value = float(line.split(':')[1].strip())
            # value = float(line.split(':')[-1].strip())
            return m_n, value
    return None


def parse_log(p, n, names=['Average Travel Time']):
    print(f"Parsing log: {n}")
    with open(str(p / n), 'r') as f:
        lines = f.readlines()
        episodes = []
        metric_list = []
        # metric_list = defaultdict(list)

        log_length = len(lines)
        # divide each episode
        line_numbs = []
        for line_num in range(log_length):
            if 'episode' and 'steps' in lines[line_num]:
            # if 'episode' in lines[line_num][:7]:
                line_numbs.append(line_num)
        line_numbs.append(log_length)

        for i in range(len(line_numbs) - 1):
            cur_lines = lines[line_numbs[i]:line_numbs[i+1]]
            for line in cur_lines:
                if 'episode' in line and 'steps' in line:
                # if 'episode' in line[:7]:
                    items = line.split(',')[0].split(':')
                    episode_num = int(items[1].split('/')[0])
                    episodes.append(episode_num)
                    # res = find_metric(line, names)
                    # if res:
                    #     m_n, v = res
                    #     # metric_list[m_n].append(v)
                    #     metric_list.append(v)
                else:
                    res = find_metric(line, names)
                    if res:
                        m_n, v = res
                        # metric_list[m_n].append(v)
                        metric_list.append(v)
        print(len(episodes))
        # print(len(metric_list))
        # print(np.min(metric_list))
        # episodes = episodes[:100]
        # metric_list = metric_list[:100]
        return np.array(episodes), np.array(metric_list)


# jinan compare ma_none, ma_federated, ma_distributed
# synthetic compare queue_size ma_none, ma_federated, ma_distributed and gat_rnn_ma_none
if __name__ == '__main__':
    # file_path = Path("log/dqn")
    # file_path = Path("log/backup/without_yellow_light")
    file_path = Path("log/backup/dqn_wphase")
    # file_path = Path("log/backup/with_yellow_light")
    # file_path = Path("log/dqn_sa")

    # Jinan_3_4
    # file_names = ["20220930-222753_jinan_ma_none.log",
    #               "20221005-111822_jinan_ma_federated.log", "20221005-111825_jinan_ma_distributed.log"]
    # file_names = ["20220930-222753_jinan_ma_none.log", "20221005-111822.log", "20221005-111825.log"]
    # plot_labels = ['GAT_RNN_map_ma_none', 'GAT_RNN_map_ma_federated', 'GAT_RNN_map_ma_distributed']
    # file_names = ["FC_queue.out", "GAT_RNN_queue.out", "CNN_map.out",
    #               "CNN_RNN_map.out", "20220930-222753_jinan_ma_none.log"]
    # plot_labels = ['FC_Queue', 'GAT_RNN_Queue', 'CNN_Map', 'CNN_RNN_map', 'GAT_RNN_Map_ma_none']
    # plot_labels = ['FC_Queue', 'RNN_Queue', 'CNN_Map', 'CNN_RNN_Map', 'GAT+CNN_Map+RNN_Queue']
    # file_names = ["20221021-011431.log", "20221021-105719.log", "20221021-211938.log", "20221022-004737.log"]
    # plot_labels = ['Dueling', 'Vanilla', 'Prioritized Experience Replay', 'Dueling+PER']

    # With yellow light phase
    # file_names = ["20221107-121801.log", "20221107-121828.log", "20221107-121839.log",
    #               "20221115-201652.log", "20221116-222341.log", "20221116-222617.log"]
    # file_names = ["20221114-184502.log", "20221114-184513.log", "20221114-184524.log",
    #               "20221115-201652.log", "20221116-222341.log", "20221116-222617.log"]
    # plot_labels = ['CNN_Map_None', 'CNN_Map_Federated', 'CNN_Map_Distributed',
    #                'GAT_RNN_CNN_None', 'GAT_RNN_CNN_Federated', 'GAT_RNN_CNN_Distributed']
    # file_names = ["20221107-121801.log", "20221107-121828.log", "20221107-121839.log"]
    # file_names = ["20221108-114135.log", "20221108-114140.log", "20221112-104225.log"]
    # plot_labels = ['CNN_Map_None', 'CNN_Map_Federated', 'CNN_Map_Distributed']
    # plot_labels = ['GAT_RNN_CNN_Federated', 'GAT_RNN_CNN_Distributed', 'GAT_RNN_CNN_None']
    file_names = [
                  "20221219-194939_CNN_None.log",
                  # "20221220-223720_CNN_Federated.log",
                  # "20221220-223730_CNN_Distributed.log",
                  "20221228-154254_RNN_None.log",
                  # "20221228-154259_RNN_Federated.log", "20221228-154307_RNN_Distributed.log",
                  # "20221226-164534_Colight.log", "20221226-173230_MPLight.log", "20221226-175358_PressLight.log",
                ]
    plot_labels = [fn.split('.')[0][16:] for fn in file_names]

    # file_names = ["FC_ma_none.log", "FC_ma_Federated.log", "FC_ma_Distributed.log"]
    # plot_labels = ["FC_ma_none", "FC_ma_Federated", "FC_ma_Distributed"]
    # file_names = ["CNN_ma_none_w_LN.log", "CNN_ma_Federated_w_LN.log", "CNN_ma_Distributed_w_LN.log"]
    # file_names = ["20221031-155922.log", "20221031-155923.log", "20221031-155924.log"]
    # plot_labels = ["CNN_ma_none", "CNN_ma_Federated", "CNN_ma_Distributed"]
    # file_names = ["20221101-113621.log", "20221101-141345.log", "20221101-141435.log", "20221101-141525.log"]
    # plot_labels = ["CNN_ma_none_with_LN", "CNN_ma_Federated_wo_LN", "CNN_ma_Distributed_wo_LN", "CNN_ma_none_wo_LN"]
    # file_names = ["20221102-093351.log", "20221102-093352.log", "20221102-093354.log"]
    # file_names = ["20221102-093351.log", "20221106-205206.log", "20221102-093354.log"]
    # file_names = ["20221107-121801.log", "20221107-121828.log", "20221107-121839.log"]
    # file_names = ["20221104-090836.log", "20221104-090843.log", "20221104-090850.log"]
    # plot_labels = ["CNN_ma_none", "CNN_ma_Federated", "CNN_ma_Distributed"]
    # file_names = ["FC_ma_none.log", "CNN_ma_none.log", "FC_ma_Federated.log", "CNN_ma_Federated.log",
    #               "FC_ma_Distributed.log", "CNN_ma_Distributed.log"]
    # plot_labels = ["FC_ma_none", "CNN_ma_none", "FC_ma_Federated", "CNN_ma_Federated",
    #                "FC_ma_Distributed", "CNN_ma_Distributed"]

    # Synthetic_4_4
    # file_names = ["20220930-223126_synthetic_ma_none.log", "synthetic_map_ma_none.log",
    #               "synthetic_map_ma_federated.log", "synthetic_map_ma_distributed.log"]
    # plot_labels = ['GAT_RNN_Map_ma_none', 'CNN_Map_ma_none', 'CNN_Map_ma_federated', 'CNN_Map_ma_distributed']
    # fig_title = "Travel Average Time on the Synthetic_4_4 dataset (training curve)"

    # just repeatedly iterate over the file
    # metric_names = ['Average Travel Time', 'Lane waiting count', 'Average Throughput']
    # metric_names = ['Average Travel Time', 'Average Speed Score', 'Average Throughput', 'Lane waiting count']
    # metric_names = ['average travel time']
    metric_names = ['Average Travel Time']
    for m_i, metric_name in enumerate(metric_names):
        fig_title = f"{metric_name} on the Jinan_3_4 dataset (training curve)"
        fig = plt.figure(figsize=(12, 8))
        plt.style.use(('science', 'no-latex', 'bright'))
        plt.rcParams.update({
            # "font.family": "serif",  # specify font family here
            # "font.serif": ["Times"],  # specify font here
            "font.size": 13})  # specify font size here
        plt.title(fig_title)
        plt.xlabel('Episode num')
        plt.ylabel(metric_name)
        for i in range(len(file_names)):
            file_name = file_names[i]
            e_nums, metrics = parse_log(file_path, file_name, [metric_name])
            # if plot_labels[i] == 'CNN_Map' and metric_name == 'Average Travel Time':
            #     metrics += 3.5
            # if plot_labels[i] == 'CNN_Map' and metric_name == 'Average Speed Score':
            #     metrics -= 0.01
            # if plot_labels[i] == 'CNN_Map' and metric_name == 'Average Throughput':
            #     metrics -= 20
            print(plot_labels[i])
            print(f'min {metric_name}', min(metrics))
            plt.plot(e_nums, metrics, label=plot_labels[i])

        plt.legend()
        plt.savefig(f"test_{m_i}.png", bbox_inches='tight', transparent=True)
        plt.show()
