import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 3)
np.set_printoptions(precision=3)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("mkdir at {}".format(path))
    return path


# generate a dict from a list of yaml files in config_names
def config_args(*config_names):
    config_dicts = dict()
    for config_name in config_names:
        if config_name is not None:
            with open(os.path.join(os.path.dirname(__file__), "../", "{}.yaml".format(config_name)),
                      "r", encoding="UTF-8") as f:
                try:
                    config_dict = yaml.load(f, Loader=yaml.Loader)
                    if config_dict:
                        config_dicts.update(config_dict)
                except yaml.YAMLError as exc:
                    assert False, "{}.yaml error: {}".format(config_name, exc)
    return config_dicts


def save_args_to_json(args, path, display=True):
    with open(path, "w") as f:
        json.dump(args, f, indent=4)
    if display:
        print(json.dumps(args, indent=4))
    print("Args have saved at", path)


def load_args_from_json(path, display=True):
    with open(path, 'r') as f:
        args = json.load(f)
    if display:
        print(json.dumps(args, indent=4))
    # print("Args have loaded at", path)
    return args


def plot_trace(*aircrafts, render_info=None):
    if render_info is None:
        return
    aircrafts = list(aircrafts)
    figure = plt.figure()
    ax = figure.gca(projection="3d")
    legend_list = []
    for ac in aircrafts:
        x = [point[0] for point in ac.pos_list]
        y = [point[1] for point in ac.pos_list]
        z = [point[2] for point in ac.pos_list]
        ax.plot3D(x, y, z, color=ac.side)
        ax.scatter(x[0], y[0], z[0], marker='*', color=ac.side)
        # ax.scatter(x[-1], y[-1], z[-1], marker='x', color=ac.side)
        ax.text(x[0], y[0], z[0], "{:.0f} {:.0f} {:.0f} ".format(x[0], y[0], z[0]))
        legend_list.append(str(len(ac.pos_list)) + "steps")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(legend_list)
    title = "aircraft trajectory of red and blue side\n"
    if render_info.get("winner", None) and render_info.get("done_info", None):
        title += render_info["winner"] + " win with: " + render_info["done_info"]
    ax.set_title(title)
    # plt.style.use("ggplot")
    if render_info.get("save", False):
        plt.savefig(render_info["save_path"])
        print("position fig saved at", render_info["save_path"])
    if render_info["render"]:
        plt.show()
    plt.close()


def plot(data_list, xlabel="x", ylabel="y", info=None):
    # plot_info = {"render": args["render"], "save": args["save"], "save_path": ""}
    if info is None:
        return
    plt.clf()
    plt.plot(range(len(data_list)), data_list, color="k")
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if "acc" in ylabel:
        plt.ylim(0, 1)
    if info.get("title", False):
        plt.title(info["title"])

    if info.get("save", False):
        save_name = info.get("save_name", "") + "_" + ylabel
        save_path = info["save_path"] + save_name
        plt.savefig(save_path)
        print("fig saved at", save_path)
    if info.get("render", False):
        plt.show()
    plt.close()


def plot_rate(win_rate, ylabel, info=None):
    plt.clf()
    red = [rate["red"] for rate in win_rate]
    blue = [rate["blue"] for rate in win_rate]
    draw = [rate["draw"] for rate in win_rate]

    plt.plot(range(len(win_rate)), red, color="r")
    plt.plot(range(len(win_rate)), blue, color="b")
    plt.plot(range(len(win_rate)), draw, color="k")
    plt.grid()
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    legend_list = ["red", "blue", "draw"]
    plt.legend(legend_list)

    if info["render"]:
        plt.show()
    if info["save"]:
        save_name = info["save_name"] + "_" + ylabel
        save_path = info["save_path"] + save_name
        plt.savefig(save_path)
        print("fig saved at", save_path)
    plt.close()


def plot_parallel(reward_dict, xlabel="x", ylabel="y", info=None):
    # plot_info = {"render": args["render"], "save": args["save"], "save_path": ""}
    if info is None:
        return
    plt.clf()
    rewards = []
    for key, reward_list in reward_dict.items():
        plt.plot(reward_list, label=key, linewidth=0.5)
        rewards.append(reward_list)

    if info.get("fill", False):
        rewards = np.array(rewards)
        reward_mean = rewards.mean(axis=0)
        reward_std = rewards.std(axis=0)
        reward_upper = list(map(lambda x: x[0] + x[1], zip(reward_mean, reward_std)))
        reward_floor = list(map(lambda x: x[0] - x[1], zip(reward_mean, reward_std)))
        plt.plot(reward_mean, color="r", linewidth=2)
        # plt.plot(reward_upper, label="upper", color="g", linewidth=0.1)
        # plt.plot(reward_floor, label="floor", color="b", linewidth=0.1)
        plt.fill_between(range(len(reward_mean)), reward_floor, reward_upper, facecolor="lightsalmon", alpha=0.5)

    plt.grid()
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if info.get("title", False):
        plt.title(info["title"])

    if info.get("save", False):
        save_name = info.get("save_name", "") + "_" + ylabel
        save_path = info["save_path"] + save_name
        plt.savefig(save_path)
        print("fig saved at", save_path)
    if info.get("render", False):
        plt.show()
    plt.close()


def plot_mean_parallel(reward_dict, xlabel="x", ylabel="y", info=None):
    # plot_info = {"render": args["render"], "save": args["save"], "save_path": ""}
    ylabel_name = {"dqn": "DQN", "d3qn": "D3QN", "d3qn_pi": "D3QN-OAP"}
    if info is None:
        return
    plt.clf()
    for alg_name, reward_list in reward_dict.items():
        rewards = np.array(reward_list)
        reward_mean = rewards.mean(axis=0)
        reward_std = rewards.std(axis=0)
        reward_upper = list(map(lambda x: x[0] + x[1], zip(reward_mean, reward_std)))
        reward_floor = list(map(lambda x: x[0] - x[1], zip(reward_mean, reward_std)))
        # plt.plot(reward_mean, label=alg_name.upper().replace("_", "-"), color=info["line_colors"][alg_name], linewidth=2)
        plt.plot(reward_mean, label=ylabel_name[alg_name], color=info["line_colors"][alg_name], linewidth=2)
        # plt.fill_between(range(len(reward_mean)), reward_floor, reward_upper, facecolor="pink", alpha=0.2)
        # plt.fill_between(range(len(reward_mean)), reward_floor, reward_upper, facecolor="lightsalmon", alpha=0.2)
        plt.fill_between(range(len(reward_mean)), reward_floor, reward_upper, facecolor=info["line_colors"][alg_name], alpha=0.1)

    plt.legend()
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if info.get("title", False):
        plt.title(info["title"])

    if info.get("save", False):
        save_name = info.get("save_name", "") + "_" + ylabel
        save_path = info["save_path"] + save_name
        plt.savefig(save_path)
        print("fig saved at", save_path)
    if info.get("render", False):
        plt.show()
    plt.close()


def analyse_statistics(path, fields1=["win_rate"], fields2=["reward"]):  # 清洗数据
    results = load_args_from_json(path, display=False)
    datas = {}
    for field in fields1:
        print("-----------------------{}-----------------------".format(field))
        for stu, result in results[field].items():  # stu
            datas[stu] = pd.DataFrame(data=result).T
            print(stu, datas[stu], sep="\n", end="\n\n")

    datas = {}
    for field in fields2:
        print("-----------------------{}-----------------------".format(field))
        for stu, result in results[field].items():  # stu
            datas[stu] = pd.DataFrame(columns=["min", "average", "max", "std"], index=list(result.keys()))
            for name, lst in result.items():
                datas[stu]["min"][name] = np.min(lst)
                datas[stu]["average"][name] = np.mean(lst)
                datas[stu]["max"][name] = np.max(lst)
                datas[stu]["std"][name] = np.std(lst)
            print(stu, datas[stu], sep="\n", end="\n\n")
    return datas


def analyse_eval_info(path, eval_field, init_mode, results):
    eval_info = load_args_from_json(path, display=False)
    blue_list = ["normal", "random", "rule"]
    convert = lambda lst: np.mean(lst) if isinstance(lst, list) else lst
    for field in eval_field:
        value_dict = {blue: {} for blue in blue_list}
        for name, field_value_dict in eval_info[field][init_mode].items():
            red, blue = name.replace(" ", "").split("V")
            # print(red, blue, v)
            if blue == "random":
                value_dict["random"][blue] = convert(field_value_dict)
            if "normal" in blue:
                value_dict["normal"][blue] = convert(field_value_dict)
            if "rule" in blue:
                value_dict["rule"][blue] = convert(field_value_dict)
        for blue, values in value_dict.items():
            # print(blue, values)
            if field in ["reward", "done_step"]:
                # results[field][alg_name][blue].append(np.mean(list(values.values())))
                results[field][blue].append(pd.Series(values).mean())
            if field in ["win_rate"]:
                # print(field, values.keys())
                results[field][blue].append(pd.DataFrame(values).mean(axis=1).to_dict())
    return results


def smooth(data, smooth_type, seq_len, index):
    if smooth_type == "average":
        return np.average(data[index:index + seq_len])
    elif smooth_type == "min":
        return np.min(data[index:index + seq_len])
    elif smooth_type == "max":
        return np.max(data[index:index + seq_len])
    elif smooth_type == "median":
        return np.median(data[index:index + seq_len])


def sliding_smooth(origin_data, smooth_type, seq_len=20):
    index_list = [_ for _ in range(len(origin_data) - seq_len)]
    from functools import partial
    return list(map(partial(smooth, origin_data, smooth_type, seq_len), index_list))


def weight_smooth(origin_data, weight=0.9):
    plot_data = [origin_data[0]]
    for point in origin_data:
        smoothed_val = plot_data[-1] * weight + (1 - weight) * point
        plot_data.append(smoothed_val)
    return plot_data


"""
rename for file
"""


def rename(path, condition=lambda x: "" in x, new_name=lambda x: x):
    for dirs in os.listdir(path):
        p = path + dirs
        if os.path.isdir(p):
            # print(p + " is dir")
            if condition(p):
                new = new_name(p)
                print("rename dir {} to {}".format(p, new))
                # os.rename(p, new)
            else:
                rename(p + "/", condition, new_name)
        else:
            # print(p + " is file")
            dir_, file_name = os.path.split(p)
            if condition(file_name):
                new = "{}/{}".format(dir_, new_name(file_name))
                print("rename file {} to {}".format(p, new))
                # os.rename(p, new)


"""
remove for dir
"""


def remove(path, condition=lambda x: "" in x):
    for dirs in os.listdir(path):
        p = path + dirs
        if os.path.isdir(p):
            # print(p + " is dir")
            if condition(p):
                remove(p + "/")  # todo 递归删除文件
                print("remove {}".format(p))
            else:
                remove(p + "/", condition)  # todo 递归查找删除
        else:
            # print(p + " is file")
            dir_, file_name = os.path.split(p)
            if condition(file_name):
                os.remove(p)  # todo 删除文件
                print("remove {}".format(p))
    if condition(path):
        os.rmdir(path)
        print("remove {}".format(path))


if __name__ == '__main__':
    # remove("../results/d3qn_pi_soccer_1v1/", condition=lambda x: "parallel" in x and os.path.isdir(x))
    remove("../", condition=lambda x: "__pycache__" in x and os.path.isdir(x))
