import matplotlib.pyplot as plt
import json

# 提供的数据示例
data = [
    {
        "data": [
            {
                "x": [0], "y": [0], "name": "default", "type": "scatter", "mode": "lines",
                "textposition": "right", "line": {}, "marker": {"size": 10, "symbol": "dot", "line": {"color": "#000000", "width": 0.5}}
            }
        ],
        "win": "window_3d2ec7174b55ca",
        "eid": "acrobot--novelty_reward_with_d_step_q_planning_2024-07-28--12-23-40_0",
        "layout": {
            "showlegend": False, "title": "Default Plot", "margin": {"l": 60, "r": 60, "t": 60, "b": 60},
            "xaxis": {"title": "training_steps"}, "yaxis": {"title": "hashed_unique_state_counts"}
        },
        "opts": {"title": "Default Plot", "xlabel": "training_steps", "ylabel": "hashed_unique_state_counts"}
    },
    {
        "data": [
            {
                "x": [0.0], "y": [-1.0], "name": "default", "type": "scatter", "mode": "lines",
                "textposition": "right", "line": {}, "marker": {"size": 10, "symbol": "dot", "line": {"color": "#000000", "width": 0.5}}
            }
        ],
        "win": "window_3d2ec71781322e",
        "eid": "acrobot--novelty_reward_with_d_step_q_planning_2024-07-28--12-23-40_0",
        "layout": {
            "showlegend": False, "title": "Extrinsic Rewards", "margin": {"l": 60, "r": 60, "t": 60, "b": 60},
            "xaxis": {"title": "training_steps"}, "yaxis": {"title": "extrinsic_rewards"}
        },
        "opts": {"title": "Extrinsic Rewards", "xlabel": "training_steps", "ylabel": "extrinsic_rewards"}
    }
]

# 将数据解析并绘制图表
for plot_data in data:
    for trace in plot_data["data"]:
        x = trace["x"]
        y = trace["y"]
        plt.plot(x, y, label=trace["name"], marker=trace["marker"]["symbol"])

    layout = plot_data["layout"]
    plt.title(layout["title"])
    plt.xlabel(layout["xaxis"]["title"])
    plt.ylabel(layout["yaxis"]["title"])
    plt.legend()
    plt.show()
