import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import binned_statistic_dd
import umap
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib
from moviepy.editor import VideoFileClip

def compute_space_coverage(data, dx):
    """
    归一化数据、离散化并计算空间覆盖率。
    
    参数:
    data: ndarray, 输入数据，形状为(N, dim)
    dx: float, 离散化步长
    
    返回:
    coverage_rate: float, 空间覆盖率
    """
    # 步骤1: 归一化数据到 [0, 1]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    
    # 步骤2: 计算每个维度的bins数量
    n_dims = normalized_data.shape[1]
    bins = [int(1 / dx) for _ in range(n_dims)]  # 因为数据已经归一化，范围为 [0, 1]
    
    # 步骤3: 离散化并统计非空网格数
    hist, edges, binnumber = binned_statistic_dd(normalized_data, None, statistic='count', bins=bins, range=[[0, 1]]*n_dims)
    
    # 计算总的网格数量
    total_grids = np.prod(bins)
    
    # 计算非空的网格数
    non_empty_bins = np.count_nonzero(hist)
    
    # 计算空间覆盖率
    coverage_rate = non_empty_bins / total_grids
    
    return coverage_rate

def umap_visualization(data):
    '''data: [N, dim]'''
    data = np.random.rand(num, dim)  # 替换为实际数据

    # 创建UMAP对象
    reducer = umap.UMAP(n_neighbors=15)

    # 执行降维
    embedding = reducer.fit_transform(data)

    # 可视化
    #使用Plotly进行可视化
    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        title='UMAP Dimensionality Reduction'
    )

    fig.update_layout(
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2'
    )

    # 显示图形
    fig.show()

def make_umap_gif():
    '''自动提取目录下的所有npy文件，并生成umap图，最后合成gif'''
    # 设置目录路径
    data_dir = '/home/user/Desktop/nsrs/examples/gym/experiments/acrobot--novelty_reward_with_d_step_q_planning_2024-08-23--22-57-25_0/encode states/'
    output_gif_path = data_dir + 'umap.gif'

    # 获取目录中的所有.npy文件
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    npy_files.sort()  # 根据需要排序文件

    # 保存每个图的图像路径
    image_files = []
    cnt = 0
    for npy_file in npy_files:
        # 加载数据
        cnt += 1
        data = np.load(os.path.join(data_dir, npy_file))

        # 截取前两个维度
        data = data[:, :2]

        # 创建UMAP对象并降维
        reducer = umap.UMAP(n_neighbors=50)
        embedding = reducer.fit_transform(data)

        # 可视化并保存图像
        fig = px.scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
            title=f'UMAP Reduction for {npy_file}'
        )

        fig.update_layout(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2'
        )

        # 保存为图像文件
        image_path = os.path.join(data_dir, f'{npy_file}.png')
        fig.write_image(image_path)
        image_files.append(image_path)
        print("finish state ", cnt)

    # 使用PIL将图像合成动图
    images = [Image.open(image_file) for image_file in image_files]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=500,  # 每帧持续时间，毫秒
        loop=0  # 循环次数，0表示无限循环
    )

    print(f'动图已保存至 {output_gif_path}')

def draw_origin_point():
    matplotlib.use('Agg')  # 使用 Agg 后端

    # 加载数据
    data = np.load('/home/user/Desktop/nsrs/examples/gym/experiments/acrobot--novelty_reward_with_d_step_q_planning_2024-08-23--14-15-23_0/encode states/step_145.npy')

    # 假设数据的形状是 (num_points, num_dimensions)
    # 截取前两维
    x = data[:, 0]
    y = data[:, 1]

    # 使用单调的彩色渐变（例如 'plasma'）
    cmap = cm.plasma  # 选择 'plasma' 颜色映射
    colors = cmap(np.linspace(0, 1, len(x)))

    fig, ax = plt.subplots(figsize=(10, 8))

    # 设置自动调整的画布范围
    padding = 0.1  # 画布范围的额外填充
    x_min, x_max = x.min() - padding, x.max() + padding
    y_min, y_max = y.min() - padding, y.max() + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 初始化画布上的散点和箭头
    scat = ax.scatter([], [], color=[], s=50)
    arrows = []

    # 添加颜色条（图例），指定 `ax` 参数
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])  # 必须设置一个空数组
    fig.colorbar(sm, ax=ax, label='Data Point Order')

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('2D Projection of High-dimensional Data with Monotonic Arrows')
    ax.grid(True)

    # 更新每一帧的数据
    def update(frame):
        if frame > 0:
            arrow = ax.arrow(x[frame-1], y[frame-1], x[frame] - x[frame-1], y[frame] - y[frame-1], 
                             head_width=0.05, head_length=0.05, 
                             fc=colors[frame-1], ec=colors[frame-1], linestyle='dashed')
            arrows.append(arrow)
        scat.set_offsets(np.c_[x[:frame+1], y[:frame+1]])
        scat.set_color(colors[:frame+1])

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(x), interval=200, repeat=False)

    # 保存为gif
    ani.save('output_animation.gif', writer='imagemagick')

    plt.close(fig)

def find_last_npy_path(main_path):
    '''查找目录下最后一个npy文件路径'''
    import os
    import re

    # 构建encode states子目录的路径
    encode_states_path = os.path.join(main_path, "encode states")

    # 正则表达式用于匹配step_*.npy格式的文件
    pattern = re.compile(r'step_(\d+)\.npy')

    # 初始化最大步数和相应的文件名
    max_step = -1
    max_step_file = None

    # 遍历encode states子目录中的所有文件
    for filename in os.listdir(encode_states_path):
        match = pattern.match(filename)
        if match:
            # 提取文件名中的数字部分
            step_number = int(match.group(1))
            # 更新最大步数及其文件名
            if step_number > max_step:
                max_step = step_number
                max_step_file = filename
    
    return main_path + "/encode states/" + max_step_file

def draw_origin_point_with_video(main_path):
    matplotlib.use('Agg')  # 使用 Agg 后端
    
    main_path = main_path.replace("\\", "\ ")
    data_path = find_last_npy_path(main_path)
    video_path =  main_path + '/merged_video.mp4'
    
    video_stitching(main_path)
    

    # 加载数据
    
    data = np.load(data_path)
    # 假设数据的形状是 (num_points, num_dimensions)
    # 截取前两维
    print("data shape", data.shape)
    x = data[:, 2]
    y = data[:, 3]

    # 使用单调的彩色渐变（例如 'plasma'）
    cmap = cm.plasma  # 选择 'plasma' 颜色映射
    colors = cmap(np.linspace(0, 1, len(x)))

    # 创建两个子图，一个用于显示点的动图，一个用于播放视频
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # 设置点的动图的画布范围
    padding = 0.1  # 画布范围的额外填充
    x_min, x_max = x.min() - padding, x.max() + padding
    y_min, y_max = y.min() - padding, y.max() + padding
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    # 初始化画布上的散点和箭头
    scat = ax1.scatter([], [], color=[], s=50)
    arrows = []

    # 添加颜色条（图例），指定 `ax1` 参数
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])  # 必须设置一个空数组
    fig.colorbar(sm, ax=ax1, label='Data Point Order')

    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.set_title('2D Projection of High-dimensional Data with Monotonic Arrows')
    ax1.grid(True)

    # 加载视频
    video_clip = VideoFileClip(video_path)
    img = ax2.imshow(video_clip.get_frame(0))  # 初始化显示第一帧
    ax2.axis('off')

    start_frame = 5  # 定义从第5帧开始

    # 更新每一帧的数据和视频
    def update(frame):
        video_frame_index = start_frame + frame  # 从第5帧开始

        if video_frame_index < len(x):  # 确保视频帧索引不超过数据的长度
            if frame > 0:
                arrow = ax1.arrow(x[frame-1], y[frame-1], x[frame] - x[frame-1], y[frame] - y[frame-1], 
                                 head_width=0.05, head_length=0.05, 
                                 fc=colors[frame-1], ec=colors[frame-1], linestyle='dashed')
                arrows.append(arrow)
            scat.set_offsets(np.c_[x[:frame+1], y[:frame+1]])
            scat.set_color(colors[:frame+1])
            img.set_array(video_clip.get_frame(video_frame_index / len(x) * video_clip.duration))
        
        print("frame: ", frame, "/", len(x) - start_frame)

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(x) - start_frame, interval=100, repeat=False)

    # 保存为mp4
    ani.save(main_path + 'original_state_video.mp4', writer='ffmpeg')

    plt.close(fig)

def draw_umap_projection_with_video(file_path):
    matplotlib.use('Agg')  # 使用 Agg 后端

    # 加载高维数据
    data = np.load(file_path)

    # 使用 UMAP 将数据压缩到二维
    reducer = umap.UMAP(n_neighbors=100)
    umap_embedding = reducer.fit_transform(data)

    # 截取前两维（UMAP 结果本身就是二维的）
    x = umap_embedding[:, 0]
    y = umap_embedding[:, 1]

    # 使用单调的彩色渐变（例如 'plasma'）
    cmap = cm.plasma  # 选择 'plasma' 颜色映射
    colors = cmap(np.linspace(0, 1, len(x)))

    # 创建两个子图，一个用于显示点的动图，一个用于播放视频
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # 设置点的动图的画布范围
    padding = 0.1  # 画布范围的额外填充
    x_min, x_max = x.min() - padding, x.max() + padding
    y_min, y_max = y.min() - padding, y.max() + padding
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    # 初始化画布上的散点和箭头
    scat = ax1.scatter([], [], color=[], s=50)
    arrows = []

    # 添加颜色条（图例），指定 `ax1` 参数
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])  # 必须设置一个空数组
    fig.colorbar(sm, ax=ax1, label='Data Point Order')

    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    ax1.set_title('2D UMAP Projection with Monotonic Arrows')
    ax1.grid(True)

    # 加载视频
    video_clip = VideoFileClip("/home/user/Desktop/nsrs/examples/gym/experiments/acrobot--novelty_reward_with_d_step_q_planning_2024-08-23--14-15-23_0/" + 'merged_video.mp4')
    img = ax2.imshow(video_clip.get_frame(0))  # 初始化显示第一帧
    ax2.axis('off')

    start_frame = 5  # 定义从第5帧开始

    # 更新每一帧的数据和视频
    def update(frame):
        video_frame_index = start_frame + frame  # 从第5帧开始

        if video_frame_index < len(x):  # 确保视频帧索引不超过数据的长度
            if frame > 0:
                arrow = ax1.arrow(x[frame-1], y[frame-1], x[frame] - x[frame-1], y[frame] - y[frame-1], 
                                 head_width=0.05, head_length=0.05, 
                                 fc=colors[frame-1], ec=colors[frame-1], linestyle='dashed')
                arrows.append(arrow)
            scat.set_offsets(np.c_[x[:frame+1], y[:frame+1]])
            scat.set_color(colors[:frame+1])
            img.set_array(video_clip.get_frame(video_frame_index / len(x) * video_clip.duration))

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(x) - start_frame, interval=200, repeat=False)

    # 保存为mp4
    ani.save('output_umap_animation_with_video.mp4', writer='ffmpeg')

    plt.close(fig)

def count_frame():
    import cv2
    video_path = "/home/user/Desktop/nsrs/examples/gym/experiments/acrobot--novelty_reward_with_d_step_q_planning_2024-08-23--14-15-23_0/merged_video.mp4"
    video = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not video.isOpened():
        print("Error: Unable to open video file.")
        return 0
    
    # 获取视频的总帧数
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 释放视频对象
    video.release()
    print("frame_count", frame_count)

def video_stitching(main_path):
    import os
    import re
    from moviepy.editor import VideoFileClip, concatenate_videoclips

    # 设置你的视频文件所在的目录
    main_path = main_path.replace("\\", "/")
    

    # 获取目录下所有的mp4文件并按数字顺序排序
    video_files = sorted([f for f in os.listdir( main_path) if f.endswith('.mp4')],
                        key=lambda x: int(re.search(r'\d+', x).group()))

    # 输出排序后的视频文件名称
    print("排序后的视频文件列表:")
    for video in video_files:
        print(video)
        
    # 加载所有的视频文件
    clips = [VideoFileClip(os.path.join( main_path, file)) for file in video_files]

    # 将视频合并成一个
    final_clip = concatenate_videoclips(clips)

    # 导出最终的视频文件
    output_file = os.path.join( main_path, "merged_video.mp4")
    final_clip.write_videofile(output_file, codec="libx264")

    print(f"视频合并完成，已保存为: {output_file}")


if __name__ == "__main__":
   
    # make_umap_gif()
    # draw_origin_point()
    
    #origin point with video
    # main_path = "C:/Users/15210/Desktop/nsrs测试数据/renyi1_avg271-std44/renyi1_sigma05_alpha101_1141_unfinish/acrobot--novelty_reward_with_d_step_q_planning_2024-08-23--14-14-03_0"
    main_path = "C:/Users/15210/Desktop/nsrs测试数据/renyi0+kde_avg1636_std668/renyi0_kde_1214/acrobot--novelty_reward_with_d_step_q_planning_2024-09-04--04-00-18_0"
    draw_origin_point_with_video(main_path)
    
    #umap with video
    # file_path = "C:/Users/15210/Desktop/nsrs测试数据/renyi0+kde/renyi0_kde_2349/acrobot--novelty_reward_with_d_step_q_planning_2024-09-02--08-11-36_0/encode states/step_1523.npy"
    # draw_umap_projection_with_video(file_path=file_path)
    # count_frame()




    # 生成示例数据，替换为你的数据
    # num = 1000
    # dim = 4
    # # data = np.random.rand(num, dim)  # 替换为实际数据
    # data = np.load('/home/user/Desktop/nsrs/examples/gym/experiments/acrobot--novelty_reward_with_d_step_q_planning_2024-08-23--22-57-25_0/encode states/step_87.npy')
    # # data = np.load('/home/user/Desktop/nsrs/examples/gym/experiments/acrobot--novelty_reward_with_d_step_q_planning_2024-08-23--22-57-25_0/action states/all_action.npy')
    # # print("data", data)
    # #cut data to [N, 2]
    # # data = data[:][:2]
    # print("size", data.shape)
    # data = data[:, :2]
    # print("size", data.shape)

    # # 创建UMAP对象
    # reducer = umap.UMAP(n_neighbors=50)

    # # 执行降维
    # embedding = reducer.fit_transform(data)

    # # 可视化
    # #使用Plotly进行可视化
    # fig = px.scatter(
    #     x=embedding[:, 0],
    #     y=embedding[:, 1],
    #     labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
    #     title='UMAP Dimensionality Reduction'
    # )

    # fig.update_layout(
    #     xaxis_title='UMAP Dimension 1',
    #     yaxis_title='UMAP Dimension 2'
    # )

    # # 显示图形
    # fig.show()
