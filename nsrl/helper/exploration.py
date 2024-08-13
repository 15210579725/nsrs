"""
Exploration helpers. Requires PyTorch
"""
import torch
from nsrl.helper.knn import ranked_avg_knn_scores, avg_knn_scores, batch_knn, batch_count_scaled_knn
from nsrl.helper.pytorch import device, calculate_large_batch

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from nsrl.learning_algos.space_cover import compute_space_coverage
import os


def calculate_unpredictability_estimate(states, target_network, predictor_network):
    """
    Calculating unpredictability of given states.

    Parameters
    ----------
    states: States to calculate scores from of size [batch_size x (state_dim)]
    target_network: randomly initialized fixed encoder network
    predictor_network: network to try and predict predictor network.

    Returns
    -------
    Scores of size [batch_size]
    """
    pass


def calculate_scores(states, memory, encoder=None, k=10, dist_score=ranked_avg_knn_scores,
                     knn=batch_count_scaled_knn, plotter=None, _count = None):
    """
    Calculating KNN scores for each of the states. We want to
    optionally encode ALL the states in the buffer to calculate things.

    Parameters
    ----------
    states: States to calculate scores from of size [batch_size x (state_dim)]
    encoder: Encoder that takes in [batch_size x (state_dim)] and returns [batch_size x encoded_size]

    Returns
    -------
    Scores of size [batch_size]
    """
    # don't calculate gradients here!
    # print("states shape: ", states.shape)
    # print("memory shape: ", memory.shape)


    with torch.no_grad():
        if encoder is None:
            encoder = lambda x: x

        # one big bad batch
        # encoded_memory = encoder(torch.tensor(memory, dtype=torch.float).to(device))
        encoded_memory = calculate_large_batch(encoder, memory)

        encoded_states = states
        # REFACTOR THIS
        if encoded_states.shape[-1] != encoded_memory.shape[-1]:
            # encoded_states = encoder(torch.tensor(states, dtype=torch.float).to(device))
            encoded_states = calculate_large_batch(encoder, states)
        scores = dist_score(encoded_states.cpu().detach().numpy(),
                            encoded_memory.cpu().detach().numpy(), k=k,
                            knn=knn)
        if encoded_states.shape[0] != 1 and plotter:
            print("encoded_states shape: ", encoded_states.shape)
            plotter.plot("newest knn scores", np.array([_count]), np.array([scores[-1]]), "newest knn scores")
            plotter.plot("knn scores 0", np.array([_count]), np.array([scores[0]]), "knn scores 0")

    return scores




def reshape_data(data:np.array):
    '''把数据除了第一维以外全部压缩成一维, 并且转置, 返回dim*bs'''
    shape = data.shape
    data = data.reshape(shape[0], -1) 
    return data.T

def calculate_scores_kde(states, memory, encoder=None, band_witdth=None, k=10, dist_score=ranked_avg_knn_scores,
                     knn=batch_count_scaled_knn, plotter = None, _count = None):    #后三个参数保持接口一致，位置参数是必须传这个参数，关键字是可以调换参数，也可以没有，必须在后面
    """
    Calculating KDE scores for each of the states. 
    We want to
    optionally encode ALL the states in the buffer to calculate things.

    Parameters
    ----------
    states: States to calculate scores from of size [batch_size x (state_dim)]
    encoder: Encoder that takes in [batch_size x (state_dim)] and returns [batch_size x encoded_size]
    batch_size: batch size of the states ie.[62, 4]
    state_dim: state dimension ie. [32, 32]

    Returns
    -------
    Scores of size [batch_size]
    """
    # don't calculate gradients here!
    with torch.no_grad():
        if encoder is None:
            encoder = lambda x: x

        # one big bad batch
        encoded_memory = encoder(torch.tensor(memory, dtype=torch.float).to(device))
        #small_batch 32
        # encoded_memory = calculate_large_batch(encoder, memory)

        if encoded_memory.shape[0] < np.prod(encoded_memory.shape[1:-1]):
            raise ValueError("!!!!!!!!!!!!!!!Encoded memory batch size too small!!!!!!!!!!!!!")

        encoded_states = states
        # REFACTOR THIS
        if encoded_states.shape[-1] != encoded_memory.shape[-1]:
            # encoded_states = encoder(torch.tensor(states, dtype=torch.float).to(device))
            encoded_states = calculate_large_batch(encoder, states)
        

        
        #calculate kde
        reshape_states = reshape_data(encoded_states.cpu().detach().numpy())
        reshape_memory = reshape_data(encoded_memory.cpu().detach().numpy())

        # 保存数组到文件中
        if _count:
            save_dir = './encode states/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = f'step_{_count}.npy'
            file_path = os.path.join(save_dir, file_name)
            np.save(file_path, reshape_memory.T)
            print("Saved array to file:", file_path)

        # 从文件中加载数组
        # loaded_array = np.load('array.npy')
        # print("reshape_states", reshape_states)
        # print("reshape_memory", reshape_memory)
        # max_values = np.max(reshape_memory, axis=1)
        # min_values = np.min(reshape_memory, axis=1)
        # mean_values = np.mean(reshape_memory, axis=1)
        # print("max_values", max_values)
        # print("min_values", min_values)
        # print("mean_values", mean_values)

        coverage = compute_space_coverage(reshape_memory.T,  0.2)
        print("coverage", coverage * 100.0, "%")

        # print("reshape_states.shape", reshape_states.shape)
        # print("reshape_memory.shape", reshape_memory.shape)


        kde = gaussian_kde(reshape_memory)    #dims * bs
        result = kde.evaluate(reshape_states) #dims * bs

        scores = 1 / result
        log_scores = np.log(scores)
        # print("result", result)
        # print("scores", scores)
        # print("log scores", log_scores)
        # self._plotter.plot("intrinsic_mean_rewards", np.array([self._count]), [np.mean(intr_rewards)], title_name="Intrinsic Rewards")
        if plotter:
            plotter.plot("result0", np.array([_count]), np.array([result[0]]), title_name="result0")
            plotter.plot("density newest", np.array([_count]), np.array([result[-1]]), title_name="density newest")
            plotter.plot("coverage", np.array([_count]), np.array([coverage]), title_name="coverage")

    return log_scores



if __name__ == '__main__':
    # batch_size = [5,1]
    # query_size = [3, 1]
    # data_dim = [2]
    
    # # data = np.array([[0,0], [1,1], [1,-1],[-1,1],[-1,-1]])
    # # query = np.array([[-1, 0], [0, 1], [-2, -1]])
    # data = np.random.rand(13,4,3,3)
    # query = np.random.rand(4,4,3,3)
    
    
    
    # data = torch.tensor(data).to('cuda')
    # query = torch.tensor(query).to('cuda')
    # calculate_scores_kde(query, data, encoder=None, band_witdth=None)

    # 创建一个NumPy数组
    array = np.array([1, 2, 3, 4, 5])

    # 保存数组到文件中
    np.save('array.npy', array)

    # 从文件中加载数组
    loaded_array = np.load('array.npy')

    # 打印加载的数组
    print(loaded_array)