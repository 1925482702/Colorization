"""通用的图像到图像翻译训练脚本。

该脚本适用于不同的模型（通过选项 '--model'，例如 pix2pix、cyclegan、colorization）和不同的数据集（通过选项 '--dataset_mode'，例如 aligned、unaligned、single、colorization）。
您需要指定数据集（'--dataroot'）、实验名称（'--name'）和模型（'--model'）。

它首先根据选项创建模型、数据集和可视化器。
然后进行标准的网络训练。在训练过程中，它还会可视化/保存图像、打印/保存损失图表，并保存模型。
脚本支持继续/恢复训练。使用 '--continue_train' 来恢复先前的训练。

示例:

    训练一个 pix2pix 模型:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

"""

import time
from cGANs_Colorization.options.train_options import TrainOptions
from cGANs_Colorization.data import create_dataset
from cGANs_Colorization.models import create_model
from cGANs_Colorization.util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # 获取训练选项
    dataset = create_dataset(opt)  # 根据 opt.dataset_mode 和其他选项创建数据集
    dataset_size = len(dataset)    # 获取数据集中的图像数量
    print('The number of training = %d' % dataset_size)

    model = create_model(opt)      # 根据 opt.model 和其他选项创建模型
    model.setup(opt)               # 常规设置：加载和打印网络；创建调度器
    visualizer = Visualizer(opt)   # 创建一个可显示/保存图像和图表的可视化器
    total_iters = 0                # 训练迭代的总数

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # 外部循环不同的时代；我们通过 <epoch_count>, <epoch_count>+<save_latest_freq> 来保存模型
        epoch_start_time = time.time()  # 整个时代的计时器
        iter_data_time = time.time()    # 每次迭代的数据加载计时器
        epoch_iter = 0                  # 当前时代的训练迭代次数，在每个时代开始时重置为0
        visualizer.reset()              # 重置可视化器

        for i, data in enumerate(dataset):  # 内部循环在一个时代内
            iter_start_time = time.time()  # 每次迭代的计算计时器
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # 加载数据并进行预处理
            model.optimize_parameters()   # 计算损失、获取梯度、更新权重

            if total_iters % opt.display_freq == 0:   # 在 visdom 上显示图像并将图像保存到 HTML 文件
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # 打印训练损失和保存日志
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # 每 <save_latest_freq> 次迭代缓存我们的最新模型
                print('saving latest model (time %d, total num %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # 每 <save_epoch_freq> 个时代缓存我们的模型
            print('saving latest model (time %d, total num %d)' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('time %d / %d end \t use: %d s' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # 每个时代的最后更新学习率
