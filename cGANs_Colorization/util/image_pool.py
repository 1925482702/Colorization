import random
import torch


class ImagePool():
    """这个类实现了一个存储先前生成图像的图像缓冲区。

    这个缓冲区使我们能够使用先前生成的图像历史来更新判别器，而不是使用最新生成器生成的图像。
    """

    def __init__(self, pool_size):
        """初始化ImagePool类

        参数:
            pool_size (int) -- 图像缓冲区的大小，如果pool_size=0，则不会创建缓冲区
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # 创建一个空的缓冲区
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """从缓冲区返回一个图像。

        参数:
            images: 生成器生成的最新图像

        返回从缓冲区返回的图像。

        以50/100的概率，缓冲区将返回输入图像。
        以50/100的概率，缓冲区将返回先前存储在缓冲区中的图像，并将当前图像插入到缓冲区中。
        """
        if self.pool_size == 0:  # 如果缓冲区大小为0，则不做任何操作
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # 如果缓冲区未满；继续将当前图像插入到缓冲区中
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # 以50%的概率，缓冲区将返回一个先前存储的图像，并将当前图像插入到缓冲区中
                    random_id = random.randint(0, self.pool_size - 1)  # randint是包含的
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # 另外50%的概率，缓冲区将返回当前图像
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # 收集所有图像并返回
        return return_images
