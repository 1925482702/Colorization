import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os


class HTML:
    """这个HTML类允许我们将图像和文本保存到一个HTML文件中。

    它包括了诸如<add_header>（向HTML文件添加文本标题）、
    <add_images>（向HTML文件添加一行图像）和<save>（将HTML保存到磁盘）等函数。
    它基于Python库'dominate'，这是一个用于创建和操作HTML文档的Python库，使用DOM API。
    """

    def __init__(self, web_dir, title, refresh=0):
        """初始化HTML类

        参数:
            web_dir (str) -- 存储网页的目录。HTML文件将被创建在<web_dir>/index.html；图像将被保存在<web_dir/images/。
            title (str)   -- 网页名称
            refresh (int) -- 网站自我刷新的频率；如果为0；则不刷新
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """返回存储图像的目录"""
        return self.img_dir

    def add_header(self, text):
        """向HTML文件插入标题

        参数:
            text (str) -- 标题文本
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """向HTML文件添加图像

        参数:
            ims (str list)   -- 图像路径的列表
            txts (str list)  -- 在网站上显示的图像名称列表
            links (str list) -- 超链接列表；当单击图像时，将重定向到一个新页面
        """
        self.t = table(border=1, style="table-layout: fixed;")  # 插入一个表格
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        """将当前内容保存到HTML文件"""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':  # 这里展示一个示例用法。
    html = HTML('web/', 'test_html')
    html.add_header('你好，世界')

    ims, txts, links = [], [], []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('文本_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
