import numpy as np
import matplotlib.pyplot as plt 
import argparse

start_plot_idx = 1
def parse_args():
   parser = argparse.ArgumentParser(description='show loss acc')
   parser.add_argument('--file', dest='lossfile', help='the model', default='loss.refine', type=str)
   args = parser.parse_args()
   return args

def get_loss_ganG(loss_str,start,end):
   str_start_idx = loss_str.find(start)+6
   str_end_idx = loss_str.find(end)-1
   tmp = loss_str[str_start_idx+1:str_end_idx]
   return float(tmp)

def get_loss_ganDreal(loss_str,start,end):
   str_start_idx = loss_str.find(start)+7
   str_end_idx = loss_str.find(end)-1
   tmp = loss_str[str_start_idx+1:str_end_idx]
   return float(tmp)

def get_loss_L1(loss_str,start,end):
   str_start_idx = loss_str.find(start)+5
   str_end_idx = loss_str.find(end)-1
   tmp = loss_str[str_start_idx+1:str_end_idx]
   return float(tmp)

def show_loss(lossfile,statistic_interval,lineset,scale=1.0,loss_type='ganG'):
   loss_file = open(lossfile, 'r')
   loss_total = loss_file.readlines()
   loss_num = len(loss_total)
   loss_res = np.zeros(loss_num)
   loss_idx = np.arange(loss_num)

   for idx in range(loss_num) :
       loss_str = loss_total[idx]
       if loss_type == 'ganG':
           loss_res[idx] = scale * get_loss_ganG(loss_str,start='G_GAN:',end='G_L1')
       elif loss_type == 'L1':
           loss_res[idx] = scale * get_loss_L1(loss_str,start='G_L1:',end='D_real:')
       elif loss_type == 'D_real':
           loss_res[idx] = scale * get_loss_ganDreal(loss_str,start='D_real:',end='D_fake:')
       
   statistic_len = int((loss_num + statistic_interval - 1)/statistic_interval)
   statistic_idx = np.arange(statistic_len) * statistic_interval
   statistic_res_mean = np.zeros(statistic_len)
   
   for idx in range(statistic_len) :
       loss_start_idx = idx*statistic_interval
       loss_end_idx = min(loss_start_idx + statistic_interval, loss_num)
       loss_part = loss_res[loss_start_idx : loss_end_idx]
       statistic_res_mean[idx] = np.mean(loss_part)
       
   plt.plot(statistic_idx[start_plot_idx:], statistic_res_mean[start_plot_idx:],lineset)

if __name__ == '__main__':
    args = parse_args()
    show_loss('loss.refine',1,'r-',loss_type='ganG')
    show_loss('loss.refine',1,'g-',loss_type='L1')
    show_loss('loss.refine',1,'b-',loss_type='D_real')
    show_loss('loss.refine',1,'k-')
    plt.legend(('G GAN loss','L1 loss','D real loss','w total loss'))
    plt.title('train_loss')
    plt.xlabel('niters')
    plt.show()
