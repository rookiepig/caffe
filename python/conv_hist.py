#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Zhang Kang
# @Email: zhangkang-pd@360.cn
# @Date:a2016/09/06
# @Content:  using histogram to visualize model weights
#            input  param: [model_file_name]
#            output param: [hist_img_dir]

import matplotlib
matplotlib.use('Agg')    # do not use X11
import matplotlib.pyplot as plt
import numpy as np
import caffe.proto.caffe_pb2 as pb
import argparse
import sys
import os

def parse_args():
    """ parsing command line arguments
    
    Args:
        
    
    Returns:
        argument class	
    
    """
    parser = argparse.ArgumentParser(description='using histogram to'
            'visualize model weights')
    parser.add_argument('--model_file', dest='model_file',
                        help='full path and file name for model file',
                        default='test.caffemodel', type=str)
    parser.add_argument('--hist_img_dir', dest='hist_img_dir',
                        help='dir for store histogram images',
                        default='/tmp/', type=str)
    parser.add_argument('--bin_num', dest='bin_num',
                        help='histgram bin number',
                        default='64', type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    args = parser.parse_args()
    return args
 
def main():
    args = parse_args()
    # make dir
    if not os.path.isdir(args.hist_img_dir):
        print 'make dir: ' + args.hist_img_dir
        os.mkdir(args.hist_img_dir)
    # load model
    net = pb.NetParameter.FromString(open(args.model_file).read())
    # iterate all layers
    for layer in net.layer:
        if (layer.type == u'Convolution'):
            print '\t name: ' + layer.name
            print '\t type: ' + layer.type
            output_channel = layer.blobs[0].shape.dim[0]
            input_channel  = layer.blobs[0].shape.dim[1]
            kernel_size = layer.blobs[0].shape.dim[2]
            tmp = layer.blobs[0].shape.dim[3]
            if (tmp != kernel_size):
                print 'Error: kernel not square'
                exit()
            print '\t output_channel: ' + str(output_channel)
            print '\t input_channel: ' + str(input_channel)
            print '\t kernel_size: ' + str(kernel_size)
            plt.hist(layer.blobs[0].data, bins=args.bin_num)
            plt.title(layer.name)
            # save fig
            file_prefix = layer.name.replace("/", "_")
            print '\t save to file: ' + file_prefix
            plt.savefig(os.path.join(args.hist_img_dir, file_prefix + '.png'))
            plt.close()
            print '\n'

if __name__ == "__main__":
    main()
