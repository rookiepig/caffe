import caffe.proto.caffe_pb2 as pb


net = pb.NetParameter.FromString(open('/home/zhangkang-pd/project/cnn_acc/faster-rcnn/faster-rcnn-2/models/FD_ML/faster_rcnn_alt_opt_ml/full_cross_gemmlowp_conv1-5_wildface_newdat__uint8_iter_480000.caffemodel').read())


for layer in net.layer:
    if (layer.type == u'Convolution' and layer.convolution_param.use_cross_conv == 1):
        print 'name: ' + layer.name
        print 'type: ' + layer.type
        print 'use_cross_conv: ' + str(layer.convolution_param.use_cross_conv)
        output_channel = layer.blobs[0].shape.dim[0]
        input_channel  = layer.blobs[0].shape.dim[1]
        kernel_size = layer.blobs[0].shape.dim[2]
        tmp = layer.blobs[0].shape.dim[3]
        if (tmp != kernel_size):
            print 'Error: kernel not square'
            exit()
        print 'output_channel: ' + str(output_channel)
        print 'input_channel: ' + str(input_channel)
        print 'kernel_size: ' + str(kernel_size)
        half_kernel_size = kernel_size / 2;
        idx = 0
        for output_idx in range(output_channel):
            for input_idx in range(input_channel):
                for kernel_row in range(kernel_size):
                    for kernel_col in range(kernel_size):
                        if (kernel_row != half_kernel_size and kernel_col != half_kernel_size):
                            layer.blobs[0].data[idx] = 0.0
                        idx = idx + 1
        print layer.blobs[0].data[0 : 25]

with open('/home/zhangkang-pd/project/cnn_acc/faster-rcnn/faster-rcnn-2/models/FD_ML/faster_rcnn_alt_opt_ml/no_cross_full_cross_gemmlowp_conv1-5_wildface_newdat__uint8_iter_480000.caffemodel', 'w') as file:
    print 'Write to file'
    file.write(net.SerializeToString())
