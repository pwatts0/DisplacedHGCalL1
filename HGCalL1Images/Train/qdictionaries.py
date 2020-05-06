qDicts = {}

q_dict_dense2_binary={
          'dense_2': {'activation': 'quantized_relu(32,16)',  #third dense layer, dense_2
          'bias_quantizer': 'binary()',
          'kernel_quantizer': 'binary()',
          }}
q_dict_conv2d_binary={
          'conv2d': {'activation': 'quantized_relu(32,16)', # first convolutional layer, conv2d
          'bias_quantizer': 'binary()',
          'kernel_quantizer': 'binary()',
          }}          

q_dict_4bit = {
        'dense_2': {'activation': 'quantized_relu(4,0)',  #third dense layer, dense_2. Get error when doing print_qstats if passing dense and dense_1? Try setting per-layer names properly?
        'bias_quantizer': 'quantized_bits(4,0,1)',
        'kernel_quantizer': 'quantized_bits(4,0,1)',
        },
        "QConv2D": {'activation': 'quantized_relu(32,16)', # All 2D convs
        "kernel_quantizer": "quantized_bits(4,0,1)",
        "bias_quantizer": "quantized_bits(4,0,1)",
        }}
                    
qDicts['dense2_binary']   = q_dict_dense2_binary
qDicts['conv2d_binary']   = q_dict_conv2d_binary
qDicts['4_bit']           = q_dict_4bit       