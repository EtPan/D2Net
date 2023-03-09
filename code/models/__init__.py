from .d2net import D2NET

def d2net():
    net = D2NET(in_channels=1, out_channels=1, num_features=32,num_rcab=4)
    net.use_2dconv = False
    net.bandwise = False
    return net

