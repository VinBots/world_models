from utils import DotDict
from dataclasses import dataclass
#from rnn.config import Configuration
import rnn.config, vae.config, controller.config
from dacite import from_dict
import dacite

@dataclass
class Configuration:
    vae: vae.config.Configuration
    rnn: rnn.config.Configuration
    controller: controller.config.Configuration

vae_params = {
    "conv_layers": 4,
    "deconv_layers": 4,
    "stride": 2,
    "conv_1": ("relu", 32, 4),
    "conv_2": ("relu", 64, 4),
    "conv_3": ("relu", 128, 4),
    "conv_4": ("relu", 256, 4),
    "deconv_1": ("relu", 128, 5),
    "deconv_2": ("relu", 64, 5),
    "deconv_3": ("relu", 32, 6),
    "deconv_4": ("sigmoid", 3, 6),
    "tr_epochs" : 100,
}

raw_data = {
    "vae": {"version": "1.0", "vae_params": vae_params},
    "rnn":{"version":"1.0"},
    "controller":{"version":"1.0"}
}


converters = {}

config = dacite.from_dict(
    data_class = Configuration, 
    data = raw_data,
    config = dacite.Config(type_hooks = converters)
)

if config:
    print ("success")



'''
# Task settings


# VAE settings
vae_params = DotDict({
    "resize": (64,64),

})

# conv_x have a format as (activation, output_channels, filter_size)




mdn_rnn_params = DotDict({
    "tr_epochs" : 20,
    "h_units" : 256,
})

# Task settings
car_racing = DotDict({
    "N_z" = 32
})

doom = DotDict({
    "N_z" = 32

})
'''