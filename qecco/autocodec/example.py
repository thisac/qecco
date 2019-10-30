from ..decoder import Decoder
from ..encoder import Encoder
from ..autocodec import Autocodec

encoder_params = {}
decoder_params = {}

encoder = Encoder(encoder_params)
decoder = Decoder(decoder_params)

while True:
    ac = Autocodec(encoder, decoder)

    ac.optimize_encoder()
    ac.optimize_decoder()
